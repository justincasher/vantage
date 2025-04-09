# File: lean_lsp_analyzer.py

import asyncio
import json
import logging
import os
import pathlib
import re
import sys
import uuid
from typing import Any, Dict, Optional, Tuple, List, AsyncGenerator

# --- Logging Configuration ---
# Configure logging level using the LOGLEVEL environment variable (common practice).
# The logging format can also be configured here or set globally elsewhere.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
CONTENT_LENGTH_HEADER = b"Content-Length: "
HEADER_SEPARATOR = b"\r\n\r\n"
DEFAULT_LSP_TIMEOUT = 30 # Seconds for typical LSP request/response

# --- Helper Functions ---

def strip_lean_comments(lean_code: str) -> str:
    """Strips single-line Lean comments (--...) from code."""
    lines = lean_code.splitlines()
    # Remove comments, but keep line structure (even if line becomes empty)
    stripped_lines = [re.sub(r'--.*$', '', line) for line in lines]
    return "\n".join(stripped_lines)

# --- LSP Client Class ---

class LeanLspClient:
    """Manages communication with a lean --server process via LSP over stdio.

    Handles process startup, message framing (JSON-RPC over stdio with
    Content-Length headers), request/response correlation, and asynchronous
    notification handling using asyncio.

    Attributes:
        lean_executable_path (str): Path to the lean executable.
        cwd (str): Working directory for the lean server process.
        timeout (int): Default timeout in seconds for LSP requests.
        process (Optional[asyncio.subprocess.Process]): The subprocess object.
        writer (Optional[asyncio.StreamWriter]): Stream writer for process stdin.
        reader (Optional[asyncio.StreamReader]): Stream reader for process stdout.
    """
    def __init__(self, lean_executable_path: str, cwd: str, timeout: int = DEFAULT_LSP_TIMEOUT):
        """Initializes the LeanLspClient.

        Args:
            lean_executable_path (str): The file path to the 'lean' executable.
            cwd (str): The directory where the 'lean --server' process should be run.
                This is important for the server to find project context (e.g., lakefile).
            timeout (int): The default timeout in seconds for waiting for LSP responses.
        """
        self.lean_executable_path = lean_executable_path
        self.cwd = cwd
        self.timeout = timeout
        self.process: Optional[asyncio.subprocess.Process] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self._message_id_counter = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._notifications_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None # Task for reading stderr
        self._closed = False

    async def start_server(self) -> None:
        """Starts the lean --server subprocess and establishes communication streams.

        Initializes the stdin, stdout, and stderr pipes for interaction. Starts
        background tasks to read stdout (for LSP messages) and stderr (for logging).

        Raises:
            FileNotFoundError: If the lean executable path is invalid.
            ConnectionError: If streams could not be established or the server fails to start.
            Exception: For other potential errors during subprocess creation.
        """
        if self.process and self.process.returncode is None:
            logger.warning("Lean server process already running.")
            return

        logger.info(f"Starting lean server: {self.lean_executable_path} --server in {self.cwd}")
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.lean_executable_path, '--server',
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, # Capture stderr for debugging
                cwd=self.cwd,
                env=os.environ.copy() # Inherit environment
            )
            if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
                 raise ConnectionError("Failed to get streams from lean server process.")

            # Setup writer and reader using the process streams
            self.writer = asyncio.StreamWriter(self.process.stdin, self._get_loop(), None) # type: ignore[arg-type]
            self.reader = asyncio.StreamReader(self._get_loop())
            protocol = asyncio.StreamReaderProtocol(self.reader, self._get_loop())
            # Connect the process's stdout to the reader via the protocol
            transport, _ = await self._get_loop().connect_read_pipe(lambda: protocol, self.process.stdout) # type: ignore[arg-type]

            # Start task to read stderr separately for debugging
            self._stderr_task = asyncio.create_task(self._read_stderr(), name="lsp_stderr_reader")

            # Start the main message reader task
            self._reader_task = asyncio.create_task(self._message_reader_loop(), name="lsp_message_reader")
            logger.info("Lean server started successfully.")

        except FileNotFoundError:
            logger.error(f"Lean executable not found at '{self.lean_executable_path}'.")
            raise
        except Exception as e:
            logger.exception(f"Failed to start lean server: {e}")
            await self.close() # Ensure cleanup on failure
            raise ConnectionError(f"Failed to start lean server: {e}") from e

    async def _read_stderr(self) -> None:
        """Reads and logs stderr from the Lean server process asynchronously."""
        if not self.process or not self.process.stderr:
            logger.warning("Stderr stream not available for reading.")
            return
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    logger.debug("Stderr stream EOF reached.")
                    break
                logger.warning(f"Lean Server STDERR: {line.decode('utf-8', errors='replace').strip()}")
        except asyncio.CancelledError:
             logger.info("Stderr reader task cancelled.")
        except Exception as e:
            if not self._closed: # Avoid logging errors during graceful shutdown
                logger.error(f"Error reading lean server stderr: {e}")
        finally:
             logger.info("Stderr reader task finished.")

    def _get_loop(self):
        """Gets the current asyncio event loop.

        Returns:
            asyncio.AbstractEventLoop: The running event loop. Creates one if none exists.
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("No running event loop, creating a new one (may not be intended).")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    async def _read_message(self) -> Optional[Dict[str, Any]]:
        """Reads a single LSP message (header + JSON body) from the server's stdout.

        Parses the 'Content-Length' header and reads the specified number of bytes
        for the JSON payload.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON message as a dictionary,
            or None if reading failed (e.g., timeout, EOF, invalid format).

        Raises:
            ConnectionError: If the connection closes unexpectedly during read.
        """
        if not self.reader or self.reader.at_eof():
             logger.debug("LSP reader is None or at EOF.")
             return None
        try:
            # Read Content-Length header line by line until separator found
            header_lines = []
            while True:
                 line = await asyncio.wait_for(self.reader.readline(), timeout=self.timeout * 2)
                 if not line: raise asyncio.IncompleteReadError("EOF before headers end", None)
                 header_lines.append(line)
                 if line == b'\r\n': # Empty line signifies end of headers
                      break

            header_str = b"".join(header_lines).decode('ascii') # Headers are ASCII
            content_length = -1
            for h_line in header_str.splitlines():
                 if h_line.lower().startswith('content-length:'):
                      try:
                           content_length = int(h_line.split(':', 1)[1].strip())
                           break
                      except ValueError:
                           logger.error(f"Invalid Content-Length value: {h_line}")
                           return None
            if content_length == -1:
                 logger.error(f"Content-Length header not found in received headers: {header_str!r}")
                 return None

            # Read JSON body
            json_body_bytes = await asyncio.wait_for(self.reader.readexactly(content_length), timeout=self.timeout)
            json_body_str = json_body_bytes.decode('utf-8') # Body is UTF-8

            message = json.loads(json_body_str)
            # logger.debug(f"LSP Received: {json.dumps(message, indent=2)}") # Verbose logging
            return message

        except asyncio.TimeoutError:
            logger.error("Timeout reading LSP message header or body from server.")
            await self.close() # Close connection on timeout
            return None
        except asyncio.IncompleteReadError as e:
            if not self._closed: # Ignore if we initiated the close
                logger.error(f"Server closed connection unexpectedly: {e}")
            await self.close()
            raise ConnectionError("LSP connection closed unexpectedly.") from e # Raise to signal failure
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from server: {e}\nReceived: {json_body_str!r}")
            return None # Don't raise, just log and return None for this message
        except Exception as e:
            if not self._closed:
                 logger.exception(f"Error reading or processing LSP message: {e}")
            await self.close() # Close on unexpected errors
            raise ConnectionError(f"Unexpected error reading LSP message: {e}") from e # Raise

    async def _message_reader_loop(self) -> None:
        """Continuously reads messages from stdout and dispatches them.

        Runs as a background task. Reads messages using `_read_message`.
        If a message is a response, it finds the corresponding waiting request
        future and sets its result or exception. If it's a notification, it
        puts it onto the `_notifications_queue`.
        """
        logger.debug("Starting LSP message reader loop.")
        try:
            while self.process and self.process.returncode is None and not self._closed:
                try:
                    message = await self._read_message()
                except ConnectionError: # Raised by _read_message on critical errors
                     logger.warning("Connection error in reader loop. Exiting.")
                     break

                if message is None:
                    if not self._closed:
                        logger.warning("Message reader received None, likely connection closed or parse error. Exiting loop.")
                    break # Exit loop if read fails or connection closes

                if 'id' in message: # It's a response or error response
                    request_id = int(message['id'])
                    if request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.cancelled():
                            if 'error' in message:
                                future.set_exception(LspResponseError(message['error']))
                            else:
                                future.set_result(message.get('result'))
                        else:
                             logger.warning(f"Received response for already cancelled request ID {request_id}")
                    else:
                        logger.warning(f"Received unexpected response for ID {request_id}")
                elif 'method' in message: # It's a notification or request from server
                    # We only care about notifications for now
                    if 'id' not in message:
                        logger.debug(f"Received notification: {message['method']}")
                        await self._notifications_queue.put(message)
                    else:
                         logger.warning(f"Received unexpected request from server (ignoring): {message}")
                else:
                    logger.warning(f"Received message with unknown structure: {message}")
        except asyncio.CancelledError:
            logger.info("LSP message reader loop cancelled.")
        except Exception as e:
            logger.exception(f"Unexpected error in LSP message reader loop: {e}")
        finally:
            logger.info("LSP message reader loop finished.")
            # Cancel any remaining pending requests when the loop exits
            for req_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.cancel(f"LSP reader loop exited while request {req_id} was pending.")
                self._pending_requests.pop(req_id, None)


    async def _write_message(self, message: Dict[str, Any]) -> None:
        """Formats a message as JSON-RPC and writes it to the server's stdin.

        Prepends the required 'Content-Length' header.

        Args:
            message (Dict[str, Any]): The request or notification payload.

        Raises:
            ConnectionError: If the writer is unavailable or if a connection
                             error occurs during writing.
            Exception: For other errors during JSON serialization or writing.
        """
        if not self.writer or self.writer.is_closing():
            raise ConnectionError("LSP writer is not available or closing.")

        try:
            json_body = json.dumps(message).encode('utf-8')
            header = f"Content-Length: {len(json_body)}\r\n\r\n".encode('utf-8')

            # logger.debug(f"LSP Sending: {json.dumps(message, indent=2)}") # Verbose logging
            self.writer.write(header)
            self.writer.write(json_body)
            await self.writer.drain()
        except (ConnectionResetError, BrokenPipeError) as e:
             logger.error(f"Connection error writing LSP message: {e}")
             await self.close()
             raise ConnectionError(f"Connection error writing LSP message: {e}") from e
        except Exception as e:
            logger.exception(f"Error writing LSP message: {e}")
            await self.close()
            raise ConnectionError(f"Error writing LSP message: {e}") from e

    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Sends an LSP request and waits asynchronously for its response.

        Args:
            method (str): The LSP method name (e.g., "initialize", "$/lean/plainGoal").
            params (Optional[Dict[str, Any]]): The parameters for the request.

        Returns:
            Any: The 'result' field from the LSP response payload.

        Raises:
            ConnectionError: If the client is closed or writing fails.
            asyncio.TimeoutError: If no response is received within the timeout period.
            LspResponseError: If the server returns an error response payload.
            asyncio.CancelledError: If the waiting future is cancelled.
            Exception: For other unexpected errors during the process.
        """
        if self._closed: raise ConnectionError("Client is closed.")

        request_id = self._message_id_counter
        self._message_id_counter += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {}
        }
        future: asyncio.Future = self._get_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await self._write_message(request)
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response to request {request_id} ({method}).")
            # Clean up pending request if timed out
            if request_id in self._pending_requests:
                 pending_future = self._pending_requests.pop(request_id)
                 if not pending_future.done():
                     pending_future.cancel(f"Timeout waiting for response to request {request_id}")
            raise # Re-raise timeout
        except asyncio.CancelledError:
            logger.warning(f"Request {request_id} ({method}) was cancelled.")
            # Ensure cleanup if cancelled externally
            self._pending_requests.pop(request_id, None)
            raise # Re-raise cancellation
        except Exception as e: # Includes ConnectionError from _write_message, LspResponseError
            logger.error(f"Error sending request {request_id} ({method}) or awaiting response: {e}")
            self._pending_requests.pop(request_id, None) # Clean up on other errors too
            raise # Re-raise original error

    async def send_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Sends an LSP notification (no response expected).

        Args:
            method (str): The LSP notification method name (e.g., "initialized").
            params (Optional[Dict[str, Any]]): The parameters for the notification.

        Raises:
            ConnectionError: If the client is closed or writing fails.
            Exception: For other unexpected errors during the process.
        """
        if self._closed: raise ConnectionError("Client is closed.")
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        await self._write_message(notification)

    async def initialize(self) -> Dict[str, Any]:
        """Performs the LSP initialization handshake.

        Sends 'initialize' request and 'initialized' notification.

        Returns:
            Dict[str, Any]: The server capabilities returned in the InitializeResult.

        Raises:
            ConnectionError: If the handshake fails due to connection issues, timeouts,
                             or server errors during initialization.
        """
        logger.info("Sending LSP initialize request.")
        # Basic capabilities sufficient for diagnostics and goal querying
        init_params = {
            "processId": os.getpid(),
            "rootUri": pathlib.Path(self.cwd).as_uri(),
            "capabilities": {
                 "textDocument": {
                     "synchronization": {"dynamicRegistration": False, "willSave": False, "willSaveWaitUntil": False, "didSave": True},
                     "publishDiagnostics": {"relatedInformation": True},
                 },
                 "workspace": {}
            },
            "trace": "off",
            "workspaceFolders": [{"uri": pathlib.Path(self.cwd).as_uri(), "name": os.path.basename(self.cwd)}]
        }
        try:
            # Add lean specific initialization options if known/needed
            # init_params["initializationOptions"] = { ... }
            response = await self.send_request("initialize", init_params)
            logger.info("Received initialize response. Sending initialized notification.")
            await self.send_notification("initialized", {})
            logger.info("LSP Handshake Complete.")
            return response # Return server capabilities
        except Exception as e:
            logger.exception(f"LSP Initialization failed: {e}")
            await self.close() # Ensure cleanup on failure
            raise ConnectionError(f"LSP Initialization failed: {e}") from e

    async def did_open(self, file_uri: str, language_id: str, version: int, text: str) -> None:
        """Sends a textDocument/didOpen notification to the server.

        Args:
            file_uri (str): The URI of the document being opened (e.g., 'file:///...').
            language_id (str): The language ID (e.g., "lean").
            version (int): The initial version number of the document (typically 1).
            text (str): The full text content of the document.

        Raises:
            ConnectionError: If the client is closed or writing fails.
            Exception: For other unexpected errors.
        """
        logger.info(f"Sending textDocument/didOpen for {file_uri}")
        params = {
            "textDocument": {
                "uri": file_uri,
                "languageId": language_id,
                "version": version,
                "text": text
            }
        }
        await self.send_notification("textDocument/didOpen", params)

    async def get_goal(self, file_uri: str, line: int, character: int) -> Optional[Dict[str, Any]]:
        """Sends the custom request to get the Lean proof goal state at a position.

        Uses '$/lean/plainGoal' by default, but this might need verification.

        Args:
            file_uri (str): The URI of the document.
            line (int): The zero-based line number for the position.
            character (int): The zero-based character offset for the position.

        Returns:
            Optional[Dict[str, Any]]: The result payload from the server, which likely
            contains the goal state information. Returns None if the request fails
            (e.g., timeout, server error response, connection error). The exact
            structure of the returned dict needs to be handled by the caller.
        """
        # NOTE: The method name '$/lean/plainGoal' is assumed.
        # Verify against Lean LSP source/docs for the target Lean version.
        method = "$/lean/plainGoal"
        logger.debug(f"Sending {method} request for {file_uri} at {line}:{character}")
        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character}
        }
        try:
            result = await self.send_request(method, params)
            return result
        except LspResponseError as e:
            # Specific errors might be expected if position is invalid, log as warning
            logger.warning(f"{method} request failed with LSP error: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"{method} request timed out for {file_uri} at {line}:{character}")
            return None # Return None on timeout
        except ConnectionError:
             logger.error(f"Connection error during {method} request.")
             return None # Return None if connection failed during request
        except Exception as e:
            logger.error(f"Unexpected error sending {method} request: {e}")
            return None # Return None on other errors

    async def shutdown(self) -> None:
        """Sends the 'shutdown' request to the server.

        Politely asks the server to prepare for termination.
        """
        if self._closed or not self.process or self.process.returncode is not None: return
        logger.info("Sending LSP shutdown request.")
        try:
            # Shutdown expects a response (null) according to LSP spec
            await self.send_request("shutdown")
            logger.info("LSP shutdown request acknowledged by server.")
        except (ConnectionError, asyncio.TimeoutError, LspResponseError) as e:
            # Log error but proceed to exit/close anyway, as server might be unresponsive
            logger.warning(f"Error during LSP shutdown request (proceeding to exit/close): {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during shutdown request: {e}")

    async def exit(self) -> None:
        """Sends the 'exit' notification to the server.

        Informs the server it should terminate. Does not wait for confirmation.
        """
        if self._closed or not self.process or self.process.returncode is not None: return
        logger.info("Sending LSP exit notification.")
        try:
            # Exit is a notification, no response expected
            await self.send_notification("exit")
        except ConnectionError as e:
             # Server might have already exited after shutdown, this is expected
             logger.warning(f"Connection error during LSP exit notification (may be expected): {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during exit notification: {e}")

    async def close(self) -> None:
        """Closes the connection and terminates the server process forcefully.

        Cancels background tasks, closes streams, and terminates/kills the
        subprocess. Should be called to ensure cleanup.
        """
        if self._closed:
            return
        self._closed = True
        logger.info("Closing LSP client connection and terminating server.")

        # 1. Cancel reader/stderr tasks
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()

        # 2. Close streams (helps tasks terminate if blocked on read/write)
        if self.writer and not self.writer.is_closing():
             try:
                 self.writer.close()
                 await self.writer.wait_closed()
             except Exception as e: logger.warning(f"Error closing LSP writer: {e}")
        self.writer = None
        # Reader is associated with the pipe/protocol, closing pipe implicitly handles it.

        # 3. Terminate process if still running
        if self.process and self.process.returncode is None:
            logger.info("Terminating lean server process...")
            try:
                self.process.terminate()
                # Wait briefly for termination, then kill if necessary
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info(f"Lean server process terminated with code {self.process.returncode}.")
            except asyncio.TimeoutError:
                logger.warning("Lean server process did not terminate gracefully after 5s, killing.")
                try:
                    self.process.kill()
                    await self.process.wait() # Wait for kill completion
                    logger.info("Lean server process killed.")
                except ProcessLookupError:
                     logger.warning("Process already killed or finished by the time kill was attempted.")
                except Exception as kill_e:
                     logger.error(f"Error killing lean process: {kill_e}")
            except ProcessLookupError:
                 logger.warning("Process already finished before termination attempt.") # Handle race condition
            except Exception as e:
                logger.error(f"Error during lean server process termination: {e}")
        self.process = None

        # 4. Clear any remaining pending requests
        for req_id, future in list(self._pending_requests.items()):
            if not future.done():
                 future.cancel(f"LSP Client closed while request {req_id} was pending.")
            self._pending_requests.pop(req_id, None)

        # Allow tasks to finish cleanup if needed
        await asyncio.sleep(0.1)
        logger.info("LSP client closed.")

    async def get_diagnostics(self, timeout: Optional[float] = 0.1) -> List[Dict[str, Any]]:
        """Retrieves diagnostic notifications received from the server.

        Checks the internal queue for 'textDocument/publishDiagnostics' messages.
        This is non-blocking beyond the short timeout if the queue is empty.

        Args:
            timeout (Optional[float]): Max time to wait for a message if queue is empty.
                                        Defaults to 0.1s. Use 0 for immediate check.

        Returns:
            List[Dict[str, Any]]: A list of diagnostic objects received since the
            last call (or accumulated if queue wasn't drained). Each dict typically
            contains 'range', 'severity', 'message', etc.
        """
        diagnostics = []
        try:
             while True: # Drain the queue quickly
                 notification = await asyncio.wait_for(self._notifications_queue.get(), timeout=timeout)
                 if notification.get("method") == "textDocument/publishDiagnostics":
                      # Could add filtering by URI here if managing multiple documents
                      # params = notification.get("params", {})
                      # doc_uri = params.get("uri")
                      # if doc_uri == self.target_uri: ...
                      diags = notification.get("params", {}).get("diagnostics", [])
                      logger.debug(f"Retrieved {len(diags)} diagnostics from queue.")
                      diagnostics.extend(diags)
                 else:
                      # Handle other potentially interesting notifications if needed in the future
                      logger.debug(f"Ignoring notification in get_diagnostics: {notification.get('method')}")
                 self._notifications_queue.task_done()
                 # Reset timeout after finding one message? Or keep draining fast? Drain fast.
                 timeout = 0 # Check rest of queue immediately
        except (asyncio.TimeoutError, asyncio.QueueEmpty):
             pass # Expected when queue is empty
        except Exception as e:
             logger.error(f"Error getting diagnostics from queue: {e}")

        return diagnostics


class LspResponseError(Exception):
    """Custom exception for LSP error responses."""
    def __init__(self, error_payload: Dict[str, Any]):
        self.code = error_payload.get('code', 'Unknown')
        self.message = error_payload.get('message', 'Unknown error')
        self.data = error_payload.get('data')
        super().__init__(f"LSP Error Code {self.code}: {self.message}")

# --- Main Analysis Function ---

async def analyze_lean_failure(
    lean_code: str,
    lean_executable_path: str,
    cwd: str,
    timeout_seconds: int = DEFAULT_LSP_TIMEOUT,
    fallback_error: str = "LSP analysis failed."
) -> str:
    """
    Analyzes failing Lean code using LSP to generate an annotated error log.

    Args:
        lean_code (str): The Lean code snippet that failed lake build.
        lean_executable_path (str): Path to the 'lean' executable.
        cwd (str): The directory context in which lean --server should run.
             This should be the temporary project directory where lake build failed.
        timeout_seconds (int): Timeout for individual LSP request/response operations.
        fallback_error (str): Error message to include if LSP analysis fails.

    Returns:
        str: An annotated string containing code lines, goal states, and the LSP error,
             or a fallback error message if LSP analysis fails.
    """
    client = LeanLspClient(lean_executable_path, cwd, timeout=timeout_seconds)
    annotated_lines: List[str] = []
    processed_ok = False
    # Use a consistent temporary filename within the CWD for the URI
    temp_filename = "temp_analysis_file.lean"
    temp_file_path = pathlib.Path(cwd) / temp_filename
    temp_file_uri = temp_file_path.as_uri()
    # We don't actually need to write the file to disk for LSP didOpen

    try:
        await client.start_server()
        await client.initialize()

        stripped_code = strip_lean_comments(lean_code)
        code_lines = stripped_code.splitlines()

        await client.did_open(temp_file_uri, "lean", 1, stripped_code)
        logger.info(f"Sent textDocument/didOpen for URI {temp_file_uri}")

        # Short pause to allow server to process didOpen and potentially send initial diagnostics
        await asyncio.sleep(0.8) # Adjust pause as needed

        last_successful_goal_str = "Initial state (no goal requested)"
        found_error_diagnostic = None
        error_line_index = -1

        for i, line_content in enumerate(code_lines):
            # 1. Query Goal State *before* this line
            current_goal_str = "Error: Could not retrieve goal state" # Default msg
            try:
                # Position is start of the current line (line i, char 0)
                goal_result = await client.get_goal(temp_file_uri, line=i, character=0)

                # --- Parse Goal Result ---
                # This parsing logic needs validation against your Lean version's LSP output.
                if goal_result and isinstance(goal_result, dict):
                     if "rendered" in goal_result:
                          current_goal_str = goal_result["rendered"].strip()
                     elif "plainGoal" in goal_result:
                          current_goal_str = goal_result["plainGoal"].strip()
                     elif "goals" in goal_result and isinstance(goal_result["goals"], list) and goal_result["goals"]:
                          first_goal = goal_result["goals"][0]
                          if "rendered" in first_goal: current_goal_str = first_goal["rendered"].strip()
                          elif "goalState" in first_goal: current_goal_str = first_goal["goalState"].strip()
                          else: current_goal_str = f"Goal 1/{len(goal_result['goals'])} fmt unknown: {str(first_goal)[:100]}"
                     else: current_goal_str = f"Goal state fmt unknown: {str(goal_result)[:100]}"
                elif isinstance(goal_result, str):
                    current_goal_str = goal_result.strip()
                elif goal_result is not None:
                    current_goal_str = f"Unexpected goal result type: {str(goal_result)[:100]}"
                else:
                     current_goal_str = "No goal state reported" # Goal query returned None

                # Standardize 'no goals' message
                if not current_goal_str.strip() or "no goals" in current_goal_str.lower():
                    current_goal_str = "goals accomplished"

            except asyncio.TimeoutError:
                 logger.error(f"Timeout retrieving goal state before line {i+1}")
                 current_goal_str = "Error: Timeout retrieving goal state"
            # Catch other exceptions during goal retrieval
            except Exception as goal_e:
                logger.error(f"Error retrieving goal state before line {i+1}: {goal_e}")
                # current_goal_str keeps default error message

            annotated_lines.append(f"-- Goal: {current_goal_str}")
            last_successful_goal_str = current_goal_str # Store goal before the line

            # 2. Append the code line itself
            annotated_lines.append(line_content)

            # 3. Check for diagnostics reported for this line index 'i'
            await asyncio.sleep(0.1) # Brief pause for server reaction/diagnostic dispatch
            diagnostics = await client.get_diagnostics(timeout=0.2)

            for diag in diagnostics:
                diag_range = diag.get("range", {})
                diag_start = diag_range.get("start")
                if not isinstance(diag_start, dict): continue
                diag_line = diag_start.get("line") # 0-based line number
                severity = diag.get("severity", 0) # 1: Error

                # Check if it's an Error diagnostic *starting* on the current line
                if diag_line == i and severity == 1:
                    error_message = diag.get("message", "Unknown Lean error.")
                    logger.warning(f"Found error originating on line {i+1}: {error_message}")
                    found_error_diagnostic = error_message
                    error_line_index = i
                    # Add the error message immediately after the failing line
                    annotated_lines.append(f"-- Error: {error_message}")
                    break # Stop processing lines after finding first error on this line

            if found_error_diagnostic:
                break # Exit outer loop (line iteration) as well

        # Check for errors reported after iterating through all lines
        if not found_error_diagnostic:
             logger.info("Completed line iteration without finding line-specific error. Checking final diagnostics.")
             await asyncio.sleep(0.2) # Final check
             diagnostics = await client.get_diagnostics(timeout=0.2)
             for diag in diagnostics:
                 severity = diag.get("severity", 0)
                 if severity == 1: # Found any error at all
                      diag_range = diag.get("range", {})
                      diag_start = diag_range.get("start")
                      if not isinstance(diag_start, dict): continue
                      diag_line = diag_start.get("line", -1)
                      error_message = diag.get("message", "Unknown Lean error.")
                      logger.warning(f"Found error after processing all lines (reported for line {diag_line+1}): {error_message}")
                      # Append error at the end; less precise but better than nothing
                      annotated_lines.append(f"-- Error: (Reported at end for line ~{diag_line+1}): {error_message}")
                      found_error_diagnostic = error_message # Mark error as found
                      break

        processed_ok = True # Indicate processing loop completed/exited cleanly

    except ConnectionError as e:
        logger.error(f"LSP Connection Error during analysis: {e}")
        annotated_lines.append(f"-- Error: LSP Connection Failed: {e}")
    except asyncio.TimeoutError as e:
         logger.error(f"LSP Timeout Error during analysis: {e}")
         annotated_lines.append(f"-- Error: LSP Timeout: {e}")
    except Exception as e:
        logger.exception(f"Unhandled exception during LSP analysis: {e}")
        annotated_lines.append(f"-- Error: Unexpected analysis failure: {e}")
    finally:
        # Ensure server shutdown/cleanup happens regardless of errors
        if client:
            try:
                # Attempt graceful shutdown first
                await client.shutdown()
                await client.exit()
            except Exception as shutdown_e:
                 logger.warning(f"Error during graceful shutdown/exit (forcing close): {shutdown_e}")
            finally: # Ensure close runs even if shutdown/exit fail
                 await client.close() # Forcefully closes and terminates

    # Construct final output string based on outcome
    if processed_ok and found_error_diagnostic:
        # LSP analysis ran and found an error. Return the annotated lines.
        return "\n".join(annotated_lines)
    elif processed_ok and not found_error_diagnostic:
         # LSP analysis ran but found no errors (contradicts initial lake build failure).
         logger.warning("LSP analysis completed without finding errors, but lake build failed.")
         # Return the stripped code and the original fallback error for context.
         return f"-- LSP analysis completed without reporting errors. --\n{strip_lean_comments(lean_code)}\n-- Original Failure Report --\n{fallback_error}"
    else:
        # LSP analysis itself failed (e.g., connection, timeout, unexpected exception).
        logger.error("LSP analysis process failed or was incomplete.")
        # Return whatever annotations were gathered plus the original fallback error.
        return f"-- LSP Analysis Incomplete --\n" + "\n".join(annotated_lines) + f"\n-- Original Failure Report --\n{fallback_error}"