# File: lean_automator/lean/lsp_analyzer.py

"""Provides detailed analysis of Lean code failures for LLM self-correction.

This module is a key component of an autoformalization system, designed to
give a Large Language Model (LLM) the necessary feedback to debug and refine
its own generated Lean code. It features the `LeanLspClient` class, an
asynchronous client that interfaces with the Lean Language Server (`lean --server`)
via stdio. This client manages the server process, handles LSP message protocols,
processes diagnostics, and queries proof states.

The primary function, `analyze_lean_failure`, takes Lean code (typically output
by an LLM that failed verification) and performs the following steps:

1. Starts and initializes the `lean --server` with the correct project context.

2. Sends the Lean code snippet to the server for analysis.

3. Captures detailed diagnostic information (errors, warnings) reported by Lean.

4. Queries the specific proof goal state (`$/lean/plainGoal`) before each relevant line.

5. Produces an annotated version of the original code, interspersing the goal states
   and diagnostics as comments directly preceding the lines they pertain to.

6. Appends the original high-level build error (e.g., from `lake build`).

The output from `analyze_lean_failure` is formatted specifically to be fed back
into the LLM, providing it with fine-grained, contextual information about the
errors and proof obligations, enabling it to attempt automated repairs on its
previously generated, incorrect Lean code.
"""

import asyncio
import json
import logging
import os
import pathlib
import re
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# --- Logging Configuration ---
# Configure logging level using the LOGLEVEL environment variable (common practice).
# The logging format can also be configured here or set globally elsewhere.
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Constants ---
CONTENT_LENGTH_HEADER = b"Content-Length: "
HEADER_SEPARATOR = b"\r\n\r\n"
DEFAULT_LSP_TIMEOUT = 30  # Seconds for typical LSP request/response

# --- Helper Functions ---


def strip_lean_comments(lean_code: str) -> str:
    """Strips single-line Lean comments (--...) from code."""
    lines = lean_code.splitlines()
    # Remove comments, but keep line structure (even if line becomes empty)
    stripped_lines = [re.sub(r"--.*$", "", line) for line in lines]
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

    def __init__(
        self,
        lean_executable_path: str,
        cwd: str,
        timeout: int = DEFAULT_LSP_TIMEOUT,
        shared_lib_path: Optional[pathlib.Path] = None,
    ):
        """Initializes the LeanLspClient.

        Args:
            lean_executable_path (str): The file path to the 'lean' executable.
            cwd (str): The directory where the 'lean --server' process should be run.
                This is important for the server to find project context
                (e.g., lakefile).
            timeout (int): The default timeout in seconds for waiting for LSP responses.
        """
        self.lean_executable_path = lean_executable_path
        self.cwd = cwd
        self.timeout = timeout
        self.shared_lib_path = shared_lib_path
        self.process: Optional[asyncio.subprocess.Process] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self._message_id_counter = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._notifications_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None  # Task for reading stderr
        self._closed = False

    async def start_server(self) -> None:
        """Starts the lean --server subprocess and assigns communication streams.

        Initializes the stdin, stdout, and stderr pipes for interaction. Starts
        background tasks to read stdout (for LSP messages) and stderr (for logging).

        Raises:
            FileNotFoundError: If the lean executable path is invalid.
            ConnectionError: If streams could not be established or the server fails
                to start.
            Exception: For other potential errors during subprocess creation.
        """
        if self.process and self.process.returncode is None:
            logger.warning("Lean server process already running.")
            return

        logger.info(
            f"Starting lean server: {self.lean_executable_path} --server in {self.cwd}"
        )
        try:
            # --- Construct LEAN_PATH for LSP Server ---
            subprocess_env = os.environ.copy()
            lean_paths: List[str] = []

            # 1. Detect Stdlib Path
            std_lib_path: Optional[str] = None
            try:
                logger.debug(
                    "LSP Env: Detecting stdlib path using "
                    f"'{self.lean_executable_path} --print-libdir'"
                )
                # Using blocking call here for simplicity before async process starts.
                lean_path_proc = subprocess.run(
                    [self.lean_executable_path, "--print-libdir"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                    encoding="utf-8",
                )
                path_candidate = lean_path_proc.stdout.strip()
                if path_candidate and pathlib.Path(path_candidate).is_dir():
                    std_lib_path = path_candidate
                    logger.info(f"LSP Env: Detected Lean stdlib path: {std_lib_path}")
                    lean_paths.append(std_lib_path)
                else:
                    logger.warning(
                        "LSP Env: Command "
                        f"'{self.lean_executable_path} --print-libdir' "
                        f"did not return valid directory: '{path_candidate}'"
                    )
            except Exception as e:
                logger.warning(
                    "LSP Env: Failed to detect Lean stdlib path via "
                    f"--print-libdir: {e}. Build path might be incomplete."
                )

            # 2. Add Shared Library Build Path (if provided via __init__)
            # Use hasattr for safety in case old instances exist
            # without the attribute during development
            if (
                hasattr(self, "shared_lib_path")
                and self.shared_lib_path
                and self.shared_lib_path.is_dir()
            ):
                # Standard lake build path relative to the library root
                shared_lib_build_path = self.shared_lib_path / ".lake" / "build" / "lib"
                if shared_lib_build_path.is_dir():  # Check if it actually exists
                    abs_shared_lib_build_path = str(shared_lib_build_path.resolve())
                    logger.info(
                        "LSP Env: Adding shared lib build path: "
                        f"{abs_shared_lib_build_path}"
                    )
                    lean_paths.append(abs_shared_lib_build_path)
                else:
                    logger.warning(
                        "LSP Env: Shared library path provided "
                        f"({self.shared_lib_path}), but build dir not found at "
                        f"{shared_lib_build_path}"
                    )
            elif hasattr(
                self, "shared_lib_path"
            ):  # Log if attribute exists but wasn't valid/provided
                logger.debug(
                    "LSP Env: shared_lib_path attribute exists but is not a valid "
                    "directory or was not provided."
                )

            # 3. Add Temporary Project's *Own* Build Path
            temp_project_build_path = pathlib.Path(self.cwd) / ".lake" / "build" / "lib"
            abs_temp_project_build_path = str(temp_project_build_path.resolve())
            logger.debug(
                "LSP Env: Adding temp project's own potential build path: "
                f"{abs_temp_project_build_path}"
            )
            lean_paths.append(
                abs_temp_project_build_path
            )  # Add even if it doesn't exist yet

            # 4. Combine with existing LEAN_PATH (if any)
            existing_lean_path = subprocess_env.get("LEAN_PATH")
            if existing_lean_path:
                # Avoid adding duplicates if already present
                if existing_lean_path not in lean_paths:
                    lean_paths.append(existing_lean_path)
                else:
                    logger.debug(
                        "LSP Env: Existing LEAN_PATH "
                        f"'{existing_lean_path}' is already covered by detected paths."
                    )

            # Set the final LEAN_PATH
            if lean_paths:
                # Filter out potential duplicates just in case before joining
                unique_lean_paths = []
                seen_paths = set()
                for p in lean_paths:
                    if p not in seen_paths:
                        unique_lean_paths.append(p)
                        seen_paths.add(p)

                final_lean_path = os.pathsep.join(unique_lean_paths)
                subprocess_env["LEAN_PATH"] = final_lean_path
                logger.info(
                    f"LSP Env: Setting LEAN_PATH for lean --server: {final_lean_path}"
                )
            else:
                logger.warning("LSP Env: Could not determine any paths for LEAN_PATH.")
            # --- End LEAN_PATH Construction ---

            # Now create the subprocess with the modified environment
            self.process = await asyncio.create_subprocess_exec(
                self.lean_executable_path,
                "--server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=subprocess_env,
            )

            # --- Simplified Stream Assignment ---
            # Assign the streams directly provided by the subprocess
            self.reader = self.process.stdout
            self.writer = self.process.stdin
            # --- End Simplified Assignment ---

            # Start task to read stderr separately for debugging
            self._stderr_task = asyncio.create_task(
                self._read_stderr(), name="lsp_stderr_reader"
            )

            # Start the main message reader task (reads from self.reader)
            self._reader_task = asyncio.create_task(
                self._message_reader_loop(), name="lsp_message_reader"
            )
            logger.info("Lean server started successfully.")

        except FileNotFoundError:
            logger.error(f"Lean executable not found at '{self.lean_executable_path}'.")
            raise
        except Exception as e:
            logger.exception(f"Failed to start lean server: {e}")
            await self.close()  # Ensure cleanup on failure
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
                logger.warning(
                    "Lean Server STDERR: "
                    f"{line.decode('utf-8', errors='replace').strip()}"
                )
        except asyncio.CancelledError:
            logger.info("Stderr reader task cancelled.")
        except Exception as e:
            if not self._closed:  # Avoid logging errors during graceful shutdown
                logger.error(f"Error reading lean server stderr: {e}")
        finally:
            logger.info("Stderr reader task finished.")

    def _get_loop(self):
        """Gets the current asyncio event loop.

        Returns:
            asyncio.AbstractEventLoop: The running event loop. Creates one if
            none exists.
        """
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "No running event loop, creating a new one (may not be intended)."
            )
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
                line = await asyncio.wait_for(
                    self.reader.readline(), timeout=self.timeout * 2
                )
                if not line:
                    raise asyncio.IncompleteReadError("EOF before headers end", None)
                header_lines.append(line)
                if line == b"\r\n":  # Empty line signifies end of headers
                    break

            header_str = b"".join(header_lines).decode("ascii")  # Headers are ASCII
            content_length = -1
            for h_line in header_str.splitlines():
                if h_line.lower().startswith("content-length:"):
                    try:
                        content_length = int(h_line.split(":", 1)[1].strip())
                        break
                    except ValueError:
                        logger.error(f"Invalid Content-Length value: {h_line}")
                        return None
            if content_length == -1:
                logger.error(
                    "Content-Length header not found in received headers: "
                    f"{header_str!r}"
                )
                return None

            # Read JSON body
            json_body_bytes = await asyncio.wait_for(
                self.reader.readexactly(content_length), timeout=self.timeout
            )
            json_body_str = json_body_bytes.decode("utf-8")  # Body is UTF-8

            message = json.loads(json_body_str)

            # logger.debug(f"LSP Received: {json.dumps(message, indent=2)}") # Verbose
            return message

        except asyncio.TimeoutError:
            logger.error("Timeout reading LSP message header or body from server.")
            await self.close()  # Close connection on timeout
            return None
        except asyncio.IncompleteReadError as e:
            if not self._closed:  # Ignore if we initiated the close
                logger.error(f"Server closed connection unexpectedly: {e}")
            await self.close()
            raise ConnectionError(
                "LSP connection closed unexpectedly."
            ) from e  # Raise to signal failure
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from server: {e}\nReceived: {json_body_str!r}"
            )
            return None  # Don't raise, just log and return None for this message
        except Exception as e:
            if not self._closed:
                logger.exception(f"Error reading or processing LSP message: {e}")
            await self.close()  # Close on unexpected errors
            raise ConnectionError(
                f"Unexpected error reading LSP message: {e}"
            ) from e  # Raise

    async def _message_reader_loop(self) -> None:
        """Continuously reads messages from stdout and dispatches them.

        Runs as a background task. Reads messages using `_read_message`.
        Handles responses, notifications, and ignores server-to-client requests.
        """
        logger.debug("Starting LSP message reader loop.")
        try:
            while self.process and self.process.returncode is None and not self._closed:
                try:
                    message = await self._read_message()
                except ConnectionError:  # Raised by _read_message on critical errors
                    logger.warning("Connection error in reader loop. Exiting.")
                    break

                if message is None:
                    if not self._closed:
                        logger.warning(
                            "Message reader received None, likely connection closed "
                            "or parse error. Exiting loop."
                        )
                    break  # Exit loop if read fails or connection closes

                msg_id = message.get("id")
                msg_method = message.get("method")

                # --- Case 1: Response to our request ---
                # Has 'id', does NOT have 'method'. Our client uses integer IDs.
                if msg_id is not None and msg_method is None:
                    try:
                        request_id = int(msg_id)  # Convert expected integer ID
                        if request_id in self._pending_requests:
                            future = self._pending_requests.pop(request_id)
                            if not future.cancelled():
                                if "error" in message:
                                    logger.debug(
                                        f"Received error response for ID {request_id}"
                                    )
                                    future.set_exception(
                                        LspResponseError(message["error"])
                                    )
                                else:
                                    logger.debug(
                                        f"Received result response for ID {request_id}"
                                    )
                                    future.set_result(message.get("result"))
                            else:
                                logger.warning(
                                    "Received response for already cancelled "
                                    f"request ID {request_id}"
                                )
                        else:
                            # Could be a response to a request that already
                            # timed out/cancelled, or unexpected
                            logger.warning(
                                f"Received response for ID {request_id}, but it was "
                                "not pending."
                            )
                    except ValueError:
                        # This happens if server sends a response with a non-integer ID,
                        # which shouldn't occur if it's replying to us.
                        logger.warning(
                            f"Received response with non-integer ID '{msg_id}', "
                            "which was not expected from our requests."
                        )
                    except Exception as e:
                        # Catch other potential errors during future handling
                        logger.exception(
                            f"Error processing response for ID {msg_id}: {e}"
                        )
                        # Ensure future is removed if error occurs during processing
                        if (
                            "request_id" in locals()
                            and request_id in self._pending_requests
                        ):
                            future = self._pending_requests.pop(request_id)
                            if not future.done():
                                future.set_exception(e)

                # --- Case 2: Notification from server ---
                # Has 'method', does NOT have 'id'.
                elif msg_method is not None and msg_id is None:
                    logger.debug(f"Received notification: {msg_method}")
                    # Handle specific notifications we care about
                    if msg_method == "textDocument/publishDiagnostics":
                        diags_data = message.get("params", {}).get("diagnostics", [])
                        uri = message.get("params", {}).get("uri")
                        logger.info(
                            "Received textDocument/publishDiagnostics with "
                            f"{len(diags_data)} diagnostics for URI {uri}."
                        )
                        logger.debug(
                            "Diagnostics data (first item snippet): "
                            f"{str(diags_data[0])[:200]}..."
                            if diags_data
                            else "No diagnostics in this message"
                        )
                        logger.debug(
                            "Queueing publishDiagnostics notification. "
                            "Queue size approx: "
                            f"{self._notifications_queue.qsize() + 1}"
                        )
                        await self._notifications_queue.put(message)
                    elif msg_method == "$/lean/fileProgress":
                        # Lean sends progress updates. Log them for debugging if needed.
                        progress_params = message.get("params", {})
                        logger.debug(f"Lean File Progress: {progress_params}")
                        # Optionally put on queue if needed downstream:
                        # await self._notifications_queue.put(message)
                    else:
                        logger.debug(f"Ignoring unhandled notification: {msg_method}")

                # --- Case 3: Request from server ---
                # Has 'method' AND 'id'. The server wants US to do something.
                elif msg_method is not None and msg_id is not None:
                    # This is where 'register_lean_watcher' likely falls.
                    logger.warning(
                        "Received unexpected request from server (Method: "
                        f"{msg_method}, ID: {msg_id}). Ignoring."
                    )
                    # For this tool's purpose, we don't need to respond to server
                    # requests.
                    # A fully compliant client might send back a MethodNotFound error
                    # response.

                # --- Case 4: Unknown message structure ---
                else:
                    logger.warning(
                        f"Received message with unknown structure: {message}"
                    )

        except asyncio.CancelledError:
            logger.info("LSP message reader loop cancelled.")
        except Exception as e:
            # Log unexpected errors in the loop itself
            if not self._closed:  # Avoid logging errors during graceful shutdown
                logger.exception(f"Unexpected error in LSP message reader loop: {e}")
        finally:
            logger.info("LSP message reader loop finished.")
            # Cancel any remaining pending requests when the loop exits
            for req_id, future in list(self._pending_requests.items()):
                if not future.done():
                    future.cancel(
                        f"LSP reader loop exited while request {req_id} was pending."
                    )
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
            json_body = json.dumps(message).encode("utf-8")
            header = f"Content-Length: {len(json_body)}\r\n\r\n".encode()

            # logger.debug(f"LSP Sending: {json.dumps(message, indent=2)}") # Verbose
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

    async def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
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
        if self._closed:
            raise ConnectionError("Client is closed.")

        request_id = self._message_id_counter
        self._message_id_counter += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        future: asyncio.Future = self._get_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            await self._write_message(request)
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout waiting for response to request {request_id} ({method})."
            )
            # Clean up pending request if timed out
            if request_id in self._pending_requests:
                pending_future = self._pending_requests.pop(request_id)
                if not pending_future.done():
                    pending_future.cancel(
                        f"Timeout waiting for response to request {request_id}"
                    )
            raise  # Re-raise timeout
        except asyncio.CancelledError:
            logger.warning(f"Request {request_id} ({method}) was cancelled.")
            # Ensure cleanup if cancelled externally
            self._pending_requests.pop(request_id, None)
            raise  # Re-raise cancellation
        except (
            Exception
        ) as e:  # Includes ConnectionError from _write_message, LspResponseError
            logger.error(
                "Error sending request "
                f"{request_id} ({method}) or awaiting response: {e}"
            )
            self._pending_requests.pop(request_id, None)  # Clean up on other errors too
            raise  # Re-raise original error

    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Sends an LSP notification (no response expected).

        Args:
            method (str): The LSP notification method name (e.g., "initialized").
            params (Optional[Dict[str, Any]]): The parameters for the notification.

        Raises:
            ConnectionError: If the client is closed or writing fails.
            Exception: For other unexpected errors during the process.
        """
        if self._closed:
            raise ConnectionError("Client is closed.")
        notification = {"jsonrpc": "2.0", "method": method, "params": params or {}}
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
                    "synchronization": {
                        "dynamicRegistration": False,
                        "willSave": False,
                        "willSaveWaitUntil": False,
                        "didSave": True,
                    },
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {},
            },
            "trace": "off",
            "workspaceFolders": [
                {
                    "uri": pathlib.Path(self.cwd).as_uri(),
                    "name": os.path.basename(self.cwd),
                }
            ],
        }
        try:
            # Add lean specific initialization options if known/needed
            # init_params["initializationOptions"] = { ... }
            response = await self.send_request("initialize", init_params)
            logger.info(
                "Received initialize response. Sending initialized notification."
            )
            await self.send_notification("initialized", {})
            logger.info("LSP Handshake Complete.")
            return response  # Return server capabilities
        except Exception as e:
            logger.exception(f"LSP Initialization failed: {e}")
            await self.close()  # Ensure cleanup on failure
            raise ConnectionError(f"LSP Initialization failed: {e}") from e

    async def did_open(
        self, file_uri: str, language_id: str, version: int, text: str
    ) -> None:
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
                "text": text,
            }
        }
        await self.send_notification("textDocument/didOpen", params)

    async def get_goal(
        self, file_uri: str, line: int, character: int
    ) -> Optional[Dict[str, Any]]:
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
        method = "$/lean/plainGoal" # <<< Still assuming this method name
        logger.debug(f"Sending {method} request for {file_uri} at {line}:{character}")
        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character},
        }
        try:
            result = await self.send_request(method, params)
            logger.debug(f"Received result for get_goal({line}:{character}): {result!r}")
            return result
        except LspResponseError as e:
            # Specific errors might be expected if position is invalid, log as warning
            logger.warning(f"{method} request failed with LSP error: {e}")
            logger.debug(f"Returning None due to LspResponseError for get_goal({line}:{character})")
            return None
        except asyncio.TimeoutError:
            logger.error(
                f"{method} request timed out for {file_uri} at {line}:{character}"
            )
            logger.debug(f"Returning None due to TimeoutError for get_goal({line}:{character})")
            return None  # Return None on timeout
        except ConnectionError:
            logger.error(f"Connection error during {method} request.")
            logger.debug(f"Returning None due to ConnectionError for get_goal({line}:{character})")
            return None  # Return None if connection failed during request
        except Exception as e:
            logger.error(f"Unexpected error sending {method} request: {e}")
            logger.debug(f"Returning None due to Exception '{e}' for get_goal({line}:{character})")
            return None  # Return None on other errors


    async def shutdown(self) -> None:
        """Sends the 'shutdown' request to the server.

        Politely asks the server to prepare for termination.
        """
        if self._closed or not self.process or self.process.returncode is not None:
            return
        logger.info("Sending LSP shutdown request.")
        try:
            # Shutdown expects a response (null) according to LSP spec
            await self.send_request("shutdown")
            logger.info("LSP shutdown request acknowledged by server.")
        except (ConnectionError, asyncio.TimeoutError, LspResponseError) as e:
            # Log error but proceed to exit/close anyway, as server might be
            # unresponsive
            logger.warning(
                f"Error during LSP shutdown request (proceeding to exit/close): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during shutdown request: {e}")

    async def exit(self) -> None:
        """Sends the 'exit' notification to the server.

        Informs the server it should terminate. Does not wait for confirmation.
        """
        if self._closed or not self.process or self.process.returncode is not None:
            return
        logger.info("Sending LSP exit notification.")
        try:
            # Exit is a notification, no response expected
            await self.send_notification("exit")
        except ConnectionError as e:
            # Server might have already exited after shutdown, this is expected
            logger.warning(
                f"Connection error during LSP exit notification (may be expected): {e}"
            )
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
            except Exception as e:
                logger.warning(f"Error closing LSP writer: {e}")
        self.writer = None
        # Reader is associated with the pipe/protocol, closing pipe implicitly
        # handles it.

        # 3. Terminate process if still running
        if self.process and self.process.returncode is None:
            logger.info("Terminating lean server process...")
            try:
                self.process.terminate()
                # Wait briefly for termination, then kill if necessary
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
                logger.info(
                    "Lean server process terminated with code "
                    f"{self.process.returncode}."
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Lean server process did not terminate gracefully after 5s, "
                    "killing."
                )
                try:
                    self.process.kill()
                    await self.process.wait()  # Wait for kill completion
                    logger.info("Lean server process killed.")
                except ProcessLookupError:
                    logger.warning(
                        "Process already killed or finished by the time kill was "
                        "attempted."
                    )
                except Exception as kill_e:
                    logger.error(f"Error killing lean process: {kill_e}")
            except ProcessLookupError:
                logger.warning(
                    "Process already finished before termination attempt."
                )  # Handle race condition
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

    async def get_diagnostics(
        self, timeout: Optional[float] = 0.1
    ) -> List[Dict[str, Any]]:
        """Retrieves diagnostic notifications received from the server.

        Checks the internal queue for 'textDocument/publishDiagnostics' messages.
        It attempts to drain the queue using small timeouts.

        Args:
            timeout (Optional[float]): Initial timeout to wait for the first message
                                        if the queue might be initially empty.
                                        Defaults to 0.1s. Subsequent checks use
                                        a near-zero timeout to drain quickly.

        Returns:
            List[Dict[str, Any]]: A list of diagnostic objects received since the
            last call (or accumulated if queue wasn't drained). Each dict typically
            contains 'range', 'severity', 'message', etc.
        """
        diagnostics = []
        first_attempt = True
        # Use a very small timeout for draining attempts after the first
        drain_timeout = 0.01

        logger.debug(
            "Entering get_diagnostics. Approx queue size: "
            f"{self._notifications_queue.qsize()}"
        )

        while True:
            current_timeout = timeout if first_attempt else drain_timeout
            first_attempt = False  # Use drain_timeout for subsequent attempts

            try:
                # Wait for an item with the current timeout
                notification = await asyncio.wait_for(
                    self._notifications_queue.get(), timeout=current_timeout
                )
                logger.debug(
                    "Retrieved notification from queue: "
                    f"method='{notification.get('method')}'"
                )

                if notification.get("method") == "textDocument/publishDiagnostics":
                    # Could add filtering by URI here if managing multiple documents
                    # params = notification.get("params", {})
                    # doc_uri = params.get("uri")
                    # if doc_uri == self.target_uri: ...
                    diags = notification.get("params", {}).get("diagnostics", [])
                    logger.debug(
                        f"Retrieved {len(diags)} diagnostics from notification."
                    )
                    if diags:  # Log content snippet only if diagnostics are present
                        logger.debug(
                            "Diagnostic content (first item snippet): "
                            f"{str(diags[0])[:200]}..."
                        )
                    else:
                        logger.debug("No diagnostic items within this notification.")
                    diagnostics.extend(diags)
                else:
                    # Handle other potentially interesting notifications if needed in
                    # the future
                    logger.debug(
                        "Ignoring notification type during diagnostic collection: "
                        f"{notification.get('method')}"
                    )

                self._notifications_queue.task_done()

            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                # This is the expected way to exit the loop when the queue is empty
                logger.debug(
                    f"Queue appears empty (timeout={current_timeout}s). Exiting "
                    "diagnostic collection loop."
                )
                break
            except Exception as e:
                logger.error(
                    f"Error getting diagnostics from queue: {e}", exc_info=True
                )
                break  # Exit loop on unexpected errors

        logger.debug(f"Exiting get_diagnostics. Total collected: {len(diagnostics)}")
        return diagnostics


class LspResponseError(Exception):
    """Custom exception for LSP error responses."""

    def __init__(self, error_payload: Dict[str, Any]):
        self.code = error_payload.get("code", "Unknown")
        self.message = error_payload.get("message", "Unknown error")
        self.data = error_payload.get("data")
        super().__init__(f"LSP Error Code {self.code}: {self.message}")


# --- Main Analysis Function ---


async def analyze_lean_failure(
    lean_code: str,
    lean_executable_path: str,
    cwd: str,
    shared_lib_path: Optional[pathlib.Path],
    timeout_seconds: int = DEFAULT_LSP_TIMEOUT,
    fallback_error: str = "LSP analysis failed.",
) -> str:
    """
    Analyzes failing Lean code using LSP and direct compilation check.
    Annotates the code with goal states (or 'N/A' after first detected error),
    inline LSP diagnostics, and the original build failure message.

    Args:
        lean_code (str): The Lean code string to analyze.
        lean_executable_path (str): Path to the 'lean' executable.
        cwd (str): Working directory for the lean server process (temp project).
        shared_lib_path (Optional[pathlib.Path]): Path to the root of the shared
            library dependency.
        timeout_seconds (int): Timeout for individual LSP requests (like get_goal).
        fallback_error (str): Error message (typically lake build output) to include.

    Returns:
        str: Annotated Lean code with goals/N/A, LSP diagnostics, and the build error.
    """

    # --- Helper function for LEAN_PATH (to avoid duplication) ---
    def _get_lean_path_env(
        exec_path: str,
        project_cwd: str,
        lib_path: Optional[pathlib.Path]
    ) -> Dict[str, str]:
        """Constructs the environment dictionary with LEAN_PATH set."""
        subprocess_env = os.environ.copy()
        lean_paths: List[str] = []

        # 1. Detect Stdlib Path (synchronous call before async process)
        try:
            lean_path_proc = subprocess.run(
                [exec_path, "--print-libdir"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
                encoding="utf-8",
            )
            path_candidate = lean_path_proc.stdout.strip()
            if path_candidate and pathlib.Path(path_candidate).is_dir():
                std_lib_path = path_candidate
                logger.debug(f"Path Env: Detected Lean stdlib path: {std_lib_path}")
                lean_paths.append(std_lib_path)
            else:
                logger.warning(f"Path Env: --print-libdir invalid output: '{path_candidate}'")
        except Exception as e:
            logger.warning(f"Path Env: Failed to detect Lean stdlib path: {e}")

        # 2. Add Shared Library Build Path
        if lib_path and lib_path.is_dir():
            shared_lib_build_path = lib_path / ".lake" / "build" / "lib"
            if shared_lib_build_path.is_dir():
                abs_path = str(shared_lib_build_path.resolve())
                logger.debug(f"Path Env: Adding shared lib build path: {abs_path}")
                lean_paths.append(abs_path)
            else:
                 logger.warning(f"Path Env: Shared lib path provided but build dir not found: {shared_lib_build_path}")
        elif lib_path:
             logger.warning(f"Path Env: Provided shared_lib_path is not a valid directory: {lib_path}")


        # 3. Add Temporary Project's *Own* Build Path
        temp_project_build_path = pathlib.Path(project_cwd) / ".lake" / "build" / "lib"
        abs_temp_path = str(temp_project_build_path.resolve())
        logger.debug(f"Path Env: Adding temp project's own build path: {abs_temp_path}")
        lean_paths.append(abs_temp_path) # Add even if it doesn't exist yet

        # 4. Combine with existing LEAN_PATH
        existing_lean_path = subprocess_env.get("LEAN_PATH")
        if existing_lean_path:
             if existing_lean_path not in lean_paths: # Avoid adding if already covered
                 logger.debug(f"Path Env: Adding existing LEAN_PATH: {existing_lean_path}")
                 lean_paths.append(existing_lean_path)

        # Set the final LEAN_PATH
        if lean_paths:
            unique_lean_paths = list(dict.fromkeys(lean_paths)) # Order-preserving unique
            final_lean_path = os.pathsep.join(unique_lean_paths)
            subprocess_env["LEAN_PATH"] = final_lean_path
            logger.info(f"Path Env: Setting LEAN_PATH: {final_lean_path}")
        else:
            logger.warning("Path Env: Could not determine any paths for LEAN_PATH.")

        return subprocess_env

    # --- Pre-check using direct lean execution ---
    first_compiler_error_line = float('inf') # Use 1-based indexing from compiler
    temp_precheck_filename = "precheck_analysis_file.lean"
    temp_precheck_path = pathlib.Path(cwd) / temp_precheck_filename
    stripped_code_for_precheck = strip_lean_comments(lean_code) # Use stripped code

    try:
        logger.info(f"Pre-check: Writing code to {temp_precheck_path}")
        temp_precheck_path.write_text(stripped_code_for_precheck, encoding='utf-8')

        logger.info(f"Pre-check: Running: {lean_executable_path} {temp_precheck_filename}")
        precheck_timeout = 25 # Slightly longer timeout
        precheck_env = _get_lean_path_env(lean_executable_path, cwd, shared_lib_path)

        result = subprocess.run(
            [lean_executable_path, temp_precheck_filename],
            capture_output=True,
            text=True, # Get text output
            cwd=cwd,
            timeout=precheck_timeout,
            encoding='utf-8', # Specify encoding
            errors='replace', # Handle potential decoding errors
            env=precheck_env
        )

        # Attempt to parse stdout for the first error line, as logs showed errors there
        error_pattern = re.compile(r".*?:(\d+):\d+:\s*error:")
        output_lines = result.stdout.splitlines() # Read from stdout
        logger.debug(f"Pre-check stdout ({len(output_lines)} lines): {result.stdout[:500]}...") # Log stdout snippet

        for line in output_lines: # Loop through stdout lines
            match = error_pattern.match(line)
            if match:
                first_compiler_error_line = int(match.group(1))
                logger.info(f"Pre-check: Found first compiler error on line: {first_compiler_error_line} (from stdout)")
                break # Stop after finding the first one

        if first_compiler_error_line == float('inf'):
             # Check stderr as a fallback or just log it wasn't found in stdout
             logger.info("Pre-check: No errors found matching pattern in compiler stdout.")
             # Optionally parse stderr here too if errors might appear in either place
             # For now, just log stderr if it exists
             if result.stderr:
                  logger.debug(f"Pre-check stderr content: {result.stderr[:500]}...")


    except FileNotFoundError:
         logger.error(f"Pre-check Failed: Lean executable not found at '{lean_executable_path}'.")
         # Decide if this is fatal or if LSP analysis should proceed anyway
         # For now, we'll let LSP proceed, first_compiler_error_line remains 'inf'
    except subprocess.TimeoutExpired:
        logger.warning(f"Pre-check: Timed out after {precheck_timeout}s.")
    except Exception as precheck_e:
        # Catch other potential errors like permission issues, etc.
        logger.warning(f"Pre-check: Failed with unexpected error: {precheck_e}", exc_info=True)
    finally:
        # Clean up the temporary file used for pre-check
        if temp_precheck_path.exists():
            try:
                temp_precheck_path.unlink()
                logger.debug(f"Pre-check: Cleaned up {temp_precheck_path}")
            except OSError as e:
                logger.warning(f"Pre-check: Could not delete temp file {temp_precheck_path}: {e}")

    # --- Initialize LSP Client ---
    client_timeout = max(timeout_seconds, 60)
    # NOTE: LeanLspClient should ideally also use the _get_lean_path_env helper,
    # assuming it's refactored out or the logic here is reliable.
    # For now, the client uses its internal LEAN_PATH logic.
    client = LeanLspClient(
        lean_executable_path,
        cwd,
        timeout=client_timeout,
        shared_lib_path=shared_lib_path,
    )

    # --- Analysis Variables ---
    analysis_succeeded = False
    collected_diagnostics: List[Dict[str, Any]] = []
    diagnostics_by_line: Dict[int, List[str]] = defaultdict(list)
    first_lsp_error_line_idx = float('inf') # Use 0-based indexing for LSP
    effective_error_line_idx = float('inf') # 0-based index

    # --- Helper functions for goal/diagnostic processing ---
    def _parse_goal_result(goal_result: Optional[Any]) -> str:
        """Parses the LSP goal result into a string."""
        if goal_result and isinstance(goal_result, dict):
            if "rendered" in goal_result: return goal_result["rendered"].strip()
            if "plainGoal" in goal_result: return goal_result["plainGoal"].strip()
            if ("goals" in goal_result and isinstance(goal_result["goals"], list)
                    and goal_result["goals"]):
                # Format multiple goals concisely
                return (f"{len(goal_result['goals'])} goal(s): " +
                        " | ".join([
                            g.get("rendered", str(g)).replace("\n", " ")
                            for g in goal_result['goals']
                        ]))
            return f"Goal state fmt unknown: {str(goal_result)[:300]}"
        elif isinstance(goal_result, str):
            return goal_result.strip()
        elif goal_result is not None:
            return f"Unexpected goal result type: {str(goal_result)[:100]}"
        else:
            # This case hit if client.get_goal returned None (e.g., timeout, error)
            # The outer try/except in the loop handles setting specific error messages.
            # Returning this implies the request succeeded but returned null/empty.
            return "No goal state reported" # Or potentially ""

    def _clean_goal_string(goal_str: str) -> Tuple[str, str]:
        """Cleans goal string and creates single-line version."""
        cleaned = goal_str.strip()
        if cleaned.startswith("```lean"): cleaned = cleaned.removeprefix("```lean").strip()
        if cleaned.endswith("```"): cleaned = cleaned.removesuffix("```").strip()
        if not cleaned or "no goals" in cleaned.lower():
            cleaned = "goals accomplished"
        single_line = cleaned.replace("\n", "; ")
        return cleaned, single_line

    def _format_diagnostic_line(diag: Dict[str, Any], line_idx: int) -> str:
         """Formats a single diagnostic into a comment line."""
         severity_map = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}
         severity = severity_map.get(diag.get("severity", 0), "Unknown")
         message = diag.get("message", "Unknown Lean diagnostic.")
         diag_range = diag.get("range", {})
         start_pos = diag_range.get("start", {})
         start_char_disp = start_pos.get("character", -1) + 1
         end_pos = diag_range.get("end", {})
         # Use the line_idx passed in for start line, as LSP might report range across lines
         start_line_disp = line_idx + 1
         end_line_disp = end_pos.get("line", -1) + 1
         end_char_disp = end_pos.get("character", -1) + 1

         # Format the diagnostic message
         return (f"-- {severity}: (Reported range "
                 f"L{start_line_disp}:{start_char_disp}"
                 f"-L{end_line_disp}:{end_char_disp}): {message}")

    # --- File URI ---
    temp_filename = "temp_analysis_file.lean"
    temp_file_path = pathlib.Path(cwd) / temp_filename
    temp_file_uri = temp_file_path.as_uri()

    try:
        # --- Start LSP and Open Document ---
        await client.start_server()
        await client.initialize()

        # Use the *same* stripped code for LSP as for pre-check
        stripped_code = strip_lean_comments(lean_code)
        code_lines = stripped_code.splitlines()

        await client.did_open(temp_file_uri, "lean", 1, stripped_code)
        logger.info(f"Sent textDocument/didOpen for URI {temp_file_uri}")

        # --- Wait for initial diagnostics ---
        # Give Lean LSP time to process the file after opening.
        initial_wait_time = 5.0 # Adjust based on typical project size/complexity
        logger.info(f"Waiting {initial_wait_time}s for initial diagnostics...")
        await asyncio.sleep(initial_wait_time)
        # Try to collect diagnostics generated so far
        collected_diagnostics = await client.get_diagnostics(timeout=1.0)
        logger.info(f"Collected {len(collected_diagnostics)} initial diagnostics.")

        # --- Determine effective error line ---
        for diag in collected_diagnostics:
             if diag.get("severity") == 1: # 1 is Error severity in LSP
                 line_idx = diag.get("range", {}).get("start", {}).get("line", float('inf'))
                 # Ensure line_idx is treated as float for min() comparison initially
                 try:
                     current_min_idx = float(line_idx)
                     first_lsp_error_line_idx = min(first_lsp_error_line_idx, current_min_idx)
                 except (ValueError, TypeError):
                     logger.warning(f"Could not parse line number from diagnostic start: {line_idx}")


        # Combine pre-check (1-based) and LSP error (0-based)
        # Ensure comparison is between numbers (float('inf') handles cases where one is not found)
        effective_error_line_idx = min(float(first_compiler_error_line) - 1, first_lsp_error_line_idx)

        if effective_error_line_idx != float('inf'):
             # Convert back to int for display if not infinity
             effective_error_line_idx = int(effective_error_line_idx)
             logger.info(f"Effective first error detected near line index: {effective_error_line_idx} "
                         f"(Line number {effective_error_line_idx + 1})")
        else:
             logger.info("No errors detected from pre-check or initial LSP diagnostics.")
             effective_error_line_idx = -1 # Use -1 to indicate no error found, easier than inf

        # --- Goal fetching loop with N/A marking ---
        logger.info("Starting line-by-line goal annotation (marking N/A after error)...")
        goal_lines_buffer: List[Optional[str]] = [] # Store goals or None
        code_lines_buffer: List[str] = code_lines # Keep original code lines

        for i, line_content in enumerate(code_lines_buffer):
            goal_comment_line: Optional[str] = None # Default to no goal comment

            # Only process goals for non-empty lines AND before the effective error
            if line_content.strip():
                if effective_error_line_idx != -1 and i >= effective_error_line_idx:
                    # Error detected earlier, mark goal as N/A
                    goal_comment_line = f"-- Goal: N/A (error detected near line {effective_error_line_idx + 1})"
                    # Log moved inside this block
                    logger.debug(f"Marking goal N/A for code line index {i}")
                else:
                    # No error detected yet, try to fetch the goal
                    current_goal_str = "Error: Could not retrieve goal state" # Default msg
                    try:
                        goal_result = await asyncio.wait_for(
                            client.get_goal(temp_file_uri, line=i, character=0),
                            timeout=timeout_seconds,
                        )
                        logger.debug(f"Raw goal_result before line {i+1}: {goal_result!r}")
                        parsed_goal = _parse_goal_result(goal_result)
                        _, single_line_goal = _clean_goal_string(parsed_goal)
                        goal_comment_line = f"-- Goal: {single_line_goal}"

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout ({timeout_seconds}s) retrieving goal before line {i + 1}")
                        goal_comment_line = "-- Goal: Error: Timeout retrieving goal state"
                    except LspResponseError as goal_lsp_e:
                         logger.warning(f"LSP error retrieving goal before line {i + 1}: {goal_lsp_e}")
                         # Provide a slightly more informative error message
                         goal_comment_line = f"-- Goal: Error retrieving goal (LSP: {goal_lsp_e.code})"
                    except ConnectionError as goal_conn_e:
                        logger.error(f"Connection error retrieving goal before line {i+1}: {goal_conn_e}")
                        goal_comment_line = "-- Goal: Error: Connection failed during goal retrieval"
                        # Maybe re-raise or break if connection is lost? For now, just mark goal.
                    except Exception as goal_e:
                        logger.error(f"Error retrieving goal state before line {i + 1}: {goal_e}", exc_info=True) # Log full trace for unexpected
                        goal_comment_line = f"-- Goal: Error retrieving goal: {type(goal_e).__name__}"

            if i == 2: # Index corresponding to the 'theorem' line
                logger.debug(f"State for line index 2: line_content='{line_content}', effective_error_line_idx={effective_error_line_idx}")
                # Add logs for the variables calculated *inside* the else block if needed
                # logger.debug(f"State for line index 2 (else block): goal_result={goal_result}, parsed_goal='{parsed_goal}', single_line_goal='{single_line_goal}'") # If needed
                logger.debug(f"State for line index 2: Calculated goal_comment_line='{goal_comment_line}'")

            goal_lines_buffer.append(goal_comment_line) # Append goal/N/A/None

        # --- Collect final diagnostics and process ---
        diagnostic_collection_timeout = 5.0
        logger.info(f"Collecting final diagnostics (timeout={diagnostic_collection_timeout}s)...")
        # Fetch diagnostics again in case more arrived during goal fetching
        more_diagnostics = await client.get_diagnostics(timeout=diagnostic_collection_timeout)
        collected_diagnostics.extend(more_diagnostics)
        logger.info(f"Total collected diagnostics: {len(collected_diagnostics)}")

        if collected_diagnostics:
            logger.debug(f"Processing {len(collected_diagnostics)} total diagnostics for inline reporting...")
            formatted_diags_count = 0
            seen_diags = set()
            for diag in collected_diagnostics:
                # De-duplicate diagnostics (same as original code)
                diag_sig = (
                    diag.get("severity"),
                    diag.get("range", {}).get("start", {}).get("line"),
                    diag.get("range", {}).get("start", {}).get("character"),
                    diag.get("range", {}).get("end", {}).get("line"),
                    diag.get("range", {}).get("end", {}).get("character"),
                    diag.get("message"),
                )
                if diag_sig in seen_diags: continue
                seen_diags.add(diag_sig)

                # Get the primary line index for mapping
                start_line_idx = diag.get("range", {}).get("start", {}).get("line", -1)

                if start_line_idx != -1:
                    diag_log_line = _format_diagnostic_line(diag, start_line_idx)
                    diagnostics_by_line[start_line_idx].append(diag_log_line)
                    formatted_diags_count += 1
            logger.info(f"Processed {formatted_diags_count} unique diagnostics into line map.")

        # --- Assemble final output ---
        logger.debug(f"Goal lines buffer before assembly: {goal_lines_buffer}")
        logger.debug(f"Diagnostics by line map: {diagnostics_by_line}")
        logger.info("Assembling final annotated output...")
        final_output_lines = []
        for i, code_line in enumerate(code_lines_buffer):
            # 1. Add goal comment from buffer (if one exists for this line)
            goal_comment = goal_lines_buffer[i]
            if goal_comment is not None:
                final_output_lines.append(goal_comment)

            # 2. Add diagnostics reported for this line index *before* the code
            if i in diagnostics_by_line:
                # Add indentation for readability
                indented_diags = ["  " + diag for diag in diagnostics_by_line[i]]
                final_output_lines.extend(indented_diags)
                logger.debug(f"Inserted {len(indented_diags)} diagnostics before line index {i}")

            # 3. Add the original code line
            final_output_lines.append(code_line)

        annotated_lines = final_output_lines # Use the newly assembled list
        logger.info("Finished assembling final annotated output.")

        analysis_succeeded = True

    except ConnectionError as e:
        logger.error(f"LSP Connection Error during analysis: {e}")
        # Ensure final_output_lines has something to join if error happens early
        if not ('final_output_lines' in locals() and final_output_lines):
             final_output_lines = ['-- Error: LSP Connection Failed early in analysis']
        else: # Append error if analysis was partially done
             final_output_lines.append(f"-- Error: LSP Connection Failed: {e}")
        analysis_succeeded = False # Mark as failed
    except asyncio.TimeoutError as e:
        logger.error(f"LSP Overall Timeout Error during analysis: {e}")
        if not ('final_output_lines' in locals() and final_output_lines):
             final_output_lines = ['-- Error: LSP Timeout early in analysis']
        else:
             final_output_lines.append(f"-- Error: LSP Timeout: {e}")
        analysis_succeeded = False
    except Exception as e:
        logger.exception(f"Unhandled exception during LSP analysis: {e}")
        if not ('final_output_lines' in locals() and final_output_lines):
             final_output_lines = [f'-- Error: Unexpected analysis failure: {e}']
        else:
             final_output_lines.append(f"-- Error: Unexpected analysis failure: {e}")
        analysis_succeeded = False
    finally:
        # LSP Client shutdown (same as original code)
        if client and client.process: # Check if client was initialized
            logger.info("Shutting down LSP client...")
            try:
                await client.shutdown()
                await client.exit()
            except Exception as shutdown_e:
                logger.warning(f"Error during graceful shutdown/exit (forcing close): {shutdown_e}")
            finally:
                await client.close()
        else:
             logger.info("LSP client was not fully started or already closed.")


    # --- Append Build Error Section (same logic as original) ---
    # Ensure final_output_lines exists even if analysis failed very early
    if 'final_output_lines' not in locals():
         final_output_lines = ["-- Analysis failed before generating output --"]

    if analysis_succeeded:
        final_output_lines.append("\n-- Build System Output (lake build) --")
        if fallback_error and fallback_error != "LSP analysis failed.":
            fallback_lines = ["--   " + line for line in fallback_error.strip().splitlines()]
            final_output_lines.extend(fallback_lines)
        else:
            final_output_lines.append("-- (Build system output not provided or analysis failed internally)")
    else:
        logger.error("LSP analysis process failed or was incomplete.")
        # Ensure failure header is inserted correctly even if final_output_lines was created in except block
        if not final_output_lines or not final_output_lines[0].startswith("-- Error:"):
             final_output_lines.insert(0, "-- LSP Analysis Incomplete --")

        # Always append build output section on failure too
        final_output_lines.append("\n-- Original Build System Output (lake build) --")
        if fallback_error and fallback_error != "LSP analysis failed.":
            fallback_lines = ["--   " + line for line in fallback_error.strip().splitlines()]
            final_output_lines.extend(fallback_lines)
        else:
            final_output_lines.append("-- (Build system output not provided or analysis failed internally)")

    # --- Return Final Result ---
    return "\n".join(final_output_lines)