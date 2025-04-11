# src/lean_automator/lean/lsp_client.py

"""Provides an asynchronous client for interacting with the Lean Language Server.

This module defines the `LeanLspClient` class, responsible for managing a
`lean --server` subprocess, handling Language Server Protocol (LSP) communication
over stdio (JSON-RPC with Content-Length headers), managing request/response
correlation, and processing asynchronous notifications like diagnostics.
"""

import asyncio
import json
import logging
import os
import pathlib
import subprocess
from typing import Any, Dict, List, Optional

# --- Logging Configuration ---
# Each module should get its own logger instance.
# Basic configuration (level, format) should ideally be set globally
# in the application's entry point.
logger = logging.getLogger(__name__)

# --- Constants ---
CONTENT_LENGTH_HEADER = b"Content-Length: "
HEADER_SEPARATOR = b"\r\n\r\n"
DEFAULT_LSP_TIMEOUT = 30  # Seconds for typical LSP request/response


# --- Custom Exception ---


class LspResponseError(Exception):
    """Custom exception for LSP error responses.

    Attributes:
        code (Any): The error code from the LSP response. Defaults to "Unknown".
        message (str): The error message from the LSP response. Defaults to
            "Unknown error".
        data (Any): Optional additional data provided with the error.
    """

    def __init__(self, error_payload: Dict[str, Any]):
        """Initializes LspResponseError.

        Args:
            error_payload (Dict[str, Any]): The 'error' object from the LSP response.
        """
        self.code = error_payload.get("code", "Unknown")
        self.message = error_payload.get("message", "Unknown error")
        self.data = error_payload.get("data")
        super().__init__(f"LSP Error Code {self.code}: {self.message}")


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
        shared_lib_path (Optional[pathlib.Path]): Optional path to a shared
            dependency library root for LEAN_PATH construction.
        process (Optional[asyncio.subprocess.Process]): The subprocess object for
            `lean --server`. Initialized after `start_server()` is called.
        writer (Optional[asyncio.StreamWriter]): The stream writer connected to the
            subprocess's standard input. Initialized after `start_server()`.
        reader (Optional[asyncio.StreamReader]): The stream reader connected to the
            subprocess's standard output. Initialized after `start_server()`.
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
            timeout (int): The default timeout in seconds for waiting for LSP
                responses. Defaults to `DEFAULT_LSP_TIMEOUT`.
            shared_lib_path (Optional[pathlib.Path]): Path to the root of a shared
                library dependency, used to help construct LEAN_PATH for the server.
                Defaults to None.
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
        """Starts the lean --server subprocess and establishes communication.

        Initializes the stdin, stdout, and stderr pipes for interaction. Starts
        background tasks to read stdout (for LSP messages) and stderr (for logging
        server errors). Constructs an appropriate LEAN_PATH environment variable
        for the server, considering the standard library, an optional shared library
        path, and the current working directory's build path.

        Raises:
            FileNotFoundError: If the lean executable path specified during
                initialization is invalid or not found.
            ConnectionError: If the subprocess fails to start, or if the stdin/stdout
                streams cannot be established.
            Exception: For other potential errors during subprocess creation or
                environment setup.
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
            if self.shared_lib_path and self.shared_lib_path.is_dir():
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
            elif self.shared_lib_path:  # Log if path was provided but invalid
                logger.warning(
                    f"LSP Env: Provided shared_lib_path "
                    f"({self.shared_lib_path}) is not a valid directory."
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
                    logger.debug(
                        "LSP Env: Adding existing LEAN_PATH from environment: "
                        f"{existing_lean_path}"
                    )
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
                    # Resolve paths to handle potential symbolic links or case
                    # differences
                    resolved_p = str(pathlib.Path(p).resolve())
                    if resolved_p not in seen_paths:
                        unique_lean_paths.append(p)  # Append original path string
                        seen_paths.add(resolved_p)

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

            # Assign the streams directly provided by the subprocess
            self.reader = self.process.stdout
            self.writer = self.process.stdin

            if not self.reader or not self.writer:
                raise ConnectionError(
                    "Failed to get stdout/stdin streams from subprocess."
                )

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
        """Reads and logs stderr from the Lean server process asynchronously.

        Runs as a background task, reading line by line from the server's stderr
        stream and logging each line with a 'warning' level. Exits when the
        stream reaches EOF or the task is cancelled.
        """
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
            # Avoid logging errors during graceful shutdown or if process already exited
            if not self._closed and self.process and self.process.returncode is None:
                logger.error(f"Error reading lean server stderr: {e}")
        finally:
            logger.info("Stderr reader task finished.")

    def _get_loop(self):
        """Gets the current asyncio event loop.

        Attempts `asyncio.get_running_loop()`. If no loop is running (which
        might happen in certain contexts like thread executors), it logs a warning,
        creates a new event loop using `asyncio.new_event_loop()`, sets it as the
        current loop for the current OS thread using `asyncio.set_event_loop()`,
        and returns it.

        Returns:
            asyncio.AbstractEventLoop: The current or newly created event loop.
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
        for the JSON payload. Handles potential timeouts, end-of-file conditions,
        invalid header formats, and JSON decoding errors.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON message as a dictionary if
            successful. Returns None if reading failed due to EOF, timeout, invalid
            header, or JSON decoding errors.

        Raises:
            ConnectionError: If the connection closes unexpectedly during a read
                operation (e.g., `asyncio.IncompleteReadError`) or if other
                critical stream errors occur.
        """
        if not self.reader or self.reader.at_eof():
            logger.debug("LSP reader is None or at EOF.")
            return None

        header_str = ""
        json_body_str = ""
        try:
            # Read Content-Length header line by line until separator found
            header_lines_bytes = bytearray()
            while True:
                # Use a longer timeout for reading headers, as server might be slow
                # initially
                line_bytes = await asyncio.wait_for(
                    self.reader.readline(), timeout=self.timeout * 3
                )
                if not line_bytes:
                    raise asyncio.IncompleteReadError("EOF before headers end", None)
                header_lines_bytes.extend(line_bytes)
                if (
                    line_bytes == HEADER_SEPARATOR[:2]
                ):  # Check for \r\n signifies end of line
                    # Check if the last four bytes form the full separator
                    if header_lines_bytes.endswith(HEADER_SEPARATOR):
                        # Remove the final separator itself for parsing
                        header_lines_bytes = header_lines_bytes[
                            : -len(HEADER_SEPARATOR)
                        ]
                        break
                    elif header_lines_bytes.endswith(b"\n\n"):  # Tolerate \n\n
                        logger.warning(
                            "LSP message used non-standard \\n\\n header separator."
                        )
                        header_lines_bytes = header_lines_bytes[:-2]
                        break
                # Safety break if header gets excessively long
                if len(header_lines_bytes) > 4096:
                    raise ValueError("Excessively long LSP header received.")

            header_str = header_lines_bytes.decode("ascii")  # Headers must be ASCII
            content_length = -1
            for h_line in header_str.splitlines():
                if h_line.lower().startswith("content-length:"):
                    try:
                        content_length = int(h_line.split(":", 1)[1].strip())
                        break
                    except ValueError:
                        logger.error(f"Invalid Content-Length value: {h_line}")
                        return None  # Bad message format

            if content_length == -1:
                logger.error(
                    "Content-Length header not found in received headers: "
                    f"{header_str!r}"
                )
                return None  # Bad message format

            # Read JSON body
            json_body_bytes = await asyncio.wait_for(
                self.reader.readexactly(content_length), timeout=self.timeout
            )
            json_body_str = json_body_bytes.decode("utf-8")  # Body is UTF-8

            message = json.loads(json_body_str)
            # logger.debug(f"LSP Received Raw: {json_body_str}") # Very Verbose
            return message

        except asyncio.TimeoutError:
            logger.error(
                f"Timeout reading LSP message header or body (timeout={self.timeout}s)."
            )
            await self.close()  # Close connection on timeout
            return None
        except asyncio.IncompleteReadError as e:
            if not self._closed:  # Ignore if we initiated the close
                logger.error(
                    f"Server closed connection unexpectedly while reading: {e}"
                )
            await self.close()
            raise ConnectionError(
                "LSP connection closed unexpectedly."
            ) from e  # Raise to signal failure
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to decode JSON from server: {e}\n"
                f"Received Body: {json_body_str!r}"
            )
            return None  # Don't raise, just log and return None for this message
        except ValueError as e:  # Catch header parsing errors like long header
            logger.error(
                f"Error parsing LSP message header: {e}. "
                f"Header received: {header_str!r}"
            )
            await self.close()
            return None
        except Exception as e:
            if not self._closed:
                logger.exception(f"Error reading or processing LSP message: {e}")
            await self.close()  # Close on unexpected errors
            raise ConnectionError(
                f"Unexpected error reading LSP message: {e}"
            ) from e  # Raise

    async def _message_reader_loop(self) -> None:
        """Continuously reads and dispatches messages from the server's stdout.

        Runs as a background task initiated by `start_server`. Uses
        `_read_message` to get individual LSP messages. Based on the message
        structure (presence/absence of 'id' and 'method' fields), it handles:
        1.  Responses: Matches the 'id' to a pending request's Future in
            `_pending_requests`, sets the Future's result or exception
            (if an 'error' field is present).
        2.  Notifications: Checks if the 'method' is one we handle (currently
            `textDocument/publishDiagnostics` or `$/lean/fileProgress`). If so,
            puts the message onto the `_notifications_queue`. Logs and ignores
            other notification methods.
        3.  Server-to-Client Requests: Logs a warning and ignores requests
            initiated by the server (messages with both 'id' and 'method').
        4.  Unknown Messages: Logs a warning for messages that don't fit the above.

        The loop terminates if `_read_message` returns None (indicating EOF or
        unrecoverable read error), if a `ConnectionError` occurs during reading,
        if the task is cancelled, or if an unexpected exception occurs within the
        loop. Upon termination, it cancels any remaining pending request Futures.
        """
        logger.debug("Starting LSP message reader loop.")
        try:
            # Ensure process is running and we haven't initiated close
            while self.process and self.process.returncode is None and not self._closed:
                message = None  # Reset message for each loop iteration
                try:
                    message = await self._read_message()
                except ConnectionError:  # Raised by _read_message on critical errors
                    logger.warning("Connection error in reader loop. Exiting.")
                    break  # Exit the loop cleanly

                if message is None:
                    # Check if connection still seems valid or if we are closing
                    if (
                        not self._closed
                        and self.process
                        and self.process.returncode is None
                    ):
                        logger.warning(
                            "Message reader received None, but connection appears "
                            "active. Possible parse error or unexpected server "
                            "behavior."
                        )
                        # Optional: add a small delay before retrying?
                        # await asyncio.sleep(0.1)
                    else:
                        logger.info(
                            "Message reader received None, likely connection closed. "
                            "Exiting loop."
                        )
                        break  # Exit loop if read fails or connection closing

                # ----- Message Processing Logic -----
                # Avoid processing if message is None
                if message:
                    msg_id = message.get("id")
                    msg_method = message.get("method")
                    # Initialize request_id for potential use in error handling
                    request_id = None

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
                                            "Received error response for ID "
                                            f"{request_id}"
                                        )
                                        future.set_exception(
                                            LspResponseError(message["error"])
                                        )
                                    else:
                                        logger.debug(
                                            "Received result response for ID "
                                            f"{request_id}"
                                        )
                                        future.set_result(message.get("result"))
                                else:
                                    logger.warning(
                                        "Received response for already cancelled "
                                        f"request ID {request_id}"
                                    )
                            else:
                                # Could be response to request that already timed
                                # out/cancelled
                                logger.warning(
                                    f"Received response for ID {request_id}, but "
                                    "it was not pending."
                                )
                        except ValueError:
                            logger.warning(
                                f"Received response with non-integer ID '{msg_id}', "
                                "ignoring."
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error processing response for ID {msg_id}: {e}"
                            )
                            # Ensure future is removed if error occurs during processing
                            if (
                                request_id is not None
                                and request_id in self._pending_requests
                            ):
                                future = self._pending_requests.pop(request_id)
                                if not future.done():
                                    future.set_exception(e)

                    # --- Case 2: Notification from server ---
                    # Has 'method', does NOT have 'id'.
                    elif msg_method is not None and msg_id is None:
                        logger.debug(f"Received notification: {msg_method}")
                        # We only queue specific notifications we might care about
                        if msg_method in [
                            "textDocument/publishDiagnostics",
                            "$/lean/fileProgress",
                        ]:
                            q_size = self._notifications_queue.qsize() + 1
                            logger.debug(
                                f"Queueing notification '{msg_method}'. "
                                f"Queue size approx: {q_size}"
                            )
                            await self._notifications_queue.put(message)
                        else:
                            logger.debug(
                                f"Ignoring unhandled notification: {msg_method}"
                            )

                    # --- Case 3: Request from server ---
                    # Has 'method' AND 'id'. The server wants US to do something.
                    elif msg_method is not None and msg_id is not None:
                        logger.warning(
                            "Received request from server "
                            f"(Method: {msg_method}, ID: {msg_id}). Ignoring."
                        )
                        # A fully compliant client might send back MethodNotFound error.

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
        """Formats and writes a JSON-RPC message to the server's stdin.

        Encodes the message dictionary as JSON (UTF-8), prepends the required
        'Content-Length' header and separator (`\r\n\r\n`), writes the header
        and body to the process's stdin stream (`self.writer`), and drains the
        writer buffer.

        Args:
            message (Dict[str, Any]): The request or notification payload dictionary
                to be sent.

        Raises:
            ConnectionError: If the writer stream (`self.writer`) is unavailable
                (None or closing), or if a `ConnectionResetError` or
                `BrokenPipeError` occurs during writing, indicating a lost
                connection. Also raised for other unexpected exceptions during
                writing.
            Exception: For JSON serialization errors or other low-level write errors.
        """
        if not self.writer or self.writer.is_closing():
            raise ConnectionError("LSP writer is not available or closing.")

        try:
            json_body = json.dumps(message).encode("utf-8")
            header = (
                CONTENT_LENGTH_HEADER
                + str(len(json_body)).encode("ascii")
                + HEADER_SEPARATOR
            )

            # logger.debug(f"LSP Sending: {json.dumps(message, indent=2)}") # Verbose
            self.writer.write(header)
            self.writer.write(json_body)
            await self.writer.drain()
        except (ConnectionResetError, BrokenPipeError) as e:
            logger.error(f"Connection error writing LSP message: {e}")
            await self.close()  # Close connection on write failure
            raise ConnectionError(f"Connection error writing LSP message: {e}") from e
        except Exception as e:
            logger.exception(f"Error writing LSP message: {e}")
            await self.close()  # Close on other write errors too
            raise ConnectionError(f"Error writing LSP message: {e}") from e

    async def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Sends an LSP request and waits asynchronously for its response.

        Constructs a JSON-RPC request payload with a unique integer ID, the given
        method, and parameters. Sends the message using `_write_message`. Creates
        an `asyncio.Future` and stores it in `_pending_requests` keyed by the
        request ID. Waits for this Future to be completed by the
        `_message_reader_loop` when the corresponding response arrives.

        Args:
            method (str): The LSP method name (e.g., "initialize",
                "$/lean/plainGoal").
            params (Optional[Dict[str, Any]]): The parameters dictionary for the
                request. Defaults to an empty dictionary if None.

        Returns:
            Any: The 'result' field from the LSP response payload. Can be None if
                 the server explicitly returns null.

        Raises:
            ConnectionError: If the client is already closed, the server process
                is not running, or if a connection error occurs during writing the
                request or reading the response (handled by `_write_message` or
                `_message_reader_loop`).
            asyncio.TimeoutError: If no response is received from the server within
                the client's configured `timeout` period.
            LspResponseError: If the server returns an error response payload
                (indicated by the 'error' field in the response message).
            asyncio.CancelledError: If the waiting future is cancelled externally
                or by the client closing down before the response arrives.
            Exception: For other unexpected errors during the process, such as
                JSON encoding issues or internal asyncio errors.
        """
        if self._closed:
            raise ConnectionError("Client is closed.")
        if not self.process or self.process.returncode is not None:
            raise ConnectionError("Lean server process is not running.")

        request_id = self._message_id_counter
        self._message_id_counter += 1
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params if params is not None else {},  # Ensure params is dict
        }
        future: asyncio.Future = self._get_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            logger.debug(f"Sending request {request_id}: {method}")
            await self._write_message(request)
            # Wait for the future associated with this request ID
            return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout waiting for response to request {request_id} ({method})."
            )
            # Clean up pending request if timed out
            pending_future = self._pending_requests.pop(request_id, None)
            if pending_future and not pending_future.done():
                pending_future.cancel(
                    f"Timeout waiting for response to request {request_id}"
                )
            raise  # Re-raise timeout
        except asyncio.CancelledError:
            logger.warning(f"Request {request_id} ({method}) was cancelled.")
            # Ensure cleanup if cancelled externally
            self._pending_requests.pop(
                request_id, None
            )  # Future might already be removed by reader loop
            raise  # Re-raise cancellation
        except LspResponseError as e:  # If future gets an error set by reader loop
            logger.warning(
                f"Request {request_id} ({method}) failed with LSP Error: {e}"
            )
            # No need to cleanup future here, reader loop already did
            raise  # Re-raise LspResponseError
        except ConnectionError as e:  # Connection error during write or read
            logger.error(
                f"Connection error during request {request_id} ({method}): {e}"
            )
            self._pending_requests.pop(request_id, None)  # Clean up future
            raise  # Re-raise ConnectionError
        except Exception as e:  # Other unexpected errors
            logger.exception(
                f"Unexpected error sending request {request_id} ({method}) "
                f"or awaiting response: {e}"
            )
            # Clean up future on other errors too
            pending_future = self._pending_requests.pop(request_id, None)
            if pending_future and not pending_future.done():
                # If future wasn't cancelled/set_exception'd by reader loop yet
                pending_future.set_exception(e)
            raise  # Re-raise original error

    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Sends an LSP notification (fire-and-forget).

        Constructs a JSON-RPC notification payload (without an 'id' field) and
        sends it using `_write_message`. Does not wait for any acknowledgment.

        Args:
            method (str): The LSP notification method name (e.g., "initialized",
                "textDocument/didOpen").
            params (Optional[Dict[str, Any]]): The parameters dictionary for the
                notification. Defaults to an empty dictionary if None.

        Raises:
            ConnectionError: If the client is already closed, the server process
                is not running, or if a connection error occurs during writing
                (handled by `_write_message`).
            Exception: For other unexpected errors, such as JSON encoding issues.
        """
        if self._closed:
            raise ConnectionError("Client is closed.")
        if not self.process or self.process.returncode is not None:
            raise ConnectionError("Lean server process is not running.")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params if params is not None else {},  # Ensure params is dict
        }
        logger.debug(f"Sending notification: {method}")
        await self._write_message(notification)

    async def initialize(self) -> Dict[str, Any]:
        """Performs the LSP initialization handshake.

        Sends the `initialize` request with client capabilities (including basic
        text document synchronization and workspace folder support) and project
        details (root URI, process ID). Waits for the `InitializeResult` response
        from the server. If successful, sends the `initialized` notification.

        Returns:
            Dict[str, Any]: The server capabilities dictionary returned in the
                'capabilities' field of the `InitializeResult` response. Returns
                an empty dictionary if the server response has no result field.

        Raises:
            ConnectionError: If the handshake fails due to connection issues,
                timeouts (`asyncio.TimeoutError`), server errors during
                initialization (`LspResponseError`), or other exceptions during
                request/notification sending. The client connection will likely
                be closed if this occurs.
        """
        logger.info("Sending LSP initialize request.")
        # Basic capabilities sufficient for diagnostics and goal querying
        init_params = {
            "processId": os.getpid(),
            "rootUri": pathlib.Path(self.cwd).as_uri(),
            "capabilities": {
                "textDocument": {
                    # Advertise basic sync kind
                    "synchronization": {
                        "dynamicRegistration": False,
                        "willSave": False,
                        "willSaveWaitUntil": False,
                        "didSave": True,  # We will send didOpen/didChange/didSave
                        "didOpen": True,
                    },
                    "publishDiagnostics": {"relatedInformation": True},
                    # If we need hover, completion etc., add capabilities here
                },
                "workspace": {
                    "workspaceFolders": True  # Indicate support for workspace folders
                },
            },
            "trace": "off",  # Set to "verbose" for more LSP logging if needed
            "workspaceFolders": [
                {
                    "uri": pathlib.Path(self.cwd).as_uri(),
                    "name": os.path.basename(self.cwd),
                }
            ],
            # Add lean specific initialization options if required by the server version
            # "initializationOptions": { ... }
        }
        try:
            response = await self.send_request("initialize", init_params)
            logger.info(
                "Received initialize response. Sending initialized notification."
            )
            await self.send_notification("initialized", {})
            logger.info("LSP Handshake Complete.")
            # Extract server capabilities, default to empty dict if 'result' is null
            # or missing
            server_capabilities = response.get("capabilities", {}) if response else {}
            return server_capabilities
        except Exception as e:
            logger.exception(f"LSP Initialization failed: {e}")
            await self.close()  # Ensure cleanup on failure
            raise ConnectionError(f"LSP Initialization failed: {e}") from e

    async def did_open(
        self, file_uri: str, language_id: str, version: int, text: str
    ) -> None:
        """Sends a `textDocument/didOpen` notification to the server.

        Informs the server that a document has been opened in the client.

        Args:
            file_uri (str): The URI of the document being opened (e.g.,
                'file:///path/to/your/file.lean').
            language_id (str): The language identifier for the document (typically
                "lean").
            version (int): The initial version number of the document in the client
                (usually 1).
            text (str): The full text content of the document as it was opened.

        Raises:
            ConnectionError: If the client is closed, the server process is not
                running, or writing the notification fails.
            Exception: For other unexpected errors (e.g., JSON encoding).
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
        """Sends a request to get the Lean proof goal state at a specific position.

        Uses the custom Lean LSP method `$/lean/plainGoal`.

        Args:
            file_uri (str): The URI of the document (e.g.,
                'file:///path/to/your/file.lean').
            line (int): The zero-based line number for the position within the document.
            character (int): The zero-based character offset (UTF-16 code units)
                within the line.

        Returns:
            Optional[Dict[str, Any]]: The result payload dictionary from the server
            if the request is successful. This typically contains goal state
            information like 'rendered' or 'plainGoal'. Returns `None` if the
            request fails due to timeout, server error (`LspResponseError`),
            connection error, or other unexpected exceptions. Also returns `None`
            if the server explicitly returns a null result.
        """
        method = "$/lean/plainGoal"
        logger.debug(f"Sending {method} request for {file_uri} at {line}:{character}")
        params = {
            "textDocument": {"uri": file_uri},
            "position": {"line": line, "character": character},
        }
        try:
            result = await self.send_request(method, params)
            # logger.debug(f"Received result for get_goal({line}:{character}): "
            #              f"{result!r}")
            return result  # Result can be None if server sends null result
        except LspResponseError as e:
            logger.warning(f"{method} request failed for {line}:{character}: {e}")
            return None  # Return None on LSP error responses
        except asyncio.TimeoutError:
            logger.error(f"{method} request timed out for {line}:{character}")
            return None  # Return None on timeout
        except ConnectionError:
            logger.error(
                f"Connection error during {method} request for {line}:{character}."
            )
            return None  # Return None if connection failed
        except Exception as e:  # Catch other unexpected errors
            logger.error(
                f"Unexpected error sending {method} request for {line}:{character}: {e}"
            )
            return None  # Return None on other errors

    async def shutdown(self) -> None:
        """Sends the `shutdown` request to the server.

        Politely asks the server to prepare for termination. According to the LSP
        specification, the client should wait for the response to `shutdown` before
        sending the `exit` notification. This method sends the request and waits
        for the (usually null) response. Errors during this process (like timeouts
        or connection errors if the server exits prematurely) are logged as warnings,
        as the client will proceed to `exit` and close regardless.
        """
        if self._closed or not self.process or self.process.returncode is not None:
            return
        logger.info("Sending LSP shutdown request.")
        try:
            # Shutdown expects a response (null) according to LSP spec
            await self.send_request("shutdown")
            logger.info("LSP shutdown request acknowledged by server.")
        except (ConnectionError, asyncio.TimeoutError, LspResponseError) as e:
            # Log error but proceed to exit/close anyway
            logger.warning(
                f"Error during LSP shutdown request (proceeding to exit/close): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during shutdown request: {e}")

    async def exit(self) -> None:
        """Sends the `exit` notification to the server.

        Informs the server that it should terminate immediately. This is a
        notification, so no response is expected. It should typically be called
        after the `shutdown` request has been sent (and potentially acknowledged
        or failed). Errors during sending (like `ConnectionError` if the server
        already terminated) are logged as warnings.
        """
        if self._closed or not self.process or self.process.returncode is not None:
            # Avoid sending exit if server already stopped or client closing
            if not self._closed:
                logger.debug(
                    "Skipping exit notification as server process is not running."
                )
            return

        logger.info("Sending LSP exit notification.")
        try:
            # Exit is a notification, no response expected
            await self.send_notification("exit")
        except ConnectionError as e:
            # Server might have already exited after shutdown acknowledgement,
            # this is okay
            logger.warning(
                "Connection error during LSP exit notification "
                f"(may be expected if server stopped quickly): {e}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error during exit notification: {e}")

    async def close(self) -> None:
        """Closes the LSP client connection and terminates the server process
        gracefully.

        This method performs the following steps:
        1. Marks the client as closed (`self._closed = True`).
        2. If the server process appears to be running, attempts a graceful shutdown
           by sending the `shutdown` request (with a short timeout).
        3. Sends the `exit` notification (regardless of `shutdown` success).
        4. Cancels the background message reader (`_reader_task`) and stderr reader
           (`_stderr_task`) tasks.
        5. Closes the stdin writer stream (`self.writer`).
        6. If the server process is still running, attempts to terminate it using
           `process.terminate()`. Waits briefly for termination.
        7. If termination times out, forcefully kills the process using
           `process.kill()`.
        8. Cleans up any remaining pending request Futures by cancelling them.
        9. Waits for the cancelled reader and stderr tasks to complete.
        10. Sets internal state variables (`process`, `writer`, `reader`, tasks)
            to None.

        This method is idempotent; calling it multiple times will have no further effect
        after the first call.
        """
        if self._closed:
            return
        self._closed = True  # Mark as closed immediately to prevent race conditions
        logger.info("Closing LSP client connection and terminating server.")

        # 1. Graceful shutdown sequence (if process still seems alive)
        if self.process and self.process.returncode is None:
            try:
                # Try shutdown first (with a shorter timeout than default request)
                await asyncio.wait_for(self.shutdown(), timeout=5.0)
            except (
                ConnectionError,
                asyncio.TimeoutError,
                LspResponseError,
                Exception,
            ) as shutdown_e:
                logger.warning(
                    f"Graceful shutdown failed or timed out ({shutdown_e}), proceeding."
                )

            try:
                # Send exit regardless of shutdown success
                await self.exit()
                # Give a brief moment for exit to potentially be processed
                await asyncio.sleep(0.1)
            except (ConnectionError, Exception) as exit_e:
                logger.warning(
                    f"Error sending exit notification during close: {exit_e}"
                )

        # 2. Cancel reader/stderr tasks
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel("Client closing")
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel("Client closing")

        # 3. Close streams (helps tasks terminate if blocked on read/write)
        if self.writer and not self.writer.is_closing():
            try:
                self.writer.close()
                # Don't wait indefinitely here, process termination handles it
                # await self.writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error closing LSP writer: {e}")
        self.writer = None
        # Reader is associated with the process stdout pipe, closing pipe handles it
        self.reader = None

        # 4. Terminate process if still running after graceful attempts
        proc = self.process  # Capture process locally
        self.process = None  # Nullify instance variable
        if proc and proc.returncode is None:
            logger.info("Terminating lean server process...")
            try:
                proc.terminate()
                # Wait briefly for termination, then kill if necessary
                return_code = await asyncio.wait_for(proc.wait(), timeout=5.0)
                logger.info(f"Lean server process terminated with code {return_code}.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Lean server process did not terminate gracefully after 5s, "
                    "killing."
                )
                try:
                    proc.kill()
                    await proc.wait()  # Wait for kill completion
                    logger.info("Lean server process killed.")
                except ProcessLookupError:
                    logger.warning("Process already killed or finished.")
                except Exception as kill_e:
                    logger.error(f"Error killing lean process: {kill_e}")
            except ProcessLookupError:
                logger.warning(
                    "Process already finished before final termination attempt."
                )
            except Exception as e:
                logger.error(f"Error during lean server process termination: {e}")

        # 5. Clear any remaining pending requests
        for req_id, future in list(self._pending_requests.items()):
            if not future.done():
                future.cancel(f"LSP Client closed while request {req_id} was pending.")
            self._pending_requests.pop(req_id, None)

        # 6. Wait for tasks to finish cancellation
        try:
            tasks_to_wait = []
            if self._reader_task:
                tasks_to_wait.append(self._reader_task)
            if self._stderr_task:
                tasks_to_wait.append(self._stderr_task)

            if tasks_to_wait:
                await asyncio.gather(*tasks_to_wait, return_exceptions=True)
                logger.debug("Reader and stderr tasks finished after cancellation.")
        except Exception as e:
            logger.warning(f"Exception during task cleanup: {e}")

        self._reader_task = None
        self._stderr_task = None

        logger.info("LSP client closed.")

    async def get_diagnostics(
        self, timeout: Optional[float] = 0.1
    ) -> List[Dict[str, Any]]:
        """Retrieves pending diagnostic notifications from the internal queue.

        Checks the `_notifications_queue` for messages with the method
        `textDocument/publishDiagnostics`. It attempts to drain the queue by
        repeatedly calling `queue.get()` with short timeouts until the queue
        is empty (`asyncio.TimeoutError` is caught).

        For each `publishDiagnostics` notification found, it extracts the list of
        diagnostic objects from the `params.diagnostics` field and appends these
        individual diagnostic objects to the result list.

        Args:
            timeout (Optional[float]): The initial timeout in seconds to wait for the
                first notification if the queue might be empty. Subsequent attempts
                to drain the queue use a very short timeout (0.01s). Defaults to 0.1s.

        Returns:
            List[Dict[str, Any]]: A list containing all individual diagnostic objects
            extracted from the `textDocument/publishDiagnostics` notifications found
            in the queue. Each element in the list is a single diagnostic dictionary
            (e.g., containing 'range', 'severity', 'message', etc.). Returns an
            empty list if no relevant notifications are found in the queue.
        """
        diagnostics_params = []
        first_attempt = True
        drain_timeout = 0.01  # Use a very small timeout for subsequent checks

        logger.debug(
            "Entering get_diagnostics. Approx queue size: "
            f"{self._notifications_queue.qsize()}"
        )

        while True:
            current_timeout = timeout if first_attempt else drain_timeout
            first_attempt = False  # Use drain_timeout after the first attempt

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
                    params = notification.get("params", {})
                    # Extract the list of diagnostic items
                    diags_list = params.get("diagnostics", [])
                    # We return the list of individual diagnostic objects
                    if diags_list:  # Only add if there are actual diagnostics
                        uri = params.get("uri")
                        logger.debug(
                            f"Extending with {len(diags_list)} diagnostics from "
                            f"notification for URI {uri}."
                        )
                        diagnostics_params.extend(diags_list)
                    else:
                        uri = params.get("uri")
                        logger.debug(
                            "Notification contained empty diagnostics list "
                            f"for URI {uri}."
                        )

                # Ignore other notification types like $/lean/fileProgress here
                # else: logger.debug("Ignoring notification type: "
                #                    f"{notification.get('method')}")

                self._notifications_queue.task_done()

            except asyncio.TimeoutError:
                # This is the expected way to exit when the queue is empty
                logger.debug(
                    f"Queue appears empty (timeout={current_timeout}s). "
                    "Exiting diagnostic collection."
                )
                break
            except (
                asyncio.QueueEmpty
            ):  # Should be caught by TimeoutError, but just in case
                logger.debug("Queue empty exception. Exiting diagnostic collection.")
                break
            except Exception as e:
                logger.error(
                    f"Error getting diagnostics from queue: {e}", exc_info=True
                )
                break  # Exit loop on unexpected errors

        logger.debug(
            "Exiting get_diagnostics. Total collected diagnostic items: "
            f"{len(diagnostics_params)}"
        )
        # Return the list of individual diagnostic objects
        return diagnostics_params
