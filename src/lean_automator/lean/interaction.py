# File: lean_automator/lean/interaction.py

"""Provides functions and classes to interact with the Lean prover via Lake.

This module handles the validation of Lean code associated with Knowledge Base
items (`KBItem`). It defines mechanisms to:

- Directly write Lean code for a KBItem into the configured shared library.
- Execute `lake build` commands asynchronously within the shared library context.
- Keep the code file in the shared library upon successful verification.
- Remove the code file from the shared library upon verification failure.
- Manage database status updates based on validation outcomes.
- On validation failure, use lean_lsp_analyzer to generate detailed error logs
  within the shared library context.
"""

import asyncio
import functools
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile # Keep for TemporaryDirectory potentially in LSP? Though likely not needed now.
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Use absolute imports assume src is in path or package installed
try:
    from lean_automator.config.loader import (
        APP_CONFIG,
        get_lean_automator_shared_lib_path,
    )
except ImportError:
    warnings.warn(
        "config_loader not found. Configuration system unavailable.", ImportWarning
    )
    APP_CONFIG = {}
    def get_lean_automator_shared_lib_path() -> Optional[str]:
        return os.getenv("LEAN_AUTOMATOR_SHARED_LIB_PATH")


try:
    from lean_automator.kb.storage import (
        DEFAULT_DB_PATH,
        ItemStatus,
        ItemType,
        KBItem,
        get_kb_item_by_name,
        save_kb_item,
    )
except ImportError:
    print("Error: Failed to import kb_storage.", file=sys.stderr)
    # Define dummy types/functions
    KBItem = type("KBItem", (object,), {"lean_code": "", "unique_name": "DummyItem"}) # type: ignore
    ItemStatus = type("ItemStatus", (object,), {"ERROR": "ERROR", "PROVEN": "PROVEN", "LEAN_VALIDATION_FAILED": "LEAN_VALIDATION_FAILED"}) # type: ignore
    ItemType = type("ItemType", (object,), {}) # type: ignore
    def get_kb_item_by_name(*args: Any, **kwargs: Any) -> None: return None # type: ignore
    async def save_kb_item(*args: Any, **kwargs: Any) -> None: pass # type: ignore
    DEFAULT_DB_PATH = "knowledge_base.sqlite"


# --- Import the new LSP analyzer ---
try:
    from lean_automator.lean.lsp_analyzer import analyze_lean_failure
except ImportError:
    print("Error: Failed to import lean_lsp_analyzer.", file=sys.stderr)
    async def analyze_lean_failure(*args: Any, **kwargs: Any) -> str: # type: ignore
        logger.error("lean_lsp_analyzer.analyze_lean_failure is unavailable!")
        fallback = kwargs.get("fallback_error", "LSP analyzer unavailable.")
        return f"-- LSP Analysis Skipped (Import Error) --\n{fallback}"


# --- Logging Configuration ---
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --- Helper Functions (Remain outside class) ---

def _module_name_to_path(module_name: str) -> pathlib.Path:
    """Converts a Lean module name (dot-separated) to a relative Path object.

    Args:
        module_name: The dot-separated Lean module name.

    Returns:
        The corresponding relative path.

    Raises:
        ValueError: If the module name format is invalid.
    """
    # Allow dots in the final component (e.g., file name can have dots)
    # but split directories by dots. Assume library name is part of the path.
    # Example: VantageLib.Test.Module -> VantageLib/Test/Module
    safe_module_name = module_name.replace("\\", "/")
    parts = safe_module_name.split('.')
    valid_parts = [part for part in parts if part]

    if not valid_parts:
        raise ValueError(f"Invalid module name format: '{module_name}'")

    # Reconstruct path, ensuring the final part keeps internal dots if any (unlikely for modules)
    # For simplicity here, assume standard module naming conventions.
    return pathlib.Path(*valid_parts)


def _write_text_sync(p: pathlib.Path, c: str):
    """Synchronous helper to write text to a file.

    Args:
        p: The path to the file.
        c: The content string to write.
    """
    p.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    p.write_text(c, encoding="utf-8")


# --- Core Verification Class ---

class LeanVerifier:
    """Manages the direct verification and integration of Lean code into a shared library.

    This class encapsulates the logic for writing Lean code directly into the
    shared library structure, running Lake builds within that library's context,
    analyzing failures, and managing database status updates for KBItems.
    Successfully verified files are left in the shared library; failed files are removed.

    Attributes:
        db_path: Path to the knowledge base database.
        lake_executable_path: Path to the `lake` executable.
        timeout_seconds: Timeout for Lake subprocess executions.
        shared_lib_path: Path to the persistent shared Lean library root.
        shared_lib_package_name: Name of the package in the shared library's lakefile.
        shared_lib_src_dir_name: Name of the source directory within the shared library.
        loop: The asyncio event loop.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        lake_executable_path: str = "lake",
        timeout_seconds: int = 120,
        shared_lib_path_str: Optional[str] = None,
        shared_lib_package_name: Optional[str] = None,
        shared_lib_src_dir_name: Optional[str] = None,
    ):
        """Initializes the LeanVerifier instance.

        Args:
            db_path: Path to the knowledge base database file.
            lake_executable_path: Path to the `lake` command-line tool.
            timeout_seconds: Timeout for Lake subprocess executions.
            shared_lib_path_str: Path string for the persistent shared library.
            shared_lib_package_name: Name of the shared library package.
            shared_lib_src_dir_name: Name of the shared library source directory.

        Raises:
            ValueError: If the shared library path is required but not configured
                or invalid.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.lake_executable_path = lake_executable_path
        self.timeout_seconds = timeout_seconds
        self.loop = asyncio.get_running_loop()

        # Load Configuration
        if shared_lib_path_str is None:
            shared_lib_path_str = get_lean_automator_shared_lib_path()

        lean_paths_config = APP_CONFIG.get("lean_paths", {})
        self.shared_lib_package_name = shared_lib_package_name or lean_paths_config.get(
            "shared_lib_package_name", "vantage_lib"
        )
        # Ensure src_dir_name aligns with module structure (e.g., "VantageLib")
        self.shared_lib_src_dir_name = shared_lib_src_dir_name or lean_paths_config.get(
            "shared_lib_src_dir_name", "VantageLib" # This should be the root source folder name
        )

        logger.info(f"Using shared library package name: {self.shared_lib_package_name}")
        logger.info(f"Using shared library source directory name: {self.shared_lib_src_dir_name}")

        # Validate Shared Library Path
        self.shared_lib_path = self._resolve_shared_lib_path(shared_lib_path_str)
        if not self.shared_lib_path:
            raise ValueError(
                "Shared library path (LEAN_AUTOMATOR_SHARED_LIB_PATH) is not configured "
                "correctly or the directory does not exist. Cannot proceed."
            )
        logger.info(f"Using validated shared library path: {self.shared_lib_path}")

        # Internal state/cache
        self._lean_executable_path: Optional[str] = None
        self._subprocess_env: Optional[Dict[str, str]] = None

    def _resolve_shared_lib_path(self, path_str: Optional[str]) -> Optional[pathlib.Path]:
        """Resolves and validates the shared library path string."""
        if not path_str:
            logger.error( "Shared library path not configured.")
            return None
        try:
            resolved_path = pathlib.Path(path_str).resolve()
            if resolved_path.is_dir():
                if not (
                    (resolved_path / "lakefile.lean").is_file()
                    or (resolved_path / "lakefile.toml").is_file()
                ):
                    logger.warning(f"Shared library path {resolved_path} lacks a lakefile.")
                return resolved_path
            else:
                logger.error(f"Resolved path '{resolved_path}' is not a directory.")
                return None
        except Exception as e:
            logger.error(f"Error resolving shared library path '{path_str}': {e}")
            return None

    async def _get_item(self, unique_name: str) -> Optional[KBItem]:
        """Fetches a KBItem from the database."""
        try:
            # Ensure KBItem type is correctly defined or imported
            item: Optional[KBItem] = get_kb_item_by_name(unique_name, db_path=self.db_path)
            if item is None:
                logger.error(f"Target item '{unique_name}' not found.")
            return item
        except Exception as e:
            logger.exception(f"Error fetching item {unique_name}: {e}")
            return None

    async def _update_item_status(
        self, item: KBItem, status: ItemStatus, error_log: Optional[str] = None, increment_failure: bool = False
    ):
        """Updates the status and optionally error log/failure count of a KBItem."""
        if not item: return
        try:
            # Ensure update_status and increment_failure_count methods exist on KBItem
            item.update_status(status, error_log=error_log)
            if increment_failure:
                item.increment_failure_count()
            await save_kb_item(item, client=None, db_path=self.db_path)
            logger.info(f"DB status updated to {status.name if hasattr(status, 'name') else status} for {item.unique_name}.")
        except Exception as e:
            logger.error(f"Failed to update DB status for {item.unique_name}: {e}", exc_info=True)


    def _generate_imports_for_target(self, item: KBItem) -> str:
        """Generates Lean import statements based on plan_dependencies."""
        if not item or not hasattr(item, 'plan_dependencies') or not item.plan_dependencies:
            return ""

        import_lines: List[str] = []
        for dep_name in item.plan_dependencies:
            if dep_name != item.unique_name:
                # Assume dependency names are fully qualified relative to the source dir
                # Example: if src_dir is VantageLib, dep_name "VantageLib.Utils.Helper" is used directly
                # Example: if dep_name is "Utils.Helper", it becomes "import VantageLib.Utils.Helper"
                if not dep_name.startswith(self.shared_lib_src_dir_name + "."):
                     full_import_name = f"{self.shared_lib_src_dir_name}.{dep_name}"
                else:
                     full_import_name = dep_name # Assume already fully qualified including src dir name
                import_lines.append(f"import {full_import_name}")

        if not import_lines: return ""
        sorted_imports = sorted(list(set(import_lines)))
        return "-- Auto-generated imports based on plan_dependencies --\n" + "\n".join(sorted_imports)


    async def _prepare_subprocess_env(self) -> Dict[str, str]:
        """Prepares the environment dictionary for Lake subprocesses (cached)."""
        if self._subprocess_env is not None:
             return self._subprocess_env

        # (Implementation is the same as before - detect lean, stdlib, set LEAN_PATH, LAKE_HOME)
        # ... (kept identical to previous refactored version for brevity) ...
        subprocess_env = os.environ.copy()
        std_lib_path = None
        lean_exe = "lean" # Default

        # --- Detect Lean Executable ---
        if self._lean_executable_path is None:
            try:
                lake_path_obj = pathlib.Path(self.lake_executable_path).resolve()
                lean_exe_candidate = lake_path_obj.parent / "lean"
                if await self.loop.run_in_executor(None, lean_exe_candidate.is_file):
                    lean_exe = str(lean_exe_candidate)
                    logger.info(f"Detected 'lean' executable relative to lake: {lean_exe}")
                else:
                    lean_exe_from_path = await self.loop.run_in_executor( None, shutil.which, "lean" )
                    if lean_exe_from_path:
                        lean_exe = lean_exe_from_path
                        logger.info(f"Using 'lean' executable found in PATH: {lean_exe}")
                    else:
                        logger.warning("Could not find 'lean' relative to lake or in PATH. Assuming default 'lean'.")
            except Exception as path_err:
                logger.warning(f"Error resolving lean executable path from lake path: {path_err}. Assuming default 'lean'.")
            self._lean_executable_path = lean_exe
        else:
             lean_exe = self._lean_executable_path

        # --- Detect Stdlib Path ---
        if not await self.loop.run_in_executor(None, shutil.which, lean_exe):
             logger.warning(f"Lean executable '{lean_exe}' not found in PATH. Cannot detect stdlib path via --print-libdir.")
        else:
            try:
                logger.debug(f"Attempting stdlib detection using: {lean_exe} --print-libdir")
                lean_path_proc = await self.loop.run_in_executor(
                    None,
                    functools.partial(
                        subprocess.run,
                        [lean_exe, "--print-libdir"],
                        capture_output=True, text=True, check=True, timeout=10, encoding="utf-8"
                    )
                )
                path_candidate = lean_path_proc.stdout.strip()
                if path_candidate and await self.loop.run_in_executor(None, pathlib.Path(path_candidate).is_dir):
                    std_lib_path = path_candidate
                    logger.info(f"Detected Lean stdlib path: {std_lib_path}")
                else:
                    logger.warning(f"Command '{lean_exe} --print-libdir' did not return valid directory: '{path_candidate}'")
            except Exception as e:
                logger.warning(f"Failed to detect Lean stdlib path via '{lean_exe} --print-libdir': {e}")

        # Fallback detection
        if not std_lib_path:
             try:
                 lake_path_obj = pathlib.Path(self.lake_executable_path).resolve()
                 toolchain_dir = lake_path_obj.parent.parent
                 fallback_path = toolchain_dir / "lib" / "lean"
                 if await self.loop.run_in_executor(None, fallback_path.is_dir):
                     std_lib_path = str(fallback_path.resolve())
                     logger.warning(f"Assuming stdlib path relative to toolchain: {std_lib_path}")
             except Exception: pass

        # --- Set LEAN_PATH ---
        if std_lib_path:
            existing_lean_path = subprocess_env.get("LEAN_PATH")
            separator = os.pathsep
            subprocess_env["LEAN_PATH"] = f"{std_lib_path}{separator}{existing_lean_path}" if existing_lean_path else std_lib_path
            logger.debug(f"Setting LEAN_PATH for subprocess: {subprocess_env['LEAN_PATH']}")
        else:
            logger.warning("Stdlib path not found. LEAN_PATH will not include it.")

        # --- Set LAKE_HOME ---
        persistent_cache_path = os.getenv("LEAN_AUTOMATOR_LAKE_CACHE")
        if persistent_cache_path:
            persistent_cache_path_obj = pathlib.Path(persistent_cache_path).resolve()
            try:
                await self.loop.run_in_executor(None, functools.partial(os.makedirs, persistent_cache_path_obj, exist_ok=True))
                subprocess_env["LAKE_HOME"] = str(persistent_cache_path_obj)
                logger.info(f"Setting LAKE_HOME for subprocess: {persistent_cache_path_obj}")
            except OSError as e:
                logger.error(f"Failed to create/use persistent Lake cache directory '{persistent_cache_path_obj}': {e}")
        else:
            logger.debug("LEAN_AUTOMATOR_LAKE_CACHE not set. Lake will use default caching.")

        self._subprocess_env = subprocess_env
        return subprocess_env


    async def _run_lake_build(
        self, build_target: str, cwd: pathlib.Path, env: Dict[str, str], description: str = "Lake build"
    ) -> subprocess.CompletedProcess:
        """Runs a `lake build` command asynchronously. (Identical to previous)"""
        command = [self.lake_executable_path, "build", build_target]
        logger.info(f"Running {description}: {' '.join(command)} in {cwd}")
        try:
            run_args = {
                "args": command, "capture_output": True, "text": True,
                "timeout": self.timeout_seconds, "encoding": "utf-8", "errors": "replace",
                "cwd": str(cwd), "check": False, "env": env,
            }
            result = await self.loop.run_in_executor( None, functools.partial(subprocess.run, **run_args) )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            logger.debug(f"{description} return code: {result.returncode}")
            if stdout: logger.debug(f"{description} stdout:\n{stdout}")
            if stderr: logger.debug(f"{description} stderr:\n{stderr}")
            return result
        except subprocess.TimeoutExpired as e:
            logger.error(f"{description} timed out after {self.timeout_seconds}s for '{build_target}'.")
            raise e
        except FileNotFoundError as e:
            logger.error(f"Lake executable not found at '{self.lake_executable_path}' during {description}.")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error running {description} for '{build_target}': {e}")
            raise RuntimeError(f"Subprocess execution error during {description}: {e}") from e


    async def _analyze_failure_direct(
        self, item: KBItem, build_output: str, failed_file_path: pathlib.Path
        # Note: failed_file_path argument might be redundant now, but kept for context if needed later
    ) -> str:
        """Analyzes a Lean build failure using LSP within the shared library context.

        Args:
            item: The KBItem that failed verification.
            build_output: The stderr/stdout from the failed lake build command.
            failed_file_path: The path where the .lean file was written before failing.
                              (Used mainly for logging context here).

        Returns:
            A string containing the detailed error log.
        """
        logger.warning(f"Direct verification failed for {item.unique_name} at {failed_file_path}. Starting LSP analysis.")
        final_error_log = f"Lake build failed.\n{build_output}" # Default log

        try:
            # Ensure item has lean_code attribute
            current_lean_code = getattr(item, "lean_code", None)
            if not current_lean_code:
                 logger.warning(f"Skipping LSP analysis for {item.unique_name}: Lean code missing.")
                 return final_error_log

            # Ensure lean_exe path is available
            if self._lean_executable_path is None:
                 await self._prepare_subprocess_env() # Ensure lean_exe is detected
            lean_exe = self._lean_executable_path or "lean"

            # The LSP analyzer likely creates its own temp file within the CWD.
            # It needs the CWD to be the project root for context.
            logger.info(f"Running LSP analysis for {item.unique_name} code within shared lib context ({self.shared_lib_path})")

            # Call analyze_lean_failure with arguments matching its documented signature
            lsp_analysis_result = await analyze_lean_failure(
                lean_code=str(current_lean_code),
                lean_executable_path=lean_exe,
                cwd=str(self.shared_lib_path),        # CWD is the project root
                shared_lib_path=self.shared_lib_path, # For LEAN_PATH setup inside analyzer
                timeout_seconds=self.timeout_seconds,
                fallback_error=build_output,
            )

            logger.info(f"LSP analysis completed for {item.unique_name}.")
            return lsp_analysis_result

        except Exception as lsp_err:
            logger.error(f"LSP analysis execution failed for {item.unique_name}: {lsp_err}", exc_info=True)
            return f"{final_error_log}\n\n--- LSP Analysis Failed: {lsp_err} ---"

    async def verify_and_integrate_item(self, unique_name: str) -> Tuple[bool, str]:
        """Verifies Lean code by writing it directly into the shared library and building.

        Writes the item's code to the calculated path within the shared library's
        source structure. Runs `lake build` on that target within the library root.
        If successful, the file is kept and status updated to PROVEN.
        If failed, LSP analysis is run, the file is removed, and status updated.

        Args:
            unique_name: The unique name of the KBItem to check and compile.

        Returns:
            A tuple containing:
                - bool: True if verification succeeded and file was integrated, False otherwise.
                - str: A message describing the outcome.
        """
        logger.info(f"Starting direct Lean validation for: {unique_name} in shared library.")
        assert self.shared_lib_path is not None # Should be guaranteed by __init__

        # --- 1. Get Item and Validate ---
        item = await self._get_item(unique_name)
        if item is None: return False, f"Target item '{unique_name}' not found."
        if not hasattr(item, "lean_code") or not item.lean_code:
            msg = f"Target item '{unique_name}' has no lean_code to compile."
            logger.error(msg)
            await self._update_item_status(item, ItemStatus.ERROR, msg)
            return False, msg

        # --- 2. Determine Target Path & Module Name ---
        target_path: Optional[pathlib.Path] = None
        target_module_name: Optional[str] = None
        try:
             # Module name should be relative to src dir, e.g., Test.Module
             # We need to strip the library name prefix if present in unique_name
             relative_module_name = unique_name
             prefix = self.shared_lib_src_dir_name + "."
             if unique_name.startswith(prefix):
                  relative_module_name = unique_name[len(prefix):]

             module_path_rel = _module_name_to_path(relative_module_name)
             lean_file_rel = module_path_rel.with_suffix(".lean")
             # Path is relative to the source directory within the shared library
             target_path = self.shared_lib_path / self.shared_lib_src_dir_name / lean_file_rel
             # Module name for build is fully qualified including the source dir name
             target_module_name = f"{self.shared_lib_src_dir_name}.{relative_module_name}"

             logger.info(f"Target path for verification: {target_path}")
             logger.info(f"Target module for build: {target_module_name}")
        except ValueError as e:
             msg = f"Invalid unique name '{unique_name}' for path conversion: {e}"
             logger.error(msg)
             await self._update_item_status(item, ItemStatus.ERROR, msg)
             return False, msg
        except Exception as e:
             msg = f"Unexpected error calculating target path for {unique_name}: {e}"
             logger.exception(msg)
             await self._update_item_status(item, ItemStatus.ERROR, msg)
             return False, msg

        if not target_path or not target_module_name:
             msg = f"Internal error: Failed to determine target path or module name for {unique_name}."
             logger.error(msg)
             await self._update_item_status(item, ItemStatus.ERROR, msg)
             return False, msg

        # --- 3. Write File, Build, Analyze/Cleanup ---
        file_written = False
        build_success = False
        final_message = ""

        try:
            # Prepare code content
            import_block = self._generate_imports_for_target(item)
            separator = "\n\n" if import_block and item.lean_code else ""
            lean_code_str = str(item.lean_code)
            full_code = f"{import_block}{separator}{lean_code_str}"

            # Write the file using helper (ensures dir exists)
            await self.loop.run_in_executor(None, _write_text_sync, target_path, full_code)
            file_written = True
            logger.info(f"Wrote source for {unique_name} to {target_path} for verification.")

            # Attempt build in shared library context
            env = await self._prepare_subprocess_env()
            build_result = await self._run_lake_build(
                build_target=target_module_name,
                cwd=self.shared_lib_path, # Run IN the shared lib root
                env=env,
                description="Direct verification build"
            )

            # --- Process Result ---
            if build_result.returncode == 0:
                logger.info(f"Successfully verified {unique_name} directly in shared library {self.shared_lib_path}.")
                await self._update_item_status(item, ItemStatus.PROVEN, error_log=None)
                final_message = f"Lean code verified and integrated into shared library. Status: {ItemStatus.PROVEN.name if hasattr(ItemStatus.PROVEN, 'name') else ItemStatus.PROVEN}."
                build_success = True
                # File intentionally left in place

            else:
                # Build failed
                exit_code = build_result.returncode
                build_output = f"--- STDOUT ---\n{build_result.stdout or ''}\n--- STDERR ---\n{build_result.stderr or ''}".strip()
                logger.warning(f"Direct verification failed for {unique_name}. Exit code: {exit_code}")

                error_log = await self._analyze_failure_direct(
                     item, build_output, target_path
                )

                await self._update_item_status(
                    item, ItemStatus.LEAN_VALIDATION_FAILED, error_log, increment_failure=True
                )
                final_message = (
                    f"Lean validation failed (Exit code: {exit_code}). "
                    f"LSP analysis attempted. See KBItem error log. File removed."
                )
                build_success = False
                # File will be removed in finally block

        except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
            msg = f"Build execution error during direct verification: {e}"
            # Error already logged by _run_lake_build or wrapper
            await self._update_item_status(item, ItemStatus.ERROR, msg)
            build_success = False
            final_message = msg
        except OSError as e:
            msg = f"File system error writing {target_path}: {e}"
            logger.error(msg, exc_info=True)
            await self._update_item_status(item, ItemStatus.ERROR, msg)
            build_success = False
            final_message = msg
            file_written = False # Don't try to remove if write failed
        except Exception as e:
            msg = f"Unhandled exception during direct verification for {unique_name}: {e}"
            logger.exception(msg)
            await self._update_item_status(item, ItemStatus.ERROR, msg)
            build_success = False
            final_message = msg

        finally:
            # --- Cleanup: Remove the file ONLY if it was written AND build failed ---
            if file_written and not build_success:
                try:
                    logger.warning(f"Removing failed file: {target_path}")
                    await self.loop.run_in_executor(None, os.remove, target_path)
                    logger.info(f"Successfully removed failed file: {target_path}")
                except OSError as e:
                    logger.error(f"Failed to remove file {target_path} after verification failure: {e}")

        return build_success, final_message


# --- Public API Function ---

async def check_and_compile_item(
    unique_name: str,
    db_path: Optional[str] = None,
    lake_executable_path: str = "lake",
    timeout_seconds: int = 120,
) -> Tuple[bool, str]:
    """Verifies Lean code by writing it directly into the shared library and building.

    Instantiates a LeanVerifier and calls its `verify_and_integrate_item` method.
    This method attempts to write the item's Lean code to the correct path within
    the configured shared library, then runs `lake build` within that library.
    If the build succeeds, the file is kept; otherwise, it's removed after
    LSP analysis. The KBItem status is updated accordingly.

    Args:
        unique_name: The unique name of the KBItem (e.g., VantageLib.Test.Module).
        db_path: Path to the knowledge base database file.
        lake_executable_path: Path to the `lake` command-line tool.
        timeout_seconds: Timeout duration for `lake build` execution.

    Returns:
        A tuple containing:
            - bool: True if verification succeeded and file was integrated, False otherwise.
            - str: A message describing the overall outcome.
    """
    try:
        verifier = LeanVerifier(
            db_path=db_path,
            lake_executable_path=lake_executable_path,
            timeout_seconds=timeout_seconds,
        )
        return await verifier.verify_and_integrate_item(unique_name)
    except ValueError as config_err:
        logger.error(f"Configuration error preventing verification: {config_err}")
        return False, f"Configuration error: {config_err}"
    except Exception as e:
        logger.exception(f"Failed to initialize LeanVerifier or run verification for {unique_name}: {e}")
        return False, f"Initialization or critical runtime error: {e}"