# File: lean_interaction.py

"""Provides functions to interact with the Lean prover via Lake.

This module handles the validation of Lean code associated with Knowledge Base
items (`KBItem`). It defines functions to:

- Create temporary Lake environments for isolated verification builds.
- Execute `lake build` commands asynchronously.
- Update a persistent shared Lean library with successfully verified code.
- Manage database status updates based on validation outcomes.

Configuration relies on environment variables like `LEAN_AUTOMATOR_SHARED_LIB_PATH`
for the location of the persistent shared library's root directory and optionally
`LEAN_AUTOMATOR_LAKE_CACHE` for caching.
"""

import asyncio
import subprocess
import os
import tempfile
import sys
import shutil
from typing import Tuple, Dict, Set, Optional, Any, List
import logging
import pathlib
import functools

# Use absolute imports assume src is in path or package installed
try:
    # NOTE: Ensure kb_storage defines ItemStatus, KBItem etc. correctly
    from lean_automator.kb_storage import (
        KBItem,
        ItemStatus,
        ItemType, # Make sure ItemType is imported if used internally
        get_kb_item_by_name,
        save_kb_item,
        DEFAULT_DB_PATH # Import if needed for default path logic
    )
except ImportError:
     print("Error: Failed to import kb_storage.", file=sys.stderr)
     # Define dummy types to allow script to load, but it will fail at runtime
     KBItem = type('KBItem', (object,), {'lean_code': ''}) # type: ignore
     ItemStatus = type('ItemStatus', (object,), {}) # type: ignore
     ItemType = type('ItemType', (object,), {}) # type: ignore
     def get_kb_item_by_name(*args: Any, **kwargs: Any) -> None: return None # type: ignore
     async def save_kb_item(*args: Any, **kwargs: Any) -> None: pass # type: ignore
     DEFAULT_DB_PATH = 'knowledge_base.sqlite' # Define dummy default
     # raise # Or re-raise the error if preferred

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for Persistent Shared Library ---
# Example: /path/to/your/project/vantage_lib (Root of the Lake project)
SHARED_LIB_PATH_STR = os.getenv('LEAN_AUTOMATOR_SHARED_LIB_PATH')
# The specific package/module name is now handled internally based on assumptions or reading config.

# Hardcoded names based on previous discussion for vantage_lib example
# The name of the package defined in the shared library's lakefile (.lean or .toml)
SHARED_LIB_PACKAGE_NAME = "vantage_lib"
# The name of the primary library target / source directory within that package
SHARED_LIB_SRC_DIR_NAME = "VantageLib"


if not SHARED_LIB_PATH_STR:
    logger.warning("LEAN_AUTOMATOR_SHARED_LIB_PATH environment variable not set. Persistent library features will likely fail.")
    SHARED_LIB_PATH = None
else:
    SHARED_LIB_PATH = pathlib.Path(SHARED_LIB_PATH_STR).resolve()
    if not SHARED_LIB_PATH.is_dir():
        logger.warning(f"Shared library path {SHARED_LIB_PATH} does not exist or is not a directory.")
        SHARED_LIB_PATH = None
    else:
        # Optional: Add check for lakefile.lean or lakefile.toml here?
        if not (SHARED_LIB_PATH / "lakefile.lean").is_file() and not (SHARED_LIB_PATH / "lakefile.toml").is_file():
             logger.warning(f"Shared library path {SHARED_LIB_PATH} does not appear to contain a lakefile.lean or lakefile.toml.")
        # Optional: Add check for source directory?
        # if not (SHARED_LIB_PATH / SHARED_LIB_SRC_DIR_NAME).is_dir():
        #      logger.warning(f"Shared library source directory {SHARED_LIB_PATH / SHARED_LIB_SRC_DIR_NAME} not found.")


# --- Helper Functions ---

def _module_name_to_path(module_name: str) -> pathlib.Path:
    """Converts a Lean module name (dot-separated) to a relative Path object. (Internal Helper)

    Example: "Category.Theory.Limits" -> Path("Category/Theory/Limits")

    Args:
        module_name (str): The dot-separated Lean module name.

    Returns:
        pathlib.Path: The corresponding relative path.

    Raises:
        ValueError: If the module name format is invalid (e.g., empty or contains only dots).
    """
    safe_module_name = module_name.replace('\\', '/') # Normalize separators
    parts = safe_module_name.split('.')
    valid_parts = [part for part in parts if part] # Filter out empty parts

    if not valid_parts:
        raise ValueError(f"Invalid module name format: '{module_name}' resulted in no valid path parts.")
    # Construct path from valid parts
    return pathlib.Path(*valid_parts)


def _generate_imports_for_target(item: KBItem) -> str:
    """Generates Lean import statements based on the item's plan_dependencies. (Internal Helper)

    Creates standard `import Module.Name` lines for each dependency listed in
    `item.plan_dependencies`, excluding self-imports. Sorts the imports for
    consistency. It assumes the dependency names directly map to Lean modules within
    the shared library structure (e.g., dep_name "MyDep.Helper" corresponds to
    import `VantageLib.MyDep.Helper` if SHARED_LIB_SRC_DIR_NAME is `VantageLib`).

    Args:
        item (KBItem): The KBItem for which to generate imports.

    Returns:
        str: A string containing the formatted import block, potentially empty
        if no dependencies are listed or after filtering. Includes a header comment.
    """
    if not item or not item.plan_dependencies:
        return "" # Return empty string if no item or no dependencies

    import_lines: List[str] = []
    # NOTE: This assumes dependency names are relative to the shared library root module
    # e.g., if dep_name is "Sub.Module", the import becomes "import VantageLib.Sub.Module"
    # If dependency names are already fully qualified like "VantageLib.Sub.Module",
    # this logic might need adjustment or the `plan_dependencies` need to be stored differently.
    for dep_name in item.plan_dependencies:
        # Avoid self-imports
        if dep_name != item.unique_name:
            # Prepend the shared library source dir name if it's not already there
            # This assumes unique_names don't already include the library prefix. Adjust if needed.
            if not dep_name.startswith(SHARED_LIB_SRC_DIR_NAME + "."):
                full_import_name = f"{SHARED_LIB_SRC_DIR_NAME}.{dep_name}"
            else:
                full_import_name = dep_name # Assume already fully qualified
            import_lines.append(f"import {full_import_name}")

    if not import_lines:
        return "" # Return empty if only self-imports or empty list initially

    # Use set to ensure uniqueness, then sort for consistent ordering
    sorted_imports = sorted(list(set(import_lines)))
    # Add a comment indicating auto-generation
    return "-- Auto-generated imports based on plan_dependencies --\n" + "\n".join(sorted_imports)


def _create_temp_env_for_verification(
    target_item: KBItem,
    temp_dir_base: str,
    shared_lib_path: pathlib.Path,
    temp_lib_name: str = "TempVerifyLib"
) -> Tuple[str, str]:
    """Creates a temporary Lake project for isolated verification. (Internal Helper)

    Sets up a directory structure with a `lakefile.lean` that explicitly requires
    the persistent shared library package (hardcoded as SHARED_LIB_PACKAGE_NAME)
    located at `shared_lib_path`. It then writes *only* the Lean source code (`lean_code`
    prefixed with generated imports from `plan_dependencies`) of the `target_item`
    into the appropriate subdirectory within this temporary project.

    Args:
        target_item (KBItem): The item whose code needs verification.
        temp_dir_base (str): The base path of the temporary directory to use.
        shared_lib_path (pathlib.Path): The resolved path to the persistent
            shared library directory (root of the Lake project).
        temp_lib_name (str): The name to use for the temporary Lean library
            within the temporary project (defaults to "TempVerifyLib").

    Returns:
        Tuple[str, str]: A tuple containing:
            - The absolute path string to the root of the created temporary project.
            - The full Lean module name of the target item within the temporary
              project context (e.g., "TempVerifyLib.Target.Module.Name").

    Raises:
        ValueError: If `shared_lib_path` or `target_item.lean_code` is invalid
            or missing, or if the target item's `unique_name` cannot be converted
            to a valid path.
        OSError: If file system operations (creating directories, writing files) fail.
        TypeError: If `target_item.lean_code` is not a string or string-convertible.
    """
    if not shared_lib_path or not shared_lib_path.is_dir():
            raise ValueError(f"Shared library path '{shared_lib_path}' is invalid or not configured.")

    temp_project_path = pathlib.Path(temp_dir_base)
    # Directory where source code for the temporary library will live
    lib_source_dir = temp_project_path / temp_lib_name

    # --- Create lakefile.lean ---
    # Use resolved absolute path for the required library for robustness
    shared_lib_abs_path = str(shared_lib_path.resolve()).replace('\\', '/') # Normalize path separators

    # Use the hardcoded SHARED_LIB_PACKAGE_NAME in the require statement
    lakefile_content = f"""
import Lake
open Lake DSL

-- Require the persistent shared library using its absolute path
-- Using package name '{SHARED_LIB_PACKAGE_NAME}'
require {SHARED_LIB_PACKAGE_NAME} from "{shared_lib_abs_path}"

-- Define the temporary package
package _{temp_lib_name.lower()}_project where
  -- package configuration options

-- Define the temporary library within the package
lean_lib {temp_lib_name} where
  -- library configuration options
"""
    lakefile_path = temp_project_path / "lakefile.lean"
    logger.debug(f"Creating temporary lakefile requiring '{SHARED_LIB_PACKAGE_NAME}' from '{shared_lib_abs_path}': {lakefile_path}")
    try:
        # Ensure parent directory exists and write the lakefile
        lakefile_path.parent.mkdir(parents=True, exist_ok=True)
        lakefile_path.write_text(lakefile_content, encoding='utf-8')
    except OSError as e:
        raise OSError(f"Failed to write temporary lakefile.lean: {e}") from e

    # --- Write ONLY the target item's .lean source file ---
    if not target_item or not hasattr(target_item, 'lean_code') or not target_item.lean_code:
        raise ValueError(f"Target item '{target_item.unique_name if target_item else 'None'}' has no lean_code to write.")

    try:
        # Convert the item's unique name (e.g., "My.Target.Module") to a relative path ("My/Target/Module")
        # This assumes unique_name does NOT contain the library root module prefix (e.g. "VantageLib.")
        # If it does, _module_name_to_path might need adjustment or the name needs cleaning first.
        module_path_rel = _module_name_to_path(target_item.unique_name)
    except ValueError as e:
        logger.error(f"Invalid module name '{target_item.unique_name}' for target item. Cannot create file path.")
        raise ValueError(f"Invalid module name for target item: {e}") from e

    lean_file_rel = module_path_rel.with_suffix('.lean')
    # Place the source file inside the temporary library's source directory structure
    lean_file_abs = lib_source_dir / lean_file_rel

    logger.debug(f"Target source file path (absolute): {lean_file_abs}")
    try:
        # Ensure the target directory exists within the temp structure
        lean_file_abs.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create source directory '{lean_file_abs.parent}' for target item {target_item.unique_name}: {e}") from e

    # Generate standard import statements based on the item's plan_dependencies
    # These imports should be fully qualified including the SHARED_LIB_SRC_DIR_NAME prefix
    import_block = _generate_imports_for_target(target_item)
    logger.debug(f"Generated imports for target {target_item.unique_name}:\n{import_block}")

    # Combine imports and the item's Lean code
    separator = "\n\n" if import_block and target_item.lean_code else ""
    try:
        # Ensure lean_code is a string before combining
        lean_code_str = str(target_item.lean_code)
    except Exception as e:
         raise TypeError(f"Failed to convert lean_code to string for target {target_item.unique_name} (type: {type(target_item.lean_code)}): {e}") from e

    full_code = f"{import_block}{separator}{lean_code_str}"

    logger.debug(f"Writing target source code to: {lean_file_abs}")
    try:
        # Write the combined code to the target file
        lean_file_abs.write_text(full_code, encoding='utf-8')
    except OSError as e:
        raise OSError(f"Failed to write .lean file '{lean_file_abs}' for target item {target_item.unique_name}: {e}") from e
    except Exception as e: # Catch other potential errors like encoding issues
         raise RuntimeError(f"Unexpected error writing lean_code for target {target_item.unique_name} to {lean_file_abs}: {e}") from e

    # Construct the full module name required for the 'lake build' command in the temp env
    # This assumes unique_name represents the path relative to the library root.
    full_target_module_name = f"{temp_lib_name}.{target_item.unique_name}"

    return str(temp_project_path.resolve()), full_target_module_name


async def _update_persistent_library(
    item: KBItem,
    shared_lib_path: pathlib.Path,
    lake_executable_path: str,
    timeout_seconds: int,
    loop: asyncio.AbstractEventLoop
    ) -> bool:
    """Adds verified item code to the persistent library and builds it. (Internal Helper)

    Takes a successfully verified KBItem, writes its source code (including any
    necessary imports derived from `plan_dependencies`) to the correct location
    within the persistent shared library file structure (`shared_lib_path`), using
    the hardcoded SHARED_LIB_SRC_DIR_NAME for the source directory. It then
    executes `lake build <item.unique_name>` within that library's directory
    to integrate the new/updated item.

    Args:
        item (KBItem): The verified KBItem containing the `lean_code`.
        shared_lib_path (pathlib.Path): The path to the root of the persistent
            shared library Lake project.
        lake_executable_path (str): Path to the `lake` executable.
        timeout_seconds (int): Timeout duration for the `lake build` command.
        loop (asyncio.AbstractEventLoop): The running asyncio event loop.

    Returns:
        bool: True if the source file was written and the subsequent `lake build`
        command in the persistent library succeeded (exit code 0). False otherwise
        (e.g., file system errors, invalid module name, build failure, timeout).
    """
    if not shared_lib_path or not shared_lib_path.is_dir():
         logger.error(f"Cannot update persistent library: Path '{shared_lib_path}' is invalid.")
         return False
    if not item or not hasattr(item, 'lean_code') or not item.lean_code:
         logger.error(f"Cannot update persistent library: Item '{item.unique_name if item else 'None'}' has no lean_code.")
         return False

    logger.info(f"Updating persistent library '{SHARED_LIB_SRC_DIR_NAME}' at {shared_lib_path} with item '{item.unique_name}'.")

    # 1. Determine destination path for the .lean source file within the shared library
    try:
        # Convert item unique name to relative path (e.g., "My.Module" -> Path("My/Module"))
        # Assumes unique_name does not include the library prefix.
        module_path_rel = _module_name_to_path(item.unique_name)
        lean_file_rel = module_path_rel.with_suffix('.lean')
        # Destination path is relative to the shared library's *source directory* (hardcoded name)
        # Example: /path/to/shared/VantageLib/My/Module.lean
        dest_file_abs = shared_lib_path / SHARED_LIB_SRC_DIR_NAME / lean_file_rel
        logger.debug(f"Destination path in persistent library: {dest_file_abs}")
    except ValueError as e:
        logger.error(f"Invalid module name '{item.unique_name}'. Cannot determine persistent path: {e}")
        return False

    # 2. Write the source file (with imports) to the persistent library
    try:
        # Ensure parent directory exists (run sync os.makedirs in executor)
        await loop.run_in_executor(None, functools.partial(os.makedirs, dest_file_abs.parent, exist_ok=True))

        # Generate imports based on plan_dependencies (these should be fully qualified)
        import_block = _generate_imports_for_target(item)
        separator = "\n\n" if import_block and item.lean_code else ""
        lean_code_str = str(item.lean_code) # Ensure string type
        full_code = f"{import_block}{separator}{lean_code_str}"

        logger.debug(f"Writing source file to persistent library: {dest_file_abs}")
        # Use a lambda for write_text within run_in_executor
        write_lambda = lambda p, c: p.write_text(c, encoding='utf-8')
        await loop.run_in_executor(None, write_lambda, dest_file_abs, full_code)
        logger.info(f"Successfully wrote source for {item.unique_name} to persistent library.")

    except OSError as e:
        logger.error(f"Failed to write source file {dest_file_abs} for {item.unique_name} to persistent library: {e}")
        return False # Fail if cannot write the file
    except Exception as e: # Catch other errors like TypeError from str() or encoding issues
        logger.error(f"Unexpected error writing source file {dest_file_abs}: {e}")
        return False

    # 3. Trigger 'lake build' within the persistent library directory
    # We build the specific module that was just added/updated.
    # The target name needs to be fully qualified relative to the package root.
    target_build_name = f"{SHARED_LIB_SRC_DIR_NAME}.{item.unique_name}"

    command = [lake_executable_path, 'build', target_build_name]
    logger.info(f"Triggering persistent library build: {' '.join(command)} in {shared_lib_path}")

    persistent_build_success = False
    try:
        # Prepare arguments for subprocess.run
        run_args = {
            'args': command,
            'capture_output': True, 'text': True, # Capture stdout/stderr as text
            'timeout': timeout_seconds,
            'encoding': 'utf-8', 'errors': 'replace', # Handle potential encoding errors
            'cwd': str(shared_lib_path), # IMPORTANT: Run IN the shared library directory
            'check': False, # We'll check the return code manually
            'env': os.environ.copy() # Inherit environment, may need LEAN_PATH adjustments if complex deps
        }
        # Run the potentially blocking subprocess in an executor thread
        lake_process_result = await loop.run_in_executor(None, functools.partial(subprocess.run, **run_args))

        stdout_output = lake_process_result.stdout or ""
        stderr_output = lake_process_result.stderr or ""
        log_output = f"--- STDOUT ---\n{stdout_output}\n--- STDERR ---\n{stderr_output}"

        if lake_process_result.returncode == 0:
            logger.info(f"Persistent library build successful for target '{target_build_name}'.")
            persistent_build_success = True
        else:
            # Log detailed error if the build command fails
            logger.error(f"Persistent library build FAILED for target '{target_build_name}'. Exit code: {lake_process_result.returncode}")
            logger.error(f"Persistent build output:\n{log_output}")
            persistent_build_success = False # Build failed

    except subprocess.TimeoutExpired:
         logger.error(f"Persistent library build timed out after {timeout_seconds} seconds for target '{target_build_name}'.")
         persistent_build_success = False
    except FileNotFoundError:
         logger.error(f"Lake executable '{lake_executable_path}' not found during persistent build.")
         persistent_build_success = False
    except Exception as e:
        # Catch any other unexpected errors during subprocess execution
        logger.exception(f"Unexpected error running persistent Lake build subprocess for {target_build_name}: {e}")
        persistent_build_success = False

    return persistent_build_success # Return the success status of the persistent build step


# --- Core Function ---

async def check_and_compile_item(
    unique_name: str,
    db_path: Optional[str] = None,
    lake_executable_path: str = 'lake',
    timeout_seconds: int = 120,
) -> Tuple[bool, str]:
    """Verifies Lean code using Lake and updates a persistent shared library.

    This function implements a two-stage validation process for a `KBItem`:
    1.  **Temporary Verification:** Creates an isolated temporary Lake project.
        This project contains only the target item's source code (with generated
        imports) and a `lakefile.lean` that requires the persistent shared
        library package (identified by SHARED_LIB_PACKAGE_NAME) using its path
        (LEAN_AUTOMATOR_SHARED_LIB_PATH). It runs `lake build` within this
        temporary environment.
    2.  **Persistent Update:** If the temporary verification succeeds (build completes
        with exit code 0), the function writes the item's source code to the
        correct directory within the persistent shared library (using
        SHARED_LIB_SRC_DIR_NAME) and runs `lake build` within the root of *that*
        library project to integrate the new item.

    Updates the `KBItem` status in the database to `PROVEN` upon successful
    verification and persistent update, `LEAN_VALIDATION_FAILED` if the temporary
    verification fails, or `ERROR` for configuration or unexpected issues.

    Args:
        unique_name (str): The unique name of the KBItem to check and compile.
            Assumed to be relative to the library root (e.g., "MyModule.MySub").
        db_path (Optional[str]): Path to the knowledge base database file. Uses
            `DEFAULT_DB_PATH` if None.
        lake_executable_path (str): The path to the `lake` command-line tool.
            Defaults to 'lake' (assumes it's in the system PATH).
        timeout_seconds (int): Timeout duration in seconds for each `lake build`
            subprocess execution (both temporary and persistent). Defaults to 120.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the temporary verification was successful, False otherwise.
              Note: This reflects the success of the primary verification step,
              even if the subsequent persistent library update fails (which will
              be logged as a warning).
            - str: A message describing the outcome (e.g., "Lean code verified.",
              "Lean validation failed.", "Configuration error.", "Timeout.",
              "Warning: Persistent library update failed."). Includes error details
              or status updates where applicable.
    """
    logger.info(f"Starting Lean validation process for: {unique_name} using shared library.")
    effective_db_path = db_path or DEFAULT_DB_PATH
    loop = asyncio.get_running_loop()
    target_item: Optional[KBItem] = None # Initialize target_item

    # --- Check Shared Library Configuration ---
    if not SHARED_LIB_PATH or not SHARED_LIB_PATH.is_dir():
         msg = "Shared library path (LEAN_AUTOMATOR_SHARED_LIB_PATH) not configured correctly or directory does not exist."
         logger.error(msg)
         return False, msg
    logger.debug(f"Using shared library path: {SHARED_LIB_PATH}")

    # --- Fetch Target Item ---
    try:
        target_item = get_kb_item_by_name(unique_name, db_path=effective_db_path)
        if target_item is None:
            msg = f"Target item '{unique_name}' not found in database."
            logger.error(msg)
            return False, msg
        if not hasattr(target_item, 'lean_code') or not target_item.lean_code:
             msg = f"Target item '{unique_name}' has no lean_code to compile."
             logger.error(msg)
             try:
                 target_item.update_status(ItemStatus.ERROR, "Missing lean_code for compilation.")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
             except Exception as db_err: logger.error(f"Failed to update status for item missing code: {db_err}")
             return False, msg
        logger.info(f"Fetched target item {unique_name} for validation.")

    except Exception as e:
        logger.exception(f"Unexpected error fetching target item {unique_name}: {e}")
        if target_item and isinstance(target_item, KBItem):
            try:
                target_item.update_status(ItemStatus.ERROR, f"Unexpected fetch error: {e}")
                await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception as db_err: logger.error(f"Failed to update status after fetch error: {db_err}")
        return False, f"Unexpected error fetching target: {e}"

    # --- Create Temporary Environment ---
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    temp_project_path_str : Optional[str] = None
    target_temp_module_name : Optional[str] = None
    temp_lib_name_used = "TempVerifyLib"

    try:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir_base = temp_dir_obj.name
        logger.info(f"Creating temporary verification environment in: {temp_dir_base}")
        # Pass the hardcoded package name determined earlier
        temp_project_path_str, target_temp_module_name = _create_temp_env_for_verification(
             target_item, temp_dir_base, SHARED_LIB_PATH,
             temp_lib_name_used
        )
        logger.info(f"Temporary Lake project created at: {temp_project_path_str}")
        logger.info(f"Target module name for temporary build: {target_temp_module_name}")

    except (ValueError, OSError, TypeError) as e:
         logger.error(f"Failed to create temporary verification environment for {unique_name}: {e}")
         if target_item:
             try:
                 target_item.update_status(ItemStatus.ERROR, f"Verification env creation error: {str(e)[:500]}")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
             except Exception as db_err: logger.error(f"Failed to update status after env creation error: {db_err}")
         if temp_dir_obj:
             try: await loop.run_in_executor(None, temp_dir_obj.cleanup)
             except Exception: pass
         return False, f"Verification environment creation error: {e}"
    except Exception as e:
        logger.exception(f"Unexpected error during verification environment setup for {unique_name}: {e}")
        if target_item:
            try:
                target_item.update_status(ItemStatus.ERROR, f"Unexpected env setup error: {str(e)[:500]}")
                await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception as db_err: logger.error(f"Failed to update status after unexpected env setup error: {db_err}")
        if temp_dir_obj:
            try: await loop.run_in_executor(None, temp_dir_obj.cleanup)
            except Exception: pass
        return False, f"Unexpected setup error: {e}"

    if not temp_project_path_str or not target_temp_module_name:
        msg = "Internal error: Temporary environment setup failed to return valid paths."
        logger.error(f"{msg} for {unique_name}.")
        if target_item:
            try:
                target_item.update_status(ItemStatus.ERROR, msg)
                await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception: pass
        if temp_dir_obj:
             try: await loop.run_in_executor(None, temp_dir_obj.cleanup)
             except Exception: pass
        return False, msg

    # --- Execute Lake Build (Temporary Verification) ---
    try:
        subprocess_env = os.environ.copy()
        std_lib_path = None
        lean_executable_path_obj = None
        try:
            lean_executable_path_obj = pathlib.Path(lake_executable_path).resolve()
            lean_exe_candidate = lean_executable_path_obj.parent / 'lean'
            lean_exe = str(lean_exe_candidate) if lean_exe_candidate.is_file() else 'lean'
        except Exception as path_err:
            logger.warning(f"Could not resolve lean executable path from lake path '{lake_executable_path}': {path_err}. Assuming 'lean' is in PATH.")
            lean_exe = 'lean'

        if lean_exe == 'lean' and not shutil.which('lean'):
             logger.warning("Could not find 'lean' executable relative to lake or in PATH. Cannot detect stdlib path.")
        else:
            try:
                logger.debug(f"Attempting stdlib detection using: {lean_exe} --print-libdir")
                lean_path_proc = await loop.run_in_executor(None, functools.partial(
                    subprocess.run, [lean_exe, '--print-libdir'],
                    capture_output=True, text=True, check=True, timeout=10, encoding='utf-8'
                ))
                path_candidate = lean_path_proc.stdout.strip()
                if path_candidate and pathlib.Path(path_candidate).is_dir():
                    std_lib_path = path_candidate
                    logger.info(f"Detected Lean stdlib path: {std_lib_path}")
                else:
                    logger.warning(f"Command '{lean_exe} --print-libdir' did not return valid directory: '{path_candidate}'")
            except Exception as e:
                logger.warning(f"Failed to detect Lean stdlib path via '{lean_exe} --print-libdir': {e}")

        if not std_lib_path and lean_executable_path_obj:
            try:
                toolchain_dir = lean_executable_path_obj.parent.parent
                fallback_path = toolchain_dir / "lib" / "lean"
                if await loop.run_in_executor(None, fallback_path.is_dir):
                    std_lib_path = str(fallback_path.resolve())
                    logger.warning(f"Assuming stdlib path relative to toolchain: {std_lib_path}")
            except Exception as fallback_e:
                logger.warning(f"Error during fallback stdlib detection: {fallback_e}")

        if std_lib_path:
            existing_lean_path = subprocess_env.get('LEAN_PATH')
            separator = os.pathsep
            subprocess_env['LEAN_PATH'] = f"{std_lib_path}{separator}{existing_lean_path}" if existing_lean_path else std_lib_path
            logger.debug(f"Setting LEAN_PATH for subprocess: {subprocess_env['LEAN_PATH']}")
        else:
            logger.warning("Stdlib path not found. LEAN_PATH will not include it. Build might fail if stdlib needed implicitly.")

        persistent_cache_path = os.getenv('LEAN_AUTOMATOR_LAKE_CACHE')
        if persistent_cache_path:
            persistent_cache_path_obj = pathlib.Path(persistent_cache_path).resolve()
            try:
                await loop.run_in_executor(None, functools.partial(os.makedirs, persistent_cache_path_obj, exist_ok=True))
                subprocess_env['LAKE_HOME'] = str(persistent_cache_path_obj)
                logger.info(f"Setting LAKE_HOME for subprocess: {persistent_cache_path_obj}")
            except OSError as e:
                logger.error(f"Failed to create/use persistent Lake cache directory '{persistent_cache_path_obj}': {e}")
        else:
            logger.warning("LEAN_AUTOMATOR_LAKE_CACHE not set. Lake will use default caching.")

        command = [lake_executable_path, 'build', target_temp_module_name]
        lake_process_result: Optional[subprocess.CompletedProcess] = None
        full_output = ""
        logger.info(f"Running verification build: {' '.join(command)} in {temp_project_path_str}")

        try:
            run_args = {
                'args': command, 'capture_output': True, 'text': True,
                'timeout': timeout_seconds, 'encoding': 'utf-8', 'errors': 'replace',
                'cwd': temp_project_path_str,
                'check': False,
                'env': subprocess_env
            }
            lake_process_result = await loop.run_in_executor(None, functools.partial(subprocess.run, **run_args))

            stdout_output = lake_process_result.stdout or ""
            stderr_output = lake_process_result.stderr or ""
            logger.debug(f"Verification build return code: {lake_process_result.returncode}")
            if stdout_output:
                 logger.debug(f"Verification build stdout:\n{stdout_output}")
            if stderr_output:
                 logger.debug(f"Verification build stderr:\n{stderr_output}")
            full_output = f"--- STDOUT ---\n{stdout_output}\n--- STDERR ---\n{stderr_output}"

        except subprocess.TimeoutExpired:
            msg = f"Timeout after {timeout_seconds}s during temporary verification build"
            logger.error(f"{msg} for {unique_name}.")
            if target_item:
                 target_item.update_status(ItemStatus.ERROR, msg)
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
            return False, msg
        except FileNotFoundError:
             msg = f"Lake executable not found at '{lake_executable_path}'"
             logger.error(msg)
             return False, msg
        except Exception as e:
            msg = f"Subprocess execution error during verification build: {e}"
            logger.exception(f"Unexpected error running verification build subprocess for {unique_name}: {e}")
            if target_item:
                 target_item.update_status(ItemStatus.ERROR, msg)
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
            return False, msg

        if not lake_process_result:
             error_msg = "Internal error: Lake process result missing after execution."
             logger.error(error_msg)
             if target_item:
                  target_item.update_status(ItemStatus.ERROR, error_msg)
                  await save_kb_item(target_item, client=None, db_path=effective_db_path)
             return False, error_msg

        if lake_process_result.returncode == 0:
            logger.info(f"Successfully verified {unique_name} in temporary environment.")
            target_item = get_kb_item_by_name(unique_name, db_path=effective_db_path)
            if not target_item:
                 msg = f"Item {unique_name} disappeared from DB after successful verification."
                 logger.error(msg)
                 return False, msg

            target_item.update_status(ItemStatus.PROVEN, error_log=None)
            await save_kb_item(target_item, client=None, db_path=effective_db_path)
            logger.info(f"Database status updated to PROVEN for {unique_name}.")

            # Pass the hardcoded library src dir name determined earlier
            persist_success = await _update_persistent_library(
                target_item, SHARED_LIB_PATH,
                lake_executable_path, timeout_seconds, loop
            )
            if persist_success:
                logger.info(f"Successfully updated persistent library with {unique_name}.")
                final_msg = f"Lean code verified. Status: {ItemStatus.PROVEN.name}. Persistent library updated."
                return True, final_msg
            else:
                logger.warning(f"Verification succeeded for {unique_name}, BUT failed to update/build persistent library.")
                final_msg = f"Lean code verified. Status: {ItemStatus.PROVEN.name}. WARNING: Persistent library update failed (see logs)."
                return True, final_msg

        else: # Verification build failed (return code != 0)
            final_error_message = full_output.strip() if full_output.strip() else "Unknown Lake build error (no output captured)."
            exit_code = lake_process_result.returncode
            logger.warning(f"Lean verification failed for {unique_name}. Exit code: {exit_code}")
            logger.debug(f"Captured output for failed verification build of {unique_name}:\n{final_error_message[:1000]}...")

            target_item = get_kb_item_by_name(unique_name, db_path=effective_db_path)
            if not target_item:
                msg = f"Item {unique_name} disappeared from DB after failed verification."
                logger.error(msg)
                return False, msg

            target_item.update_status(ItemStatus.LEAN_VALIDATION_FAILED, error_log=final_error_message)
            target_item.increment_failure_count()
            await save_kb_item(target_item, client=None, db_path=effective_db_path)
            logger.info(f"Database status updated to LEAN_VALIDATION_FAILED for {unique_name}.")
            final_msg = f"Lean validation failed (Exit code: {exit_code}). See logs or KBItem error log."
            return False, final_msg

    except Exception as e:
         logger.exception(f"Unhandled exception during check_and_compile for {unique_name}: {e}")
         if target_item and isinstance(target_item, KBItem) and getattr(target_item, 'status', None) != ItemStatus.PROVEN:
              try:
                  current_item_state = get_kb_item_by_name(unique_name, db_path=effective_db_path)
                  if current_item_state and current_item_state.status != ItemStatus.PROVEN:
                      current_item_state.update_status(ItemStatus.ERROR, f"Unhandled exception: {str(e)[:500]}")
                      await save_kb_item(current_item_state, client=None, db_path=effective_db_path)
              except Exception as db_err:
                  logger.error(f"Failed to update item status after unhandled exception: {db_err}")
         return False, f"Unhandled exception: {e}"
    finally:
        if temp_dir_obj:
            try:
                await loop.run_in_executor(None, temp_dir_obj.cleanup)
                logger.debug(f"Successfully cleaned up temporary directory: {temp_dir_obj.name}")
            except Exception as cleanup_err:
                 logger.error(f"Error cleaning up temporary directory {temp_dir_obj.name}: {cleanup_err}")