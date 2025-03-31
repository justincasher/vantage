# File: lean_interaction.py

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
    # NOTE: Assumes KBItem no longer has lean_olean field after user modification
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
# Example: /path/to/your/project/lean_proven_kb
SHARED_LIB_PATH_STR = os.getenv('LEAN_AUTOMATOR_SHARED_LIB_PATH')
SHARED_LIB_MODULE_NAME = os.getenv('LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME', 'ProvenKB') # e.g., the library name defined in the shared lakefile

if not SHARED_LIB_PATH_STR:
    logger.warning("LEAN_AUTOMATOR_SHARED_LIB_PATH environment variable not set. Persistent library features will likely fail.")
    SHARED_LIB_PATH = None
else:
    SHARED_LIB_PATH = pathlib.Path(SHARED_LIB_PATH_STR).resolve()
    if not SHARED_LIB_PATH.is_dir():
        logger.warning(f"Shared library path {SHARED_LIB_PATH} does not exist or is not a directory.")
        # Optionally try to create it? For now, just warn.
        # try:
        #     SHARED_LIB_PATH.mkdir(parents=True, exist_ok=True)
        #     logger.info(f"Created shared library directory: {SHARED_LIB_PATH}")
        #     # Need to run `lake init <SHARED_LIB_MODULE_NAME>` here? Better to do manually.
        # except OSError as e:
        #     logger.error(f"Failed to create shared library directory {SHARED_LIB_PATH}: {e}")
        #     SHARED_LIB_PATH = None


# --- Helper Functions ---

def _module_name_to_path(module_name: str) -> pathlib.Path:
    """Converts Lean module name to relative Path."""
    safe_module_name = module_name.replace('\\', '/')
    # Strip leading library name if present (e.g., ProvenKB.MyModule -> MyModule)
    # This assumes the shared library name isn't part of the unique_name logic itself.
    # Adjust if unique_names are expected to contain the shared lib name.
    # parts = safe_module_name.split('.')
    # if parts[0] == SHARED_LIB_MODULE_NAME:
    #     parts = parts[1:]
    parts = safe_module_name.split('.')

    valid_parts = [part for part in parts if part]
    if not valid_parts:
        raise ValueError(f"Invalid module name format: '{module_name}'")
    return pathlib.Path(*valid_parts)

# Modified: Generates standard imports, not prefixed with temp lib name
def _generate_imports_for_target(item: KBItem) -> str:
    """Generates the Lean import block string for a given item's plan_dependencies."""
    if not item or not item.plan_dependencies:
        return ""

    import_lines: List[str] = []
    for dep_name in item.plan_dependencies:
        # Basic check to avoid self-imports, although Lean handles them
        if dep_name != item.unique_name:
            # Generate standard import statements
            import_lines.append(f"import {dep_name}")

    if not import_lines:
        return ""

    # Sort for consistency and add a comment
    sorted_imports = sorted(list(set(import_lines))) # Use set to ensure uniqueness
    return "-- Auto-generated imports based on plan_dependencies --\n" + "\n".join(sorted_imports)


def _fetch_recursive_dependencies(
    unique_name: str,
    db_path: Optional[str],
    visited_names: Set[str],
    fetched_items: Dict[str, KBItem]
) -> None:
    """
    Recursively fetches an item and its PRE-DEFINED dependencies from the database.
    Uses the 'plan_dependencies' field of KBItem.
    (No changes needed here)
    """
    if unique_name in visited_names:
        if unique_name not in fetched_items:
            logger.warning(f"Cycle detected or item visited but not fetched: {unique_name}")
        return

    visited_names.add(unique_name)

    logger.debug(f"Fetching item and plan dependencies for: {unique_name}")
    item = get_kb_item_by_name(unique_name, db_path=db_path)
    if item is None:
        visited_names.remove(unique_name)
        raise FileNotFoundError(f"Item '{unique_name}' not found in the database during dependency fetch.")

    fetched_items[unique_name] = item

    logger.debug(f"Item '{unique_name}' plan_dependencies: {item.plan_dependencies}")
    dependencies_to_fetch = item.plan_dependencies if item.plan_dependencies else []
    for dep_name in dependencies_to_fetch:
        if dep_name == unique_name: continue

        if dep_name not in fetched_items:
             if dep_name not in visited_names:
                 try:
                      _fetch_recursive_dependencies(dep_name, db_path, visited_names, fetched_items)
                 except FileNotFoundError as e:
                      logger.error(f"Failed to fetch sub-dependency '{dep_name}' required by '{unique_name}'.")
                      raise e


# Modified: Creates temp env with ONLY target source and requires shared lib
def _create_temp_env_for_verification(
    target_item: KBItem,
    temp_dir_base: str,
    shared_lib_path: pathlib.Path,
    shared_lib_module_name: str,
    temp_lib_name: str = "TempVerifyLib"
) -> Tuple[str, str]:
    """
    Creates a temporary Lake project containing ONLY the target item's source code.
    The lakefile requires the persistent shared library.
    """
    if not shared_lib_path or not shared_lib_path.is_dir():
         raise ValueError(f"Shared library path '{shared_lib_path}' is invalid or not configured.")
    if not shared_lib_module_name:
         raise ValueError("Shared library module name is not configured.")

    temp_project_path = pathlib.Path(temp_dir_base)
    lib_source_dir = temp_project_path / temp_lib_name # Use temp lib name for source dir

    # --- Create lakefile.lean ---
    # Use absolute path for the required library for robustness
    shared_lib_abs_path = str(shared_lib_path.resolve())

    lakefile_content = f"""
import Lake
open Lake DSL

require {shared_lib_module_name} from "{shared_lib_abs_path}"

package _{temp_lib_name.lower()}_project where
  -- add package configuration options here

lean_lib {temp_lib_name} where
  -- add library configuration options here
"""
    lakefile_path = temp_project_path / "lakefile.lean"
    logger.debug(f"Creating temporary lakefile requiring '{shared_lib_module_name}': {lakefile_path}")
    try:
        lakefile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lakefile_path, 'w', encoding='utf-8') as f:
            f.write(lakefile_content)
    except OSError as e:
        raise OSError(f"Failed to write temporary lakefile.lean: {e}") from e

    # --- Write ONLY the target item's .lean source file ---
    if not target_item or not target_item.lean_code:
        raise ValueError(f"Target item '{target_item.unique_name}' has no lean_code to write.")

    try:
        # Use the item's unique name to determine the path WITHIN the temp lib structure
        module_path_rel = _module_name_to_path(target_item.unique_name)
    except ValueError as e:
        logger.error(f"Invalid module name '{target_item.unique_name}' for target item. Cannot create file.")
        raise ValueError(f"Invalid module name for target: {e}") from e

    lean_file_rel = module_path_rel.with_suffix('.lean')
    # Place it inside the temporary library's source directory
    lean_file_abs = lib_source_dir / lean_file_rel

    # Ensure directory exists within the temp structure
    logger.debug(f"Ensuring directory for target source: {lean_file_abs.parent}")
    try:
        lean_file_abs.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create source directory for target item {target_item.unique_name}: {e}") from e

    # Generate standard imports based on plan_dependencies
    import_block = _generate_imports_for_target(target_item)
    logger.debug(f"Generated imports for target {target_item.unique_name}:\n{import_block}")

    # Combine imports and code
    separator = "\n\n" if import_block and target_item.lean_code else ""
    full_code = f"{import_block}{separator}{str(target_item.lean_code)}" # Ensure code is string

    logger.debug(f"Writing target source: {lean_file_abs} for item '{target_item.unique_name}'")
    try:
        with open(lean_file_abs, 'w', encoding='utf-8') as f:
            f.write(full_code)
    except OSError as e:
        raise OSError(f"Failed to write .lean file for target item {target_item.unique_name}: {e}") from e
    except Exception as e:
         raise TypeError(f"Error writing lean_code for target {target_item.unique_name} (type: {type(target_item.lean_code)}): {e}") from e

    # The target module name within the temporary library context
    full_target_module_name = f"{temp_lib_name}.{target_item.unique_name}"

    return str(temp_project_path), full_target_module_name


async def _update_persistent_library(
    item: KBItem,
    shared_lib_path: pathlib.Path,
    shared_lib_module_name: str,
    lake_executable_path: str,
    timeout_seconds: int,
    loop: asyncio.AbstractEventLoop
    ) -> bool:
    """
    Adds the item's source code to the persistent shared library and triggers a build.
    """
    if not shared_lib_path or not shared_lib_path.is_dir():
         logger.error(f"Cannot update persistent library: Path '{shared_lib_path}' is invalid.")
         return False
    if not item or not item.lean_code:
         logger.error(f"Cannot update persistent library: Item '{item.unique_name}' has no lean_code.")
         return False

    logger.info(f"Updating persistent library '{shared_lib_module_name}' at {shared_lib_path} with item '{item.unique_name}'.")

    # 1. Determine destination path for the .lean source file
    try:
        module_path_rel = _module_name_to_path(item.unique_name)
        lean_file_rel = module_path_rel.with_suffix('.lean')
        # Destination is relative to the shared library's module source root
        dest_file_abs = shared_lib_path / shared_lib_module_name / lean_file_rel
    except ValueError as e:
        logger.error(f"Invalid module name '{item.unique_name}'. Cannot determine persistent path: {e}")
        return False

    # 2. Write the source file to the persistent library
    try:
        logger.debug(f"Ensuring directory exists: {dest_file_abs.parent}")
        await loop.run_in_executor(None, functools.partial(os.makedirs, dest_file_abs.parent, exist_ok=True))

        logger.debug(f"Writing source file to persistent library: {dest_file_abs}")
        # Assume item.lean_code already contains necessary standard imports if generated by processor
        write_lambda = lambda p, c: p.write_text(c, encoding='utf-8')
        await loop.run_in_executor(None, write_lambda, dest_file_abs, str(item.lean_code))
        logger.info(f"Successfully wrote source for {item.unique_name} to persistent library.")

    except OSError as e:
        logger.error(f"Failed to write source file {dest_file_abs} for {item.unique_name} to persistent library: {e}")
        # Consider if we should revert DB status here? For now, just log and return failure.
        return False
    except Exception as e:
        logger.error(f"Unexpected error writing source file {dest_file_abs}: {e}")
        return False

    # 3. Trigger 'lake build' within the persistent library directory
    # We build the specific module that was just added/updated.
    # Building the whole library might be safer but slower (`lake build SharedLib`)
    target_build_name = item.unique_name # Use the item's fully qualified name

    command = [lake_executable_path, 'build', target_build_name]
    logger.info(f"Triggering persistent library build: {' '.join(command)} in {shared_lib_path}")

    persistent_build_success = False
    try:
        run_args = {
             'args': command, 'capture_output': True, 'text': True,
             'timeout': timeout_seconds, 'encoding': 'utf-8', 'errors': 'replace',
             'cwd': str(shared_lib_path), # Run IN the shared library directory
             'check': False, # Handle non-zero exit code manually
             'env': os.environ.copy() # Use current env, maybe inherit LEAN_PATH? ok for now.
        }
        # Using run_in_executor as this is called from within the async check_and_compile_item
        lake_process_result = await loop.run_in_executor(None, functools.partial(subprocess.run, **run_args))

        stdout_output = lake_process_result.stdout or ""
        stderr_output = lake_process_result.stderr or ""
        log_output = f"--- STDOUT ---\n{stdout_output}\n--- STDERR ---\n{stderr_output}"

        if lake_process_result.returncode == 0:
            logger.info(f"Persistent library build successful for target '{target_build_name}'.")
            persistent_build_success = True
        else:
            logger.error(f"Persistent library build FAILED for target '{target_build_name}'. Exit code: {lake_process_result.returncode}")
            logger.error(f"Persistent build output:\n{log_output}")
            # Even if the persistent build fails, the item was verified successfully in temp env.
            # We log this as a serious error/inconsistency but don't fail the overall verification result.
            persistent_build_success = False # Indicate failure, but verification already succeeded

    except subprocess.TimeoutExpired:
         logger.error(f"Persistent library build timed out after {timeout_seconds} seconds for target '{target_build_name}'.")
         persistent_build_success = False
    except FileNotFoundError:
         logger.error(f"Lake executable '{lake_executable_path}' not found during persistent build.")
         persistent_build_success = False
    except Exception as e:
        logger.exception(f"Unexpected error running persistent Lake build subprocess for {target_build_name}: {e}")
        persistent_build_success = False

    return persistent_build_success # Return status of the persistent build itself


# --- Core Function ---

async def check_and_compile_item(
    unique_name: str,
    db_path: Optional[str] = None,
    lake_executable_path: str = 'lake',
    timeout_seconds: int = 120,
    # Removed temp_lib_name as it's hardcoded in _create_temp... now
) -> Tuple[bool, str]:
    """
    Checks and attempts to compile/validate a KBItem using Lake, requiring a
    persistent shared library (`LEAN_AUTOMATOR_SHARED_LIB_PATH`).

    On successful validation in a temporary environment, it adds the item's
    source code to the persistent library and triggers a build there.
    """
    logger.info(f"Starting Lean validation process for: {unique_name} using shared library.")
    effective_db_path = db_path or DEFAULT_DB_PATH
    loop = asyncio.get_running_loop()

    # --- Check Configuration ---
    if not SHARED_LIB_PATH or not SHARED_LIB_PATH.is_dir():
         logger.error("Shared library path not configured correctly or directory does not exist.")
         return False, "Shared library path configuration error."
    logger.debug(f"Using shared library: {SHARED_LIB_PATH} (Module: {SHARED_LIB_MODULE_NAME})")


    # 1. Fetch Target Item (Only target needed initially for source code)
    target_item: Optional[KBItem] = None
    try:
        target_item = get_kb_item_by_name(unique_name, db_path=effective_db_path)
        if target_item is None:
            logger.error(f"Target item '{unique_name}' not found.")
            return False, f"Target item '{unique_name}' not found."
        if not target_item.lean_code:
             logger.error(f"Target item '{unique_name}' has no lean_code to compile.")
             # Try to update status, but main failure is missing code
             try:
                 target_item.update_status(ItemStatus.ERROR, "Missing lean_code for compilation.")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
             except Exception: pass # Ignore DB error if item already missing code
             return False, "Missing lean_code for compilation."

        # We don't strictly need to fetch *all* dependencies anymore for the temp build,
        # as Lake will resolve them via the 'require' statement.
        # However, _generate_imports_for_target still uses item.plan_dependencies,
        # so the target_item fetched here should have them populated correctly.
        logger.info(f"Fetched target item {unique_name}.")

    except FileNotFoundError as e: # Should be caught by initial check
         logger.error(f"Error fetching target item {unique_name}: {e}")
         return False, f"Error fetching target: {e}"
    except Exception as e:
        logger.exception(f"Unexpected error fetching target item {unique_name}: {e}")
        # Try to update status if possible
        if target_item and isinstance(target_item, KBItem):
            try:
                target_item.update_status(ItemStatus.ERROR, f"Unexpected fetch error: {e}")
                await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception: pass
        return False, f"Unexpected error fetching target: {e}"

    # 2. Create Temporary Environment for Verification
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_base = temp_dir_obj.name
    temp_project_path_str : Optional[str] = None
    target_temp_module_name : Optional[str] = None
    temp_lib_name_used = "TempVerifyLib" # Define here or pass if needed

    try:
        logger.info(f"Creating temporary verification environment in: {temp_dir_base}")
        temp_project_path_str, target_temp_module_name = _create_temp_env_for_verification(
             target_item, temp_dir_base, SHARED_LIB_PATH, SHARED_LIB_MODULE_NAME, temp_lib_name_used
        )
        logger.info(f"Temporary Lake project created at: {temp_project_path_str}")
        logger.info(f"Target module name for temporary build: {target_temp_module_name}")

    except (ValueError, OSError, TypeError) as e:
         logger.error(f"Failed to create temporary verification environment for {unique_name}: {e}")
         # Attempt to update status if target_item exists
         if target_item:
             try:
                 target_item.update_status(ItemStatus.ERROR, f"Verification env creation error: {e}")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
             except Exception as db_err: logger.error(f"Failed to update status after env creation error: {db_err}")
         try: temp_dir_obj.cleanup()
         except Exception: pass
         return False, f"Verification environment creation error: {e}"
    except Exception as e: # Catch any other unexpected error during setup
        logger.exception(f"Unexpected error during verification environment setup for {unique_name}: {e}")
        if target_item:
            try:
                target_item.update_status(ItemStatus.ERROR, f"Unexpected env setup error: {e}")
                await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception as db_err: logger.error(f"Failed to update status after unexpected env setup error: {db_err}")
        try: temp_dir_obj.cleanup()
        except Exception: pass
        return False, f"Unexpected setup error: {e}"

    # Check if paths were successfully created before proceeding
    if not temp_project_path_str or not target_temp_module_name:
        logger.error(f"Verification environment setup failed to return valid paths for {unique_name}.")
        if target_item:
            try:
                 target_item.update_status(ItemStatus.ERROR, "Internal error: Env setup path generation failed.")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
            except Exception: pass
        try: temp_dir_obj.cleanup()
        except Exception: pass
        return False, "Internal error: Environment setup failed."

    # 3. Prepare and Execute Lake Build (Temporary Verification)
    try:
        # --- Prepare Environment for Subprocess ---
        subprocess_env = os.environ.copy()
        # LEAN_PATH might not be strictly necessary if dependencies are resolved via 'require',
        # but setting it for the stdlib is still good practice.
        # (LEAN_PATH detection logic remains the same as before)
        # --- Set LEAN_PATH ---
        std_lib_path = None
        lean_executable_path_obj = None
        try:
            lean_executable_path_obj = pathlib.Path(lake_executable_path).resolve() # Resolve path
            lean_exe_candidate = lean_executable_path_obj.parent / 'lean'
            lean_exe = str(lean_exe_candidate) if lean_exe_candidate.is_file() else 'lean'
        except Exception as path_err:
            logger.warning(f"Could not resolve lean executable path from lake path '{lake_executable_path}': {path_err}. Assuming 'lean' is in PATH.")
            lean_exe = 'lean'

        if lean_exe == 'lean' and not shutil.which('lean'):
             logger.warning("Could not find 'lean' executable relative to lake or in PATH. Cannot detect stdlib path.")
        else:
            try:
                logger.debug(f"Attempting to detect stdlib path using: {lean_exe} --print-libdir")
                lean_path_proc = subprocess.run(
                    [lean_exe, '--print-libdir'],
                    capture_output=True, text=True, check=True, timeout=10, encoding='utf-8'
                )
                path_candidate = lean_path_proc.stdout.strip()
                if path_candidate and pathlib.Path(path_candidate).is_dir():
                     std_lib_path = path_candidate
                     logger.info(f"Detected Lean standard library path: {std_lib_path}")
                else:
                     logger.warning(f"Lean command '{lean_exe} --print-libdir' did not return a valid directory path: '{path_candidate}'")
            except FileNotFoundError:
                 logger.warning(f"'{lean_exe}' not found when trying to detect stdlib path.")
            except subprocess.TimeoutExpired:
                 logger.warning(f"Timeout trying to detect stdlib path using '{lean_exe} --print-libdir'.")
            except subprocess.CalledProcessError as e:
                 logger.warning(f"Error running '{lean_exe} --print-libdir' (return code {e.returncode}): {e.stderr or e.stdout}")
            except Exception as e:
                logger.warning(f"Could not automatically detect Lean stdlib path using '{lean_exe} --print-libdir': {e}")

        if not std_lib_path and lean_executable_path_obj: # Fallback stdlib detection
            try:
                 toolchain_dir = lean_executable_path_obj.parent.parent
                 fallback_path = toolchain_dir / "lib" / "lean"
                 if fallback_path.is_dir():
                     std_lib_path = str(fallback_path.resolve())
                     logger.warning(f"Assuming stdlib path relative to toolchain: {std_lib_path}")
            except NameError:
                 logger.warning("Cannot attempt fallback stdlib detection due to earlier path resolution error.")
            except Exception as fallback_e:
                 logger.warning(f"Error during fallback stdlib path detection: {fallback_e}")

        if std_lib_path:
            existing_lean_path = subprocess_env.get('LEAN_PATH')
            separator = os.pathsep
            if existing_lean_path: subprocess_env['LEAN_PATH'] = f"{std_lib_path}{separator}{existing_lean_path}"
            else: subprocess_env['LEAN_PATH'] = std_lib_path
            logger.debug(f"Setting LEAN_PATH for subprocess: {subprocess_env['LEAN_PATH']}")
        else: logger.warning("Stdlib path not found or detection failed, LEAN_PATH will not include it.")


        # --- Set LAKE_HOME from Environment Variable (Still useful for external caching if any needed by shared lib) ---
        persistent_cache_path = os.getenv('LEAN_AUTOMATOR_LAKE_CACHE')
        if persistent_cache_path:
            persistent_cache_path_obj = pathlib.Path(persistent_cache_path).resolve()
            # Ensure directory exists (using sync os call within run_in_executor)
            if not await loop.run_in_executor(None, persistent_cache_path_obj.is_dir):
                logger.info(f"Persistent Lake cache directory specified but does not exist: {persistent_cache_path_obj}")
                try:
                    await loop.run_in_executor(None, functools.partial(os.makedirs, persistent_cache_path_obj, exist_ok=True))
                    logger.info(f"Successfully created persistent Lake cache directory: {persistent_cache_path_obj}")
                except OSError as e:
                    logger.error(f"Failed to create persistent Lake cache directory '{persistent_cache_path_obj}': {e}.") # Removed potentially confusing part
            subprocess_env['LAKE_HOME'] = str(persistent_cache_path_obj)
            logger.info(f"Setting LAKE_HOME for subprocess: {persistent_cache_path_obj}")
        else:
            logger.warning("LEAN_AUTOMATOR_LAKE_CACHE environment variable not set. Lake will use default caching behavior.")


        # --- Execute Lake Build (Async Subprocess in Temp Dir) ---
        command = [lake_executable_path, 'build', target_temp_module_name] # Build the temp module name
        lake_process_result: Optional[subprocess.CompletedProcess] = None
        full_output = ""
        logger.info(f"Running verification build: {' '.join(command)} in {temp_project_path_str}")

        try:
            run_args = {
                 'args': command, 'capture_output': True, 'text': True,
                 'timeout': timeout_seconds, 'encoding': 'utf-8', 'errors': 'replace',
                 'cwd': temp_project_path_str, 'check': False,
                 'env': subprocess_env
            }
            lake_process_result = await loop.run_in_executor(None, functools.partial(subprocess.run, **run_args))

            stdout_output = lake_process_result.stdout or ""
            stderr_output = lake_process_result.stderr or ""
            logger.debug(f"Verification build return code: {lake_process_result.returncode}")
            if stdout_output: logger.debug(f"Verification build stdout:\n{stdout_output}")
            if stderr_output: logger.debug(f"Verification build stderr:\n{stderr_output}")
            full_output = f"--- STDOUT ---\n{stdout_output}\n--- STDERR ---\n{stderr_output}"

        except subprocess.TimeoutExpired:
            logger.error(f"Verification build timed out after {timeout_seconds} seconds for {unique_name}.")
            if target_item:
                 target_item.update_status(ItemStatus.ERROR, f"Timeout after {timeout_seconds}s during verification build")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
            return False, f"Timeout after {timeout_seconds}s"
        except FileNotFoundError:
             logger.error(f"Lake executable '{lake_executable_path}' not found.")
             return False, f"Lake executable not found at '{lake_executable_path}'"
        except Exception as e:
            logger.exception(f"Unexpected error running verification build subprocess for {unique_name}: {e}")
            if target_item:
                 target_item.update_status(ItemStatus.ERROR, f"Subprocess execution error: {e}")
                 await save_kb_item(target_item, client=None, db_path=effective_db_path)
            return False, f"Subprocess error during verification build: {e}"

        # --- 4. Process Verification Results ---
        if not lake_process_result:
             error_msg = "Internal error: Lake process result missing after execution."
             logger.error(error_msg)
             if target_item:
                  target_item.update_status(ItemStatus.ERROR, error_msg)
                  await save_kb_item(target_item, client=None, db_path=effective_db_path)
             return False, error_msg

        if lake_process_result.returncode == 0:
             # Verification successful!
             logger.info(f"Successfully verified {unique_name} in temporary environment.")
             # Update DB status first
             target_item.update_status(ItemStatus.PROVEN, error_log=None) # Clear error log on success
             # target_item.update_olean(None) # Ensure olean field is cleared/ignored
             await save_kb_item(target_item, client=None, db_path=effective_db_path)
             logger.info(f"Database status updated to PROVEN for {unique_name}.")

             # Now, update the persistent library
             persist_success = await _update_persistent_library(
                 target_item, SHARED_LIB_PATH, SHARED_LIB_MODULE_NAME,
                 lake_executable_path, timeout_seconds, loop
             )
             if persist_success:
                 logger.info(f"Successfully updated persistent library with {unique_name}.")
                 return True, f"Lean code verified. Status: {ItemStatus.PROVEN.name}. Persistent library updated."
             else:
                 # Logged as error in helper, but verification itself succeeded.
                 logger.warning(f"Verification succeeded for {unique_name}, but failed to update/build persistent library.")
                 # Return success for verification, but maybe indicate the persistent lib issue.
                 return True, f"Lean code verified. Status: {ItemStatus.PROVEN.name}. WARNING: Persistent library update failed (see logs)."

        else: # Verification build failed
            final_error_message = full_output.strip() if full_output else "Unknown Lake build error (no output captured)."
            exit_code = lake_process_result.returncode
            logger.warning(f"Lean verification failed for {unique_name}. Exit code: {exit_code}")
            logger.debug(f"Captured output for failed verification build of {unique_name}:\n{full_output}")

            target_item.update_status(ItemStatus.LEAN_VALIDATION_FAILED, error_log=final_error_message)
            # target_item.update_olean(None) # Ensure olean field is cleared/ignored
            target_item.increment_failure_count()
            await save_kb_item(target_item, client=None, db_path=effective_db_path)
            return False, f"Lean validation failed (Exit code: {exit_code}). See logs or KBItem error log for details."

    # --- General Exception Handling for the whole process ---
    except Exception as e:
         logger.exception(f"Unhandled exception during check_and_compile for {unique_name}: {e}")
         if target_item and isinstance(target_item, KBItem) and getattr(target_item, 'status', None) != ItemStatus.PROVEN:
              try:
                  target_item.update_status(ItemStatus.ERROR, f"Unhandled exception: {e}")
                  # target_item.update_olean(None) # Ensure olean field is cleared/ignored
                  await save_kb_item(target_item, client=None, db_path=effective_db_path)
              except Exception as db_err: logger.error(f"Failed to update item status after unhandled exception: {db_err}")
         return False, f"Unhandled exception: {e}"
    finally:
        # --- Cleanup Temporary Directory ---
        try:
            if 'temp_dir_obj' in locals() and temp_dir_obj:
                 # Use run_in_executor for potentially blocking cleanup
                 await loop.run_in_executor(None, temp_dir_obj.cleanup)
                 logger.debug(f"Successfully cleaned up temporary directory: {temp_dir_base}")
        except Exception as cleanup_err:
            logger.error(f"Error cleaning up temporary directory {temp_dir_base}: {cleanup_err}")