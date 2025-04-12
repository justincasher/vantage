# File: scripts/test_lean_lsp_analyzer.py

import asyncio
import logging
import os
import pathlib
import shutil
import sys
import time

# Removed tempfile import
# --- Load Environment Variables or Exit ---
from dotenv import load_dotenv

env_loaded_successfully = load_dotenv()

if not env_loaded_successfully:
    print("\nCRITICAL ERROR: Could not find or load the .env file.", file=sys.stderr)
    print(
        "Please ensure a .env file exists in the current directory or parent.",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Add project root to path to allow imports ---
project_root = os.getcwd()
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# --- End Path Setup ---

# --- Module Imports ---
try:
    from lean_automator.lean.lsp_analyzer import analyze_lean_failure
except ImportError as e:
    print(f"Error importing project modules: {e}", file=sys.stderr)
    print(
        "Ensure the script is run from the project root directory ('vantage')",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Configuration ---
LAKE_EXECUTABLE_PATH_STR = os.getenv("LAKE_EXECUTABLE_PATH", "lake")
LEAN_AUTOMATOR_SHARED_LIB_PATH_STR = os.getenv("LEAN_AUTOMATOR_SHARED_LIB_PATH")

if not LEAN_AUTOMATOR_SHARED_LIB_PATH_STR:
    print(
        "CRITICAL ERROR: 'LEAN_AUTOMATOR_SHARED_LIB_PATH' not found in environment.",
        file=sys.stderr,
    )
    print("Please set it in your .env file.", file=sys.stderr)
    sys.exit(1)

try:
    SHARED_LIB_ROOT_PATH = pathlib.Path(LEAN_AUTOMATOR_SHARED_LIB_PATH_STR).resolve(
        strict=True
    )  # Use strict=True to ensure it exists
except FileNotFoundError:
    print(
        f"CRITICAL ERROR: LEAN_AUTOMATOR_SHARED_LIB_PATH "
        f"'{LEAN_AUTOMATOR_SHARED_LIB_PATH_STR}' does not exist or is not accessible.",
        file=sys.stderr,
    )
    sys.exit(1)
except Exception as path_e:
    print(
        f"CRITICAL ERROR: Error resolving LEAN_AUTOMATOR_SHARED_LIB_PATH "
        f"'{LEAN_AUTOMATOR_SHARED_LIB_PATH_STR}': {path_e}",
        file=sys.stderr,
    )
    sys.exit(1)

if not SHARED_LIB_ROOT_PATH.is_dir():
    print(
        f"CRITICAL ERROR: LEAN_AUTOMATOR_SHARED_LIB_PATH "
        f"'{SHARED_LIB_ROOT_PATH}' is not a directory.",
        file=sys.stderr,
    )
    sys.exit(1)


LSP_TIMEOUT_SECONDS = int(os.getenv("LSP_ANALYZER_TIMEOUT_SECONDS", "60"))

# --- Logging Setup ---
log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
# Slightly rename logger to reflect the new approach
logger = logging.getLogger("TestLeanLspAnalyzerInVantageLib")

# --- Failing Lean Code to Analyze ---
FAILING_LEAN_CODE = """
import Lean -- Not strictly needed for tactics, but good practice

theorem SimpleAndCommutes (p q : Prop) : p ∧ q → q ∧ p := by
  intro h_and -- Introduce hypothesis h_and : p ∧ q. Goal is now q ∧ p
  -- Try to rewrite using a lemma that doesn't exist.
  -- This is the point of failure.
  rw [non_existent_lemma] -- Error: unknown identifier 'non_existent_lemma'
  -- The following lines won't be effectively reached by the compiler,
  -- but we include them to see how the analyzer handles code post-error.
  cases h_and with
  | intro hp hq =>
    apply And.intro
    exact hq
    exact hp

-- Add another simple definition to see if analysis picks it up
def simpleVal : Nat := 5
"""

# Removed MINIMAL_LAKEFILE_CONTENT and LAKEFILE_CONTENT_WITH_DEP


async def main():
    """Main execution function to test lean_lsp_analyzer within vantage_lib."""
    logger.info("--- Starting Lean LSP Analyzer Test (within vantage_lib) ---")
    loop = asyncio.get_running_loop()

    # --- Determine Lean Executable Path ---
    lean_exe = "lean"  # Default value
    logger.info(f"Using Lake executable path hint: {LAKE_EXECUTABLE_PATH_STR}")
    try:
        # Find lean based on lake path or system PATH (logic unchanged)
        resolved_lake_path_str = await loop.run_in_executor(
            None, shutil.which, LAKE_EXECUTABLE_PATH_STR
        )
        if resolved_lake_path_str:
            logger.info(f"Found 'lake' executable at: {resolved_lake_path_str}")
            resolved_lake_path = pathlib.Path(resolved_lake_path_str)
            lean_exe_candidate = resolved_lake_path.parent / "lean"
            is_file = await loop.run_in_executor(None, lean_exe_candidate.is_file)
            if is_file:
                lean_exe = str(lean_exe_candidate)
                logger.info(f"Found 'lean' relative to 'lake': {lean_exe}")
            else:
                logger.debug(
                    "'lean' not found relative to 'lake'. Checking PATH for 'lean'."
                )
                lean_exe_from_path = await loop.run_in_executor(
                    None, shutil.which, "lean"
                )
                if lean_exe_from_path:
                    lean_exe = lean_exe_from_path
                    logger.info(f"Found 'lean' executable in PATH: {lean_exe}")
                else:
                    logger.warning(
                        "Could not find 'lean' relative to 'lake' or in PATH. "
                        "Using default 'lean'."
                    )
                    lean_exe = "lean"
        else:
            logger.warning(
                f"Could not find '{LAKE_EXECUTABLE_PATH_STR}' executable. "
                "Checking PATH directly for 'lean'."
            )
            lean_exe_from_path = await loop.run_in_executor(None, shutil.which, "lean")
            if lean_exe_from_path:
                lean_exe = lean_exe_from_path
                logger.info(f"Found 'lean' executable in PATH: {lean_exe}")
            else:
                logger.warning(
                    "Could not find 'lake' or 'lean' in PATH. Using default 'lean'."
                )
                lean_exe = "lean"
    except Exception as path_err:
        logger.error(
            f"Unexpected error determining lean executable path: {path_err}. "
            "Using default 'lean'."
        )
        lean_exe = "lean"
    logger.info(f"Using Lean executable path for LSP: {lean_exe}")
    # --- End Determine Lean Executable Path ---

    # --- Define Target File Path within vantage_lib ---
    # analyze_lean_failure internally uses "temp_analysis_file.lean" in the cwd
    temp_file_name = "temp_analysis_file.lean"
    temp_file_abs_path = SHARED_LIB_ROOT_PATH / temp_file_name
    logger.info(f"LSP analyzer will use temporary file: {temp_file_abs_path}")

    # Define a simple fallback error message
    fallback_error_msg = "Simulated build failure triggering LSP analysis."

    analysis_result = ""  # Initialize result var
    duration = 0.0
    start_time = None  # Initialize start_time

    try:
        # --- Time the analysis ---
        # analyze_lean_failure will handle writing the code to temp_file_abs_path
        # because we set cwd to SHARED_LIB_ROOT_PATH.
        logger.info(f"Calling analyze_lean_failure with cwd={SHARED_LIB_ROOT_PATH}")
        start_time = time.perf_counter()

        analysis_result = await analyze_lean_failure(
            lean_code=FAILING_LEAN_CODE,
            lean_executable_path=lean_exe,
            cwd=str(SHARED_LIB_ROOT_PATH),  # Use vantage_lib root as cwd
            shared_lib_path=SHARED_LIB_ROOT_PATH,  # Pass vantage_lib path here too
            timeout_seconds=LSP_TIMEOUT_SECONDS,
            fallback_error=fallback_error_msg,
        )

        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"analyze_lean_failure completed in {duration:.4f} seconds.")

    except Exception as e:
        logger.exception(f"An exception occurred during analyze_lean_failure: {e}")
        analysis_result = f"-- Analysis Failed due to Exception: {e} --"
        # Ensure timer ends if exception happens before end_time is set
        if start_time is not None and not duration:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"analyze_lean_failure failed after {duration:.4f} seconds.")

    finally:
        # --- Cleanup Temporary File ---
        logger.info(f"Attempting cleanup of temporary file: {temp_file_abs_path}")
        try:
            # Use run_in_executor for potentially blocking file check/removal
            if await loop.run_in_executor(None, temp_file_abs_path.is_file):
                await loop.run_in_executor(None, os.remove, temp_file_abs_path)
                logger.info(f"Successfully removed {temp_file_abs_path}")
            else:
                # This is expected if analyze_lean_failure failed before writing
                logger.warning(
                    f"Temporary file {temp_file_abs_path} not found during cleanup "
                    "(might be expected if analysis failed early)."
                )
        except Exception as cleanup_err:
            logger.error(f"Error during cleanup of {temp_file_abs_path}: {cleanup_err}")

    # --- Print the result ---
    print("\n" + "=" * 20 + " Analysis Result " + "=" * 20)
    print(analysis_result)
    print("=" * (40 + len(" Analysis Result ")))

    logger.info("--- Lean LSP Analyzer Test (within vantage_lib) Finished ---")


if __name__ == "__main__":
    # Ensure the script is run from the project root
    if not os.path.exists("pyproject.toml") or not os.path.exists("src"):
        print(
            "ERROR: This script must be run from the project root directory "
            "('vantage').",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
