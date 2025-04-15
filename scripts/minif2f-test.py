#!/usr/bin/env python3
import asyncio
import json
import pathlib
import sys
import os
import uuid
import subprocess # Using asyncio.create_subprocess_exec later

# --- Dependencies from lean_automator ---
# Assumption: lean_automator is installed or accessible in the PYTHONPATH.
# Adjust the imports based on your project structure if necessary.
try:
    # Import the application config singleton and potentially specific accessors
    from lean_automator.config.loader import APP_CONFIG, get_gemini_api_key
    from lean_automator.llm.caller import GeminiClient
    from lean_automator.lean.llm_interface import extract_lean_code
    from lean_automator.lean.lsp_analyzer import analyze_lean_failure
except ImportError as e:
    print(f"Error importing from lean_automator: {e}")
    print("Please ensure lean_automator is installed or accessible in your PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Script-Specific Configuration ---
# These settings are specific to this test script and are not typically
# part of the main application config loaded into APP_CONFIG.
# ---------------------------------------------

# --- Determine Project Paths Dynamically ---
# Assumes this script is located in a subdirectory (e.g., 'scripts')
# directly under the main project root.
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent # Go one level up from script dir
if not (PROJECT_ROOT / "pyproject.toml").exists(): # Basic check for project root
     print(f"Warning: Could not reliably determine project root from script location {SCRIPT_DIR}. Adjust PROJECT_ROOT if needed.")
     # Fallback or error, depending on requirements
     # PROJECT_ROOT = pathlib.Path("/absolute/path/to/vantage") # Example fallback


# Problem Selection
PROBLEM_ID_TO_TEST = "amc12a_2019_21" # <<< EDIT THIS: The ID of the minif2f problem to test

# File Paths for Test Data and Environment (Derived from Project Root)
TEST_ENV_DIR_NAME = "minif2f_lean_test_env" # Name of the test env directory
MINIF2F_DATA_FILENAME = "minif2f_lean4.jsonl" # Name of the data file

TEST_ENV_DIR = PROJECT_ROOT / TEST_ENV_DIR_NAME
MINIF2F_DATA_PATH = TEST_ENV_DIR / MINIF2F_DATA_FILENAME

# Subdirectory within TEST_ENV_DIR for temporary source files
TEST_SRC_SUBDIR = "MiniF2F"

# Execution Parameters
TIMEOUT_SECONDS = 180 # Timeout in seconds for Lean verification

# --- Fixed Header ---
# This header will be used for ALL problems, ignoring the one in the JSONL file.
FIXED_LEAN_HEADER = """\
import MiniF2F.Minif2fImport
open BigOperators Real Nat Topology
"""
# ---------------------------------------------
# END OF SCRIPT-SPECIFIC CONFIGURATION SECTION


# --- Configuration from lean_automator.config.loader ---
# General configuration like model names, API keys (via env), etc.,
# should be sourced from APP_CONFIG or helper functions.

# Get the LLM model name from the loaded application configuration
# Provides a fallback if the config structure is missing keys
MODEL_NAME = APP_CONFIG.get("llm", {}).get("default_gemini_model", "gemini-1.5-flash-latest")
# Note: The GeminiClient might internally use other APP_CONFIG values
# like 'gemini_max_retries', 'gemini_backoff_factor'.
# The API key is typically handled by the underlying google-generativeai library
# using the GOOGLE_API_KEY env var (which dotenv loads via loader.py)
# or potentially via get_gemini_api_key() if GeminiClient uses it explicitly.


# --- Helper Functions ---

def load_minif2f_data(jsonl_filepath: pathlib.Path) -> list[dict]:
    """Loads minif2f problems from a JSON Lines file."""
    data = []
    if not jsonl_filepath.is_file():
        print(f"Error: Cannot find minif2f data file: {jsonl_filepath}")
        return []
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        problem_data = json.loads(line)
                        data.append(problem_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line {i+1}: {line.strip()} - Error: {e}")
    except Exception as e:
        print(f"Error reading or parsing {jsonl_filepath}: {e}")
    print(f"Loaded {len(data)} problems from {jsonl_filepath.name}")
    return data

async def run_lean_verification(
    lean_code: str,
    problem_id: str,
    test_env_dir: pathlib.Path,
    test_src_subdir: str,
    timeout: int
) -> tuple[bool, str, str]:
    """
    Writes Lean code to a temporary file within the specified test environment
    and attempts to compile it using 'lake build'.

    Args:
        lean_code: The full Lean code string (including fixed header and theorem body).
        problem_id: The identifier of the problem being tested.
        test_env_dir: Path object pointing to the root of the Lean test environment directory.
        test_src_subdir: The name of the subdirectory within test_env_dir where source files reside.
        timeout: Maximum time in seconds to allow for the verification process.

    Returns:
        A tuple containing:
        - bool: True if verification succeeded (lake build returned 0), False otherwise.
        - str: The standard output from the lake build process.
        - str: The standard error from the lake build process (or timeout/exception message).
    """
    src_dir = test_env_dir / test_src_subdir
    src_dir.mkdir(exist_ok=True) # Ensure the source subdirectory exists

    # Create a safe filename from the problem ID and add a UUID for uniqueness
    safe_problem_id = "".join(c if c.isalnum() else "_" for c in problem_id)
    temp_filename_base = f"temp_{safe_problem_id}_{uuid.uuid4().hex[:8]}"
    temp_lean_file = src_dir / f"{temp_filename_base}.lean"

    # Construct the Lean module name based on directory structure
    module_name = f"{test_src_subdir}.{temp_filename_base}"

    # Use 'lake' command directly, assuming it's in the system PATH
    lake_executable = "lake"

    print(f"Attempting verification for {problem_id} in {temp_lean_file}...")
    try:
        # Write the assembled Lean code to the temporary file
        with open(temp_lean_file, 'w', encoding='utf-8') as f:
            f.write(lean_code)

        # Execute 'lake build <module_name>' within the test environment directory
        print(f"Running: {lake_executable} build {module_name} in {test_env_dir}")
        process = await asyncio.create_subprocess_exec(
            lake_executable, 'build', module_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(test_env_dir) # Set the working directory for lake
        )

        # Wait for the process to complete or timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Error: Lean verification timed out after {timeout} seconds.")
            # Attempt to kill the timed-out process
            try:
                process.kill()
            except ProcessLookupError:
                pass # Process already terminated
            await process.wait() # Ensure resources are cleaned up
            return False, "", f"TimeoutError: Verification exceeded {timeout} seconds."
        except Exception as comm_err:
             # Handle potential errors during communication (less common)
             print(f"Error during process communication: {comm_err}")
             try:
                 process.kill()
             except ProcessLookupError:
                 pass
             await process.wait()
             return False, "", f"Communication Error: {comm_err}"

        # Decode stdout and stderr
        stdout = stdout_bytes.decode('utf-8', errors='ignore')
        stderr = stderr_bytes.decode('utf-8', errors='ignore')

        # Check the return code of the lake process
        success = (process.returncode == 0)

        if success:
            print(f"Verification SUCCESSFUL for {problem_id} (Return Code: {process.returncode})")
        else:
            print(f"Verification FAILED for {problem_id} (Return Code: {process.returncode})")

        return success, stdout, stderr

    except Exception as e:
        # Catch any other exceptions during the process setup or file writing
        print(f"Error during Lean verification process: {e}")
        return False, "", f"Exception during verification: {e}"
    finally:
        # Ensure the temporary file is deleted even if errors occur
        if 'temp_lean_file' in locals() and temp_lean_file.exists():
            try:
                temp_lean_file.unlink()
            except OSError as e:
                # Log a warning if deletion fails, but don't crash
                print(f"Warning: Could not delete temporary file {temp_lean_file}: {e}")

async def run_lsp_analysis(
    lean_code: str,
    problem_id: str,
    test_env_dir: pathlib.Path,
    build_stderr: str
) -> str:
    """
    Runs the LSP analyzer (from lean_automator) on failed Lean code to get diagnostics.

    Args:
        lean_code: The full Lean code string that failed verification.
        problem_id: The identifier of the problem.
        test_env_dir: Path object pointing to the root of the Lean test environment directory.
        build_stderr: The stderr output from the failed 'lake build' command, used as fallback.

    Returns:
        A string containing the LSP analysis feedback or an error message.
    """
    # Use 'lean' command directly, assuming it's in the system PATH
    lean_executable = "lean"

    print(f"Running LSP analysis for failed problem {problem_id}...")
    try:
        # Call the imported analyzer function
        analysis_result = await analyze_lean_failure(
            lean_code=lean_code,
            lean_executable_path=lean_executable, # Pass the command name
            cwd=str(test_env_dir), # Analyzer needs the context of the project
            shared_lib_path=None, # Assuming not needed unless specified by lean_automator docs
            fallback_error=build_stderr # Provide build error as context
        )
        return analysis_result
    except Exception as e:
        # Catch errors specifically from the analysis step
        print(f"Error during LSP analysis: {e}")
        return f"LSP analysis failed with exception: {e}"

# --- Main Execution Logic ---

async def main():
    """
    Main asynchronous function to orchestrate the theorem proving test.
    """
    # --- Configuration Setup ---
    problem_id_to_run = PROBLEM_ID_TO_TEST
    data_file_path_obj = MINIF2F_DATA_PATH
    test_env_path_obj = TEST_ENV_DIR
    test_src_subdir_name = TEST_SRC_SUBDIR
    timeout_val = TIMEOUT_SECONDS
    model_name_from_config = MODEL_NAME

    # --- Path Validation ---
    if not data_file_path_obj.is_file():
        print(f"Error: Data file not found at {data_file_path_obj}")
        sys.exit(1)
    if not test_env_path_obj.is_dir():
        print(f"Error: Lean test environment directory not found at {test_env_path_obj}")
        sys.exit(1)
    if not (test_env_path_obj / "lakefile.lean").is_file():
         print(f"Error: No 'lakefile.lean' found in {test_env_path_obj}. This is required for 'lake build'.")
         sys.exit(1)

    # --- Load Problem Data ---
    all_problems = load_minif2f_data(data_file_path_obj)
    if not all_problems:
        print("Exiting due to failure loading problems.")
        sys.exit(1)

    # --- Select Problem ---
    problem_to_test = next((p for p in all_problems if p.get("id") == problem_id_to_run), None)

    if not problem_to_test:
        print(f"Error: Problem with ID '{problem_id_to_run}' not found in {data_file_path_obj.name}.")
        sys.exit(1)

    # --- Extract Problem Details ---
    problem_id = problem_to_test.get("id")
    formal_statement = problem_to_test.get("formal_statement")
    # The 'header' field from the JSONL is intentionally ignored here.

    # Validate extracted fields
    if not all([problem_id, formal_statement]):
        print(f"Error: Missing required fields (id, formal_statement) for problem {problem_id}")
        sys.exit(1)

    print(f"--- Testing Problem: {problem_id} ---")
    print(f"Model: {model_name_from_config}")
    print(f"Lean Env: {test_env_path_obj}")
    print(f"Using Fixed Header:\n{FIXED_LEAN_HEADER}")

    # --- Initialize LLM Client ---
    try:
        # Assumes GeminiClient uses GOOGLE_API_KEY env var or other config internally
        client = GeminiClient(default_generation_model=model_name_from_config)
    except Exception as e:
        print(f"Error initializing GeminiClient: {e}")
        sys.exit(1)

    # --- Prepare LLM Prompt ---
    # The prompt asks the LLM *only* for the theorem body, without imports/opens.
    # The FIXED_LEAN_HEADER will be prepended later.
    prompt = f"""
You are an expert Lean 4 programmer using mathlib4.
Your goal is to complete the following Lean theorem statement by replacing ':= sorry' with a valid proof.
The necessary imports and open directives will be provided separately.

**CRUCIAL SYNTAX INSTRUCTIONS:**
* You **MUST** use Lean 4 syntax and tactics exclusively.
* Tactic proofs **MUST** start with the `by` keyword (e.g., `theorem name : Type := by tactic1 tactic2 ...`).
* You **MUST NOT** use any Lean 3 syntax.
* Specifically, **DO NOT use `begin...end` blocks.** Using `begin...end` will result in incorrect code.
* **Do NOT add any new `import` statements.** Use only the imports provided externally.

**Instructions for Proof Steps:**
Before each tactic line in your proof (e.g., `rw [...]`, `apply ...`, `simp`, `exact ...`), add two comments on the preceding lines:
1.  A comment starting with `-- Goal:` describing the current proof goal state shown by Lean.
2.  A comment starting with `-- Action:` describing the action the *next* tactic performs to modify the goal.

Example of desired output format (using Lean 4 `by` syntax and comments):
```lean
theorem example_proof (a b : Nat) (h : a = b) : a + 0 = b := by
  -- Goal: a + 0 = b
  -- Action: Rewrite using the equation h (a = b).
  rw [h]
  -- Goal: b + 0 = b
  -- Action: Simplify the expression using the definition of addition with zero.
  simp only [Nat.add_zero]
  -- Goal: b = b
  -- Action: The goal is now trivial, apply reflexivity.
  rfl
```

**Theorem to Prove (replace `sorry`):**
```lean
{formal_statement}
```

Provide ONLY the Lean code starting from `theorem ...` including the signature and the proof tactics (with comments as instructed) replacing `:= sorry`.
Start your response with ```lean and end it with ```. Do NOT include `import` or `open` lines.
"""

    # --- Call LLM for Proof Generation ---
    print("Generating proof with LLM...")
    try:
        llm_response = await client.generate(prompt, model=model_name_from_config)
    except Exception as e:
        print(f"Error calling GeminiClient generate: {e}")
        sys.exit(1)

    # --- Extract Code from LLM Response ---
    print("Extracting code from LLM response...")
    # Expects only the theorem definition and proof body (e.g., "theorem ... := by ...")
    extracted_theorem_body = extract_lean_code(llm_response)

    if not extracted_theorem_body:
        print("Error: Could not extract Lean code block (```lean ... ```) from LLM response:")
        print("--- LLM Response ---")
        print(llm_response)
        print("--------------------")
        sys.exit(1)

    # --- Assemble Full Lean Code ---
    # Combine the predefined FIXED_LEAN_HEADER with the LLM's generated theorem body.
    full_lean_code_to_verify = f"{FIXED_LEAN_HEADER}\n\n{extracted_theorem_body}"

    print("--- Assembled Code for Verification ---")
    print(full_lean_code_to_verify)
    print("---------------------------------------")

    # --- Run Lean Verification ---
    success, stdout, stderr = await run_lean_verification(
        lean_code=full_lean_code_to_verify,
        problem_id=problem_id,
        test_env_dir=test_env_path_obj,
        test_src_subdir=test_src_subdir_name,
        timeout=timeout_val
    )

    # --- Report Results & Analyze Failures ---
    print("\n--- Result ---")
    if success:
        print(f"✅ SUCCESS: Proof for {problem_id} verified successfully.")
    else:
        print(f"❌ FAILURE: Proof for {problem_id} failed verification.")
        print("--- Build Error (stderr) ---")
        print(stderr if stderr else "(No stderr captured)")
        print("--------------------------")

        # Run LSP analysis on the failed code
        lsp_feedback = await run_lsp_analysis(
            lean_code=full_lean_code_to_verify, # Analyze the full code that failed
            problem_id=problem_id,
            test_env_dir=test_env_path_obj,
            build_stderr=stderr # Pass stderr from build failure
        )
        print("\n--- LSP Analysis Feedback ---")
        print(lsp_feedback)
        print("---------------------------")

    print(f"--- Finished Problem: {problem_id} ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure lean_automator modules can be found if the script is run directly
    # and lean_automator is not installed system-wide or in the standard PYTHONPATH.
    # This adds the parent directory of the script's location (PROJECT_ROOT) to sys.path.
    if str(PROJECT_ROOT) not in sys.path:
         sys.path.insert(0, str(PROJECT_ROOT))
         print(f"Added {PROJECT_ROOT} to sys.path to locate lean_automator modules.")

    # Check if APP_CONFIG was loaded successfully by the import mechanism
    if "APP_CONFIG" not in globals():
         print("Error: APP_CONFIG not loaded. Check lean_automator.config.loader import and configuration setup.")
         sys.exit(1)
    print("APP_CONFIG loaded successfully.")

    # Run the main asynchronous function
    asyncio.run(main())
