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
TEST_SRC_SUBDIR = "Minif2fTest"

# Execution Parameters
TIMEOUT_SECONDS = 180 # Timeout in seconds for Lean verification
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


# --- Helper Functions --- (Identical to previous version)

def load_minif2f_data(jsonl_filepath: pathlib.Path) -> list[dict]:
    """Loads minif2f problems from a JSON Lines file."""
    data = []
    # filepath = pathlib.Path(jsonl_filepath) # Path object now passed directly
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
    # lake_executable: str, # Removed
    timeout: int
) -> tuple[bool, str, str]:
    """Writes Lean code to a temporary file and attempts to compile it using lake build."""
    src_dir = test_env_dir / test_src_subdir
    src_dir.mkdir(exist_ok=True)
    safe_problem_id = "".join(c if c.isalnum() else "_" for c in problem_id)
    temp_filename_base = f"temp_{safe_problem_id}_{uuid.uuid4().hex[:8]}"
    temp_lean_file = src_dir / f"{temp_filename_base}.lean"
    module_name = f"{test_src_subdir}.{temp_filename_base}"
    lake_executable = "lake" # Use command name directly
    print(f"Attempting verification for {problem_id} in {temp_lean_file}...")
    try:
        with open(temp_lean_file, 'w', encoding='utf-8') as f:
            f.write(lean_code)
        print(f"Running: {lake_executable} build {module_name} in {test_env_dir}")
        process = await asyncio.create_subprocess_exec(
            lake_executable, 'build', module_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(test_env_dir)
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"Error: Lean verification timed out after {timeout} seconds.")
            try: process.kill()
            except ProcessLookupError: pass
            await process.wait()
            return False, "", f"TimeoutError: Verification exceeded {timeout} seconds."
        except Exception as comm_err:
             print(f"Error during process communication: {comm_err}")
             try: process.kill()
             except ProcessLookupError: pass
             await process.wait()
             return False, "", f"Communication Error: {comm_err}"
        stdout = stdout_bytes.decode('utf-8', errors='ignore')
        stderr = stderr_bytes.decode('utf-8', errors='ignore')
        success = (process.returncode == 0)
        if success: print(f"Verification SUCCESSFUL for {problem_id} (Return Code: {process.returncode})")
        else: print(f"Verification FAILED for {problem_id} (Return Code: {process.returncode})")
        return success, stdout, stderr
    except Exception as e:
        print(f"Error during Lean verification process: {e}")
        return False, "", f"Exception during verification: {e}"
    finally:
        if 'temp_lean_file' in locals() and temp_lean_file.exists():
            try: temp_lean_file.unlink()
            except OSError as e: print(f"Warning: Could not delete temporary file {temp_lean_file}: {e}")

async def run_lsp_analysis(
    lean_code: str,
    problem_id: str,
    test_env_dir: pathlib.Path,
    # lean_executable: str, # Removed
    build_stderr: str
) -> str:
    """Runs the LSP analyzer on failed code."""
    lean_executable = "lean" # Use command name directly
    print(f"Running LSP analysis for failed problem {problem_id}...")
    try:
        analysis_result = await analyze_lean_failure(
            lean_code=lean_code,
            lean_executable_path=lean_executable,
            cwd=str(test_env_dir),
            shared_lib_path=None,
            fallback_error=build_stderr
        )
        return analysis_result
    except Exception as e:
        print(f"Error during LSP analysis: {e}")
        return f"LSP analysis failed with exception: {e}"

# --- Main Execution Logic ---

async def main():
    # Use script-specific configuration variables defined at the top
    problem_id_to_run = PROBLEM_ID_TO_TEST
    # Paths are now pathlib objects derived dynamically
    data_file_path_obj = MINIF2F_DATA_PATH
    test_env_path_obj = TEST_ENV_DIR
    test_src_subdir_name = TEST_SRC_SUBDIR
    # lake_executable_path = LAKE_EXECUTABLE # Removed
    # lean_executable_path = LEAN_EXECUTABLE # Removed
    timeout_val = TIMEOUT_SECONDS

    # Use model name from APP_CONFIG
    model_name_from_config = MODEL_NAME # Uses variable defined above from APP_CONFIG

    # Validate paths (using the derived pathlib objects)
    if not data_file_path_obj.is_file():
        print(f"Error: Data file not found at {data_file_path_obj}")
        sys.exit(1)
    if not test_env_path_obj.is_dir():
        print(f"Error: Lean test environment directory not found at {test_env_path_obj}")
        sys.exit(1)
    if not (test_env_path_obj / "lakefile.lean").is_file():
         print(f"Error: No 'lakefile.lean' found in {test_env_path_obj}")
         sys.exit(1)

    # Load data (pass pathlib object)
    all_problems = load_minif2f_data(data_file_path_obj)
    if not all_problems:
        sys.exit(1)

    # Find the selected problem
    problem_to_test = next((p for p in all_problems if p.get("id") == problem_id_to_run), None)

    if not problem_to_test:
        print(f"Error: Problem with ID '{problem_id_to_run}' not found in data file.")
        sys.exit(1)

    # Extract necessary fields
    problem_id = problem_to_test.get("id")
    formal_statement = problem_to_test.get("formal_statement")
    header = problem_to_test.get("header") # Get the header provided in the data

    if not all([problem_id, formal_statement, header]):
        print(f"Error: Missing required fields (id, formal_statement, header) for problem {problem_id}")
        sys.exit(1)

    print(f"--- Testing Problem: {problem_id} ---")
    # Print the model name being used (sourced from APP_CONFIG)
    print(f"Model: {model_name_from_config}")
    print(f"Lean Env: {test_env_path_obj}") # Use the path object
    # Don't print the full statement here, as the LLM only needs the part after header
    # print(f"Statement:\n{formal_statement}\n") # Optional: print if needed

    # Initialize Gemini Client
    try:
        client = GeminiClient(default_generation_model=model_name_from_config)
    except Exception as e:
        print(f"Error initializing GeminiClient: {e}")
        sys.exit(1)

    # Format the prompt - ** UPDATED WITH LEAN 4 SYNTAX RULES **
    # Ask LLM only for the theorem body, excluding header/imports
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

    # Call LLM
    print("Generating proof with LLM...")
    try:
        llm_response = await client.generate(prompt, model=model_name_from_config)
    except Exception as e:
        print(f"Error calling GeminiClient generate: {e}")
        sys.exit(1)

    # Extract Lean code from response
    print("Extracting code from LLM response...")
    # This now expects only the theorem body (theorem... := ... end)
    extracted_theorem_body = extract_lean_code(llm_response)

    if not extracted_theorem_body:
        print("Error: Could not extract Lean code block (```lean ... ```) from LLM response:")
        print("--- LLM Response ---")
        print(llm_response)
        print("--------------------")
        sys.exit(1)

    # --- Assemble the full code for verification ---
    # Combine the header from JSONL with the LLM's generated theorem body
    full_lean_code_to_verify = f"{header}\n\n{extracted_theorem_body}"

    print("--- Assembled Code for Verification ---")
    print(full_lean_code_to_verify)
    print("---------------------------------------")

    # --- Verification ---
    success, stdout, stderr = await run_lean_verification(
        lean_code=full_lean_code_to_verify,
        problem_id=problem_id,
        test_env_dir=test_env_path_obj, # Use path object
        test_src_subdir=test_src_subdir_name,
        # lake_executable=lake_executable_path, # Removed
        timeout=timeout_val
    )

    # --- Reporting & Analysis --- (Identical to previous version)
    print("\n--- Result ---")
    if success:
        print(f"✅ SUCCESS: Proof for {problem_id} verified successfully.")
    else:
        print(f"❌ FAILURE: Proof for {problem_id} failed verification.")
        print("--- Build Error (stderr) ---")
        print(stderr if stderr else "(No stderr captured)")
        print("--------------------------")
        lsp_feedback = await run_lsp_analysis(
            lean_code=full_lean_code_to_verify, # Analyze the combined code
            problem_id=problem_id,
            test_env_dir=test_env_path_obj, # Use path object
            # lean_executable=lean_executable_path, # Removed
            build_stderr=stderr
        )
        print("\n--- LSP Analysis Feedback ---")
        print(lsp_feedback)
        print("---------------------------")

    print(f"--- Finished Problem: {problem_id} ---")


if __name__ == "__main__":
    # Ensure lean_automator modules can be found if script is run directly
    # and lean_automator is not installed system-wide.
    # This adds the parent directory of the script's location (PROJECT_ROOT) to sys.path,
    # assuming the script is e.g. inside 'scripts/'.
    if str(PROJECT_ROOT) not in sys.path:
         sys.path.insert(0, str(PROJECT_ROOT))
         print(f"Added {PROJECT_ROOT} to sys.path")

    # Ensure APP_CONFIG is loaded before running main
    if "APP_CONFIG" not in globals():
         print("Error: APP_CONFIG not loaded. Check config loader import.")
         sys.exit(1)
    print("APP_CONFIG loaded successfully.") # Optional confirmation

    asyncio.run(main())
