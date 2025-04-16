# File: scripts/minif2f_test.py

import asyncio
import json
import pathlib
import random  # For sampling problems
import re  # For potentially cleaning up LLM response
import sys
import uuid

# --- Dependencies from lean_automator ---
try:
    # Ensure lean_automator is accessible (e.g., in PYTHONPATH or installed)
    from lean_automator.config.loader import APP_CONFIG, get_gemini_api_key
    from lean_automator.lean.llm_interface import extract_lean_code
    from lean_automator.lean.lsp_analyzer import analyze_lean_failure
    from lean_automator.llm.caller import GeminiClient

except ImportError as e:
    print(f"Error importing from lean_automator: {e}")
    print("Please ensure lean_automator is installed or accessible in your PYTHONPATH.")
    # Determine project root relative to this script to add to path if necessary
    SCRIPT_DIR_FOR_IMPORT = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR_FOR_IMPORT.parent
    if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
        print(
            f"Added {PROJECT_ROOT_FOR_IMPORT} to sys.path. "
            "Please try running the script again."
        )
    else:
        print("Project root already in sys.path.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Script-Specific Configuration ---
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if not (PROJECT_ROOT / "pyproject.toml").exists():
    print(
        f"Warning: Could not reliably determine project root from script "
        f"location {SCRIPT_DIR}. Adjust PROJECT_ROOT if needed."
    )

# Problem Selection
NUM_PROBLEMS_TO_SAMPLE = (
    5  # <<< EDIT THIS - Number of problems to randomly sample and test
)

# File Paths
TEST_ENV_DIR_NAME = "minif2f_lean_test_env"
MINIF2F_DATA_FILENAME = "minif2f_lean4.jsonl"
TEST_ENV_DIR = PROJECT_ROOT / TEST_ENV_DIR_NAME
MINIF2F_DATA_PATH = TEST_ENV_DIR / MINIF2F_DATA_FILENAME
TEST_SRC_SUBDIR = "MiniF2F"

# Execution Parameters
TIMEOUT_SECONDS = 180
MAX_ITERS_PER_PROBLEM = 5  # Maximum number of LLM calls (iterations) per problem

# --- Fixed Header ---
# This header will be prepended to the LLM's generated proof tactics.
# The LLM should NOT generate this part.
FIXED_LEAN_HEADER = """\
import MiniF2F.Minif2fImport
open BigOperators Real Nat Topology
"""
# --- Configuration from lean_automator.config.loader ---
MODEL_NAME = APP_CONFIG.get("llm", {}).get(
    "default_gemini_model", "gemini-1.5-pro-preview-0409"
)  # Use a capable model
# ---------------------------------------------

# --- Prompt Templates (Revised with Consistent Instructions and Mandatory Comments) ---

# Initial prompt asks for the *next step* in the tactic block.
initial_prompt_template = """
You are an expert Lean 4 programmer using mathlib4. Your task is to propose the
next step in proving a given theorem.

**Instructions:**
1.  I will provide the theorem statement.
2.  You must provide the **next step (typically one or a few lines) of the
    proof tactic block** that should come *after* the `:= by` keyword or the
    previous tactic.
3.  **Output ONLY the Lean 4 tactics for the next step (e.g., a few lines of
    Lean).** Do NOT include the theorem signature (e.g., `theorem name (...) :
    ... := by`) or any import statements.
4.  Use Lean 4 syntax. Do NOT use `begin...end` or `sorry`.
5.  **Ensure proper tactic separation:** If your suggested step is followed by
    another tactic (not the end of the proof), make sure it ends appropriately
    (e.g., with a comma `,`) so the parser correctly recognizes the start of the
    next command.
6.  **Include extensive, descriptive comments before the tactic(s):** Explain
    the current state Lean needs to prove (you should infer this state) and the
    purpose of the tactic(s) you are proposing to address that state. Make your
    reasoning clear through comments.

**Theorem Statement:**
```lean
{{formal_statement}}
```

**Your Task:**
Propose the **next step** in the Lean 4 tactic block for the proof, starting
immediately after `:= by` (or continuing the existing proof). Enclose your
response ONLY in ```lean ... ```.

Example Response Format (for one step):
```lean
  -- The current goal is to show that n + 0 equals n.
  -- We can use the simplification tactic `simp` which automatically applies
  -- relevant lemmas, including the definition of addition with zero.
  simp
```
Or for a multi-line step:
```lean
  -- The goal is currently <some complex goal>.
  -- First, introduce the variables a and b, and the hypothesis h into the
  -- local context.
  intro a b h
  -- Now the goal is <goal after intro>.
  -- We can rewrite the goal using the equality provided by hypothesis h.
  rw [h]
```
"""

# Subsequent prompt asks for the *next step*, revising based on feedback.
subsequent_prompt_template = """
You are an expert Lean 4 programmer using mathlib4. Your task is to propose the
next step in proving a given theorem.

**Instructions:**
1.  Based *only* on the theorem statement and the Lean feedback provided above,
    provide the **next step (typically one or a few lines) for the proof tactic
    block**.
2.  **Output ONLY the Lean 4 tactics for the next step (e.g., the next few
    lines of Lean plus your previous attempt code).**
3.  Do NOT include the theorem signature or imports in your answer. Remember:
    Everything you write will go after "by".
4.  Your goal is to propose a tactic step that makes progress, addressing any
    errors or remaining goals shown in the feedback.
5.  Use Lean 4 syntax. Do not use `begin...end` or `sorry`.
6.  **Include extensive, descriptive comments before the tactic(s):** Explain
    the current state Lean needs to prove (considering the feedback) and the
    purpose of the tactic(s) you are proposing to address that state or correct
    previous errors. Make your reasoning clear through comments.
7.  Enclose your response ONLY in ```lean ... ```

**Theorem Statement:**
```lean
{{formal_statement}}
```

**Feedback from Previous Attempt:**
The following output comes from analyzing the previous proof attempt using Lean's
language server. It includes the code with annotations showing goal states,
errors, or other diagnostics. Use this feedback to determine the next step. You
should aim to directly incorporate this code into your proof.

```lean
{{feedback_section}}
```

**Your Task:**
Provide the **next step** for the Lean 4 tactic block, revising based on the
feedback and according to the instructions. Enclose your response ONLY in
```lean ... ```.

Example Response Format (revising a previous step or adding a new one):
```lean
  -- The feedback indicates the previous tactic failed because <reason>.
  -- The current goal state is <updated goal state based on feedback>.
  -- To address this, we will now try rewriting with `some_lemma` instead.
  -- This might be a correction or the next logical step
  rw [some_lemma]
```
"""


# --- Helper Functions ---


def load_minif2f_data(jsonl_filepath: pathlib.Path) -> list[dict]:
    """Loads minif2f problems from a JSON Lines file."""
    data = []
    if not jsonl_filepath.is_file():
        print(f"Error: Cannot find minif2f data file: {jsonl_filepath}")
        return []
    try:
        # UP015: No need for "r" mode explicitly
        with open(jsonl_filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        problem_data = json.loads(line)
                        # Basic validation
                        if "id" in problem_data and "formal_statement" in problem_data:
                            data.append(problem_data)
                        else:
                            print(
                                f"Warning: Skipping line {i + 1} due to missing 'id' "
                                f"or 'formal_statement': {line.strip()}"
                            )
                    except json.JSONDecodeError as e:
                        print(
                            f"Warning: Skipping invalid JSON line {i + 1}: "
                            f"{line.strip()} - Error: {e}"
                        )
    except Exception as e:
        print(f"Error reading or parsing {jsonl_filepath}: {e}")
    print(f"Loaded {len(data)} valid problems from {jsonl_filepath.name}")
    return data


async def run_lean_verification(
    lean_code: str,
    problem_id: str,
    iteration: int,
    test_env_dir: pathlib.Path,
    test_src_subdir: str,
    timeout: int,
) -> tuple[bool, str, str]:
    """
    Writes Lean code to a temporary file and attempts to compile it using 'lake build'.
    Modified to include iteration number in filename for easier debugging.
    """
    src_dir = test_env_dir / test_src_subdir
    src_dir.mkdir(exist_ok=True)

    safe_problem_id = "".join(c if c.isalnum() else "_" for c in problem_id)
    # Include iteration in filename
    temp_filename_base = (
        f"temp_{safe_problem_id}_iter{iteration}_{uuid.uuid4().hex[:8]}"
    )
    temp_lean_file = src_dir / f"{temp_filename_base}.lean"
    # Construct the Lean module name based on the subdirectory and filename
    module_name = f"{test_src_subdir}.{temp_filename_base}"
    lake_executable = "lake"  # Assumes lake is in PATH

    try:
        with open(temp_lean_file, "w", encoding="utf-8") as f:
            f.write(lean_code)

        process = await asyncio.create_subprocess_exec(
            lake_executable,
            "build",
            module_name,  # Pass module name correctly
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(test_env_dir),  # Ensure lake runs in the correct project directory
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            print(
                f"Error: Lean verification timed out after {timeout} seconds "
                f"for {module_name}."
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return False, "", f"TimeoutError: Verification exceeded {timeout} seconds."
        except Exception as comm_err:
            print(f"Error during process communication for {module_name}: {comm_err}")
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
            return False, "", f"Communication Error: {comm_err}"

        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")
        success = process.returncode == 0

        return success, stdout, stderr

    except FileNotFoundError:
        print(
            f"Error: '{lake_executable}' command not found. Make sure Lean and Lake "
            "are installed and in your system's PATH."
        )
        return False, "", f"ExecutableNotFound: '{lake_executable}' not found."
    except Exception as e:
        print(f"Error during Lean verification process for {module_name}: {e}")
        return False, "", f"Exception during verification: {e}"
    finally:
        # Keep temporary files for debugging if needed
        if "temp_lean_file" in locals() and temp_lean_file.exists():
            try:
                # pass # Keep files for debugging
                temp_lean_file.unlink()  # Uncomment to delete files automatically
            except OSError as e:
                print(f"Warning: Could not delete temporary file {temp_lean_file}: {e}")


async def run_lsp_analysis(
    lean_code: str,
    problem_id: str,
    iteration: int,
    test_env_dir: pathlib.Path,
    build_stderr: str,  # Pass stderr from build as potential fallback
) -> str:
    """
    Runs the LSP analyzer on Lean code (typically after a failed build or to
    check goals). Includes iteration number.
    """
    lean_executable = "lean"  # Assumes lean is in PATH
    try:
        # Assuming analyze_lean_failure from lean_automator handles LSP interaction
        # It should return a formatted string with code annotations (errors, goals)
        analysis_result = await analyze_lean_failure(
            lean_code=lean_code,
            lean_executable_path=lean_executable,  # Pass the executable name/path
            cwd=str(test_env_dir),  # Run analysis in the project context
            shared_lib_path=None,  # Adjust if specific library paths are needed
            # Provide build error context if LSP fails severely
            fallback_error=build_stderr,
        )
        return analysis_result
    except FileNotFoundError:
        print(
            f"Error: '{lean_executable}' command not found for LSP analysis. "
            "Make sure Lean is installed and in your system's PATH."
        )
        return f"LSP analysis failed: '{lean_executable}' not found."
    except Exception as e:
        print(f"Error during LSP analysis: {e}")
        # Include the original build stderr in the feedback if LSP fails completely
        error_message = f"LSP analysis failed with exception: {e}"
        if build_stderr:
            error_message += f"\n\n--- Original Build Stderr ---\n{build_stderr}"
        return error_message


# --- Per-Problem Solving Logic ---


async def solve_problem(
    problem_data: dict,
    client: GeminiClient,
    model_name: str,
    max_iterations: int,
    timeout_val: int,
    test_env_path_obj: pathlib.Path,
    test_src_subdir_name: str,
) -> tuple[bool, str, str, str, int]:
    """
    Attempts to solve a single Lean theorem iteratively using the LLM.

    Args:
        problem_data: Dictionary containing 'id' and 'formal_statement'.
        client: The initialized GeminiClient instance (shared for cost tracking).
        model_name: The name of the Gemini model to use.
        max_iterations: Maximum number of LLM calls for this problem.
        timeout_val: Timeout in seconds for Lean verification.
        test_env_path_obj: Path to the Lean test environment directory.
        test_src_subdir_name: Name of the source subdirectory within the test env.

    Returns:
        A tuple containing:
        - bool: True if the proof succeeded, False otherwise.
        - str: The problem ID.
        - str: The final Lean code attempted (or the successful code).
        - str: The final LSP feedback or build error message.
        - int: The number of iterations performed for this problem.
    """
    problem_id = problem_data.get("id")
    formal_statement = problem_data.get("formal_statement")

    if not all([problem_id, formal_statement]):
        print(
            "Error: Skipping problem due to missing 'id' or 'formal_statement': "
            f"{problem_data}"
        )
        return (
            False,
            problem_data.get("id", "UNKNOWN"),
            "",
            "Missing required fields",
            0,
        )

    # Extract the theorem signature part (everything up to and including ':=')
    theorem_signature_part = ""
    if ":=" in formal_statement:
        theorem_signature_part = formal_statement.split(":=", 1)[0].strip() + " :="
    else:
        print(
            f"Error: Could not find ':=' in formal statement for {problem_id}. "
            "Cannot reliably separate signature."
        )
        print(f"Statement: {formal_statement}")
        return (
            False,
            problem_id,
            formal_statement,
            "Could not parse theorem signature",
            0,
        )

    if not theorem_signature_part:
        print(f"Error: Failed to extract theorem signature for {problem_id}.")
        return (
            False,
            problem_id,
            formal_statement,
            "Failed to extract theorem signature",
            0,
        )

    print(f"\n--- Starting Test for Problem: {problem_id} ---")
    # Print the full formal statement for clarity
    print(f"Formal Statement:\n```lean\n{formal_statement}\n```")
    # Less verbose version if preferred:
    # print(f"Theorem Signature:\n{theorem_signature_part} by\n")

    # --- Iteration State (Reset for each problem) ---
    current_full_tactic_block = ""
    proof_succeeded = False
    lsp_feedback_for_next_prompt = "LSP Analysis not yet performed."
    last_build_stderr = ""
    full_lean_code_to_verify = ""
    # Default final code
    final_code_output = (
        f"{FIXED_LEAN_HEADER}\n\n{theorem_signature_part} by\n"
        "  -- No tactics generated yet"
    )
    final_feedback_output = "(No feedback generated yet)"
    iterations_done = 0

    # --- Iteration Loop ---
    for current_iteration in range(1, max_iterations + 1):
        iterations_done = current_iteration
        print(
            f"--- Problem: {problem_id} | "
            f"Iteration {current_iteration}/{max_iterations} ---"
        )

        # --- Construct Prompt ---
        if current_iteration == 1:
            prompt = initial_prompt_template.format(formal_statement=formal_statement)
        else:
            feedback_section = lsp_feedback_for_next_prompt
            if (
                not feedback_section
                or feedback_section == "LSP Analysis not yet performed."
            ):
                if last_build_stderr:
                    feedback_section = (
                        "Build failed in the previous iteration. No specific LSP "
                        "analysis available.\nBuild stderr:\n"
                        f"```\n{last_build_stderr}\n```"
                    )
                else:
                    feedback_section = (
                        "No specific feedback available from the previous iteration. "
                        "Please attempt the proof."
                    )
            prompt = subsequent_prompt_template.format(
                formal_statement=formal_statement, feedback_section=feedback_section
            )

        # --- Call LLM ---
        try:
            print(
                f"Sending prompt to LLM (Problem: {problem_id}, "
                f"Iter: {current_iteration})..."
            )
            llm_response = await client.generate(prompt, model=model_name)

            # Print cumulative cost after this call
            cost_summary = client.cost_tracker.get_summary()
            total_cost = cost_summary.get("total_estimated_cost", 0.0)
            print(
                f"Cumulative Estimated Cost after Iter {current_iteration}: "
                f"${total_cost:.6f}"
            )

            # Optionally print raw LLM response for debugging:
            # print(f"LLM Raw Response:\n{llm_response}\n--------------------")
        except Exception as e:
            print(
                f"Error calling GeminiClient generate for {problem_id} in "
                f"iteration {current_iteration}: {e}"
            )
            final_feedback_output = f"LLM Error: {e}"
            break  # Exit the loop for this problem

        # --- Extract and Clean Tactic Block ---
        extracted_code = extract_lean_code(llm_response)
        if not extracted_code:
            print(
                f"Warning: Could not extract Lean code block from LLM response for "
                f"{problem_id} in iter {current_iteration}."
            )
            final_feedback_output = "LLM did not return a parsable code block."
            last_build_stderr = "(No build attempted - LLM failed to provide code)"
            lsp_feedback_for_next_prompt = (
                "(No LSP analysis performed - LLM failed to provide code)"
            )
            break

        cleaned_tactic_block = re.sub(
            r"^```(?:lean)?\s*", "", extracted_code, flags=re.IGNORECASE
        ).strip()
        cleaned_tactic_block = re.sub(r"\s*```$", "", cleaned_tactic_block).strip()

        if (
            theorem_signature_part.split(" ", 1)[0]
            in cleaned_tactic_block.split("\n")[0]
        ):
            print(
                f"Warning: LLM response might have included the theorem signature "
                f"for {problem_id} in iter {current_iteration}."
            )

        if not cleaned_tactic_block:
            print(
                f"Warning: Extracted code was empty after cleanup for {problem_id} "
                f"in iter {current_iteration}."
            )
            final_feedback_output = "LLM returned an empty code block."
            last_build_stderr = "(No build attempted - LLM provided empty code)"
            lsp_feedback_for_next_prompt = (
                "(No LSP analysis performed - LLM provided empty code)"
            )
            break

        current_full_tactic_block = "\n".join(
            [f"  {line}" for line in cleaned_tactic_block.splitlines()]
        )
        full_lean_code_to_verify = (
            f"{FIXED_LEAN_HEADER}\n\n{theorem_signature_part} by\n"
            f"{current_full_tactic_block}"
        )
        final_code_output = (
            full_lean_code_to_verify  # Update final code with the latest attempt
        )

        # Print the code that will be verified in this iteration
        print(
            f"\n--- Code to Verify (Problem: {problem_id}, "
            f"Iter: {current_iteration}) ---"
        )
        print(full_lean_code_to_verify)
        print("------------------------------------------")

        # --- Run Lean Verification ---
        success, stdout, stderr = await run_lean_verification(
            lean_code=full_lean_code_to_verify,
            problem_id=problem_id,
            iteration=current_iteration,
            test_env_dir=test_env_path_obj,
            test_src_subdir=test_src_subdir_name,
            timeout=timeout_val,
        )
        last_build_stderr = stderr
        final_feedback_output = stderr  # Default feedback is stderr if things go wrong

        # --- Process Verification Result ---
        if success:
            print(
                f"Build SUCCEEDED for {problem_id} (Iter: {current_iteration}). "
                "Checking goals via LSP..."
            )
            lsp_feedback = await run_lsp_analysis(
                lean_code=full_lean_code_to_verify,
                problem_id=problem_id,
                iteration=current_iteration,
                test_env_dir=test_env_path_obj,
                build_stderr=stderr,
            )
            lsp_feedback_for_next_prompt = lsp_feedback
            final_feedback_output = lsp_feedback  # Update final feedback

            # Print LSP feedback for this iteration
            print(
                f"\n--- LSP Feedback (Problem: {problem_id}, "
                f"Iter: {current_iteration}, Build Success) ---"
            )
            print(lsp_feedback)
            print(
                "---------------------------------------------------------------------"
            )

            if (
                "unsolved goals" not in lsp_feedback.lower()
                and "⊢" not in lsp_feedback
                and "error" not in lsp_feedback.lower()
            ):
                proof_succeeded = True
                print(
                    f"✅ SUCCESS: Proof for {problem_id} verified successfully in "
                    f"iteration {current_iteration}."
                )
                break  # Exit the loop for this problem successfully
            else:
                print(
                    f"Build succeeded for {problem_id}, but goals remain or warnings "
                    "present. Preparing for next iteration..."
                )
                pass  # Continue to next iteration

        else:  # Build failed
            print(
                f"Build FAILED for {problem_id} (Iter: {current_iteration}). "
                "Running LSP analysis for error details..."
            )
            lsp_feedback = await run_lsp_analysis(
                lean_code=full_lean_code_to_verify,
                problem_id=problem_id,
                iteration=current_iteration,
                test_env_dir=test_env_path_obj,
                build_stderr=stderr,  # Crucial to pass stderr here
            )
            lsp_feedback_for_next_prompt = lsp_feedback
            final_feedback_output = lsp_feedback  # Update final feedback

            # Print LSP feedback / build error for this iteration
            print(
                f"\n--- LSP Feedback / Build Error (Problem: {problem_id}, "
                f"Iter: {current_iteration}, Build Failed) ---"
            )
            print(
                lsp_feedback
            )  # This now likely contains formatted errors from LSP or fallback stderr
            print(
                "--------------------------------------------------------------------------"
            )

            if current_iteration == max_iterations:
                print(f"Max iterations reached for {problem_id} after build failure.")
            # Continue to the next iteration (or loop exit if max reached)

    # --- Return results for this problem ---
    status = "Success" if proof_succeeded else "Failure"
    print(
        f"--- Finished Problem: {problem_id} ({status} after "
        f"{iterations_done} iterations) ---"
    )
    return (
        proof_succeeded,
        problem_id,
        final_code_output,
        final_feedback_output,
        iterations_done,
    )


# --- Main Execution Logic ---


async def main():
    """
    Main asynchronous function orchestrating the testing of multiple sampled
    problems sequentially.
    """
    # --- Configuration Setup ---
    num_to_sample = NUM_PROBLEMS_TO_SAMPLE
    data_file_path_obj = MINIF2F_DATA_PATH
    test_env_path_obj = TEST_ENV_DIR
    test_src_subdir_name = TEST_SRC_SUBDIR
    timeout_val = TIMEOUT_SECONDS
    model_name_from_config = MODEL_NAME
    max_iterations_per_prob = MAX_ITERS_PER_PROBLEM

    # --- Path Validation ---
    if not data_file_path_obj.is_file():
        print(f"Error: Data file not found at {data_file_path_obj}")
        sys.exit(1)
    if not test_env_path_obj.is_dir():
        print(
            f"Error: Lean test environment directory not found at {test_env_path_obj}"
        )
        sys.exit(1)
    if not (test_env_path_obj / "lakefile.lean").is_file():
        print(
            f"Error: No 'lakefile.lean' found in {test_env_path_obj}. "
            "This is required for 'lake build'."
        )
        sys.exit(1)

    # --- Load Problem Data ---
    all_problems = load_minif2f_data(data_file_path_obj)
    if not all_problems:
        print("Exiting due to failure loading problems.")
        sys.exit(1)

    total_problems_available = len(all_problems)
    print(f"Total problems loaded: {total_problems_available}")

    # --- Select Problems ---
    if num_to_sample >= total_problems_available:
        print(
            f"Requested sample size ({num_to_sample}) is >= total available "
            f"problems ({total_problems_available}). Using the entire dataset."
        )
        problems_to_test = all_problems
        actual_num_tested = total_problems_available
    else:
        print(f"Randomly sampling {num_to_sample} problems to test...")
        problems_to_test = random.sample(all_problems, num_to_sample)
        actual_num_tested = num_to_sample
    print(f"Will attempt to solve {len(problems_to_test)} problems.")

    # --- Initialize LLM Client (ONCE for cumulative cost tracking) ---
    try:
        api_key = get_gemini_api_key()
        if not api_key:
            print(
                "Warning: Gemini API key not found. Ensure it's set in config or "
                "environment."
            )
        # Initialize client here, outside the loop
        shared_client = GeminiClient(
            api_key=api_key, default_generation_model=model_name_from_config
        )
    except Exception as e:
        print(f"Error initializing GeminiClient: {e}")
        sys.exit(1)

    # F541: Removed f prefix
    print("\n--- Starting Batch Test (Sequential Execution) ---")
    print(f"Model: {model_name_from_config}")
    print(f"Max Iterations per Problem: {max_iterations_per_prob}")
    print(f"Timeout per Iteration: {timeout_val}s")

    # --- Run Tests Sequentially ---
    results = []
    for problem_data in problems_to_test:
        # Await each problem directly in the loop
        result = await solve_problem(
            problem_data=problem_data,
            client=shared_client,  # Pass the shared client
            model_name=model_name_from_config,
            max_iterations=max_iterations_per_prob,
            timeout_val=timeout_val,
            test_env_path_obj=test_env_path_obj,
            test_src_subdir_name=test_src_subdir_name,
        )
        results.append(result)
        # Add a small delay if needed, e.g., await asyncio.sleep(0.1)
        # This can sometimes help with resource contention or rate limits,
        # but is not strictly necessary for sequential execution.

    # --- Final Summary Report ---
    # F541: Removed f prefix
    print("\n\n--- Batch Test Summary ---")
    success_count = sum(1 for r in results if r[0])
    failure_count = actual_num_tested - success_count

    print(f"Total Problems Attempted: {actual_num_tested}")
    print(f"✅ Successes: {success_count}")
    print(f"❌ Failures: {failure_count}")

    # Print cumulative cost from the shared client (still useful for final total)
    cost_summary = shared_client.cost_tracker.get_summary()
    total_cost = cost_summary.get("total_estimated_cost", 0.0)
    print(f"\nFinal Cumulative Estimated LLM Cost for this run: ${total_cost:.6f}")

    # Optional: Print details for failures
    print("\n--- Failure Details ---")
    found_failures = False
    for success, prob_id, final_code, final_feedback, iters in results:
        if not success:
            found_failures = True
            print(f"\nProblem ID: {prob_id} (Failed after {iters} iterations)")
            # print(f"Final Code Attempt:\n{final_code}") # Can be very long
            print(f"Final Feedback/Error:\n{final_feedback}")
            print("-" * 20)
    if not found_failures:
        print("No failures to report.")
    print("-----------------------")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure project root is in path if running script directly
    SCRIPT_DIR_MAIN = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT_MAIN = SCRIPT_DIR_MAIN.parent
    if str(PROJECT_ROOT_MAIN) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_MAIN))
        print(
            f"Added {PROJECT_ROOT_MAIN} to sys.path to locate lean_automator modules."
        )

    # Check if config loaded (it should have been checked during import)
    if "APP_CONFIG" not in globals():
        print(
            "Error: APP_CONFIG not loaded. Check lean_automator.config.loader "
            "import and configuration setup."
        )
        sys.exit(1)

    # Run the main asynchronous function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
