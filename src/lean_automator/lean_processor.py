# File: lean_processor.py

"""Orchestrates Lean code generation and verification for Knowledge Base items.

This module manages the process of generating formal Lean 4 code (both statement
signatures and proofs) for mathematical items (`KBItem`) stored in the knowledge
base. It interacts with an LLM client (`GeminiClient`) to generate the code based
on LaTeX statements, informal proofs, and dependency context.

The core logic involves:

1. Generating a Lean statement signature (`... := sorry`).

2. If the item requires proof, generating Lean proof tactics to replace `sorry`.

3. Calling the `lean_interaction` module to verify the generated code using `lake`
    in a temporary environment that requires a persistent shared library.

4. Handling retries for LLM generation and potentially invoking proof repair logic.

5. Updating the `KBItem` status and content in the database based on the outcome.

6. Providing batch processing capabilities for items pending Lean processing.
"""

import asyncio
import warnings
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any
import time

# --- Imports from other project modules ---
try:
    from lean_automator.kb_storage import (
        KBItem,
        ItemStatus,
        ItemType,
        get_kb_item_by_name,
        save_kb_item,
        get_items_by_status,
        DEFAULT_DB_PATH
    )
    from lean_automator.llm_call import GeminiClient
    # lean_interaction module now uses the persistent library strategy
    from lean_automator import lean_interaction
    from lean_automator import lean_proof_repair # Import the repair module
except ImportError as e:
    warnings.warn(f"lean_processor: Required modules not found: {e}")
    # Define dummy types/functions to allow script loading without crashing
    KBItem = None; ItemStatus = None; ItemType = None; get_kb_item_by_name = None; save_kb_item = None; get_items_by_status = None; GeminiClient = None; DEFAULT_DB_PATH = None; lean_interaction = None; lean_proof_repair = None; # type: ignore

# --- Load Environment Variables or Exit ---
from dotenv import load_dotenv
env_loaded_successfully = load_dotenv()

if not env_loaded_successfully:
    # Print error message to standard error
    print("\nCRITICAL ERROR: Could not find or load the .env file.", file=sys.stderr)
    print("This script relies on environment variables defined in that file.", file=sys.stderr)
    # Show where it looked relative to, which helps debugging
    print(f"Please ensure a .env file exists in the current directory ({os.getcwd()}) or its parent directories.", file=sys.stderr)
    sys.exit(1) # Exit the script with a non-zero status code indicating failure

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS = 2
DEFAULT_LEAN_PROOF_MAX_ATTEMPTS = 3

try:
    LEAN_STATEMENT_MAX_ATTEMPTS = int(os.getenv('LEAN_STATEMENT_MAX_ATTEMPTS', DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS))
    if LEAN_STATEMENT_MAX_ATTEMPTS < 1:
         logger.warning(f"LEAN_STATEMENT_MAX_ATTEMPTS must be >= 1. Using default {DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS}.")
         LEAN_STATEMENT_MAX_ATTEMPTS = DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS
except (ValueError, TypeError):
    logger.warning(f"Invalid value for LEAN_STATEMENT_MAX_ATTEMPTS. Using default {DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS}.")
    LEAN_STATEMENT_MAX_ATTEMPTS = DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS

try:
    LEAN_PROOF_MAX_ATTEMPTS = int(os.getenv('LEAN_PROOF_MAX_ATTEMPTS', DEFAULT_LEAN_PROOF_MAX_ATTEMPTS))
    if LEAN_PROOF_MAX_ATTEMPTS < 1:
         logger.warning(f"LEAN_PROOF_MAX_ATTEMPTS must be >= 1. Using default {DEFAULT_LEAN_PROOF_MAX_ATTEMPTS}.")
         LEAN_PROOF_MAX_ATTEMPTS = DEFAULT_LEAN_PROOF_MAX_ATTEMPTS
except (ValueError, TypeError):
    logger.warning(f"Invalid value for LEAN_PROOF_MAX_ATTEMPTS. Using default {DEFAULT_LEAN_PROOF_MAX_ATTEMPTS}.")
    LEAN_PROOF_MAX_ATTEMPTS = DEFAULT_LEAN_PROOF_MAX_ATTEMPTS

logger.info(f"Using LEAN_STATEMENT_MAX_ATTEMPTS = {LEAN_STATEMENT_MAX_ATTEMPTS}")
logger.info(f"Using LEAN_PROOF_MAX_ATTEMPTS = {LEAN_PROOF_MAX_ATTEMPTS}")


# --- Prompt Templates ---
# Prompts remain the same - they already instructed the LLM NOT to write dependency imports.
LEAN_STATEMENT_GENERATOR_PROMPT = """
You are an expert Lean 4 programmer translating mathematical statements into formal Lean code. **You are working in a restricted environment with ONLY the Lean 4 prelude and explicitly provided dependencies.**

**Goal:** Generate the Lean 4 statement signature (including necessary standard imports if absolutely needed, usually none) for the item named `{unique_name}` ({item_type_name}), based on its LaTeX statement.

**LaTeX Statement:**
--- BEGIN LATEX STATEMENT ---
{latex_statement}
--- END LATEX STATEMENT ---

**Available Proven Dependencies (For Context - Names and Types):**
{dependency_context_for_statement}

**Refinement Feedback (If Applicable):**
{statement_error_feedback}

**Instructions:**
1.  Translate the LaTeX Statement into a formal Lean 4 theorem/definition signature (e.g., `theorem MyTheorem (n : Nat) : ...`). **IMPORTANT: Choose a Lean formulation that uses ONLY features available in the standard Lean 4 prelude (like basic types `Nat`, `List`, `Prop`, `Type`, logical connectives `∀`, `∃`, `∧`, `∨`, `¬`, basic arithmetic `+`, `*`, `>`, `=`, induction principles) and the provided dependencies.** For example, expressing 'infinitely many primes' as `∀ n, ∃ p > n, Nat.IsPrime p` is preferred over using `Set.Infinite` which requires extra libraries.
2.  Include necessary minimal standard imports ONLY if required beyond the prelude (e.g., often no imports are needed). **DO NOT generate imports for the dependencies listed above; they will be handled automatically.**
3.  **CRITICAL: DO NOT use or import ANYTHING from `mathlib` or `Std` unless explicitly provided in the dependencies.** Code relying on concepts like `Set`, `Finset`, `Data.`, `Mathlib.` etc., is INCORRECT for this task.
4.  Append ` := sorry` to the end of the statement signature.
5.  Output **only** the Lean code containing any necessary standard imports and the complete signature ending in `sorry`, marked between `--- BEGIN LEAN HEADER ---` and `--- END LEAN HEADER ---`. Here is an example output:

--- BEGIN LEAN HEADER ---
theorem MyTheorem (n : Nat) : Exists (m : Nat), m > n := sorry
--- END LEAN HEADER ---
"""

LEAN_PROOF_GENERATOR_PROMPT = """
You are an expert Lean 4 programmer completing a formal proof. **You are working in a restricted environment with ONLY the Lean 4 prelude and explicitly provided dependencies.**

**Goal:** Complete the Lean proof below by replacing `sorry`.

**Lean Statement Shell (Target):**
--- BEGIN LEAN HEADER ---
{lean_statement_shell}
--- END LEAN HEADER ---

**Informal LaTeX Proof (Use as Guidance):**
(This informal proof might contain errors, but use it as a guide for the formal Lean proof structure and steps.)
--- BEGIN LATEX PROOF ---
{latex_proof}
--- END LATEX PROOF ---

**Available Proven Dependencies (Lean Code):**
(You MUST use these definitions and theorems. **Assume they are correctly imported automatically.**)
{dependency_context_for_proof}

**Previous Attempt Error (If Applicable):**
(The previous attempt to compile the generated Lean code failed with the following error. Please fix the proof tactics.)
--- BEGIN LEAN ERROR ---
{lean_error_log}
--- END LEAN ERROR ---

**Instructions:**
1.  Write the Lean 4 proof tactics to replace the `sorry` in the provided Lean Statement Shell.
2.  Ensure the proof strictly follows Lean 4 syntax and logic.
3.  You have access ONLY to Lean 4 prelude features (basic types, logic, induction, basic tactics like `rw`, `simp`, `intro`, `apply`, `exact`, `have`, `let`, `by_contra`, `cases`, `induction`, `rfl`) and the 'Available Proven Dependencies' provided above. **Use `simp` frequently to simplify goals and unfold definitions (like the definition of `List.append` when applied to `::`).** Use `rw [axiom_name]` or `rw [← axiom_name]` for intermediate steps. **Do NOT try to `rw` using function names (like `List.append`) or constructor names (like `List.cons`).**
4.  **CRITICAL: DO NOT use or import ANYTHING from `mathlib` or `Std` unless explicitly provided in the dependencies.** Code using `Set`, `Finset`, advanced tactics (like `linarith`, `ring`), or library functions beyond the prelude or provided dependencies is INCORRECT.
5.  **CRITICAL: DO NOT generate any `import` statements. Assume necessary dependency imports are already present.**
6.  Use the Informal LaTeX Proof as a *guide* but prioritize formal correctness using ONLY the allowed features.
7.  Before significant tactics (`rw`, `simp` variants, `apply`, `induction` steps, `cases`, `have`, `let`), add **two** comment lines in the following format:
    * `-- Goal: [Brief summary or key part of the current proof goal]`
    * `-- Action: [Explain the planned tactic, which rule/hypothesis (like 'ih') is used, and why (note rw ← if needed)]`
* Do **not** add these comments for simple tactics like `rfl`, `exact ...`, `done`, or simple structural syntax.
* **Example:**
    ```lean
    -- Goal: List.reverse l' = List.append (List.reverse l') []
    -- Action: Apply List.append_nil_ax to simplify the RHS
    rw [List.append_nil_ax]
    ```
* **Example:**
    ```lean
    -- Goal: List.append (List.append A B) [x] = List.append A (List.reverse (x :: xs))
    -- Action: Apply List.reverse_cons_ax to the RHS
    rw [List.reverse_cons_ax]
    ```
* **Example:**
    ```lean
    -- Goal: List.reverse (xs ++ l') ++ [x] = ...
    -- Action: Apply the induction hypothesis 'ih' to the LHS
    rw [ih]
    ```
8.  **TACTIC NOTE (Reflexivity):** After applying tactics like `rw [...]` or `simp [...]`, check if the resulting goal is of the form `X = X`. If it is, the goal is solved by reflexivity. **DO NOT add `rfl` in this case.** Simply proceed or end the branch. Avoid redundant tactics.
9.  **TACTIC NOTE (Finishing with Axioms/Hypotheses):** If the goal is an equality `LHS = RHS` and the *final* step is to apply a single axiom or hypothesis `h : LHS = RHS` (or `h : RHS = LHS`), **prefer using `exact h` (or `exact h.symm`)** instead of using `rw [h]` or `rw [← h]` as the very last tactic for that goal branch. Use `rw` for intermediate steps.
10. **Error Handling:** If fixing an error based on 'Previous Attempt Error', carefully analyze the error message and modify the proof tactics accordingly. **Do NOT change the theorem signature provided.**
    * **Specifically for "no goals to be solved" errors:** If the error log contains `error: ...:N:M: no goals to be solved` pointing to a line `N` containing a tactic (like `rfl`), it almost always means the goal was already solved implicitly by the tactic on the line *before* `N`. You should **remove the superfluous tactic on line `N`** in your corrected proof attempt.
11. Ensure the proof block is correctly terminated (e.g., no stray `end`).
12. Output **only** the complete Lean code block, including the *unchanged* statement signature, and the full proof replacing sorry (with comments), marked between `--- BEGIN LEAN CODE ---` and `--- END LEAN CODE ---`. Here is an example output:

--- BEGIN LEAN CODE ---
theorem MyTheorem (n : Nat) : Exists (m : Nat), m > n := by
  -- Use existence introduction
  apply Exists.intro (n + 1)
  -- Apply the definition of successor and less than
  simp [Nat.succ_eq_add_one, Nat.lt_succ_self]
--- END LEAN CODE ---
"""


# --- Helper Functions ---

def _extract_lean_header(text: Optional[str]) -> Optional[str]:
    """Extracts Lean header (signature ending in ':= sorry') from LLM output. (Internal Helper)

    Tries to parse the header using custom markers (`--- BEGIN/END LEAN HEADER ---`)
    first, then falls back to parsing a strict markdown code block (` ```lean ... ``` `)
    if markers are not found. Ensures the extracted header ends with `:= sorry`,
    attempting to append it if missing.

    Args:
        text (Optional[str]): The raw text output from the Lean statement generator LLM.

    Returns:
        Optional[str]: The extracted and cleaned Lean header string ending in
        `:= sorry`, or None if parsing fails or the input text is empty.
    """
    if not text: return None
    header = None
    stripped_text = text.strip()
    # Try custom markers first
    custom_marker_regex = r"---\s*BEGIN LEAN HEADER\s*---\s*(.*?)\s*---\s*END LEAN HEADER\s*---"
    match_custom = re.search(custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE)
    if match_custom:
        header = match_custom.group(1).strip()
        logger.debug("Extracted header using custom markers.")
    else:
        logger.warning("Could not find '--- BEGIN/END LEAN HEADER ---' markers. Trying strict markdown block...")
        # Fallback: Try strict markdown block encompassing the whole response
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            header = match_md.group(1).strip()
            logger.debug("Extracted header using strict markdown block.")
        else:
            logger.warning("Could not find strict markdown block ('```lean...```') encompassing the whole response.")

    # Post-process: Ensure it ends with ':= sorry'
    if header:
        stripped_header = header.rstrip()
        # Remove trailing comments before checking/appending ':= sorry'
        header_no_comments = re.sub(r"\s*--.*$", "", stripped_header, flags=re.MULTILINE).rstrip()

        if not header_no_comments.endswith(':= sorry'):
            logger.warning("Parsed Lean header does not end with ':= sorry'. Attempting to append.")
            # Attempt to fix common near misses before just appending
            if header_no_comments.endswith(' :='):
                header = header_no_comments + " sorry"
            elif header_no_comments.endswith(' :'):
                header = header_no_comments + "= sorry"
            else: # Append directly if no obvious near miss
                header = header_no_comments + " := sorry"
        else:
            # If it already ends correctly (after stripping comments), use the original (potentially with comments)
            header = stripped_header

        logger.debug(f"Final extracted/corrected header: '{header}'")
        return header
    else:
        # Log error if extraction failed completely
        logger.error(f"Failed to extract Lean header using any method. Raw text received: {repr(text)[:500]}...")
        return None

def _extract_lean_code(text: Optional[str]) -> Optional[str]:
    """Extracts a block of Lean code from LLM output. (Internal Helper)

    Tries to parse the code using custom markers (`--- BEGIN/END LEAN CODE ---`)
    first, then falls back to parsing a strict markdown code block (` ```lean ... ``` `)
    if markers are not found.

    Args:
        text (Optional[str]): The raw text output from the Lean proof generator LLM.

    Returns:
        Optional[str]: The extracted Lean code string (including statement and proof),
        or None if parsing fails or the input text is empty.
    """
    if not text: return None
    code = None
    stripped_text = text.strip()
    # Try custom markers first
    custom_marker_regex = r"---\s*BEGIN LEAN CODE\s*---\s*(.*?)\s*---\s*END LEAN CODE\s*---"
    match_custom = re.search(custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE)
    if match_custom:
        code = match_custom.group(1).strip()
        logger.debug("Extracted code using custom markers.")
        return code
    else:
        logger.warning("Could not find '--- BEGIN/END LEAN CODE ---' markers. Trying strict markdown block...")
        # Fallback: Try strict markdown block encompassing the whole response
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            code = match_md.group(1).strip()
            logger.debug("Extracted code using strict markdown block.")
            return code
        else:
             logger.warning("Could not find strict markdown block ('```lean...```') encompassing the whole response.")

    # Log error if extraction failed completely
    logger.error(f"Failed to extract Lean code using any method. Raw text received: {repr(text)[:500]}...")
    return None


def _build_lean_dependency_context_for_statement(dependencies: List[KBItem]) -> str:
    """Builds a simple context string listing dependency names and types. (Internal Helper)

    Used for the statement generation prompt to give the LLM awareness of available
    item names and types without including their full code.

    Args:
        dependencies (List[KBItem]): A list of dependency KBItems.

    Returns:
        str: A formatted string listing dependency names and types, or "(None)"
        if the list is empty.
    """
    if not dependencies: return "(None)"
    ctx = ""
    # Get attributes safely
    for dep in dependencies:
        dep_name = getattr(dep, 'unique_name', 'UNKNOWN_DEP')
        dep_type_name = getattr(getattr(dep, 'item_type', None), 'name', 'UNKNOWN_TYPE')
        ctx += f"- {dep_name} ({dep_type_name})\n"
    return ctx.strip()

def _build_lean_dependency_context_for_proof(dependencies: List[KBItem]) -> str:
    """Builds context string with full Lean code of PROVEN dependencies. (Internal Helper)

    Filters the provided list of dependencies to include only those that are
    considered "proven" (status PROVEN, AXIOM_ACCEPTED, DEFINITION_ADDED) and
    have Lean code. Formats the output with markers for clarity in the LLM prompt.

    Args:
        dependencies (List[KBItem]): A list of potential dependency KBItems.

    Returns:
        str: A formatted string containing the Lean code of valid, proven
        dependencies, or a default message if none are found.
    """
    dependency_context = ""
    # Define statuses indicating a dependency is ready to be used in a proof
    valid_statuses = {ItemStatus.PROVEN, ItemStatus.AXIOM_ACCEPTED, ItemStatus.DEFINITION_ADDED}
    # Filter dependencies: must exist, have lean_code, and have a valid status
    proven_deps = [d for d in dependencies if d and getattr(d, 'lean_code', None) and getattr(d, 'status', None) in valid_statuses]

    if proven_deps:
        for dep in proven_deps:
            # Safely get attributes
            dep_name = getattr(dep, 'unique_name', 'UNKNOWN_DEP')
            dep_type_name = getattr(getattr(dep, 'item_type', None), 'name', 'UNKNOWN_TYPE')
            dep_code = getattr(dep, 'lean_code', '# Error: Code missing')
            # Format clearly for the LLM prompt
            dependency_context += f"-- Dependency: {dep_name} ({dep_type_name})\n"
            dependency_context += f"-- BEGIN {dep_name} LEAN --\n"
            dependency_context += f"{dep_code.strip()}\n" # Use the actual proven code
            dependency_context += f"-- END {dep_name} LEAN --\n\n"
    else:
        # Provide a clear message if no suitable dependencies were found
        dependency_context = "-- (No specific proven dependencies provided from KB. Rely on Lean prelude.) --\n"

    return dependency_context.strip()


# --- LLM Caller Functions ---
async def _call_lean_statement_generator(
    item: KBItem, dependencies: List[KBItem], statement_error_feedback: Optional[str], client: GeminiClient
) -> Optional[str]:
    """Calls the LLM to generate the Lean statement signature (`... := sorry`). (Internal Helper)

    Formats the prompt using `LEAN_STATEMENT_GENERATOR_PROMPT`, including item
    details (LaTeX statement), dependency context (names/types), and optional
    feedback from previous failed attempts.

    Args:
        item (KBItem): The KBItem for which to generate the statement shell.
        dependencies (List[KBItem]): List of dependency items for context.
        statement_error_feedback (Optional[str]): Feedback from a previous failed
            statement generation attempt, if applicable.
        client (GeminiClient): The initialized LLM client instance.

    Returns:
        Optional[str]: The raw text response from the LLM, or None if the client
        is unavailable, prompt formatting fails, or the API call errors.

    Raises:
        ValueError: If the `client` is None.
    """
    if not client: raise ValueError("GeminiClient not available for Lean statement generation.")
    # Safely get attributes from item, providing defaults or raising errors if critical info missing
    item_unique_name = getattr(item, 'unique_name', 'UNKNOWN_ITEM')
    item_type = getattr(item, 'item_type', None)
    item_type_name = getattr(item_type, 'name', 'UNKNOWN_TYPE') if item_type else 'UNKNOWN_TYPE'
    item_latex_statement = getattr(item, 'latex_statement', None)

    if not item_latex_statement:
         logger.error(f"Cannot generate Lean statement for {item_unique_name}: Missing latex_statement.")
         return None # Cannot proceed without LaTeX statement

    # Build context string from dependencies
    dep_context = _build_lean_dependency_context_for_statement(dependencies)

    try:
        base_prompt = LEAN_STATEMENT_GENERATOR_PROMPT
        # Conditionally format the prompt based on whether feedback is provided
        if not statement_error_feedback:
             # Remove the optional feedback section if not needed
             base_prompt = re.sub(r"\n\*\*Refinement Feedback \(If Applicable\):\*\*.*?\n(\*\*Instructions:\*\*)", r"\n\1", base_prompt, flags=re.DOTALL | re.MULTILINE)
             prompt = base_prompt.format(
                 unique_name=item_unique_name,
                 item_type_name=item_type_name,
                 latex_statement=item_latex_statement, # Already checked it exists
                 dependency_context_for_statement=dep_context
                 # statement_error_feedback key is removed from template here
             )
        else:
             # Format with feedback included
             prompt = base_prompt.format(
                 unique_name=item_unique_name,
                 item_type_name=item_type_name,
                 latex_statement=item_latex_statement,
                 dependency_context_for_statement=dep_context,
                 statement_error_feedback=statement_error_feedback # Provide feedback
             )
    except KeyError as e:
        logger.error(f"Lean Statement Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        # Catch other potential formatting issues
        logger.error(f"Unexpected error formatting Lean statement prompt for {item_unique_name}: {e}")
        return None

    # Call the LLM client
    try:
        logger.debug(f"Sending Lean statement generation prompt for {item_unique_name}")
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        logger.error(f"Error calling Lean statement generator LLM for {item_unique_name}: {e}")
        return None

async def _call_lean_proof_generator(
    lean_statement_shell: str,
    latex_proof: Optional[str],
    unique_name: str,
    item_type_name: str,
    dependencies: List[KBItem],
    lean_error_log: Optional[str],
    client: GeminiClient
) -> Optional[str]:
    """Calls the LLM to generate Lean proof tactics for a given statement shell. (Internal Helper)

    Formats the prompt using `LEAN_PROOF_GENERATOR_PROMPT`, providing the
    statement shell (`... := sorry`), informal LaTeX proof (as guidance),
    dependency context (full Lean code), and optional error logs from previous
    failed compilation attempts.

    Args:
        lean_statement_shell (str): The Lean statement signature ending in `:= sorry`.
        latex_proof (Optional[str]): The informal LaTeX proof text (guidance only).
        unique_name (str): The unique name of the KBItem (for logging/context).
        item_type_name (str): The type name of the KBItem (for context).
        dependencies (List[KBItem]): List of proven dependency items (for context).
        lean_error_log (Optional[str]): Error output from the previous failed
            Lean compilation attempt, if applicable.
        client (GeminiClient): The initialized LLM client instance.

    Returns:
        Optional[str]: The raw text response from the LLM, expected to contain the
        complete Lean code (statement + proof tactics), or None if the client is
        unavailable, prompt formatting fails, the API call errors, or input validation fails.

    Raises:
        ValueError: If `client` is None or `lean_statement_shell` is invalid.
    """
    if not client: raise ValueError("GeminiClient not available for Lean proof generation.")
    # Validate the input shell
    if not lean_statement_shell or ":= sorry" not in lean_statement_shell:
         msg = f"Internal Error: _call_lean_proof_generator for {unique_name} called without a valid shell ending in ':= sorry'. Shell: {lean_statement_shell}"
         logger.error(msg)
         # Returning None instead of raising here as it's likely an internal logic error
         return None

    # Build dependency context string with full Lean code
    dep_context = _build_lean_dependency_context_for_proof(dependencies)

    try:
        base_prompt = LEAN_PROOF_GENERATOR_PROMPT
        # Conditionally format the prompt based on whether an error log is provided
        if not lean_error_log:
             # Remove the optional error log section if not needed
             base_prompt = re.sub(r"\n\*\*Previous Attempt Error \(If Applicable\):\*\*.*?\n(\*\*Instructions:\*\*)", r"\n\1", base_prompt, flags=re.DOTALL | re.MULTILINE)
             prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                unique_name=unique_name, # Added for better LLM context
                item_type_name=item_type_name # Added for better LLM context
                # lean_error_log key removed from template here
            )
        else:
             # Format with error log included
             prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                lean_error_log=lean_error_log, # Provide error log
                unique_name=unique_name, # Added for better LLM context
                item_type_name=item_type_name # Added for better LLM context
            )
    except KeyError as e:
        logger.error(f"Lean Proof Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        # Catch other potential formatting issues
        logger.error(f"Unexpected error formatting Lean proof prompt for {unique_name}: {e}")
        return None

    # Call the LLM client
    try:
        logger.debug(f"Sending Lean proof generation prompt for {unique_name}")
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        logger.error(f"Error calling Lean proof generator LLM for {unique_name}: {e}")
        return None


# --- Main Processing Function ---

async def generate_and_verify_lean(
    unique_name: str,
    client: GeminiClient,
    db_path: Optional[str] = None,
    lake_executable_path: str = 'lake',
    timeout_seconds: int = 120
) -> bool:
    """Generates, verifies, and potentially repairs Lean code for a KBItem.

    This function orchestrates the full Lean processing pipeline for an item:
    1.  Fetches the item and checks its status and prerequisites (LaTeX statement,
        proven dependencies).
    2.  Generates the Lean statement signature (`... := sorry`) using an LLM if
        it doesn't exist, with retries (`LEAN_STATEMENT_MAX_ATTEMPTS`).
    3.  If the item requires proof:
        a. Enters a loop (up to `LEAN_PROOF_MAX_ATTEMPTS`) to generate proof tactics.
        b. Calls the LLM proof generator, providing the statement shell, informal
           proof, dependency code, and any previous compilation errors.
        c. Parses the full Lean code from the LLM response.
        d. Calls `lean_interaction.check_and_compile_item` to verify the code
           in a temporary environment and update the persistent shared library on success.
        e. If verification succeeds, updates status to `PROVEN` and returns True.
        f. If verification fails, stores the error log and continues the loop
           (potentially after attempting automated repair via `lean_proof_repair`).
    4.  If the item does not require proof, marks it as `PROVEN` if a valid
        statement exists.
    5.  Updates the item's status (`PROVEN`, `LEAN_VALIDATION_FAILED`, `ERROR`)
        and stores relevant logs/responses in the database throughout the process.

    Args:
        unique_name (str): The unique name of the KBItem to process.
        client (GeminiClient): An initialized LLM client instance.
        db_path (Optional[str]): Path to the knowledge base database file. Uses
            `DEFAULT_DB_PATH` if None.
        lake_executable_path (str): Path to the `lake` executable.
        timeout_seconds (int): Timeout for `lake build` commands executed by
            `lean_interaction.check_and_compile_item`.

    Returns:
        bool: True if the Lean code is successfully generated (if needed) and
        verified (passing `check_and_compile_item`), False otherwise.
    """
    start_time = time.time()
    # Check if dependencies are properly loaded
    kbitem_available = KBItem is not None and not isinstance(KBItem, type('Dummy', (object,), {})) # Check if real type
    if kbitem_available:
        if not all([KBItem, ItemStatus, ItemType, get_kb_item_by_name, save_kb_item, lean_interaction, get_items_by_status, lean_proof_repair]):
            logger.critical("Required modules (kb_storage, lean_interaction, lean_proof_repair) not fully loaded. Cannot process Lean.")
            return False
    else:
         warnings.warn("Running Lean processor with dummy KB types.", RuntimeWarning)

    if not client:
        logger.error(f"GeminiClient missing. Cannot process Lean for {unique_name}.")
        return False
    if not hasattr(lean_interaction, 'check_and_compile_item'):
         logger.critical(f"lean_interaction.check_and_compile_item missing. Cannot process Lean for {unique_name}.")
         return False
    if not hasattr(lean_proof_repair, 'attempt_proof_repair'):
         logger.critical(f"lean_proof_repair.attempt_proof_repair missing. Cannot process Lean for {unique_name}.")
         return False

    effective_db_path = db_path or DEFAULT_DB_PATH

    # --- Initial Item Fetch and Status Checks ---
    item = get_kb_item_by_name(unique_name, effective_db_path)
    if not item:
        logger.error(f"Lean Proc: Item not found: {unique_name}")
        return False

    item_status = getattr(item, 'status', None)
    item_type = getattr(item, 'item_type', None)
    item_lean_code = getattr(item, 'lean_code', None)
    item_latex_statement = getattr(item, 'latex_statement', None)
    item_plan_dependencies = getattr(item, 'plan_dependencies', [])
    item_latex_proof = getattr(item, 'latex_proof', None) # For proof gen context

    if item_status == ItemStatus.PROVEN:
        logger.info(f"Lean Proc: Item {unique_name} is already PROVEN. Skipping.")
        return True

    # Define statuses that trigger Lean processing
    trigger_statuses = {ItemStatus.LATEX_ACCEPTED, ItemStatus.PENDING_LEAN, ItemStatus.LEAN_VALIDATION_FAILED}
    if item_status not in trigger_statuses:
        logger.warning(f"Lean Proc: Item {unique_name} not in a trigger status ({ {s.name for s in trigger_statuses} }). Current: {item_status.name if item_status else 'None'}. Skipping.")
        return False

    # Check prerequisite: LaTeX statement must exist
    if not item_latex_statement:
         logger.error(f"Lean Proc: Cannot process {unique_name}, missing required latex_statement.")
         if hasattr(item, 'update_status'):
             item.update_status(ItemStatus.ERROR, "Missing latex_statement for Lean generation.")
             await save_kb_item(item, client=None, db_path=effective_db_path)
         return False

    # Determine if proof is needed based on item type
    proof_required = item_type.requires_proof() if item_type else False # Assume proof needed if type unknown

    # Handle non-provable types early IF they already have Lean code
    if not proof_required and item_lean_code:
        logger.info(f"Lean Proc: No proof required for {unique_name} ({item_type.name if item_type else 'N/A'}) and Lean code exists. Marking PROVEN.")
        if hasattr(item, 'update_status'):
            item.update_status(ItemStatus.PROVEN)
            await save_kb_item(item, client=None, db_path=effective_db_path)
        return True
    # Note: If proof not required but code missing, we fall through to statement generation.

    # --- Check Dependencies ---
    dependency_items: List[KBItem] = []
    valid_dep_statuses = {ItemStatus.PROVEN, ItemStatus.AXIOM_ACCEPTED, ItemStatus.DEFINITION_ADDED}
    logger.debug(f"Checking {len(item_plan_dependencies)} dependencies for {unique_name}...")
    all_deps_ready = True
    for dep_name in item_plan_dependencies:
        dep_item = get_kb_item_by_name(dep_name, effective_db_path)
        dep_status = getattr(dep_item, 'status', None)
        dep_code_exists = bool(getattr(dep_item, 'lean_code', None))

        # Dependency must exist, be in a valid status, and have lean code
        if not dep_item or dep_status not in valid_dep_statuses or not dep_code_exists:
            dep_status_name = dep_status.name if dep_status else 'MISSING_ITEM'
            logger.error(f"Dependency '{dep_name}' for '{unique_name}' not ready (Status: {dep_status_name}, Code Exists: {dep_code_exists}). Cannot proceed.")
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.ERROR, f"Prerequisite dependency '{dep_name}' is not PROVEN or lacks code.")
                await save_kb_item(item, client=None, db_path=effective_db_path)
            all_deps_ready = False
            break # Stop checking other dependencies
        dependency_items.append(dep_item)

    if not all_deps_ready:
        return False # Exit if any dependency is not ready

    logger.debug(f"All {len(dependency_items)} dependencies for {unique_name} confirmed available and proven.")

    # --- Statement Generation Phase (if needed) ---
    original_lean_statement_shell = item_lean_code
    statement_error_feedback = None # Feedback for refinement
    needs_statement_generation = not item_lean_code or item_status in {ItemStatus.LATEX_ACCEPTED, ItemStatus.PENDING_LEAN}

    if needs_statement_generation:
        logger.info(f"Starting Lean statement generation phase for {unique_name}")
        statement_accepted = False
        for attempt in range(LEAN_STATEMENT_MAX_ATTEMPTS):
            logger.info(f"Lean Statement Generation Attempt {attempt + 1}/{LEAN_STATEMENT_MAX_ATTEMPTS} for {unique_name}")
            # Update status to show progress
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.LEAN_GENERATION_IN_PROGRESS, f"Statement attempt {attempt + 1}")
                await save_kb_item(item, client=None, db_path=effective_db_path) # Save status only

            # Call LLM for statement generation
            raw_response = await _call_lean_statement_generator(item, dependency_items, statement_error_feedback, client)
            parsed_header = _extract_lean_header(raw_response)

            # Refresh item state after LLM call
            item = get_kb_item_by_name(unique_name, effective_db_path)
            if not item: raise Exception(f"Item {unique_name} vanished during statement generation.")

            if not parsed_header:
                 logger.warning(f"Failed to parse Lean header on attempt {attempt + 1}.")
                 statement_error_feedback = f"LLM output did not contain a valid Lean header (attempt {attempt+1}). Raw response: {repr(raw_response[:500])}"
                 # Save raw response and error feedback for potential debugging or next attempt
                 if hasattr(item, 'raw_ai_response'): item.raw_ai_response = raw_response
                 if hasattr(item, 'lean_error_log'): item.lean_error_log = statement_error_feedback # Store feedback as error
                 if hasattr(item, 'generation_prompt'): item.generation_prompt = "Statement Gen Prompt (see logs/code)"
                 await save_kb_item(item, client=None, db_path=effective_db_path)
                 continue # Try next attempt
            else:
                # Successfully parsed header
                logger.info(f"Lean statement shell generated successfully for {unique_name} on attempt {attempt + 1}.")
                original_lean_statement_shell = parsed_header
                # Update item with the generated shell
                if hasattr(item, 'lean_code'): item.lean_code = original_lean_statement_shell
                if hasattr(item, 'generation_prompt'): item.generation_prompt = "Statement Gen Prompt (see logs/code)"
                if hasattr(item, 'raw_ai_response'): item.raw_ai_response = raw_response
                if hasattr(item, 'lean_error_log'): item.lean_error_log = None # Clear previous statement error feedback
                await save_kb_item(item, client=None, db_path=effective_db_path)
                statement_accepted = True
                break # Exit loop on success

        # Check if statement generation ultimately failed
        if not statement_accepted:
            logger.error(f"Failed to generate valid Lean statement shell for {unique_name} after {LEAN_STATEMENT_MAX_ATTEMPTS} attempts.")
            if hasattr(item, 'update_status'): # Ensure item object still exists
                item.update_status(ItemStatus.ERROR, f"Failed Lean statement generation after {LEAN_STATEMENT_MAX_ATTEMPTS} attempts.")
                # Keep last error feedback in lean_error_log
                await save_kb_item(item, client=None, db_path=effective_db_path)
            return False # Cannot proceed without a statement shell
    else:
         # Use existing Lean code as the statement shell
         logger.info(f"Skipping Lean statement generation for {unique_name}, using existing lean_code.")
         original_lean_statement_shell = item_lean_code
         # Validate existing shell format
         if not original_lean_statement_shell or ":= sorry" not in original_lean_statement_shell:
             logger.error(f"Existing lean_code for {unique_name} is invalid (missing ':= sorry'). Cannot proceed.")
             if hasattr(item, 'update_status'):
                 item.update_status(ItemStatus.ERROR, "Invalid existing lean_code shell (missing ':= sorry').")
                 await save_kb_item(item, client=None, db_path=effective_db_path)
             return False

    # --- Handle non-provable items after potential statement generation ---
    if not proof_required:
        logger.info(f"Lean Proc: Statement generated/present for non-provable {unique_name}. Marking PROVEN.")
        item_final = get_kb_item_by_name(unique_name, effective_db_path) # Re-fetch latest state
        if item_final and hasattr(item_final, 'status') and item_final.status != ItemStatus.PROVEN:
            if hasattr(item_final, 'update_status'):
                item_final.update_status(ItemStatus.PROVEN)
                await save_kb_item(item_final, client=None, db_path=effective_db_path)
        elif not item_final: logger.error(f"Item {unique_name} vanished before final PROVEN update.")
        return True # Success for non-provable item with statement

    # --- Proof Generation and Verification Phase ---
    logger.info(f"Starting Lean proof generation and verification phase for {unique_name}")
    lean_verification_success = False

    # Fetch item state just before the proof loop
    item_before_proof_loop = get_kb_item_by_name(unique_name, effective_db_path)
    if not item_before_proof_loop or not original_lean_statement_shell:
         logger.error(f"Cannot proceed to proof generation: Item {unique_name} missing or statement shell invalid before proof loop.")
         return False

    # Clear previous error log if starting fresh proof attempts
    if getattr(item_before_proof_loop, 'status', None) != ItemStatus.LEAN_VALIDATION_FAILED:
         if hasattr(item_before_proof_loop, 'lean_error_log') and getattr(item_before_proof_loop, 'lean_error_log', None) is not None:
             logger.debug(f"Clearing previous lean_error_log for {unique_name} (Status: {item_before_proof_loop.status.name if item_before_proof_loop.status else 'None'}).")
             item_before_proof_loop.lean_error_log = None
             await save_kb_item(item_before_proof_loop, client=None, db_path=effective_db_path) # Save cleared log

    # --- Proof Attempt Loop ---
    for attempt in range(LEAN_PROOF_MAX_ATTEMPTS):
        logger.info(f"Lean Proof Generation Attempt {attempt + 1}/{LEAN_PROOF_MAX_ATTEMPTS} for {unique_name}")
        # Fetch the latest item state, including any error log from previous verification attempt
        current_item_state = get_kb_item_by_name(unique_name, effective_db_path)
        if not current_item_state:
            logger.error(f"Item {unique_name} vanished mid-proof attempts!")
            return False

        # Set status to indicate generation is starting for this attempt
        if hasattr(current_item_state, 'update_status'):
            current_item_state.update_status(ItemStatus.LEAN_GENERATION_IN_PROGRESS, f"Proof attempt {attempt + 1}")
            await save_kb_item(current_item_state, client=None, db_path=effective_db_path)

        # Safely get attributes needed for the LLM call
        current_latex_proof = getattr(current_item_state, 'latex_proof', None)
        current_item_type_name = getattr(getattr(current_item_state, 'item_type', None), 'name', 'UNKNOWN_TYPE')
        current_lean_error_log = getattr(current_item_state, 'lean_error_log', None) # Use error from previous failed attempt

        # --- Call LLM for Proof Generation ---
        raw_llm_response = await _call_lean_proof_generator(
            lean_statement_shell=original_lean_statement_shell,
            latex_proof=current_latex_proof,
            unique_name=unique_name,
            item_type_name=current_item_type_name,
            dependencies=dependency_items,
            lean_error_log=current_lean_error_log, # Provide error log for refinement
            client=client
        )

        # Refresh item state after LLM call
        item_after_gen = get_kb_item_by_name(unique_name, effective_db_path)
        if not item_after_gen: raise Exception(f"Item {unique_name} vanished after proof generation call.")

        if not raw_llm_response:
            logger.warning(f"Lean proof generator returned no response on attempt {attempt + 1}")
            if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS: # If final attempt failed
                 if hasattr(item_after_gen, 'update_status'):
                     item_after_gen.update_status(ItemStatus.ERROR, f"LLM failed to generate proof on final attempt {attempt + 1}")
                     if hasattr(item_after_gen, 'lean_error_log'): item_after_gen.lean_error_log = "LLM provided no output."
                     await save_kb_item(item_after_gen, client=None, db_path=effective_db_path)
            continue # Try next attempt

        # --- Parse LLM Response ---
        generated_lean_code_from_llm = _extract_lean_code(raw_llm_response)
        if not generated_lean_code_from_llm:
            logger.warning(f"Failed to parse Lean code from proof generator on attempt {attempt + 1}.")
            error_message = f"LLM output parsing failed on proof attempt {attempt + 1}. Raw: {repr(raw_llm_response[:500])}"
            # Save raw response and parsing error
            if hasattr(item_after_gen, 'raw_ai_response'): item_after_gen.raw_ai_response = raw_llm_response
            if hasattr(item_after_gen, 'lean_error_log'): item_after_gen.lean_error_log = error_message
            if hasattr(item_after_gen, 'generation_prompt'): item_after_gen.generation_prompt = "Proof Gen Prompt (see logs/code)"
            if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS: # If final attempt failed parsing
                if hasattr(item_after_gen, 'update_status'): item_after_gen.update_status(ItemStatus.ERROR, f"LLM output parsing failed on final Lean proof attempt")
            await save_kb_item(item_after_gen, client=None, db_path=effective_db_path)
            continue # Try next attempt

        # Log generated code before attempting verification
        logger.info(f"--- LLM Generated Lean Code (Attempt {attempt + 1}) ---")
        logger.info(f"\n{generated_lean_code_from_llm}\n")
        logger.info(f"--- End LLM Generated Code ---")

        # --- Prepare and Verify Code ---
        # Update item with the latest generated code and set status for verification
        if hasattr(item_after_gen, 'lean_code'): item_after_gen.lean_code = generated_lean_code_from_llm
        if hasattr(item_after_gen, 'generation_prompt'): item_after_gen.generation_prompt = "Proof Gen Prompt (see logs/code)"
        if hasattr(item_after_gen, 'raw_ai_response'): item_after_gen.raw_ai_response = raw_llm_response # Store the successful generation
        if hasattr(item_after_gen, 'update_status'): item_after_gen.update_status(ItemStatus.LEAN_VALIDATION_PENDING)
        await save_kb_item(item_after_gen, client=None, db_path=effective_db_path) # Save before verification

        logger.info(f"Calling lean_interaction.check_and_compile_item for {unique_name} (Proof Attempt {attempt + 1})")
        try:
            # Call verification function (handles temp env, lake build, persistent update)
            verified, message = await lean_interaction.check_and_compile_item(
                unique_name=unique_name,
                db_path=effective_db_path,
                lake_executable_path=lake_executable_path,
                timeout_seconds=timeout_seconds
            )

            if verified:
                logger.info(f"Successfully verified Lean code for: {unique_name} on attempt {attempt + 1}. Message: {message}")
                # Status should now be PROVEN (set by check_and_compile_item)
                lean_verification_success = True
                break # Exit proof attempt loop successfully
            else:
                # Verification failed
                logger.warning(f"Verification failed for {unique_name} on proof attempt {attempt + 1}. Message: {message[:500]}...")
                # Status should be LEAN_VALIDATION_FAILED (set by check_and_compile_item)
                # Error log should also be saved by check_and_compile_item

                # Fetch item again to see the error log saved by check_and_compile
                item_after_fail = get_kb_item_by_name(unique_name, effective_db_path)
                latest_error_log = getattr(item_after_fail, 'lean_error_log', None) if item_after_fail else None

                if latest_error_log:
                    logger.warning(f"--- Lean Error Log (Attempt {attempt + 1}) ---")
                    logger.warning(f"\n{latest_error_log}\n")
                    logger.warning(f"--- End Lean Error Log ---")

                    # --- Optional: Attempt Automated Proof Repair ---
                    # Check if repair module is available and item code exists
                    if lean_proof_repair and hasattr(item_after_fail, 'lean_code') and item_after_fail.lean_code:
                        logger.info(f"Attempting automated proof repair for {unique_name} based on error log.")
                        try:
                            fix_applied, _ = lean_proof_repair.attempt_proof_repair(
                                item_after_fail.lean_code,
                                latest_error_log
                            )
                            if fix_applied:
                                logger.info(f"Automated proof repair heuristic applied for {unique_name}. Note: Re-verification within this loop is not implemented.")
                                # The next LLM attempt will receive the original error log,
                                # not reflecting the automated repair attempt.
                                # Could potentially save the repaired code back to DB here,
                                # but adds complexity (might overwrite user changes, needs re-verify logic).
                            else:
                                logger.debug(f"No automated fix applied by lean_proof_repair for {unique_name}.")
                        except Exception as repair_err:
                             logger.error(f"Error during automated proof repair attempt for {unique_name}: {repair_err}")
                    # --- End Optional Proof Repair ---
                else:
                    logger.warning(f"No specific error log captured in DB after failed verification attempt {attempt + 1}, message: {message}")

                # Continue to the next LLM proof generation attempt loop iteration
                continue

        except Exception as verify_err:
            # Catch errors from check_and_compile_item itself
            logger.exception(f"check_and_compile_item crashed unexpectedly for {unique_name} on proof attempt {attempt + 1}: {verify_err}")
            # Attempt to set item status to ERROR
            item_err_state = get_kb_item_by_name(unique_name, effective_db_path)
            if item_err_state and hasattr(item_err_state, 'update_status'):
                err_log_message = f"Verification system crashed: {str(verify_err)[:500]}"
                item_err_state.update_status(ItemStatus.ERROR, f"Lean verification system error")
                if hasattr(item_err_state, 'lean_error_log'): item_err_state.lean_error_log = err_log_message
                if hasattr(item_err_state, 'increment_failure_count'): item_err_state.increment_failure_count()
                await save_kb_item(item_err_state, client=None, db_path=effective_db_path)
                # Log error message if saved
                if getattr(item_err_state, 'lean_error_log', None):
                     logger.warning(f"--- Lean Error Log (Verification Crash - Attempt {attempt + 1}) ---")
                     logger.warning(f"\n{item_err_state.lean_error_log}\n")
                     logger.warning(f"--- End Lean Error Log ---")

            continue # Try next LLM attempt


    # --- Loop Finished ---
    end_time = time.time()
    duration = end_time - start_time

    if lean_verification_success:
        logger.info(f"Lean processing SUCCEEDED for {unique_name} in {duration:.2f} seconds.")
        return True
    else:
        logger.error(f"Failed to generate and verify Lean proof for {unique_name} after all attempts ({LEAN_PROOF_MAX_ATTEMPTS} attempts). Total time: {duration:.2f} seconds.")
        # Final status should be LEAN_VALIDATION_FAILED or ERROR, already set within the loop.
        # Fetch final state just for logging consistency.
        final_item = get_kb_item_by_name(unique_name, effective_db_path)
        final_status_name = getattr(getattr(final_item, 'status', None), 'name', 'UNKNOWN')
        logger.error(f"Final status for {unique_name}: {final_status_name}")
        # Double-check if somehow marked Proven - indicates a logic error somewhere.
        if final_status_name == ItemStatus.PROVEN.name:
            logger.warning(f"Item {unique_name} ended as PROVEN despite loop indicating failure. Assuming success due to final status.")
            return True
        # Otherwise, the failure outcome is correct.
        return False


# --- Batch Processing Function ---
async def process_pending_lean_items(
        client: GeminiClient,
        db_path: Optional[str] = None,
        limit: Optional[int] = None,
        process_statuses: Optional[List[ItemStatus]] = None,
        **kwargs # Pass other args like lake path, timeout to generate_and_verify_lean
    ):
    """Processes multiple items requiring Lean code generation and verification.

    Queries the database for items in specified statuses (defaulting to
    LATEX_ACCEPTED, PENDING_LEAN, LEAN_VALIDATION_FAILED). It then iterates
    through eligible items (checking status again before processing), calls
    `generate_and_verify_lean` for each one up to an optional limit, and logs
    summary statistics. Also handles non-provable items that only need statement
    generation.

    Args:
        client (GeminiClient): An initialized LLM client instance.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.
        limit (Optional[int]): Max number of items to process in this batch.
        process_statuses (Optional[List[ItemStatus]]): List of statuses to query for
            processing. Defaults to LATEX_ACCEPTED, PENDING_LEAN, LEAN_VALIDATION_FAILED.
        **kwargs: Additional keyword arguments (e.g., `lake_executable_path`,
            `timeout_seconds`) passed directly to `generate_and_verify_lean`.
    """
    # Dependency checks
    kbitem_available = KBItem is not None and not isinstance(KBItem, type('Dummy', (object,), {}))
    if kbitem_available:
        if not all([ItemStatus, ItemType, get_items_by_status, get_kb_item_by_name, save_kb_item, lean_proof_repair, lean_interaction]):
             logger.critical("Required modules not fully loaded. Cannot batch process Lean items.")
             return
    else:
        warnings.warn("Running Lean batch process with dummy KB types.", RuntimeWarning)

    effective_db_path = db_path or DEFAULT_DB_PATH
    # Default statuses to query if none provided
    if process_statuses is None:
         process_statuses_set = {ItemStatus.PENDING_LEAN, ItemStatus.LATEX_ACCEPTED, ItemStatus.LEAN_VALIDATION_FAILED}
    else:
         process_statuses_set = set(process_statuses) # Use a set for efficient lookup


    processed_count = 0
    success_count = 0
    fail_count = 0
    items_to_process_names = []

    logger.info(f"Starting Lean batch processing. Querying for items with statuses: {[s.name for s in process_statuses_set]}.")
    # --- Collect eligible items ---
    for status in process_statuses_set:
        if limit is not None and len(items_to_process_names) >= limit: break
        try:
            items_gen = get_items_by_status(status, effective_db_path)
            count_for_status = 0
            for item in items_gen:
                if limit is not None and len(items_to_process_names) >= limit: break
                item_unique_name = getattr(item, 'unique_name', None)
                if not item_unique_name: continue # Skip unnamed items

                # Handle non-provable items correctly during collection
                item_type = getattr(item, 'item_type', None)
                item_status = getattr(item, 'status', None) # Should match loop status
                item_lean_code = getattr(item, 'lean_code', None)
                needs_proof = item_type.requires_proof() if item_type else True

                if not needs_proof and item_status == ItemStatus.LATEX_ACCEPTED:
                     # If non-provable and already has code, mark PROVEN and skip adding to batch
                     if item_lean_code:
                         logger.info(f"Found non-provable item {item_unique_name} with code. Marking PROVEN directly.")
                         if hasattr(item, 'update_status'):
                             item.update_status(ItemStatus.PROVEN)
                             await save_kb_item(item, client=None, db_path=effective_db_path)
                         continue # Don't add to processing list
                     else:
                         # Needs statement generation, fall through to add
                         logger.info(f"Found non-provable item {item_unique_name} needing statement generation. Adding to list.")
                         pass

                # Add all other eligible items (provable, or non-provable needing statement)
                if item_unique_name not in items_to_process_names:
                    items_to_process_names.append(item_unique_name)
                    count_for_status += 1

            logger.debug(f"Found {count_for_status} potential items with status {status.name}")
            if limit is not None and len(items_to_process_names) >= limit: break
        except Exception as e:
            logger.error(f"Failed to retrieve items with status {status.name}: {e}")

    if not items_to_process_names:
         logger.info("No eligible items found requiring Lean processing in the specified statuses.")
         return

    logger.info(f"Collected {len(items_to_process_names)} unique items for Lean processing.")

    # --- Process collected items ---
    for unique_name in items_to_process_names:
        # Re-fetch item state before processing to ensure it's still eligible
        try:
            current_item_state = get_kb_item_by_name(unique_name, effective_db_path)
            if not current_item_state:
                 logger.warning(f"Skipping {unique_name}: Item disappeared before processing.")
                 continue

            item_status = getattr(current_item_state, 'status', None)
            item_type = getattr(current_item_state, 'item_type', None)
            item_lean_code = getattr(current_item_state, 'lean_code', None)
            needs_proof = item_type.requires_proof() if item_type else True

            # Double-check eligibility based on latest status and whether processing is needed
            is_eligible_status = item_status in process_statuses_set
            # Processing is needed if it requires proof, OR if it doesn't but lacks code
            needs_processing = needs_proof or (not needs_proof and not item_lean_code)

            if not is_eligible_status or not needs_processing:
                 logger.info(f"Skipping {unique_name}: Status ({item_status.name if item_status else 'None'}) or state changed, no longer eligible for processing.")
                 continue

        except Exception as fetch_err:
             logger.error(f"Error re-fetching state for {unique_name} before processing: {fetch_err}. Skipping.")
             continue

        # Proceed with processing the eligible item
        logger.info(f"--- Processing Lean for: {unique_name} (ID: {getattr(current_item_state, 'id', 'N/A')}, Status: {item_status.name if item_status else 'None'}) ---")
        processed_count += 1
        try:
            # Call the main processing function, passing through kwargs
            success = await generate_and_verify_lean(
                unique_name, client, effective_db_path, **kwargs
            )
            if success:
                success_count += 1
                logger.info(f"Successfully processed Lean for {unique_name}.")
            else:
                fail_count += 1
                logger.warning(f"Failed to process Lean for {unique_name} (see previous logs).")
        except Exception as e:
             # Catch unexpected errors from generate_and_verify_lean
             logger.exception(f"Critical error during batch processing of {unique_name}: {e}")
             fail_count += 1
             # Attempt to mark item as ERROR in DB
             try:
                  err_item = get_kb_item_by_name(unique_name, effective_db_path)
                  # Only set ERROR if not already PROVEN (shouldn't happen if exception occurred)
                  if err_item and getattr(err_item, 'status', None) != ItemStatus.PROVEN:
                     if hasattr(err_item, 'update_status'):
                         err_item.update_status(ItemStatus.ERROR, f"Batch Lean processing crashed: {str(e)[:500]}")
                         await save_kb_item(err_item, client=None, db_path=effective_db_path)
             except Exception as save_err:
                  logger.error(f"Failed to save ERROR status for {unique_name} after batch crash: {save_err}")

        logger.info(f"--- Finished processing Lean for {unique_name} ---")

    logger.info(f"Lean Batch Processing Complete. Total Processed: {processed_count}, Succeeded: {success_count}, Failed: {fail_count}")