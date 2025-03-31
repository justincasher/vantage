# File: lean_processor.py

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
    KBItem = None; ItemStatus = None; ItemType = None; get_kb_item_by_name = None; save_kb_item = None; get_items_by_status = None; GeminiClient = None; DEFAULT_DB_PATH = None; lean_interaction = None; lean_proof_repair = None;

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
    """Extracts Lean header code (signature + sorry) using markers or markdown."""
    if not text: return None
    header = None
    extracted_via = None
    stripped_text = text.strip()
    custom_marker_regex = r"---\s*BEGIN LEAN HEADER\s*---\s*(.*?)\s*---\s*END LEAN HEADER\s*---"
    match_custom = re.search(custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE)
    if match_custom:
        header = match_custom.group(1).strip()
        extracted_via = "custom markers"
        logger.debug(f"Extracted header using {extracted_via}.")
    else:
        logger.warning("Could not find '--- BEGIN/END LEAN HEADER ---' markers. Trying strict markdown block...")
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            header = match_md.group(1).strip()
            extracted_via = "strict markdown block"
            logger.debug(f"Extracted header using {extracted_via}.")
        else:
            logger.warning("Could not find strict markdown block ('```lean...```') encompassing the whole response.")

    if header:
        stripped_header = header.rstrip()
        if not stripped_header.endswith(':= sorry'):
            logger.warning("Parsed Lean header does not end with ':= sorry'. Appending it.")
            if stripped_header.endswith(' :='): header = stripped_header + " sorry"
            elif stripped_header.endswith(' :'): header = stripped_header + "= sorry"
            else:
                header_no_comments = re.sub(r"\s*--.*$", "", stripped_header, flags=re.MULTILINE).rstrip()
                if header_no_comments.endswith(' :='): header = header_no_comments + " sorry"
                elif header_no_comments.endswith(' :'): header = header_no_comments + "= sorry"
                else: header = header_no_comments + " := sorry"
        return header
    else:
        logger.error(f"Failed to extract Lean header using any method. Raw text received: {repr(text)}")
        return None

def _extract_lean_code(text: Optional[str]) -> Optional[str]:
    """Extracts full Lean code using markers or markdown."""
    if not text: return None
    code = None
    extracted_via = None
    stripped_text = text.strip()
    custom_marker_regex = r"---\s*BEGIN LEAN CODE\s*---\s*(.*?)\s*---\s*END LEAN CODE\s*---"
    match_custom = re.search(custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE)
    if match_custom:
        code = match_custom.group(1).strip()
        extracted_via = "custom markers"; logger.debug(f"Extracted code using {extracted_via}.")
        return code
    else:
        logger.warning("Could not find '--- BEGIN/END LEAN CODE ---' markers. Trying strict markdown block...")
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            code = match_md.group(1).strip()
            extracted_via = "strict markdown block"; logger.debug(f"Extracted code using {extracted_via}.")
            return code
        else:
            logger.warning("Could not find strict markdown block ('```lean...```') encompassing the whole response.")

    logger.error(f"Failed to extract Lean code using any method. Raw text received: {repr(text)}")
    return None


def _build_lean_dependency_context_for_statement(dependencies: List[KBItem]) -> str:
    """Builds a simple text list of dependency names and types for the statement prompt."""
    if not dependencies: return "(None)"
    ctx = ""
    for dep in dependencies: ctx += f"- {dep.unique_name} ({dep.item_type.name})\n"
    return ctx.strip()

def _build_lean_dependency_context_for_proof(dependencies: List[KBItem]) -> str:
    """Builds a context string containing the full Lean code of PROVEN dependencies for the proof prompt."""
    dependency_context = ""
    valid_statuses = {ItemStatus.PROVEN, ItemStatus.AXIOM_ACCEPTED, ItemStatus.DEFINITION_ADDED}
    # Ensure KBItem has lean_code attribute even in dummy definition
    proven_deps = [d for d in dependencies if d and getattr(d, 'lean_code', None) and getattr(d, 'status', None) in valid_statuses]

    if proven_deps:
        for dep in proven_deps:
            dependency_context += f"-- Dependency: {dep.unique_name} ({dep.item_type.name})\n"
            dependency_context += f"-- BEGIN {dep.unique_name} LEAN --\n"
            dependency_context += f"{dep.lean_code.strip()}\n"
            dependency_context += f"-- END {dep.unique_name} LEAN --\n\n"
    else:
        dependency_context = "-- (No specific proven dependencies provided from KB. Rely on Lean prelude.) --\n"

    return dependency_context.strip()

# _generate_dependency_import_block function removed as it's no longer correct/needed


# --- LLM Caller Functions ---
async def _call_lean_statement_generator(
    item: KBItem, dependencies: List[KBItem], statement_error_feedback: Optional[str], client: GeminiClient
) -> Optional[str]:
    """Calls the LLM to generate the Lean statement signature."""
    if not client: raise ValueError("GeminiClient not available.")
    # Ensure KBItem has attributes even in dummy definition
    item_unique_name = getattr(item, 'unique_name', 'UNKNOWN_ITEM')
    item_type_name = getattr(getattr(item, 'item_type', None), 'name', 'UNKNOWN_TYPE')
    item_latex_statement = getattr(item, 'latex_statement', None)


    dep_context = _build_lean_dependency_context_for_statement(dependencies)

    try:
        base_prompt = LEAN_STATEMENT_GENERATOR_PROMPT
        if not statement_error_feedback:
             # Remove the optional feedback section if not provided
             base_prompt = re.sub(r"\n\*\*Refinement Feedback \(If Applicable\):\*\*\n.*?(\n\*\*Instructions:\*\*)", r"\1", base_prompt, flags=re.DOTALL | re.MULTILINE)
             prompt = base_prompt.format(
                 unique_name=item_unique_name,
                 item_type_name=item_type_name,
                 latex_statement=item_latex_statement or "(LaTeX Statement Missing!)",
                 dependency_context_for_statement=dep_context
             )
        else:
             prompt = base_prompt.format(
                 unique_name=item_unique_name,
                 item_type_name=item_type_name,
                 latex_statement=item_latex_statement or "(LaTeX Statement Missing!)",
                 dependency_context_for_statement=dep_context,
                 statement_error_feedback=statement_error_feedback
             )
    except KeyError as e:
        logger.error(f"Statement Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error formatting statement prompt for {item_unique_name}: {e}")
        return None


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
    """Calls the LLM to generate the Lean proof using the provided shell."""
    if not client: raise ValueError("GeminiClient not available.")
    if not lean_statement_shell or ":= sorry" not in lean_statement_shell:
         logger.error(f"Internal Error: _call_lean_proof_generator for {unique_name} called without a valid shell ending in ':= sorry'. Shell: {lean_statement_shell}")
         raise ValueError("Internal Error: _call_lean_proof_generator called without a valid shell ending in ':= sorry'.")

    dep_context = _build_lean_dependency_context_for_proof(dependencies)

    try:
        base_prompt = LEAN_PROOF_GENERATOR_PROMPT
        if not lean_error_log:
             # Remove the optional error log section if not provided
             base_prompt = re.sub(r"\n\*\*Previous Attempt Error \(If Applicable\):\*\*(.*?)\n\*\*Instructions:\*\*", "\n**Instructions:**", base_prompt, flags=re.DOTALL | re.MULTILINE)
             prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                unique_name=unique_name,
                item_type_name=item_type_name
            )
        else:
             prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                lean_error_log=lean_error_log,
                unique_name=unique_name,
                item_type_name=item_type_name
            )
    except KeyError as e:
        logger.error(f"Proof Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error formatting proof prompt for {unique_name}: {e}")
        return None

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
    """
    Generates and verifies Lean code for a KBItem using a two-step LLM process.
    Relies on lean_interaction.check_and_compile_item which uses a persistent
    shared library for dependencies. Updates persistent library on success.
    """
    start_time = time.time()
    # Check if actual KBItem type is available, otherwise use dummy check
    kbitem_available = KBItem is not None and not isinstance(KBItem, type('Dummy', (object,), {}))

    if kbitem_available:
        if not all([KBItem, ItemStatus, ItemType, get_kb_item_by_name, save_kb_item, lean_interaction, get_items_by_status, lean_proof_repair]):
            logger.critical("Required real modules not loaded correctly. Cannot process Lean.")
            return False
    else: # If using dummy types, just log a warning
         warnings.warn("Running with dummy KB types. Full functionality unavailable.", RuntimeWarning)

    if not client:
        logger.error("GeminiClient missing.")
        return False
    if not hasattr(lean_interaction, 'check_and_compile_item'):
         logger.critical("lean_interaction.check_and_compile_item missing.")
         return False
    if not hasattr(lean_proof_repair, 'attempt_proof_repair'):
         logger.critical("lean_proof_repair.attempt_proof_repair missing.")
         return False

    effective_db_path = db_path or DEFAULT_DB_PATH

    item = get_kb_item_by_name(unique_name, effective_db_path)
    if not item:
        logger.error(f"Lean Proc: Item not found: {unique_name}")
        return False

    # Get attributes safely in case using dummy item
    item_status = getattr(item, 'status', None)
    item_type = getattr(item, 'item_type', None)
    item_lean_code = getattr(item, 'lean_code', None)
    item_latex_statement = getattr(item, 'latex_statement', None)
    item_plan_dependencies = getattr(item, 'plan_dependencies', [])
    item_latex_proof = getattr(item, 'latex_proof', None)

    if item_status == ItemStatus.PROVEN:
        logger.info(f"Lean Proc: Already PROVEN: {unique_name}")
        return True

    trigger_statuses = {ItemStatus.LATEX_ACCEPTED, ItemStatus.PENDING_LEAN, ItemStatus.LEAN_VALIDATION_FAILED}
    if item_status not in trigger_statuses:
        logger.warning(f"Lean Proc: Item {unique_name} not in a trigger status. Status: {item_status.name if item_status else 'None'}")
        return False

    if not item_latex_statement:
         logger.error(f"Lean Proc: Missing latex_statement for {unique_name}")
         if hasattr(item, 'update_status'):
             item.update_status(ItemStatus.ERROR, "Missing latex_statement for Lean generation.")
             await save_kb_item(item, client=None, db_path=effective_db_path)
         return False

    proof_required = item_type.requires_proof() if item_type else False
    if not proof_required:
        logger.info(f"Lean Proc: No proof required for {unique_name} ({item_type.name if item_type else 'N/A'}). Marking PROVEN.")
        if not item_lean_code:
             logger.warning(f"Lean Proc: Non-provable item {unique_name} has no Lean code. Requires statement generation first.")
             # Need statement generation even if no proof needed
             # Fall through to statement generation
             pass
        else:
            # Already has code, no proof needed, mark proven
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.PROVEN)
                await save_kb_item(item, client=None, db_path=effective_db_path)
            return True

    dependency_items: List[KBItem] = []
    valid_dep_statuses = {ItemStatus.PROVEN, ItemStatus.AXIOM_ACCEPTED, ItemStatus.DEFINITION_ADDED}
    logger.debug(f"Checking dependencies for {unique_name}: {item_plan_dependencies}")
    all_deps_ready = True
    for dep_name in item_plan_dependencies:
        dep_item = get_kb_item_by_name(dep_name, effective_db_path)
        dep_status = getattr(dep_item, 'status', None)
        dep_code_exists = bool(getattr(dep_item, 'lean_code', None))

        if not dep_item or dep_status not in valid_dep_statuses or not dep_code_exists:
            logger.error(f"Dependency '{dep_name}' for '{unique_name}' not ready (status={dep_status.name if dep_status else 'MISSING'}, code_exists={dep_code_exists}). Cannot proceed.")
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.ERROR, f"Dependency '{dep_name}' not ready.")
                await save_kb_item(item, client=None, db_path=effective_db_path)
            all_deps_ready = False
            break # No need to check further dependencies
        dependency_items.append(dep_item)

    if not all_deps_ready:
        return False # Dependency check failed

    logger.debug(f"All {len(dependency_items)} dependencies for {unique_name} are assumed available in shared library.")

    # No longer need to generate import block here - lean_interaction handles it for the target file
    # dependency_import_block = _generate_dependency_import_block(dependency_items) # Removed
    # logger.debug(f"Generated import block for {unique_name}:\n{dependency_import_block}") # Removed

    original_lean_statement_shell = item_lean_code
    statement_error_feedback = None

    needs_statement_generation = not item_lean_code or item_status in {ItemStatus.LATEX_ACCEPTED, ItemStatus.PENDING_LEAN}

    if needs_statement_generation:
        logger.info(f"Starting Lean statement generation phase for {unique_name}")
        statement_accepted = False
        for attempt in range(LEAN_STATEMENT_MAX_ATTEMPTS):
            logger.info(f"Lean Statement Attempt {attempt + 1}/{LEAN_STATEMENT_MAX_ATTEMPTS} for {unique_name}")
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.LEAN_GENERATION_IN_PROGRESS, f"Statement attempt {attempt + 1}")
                await save_kb_item(item, client=None, db_path=effective_db_path)

            # Fetch potentially updated dependency items? Usually not needed for statement gen.
            raw_response = await _call_lean_statement_generator(item, dependency_items, statement_error_feedback, client)
            parsed_header = _extract_lean_header(raw_response)

            if not parsed_header:
                 logger.warning(f"Failed to parse Lean header on attempt {attempt + 1}.")
                 statement_error_feedback = f"LLM output did not contain a valid Lean header (attempt {attempt+1}). Raw response: {repr(raw_response[:500])}"
                 if hasattr(item, 'raw_ai_response'): item.raw_ai_response = raw_response
                 if hasattr(item, 'generation_prompt'): item.generation_prompt = "Statement Gen Prompt (see logs/code)" # Placeholder
                 await save_kb_item(item, client=None, db_path=effective_db_path)
                 continue

            logger.info(f"Lean statement shell generated successfully for {unique_name}")
            original_lean_statement_shell = parsed_header
            if hasattr(item, 'lean_code'): item.lean_code = original_lean_statement_shell
            if hasattr(item, 'generation_prompt'): item.generation_prompt = "Statement Gen Prompt (see logs/code)" # Placeholder
            if hasattr(item, 'raw_ai_response'): item.raw_ai_response = raw_response
            await save_kb_item(item, client=None, db_path=effective_db_path)
            statement_accepted = True
            break # Exit loop on success

        if not statement_accepted:
            logger.error(f"Failed to generate valid Lean statement shell for {unique_name} after {LEAN_STATEMENT_MAX_ATTEMPTS} attempts.")
            if hasattr(item, 'update_status'):
                item.update_status(ItemStatus.ERROR, f"Failed Lean statement generation after {LEAN_STATEMENT_MAX_ATTEMPTS} attempts. Last feedback: {statement_error_feedback}")
                if hasattr(item, 'lean_error_log'): item.lean_error_log = statement_error_feedback
                await save_kb_item(item, client=None, db_path=effective_db_path)
            return False
    else:
         # Already has lean code, use it as the shell
         logger.info(f"Skipping Lean statement generation for {unique_name}, using existing shell.")
         original_lean_statement_shell = item_lean_code
         if not original_lean_statement_shell or ":= sorry" not in original_lean_statement_shell:
             logger.error(f"Existing lean_code for {unique_name} is invalid or missing ':= sorry'. Cannot proceed to proof generation.")
             if hasattr(item, 'update_status'):
                 item.update_status(ItemStatus.ERROR, "Invalid existing lean_code shell for proof generation.")
                 await save_kb_item(item, client=None, db_path=effective_db_path)
             return False

    # Handle non-provable items again after potential statement generation
    if not proof_required:
        logger.info(f"Lean Proc: Statement generated/present for non-provable {unique_name}. Marking PROVEN.")
        # Re-fetch item to ensure we have the latest version before updating status
        item_final = get_kb_item_by_name(unique_name, effective_db_path)
        if item_final and getattr(item_final, 'status', None) != ItemStatus.PROVEN:
            if hasattr(item_final, 'update_status'):
                item_final.update_status(ItemStatus.PROVEN)
                await save_kb_item(item_final, client=None, db_path=effective_db_path)
        elif not item_final:
             logger.error(f"Item {unique_name} disappeared before final PROVEN update for non-provable.")
        return True # Return True as statement exists and no proof needed


    # --- Proof Generation Phase ---
    logger.info(f"Starting Lean proof generation phase for {unique_name}")
    lean_verification_success = False

    # Fetch item state just before the loop begins
    item_before_proof_loop = get_kb_item_by_name(unique_name, effective_db_path)
    if not item_before_proof_loop or not original_lean_statement_shell:
         logger.error(f"Cannot proceed to proof generation: Item {unique_name} missing or shell invalid before proof loop.")
         return False

    # Clear error log if status is not already LEAN_VALIDATION_FAILED
    if getattr(item_before_proof_loop, 'status', None) != ItemStatus.LEAN_VALIDATION_FAILED:
         if hasattr(item_before_proof_loop, 'lean_error_log') and getattr(item_before_proof_loop, 'lean_error_log', None) is not None:
             logger.debug(f"Clearing previous error log for {unique_name} as status is {item_before_proof_loop.status.name if item_before_proof_loop.status else 'None'}")
             item_before_proof_loop.lean_error_log = None
             await save_kb_item(item_before_proof_loop, client=None, db_path=effective_db_path)


    for attempt in range(LEAN_PROOF_MAX_ATTEMPTS):
        logger.info(f"Lean Proof Attempt {attempt + 1}/{LEAN_PROOF_MAX_ATTEMPTS} for {unique_name}")
        # Fetch the latest state within the loop for error log feedback
        current_item_state = get_kb_item_by_name(unique_name, effective_db_path)
        if not current_item_state:
            logger.error(f"Item {unique_name} disappeared mid-proof attempts!")
            return False # Critical error

        # Update status to indicate generation is in progress
        if hasattr(current_item_state, 'update_status'):
            current_item_state.update_status(ItemStatus.LEAN_GENERATION_IN_PROGRESS, f"Proof attempt {attempt + 1}")
            await save_kb_item(current_item_state, client=None, db_path=effective_db_path)

        # Get attributes needed for the call safely
        current_latex_proof = getattr(current_item_state, 'latex_proof', None)
        current_item_type_name = getattr(getattr(current_item_state, 'item_type', None), 'name', 'UNKNOWN_TYPE')
        current_lean_error_log = getattr(current_item_state, 'lean_error_log', None)


        raw_llm_response = await _call_lean_proof_generator(
            lean_statement_shell=original_lean_statement_shell,
            latex_proof=current_latex_proof,
            unique_name=unique_name, # Use unique_name directly
            item_type_name=current_item_type_name,
            dependencies=dependency_items, # Pass fetched dependencies for context
            lean_error_log=current_lean_error_log,
            client=client
        )

        if not raw_llm_response:
            logger.warning(f"Lean proof generator returned no response on attempt {attempt + 1}")
            if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS:
                 # Update status on final attempt failure
                 final_item_state = get_kb_item_by_name(unique_name, effective_db_path)
                 if final_item_state and hasattr(final_item_state, 'update_status'):
                     final_item_state.update_status(ItemStatus.ERROR, f"LLM failed to generate proof on final attempt {attempt + 1}")
                     if hasattr(final_item_state, 'lean_error_log'): final_item_state.lean_error_log = "LLM provided no output."
                     await save_kb_item(final_item_state, client=None, db_path=effective_db_path)
            continue # Try next attempt

        generated_lean_code_from_llm = _extract_lean_code(raw_llm_response)
        if not generated_lean_code_from_llm:
            logger.warning(f"Failed to parse Lean code from proof generator on attempt {attempt + 1}.")
            item_parse_fail = get_kb_item_by_name(unique_name, effective_db_path)
            if item_parse_fail:
                if hasattr(item_parse_fail, 'raw_ai_response'): item_parse_fail.raw_ai_response = raw_llm_response
                error_message = f"LLM output parsing failed on proof attempt {attempt + 1}. Raw: {repr(raw_llm_response[:500])}"
                if hasattr(item_parse_fail, 'lean_error_log'): item_parse_fail.lean_error_log = error_message
                if hasattr(item_parse_fail, 'generation_prompt'): item_parse_fail.generation_prompt = "Proof Gen Prompt (see logs/code)" # Placeholder
                if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS:
                    if hasattr(item_parse_fail, 'update_status'): item_parse_fail.update_status(ItemStatus.ERROR, f"LLM output parsing failed on final Lean proof attempt")
                await save_kb_item(item_parse_fail, client=None, db_path=effective_db_path)
            continue # Try next attempt

        # Log the generated code before verification
        logger.info(f"--- LLM Generated Code (Attempt {attempt + 1}) ---")
        logger.info(f"\n{generated_lean_code_from_llm}\n")
        logger.info(f"--- End LLM Generated Code (Attempt {attempt + 1}) ---")

        # --- Prepare item for verification ---
        # The generated code should contain the full definition/theorem including proof
        # No need to prepend import block here - lean_interaction._create_temp_env... handles imports for the target file
        final_lean_code = generated_lean_code_from_llm

        item_to_verify = get_kb_item_by_name(unique_name, effective_db_path)
        if not item_to_verify:
            logger.error(f"Item {unique_name} vanished before Lean verification attempt {attempt + 1}!")
            return False # Critical error

        # Update item with the code to be verified
        if hasattr(item_to_verify, 'lean_code'): item_to_verify.lean_code = final_lean_code
        if hasattr(item_to_verify, 'generation_prompt'): item_to_verify.generation_prompt = "Proof Gen Prompt (see logs/code)" # Placeholder
        if hasattr(item_to_verify, 'raw_ai_response'): item_to_verify.raw_ai_response = raw_llm_response
        if hasattr(item_to_verify, 'update_status'): item_to_verify.update_status(ItemStatus.LEAN_VALIDATION_PENDING)
        await save_kb_item(item_to_verify, client=None, db_path=effective_db_path)

        logger.info(f"Calling lean_interaction.check_and_compile_item for {unique_name} (Proof Attempt {attempt + 1})")

        try:
            # Call the updated lean_interaction function
            # It now handles requiring the shared library and updating it on success
            verified, message = await lean_interaction.check_and_compile_item(
                unique_name=unique_name,
                db_path=effective_db_path,
                lake_executable_path=lake_executable_path,
                timeout_seconds=timeout_seconds
                # Pass shared lib info if check_and_compile_item needs it explicitly
                # (currently reads from env vars/constants within lean_interaction)
            )

            if verified:
                logger.info(f"Successfully verified Lean code for: {unique_name} on attempt {attempt + 1}.")
                # check_and_compile_item now handles DB status update to PROVEN and persistent lib update
                lean_verification_success = True
                break # Exit loop on success
            else:
                # Verification failed
                logger.warning(f"Verification failed for {unique_name} on proof attempt {attempt + 1}. Message: {message[:500]}...")
                # check_and_compile_item already updated status to LEAN_VALIDATION_FAILED and saved error log.

                # Fetch the item again to get the error log for the next LLM attempt or final state
                item_after_fail = get_kb_item_by_name(unique_name, effective_db_path)
                if item_after_fail:
                     latest_error_log = getattr(item_after_fail, 'lean_error_log', None)
                     if latest_error_log:
                         logger.warning(f"--- Lean Error Log (Attempt {attempt + 1}) ---")
                         logger.warning(f"\n{latest_error_log}\n")
                         logger.warning(f"--- End Lean Error Log (Attempt {attempt + 1}) ---")
                     else: # Should not happen if check_and_compile saves log on failure
                         logger.warning(f"No specific error log captured in DB for attempt {attempt + 1}, message: {message}")

                     # --- Optional: Proof Repair ---
                     # Consider if repair logic should be attempted here.
                     # Current repair logic is minimal/disabled.
                     if hasattr(item_after_fail, 'lean_code') and latest_error_log:
                         fix_applied, _ = lean_proof_repair.attempt_proof_repair(
                             item_after_fail.lean_code,
                             latest_error_log
                         )
                         if fix_applied:
                              logger.info(f"Automated proof repair applied for {unique_name}. Re-verification *not* implemented in this loop.")
                              # If repair modifies code, the next LLM attempt might get stale code?
                              # Or should we re-verify immediately after repair? Adds complexity.
                              # For now, just log that repair was applied. The next LLM attempt
                              # will use the error log from the *original* failed verification.
                         else:
                              logger.debug(f"No automated fix applied for {unique_name} on attempt {attempt + 1}.")
                     # --- End Optional Proof Repair ---

                else: # Should not happen if item exists
                     logger.error(f"Item {unique_name} vanished after failed verification attempt {attempt + 1}!")
                     return False

                # Continue to the next LLM attempt loop iteration (if attempts remain)
                continue # Go to next attempt

        except Exception as verify_err:
            logger.exception(f"check_and_compile_item failed unexpectedly for {unique_name} on proof attempt {attempt + 1}: {verify_err}")
            item_err_state = get_kb_item_by_name(unique_name, effective_db_path)
            if item_err_state and hasattr(item_err_state, 'update_status'):
                err_log_message = f"Verification system crashed: {verify_err}"
                item_err_state.update_status(ItemStatus.ERROR, f"Lean verification system error: {verify_err}")
                if hasattr(item_err_state, 'lean_error_log'): item_err_state.lean_error_log = err_log_message
                if hasattr(item_err_state, 'increment_failure_count'): item_err_state.increment_failure_count()
                await save_kb_item(item_err_state, client=None, db_path=effective_db_path)
                # Log error if available
                if getattr(item_err_state, 'lean_error_log', None):
                     logger.warning(f"--- Lean Error Log (Verification Crash - Attempt {attempt + 1}) ---")
                     logger.warning(f"\n{item_err_state.lean_error_log}\n")
                     logger.warning(f"--- End Lean Error Log ---")

            continue # Try next attempt


    # --- Loop Finished ---
    end_time = time.time()
    duration = end_time - start_time

    if lean_verification_success:
        logger.info(f"Lean processing SUCCEEDED for {unique_name} in {duration:.2f} seconds.")
        return True
    else:
        logger.error(f"Failed to generate/verify Lean proof for {unique_name} after all attempts. Total time: {duration:.2f} seconds.")
        # Final status should be LEAN_VALIDATION_FAILED or ERROR, set by check_and_compile or exception handling
        # Re-fetch just to double-check status if needed for logging/return value, but don't change DB state here.
        final_item = get_kb_item_by_name(unique_name, effective_db_path)
        if final_item and getattr(final_item, 'status', None) == ItemStatus.PROVEN:
             # This case indicates maybe check_and_compile succeeded but loop logic was wrong? Unlikely with current structure.
            logger.warning(f"Item {unique_name} is PROVEN despite loop indicating failure. Assuming success.")
            return True
        # Otherwise, failure is the correct outcome
        return False


# --- Batch Processing Function ---
async def process_pending_lean_items(
        client: GeminiClient,
        db_path: Optional[str] = None,
        limit: Optional[int] = None,
        process_statuses: Optional[List[ItemStatus]] = None,
        **kwargs # Pass other args like lake path, timeout
    ):
    """Processes items in specified statuses using generate_and_verify_lean."""
    # Check if actual KBItem type is available
    kbitem_available = KBItem is not None and not isinstance(KBItem, type('Dummy', (object,), {}))
    if kbitem_available:
        if not all([ItemStatus, ItemType, get_items_by_status, get_kb_item_by_name, save_kb_item, lean_proof_repair]):
             logger.critical("KB Storage components or proof repair module not loaded correctly. Cannot batch process Lean items.")
             return
    else:
        warnings.warn("Running batch process with dummy KB types. Full functionality unavailable.", RuntimeWarning)


    effective_db_path = db_path or DEFAULT_DB_PATH
    if process_statuses is None:
         process_statuses = {ItemStatus.PENDING_LEAN, ItemStatus.LATEX_ACCEPTED, ItemStatus.LEAN_VALIDATION_FAILED}
    else:
         process_statuses = set(process_statuses) # Ensure it's a set


    processed_count = 0
    success_count = 0
    fail_count = 0
    items_to_process_names = []

    logger.info(f"Querying for items with statuses: {[s.name for s in process_statuses]} for Lean processing.")
    for status in process_statuses:
        try:
            items_gen = get_items_by_status(status, effective_db_path)
            count_for_status = 0
            for item in items_gen:
                # Check limit
                if limit is not None and len(items_to_process_names) >= limit:
                    logger.info(f"Reached processing limit of {limit} items.")
                    break # Break inner loop

                item_unique_name = getattr(item, 'unique_name', None)
                if not item_unique_name: continue # Skip items without a name

                # Handle non-provable items needing only statement generation
                item_type = getattr(item, 'item_type', None)
                item_status = getattr(item, 'status', None)
                item_lean_code = getattr(item, 'lean_code', None)
                needs_proof = item_type.requires_proof() if item_type else True # Assume proof needed if type unknown

                if not needs_proof and item_status == ItemStatus.LATEX_ACCEPTED:
                     logger.info(f"Found non-provable item {item_unique_name} ({item_type.name if item_type else 'N/A'}) with status LATEX_ACCEPTED.")
                     if item_lean_code:
                         logger.info(f"Item {item_unique_name} has code, marking PROVEN.")
                         if hasattr(item, 'update_status'):
                             item.update_status(ItemStatus.PROVEN)
                             await save_kb_item(item, client=None, db_path=effective_db_path)
                         # Don't add to processing list
                         continue
                     else:
                         logger.info(f"Item {item_unique_name} needs statement generation. Adding to process list.")
                         # Fall through to add to list below

                # Add eligible items to list if not already present
                if item_unique_name not in items_to_process_names:
                    items_to_process_names.append(item_unique_name)
                    count_for_status += 1

            logger.debug(f"Found {count_for_status} potential items with status {status.name}")
            # Check limit again after iterating through a status
            if limit is not None and len(items_to_process_names) >= limit:
                break # Break outer loop

        except Exception as e:
            logger.error(f"Failed to retrieve items with status {status.name}: {e}")

    if not items_to_process_names:
         logger.info("No items found requiring Lean proof processing in the specified statuses.")
         return

    logger.info(f"Found {len(items_to_process_names)} unique items for Lean processing.")

    # Process items one by one
    for unique_name in items_to_process_names:
        # Re-fetch item state before processing to ensure it's still eligible
        current_item_state = get_kb_item_by_name(unique_name, effective_db_path)
        if not current_item_state:
             logger.warning(f"Skipping {unique_name} as it could not be re-fetched before processing.")
             continue

        item_status = getattr(current_item_state, 'status', None)
        item_type = getattr(current_item_state, 'item_type', None)
        item_lean_code = getattr(current_item_state, 'lean_code', None)
        needs_proof = item_type.requires_proof() if item_type else True

        # Check eligibility again
        is_eligible_status = item_status in process_statuses
        # Needs processing if it requires proof OR if it doesn't require proof but lacks code
        needs_processing = needs_proof or (not needs_proof and not item_lean_code)

        if not is_eligible_status or not needs_processing:
             logger.info(f"Skipping {unique_name} as its status ({item_status.name if item_status else 'None'}) or state changed, making it ineligible.")
             continue

        logger.info(f"--- Processing Lean for: {unique_name} (ID: {getattr(current_item_state, 'id', 'N/A')}, Status: {item_status.name if item_status else 'None'}) ---")
        try:
            # Call the main processing function
            success = await generate_and_verify_lean(
                unique_name, client, effective_db_path, **kwargs
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
             # Catch unexpected errors during the processing of a single item
             logger.exception(f"Critical error during generate_and_verify_lean for {unique_name}: {e}")
             fail_count += 1
             # Attempt to mark the item with an error status in the DB
             try:
                  err_item = get_kb_item_by_name(unique_name, effective_db_path)
                  if err_item and getattr(err_item, 'status', None) != ItemStatus.PROVEN:
                     if hasattr(err_item, 'update_status'):
                         err_item.update_status(ItemStatus.ERROR, f"Batch Lean processing crashed: {e}")
                         await save_kb_item(err_item, client=None, db_path=effective_db_path)
             except Exception as save_err:
                  logger.error(f"Failed to save ERROR status for {unique_name} after batch processing crash: {save_err}")

        processed_count += 1
        logger.info(f"--- Finished processing Lean for {unique_name} ---")

    logger.info(f"Lean Batch Processing Complete. Total Processed: {processed_count}, Succeeded: {success_count}, Failed: {fail_count}")