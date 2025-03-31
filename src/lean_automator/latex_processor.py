# File: latex_processor.py

import asyncio
import warnings
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any

# --- Imports from other project modules ---
try:
    # Need KBItem definition, DB access functions, Status Enum, ItemType
    from lean_automator.kb_storage import (
        KBItem,
        ItemStatus,
        ItemType, # Import ItemType
        get_kb_item_by_name,
        save_kb_item,
        DEFAULT_DB_PATH
    )
    # Need LLM client
    from lean_automator.llm_call import GeminiClient
except ImportError:
    warnings.warn("latex_processor: Required modules (kb_storage, llm_call) not found.")
    KBItem = None; ItemStatus = None; ItemType = None; get_kb_item_by_name = None; save_kb_item = None; GeminiClient = None; DEFAULT_DB_PATH = None

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_REVIEW_CYCLES = 3
try:
    # Read from environment variable, fallback to default if not set or invalid
    MAX_REVIEW_CYCLES = int(os.getenv('LATEX_MAX_REVIEW_CYCLES', DEFAULT_MAX_REVIEW_CYCLES))
    if MAX_REVIEW_CYCLES < 0: # Ensure non-negative
         logger.warning(f"LATEX_MAX_REVIEW_CYCLES cannot be negative. Using default {DEFAULT_MAX_REVIEW_CYCLES}.")
         MAX_REVIEW_CYCLES = DEFAULT_MAX_REVIEW_CYCLES
except (ValueError, TypeError):
    logger.warning(f"Invalid value for LATEX_MAX_REVIEW_CYCLES environment variable. Using default {DEFAULT_MAX_REVIEW_CYCLES}.")
    MAX_REVIEW_CYCLES = DEFAULT_MAX_REVIEW_CYCLES

logger.info(f"Using MAX_REVIEW_CYCLES = {MAX_REVIEW_CYCLES}")

# --- Prompt Templates (Combined Statement & Proof) ---

# Note: This template now needs conditional logic based on item_type
COMBINED_GENERATOR_PROMPT_TEMPLATE = """
You are an expert mathematician assisting in building a knowledge base.

**Goal:** Generate the formal LaTeX for the item named `{unique_name}` ({item_type_name}).
**Description:** {nl_description}

**Context: This item builds upon the following prerequisite items:**
{dependency_context}

**Refinement (If Applicable):**
The previous attempt's statement was:
--- BEGIN STATEMENT ---
{current_statement}
--- END STATEMENT ---
{proof_context_for_refinement}
Reviewer Feedback: {review_feedback}
Please revise based on the feedback. Address issues in both the statement and the proof (if applicable).

**Instructions:**
1. Determine the item type: {item_type_name}.
2. Generate the LaTeX for the mathematical **statement**.
3. **If** the item type is one that requires a proof (Theorem, Lemma, Proposition, Corollary, Example), **also** generate a rigorous-style informal **proof** in LaTeX, justifying each step clearly and using only the provided Context items or standard mathematical reasoning based on the Lean 4 prelude (Nat, List, basic logic).
4. Ensure all LaTeX is syntactically correct and uses standard mathematical notation.
5. The statement should accurately reflect the Description and Context.
6. The proof (if generated) should logically follow from the Context items and prelude assumptions.
7. **Output Format:** Structure your response EXACTLY like this, including the markers:
--- BEGIN LATEX ---
{item_type_marker}
[LaTeX code for the statement...]
{proof_marker_and_content}
--- END LATEX ---

* Replace `{item_type_marker}` with the capitalized item type (e.g., THEOREM, DEFINITION).
* If generating a proof, include a `PROOF` line, followed by the LaTeX proof content. If not, omit the `PROOF` line and content entirely. Example for a theorem:
    --- BEGIN LATEX ---
    THEOREM
    [Statement LaTeX...]

    PROOF
    [Proof LaTeX...]
    --- END LATEX ---
* Example for a definition:
    --- BEGIN LATEX ---
    DEFINITION
    [Statement LaTeX...]
    --- END LATEX ---
"""

# Note: This template now needs conditional review instructions
COMBINED_REVIEWER_PROMPT_TEMPLATE = """
You are an expert mathematical reviewer evaluating a generated LaTeX statement and its accompanying informal proof (if applicable).

**Item Name:** {unique_name} ({item_type_name})
**Item Description:** {nl_description}

**Content to Review:**
--- BEGIN LATEX ---
{item_type_marker}
{latex_statement_to_review}
{proof_marker_and_content_to_review}
--- END LATEX ---

**Context: This item relies on the following prerequisite items:**
{dependency_context}

**Review Tasks for STATEMENT ({item_type_marker}):**
1.  **LaTeX Syntax:** Is the statement's LaTeX code syntactically valid?
2.  **Faithfulness:** Does the statement accurately represent the provided Description?
3.  **Consistency:** Is the notation/terminology consistent with the provided Context items?
4.  **Mathematical Plausibility:** Does the statement seem *mathematically plausible* given the Context? (Heuristic check for obvious errors).
5.  **Clarity & Notation:** Is the statement clear and using standard notation?

**Review Tasks for PROOF (If present):**
6.  **LaTeX Syntax:** Is the proof's LaTeX code syntactically valid?
7.  **Structure & Clarity:** Is the proof well-structured (e.g., induction steps clear, cases covered) and easy to follow?
8.  **Use of Context:** Does the proof correctly use/refer to the provided Context items?
9.  **Obvious Flaws:** Are there *obvious* logical gaps, contradictions, unjustified steps, or clear errors? (This is a heuristic check, NOT a guarantee of full logical rigor).

**Output Format:**
Provide your assessment in the following format EXACTLY:

Statement Judgment: [Accepted OR Rejected]
Statement Feedback: [Specific feedback if Rejected, otherwise "OK"]
Proof Judgment: [Accepted OR Rejected OR NA] (Use NA if no proof was expected/provided)
Proof Feedback: [Specific feedback if Rejected, otherwise "OK" or "NA"]
Overall Judgment: [Accepted OR Rejected] (Must be Accepted ONLY if both applicable Statement and Proof Judgments are Accepted)
"""

# --- Helper Functions ---

def _parse_combined_latex(text: Optional[str], item_type: ItemType) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts LaTeX statement and proof from the combined generator output.
    Returns (statement, proof) or (None, None) on failure.
    Proof is None if not expected or not found.
    """
    if not text:
        return None, None

    outer_match = re.search(r"---\s*BEGIN LATEX\s*---\s*(.*?)\s*---\s*END LATEX\s*---", text, re.DOTALL | re.IGNORECASE)
    if not outer_match:
        logger.warning("Could not find '--- BEGIN LATEX ---...--- END LATEX ---' block.")
        return None, None

    content = outer_match.group(1).strip()
    item_type_str = item_type.name # Get the expected type marker (e.g., "THEOREM")

    # Find the statement part after the type marker
    statement_match = re.search(rf"^\s*{item_type_str}\s*\n(.*?)(?:^\s*PROOF\s*\n|\Z)", content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if not statement_match:
        logger.warning(f"Could not find item type marker '{item_type_str}' or statement content.")
        return None, None

    statement = statement_match.group(1).strip()
    proof = None

    if item_type.requires_proof():
        # Find the proof part after the PROOF marker
        proof_match = re.search(r"^\s*PROOF\s*\n(.*)", content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if proof_match:
            proof = proof_match.group(1).strip()
        else:
            logger.warning(f"Expected PROOF marker/content for item type {item_type_str}, but not found.")
            # Return statement but no proof? Or fail parsing? Let's return statement for now.
            pass

    return statement, proof


def _parse_combined_review(text: Optional[str], proof_expected: bool) -> Tuple[str, str, Optional[str]]:
    """
    Parses the combined reviewer response.
    Returns: (overall_judgment: str, combined_feedback: str, error_location: Optional[str])
    error_location might be "Statement", "Proof", or None if accepted/unclear.
    """
    if not text:
        return "Rejected", "Reviewer response was empty.", "Unknown"

    # Defaults
    overall_judgment = "Rejected"
    stmt_judgment = "Rejected"
    proof_judgment = "Rejected" if proof_expected else "NA"
    stmt_feedback = ""
    proof_feedback = ""
    error_location = None

    # Extract judgments
    oj_match = re.search(r"Overall Judgment:\s*(Accepted|Rejected)", text, re.IGNORECASE)
    if oj_match:
        overall_judgment = "Accepted" if oj_match.group(1).strip().lower() == "accepted" else "Rejected"

    sj_match = re.search(r"Statement Judgment:\s*(Accepted|Rejected)", text, re.IGNORECASE)
    if sj_match:
        stmt_judgment = "Accepted" if sj_match.group(1).strip().lower() == "accepted" else "Rejected"

    pj_match = re.search(r"Proof Judgment:\s*(Accepted|Rejected|NA)", text, re.IGNORECASE)
    if pj_match:
        proof_judgment_str = pj_match.group(1).strip().upper()
        if proof_judgment_str in ["ACCEPTED", "REJECTED", "NA"]:
             proof_judgment = proof_judgment_str

    # Extract feedback sections (capture everything after the tag)
    sf_match = re.search(r"Statement Feedback:\s*(.*?)(?:Proof Judgment:|Proof Feedback:|Overall Judgment:|$)", text, re.DOTALL | re.IGNORECASE)
    if sf_match:
        stmt_feedback = sf_match.group(1).strip()

    pf_match = re.search(r"Proof Feedback:\s*(.*?)(?:Overall Judgment:|$)", text, re.DOTALL | re.IGNORECASE)
    if pf_match:
        proof_feedback = pf_match.group(1).strip()

    # Determine combined feedback and error location
    combined_feedback_parts = []
    if stmt_judgment != "Accepted" and stmt_feedback and stmt_feedback.lower() != 'ok':
        combined_feedback_parts.append(f"Statement Feedback: {stmt_feedback}")
        error_location = "Statement"
    if proof_judgment == "Rejected" and proof_feedback and proof_feedback.lower() != 'ok':
        combined_feedback_parts.append(f"Proof Feedback: {proof_feedback}")
        # If statement also failed, keep error_location as Statement? Or set to Both? Let's prioritize statement error.
        if error_location is None:
            error_location = "Proof"
        elif error_location == "Statement":
             error_location = "Both"


    combined_feedback = "\n".join(combined_feedback_parts) if combined_feedback_parts else "No specific feedback provided."

    # Sanity check: Overall judgment should be Rejected if any part is Rejected
    if overall_judgment == "Accepted" and (stmt_judgment == "Rejected" or proof_judgment == "Rejected"):
         logger.warning(f"Reviewer inconsistencies: Overall judgment is Accepted but Statement is {stmt_judgment} and Proof is {proof_judgment}. Overriding Overall to Rejected.")
         overall_judgment = "Rejected"
    # Sanity check: Overall must be Accepted if both parts are Accepted (or NA for proof)
    elif overall_judgment == "Rejected" and stmt_judgment == "Accepted" and proof_judgment in ["Accepted", "NA"]:
         logger.warning(f"Reviewer inconsistencies: Overall judgment is Rejected but Statement is Accepted and Proof is {proof_judgment}. Keeping Overall as Rejected based on text.")
         # Keep overall_judgment as Rejected, maybe feedback explains why

    if overall_judgment == "Accepted":
        combined_feedback = "OK" # Simplify feedback if accepted

    return overall_judgment, combined_feedback, error_location


def _build_dependency_context_string(dependencies: List[KBItem]) -> str:
    """Builds the context string for dependencies using statement LaTeX."""
    dependency_context = ""
    # Dependencies should have LATEX_ACCEPTED status
    valid_deps = [d for d in dependencies if d and d.latex_statement and d.status in {ItemStatus.PROVEN, ItemStatus.AXIOM_ACCEPTED, ItemStatus.DEFINITION_ADDED, ItemStatus.LATEX_ACCEPTED}]
    if valid_deps:
        for dep in valid_deps:
            # Use --- BEGIN/END STATEMENT --- marker for clarity? Or just the latex? Let's just use latex.
            dependency_context += f"- {dep.unique_name} ({dep.item_type.name}):\n  ```latex\n  {dep.latex_statement}\n  ```\n" # Using ```latex for context is ok for LLM
    else:
        dependency_context = "(None explicitly provided from KB)\n"
    return dependency_context


async def _call_latex_statement_and_proof_generator(
    item: KBItem, # Pass the whole item
    dependencies: List[KBItem],
    current_statement: Optional[str], # For refinement
    current_proof: Optional[str],     # For refinement
    review_feedback: Optional[str],   # For refinement
    client: GeminiClient
) -> Optional[str]:
    """Calls the LLM to generate combined LaTeX statement and proof (if needed). Returns raw response."""
    if not client:
        logger.error("_call_latex_generator: GeminiClient is not available.")
        raise ValueError("GeminiClient is not available.")

    dependency_context_str = _build_dependency_context_string(dependencies)
    item_type_name = item.item_type.name
    item_type_marker = item_type_name.upper() # E.g., THEOREM
    proof_expected = item.item_type.requires_proof()

    # Format refinement context based on what's available
    proof_context_for_refinement = ""
    if proof_expected and current_proof:
        proof_context_for_refinement = f"The previous proof attempt was:\n--- BEGIN PROOF ---\n{current_proof}\n--- END PROOF ---"
    elif proof_expected and not current_proof:
         proof_context_for_refinement = "(No previous proof attempt available)"
    # else: proof not expected, leave empty

    try:
        # Handle conditional formatting for refinement section
        base_prompt = COMBINED_GENERATOR_PROMPT_TEMPLATE
        if not (current_statement and review_feedback):
            # Remove the refinement section if it's the first attempt
            base_prompt = re.sub(r"\*\*Refinement \(If Applicable\):\*\*(.*?)\*\*Instructions:\*\*", "**Instructions:**", base_prompt, flags=re.DOTALL)
            prompt = base_prompt.format(
                unique_name=item.unique_name,
                item_type_name=item_type_name,
                nl_description=item.description_nl or "(No description provided)",
                dependency_context=dependency_context_str,
                item_type_marker=item_type_marker, # For instructions example
                proof_marker_and_content="" # Placeholder, handled by LLM logic
            )
        else:
            # Include refinement section
             prompt = base_prompt.format(
                unique_name=item.unique_name,
                item_type_name=item_type_name,
                nl_description=item.description_nl or "(No description provided)",
                dependency_context=dependency_context_str,
                current_statement=current_statement or "(Not available)",
                proof_context_for_refinement=proof_context_for_refinement,
                review_feedback=review_feedback or "(Not available)",
                item_type_marker=item_type_marker, # For instructions example
                proof_marker_and_content="" # Placeholder, handled by LLM logic
            )

        # Add final instruction about proof based on type
        if proof_expected:
             prompt = prompt.replace("{proof_marker_and_content}", "PROOF\n[LaTeX code for the proof...]")
        else:
             prompt = prompt.replace("{proof_marker_and_content}", "") # Remove placeholder entirely

    except KeyError as e:
        logger.error(f"Missing key in COMBINED_GENERATOR_PROMPT_TEMPLATE formatting: {e}")
        return None

    try:
        logger.debug(f"Sending combined LaTeX generation prompt for {item.unique_name}")
        response_text = await client.generate(prompt=prompt)
        return response_text # Return raw text for external parsing
    except Exception as e:
        logger.error(f"Error calling combined LaTeX generator LLM for {item.unique_name}: {e}")
        return None


async def _call_latex_statement_and_proof_reviewer(
    item_type: ItemType,
    latex_statement: str,
    latex_proof: Optional[str], # Only if proof_expected
    unique_name: str,
    nl_description: Optional[str],
    dependencies: List[KBItem],
    client: GeminiClient
) -> Optional[str]: # Return raw response text
    """Calls the LLM to review combined LaTeX statement and proof. Returns raw response."""
    if not client:
        logger.error("_call_latex_reviewer: GeminiClient is not available.")
        raise ValueError("GeminiClient is not available.")

    dependency_context_str = _build_dependency_context_string(dependencies)
    item_type_name = item_type.name
    item_type_marker = item_type_name.upper()
    proof_expected = item_type.requires_proof()

    proof_marker_and_content_to_review = ""
    if proof_expected and latex_proof:
        proof_marker_and_content_to_review = f"\n\nPROOF\n{latex_proof}"
    elif proof_expected and not latex_proof:
        # This case indicates an error - generator failed to provide expected proof
        logger.error(f"Reviewer called for provable type {item_type_name} but no proof provided.")
        return None # Cannot review
    # else: proof not expected, leave empty

    try:
        prompt = COMBINED_REVIEWER_PROMPT_TEMPLATE.format(
            unique_name=unique_name,
            item_type_name=item_type_name,
            nl_description=nl_description or "(No description provided)",
            item_type_marker=item_type_marker,
            latex_statement_to_review=latex_statement,
            proof_marker_and_content_to_review=proof_marker_and_content_to_review,
            dependency_context=dependency_context_str
        )
        # Adjust prompt if proof not expected - remove proof review tasks?
        if not proof_expected:
             prompt = re.sub(r"\*\*Review Tasks for PROOF \(If present\):\*\*(.*?)\*\*Output Format:\*\*", "**Output Format:**", prompt, flags=re.DOTALL)
             # Also adjust output format instruction if needed, tell it to output NA for proof parts

    except KeyError as e:
        logger.error(f"Missing key in COMBINED_REVIEWER_PROMPT_TEMPLATE formatting: {e}")
        return None

    try:
        logger.debug(f"Sending combined LaTeX review prompt for {unique_name}")
        response_text = await client.generate(prompt=prompt)
        return response_text # Return raw text for external parsing
    except Exception as e:
        logger.error(f"Error calling combined LaTeX reviewer LLM for {unique_name}: {e}")
        return None


# --- Main Orchestration Function ---

async def generate_and_review_latex(
    unique_name: str,
    client: GeminiClient,
    db_path: Optional[str] = None
) -> bool:
    """
    Manages the combined process of generating, reviewing, and refining LaTeX statement
    and proof (if applicable) for a given KBItem. Updates the item in the database.

    Returns:
        bool: True if LaTeX was successfully generated and accepted, False otherwise.
    """
    if not all([KBItem, ItemStatus, ItemType, get_kb_item_by_name, save_kb_item]):
        logger.critical("KB Storage module components not loaded correctly. Cannot process LaTeX.")
        return False
    if not client:
        logger.error("GeminiClient not provided. Cannot process LaTeX.")
        return False

    effective_db_path = db_path or DEFAULT_DB_PATH
    item = get_kb_item_by_name(unique_name, effective_db_path)

    if not item:
        logger.error(f"LaTeX Processing: Item not found: {unique_name}")
        return False

    # Define statuses that trigger processing
    trigger_statuses = {ItemStatus.PENDING, ItemStatus.PENDING_LATEX, ItemStatus.LATEX_REJECTED_FINAL}
    # Define statuses that mean LaTeX is already acceptable
    already_accepted_statuses = {ItemStatus.LATEX_ACCEPTED, ItemStatus.PROVEN, ItemStatus.DEFINITION_ADDED, ItemStatus.AXIOM_ACCEPTED}

    if item.status in already_accepted_statuses:
        logger.info(f"LaTeX Processing: Item {unique_name} status ({item.status.name}) indicates LaTeX is already acceptable. Skipping.")
        return True

    if item.status not in trigger_statuses:
        logger.warning(f"LaTeX Processing: Item {unique_name} not in a trigger status ({ {s.name for s in trigger_statuses} }). Current: {item.status.name}. Skipping.")
        return False

    original_status = item.status

    # --- Fetch Dependencies ---
    dependency_items: List[KBItem] = []
    logger.debug(f"Fetching dependencies for {unique_name}: {item.plan_dependencies}")
    for dep_name in item.plan_dependencies:
        dep_item = get_kb_item_by_name(dep_name, effective_db_path)
        if dep_item:
            # Need dependency's statement to be accepted
            if dep_item.latex_statement and dep_item.status in already_accepted_statuses:
                 dependency_items.append(dep_item)
            else:
                 logger.warning(f"Dependency '{dep_name}' for '{unique_name}' does not have accepted LaTeX statement (status: {dep_item.status.name}). Excluding from context.")
        else:
            logger.error(f"Dependency '{dep_name}' not found in KB for target '{unique_name}' during LaTeX processing.")


    # --- Start Processing ---
    try:
        item.update_status(ItemStatus.LATEX_GENERATION_IN_PROGRESS)
        await save_kb_item(item, client=None, db_path=effective_db_path)

        current_statement: Optional[str] = item.latex_statement # Start with existing if any
        current_proof: Optional[str] = item.latex_proof       # Start with existing if any
        review_feedback: Optional[str] = item.latex_review_feedback # Use previous feedback
        accepted = False
        proof_expected = item.item_type.requires_proof()

        for cycle in range(MAX_REVIEW_CYCLES):
            logger.info(f"Combined LaTeX Cycle {cycle + 1}/{MAX_REVIEW_CYCLES} for {unique_name}")

            # --- Step (a): Generate / Refine Combined LaTeX ---
            item.update_status(ItemStatus.LATEX_GENERATION_IN_PROGRESS)
            await save_kb_item(item, client=None, db_path=effective_db_path)

            raw_generator_response = await _call_latex_statement_and_proof_generator(
                item=item,
                dependencies=dependency_items,
                current_statement=current_statement,
                current_proof=current_proof,
                review_feedback=review_feedback,
                client=client
            )

            if not raw_generator_response:
                logger.warning(f"Combined LaTeX generator failed or returned no response for {unique_name} in cycle {cycle + 1}")
                item.update_status(ItemStatus.ERROR, f"LaTeX generator failed in cycle {cycle + 1}")
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False

            # --- Step (a.2): Parse Generator Output ---
            parsed_statement, parsed_proof = _parse_combined_latex(raw_generator_response, item.item_type)

            if not parsed_statement:
                logger.warning(f"Failed to parse statement from generator output for {unique_name} in cycle {cycle + 1}. Raw: {raw_generator_response[:500]}")
                item.update_status(ItemStatus.ERROR, f"LaTeX generator output parsing failed (statement) in cycle {cycle + 1}")
                item.raw_ai_response = raw_generator_response # Save raw for debugging
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False # Exit on parsing failure

            if proof_expected and not parsed_proof:
                logger.warning(f"Failed to parse proof from generator output for {unique_name} (type: {item.item_type.name}) in cycle {cycle + 1}. Raw: {raw_generator_response[:500]}")
                # Allow continuing to review? Maybe reviewer catches it. Or fail here? Let's fail for now if proof expected but missing.
                item.update_status(ItemStatus.ERROR, f"LaTeX generator output parsing failed (proof) in cycle {cycle + 1}")
                item.raw_ai_response = raw_generator_response # Save raw for debugging
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False


            current_statement = parsed_statement # Update working versions
            current_proof = parsed_proof

            # --- Step (b): Review Combined LaTeX ---
            item.update_status(ItemStatus.LATEX_REVIEW_IN_PROGRESS)
            # Save candidate statement/proof before review? Or just in raw response? Let's store in raw.
            item.raw_ai_response = raw_generator_response # Store generator output
            await save_kb_item(item, client=None, db_path=effective_db_path)

            raw_reviewer_response = await _call_latex_statement_and_proof_reviewer(
                item_type=item.item_type,
                latex_statement=current_statement,
                latex_proof=current_proof, # Pass None if not expected/parsed
                unique_name=unique_name,
                nl_description=item.description_nl,
                dependencies=dependency_items,
                client=client
            )

            if not raw_reviewer_response:
                 logger.warning(f"LaTeX reviewer failed or returned no response for {unique_name} in cycle {cycle + 1}")
                 item.update_status(ItemStatus.ERROR, f"LaTeX reviewer failed in cycle {cycle + 1}")
                 await save_kb_item(item, client=None, db_path=effective_db_path)
                 return False # Exit on reviewer failure

            # --- Step (c): Parse and Process Review ---
            judgment, feedback, error_loc = _parse_combined_review(raw_reviewer_response, proof_expected)

            # Get fresh item state before update
            item = get_kb_item_by_name(unique_name, effective_db_path)
            if not item: raise Exception("Item vanished during review processing")

            if judgment == "Accepted":
                logger.info(f"Combined LaTeX for {unique_name} accepted by reviewer in cycle {cycle + 1}.")
                accepted = True
                item.latex_statement = current_statement # Set accepted content
                item.latex_proof = current_proof         # Set accepted content (will be None if not proof_expected)
                item.update_status(ItemStatus.LATEX_ACCEPTED, review_feedback=None) # Clear feedback
                # Save final state - trigger embedding generation for latex_statement
                await save_kb_item(item, client=client, db_path=effective_db_path)
                break # Exit loop successfully
            else:
                logger.warning(f"Combined LaTeX for {unique_name} rejected by reviewer in cycle {cycle + 1}. Location: {error_loc or 'N/A'}. Feedback: {feedback[:200]}...")
                review_feedback = feedback # Store feedback for the next generation attempt
                item.latex_review_feedback = review_feedback # Save feedback to item
                item.update_status(ItemStatus.PENDING_LATEX_REVIEW) # Indicate needs another cycle
                await save_kb_item(item, client=None, db_path=effective_db_path) # Save feedback and status

        # --- After Loop ---
        if not accepted:
            logger.error(f"Combined LaTeX for {unique_name} rejected after {MAX_REVIEW_CYCLES} cycles.")
            item = get_kb_item_by_name(unique_name, effective_db_path) # Get final state
            if item:
                 item.update_status(ItemStatus.LATEX_REJECTED_FINAL, review_feedback=review_feedback) # Keep last feedback
                 await save_kb_item(item, client=None, db_path=effective_db_path)
            return False
        else:
             return True # Was accepted within the loop

    except Exception as e:
        logger.exception(f"Unhandled exception during combined LaTeX processing for {unique_name}: {e}")
        try:
             item_err = get_kb_item_by_name(unique_name, effective_db_path)
             if item_err and item_err.status not in already_accepted_statuses and item_err.status != ItemStatus.LATEX_REJECTED_FINAL:
                 item_err.update_status(ItemStatus.ERROR, f"Unhandled LaTeX processor exception: {e}")
                 await save_kb_item(item_err, client=None, db_path=effective_db_path)
        except Exception as final_err:
             logger.error(f"Failed to save final error state for {unique_name} after exception: {final_err}")
        return False


# --- Optional: Batch Processing Function ---
# (This function remains largely the same, just calls the updated generate_and_review_latex)
async def process_pending_latex_items(
        client: GeminiClient,
        db_path: Optional[str] = None,
        limit: Optional[int] = None,
        process_statuses: Optional[List[ItemStatus]] = None
    ):
    """Finds items needing LaTeX processing and runs the generate/review cycle."""
    if not all([ItemStatus, get_items_by_status, get_kb_item_by_name, save_kb_item]): # Add checks
         logger.critical("KB Storage components not loaded correctly. Cannot batch process.")
         return

    effective_db_path = db_path or DEFAULT_DB_PATH
    if process_statuses is None:
         process_statuses = {ItemStatus.PENDING_LATEX, ItemStatus.PENDING, ItemStatus.LATEX_REJECTED_FINAL}

    processed_count = 0
    success_count = 0
    fail_count = 0
    items_to_process = []

    logger.info(f"Querying for items with statuses: {[s.name for s in process_statuses]}")
    for status in process_statuses:
        try:
            items_gen = get_items_by_status(status, effective_db_path)
            count_for_status = 0
            for item in items_gen:
                 if limit is not None and len(items_to_process) >= limit:
                     break
                 items_to_process.append(item)
                 count_for_status += 1
            logger.debug(f"Found {count_for_status} items with status {status.name}")
            if limit is not None and len(items_to_process) >= limit:
                 break
        except Exception as e:
            logger.error(f"Failed to retrieve items with status {status.name}: {e}")

    if not items_to_process:
         logger.info("No items found requiring LaTeX processing in the specified statuses.")
         return

    logger.info(f"Found {len(items_to_process)} items for LaTeX processing.")

    for item in items_to_process:
        # Check status again before processing
        current_item_state = get_kb_item_by_name(item.unique_name, effective_db_path)
        if not current_item_state or current_item_state.status not in process_statuses:
            logger.info(f"Skipping {item.unique_name} as its status changed ({current_item_state.status.name if current_item_state else 'deleted'}) or item disappeared.")
            continue

        logger.info(f"--- Processing LaTeX for: {item.unique_name} (ID: {item.id}, Status: {item.status.name}) ---")
        try:
            # Call the main combined processing function
            success = await generate_and_review_latex(item.unique_name, client, effective_db_path)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
             logger.error(f"Error processing item {item.unique_name} in batch: {e}")
             fail_count += 1
             try:
                  err_item = get_kb_item_by_name(item.unique_name, effective_db_path)
                  if err_item and err_item.status not in {ItemStatus.PROVEN, ItemStatus.LATEX_ACCEPTED}:
                     err_item.update_status(ItemStatus.ERROR, f"Batch processing error: {e}")
                     await save_kb_item(err_item, client=None, db_path=effective_db_path)
             except Exception as save_err:
                  logger.error(f"Failed to save error status for {item.unique_name} during batch: {save_err}")

        processed_count += 1
        logger.info(f"--- Finished processing {item.unique_name} ---")
        # await asyncio.sleep(0.5) # Optional delay

    logger.info(f"LaTeX Batch Processing Complete. Total Processed: {processed_count}, Succeeded: {success_count}, Failed: {fail_count}")