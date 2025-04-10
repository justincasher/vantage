# File: lean_automator/latex/processor.py

"""Handles LaTeX generation and review for Knowledge Base items using an LLM.

This module orchestrates the process of generating LaTeX representations (both
statement and optional informal proof) for mathematical items stored in the
Knowledge Base (`KBItem`). It uses a provided LLM client (`GeminiClient`)
to generate and subsequently review the LaTeX content based on predefined prompts
and the item's context (description, dependencies). The process involves iterative
refinement cycles until the LaTeX is accepted by the LLM reviewer or a maximum
number of cycles is reached. It updates the status and content of the KBItem
in the database accordingly.
"""

import asyncio
import warnings
import logging
import os
import re
from typing import Optional, Tuple, List, Dict, Any

# --- Imports from other project modules ---
try:
    from lean_automator.config.loader import APP_CONFIG
except ImportError:
    warnings.warn(
        "config_loader.APP_CONFIG not found. Default settings may be used.",
        ImportWarning,
    )
    APP_CONFIG = {}  # Provide an empty dict as a fallback

try:
    # Need KBItem definition, DB access functions, Status Enum, ItemType
    from lean_automator.kb.storage import (
        KBItem,
        ItemStatus,
        ItemType,  # Import ItemType
        get_kb_item_by_name,
        save_kb_item,
        DEFAULT_DB_PATH,
        get_items_by_status,  # Added for batch processing
    )

    # Need LLM client
    from lean_automator.llm.caller import GeminiClient
except ImportError:
    warnings.warn("latex_processor: Required modules (kb_storage, llm_call) not found.")
    KBItem = None
    ItemStatus = None
    ItemType = None
    get_kb_item_by_name = None
    save_kb_item = None
    GeminiClient = None
    DEFAULT_DB_PATH = None
    get_items_by_status = None  # type: ignore

# --- Logging ---
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MAX_REVIEW_CYCLES = 3

# Get max review cycles from loaded configuration, falling back to default
# Assumes APP_CONFIG['latex']['max_review_cycles'] is loaded as an int by config_loader
# or provides the default if keys are missing.
MAX_REVIEW_CYCLES = APP_CONFIG.get("latex", {}).get(
    "max_review_cycles", DEFAULT_MAX_REVIEW_CYCLES
)

# Validate the value (ensure it's an int and non-negative)
if not isinstance(MAX_REVIEW_CYCLES, int) or MAX_REVIEW_CYCLES < 0:
    logger.warning(
        f"Invalid or negative value found for config ['latex']['max_review_cycles'] ('{MAX_REVIEW_CYCLES}'). "
        f"Using default {DEFAULT_MAX_REVIEW_CYCLES}."
    )
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


def _parse_combined_latex(
    text: Optional[str], item_type: ItemType
) -> Tuple[Optional[str], Optional[str]]:
    """Extracts LaTeX statement and proof from combined generator output. (Internal Helper)

    Parses text expected to contain '--- BEGIN LATEX --- ... --- END LATEX ---' blocks,
    extracting the statement following the item type marker and the proof following
    the 'PROOF' marker (if the item type requires a proof).

    Args:
        text (Optional[str]): The raw text output from the LaTeX generator LLM.
        item_type (ItemType): The expected type of the item, used to locate the
            start of the statement and determine if a proof is expected.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - The extracted LaTeX statement string, or None if parsing fails.
            - The extracted LaTeX proof string if the item type requires proof and
              it was found, otherwise None.
        Returns (None, None) if the main block markers are not found or the
        statement cannot be parsed.
    """
    if not text:
        return None, None

    outer_match = re.search(
        r"---\s*BEGIN LATEX\s*---\s*(.*?)\s*---\s*END LATEX\s*---",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not outer_match:
        logger.warning(
            "Could not find '--- BEGIN LATEX ---...--- END LATEX ---' block."
        )
        return None, None

    content = outer_match.group(1).strip()
    item_type_str = item_type.name  # Get the expected type marker (e.g., "THEOREM")

    # Find the statement part after the type marker
    # Match from start of content, looking for type marker, newline, then capture group
    # Stop capture group before optional PROOF marker or end of string (\Z)
    statement_match = re.search(
        rf"^\s*{re.escape(item_type_str)}\s*\n(.*?)(?:^\s*PROOF\s*\n|\Z)",
        content,
        re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if not statement_match:
        logger.warning(
            f"Could not find item type marker '{item_type_str}' or statement content."
        )
        return None, None

    statement = statement_match.group(1).strip()
    proof = None

    if item_type.requires_proof():
        # Find the proof part after the PROOF marker, starting search from beginning of content block
        proof_match = re.search(
            r"^\s*PROOF\s*\n(.*)", content, re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        if proof_match:
            proof = proof_match.group(1).strip()
        else:
            logger.warning(
                f"Expected PROOF marker/content for item type {item_type_str}, but not found."
            )
            # Proceed with the successfully parsed statement, proof remains None
            pass

    return statement, proof


def _parse_combined_review(
    text: Optional[str], proof_expected: bool
) -> Tuple[str, str, Optional[str]]:
    """Parses the structured response from the LaTeX reviewer LLM. (Internal Helper)

    Extracts the overall judgment ("Accepted" or "Rejected"), combines feedback
    from statement and proof sections (if rejected), and identifies the primary
    location of the rejection ("Statement", "Proof", "Both", or None if accepted).

    Args:
        text (Optional[str]): The raw text output from the LaTeX reviewer LLM.
        proof_expected (bool): Whether a proof was expected for this item type,
            used to interpret the Proof Judgment field.

    Returns:
        Tuple[str, str, Optional[str]]: A tuple containing:
            - overall_judgment (str): "Accepted" or "Rejected".
            - combined_feedback (str): Consolidated feedback string if rejected,
              or "OK" if accepted.
            - error_location (Optional[str]): "Statement", "Proof", "Both" if
              rejected, None if accepted or location unclear. Defaults to "Rejected"
              with generic feedback if parsing fails completely.
    """
    if not text:
        return "Rejected", "Reviewer response was empty.", "Unknown"

    # Defaults in case parsing fails
    overall_judgment = "Rejected"
    stmt_judgment = "Rejected"
    proof_judgment = (
        "Rejected" if proof_expected else "NA"
    )  # Match review prompt options
    stmt_feedback = ""
    proof_feedback = ""
    error_location = None  # None means accepted or unclear rejection location

    # Extract judgments using regex, case-insensitive
    oj_match = re.search(
        r"Overall Judgment:\s*(Accepted|Rejected)", text, re.IGNORECASE
    )
    if oj_match:
        overall_judgment = (
            "Accepted"
            if oj_match.group(1).strip().lower() == "accepted"
            else "Rejected"
        )

    sj_match = re.search(
        r"Statement Judgment:\s*(Accepted|Rejected)", text, re.IGNORECASE
    )
    if sj_match:
        stmt_judgment = (
            "Accepted"
            if sj_match.group(1).strip().lower() == "accepted"
            else "Rejected"
        )

    pj_match = re.search(
        r"Proof Judgment:\s*(Accepted|Rejected|NA)", text, re.IGNORECASE
    )
    if pj_match:
        proof_judgment_str = pj_match.group(1).strip().upper()
        # Validate against expected values
        if proof_judgment_str in ["ACCEPTED", "REJECTED", "NA"]:
            proof_judgment = proof_judgment_str
        else:  # If unexpected value, treat as rejected if proof was expected
            proof_judgment = "Rejected" if proof_expected else "NA"
            logger.warning(
                f"Unexpected Proof Judgment value '{proof_judgment_str}', treating as {proof_judgment}."
            )

    # Extract feedback sections robustly (capture everything until next known section or end)
    sf_match = re.search(
        r"Statement Feedback:\s*(.*?)(?:Proof Judgment:|Proof Feedback:|Overall Judgment:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if sf_match:
        stmt_feedback = sf_match.group(1).strip()

    pf_match = re.search(
        r"Proof Feedback:\s*(.*?)(?:Overall Judgment:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if pf_match:
        proof_feedback = pf_match.group(1).strip()

    # Determine combined feedback and primary error location if rejected
    combined_feedback_parts = []
    if stmt_judgment != "Accepted" and stmt_feedback and stmt_feedback.lower() != "ok":
        combined_feedback_parts.append(f"Statement Feedback: {stmt_feedback}")
        error_location = "Statement"  # Statement rejected

    # Check proof only if it was expected and judged
    if (
        proof_expected
        and proof_judgment == "Rejected"
        and proof_feedback
        and proof_feedback.lower() != "ok"
    ):
        combined_feedback_parts.append(f"Proof Feedback: {proof_feedback}")
        # Determine combined error location
        if error_location == "Statement":
            error_location = "Both"  # Both rejected
        else:
            error_location = "Proof"  # Only proof rejected

    combined_feedback = "\n".join(combined_feedback_parts).strip()
    # Provide default feedback if rejected but no specific feedback captured
    if overall_judgment == "Rejected" and not combined_feedback:
        combined_feedback = (
            "Rejected by reviewer, but specific feedback could not be parsed."
        )
        if error_location is None:  # Try to assign general location if possible
            if stmt_judgment == "Rejected":
                error_location = "Statement"
            elif proof_judgment == "Rejected":
                error_location = "Proof"
            else:
                error_location = "Unknown"

    # Sanity Checks / Overrides based on component judgments
    # If Overall says Accepted but a component is Rejected -> Force Reject
    if overall_judgment == "Accepted" and (
        stmt_judgment == "Rejected" or (proof_expected and proof_judgment == "Rejected")
    ):
        logger.warning(
            f"Reviewer inconsistency: Overall Judgment 'Accepted' but Statement='{stmt_judgment}', Proof='{proof_judgment}'. Forcing Overall Judgment to 'Rejected'."
        )
        overall_judgment = "Rejected"
        if (
            not combined_feedback or combined_feedback == "OK"
        ):  # Ensure feedback exists if forced reject
            combined_feedback = "Rejected due to component judgment inconsistency (Statement or Proof rejected)."
            # Update error location if needed
            if error_location is None:
                if stmt_judgment == "Rejected":
                    error_location = "Statement"
                if proof_judgment == "Rejected":
                    error_location = "Proof" if error_location is None else "Both"

    # If overall judgment is finally accepted, simplify feedback
    if overall_judgment == "Accepted":
        combined_feedback = "OK"
        error_location = None

    return overall_judgment, combined_feedback, error_location


def _build_dependency_context_string(dependencies: List[KBItem]) -> str:
    """Builds a formatted string of dependency statements for LLM context. (Internal Helper)

    Filters dependencies to include only those with an accepted LaTeX statement
    (status indicates PROVEN, LATEX_ACCEPTED, etc.) and formats them as a list
    with their unique name, type, and LaTeX statement.

    Args:
        dependencies (List[KBItem]): A list of KBItem objects representing the
            dependencies of the item being processed.

    Returns:
        str: A formatted string containing the context of valid dependencies,
        or a default message if no valid dependencies are found.
    """
    dependency_context = ""
    # Define statuses indicating acceptable LaTeX statement for context
    already_accepted_statuses = {
        ItemStatus.LATEX_ACCEPTED,
        ItemStatus.PROVEN,
        ItemStatus.DEFINITION_ADDED,
        ItemStatus.AXIOM_ACCEPTED,
    }
    # Filter dependencies: must exist, have a statement, and have an accepted status
    valid_deps = [
        d
        for d in dependencies
        if d and d.latex_statement and d.status in already_accepted_statuses
    ]

    if valid_deps:
        for dep in valid_deps:
            # Format each dependency clearly for the LLM
            dependency_context += f"- {dep.unique_name} ({dep.item_type.name}):\n  ```latex\n  {dep.latex_statement}\n  ```\n"
    else:
        dependency_context = "(None explicitly provided from KB or dependencies lack accepted LaTeX statements)\n"
    return dependency_context


async def _call_latex_statement_and_proof_generator(
    item: KBItem,
    dependencies: List[KBItem],
    current_statement: Optional[str],
    current_proof: Optional[str],
    review_feedback: Optional[str],
    client: GeminiClient,
) -> Optional[str]:
    """Calls the LLM to generate/refine LaTeX statement and proof. (Internal Helper)

    Formats the prompt using `COMBINED_GENERATOR_PROMPT_TEMPLATE`, including
    item details, dependency context, and potentially refinement instructions based
    on previous attempts (current statement/proof and feedback).

    Args:
        item (KBItem): The KBItem being processed.
        dependencies (List[KBItem]): List of dependency KBItems for context.
        current_statement (Optional[str]): The LaTeX statement from the previous
            cycle, used for refinement context. None for the first attempt.
        current_proof (Optional[str]): The LaTeX proof from the previous cycle,
            used for refinement context (if applicable). None for the first attempt.
        review_feedback (Optional[str]): Feedback from the previous review cycle,
            used for refinement context. None for the first attempt.
        client (GeminiClient): The initialized LLM client instance.

    Returns:
        Optional[str]: The raw text response from the LLM generator, or None if
        the client is unavailable, prompt formatting fails, or the API call errors.

    Raises:
        ValueError: If the GeminiClient is not available.
    """
    if not client:
        logger.error("_call_latex_generator: GeminiClient is not available.")
        raise ValueError("GeminiClient is not available.")

    dependency_context_str = _build_dependency_context_string(dependencies)
    item_type_name = item.item_type.name
    item_type_marker = item_type_name.upper()
    proof_expected = item.item_type.requires_proof()

    # Format context about the previous proof attempt for refinement prompt
    proof_context_for_refinement = ""
    if proof_expected:
        if current_proof:
            proof_context_for_refinement = f"The previous proof attempt was:\n--- BEGIN PROOF ---\n{current_proof}\n--- END PROOF ---"
        else:
            # Indicate if proof expected but none available from previous attempt
            proof_context_for_refinement = (
                "(No previous proof attempt available or applicable)"
            )

    try:
        # Choose base prompt and format, removing refinement section if first attempt
        base_prompt = COMBINED_GENERATOR_PROMPT_TEMPLATE
        is_first_attempt = not (current_statement and review_feedback)

        if is_first_attempt:
            # Remove the whole refinement section using regex for robustness
            base_prompt = re.sub(
                r"\*\*Refinement \(If Applicable\):\*\*.*?\*\*Instructions:\*\*",
                "**Instructions:**",
                base_prompt,
                flags=re.DOTALL,
            )
            # Format without refinement placeholders
            prompt = base_prompt.format(
                unique_name=item.unique_name,
                item_type_name=item_type_name,
                nl_description=item.description_nl or "(No description provided)",
                dependency_context=dependency_context_str,
                item_type_marker=item_type_marker,  # For example in instructions
                proof_marker_and_content="{proof_marker_and_content}",  # Placeholder still needed
            )
        else:
            # Format with refinement placeholders
            prompt = base_prompt.format(
                unique_name=item.unique_name,
                item_type_name=item_type_name,
                nl_description=item.description_nl or "(No description provided)",
                dependency_context=dependency_context_str,
                current_statement=current_statement or "(Not available)",
                proof_context_for_refinement=proof_context_for_refinement,
                review_feedback=review_feedback or "(Not available)",
                item_type_marker=item_type_marker,  # For example in instructions
                proof_marker_and_content="{proof_marker_and_content}",  # Placeholder still needed
            )

        # Replace the final placeholder based on whether proof is expected
        if proof_expected:
            prompt = prompt.replace(
                "{proof_marker_and_content}", "PROOF\n[LaTeX code for the proof...]"
            )
        else:
            prompt = prompt.replace("{proof_marker_and_content}", "")  # Remove entirely

    except KeyError as e:
        logger.error(
            f"Missing key in COMBINED_GENERATOR_PROMPT_TEMPLATE formatting: {e}. Prompt state: {prompt[:500] if 'prompt' in locals() else 'N/A'}"
        )
        return None

    try:
        logger.debug(
            f"Sending combined LaTeX generation prompt for {item.unique_name} (Cycle context: {'Refinement' if not is_first_attempt else 'First attempt'})"
        )
        # Assuming client.generate handles the API call and returns text
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        # Catch exceptions during the API call itself
        logger.error(
            f"Error calling combined LaTeX generator LLM for {item.unique_name}: {e}"
        )
        return None


async def _call_latex_statement_and_proof_reviewer(
    item_type: ItemType,
    latex_statement: str,
    latex_proof: Optional[str],
    unique_name: str,
    nl_description: Optional[str],
    dependencies: List[KBItem],
    client: GeminiClient,
) -> Optional[str]:
    """Calls the LLM to review the generated LaTeX statement and proof. (Internal Helper)

    Formats the prompt using `COMBINED_REVIEWER_PROMPT_TEMPLATE`, providing the
    generated LaTeX, item details, and dependency context for the LLM reviewer.

    Args:
        item_type (ItemType): The type of the KBItem being reviewed.
        latex_statement (str): The generated LaTeX statement to be reviewed.
        latex_proof (Optional[str]): The generated LaTeX proof to be reviewed.
            Should be None if the `item_type` does not require a proof.
        unique_name (str): The unique name of the KBItem.
        nl_description (Optional[str]): The natural language description for context.
        dependencies (List[KBItem]): List of dependency KBItems for context.
        client (GeminiClient): The initialized LLM client instance.

    Returns:
        Optional[str]: The raw text response from the LLM reviewer, or None if
        the client is unavailable, prompt formatting fails, the API call errors,
        or if a proof was expected but not provided.

    Raises:
        ValueError: If the GeminiClient is not available.
    """
    if not client:
        logger.error("_call_latex_reviewer: GeminiClient is not available.")
        raise ValueError("GeminiClient is not available.")

    dependency_context_str = _build_dependency_context_string(dependencies)
    item_type_name = item_type.name
    item_type_marker = item_type_name.upper()
    proof_expected = item_type.requires_proof()

    # Format the proof part for the review prompt
    proof_marker_and_content_to_review = ""
    if proof_expected:
        if latex_proof:
            # Include PROOF marker and content if expected and provided
            proof_marker_and_content_to_review = f"\n\nPROOF\n{latex_proof}"
        else:
            # Error condition: proof expected based on type, but not provided
            logger.error(
                f"Reviewer called for provable type {item_type_name} ('{unique_name}') but no proof content was provided to review."
            )
            return None  # Cannot proceed with review if expected proof is missing

    try:
        # Format the base review prompt
        prompt = COMBINED_REVIEWER_PROMPT_TEMPLATE.format(
            unique_name=unique_name,
            item_type_name=item_type_name,
            nl_description=nl_description or "(No description provided)",
            item_type_marker=item_type_marker,
            latex_statement_to_review=latex_statement,
            proof_marker_and_content_to_review=proof_marker_and_content_to_review,
            dependency_context=dependency_context_str,
        )
        # If proof is not expected, remove the proof review tasks section
        if not proof_expected:
            prompt = re.sub(
                r"\*\*Review Tasks for PROOF \(If present\):\*\*.*?\*\*Output Format:\*\*",
                "**Output Format:**",
                prompt,
                flags=re.DOTALL,
            )
            # Optionally, could add instruction to output NA for proof judgment/feedback

    except KeyError as e:
        logger.error(
            f"Missing key in COMBINED_REVIEWER_PROMPT_TEMPLATE formatting: {e}. Prompt state: {prompt[:500] if 'prompt' in locals() else 'N/A'}"
        )
        return None

    try:
        logger.debug(f"Sending combined LaTeX review prompt for {unique_name}")
        # Assuming client.generate handles the API call and returns text
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        # Catch exceptions during the API call itself
        logger.error(
            f"Error calling combined LaTeX reviewer LLM for {unique_name}: {e}"
        )
        return None


# --- Main Orchestration Function ---


async def generate_and_review_latex(
    unique_name: str, client: GeminiClient, db_path: Optional[str] = None
) -> bool:
    """Generates, reviews, and refines LaTeX for a KBItem iteratively.

    This function orchestrates the main workflow for obtaining accepted LaTeX
    for a given `KBItem` identified by its `unique_name`.

    1. Fetches the item and its dependencies from the database.

    2. Checks if the item's current status requires LaTeX processing.

    3. Enters a loop (up to `MAX_REVIEW_CYCLES`):

        a. Calls the LLM generator (`_call_latex_statement_and_proof_generator`)
           to produce/refine the LaTeX statement and proof (if applicable),
           providing previous feedback if it's not the first cycle.

        b. Parses the generator's output using `_parse_combined_latex`.

        c. Calls the LLM reviewer (`_call_latex_statement_and_proof_reviewer`)
           to evaluate the generated LaTeX.

        d. Parses the reviewer's judgment and feedback using `_parse_combined_review`.

        e. If accepted, updates the KBItem with the accepted LaTeX, sets status to
           `LATEX_ACCEPTED`, saves it (triggering embedding generation), and returns True.

        f. If rejected, stores the feedback, updates status, saves, and continues the loop.

    4. If the loop finishes without acceptance, sets status to `LATEX_REJECTED_FINAL`
       and returns False.

    Handles errors during the process, updating the item status to `ERROR`.

    Args:
        unique_name (str): The unique name of the KBItem to process.
        client (GeminiClient): An initialized LLM client instance.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.

    Returns:
        bool: True if LaTeX was successfully generated and accepted within the
        allowed review cycles, False otherwise (including errors or final rejection).
    """
    # Check if necessary storage functions are available
    if not all([KBItem, ItemStatus, ItemType, get_kb_item_by_name, save_kb_item]):
        logger.critical(
            "KB Storage module components not loaded correctly. Cannot process LaTeX."
        )
        return False
    if not client:
        logger.error(
            f"GeminiClient not provided for LaTeX processing of {unique_name}."
        )
        return False

    effective_db_path = db_path or DEFAULT_DB_PATH
    item = get_kb_item_by_name(unique_name, effective_db_path)

    if not item:
        logger.error(f"LaTeX Processing: Item not found: {unique_name}")
        return False

    # --- Status Checks ---
    # Define statuses that trigger processing
    trigger_statuses = {
        ItemStatus.PENDING,
        ItemStatus.PENDING_LATEX,
        ItemStatus.LATEX_REJECTED_FINAL,
        ItemStatus.PENDING_LATEX_REVIEW,
    }  # Added review status
    # Define statuses that mean LaTeX is already acceptable
    already_accepted_statuses = {
        ItemStatus.LATEX_ACCEPTED,
        ItemStatus.PROVEN,
        ItemStatus.DEFINITION_ADDED,
        ItemStatus.AXIOM_ACCEPTED,
    }

    if item.status in already_accepted_statuses:
        logger.info(
            f"LaTeX Processing: Item {unique_name} status ({item.status.name}) indicates LaTeX is already acceptable. Skipping."
        )
        return True

    # Ensure item is in a state where LaTeX generation should start/resume
    if item.status not in trigger_statuses:
        logger.warning(
            f"LaTeX Processing: Item {unique_name} not in a trigger status ({ {s.name for s in trigger_statuses} }). Current: {item.status.name}. Skipping."
        )
        return False

    original_status = item.status
    logger.info(
        f"Starting combined LaTeX processing for {unique_name} (Status: {original_status.name})"
    )

    # --- Fetch Dependencies ---
    dependency_items: List[KBItem] = []
    logger.debug(f"Fetching dependencies for {unique_name}: {item.plan_dependencies}")
    if item.plan_dependencies:
        for dep_name in item.plan_dependencies:
            dep_item = get_kb_item_by_name(dep_name, effective_db_path)
            if dep_item:
                # Check if dependency has acceptable LaTeX statement
                if (
                    dep_item.latex_statement
                    and dep_item.status in already_accepted_statuses
                ):
                    dependency_items.append(dep_item)
                    logger.debug(
                        f"  - Including valid dependency: {dep_name} ({dep_item.status.name})"
                    )
                else:
                    logger.warning(
                        f"  - Dependency '{dep_name}' for '{unique_name}' excluded: Lacks accepted LaTeX statement (Status: {dep_item.status.name if dep_item.status else 'N/A'})."
                    )
            else:
                logger.error(
                    f"  - Dependency '{dep_name}' not found in KB for target '{unique_name}'."
                )
    else:
        logger.debug(f"No plan dependencies listed for {unique_name}.")

    # --- Start Processing Cycle ---
    try:
        # Initial status update to indicate processing has started
        item.update_status(ItemStatus.LATEX_GENERATION_IN_PROGRESS)
        await save_kb_item(
            item, client=None, db_path=effective_db_path
        )  # Save status update without generating embeddings yet

        # Initialize variables for the loop
        current_statement: Optional[str] = (
            item.latex_statement
        )  # Use existing LaTeX as starting point if available
        current_proof: Optional[str] = (
            item.latex_proof
        )  # Use existing proof if available
        review_feedback: Optional[str] = (
            item.latex_review_feedback
        )  # Use previous feedback if resuming
        accepted = False
        proof_expected = item.item_type.requires_proof()

        # --- Generation and Review Loop ---
        for cycle in range(MAX_REVIEW_CYCLES):
            logger.info(
                f"Combined LaTeX Cycle {cycle + 1}/{MAX_REVIEW_CYCLES} for {unique_name}"
            )

            # --- Step (a): Generate / Refine Combined LaTeX ---
            item.update_status(
                ItemStatus.LATEX_GENERATION_IN_PROGRESS
            )  # Mark as generating
            await save_kb_item(
                item, client=None, db_path=effective_db_path
            )  # Save status

            raw_generator_response = await _call_latex_statement_and_proof_generator(
                item=item,
                dependencies=dependency_items,
                current_statement=current_statement,
                current_proof=current_proof,
                review_feedback=review_feedback,
                client=client,
            )

            item = get_kb_item_by_name(
                unique_name, effective_db_path
            )  # Refresh item state after potentially long LLM call
            if not item:
                raise Exception(f"Item {unique_name} vanished during LaTeX generation.")
            item.generation_prompt = "See _call_latex_statement_and_proof_generator"  # Placeholder, actual prompt is complex
            item.raw_ai_response = raw_generator_response  # Store raw response

            if not raw_generator_response:
                logger.warning(
                    f"Combined LaTeX generator failed or returned empty response for {unique_name} in cycle {cycle + 1}."
                )
                item.update_status(
                    ItemStatus.ERROR,
                    f"LaTeX generator failed/empty in cycle {cycle + 1}",
                )
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False  # Critical failure

            # --- Step (a.2): Parse Generator Output ---
            parsed_statement, parsed_proof = _parse_combined_latex(
                raw_generator_response, item.item_type
            )

            if (
                parsed_statement is None
            ):  # Only statement parsing failure is always critical
                logger.warning(
                    f"Failed to parse statement from generator output for {unique_name} in cycle {cycle + 1}. Raw: {raw_generator_response[:500]}"
                )
                item.update_status(
                    ItemStatus.ERROR,
                    f"LaTeX generator output parsing failed (statement) in cycle {cycle + 1}",
                )
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False  # Exit on critical parsing failure

            if proof_expected and parsed_proof is None:
                # If proof was expected but not parsed, log it but maybe let reviewer catch it?
                # Or fail definitively? Let's log warning and proceed to review. Reviewer should reject.
                logger.warning(
                    f"Expected proof for {unique_name} (type: {item.item_type.name}) but failed to parse from generator output in cycle {cycle + 1}."
                )
                # Proceed with parsed_statement and parsed_proof=None

            current_statement = parsed_statement  # Update working versions for review
            current_proof = parsed_proof

            # --- Step (b): Review Combined LaTeX ---
            item.update_status(ItemStatus.LATEX_REVIEW_IN_PROGRESS)  # Mark as reviewing
            # Save state before review (includes raw generator response)
            await save_kb_item(item, client=None, db_path=effective_db_path)

            raw_reviewer_response = await _call_latex_statement_and_proof_reviewer(
                item_type=item.item_type,
                latex_statement=current_statement,
                latex_proof=current_proof,  # Pass None if not expected/parsed
                unique_name=unique_name,
                nl_description=item.description_nl,
                dependencies=dependency_items,
                client=client,
            )

            item = get_kb_item_by_name(
                unique_name, effective_db_path
            )  # Refresh item state
            if not item:
                raise Exception(f"Item {unique_name} vanished during LaTeX review.")
            # Could store reviewer prompt/response too if needed
            item.raw_ai_response = (
                raw_reviewer_response  # Overwrite raw response with reviewer's output
            )

            if not raw_reviewer_response:
                logger.warning(
                    f"LaTeX reviewer failed or returned empty response for {unique_name} in cycle {cycle + 1}."
                )
                item.update_status(
                    ItemStatus.ERROR,
                    f"LaTeX reviewer failed/empty in cycle {cycle + 1}",
                )
                await save_kb_item(item, client=None, db_path=effective_db_path)
                return False  # Critical failure

            # --- Step (c): Parse and Process Review ---
            judgment, feedback, error_loc = _parse_combined_review(
                raw_reviewer_response, proof_expected
            )

            if judgment == "Accepted":
                logger.info(
                    f"Combined LaTeX for {unique_name} ACCEPTED by reviewer in cycle {cycle + 1}."
                )
                accepted = True
                item.latex_statement = current_statement  # Store accepted content
                item.latex_proof = current_proof  # Store accepted proof (or None)
                item.update_status(
                    ItemStatus.LATEX_ACCEPTED, review_feedback=None
                )  # Clear feedback on acceptance
                # Save final accepted state, potentially triggering embedding generation in save_kb_item
                await save_kb_item(item, client=client, db_path=effective_db_path)
                break  # Exit review loop successfully
            else:
                # Rejected case
                logger.warning(
                    f"Combined LaTeX for {unique_name} REJECTED by reviewer in cycle {cycle + 1}. Location: {error_loc or 'N/A'}. Feedback: {feedback[:200]}..."
                )
                review_feedback = (
                    feedback  # Store feedback for the next generation attempt
                )
                item.latex_review_feedback = (
                    review_feedback  # Save feedback to item DB field
                )
                item.update_status(
                    ItemStatus.PENDING_LATEX_REVIEW
                )  # Set status indicating rejection needs rework
                # Save feedback and status update (no embedding generation needed here)
                await save_kb_item(item, client=None, db_path=effective_db_path)
                # Continue to the next cycle

        # --- After Loop ---
        if not accepted:
            logger.error(
                f"Combined LaTeX for {unique_name} was REJECTED after {MAX_REVIEW_CYCLES} cycles."
            )
            # Get final item state before updating status
            item = get_kb_item_by_name(unique_name, effective_db_path)
            if item:
                # Set final rejected status, keeping the last feedback
                item.update_status(
                    ItemStatus.LATEX_REJECTED_FINAL, review_feedback=review_feedback
                )
                await save_kb_item(item, client=None, db_path=effective_db_path)
            return False  # Return False indicating failure after max cycles
        else:
            return True  # Return True as it was accepted within the loop

    except Exception as e:
        # Catch any unexpected errors during the process
        logger.exception(
            f"Unhandled exception during combined LaTeX processing for {unique_name}: {e}"
        )
        try:
            # Attempt to set the item status to ERROR
            item_err = get_kb_item_by_name(unique_name, effective_db_path)
            if (
                item_err
                and item_err.status not in already_accepted_statuses
                and item_err.status != ItemStatus.LATEX_REJECTED_FINAL
            ):
                item_err.update_status(
                    ItemStatus.ERROR,
                    f"Unhandled LaTeX processor exception: {str(e)[:500]}",
                )
                await save_kb_item(item_err, client=None, db_path=effective_db_path)
        except Exception as final_err:
            # Log if even saving the error state fails
            logger.error(
                f"Failed to save final error state for {unique_name} after exception: {final_err}"
            )
        return False


# --- Optional: Batch Processing Function ---
async def process_pending_latex_items(
    client: GeminiClient,
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
    process_statuses: Optional[List[ItemStatus]] = None,
):
    """Processes multiple items requiring LaTeX generation and review.

    Queries the database for items in specified statuses (defaulting to
    PENDING_LATEX, PENDING, LATEX_REJECTED_FINAL), then iterates through them,
    calling `generate_and_review_latex` for each one up to an optional limit.
    Logs summary statistics upon completion.

    Args:
        client (GeminiClient): An initialized LLM client instance passed to the
            processing function for each item.
        db_path (Optional[str]): Path to the database file. If None, uses
            `DEFAULT_DB_PATH`.
        limit (Optional[int]): The maximum number of items to process in this batch.
            If None, processes all found items.
        process_statuses (Optional[List[ItemStatus]]): A list of `ItemStatus` enums
            indicating which items should be processed. If None, defaults to
            `[PENDING_LATEX, PENDING, LATEX_REJECTED_FINAL]`.
    """
    # Check if necessary components are available
    if not all([ItemStatus, get_items_by_status, get_kb_item_by_name, save_kb_item]):
        logger.critical(
            "KB Storage or required components not loaded correctly. Cannot batch process LaTeX items."
        )
        return

    effective_db_path = db_path or DEFAULT_DB_PATH
    # Default statuses to process if none provided
    if process_statuses is None:
        process_statuses = [
            ItemStatus.PENDING_LATEX,
            ItemStatus.PENDING,
            ItemStatus.LATEX_REJECTED_FINAL,
            ItemStatus.PENDING_LATEX_REVIEW,
        ]  # Added review

    processed_count = 0
    success_count = 0
    fail_count = 0
    items_to_process = []

    logger.info(
        f"Starting LaTeX batch processing. Querying for items with statuses: {[s.name for s in process_statuses]}"
    )
    # Collect items to process from all specified statuses
    for status in process_statuses:
        if limit is not None and len(items_to_process) >= limit:
            break  # Stop querying if limit already reached
        try:
            # Use the generator function to get items for the current status
            items_gen = get_items_by_status(status, effective_db_path)
            count_for_status = 0
            for item in items_gen:
                if limit is not None and len(items_to_process) >= limit:
                    break  # Stop adding if limit reached
                items_to_process.append(item)
                count_for_status += 1
            logger.debug(f"Found {count_for_status} items with status {status.name}")
        except Exception as e:
            logger.error(f"Failed to retrieve items with status {status.name}: {e}")
            # Continue to next status even if one fails

    if not items_to_process:
        logger.info(
            "No items found requiring LaTeX processing in the specified statuses."
        )
        return

    logger.info(
        f"Found {len(items_to_process)} items matching criteria for LaTeX processing."
    )

    # Process collected items
    for item in items_to_process:
        # Check item status again before processing, in case it changed concurrently
        try:
            current_item_state = get_kb_item_by_name(
                item.unique_name, effective_db_path
            )
            # Ensure item exists and is still in one of the target processing statuses
            if (
                not current_item_state
                or current_item_state.status not in process_statuses
            ):
                logger.info(
                    f"Skipping {item.unique_name} as its status changed ({current_item_state.status.name if current_item_state else 'deleted'}) or item disappeared before processing."
                )
                continue
        except Exception as fetch_err:
            logger.error(
                f"Error fetching current state for {item.unique_name} before processing: {fetch_err}. Skipping."
            )
            continue

        logger.info(
            f"--- Processing LaTeX for: {item.unique_name} (ID: {item.id}, Status: {current_item_state.status.name}) ---"
        )
        processed_count += 1
        try:
            # Call the main combined processing function for the item
            success = await generate_and_review_latex(
                item.unique_name, client, effective_db_path
            )
            if success:
                success_count += 1
                logger.info(f"Successfully processed LaTeX for {item.unique_name}.")
            else:
                fail_count += 1
                logger.warning(
                    f"Failed to process LaTeX for {item.unique_name} (check previous logs for details)."
                )
        except Exception as e:
            # Catch unexpected errors from generate_and_review_latex itself
            logger.error(
                f"Unhandled error processing item {item.unique_name} in batch: {e}"
            )
            fail_count += 1
            # Attempt to mark the item as ERROR if processing failed critically
            try:
                err_item = get_kb_item_by_name(item.unique_name, effective_db_path)
                # Avoid overwriting already accepted/rejected statuses with ERROR
                if err_item and err_item.status not in {
                    ItemStatus.PROVEN,
                    ItemStatus.LATEX_ACCEPTED,
                    ItemStatus.LATEX_REJECTED_FINAL,
                }:
                    err_item.update_status(
                        ItemStatus.ERROR, f"Batch processing error: {str(e)[:500]}"
                    )
                    await save_kb_item(err_item, client=None, db_path=effective_db_path)
            except Exception as save_err:
                logger.error(
                    f"Failed to save error status for {item.unique_name} during batch exception handling: {save_err}"
                )

        logger.info(f"--- Finished processing {item.unique_name} ---")
        # Optional delay between items if needed
        # await asyncio.sleep(0.5)

    logger.info(
        f"LaTeX Batch Processing Complete. Total Attempted: {processed_count}, Succeeded: {success_count}, Failed: {fail_count}"
    )
