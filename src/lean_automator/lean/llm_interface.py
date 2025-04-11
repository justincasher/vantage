# src/lean_automator/lean/llm_interface.py

"""Handles LLM interactions for Lean code generation.

This module provides functions to:

- Format prompts for Lean statement and proof generation.
- Build context strings from dependencies.
- Call the designated LLM client (e.g., GeminiClient).
- Parse Lean code snippets from the LLM's responses.
"""

import logging
import re
import warnings
from typing import List, Optional

# --- Imports from other project modules ---
try:
    # Assumes KBItem, ItemStatus, ItemType might be needed for context building
    from lean_automator.kb.storage import ItemStatus, ItemType, KBItem
except ImportError:
    warnings.warn(
        "llm_interface: Could not import KB types from lean_automator.kb.storage. "
        "Dependency context building might fail.",
        ImportWarning,
    )
    # Define dummy types to allow loading
    KBItem = type("DummyKBItem", (object,), {})
    ItemStatus = type("DummyItemStatus", (object,), {})  # type: ignore
    ItemType = type("DummyItemType", (object,), {})  # type: ignore

try:
    # Assumes GeminiClient is the expected LLM client type hint
    from lean_automator.llm.caller import GeminiClient
except ImportError:
    warnings.warn(
        "llm_interface: Could not import GeminiClient from lean_automator.llm.caller. "
        "LLM calls will likely fail.",
        ImportWarning,
    )
    # Define dummy type
    GeminiClient = type("DummyGeminiClient", (object,), {})  # type: ignore

try:
    # Import the prompt templates from the sibling module
    from .prompts import LEAN_PROOF_GENERATOR_PROMPT, LEAN_STATEMENT_GENERATOR_PROMPT
except ImportError as e:
    # Log critical error and re-raise, as these are essential.
    logging.getLogger(__name__).critical(
        "FATAL: Failed to import required prompt templates from "
        "lean_automator.lean.prompts. "
        f"Original error: {e}"
    )
    raise ImportError(
        "Essential prompt templates could not be loaded from "
        "lean_automator.lean.prompts."
    ) from e


# --- Logging ---
logger = logging.getLogger(__name__)


# --- Parsing Functions ---


def extract_lean_header(text: Optional[str]) -> Optional[str]:
    """Extracts Lean header (signature ending in ':= sorry') from LLM output.

    Tries to parse the header using custom markers (`--- BEGIN/END LEAN HEADER ---`)
    first, then falls back to parsing a strict markdown code block (` ```lean ... ``` `)
    if markers are not found. Ensures the extracted header ends with `:= sorry`,
    attempting to append it if missing.

    Args:
        text: The raw text output from the Lean statement generator LLM.

    Returns:
        The extracted and cleaned Lean header string ending in `:= sorry`,
        or None if parsing fails or the input text is empty.
    """
    if not text:
        return None
    header = None
    stripped_text = text.strip()
    # Try custom markers first
    custom_marker_regex = (
        r"---\s*BEGIN LEAN HEADER\s*---\s*(.*?)\s*---\s*END LEAN HEADER\s*---"
    )
    match_custom = re.search(
        custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE
    )
    if match_custom:
        header = match_custom.group(1).strip()
        logger.debug("Extracted header using custom markers.")
    else:
        logger.warning(
            "Could not find '--- BEGIN/END LEAN HEADER ---' markers. "
            "Trying strict markdown block..."
        )
        # Fallback: Try strict markdown block encompassing the whole response
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            header = match_md.group(1).strip()
            logger.debug("Extracted header using strict markdown block.")
        else:
            logger.warning(
                "Could not find strict markdown block ('```lean...```') "
                "encompassing the whole response."
            )

    # Post-process: Ensure it ends with ':= sorry'
    if header:
        stripped_header = header.rstrip()
        # Remove trailing comments before checking/appending ':= sorry'
        header_no_comments = re.sub(
            r"\s*--.*$", "", stripped_header, flags=re.MULTILINE
        ).rstrip()

        if not header_no_comments.endswith(":= sorry"):
            logger.warning(
                "Parsed Lean header does not end with ':= sorry'. Attempting to append."
            )
            # Attempt to fix common near misses before just appending
            if header_no_comments.endswith(" :="):
                header = header_no_comments + " sorry"
            elif header_no_comments.endswith(" :"):
                header = header_no_comments + "= sorry"
            else:  # Append directly if no obvious near miss
                header = header_no_comments + " := sorry"
        else:
            # If it already ends correctly (after stripping comments),
            # use the original (potentially with comments)
            header = stripped_header

        logger.debug(f"Final extracted/corrected header: '{header}'")
        return header
    else:
        # Log error if extraction failed completely
        logger.error(
            "Failed to extract Lean header using any method. Raw text "
            f"received: {repr(text)[:500]}..."
        )
        return None


def extract_lean_code(text: Optional[str]) -> Optional[str]:
    """Extracts a block of Lean code from LLM output.

    Tries to parse the code using custom markers (`--- BEGIN/END LEAN CODE ---`)
    first, then falls back to parsing a strict markdown code block (` ```lean ... ``` `)
    if markers are not found.

    Args:
        text: The raw text output from the Lean proof generator LLM.

    Returns:
        The extracted Lean code string (including statement and proof),
        or None if parsing fails or the input text is empty.
    """
    if not text:
        return None
    code = None
    stripped_text = text.strip()
    # Try custom markers first
    custom_marker_regex = (
        r"---\s*BEGIN LEAN CODE\s*---\s*(.*?)\s*---\s*END LEAN CODE\s*---"
    )
    match_custom = re.search(
        custom_marker_regex, stripped_text, re.DOTALL | re.IGNORECASE
    )
    if match_custom:
        code = match_custom.group(1).strip()
        logger.debug("Extracted code using custom markers.")
        return code
    else:
        logger.warning(
            "Could not find '--- BEGIN/END LEAN CODE ---' markers. "
            "Trying strict markdown block..."
        )
        # Fallback: Try strict markdown block encompassing the whole response
        markdown_regex_strict = r"^```lean\s*(.*?)\s*```$"
        match_md = re.match(markdown_regex_strict, stripped_text, re.DOTALL)
        if match_md:
            code = match_md.group(1).strip()
            logger.debug("Extracted code using strict markdown block.")
            return code
        else:
            logger.warning(
                "Could not find strict markdown block ('```lean...```') "
                "encompassing the whole response."
            )

    # Log error if extraction failed completely
    logger.error(
        "Failed to extract Lean code using any method. Raw text received: "
        f"{repr(text)[:500]}..."
    )
    return None


# --- Context Building Functions ---


def build_lean_dependency_context_for_statement(dependencies: List[KBItem]) -> str:
    """Builds simple context string: dependency names and types.

    Used for the statement generation prompt to give the LLM awareness of available
    item names and types without including their full code.

    Args:
        dependencies: A list of dependency KBItems.

    Returns:
        A formatted string listing dependency names and types, or "(None)"
        if the list is empty.
    """
    if not dependencies:
        return "(None)"
    # Check if real KBItem type is available
    if isinstance(KBItem, type("DummyKBItem", (object,), {})):
        logger.warning("Cannot build dependency context: KBItem type not loaded.")
        return "(Error: KBItem type not available)"

    ctx = ""
    # Get attributes safely
    for dep in dependencies:
        # Basic type check to ensure we have something resembling a KBItem
        if not hasattr(dep, "unique_name") or not hasattr(dep, "item_type"):
            logger.warning(f"Skipping invalid dependency object in context: {dep}")
            continue
        dep_name = getattr(dep, "unique_name", "UNKNOWN_DEP")
        dep_item_type = getattr(dep, "item_type", None)
        dep_type_name = (
            getattr(dep_item_type, "name", "UNKNOWN_TYPE")
            if dep_item_type
            else "UNKNOWN_TYPE"
        )
        ctx += f"- {dep_name} ({dep_type_name})\n"
    return ctx.strip()


def build_lean_dependency_context_for_proof(dependencies: List[KBItem]) -> str:
    """Builds context string with full Lean code of PROVEN dependencies.

    Filters the provided list of dependencies to include only those that are
    considered "proven" (status PROVEN, AXIOM_ACCEPTED, DEFINITION_ADDED) and
    have Lean code. Formats the output with markers for clarity in the LLM prompt.

    Args:
        dependencies: A list of potential dependency KBItems.

    Returns:
        A formatted string containing the Lean code of valid, proven
        dependencies, or a default message if none are found.
    """
    # Check if real KB types are available
    if isinstance(KBItem, type("DummyKBItem", (object,), {})) or isinstance(
        ItemStatus, type("DummyItemStatus", (object,), {})
    ):
        logger.warning(
            "Cannot build dependency context: KBItem/ItemStatus types not loaded."
        )
        return "-- (Error: KBItem/ItemStatus types not available) --"

    dependency_context = ""
    # Define statuses indicating a dependency is ready to be used in a proof
    # Ensure ItemStatus was loaded correctly before using its members
    try:
        valid_statuses = {
            ItemStatus.PROVEN,
            ItemStatus.AXIOM_ACCEPTED,
            ItemStatus.DEFINITION_ADDED,
        }
    except AttributeError:
        logger.error(
            "Cannot build dependency context: ItemStatus members not accessible."
        )
        return "-- (Error: ItemStatus members not accessible) --"

    # Filter dependencies: must exist, have lean_code, and have a valid status
    proven_deps = [
        d
        for d in dependencies
        if d  # Check if dependency object itself exists
        and hasattr(d, "lean_code")
        and getattr(d, "lean_code", None)  # Has code
        and hasattr(d, "status")
        and getattr(d, "status", None) in valid_statuses  # Has valid status
        and hasattr(d, "unique_name")  # Has a name
        and hasattr(d, "item_type")  # Has a type attribute
    ]

    if proven_deps:
        for dep in proven_deps:
            # Safely get attributes (existence checked in list comprehension)
            dep_name = getattr(dep, "unique_name")
            dep_item_type = getattr(dep, "item_type", None)
            dep_type_name = (
                getattr(dep_item_type, "name", "UNKNOWN_TYPE")
                if dep_item_type
                else "UNKNOWN_TYPE"
            )
            dep_code = getattr(dep, "lean_code")
            # Format clearly for the LLM prompt
            dependency_context += f"-- Dependency: {dep_name} ({dep_type_name})\n"
            dependency_context += f"-- BEGIN {dep_name} LEAN --\n"
            dependency_context += f"{dep_code.strip()}\n"  # Use the actual proven code
            dependency_context += f"-- END {dep_name} LEAN --\n\n"
    else:
        # Provide a clear message if no suitable dependencies were found
        dependency_context = (
            "-- (No specific proven dependencies provided from KB. "
            "Rely on Lean prelude.) --\n"
        )

    return dependency_context.strip()


# --- LLM Calling Functions ---


async def call_lean_statement_generator(
    item: KBItem,
    dependencies: List[KBItem],
    statement_error_feedback: Optional[str],
    client: GeminiClient,
) -> Optional[str]:
    """Calls the LLM to generate a Lean statement signature (`... := sorry`).

    Formats the prompt using `LEAN_STATEMENT_GENERATOR_PROMPT`, including item
    details (LaTeX statement), dependency context (names/types), and optional
    feedback from previous failed attempts.

    Args:
        item: The KBItem for which to generate the statement shell.
        dependencies: List of dependency items for context.
        statement_error_feedback: Feedback from a previous failed
            statement generation attempt, if applicable.
        client: The initialized LLM client instance.

    Returns:
        The raw text response from the LLM, or None if the client
        is unavailable, prompt formatting fails, or the API call errors.

    Raises:
        ValueError: If the `client` is None or not a valid client instance.
        TypeError: If `item` is not a valid KBItem instance.
    """
    # Check if real types are available
    if (
        isinstance(GeminiClient, type("DummyGeminiClient", (object,), {}))
        or isinstance(KBItem, type("DummyKBItem", (object,), {}))
        or isinstance(ItemType, type("DummyItemType", (object,), {}))
    ):
        logger.error(
            "Cannot call LLM: Required types (GeminiClient, KBItem, ItemType) "
            "not loaded."
        )
        return None

    if not client or not isinstance(client, GeminiClient):
        raise ValueError(
            "GeminiClient not available or invalid for Lean statement generation."
        )
    if not item or not isinstance(item, KBItem):
        raise TypeError(
            f"Invalid KBItem provided to call_lean_statement_generator: {item}"
        )

    # Safely get attributes from item, providing defaults or raising errors
    # if critical info missing
    item_unique_name = getattr(item, "unique_name", "UNKNOWN_ITEM")
    item_type = getattr(item, "item_type", None)
    item_type_name = (
        getattr(item_type, "name", "UNKNOWN_TYPE") if item_type else "UNKNOWN_TYPE"
    )
    item_latex_statement = getattr(item, "latex_statement", None)

    if not item_latex_statement:
        logger.error(
            f"Cannot generate Lean statement for {item_unique_name}: "
            f"Missing latex_statement."
        )
        return None  # Cannot proceed without LaTeX statement

    # Build context string from dependencies
    dep_context = build_lean_dependency_context_for_statement(dependencies)

    try:
        base_prompt = LEAN_STATEMENT_GENERATOR_PROMPT
        # Conditionally format the prompt based on whether feedback is provided
        if not statement_error_feedback:
            # Remove the optional feedback section if not needed
            feedback_pattern = (
                r"\n\*\*Refinement Feedback \(If Applicable\):\*\*.*?\n"
                r"(\*\*Instructions:\*\*)"
            )
            base_prompt = re.sub(
                feedback_pattern,
                r"\n\1",
                base_prompt,
                flags=re.DOTALL | re.MULTILINE,
            )
            prompt = base_prompt.format(
                unique_name=item_unique_name,
                item_type_name=item_type_name,
                latex_statement=item_latex_statement,  # Already checked it exists
                dependency_context_for_statement=dep_context,
                # statement_error_feedback key is removed from template here
            )
        else:
            # Format with feedback included
            prompt = base_prompt.format(
                unique_name=item_unique_name,
                item_type_name=item_type_name,
                latex_statement=item_latex_statement,
                dependency_context_for_statement=dep_context,
                statement_error_feedback=statement_error_feedback,  # Provide feedback
            )
    except KeyError as e:
        logger.error(f"Lean Statement Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        # Catch other potential formatting issues
        logger.error(
            f"Unexpected error formatting Lean statement prompt for "
            f"{item_unique_name}: {e}"
        )
        return None

    # Call the LLM client
    try:
        logger.debug(f"Sending Lean statement generation prompt for {item_unique_name}")
        # Ensure the client has the expected async generate method
        if not hasattr(client, "generate") or not callable(client.generate):
            logger.error(
                f"LLM client for {item_unique_name} missing 'generate' method."
            )
            return None
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        logger.error(
            f"Error calling Lean statement generator LLM for {item_unique_name}: {e}"
        )
        return None


async def call_lean_proof_generator(
    lean_statement_shell: str,
    latex_proof: Optional[str],
    unique_name: str,
    item_type_name: str,
    dependencies: List[KBItem],
    lean_error_log: Optional[str],
    client: GeminiClient,
) -> Optional[str]:
    """Calls the LLM to generate Lean proof tactics for a statement shell.

    Formats the prompt using `LEAN_PROOF_GENERATOR_PROMPT`, providing the
    statement shell (`... := sorry`), informal LaTeX proof (as guidance),
    dependency context (full Lean code), and optional error logs from previous
    failed compilation attempts.

    Args:
        lean_statement_shell: The Lean statement signature ending in `:= sorry`.
        latex_proof: The informal LaTeX proof text (guidance only).
        unique_name: The unique name of the KBItem (for logging/context).
        item_type_name: The type name of the KBItem (for context).
        dependencies: List of proven dependency items (for context).
        lean_error_log: Error output from the previous failed
            Lean compilation attempt, if applicable.
        client: The initialized LLM client instance.

    Returns:
        The raw text response from the LLM, expected to contain
        the complete Lean code (statement + proof tactics), or None if the client
        is unavailable, prompt formatting fails, the API call errors, or input
        validation fails.

    Raises:
        ValueError: If `client` is None or not a valid client instance.
    """
    # Check if real types are available
    if isinstance(GeminiClient, type("DummyGeminiClient", (object,), {})) or isinstance(
        KBItem, type("DummyKBItem", (object,), {})
    ):
        logger.error(
            "Cannot call LLM: Required types (GeminiClient, KBItem) not loaded."
        )
        return None

    if not client or not isinstance(client, GeminiClient):
        raise ValueError(
            "GeminiClient not available or invalid for Lean proof generation."
        )

    # Validate the input shell
    if not lean_statement_shell or ":= sorry" not in lean_statement_shell:
        msg = (
            f"Internal Error: call_lean_proof_generator for {unique_name} "
            f"called without a valid shell ending in ':= sorry'. "
            f"Shell: {lean_statement_shell}"
        )
        logger.error(msg)
        # Returning None instead of raising here as it's likely an internal logic error
        return None

    # Build dependency context string with full Lean code
    dep_context = build_lean_dependency_context_for_proof(dependencies)

    try:
        base_prompt = LEAN_PROOF_GENERATOR_PROMPT
        # Conditionally format the prompt based on whether an error log is provided
        if not lean_error_log:
            # Remove the optional error log section if not needed
            error_log_pattern = (
                r"\n\*\*Previous Attempt Error \(If Applicable\):\*\*.*?\n"
                r"(\*\*Instructions:\*\*)"
            )
            base_prompt = re.sub(
                error_log_pattern,
                r"\n\1",
                base_prompt,
                flags=re.DOTALL | re.MULTILINE,
            )
            prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                unique_name=unique_name,  # Added for better LLM context
                item_type_name=item_type_name,  # Added for better LLM context
                # lean_error_log key removed from template here
            )
        else:
            # Format with error log included
            prompt = base_prompt.format(
                lean_statement_shell=lean_statement_shell,
                latex_proof=latex_proof or "(No informal proof provided)",
                dependency_context_for_proof=dep_context,
                lean_error_log=lean_error_log,  # Provide error log
                unique_name=unique_name,  # Added for better LLM context
                item_type_name=item_type_name,  # Added for better LLM context
            )
    except KeyError as e:
        logger.error(f"Lean Proof Gen Prompt Formatting Error: Missing key {e}")
        return None
    except Exception as e:
        # Catch other potential formatting issues
        logger.error(
            f"Unexpected error formatting Lean proof prompt for {unique_name}: {e}"
        )
        return None

    # Call the LLM client
    try:
        logger.debug(f"Sending Lean proof generation prompt for {unique_name}")
        # Ensure the client has the expected async generate method
        if not hasattr(client, "generate") or not callable(client.generate):
            logger.error(f"LLM client for {unique_name} missing 'generate' method.")
            return None
        response_text = await client.generate(prompt=prompt)
        return response_text
    except Exception as e:
        logger.error(f"Error calling Lean proof generator LLM for {unique_name}: {e}")
        return None
