# File: lean_automator/lean/processor.py

"""Orchestrates Lean code generation and verification for Knowledge Base items.

This module manages the process of generating formal Lean 4 code (both statement
signatures and proofs) for mathematical items (`KBItem`) stored in the knowledge
base. It interacts with an LLM client (`GeminiClient`) via the `llm_interface`
module to generate the code based on LaTeX statements, informal proofs, and
dependency context.

The core logic, orchestrated by `generate_and_verify_lean`, involves:
1. Checking prerequisites and dependencies (`_check_prerequisites_and_dependencies`).
2. Generating a Lean statement signature (`_generate_statement_shell`).
3. Generating and verifying Lean proof tactics (`_generate_and_verify_proof`).
4. Updating the KBItem status and content throughout the process.
5. Providing batch processing capabilities (`process_pending_lean_items`).
"""

import logging
import time
import warnings
from typing import List, Optional, Tuple

# --- Imports from other project modules ---
try:
    from lean_automator.config.loader import APP_CONFIG
except ImportError:
    warnings.warn(
        "config_loader.APP_CONFIG not found. Default settings may be used.",
        ImportWarning,
    )
    APP_CONFIG = {}  # Provide an empty dict as a fallback

# Setup logger early for critical failures during imports
logging.basicConfig(level=logging.INFO)  # Basic config if logger not set up yet
logger = logging.getLogger(__name__)

try:
    # Prompts are now external (imports not used directly, loaded by llm_interface)
    _ = None  # Placeholder to potentially load prompts.py if needed later
except ImportError as e:
    logger.critical(
        "FATAL: Failed to import required prompt templates from "
        "lean_automator.lean.prompts. "
        "The Lean processor cannot function without these prompts. "
        "Ensure 'prompts.py' exists in the same directory as 'processor.py'. "
        f"Original error: {e}"
    )
    raise ImportError(
        "Essential prompt templates could not be loaded from "
        "lean_automator.lean.prompts. Processor cannot start."
    ) from e

try:
    # LLM interaction logic is now external
    from . import llm_interface
except ImportError as e:
    logger.critical(
        "FATAL: Failed to import required LLM interface functions from "
        "lean_automator.lean.llm_interface. "
        "The Lean processor cannot function without these. "
        "Ensure 'llm_interface.py' exists in the same directory as 'processor.py'. "
        f"Original error: {e}"
    )
    raise ImportError(
        "Essential LLM interface functions could not be loaded from "
        "lean_automator.lean.llm_interface."
    ) from e

try:
    from lean_automator.kb.storage import (
        DEFAULT_DB_PATH,
        ItemStatus,
        ItemType,
        KBItem,
        get_items_by_status,
        get_kb_item_by_name,
        save_kb_item,
    )
    from lean_automator.llm.caller import GeminiClient

    # lean_interaction module now uses the persistent library strategy
    from . import interaction as lean_interaction
    from . import proof_repair as lean_proof_repair  # Import the repair module
except ImportError as e:
    warnings.warn(f"lean_processor: Required modules not found: {e}")
    # Define dummy types/functions to allow script loading without crashing
    KBItem = type("DummyKBItem", (object,), {})
    ItemStatus = type("DummyItemStatus", (object,), {})  # type: ignore
    ItemType = type("DummyItemType", (object,), {})  # type: ignore

    def _dummy_get_kb_item_by_name(name, path):
        return None

    get_kb_item_by_name = _dummy_get_kb_item_by_name  # type: ignore

    def _dummy_save_kb_item(item, client, db_path):
        return None

    save_kb_item = _dummy_save_kb_item  # type: ignore

    def _dummy_get_items_by_status(status, path):
        return iter([])

    get_items_by_status = _dummy_get_items_by_status  # type: ignore
    GeminiClient = type("DummyGeminiClient", (object,), {})  # type: ignore
    DEFAULT_DB_PATH = None
    lean_interaction = None
    lean_proof_repair = None

# --- Constants ---
DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS = 2
DEFAULT_LEAN_PROOF_MAX_ATTEMPTS = 3

# Get configuration safely from APP_CONFIG with fallbacks to defaults
lean_config = APP_CONFIG.get("lean_processing", {})

LEAN_STATEMENT_MAX_ATTEMPTS = lean_config.get(
    "statement_max_attempts", DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS
)
# Validate the loaded/default value
if not isinstance(LEAN_STATEMENT_MAX_ATTEMPTS, int) or LEAN_STATEMENT_MAX_ATTEMPTS < 1:
    logger.warning(
        f"Invalid value '{LEAN_STATEMENT_MAX_ATTEMPTS}' for "
        f"lean_processing.statement_max_attempts (must be int >= 1). "
        f"Using default {DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS}."
    )
    LEAN_STATEMENT_MAX_ATTEMPTS = DEFAULT_LEAN_STATEMENT_MAX_ATTEMPTS

LEAN_PROOF_MAX_ATTEMPTS = lean_config.get(
    "proof_max_attempts", DEFAULT_LEAN_PROOF_MAX_ATTEMPTS
)
# Validate the loaded/default value
if not isinstance(LEAN_PROOF_MAX_ATTEMPTS, int) or LEAN_PROOF_MAX_ATTEMPTS < 1:
    logger.warning(
        f"Invalid value '{LEAN_PROOF_MAX_ATTEMPTS}' for "
        f"lean_processing.proof_max_attempts (must be int >= 1). "
        f"Using default {DEFAULT_LEAN_PROOF_MAX_ATTEMPTS}."
    )
    LEAN_PROOF_MAX_ATTEMPTS = DEFAULT_LEAN_PROOF_MAX_ATTEMPTS

logger.info(f"Using LEAN_STATEMENT_MAX_ATTEMPTS = {LEAN_STATEMENT_MAX_ATTEMPTS}")
logger.info(f"Using LEAN_PROOF_MAX_ATTEMPTS = {LEAN_PROOF_MAX_ATTEMPTS}")


# --- Helper Functions for Processing Steps ---


def _check_prerequisites_and_dependencies(
    unique_name: str, db_path: str
) -> Tuple[Optional[KBItem], List[KBItem], Optional[str]]:
    """Fetches item, checks status, prerequisites, and dependency readiness.

    Args:
        unique_name: The unique name of the KBItem to process.
        db_path: Path to the knowledge base database file.

    Returns:
        A tuple containing:
        - The fetched KBItem if prerequisites are met so far, else None.
        - A list of validated dependency KBItems (empty if failed).
        - An error message string if checks fail, else None.
    """
    # Ensure necessary types/functions are loaded
    if not all([KBItem, ItemStatus, ItemType, get_kb_item_by_name]):
        return None, [], "Core KB types or functions not loaded."

    # --- Initial Item Fetch and Status Checks ---
    item = get_kb_item_by_name(unique_name, db_path)
    if not item:
        logger.error(f"Lean Proc: Item not found: {unique_name}")
        return None, [], f"Item not found: {unique_name}"

    item_status = getattr(item, "status", None)
    item_latex_statement = getattr(item, "latex_statement", None)
    item_plan_dependencies = getattr(item, "plan_dependencies", [])

    if item_status == ItemStatus.PROVEN:
        logger.info(f"Lean Proc: Item {unique_name} is already PROVEN. Skipping.")
        # Return item, but signal no further action needed maybe? Or let caller handle.
        # Let's return item and None error message, caller handles PROVEN status.
        return item, [], None

    # Define statuses that trigger Lean processing
    try:
        trigger_statuses = {
            ItemStatus.LATEX_ACCEPTED,
            ItemStatus.PENDING_LEAN,
            ItemStatus.LEAN_VALIDATION_FAILED,
        }
    except AttributeError:
        logger.critical("ItemStatus enum members missing.")
        return item, [], "ItemStatus enum members missing."

    if item_status not in trigger_statuses:
        trigger_names = {s.name for s in trigger_statuses if hasattr(s, "name")}
        status_name = getattr(item_status, "name", "None")
        msg = (
            f"Item {unique_name} not in a trigger status "
            f"({trigger_names}). Current: {status_name}. Skipping."
        )
        logger.warning(f"Lean Proc: {msg}")
        return item, [], msg  # Return item, but with error message

    # Check prerequisite: LaTeX statement must exist
    if not item_latex_statement:
        msg = f"Cannot process {unique_name}, missing required latex_statement."
        logger.error(f"Lean Proc: {msg}")
        # Caller should handle saving ERROR status if needed
        return item, [], msg

    # --- Check Dependencies ---
    dependency_items: List[KBItem] = []
    try:
        valid_dep_statuses = {
            ItemStatus.PROVEN,
            ItemStatus.AXIOM_ACCEPTED,
            ItemStatus.DEFINITION_ADDED,
        }
    except AttributeError:
        logger.error("Cannot check dependencies: ItemStatus members not accessible.")
        return item, [], "ItemStatus members not accessible for dependency check."

    logger.debug(
        f"Checking {len(item_plan_dependencies)} dependencies for {unique_name}..."
    )
    for dep_name in item_plan_dependencies:
        dep_item = get_kb_item_by_name(dep_name, db_path)
        dep_status = getattr(dep_item, "status", None)
        dep_code_exists = bool(getattr(dep_item, "lean_code", None))

        # Dependency must exist, be in a valid status, and have lean code
        if not dep_item or dep_status not in valid_dep_statuses or not dep_code_exists:
            dep_status_name = getattr(dep_status, "name", "MISSING_ITEM")
            msg = (
                f"Dependency '{dep_name}' for '{unique_name}' not ready "
                f"(Status: {dep_status_name}, Code Exists: {dep_code_exists}). "
                f"Cannot proceed."
            )
            logger.error(msg)
            # Caller handles saving ERROR status
            return item, [], msg
        dependency_items.append(dep_item)

    logger.debug(
        f"All {len(dependency_items)} dependencies for {unique_name} "
        f"confirmed available and proven."
    )
    return item, dependency_items, None  # Prerequisites met


async def _generate_statement_shell(
    item: KBItem, dependencies: List[KBItem], client: GeminiClient, db_path: str
) -> Optional[str]:
    """Generates the Lean statement shell (`... := sorry`) using the LLM.

    Handles retries, status updates, and saving the item state during generation.

    Args:
        item: The KBItem to process (will be modified and saved).
        dependencies: List of dependency items for context.
        client: The initialized LLM client instance.
        db_path: Path to the knowledge base database file.

    Returns:
        The generated Lean statement shell string on success, None on failure.
    """
    logger.info(f"Starting Lean statement generation phase for {item.unique_name}")
    statement_error_feedback = None
    statement_accepted = False
    generated_shell = None

    # Ensure necessary functions/types are available
    if not all([save_kb_item, llm_interface]):
        logger.error(
            "Cannot generate statement shell: Missing save_kb_item or llm_interface."
        )
        return None

    for attempt in range(LEAN_STATEMENT_MAX_ATTEMPTS):
        logger.info(
            f"Lean Statement Generation Attempt {attempt + 1}/"
            f"{LEAN_STATEMENT_MAX_ATTEMPTS} for {item.unique_name}"
        )
        # Update status to show progress
        if hasattr(item, "update_status") and hasattr(
            ItemStatus, "LEAN_GENERATION_IN_PROGRESS"
        ):
            item.update_status(
                ItemStatus.LEAN_GENERATION_IN_PROGRESS,
                f"Statement attempt {attempt + 1}",
            )
            # Save status only - avoid overwriting potentially useful logs
            # from previous full run
            await save_kb_item(item, client=None, db_path=db_path)

        # Call LLM for statement generation via llm_interface
        raw_response = await llm_interface.call_lean_statement_generator(
            item, dependencies, statement_error_feedback, client
        )
        # Parse response via llm_interface
        parsed_header = llm_interface.extract_lean_header(raw_response)

        # --- Re-fetch item state after LLM call and save ---
        # This is crucial if save_kb_item doesn't return the updated object
        # or if external modifications could occur.
        refreshed_item = get_kb_item_by_name(item.unique_name, db_path)
        if not refreshed_item:
            logger.critical(
                f"Item {item.unique_name} vanished during statement generation!"
            )
            # This is a critical error, potentially raise exception
            return None
        item = refreshed_item  # Use the latest state
        # ----------------------------------------------------

        if not parsed_header:
            logger.warning(
                f"Failed to parse Lean header on attempt {attempt + 1} for "
                f"{item.unique_name}."
            )
            statement_error_feedback = (
                f"LLM output did not contain a valid Lean header "
                f"(attempt {attempt + 1}). Raw response: "
                f"{repr(raw_response[:500])}"
            )
            # Save raw response and error feedback
            if hasattr(item, "raw_ai_response"):
                item.raw_ai_response = raw_response
            if hasattr(item, "lean_error_log"):
                item.lean_error_log = statement_error_feedback
            if hasattr(item, "generation_prompt"):
                item.generation_prompt = "Statement Gen Prompt (see logs/code)"
            await save_kb_item(item, client=None, db_path=db_path)  # Save full state
            continue  # Try next attempt
        else:
            # Successfully parsed header
            logger.info(
                f"Lean statement shell generated successfully for {item.unique_name} "
                f"on attempt {attempt + 1}."
            )
            generated_shell = parsed_header
            # Update item with the generated shell
            if hasattr(item, "lean_code"):
                item.lean_code = generated_shell
            if hasattr(item, "generation_prompt"):
                item.generation_prompt = "Statement Gen Prompt (see logs/code)"
            if hasattr(item, "raw_ai_response"):
                item.raw_ai_response = raw_response
            if hasattr(item, "lean_error_log"):
                item.lean_error_log = None  # Clear error log
            # Status remains LEAN_GENERATION_IN_PROGRESS until proof phase
            # or marked PROVEN
            await save_kb_item(
                item, client=None, db_path=db_path
            )  # Save successful result
            statement_accepted = True
            break  # Exit loop on success

    # Check if statement generation ultimately failed
    if not statement_accepted:
        logger.error(
            f"Failed to generate valid Lean statement shell for {item.unique_name} "
            f"after {LEAN_STATEMENT_MAX_ATTEMPTS} attempts."
        )
        # Final status update to ERROR (item state already saved with last error
        # in loop)
        if hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
            item.update_status(
                ItemStatus.ERROR,
                f"Failed Lean statement generation after "
                f"{LEAN_STATEMENT_MAX_ATTEMPTS} attempts.",
            )
            await save_kb_item(item, client=None, db_path=db_path)
        return None

    return generated_shell


async def _generate_and_verify_proof(
    item: KBItem,
    statement_shell: str,
    dependencies: List[KBItem],
    client: GeminiClient,
    db_path: str,
    lake_executable_path: str,
    timeout_seconds: int,
) -> bool:
    """Generates and verifies the Lean proof using LLM and lean_interaction.

    Handles retries, status updates, saving item state, and optional proof repair.

    Args:
        item: The KBItem to process (will be modified and saved).
        statement_shell: The valid Lean statement shell (`... := sorry`).
        dependencies: List of dependency items for context.
        client: The initialized LLM client instance.
        db_path: Path to the knowledge base database file.
        lake_executable_path: Path to the `lake` executable.
        timeout_seconds: Timeout for `lake build` commands.

    Returns:
        True if verification succeeds within the attempts, False otherwise.
    """
    logger.info(
        f"Starting Lean proof generation and verification phase for {item.unique_name}"
    )
    lean_verification_success = False

    # Ensure necessary functions/types are available
    if not all([save_kb_item, llm_interface, lean_interaction, get_kb_item_by_name]):
        logger.error("Cannot generate/verify proof: Missing core functions or modules.")
        return False
    # Check optional module
    repair_available = lean_proof_repair and hasattr(
        lean_proof_repair, "attempt_proof_repair"
    )

    # Clear previous error log if starting fresh proof attempts
    # (only if status wasn't already failure)
    current_status = getattr(item, "status", None)
    needs_log_clear = (
        hasattr(ItemStatus, "LEAN_VALIDATION_FAILED")
        and current_status != ItemStatus.LEAN_VALIDATION_FAILED
    )
    if needs_log_clear:
        if (
            hasattr(item, "lean_error_log")
            and getattr(item, "lean_error_log", None) is not None
        ):
            status_name = getattr(current_status, "name", "None")
            logger.debug(
                f"Clearing previous lean_error_log for {item.unique_name} "
                f"(Status: {status_name})."
            )
            item.lean_error_log = None
            await save_kb_item(item, client=None, db_path=db_path)

    # --- Proof Attempt Loop ---
    for attempt in range(LEAN_PROOF_MAX_ATTEMPTS):
        logger.info(
            f"Lean Proof Generation Attempt {attempt + 1}/"
            f"{LEAN_PROOF_MAX_ATTEMPTS} for {item.unique_name}"
        )
        # Fetch the latest item state before each attempt
        current_item_state = get_kb_item_by_name(item.unique_name, db_path)
        if not current_item_state:
            logger.critical(f"Item {item.unique_name} vanished mid-proof attempts!")
            return False  # Treat as failure if item disappears
        item = current_item_state  # Use latest state

        # Set status to indicate generation is starting for this attempt
        if hasattr(item, "update_status") and hasattr(
            ItemStatus, "LEAN_GENERATION_IN_PROGRESS"
        ):
            item.update_status(
                ItemStatus.LEAN_GENERATION_IN_PROGRESS, f"Proof attempt {attempt + 1}"
            )
            await save_kb_item(item, client=None, db_path=db_path)

        # Safely get attributes needed for the LLM call
        current_latex_proof = getattr(item, "latex_proof", None)
        current_item_type = getattr(item, "item_type", None)
        current_item_type_name = (
            getattr(current_item_type, "name", "UNKNOWN_TYPE")
            if current_item_type
            else "UNKNOWN_TYPE"
        )
        current_lean_error_log = getattr(
            item, "lean_error_log", None
        )  # Use error from previous failed attempt

        # --- Call LLM for Proof Generation via llm_interface ---
        raw_llm_response = await llm_interface.call_lean_proof_generator(
            lean_statement_shell=statement_shell,
            latex_proof=current_latex_proof,
            unique_name=item.unique_name,
            item_type_name=current_item_type_name,
            dependencies=dependencies,
            lean_error_log=current_lean_error_log,
            client=client,
        )

        # Re-fetch item state after LLM call before saving LLM output
        item_after_gen = get_kb_item_by_name(item.unique_name, db_path)
        if not item_after_gen:
            logger.critical(
                f"Item {item.unique_name} vanished after proof generation call!"
            )
            return False
        item = item_after_gen  # Use latest state

        if not raw_llm_response:
            logger.warning(
                f"Lean proof generator returned no response on attempt "
                f"{attempt + 1} for {item.unique_name}"
            )
            # Save this state before potentially finishing the loop
            if hasattr(item, "raw_ai_response"):
                item.raw_ai_response = None
            if hasattr(item, "lean_error_log"):
                item.lean_error_log = "LLM provided no output."
            await save_kb_item(item, client=None, db_path=db_path)

            if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS:
                if hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
                    item.update_status(
                        ItemStatus.ERROR,
                        f"LLM failed to generate proof on final attempt {attempt + 1}",
                    )
                    await save_kb_item(item, client=None, db_path=db_path)
            continue

        # --- Parse LLM Response via llm_interface ---
        generated_lean_code_from_llm = llm_interface.extract_lean_code(raw_llm_response)
        if not generated_lean_code_from_llm:
            logger.warning(
                f"Failed to parse Lean code from proof generator on attempt "
                f"{attempt + 1} for {item.unique_name}."
            )
            error_message = (
                f"LLM output parsing failed on proof attempt {attempt + 1}. "
                f"Raw: {repr(raw_llm_response[:500])}"
            )
            if hasattr(item, "raw_ai_response"):
                item.raw_ai_response = raw_llm_response
            if hasattr(item, "lean_error_log"):
                item.lean_error_log = error_message
            if hasattr(item, "generation_prompt"):
                item.generation_prompt = "Proof Gen Prompt (see logs/code)"
            await save_kb_item(item, client=None, db_path=db_path)

            if attempt + 1 == LEAN_PROOF_MAX_ATTEMPTS:
                if hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
                    item.update_status(
                        ItemStatus.ERROR,
                        "LLM output parsing failed on final Lean proof attempt",
                    )
                    await save_kb_item(item, client=None, db_path=db_path)
            continue

        # --- Prepare and Verify Code ---
        logger.info(
            f"--- LLM Generated Lean Code (Attempt {attempt + 1} for "
            f"{item.unique_name}) ---"
        )
        logger.info(f"\n{generated_lean_code_from_llm}\n")
        logger.info("--- End LLM Generated Code ---")

        # Update item with the latest generated code and set status for verification
        if hasattr(item, "lean_code"):
            item.lean_code = generated_lean_code_from_llm
        if hasattr(item, "generation_prompt"):
            item.generation_prompt = "Proof Gen Prompt (see logs/code)"
        if hasattr(item, "raw_ai_response"):
            item.raw_ai_response = raw_llm_response
        if hasattr(item, "update_status") and hasattr(
            ItemStatus, "LEAN_VALIDATION_PENDING"
        ):
            item.update_status(ItemStatus.LEAN_VALIDATION_PENDING)
        await save_kb_item(
            item, client=None, db_path=db_path
        )  # Save before verification

        logger.info(
            f"Calling lean_interaction.check_and_compile_item for "
            f"{item.unique_name} (Proof Attempt {attempt + 1})"
        )
        try:
            # Ensure lean_interaction and its method exist before calling
            if not lean_interaction or not hasattr(
                lean_interaction, "check_and_compile_item"
            ):
                logger.error(
                    f"lean_interaction.check_and_compile_item not available. "
                    f"Cannot verify {item.unique_name}"
                )
                # Set item to error and fail this attempt
                if hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
                    item.update_status(
                        ItemStatus.ERROR, "Lean interaction module misconfiguration"
                    )
                    await save_kb_item(item, client=None, db_path=db_path)
                continue  # Or return False if this is unrecoverable

            verified, message = await lean_interaction.check_and_compile_item(
                unique_name=item.unique_name,
                db_path=db_path,
                lake_executable_path=lake_executable_path,
                timeout_seconds=timeout_seconds,
            )

            if verified:
                logger.info(
                    f"Successfully verified Lean code for: {item.unique_name} on "
                    f"attempt {attempt + 1}. Message: {message}"
                )
                lean_verification_success = True
                # Item status updated to PROVEN by check_and_compile_item
                break  # Exit proof attempt loop successfully
            else:
                # Verification failed
                logger.warning(
                    f"Verification failed for {item.unique_name} on proof attempt "
                    f"{attempt + 1}. Message: {message[:500]}..."
                )
                # Status/log updated by check_and_compile_item

                # Fetch item again to see the error log saved by check_and_compile
                item_after_fail = get_kb_item_by_name(item.unique_name, db_path)
                if not item_after_fail:
                    logger.error(
                        f"Item {item.unique_name} vanished after failed "
                        f"verification attempt!"
                    )
                    continue  # Try next attempt if possible, though this is bad

                latest_error_log = getattr(item_after_fail, "lean_error_log", None)

                if latest_error_log:
                    logger.warning(
                        f"--- Lean Error Log (Attempt {attempt + 1} for "
                        f"{item.unique_name}) ---"
                    )
                    logger.warning(f"\n{latest_error_log}\n")
                    logger.warning("--- End Lean Error Log ---")

                    # Optional: Attempt Automated Proof Repair
                    if (
                        repair_available
                        and hasattr(item_after_fail, "lean_code")
                        and item_after_fail.lean_code
                    ):
                        logger.info(
                            f"Attempting automated proof repair for {item.unique_name} "
                            f"based on error log."
                        )
                        try:
                            fix_applied, _ = lean_proof_repair.attempt_proof_repair(  # type: ignore
                                item_after_fail.lean_code, latest_error_log
                            )
                            if fix_applied:
                                logger.info(
                                    f"Automated proof repair heuristic applied for "
                                    f"{item.unique_name}."
                                )
                            else:
                                logger.debug(
                                    f"No automated fix applied by lean_proof_repair "
                                    f"for {item.unique_name}."
                                )
                        except Exception as repair_err:
                            logger.error(
                                f"Error during automated proof repair attempt for "
                                f"{item.unique_name}: {repair_err}"
                            )
                else:
                    logger.warning(
                        f"No specific error log captured in DB after failed "
                        f"verification attempt {attempt + 1} for {item.unique_name}, "
                        f"message: {message}"
                    )

                # Continue to the next LLM proof generation attempt loop iteration
                continue

        except Exception as verify_err:
            logger.exception(
                f"check_and_compile_item crashed unexpectedly for {item.unique_name} "
                f"on proof attempt {attempt + 1}: {verify_err}"
            )
            # Attempt to set item status to ERROR
            item_err_state = get_kb_item_by_name(item.unique_name, db_path)  # Re-fetch
            if (
                item_err_state
                and hasattr(item_err_state, "update_status")
                and hasattr(ItemStatus, "ERROR")
            ):
                err_log_message = (
                    f"Verification system crashed: {str(verify_err)[:500]}"
                )
                item_err_state.update_status(
                    ItemStatus.ERROR, "Lean verification system error"
                )
                if hasattr(item_err_state, "lean_error_log"):
                    item_err_state.lean_error_log = err_log_message
                if hasattr(item_err_state, "increment_failure_count") and callable(
                    getattr(item_err_state, "increment_failure_count", None)
                ):
                    item_err_state.increment_failure_count()
                await save_kb_item(item_err_state, client=None, db_path=db_path)
            continue  # Try next LLM attempt

    # --- Loop Finished ---
    # Final status should be PROVEN or LEAN_VALIDATION_FAILED/ERROR set within
    # loop/check_and_compile
    return lean_verification_success


# --- Main Orchestrator Function ---


async def generate_and_verify_lean(
    unique_name: str,
    client: GeminiClient,
    db_path: Optional[str] = None,
    lake_executable_path: str = "lake",
    timeout_seconds: int = 120,
) -> bool:
    """Orchestrates the full Lean processing pipeline for a single KBItem.

    Calls helper functions to check prerequisites, generate statement, and
    generate/verify proof as needed.

    Args:
        unique_name: The unique name of the KBItem to process.
        client: An initialized LLM client instance.
        db_path: Path to the knowledge base database file. Uses
            `DEFAULT_DB_PATH` if None.
        lake_executable_path: Path to the `lake` executable.
        timeout_seconds: Timeout for `lake build` commands.

    Returns:
        True if the item is successfully processed (ends in PROVEN status),
        False otherwise.
    """
    start_time = time.time()
    effective_db_path = db_path or DEFAULT_DB_PATH
    overall_success = False

    # --- Basic readiness checks ---
    if not client or isinstance(client, type("DummyGeminiClient", (object,), {})):
        logger.error(
            f"GeminiClient missing or invalid. Cannot process Lean for {unique_name}."
        )
        return False
    if not all(
        [
            KBItem,
            ItemStatus,
            ItemType,
            get_kb_item_by_name,
            save_kb_item,
            lean_interaction,
            lean_proof_repair,
            llm_interface,
        ]
    ):
        logger.critical(
            f"Core modules/types missing for {unique_name}. Cannot process."
        )
        return False

    try:
        # --- Step 1: Check prerequisites and dependencies ---
        item, dependency_items, error_msg = _check_prerequisites_and_dependencies(
            unique_name, effective_db_path
        )

        if error_msg:
            logger.warning(f"Prerequisite check failed for {unique_name}: {error_msg}")
            # Save ERROR status only if item exists and status indicates an error
            # occurred during processing
            if item and hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
                # Avoid overwriting PROVEN status if that was the reason
                # for the 'error' message
                if getattr(item, "status", None) != ItemStatus.PROVEN:
                    # Check if the error requires setting ERROR status
                    if (
                        "missing required latex_statement" in error_msg
                        or "not ready" in error_msg
                        or "not accessible" in error_msg
                    ):
                        item.update_status(ItemStatus.ERROR, error_msg)
                        await save_kb_item(item, client=None, db_path=effective_db_path)
            return False  # Prerequisite failed

        if not item:  # Should not happen if error_msg is None, but check defensively
            logger.error(
                f"Prerequisite check returned no item and no error for {unique_name}."
            )
            return False

        # If item is already proven, the check function handles logging,
        # we just exit successfully.
        if (
            hasattr(ItemStatus, "PROVEN")
            and getattr(item, "status", None) == ItemStatus.PROVEN
        ):
            return True

        # --- Step 2: Determine if statement generation is needed ---
        needs_statement_generation = not getattr(item, "lean_code", None) or getattr(
            item, "status", None
        ) in {ItemStatus.LATEX_ACCEPTED, ItemStatus.PENDING_LEAN}
        statement_shell = None

        if needs_statement_generation:
            statement_shell = await _generate_statement_shell(
                item, dependency_items, client, effective_db_path
            )
            if not statement_shell:
                logger.error(f"Failed to generate statement shell for {unique_name}.")
                # Status already set to ERROR inside helper function
                return False
            # Re-fetch item state after generation as it was modified
            item = get_kb_item_by_name(unique_name, effective_db_path)
            if not item:
                logger.critical(
                    f"Item {unique_name} vanished after statement generation!"
                )
                return False  # Or raise
        else:
            logger.info(
                f"Skipping Lean statement generation for {unique_name}, "
                f"using existing lean_code."
            )
            statement_shell = getattr(item, "lean_code", None)
            # Validate existing shell
            if not statement_shell or ":= sorry" not in statement_shell.strip():
                error_msg = (
                    f"Existing lean_code for {unique_name} is invalid "
                    f"(missing ':= sorry')."
                )
                logger.error(error_msg)
                if hasattr(item, "update_status") and hasattr(ItemStatus, "ERROR"):
                    item.update_status(ItemStatus.ERROR, error_msg)
                    await save_kb_item(item, client=None, db_path=effective_db_path)
                return False

        # --- Step 3: Handle non-provable items ---
        item_type = getattr(item, "item_type", None)
        proof_required = getattr(item_type, "requires_proof", lambda: True)()

        if not proof_required:
            logger.info(
                f"Lean Proc: Statement generated/present for non-provable "
                f"{unique_name}. Marking PROVEN."
            )
            if hasattr(item, "update_status") and hasattr(ItemStatus, "PROVEN"):
                # Only update if not already proven
                if getattr(item, "status", None) != ItemStatus.PROVEN:
                    item.update_status(ItemStatus.PROVEN)
                    await save_kb_item(item, client=None, db_path=effective_db_path)
                overall_success = True  # Mark as success
            else:
                logger.error(
                    f"Failed to mark non-provable item {unique_name} as PROVEN."
                )
                overall_success = False  # Should not happen if checks passed
        else:
            # --- Step 4: Generate and verify proof ---
            proof_success = await _generate_and_verify_proof(
                item,
                statement_shell,
                dependency_items,
                client,
                effective_db_path,
                lake_executable_path,
                timeout_seconds,
            )
            overall_success = proof_success

    except Exception as e:
        logger.exception(
            f"Unhandled exception during generate_and_verify_lean for "
            f"{unique_name}: {e}"
        )
        # Attempt to mark item as error if possible
        try:
            item_err = get_kb_item_by_name(unique_name, effective_db_path)
            if (
                item_err
                and hasattr(item_err, "update_status")
                and hasattr(ItemStatus, "ERROR")
            ):
                if (
                    getattr(item_err, "status", None) != ItemStatus.PROVEN
                ):  # Avoid overwriting success
                    item_err.update_status(
                        ItemStatus.ERROR, f"Unhandled exception: {str(e)[:100]}"
                    )
                    await save_kb_item(item_err, client=None, db_path=effective_db_path)
        except Exception as save_err:
            logger.error(
                f"Failed to save ERROR status after unhandled exception for "
                f"{unique_name}: {save_err}"
            )
        overall_success = False  # Ensure failure on exception

    # --- Final Logging ---
    end_time = time.time()
    duration = end_time - start_time
    if overall_success:
        logger.info(
            f"Lean processing SUCCEEDED for {unique_name} in {duration:.2f} seconds."
        )
    else:
        # Fetch final state for accurate logging
        final_item = get_kb_item_by_name(unique_name, effective_db_path)
        final_status_name = "UNKNOWN"
        if final_item:
            final_status = getattr(final_item, "status", None)
            final_status_name = getattr(final_status, "name", "UNKNOWN")
            # Check again if it somehow ended as PROVEN despite failure path
            if hasattr(ItemStatus, "PROVEN") and final_status == ItemStatus.PROVEN:
                logger.warning(
                    f"Item {unique_name} ended as PROVEN despite processing path "
                    f"indicating failure. Assuming success."
                )
                overall_success = True  # Correct the outcome based on final state
                logger.info(
                    f"Lean processing SUCCEEDED (final status) for {unique_name} "
                    f"in {duration:.2f} seconds."
                )
                return True  # Return success

        logger.error(
            f"Lean processing FAILED for {unique_name}. "
            f"Final Status: {final_status_name}. Total time: {duration:.2f} seconds."
        )

    return overall_success


# --- Batch Processing Function ---
async def process_pending_lean_items(
    client: GeminiClient,
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
    process_statuses: Optional[List[ItemStatus]] = None,
    **kwargs,  # Pass other args like lake path, timeout to generate_and_verify_lean
):
    """Processes multiple items requiring Lean code generation and verification.

    Queries the database for items in specified statuses (defaulting to
    LATEX_ACCEPTED, PENDING_LEAN, LEAN_VALIDATION_FAILED). It then iterates
    through eligible items (checking status again before processing), calls
    `generate_and_verify_lean` for each one up to an optional limit, and logs
    summary statistics.

    Args:
        client: An initialized LLM client instance.
        db_path: Path to the database file. If None, uses `DEFAULT_DB_PATH`.
        limit: Max number of items to process in this batch.
        process_statuses: List of statuses to query for processing. Defaults
            to LATEX_ACCEPTED, PENDING_LEAN, LEAN_VALIDATION_FAILED.
        **kwargs: Additional keyword arguments passed to `generate_and_verify_lean`.
    """
    # Basic dependency checks
    if not all(
        [
            KBItem,
            ItemStatus,
            ItemType,
            get_items_by_status,
            get_kb_item_by_name,
            save_kb_item,
        ]
    ):
        logger.critical(
            "Required KB modules/functions not fully loaded. "
            "Cannot batch process Lean items."
        )
        return
    if not client or isinstance(client, type("DummyGeminiClient", (object,), {})):
        logger.error("GeminiClient missing or invalid. Cannot batch process.")
        return

    effective_db_path = db_path or DEFAULT_DB_PATH
    # Determine target statuses
    try:
        default_statuses = {
            ItemStatus.PENDING_LEAN,
            ItemStatus.LATEX_ACCEPTED,
            ItemStatus.LEAN_VALIDATION_FAILED,
        }
        if process_statuses is None:
            process_statuses_set = default_statuses
        else:
            valid_statuses = {s for s in process_statuses if isinstance(s, ItemStatus)}
            if len(valid_statuses) != len(process_statuses):
                logger.warning(
                    "Invalid status types provided to process_pending_lean_items."
                )
            process_statuses_set = (
                valid_statuses if valid_statuses else default_statuses
            )
            if not valid_statuses:
                logger.warning("No valid process_statuses provided, using defaults.")
    except AttributeError:
        logger.critical(
            "Cannot determine batch processing statuses: "
            "ItemStatus members not accessible."
        )
        return

    processed_count = 0
    success_count = 0
    fail_count = 0
    items_to_process_names = []

    status_names = [getattr(s, "name", "UNKNOWN") for s in process_statuses_set]
    logger.info(
        f"Starting Lean batch processing. Querying for statuses: {status_names}."
    )

    # --- Collect eligible items ---
    for status in process_statuses_set:
        if limit is not None and len(items_to_process_names) >= limit:
            break
        try:
            if not callable(get_items_by_status):
                continue  # Skip if function not loaded

            items_gen = get_items_by_status(status, effective_db_path)
            count_for_status = 0
            for item in items_gen:
                if limit is not None and len(items_to_process_names) >= limit:
                    break
                item_unique_name = getattr(item, "unique_name", None)
                if not item_unique_name:
                    continue

                # Add item if not already collected
                if item_unique_name not in items_to_process_names:
                    # Check if non-provable item with code can be marked PROVEN directly
                    item_type = getattr(item, "item_type", None)
                    item_lean_code = getattr(item, "lean_code", None)
                    needs_proof = getattr(item_type, "requires_proof", lambda: True)()
                    current_item_status = getattr(
                        item, "status", None
                    )  # Should be same as loop 'status'

                    if (
                        not needs_proof
                        and item_lean_code
                        and current_item_status == ItemStatus.LATEX_ACCEPTED
                    ):
                        logger.info(
                            f"Pre-marking non-provable item {item_unique_name} "
                            f"with code as PROVEN."
                        )
                        if hasattr(item, "update_status") and hasattr(
                            ItemStatus, "PROVEN"
                        ):
                            item.update_status(ItemStatus.PROVEN)
                            await save_kb_item(
                                item, client=None, db_path=effective_db_path
                            )
                        continue  # Skip adding to main processing list

                    # Otherwise, add to list for processing
                    items_to_process_names.append(item_unique_name)
                    count_for_status += 1

            logger.debug(
                f"Found {count_for_status} potential items with status "
                f"{getattr(status, 'name', 'UNKNOWN')}"
            )

        except Exception as e:
            logger.error(
                f"Failed to retrieve items with status "
                f"{getattr(status, 'name', 'UNKNOWN')}: {e}"
            )

    if not items_to_process_names:
        logger.info("No eligible items found requiring Lean processing.")
        return

    logger.info(
        f"Collected {len(items_to_process_names)} unique items for Lean processing."
    )

    # --- Process collected items ---
    for unique_name in items_to_process_names:
        # Re-fetch item state before processing to double-check eligibility
        try:
            current_item_state = get_kb_item_by_name(unique_name, effective_db_path)
            if not current_item_state:
                logger.warning(
                    f"Skipping {unique_name}: Item disappeared before processing."
                )
                continue

            item_status = getattr(current_item_state, "status", None)
            # Check if status is still one of the target statuses
            if item_status not in process_statuses_set:
                status_name = getattr(item_status, "name", "None")
                logger.info(
                    f"Skipping {unique_name}: Status ({status_name}) changed, "
                    f"no longer eligible."
                )
                continue

        except Exception as fetch_err:
            logger.error(
                f"Error re-fetching state for {unique_name}: {fetch_err}. Skipping."
            )
            continue

        # Proceed with processing the eligible item
        item_id = getattr(current_item_state, "id", "N/A")
        status_name = getattr(item_status, "name", "None")
        logger.info(
            f"--- Processing Lean for: {unique_name} (ID: {item_id}, "
            f"Status: {status_name}) ---"
        )
        processed_count += 1
        try:
            # Call the main orchestrator function
            success = await generate_and_verify_lean(
                unique_name, client, effective_db_path, **kwargs
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
            # Detailed outcome logged within generate_and_verify_lean
        except Exception as e:
            # Catch unexpected errors from the orchestrator
            logger.exception(
                f"Critical error during batch processing of {unique_name}: {e}"
            )
            fail_count += 1
            # Attempt to mark item as ERROR in DB
            try:
                err_item = get_kb_item_by_name(unique_name, effective_db_path)
                if (
                    err_item
                    and hasattr(err_item, "update_status")
                    and hasattr(ItemStatus, "ERROR")
                ):
                    current_err_status = getattr(err_item, "status", None)
                    is_proven = (
                        hasattr(ItemStatus, "PROVEN")
                        and current_err_status == ItemStatus.PROVEN
                    )
                    if not is_proven:  # Don't overwrite success
                        err_item.update_status(
                            ItemStatus.ERROR,
                            f"Batch processing crashed: {str(e)[:100]}",
                        )
                        await save_kb_item(
                            err_item, client=None, db_path=effective_db_path
                        )
            except Exception as save_err:
                logger.error(
                    f"Failed to save ERROR status for {unique_name} after batch crash: "
                    f"{save_err}"
                )

        logger.info(f"--- Finished processing Lean for {unique_name} ---")

    logger.info(
        f"Lean Batch Processing Complete. Total Processed: {processed_count}, "
        f"Succeeded: {success_count}, Failed: {fail_count}"
    )
