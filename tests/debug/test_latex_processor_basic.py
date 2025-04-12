# File: test_latex_processor_basic.py

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# --- Add project root to path to allow imports ---
# Adjust this path if your script is located elsewhere relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(
    script_dir
)  # Assumes script is in a 'scripts' or 'tests' dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Module Imports (after path setup and dotenv) ---
try:
    from lean_automator.kb import storage as kb_storage
    from lean_automator.kb.storage import ItemStatus, ItemType
    from lean_automator.latex import processor as latex_processor
    from lean_automator.llm import caller as llm_call
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Ensure the script is run from the project root or adjust PYTHONPATH.")
    sys.exit(1)

# --- Configuration ---
TEST_DB_PATH = "./test_latex_proc_primes_kb.sqlite"  # Use a new DB file for this test

# --- Test Item: Infinitely Many Primes ---
TARGET_ITEM_NAME = "Euclid.theorem_inf_primes"
DEP_PRIME_DEF_NAME = "Nat.is_prime"
DEP_DIVIDES_DEF_NAME = "Nat.divides"
DEP_FACTOR_THM_NAME = "Nat.exists_prime_factor"

# --- Logging Setup ---
log_level = os.getenv("LOGLEVEL", "INFO").upper()
# Avoid duplicate handlers if run multiple times in same session
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("TestLatexProcessorPrimes")


async def setup_test_data(db_path: str):
    """Initializes DB and creates items for 'infinitely many primes' test."""
    logger.info(f"Setting up test database: {db_path}")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.debug("Removed existing test database.")
        except OSError as e:
            logger.error(f"Error removing existing database {db_path}: {e}")
            # Decide if this is fatal or if we can continue (maybe DB is locked)
            return False  # Indicate setup failure

    kb_storage.initialize_database(db_path)

    # --- Create Dependency Items ---
    # Manually provide accepted LaTeX statements for context.

    # 1. Definition of Prime
    dep_prime = kb_storage.KBItem(
        unique_name=DEP_PRIME_DEF_NAME,
        item_type=ItemType.DEFINITION,
        description_nl=(
            "A natural number p is prime if it is greater than 1 "
            "and its only divisors are 1 and p."
        ),
        latex_statement=(
            r"p \in \mathbb{N} \text{ is prime} \iff p > 1 \land "
            r"(\forall d \in \mathbb{N}, d | p \implies d = 1 \lor d = p)"
        ),
        latex_proof=None,  # Definitions don't have proofs
        status=ItemStatus.LATEX_ACCEPTED,  # Mark as ready
    )
    await kb_storage.save_kb_item(dep_prime, client=None, db_path=db_path)
    logger.info(
        f"Created dependency item: {dep_prime.unique_name} "
        f"with status {dep_prime.status.name}"
    )

    # 2. Definition of Divides
    dep_divides = kb_storage.KBItem(
        unique_name=DEP_DIVIDES_DEF_NAME,
        item_type=ItemType.DEFINITION,
        description_nl=(
            "A natural number d divides a natural number n if there exists "
            "a natural number k such that n = d * k."
        ),
        latex_statement=r"d | n \iff \exists k \in \mathbb{N}, n = d \times k",
        latex_proof=None,
        status=ItemStatus.LATEX_ACCEPTED,
    )
    await kb_storage.save_kb_item(dep_divides, client=None, db_path=db_path)
    logger.info(
        f"Created dependency item: {dep_divides.unique_name} "
        f"with status {dep_divides.status.name}"
    )

    # 3. Existence of Prime Factor Theorem
    dep_factor = kb_storage.KBItem(
        unique_name=DEP_FACTOR_THM_NAME,
        item_type=ItemType.THEOREM,
        description_nl=(
            "Every natural number greater than 1 has at least one prime divisor."
        ),
        latex_statement=(
            r"\forall n \in \mathbb{N}, n > 1 \implies "
            r"(\exists p \in \mathbb{N}, p \text{ is prime} \land p | n)"
        ),
        latex_proof=None,  # We only need the statement as context for the
        # target theorem's statement/proof generation
        status=ItemStatus.LATEX_ACCEPTED,  # Assume this statement is accepted
    )
    await kb_storage.save_kb_item(dep_factor, client=None, db_path=db_path)
    logger.info(
        f"Created dependency item: {dep_factor.unique_name} "
        f"with status {dep_factor.status.name}"
    )

    # --- Create Target Item ---
    target_item = kb_storage.KBItem(
        unique_name=TARGET_ITEM_NAME,
        item_type=ItemType.THEOREM,
        description_nl="There are infinitely many prime numbers.",
        plan_dependencies=[
            DEP_PRIME_DEF_NAME,
            DEP_DIVIDES_DEF_NAME,
            DEP_FACTOR_THM_NAME,
        ],  # List dependencies
        latex_statement=None,  # To be generated
        latex_proof=None,  # To be generated
        lean_code=(
            f"theorem {TARGET_ITEM_NAME} : "
            f"Set.Infinite {{ p : Nat // Nat.Prime p }} := sorry"
        ),  # Example Lean shell (not used in this test)
        status=ItemStatus.PENDING_LATEX,  # Ready for the processor
    )
    await kb_storage.save_kb_item(target_item, client=None, db_path=db_path)
    logger.info(
        f"Created target item: {target_item.unique_name} "
        f"with status {target_item.status.name}"
    )
    return True  # Indicate success


async def main():
    """Main execution function."""
    logger.info("--- Starting Basic LaTeX Processor Test (Infinitely Many Primes) ---")

    # --- Load Environment Variables ---
    if not load_dotenv():
        logger.warning(
            "Could not find .env file. Ensure environment variables are set."
        )
    if not os.getenv("GEMINI_API_KEY"):
        logger.critical("GEMINI_API_KEY environment variable not set. Exiting.")
        return

    # --- Setup Database and Test Data ---
    if not await setup_test_data(TEST_DB_PATH):
        logger.critical("Failed to set up test data. Exiting.")
        return

    # --- Initialize LLM Client ---
    client = None
    try:
        # Ensure llm_call is importable and GeminiClient exists
        if llm_call and hasattr(llm_call, "GeminiClient"):
            client = llm_call.GeminiClient()
            logger.info("GeminiClient initialized successfully.")
        else:
            raise ImportError("llm_call or GeminiClient not available.")
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiClient: {e}")
        return

    # --- Run the LaTeX Processor ---
    logger.info(f"Running generate_and_review_latex for item: {TARGET_ITEM_NAME}")
    success = False
    try:
        # Ensure latex_processor module and function are available
        if latex_processor and hasattr(latex_processor, "generate_and_review_latex"):
            success = await latex_processor.generate_and_review_latex(
                unique_name=TARGET_ITEM_NAME, client=client, db_path=TEST_DB_PATH
            )
            logger.info(
                "generate_and_review_latex finished. Result: "
                f"{'Success' if success else 'Failure'}"
            )
        else:
            raise ImportError(
                "latex_processor or generate_and_review_latex not available."
            )
    except Exception as e:
        logger.exception(f"An error occurred during generate_and_review_latex: {e}")

    # --- Verify Final State ---
    logger.info(f"Verifying final state of item: {TARGET_ITEM_NAME}")
    final_item = kb_storage.get_kb_item_by_name(TARGET_ITEM_NAME, TEST_DB_PATH)

    if final_item:
        logger.info(f"  Final Status: {final_item.status.name}")
        logger.info(
            f"  Final LaTeX Statement:\n--- START ---\n"
            f"{final_item.latex_statement}\n--- END ---"
        )
        if final_item.item_type.requires_proof():
            logger.info(
                f"  Final LaTeX Proof:\n--- START ---\n"
                f"{final_item.latex_proof}\n--- END ---"
            )
        if final_item.status == kb_storage.ItemStatus.LATEX_REJECTED_FINAL:
            logger.warning(
                f"  Final Review Feedback:\n--- START ---\n"
                f"{final_item.latex_review_feedback}\n--- END ---"
            )
        elif (
            final_item.latex_review_feedback
        ):  # Log feedback even if accepted, just in case
            logger.debug(f"  Last Review Feedback: {final_item.latex_review_feedback}")
    else:
        logger.error(f"Could not retrieve final state for item {TARGET_ITEM_NAME}")

    logger.info("--- Basic LaTeX Processor Test (Infinitely Many Primes) Finished ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
