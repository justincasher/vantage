# File: scripts/test_lean_processor_basic.py

import asyncio
import os
import logging
import sys

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

# --- Add project root to path to allow imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Module Imports ---
try:
    # Use src prefix if running script directly and src is not installed
    from src.lean_automator import kb_storage
    from src.lean_automator import lean_processor # Import the lean processor
    from src.lean_automator import llm_call
    from src.lean_automator.kb_storage import ItemType, ItemStatus
except ImportError:
    # Fallback if running as installed package
    try:
        from lean_automator import kb_storage
        from lean_automator import lean_processor
        from lean_automator import llm_call
        from lean_automator.kb_storage import ItemType, ItemStatus
    except ImportError as e:
        print(f"Error importing project modules: {e}")
        print("Ensure the script is run from the project root or the package is installed.")
        sys.exit(1)

# --- Configuration ---
TEST_DB_PATH = "./test_lean_proc_list_kb.sqlite" # Use same DB file

# --- Test Item: List Reverse Append (using a unique name) ---
TARGET_ITEM_NAME = "MyProof.list_reverse_append" # Using unique name
DEP_APPEND_ASSOC_AX_NAME = "List.append_assoc_ax"
DEP_REVERSE_CONS_AX_NAME = "List.reverse_cons_ax"
DEP_REVERSE_NIL_AX_NAME = "List.reverse_nil_ax"
DEP_APPEND_NIL_AX_NAME = "List.append_nil_ax" # l ++ [] = l
DEP_NIL_APPEND_AX_NAME = "List.nil_append_ax"   # [] ++ l = l  <- Added

# --- Logging Setup ---
log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("TestLeanProcessorList")

async def setup_test_data(db_path: str):
    """Initializes DB and creates items for MyProof.list_reverse_append test."""
    logger.info(f"Setting up test database with list axioms: {db_path}")
    if os.path.exists(db_path):
        try: os.remove(db_path); logger.debug("Removed existing test database.")
        except OSError as e: logger.error(f"Error removing DB: {e}"); return False

    kb_storage.initialize_database(db_path)
    client = None

    items_to_create = []
    dependency_names = []

    # --- Create Dependency Items as Axioms (marked as PROVEN) ---

    # 1. List.append_assoc (Axiom)
    assoc_ax = kb_storage.KBItem(
        unique_name=DEP_APPEND_ASSOC_AX_NAME, item_type=ItemType.AXIOM,
        description_nl="List append (++) is associative.",
        latex_statement=r"\forall \alpha : \text{Type}, \forall (l_1 l_2 l_3 : \text{List } \alpha), (l_1 ++ l_2) ++ l_3 = l_1 ++ (l_2 ++ l_3)",
        lean_code=f"universe u\naxiom {DEP_APPEND_ASSOC_AX_NAME} {{α : Type u}} (l₁ l₂ l₃ : List α) : (l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃)",
        status=ItemStatus.PROVEN
    )
    items_to_create.append(assoc_ax)
    dependency_names.append(assoc_ax.unique_name)

    # 2. List.reverse_cons (Axiom)
    rev_cons_ax = kb_storage.KBItem(
        unique_name=DEP_REVERSE_CONS_AX_NAME, item_type=ItemType.AXIOM,
        description_nl="Interaction of reverse with cons.",
        latex_statement=r"\forall \alpha : \text{Type}, \forall (x : \alpha) (xs : \text{List } \alpha), \text{List.reverse} (x :: xs) = \text{List.append } (\text{List.reverse } xs) [x]",
        lean_code=f"universe u\naxiom {DEP_REVERSE_CONS_AX_NAME} {{α : Type u}} (x : α) (xs : List α) : List.reverse (x :: xs) = List.append (List.reverse xs) [x]",
        status=ItemStatus.PROVEN
    )
    items_to_create.append(rev_cons_ax)
    dependency_names.append(rev_cons_ax.unique_name)

    # 3. List.reverse_nil (Axiom)
    rev_nil_ax = kb_storage.KBItem(
        unique_name=DEP_REVERSE_NIL_AX_NAME, item_type=ItemType.AXIOM,
        description_nl="Reverse of the empty list is the empty list.",
        latex_statement=r"\forall \alpha : \text{Type}, \text{List.reverse} (@List.nil \alpha) = @List.nil \alpha",
        lean_code=f"universe u\naxiom {DEP_REVERSE_NIL_AX_NAME} {{α : Type u}} : List.reverse (@List.nil α) = @List.nil α",
        status=ItemStatus.PROVEN
    )
    items_to_create.append(rev_nil_ax)
    dependency_names.append(rev_nil_ax.unique_name)

    # 4. List.append_nil (Axiom) - l ++ [] = l
    app_nil_ax = kb_storage.KBItem(
        unique_name=DEP_APPEND_NIL_AX_NAME, item_type=ItemType.AXIOM,
        description_nl="Appending nil to a list results in the original list.",
        latex_statement=r"\forall \alpha : \text{Type}, \forall (l : \text{List } \alpha), l ++ [] = l",
        lean_code=f"universe u\naxiom {DEP_APPEND_NIL_AX_NAME} {{α : Type u}} (l : List α) : l ++ [] = l",
        status=ItemStatus.PROVEN
    )
    items_to_create.append(app_nil_ax)
    dependency_names.append(app_nil_ax.unique_name)

    # 5. List.nil_append (Axiom) - [] ++ l = l 
    nil_app_ax = kb_storage.KBItem(
        unique_name=DEP_NIL_APPEND_AX_NAME, item_type=ItemType.AXIOM,
        description_nl="Prepending nil to a list results in the original list.",
        latex_statement=r"\forall \alpha : \text{Type}, \forall (l : \text{List } \alpha), [] ++ l = l",
        lean_code=f"universe u\naxiom {DEP_NIL_APPEND_AX_NAME} {{α : Type u}} (l : List α) : [] ++ l = l", # Corrected definition
        status=ItemStatus.PROVEN
    )
    items_to_create.append(nil_app_ax)
    dependency_names.append(nil_app_ax.unique_name)


    # --- Save all dependency items ---
    for item in items_to_create:
        try:
            await kb_storage.save_kb_item(item, client=None, db_path=db_path)
            logger.info(f"Created dependency item: {item.unique_name} with status {item.status.name}")
        except Exception as e:
            logger.error(f"Failed to save dependency {item.unique_name}: {e}")
            return False

    # --- Create Target Item ---
    target_latex_statement = r"""For any lists $l$ and $l'$ (over some type $\alpha$), reversing their concatenation yields the concatenation of their reverses in opposite order:
$$ \text{reverse} (l ++ l') = (\text{reverse } l') ++ (\text{reverse } l) $$"""
    target_latex_proof = r"""
We prove the statement by induction on the list $l$.

\textbf{Base Case:} $l = []$.
We need to show $\text{reverse} ([] ++ l') = (\text{reverse } l') ++ (\text{reverse } [])$.
The left side simplifies: $\text{reverse} ([] ++ l') = \text{reverse } l'$.
The right side also simplifies: $(\text{reverse } l') ++ (\text{reverse } []) = (\text{reverse } l') ++ [] = \text{reverse } l'$.
Since both sides equal $\text{reverse } l'$, the base case holds.

\textbf{Inductive Step:} Assume the property holds for some list $xs$.
That is, assume $\text{reverse} (xs ++ l') = (\text{reverse } l') ++ (\text{reverse } xs)$ (Inductive Hypothesis, IH).
We want to show the property holds for $l = x :: xs$:
$$ \text{reverse} ((x :: xs) ++ l') = (\text{reverse } l') ++ (\text{reverse } (x :: xs)) $$
Let's analyze the left-hand side (LHS):
\begin{align*} \label{eq:1} \text{LHS} &= \text{reverse} ((x :: xs) ++ l') \\ &= \text{reverse} (x :: (xs ++ l')) \quad (\text{by def of } ++) \\ &= (\text{reverse } (xs ++ l')) ++ [x] \quad (\text{by property of reverse}) \\ &= ((\text{reverse } l') ++ (\text{reverse } xs)) ++ [x] \quad (\text{by IH})\end{align*} 
Now let's analyze the right-hand side (RHS):
\begin{align*} \label{eq:2} \text{RHS} &= (\text{reverse } l') ++ (\text{reverse } (x :: xs)) \\ &= (\text{reverse } l') ++ ((\text{reverse } xs) ++ [x]) \quad (\text{by property of reverse})\end{align*} 
Comparing the final forms of the LHS and RHS, we have:
$$ ((\text{reverse } l') ++ (\text{reverse } xs)) ++ [x] \quad \text{vs} \quad (\text{reverse } l') ++ ((\text{reverse } xs) ++ [x]) $$
These are equal by the associativity of list append ($++$).
Thus, the property holds for $x :: xs$.

By the principle of induction, the statement holds for all lists $l$. QED.
""".strip()

    target_item = kb_storage.KBItem(
        unique_name=TARGET_ITEM_NAME,
        item_type=ItemType.THEOREM,
        description_nl="Reverse distributes over append in reverse order.",
        plan_dependencies=dependency_names, # Includes the new axiom name
        latex_statement=target_latex_statement,
        latex_proof=target_latex_proof,
        lean_code="",
        status=ItemStatus.PENDING_LEAN
    )
    try:
        await kb_storage.save_kb_item(target_item, client=None, db_path=db_path)
        logger.info(f"Created target item: {target_item.unique_name} with status {target_item.status.name}")
    except Exception as e:
        logger.error(f"Failed to save target item {target_item.unique_name}: {e}")
        return False

    return True


async def main():
    """Main execution function."""
    logger.info(f"--- Starting Basic Lean Processor Test ({TARGET_ITEM_NAME}) ---")

    if not os.getenv("GEMINI_API_KEY"): logger.critical("GEMINI_API_KEY missing."); return

    if not await setup_test_data(TEST_DB_PATH):
         logger.critical("Failed to set up test data. Exiting."); return

    client = None
    try:
        if llm_call and hasattr(llm_call, 'GeminiClient'): client = llm_call.GeminiClient()
        else: raise ImportError("llm_call or GeminiClient not available.")
        logger.info("GeminiClient initialized successfully.")
    except Exception as e: logger.critical(f"Failed to initialize GeminiClient: {e}"); return

    logger.info(f"Running generate_and_verify_lean for item: {TARGET_ITEM_NAME}")
    lean_success = False
    try:
        if lean_processor and hasattr(lean_processor, 'generate_and_verify_lean'):
            lean_success = await lean_processor.generate_and_verify_lean(
                unique_name=TARGET_ITEM_NAME,
                client=client,
                db_path=TEST_DB_PATH
            )
            logger.info(f"generate_and_verify_lean finished. Result: {'Success' if lean_success else 'Failure'}")
        else: raise ImportError("lean_processor or generate_and_verify_lean not available.")
    except Exception as e: logger.exception(f"An error occurred during generate_and_verify_lean: {e}")

    logger.info(f"Verifying final state of item: {TARGET_ITEM_NAME}")
    final_item = kb_storage.get_kb_item_by_name(TARGET_ITEM_NAME, TEST_DB_PATH)
    if final_item:
        logger.info(f"  Final Status: {final_item.status.name}")
        logger.info(f"  Final Lean Code (len={len(final_item.lean_code or '')}):\n--- START ---\n{final_item.lean_code}\n--- END ---")
        if final_item.status == kb_storage.ItemStatus.LEAN_VALIDATION_FAILED:
            logger.warning(f"  Lean Error Log:\n--- START ---\n{final_item.lean_error_log}\n--- END ---")
        elif final_item.lean_error_log:
             logger.debug(f"  Last Lean Error Log: {final_item.lean_error_log}")
    else: logger.error(f"Could not retrieve final state for item {TARGET_ITEM_NAME}")

    logger.info(f"--- Basic Lean Processor Test ({TARGET_ITEM_NAME}) Finished ---")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: logger.info("Test interrupted by user.")
    except Exception as e: logger.exception(f"An unexpected error occurred in main: {e}")