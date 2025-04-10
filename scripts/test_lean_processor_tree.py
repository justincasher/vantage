# File: scripts/test_lean_processor_tree.py

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# --- Add project root to path to allow imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Module Imports ---
try:
    # Use src prefix if running script directly and src is not installed
    from src.lean_automator.kb import storage as kb_storage
    from src.lean_automator.kb.storage import ItemStatus, ItemType
    from src.lean_automator.lean import processor as lean_processor
    from src.lean_automator.llm import caller as llm_call
except ImportError:
    # Fallback if running as installed package
    try:
        from lean_automator.kb import storage as kb_storage
        from lean_automator.kb.storage import ItemStatus, ItemType
        from lean_automator.lean import processor as lean_processor
        from lean_automator.llm import caller as llm_call
    except ImportError as e:
        print(f"Error importing project modules: {e}")
        print(
            "Ensure the script is run from the project root or the package is installed."
        )
        sys.exit(1)

# --- Configuration ---
TEST_DB_PATH = "./test_lean_proc_tree_kb.sqlite"  # Use a separate DB for this test

# --- Test Item: Tree Mirror Mirror ---
DEF_TREE_TYPE_NAME = "MyDefs.MyTree"
# DEF_MIRROR_FUNC_NAME = "MyDefs.MyTree.mirror" # OLD - Incorrect structure/name
DEF_MIRROR_FUNC_NAME = "MyDefs.mirror"  # NEW - Correct structure/standard name
TARGET_ITEM_NAME = "MyProofs.MyTree.mirror_mirror"

# --- Logging Setup ---
log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("TestLeanProcessorTree")


async def setup_test_data(db_path: str):
    """Initializes DB and creates items for MyProofs.MyTree.mirror_mirror test."""
    logger.info(f"Setting up test database with tree definitions: {db_path}")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.debug("Removed existing test database.")
        except OSError as e:
            logger.error(f"Error removing DB: {e}")
            return False

    kb_storage.initialize_database(db_path)
    client = None

    items_to_create = []
    dependency_names = []

    # --- Create Dependency Items (Definitions) ---

    # 1. MyTree Type Definition (Keep namespace)
    tree_type_lean_code = """
universe u
namespace MyDefs
inductive MyTree (α : Type u) where
  | leaf : MyTree α
  | node : MyTree α → α → MyTree α → MyTree α
end MyDefs
""".strip()
    tree_type_item = kb_storage.KBItem(
        unique_name=DEF_TREE_TYPE_NAME,  # "MyDefs.MyTree"
        item_type=ItemType.DEFINITION,
        description_nl="A simple binary tree structure.",
        latex_statement=r"Inductive type $\texttt{MyTree } \alpha$ with constructors $\texttt{leaf}$ and $\texttt{node}$.",
        lean_code=tree_type_lean_code,
        status=ItemStatus.DEFINITION_ADDED,
    )
    items_to_create.append(tree_type_item)
    dependency_names.append(
        tree_type_item.unique_name
    )  # Add "MyDefs.MyTree" to dependencies list

    # 2. mirror Function Definition (Corrected structure and name)
    mirror_func_lean_code = """
universe u
namespace MyDefs

-- Requires MyTree definition to be available (implicitly, as it's in the same namespace)

-- Define the mirror function within the MyDefs namespace
def mirror {α : Type u} (t : MyTree α) : MyTree α :=
  match t with
  | MyTree.leaf => MyTree.leaf -- Can use shorter names within the namespace
  | MyTree.node l x r => MyTree.node (mirror r) x (mirror l) -- Recursive calls use shorter name

end MyDefs
""".strip()
    mirror_func_item = kb_storage.KBItem(
        unique_name=DEF_MIRROR_FUNC_NAME,  # "MyDefs.mirror"
        item_type=ItemType.DEFINITION,
        description_nl="Mirrors a binary tree by swapping left and right children recursively.",
        latex_statement=r"Function $\text{mirror}(t)$ defined recursively on trees.",
        lean_code=mirror_func_lean_code,
        status=ItemStatus.DEFINITION_ADDED,
        plan_dependencies=[
            DEF_TREE_TYPE_NAME
        ],  # The definition depends on the Tree type
    )
    items_to_create.append(mirror_func_item)
    dependency_names.append(
        mirror_func_item.unique_name
    )  # Add "MyDefs.mirror" to dependencies list

    # --- Save dependency items ---
    for item in items_to_create:
        try:
            await kb_storage.save_kb_item(item, client=None, db_path=db_path)
            logger.info(
                f"Created dependency/definition item: {item.unique_name} with status {item.status.name if item.status else 'None'}"
            )  # Adjusted status logging
        except Exception as e:
            logger.error(f"Failed to save dependency {item.unique_name}: {e}")
            return False

    items_to_create = []  # Reset

    # --- Create Target Item (Theorem) ---
    # (Theorem definition remains the same)
    target_latex_statement = r"""
For any binary tree $t$ (over some type $\alpha$), let $\text{mirror}(t)$ be the tree obtained by recursively swapping left and right children. Then $\text{mirror}(\text{mirror}(t)) = t$.
""".strip()
    target_latex_proof = r"""
We prove by induction on the structure of the tree $t$.

\textbf{Base Case:} $t = \text{leaf}$.
We need to show $\text{mirror}(\text{mirror}(\text{leaf})) = \text{leaf}$.
By definition, $\text{mirror}(\text{leaf}) = \text{leaf}$.
So, $\text{mirror}(\text{mirror}(\text{leaf})) = \text{mirror}(\text{leaf}) = \text{leaf}$. The base case holds.

\textbf{Inductive Step:} $t = \text{node}(l, x, r)$.
Assume the property holds for the subtrees $l$ and $r$.
Inductive Hypotheses (IH): $\text{mirror}(\text{mirror}(l)) = l$ (IH$_l$) and $\text{mirror}(\text{mirror}(r)) = r$ (IH$_r$).
We need to show $\text{mirror}(\text{mirror}(\text{node}(l, x, r))) = \text{node}(l, x, r)$.
Let's compute the left side:
\begin{align*} \text{LHS} &= \text{mirror}(\text{mirror}(\text{node}(l, x, r))) \\ &= \text{mirror}(\text{node}(\text{mirror}(r), x, \text{mirror}(l))) \quad (\text{by def of mirror}) \\ &= \text{node}(\text{mirror}(\text{mirror}(l)), x, \text{mirror}(\text{mirror}(r))) \quad (\text{by def of mirror}) \\ &= \text{node}(l, x, r) \quad (\text{by IH}_l \text{ and IH}_r)\end{align*}
This matches the right side. The inductive step holds.

By induction, the property holds for all trees $t$. QED.
""".strip()

    target_item = kb_storage.KBItem(
        unique_name=TARGET_ITEM_NAME,  # "MyProofs.MyTree.mirror_mirror"
        item_type=ItemType.THEOREM,
        description_nl="Mirroring a tree twice returns the original tree.",
        plan_dependencies=dependency_names,  # Use the collected list: ["MyDefs.MyTree", "MyDefs.mirror"]
        latex_statement=target_latex_statement,
        latex_proof=target_latex_proof,
        lean_code="",  # Start with no lean code for the theorem
        status=ItemStatus.PENDING_LEAN,
    )
    try:
        await kb_storage.save_kb_item(target_item, client=None, db_path=db_path)
        logger.info(
            f"Created target item: {target_item.unique_name} with status {target_item.status.name if target_item.status else 'None'}"
        )  # Adjusted status logging
    except Exception as e:
        logger.error(f"Failed to save target item {target_item.unique_name}: {e}")
        return False

    return True


async def main():
    """Main execution function."""
    logger.info(f"--- Starting Lean Processor Test ({TARGET_ITEM_NAME}) ---")

    if not load_dotenv():
        logger.warning("Could not find .env file.")
    if not os.getenv("GEMINI_API_KEY"):
        logger.critical("GEMINI_API_KEY missing.")
        return

    if not await setup_test_data(TEST_DB_PATH):
        logger.critical("Failed to set up test data. Exiting.")
        return

    client = None
    try:
        if llm_call and hasattr(llm_call, "GeminiClient"):
            client = llm_call.GeminiClient()
        else:
            raise ImportError("llm_call or GeminiClient not available.")
        logger.info("GeminiClient initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiClient: {e}")
        return

    logger.info(f"Running generate_and_verify_lean for item: {TARGET_ITEM_NAME}")
    lean_success = False
    try:
        if lean_processor and hasattr(lean_processor, "generate_and_verify_lean"):
            # Pass any necessary kwargs like lake_executable_path if not default
            lean_success = await lean_processor.generate_and_verify_lean(
                unique_name=TARGET_ITEM_NAME,
                client=client,
                db_path=TEST_DB_PATH,
                # lake_executable_path='path/to/lake' # Optional: if lake is not in PATH
                # timeout_seconds=180 # Optional: increase timeout
            )
            logger.info(
                f"generate_and_verify_lean finished. Result: {'Success' if lean_success else 'Failure'}"
            )
        else:
            logger.critical("lean_processor or generate_and_verify_lean not available.")
            raise ImportError(
                "lean_processor or generate_and_verify_lean not available."
            )
    except Exception as e:
        logger.exception(f"An error occurred during generate_and_verify_lean: {e}")

    logger.info(f"Verifying final state of item: {TARGET_ITEM_NAME}")
    final_item = kb_storage.get_kb_item_by_name(TARGET_ITEM_NAME, TEST_DB_PATH)
    if final_item:
        logger.info(
            f"  Final Status: {final_item.status.name if final_item.status else 'None'}"
        )  # Adjusted status logging
        logger.info(
            f"  Final Lean Code (len={len(final_item.lean_code or '')}):\n--- START ---\n{final_item.lean_code}\n--- END ---"
        )
        if final_item.status == kb_storage.ItemStatus.LEAN_VALIDATION_FAILED:
            logger.warning(
                f"  Lean Error Log:\n--- START ---\n{final_item.lean_error_log}\n--- END ---"
            )
        elif final_item.lean_error_log:
            logger.debug(f"  Last Lean Error Log: {final_item.lean_error_log}")
    else:
        logger.error(f"Could not retrieve final state for item {TARGET_ITEM_NAME}")

    logger.info(f"--- Lean Processor Test ({TARGET_ITEM_NAME}) Finished ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
