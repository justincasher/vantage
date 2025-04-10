# File: scripts/test_lean_processor_basic.py

import asyncio
import logging
import os
import sys

# --- Load Environment Variables or Exit ---
from dotenv import load_dotenv

env_loaded_successfully = load_dotenv()

if not env_loaded_successfully:
    # Print error message to standard error
    print("\nCRITICAL ERROR: Could not find or load the .env file.", file=sys.stderr)
    print(
        "This script relies on environment variables defined in that file.",
        file=sys.stderr,
    )
    # Show where it looked relative to, which helps debugging
    print(
        f"Please ensure a .env file exists in the current directory ({os.getcwd()}) "
        "or its parent directories.",
        file=sys.stderr,
    )
    sys.exit(1)  # Exit the script with a non-zero status code indicating failure

# --- Add project root to path to allow imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# --- Module Imports ---
try:
    from src.lean_automator.kb import storage as kb_storage
    from src.lean_automator.kb.storage import ItemStatus, ItemType
    from src.lean_automator.lean import interaction as lean_interaction
    from src.lean_automator.lean import processor as lean_processor
    from src.lean_automator.llm import caller as llm_call
except ImportError:
    # Fallback if running as installed package
    try:
        from lean_automator.kb import storage as kb_storage
        from lean_automator.kb.storage import ItemStatus, ItemType
        from lean_automator.lean import interaction as lean_interaction
        from lean_automator.lean import processor as lean_processor
        from lean_automator.llm import caller as llm_call
    except ImportError as e:
        print(f"Error importing project modules: {e}")
        print(
            "Ensure the script is run from the project root or the package is "
            "installed."
        )
        sys.exit(1)

# --- Configuration ---
TEST_DB_PATH = "./test_lean_proc_list_kb.sqlite"  # Use dedicated DB file for this test
LAKE_EXECUTABLE_PATH = os.getenv(
    "LAKE_EXECUTABLE_PATH", "lake"
)  # Get lake path from env or default
LAKE_TIMEOUT_SECONDS = int(
    os.getenv("LAKE_TIMEOUT_SECONDS", "120")
)  # Get timeout from env or default

# --- Test Item Names (with test-specific namespace) ---
NAMESPACE = "VantageLib.TestLeanProcessorBasic"
TARGET_ITEM_NAME = f"{NAMESPACE}.MyProof.list_reverse_append"
DEP_APPEND_ASSOC_AX_NAME = f"{NAMESPACE}.List.append_assoc_ax"
DEP_REVERSE_CONS_AX_NAME = f"{NAMESPACE}.List.reverse_cons_ax"
DEP_REVERSE_NIL_AX_NAME = f"{NAMESPACE}.List.reverse_nil_ax"
DEP_APPEND_NIL_AX_NAME = f"{NAMESPACE}.List.append_nil_ax"
DEP_NIL_APPEND_AX_NAME = f"{NAMESPACE}.List.nil_append_ax"

# List of dependency axiom names for easier iteration
AXIOM_NAMES = [
    DEP_APPEND_ASSOC_AX_NAME,
    DEP_REVERSE_CONS_AX_NAME,
    DEP_REVERSE_NIL_AX_NAME,
    DEP_APPEND_NIL_AX_NAME,
    DEP_NIL_APPEND_AX_NAME,
]

# --- Logging Setup ---
log_level = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("TestLeanProcessorList")


async def setup_test_data(db_path: str):
    """Initializes DB and creates items for the test."""
    logger.info(f"Setting up test database: {db_path}")
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            logger.debug("Removed existing test database.")
        except OSError as e:
            logger.error(f"Error removing DB: {e}")
            return False

    kb_storage.initialize_database(db_path)
    # client = None # No LLM needed for setup phase -> F841 Removed

    items_to_create = []

    # --- Create Dependency Items as Axioms ---
    # Status is set to AXIOM_ACCEPTED, indicating the statement is defined.
    # They still need to be placed in the shared lib via check_and_compile_item later.

    # 1. List.append_assoc (Axiom)
    assoc_ax = kb_storage.KBItem(
        unique_name=DEP_APPEND_ASSOC_AX_NAME,
        item_type=ItemType.AXIOM,
        description_nl="List append (++) is associative.",
        latex_statement=(
            r"\forall \alpha : \text{Type}, "
            r"\forall (l_1 l_2 l_3 : \text{List } \alpha), "
            r"(l_1 ++ l_2) ++ l_3 = l_1 ++ (l_2 ++ l_3)"
        ),
        lean_code=(
            f"universe u\naxiom {DEP_APPEND_ASSOC_AX_NAME} {{α : Type u}} "
            f"(l₁ l₂ l₃ : List α) : "
            f"(l₁ ++ l₂) ++ l₃ = l₁ ++ (l₂ ++ l₃)"
        ),
        status=ItemStatus.AXIOM_ACCEPTED,  # Indicate statement is ready for placement
    )
    items_to_create.append(assoc_ax)

    # 2. List.reverse_cons (Axiom)
    rev_cons_ax = kb_storage.KBItem(
        unique_name=DEP_REVERSE_CONS_AX_NAME,
        item_type=ItemType.AXIOM,
        description_nl="Interaction of reverse with cons.",
        latex_statement=(
            r"\forall \alpha : \text{Type}, \forall (x : \alpha) "
            r"(xs : \text{List } \alpha), \text{List.reverse} (x :: xs) = "
            r"\text{List.append } (\text{List.reverse } xs) [x]"
        ),
        lean_code=(
            f"universe u\naxiom {DEP_REVERSE_CONS_AX_NAME} {{α : Type u}} "
            f"(x : α) (xs : List α) : "
            f"List.reverse (x :: xs) = List.append (List.reverse xs) [x]"
        ),
        status=ItemStatus.AXIOM_ACCEPTED,
    )
    items_to_create.append(rev_cons_ax)

    # 3. List.reverse_nil (Axiom)
    rev_nil_ax = kb_storage.KBItem(
        unique_name=DEP_REVERSE_NIL_AX_NAME,
        item_type=ItemType.AXIOM,
        description_nl="Reverse of the empty list is the empty list.",
        latex_statement=(
            r"\forall \alpha : \text{Type}, "
            r"\text{List.reverse} (@List.nil \alpha) = @List.nil \alpha"
        ),
        lean_code=(
            f"universe u\naxiom {DEP_REVERSE_NIL_AX_NAME} {{α : Type u}} : "
            f"List.reverse (@List.nil α) = @List.nil α"
        ),
        status=ItemStatus.AXIOM_ACCEPTED,
    )
    items_to_create.append(rev_nil_ax)

    # 4. List.append_nil (Axiom) - l ++ [] = l
    app_nil_ax = kb_storage.KBItem(
        unique_name=DEP_APPEND_NIL_AX_NAME,
        item_type=ItemType.AXIOM,
        description_nl="Appending nil to a list results in the original list.",
        latex_statement=(
            r"\forall \alpha : \text{Type}, "
            r"\forall (l : \text{List } \alpha), l ++ [] = l"
        ),
        lean_code=(
            f"universe u\naxiom {DEP_APPEND_NIL_AX_NAME} {{α : Type u}} "
            f"(l : List α) : l ++ [] = l"
        ),
        status=ItemStatus.AXIOM_ACCEPTED,
    )
    items_to_create.append(app_nil_ax)

    # 5. List.nil_append (Axiom) - [] ++ l = l
    nil_app_ax = kb_storage.KBItem(
        unique_name=DEP_NIL_APPEND_AX_NAME,
        item_type=ItemType.AXIOM,
        description_nl="Prepending nil to a list results in the original list.",
        latex_statement=(
            r"\forall \alpha : \text{Type}, "
            r"\forall (l : \text{List } \alpha), [] ++ l = l"
        ),
        lean_code=(
            f"universe u\naxiom {DEP_NIL_APPEND_AX_NAME} {{α : Type u}} "
            f"(l : List α) : [] ++ l = l"
        ),
        status=ItemStatus.AXIOM_ACCEPTED,
    )
    items_to_create.append(nil_app_ax)

    # --- Save all dependency items ---
    for item in items_to_create:
        try:
            # No client needed here as we are providing the code and not
            # generating embeddings initially
            await kb_storage.save_kb_item(item, client=None, db_path=db_path)
            logger.info(
                f"Created dependency item entry: {item.unique_name} "
                f"with status {item.status.name}"
            )
        except Exception as e:
            logger.error(f"Failed to save dependency {item.unique_name}: {e}")
            return False

    # --- Create Target Item ---
    target_latex_statement = (
        r"""For any lists $l$ and $l'$ (over some type $\alpha$), reversing """
        r"""their concatenation yields the concatenation of their reverses """
        r"""in opposite order:
$$ \text{reverse} (l ++ l') = (\text{reverse } l') ++ (\text{reverse } l) $$"""
    )
    target_latex_proof = r"""
We prove the statement by induction on the list $l$.

\textbf{Base Case:} $l = []$.
We need to show
$\text{reverse} ([] ++ l') = (\text{reverse } l') ++ (\text{reverse } [])$.
The left side simplifies:
$\text{reverse} ([] ++ l') = \text{reverse } l'$. (using nil_append axiom)
The right side also simplifies:
$(\text{reverse } l') ++ (\text{reverse } []) = (\text{reverse } l') ++ []$
$= \text{reverse } l'$.
(using reverse_nil and append_nil axioms)
Since both sides equal $\text{reverse } l'$, the base case holds.

\textbf{Inductive Step:} Assume the property holds for some list $xs$.
That is, assume
$\text{reverse} (xs ++ l') = (\text{reverse } l') ++ (\text{reverse } xs)$
(Inductive Hypothesis, IH).
We want to show the property holds for $l = x :: xs$:
$$ \text{reverse} ((x :: xs) ++ l') =
   (\text{reverse } l') ++ (\text{reverse } (x :: xs)) $$
Let's analyze the left-hand side (LHS):
\begin{align*} \label{eq:1}
\text{LHS} &= \text{reverse} ((x :: xs) ++ l') \\
&= \text{reverse} (x :: (xs ++ l')) \quad (\text{by def of } ++) \\
&= (\text{reverse } (xs ++ l')) ++ [x] \quad (\text{by reverse_cons axiom}) \\
&= ((\text{reverse } l') ++ (\text{reverse } xs)) ++ [x] \quad (\text{by IH})
\end{align*}
Now let's analyze the right-hand side (RHS):
\begin{align*} \label{eq:2}
\text{RHS} &= (\text{reverse } l') ++ (\text{reverse } (x :: xs)) \\
&= (\text{reverse } l') ++ ((\text{reverse } xs) ++ [x])
   \quad (\text{by reverse_cons axiom})
\end{align*}
Comparing the final forms of the LHS and RHS, we have:
$$ ((\text{reverse } l') ++ (\text{reverse } xs)) ++ [x] \quad \text{vs} \quad
(\text{reverse } l') ++ ((\text{reverse } xs) ++ [x]) $$
These are equal by the associativity of list append ($++$) (using append_assoc axiom).
Thus, the property holds for $x :: xs$.

By the principle of induction, the statement holds for all lists $l$. QED.
""".strip()

    target_item = kb_storage.KBItem(
        unique_name=TARGET_ITEM_NAME,
        item_type=ItemType.THEOREM,
        description_nl="Reverse distributes over append in reverse order.",
        plan_dependencies=AXIOM_NAMES,  # Use the list of new axiom names
        latex_statement=target_latex_statement,
        latex_proof=target_latex_proof,
        lean_code="",  # To be generated
        status=ItemStatus.PENDING_LEAN,  # Ready for LLM generation and verification
    )
    try:
        # No client needed here
        await kb_storage.save_kb_item(target_item, client=None, db_path=db_path)
        logger.info(
            f"Created target item entry: {target_item.unique_name} "
            f"with status {target_item.status.name}"
        )
    except Exception as e:
        logger.error(f"Failed to save target item {target_item.unique_name}: {e}")
        return False

    return True


async def main():
    """Main execution function."""
    logger.info(f"--- Starting Lean Processor Test with Namespace ({NAMESPACE}) ---")

    if not os.getenv("GEMINI_API_KEY"):
        logger.critical("GEMINI_API_KEY missing from environment variables.")
        return
    if not os.getenv("LEAN_AUTOMATOR_SHARED_LIB_PATH"):
        logger.critical(
            "LEAN_AUTOMATOR_SHARED_LIB_PATH missing from environment variables."
        )
        return

    # --- Setup Database ---
    if not await setup_test_data(TEST_DB_PATH):
        logger.critical("Failed to set up test data. Exiting.")
        return

    # --- Initialize LLM Client ---
    # Although axioms don't need generation, the target theorem does.
    client = None
    try:
        if llm_call and hasattr(llm_call, "GeminiClient"):
            client = llm_call.GeminiClient()
            logger.info("GeminiClient initialized successfully.")
        else:
            raise ImportError("llm_call or GeminiClient not available.")
    except Exception as e:
        logger.critical(f"Failed to initialize GeminiClient: {e}")
        return

    # --- Process Axioms: Place and Compile into Shared Library ---
    logger.info("--- Phase 1: Placing and Compiling Axioms ---")
    all_axioms_placed = True
    for axiom_name in AXIOM_NAMES:
        logger.info(f"Processing axiom: {axiom_name}")
        try:
            # Use lean_interaction directly to place/compile predefined axioms
            (
                placed_successfully,
                message,
            ) = await lean_interaction.check_and_compile_item(
                unique_name=axiom_name,
                db_path=TEST_DB_PATH,
                lake_executable_path=LAKE_EXECUTABLE_PATH,
                timeout_seconds=LAKE_TIMEOUT_SECONDS,
            )

            if placed_successfully:
                # Check if the status was indeed updated to PROVEN in the DB
                axiom_item = kb_storage.get_kb_item_by_name(axiom_name, TEST_DB_PATH)
                if axiom_item and axiom_item.status == ItemStatus.PROVEN:
                    logger.info(
                        f"Axiom {axiom_name} successfully placed and compiled. "
                        f"Status: PROVEN."
                    )
                elif axiom_item:
                    logger.warning(
                        f"Axiom {axiom_name} check_and_compile returned success, "
                        f"but DB status is {axiom_item.status.name}."
                    )
                    # Might still be okay if persistent build passed, but worth noting
                else:
                    logger.warning(
                        f"Axiom {axiom_name} check_and_compile returned success, "
                        f"but could not re-fetch item from DB."
                    )

            else:
                logger.error(
                    f"Failed to place and compile axiom {axiom_name}. Reason: {message}"
                )
                all_axioms_placed = False
                # Optionally retrieve item to log specific lean_error_log if needed
                axiom_item = kb_storage.get_kb_item_by_name(axiom_name, TEST_DB_PATH)
                if axiom_item and axiom_item.lean_error_log:
                    logger.error(
                        f"Lean Error Log for {axiom_name}:\n{axiom_item.lean_error_log}"
                    )
                break  # Stop processing axioms if one fails

        except Exception as e:
            logger.exception(
                "An unexpected error occurred during check_and_compile_item "
                f"for axiom {axiom_name}: {e}"
            )
            all_axioms_placed = False
            break

    # --- Process Target Theorem (only if all axioms are ready) ---
    if all_axioms_placed:
        logger.info("--- Phase 2: Generating and Verifying Target Theorem ---")
        logger.info(
            f"Running generate_and_verify_lean for target item: {TARGET_ITEM_NAME}"
        )
        lean_success = False
        try:
            if lean_processor and hasattr(lean_processor, "generate_and_verify_lean"):
                lean_success = await lean_processor.generate_and_verify_lean(
                    unique_name=TARGET_ITEM_NAME,
                    client=client,
                    db_path=TEST_DB_PATH,
                    lake_executable_path=LAKE_EXECUTABLE_PATH,
                    timeout_seconds=LAKE_TIMEOUT_SECONDS,
                )
                logger.info(
                    "generate_and_verify_lean finished for target. "
                    f"Result: {'Success' if lean_success else 'Failure'}"
                )
            else:
                raise ImportError(
                    "lean_processor or generate_and_verify_lean not available."
                )
        except Exception as e:
            logger.exception(
                "An error occurred during generate_and_verify_lean "
                f"for target {TARGET_ITEM_NAME}: {e}"
            )

    else:
        logger.error(
            "--- Phase 2 Skipped: "
            "Not all axioms were successfully placed and compiled. ---"
        )

    # --- Final Verification ---
    logger.info("--- Final State Verification ---")
    final_target_item = kb_storage.get_kb_item_by_name(TARGET_ITEM_NAME, TEST_DB_PATH)
    if final_target_item:
        logger.info(f"Target Item: {TARGET_ITEM_NAME}")
        logger.info(f"  Final Status: {final_target_item.status.name}")
        logger.info(
            f"  Final Lean Code (len={len(final_target_item.lean_code or '')}):\n"
            f"--- START ---\n{final_target_item.lean_code}\n--- END ---"
        )
        if (
            final_target_item.status == kb_storage.ItemStatus.LEAN_VALIDATION_FAILED
            and final_target_item.lean_error_log
        ):
            logger.warning(
                "  Lean Error Log:\n--- START ---\n"
                f"{final_target_item.lean_error_log}\n--- END ---"
            )
        elif final_target_item.lean_error_log:
            logger.debug(
                "  Last Lean Error Log (present even on success?):\n"
                f"{final_target_item.lean_error_log}"
            )
    else:
        logger.error(
            f"Could not retrieve final state for target item {TARGET_ITEM_NAME}"
        )

    # Log final status of axioms too
    logger.info("Final Axiom Statuses:")
    for name in AXIOM_NAMES:
        item = kb_storage.get_kb_item_by_name(name, TEST_DB_PATH)
        status_name = item.status.name if item else "NOT_FOUND"
        logger.info(f"  {name}: {status_name}")

    logger.info(f"--- Lean Processor Test ({NAMESPACE}) Finished ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main: {e}")
