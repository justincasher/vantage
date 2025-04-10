# Lean Automator: Usage Examples

This document provides examples demonstrating the core workflow of the Lean Automator system. It aims to give a practical overview of how the different modules (`kb_storage`, `llm_call`, `latex_processor`, `lean_processor`, `kb_search`) interact to build and manage the mathematical knowledge base.

**Prerequisites:**

* Python environment set up with dependencies installed as per the [Installation & Setup](../getting-started/index.md) guide.
* Lean 4 and Lake installed correctly.
* **Application configured:** Ensure you have followed the configuration steps outlined in the [Installation & Setup](../getting-started/index.md) and [Configuration](../getting-started/configuration.md) documents. This typically involves:
    * Creating a `.env` file in the project root.
    * Setting your `GEMINI_API_KEY` in the `.env` file.
    * Setting the `LEAN_AUTOMATOR_SHARED_LIB_PATH` in the `.env` file to the absolute path of your initialized shared Lean library.
* The shared Lean library (specified by `LEAN_AUTOMATOR_SHARED_LIB_PATH`) must be initialized (Steps 7-11 in the Installation guide).

*(These examples assume async execution using `asyncio.run()`. Due to the asynchronous nature and potential subprocess usage (especially for Lean interaction), running these examples directly in a Python script or an interactive terminal like `ipython` is recommended over standard Jupyter Notebook environments.)*

---

## 1. Initialization

First, ensure the database schema exists and initialize the Gemini client. The configuration system will load defaults and apply any overrides from your `.env` file.

```python
import asyncio
from lean_automator.kb import storage as kb_storage
from lean_automator.llm import caller as llm_call

# Initialize the database (creates tables if they don't exist)
# Uses the configured path (env var KB_DB_PATH or default from config.yml)
print("Initializing database...")
kb_storage.initialize_database() # Assumes this function uses the configured path
print("Database initialized.")

# Initialize the Gemini Client
# Reads API key, model names, retry settings etc. from the environment
# (using values from .env or defaults from config.yml/model_costs.json via config loader)
try:
    print("Initializing Gemini client...")
    client = llm_call.GeminiClient() # Assumes GeminiClient uses the config loader internally
    print("Gemini client initialized.")
    # Accessing config might be different now, depending on GeminiClient implementation
    # Check how config is exposed (e.g., via APP_CONFIG) if needed:
    # from lean_automator.config_loader import APP_CONFIG
    # print(f"Using Generation Model: {APP_CONFIG.llm.default_gemini_model}")
    # print(f"Using Embedding Model: {APP_CONFIG.embedding.default_embedding_model}")
except Exception as e:
    print(f"FATAL: Failed to initialize Gemini Client: {e}")
    # Cannot proceed without the client
    exit()
```

---

## 2. Creating and Saving a Basic KB Item (Definition)

Let's create a simple definition and save it to the knowledge base.

```python
# Continued from previous block...
from lean_automator.kb.storage import KBItem, ItemType, ItemStatus, save_kb_item

async def add_definition():
    print("\nCreating a new definition KBItem...")
    definition_item = KBItem(
        unique_name="MyProject.BasicDefs.ZeroIsNatural", # Use Lean-like naming convention
        item_type=ItemType.DEFINITION,
        description_nl="Defines that the number zero is considered a natural number in our system.",
        topic="MyProject.BasicDefs",
        # No LaTeX or Lean code initially
        status=ItemStatus.PENDING # Start as Pending
    )

    print(f"Saving item '{definition_item.unique_name}'...")
    try:
        # save_kb_item is async and handles INSERT or UPDATE.
        # If description_nl or latex_statement changes and client is provided,
        # it can automatically trigger embedding generation.
        saved_item = await save_kb_item(definition_item, client=client) # Assumes client is initialized
        print(f"Item saved successfully with ID: {saved_item.id} and status: {saved_item.status.name}")
        # If description_nl was present, embedding_nl might now be populated (as bytes)
        print(f"Embedding NL present: {saved_item.embedding_nl is not None}")
        return saved_item.unique_name # Return name for dependency use
    except Exception as e:
        print(f"Error saving item: {e}")
        return None

# Run the async function
definition_name = asyncio.run(add_definition())
```

---

## 3. Adding a Dependent Item (Theorem)

Now, add a theorem that depends on the definition created above.

```python
# Continued from previous block...
async def add_theorem(dependency_name: str):
    if not dependency_name:
        print("\nSkipping theorem creation, definition dependency failed.")
        return None

    print("\nCreating a new theorem KBItem...")
    theorem_item = KBItem(
        unique_name="MyProject.Theorems.ZeroProperty",
        item_type=ItemType.THEOREM,
        description_nl="States a simple property involving the natural number zero, based on its definition.",
        topic="MyProject.Theorems",
        plan_dependencies=[dependency_name], # List the unique_name of the definition
        status=ItemStatus.PENDING # Start as Pending, needs processing
    )

    print(f"Saving item '{theorem_item.unique_name}'...")
    try:
        saved_item = await save_kb_item(theorem_item, client=client) # Assumes client is initialized
        print(f"Item saved successfully with ID: {saved_item.id} and status: {saved_item.status.name}")
        print(f"Plan dependencies: {saved_item.plan_dependencies}")
        print(f"Embedding NL present: {saved_item.embedding_nl is not None}")
        return saved_item.unique_name
    except Exception as e:
        print(f"Error saving item: {e}")
        return None

# Run the async function, passing the name of the definition
theorem_name = asyncio.run(add_theorem(definition_name))

```

---

## 4. Processing LaTeX

Items often start with just a description. The `latex_processor` uses the LLM to generate and review the LaTeX statement and proof (if required).

```python
# Continued from previous block...
from lean_automator.latex import processor as latex_processor
from lean_automator.kb import storage as kb_storage 
from lean_automator.kb.storage import ItemStatus

async def process_latex(item_name: str):
    if not item_name:
        print("\nSkipping LaTeX processing, theorem creation failed.")
        return False

    print(f"\nProcessing LaTeX for item '{item_name}'...")
    # This function calls the LLM multiple times (generate, review, potentially refine).
    # It updates the item in the database.
    # Assumes the dependency 'MyProject.BasicDefs.ZeroIsNatural' has acceptable LaTeX (or is simple enough).
    # For this example, let's assume the definition doesn't need complex LaTeX processing first.
    try:
        success = await latex_processor.generate_and_review_latex(
            unique_name=item_name,
            client=client # Assumes client is initialized
            # db_path=... # Optional override for DB path if needed
        )

        if success:
            print(f"LaTeX processing successful for {item_name}.")
            # Check the item status in DB - should be LATEX_ACCEPTED
            item_after_latex = kb_storage.get_kb_item_by_name(item_name) # Assumes uses configured DB path
            if item_after_latex:
                print(f"New status: {item_after_latex.status.name}")
                print(f"LaTeX Statement present: {item_after_latex.latex_statement is not None}")
                print(f"LaTeX Proof present: {item_after_latex.latex_proof is not None}")
                # Saving the accepted statement triggered embedding generation
                print(f"Embedding LaTeX present: {item_after_latex.embedding_latex is not None}")
            return True
        else:
            print(f"LaTeX processing failed for {item_name}. Check logs and DB status/feedback.")
            # Status might be LATEX_REJECTED_FINAL or ERROR
            item_after_latex = kb_storage.get_kb_item_by_name(item_name) # Assumes uses configured DB path
            if item_after_latex:
                 print(f"Final status: {item_after_latex.status.name}")
                 print(f"Review Feedback: {item_after_latex.latex_review_feedback}")
            return False
    except Exception as e:
        print(f"Error during LaTeX processing call: {e}")
        return False

# Run the async function
latex_success = asyncio.run(process_latex(theorem_name))
```

---

## 5. Processing Lean Code

Once LaTeX is accepted (`LATEX_ACCEPTED`), the `lean_processor` can generate and verify the Lean 4 code.

```python
# Continued from previous block...
from lean_automator.lean import processor as lean_processor
from lean_automator.kb import storage as kb_storage
from lean_automator.kb.storage import ItemStatus

async def process_lean(item_name: str, previous_step_success: bool):
    if not item_name or not previous_step_success:
        print("\nSkipping Lean processing due to previous step failure or missing item.")
        return False

    # Fetch item to ensure it's in the correct state
    item = kb_storage.get_kb_item_by_name(item_name) # Assumes uses configured DB path
    if not item or item.status != ItemStatus.LATEX_ACCEPTED:
        print(f"\nSkipping Lean processing for {item_name}. Expected LATEX_ACCEPTED status, found {item.status.name if item else 'MISSING'}.")
        return False

    print(f"\nProcessing Lean code for item '{item_name}'...")
    # This function involves LLM calls (statement gen, proof gen) and
    # calls lean_interaction.check_and_compile_item, which uses 'lake'
    # and interacts with the persistent shared library (using configured path).
    # Assumes the dependency ('MyProject.BasicDefs.ZeroIsNatural') has been
    # successfully processed to PROVEN status and exists in the shared library.
    # For this demo, we'll assume that happened somehow (e.g., manual addition or prior run).
    try:
        # Make sure the dependency is marked PROVEN conceptually for this to work
        # In a real run, you'd process dependencies first.
        print("INFO: Assuming dependency 'MyProject.BasicDefs.ZeroIsNatural' is PROVEN in the shared library.")

        success = await lean_processor.generate_and_verify_lean(
            unique_name=item_name,
            client=client # Assumes client is initialized
            # db_path=..., # Optional override for DB path
            # lake_executable_path=..., # Optional override for lake path
            # timeout_seconds=... # Optional override for lean interaction timeout
        )

        # Check final status
        item_after_lean = kb_storage.get_kb_item_by_name(item_name) # Assumes uses configured DB path
        status_after = item_after_lean.status.name if item_after_lean else "MISSING"

        if success:
            print(f"Lean processing successful for {item_name}. Final Status: {status_after}")
            # Item status should be PROVEN, and code added to the shared library.
            print(f"Lean code present: {item_after_lean.lean_code is not None and 'sorry' not in item_after_lean.lean_code}")
            return True
        else:
            print(f"Lean processing failed for {item_name}. Final Status: {status_after}")
            # Status might be LEAN_VALIDATION_FAILED or ERROR. Check lean_error_log.
            print(f"Lean Error Log present: {item_after_lean.lean_error_log is not None}")
            # print(f"Error Log Snippet: {item_after_lean.lean_error_log[:500] if item_after_lean.lean_error_log else 'N/A'}")
            return False
    except Exception as e:
        print(f"Error during Lean processing call: {e}")
        # Manually check DB status if needed
        item_after_crash = kb_storage.get_kb_item_by_name(item_name) # Assumes uses configured DB path
        status_after = item_after_crash.status.name if item_after_crash else "MISSING"
        print(f"Status after crash: {status_after}")
        return False

# Run the async function
lean_success = asyncio.run(process_lean(theorem_name, latex_success))
```

---

## 6. Semantic Search

After processing, items have embeddings (from description or LaTeX statement). We can search for items related to a query.

```python
# Continued from previous block...
from lean_automator.kb import search as kb_search

async def perform_search(query: str):
    if not query:
        print("\nSkipping search, no query provided.")
        return

    print(f"\nPerforming semantic search for: '{query}'")
    try:
        # Search against the natural language description embeddings ('nl')
        # Alternatively, use 'latex' to search against latex_statement embeddings
        # Assumes client is initialized and kb_search uses configured DB path
        similar_items = await kb_search.find_similar_items(
            query_text=query,
            search_field='nl', # 'nl' or 'latex'
            client=client, # Assumes client is initialized
            top_n=3
            # db_path=... # Optional override for DB path
        )

        if similar_items:
            print("Found similar items:")
            for item, score in similar_items:
                print(f"- {item.unique_name} (Type: {item.item_type.name}, Status: {item.status.name}) - Score: {score:.4f}")
        else:
            print("No similar items found.")

    except Exception as e:
        print(f"Error during semantic search: {e}")

# Construct a query related to the theorem we added
search_query = "natural number zero property"
if theorem_name: # Only search if the theorem was likely created
    asyncio.run(perform_search(search_query))
else:
     asyncio.run(perform_search("Example search query about mathematics"))

```

---

## 7. Direct Embedding Generation

You can also generate embeddings directly using the client.

```python
# Continued from previous block...
async def generate_single_embedding():
    print("\nGenerating a direct embedding...")
    text_to_embed = "Cosine similarity measures the cosine of the angle between two non-zero vectors."
    task = "SEMANTIC_SIMILARITY" # Or RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT etc.
    try:
        # Assumes client is initialized
        embeddings = await client.embed_content(
            contents=text_to_embed,
            task_type=task
        )
        if embeddings:
            print(f"Generated embedding for task '{task}'. Vector dimensions: {len(embeddings[0])}")
            # print(f"Embedding vector (first 10 dims): {embeddings[0][:10]}")
        else:
            print("Embedding generation returned empty result.")
    except Exception as e:
        print(f"Error generating direct embedding: {e}")

asyncio.run(generate_single_embedding())
```

---

## 8. Retrieving Cost Summary

Track estimated API costs using the cost tracker integrated into the `GeminiClient` (assuming it's implemented this way and accessible).

```python
# Continued from previous block...

print("\nRetrieving API Cost Summary...")
# The client's cost_tracker accumulates usage across all calls (generate, embed)
# Assumes client is initialized and has cost tracking integrated
try:
    # Accessing cost_tracker might depend on GeminiClient implementation
    summary = client.cost_tracker.get_summary()
    import json
    print(json.dumps(summary, indent=2))

    # You can also get just the total
    total_cost = client.cost_tracker.get_total_cost()
    print(f"\nEstimated Total Cost: ${total_cost:.6f}")
except AttributeError:
    print("Cost tracking might not be implemented or accessible via client.cost_tracker.")
except Exception as e:
    print(f"Error retrieving cost summary: {e}")

```

---

This concludes the basic usage examples, demonstrating the flow from item creation through LaTeX and Lean processing, leveraging LLMs and the Lean compiler, and utilizing semantic search capabilities, all based on the configured environment.
