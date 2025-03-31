# Lean Automator

Lean Automator is a Python project designed to build and manage a mathematical knowledge base (KB). It leverages Lean 4 for formal verification, integrates with Large Language Models (specifically Google's Gemini) for potential code/description generation, and uses an SQLite database for persistent storage of mathematical items (source code, metadata, embeddings).

## Features

* **Knowledge Base Storage:** Defines and stores mathematical items (`KBItem`: Theorems, Definitions, Axioms, etc.) including their Lean source code, metadata, and embeddings in an SQLite database (`kb_storage.py`).
* **Lean Interaction & Persistent Library:** Verifies Lean 4 code snippets using the `lake` build tool. It manages dependencies by linking against a **persistent, incrementally growing shared Lean library** (`vantage_lib` by default). When an item is successfully verified:
    * Its status is updated to `PROVEN` in the database.
    * Its `.lean` source code is added to the persistent shared library.
    * The persistent shared library is rebuilt using `lake` to incorporate the new item and update its compiled (`.olean`) artifacts.
    * This allows subsequent verifications to reuse the compiled artifacts from the shared library, significantly speeding up builds (`lean_interaction.py`).
* **LLM Integration:** Interacts with the Google Gemini API for tasks like generating Lean code or natural language descriptions. Includes features like asynchronous calls, automatic retries with exponential backoff, and cost tracking (`llm_call.py`).
* **Structured Data:** Uses dataclasses (`KBItem`, `LatexLink`, `ItemStatus`, `ItemType`) for well-defined knowledge representation.
* **Dependency Management:** Tracks planned dependencies between KB items (`plan_dependencies`).
* **State Tracking:** Monitors the status of each KB item (e.g., `PENDING_LEAN`, `LEAN_VALIDATION_FAILED`, `PROVEN`, `ERROR`).
* **Cost Tracking:** Tracks token usage and estimates costs for Gemini API calls based on configurable rates.

## Project Structure

```
.
├── LICENSE
├── README.md
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── scripts
│   ├── test_latex_processor_basic.py
│   ├── test_lean_processor_basic.py
│   └── test_lean_processor_tree.py
├── src
│   ├── lean_automator
│   │   ├── __init__.py
│   │   ├── kb_search.py
│   │   ├── kb_storage.py
│   │   ├── latex_processor.py
│   │   ├── lean_interaction.py
│   │   ├── lean_processor.py
│   │   ├── lean_proof_repair.py
│   │   └── llm_call.py
│   └── vantage.egg-info
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       └── top_level.txt
├── test_lean_proc_list_kb.sqlite
├── test_lean_proc_tree_kb.sqlite
└── tests
    ├── __init__.py
    ├── integration
    │   ├── __init__.py
    │   ├── test_kb_search_integration.py
    │   ├── test_kb_storage_integration.py
    │   ├── test_lean_interaction_integration.py
    │   └── test_llm_call_integration.py
    └── unit
        ├── __init__.py
        ├── test_kb_search_unit.py
        ├── test_kb_storage_unit.py
        ├── test_lean_interaction_unit.py
        └── test_llm_call_unit.py
```

*(Note: `.sqlite` database files and `.env` are typically generated/created in the root but omitted from the source structure example)*

## Prerequisites

* **Python:** Version 3.8 or higher is recommended.
* **Lean 4 & Lake:** You need a working installation of Lean 4 and its build tool, Lake. Ensure the `lake` executable is available in your system's PATH. See [Lean Installation Guide](https://lean-lang.org/toolchains/).
* **Google AI API Key:** To use the LLM features, you need an API key for Google's Generative AI (Gemini). You can get one from [Google AI Studio](https://aistudio.google.com/).

## Installation & Setup

Follow these steps to get the Lean Automator project running. 

*The shared library initialization (Steps 5-10) only needs to be done once.* The `lean_processor` will automatically add new `.lean` files to the library's source directory (e.g., `vantage_lib/VantageLib/`) and trigger builds within the shared library directory later.


1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name> # e.g., cd vantage
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the project root directory (e.g., `vantage/.env`). Add your API key and placeholders for library configuration initially. You will confirm the exact values in later steps.
    ```dotenv
    # Required:
    GEMINI_API_KEY=YOUR_GEMINI_KEY_HERE
    DEFAULT_GEMINI_MODEL=gemini-1.5-flash-latest # Or your preferred model

    # --- Shared Library Configuration (Confirm values in steps below) ---
    # Path (absolute recommended) to the directory created in Step 5
    LEAN_AUTOMATOR_SHARED_LIB_PATH=/replace/with/absolute/path/to/vantage_lib
    # The Lean library module name defined inside the shared library project (Step 7)
    LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME=VantageLib # Default if using steps below

    # Optional (Defaults shown):
    # GEMINI_MODEL_COSTS='{"gemini-1.5-flash-latest": {"input": 0.35, "output": 0.70}}'
    # KB_DB_PATH=knowledge_base.sqlite
    # GEMINI_MAX_RETRIES=3
    # GEMINI_BACKOFF_FACTOR=1.0
    # LEAN_STATEMENT_MAX_ATTEMPTS=2
    # LEAN_PROOF_MAX_ATTEMPTS=3
    # LEAN_AUTOMATOR_LAKE_CACHE=.lake_cache # Optional: For external Lake deps cache
    ```
    * Replace `YOUR_GEMINI_KEY_HERE` with your actual key.
    * You will set the correct **absolute path** for `LEAN_AUTOMATOR_SHARED_LIB_PATH` and confirm `LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME` in Step 10.

5.  **Create Shared Library Directory:**
    This directory will hold the Lake project for your persistent Lean library. Create it and navigate into it from your project root (e.g., `vantage/`).
    ```bash
    # Use a consistent name (e.g., vantage_lib)
    # (If it already exists from a previous attempt, ensure it's empty first)
    mkdir vantage_lib
    cd vantage_lib
    ```

6.  **Initialize Shared Library with Lake:**
    Initialize a default Lake project within the new directory. On older Lake versions using `lakefile.toml`, this often configures both a library (e.g., `VantageLib`) and an executable (`Main.lean`), deriving names from the directory.
    ```bash
    # Initialize default Lake project in the current directory (e.g., vantage_lib/)
    lake init .
    ```

7.  **Configure Shared Library (`lakefile.toml`):**
    Manually edit the generated `lakefile.toml` to configure the project as *library-only*.
    * Open `lakefile.toml`.
    * Find the library definition section (e.g., `[[lean_lib]]`). Ensure the `name` attribute matches your desired module name (e.g., `name = "VantageLib"`). **Keep this section.** This name *must* match the `LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME` environment variable.
    * Find the executable definition section (e.g., `[[lean_exe]]`). **Delete this entire section** (usually 3 lines).
    * *(Optional but Recommended)* Find the `defaultTargets = [...]` line near the top and **delete it** to avoid potential build issues.
    * Save the changes to `lakefile.toml`.

8.  **Clean Up Shared Library Files:**
    Delete the unnecessary files generated by the default `lake init` associated with the executable or placeholders. Run these commands while still inside the shared library directory (e.g., `vantage_lib/`).
    ```bash
    # Remove executable entry point and potential library root file:
    rm -f Main.lean VantageLib.lean

    # Remove default placeholder inside the library source directory:
    # (Replace VantageLib if you configured a different library name in Step 7)
    rm -f VantageLib/Basic.lean

    # Optional: remove default readme:
    rm -f README.md
    ```

9.  **Optional: Build Check Shared Library:**
    Verify the Lake configuration by running a build command. This ensures the `lakefile.toml` and directory structure are valid. Run this while still inside the shared library directory (e.g., `vantage_lib/`).
    ```bash
    lake build
    ```
    *Note: If this fails with `error: package 'vantage_lib' has no target 'vantage_lib'`, double-check that you deleted the `defaultTargets = ["vantage_lib"]` line from your `lakefile.toml` in Step 7.* A successful command indicates the setup is correct so far.

10. **Finalize Setup:**
    Navigate back to the project root directory and ensure your environment variables are correctly set.
    ```bash
    # Navigate back to the project root (e.g., vantage/)
    cd ..
    ```
    * **CRITICAL:** Now, update your `.env` file (from Step 4) or export the environment variables:
        * Set `LEAN_AUTOMATOR_SHARED_LIB_PATH` to the **absolute path** of the shared library directory you created (e.g., the output of `pwd` when you were inside `vantage_lib`).
        * Confirm `LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME` matches the library name you configured in Step 7 (e.g., `VantageLib`).

## Configuration Summary

The application uses environment variables (preferably set via a `.env` file in the project root).

**Required:**

* `GEMINI_API_KEY`: Your Google AI (Gemini) API key.
* `DEFAULT_GEMINI_MODEL`: Default Gemini model name (e.g., `gemini-1.5-flash-latest`).
* `LEAN_AUTOMATOR_SHARED_LIB_PATH`: Path (absolute recommended, or relative to execution dir) to the persistent shared Lean library directory (e.g., `vantage_lib`). **Must match the directory created in setup.**

**Optional (with defaults):**

* `LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME`: The Lean module name for the shared library (used in `lake init`). **Default:** `ProvenKB`.
* `GEMINI_MODEL_COSTS`: JSON string for model costs per million tokens (see example above). **Default:** `{}`.
* `KB_DB_PATH`: Path for the SQLite database. **Default:** `knowledge_base.sqlite`.
* `GEMINI_MAX_RETRIES`: Max Gemini API retries. **Default:** `3`.
* `GEMINI_BACKOFF_FACTOR`: Retry delay factor. **Default:** `1.0`.
* `LEAN_STATEMENT_MAX_ATTEMPTS`: Max attempts for LLM statement generation. **Default:** `2`.
* `LEAN_PROOF_MAX_ATTEMPTS`: Max attempts for LLM proof generation. **Default:** `3`.
* `LEAN_AUTOMATOR_LAKE_CACHE`: Path for caching external Lake dependencies (if any are used later). **Default:** Not set (Lake uses user default). Example: `.lake_cache`.

## Usage

*(This section provides high-level examples. Integrate into your workflow.)*

1.  **Initialize the Database:**
    ```python
    from lean_automator.kb_storage import initialize_database
    # Uses KB_DB_PATH env var or default
    initialize_database()
    ```

2.  **Create and Save a Knowledge Base Item:**
    ```python
    from lean_automator.kb_storage import KBItem, ItemType, ItemStatus, save_kb_item
    import asyncio # save_kb_item is async

    async def add_item():
        new_def = KBItem(
            unique_name="MyDefs.MyType", # Use Lean-style naming
            item_type=ItemType.DEFINITION,
            lean_code="namespace MyDefs\ninductive MyType where\n| constructor1 : MyType\nend MyDefs",
            description_nl="A simple type definition.",
            topic="MyDefs",
            status=ItemStatus.PENDING_LEAN # Or LATEX_ACCEPTED if applicable
        )
        # Save it (client=None as we aren't generating embeddings here)
        saved_def = await save_kb_item(new_def, client=None)
        print(f"Saved item with ID: {saved_def.id}")

    asyncio.run(add_item())
    ```

3.  **Interact with the Gemini API:** (Example unchanged, still valid)
    ```python
    # ... (see previous README version) ...
    ```

4.  **Generate and Verify a Lean Item:**
    This function attempts to generate Lean code (if missing) using the LLM and then verifies it using `lean_interaction.check_and_compile_item`. On success, the item's status is set to `PROVEN` in the DB, and its source code is added and compiled within the persistent shared library (`vantage_lib`).
    ```python
    import asyncio
    from lean_automator import lean_processor, llm_call, kb_storage

    # Assume:
    # - DB is initialized
    # - Shared library ('vantage_lib') is initialized
    # - Environment variables (API key, shared lib path) are set
    # - A KBItem 'MyDefs.MyType' exists with status DEFINITION_ADDED / PROVEN
    # - A KBItem 'MyTheorems.UseMyType' exists with status PENDING_LEAN,
    #   latex_statement defined, and plan_dependencies=["MyDefs.MyType"]

    async def process_item():
        client = llm_call.GeminiClient() # Reads config from env
        item_name = "MyTheorems.UseMyType"

        success = await lean_processor.generate_and_verify_lean(
            unique_name=item_name,
            client=client
            # db_path=..., # Optional override
            # lake_executable_path=..., # Optional override
            # timeout_seconds=... # Optional override
        )

        if success:
            print(f"Processing successful for {item_name}.")
            # The item is now PROVEN in the DB and added to the shared library
        else:
            print(f"Processing failed for {item_name}. Check logs and DB for error details.")
            # The item is now LEAN_VALIDATION_FAILED or ERROR in the DB

    asyncio.run(process_item())
    ```

## Testing

The project includes unit and integration tests using `pytest`.

1.  **Install test dependencies:**
    ```bash
    pip install pytest pytest-asyncio
    ```

2.  **Run tests:**
    Navigate to the project root directory (where `pytest.ini` is located).

    * **Run all tests:**
        ```bash
        pytest
        ```
        * **Note:** Running all tests includes integration tests. These tests (`tests/integration`) interact with the actual database, the Gemini API, and the Lean toolchain. They are marked with the `integration` marker and might be slow or incur costs (for API calls). Ensure you have set the necessary environment variables (`GEMINI_API_KEY`, potentially `KB_DB_PATH` if not using the default) and that `lake` is installed and accessible in your PATH before running them.

    * **Run only unit tests (excluding integration tests):**
        The project uses markers defined in `pytest.ini` to categorize tests. To run faster checks without external dependencies, you can exclude integration tests:
        ```bash
        pytest -m "not integration"
        ```

    * **Run only integration tests:**
        If you specifically want to run only the integration tests:
        ```bash
        pytest -m integration
        ```

    * **Exclude slow tests:**
        Some tests might be marked as `slow`. To exclude these:
        ```bash
        pytest -m "not slow"
        ```

    You can combine markers as needed (e.g., `pytest -m "integration and not slow"`).

## Contributing

Contributions are welcome! Please follow standard practices like creating issues for bugs or feature requests and submitting pull requests for changes. (Add more specific guidelines if desired).

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for the full text.