# Configuration

The application uses environment variables, typically loaded from a `.env` file in the project root directory using a library like `python-dotenv`.

## Required Variables

These variables must be set for the application to function correctly.

* **`GEMINI_API_KEY`**
    * **Description:** Your API key for Google AI (Gemini). Obtain one from [Google AI Studio](https://aistudio.google.com/).
    * **Example Value:** `YOUR_API_KEY_HERE` *(Do not commit your actual key to version control)*

* **`DEFAULT_GEMINI_MODEL`**
    * **Description:** The default Google Gemini model name to use for generation tasks.
    * **Example Value:** `gemini-1.5-flash-latest` or `gemini-2.0-flash`

* **`LEAN_AUTOMATOR_SHARED_LIB_PATH`**
    * **Description:** The path (absolute path recommended) to the directory containing the persistent shared Lean library project (e.g., `vantage_lib`). This *must* match the directory created and configured during the [Installation & Setup](../getting-started/index.md).
    * **Example Value:** `/path/to/your/project/vantage_lib`

## Optional Variables

These variables have default values or are only needed for specific features. You can override them in your `.env` file.

* **`LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME`**
    * **Description:** The Lean module name of the shared library, as defined in its `lakefile.toml` (e.g., `name = "VantageLib"`). Should match the configuration in the setup guide.
    * **Default:** `VantageLib` *(Note: This should match the `name` configured in the shared library's `lakefile.toml` during setup)*

* **`DEFAULT_EMBEDDING_MODEL`**
    * **Description:** The model used for generating vector embeddings for knowledge base items (used for similarity search). Required if using embedding features.
    * **Example Value:** `text-embedding-004`

* **`KB_DB_PATH`**
    * **Description:** Path and filename for the SQLite database storing the knowledge base.
    * **Default:** `knowledge_base.sqlite`

* **`GEMINI_MODEL_COSTS`**
    * **Description:** JSON string defining the estimated cost per million units (e.g., tokens) for input and output for specific models. Used for cost tracking. Verify units and costs on the model provider's pricing page.
    * **Default:** `{}`
    * **Example Format:** `'{"model-name": {"input": cost_input, "output": cost_output}}'`

* **`GEMINI_MAX_RETRIES`**
    * **Description:** Maximum number of times to retry a failed API call to the Gemini model.
    * **Default:** `3`

* **`GEMINI_BACKOFF_FACTOR`**
    * **Description:** Factor determining the delay between retries (exponential backoff). A factor of 1.0 means delays of 1s, 2s, 4s, etc.
    * **Default:** `1.0`

* **`LEAN_STATEMENT_MAX_ATTEMPTS`**
    * **Description:** Maximum attempts the system will make to generate a valid Lean statement using the LLM (if applicable to the workflow).
    * **Default:** `2`

* **`LEAN_PROOF_MAX_ATTEMPTS`**
    * **Description:** Maximum attempts the system will make to generate a valid Lean proof using the LLM (if applicable to the workflow).
    * **Default:** `3`

* **`LATEX_MAX_REVIEW_CYCLES`**
    * **Description:** Maximum number of review cycles involving the LLM for generating or validating LaTeX output (if applicable to the workflow).
    * **Example Value:** `3` *(No hard default specified, depends on implementation)*

* **`LEAN_AUTOMATOR_LAKE_CACHE`**
    * **Description:** Optional path to a directory for caching external Lake dependencies, potentially speeding up builds if the shared library uses external Lean libraries.
    * **Default:** Not set (Lake uses its own default cache location).
    * **Example Value:** `.lake_cache`

## Example `.env` File

Here is an example `.env` file structure incorporating some common settings and overrides, based on the variables described above. **Remember to replace placeholders with your actual values and keep sensitive information private.**

```dotenv
# Environment variables for the Lean Automator project

# --- Database ---
# KB_DB_PATH=knowledge_base.sqlite # Using default

# --- LLM Configuration (Google Gemini) ---
# IMPORTANT: Keep your API key secure! Do not commit it.
GEMINI_API_KEY=YOUR_API_KEY_HERE
DEFAULT_GEMINI_MODEL=gemini-2.0-flash

# --- Vector Embedding Configuration ---
# Required if using embedding/search features
DEFAULT_EMBEDDING_MODEL=text-embedding-004

# --- Lean/Lake Configuration ---
# Use the absolute path to your shared library directory
LEAN_AUTOMATOR_SHARED_LIB_PATH=/path/to/your/project/vantage_lib
# Ensure this matches the 'name' in vantage_lib/lakefile.toml
LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME=VantageLib

# --- Cost Tracking ---
# Format: '{"model-name": {"input": cost_per_million_input_units, "output": cost_per_million_output_units}}'
# Costs are examples, verify current pricing. Units might be tokens/chars.
GEMINI_MODEL_COSTS='{"gemini-2.0-flash": {"input": 0.1, "output": 0.4}, "models/text-embedding-004": {"input": 0.0, "output": 0.0}}'

# --- LLM Retry/Backoff Overrides (Optional) ---
GEMINI_MAX_RETRIES=5
GEMINI_BACKOFF_FACTOR=1.5

# --- LaTeX Generation (Optional) ---
LATEX_MAX_REVIEW_CYCLES=3

# --- Other Optional Settings ---
# LEAN_STATEMENT_MAX_ATTEMPTS=2 # Using default
# LEAN_PROOF_MAX_ATTEMPTS=3 # Using default
# LEAN_AUTOMATOR_LAKE_CACHE=.lake_cache # Using default Lake cache
```