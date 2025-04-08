# Configuration

This application utilizes a layered configuration system to manage settings, separating defaults, secrets, and local overrides.

## Configuration Layers

1.  **Default Configuration (`config.yml`):** Contains standard application settings and defaults. This file is version-controlled.
2.  **Model Cost Data (`model_costs.json`):** Stores structured data for API cost estimation. This file is version-controlled.
3.  **Environment Variables (`.env` file):** Used for secrets (like API keys), user-specific paths, and overriding default settings. This file is loaded into the environment, typically using a local `.env` file in the project root. **The `.env` file should *not* be committed to version control.**

Settings are loaded in the order above, with environment variables taking precedence and overriding values from `config.yml` or `model_costs.json`.

## The `.env` File and `.env.example`

You will need to create a `.env` file in the project root directory to provide required secrets and paths, and optionally override default configurations.

The `.env.example` file in the repository serves as a **template and provides examples** of commonly used environment variables. It shows the required format but may not list every single configuration key that *could* potentially be overridden.

**[Link to `.env.example` in repository - *to be added after merge*]**

Copy `.env.example` to `.env` and populate it with your specific values.

## Environment Variable Details

These are the key environment variables recognized by the application, typically set in your `.env` file.

### Required Variables

These **must** be defined in your `.env` file for the application to function.

* **`GEMINI_API_KEY`**
    * **Description:** Your API key for Google AI (Gemini).
    * **How to Obtain:** Get one from [Google AI Studio](https://aistudio.google.com/).
    * **Example Value:** `AIzaSy...` *(Keep your actual key secure and private)*

* **`LEAN_AUTOMATOR_SHARED_LIB_PATH`**
    * **Description:** The **absolute path** to the directory containing the persistent shared Lean library project (e.g., `vantage_lib`). This path is specific to your local machine.
    * **How to Obtain:** This directory is created during the [Installation & Setup](../getting-started/index.md). Use the full path identified in Step 12 of the setup guide.
    * **Example Value:** `/Users/yourname/projects/vantage/vantage_lib`

### Optional Variables (Overrides)

These variables can be added to your `.env` file to override the default settings loaded from `config.yml` or `model_costs.json`.

* **`DEFAULT_GEMINI_MODEL`**
    * **Description:** Overrides the default Google Gemini model name used for generation tasks.
    * **Default From:** `config.yml` (`llm.default_gemini_model`)
    * **Example Value:** `gemini-1.5-pro-latest`

* **`DEFAULT_EMBEDDING_MODEL`**
    * **Description:** Overrides the default model used for generating vector embeddings.
    * **Default From:** `config.yml` (`embedding.default_embedding_model`)
    * **Example Value:** `text-embedding-preview-0409`

* **`KB_DB_PATH`**
    * **Description:** Overrides the default path and filename for the SQLite knowledge base database.
    * **Default From:** `config.yml` (`database.kb_db_path`)
    * **Example Value:** `./data/my_project_kb.sqlite`

* **`LEAN_AUTOMATOR_SHARED_LIB_MODULE_NAME`**
    * **Description:** Specifies the Lean module name of the shared library, as defined in its `lakefile.toml` (e.g., `name = "MyLeanLib"`). Only needed if you deviate from the name used during setup.
    * **Default Logic:** The application might derive a default or expect a common name (e.g., based on the directory name or a hardcoded default like `VantageLib` if not set - check `config_loader.py` or setup guide recommendations).
    * **Example Value:** `MyCustomLeanLib`

* **`GEMINI_MODEL_COSTS`**
    * **Description:** Overrides the default model cost data loaded from `model_costs.json`. Provide a JSON string defining estimated costs per million units (verify units/costs with Google's pricing).
    * **Default From:** `model_costs.json`
    * **Example Format:** `'{"gemini-1.5-flash-latest": {"input": 0.35, "output": 0.70}, "models/text-embedding-004": {"input": 0.02, "output": 0.02}}'`

* **`GEMINI_MAX_RETRIES`**
    * **Description:** Overrides the maximum number of times to retry a failed API call to the Gemini model.
    * **Default From:** `config.yml` (`llm.gemini_max_retries`)
    * **Example Value:** `5`

* **`GEMINI_BACKOFF_FACTOR`**
    * **Description:** Overrides the factor determining the delay between retries (exponential backoff).
    * **Default From:** `config.yml` (`llm.gemini_backoff_factor`)
    * **Example Value:** `2.0`

* **`LATEX_MAX_REVIEW_CYCLES`**
    * **Description:** Overrides the maximum number of LLM review cycles for LaTeX generation/validation.
    * **Default From:** `config.yml` (`latex.max_review_cycles`)
    * **Example Value:** `2`

* **`LEAN_STATEMENT_MAX_ATTEMPTS`**
    * **Description:** Overrides the maximum attempts to generate a valid Lean statement using the LLM.
    * **Default Logic:** Likely defined within the `lean_processor` module or potentially added to `config.yml`. Check implementation for default behavior.
    * **Example Value:** `3`

* **`LEAN_PROOF_MAX_ATTEMPTS`**
    * **Description:** Overrides the maximum attempts to generate a valid Lean proof using the LLM.
    * **Default Logic:** Likely defined within the `lean_processor` module or potentially added to `config.yml`. Check implementation for default behavior.
    * **Example Value:** `4`

* **`LEAN_AUTOMATOR_LAKE_CACHE`**
    * **Description:** Specifies a path for caching external Lake dependencies. Can speed up builds if the shared library uses external Lean libraries.
    * **Default Logic:** If not set, Lake uses its own default cache location.
    * **Example Value:** `/path/to/shared/.lake_cache`

---

Always ensure your `.env` file contains the required variables and any overrides you need for your specific setup. Remember to keep secrets secure and avoid committing your `.env` file.
