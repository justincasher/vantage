# File: lean_automator/config/loader.py

"""Loads and manages application configuration from multiple sources.

This module provides functions to load configuration settings from YAML files
(defaults), JSON files (e.g., costs), and environment variables (overrides, secrets).
It automatically determines the paths for the default configuration files
(`config.yml`, `model_costs.json`) by:
1. Checking specific environment variables (`LEAN_AUTOMATOR_CONFIG_FILE`, `LEAN_AUTOMATOR_COSTS_FILE`).
2. Searching upwards from this file's location for a project root marker (`pyproject.toml`)
   and looking for the files in that root directory.
3. As a fallback, looking in the current working directory (with a warning).

It exposes the loaded configuration via a singleton dictionary `APP_CONFIG`
and provides helper functions to access specific sensitive values directly
from the environment.
"""

import json
import logging
import os
import pathlib # Added for path manipulation
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from dotenv import load_dotenv

# Initialize logging for this module
logger = logging.getLogger(__name__)
# Ensure logging is configured (e.g., by a root logger setup elsewhere or basicConfig)
# If run standalone or root logger isn't set, this provides a default.
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
    )

# Default filenames
DEFAULT_CONFIG_FILENAME = "config.yml"
DEFAULT_COSTS_FILENAME = "model_costs.json"
PROJECT_ROOT_MARKER = "pyproject.toml" # File to indicate project root

# Environment variables for specifying config file paths explicitly
ENV_CONFIG_PATH = "LEAN_AUTOMATOR_CONFIG_FILE"
ENV_COSTS_PATH = "LEAN_AUTOMATOR_COSTS_FILE"

# Defines mappings from environment variables to nested configuration dictionary keys
# for OVERRIDING specific values within the loaded config.
# Format: (ENV_VARIABLE_NAME, [list, of, config, keys], target_type_for_conversion)
ENV_OVERRIDES: List[Tuple[str, List[str], Type]] = [
    ("KB_DB_PATH", ["database", "kb_db_path"], str),
    ("DEFAULT_GEMINI_MODEL", ["llm", "default_gemini_model"], str),
    ("GEMINI_MAX_RETRIES", ["llm", "gemini_max_retries"], int),
    ("GEMINI_BACKOFF_FACTOR", ["llm", "gemini_backoff_factor"], float),
    ("DEFAULT_EMBEDDING_MODEL", ["embedding", "default_embedding_model"], str),
    ("LATEX_MAX_REVIEW_CYCLES", ["latex", "max_review_cycles"], int),
]


def _find_project_root(
    start_path: pathlib.Path, marker_filename: str = PROJECT_ROOT_MARKER
) -> Optional[pathlib.Path]:
    """Searches upward from start_path for a directory containing marker_filename.

    Args:
        start_path: The directory path to begin the search from.
        marker_filename: The filename to look for as the project root indicator.

    Returns:
        The Path object for the directory containing the marker file, or None if
        not found before reaching the filesystem root.
    """
    current_path = start_path.resolve()
    while True:
        if (current_path / marker_filename).is_file():
            logger.debug(f"Found project root marker '{marker_filename}' at '{current_path}'")
            return current_path
        parent_path = current_path.parent
        if parent_path == current_path:
            # Reached the filesystem root
            logger.debug(f"Project root marker '{marker_filename}' not found searching from '{start_path}'.")
            return None
        current_path = parent_path


def _update_nested_dict(d: Dict[str, Any], keys: List[str], value: Any):
    """Safely sets a value in a nested dictionary based on a list of keys.

    Creates intermediate dictionaries if they don't exist. Logs an error
    if a path conflict occurs (e.g., expecting a dict but finding a non-dict).

    Args:
        d: The dictionary to update.
        keys: A list of strings representing the path to the key.
        value: The value to set at the specified path.
    """
    node = d
    for i, key in enumerate(keys[:-1]):
        # Ensure the node is a dictionary or create one
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            logger.error(
                f"Config structure conflict: Expected dict at '{key}' "
                f"while setting path '{'.'.join(keys)}', but found type {type(child)}. "
                f"Cannot apply value '{value}'."
            )
            return # Stop processing this override
        node = child # Move deeper

    # Set the final value
    final_key = keys[-1]
    node[final_key] = value


def load_configuration(
    dotenv_path: Optional[str] = None,
    env_override_map: List[Tuple[str, List[str], Type]] = ENV_OVERRIDES,
) -> Dict[str, Any]:
    """Loads configuration layers: YAML defaults, JSON costs, and env overrides.

    Determines the paths for the default configuration file (`config.yml`)
    and costs file (`model_costs.json`) using the following priority:
    1. Environment variables: `LEAN_AUTOMATOR_CONFIG_FILE` and `LEAN_AUTOMATOR_COSTS_FILE`.
    2. Project Root Search: Looks for `pyproject.toml` upwards from this loader's
       directory and constructs paths relative to that root.
    3. Current Working Directory (Fallback): If neither of the above yields a path,
       it tries the CWD (logging a warning).

    After determining paths, it reads the base configuration from the YAML file,
    merges data from the JSON costs file into the 'costs' key, loads environment
    variables from a .env file (if specified or found), and finally applies
    overrides defined in `env_override_map` using values from environment variables.

    Args:
        dotenv_path: Explicit path to the .env file. If None, `python-dotenv`
                     searches standard locations.
        env_override_map: The mapping defining which environment variables
                          override which configuration keys and their types.

    Returns:
        A dictionary containing the fully merged configuration. Returns an
        empty dictionary if the determined base config file cannot be found or parsed.
        Cost data or overrides might be missing if respective files/variables are
        absent or invalid.
    """
    config: Dict[str, Any] = {}

    # --- Determine Configuration File Paths ---
    effective_config_path: Optional[pathlib.Path] = None
    effective_costs_path: Optional[pathlib.Path] = None

    # 1. Check Environment Variables
    env_config_path_str = os.getenv(ENV_CONFIG_PATH)
    env_costs_path_str = os.getenv(ENV_COSTS_PATH)

    if env_config_path_str:
        effective_config_path = pathlib.Path(env_config_path_str)
        logger.info(f"Using config path from environment variable {ENV_CONFIG_PATH}: '{effective_config_path}'")
    if env_costs_path_str:
        effective_costs_path = pathlib.Path(env_costs_path_str)
        logger.info(f"Using costs path from environment variable {ENV_COSTS_PATH}: '{effective_costs_path}'")

    # 2. Find Project Root (if paths not set by env vars)
    if effective_config_path is None or effective_costs_path is None:
        project_root = _find_project_root(start_path=pathlib.Path(__file__).parent)

        if project_root:
            logger.info(f"Determined project root: '{project_root}'")
            if effective_config_path is None:
                effective_config_path = project_root / DEFAULT_CONFIG_FILENAME
                logger.info(f"Derived config path from project root: '{effective_config_path}'")
            if effective_costs_path is None:
                effective_costs_path = project_root / DEFAULT_COSTS_FILENAME
                logger.info(f"Derived costs path from project root: '{effective_costs_path}'")
        else:
            logger.warning(
                f"Could not find project root marker '{PROJECT_ROOT_MARKER}'. "
                "Falling back to current working directory for config/costs paths."
            )
            # 3. Fallback to CWD (only if project root not found AND env var not set)
            cwd = pathlib.Path.cwd()
            if effective_config_path is None:
                effective_config_path = cwd / DEFAULT_CONFIG_FILENAME
                logger.info(f"Using fallback config path in CWD: '{effective_config_path}'")
            if effective_costs_path is None:
                effective_costs_path = cwd / DEFAULT_COSTS_FILENAME
                logger.info(f"Using fallback costs path in CWD: '{effective_costs_path}'")

    # Ensure paths are resolved before use (makes error messages clearer)
    if effective_config_path:
         effective_config_path = effective_config_path.resolve()
    if effective_costs_path:
         effective_costs_path = effective_costs_path.resolve()

    # --- Load Base YAML Configuration ---
    if effective_config_path:
        try:
            with open(effective_config_path, encoding="utf-8") as f:
                loaded_yaml = yaml.safe_load(f)
                config = loaded_yaml if isinstance(loaded_yaml, dict) else {} # Ensure it's a dict
            logger.info(f"Loaded base config from '{effective_config_path}'.")
        except FileNotFoundError:
            logger.warning(f"Base config file '{effective_config_path}' not found.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML '{effective_config_path}': {e}", exc_info=True)
            return {} # Critical error parsing base config, return empty
        except Exception as e:
             logger.error(f"Unexpected error loading '{effective_config_path}': {e}", exc_info=True)
             return {}
    else:
        logger.error("Could not determine a valid path for the base configuration file.")
        return {} # Cannot proceed without base config path

    # --- Load and Merge JSON Costs Data ---
    if effective_costs_path:
        try:
            with open(effective_costs_path, encoding="utf-8") as f:
                model_costs = json.load(f)
            if "costs" not in config or not isinstance(config.get("costs"), dict):
                config["costs"] = {} # Ensure 'costs' key exists and is a dict
            # Safely update, preferring loaded costs over potential existing values
            config["costs"].update(model_costs if isinstance(model_costs, dict) else {})
            logger.info(f"Loaded and merged costs from '{effective_costs_path}'.")
        except FileNotFoundError:
            logger.warning(f"Costs file '{effective_costs_path}' not found.")
            config.setdefault("costs", {}) # Ensure 'costs' key exists even if file not found
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON costs '{effective_costs_path}': {e}", exc_info=True)
            config.setdefault("costs", {}) # Ensure 'costs' key exists after error
        except Exception as e:
            logger.error(f"Unexpected error loading '{effective_costs_path}': {e}", exc_info=True)
            config.setdefault("costs", {})
    else:
        logger.warning("Could not determine a valid path for the costs file. 'costs' section may be empty.")
        config.setdefault("costs", {}) # Ensure 'costs' key exists

    # --- Load .env file into environment variables ---
    try:
        # Pass override=True if you want .env to override existing env vars
        loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
        if loaded_env:
            logger.info(".env file loaded into environment variables (overriding existing).")
        else:
             logger.debug(".env file not found or empty.")
    except Exception as e:
        # Catch potential errors during dotenv loading (e.g., permission issues)
        logger.error(f"Error loading .env file: {e}", exc_info=True)

    # --- Apply Environment Variable Overrides for config VALUES ---
    logger.info("Checking for environment variable overrides for config values...")
    override_count = 0
    for env_var, config_keys, target_type in env_override_map:
        env_value_str = os.getenv(env_var)
        if env_value_str is not None:
            try:
                # Attempt conversion
                typed_value = target_type(env_value_str)
                # Update the nested dictionary
                _update_nested_dict(config, config_keys, typed_value)
                logger.info(
                    f"Applied value override: '{'.'.join(config_keys)}' = '{typed_value}' "
                    f"(from env '{env_var}')"
                )
                override_count += 1
            except ValueError:
                logger.warning(
                    f"Value override failed: Cannot convert env var '{env_var}' "
                    f"value '{env_value_str}' to target type {target_type.__name__}."
                )
            except Exception as e:
                logger.error(
                    f"Value override error: Unexpected issue applying env var '{env_var}': {e}",
                    exc_info=True,
                )
    if override_count > 0:
        logger.info(f"Applied {override_count} environment variable value override(s).")
    else:
        logger.info("No environment variable value overrides applied.")

    return config


# --- Singleton Configuration Instance ---
# Load the configuration once when this module is first imported.
# The load_configuration function now handles finding the paths internally.
APP_CONFIG: Dict[str, Any] = load_configuration()


# --- Direct Accessors for Sensitive/Specific Environment Variables ---
# These remain unchanged as they directly access environment variables.

def get_gemini_api_key() -> Optional[str]:
    """Retrieves the Gemini API Key directly from environment variables.

    Returns:
        The Gemini API key string if the 'GEMINI_API_KEY' environment
        variable is set, otherwise None.
    """
    return os.getenv("GEMINI_API_KEY")


def get_lean_automator_shared_lib_path() -> Optional[str]:
    """Retrieves the Lean Automator Shared Lib Path from environment variables.

    Returns:
        The path string if the 'LEAN_AUTOMATOR_SHARED_LIB_PATH' environment
        variable is set, otherwise None.
    """
    return os.getenv("LEAN_AUTOMATOR_SHARED_LIB_PATH")


# --- Example Usage / Standalone Test ---
if __name__ == "__main__":
    # This block executes only when the script is run directly
    print("\n--- Running Config Loader Standalone Test ---")

    # Use a separate logger config for standalone test if needed
    # Clear existing handlers to avoid duplicate messages if run multiple times
    # root_logger = logging.getLogger()
    # if root_logger.hasHandlers():
    #     root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
    )
    logger.setLevel(logging.INFO) # Ensure this module's logger level is also appropriate

    print("\n--- Determining Config Paths ---")
    # Re-run part of the logic to show how paths are determined in standalone mode
    test_config_path: Optional[pathlib.Path] = None
    test_costs_path: Optional[pathlib.Path] = None
    env_cfg = os.getenv(ENV_CONFIG_PATH)
    env_cst = os.getenv(ENV_COSTS_PATH)
    if env_cfg: test_config_path = pathlib.Path(env_cfg)
    if env_cst: test_costs_path = pathlib.Path(env_cst)

    if not test_config_path or not test_costs_path:
        test_root = _find_project_root(pathlib.Path(__file__).parent)
        if test_root:
             if not test_config_path: test_config_path = test_root / DEFAULT_CONFIG_FILENAME
             if not test_costs_path: test_costs_path = test_root / DEFAULT_COSTS_FILENAME
        else:
            test_cwd = pathlib.Path.cwd()
            if not test_config_path: test_config_path = test_cwd / DEFAULT_CONFIG_FILENAME
            if not test_costs_path: test_costs_path = test_cwd / DEFAULT_COSTS_FILENAME

    print(f"Config Path Used: {test_config_path.resolve() if test_config_path else 'Not Determined'}")
    print(f"Costs Path Used: {test_costs_path.resolve() if test_costs_path else 'Not Determined'}")


    print("\n--- Loaded Configuration (APP_CONFIG) ---")
    import pprint

    # Ensure APP_CONFIG is reloaded for the test if env vars might affect it
    # Note: In normal use, APP_CONFIG is loaded only once at import.
    # For standalone test, we might want to reload based on current env:
    # test_app_config = load_configuration()
    # pprint.pprint(test_app_config)
    # Or just print the already loaded one:
    pprint.pprint(APP_CONFIG)


    print("\n--- Accessing Values Example ---")
    # Access from the already loaded APP_CONFIG
    db_path = APP_CONFIG.get("database", {}).get("kb_db_path", "N/A")
    llm_model = APP_CONFIG.get("llm", {}).get("default_gemini_model", "N/A")
    print(f"Database Path: {db_path}")
    print(f"LLM Model: {llm_model}")

    # Safely access potentially missing cost info
    cost_info = APP_CONFIG.get("costs", {}).get("gemini-1.5-pro-preview-0409", {})
    print(f"Costs for gemini-1.5-pro-preview-0409: {cost_info if cost_info else 'N/A'}")


    print("\n--- Accessing Secrets/Paths Directly Example ---")
    api_key = get_gemini_api_key()
    # Avoid printing keys directly in logs or output
    print(f"Gemini API Key Set: {bool(api_key)}")
    shared_path = get_lean_automator_shared_lib_path()
    print(f"Lean Lib Path: {shared_path if shared_path else 'N/A'}")

    print("\n--- Standalone Test Complete ---")