# File: lean_automator/config/loader.py

"""Loads and manages application configuration from multiple sources.

This module provides functions to load configuration settings from YAML files
(defaults), JSON files (e.g., costs), and environment variables (overrides, secrets).
It exposes the loaded configuration via a singleton dictionary `APP_CONFIG`
and provides helper functions to access specific sensitive values directly
from the environment.
"""

import json
import logging
import os
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


# Defines mappings from environment variables to nested configuration dictionary keys.
# Format: (ENV_VARIABLE_NAME, [list, of, config, keys], target_type_for_conversion)
ENV_OVERRIDES: List[Tuple[str, List[str], Type]] = [
    ("KB_DB_PATH", ["database", "kb_db_path"], str),
    ("DEFAULT_GEMINI_MODEL", ["llm", "default_gemini_model"], str),
    ("GEMINI_MAX_RETRIES", ["llm", "gemini_max_retries"], int),
    ("GEMINI_BACKOFF_FACTOR", ["llm", "gemini_backoff_factor"], float),
    ("DEFAULT_EMBEDDING_MODEL", ["embedding", "default_embedding_model"], str),
    ("LATEX_MAX_REVIEW_CYCLES", ["latex", "max_review_cycles"], int),
]


def _update_nested_dict(d: Dict[str, Any], keys: List[str], value: Any):
    """Safely sets a value in a nested dictionary based on a list of keys.

    Creates intermediate dictionaries if they don't exist. Logs an error
    if a path conflict occurs (e.g., expecting a dict but finding a non-dict).

    Args:
        d: The dictionary to update.
        keys: A list of strings representing the path to the key.
        value: The value to set at the specified path.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
        if not isinstance(d, dict):
            logger.error(
                f"Config structure conflict: Expected dict at '{key}' "
                f"while setting '{'.'.join(keys)}', found {type(d)}. "
                f"Cannot apply value."
            )
            return
    d[keys[-1]] = value


def load_configuration(
    config_path: str = "config.yml",
    costs_path: str = "model_costs.json",
    dotenv_path: Optional[str] = None,
    env_override_map: List[Tuple[str, List[str], Type]] = ENV_OVERRIDES,
) -> Dict[str, Any]:
    """Loads configuration layers: YAML defaults, JSON costs, and env overrides.

    Reads the base configuration from a YAML file, merges data from a JSON
    file (typically model costs) into the 'costs' key, loads environment
    variables from a .env file (if present), and then applies overrides
    defined in `env_override_map` using values from environment variables.

    Args:
        config_path: Path to the YAML configuration file containing defaults.
        costs_path: Path to the JSON file containing model costs data.
        dotenv_path: Explicit path to the .env file. If None, `python-dotenv`
                     searches standard locations.
        env_override_map: The mapping defining which environment variables
                          override which configuration keys and their types.

    Returns:
        A dictionary containing the fully merged configuration. Returns an
        empty dictionary if base config file cannot be parsed. Cost data
        or overrides might be missing if respective files/variables are
        absent or invalid.
    """
    config: Dict[str, Any] = {}

    # Load Base YAML Configuration
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        logger.info(f"Loaded base config from '{config_path}'.")
    except FileNotFoundError:
        logger.warning(f"Base config file '{config_path}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML '{config_path}': {e}", exc_info=True)
        # Decide on behavior: return empty, raise error? Returning empty for robustness.
        return {}  # Critical error parsing base config, return empty

    # Load and Merge JSON Costs Data
    try:
        with open(costs_path, encoding="utf-8") as f:
            model_costs = json.load(f)
        if "costs" not in config or not isinstance(config.get("costs"), dict):
            config["costs"] = {}
        config["costs"].update(model_costs)
        logger.info(f"Loaded and merged costs from '{costs_path}'.")
    except FileNotFoundError:
        logger.warning(f"Costs file '{costs_path}' not found.")
        config.setdefault("costs", {})
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON costs '{costs_path}': {e}", exc_info=True)
        config.setdefault("costs", {})

    # Load .env file into environment variables
    try:
        loaded_env = load_dotenv(dotenv_path=dotenv_path, verbose=True)
        if loaded_env:
            logger.info(".env file loaded into environment variables.")
    except Exception as e:
        # Catch potential errors during dotenv loading (e.g., permission issues)
        logger.error(f"Error loading .env file: {e}", exc_info=True)

    # Apply Environment Variable Overrides
    logger.info("Checking for environment variable overrides...")
    override_count = 0
    for env_var, config_keys, target_type in env_override_map:
        env_value_str = os.getenv(env_var)
        if env_value_str is not None:
            try:
                typed_value = target_type(env_value_str)
                _update_nested_dict(config, config_keys, typed_value)
                logger.info(
                    f"Applied override: '{'.'.join(config_keys)}' "
                    f"(from env '{env_var}')"
                )
                override_count += 1
            except ValueError:
                logger.warning(
                    f"Override failed: Cannot convert env var '{env_var}' "
                    f"value '{env_value_str}' to {target_type.__name__}."
                )
            except Exception as e:
                logger.error(
                    f"Override error: Unexpected issue applying env var '{env_var}': "
                    f"{e}",
                    exc_info=True,
                )
    if override_count > 0:
        logger.info(f"Applied {override_count} environment variable override(s).")
    else:
        logger.info("No environment variable overrides applied.")

    return config


# --- Singleton Configuration Instance ---
# Load the configuration once when this module is first imported.
APP_CONFIG: Dict[str, Any] = load_configuration()


# --- Direct Accessors for Sensitive/Specific Environment Variables ---


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
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
    )

    print("\n--- Loaded Configuration (APP_CONFIG) ---")
    import pprint

    pprint.pprint(APP_CONFIG)

    print("\n--- Accessing Values Example ---")
    db_path = APP_CONFIG.get("database", {}).get("kb_db_path", "N/A")
    llm_model = APP_CONFIG.get("llm", {}).get("default_gemini_model", "N/A")
    print(f"Database Path: {db_path}")
    print(f"LLM Model: {llm_model}")
    cost_info = APP_CONFIG.get("costs", {}).get("gemini-2.0-flash", {})
    print(f"Costs for gemini-2.0-flash: {cost_info if cost_info else 'N/A'}")

    print("\n--- Accessing Secrets/Paths Directly Example ---")
    api_key = get_gemini_api_key()
    print(f"Gemini API Key Set: {bool(api_key)}")  # Avoid printing keys
    shared_path = get_lean_automator_shared_lib_path()
    print(f"Lean Lib Path: {shared_path if shared_path else 'N/A'}")

    print("\n--- Standalone Test Complete ---")
