# File: lean_automator/lean/proof_repair.py

"""Attempts heuristic automated repairs for common Lean compilation errors.

This module provides functionality to analyze Lean compilation error logs and
attempt simple, pattern-based repairs on the corresponding Lean code.
The goal is to automatically fix common, easily identifiable issues that might
arise from LLM code generation, potentially reducing the number of LLM retry
cycles needed.

Note: Currently, specific repair handlers (like for 'no goals to be solved')
are disabled, and the main function `attempt_proof_repair` will always return
indicating no fix was applied. Future handlers for other error patterns can be
added here.
"""

import re
import logging
from typing import Tuple, Set
import warnings

try:
    from lean_automator.config.loader import APP_CONFIG 
except ImportError:
    warnings.warn("lean_automator.config.loader.APP_CONFIG not found. Default settings may be used.", ImportWarning)
    APP_CONFIG = {} # Provide an empty dict as a fallback

logger = logging.getLogger(__name__)

# Regex to find Lean error lines reporting "no goals to be solved" and capture line number
# Format: error: <path>:<line>:<col>: no goals to be solved
NO_GOALS_ERROR_LINE_REGEX = re.compile(r"^error:.*?\.lean:(\d+):\d+:\s*no goals to be solved", re.MULTILINE)

# Regex to find Lean errors originating from a source file (path:line:col: message)
LEAN_SOURCE_ERROR_REGEX = re.compile(r"^error:.*?\.lean:\d+:\d+:.*$", re.MULTILINE)

# --- Helper functions ---

def _is_only_no_goals_error(error_log: str) -> bool:
    """Checks if the only Lean source errors are 'no goals to be solved'. (Internal Helper)

    Parses the error log, finds all lines matching the pattern of errors reported
    from a `.lean` file, and checks if *all* of those specific lines also match
    the 'no goals to be solved' error message. Ignores other lines starting with
    'error:' that don't fit the Lean source error format.

    Args:
        error_log (str): The captured stderr/stdout from the Lean compilation process.

    Returns:
        bool: True if the only errors reported from `.lean` files match the
        'no goals to be solved' pattern, False otherwise (including if no Lean
        source errors are found or other types of errors are present).
    """
    if not error_log:
        return False

    # Find all errors that look like they originate from a Lean source file
    lean_source_errors = LEAN_SOURCE_ERROR_REGEX.findall(error_log)
    num_lean_source_errors = len(lean_source_errors)

    if num_lean_source_errors == 0:
        logger.debug("No lines matching Lean source error pattern found in log for 'no goals' check.")
        return False # No relevant errors found

    # Count how many of the errors match the specific "no goals" pattern
    num_no_goals_errors = len(NO_GOALS_ERROR_LINE_REGEX.findall(error_log))

    # Check if all source file errors are specifically 'no goals' errors
    if num_lean_source_errors > 0 and num_lean_source_errors == num_no_goals_errors:
        logger.debug(f"Confirmed all {num_lean_source_errors} Lean source errors match 'no goals to be solved' pattern.")
        return True
    else:
        logger.debug(f"Not exclusively 'no goals' errors: Found {num_lean_source_errors} Lean source errors, but only {num_no_goals_errors} match 'no goals' pattern. Skipping fix.")
        return False


def _fix_no_goals_error(lean_code: str, error_log: str) -> str:
    """Replaces lines causing 'no goals' errors with the 'done' tactic. (Internal Helper - Currently Unused)

    Parses the error log to find line numbers associated with 'no goals to be solved'
    errors. It then replaces the content of those lines in the `lean_code` with the
    `done` tactic, preserving indentation.

    Note:
        This function is currently not called by `attempt_proof_repair` as this
        specific fix was found to be unreliable.

    Args:
        lean_code (str): The Lean code string.
        error_log (str): The error log containing 'no goals to be solved' errors.

    Returns:
        str: The modified Lean code with problematic lines replaced by 'done',
        or the original `lean_code` if no line numbers could be parsed or
        if other issues occur.
    """
    lines_to_replace: Set[int] = set()
    # Extract line numbers from error messages
    for match in NO_GOALS_ERROR_LINE_REGEX.finditer(error_log):
        try:
            line_num = int(match.group(1))
            if line_num > 0: # Line numbers are 1-based
                lines_to_replace.add(line_num)
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse line number from 'no goals' error match: {match.group(0)}")
            continue # Skip if line number cannot be parsed

    if not lines_to_replace:
        logger.warning("Identified 'no goals' error pattern, but failed to extract line numbers for replacement.")
        return lean_code # Return original code if no lines identified

    logger.info(f"Attempting automated fix: Replace Lean code lines {sorted(list(lines_to_replace))} with 'done'.")

    code_lines = lean_code.splitlines()
    modified_lines = []
    replaced_count = 0
    # Iterate through original code lines and replace targeted lines
    for i, line in enumerate(code_lines):
        current_line_num = i + 1 # Convert 0-based index to 1-based line number
        if current_line_num in lines_to_replace:
            # Preserve original indentation
            indentation_match = re.match(r"^(\s*)", line)
            indentation = indentation_match.group(1) if indentation_match else ""
            replacement_line = indentation + "done" # Replace with 'done'
            modified_lines.append(replacement_line)
            replaced_count += 1
            logger.debug(f"Replacing line {current_line_num} ('no goals' error) with '{replacement_line}'. Original: '{line.strip()}'")
        else:
            # Keep non-targeted lines as they are
            modified_lines.append(line)

    # Sanity check: Log if the number of replacements doesn't match expectations
    if replaced_count != len(lines_to_replace):
        logger.warning(f"Expected to replace {len(lines_to_replace)} lines for 'no goals' error, but replaced {replaced_count}. Line numbers might be incorrect or out of bounds.")

    return "\n".join(modified_lines) # Return the potentially modified code


# --- Public Function ---

def attempt_proof_repair(lean_code: str, error_log: str) -> Tuple[bool, str]:
    """Attempts to automatically repair simple, known Lean compilation errors.

    Analyzes the `error_log` for specific, fixable error patterns. If a known
    pattern is detected by an *enabled* handler, this function modifies the
    `lean_code` to attempt a fix.

    Currently, all specific repair handlers are disabled. Therefore, this function
    will always return `(False, original_lean_code)`.

    Args:
        lean_code (str): The Lean code string that failed compilation.
        error_log (str): The stderr/stdout captured from the failed `lake build`
            or `lean` command.

    Returns:
        Tuple[bool, str]: A tuple containing:

            - bool: `True` if a fix was attempted and resulted in modified code,
              `False` otherwise (including if no matching error pattern was found,
              if the relevant handler is disabled, or if an error occurred during
              the fix attempt). Currently always `False`.
              
            - str: The potentially modified Lean code string if a fix was applied,
              otherwise the original `lean_code`. Currently always the original
              `lean_code`.
    """
    logger.debug("Attempting automated proof repair (Note: Specific handlers currently disabled)...")
    original_code = lean_code # Store original code for return if no fix applied

    # Basic validation
    if not lean_code or not error_log:
        logger.debug("Skipping repair: No code or error log provided.")
        return False, original_code # Cannot repair without input

    # --- Handler 1: Only "no goals to be solved" errors (CURRENTLY DISABLED) ---
    # This handler remains disabled as the simple fix proved insufficient.
    # The logic is kept for reference or future reactivation.
    # ```python
    # if _is_only_no_goals_error(error_log):
    #     logger.info("Detected 'no goals to be solved' pattern (Handler Disabled).")
    #     # FIX DISABLED: return False, original_code
    #     # Attempt fix if handler were enabled:
    #     # try:
    #     #     modified_code = _fix_no_goals_error(lean_code, error_log)
    #     #     if modified_code != original_code and modified_code.strip():
    #     #         logger.info("Applied hypothetical fix for 'no goals to be solved'.")
    #     #         return True, modified_code
    #     #     else:
    #     #          logger.warning("Hypothetical 'no goals' fix resulted in no change or empty code.")
    #     #          return False, original_code
    #     # except Exception as e:
    #     #      logger.exception(f"Error during disabled 'no goals' fix execution: {e}")
    #     #      return False, original_code
    # ```
    # --- End Disabled Handler ---


    # --- Placeholder for Future Handlers ---
    # Add `elif` blocks here to check for other error patterns and call
    # corresponding `_fix_...` functions when handlers are developed and enabled.
    # Example:
    # elif _is_some_other_fixable_pattern(error_log):
    #     logger.info("Detected other fixable pattern...")
    #     # ... call fixer function ...
    #     # return True, modified_code


    # --- Default Case: No Enabled Handler Matched ---
    # If execution reaches here, it means no enabled handler recognized the error pattern.
    logger.debug("No enabled/matching fixable error pattern detected in error log.")
    return False, original_code # Return False and the original code