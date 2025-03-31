# File: src/lean_automator/lean_proof_repair.py

import re
import logging
from typing import Tuple, Set

logger = logging.getLogger(__name__)

# Regex to find Lean error lines reporting "no goals to be solved" and capture line number
# Format: error: <path>:<line>:<col>: no goals to be solved
NO_GOALS_ERROR_LINE_REGEX = re.compile(r"^error:.*?\.lean:(\d+):\d+:\s*no goals to be solved", re.MULTILINE)

# Regex to find Lean errors originating from a source file (path:line:col: message)
LEAN_SOURCE_ERROR_REGEX = re.compile(r"^error:.*?\.lean:\d+:\d+:.*$", re.MULTILINE)

# --- Helper functions - can be kept or commented out if no longer used ---

def _is_only_no_goals_error(error_log: str) -> bool:
    """
    Checks if the error log contains Lean source errors, and if ALL such errors
    match the 'no goals to be solved' pattern, ignoring other non-Lean-source
    lines starting with 'error:'.
    (This check might still be useful for logging/analysis even if the fix is disabled)
    """
    if not error_log:
        return False

    # Find all errors that look like they originate from a Lean source file
    lean_source_errors = LEAN_SOURCE_ERROR_REGEX.findall(error_log)
    num_lean_source_errors = len(lean_source_errors)

    if num_lean_source_errors == 0:
        logger.debug("No lines matching Lean source error pattern found in log.")
        return False

    # Count how many of the errors match the specific "no goals" pattern
    num_no_goals_errors = len(NO_GOALS_ERROR_LINE_REGEX.findall(error_log))

    if num_lean_source_errors > 0 and num_lean_source_errors == num_no_goals_errors:
        logger.debug(f"Confirmed all {num_lean_source_errors} Lean source errors match 'no goals to be solved' pattern.")
        return True
    else:
        logger.debug(f"Found {num_lean_source_errors} Lean source errors, but only {num_no_goals_errors} 'no goals' errors. Skipping fix.")
        return False


def _fix_no_goals_error(lean_code: str, error_log: str) -> str:
    """
    Replaces lines identified by 'no goals to be solved' errors with the 'done' tactic.
    (This function is currently not called by the disabled handler below)
    """
    lines_to_replace: Set[int] = set()
    original_lines_content = {} # Store original line for logging/debugging if needed
    for match in NO_GOALS_ERROR_LINE_REGEX.finditer(error_log):
        try:
            line_num = int(match.group(1))
            if line_num > 0:
                lines_to_replace.add(line_num)
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse line number from 'no goals' error match: {match.group(0)}")
            continue

    if not lines_to_replace:
        logger.warning("Identified 'no goals' error pattern, but failed to extract line numbers for replacement.")
        return lean_code

    logger.info(f"Attempting to replace Lean code lines with 'done': {sorted(list(lines_to_replace))}")

    code_lines = lean_code.splitlines()
    modified_lines = []
    replaced_count = 0
    for i, line in enumerate(code_lines):
        current_line_num = i + 1
        if current_line_num in lines_to_replace:
            original_lines_content[current_line_num] = line
            indentation = re.match(r"^(\s*)", line).group(1) if re.match(r"^(\s*)", line) else ""
            replacement_line = indentation + "done"
            modified_lines.append(replacement_line)
            replaced_count += 1
            logger.debug(f"Replacing line {current_line_num} ('no goals' error) with '{replacement_line}'. Original: {line.strip()}")
        else:
            modified_lines.append(line)

    if replaced_count != len(lines_to_replace):
        logger.warning(f"Expected to replace {len(lines_to_replace)} lines, but replaced {replaced_count}. Line numbers might be incorrect.")

    return "\n".join(modified_lines)


# --- Public Function ---

def attempt_proof_repair(lean_code: str, error_log: str) -> Tuple[bool, str]:
    """
    Attempts to automatically fix known, simple errors in generated Lean code
    based on the provided error log. Currently, known handlers are disabled.

    Args:
        lean_code: The Lean code string that failed compilation.
        error_log: The stderr/stdout captured from the failed lean/lake build.

    Returns:
        A tuple: (fix_attempted_and_applied: bool, resulting_code: str).
        Always returns (False, original_code) in the current configuration.
    """
    logger.debug("Attempting automated proof repair...")
    original_code = lean_code # Keep original for comparison

    if not lean_code or not error_log:
        logger.debug("No code or error log provided for repair.")
        return False, original_code

    # --- Handler 1: Only "no goals to be solved" errors (DISABLED) ---
    # This handler was disabled because simple local repairs (delete line or
    # replace with 'done') proved insufficient or unreliable for the underlying
    # issue where a preceding tactic implicitly solved the goal. Relying on
    # LLM retry with the original error log is preferred for this case.
    """
    if _is_only_no_goals_error(error_log):
        logger.info("Detected pattern where all Lean source errors are 'no goals to be solved'. Attempting fix...")
        try:
            modified_code = _fix_no_goals_error(lean_code, error_log) # Using replace with 'done' version
            if modified_code != original_code:
                 logger.info("Applied fix for 'no goals to be solved'.")
                 if not modified_code.strip():
                      logger.error("Automated fix resulted in empty code. Reverting.")
                      return False, original_code
                 return True, modified_code
            else:
                 logger.warning("Identified 'no goals' pattern, but generated code was unchanged by fix function (potentially failed line number extraction?).")
                 return False, original_code # No effective change made
        except Exception as e:
            logger.exception(f"Error during 'no goals' fix execution: {e}", exc_info=True)
            return False, original_code # Fix failed, return original
    """

    # --- Add elif blocks here for future error handlers ---
    # Example:
    # elif _is_some_other_fixable_error(error_log):
    #    logger.info("Detected other fixable pattern...")
    #    try:
    #        modified_code = _fix_other_error(lean_code, error_log)
    #        if modified_code != original_code:
    #            logger.info("Applied fix for other error.")
    #            return True, modified_code
    #        else:
    #            return False, original_code
    #    except Exception as e:
    #        logger.exception(f"Error during other fix execution: {e}")
    #        return False, original_code

    # --- Default: No fix applied ---
    # This is reached if no enabled handlers match the error log.
    logger.debug("No enabled/matching fixable error pattern detected.")
    return False, original_code