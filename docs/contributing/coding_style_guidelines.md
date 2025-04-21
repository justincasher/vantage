# Coding Style Guidelines

Consistency in code style is crucial for readability and maintainability. This project uses **Ruff** to enforce coding standards automatically.

## Tooling: Ruff

[**Ruff**](https://docs.astral.sh/ruff/) is used as the primary tool for both **linting** (checking for code errors and style violations) and **formatting** (automatically applying code style). Configuration is defined in the `pyproject.toml` file under the `[tool.ruff]` section.

## Running Ruff

To ensure your code adheres to the project's standards, you can run Ruff manually from your terminal. It's recommended to run these commands before committing your changes.

* **Checking for errors and style violations (Linting):**
    To scan the entire project (starting from the current directory `.`) for issues based on the rules defined in `pyproject.toml`, run:
    ```bash
    ruff check .
    ```
    You can also check specific files or directories by replacing `.` with the desired path:
    ```bash
    ruff check path/to/your/file.py
    ruff check path/to/your/directory/
    ```

* **Automatically formatting your code:**
    To automatically reformat code in the entire project (starting from the current directory `.`) to comply with the configured style, run:
    ```bash
    ruff format .
    ```
    Similarly, you can format specific files or directories:
    ```bash
    ruff format path/to/your/file.py
    ruff format path/to/your/directory/
    ```

## Code Formatting: PEP 8 enforced by Ruff

The project aims for compliance with the [**PEP 8 — Style Guide for Python Code**](https://peps.python.org/pep-0008/), with enforcement and specific settings managed by Ruff.

Key aspects enforced by the Ruff configuration (`pyproject.toml`):

* **Linter Rules:** Ruff checks for a range of issues including:
    * `E`/`W`: Pycodestyle errors and warnings (covering many PEP 8 rules).
    * `F`: Pyflakes checks (e.g., undefined names, unused imports/variables).
    * `I`: Isort checks for consistent import sorting (standard library, third-party, local application). Imports should usually be on separate lines.
    * `UP`: Pyupgrade suggestions for modernizing Python syntax.
* **Line Length:** Limit all lines to a maximum of **88 characters**. This is configured in `pyproject.toml` (`line-length = 88`). Docstrings and comments should ideally wrap earlier if practical for readability, but the hard limit is 88.
* **Indentation:** Use 4 spaces per indentation level.
* **Whitespace:** Use whitespace appropriately around operators and after commas, but not directly inside parentheses, brackets, or braces (enforced by `E`/`W` rules).
* **Naming Conventions:** Follow standard naming conventions (e.g., `lowercase` for functions and variables, `CapWords` for classes, `UPPERCASE` for constants). Ruff helps catch some deviations, but adherence is also expected during code review.
* **Target Python Version:** Code style and syntax upgrades (`UP`) consider the project's target Python version (`py38` as set in `pyproject.toml`).

## Documentation Strings: Google Style

All modules, functions, classes, and methods *must* have docstrings that follow the [**Google Python Style Guide**](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) format. While the current Ruff configuration doesn't heavily lint docstring content (no `D` rules selected), adhering to this standard is mandatory for clarity and maintainability.

Key aspects include:

* A concise summary line ending in a period.
* A more detailed explanation if necessary, separated from the summary by a blank line.
* Sections for arguments (`Args:`), return values (`Returns:`), and raised exceptions (`Raises:`), each clearly describing the item.

**Example:**

```python
def example_function(param1, param2):
  """Does something interesting.

  This function takes two parameters and performs a calculation,
  returning the result. It might also raise an error under
  certain conditions.

  Args:
    param1 (int): The first parameter, expected to be an integer.
    param2 (str): The second parameter, expected to be a string
      representation of a number.

  Returns:
    float: The result of adding param1 to the numeric value of param2.

  Raises:
    ValueError: If param1 is negative.
    TypeError: If param2 is not a valid string representation of a number.
  """
  if not isinstance(param1, int):
      # Optional: Add type check for param1 if strict typing is desired
      # although the docstring only mentions raising ValueError for negativity.
      pass # Or raise TypeError("param1 must be an integer.")

  if param1 < 0:
    raise ValueError("param1 cannot be negative.")

  try:
    # Attempt to convert param2 to a float for calculation
    param2_numeric = float(param2)
  except (ValueError, TypeError):
    # Catch errors if param2 cannot be converted (e.g., "abc" or None)
    raise TypeError("param2 must be a valid string representation of a number.")

  # Perform the calculation
  result = float(param1 + param2_numeric)

  return result
```