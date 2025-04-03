# Coding Style Guidelines

Consistency in code style is crucial for readability and maintainability. This project adheres to the following standards for Python code:

## Code Formatting: PEP 8

All Python code should follow the [**PEP 8 -- Style Guide for Python Code**](https://peps.python.org/pep-0008/).

Key aspects include (but are not limited to):
* Indentation: Use 4 spaces per indentation level.
* Line Length: Limit all lines to a maximum of 79 characters (docstrings/comments to 72).
* Imports: Imports should usually be on separate lines and grouped in the standard order (standard library, related third-party, local application/library specific).
* Whitespace: Use whitespace appropriately around operators and after commas, but not directly inside parentheses, brackets, or braces.
* Naming Conventions: Follow standard naming conventions (e.g., `lowercase` for functions and variables, `CapWords` for classes, `UPPERCASE` for constants).

## Documentation Strings: Google Style

All modules, functions, classes, and methods should have docstrings that follow the [**Google Python Style Guide**](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) format.

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
    # Note: The original docstring mentioned returning None if the
    # calculation cannot be performed, but raising exceptions for bad
    # input (as specified in Raises) is generally preferred.

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
