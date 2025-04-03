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
    param2 (str): The second parameter, expected to be a string.

  Returns:
    float: The result of the calculation based on the inputs.
    Returns None if the calculation cannot be performed.

  Raises:
    ValueError: If param1 is negative.
    TypeError: If param2 is not a valid string representation of a number.
  """
  if param1 < 0:
    raise ValueError("
