# File: gen_ref_pages.py

"""
Generate the code reference pages and an index page linking to them.

This script finds all Python files in the 'src/lean_automator' directory,
creates corresponding Markdown files in the 'docs/reference' directory,
populates them with mkdocstrings identifiers, and generates an index.md
in docs/reference/ listing links and descriptions for all generated module pages.

Module descriptions are extracted using the 'ast' module.

Includes a TEMPORARY WORKAROUND to manually insert module-level docstrings
on individual pages due to a suspected bug in mkdocstrings/griffe.

It also ensures that the edit path points back to the original source file.
"""

# Import necessary libraries
import ast  # Used for safely parsing Python files to extract docstrings
import logging  # For logging messages and errors
from pathlib import Path  # For working with file paths in an object-oriented way
from typing import Optional, Tuple, List  # For type hinting

import mkdocs_gen_files  # Library for generating files during MkDocs builds

# --- Logging Setup ---
# Get a logger specific to this plugin/script
logger = logging.getLogger(f"mkdocs.plugins.{__name__}")
# Add a NullHandler to prevent "No handler found" warnings if logging isn't configured elsewhere
logger.addHandler(logging.NullHandler())

# --- Configuration ---
# Define the root directory of the source code
src_root = Path("src")
# Define the specific package directory within the source root
package_dir = src_root / "lean_automator"

# Define the output directory for the generated reference pages (relative to the MkDocs 'docs' directory)
reference_output_dir = Path("reference")

# List to store tuples of (friendly_name, relative_md_path, module_docstring) for the index page
index_page_data: List[Tuple[str, str, Optional[str]]] = []


# --- Helper Function ---
def get_module_docstring(filepath: Path) -> Optional[str]:
    """
    Safely extracts the module-level docstring from a Python file.

    Uses the 'ast' module to parse the Python file without executing it,
    making it safe to handle potentially complex or untrusted code.

    Args:
        filepath: The Path object pointing to the Python file.

    Returns:
        The module docstring as a string if found and parsed successfully,
        otherwise returns None.
    """
    try:
        # Open and read the source file content
        with open(filepath, "r", encoding="utf-8") as source_file:
            source_code = source_file.read()
        # Parse the source code into an Abstract Syntax Tree (AST)
        tree = ast.parse(source_code, filename=str(filepath))
        # Use ast.get_docstring() to extract the docstring from the top level of the AST
        docstring = ast.get_docstring(tree)
        return docstring
    except FileNotFoundError:
        # Log an error if the file doesn't exist
        logger.error(f"File not found when trying to get docstring: {filepath}")
        return None
    except SyntaxError as e:
        # Log an error if the file has syntax errors and cannot be parsed
        logger.error(f"Syntax error parsing {filepath} for docstring: {e}")
        return None
    except Exception as e:
        # Log any other unexpected errors during parsing
        logger.error(f"Unexpected error getting docstring from {filepath}: {e}")
        return None


# --- Main Script Logic ---
print(f"Searching for Python modules in: {package_dir}")

# Iterate through all '.py' files recursively within the specified package directory
# 'sorted()' ensures a consistent order, which helps with reproducibility
for path in sorted(package_dir.rglob("*.py")):
    # Calculate the module path relative to the source root (e.g., 'lean_automator/utils/helpers')
    # '.with_suffix("")' removes the '.py' extension
    module_path = path.relative_to(src_root).with_suffix("")

    # --- Skip Top-Level __init__.py ---
    # Avoid generating a separate page for the main package's __init__.py
    if path.name == "__init__.py" and path.parent == package_dir:
        print(f"Skipping root __init__.py: {path}")
        continue  # Move to the next file in the loop

    # --- Extract Module Docstring ---
    # Call the helper function to get the module-level docstring
    # We need this for both the individual page workaround and the index page
    module_docstring = get_module_docstring(path)

    # --- Generate Individual Module Page ---
    # Create the mkdocstrings identifier string (e.g., 'lean_automator.utils.helpers')
    doc_identifier = str(module_path).replace("/", ".")
    # Calculate the output Markdown file path relative to the MkDocs 'docs' directory
    # (e.g., 'reference/utils/helpers.md')
    output_md_path = reference_output_dir / module_path.relative_to(package_dir.name).with_suffix(".md")

    print(f"  Found module: {doc_identifier}")
    print(f"    Generating doc page: {output_md_path}")

    # Use mkdocs_gen_files to open the target Markdown file for writing.
    with mkdocs_gen_files.open(output_md_path, "w") as fd:
        # Write the mkdocstrings identifier for members.
        print(f"::: {doc_identifier}", file=fd)

    # --- Set Edit Path ---
    mkdocs_gen_files.set_edit_path(output_md_path, path)

    # --- Collect Index Page Information ---
    # Calculate the link target relative to the 'reference' directory
    link_target = output_md_path.relative_to(reference_output_dir).as_posix() # Use POSIX paths
    # Create a user-friendly title
    link_title = path.stem.replace('_', ' ').title()
    if path.name == "__init__.py":
        link_title = path.parent.name.replace('_', ' ').title() + " (Package)"

    # Append the title, target path, AND the docstring to the list
    index_page_data.append((link_title, link_target, module_docstring))

print("Finished generating individual reference pages.")

# --- Generate Index Page ---
# Define the path for the index.md file within the reference directory
index_path = reference_output_dir / "index.md"
print(f"Generating index page: {index_path}")

# Open the index.md file for writing using mkdocs_gen_files
with mkdocs_gen_files.open(index_path, "w") as fd:
    # Write the title and introductory text for the index page
    print("# API Reference Index", file=fd)
    print("\nThis index lists all the modules available in the API reference.", file=fd)
    print("Each module includes a link to its detailed documentation and its description.\n", file=fd)

    # Sort the collected data alphabetically by title for a clean index
    for title, target, docstring in sorted(index_page_data):
        # Convert title to lower case
        title = title.lower()

        # Write the title as a heading with a link
        print(f"### [{title}]({target})\n", file=fd)
        # Write the docstring if it exists, otherwise mention it's missing
        if docstring:
            print(docstring, file=fd)
        else:
            print("_No description available._", file=fd)
        # Add a separator between entries
        print("\n---\n", file=fd)

# --- Set Edit Path for Index ---
# Set the edit path for the index page itself to point to this script file.
mkdocs_gen_files.set_edit_path(index_path, "scripts/gen_ref_pages.py")

print("Index page generation complete.")
