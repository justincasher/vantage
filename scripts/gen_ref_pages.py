# File: gen_ref_pages.py

"""
Generate the code reference pages and a static index page.

This script finds all Python files in the 'src/lean_automator' directory,
creates corresponding Markdown files in the 'docs/reference' directory,
and populates them with mkdocstrings identifiers.

It generates a static index.md in docs/reference/ with a general
description of the API reference section, rather than listing individual modules.

It also ensures that the edit path points back to the original source file.
"""

# Import necessary libraries
import logging  # For logging messages and errors
from pathlib import Path  # For working with file paths in an object-oriented way

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

    # --- Skip ALL other __init__.py files ---
    # Avoid generating pages for subdirectory package markers.
    if path.name == "__init__.py":
        print(f"Skipping subdirectory __init__.py: {path}")
        continue # Move to the next file in the loop

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
    # Point the 'Edit on GitHub/GitLab/etc.' link back to the original source file
    mkdocs_gen_files.set_edit_path(output_md_path, path)

    # NOTE: Removed logic for collecting index page data here.

print("Finished generating individual reference pages.")

# --- Generate Static Index Page ---
# Define the path for the index.md file within the reference directory
index_path = reference_output_dir / "index.md"
print(f"Generating static index page: {index_path}")

# Open the index.md file for writing using mkdocs_gen_files
with mkdocs_gen_files.open(index_path, "w") as fd:
    # Write the title and the static introductory text for the index page
    print("# API Reference", file=fd)
    print(f"\nWelcome to the API reference section for the `{package_dir.name}` package.", file=fd)
    print("\nThis part of the documentation provides detailed information about the", file=fd)
    print("package's modules, classes, functions, and methods.", file=fd)
    print("\nPlease use the navigation menu on the left to explore the different modules.", file=fd)
    # NOTE: Removed the loop that generated links to individual modules.

# --- Set Edit Path for Index ---
# Set the edit path for the index page itself to point to this script file.
# This is useful if someone wants to modify the static intro text.
mkdocs_gen_files.set_edit_path(index_path, "scripts/gen_ref_pages.py") # Assuming script is in scripts/

print("Index page generation complete.")