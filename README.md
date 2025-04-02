# Vantage

> *The essence of mathematical insight often comes from finding a new vantage point.*

Vantage is a Python project designed to build and manage a mathematical knowledge base (KB). It leverages Lean 4 for formal verification, integrates with Large Language Models (specifically Google's Gemini) for potential code/description generation, and uses an SQLite database for persistent storage.

For a conceptual overview and the motivation behind this project, see the introductory article: **[Introducing Vantage](https://justinasher.me/introducing_vantage)**.

## Features

* **Knowledge Base Storage:** Stores mathematical items (`KBItem`) with code, metadata, and embeddings (SQLite).
* **Lean Interaction & Persistent Library:** Verifies Lean 4 code against an incrementally growing shared Lean library (`vantage_lib`) using `lake`.
* **LLM Integration:** Interacts with Google Gemini API (async calls, retries, cost tracking).
* **Structured Data:** Uses dataclasses for knowledge representation.
* **Dependency & State Tracking:** Manages item dependencies and status (`PENDING_LEAN`, `PROVEN`, etc.).

## Prerequisites

* **Python:** Version 3.8+ recommended.
* **Lean 4 & Lake:** Working installation with `lake` in PATH. ([Lean Installation Guide](https://docs.lean-lang.org/lean4/doc/quickstart.html))
* **Google AI API Key:** For LLM features. ([Google AI Studio](https://aistudio.google.com/))

## Getting Started

For detailed instructions on setup, configuration, usage, and more, please refer to the full documentation:

* **[Full Documentation Index](docs/index.md)**
* **[Installation & Setup Guide](docs/installation.md)**
* **[Configuration Details](docs/configuration.md)**
* **[Usage Examples](docs/usage.md)**
* **[Running Tests](docs/testing.md)**

## Contributing

Vantage is under active development and contributions are highly welcome! If you're interested in helping shape the project, please review the **[Contributing Guide](docs/contributing.md)** for information on how to get started. You can also feel free to reach out directly via email at `justinchadwickasher@gmail.com` with any questions, ideas, or offers to help.

We are actively looking for collaborators! Please see the [Contributing Guide](docs/contributing.md) or email Justin Asher directly at justinchadwickasher@gmail.com.

## Project Structure

An overview of the project layout can be found in the [Project Structure](docs/project_structure.md) documentation.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for the full text.
