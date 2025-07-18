# Vantage

> [!WARNING]
> This project is incomplete and no longer being worked on for the time being. See [LeanExplore](https://github.com/justincasher/lean-explore) for my Lean search engine.

> *The essence of mathematical insight often comes from finding a new vantage point.*

Vantage envisions a future where mathematical knowledge is navigated as a vast, interconnected graph. This Python project is building the foundations for autoformalization software where each mathematical statement constitutes a node. By employing Lean 4 for rigorous verification and utilizing large language models to drive the autoformalization process, Vantage targets a highly parallelized approach. The ultimate ambition is to map, organize, and accelerate the exploration of the ever-expanding universe of mathematics with unparalleled accuracy and efficiency.

**Full Documentation:** [**https://vantageproject.org**](https://vantageproject.org)

For a conceptual overview and the motivation behind this project, see the introductory article: **[Introducing Vantage](https://justinasher.me/introducing_vantage)**.

## Features

* **Knowledge Base Storage:** Stores mathematical items (`KBItem`) with code, metadata, and embeddings using SQLite.
* **Lean Interaction & Persistent Library:** Verifies Lean 4 code against an incrementally growing shared Lean library (`vantage_lib`) using `lake`.
* **LLM Integration:** Interacts with Google Gemini API (async calls, retries, cost tracking).
* **Structured Data:** Uses Python dataclasses for knowledge representation.
* **Dependency & State Tracking:** Manages item dependencies and status (`PENDING_LEAN`, `PROVEN`, etc.).
* **Automated API Documentation:** Detailed API reference generated directly from source code docstrings.

## Prerequisites

* **Python:** Version 3.8+ recommended.
* **Lean 4 & Lake:** Working installation with `lake` in PATH. ([Lean Installation Guide](https://lean-lang.org/lean4/doc/quickstart.html))
* **Google AI API Key:** For LLM features. ([Google AI Studio](https://aistudio.google.com/))

## Getting Started & Documentation

The full documentation, including installation guides, usage details, API specifications, and contribution guidelines, is available at:

[**https://vantageproject.org**](https://vantageproject.org)

Key entry points into the documentation include:

* **[Getting Started](https://vantageproject.org/installation/)**: Installation, setup, configuration, and basic usage.
* **[API Reference](https://vantageproject.org/reference/)**: Detailed documentation for all modules and classes.
* **[Contributing](https://vantageproject.org/contributing/)**: How to contribute, testing procedures, style guides, and code of conduct.

## Contributing

Vantage is under active development and contributions are highly welcome! If you're interested in helping shape the project, please review the **[Contributing Guide](https://vantageproject.org/contributing/)**.

We are actively looking for collaborators! Feel free to explore the [issues tab](https://github.com/justincasher/vantage/issues) on GitHub, or reach out directly via email at `justinchadwickasher@gmail.com` with any questions, ideas, or offers to help.

## Repository

* **GitHub:** [https://github.com/justincasher/vantage](https://github.com/justincasher/vantage)

## License

This project is licensed under the terms of the Appache License. See the [LICENSE](LICENSE) file for the full text.
