# Welcome to Vantage

> *The essence of mathematical insight often comes from finding a new vantage point.*

Vantage envisions a future where mathematical knowledge is navigated as a vast, interconnected graph. This Python project is building the foundations for autoformalization software where each mathematical statement constitutes a node. By employing Lean 4 for rigorous verification and utilizing large language models to drive the autoformalization process, Vantage targets a highly parallelized approach. The ultimate ambition is to map, organize, and accelerate the exploration of the ever-expanding universe of mathematics with unparalleled accuracy and efficiency.

***Heads Up!*** *This project is currently under active development and is being built day-by-day. Features, documentation, and the overall structure may change frequently. Your patience and feedback are appreciated!*

This documentation provides detailed information on setting up, configuring, using, and contributing to the Vantage project.

## Features

* **Knowledge Base Storage:** Stores mathematical items (`KBItem`) with code, metadata, and embeddings using SQLite.
* **Lean Interaction & Persistent Library:** Verifies Lean 4 code against an incrementally growing shared Lean library (`vantage_lib`) using `lake`.
* **LLM Integration:** Interacts with Google Gemini API (async calls, retries, cost tracking).
* **Structured Data:** Uses Python dataclasses for knowledge representation.
* **Dependency & State Tracking:** Manages item dependencies and status (`PENDING_LEAN`, `PROVEN`, etc.).
* **Automated API Documentation:** Detailed API reference generated directly from source code docstrings.

## Documentation Overview

Navigate through the documentation using the sidebar or start with these main sections:

* **[Getting Started](installation.md):** How to install, configure, and begin using Vantage. This section covers dependencies, setup, and basic usage examples.
* **[API Reference](reference/index.md):** Detailed information on the project's modules, classes, and functions, generated directly from the source code.
* **[Contributing](contributing.md):** Guidelines for contributing to the project, including testing procedures, coding style, and the code of conduct.

## Get Involved!

As Vantage is actively being built, contributions are highly welcome! If you find this project interesting and would like to help shape its future, please:

1.  Review the **[Contributing Guidelines](contributing.md)** for information on how to get started.
2.  Feel free to reach out directly via email at `justinchadwickasher@gmail.com` with any questions, ideas, or offers to help.

---

For the source code repository, visit [GitHub](https://github.com/justincasher/vantage).