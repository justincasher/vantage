# Testing

This project utilizes both **unit tests** and **integration tests** to ensure code correctness and system reliability.

* **Unit Tests:** These tests focus on verifying individual components or functions in isolation. They are typically fast and don't rely on external services or complex setups.
* **Integration Tests:** These tests verify the interaction between different parts of the system or with external dependencies (like databases or APIs). They ensure that components work together as expected. Integration tests are often marked with `pytest` markers (e.g., `@pytest.mark.integration`) and might be slower or require specific setup.

Tests are written and executed using the `pytest` framework.

## Running Tests

1.  **Install test dependencies:**
    Ensure you have the necessary testing libraries installed. If you followed the main installation guide, these might already be included. If not, you may need to install them (potentially from a `requirements-dev.txt` file or similar):
    ```bash
    pip install pytest pytest-asyncio pytest-cov
    ```

2.  **Execute tests:**
    Navigate to the project root directory (usually where a `pytest.ini` or `pyproject.toml` configuration file is located).

    * **Run all tests (Unit & Integration):**
        ```bash
        pytest
        ```
        * **Note:** Running all tests typically includes integration tests, which require specific environment setup (like a running database or API keys) and will take longer to complete.

    * **Run only unit tests (excluding integration tests):**
        Use `pytest` markers to select tests. Assuming integration tests are marked with `@pytest.mark.integration`:
        ```bash
        pytest -m "not integration"
        ```

    * **Run only integration tests:**
        ```bash
        pytest -m integration
        ```

    * **Exclude slow tests (if marked):**
        If some tests are marked as slow (e.g., `@pytest.mark.slow`):
        ```bash
        pytest -m "not slow"
        ```

    You can combine markers as needed (e.g., `pytest -m "integration and not slow"`). Check the project's tests for specific markers used.

    (*Note: "not slow" will likely be removed in favor of a more descriptive marker.*)
