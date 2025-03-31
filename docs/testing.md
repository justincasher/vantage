# Testing

The project includes unit and integration tests using `pytest`.

1.  **Install test dependencies:**
    ```bash
    pip install pytest pytest-asyncio
    ```

2.  **Run tests:**
    Navigate to the project root directory (where `pytest.ini` is located).

    * **Run all tests:**
        ```bash
        pytest
        ```
        * **Note:** Running all tests includes integration tests... (rest of note) ...

    * **Run only unit tests (excluding integration tests):**
        ```bash
        pytest -m "not integration"
        ```

    * **Run only integration tests:**
        ```bash
        pytest -m integration
        ```

    * **Exclude slow tests:**
        ```bash
        pytest -m "not slow"
        ```

    You can combine markers as needed (e.g., `pytest -m "integration and not slow"`)