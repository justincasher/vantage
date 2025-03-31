# Usage Examples

*(This section provides high-level examples. Integrate into your workflow.)*

1.  **Initialize the Database:**
    ```python
    from lean_automator.kb_storage import initialize_database
    # Uses KB_DB_PATH env var or default
    initialize_database()
    ```

2.  **Create and Save a Knowledge Base Item:**
    ```python
    from lean_automator.kb_storage import KBItem, ItemType, ItemStatus, save_kb_item
    import asyncio # save_kb_item is async

    async def add_item():
        # ... (rest of python code block) ...

    asyncio.run(add_item())
    ```

3.  **Interact with the Gemini API:**
    ```python
    # Example assumes GeminiClient is initialized as 'client'
    import asyncio
    from lean_automator import llm_call

    async def generate_text():
        client = llm_call.GeminiClient() # Reads config from env
        prompt = "Explain the concept of mathematical induction."
        try:
            response = await client.generate_text_async(prompt)
            print("Generated Text:", response)
            print("--- Cost Info ---")
            print(client.get_cost_summary())
        except Exception as e:
            print(f"An error occurred: {e}")

    asyncio.run(generate_text())

    ```
    *(Self-correction: Added the missing Gemini API example code from the original prompt)*


4.  **Generate and Verify a Lean Item:**
    This function attempts to generate Lean code (if missing) using the LLM and then verifies it using `lean_interaction.check_and_compile_item`. On success, the item's status is set to `PROVEN` in the DB, and its source code is added and compiled within the persistent shared library (`vantage_lib`).
    ```python
    import asyncio
    from lean_automator import lean_processor, llm_call, kb_storage

    # Assume:
    # - DB is initialized
    # - Shared library ('vantage_lib') is initialized
    # - Environment variables (API key, shared lib path) are set
    # - A KBItem 'MyDefs.MyType' exists with status DEFINITION_ADDED / PROVEN
    # - A KBItem 'MyTheorems.UseMyType' exists with status PENDING_LEAN,
    #   latex_statement defined, and plan_dependencies=["MyDefs.MyType"]

    async def process_item():
        client = llm_call.GeminiClient() # Reads config from env
        item_name = "MyTheorems.UseMyType"

        success = await lean_processor.generate_and_verify_lean(
            unique_name=item_name,
            client=client
            # db_path=..., # Optional override
            # lake_executable_path=..., # Optional override
            # timeout_seconds=... # Optional override
        )

        if success:
            print(f"Processing successful for {item_name}.")
            # The item is now PROVEN in the DB and added to the shared library
        else:
            print(f"Processing failed for {item_name}. Check logs and DB for error details.")
            # The item is now LEAN_VALIDATION_FAILED or ERROR in the DB

    asyncio.run(process_item())
    ```