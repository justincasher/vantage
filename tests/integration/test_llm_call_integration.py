# File: tests/integration/test_llm_call_integration.py

"""Integration tests for the llm_call module, specifically GeminiClient.

These tests interact with the live Google Gemini API to verify the functionality
of the GeminiClient class, including text generation, embedding creation,
parameter overrides, error handling, and cost tracking updates.

Requires the `google-generativeai` library to be installed and a `GEMINI_API_KEY`
environment variable to be set. Tests interacting with the API will be skipped
if the key is not available.
"""

import os
import pytest
import asyncio
import json
import warnings # Added to check for warnings
import warnings
from typing import Tuple, List, Dict, Any

# Make sure pytest can find the src modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# config_loader imports with fallback
try:
    # Import both the config dictionary and the specific API key getter
    from lean_automator.config_loader import APP_CONFIG, get_gemini_api_key
except ImportError:
    warnings.warn("config_loader.APP_CONFIG and get_gemini_api_key not found. Using fallbacks and environment variables directly.", ImportWarning)
    # Provide fallbacks if the config loader isn't available
    APP_CONFIG = {}
    # Define a fallback getter if the real one isn't imported
    def get_gemini_api_key() -> Optional[str]:
         """Fallback: Retrieves API key directly from environment."""
         return os.getenv('GEMINI_API_KEY')

# Use absolute imports
from lean_automator.llm_call import (
    GeminiClient,
    GeminiCostTracker,
    ModelCostInfo, # Import if needed for direct assertion
    FALLBACK_MODEL_COSTS_JSON,
    DEFAULT_SAFETY_SETTINGS
)

# Import specific types/exceptions needed for assertions and setup
try:
    from google.generativeai import types as genai_types
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.api_core import exceptions as google_api_exceptions # For asserting specific API errors
except ImportError:
    # Allow tests to be collected even if library is missing,
    # api_key fixture will skip them later if key (and likely library) is unavailable.
    genai_types = None
    google_api_exceptions = None
    HarmCategory = None
    HarmBlockThreshold = None


# --- Configuration ---
# Define models known to be available and relatively cheap for testing
# Ensure these models are present in your GEMINI_MODEL_COSTS if checking costs
TEST_GEN_MODEL = "gemini-1.5-flash-latest"
TEST_EMBED_MODEL = "models/text-embedding-004"

# Fallback costs if environment variable is not set (adjust if needed)
DEFAULT_TEST_COSTS = json.dumps({
    TEST_GEN_MODEL: {"input": 0.35, "output": 0.53}, # Example costs per million tokens
    TEST_EMBED_MODEL: {"input": 0.1, "output": 0.0} # Example embedding model cost
})

# --- Fixtures ---

@pytest.fixture(scope="module")
def api_key() -> str:
    """Provides the Gemini API key from environment variables.

    Skips all tests in the module if the `GEMINI_API_KEY` environment variable
    is not set, or if the required Google libraries (`google-generativeai`,
    `google-api-core`) could not be imported.

    Returns:
        str: The Gemini API key.
    """
    key = get_gemini_api_key()
    if not key:
        pytest.skip("GEMINI_API_KEY environment variable not set. Skipping integration tests.")
    # Also skip if essential Google libraries failed to import
    if not genai_types or not google_api_exceptions:
         pytest.skip("Required google.generativeai or google.api_core libraries not found. Skipping integration tests.")
    return key

@pytest.fixture(scope="module")
def model_costs_json() -> str:
    """Provides the model cost configuration JSON string.

    Reads from the `GEMINI_MODEL_COSTS` environment variable, falling back to
    `DEFAULT_TEST_COSTS` if the environment variable is not set.
    Note: This uses a dedicated env var for testing cost scenarios, rather
    than relying on the main APP_CONFIG['costs'] loaded from model_costs.json,
    to allow for easier test-specific cost injection.

    Returns:
        str: The JSON string containing model cost information.
    """
    return os.getenv('GEMINI_MODEL_COSTS', DEFAULT_TEST_COSTS)

@pytest.fixture
def cost_tracker(model_costs_json: str) -> GeminiCostTracker:
    """Provides a new `GeminiCostTracker` instance for each test function.

    Initializes the cost tracker using the model cost JSON provided by the
    `model_costs_json` fixture.

    Args:
        model_costs_json: The JSON string defining model costs.

    Returns:
        GeminiCostTracker: A fresh instance for tracking API call costs.
    """
    costs_dict = json.loads(model_costs_json)
    return GeminiCostTracker(model_costs_override=costs_dict)

@pytest.fixture
def client(api_key: str, cost_tracker: GeminiCostTracker) -> GeminiClient:
    """Provides a configured `GeminiClient` instance for each test function.

    Initializes the client with the API key, default test models, and the
    test-specific cost tracker instance.

    Args:
        api_key: The Gemini API key.
        cost_tracker: The GeminiCostTracker instance for this test.

    Returns:
        GeminiClient: A configured client instance ready for API calls.
    """
    # Re-import locally within fixture scope if needed, though module scope should work
    # from lean_automator.llm_call import GeminiClient
    return GeminiClient(
        api_key=api_key,
        default_generation_model=TEST_GEN_MODEL,
        default_embedding_model=TEST_EMBED_MODEL,
        cost_tracker=cost_tracker # Pass the test-specific tracker
    )

# --- Test Class ---

@pytest.mark.integration
class TestGeminiClientIntegration:
    """Groups integration tests for the GeminiClient against the live API."""

    # --- Generation Tests ---

    @pytest.mark.asyncio
    async def test_successful_generation(self, client: GeminiClient):
        """Verify a basic successful text generation call using default model."""
        prompt = "Explain the concept of integration testing in one sentence."
        try:
            response = await client.generate(prompt)
            assert isinstance(response, str)
            assert len(response) > 10, "Response seems too short"
            assert "test" in response.lower() or "verify" in response.lower() or "integrate" in response.lower(), "Response missing keywords"
        except Exception as e:
            pytest.fail(f"Basic generation failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_explicit_generation_model_usage(self, client: GeminiClient):
        """Verify generation works when explicitly specifying the model argument."""
        prompt = "What is the capital of France?"
        try:
            response = await client.generate(prompt, model=TEST_GEN_MODEL)
            assert isinstance(response, str)
            assert "Paris" in response, f"Expected 'Paris' in response, got: {response}"
        except Exception as e:
            pytest.fail(f"Generation with explicit model '{TEST_GEN_MODEL}' failed: {e}")

    @pytest.mark.asyncio
    async def test_generation_config_override(self, client: GeminiClient):
        """Verify overriding generation config (temperature, max tokens) works."""
        prompt = "Write a single word that means 'very happy'."
        # Override config for deterministic, short output
        config_override = {
            "temperature": 0.0,
            "max_output_tokens": 5 # Limit output length significantly
        }
        try:
            response = await client.generate(prompt, generation_config_override=config_override)
            assert isinstance(response, str)
            assert len(response.split()) <= 2, "Response should be very short due to max_output_tokens"
            assert len(response) > 0, "Response should not be empty"
            # Simple check if it's likely a single word
            assert response.strip().isalnum(), f"Response '{response}' doesn't look like a single word"
        except Exception as e:
            pytest.fail(f"Generation with config override failed: {e}")

    @pytest.mark.asyncio
    async def test_safety_settings_override(self, client: GeminiClient):
        """Verify providing valid safety setting overrides allows the call."""
        prompt = "Tell me a short, safe story about a friendly robot."
        # Override safety settings to allow all content (use with caution)
        safety_override = [
            genai_types.SafetySettingDict(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            genai_types.SafetySettingDict(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            genai_types.SafetySettingDict(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            genai_types.SafetySettingDict(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        try:
            response = await client.generate(prompt, safety_settings_override=safety_override)
            assert isinstance(response, str)
            assert len(response) > 10, "Story response seems too short"
            assert "robot" in response.lower(), "Story response missing keyword 'robot'"
        except Exception as e:
            pytest.fail(f"Generation with safety override failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_invalid_generation_model_name_raises_error(self, client: GeminiClient):
        """Verify using an invalid generation model name raises an expected API error."""
        prompt = "This should not be generated."
        invalid_model = "invalid-model-name-does-not-exist-12345"
        # Expect an exception, likely related to resource not found or invalid argument
        with pytest.raises(Exception) as excinfo:
             await client.generate(prompt, model=invalid_model)

        # Check the cause of the exception if possible (depends on client's error wrapping)
        cause = getattr(excinfo.value, '__cause__', None)
        # Check if the exception or its cause is one of the expected Google API errors
        assert isinstance(excinfo.value, (google_api_exceptions.NotFound, google_api_exceptions.InvalidArgument, google_api_exceptions.PermissionDenied, ValueError)) or \
               isinstance(cause, (google_api_exceptions.NotFound, google_api_exceptions.InvalidArgument, google_api_exceptions.PermissionDenied, ValueError)), \
               f"Expected Google API error or ValueError for invalid model, but got {type(excinfo.value)} with cause {type(cause)}"
        print(f"\nCaught expected error for invalid generation model: {excinfo.value} (Cause: {cause})")

    @pytest.mark.asyncio
    async def test_generation_cost_tracking_updates(self, client: GeminiClient, cost_tracker: GeminiCostTracker):
        """Verify cost tracker values increase after a successful generation call."""
        prompt = "Count to three."
        target_model = TEST_GEN_MODEL # Model being tested

        initial_summary = cost_tracker.get_summary()
        initial_calls = initial_summary.get("usage_by_model", {}).get(target_model, {}).get("calls", 0)
        initial_input_units = initial_summary.get("usage_by_model", {}).get(target_model, {}).get("input_units", 0)
        initial_output_units = initial_summary.get("usage_by_model", {}).get(target_model, {}).get("output_units", 0)
        initial_total_cost = initial_summary.get("total_estimated_cost", 0.0)

        try:
            response = await client.generate(prompt)
            assert isinstance(response, str) and len(response) > 0
        except Exception as e:
            pytest.fail(f"API call for generation cost tracking test failed: {e}")

        final_summary = cost_tracker.get_summary()
        model_usage = final_summary.get("usage_by_model", {}).get(target_model, {})
        final_total_cost = final_summary.get("total_estimated_cost", 0.0)

        # Assertions on tracker updates
        assert model_usage.get("calls", 0) == initial_calls + 1, "Call count did not increment"
        assert model_usage.get("input_units", 0) > initial_input_units, "Input units did not increase"
        assert model_usage.get("output_units", 0) > initial_output_units, "Output units did not increase"

        # Check if total cost increased (requires cost data to be loaded)
        model_cost_info = cost_tracker._model_costs.get(target_model)
        if isinstance(model_cost_info, ModelCostInfo):
            # Cost should increase if model costs are known and non-zero
            assert final_total_cost > initial_total_cost, "Total estimated cost did not increase"
            print(f"\nCost Tracking: Calls={model_usage.get('calls')}, Input={model_usage.get('input_units')}, Output={model_usage.get('output_units')}, Est. Cost Increase={final_total_cost - initial_total_cost:.6f}")
        else:
            # If cost data is missing, total cost shouldn't change, print warning
            print(f"\nWarning: Cost data for model '{target_model}' missing in cost tracker. Cannot verify cost increase.")
            assert final_total_cost == pytest.approx(initial_total_cost), "Total cost changed unexpectedly despite missing model cost data"


    # --- Embedding Tests ---

    @pytest.mark.asyncio
    async def test_successful_embedding_single(self, client: GeminiClient):
        """Verify a successful embedding call for a single string."""
        content = "This is a test sentence for embedding."
        task_type = "RETRIEVAL_DOCUMENT"
        try:
            response = await client.embed_content(content, task_type)
            assert isinstance(response, list), "Response should be a list"
            assert len(response) == 1, "Response list should contain one embedding for single input"
            assert isinstance(response[0], list), "Embedding should be a list of floats"
            assert len(response[0]) > 10, "Embedding vector seems too short" # e.g., text-embedding-004 is 768
            assert all(isinstance(val, float) for val in response[0]), "Embedding vector elements should be floats"
        except Exception as e:
            pytest.fail(f"Basic single embedding failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_successful_embedding_list(self, client: GeminiClient):
        """Verify a successful embedding call for a list of strings."""
        contents = ["First sentence.", "Second sentence for embedding."]
        task_type = "SEMANTIC_SIMILARITY"
        try:
            response = await client.embed_content(contents, task_type)

            # Debugging output added based on previous run
            print(f"\nDEBUG (List Embed): Received type={type(response)}, len={len(response) if isinstance(response, list) else 'N/A'}")
            if isinstance(response, list) and len(response) > 0: print(f"DEBUG (List Embed): First element type={type(response[0])}")

            # Assertions based on expected structure for list input
            assert isinstance(response, list), "Response should be a list"
            assert len(response) == len(contents), f"Expected {len(contents)} embeddings, got {len(response)}" # Check length matches input list
            for i, vec in enumerate(response):
                 assert isinstance(vec, list), f"Element {i} in response is not a list"
                 assert len(vec) > 10, f"Embedding vector {i} seems too short"
                 assert all(isinstance(val, float) for val in vec), f"Embedding vector {i} elements should be floats"
        except Exception as e:
            pytest.fail(f"List embedding failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_embedding_optional_args(self, client: GeminiClient):
        """Verify embedding call with optional title and output_dimensionality."""
        content = "This document is about photosynthesis."
        task_type = "RETRIEVAL_DOCUMENT"
        title = "Photosynthesis Explained"
        output_dim = 128 # Choose a dimension smaller than default
        try:
            response = await client.embed_content(
                content, task_type, title=title, output_dimensionality=output_dim
            )
            assert isinstance(response, list) and len(response) == 1, "Response structure incorrect"
            assert isinstance(response[0], list), "Embedding is not a list"
            assert len(response[0]) == output_dim, f"Embedding dimension mismatch: Expected {output_dim}, Got {len(response[0])}"
        except google_api_exceptions.InvalidArgument as e:
             # Skip if the API/model specifically rejects output_dimensionality for this model
             if "output_dimensionality" in str(e).lower() and ("supported" in str(e).lower() or "invalid" in str(e).lower()):
                  pytest.skip(f"API or model {TEST_EMBED_MODEL} might not support output_dimensionality={output_dim}. Skipping. Error: {e}")
             else: # Fail on other InvalidArgument errors
                  pytest.fail(f"Embedding with optional args failed with unexpected InvalidArgument: {e}")
        except Exception as e:
             # Fail on any other exceptions
             pytest.fail(f"Embedding with optional args failed unexpectedly: {e}")

    @pytest.mark.asyncio
    async def test_embedding_explicit_model_usage(self, client: GeminiClient):
        """Verify embedding works when explicitly specifying the embedding model."""
        content = "Explicit model test."
        task_type = "RETRIEVAL_QUERY"
        try:
            response = await client.embed_content(content, task_type, model=TEST_EMBED_MODEL)
            assert isinstance(response, list) and len(response) == 1, "Response structure incorrect"
            assert isinstance(response[0], list), "Embedding is not a list"
            assert len(response[0]) > 10, "Embedding vector seems too short"
        except Exception as e:
            pytest.fail(f"Embedding with explicit model '{TEST_EMBED_MODEL}' failed: {e}")

    @pytest.mark.asyncio
    async def test_invalid_embedding_model_name_raises_error(self, client: GeminiClient):
        """Verify using an invalid embedding model name raises an expected API error."""
        content = "This should not be embedded."
        task_type = "RETRIEVAL_QUERY"
        invalid_model = "models/invalid-embedding-model-does-not-exist-xyz"
        # Expect an exception, likely NotFound or InvalidArgument
        with pytest.raises(Exception) as excinfo:
             await client.embed_content(content, task_type, model=invalid_model)

        # Check the cause if available
        cause = getattr(excinfo.value, '__cause__', None)
        # Assert that the exception or its cause is one of the expected types
        assert isinstance(excinfo.value, (google_api_exceptions.NotFound, google_api_exceptions.InvalidArgument, google_api_exceptions.PermissionDenied)) or \
               isinstance(cause, (google_api_exceptions.NotFound, google_api_exceptions.InvalidArgument, google_api_exceptions.PermissionDenied)), \
               f"Expected Google API error for invalid embedding model, but got {type(excinfo.value)} with cause {type(cause)}"
        print(f"\nCaught expected error for invalid embedding model: {excinfo.value} (Cause: {cause})")

    @pytest.mark.asyncio
    async def test_embedding_cost_tracking_updates(self, client: GeminiClient, cost_tracker: GeminiCostTracker):
        """Verify cost tracker updates correctly for embedding calls (or handles missing metadata)."""
        content = "Track embedding cost for this sentence."
        task_type = "RETRIEVAL_DOCUMENT"
        target_model = TEST_EMBED_MODEL # Embedding model being tested

        initial_summary = cost_tracker.get_summary()
        initial_calls = initial_summary.get("usage_by_model", {}).get(target_model, {}).get("calls", 0)
        initial_input_units = initial_summary.get("usage_by_model", {}).get(target_model, {}).get("input_units", 0)
        initial_total_cost = initial_summary.get("total_estimated_cost", 0.0)
        metadata_warning_found = False # Flag to track if the expected warning occurs

        try:
            # Use warnings.catch_warnings to check for the specific warning
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always") # Ensure all warnings are caught
                response = await client.embed_content(content, task_type)
                assert isinstance(response, list) and len(response) > 0, "Embedding call failed"
                # Check if the specific warning about missing metadata was issued
                metadata_warning_found = any(
                    isinstance(w.message, UserWarning) and "Usage metadata not found" in str(w.message)
                    for w in caught_warnings
                )
                if metadata_warning_found:
                    print("\nNote: Live API did not return usage_metadata for embedding call, cost tracking likely skipped.")
        except Exception as e:
            pytest.fail(f"API call for embedding cost tracking test failed: {e}")

        final_summary = cost_tracker.get_summary()
        model_usage = final_summary.get("usage_by_model", {}).get(target_model, {})
        final_calls = model_usage.get("calls", 0)
        final_input_units = model_usage.get("input_units", 0)
        final_output_units = model_usage.get("output_units", 0) # Should always be 0 for embeddings
        final_total_cost = final_summary.get("total_estimated_cost", 0.0)

        # Assertions depend on whether metadata was likely available
        if metadata_warning_found:
            # If metadata was missing, cost tracker should NOT have been updated
            assert final_calls == initial_calls, "Call count updated unexpectedly despite missing metadata"
            assert final_input_units == initial_input_units, "Input units updated unexpectedly despite missing metadata"
            assert final_total_cost == pytest.approx(initial_total_cost), "Total cost changed unexpectedly despite missing metadata"
        else:
            # If metadata *was* available (no warning), expect updates
            print("\nNote: API returned usage_metadata for embedding call.")
            assert final_calls == initial_calls + 1, "Call count did not increment"
            assert final_input_units > initial_input_units, "Input units did not increase (or metadata missing despite no warning)"
            assert final_output_units == 0, "Output units should be 0 for embeddings"
            # Verify cost increase only if cost data is present
            model_cost_info = cost_tracker._model_costs.get(target_model)
            if isinstance(model_cost_info, ModelCostInfo):
                 assert final_total_cost > initial_total_cost, "Total estimated cost did not increase"
                 print(f"\nCost Tracking (Metadata Found): Calls={final_calls}, Input={final_input_units}, Est. Cost Increase={final_total_cost - initial_total_cost:.6f}")
            else:
                 print(f"\nWarning: Cost data for model '{target_model}' missing. Cannot verify cost increase.")
                 assert final_total_cost == pytest.approx(initial_total_cost)