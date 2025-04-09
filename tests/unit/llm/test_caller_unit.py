# File: tests/unit/llm/test_caller_unit.py

import pytest
import pytest_asyncio
import json
import warnings
import os
import asyncio
from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Optional, Any, List
from unittest.mock import MagicMock, AsyncMock, patch, call # For mocking

# Assuming pytest runs from root and pytest.ini sets pythonpath=src
# Adjust the import path based on your actual project structure if src isn't in pythonpath
try:
    from lean_automator.llm.caller import (
        GeminiCostTracker,
        ModelCostInfo,
        ModelUsageStats,
        GeminiClient,
        FALLBACK_EMBEDDING_MODEL, # Import fallback for testing
        FALLBACK_MAX_RETRIES,
        FALLBACK_BACKOFF_FACTOR
        # GenerationConfig, # Import necessary types if used by client directly
        # SafetySetting,
        # HarmCategory,
    )
    # Import the genai module itself for patching, and potential types
    import google.generativeai as genai
    import google.generativeai.types as genai_types
    from google.api_core import exceptions as google_api_exceptions # For API errors

except ImportError as e:
    # If the core dependencies are missing, skip all tests in this module.
    pytest.skip(f"Skipping test module: Failed to import lean_automator components or google.generativeai. Error: {e}", allow_module_level=True)


# --- Test Constants ---
VALID_COST_JSON = json.dumps({
    "model-a": {"input": 0.5, "output": 1.5}, # Cost per Million Tokens (Generation)
    "model-b": {"input": 1.0, "output": 2.0}, # (Generation)
    "models/text-embedding-004": {"input": 0.1} # Embedding model, output cost defaults to 0
})
VALID_COST_JSON_EMBED_EXPLICIT_ZERO = json.dumps({
    "models/text-embedding-004": {"input": 0.1, "output": 0.0} # Explicit zero output cost
})
INVALID_COST_JSON = '{"model-a": {"input": 0.5, "output": 1.5}, "model-b": }' # Malformed JSON
INCOMPLETE_COST_JSON = json.dumps({
    "model-a": {"output": 0.5}, # Missing input cost (invalid)
    "model-c": {"in": 0.1, "out": 0.2} # Wrong keys (invalid)
})
MOCK_API_KEY = "TEST_API_KEY"
MOCK_DEFAULT_GEN_MODEL = "mock-gen-model-default"
MOCK_DEFAULT_EMBED_MODEL = "models/text-embedding-004" # Use a realistic name
MOCK_OTHER_GEN_MODEL = "mock-gen-model-other"
MOCK_OTHER_EMBED_MODEL = "models/text-embedding-custom"

# --- Mock google-genai structures (simplified) ---
# (Keep existing mocks: MockUsageMetadata, MockPart, MockContent, MockCandidate, MockPromptFeedback, MockGenAIResponse)
@dataclass
class MockUsageMetadata:
    """Simplified mock for google.generativeai.types.GenerateContentResponse.usage_metadata."""
    prompt_token_count: int = 0
    candidates_token_count: int = 0 # Matches the attribute name in the actual library
    total_token_count: int = field(init=False) # Often present, calculated after init

    def __post_init__(self):
        self.total_token_count = self.prompt_token_count + self.candidates_token_count

# Mock usage metadata for embedding responses (often just total tokens)
@dataclass
class MockEmbeddingUsageMetadata:
    # Example: Actual metadata might only have 'total_token_count'
    # Adjust keys based on observed responses from genai.embed_content
    total_token_count: int = 0
    # Add other keys if the library provides them (e.g., billable_characters)

@dataclass
class MockPart:
    text: Optional[str] = None

@dataclass
class MockContent:
    parts: List[MockPart] = field(default_factory=list)
    role: Optional[str] = None
    @property
    def text(self) -> Optional[str]:
        if self.parts and hasattr(self.parts[0], 'text'): return self.parts[0].text
        return None

@dataclass
class MockCandidate:
    content: MockContent = field(default_factory=lambda: MockContent(parts=[MockPart()]))
    finish_reason: Optional[Any] = None
    safety_ratings: List[Any] = field(default_factory=list)
    @property
    def text(self) -> Optional[str]: return self.content.text

@dataclass
class MockPromptFeedback:
    block_reason: Optional[Any] = None
    safety_ratings: List[Any] = field(default_factory=list)

@dataclass
class MockGenAIResponse:
    candidates: List[MockCandidate] = field(default_factory=list)
    usage_metadata: Optional[MockUsageMetadata] = None
    prompt_feedback: Optional[MockPromptFeedback] = None
    _text_manually_set: bool = False
    _text: Optional[str] = None
    @property
    def text(self) -> Optional[str]:
        if self._text_manually_set: return self._text
        try:
            if self.candidates and self.candidates[0].text is not None: return self.candidates[0].text
            elif self.candidates: return None
            else: return None
        except (AttributeError, IndexError):
            warnings.warn("Could not automatically derive text from mock candidates.")
            return None
    @text.setter
    def text(self, value: Optional[str]):
        self._text_manually_set = True; self._text = value
        if value is not None:
            if not self.candidates: self.candidates = [MockCandidate(content=MockContent(parts=[MockPart(text=value)]))]
            else:
                if not self.candidates[0].content.parts: self.candidates[0].content.parts = [MockPart(text=value)]
                else:
                    if hasattr(self.candidates[0].content.parts[0], 'text'): self.candidates[0].content.parts[0].text = value
    def __post_init__(self):
        if self.usage_metadata:
             self.usage_metadata.total_token_count = ((self.usage_metadata.prompt_token_count or 0) + (self.usage_metadata.candidates_token_count or 0))


# --- Tests for GeminiCostTracker ---

def test_cost_tracker_init_valid_with_embedding():
    """Test initialization with valid cost JSON including embedding model."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON))
    assert "model-a" in tracker._model_costs
    assert "model-b" in tracker._model_costs
    assert MOCK_DEFAULT_EMBED_MODEL in tracker._model_costs

    assert isinstance(tracker._model_costs["model-a"], ModelCostInfo)
    assert tracker._model_costs["model-a"].input_cost_per_million_units == 0.5
    assert tracker._model_costs["model-a"].output_cost_per_million_units == 1.5

    assert isinstance(tracker._model_costs[MOCK_DEFAULT_EMBED_MODEL], ModelCostInfo)
    assert tracker._model_costs[MOCK_DEFAULT_EMBED_MODEL].input_cost_per_million_units == 0.1
    # Output cost should default to 0.0 if not specified in JSON
    assert tracker._model_costs[MOCK_DEFAULT_EMBED_MODEL].output_cost_per_million_units == 0.0

def test_cost_tracker_init_valid_embed_explicit_zero():
    """Test initialization with explicit zero output cost for embedding model."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON_EMBED_EXPLICIT_ZERO))
    assert MOCK_DEFAULT_EMBED_MODEL in tracker._model_costs
    assert tracker._model_costs[MOCK_DEFAULT_EMBED_MODEL].input_cost_per_million_units == 0.1
    assert tracker._model_costs[MOCK_DEFAULT_EMBED_MODEL].output_cost_per_million_units == 0.0 # Explicitly 0.0


def test_cost_tracker_init_invalid_json():
    """Test initialization with invalid JSON string, expects warning and empty costs."""
    # Check for the specific warning about the format of the entry
    with pytest.warns(UserWarning, match="Invalid cost format for model 'invalid_structure'"):
        tracker = GeminiCostTracker(model_costs_override={"invalid_structure": 123})
    assert not tracker._model_costs # Expect empty costs dictionary

def test_cost_tracker_init_incomplete_json():
    """Test initialization with valid JSON but incomplete/wrong data format per entry."""
    with pytest.warns(UserWarning) as record:
        tracker = GeminiCostTracker(model_costs_override=json.loads(INCOMPLETE_COST_JSON))

    # Expect that entries with invalid format are skipped and warnings are issued.
    assert not tracker._model_costs # Expect empty cost dict if all entries fail validation
    warning_messages = [str(w.message) for w in record]
    # Check that warnings were issued for each problematic entry
    assert any("Invalid cost format for model 'model-a'" in msg for msg in warning_messages) # Missing 'input'
    assert any("Invalid cost format for model 'model-c'" in msg for msg in warning_messages) # Wrong keys


def test_cost_tracker_record_usage_single_embed():
    """Test recording usage for a single embedding call (output units = 0)."""
    tracker = GeminiCostTracker(model_costs_override={}) # Costs not needed
    tracker.record_usage(MOCK_DEFAULT_EMBED_MODEL, 500, 0) # Input units, 0 output units
    assert MOCK_DEFAULT_EMBED_MODEL in tracker._usage_stats
    stats = tracker._usage_stats[MOCK_DEFAULT_EMBED_MODEL]
    assert isinstance(stats, ModelUsageStats)
    assert stats.calls == 1
    assert stats.prompt_tokens == 500 # Maps to input_units
    assert stats.completion_tokens == 0 # Maps to output_units

def test_cost_tracker_record_usage_multiple():
    """Test recording usage accumulates correctly across multiple calls and models."""
    tracker = GeminiCostTracker(model_costs_override={})
    tracker.record_usage("model-a", 100, 200)
    tracker.record_usage(MOCK_DEFAULT_EMBED_MODEL, 50, 0) # Embedding call
    tracker.record_usage("model-a", 10, 20)

    assert tracker._usage_stats["model-a"].calls == 2
    assert tracker._usage_stats["model-a"].prompt_tokens == 110
    assert tracker._usage_stats["model-a"].completion_tokens == 220

    assert tracker._usage_stats[MOCK_DEFAULT_EMBED_MODEL].calls == 1
    assert tracker._usage_stats[MOCK_DEFAULT_EMBED_MODEL].prompt_tokens == 50
    assert tracker._usage_stats[MOCK_DEFAULT_EMBED_MODEL].completion_tokens == 0

def test_cost_tracker_get_total_cost_simple_embed():
    """Test calculating total cost for an embedding model."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON))
    tracker.record_usage(MOCK_DEFAULT_EMBED_MODEL, 10_000_000, 0) # 10M input units

    # Expected cost: (10M units / 1M) * $0.1 + (0 units / 1M) * $0.0 = $1.0
    expected_cost = 1.0
    assert tracker.get_total_cost() == pytest.approx(expected_cost)

def test_cost_tracker_get_total_cost_mixed_models_with_embed():
    """Test calculating total cost involving generation and embedding models."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON))
    tracker.record_usage("model-a", 500_000, 1_000_000)       # Gen: 0.5M in, 1M out
    tracker.record_usage(MOCK_DEFAULT_EMBED_MODEL, 2_000_000, 0) # Embed: 2M in, 0 out

    # Expected cost A: (0.5M / 1M) * 0.5 + (1M / 1M) * 1.5 = 0.25 + 1.5 = 1.75
    # Expected cost Embed: (2M / 1M) * 0.1 + (0 / 1M) * 0.0 = 0.2
    expected_total_cost = 1.75 + 0.2
    assert tracker.get_total_cost() == pytest.approx(expected_total_cost)

def test_cost_tracker_get_total_cost_missing_model():
    """Test cost calculation warns and excludes costs for models without price info."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON)) # Costs for A, B, embedding
    tracker.record_usage("model-a", 1_000_000, 0) # Cost = 0.5
    tracker.record_usage("model-z", 500_000, 500_000) # No cost info

    with pytest.warns(UserWarning, match="Cost information missing for model 'model-z'"):
        total_cost = tracker.get_total_cost()

    expected_cost_a = (1_000_000 / 1_000_000.0) * 0.5
    assert total_cost == pytest.approx(expected_cost_a)

def test_cost_tracker_get_summary_with_embed():
    """Test the summary output format including an embedding model."""
    tracker = GeminiCostTracker(model_costs_override=json.loads(VALID_COST_JSON))
    tracker.record_usage("model-a", 500_000, 1_000_000)       # Gen
    tracker.record_usage(MOCK_DEFAULT_EMBED_MODEL, 1_000_000, 0) # Embed
    tracker.record_usage("model-z", 100_000, 100_000)       # No cost info

    with pytest.warns(UserWarning, match="Cost information missing for model 'model-z'"):
        summary = tracker.get_summary()

    assert "total_estimated_cost" in summary
    assert "usage_by_model" in summary
    assert "model-a" in summary["usage_by_model"]
    assert MOCK_DEFAULT_EMBED_MODEL in summary["usage_by_model"]
    assert "model-z" in summary["usage_by_model"]

    # Check gen model-a
    model_a_summary = summary["usage_by_model"]["model-a"]
    expected_cost_a = (0.5 * 0.5) + (1.0 * 1.5) # 1.75
    assert model_a_summary["estimated_cost"] == pytest.approx(expected_cost_a)
    assert model_a_summary["calls"] == 1
    assert model_a_summary["input_units"] == 500_000
    assert model_a_summary["output_units"] == 1_000_000

    # Check embed model
    model_embed_summary = summary["usage_by_model"][MOCK_DEFAULT_EMBED_MODEL]
    expected_cost_embed = (1.0 * 0.1) + (0 * 0.0) # 0.1
    assert model_embed_summary["estimated_cost"] == pytest.approx(expected_cost_embed)
    assert model_embed_summary["calls"] == 1
    assert model_embed_summary["input_units"] == 1_000_000
    assert model_embed_summary["output_units"] == 0

    # Check unknown model-z
    model_z_summary = summary["usage_by_model"]["model-z"]
    assert model_z_summary["estimated_cost"] == "Unknown (cost data missing)"

    # Check total cost
    assert summary["total_estimated_cost"] == pytest.approx(expected_cost_a + expected_cost_embed)


# --- Fixtures for GeminiClient Tests ---

@pytest.fixture
def mock_config_and_key(mocker):
    """
    Provides mock APP_CONFIG and get_gemini_api_key for client initialization.
    Simulates configuration being loaded correctly for base tests.
    """
    # Mock the APP_CONFIG dictionary as it would be loaded
    mock_config = {
        'llm': {
            'default_gemini_model': MOCK_DEFAULT_GEN_MODEL,
            'gemini_max_retries': 3,
            'gemini_backoff_factor': 0.01,
        },
        'embedding': {
            'default_embedding_model': MOCK_DEFAULT_EMBED_MODEL,
        },
        'costs': json.loads(VALID_COST_JSON) # Use the parsed JSON for costs
        # Add other potential config keys if needed by __init__
    }
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', mock_config, create=True)# Use create=True if APP_CONFIG might not exist yet

    # Mock the API key getter function
    mocker.patch('lean_automator.llm.caller.get_gemini_api_key', return_value=MOCK_API_KEY)

    # Return the mocked config for potential modification in specific tests if needed
    # Return a copy to prevent tests modifying the same dict affecting each other
    return mock_config.copy()


# Enhanced mock_genai_lib fixture to also mock genai.embed_content
@pytest.fixture
def mock_genai_lib(mocker):
    """
    Mocks google.generativeai library, GenerativeModel, configure, and embed_content.
    """
    mock_genai_module = MagicMock(spec=genai)
    mock_model_instance = MagicMock(spec=genai.GenerativeModel) # For generate calls

    # --- Mock generate_content (existing) ---
    mock_response_gen = MockGenAIResponse(
        usage_metadata=MockUsageMetadata(prompt_token_count=10, candidates_token_count=20),
         candidates=[MockCandidate(content=MockContent(parts=[MockPart(text="Default Mock Response")]))]
    )
    mock_model_instance.generate_content = MagicMock(return_value=mock_response_gen)
    mock_genai_module.GenerativeModel.return_value = mock_model_instance

    # --- Mock embed_content ---
    # Default mock response for embedding (single string input)
    mock_response_embed = {
        'embedding': [0.1, 0.2, 0.3],
        'usage_metadata': {'total_token_count': 5} # Assume metadata exists for base tests
    }
    # Mock the function directly on the module object
    mock_genai_module.embed_content = MagicMock(return_value=mock_response_embed)

    # --- Mock configure (existing) ---
    mock_genai_module.configure = MagicMock()

    # --- Patch 'genai' where imported ---
    mocker.patch('lean_automator.llm.caller.genai', mock_genai_module)

    # --- Mock Types (existing) ---
    # mocker.patch('lean_automator.llm_call.GenerationConfig', spec=GenerationConfig)
    # mocker.patch('lean_automator.llm_call.SafetySetting', spec=SafetySetting)
    # mocker.patch('lean_automator.llm_call.HarmCategory', spec=HarmCategory)
    mock_genai_types_module = MagicMock(spec=genai_types)
    # Ensure SafetySettingDict is mockable if DEFAULT_SAFETY_SETTINGS is used
    mock_genai_types_module.SafetySettingDict = MagicMock()
    # Mock HarmCategory and HarmBlockThreshold if used in DEFAULT_SAFETY_SETTINGS
    mock_genai_types_module.HarmCategory = MagicMock(
        HARM_CATEGORY_HARASSMENT='HARM_CATEGORY_HARASSMENT',
        HARM_CATEGORY_HATE_SPEECH='HARM_CATEGORY_HATE_SPEECH',
        HARM_CATEGORY_SEXUALLY_EXPLICIT='HARM_CATEGORY_SEXUALLY_EXPLICIT',
        HARM_CATEGORY_DANGEROUS_CONTENT='HARM_CATEGORY_DANGEROUS_CONTENT'
    )
    mock_genai_types_module.HarmBlockThreshold = MagicMock(
        BLOCK_MEDIUM_AND_ABOVE='BLOCK_MEDIUM_AND_ABOVE'
    )
    mocker.patch('lean_automator.llm.caller.genai_types', mock_genai_types_module)
    # Ensure the actual exception classes are available to be caught
    mocker.patch('lean_automator.llm.caller.google_api_exceptions.ResourceExhausted', google_api_exceptions.ResourceExhausted)
    mocker.patch('lean_automator.llm.caller.google_api_exceptions.GoogleAPIError', google_api_exceptions.GoogleAPIError)

    # Return mocks needed for assertions
    return mock_genai_module, mock_model_instance, mock_genai_module.embed_content


@pytest.fixture
def mock_asyncio_sleep(mocker):
    """Mocks asyncio.sleep."""
    return mocker.patch('asyncio.sleep', new_callable=AsyncMock)

@pytest.fixture
def patch_asyncio_to_thread(mocker):
    """ Patches asyncio.to_thread to run sync functions directly (async wrapper). """
    async def run_sync_wrapper(func, *args, **kwargs): return func(*args, **kwargs)
    return mocker.patch('asyncio.to_thread', side_effect=run_sync_wrapper)


# --- Tests for GeminiClient ---

# -- Initialization Tests --

def test_client_init_success(mocker, mock_config_and_key, mock_genai_lib): # Use new fixture
    """Test successful client initialization including embedding model."""
    mock_genai_module, _, _ = mock_genai_lib

    # Client should load config from the mocked APP_CONFIG and get_gemini_api_key
    client = GeminiClient()

    assert client.api_key == MOCK_API_KEY
    # This assertion should now pass as APP_CONFIG provides the mock model
    assert client.default_generation_model == MOCK_DEFAULT_GEN_MODEL
    assert client.default_embedding_model == MOCK_DEFAULT_EMBED_MODEL # Check embedding model
    # Check retries/backoff came from mock_config_and_key's APP_CONFIG mock
    assert client.max_retries == 3
    assert client.backoff_factor == 0.01
    assert isinstance(client.cost_tracker, GeminiCostTracker)
    # Costs are now loaded via APP_CONFIG['costs']
    assert "model-a" in client.cost_tracker._model_costs
    assert MOCK_DEFAULT_EMBED_MODEL in client.cost_tracker._model_costs

    mock_genai_module.configure.assert_called_once_with(api_key=MOCK_API_KEY)

def test_client_init_missing_api_key(mocker, mock_genai_lib): # Doesn't need base config fixture
    """Test client initialization raises ValueError if GEMINI_API_KEY is missing."""
    _, _, _ = mock_genai_lib
    # Mock the key getter to return None
    mocker.patch('lean_automator.llm.caller.get_gemini_api_key', return_value=None)
    # Mock APP_CONFIG to provide other values so only the key is missing
    mock_config = {
        'llm': {'default_gemini_model': MOCK_DEFAULT_GEN_MODEL, 'gemini_max_retries': 3, 'gemini_backoff_factor': 0.1},
        'embedding': {'default_embedding_model': MOCK_DEFAULT_EMBED_MODEL},
        'costs': {}
    }
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', mock_config, create=True)

    with pytest.raises(ValueError, match="Gemini API key is missing"):
        GeminiClient()

def test_client_init_missing_default_gen_model(mocker, mock_config_and_key, mock_genai_lib): # Use base mock
    """Test client initialization raises ValueError if default_gemini_model is missing in APP_CONFIG."""
    _, _, _ = mock_genai_lib
    # Get the base mocked config and remove the key for this test
    base_config = mock_config_and_key.copy() # Use copy
    if 'llm' in base_config and 'default_gemini_model' in base_config['llm']:
        del base_config['llm']['default_gemini_model']
    # If 'llm' key itself is missing, that's fine too

    # Re-patch APP_CONFIG with the modified version for this specific test
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', base_config)
    # Key getter is already mocked by the fixture

    # This should now raise the error because the value isn't found in arg (None) or APP_CONFIG
    with pytest.raises(ValueError, match="Default Gemini generation model is missing"):
        GeminiClient() # Call without providing the argument


def test_client_init_missing_default_embed_model_uses_fallback(mocker, mock_config_and_key, mock_genai_lib): # Use base mock
    """Test client uses fallback embedding model if default_embedding_model is missing in APP_CONFIG."""
    _, _, _ = mock_genai_lib
    # Get the base mocked config and remove the key for this test
    base_config = mock_config_and_key.copy() # Use copy
    if 'embedding' in base_config and 'default_embedding_model' in base_config['embedding']:
        del base_config['embedding']['default_embedding_model']
    # Re-patch APP_CONFIG with the modified version
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', base_config)
    # Key getter is already mocked

    # Now, the __init__ should not find the model in arg (None) or APP_CONFIG
    # It should hit the 'if not _emb_model_name:' block and issue the fallback warning
    with pytest.warns(UserWarning, match=f"Default embedding model not set .* Using fallback: {FALLBACK_EMBEDDING_MODEL}"):
        client = GeminiClient()

    assert client.default_embedding_model == FALLBACK_EMBEDDING_MODEL


def test_client_init_embed_model_adds_prefix(mocker, mock_config_and_key, mock_genai_lib): # Use base mock
    """Test client adds 'models/' prefix to embedding model if missing in APP_CONFIG."""
    _, _, _ = mock_genai_lib
    embed_model_no_prefix = "text-embedding-004"
    expected_model_with_prefix = f"models/{embed_model_no_prefix}"

    # Modify the mocked config for this test case
    base_config = mock_config_and_key.copy() # Use copy
    base_config['embedding']['default_embedding_model'] = embed_model_no_prefix # Set value without prefix
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', base_config)
    # Key getter is already mocked

    # Now __init__ should find the model name in APP_CONFIG, detect the missing prefix,
    # add it, and issue the warning about resolution.
    with pytest.warns(UserWarning, match=f"Resolved embedding model '{embed_model_no_prefix}' did not start with 'models/'. Using '{expected_model_with_prefix}'"):
        client = GeminiClient()

    assert client.default_embedding_model == expected_model_with_prefix

def test_client_init_with_args(mocker, mock_genai_lib): # Doesn't need config fixture (args override)
    """Test initialization using direct arguments overrides config/environment."""
    mock_genai_module, _, _ = mock_genai_lib
    # Mock APP_CONFIG and key getter to return dummy values to ensure args are used
    mocker.patch('lean_automator.llm.caller.get_gemini_api_key', return_value="DUMMY_KEY_FROM_GETTER")
    mocker.patch('lean_automator.llm.caller.APP_CONFIG', {'llm': {'default_gemini_model': 'dummy-gen-model-config'}}, create=True)

    custom_api_key = "ARG_API_KEY"
    custom_gen_model = "arg-gen-model"
    custom_embed_model = "models/arg-embed-model"
    custom_retries = 1
    custom_backoff = 5.0
    custom_tracker = GeminiCostTracker(model_costs_override={})

    client = GeminiClient(
        api_key=custom_api_key,
        default_generation_model=custom_gen_model,
        default_embedding_model=custom_embed_model,
        max_retries=custom_retries,
        backoff_factor=custom_backoff,
        cost_tracker=custom_tracker
    )

    assert client.api_key == custom_api_key
    assert client.default_generation_model == custom_gen_model
    assert client.default_embedding_model == custom_embed_model
    assert client.max_retries == custom_retries
    assert client.backoff_factor == custom_backoff
    assert client.cost_tracker is custom_tracker
    mock_genai_module.configure.assert_called_once_with(api_key=custom_api_key)


# -- Generate Method Tests --

@pytest.mark.asyncio
async def test_client_generate_success_default_model(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use new fixture
    """Test successful generation using the default model and verify cost tracking."""
    mock_genai_module, mock_model_instance, _ = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')
    # Client init should now use MOCK_DEFAULT_GEN_MODEL from mocked APP_CONFIG
    client = GeminiClient()
    prompt = "Test prompt"
    response_text = await client.generate(prompt)

    assert response_text == "Default Mock Response"
    # This assertion should now pass
    mock_genai_module.GenerativeModel.assert_called_once_with(MOCK_DEFAULT_GEN_MODEL, system_instruction=None) # Check system prompt default
    mock_model_instance.generate_content.assert_called_once()
    _, call_kwargs = mock_model_instance.generate_content.call_args
    expected_contents = [{'role': 'user', 'parts': [prompt]}]
    assert call_kwargs['contents'] == expected_contents

    # Check cost tracker call uses the correct default model name
    mock_record_usage.assert_called_once_with(client.cost_tracker, MOCK_DEFAULT_GEN_MODEL, 10, 20)
    mock_asyncio_sleep.assert_not_called()

@pytest.mark.asyncio
async def test_client_generate_with_system_prompt(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use new fixture
    """Test generation call passes system_prompt correctly."""
    mock_genai_module, mock_model_instance, _ = mock_genai_lib
    # Client init should now use MOCK_DEFAULT_GEN_MODEL from mocked APP_CONFIG
    client = GeminiClient()
    prompt = "User query"
    system_p = "Act as a helpful assistant."

    await client.generate(prompt, system_prompt=system_p)

    # Verify GenerativeModel was initialized with the system_instruction
    # This assertion should now pass
    mock_genai_module.GenerativeModel.assert_called_once_with(MOCK_DEFAULT_GEN_MODEL, system_instruction=system_p)
    # Verify generate_content was called (contents argument checked in other tests)
    mock_model_instance.generate_content.assert_called_once()


# -- Embed Content Method Tests --

@pytest.mark.asyncio
async def test_client_embed_success_single_string(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test successful embedding for a single string input."""
    _, _, mock_embed_func = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')

    # Client init uses default from mocked APP_CONFIG
    client = GeminiClient()
    content_to_embed = "Embed this text."
    task_type = "RETRIEVAL_QUERY"

    # Configure mock response for single string
    expected_vector = [0.5, 0.6, 0.7]
    mock_embed_response = {'embedding': expected_vector, 'usage_metadata': {'total_token_count': 8}}
    mock_embed_func.return_value = mock_embed_response

    result_embeddings = await client.embed_content(content_to_embed, task_type)

    # Verify the result structure (list containing one vector)
    assert isinstance(result_embeddings, list)
    assert len(result_embeddings) == 1
    assert result_embeddings[0] == expected_vector

    # Verify genai.embed_content was called correctly with the default model
    mock_embed_func.assert_called_once_with(
        model=MOCK_DEFAULT_EMBED_MODEL,
        content=content_to_embed,
        task_type=task_type
        # title=None, output_dimensionality=None implicitly checked by not being present
    )

    # Verify cost tracker (using total_token_count as input, 0 output)
    mock_record_usage.assert_called_once_with(
        client.cost_tracker,
        MOCK_DEFAULT_EMBED_MODEL,
        8, # input_units from usage_metadata.total_token_count
        0  # output_units is always 0 for embeddings
    )
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_client_embed_success_list_strings(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test successful embedding for a list of strings."""
    _, _, mock_embed_func = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')

    client = GeminiClient()
    contents_to_embed = ["Embed text one.", "Embed text two."]
    task_type = "SEMANTIC_SIMILARITY"

    # Configure mock response for list input - assume key is 'embedding' containing a list
    expected_vectors = [[0.1, 0.2], [0.3, 0.4]]
    mock_embed_response = {'embedding': expected_vectors, 'usage_metadata': {'total_token_count': 15}}
    mock_embed_func.return_value = mock_embed_response

    result_embeddings = await client.embed_content(contents_to_embed, task_type)

    # Verify result structure (list containing vectors for each input string)
    assert isinstance(result_embeddings, list)
    assert len(result_embeddings) == 2
    assert result_embeddings == expected_vectors

    # Verify genai.embed_content call with default model
    mock_embed_func.assert_called_once_with(
        model=MOCK_DEFAULT_EMBED_MODEL,
        content=contents_to_embed,
        task_type=task_type
    )

    # Verify cost tracker
    mock_record_usage.assert_called_once_with(client.cost_tracker, MOCK_DEFAULT_EMBED_MODEL, 15, 0)
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_client_embed_with_optional_args(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test embedding call passes optional title and output_dimensionality."""
    _, _, mock_embed_func = mock_genai_lib
    client = GeminiClient()
    content = "Document content"
    task = "RETRIEVAL_DOCUMENT"
    doc_title = "My Document Title"
    dims = 256

    # Mock response doesn't need to reflect dimensionality truncation for this call test
    mock_embed_func.return_value = {'embedding': [0.1]*dims, 'usage_metadata': {'total_token_count': 10}}

    await client.embed_content(content, task, title=doc_title, output_dimensionality=dims)

    # Verify optional args passed to genai.embed_content along with default model
    mock_embed_func.assert_called_once_with(
        model=MOCK_DEFAULT_EMBED_MODEL,
        content=content,
        task_type=task,
        title=doc_title, # Check title passed
        output_dimensionality=dims # Check dimensionality passed
    )

@pytest.mark.asyncio
async def test_client_embed_title_wrong_task_warns(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test providing title with incorrect task_type issues a warning."""
    _, _, mock_embed_func = mock_genai_lib
    mock_warnings = mocker.patch('warnings.warn')
    client = GeminiClient()
    content = "Query text"
    task = "RETRIEVAL_QUERY" # Not RETRIEVAL_DOCUMENT
    doc_title = "Should not be used"

    # Mock response needed even though we check the call args and warning
    mock_embed_func.return_value = {'embedding': [0.1], 'usage_metadata': {'total_token_count': 3}}

    await client.embed_content(content, task, title=doc_title)

    # Verify warning was issued
    # Check all warnings, including potential ones from __init__ if mocks change
    warning_messages = [str(call_args[0]) for call_args, call_kwargs in mock_warnings.call_args_list if call_args]
    assert any("Ignoring 'title' argument as task_type is 'RETRIEVAL_QUERY'" in msg for msg in warning_messages)

    # Verify title was NOT passed to the underlying API call
    mock_embed_func.assert_called_once_with(
        model=MOCK_DEFAULT_EMBED_MODEL,
        content=content,
        task_type=task
        # title should be absent here
    )

@pytest.mark.asyncio
async def test_client_embed_model_needs_prefix(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test calling embed with a model name needing 'models/' prefix."""
    _, _, mock_embed_func = mock_genai_lib
    mock_warnings = mocker.patch('warnings.warn')
    client = GeminiClient() # Initializes with default model, but we override in call
    content = "Text"
    task = "RETRIEVAL_QUERY"
    model_no_prefix = "text-embedding-custom" # Different from default
    expected_model_with_prefix = f"models/{model_no_prefix}"

    # Mock response
    mock_embed_func.return_value = {'embedding': [0.9], 'usage_metadata': {'total_token_count': 1}}

    await client.embed_content(content, task, model=model_no_prefix) # Override model

    # Verify warning was issued about the prefix based on the *called* model
    warning_messages = [str(call_args[0]) for call_args, call_kwargs in mock_warnings.call_args_list if call_args]
    assert any(f"Embedding model name '{model_no_prefix}' should ideally start with 'models/'" in msg for msg in warning_messages)

    # Verify the API call used the name *with* the prefix (as per current implementation which warns but still uses it)
    mock_embed_func.assert_called_once_with(
        model=model_no_prefix, # The code currently warns but doesn't add prefix in embed_content itself
        content=content,
        task_type=task
    )

@pytest.mark.asyncio
async def test_client_embed_retry_logic(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test retry mechanism for embed_content."""
    _, _, mock_embed_func = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')
    mock_warnings = mocker.patch('warnings.warn')

    max_retries = 1
    backoff = 0.01
    # Initialize client with custom retry settings
    client = GeminiClient(max_retries=max_retries, backoff_factor=backoff) # Uses default models from mock_config_and_key

    expected_vector = [0.8, 0.9]
    mock_response_success = {'embedding': expected_vector, 'usage_metadata': {'total_token_count': 6}}
    api_error = google_api_exceptions.ResourceExhausted("Simulated Quota Error") # Use a specific retryable error

    mock_embed_func.side_effect = [api_error, mock_response_success]

    content = "Retry embed test"
    task = "RETRIEVAL_QUERY"
    result = await client.embed_content(content, task) # Uses default embedding model

    assert result == [expected_vector]
    assert mock_embed_func.call_count == 2 # Initial + 1 retry
    mock_asyncio_sleep.assert_called_once_with(pytest.approx(backoff * (2**0)))
    # Check cost tracker called with default model name
    mock_record_usage.assert_called_once_with(client.cost_tracker, MOCK_DEFAULT_EMBED_MODEL, 6, 0)

    # Verify warnings
    warning_messages = [str(call_args[0]) for call_args, call_kwargs in mock_warnings.call_args_list if call_args]
    assert any(f"API Quota/Rate Limit Error for {MOCK_DEFAULT_EMBED_MODEL} on attempt 1/2" in msg for msg in warning_messages)
    assert any(f"Retrying API call for {MOCK_DEFAULT_EMBED_MODEL} in {backoff * (2**0):.2f} seconds" in msg for msg in warning_messages)


@pytest.mark.asyncio
async def test_client_embed_all_retries_fail(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test embed_content raises exception after exhausting retries."""
    _, _, mock_embed_func = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')

    max_retries = 2
    backoff = 0.01
    client = GeminiClient(max_retries=max_retries, backoff_factor=backoff) # Uses defaults

    simulated_error = Exception("Persistent Embed Error")
    mock_embed_func.side_effect = simulated_error

    content = "Failure embed test"
    task = "RETRIEVAL_QUERY"
    # The final exception should be wrapped by the client's generic message, mentioning the default model
    with pytest.raises(Exception, match=f"API call to embedding model '{MOCK_DEFAULT_EMBED_MODEL}' failed after retries or during processing.") as exc_info:
        await client.embed_content(content, task)

    # Check that the original error is the cause
    assert exc_info.value.__cause__ is simulated_error

    total_expected_attempts = 1 + max_retries
    assert mock_embed_func.call_count == total_expected_attempts
    assert mock_asyncio_sleep.call_count == max_retries
    mock_record_usage.assert_not_called()

@pytest.mark.asyncio
async def test_client_embed_invalid_response_format(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test embed_content raises ValueError for invalid API response."""
    _, _, mock_embed_func = mock_genai_lib
    client = GeminiClient() # Uses defaults

    # Mock response missing 'embedding' or 'embeddings' keys
    mock_embed_func.return_value = {'wrong_key': [0.1]}

    with pytest.raises(ValueError, match=f"API call for {MOCK_DEFAULT_EMBED_MODEL} succeeded but the response dictionary is missing the expected 'embedding' or 'embeddings' key"):
        await client.embed_content("Test", "RETRIEVAL_QUERY")

    mock_embed_func.assert_called_once()
    mock_asyncio_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_client_embed_input_type_validation(mocker, mock_config_and_key, mock_genai_lib): # Use fixture
    """Test embed_content raises TypeError for invalid input types."""
    _, _, mock_embed_func = mock_genai_lib
    client = GeminiClient() # Uses defaults

    with pytest.raises(TypeError, match="Input 'contents' must be a string or a list of strings"):
        await client.embed_content(123, "RETRIEVAL_QUERY") # Invalid type

    with pytest.raises(TypeError, match="If 'contents' is provided as a list, all its items must be strings"):
        await client.embed_content(["a string", 123], "RETRIEVAL_QUERY") # List with invalid item

    mock_embed_func.assert_not_called() # Should fail before API call


@pytest.mark.asyncio
async def test_client_embed_missing_usage_metadata(mocker, mock_config_and_key, mock_genai_lib, mock_asyncio_sleep, patch_asyncio_to_thread): # Use fixture
    """Test embed_content handles missing usage metadata gracefully."""
    _, _, mock_embed_func = mock_genai_lib
    mock_record_usage = mocker.spy(GeminiCostTracker, 'record_usage')
    mock_warnings = mocker.patch('warnings.warn')
    client = GeminiClient() # Uses defaults

    # Mock response with valid embedding but no metadata
    expected_vector = [0.7, 0.8]
    mock_embed_response = {'embedding': expected_vector} # No 'usage_metadata' key
    mock_embed_func.return_value = mock_embed_response

    result = await client.embed_content("Test no metadata", "RETRIEVAL_QUERY")

    assert result == [expected_vector] # Result should still be correct

    # Verify warning about missing metadata for the default model
    warning_messages = [str(call_args[0]) for call_args, call_kwargs in mock_warnings.call_args_list if call_args]
    assert any(f"Usage metadata not found in response for embedding model '{MOCK_DEFAULT_EMBED_MODEL}'" in msg for msg in warning_messages)

    # Verify cost tracker was NOT called since metadata was missing
    mock_record_usage.assert_not_called()

    mock_asyncio_sleep.assert_not_called()