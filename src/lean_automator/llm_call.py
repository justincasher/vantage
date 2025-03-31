# File: llm_call.py

import os
import asyncio
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any, Union, List

# Google GenAI Library
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_api_exceptions # For specific API errors

# These can serve as hardcoded fallbacks if environment variables AND arguments are missing/invalid
FALLBACK_MAX_RETRIES = 3
FALLBACK_BACKOFF_FACTOR = 1.0
FALLBACK_MODEL_COSTS_JSON = '{}'
FALLBACK_EMBEDDING_MODEL = 'models/text-embedding-004' # Default if env var is missing

# Default safety settings - defined here for use as default in __init__
# Ensure genai_types is available before creating the list
DEFAULT_SAFETY_SETTINGS = [
    genai_types.SafetySettingDict(category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    genai_types.SafetySettingDict(category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    genai_types.SafetySettingDict(category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    genai_types.SafetySettingDict(category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=genai_types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
] if genai_types else None

# --- Cost Tracking ---

@dataclass
class ModelUsageStats:
    """Stores usage statistics for a specific model."""
    calls: int = 0
    prompt_tokens: int = 0 # Represents input tokens/units
    completion_tokens: int = 0 # Represents output tokens/units (often 0 for embeddings)

@dataclass
class ModelCostInfo:
    """Stores cost per MILLION units (tokens/chars) for a specific model."""
    input_cost_per_million_units: float
    output_cost_per_million_units: float

class GeminiCostTracker:
    """
    Tracks API call counts, token usage, and estimated costs for Gemini models,
    using costs specified per million units. Handles generative and embedding models.
    """
    def __init__(self, model_costs_json: Optional[str] = None):
        """
        Initializes the tracker.

        Args:
            model_costs_json: A JSON string mapping model names to their costs per million units.
                              If None, reads from GEMINI_MODEL_COSTS environment variable.
                              Example: '{"gemini-1.5-flash-latest": {"input": 0.35, "output": 0.70}, "models/text-embedding-004": {"input": 0.10, "output": 0.0}}'
        """
        effective_costs_json = model_costs_json if model_costs_json is not None else os.getenv('GEMINI_MODEL_COSTS', FALLBACK_MODEL_COSTS_JSON)
        self._usage_stats: Dict[str, ModelUsageStats] = {}
        self._model_costs: Dict[str, ModelCostInfo] = {}
        self._parse_model_costs(effective_costs_json)

    def _parse_model_costs(self, json_string: str):
        """Parses the model costs JSON string (expecting cost per million units)."""
        try:
            costs_dict = json.loads(json_string)
            for model, costs in costs_dict.items():
                # Allow 'output' to be missing or 0 for embedding models
                if isinstance(costs, dict) and 'input' in costs:
                    input_cost = float(costs['input'])
                    # Default output cost to 0 if not specified
                    output_cost = float(costs.get('output', 0.0))
                    self._model_costs[model] = ModelCostInfo(
                        input_cost_per_million_units=input_cost,
                        output_cost_per_million_units=output_cost
                    )
                else:
                    warnings.warn(f"Invalid cost format for model '{model}' in GEMINI_MODEL_COSTS. Expected at least {{'input': float}}. Found: {costs}")
        except json.JSONDecodeError:
            warnings.warn("Failed to parse GEMINI_MODEL_COSTS JSON string. Costs will not be tracked accurately.")
        except Exception as e:
            warnings.warn(f"Error processing GEMINI_MODEL_COSTS: {e}")

    def record_usage(self, model: str, input_units: int, output_units: int):
        """Records a successful API call and its unit usage."""
        if model not in self._usage_stats:
            self._usage_stats[model] = ModelUsageStats()

        stats = self._usage_stats[model]
        stats.calls += 1
        stats.prompt_tokens += input_units # Map input_units -> prompt_tokens
        stats.completion_tokens += output_units # Map output_units -> completion_tokens

    def get_total_cost(self) -> float:
        """Calculates the estimated total cost based on recorded usage and known model costs per million units."""
        total_cost = 0.0
        for model, stats in self._usage_stats.items():
            if model in self._model_costs:
                costs = self._model_costs[model]
                total_cost += (stats.prompt_tokens / 1_000_000.0) * costs.input_cost_per_million_units + \
                              (stats.completion_tokens / 1_000_000.0) * costs.output_cost_per_million_units
            else:
                warnings.warn(f"Cost information missing for model '{model}'. Usage for this model is not included in total cost.")
        return total_cost

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary dictionary of usage and estimated costs."""
        total_estimated_cost = self.get_total_cost() # Ensure warnings are potentially triggered
        summary: Dict[str, Any] = {
            "total_estimated_cost": total_estimated_cost,
            "usage_by_model": {}
        }
        for model, stats in self._usage_stats.items():
            model_summary = {
                "calls": stats.calls,
                "input_units": stats.prompt_tokens,
                "output_units": stats.completion_tokens, # Usually 0 for embeddings
                "estimated_cost": 0.0
            }
            if model in self._model_costs:
                costs = self._model_costs[model]
                model_summary["estimated_cost"] = (stats.prompt_tokens / 1_000_000.0) * costs.input_cost_per_million_units + \
                                                  (stats.completion_tokens / 1_000_000.0) * costs.output_cost_per_million_units
            else:
                 model_summary["estimated_cost"] = "Unknown (cost data missing)"

            summary["usage_by_model"][model] = model_summary
        return summary

# --- Gemini Client ---

class GeminiClient:
    """
    Client for interacting with Google's Gemini API (Generation and Embedding),
    including async calls, retries, and cost tracking integration.
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 default_generation_model: Optional[str] = None,
                 default_embedding_model: Optional[str] = None,
                 max_retries: Optional[int] = None,
                 backoff_factor: Optional[float] = None,
                 cost_tracker: Optional[GeminiCostTracker] = None,
                 safety_settings: Optional[list] = DEFAULT_SAFETY_SETTINGS):
        """
        Initializes the Gemini Client. Reads required config from environment if not passed.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
            default_generation_model: Default Gemini model for generation.
                                      If None, reads from DEFAULT_GEMINI_MODEL env var.
            default_embedding_model: Default Gemini model for embeddings.
                                     If None, reads from DEFAULT_EMBEDDING_MODEL env var or uses fallback.
                                     Ensures 'models/' prefix if needed for cost tracking consistency.
            max_retries: Max retry attempts. If None, reads from GEMINI_MAX_RETRIES env var or defaults.
            backoff_factor: Backoff factor for retries. If None, reads from GEMINI_BACKOFF_FACTOR env var or defaults.
            cost_tracker: An instance of GeminiCostTracker to record usage (optional).
            safety_settings: Default Gemini safety settings for generation (optional).
        """
        if not genai:
             raise RuntimeError("google.generativeai package is required but not found.")

        # --- Configuration Loading ---
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
             raise ValueError("Gemini API key is missing. Set via argument or GEMINI_API_KEY environment variable.")

        self.default_generation_model = default_generation_model or os.getenv('DEFAULT_GEMINI_MODEL')
        if not self.default_generation_model:
             raise ValueError("Default Gemini generation model is missing. Set via argument or DEFAULT_GEMINI_MODEL environment variable.")

        _emb_model_name = default_embedding_model or os.getenv('DEFAULT_EMBEDDING_MODEL')
        if not _emb_model_name:
             warnings.warn(f"Default embedding model not set via argument or DEFAULT_EMBEDDING_MODEL env var. Using fallback: {FALLBACK_EMBEDDING_MODEL}")
             self.default_embedding_model = FALLBACK_EMBEDDING_MODEL
        else:
             # Ensure the model name starts with 'models/' for consistency if it doesn't already
             # This helps align with cost dictionary keys like "models/text-embedding-004"
             if not _emb_model_name.startswith('models/'):
                 self.default_embedding_model = f'models/{_emb_model_name}'
                 warnings.warn(f"DEFAULT_EMBEDDING_MODEL '{_emb_model_name}' did not start with 'models/'. Using '{self.default_embedding_model}' for consistency.")
             else:
                  self.default_embedding_model = _emb_model_name

        # Max Retries
        if max_retries is not None:
            self.max_retries = max_retries
        else:
            try:
                _retries_str = os.getenv('GEMINI_MAX_RETRIES')
                self.max_retries = int(_retries_str) if _retries_str is not None else FALLBACK_MAX_RETRIES
            except (ValueError, TypeError):
                warnings.warn(f"Invalid GEMINI_MAX_RETRIES value '{os.getenv('GEMINI_MAX_RETRIES')}'. Using default {FALLBACK_MAX_RETRIES}.")
                self.max_retries = FALLBACK_MAX_RETRIES
        self.max_retries = max(0, self.max_retries) # Ensure non-negative

        # Backoff Factor
        if backoff_factor is not None:
            self.backoff_factor = backoff_factor
        else:
            try:
                 _backoff_str = os.getenv('GEMINI_BACKOFF_FACTOR')
                 self.backoff_factor = float(_backoff_str) if _backoff_str is not None else FALLBACK_BACKOFF_FACTOR
            except (ValueError, TypeError):
                 warnings.warn(f"Invalid GEMINI_BACKOFF_FACTOR value '{os.getenv('GEMINI_BACKOFF_FACTOR')}'. Using default {FALLBACK_BACKOFF_FACTOR}.")
                 self.backoff_factor = FALLBACK_BACKOFF_FACTOR
        self.backoff_factor = max(0.0, self.backoff_factor) # Ensure non-negative

        # --- Initialization ---
        self.cost_tracker = cost_tracker if cost_tracker is not None else GeminiCostTracker()
        self.safety_settings = safety_settings # Used by generate method

        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
             raise RuntimeError(f"Failed to configure Google GenAI client: {e}") from e

    # --- Private Helper for Retries ---
    async def _execute_with_retry(self, api_call_func, *args, _model_name_for_log='unknown_model', **kwargs):
        """ Executes an async API call with retry logic. """
        final_error: Optional[Exception] = None
        model_name = _model_name_for_log

        total_attempts = self.max_retries + 1
        for attempt in range(total_attempts):
            try:
                # Use asyncio.to_thread to run the synchronous SDK call in a separate thread
                response = await asyncio.to_thread(api_call_func, *args, **kwargs)
                return response # Success

            except google_api_exceptions.ResourceExhausted as e:
                # Specific handling for rate limits / quota errors - retryable
                final_error = e
                warnings.warn(f"API Quota/Rate Limit Error for {model_name} on attempt {attempt + 1}/{total_attempts}: {e}")
                # Continue to retry logic below

            except google_api_exceptions.GoogleAPIError as e:
                 # Catch other Google API errors (e.g., server errors, bad requests)
                 final_error = e
                 # Decide if retryable based on status code maybe? For now, retry most.
                 # 4xx errors are typically not retryable (Bad Request, Not Found, Invalid Argument)
                 if 400 <= getattr(e, 'code', 0) < 500:
                      warnings.warn(f"API Client Error (4xx) for {model_name} on attempt {attempt + 1}/{total_attempts}: {e}. Not retrying.")
                      break # Don't retry client errors
                 else: # Retry server errors (5xx) or unknown API errors
                     warnings.warn(f"API Server/Unknown Error for {model_name} on attempt {attempt + 1}/{total_attempts}: {e}")
                 # Continue to retry logic below

            except Exception as e:
                # Catch broader exceptions (network issues, unexpected errors)
                final_error = e
                warnings.warn(f"Unexpected Error during API call for {model_name} on attempt {attempt + 1}/{total_attempts}: {e}")
                # Continue to retry logic below


            # --- Retry Logic ---
            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2 ** attempt)
                retries_remaining = self.max_retries - attempt
                warnings.warn(
                    f"Retrying API call for {model_name} in {sleep_time:.2f} seconds... ({retries_remaining} retries remaining)"
                )
                await asyncio.sleep(sleep_time)
            else:
                # This was the final attempt
                warnings.warn(f"API call for {model_name} failed on the final attempt ({attempt + 1}/{total_attempts}).")
                break # Exit loop after final attempt

        # If loop finished without returning, raise the last captured error
        raise final_error if final_error is not None else Exception(f"Unknown error during API call to {model_name} after {total_attempts} attempts")


    # --- Public API Methods ---

    async def generate(self,
                       prompt: str,
                       *,
                       model: Optional[str] = None,
                       system_prompt: Optional[str] = None, # Note: system_instruction is arg name
                       generation_config_override: Optional[Dict[str, Any]] = None,
                       safety_settings_override: Optional[list] = None
                       ) -> str:
        """
        Generates content using the specified Gemini model, with retry logic.

        Args:
            prompt: The main user prompt.
            model: Specific Gemini model name. Defaults to client's default_generation_model.
            system_prompt: Optional system instruction.
            generation_config_override: Optional dictionary to override generation config.
            safety_settings_override: Optional list to override safety settings.

        Returns:
            The generated text content.

        Raises:
            Exception: If the API call fails after all retry attempts.
            ValueError: If the response is invalid (blocked, empty) or model init fails.
        """
        effective_model = model or self.default_generation_model
        gen_config = genai_types.GenerationConfig(**generation_config_override) if generation_config_override else None
        safety_settings = safety_settings_override if safety_settings_override is not None else self.safety_settings

        try:
             # Initialize model instance - validation happens here
             # Note: System instruction should be passed here if supported by the specific model version/SDK
             model_instance = genai.GenerativeModel(
                 effective_model,
                 system_instruction=system_prompt # Pass system prompt here
             )
        except Exception as e:
             # Catch errors during model initialization (e.g., invalid name)
             raise ValueError(f"Failed to initialize generative model '{effective_model}'. Check model name. Error: {e}") from e

        contents = [{'role': 'user', 'parts': [prompt]}]
        # System prompt is handled by model instance initialization now

        try:
            # Use the retry helper
            api_kwargs = { 
                'contents': contents,
                'generation_config': gen_config,
                'safety_settings': safety_settings,
            }
            response = await self._execute_with_retry(
                model_instance.generate_content,
                _model_name_for_log=effective_model, # Log arg
                **api_kwargs # API args
            )

            # --- Process Response ---
            generated_text = None
            prompt_tokens = 0
            completion_tokens = 0
            usage_metadata = getattr(response, 'usage_metadata', None)

            try:
                # Attempt to access generated text safely
                # response.text can raise ValueError if content is blocked
                generated_text = response.text
            except ValueError as e:
                # Handle cases where accessing .text fails (e.g., blocked content)
                 block_reason = "Unknown"
                 try:
                     if response.prompt_feedback and response.prompt_feedback.block_reason:
                         block_reason = getattr(response.prompt_feedback.block_reason, 'name', str(response.prompt_feedback.block_reason))
                 except AttributeError: pass
                 raise ValueError(f"API call failed for {effective_model}: Content blocked or invalid. Reason: {block_reason}. Original Error: {e}") from e
            except AttributeError:
                 # If .text attribute doesn't exist (shouldn't happen with valid response)
                 pass # We handle None generated_text below

            # Fallback text extraction if needed (though response.text should usually work or raise error)
            if generated_text is None:
                 try:
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                     if not generated_text or not generated_text.strip(): # Ensure we got some text
                         raise ValueError(f"API call failed for {effective_model}: Received no valid text content in response.")
                 except (AttributeError, IndexError, ValueError) as text_extract_err:
                     raise ValueError(f"API call failed for {effective_model}: Could not extract text from response structure.") from text_extract_err

            # --- Token Counting & Cost Tracking ---
            if usage_metadata:
                try:
                    # Use names consistent with google-genai v0.3+
                    prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                    completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0) # Sum of tokens across candidates
                    prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
                    completion_tokens = int(completion_tokens) if completion_tokens is not None else 0
                except (AttributeError, ValueError, TypeError) as e:
                    warnings.warn(f"Error accessing token counts from usage metadata for {effective_model}: {e}. Cost tracking may be inaccurate.")
                    prompt_tokens = 0
                    completion_tokens = 0

                if self.cost_tracker:
                    # Record usage: Map prompt_tokens -> input_units, completion_tokens -> output_units
                    self.cost_tracker.record_usage(effective_model, prompt_tokens, completion_tokens)
            else:
                warnings.warn(f"Response object for model '{effective_model}' lacks 'usage_metadata'. Cost tracking may be inaccurate.")

            return generated_text

        except ValueError as ve:
             # Catch ValueErrors raised during response processing (blocking, empty content)
             # These are definitive failures, don't wrap further
             raise ve
        except Exception as e:
             # Catch errors from _execute_with_retry (after all retries failed)
             raise Exception(f"API call to generation model '{effective_model}' failed after multiple retries.") from e


    async def embed_content(self,
                            contents: Union[str, List[str]],
                            task_type: str,
                            *,
                            model: Optional[str] = None,
                            title: Optional[str] = None, # Optional for RETRIEVAL_DOCUMENT
                            output_dimensionality: Optional[int] = None # Optional truncation
                            ) -> List[List[float]]:
        """
        Generates embeddings for the given text content(s) using the specified Gemini model,
        with retry logic.

        Args:
            contents: A single string or a list of strings to embed.
            task_type: The task type for the embedding (e.g., "RETRIEVAL_DOCUMENT",
                       "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"). See Google AI docs.
            model: Specific Gemini embedding model name. Defaults to client's default_embedding_model.
                   Should typically start with 'models/' like 'models/text-embedding-004'.
            title: An optional title when task_type="RETRIEVAL_DOCUMENT".
            output_dimensionality: Optional dimension to truncate the output embedding to.

        Returns:
            A list of embedding vectors (list of lists of floats). If a single string
            was input, the outer list will contain one vector.

        Raises:
            Exception: If the API call fails after all retry attempts.
            ValueError: If the response is invalid or parameters are incorrect.
            TypeError: If input types are wrong.
        """
        effective_model = model or self.default_embedding_model
        if not effective_model.startswith("models/"):
             # Enforce 'models/' prefix based on how cost dict and API often work
             warnings.warn(f"Embedding model name '{effective_model}' should ideally start with 'models/'. Attempting call anyway.")
             # Consider adding prefix here if API consistently fails without it:
             # effective_model = f'models/{effective_model}'

        # Validate contents type
        if not isinstance(contents, (str, list)):
             raise TypeError("contents must be a string or a list of strings.")
        if isinstance(contents, list) and not all(isinstance(item, str) for item in contents):
             raise TypeError("If contents is a list, all items must be strings.")

        # Prepare arguments for genai.embed_content
        embed_args = {
             "model": effective_model,
             "content": contents, # Pass str or list directly
             "task_type": task_type,
        }
        if title is not None and task_type == "RETRIEVAL_DOCUMENT":
            embed_args["title"] = title
        elif title is not None:
             warnings.warn(f"Ignoring 'title' argument as task_type is '{task_type}', not 'RETRIEVAL_DOCUMENT'.")

        if output_dimensionality is not None:
             embed_args["output_dimensionality"] = output_dimensionality

        try:
            # Use the retry helper
            response_dict = await self._execute_with_retry(
                genai.embed_content,
                _model_name_for_log=effective_model, # Log arg
                **embed_args # API args
            )

            # --- Process Response ---
            embeddings: List[List[float]] = []
            if 'embedding' in response_dict: # Response for single string input
                 if isinstance(response_dict['embedding'], list):
                    embeddings = [response_dict['embedding']]
                 else:
                      raise ValueError(f"Invalid embedding format received for single input in {effective_model}.")
            elif 'embeddings' in response_dict: # Response for list input
                 if isinstance(response_dict['embeddings'], list) and all(isinstance(e, list) for e in response_dict['embeddings']):
                      embeddings = response_dict['embeddings']
                 else:
                      raise ValueError(f"Invalid embedding format received for list input in {effective_model}.")
            else:
                # Should not happen if API call succeeded without error
                raise ValueError(f"API call for {effective_model} succeeded but response missing 'embedding' or 'embeddings' key.")

            # --- Token Counting & Cost Tracking (Attempt) ---
            # NOTE: Usage metadata is often NOT included in embedding responses.
            # Add specific check if response structure is known, otherwise assume unavailable.
            usage_metadata = response_dict.get('usage_metadata', None) # Assuming it might be dict
            input_units = 0
            output_units = 0 # Always 0 for embeddings

            if usage_metadata and isinstance(usage_metadata, dict):
                 # Example: Adjust key based on actual response if available
                 # E.g., 'total_token_count', 'prompt_token_count'
                 token_key = 'total_token_count' # Replace with actual key if known
                 if token_key in usage_metadata:
                     try:
                         input_units = int(usage_metadata[token_key])
                     except (ValueError, TypeError):
                         warnings.warn(f"Could not parse '{token_key}' from usage metadata for {effective_model}.")
                         input_units = 0
                 else:
                      warnings.warn(f"Expected key '{token_key}' not found in usage metadata for {effective_model}.")

                 if input_units > 0 and self.cost_tracker:
                      self.cost_tracker.record_usage(effective_model, input_units, output_units)
                 elif input_units == 0:
                      # Only warn if we expected metadata but couldn't parse/find tokens
                      warnings.warn(f"Could not determine input units from usage metadata for {effective_model}. Cost tracking may be inaccurate.")

            else:
                 # This is the expected path if usage metadata is typically absent
                 warnings.warn(f"Usage metadata not found in response for embedding model '{effective_model}'. Cost tracking skipped for this call.")

            return embeddings

        except ValueError as ve:
            # Catch ValueErrors raised during response processing
            raise ve
        except Exception as e:
            # Catch errors from _execute_with_retry (after all retries failed)
            raise Exception(f"API call to embedding model '{effective_model}' failed after multiple retries.") from e