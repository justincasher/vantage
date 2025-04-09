# File: tests/integration/kb/test_search_integration.py

"""Integration tests for the Knowledge Base semantic search functionality.

Verifies the `find_similar_items` function in `kb_search.py` by interacting
with a temporary SQLite database populated with test items having pre-defined
embeddings. Mocks LLM calls for embedding generation.
"""

import pytest
import os
import sys
import asyncio
import numpy as np
import pytest_asyncio
from unittest.mock import MagicMock
from typing import List, Tuple, Dict, Optional
import logging 

# Configure logger for this module
logger = logging.getLogger(__name__) 

# Make sure pytest can find the src modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from lean_automator.kb.storage import (
        initialize_database,
        save_kb_item, # async
        get_kb_item_by_name, # sync
        get_all_items_with_embedding, # sync
        KBItem,
        ItemStatus,
        ItemType,
        EMBEDDING_DTYPE
    )
    from lean_automator.kb.search import find_similar_items # async
    # Import GeminiClient only for type hinting if needed, we will mock its usage
    from lean_automator.llm.caller import GeminiClient
except ImportError as e:
    pytest.skip(f"Skipping test module: Failed to import components. Error: {e}", allow_module_level=True)


# --- Apply the integration mark to all tests in this module ---
pytestmark = pytest.mark.integration

# --- Constants ---
TEST_DB_FILENAME = "test_integration_search.sqlite"
DIM = 4 # Dimension for our dummy embeddings

# --- Dummy Embedding Data ---
# Define some vectors and their byte representations
VEC_A = np.array([1.0, 0.0, 0.0, 0.0], dtype=EMBEDDING_DTYPE)
VEC_B = np.array([0.9, 0.1, 0.0, 0.0], dtype=EMBEDDING_DTYPE) # Similar to A
VEC_C = np.array([0.0, 1.0, 0.0, 0.0], dtype=EMBEDDING_DTYPE) # Orthogonal to A/B
VEC_D = np.array([0.0, 0.0, 1.0, 0.0], dtype=EMBEDDING_DTYPE) # Different field
VEC_E = np.array([0.8, 0.0, 0.0, 0.1], dtype=EMBEDDING_DTYPE) # Less similar to A

BYTES_A = VEC_A.tobytes()
BYTES_B = VEC_B.tobytes()
BYTES_C = VEC_C.tobytes()
BYTES_D = VEC_D.tobytes()
BYTES_E = VEC_E.tobytes()

# --- Fixtures ---

# Filter expected warnings during DB setup where client=None prevents embedding
@pytest_asyncio.fixture(scope="function")
@pytest.mark.filterwarnings("ignore:Cannot generate")
async def test_db():
    """Sets up and tears down a temporary SQLite database for testing.

    Initializes the database schema using `initialize_database` and populates
    it with several `KBItem` objects containing pre-defined dummy embeddings
    in the `embedding_nl` or `embedding_latex` fields. Uses `save_kb_item`
    without an LLM client to prevent actual embedding generation during setup.

    Yields:
        str: The file path to the temporary database.
    """
    db_path = TEST_DB_FILENAME
    if os.path.exists(db_path):
        logger.debug(f"Removing existing test database: {db_path}")
        os.remove(db_path)

    initialize_database(db_path=db_path)

    # Create KBItems with embeddings
    items_to_save = [
        # Default type is THEOREM, requires proof
        KBItem(unique_name="item_A_nl", item_type=ItemType.REMARK, description_nl="About apples", embedding_nl=BYTES_A),
        KBItem(unique_name="item_B_nl", item_type=ItemType.REMARK, description_nl="About red apples", embedding_nl=BYTES_B),
        KBItem(unique_name="item_C_nl", item_type=ItemType.REMARK, description_nl="About oranges", embedding_nl=BYTES_C),
        # Use NOTATION type for the latex item, does not require proof
        KBItem(unique_name="item_D_latex", item_type=ItemType.NOTATION, latex_statement="\\sum x", embedding_latex=BYTES_D), # Changed latex_exposition to latex_statement
        KBItem(unique_name="item_E_nl", item_type=ItemType.REMARK, description_nl="About green apples", embedding_nl=BYTES_E),
        KBItem(unique_name="item_F_no_embed", item_type=ItemType.REMARK, description_nl="No embedding here"),
    ]

    # Save items asynchronously (use client=None to prevent embed generation)
    save_tasks = [save_kb_item(item, client=None, db_path=db_path) for item in items_to_save]
    await asyncio.gather(*save_tasks)

    yield db_path

    # Teardown: Remove the test database file
    try:
        # Add short delay before attempting removal, sometimes helps on Windows
        await asyncio.sleep(0.1)
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.debug(f"Cleaned up test database: {db_path}") # <-- Use defined logger
        else:
             logger.debug(f"Test database already removed: {db_path}")
    except OSError as e:
        # Log warning if cleanup fails but don't fail the test run
        logger.warning(f"Could not clean up test database '{db_path}': {e}") # <-- Use defined logger


@pytest.fixture
def mock_generate_query_embedding(mocker):
    """Mocks kb_search.generate_embedding to return a fixed query vector.

    This avoids actual LLM API calls during testing. The mock initially returns
    a vector designed to be similar to test vectors A and B for ranking tests.
    The test function can override `mock.return_value` if needed.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        MagicMock: The mock object patching `generate_embedding`.
    """
    # Default mock query vector (similar to VEC_A and VEC_B)
    mock_query_vec = np.array([0.95, 0.05, 0.0, 0.0], dtype=EMBEDDING_DTYPE)
    # Patch the function where it's used (within the kb_search module)
    # Use autospec=True to help ensure the mock signature matches the original
    return mocker.patch('lean_automator.kb.search.generate_embedding', return_value=mock_query_vec, autospec=True)

@pytest.fixture
def mock_gemini_client_instance(mocker) -> MagicMock:
    """Provides a basic MagicMock instance pretending to be a GeminiClient.

    Used simply to satisfy type hints and pass as an argument where a client
    instance is expected. The actual LLM calls relevant to these tests
    (`generate_embedding`) are mocked separately.

    Args:
        mocker: The pytest-mock fixture.

    Returns:
        MagicMock: A mock object simulating a GeminiClient instance.
    """
    # Create a mock with the spec of the actual GeminiClient class if available
    mock_client = mocker.MagicMock(spec=GeminiClient if GeminiClient else None)
    # Add any specific attributes or methods needed by the code under test if necessary
    # For find_similar_items, just the existence of the object is often enough
    # if generate_embedding is mocked elsewhere.
    return mock_client

# --- Test Class ---

# Ignore general warnings about GeminiClient/kb_search availability if imports fail
# Ignore specific warnings about embedding generation during setup (handled by fixture mark)
@pytest.mark.filterwarnings("ignore:GeminiClient/kb_search not available")
@pytest.mark.integration
class TestKBSearchIntegration:
    """Groups integration tests for the KB semantic search feature."""

    @pytest.mark.asyncio
    async def test_find_similar_items_nl_success(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Verifies finding and ranking similar items in the 'nl' field."""
        query = "Tell me about apples"
        search_field = "nl"
        top_n = 3

        # The mock_generate_query_embedding fixture provides the query vector
        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db,
            top_n=top_n
        )

        # --- Assertions ---
        assert len(results) == top_n, f"Expected {top_n} results, got {len(results)}"

        # Check unique names in expected order of similarity (A > B > E based on manual calc)
        result_names = [item.unique_name for item, score in results]
        expected_names = ["item_A_nl", "item_B_nl", "item_E_nl"]
        assert result_names == expected_names, f"Results order mismatch: Expected {expected_names}, Got {result_names}"

        # Check scores are descending
        scores = [score for item, score in results]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores are not descending"

        # Verify approximate scores against expected calculations
        # (Recalculating here for clarity, adjust tolerance as needed)
        query_vec = mock_generate_query_embedding.return_value
        norm_query_vec = query_vec / np.linalg.norm(query_vec) if np.linalg.norm(query_vec) > 0 else query_vec
        sim_a = np.dot(norm_query_vec, VEC_A / np.linalg.norm(VEC_A)) # Norm A=1
        sim_b = np.dot(norm_query_vec, VEC_B / np.linalg.norm(VEC_B))
        sim_e = np.dot(norm_query_vec, VEC_E / np.linalg.norm(VEC_E))

        assert results[0][1] == pytest.approx(sim_a, abs=1e-4), f"Score mismatch for A: Expected ~{sim_a:.4f}, Got {results[0][1]:.4f}"
        assert results[1][1] == pytest.approx(sim_b, abs=1e-4), f"Score mismatch for B: Expected ~{sim_b:.4f}, Got {results[1][1]:.4f}"
        assert results[2][1] == pytest.approx(sim_e, abs=1e-4), f"Score mismatch for E: Expected ~{sim_e:.4f}, Got {results[2][1]:.4f}"

        # Verify the mocked embedding generation was called correctly
        mock_generate_query_embedding.assert_called_once()
        call_args, call_kwargs = mock_generate_query_embedding.call_args
        assert call_args[0] == query # Check query text argument
        assert call_args[1] == "RETRIEVAL_QUERY" # Check default task type
        assert call_args[2] is mock_gemini_client_instance # Check client instance passed


    @pytest.mark.asyncio
    async def test_find_similar_items_latex_success(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Verifies finding the single item in the 'latex' field."""
        query = "summation formula"
        search_field = "latex"
        top_n = 1

        # Define a query vector similar to the LaTeX item D's vector
        mock_query_vec_latex = np.array([0.0, 0.0, 0.9, 0.1], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec_latex

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db,
            top_n=top_n
        )

        # --- Assertions ---
        assert len(results) == top_n, f"Expected {top_n} result for latex search, got {len(results)}"
        assert results[0][0].unique_name == "item_D_latex", "Incorrect item found for latex search"

        # Calculate expected score
        norm_query = mock_query_vec_latex / np.linalg.norm(mock_query_vec_latex)
        norm_doc_d = VEC_D / np.linalg.norm(VEC_D) # Norm D = 1
        expected_score = np.dot(norm_query, norm_doc_d)

        assert results[0][1] == pytest.approx(expected_score, abs=1e-4), f"Score mismatch for D: Expected ~{expected_score:.4f}, Got {results[0][1]:.4f}"

        # Verify mock call
        mock_generate_query_embedding.assert_called_once_with(query, "RETRIEVAL_QUERY", mock_gemini_client_instance)


    @pytest.mark.asyncio
    async def test_find_similar_items_no_matches(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Verifies behavior when the query is dissimilar to all items."""
        query = "something completely different"
        search_field = "nl"

        # Define a query vector orthogonal to most nl items
        mock_query_vec_ortho = np.array([0.0, 0.0, 0.0, 1.0], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec_ortho

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db
            # Using default top_n=5
        )

        # --- Assertions ---
        # Even orthogonal vectors can have non-zero cosine similarity due to normalization/floating point.
        # We expect results, but the top score should be low. E has a component in 4th dim.
        assert len(results) > 0, "Expected some results even for dissimilar query"

        # Calculate expected similarity for E (the only one with non-zero 4th dim)
        norm_query = mock_query_vec_ortho / np.linalg.norm(mock_query_vec_ortho) # [0,0,0,1]
        norm_vec_e = VEC_E / np.linalg.norm(VEC_E)
        expected_score_e = np.dot(norm_query, norm_vec_e) # Should be approx 0.1240

        assert results[0][0].unique_name == "item_E_nl", "Expected item E to be the top (least dissimilar) result"
        assert results[0][1] == pytest.approx(expected_score_e, abs=1e-4), "Top score for dissimilar query is unexpectedly high or low"
        assert results[0][1] < 0.2, "Top score for dissimilar query should be low"


    @pytest.mark.asyncio
    async def test_find_similar_items_top_n_limit(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Verifies that the `top_n` parameter correctly limits result count."""
        query = "apples again"
        search_field = "nl"
        top_n = 2 # Request fewer items than potentially similar ones (A, B, E)

        # Use the standard mock query vector similar to A, B, E
        mock_query_vec = np.array([0.95, 0.05, 0.0, 0.0], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db,
            top_n=top_n
        )

        # --- Assertions ---
        assert len(results) == top_n, f"Expected exactly {top_n} results, got {len(results)}"

        # Check that the top N results are the most similar ones in the correct order
        result_names = [item.unique_name for item, score in results]
        expected_top_names = ["item_A_nl", "item_B_nl"] # Based on similarity calculations
        assert result_names == expected_top_names, f"Expected top {top_n} results to be {expected_top_names}, Got {result_names}"


    @pytest.mark.asyncio
    async def test_find_similar_items_query_embed_fails(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Verifies behavior when query embedding generation returns None."""
        query = "this query will fail"
        search_field = "nl"

        # Configure the mock to simulate failure
        mock_generate_query_embedding.return_value = None

        # Expect a UserWarning and an empty list
        with pytest.warns(UserWarning, match="Failed to generate query embedding"):
            results = await find_similar_items(
                query,
                search_field,
                client=mock_gemini_client_instance,
                db_path=test_db
            )

        # --- Assertions ---
        assert results == [], "Expected empty list when query embedding fails"
        mock_generate_query_embedding.assert_called_once() # Ensure it was called


    @pytest.mark.asyncio
    async def test_find_similar_items_no_client(self, test_db, mock_generate_query_embedding):
        """Verifies behavior when the GeminiClient instance is None."""
        query = "no client query"
        search_field = "nl"

        # The mock for generate_embedding is active, but find_similar_items should check client first
        # Expect a UserWarning and an empty list
        with pytest.warns(UserWarning, match="GeminiClient not available for embedding search"):
             results = await find_similar_items(
                 query,
                 search_field,
                 client=None, # Explicitly pass None
                 db_path=test_db
             )

        # --- Assertions ---
        assert results == [], "Expected empty list when client is None"
        # Crucially, the embedding generation should not even be attempted
        mock_generate_query_embedding.assert_not_called()

