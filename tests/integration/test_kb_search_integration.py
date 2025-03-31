# File: tests/integration/test_kb_search_integration.py

import pytest
import os
import sys
import asyncio
import numpy as np
import pytest_asyncio
from unittest.mock import MagicMock
from typing import List, Tuple, Dict, Optional

# Make sure pytest can find the src modules
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

try:
    from lean_automator.kb_storage import (
        initialize_database,
        save_kb_item, # async
        get_kb_item_by_name, # sync
        get_all_items_with_embedding, # sync
        KBItem,
        ItemStatus,
        ItemType,
        EMBEDDING_DTYPE
    )
    from lean_automator.kb_search import find_similar_items # async
    # Import GeminiClient only for type hinting if needed, we will mock its usage
    from lean_automator.llm_call import GeminiClient
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

@pytest_asyncio.fixture(scope="function")
async def test_db():
    """
    Sets up an initialized temporary database, populates it with items
    having pre-defined embeddings, and yields the path. Cleans up afterwards.
    """
    db_path = TEST_DB_FILENAME
    if os.path.exists(db_path):
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

    # Teardown
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except OSError as e:
        print(f"\nWarning: Could not clean up test database '{db_path}': {e}")


@pytest.fixture
def mock_generate_query_embedding(mocker):
    """
    Mocks the kb_search.generate_embedding function to return a fixed
    query vector, avoiding actual API calls.
    """
    # This vector should be similar to VEC_A and VEC_B for testing ranking
    mock_query_vec = np.array([0.95, 0.05, 0.0, 0.0], dtype=EMBEDDING_DTYPE)
    # Patch the function *where it's imported* in the kb_search module
    return mocker.patch('lean_automator.kb_search.generate_embedding', return_value=mock_query_vec)

@pytest.fixture
def mock_gemini_client_instance(mocker) -> MagicMock:
    """ Provides a basic mock instance of GeminiClient. """
    # We don't need complex behavior, just an object to pass type checks
    # The actual embedding call will be mocked by mock_generate_query_embedding
    return mocker.MagicMock(spec=GeminiClient)

# --- Test Class ---

@pytest.mark.filterwarnings("ignore:GeminiClient/kb_search not available")
@pytest.mark.integration
class TestKBSearchIntegration:

    @pytest.mark.asyncio
    async def test_find_similar_items_nl_success(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Test finding similar items in the 'nl' field."""
        query = "Tell me about apples"
        search_field = "nl"
        top_n = 3

        # The mock_generate_query_embedding ensures generate_embedding returns our fixed query vector
        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance, # Pass the mock client
            db_path=test_db,
            top_n=top_n
        )

        # Calculate normalized query vector for comparison
        query_vec = mock_generate_query_embedding.return_value
        norm_query_vec = query_vec / np.linalg.norm(query_vec)

        # Calculate expected similarities (Cosine Sim = dot(norm_v1, norm_v2))
        # Note: Document vectors A, B, C, E are already normalized (length 1) except B and E
        norm_vec_a = VEC_A / np.linalg.norm(VEC_A) # Should be [1,0,0,0]
        norm_vec_b = VEC_B / np.linalg.norm(VEC_B)
        norm_vec_c = VEC_C / np.linalg.norm(VEC_C) # Should be [0,1,0,0]
        norm_vec_e = VEC_E / np.linalg.norm(VEC_E)

        sim_a = np.dot(norm_query_vec, norm_vec_a)
        sim_b = np.dot(norm_query_vec, norm_vec_b)
        sim_c = np.dot(norm_query_vec, norm_vec_c)
        sim_e = np.dot(norm_query_vec, norm_vec_e)

        # Expected Order based on calculated similarities: A, B, E
        # sim_a ≈ dot([0.9986, 0.0526, 0, 0], [1, 0, 0, 0]) ≈ 0.9986
        # sim_b ≈ dot([0.9986, 0.0526, 0, 0], [0.9/sqrt(0.82), 0.1/sqrt(0.82), 0, 0]) ≈ 0.9986*0.994 + 0.0526*0.110 ≈ 0.992 + 0.005 ≈ 0.997
        # sim_e ≈ dot([0.9986, 0.0526, 0, 0], [0.8/sqrt(0.65), 0, 0, 0.1/sqrt(0.65)]) ≈ 0.9986*0.992 ≈ 0.991

        # Correction: Similarity order should be A, B, E based on dot products
        # Re-evaluating dot products:
        # Query Vec: [0.95, 0.05, 0, 0], Norm ~ 0.9513
        # Norm Query: [0.9986, 0.0525, 0, 0]
        # Vec A: [1,0,0,0], Norm 1 -> Sim = 0.9986
        # Vec B: [0.9, 0.1, 0,0], Norm ~ 0.9055 -> Norm B = [0.9939, 0.1104, 0, 0] -> Sim = 0.9986*0.9939 + 0.0525*0.1104 = 0.9925 + 0.0058 = 0.9983
        # Vec C: [0,1,0,0], Norm 1 -> Sim = 0.0525
        # Vec E: [0.8, 0, 0, 0.1], Norm ~ 0.8062 -> Norm E = [0.9923, 0, 0, 0.1240] -> Sim = 0.9986*0.9923 = 0.9909

        # Expected Order: A, B, E

        assert len(results) == top_n
        assert results[0][0].unique_name == "item_A_nl" # Highest similarity
        assert results[1][0].unique_name == "item_B_nl" # Second highest
        assert results[2][0].unique_name == "item_E_nl" # Third highest

        # Check scores are descending
        assert results[0][1] >= results[1][1] >= results[2][1]
        # Check approximate scores (adjust based on recalculation)
        assert results[0][1] == pytest.approx(sim_a, abs=1e-3) # ~0.9986
        assert results[1][1] == pytest.approx(sim_b, abs=1e-3) # ~0.9983
        assert results[2][1] == pytest.approx(sim_e, abs=1e-3) # ~0.9909


        # Verify generate_embedding was called once with the query
        mock_generate_query_embedding.assert_called_once()
        call_args, _ = mock_generate_query_embedding.call_args
        assert call_args[0] == query # Check query text
        assert call_args[1] == "RETRIEVAL_QUERY" # Check default task type
        assert call_args[2] is mock_gemini_client_instance # Check client instance


    @pytest.mark.asyncio
    async def test_find_similar_items_latex_success(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Test finding similar items in the 'latex' field."""
        query = "summation formula"
        search_field = "latex"
        top_n = 1

        # Make the mock query embedder return something similar to the latex item D
        mock_query_vec_latex = np.array([0.0, 0.0, 0.9, 0.1], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec_latex

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db,
            top_n=top_n
        )

        # Only item_D_latex has latex embedding
        assert len(results) == top_n
        assert results[0][0].unique_name == "item_D_latex"

        # Calculate expected score
        norm_query = mock_query_vec_latex / np.linalg.norm(mock_query_vec_latex)
        norm_doc_d = VEC_D / np.linalg.norm(VEC_D) # Should be [0,0,1,0]
        expected_score = np.dot(norm_query, norm_doc_d) # ~ dot([0, 0, 0.994, 0.110], [0, 0, 1, 0]) = 0.994

        assert results[0][1] == pytest.approx(expected_score, abs=1e-3)

        mock_generate_query_embedding.assert_called_once_with(query, "RETRIEVAL_QUERY", mock_gemini_client_instance)


    @pytest.mark.asyncio
    async def test_find_similar_items_no_matches(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Test finding no similar items if query is dissimilar."""
        query = "something completely different"
        search_field = "nl"

        # Make the mock query embedder return something orthogonal to all nl items
        mock_query_vec_ortho = np.array([0.0, 0.0, 0.0, 1.0], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec_ortho

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db
        )

        # Calculate expected scores
        norm_query = mock_query_vec_ortho / np.linalg.norm(mock_query_vec_ortho) # [0,0,0,1]
        norm_vec_a = VEC_A / np.linalg.norm(VEC_A) # [1,0,0,0] -> Sim = 0
        norm_vec_b = VEC_B / np.linalg.norm(VEC_B) # -> Sim = 0
        norm_vec_c = VEC_C / np.linalg.norm(VEC_C) # [0,1,0,0] -> Sim = 0
        norm_vec_e = VEC_E / np.linalg.norm(VEC_E) # Approx [0.9923, 0, 0, 0.1240] -> Sim ~ 0.1240

        # Expecting scores close to 0, E might be top due to 4th dimension
        assert len(results) > 0 # It will return something based on small similarities
        assert results[0][0].unique_name == "item_E_nl" # E has a small component in 4th dim
        assert results[0][1] == pytest.approx(0.1240, abs=1e-3) # Top score should be low


    @pytest.mark.asyncio
    async def test_find_similar_items_top_n_limit(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Test that top_n correctly limits the number of results."""
        query = "apples again"
        search_field = "nl"
        top_n = 2 # Request fewer items than available matches

        # Query similar to A, B, E
        mock_query_vec = np.array([0.95, 0.05, 0.0, 0.0], dtype=EMBEDDING_DTYPE)
        mock_generate_query_embedding.return_value = mock_query_vec

        results = await find_similar_items(
            query,
            search_field,
            client=mock_gemini_client_instance,
            db_path=test_db,
            top_n=top_n
        )

        assert len(results) == top_n
        # Based on calculations in test_find_similar_items_nl_success: A, B have highest scores
        assert results[0][0].unique_name == "item_A_nl"
        assert results[1][0].unique_name == "item_B_nl"


    @pytest.mark.asyncio
    async def test_find_similar_items_query_embed_fails(self, test_db, mock_generate_query_embedding, mock_gemini_client_instance):
        """Test behavior when query embedding generation fails."""
        query = "this query will fail"
        search_field = "nl"

        # Configure mock to return None
        mock_generate_query_embedding.return_value = None

        with pytest.warns(UserWarning, match="Failed to generate query embedding"):
            results = await find_similar_items(
                query,
                search_field,
                client=mock_gemini_client_instance,
                db_path=test_db
            )

        assert results == []
        mock_generate_query_embedding.assert_called_once()


    @pytest.mark.asyncio
    async def test_find_similar_items_no_client(self, test_db, mock_generate_query_embedding):
        """Test behavior when no client is provided."""
        query = "no client query"
        search_field = "nl"

        # generate_embedding mock is still active, but find_similar_items checks client first
        with pytest.warns(UserWarning, match="GeminiClient not available for embedding search"):
             results = await find_similar_items(
                 query,
                 search_field,
                 client=None, # Pass None for client
                 db_path=test_db
             )

        assert results == []
        mock_generate_query_embedding.assert_not_called() # Should exit before calling generate