# File: tests/unit/kb/test_search_unit.py

from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest

# Assuming pytest runs from root and pytest.ini sets pythonpath=src
try:
    from lean_automator.kb.search import (
        _bytes_to_vector,
        _cosine_similarity,
        find_similar_items,
        generate_embedding,
    )
    from lean_automator.kb.storage import (
        DEFAULT_DB_PATH,
        EMBEDDING_DTYPE,
        ItemType,
        KBItem,
    )
    from lean_automator.llm.caller import GeminiClient
except ImportError as e:
    pytest.skip(
        f"Skipping test module: Failed to import components. Error: {e}",
        allow_module_level=True,
    )

# --- Constants ---
DIM = 3  # Dimension for test embeddings
TEST_DB_PATH = "/fake/search_db.sqlite"


# --- Helper Functions ---
def create_embedding_bytes(vector: List[float]) -> bytes:
    """Helper to create bytes from a list using the standard dtype."""
    return np.array(vector, dtype=EMBEDDING_DTYPE).tobytes()


# --- Fixtures ---


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """
    Provides a MagicMock instance simulating GeminiClient with
    embed_content mocked.
    """
    client = MagicMock(spec=GeminiClient)
    # Mock the async method used by generate_embedding
    client.embed_content = AsyncMock()
    return client


@pytest.fixture
def mock_kb_items_for_search() -> List[Tuple[int, str, bytes]]:
    """Provides sample data as returned by get_all_items_with_embedding."""
    return [
        (
            1,
            "item_A",
            create_embedding_bytes([1.0, 0.0, 0.0]),
        ),  # Vector pointing along X
        (
            2,
            "item_B",
            create_embedding_bytes([0.0, 1.0, 0.0]),
        ),  # Vector pointing along Y
        (3, "item_C", create_embedding_bytes([-1.0, 0.0, 0.0])),  # Vector opposite to A
        (
            4,
            "item_D",
            create_embedding_bytes([0.707, 0.707, 0.0]),
        ),  # Vector between A and B
        (5, "item_E_bad_bytes", b"invalid_bytes"),  # For testing decode error
        (6, "item_F_wrong_dim", create_embedding_bytes([1.0, 2.0])),  # Wrong dimension
    ]


@pytest.fixture
def mock_full_kb_items() -> dict:
    """Provides full KBItem objects corresponding to mock_kb_items_for_search."""
    return {
        "item_A": KBItem(
            id=1,
            unique_name="item_A",
            item_type=ItemType.THEOREM,
            lean_code="A code",
            embedding_nl=create_embedding_bytes([1.0, 0.0, 0.0]),
        ),
        "item_B": KBItem(
            id=2,
            unique_name="item_B",
            item_type=ItemType.DEFINITION,
            lean_code="B code",
            embedding_nl=create_embedding_bytes([0.0, 1.0, 0.0]),
        ),
        "item_C": KBItem(
            id=3,
            unique_name="item_C",
            item_type=ItemType.LEMMA,
            lean_code="C code",
            embedding_nl=create_embedding_bytes([-1.0, 0.0, 0.0]),
        ),
        "item_D": KBItem(
            id=4,
            unique_name="item_D",
            item_type=ItemType.THEOREM,
            lean_code="D code",
            embedding_nl=create_embedding_bytes([0.707, 0.707, 0.0]),
        ),
        # E and F aren't needed as full items if they cause errors before retrieval
    }


@pytest.fixture(autouse=True)  # Apply these patches to all tests in the module
def patch_db_functions(mocker, mock_kb_items_for_search, mock_full_kb_items):
    """Patches kb_storage functions used by kb_search."""
    mock_get_all = mocker.patch(
        "lean_automator.kb.search.get_all_items_with_embedding",
        return_value=mock_kb_items_for_search,
    )
    mock_get_by_name = mocker.patch(
        "lean_automator.kb.search.get_kb_item_by_name",
        side_effect=lambda name, db_path=None: mock_full_kb_items.get(name),
    )
    return mock_get_all, mock_get_by_name


# --- Tests for generate_embedding ---


@pytest.mark.asyncio
async def test_generate_embedding_success(mock_gemini_client):
    """Test successful embedding generation."""
    text = "Sample text"
    task_type = "RETRIEVAL_QUERY"
    expected_vector_list = [[0.1, 0.2, 0.3]]
    expected_np_array = np.array(expected_vector_list[0], dtype=EMBEDDING_DTYPE)
    mock_gemini_client.embed_content.return_value = expected_vector_list

    result_vector = await generate_embedding(text, task_type, mock_gemini_client)

    mock_gemini_client.embed_content.assert_awaited_once_with(
        contents=text, task_type=task_type
    )
    assert result_vector is not None
    np.testing.assert_array_equal(result_vector, expected_np_array)


@pytest.mark.asyncio
async def test_generate_embedding_empty_text(mock_gemini_client):
    """Test generate_embedding returns None for empty text."""
    with pytest.warns(
        UserWarning, match="Attempted to generate embedding for empty text"
    ):
        result = await generate_embedding("", "RETRIEVAL_QUERY", mock_gemini_client)
    assert result is None
    mock_gemini_client.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_generate_embedding_no_client():
    """Test generate_embedding returns None if client is missing."""
    with pytest.warns(UserWarning, match="GeminiClient not available"):
        result = await generate_embedding("Some text", "RETRIEVAL_QUERY", None)  # type: ignore
    assert result is None


@pytest.mark.asyncio
async def test_generate_embedding_api_error(mock_gemini_client):
    """Test generate_embedding handles API errors."""
    text = "Sample text"
    task_type = "RETRIEVAL_QUERY"
    mock_gemini_client.embed_content.side_effect = Exception("API Failure")

    with pytest.warns(UserWarning, match="Error generating embedding: API Failure"):
        result = await generate_embedding(text, task_type, mock_gemini_client)

    assert result is None
    mock_gemini_client.embed_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_embedding_empty_api_result(mock_gemini_client):
    """Test generate_embedding handles empty list from API."""
    mock_gemini_client.embed_content.return_value = []  # Empty list

    with pytest.warns(UserWarning, match="Embedding generation returned empty result"):
        result = await generate_embedding("Text", "TASK", mock_gemini_client)
    assert result is None

    mock_gemini_client.embed_content.return_value = None  # None result
    with pytest.warns(UserWarning, match="Embedding generation returned empty result"):
        result = await generate_embedding("Text", "TASK", mock_gemini_client)
    assert result is None


# --- Tests for _bytes_to_vector ---


def test_bytes_to_vector_success():
    """Test successful conversion from bytes back to vector."""
    original_vector = np.array([1.5, -2.5, 3.0], dtype=EMBEDDING_DTYPE)
    byte_data = original_vector.tobytes()

    result_vector = _bytes_to_vector(byte_data)

    assert result_vector is not None
    np.testing.assert_array_equal(result_vector, original_vector)
    assert result_vector.dtype == EMBEDDING_DTYPE


def test_bytes_to_vector_invalid_bytes():
    """Test conversion with invalid byte sequence."""
    invalid_byte_data = b"\x01\x02\x03"  # Incorrect length for float32

    with pytest.warns(UserWarning, match="Error decoding embedding bytes"):
        result_vector = _bytes_to_vector(invalid_byte_data)

    assert result_vector is None


# --- Tests for _cosine_similarity ---


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_opposite():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([-1.0, -2.0, -3.0])
    assert _cosine_similarity(v1, v2) == pytest.approx(-1.0)


def test_cosine_similarity_orthogonal():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert _cosine_similarity(v1, v2) == pytest.approx(0.0)


def test_cosine_similarity_known_angle():
    v1 = np.array([1.0, 0.0])  # Along X
    v2 = np.array([1.0, 1.0])  # At 45 degrees
    expected_similarity = np.cos(np.pi / 4.0)  # cos(45 degrees)
    assert _cosine_similarity(v1, v2) == pytest.approx(expected_similarity)


def test_cosine_similarity_shape_mismatch():
    v1 = np.array([1.0, 2.0])
    v2 = np.array([1.0, 2.0, 3.0])
    with pytest.warns(UserWarning, match="different shapes"):
        similarity = _cosine_similarity(v1, v2)
    assert similarity == -1.0  # Error code


def test_cosine_similarity_zero_vector():
    v1 = np.array([1.0, 2.0])
    v_zero = np.array([0.0, 0.0])
    assert _cosine_similarity(v1, v_zero) == 0.0
    assert _cosine_similarity(v_zero, v1) == 0.0
    assert _cosine_similarity(v_zero, v_zero) == 0.0


# --- Tests for find_similar_items ---


@pytest.mark.filterwarnings("ignore:Error decoding embedding bytes")
@pytest.mark.filterwarnings(
    "ignore:Could not decode embedding for item 'item_E_bad_bytes'"
)
@pytest.mark.filterwarnings("ignore:Dimension mismatch for item 'item_F_wrong_dim'")
@pytest.mark.asyncio
async def test_find_similar_items_success(mock_gemini_client, patch_db_functions):
    """Test finding similar items successfully."""
    mock_get_all, mock_get_by_name = patch_db_functions
    query_text = "Find text like A"
    search_field = "nl"  # Assumes mock embeddings are for 'nl'
    query_vector_list = [[0.9, 0.1, 0.0]]  # Query vector similar to item_A
    mock_gemini_client.embed_content.return_value = (
        query_vector_list  # Mock embedding generation
    )

    top_n = 3
    results = await find_similar_items(
        query_text, search_field, mock_gemini_client, top_n=top_n, db_path=TEST_DB_PATH
    )

    # Expected similarities (approximate):
    # Query: [0.9, 0.1, 0.0] (normalized approx [0.994, 0.110, 0.0])
    # Item A: [1.0, 0.0, 0.0] -> Score ~ 0.994
    # Item B: [0.0, 1.0, 0.0] -> Score ~ 0.110
    # Item C: [-1.0, 0.0, 0.0]-> Score ~ -0.994
    # Item D: [0.707, 0.707, 0.0] -> Score ~ 0.994*0.707 + 0.110*0.707 ~ 0.78

    # Expected order: A, D, B (top 3)
    assert len(results) == top_n
    assert results[0][0].unique_name == "item_A"
    assert results[1][0].unique_name == "item_D"
    assert results[2][0].unique_name == "item_B"
    assert results[0][1] > results[1][1] > results[2][1]  # Check scores are descending
    assert results[0][1] == pytest.approx(0.994, abs=1e-3)
    assert results[1][1] == pytest.approx(0.780, abs=1e-3)
    assert results[2][1] == pytest.approx(0.110, abs=1e-3)

    # Verify mocks
    mock_gemini_client.embed_content.assert_awaited_once()  # generate_embedding called
    mock_get_all.assert_called_once_with(
        f"embedding_{search_field}", TEST_DB_PATH or DEFAULT_DB_PATH
    )
    # Check get_kb_item_by_name called for the top N unique names found
    expected_get_calls = [
        call("item_A", TEST_DB_PATH),
        call("item_D", TEST_DB_PATH),
        call("item_B", TEST_DB_PATH),
    ]
    mock_get_by_name.assert_has_calls(
        expected_get_calls, any_order=True
    )  # Order depends on sorting


@pytest.mark.asyncio
async def test_find_similar_items_no_embeddings_in_db(
    mock_gemini_client, patch_db_functions
):
    """Test case where no items have embeddings in the specified field."""
    mock_get_all, _ = patch_db_functions
    mock_get_all.return_value = []  # Simulate empty DB result
    mock_gemini_client.embed_content.return_value = [
        [0.1, 0.2, 0.3]
    ]  # Query gen succeeds

    results = await find_similar_items("Query", "nl", mock_gemini_client)

    assert results == []
    mock_gemini_client.embed_content.assert_awaited_once()
    mock_get_all.assert_called_once()


@pytest.mark.asyncio
async def test_find_similar_items_query_embedding_fails(
    mock_gemini_client, patch_db_functions
):
    """Test case where generating the query embedding fails."""
    mock_get_all, _ = patch_db_functions
    mock_gemini_client.embed_content.side_effect = Exception("Failed to generate")

    with pytest.warns(UserWarning, match="Failed to generate query embedding"):
        results = await find_similar_items("Query", "nl", mock_gemini_client)

    assert results == []
    mock_gemini_client.embed_content.assert_awaited_once()  # Attempted
    mock_get_all.assert_not_called()  # Should exit before DB query


@pytest.mark.asyncio
async def test_find_similar_items_db_fetch_fails(
    mock_gemini_client, patch_db_functions
):
    """Test case where fetching embeddings from DB fails."""
    mock_get_all, _ = patch_db_functions
    mock_get_all.side_effect = Exception("DB Connection Error")
    mock_gemini_client.embed_content.return_value = [
        [0.1, 0.2, 0.3]
    ]  # Query gen succeeds

    with pytest.warns(UserWarning, match="Failed to retrieve embeddings from database"):
        results = await find_similar_items("Query", "nl", mock_gemini_client)

    assert results == []
    mock_gemini_client.embed_content.assert_awaited_once()
    mock_get_all.assert_called_once()


@pytest.mark.asyncio
async def test_find_similar_items_handles_bad_bytes_and_dim_mismatch(
    mock_gemini_client, patch_db_functions
):
    """Test that items with bad bytes or wrong dimensions are skipped with warnings."""
    mock_get_all, mock_get_by_name = patch_db_functions
    # mock_kb_items_for_search includes item_E_bad_bytes and item_F_wrong_dim
    mock_gemini_client.embed_content.return_value = [
        [1.0, 0.0, 0.0]
    ]  # Query vector matches dim=3

    with pytest.warns(UserWarning) as record:
        results = await find_similar_items("Query", "nl", mock_gemini_client, top_n=5)

    # Verify warnings
    warning_messages = [str(w.message) for w in record]
    assert any(
        "Could not decode embedding for item 'item_E_bad_bytes'" in msg
        for msg in warning_messages
    )
    assert any(
        "Dimension mismatch for item 'item_F_wrong_dim'" in msg
        for msg in warning_messages
    )

    # Verify results only contain valid items (A, B, C, D)
    # Expected order based on query [1,0,0]: A (1.0), D (0.707), B (0.0), C (-1.0)
    assert len(results) == 4
    assert results[0][0].unique_name == "item_A"
    assert results[1][0].unique_name == "item_D"
    assert results[2][0].unique_name == "item_B"
    assert results[3][0].unique_name == "item_C"
    # Ensure bad items are not present
    assert not any(r[0].unique_name == "item_E_bad_bytes" for r in results)
    assert not any(r[0].unique_name == "item_F_wrong_dim" for r in results)


@pytest.mark.filterwarnings("ignore:Error decoding embedding bytes")
@pytest.mark.filterwarnings(
    "ignore:Could not decode embedding for item 'item_E_bad_bytes'"
)
@pytest.mark.filterwarnings("ignore:Dimension mismatch for item 'item_F_wrong_dim'")
@pytest.mark.asyncio
async def test_find_similar_items_handles_item_retrieval_fail(
    mock_gemini_client, patch_db_functions
):
    """Test when retrieving the full KBItem fails for a top match."""
    mock_get_all, mock_get_by_name = patch_db_functions
    mock_gemini_client.embed_content.return_value = [[1.0, 0.0, 0.0]]  # Query vector

    # Simulate get_kb_item_by_name failing for the top match ('item_A')
    original_side_effect = mock_get_by_name.side_effect

    def fail_for_A(name, db_path=None):
        if name == "item_A":
            return None
        return original_side_effect(name, db_path)

    mock_get_by_name.side_effect = fail_for_A

    with pytest.warns(UserWarning, match="Could not retrieve KBItem 'item_A'"):
        results = await find_similar_items("Query", "nl", mock_gemini_client, top_n=3)

    # Expected order without A: D, B, C
    # Should return top_n valid items if available, skipping the failed one
    assert len(results) == 2  # Only D and B are successfully retrieved
    assert results[0][0].unique_name == "item_D"  # D becomes top
    assert results[1][0].unique_name == "item_B"


@pytest.mark.asyncio
async def test_find_similar_items_invalid_search_field(mock_gemini_client):
    """Test ValueError if search_field is invalid."""
    with pytest.raises(ValueError, match="search_field must be 'nl' or 'latex'"):
        await find_similar_items("Query", "invalid_field", mock_gemini_client)


@pytest.mark.asyncio
async def test_find_similar_items_no_client_provided(patch_db_functions):
    """Test warning and empty list if GeminiClient is not available."""
    # Don't pass client
    with pytest.warns(
        UserWarning, match="GeminiClient not available for embedding search"
    ):
        results = await find_similar_items("Query", "nl", None)  # type: ignore

    assert results == []
