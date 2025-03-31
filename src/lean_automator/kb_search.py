# File: kb_search.py

import asyncio
import numpy as np
import os
import warnings
from typing import List, Optional, Tuple

try:
    from lean_automator.llm_call import GeminiClient
except ImportError:
    warnings.warn("llm_call.GeminiClient not found. Embedding generation/search will fail.", ImportWarning)
    GeminiClient = None # type: ignore

try:
    from lean_automator.kb_storage import KBItem, get_kb_item_by_name, get_all_items_with_embedding, DEFAULT_DB_PATH, EMBEDDING_DTYPE
except ImportError:
     warnings.warn("kb_storage not found. KB search functionality will be limited.", ImportWarning)
     KBItem = None # type: ignore
     get_kb_item_by_name = None # type: ignore
     get_all_items_with_embedding = None # type: ignore
     DEFAULT_DB_PATH = 'knowledge_base.sqlite'
     EMBEDDING_DTYPE = np.float32


"""
Provides functions for generating embeddings and performing semantic search
within the Knowledge Base using vector similarity.
"""

# --- Embedding Generation ---

async def generate_embedding(
    text: str,
    task_type: str,
    client: GeminiClient
) -> Optional[np.ndarray]:
    """
    Generates an embedding for the given text using the configured Gemini client.

    Args:
        text: The text content to embed.
        task_type: The task type for the embedding (e.g., "RETRIEVAL_DOCUMENT").
        client: An initialized GeminiClient instance.

    Returns:
        A numpy array representing the embedding, or None if generation fails.
    """
    if not client:
        warnings.warn("GeminiClient not available for embedding generation.")
        return None
    if not text:
        warnings.warn("Attempted to generate embedding for empty text.")
        return None

    try:
        # Assumes client.embed_content returns a list of embeddings,
        # even for single input. We take the first one.
        embeddings_list = await client.embed_content(contents=text, task_type=task_type)
        if embeddings_list and embeddings_list[0]:
            return np.array(embeddings_list[0], dtype=EMBEDDING_DTYPE)
        else:
            warnings.warn(f"Embedding generation returned empty result for task '{task_type}'.")
            return None
    except Exception as e:
        warnings.warn(f"Error generating embedding: {e}")
        return None

# --- Vector Utilities ---

def _bytes_to_vector(data: bytes) -> Optional[np.ndarray]:
    """Converts raw bytes back to a numpy float vector."""
    try:
        # Assumes the bytes represent a flat array of EMBEDDING_DTYPE
        vector = np.frombuffer(data, dtype=EMBEDDING_DTYPE)
        # OPTIONAL: Dimensionality check
        # expected_dim = 768
        # if vector.size != expected_dim:
        #    warnings.warn(f"Decoded vector has unexpected size {vector.size}, expected {expected_dim}")
        #    return None
        return vector
    except Exception as e:
        warnings.warn(f"Error decoding embedding bytes: {e}")
        return None

def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    if v1.shape != v2.shape:
        warnings.warn(f"Cannot compute cosine similarity for vectors with different shapes: {v1.shape} vs {v2.shape}")
        return -1.0 # Indicate error/invalid comparison

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        # Handle zero vectors - similarity is undefined or 0 depending on convention
        return 0.0

    # Ensure vectors are float type for dot product precision
    v1_f = v1.astype(np.float64)
    v2_f = v2.astype(np.float64)

    similarity = np.dot(v1_f, v2_f) / (norm_v1 * norm_v2)
    # Clamp result to handle potential floating point inaccuracies
    return np.clip(similarity, -1.0, 1.0)


# --- Semantic Search Function ---

async def find_similar_items(
    query_text: str,
    search_field: str, # 'nl' or 'latex'
    client: GeminiClient,
    *, # Make subsequent arguments keyword-only
    task_type_query: str = "RETRIEVAL_QUERY",
    db_path: Optional[str] = None,
    top_n: int = 5
) -> List[Tuple[KBItem, float]]:
    """
    Finds KBItems with embeddings similar to the query text.

    Performs a brute-force cosine similarity search across all items
    having an embedding for the specified field.

    Args:
        query_text: The natural language query.
        search_field: Which embedding to search ('nl' or 'latex').
        client: An initialized GeminiClient instance.
        task_type_query: The task type for embedding the query (default: RETRIEVAL_QUERY).
        db_path: Optional path to the database file.
        top_n: The maximum number of similar items to return.

    Returns:
        A list of tuples, each containing a matching KBItem and its similarity
        score, sorted by score descending. Returns empty list on error or no matches.
    """
    if search_field not in ['nl', 'latex']:
        raise ValueError("search_field must be 'nl' or 'latex'")
    if not client:
        warnings.warn("GeminiClient not available for embedding search.")
        return []
    if get_all_items_with_embedding is None or get_kb_item_by_name is None:
         warnings.warn("kb_storage functions not available for search.")
         return []

    embedding_column = f"embedding_{search_field}"
    effective_db_path = db_path or DEFAULT_DB_PATH

    # 1. Generate query embedding
    query_vector = await generate_embedding(query_text, task_type_query, client)
    if query_vector is None:
        warnings.warn(f"Failed to generate query embedding for: '{query_text[:50]}...'")
        return []

    # 2. Fetch all existing document embeddings for the target field
    try:
        all_embeddings_data = get_all_items_with_embedding(embedding_column, effective_db_path)
    except Exception as e:
         warnings.warn(f"Failed to retrieve embeddings from database: {e}")
         return []

    if not all_embeddings_data:
        print(f"No items found with embeddings in field '{embedding_column}'.")
        return []

    # 3. Calculate similarities
    similarities: List[Tuple[str, float]] = [] # Store (unique_name, score)
    for item_id, unique_name, embedding_blob in all_embeddings_data:
        doc_vector = _bytes_to_vector(embedding_blob)
        if doc_vector is not None:
            # Optional: Check dimensions match query_vector here
            if query_vector.shape == doc_vector.shape:
                score = _cosine_similarity(query_vector, doc_vector)
                similarities.append((unique_name, score))
            else:
                 warnings.warn(f"Dimension mismatch for item '{unique_name}': Query={query_vector.shape}, Doc={doc_vector.shape}. Skipping.")
        else:
             warnings.warn(f"Could not decode embedding for item '{unique_name}' (ID: {item_id}). Skipping.")


    # 4. Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 5. Get top N results and retrieve full KBItem objects
    top_results: List[Tuple[KBItem, float]] = []
    for unique_name, score in similarities[:top_n]:
        item = get_kb_item_by_name(unique_name, effective_db_path)
        if item:
            top_results.append((item, score))
        else:
            # Should ideally not happen if unique_name came from the DB
            warnings.warn(f"Could not retrieve KBItem '{unique_name}' after similarity search.")

    return top_results