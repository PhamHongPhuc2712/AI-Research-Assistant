"""
hybrid_search.py — Hybrid retrieval combining semantic and keyword search.

Runs semantic search (ChromaDB) + keyword search (BM25) in parallel,
then fuses results via Reciprocal Rank Fusion.

STUB: Returns empty results until fully implemented.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Run hybrid search over the vector store and keyword index.

    Combines semantic search (ChromaDB) with keyword search (BM25) and
    fuses results using Reciprocal Rank Fusion (RRF).

    Args:
        query: The user's search query.
        top_k: Number of top results to return.

    Returns:
        A list of dicts, each containing:
            - "content": the chunk text
            - "metadata": source metadata (file, page, etc.)
            - "score": fused relevance score

    Note:
        This is a stub — returns an empty list until the full
        hybrid search pipeline is implemented.
    """
    logger.warning(
        "hybrid_search.search() is a stub — returning empty results. "
        "Implement semantic + BM25 fusion to enable retrieval."
    )
    return []
