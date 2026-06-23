"""
Retriever Module
Performs similarity search over FAISS index to find relevant chunks.
"""

import os
import numpy as np
import faiss
from typing import List, Dict, Tuple
from src.ingestion.embedder import get_embedding, load_index
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K_RESULTS", 5))


def retrieve(query: str, index: faiss.Index, chunks: List[Dict], top_k: int = TOP_K) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a query using cosine similarity.

    Args:
        query: User's question
        index: FAISS index
        chunks: List of chunk metadata dicts
        top_k: Number of results to return

    Returns:
        List of relevant chunks with similarity scores
    """
    # Embed the query
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    faiss.normalize_L2(query_embedding)

    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)

    # Build results with metadata
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx].copy()
        chunk["similarity_score"] = float(score)
        results.append(chunk)

    return results


def retrieve_with_diversity(
    query: str,
    index: faiss.Index,
    chunks: List[Dict],
    top_k: int = TOP_K,
    diversity_k: int = 10
) -> List[Dict]:
    """
    Retrieve with source diversity — ensures results come from multiple documents.
    Fetches diversity_k results then picks top_k ensuring source variety.
    """
    candidates = retrieve(query, index, chunks, top_k=diversity_k)

    # Deduplicate by source, keeping highest scoring chunk per source
    seen_sources = {}
    for chunk in candidates:
        source = chunk["source"]
        if source not in seen_sources:
            seen_sources[source] = chunk
        # If same source appears again, keep the one with higher score
        elif chunk["similarity_score"] > seen_sources[source]["similarity_score"]:
            seen_sources[source] = chunk

    # Return top_k after deduplication, sorted by score
    diverse_results = sorted(seen_sources.values(), key=lambda x: x["similarity_score"], reverse=True)
    return diverse_results[:top_k]


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    from src.ingestion.embedder import load_index

    index, chunks = load_index()
    results = retrieve("What was Apple's revenue in 2024?", index, chunks)
    for r in results:
        print(f"\n📄 {r['source']} | Page {r['page']} | Score: {r['similarity_score']:.3f}")
        print(r['text'][:200])
