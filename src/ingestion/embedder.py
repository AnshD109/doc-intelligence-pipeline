"""
Embedder Module
Generates OpenAI embeddings and builds/persists a FAISS vector index.
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks_metadata.pkl"


def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text string."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.replace("\n", " ")
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """
    Get embeddings for a list of texts in batches.
    Batching reduces API calls and speeds up indexing.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[t.replace("\n", " ") for t in batch]
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")

    return all_embeddings


def build_faiss_index(chunks: List[Dict]) -> faiss.Index:
    """
    Build a FAISS index from document chunks.
    Uses IndexFlatIP (inner product) for cosine similarity search.
    """
    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings_batch(texts)

    # Convert to numpy array
    embedding_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embedding_matrix)

    # Build flat index (exact search — suitable for our document scale)
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)

    print(f"✅ FAISS index built with {index.ntotal} vectors (dim={dimension})")
    return index


def save_index(index: faiss.Index, chunks: List[Dict]):
    """Persist FAISS index and chunk metadata to disk."""
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Index saved to {INDEX_PATH}")
    print(f"✅ Metadata saved to {CHUNKS_PATH}")


def load_index():
    """Load persisted FAISS index and chunk metadata."""
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"No index found at {INDEX_PATH}. Run ingestion first.")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
    return index, chunks


if __name__ == "__main__":
    from src.ingestion.loader import load_documents_from_folder

    chunks = load_documents_from_folder("data/documents")
    index = build_faiss_index(chunks)
    save_index(index, chunks)
