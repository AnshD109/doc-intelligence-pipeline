"""
Run this script once to index all PDFs in data/documents/
Usage: python ingest.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.loader import load_documents_from_folder
from src.ingestion.embedder import build_faiss_index, save_index

if __name__ == "__main__":
    print("=" * 50)
    print("📄 Document Intelligence Pipeline — Ingestion")
    print("=" * 50)

    print("\n📂 Loading and chunking documents...")
    chunks = load_documents_from_folder("data/documents")

    if not chunks:
        print("\n❌ No documents found in data/documents/")
        print("Please add PDF files to the data/documents/ folder and run again.")
        sys.exit(1)

    print(f"\n🔢 Building FAISS index for {len(chunks)} chunks...")
    index = build_faiss_index(chunks)

    print("\n💾 Saving index to disk...")
    save_index(index, chunks)

    print("\n✅ Ingestion complete!")
    print(f"   Documents processed: {len(set(c['source'] for c in chunks))}")
    print(f"   Total chunks indexed: {len(chunks)}")
    print("\n▶️  Next: Run the API with: uvicorn api.main:app --reload")
