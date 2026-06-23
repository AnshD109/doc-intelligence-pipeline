"""
Document Loader & Chunker
Handles PDF ingestion, text extraction, and recursive chunking.
"""

import os
import pdfplumber
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file page by page.
    Returns a list of dicts with page number, text, and source filename.
    """
    pages = []
    pdf_name = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    "page": page_num,
                    "text": text.strip(),
                    "source": pdf_name,
                    "path": pdf_path
                })

    print(f"✅ Extracted {len(pages)} pages from {pdf_name}")
    return pages


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Split extracted pages into smaller overlapping chunks
    using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
                "text": split,
                "source": page["source"],
                "page": page["page"],
            })

    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """
    Load and chunk all PDFs from a folder.
    """
    folder = Path(folder_path)
    all_chunks = []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  No PDF files found in {folder_path}")
        return []

    for pdf_file in pdf_files:
        print(f"\n📄 Processing: {pdf_file.name}")
        pages = extract_text_from_pdf(str(pdf_file))
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)

    print(f"\n✅ Total chunks across all documents: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Quick test
    chunks = load_documents_from_folder("data/documents")
    print(f"\nSample chunk:\n{chunks[0]['text'][:300]}" if chunks else "No chunks found")
