"""
FastAPI Application
REST API for the Document Intelligence Pipeline.

Endpoints:
- POST /ingest       — Upload and index PDF documents
- POST /query        — Ask a question against indexed documents
- GET  /documents    — List indexed documents
- GET  /stats        — Query statistics
- GET  /health       — Health check
"""

import os
import time
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Lazy-load heavy modules
index = None
chunks = None
chain = None

app = FastAPI(
    title="Document Intelligence Pipeline",
    description="RAG-powered document Q&A API using LangChain, FAISS, and OpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "data/documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Request/Response Models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class SourceCitation(BaseModel):
    file: str
    page: int
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    latency_ms: float

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    total_chunks: int

class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    avg_question_length: float
    avg_answer_length: float


# ── Startup ──────────────────────────────────────────────────────────────────

def load_resources():
    """Load FAISS index and RAG chain on startup."""
    global index, chunks, chain
    try:
        from src.ingestion.embedder import load_index
        from src.generation.chain import build_rag_chain
        index, chunks = load_index()
        chain = build_rag_chain()
        print("✅ Index and chain loaded successfully")
    except FileNotFoundError:
        print("⚠️  No index found. Upload documents and call /ingest first.")


@app.on_event("startup")
async def startup_event():
    load_resources()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "documents_indexed": len(set(c["source"] for c in chunks)) if chunks else 0
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """Upload PDF files and build/update the FAISS index."""
    global index, chunks, chain

    saved_files = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        dest = Path(UPLOAD_DIR) / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_files.append(str(dest))
        print(f"📥 Saved: {file.filename}")

    # Re-index all documents
    from src.ingestion.loader import load_documents_from_folder
    from src.ingestion.embedder import build_faiss_index, save_index
    from src.generation.chain import build_rag_chain

    chunks = load_documents_from_folder(UPLOAD_DIR)
    index = build_faiss_index(chunks)
    save_index(index, chunks)
    chain = build_rag_chain()

    return IngestResponse(
        message=f"Successfully indexed {len(saved_files)} file(s)",
        files_processed=len(saved_files),
        total_chunks=len(chunks)
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and get a cited answer from indexed documents."""
    if index is None or chunks is None:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Please upload PDFs via /ingest first."
        )

    start = time.time()

    from src.generation.chain import answer_question
    from src.monitoring.drift_tracker import log_query

    result = answer_question(
        question=request.question,
        index=index,
        chunks=chunks,
        chain=chain,
        top_k=request.top_k
    )

    latency_ms = round((time.time() - start) * 1000, 2)

    # Log for monitoring
    log_query(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency_ms
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[SourceCitation(**s) for s in result["sources"]],
        latency_ms=latency_ms
    )


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    pdf_files = list(Path(UPLOAD_DIR).glob("*.pdf"))
    return {
        "documents": [f.name for f in pdf_files],
        "total": len(pdf_files)
    }


@app.get("/stats")
def get_stats():
    """Return query statistics for monitoring."""
    from src.monitoring.drift_tracker import get_query_stats
    return get_query_stats()


@app.get("/drift-report")
def generate_drift_report():
    """Trigger an Evidently drift report generation."""
    from src.monitoring.drift_tracker import generate_drift_report
    report_path = generate_drift_report()
    if report_path:
        return {"message": "Report generated", "path": report_path}
    return {"message": "Not enough queries yet for drift analysis (need 100+)"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
