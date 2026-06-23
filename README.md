# 📄 Document Intelligence Pipeline

A production-style RAG (Retrieval-Augmented Generation) system for intelligent Q&A over financial documents.
Built with LangChain, FAISS, OpenAI, FastAPI, and Streamlit.

## 🏗️ Architecture

```
PDF Documents → Text Extraction (pdfplumber) → Chunking (RecursiveCharacterTextSplitter)
     → Embeddings (OpenAI text-embedding-3-small) → FAISS Vector Index
     → Query → Similarity Search → LangChain RAG Chain (GPT-4o-mini)
     → Cited Answer → FastAPI Response → Streamlit UI
     → Evidently Monitoring (query drift tracking)
```

## ✨ Features

- **Multi-document ingestion** — Process multiple PDF annual reports simultaneously
- **Semantic search** — FAISS cosine similarity with source diversity ranking
- **Citation-aware answers** — Every answer cites exact source document and page
- **Production REST API** — FastAPI with full OpenAPI docs at `/docs`
- **Interactive UI** — Streamlit dashboard with example questions
- **Drift monitoring** — Evidently tracks query patterns and latency over time
- **Docker support** — Full containerization with docker-compose

## 📁 Project Structure

```
doc-intelligence-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # PDF loading + chunking
│   │   └── embedder.py      # OpenAI embeddings + FAISS index
│   ├── retrieval/
│   │   └── retriever.py     # Similarity search + diversity ranking
│   ├── generation/
│   │   └── chain.py         # LangChain RAG chain
│   └── monitoring/
│       └── drift_tracker.py # Evidently query monitoring
├── api/
│   └── main.py              # FastAPI application
├── app/
│   └── streamlit_app.py     # Streamlit UI
├── data/
│   └── documents/           # Place your PDFs here
├── ingest.py                # Run once to index documents
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/doc-intelligence-pipeline
cd doc-intelligence-pipeline

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env       # Windows
# cp .env.example .env       # Mac/Linux
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Add Documents

Place PDF annual reports in `data/documents/`:
```
data/documents/
├── apple_annual_report_2025.pdf
├── tesla_annual_report_2024.pdf
├── sap_annual_report_2024.pdf
├── siemens_annual_report_2024.pdf
└── bmw_annual_report_2024.pdf
```

### 4. Index Documents

```bash
python ingest.py
```

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 6. Start the UI

```bash
streamlit run app/streamlit_app.py
```

UI available at: http://localhost:8501

## 🐳 Docker

```bash
copy .env.example .env    # Add your API key
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + index status |
| POST | `/ingest` | Upload and index PDF files |
| POST | `/query` | Ask a question, get cited answer |
| GET | `/documents` | List indexed documents |
| GET | `/stats` | Query statistics |
| GET | `/drift-report` | Generate Evidently drift report |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple total revenue in 2024?", "top_k": 5}'
```

## 📊 Sample Questions

- *"What was Apple's total revenue in fiscal year 2024?"*
- *"What are Tesla's main risk factors?"*
- *"How did SAP's cloud revenue grow year over year?"*
- *"Compare BMW and Siemens operating profit margins"*
- *"What dividends did Siemens pay to shareholders?"*

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (IndexFlatIP) |
| Orchestration | LangChain |
| API | FastAPI |
| UI | Streamlit |
| Monitoring | Evidently |
| Containerization | Docker |

## 📈 Monitoring

Query logs are saved to `data/query_logs.jsonl`.
Generate a drift report after 100+ queries:

```bash
curl http://localhost:8000/drift-report
```

Report saved to `data/reports/drift_report_TIMESTAMP.html`
