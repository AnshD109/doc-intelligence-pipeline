"""
Streamlit Frontend
Clean UI for the Document Intelligence Pipeline.
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path

API_URL = "http://localhost:8000"

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Intelligence Pipeline",
    page_icon="📄",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📄 Document Intelligence Pipeline")
st.markdown("*RAG-powered Q&A over financial documents — powered by LangChain, FAISS & GPT-4o-mini*")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    top_k = st.slider("Number of context chunks", min_value=3, max_value=10, value=5)

    st.divider()
    st.header("📥 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF Annual Reports",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Index Documents", type="primary"):
        with st.spinner("Indexing documents... This may take a few minutes."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
            try:
                response = requests.post(f"{API_URL}/ingest", files=files, timeout=300)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Indexed {data['files_processed']} files ({data['total_chunks']} chunks)")
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"❌ Could not connect to API: {e}")

    st.divider()

    # Show indexed documents
    st.header("📚 Indexed Documents")
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        if response.status_code == 200:
            docs = response.json()
            if docs["documents"]:
                for doc in docs["documents"]:
                    st.markdown(f"• {doc}")
            else:
                st.info("No documents indexed yet.")
    except:
        st.warning("API not connected.")

    st.divider()

    # Stats
    st.header("📊 Query Stats")
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            st.metric("Total Queries", stats.get("total_queries", 0))
            if stats.get("total_queries", 0) > 0:
                st.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.0f}ms")
    except:
        pass


# ── Main Q&A Interface ────────────────────────────────────────────────────────

# Example questions
st.subheader("💡 Example Questions")
example_cols = st.columns(3)
examples = [
    "What was Apple's total revenue in fiscal year 2024?",
    "What are Tesla's key risk factors?",
    "How did SAP's cloud business perform?",
    "Compare BMW and Siemens revenue growth",
    "What is Apple's R&D expenditure?",
    "What dividends did Siemens pay?"
]

for i, example in enumerate(examples):
    col = example_cols[i % 3]
    if col.button(example, key=f"ex_{i}", use_container_width=True):
        st.session_state["question"] = example

st.divider()

# Question input
question = st.text_input(
    "❓ Ask a question about the documents",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What was Apple's net income in 2024?",
    key="question_input"
)

col1, col2 = st.columns([1, 5])
ask_button = col1.button("🔍 Ask", type="primary", use_container_width=True)
col2.markdown("")

if ask_button and question:
    with st.spinner("Searching documents and generating answer..."):
        try:
            start = time.time()
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=60
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()

                # Answer box
                st.subheader("💬 Answer")
                st.markdown(
                    f'<div style="background:#1e3a5f;color:#ffffff;padding:1.4rem;border-radius:8px;'
                    f'border-left:5px solid #4da6ff;font-size:1rem;line-height:1.7;">'
                    f'{data["answer"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("⏱️ Latency", f"{data['latency_ms']:.0f}ms")
                m2.metric("📄 Sources Found", len(data["sources"]))
                m3.metric("🔢 Chunks Retrieved", top_k)

                # Sources
                if data["sources"]:
                    st.subheader("📚 Sources")
                    for src in data["sources"]:
                        with st.expander(f"📄 {src['file']} — Page {src['page']} (relevance: {src['score']:.3f})"):
                            st.markdown(f"**File:** {src['file']}")
                            st.markdown(f"**Page:** {src['page']}")
                            st.markdown(f"**Relevance Score:** {src['score']:.4f}")

            elif response.status_code == 503:
                st.warning("⚠️ No documents indexed. Please upload PDFs in the sidebar first.")
            else:
                st.error(f"❌ API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif ask_button and not question:
    st.warning("Please enter a question.")

# ── Chat History ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []
