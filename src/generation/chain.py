"""
RAG Chain Module
LangChain-based answer generation with citation-aware prompting.
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from src.retrieval.retriever import retrieve_with_diversity, format_context
from src.ingestion.embedder import load_index
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# System prompt — designed for citation-aware enterprise document Q&A
SYSTEM_PROMPT = """You are a professional financial document analyst assistant.
You answer questions strictly based on the provided document context.

Rules:
1. Only use information from the provided context — never use prior knowledge
2. Always cite your sources as [Source N: filename, Page X]
3. If the context doesn't contain the answer, say "I couldn't find this information in the provided documents."
4. Be concise, accurate, and professional
5. For financial figures, always include the unit (e.g., $billions, €millions)
"""

USER_PROMPT = """Context from documents:
{context}

Question: {question}

Provide a clear, cited answer based only on the context above."""


def build_rag_chain():
    """Build and return the LangChain RAG chain."""
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,  # Deterministic for factual Q&A
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT)
    ])

    chain = prompt | llm | StrOutputParser()
    return chain


def answer_question(
    question: str,
    index,
    chunks: List[Dict],
    chain,
    top_k: int = 5
) -> Dict:
    """
    Full RAG pipeline: retrieve relevant chunks → generate cited answer.

    Returns dict with answer, sources, and retrieved chunks.
    """
    # Step 1: Retrieve relevant chunks
    relevant_chunks = retrieve_with_diversity(question, index, chunks, top_k=top_k)

    if not relevant_chunks:
        return {
            "answer": "No relevant documents found. Please ensure documents are indexed.",
            "sources": [],
            "chunks": []
        }

    # Step 2: Format context
    context = format_context(relevant_chunks)

    # Step 3: Generate answer
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # Step 4: Build source citations
    sources = []
    seen = set()
    for chunk in relevant_chunks:
        key = f"{chunk['source']}_p{chunk['page']}"
        if key not in seen:
            sources.append({
                "file": chunk["source"],
                "page": chunk["page"],
                "score": round(chunk["similarity_score"], 3)
            })
            seen.add(key)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": relevant_chunks
    }


if __name__ == "__main__":
    print("Loading index...")
    index, chunks = load_index()
    chain = build_rag_chain()

    questions = [
        "What was Apple's total revenue in fiscal year 2024?",
        "What are Tesla's main risk factors?",
        "How did SAP's cloud revenue grow?"
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = answer_question(q, index, chunks, chain)
        print(f"💬 {result['answer']}")
        print(f"📚 Sources: {result['sources']}")
