import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "redjet_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Prompts
SYSTEM_PROMPT = (
    "You are a precise, pragmatic assistant. You refine user queries to maximize retrieval "
    "quality, maintain factual grounding, and respond with clear, concise explanations "
    "based strictly on provided context. If context is insufficient, state limitations."
)

DEVELOPER_PROMPT = (
    "First, refine the user’s raw query to better target the corpus. Then answer using only "
    "the retrieved chunks. Prefer technical clarity, cite chunk IDs inline where relevant, "
    "and avoid speculation."
)

REFINE_TEMPLATE = (
    "Refine the following query to maximize retrieval quality from a technical corpus.\n\n"
    "Raw Query:\n{query}\n\n"
    "Output only the refined query, no commentary."
)

ANSWER_TEMPLATE = (
    "System:\n{system}\nDeveloper:\n{dev}\n\n"
    "User Query: {query}\nRefined Query: {refined}\n\n"
    "Context Chunks (id: content):\n{context}\n\n"
    "Task: Provide a direct, well-structured answer grounded strictly in the context. "
    "Prefer concise, technical clarity. If uncertain or insufficient context, say so."
)

# FastAPI
app = FastAPI(title="Redjet RAG Service", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    min_score: float = 0.0  # optional similarity score threshold

class ChunkResult(BaseModel):
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    refined_query: str
    top_chunks: List[ChunkResult]
    answer: str

def get_llm():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing")
    return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL)

def get_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

def refine_query(llm: ChatGroq, raw_query: str) -> str:
    prompt = REFINE_TEMPLATE.format(query=raw_query)
    resp = llm.invoke(prompt)
    refined = resp.content.strip()
    logger.info(f"Refined query: {refined}")
    return refined

def retrieve_chunks(vs: Chroma, query: str, top_k: int) -> List[Dict[str, Any]]:
    # similarity_search_with_relevance_scores returns (Document, score)
    results = vs.similarity_search_with_relevance_scores(query, k=top_k)
    structured = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        structured.append({
            "chunk_id": chunk_id,
            "score": float(score),
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return structured

def format_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"{c['chunk_id']}: {c['content']}")
    return "\n---\n".join(lines)

def answer_with_context(llm: ChatGroq, user_query: str, refined_query: str, chunks: List[Dict[str, Any]]) -> str:
    context_str = format_context(chunks)
    prompt = ANSWER_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        dev=DEVELOPER_PROMPT,
        query=user_query,
        refined=refined_query,
        context=context_str
    )
    resp = llm.invoke(prompt)
    return resp.content.strip()

@app.post("/query", response_model=QueryResponse)
def query_endpoint(payload: QueryRequest):
    llm = get_llm()
    vs = get_vectorstore()

    refined = refine_query(llm, payload.query)
    raw_chunks = retrieve_chunks(vs, refined, payload.top_k)

    if payload.min_score > 0:
        filtered = [c for c in raw_chunks if c["score"] >= payload.min_score]
    else:
        filtered = raw_chunks

    # Convert to Pydantic response chunks
    top_chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            score=c["score"],
            content=c["content"],
            metadata=c["metadata"]
        )
        for c in filtered
    ]

    answer = answer_with_context(llm, payload.query, refined, filtered)

    return QueryResponse(
        refined_query=refined,
        top_chunks=top_chunks,
        answer=answer
    )

@app.get("/health")
def health():
    return {"status": "ok", "collection": CHROMA_COLLECTION}