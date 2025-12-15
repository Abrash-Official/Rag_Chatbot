import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from loguru import logger

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "redjet_collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_documents(paths: List[Path]) -> List:
    docs = []
    for p in paths:
        if p.suffix.lower() in [".pdf"]:
            logger.info(f"Loading PDF: {p}")
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".txt", ".md", ".log"]:
            logger.info(f"Loading text: {p}")
            loader = TextLoader(str(p), encoding="utf-8")
            docs.extend(loader.load())
        else:
            logger.warning(f"Skipping unsupported file: {p}")
    return docs

def chunk_documents(docs: List) -> List:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    # Attach stable metadata
    for i, c in enumerate(chunks):
        c.metadata = c.metadata or {}
        c.metadata["chunk_id"] = f"chunk-{i}"
    return chunks

def get_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vs = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vs

def upsert_chunks(vs: Chroma, chunks: List):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [m.get("chunk_id") for m in metadatas]
    logger.info(f"Upserting {len(chunks)} chunks into collection '{CHROMA_COLLECTION}'")
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    # vs.persist()
    logger.info("Persisted Chroma collection")

if __name__ == "__main__":
    data_dir = Path("./data")
    files = sorted([p for p in data_dir.glob("**/*") if p.is_file()])
    if not files:
        logger.error("No files found in ./data. Add PDFs or text files.")
        raise SystemExit(1)

    documents = load_documents(files)
    logger.info(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    vectorstore = get_vectorstore()
    upsert_chunks(vectorstore, chunks)
    logger.info("Ingestion complete")