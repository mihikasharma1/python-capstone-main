"""Chunk the documents in data/docs and load them into a persistent Chroma collection."""
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

    try:
        client.delete_collection("company_docs")
    except Exception:
        pass
    collection = client.create_collection(
        "company_docs", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )

    ids, documents, metadatas = [], [], []
    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    for path in doc_paths:
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{path.stem}-{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(ids)} chunks from {len(doc_paths)} documents into {CHROMA_DIR}")


if __name__ == "__main__":
    build()