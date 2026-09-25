"""Chunk the documents in data/docs and load them into a persistent Chroma collection."""
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DOCS_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text: str, min_chunk_size: int = 200) -> list[str]:
    """Split on paragraph breaks so a single policy statement, threshold, or process
    description never gets sliced across an arbitrary character-count boundary. Tiny
    paragraphs (e.g. a heading on its own line) get merged forward so we don't index a
    near-empty, low-signal chunk on its own."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buffer = [], ""
    for p in paragraphs:
        buffer = f"{buffer}\n\n{p}".strip() if buffer else p
        if len(buffer) >= min_chunk_size:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks

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