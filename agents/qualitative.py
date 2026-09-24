"""Qualitative RAG agent: semantic search over company docs + Gemini-generated answer with citations."""
from dataclasses import dataclass, field

import chromadb
from chromadb.utils import embedding_functions
from google import genai
from google.genai import types

from agents.gemini_utils import call_with_progress
from config import CHROMA_DIR, GEMINI_API_KEY, GENERATION_MODEL, EMBEDDING_MODEL_NAME, TOP_K, RELEVANCE_THRESHOLD
_client = genai.Client(api_key=GEMINI_API_KEY)


@dataclass
class Citation:
    source: str
    chunk_index: int
    similarity: float
    text: str


@dataclass
class QualitativeAnswer:
    answer: str
    citations: list = field(default_factory=list)
    found: bool = True


class QualitativeAgent:
    """Retrieves relevant document chunks and asks Gemini to answer using only that context."""

    def __init__(self):
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        self.collection = chroma_client.get_collection("company_docs", embedding_function=embed_fn)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Citation]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        citations = []
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            similarity = 1 - distance
            citations.append(Citation(meta["source"], meta["chunk_index"], similarity, doc))
        return citations

    def answer(self, query: str) -> QualitativeAnswer:
        citations = self.retrieve(query)
        relevant = [c for c in citations if c.similarity >= RELEVANCE_THRESHOLD]

        if not relevant:
            return QualitativeAnswer(
                answer="I couldn't find anything in the company documentation that answers this.",
                citations=citations,
                found=False,
            )

        context = "\n\n".join(f"[{c.source}#{c.chunk_index}] {c.text}" for c in relevant)
        prompt = (
            "Answer the question using ONLY the context below. "
            "Cite sources inline like [filename#chunk_index]. "
            "If the context doesn't fully answer the question, say so explicitly.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}"
        )
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=500,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                ),
            ),
            label="generating an answer from the retrieved documents",
        )
        return QualitativeAnswer(
            answer=(response.text or "I generated a response but it came back empty.").strip(),
            citations=relevant,
            found=True,
        )