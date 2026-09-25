from unittest.mock import MagicMock, patch

import chromadb
from chromadb.utils import embedding_functions

from agents.qualitative import QualitativeAgent
from config import EMBEDDING_MODEL_NAME


def make_test_collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    collection = client.create_collection(
        "company_docs", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["policy-0"],
        documents=["All employees must complete security training annually."],
        metadatas=[{"source": "security_policy.md", "chunk_index": 0}],
    )
    return collection


def test_retrieve_returns_citations(tmp_path):
    agent = QualitativeAgent.__new__(QualitativeAgent)
    agent.collection = make_test_collection(tmp_path)

    citations = agent.retrieve("What security training is required?")
    assert citations
    assert citations[0].source == "security_policy.md"


def test_answer_reports_not_found_below_threshold(tmp_path):
    agent = QualitativeAgent.__new__(QualitativeAgent)
    agent.collection = make_test_collection(tmp_path)

    result = agent.answer("What is the capital of France?")
    assert result.found is False


@patch("agents.qualitative._client")
def test_answer_calls_gemini_with_context(mock_client, tmp_path):
    agent = QualitativeAgent.__new__(QualitativeAgent)
    agent.collection = make_test_collection(tmp_path)

    mock_client.models.generate_content.return_value = MagicMock(
        text="Employees must train annually [security_policy.md#0]."
    )
    result = agent.answer("What security training is required?")

    assert "annually" in result.answer
    assert result.citations
    mock_client.models.generate_content.assert_called_once()


@patch("agents.qualitative._client")
def test_answer_handles_empty_gemini_response(mock_client, tmp_path):
    """Regression test: a model that spends its whole token budget on internal
    reasoning can return text=None — this must degrade gracefully, not crash."""
    agent = QualitativeAgent.__new__(QualitativeAgent)
    agent.collection = make_test_collection(tmp_path)

    mock_client.models.generate_content.return_value = MagicMock(text=None)
    result = agent.answer("What security training is required?")

    assert result.answer
    assert result.found is True