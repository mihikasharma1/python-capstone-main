from unittest.mock import MagicMock, patch

from agents.manager import ManagerAgent, QueryType


def make_manager(with_quantitative=False):
    manager = ManagerAgent.__new__(ManagerAgent)
    manager.qualitative_agent = MagicMock()
    manager.quantitative_agent = MagicMock() if with_quantitative else None
    return manager


@patch("agents.manager.ManagerAgent.classify")
def test_routes_qualitative_query(mock_classify):
    manager = make_manager()
    mock_classify.return_value = QueryType.QUALITATIVE
    manager.qualitative_agent.answer.return_value = MagicMock(answer="answer text", citations=[])

    response = manager.handle("What is our security policy?")

    assert response.query_type == QueryType.QUALITATIVE
    manager.qualitative_agent.answer.assert_called_once()


@patch("agents.manager.ManagerAgent.classify")
def test_routes_quantitative_query_when_agent_present(mock_classify):
    manager = make_manager(with_quantitative=True)
    mock_classify.return_value = QueryType.QUANTITATIVE
    manager.quantitative_agent.answer.return_value = MagicMock(answer="42 orders", sql="SELECT ...", rows=[])

    response = manager.handle("How many orders were placed?")

    assert response.query_type == QueryType.QUANTITATIVE
    assert response.answer == "42 orders"
    manager.quantitative_agent.answer.assert_called_once()


@patch("agents.manager.ManagerAgent.classify")
def test_quantitative_without_agent_reports_unavailable(mock_classify):
    manager = make_manager()
    mock_classify.return_value = QueryType.QUANTITATIVE

    response = manager.handle("What was Q3 revenue?")

    assert "isn't wired in" in response.answer


@patch("agents.manager.ManagerAgent.classify")
def test_unsupported_query_gets_graceful_message(mock_classify):
    manager = make_manager()
    mock_classify.return_value = QueryType.UNSUPPORTED

    response = manager.handle("What's the weather today?")

    assert response.query_type == QueryType.UNSUPPORTED


@patch("agents.manager.ManagerAgent.synthesize")
@patch("agents.manager.ManagerAgent.decompose")
@patch("agents.manager.ManagerAgent.classify")
def test_complex_query_decomposes_and_synthesizes(mock_classify, mock_decompose, mock_synthesize):
    manager = make_manager(with_quantitative=True)
    mock_classify.return_value = QueryType.COMPLEX
    mock_decompose.return_value = ("qual sub-question", "quant sub-question")
    manager.qualitative_agent.answer.return_value = MagicMock(answer="qual answer", citations=[])
    manager.quantitative_agent.answer.return_value = MagicMock(answer="quant answer", sql="SELECT 1", rows=[])
    mock_synthesize.return_value = "unified answer"

    response = manager.handle("some compound question")

    assert response.query_type == QueryType.COMPLEX
    assert "unified answer" in response.answer
    assert "qual sub-question" in response.answer
    assert "quant sub-question" in response.answer
    manager.qualitative_agent.answer.assert_called_once_with("qual sub-question")
    manager.quantitative_agent.answer.assert_called_once_with("quant sub-question")


@patch("agents.manager._client")
def test_decompose_falls_back_to_original_question_on_bad_json(mock_client):
    manager = ManagerAgent.__new__(ManagerAgent)
    mock_client.models.generate_content.return_value = MagicMock(text="not valid json")

    qual_q, quant_q = manager.decompose("some question")

    assert qual_q == "some question"
    assert quant_q == "some question"