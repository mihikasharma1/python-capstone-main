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