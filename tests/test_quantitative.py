import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from agents.quantitative import QuantitativeAgent, UnsafeSQLError


def make_test_db(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE orders (order_id INTEGER, amount REAL, region TEXT)")
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?)", [(1, 100.0, "APAC"), (2, 200.0, "EMEA")])
    conn.commit()
    conn.close()
    return db_path


def test_validate_sql_rejects_non_select():
    agent = QuantitativeAgent(db_path=None)
    with pytest.raises(UnsafeSQLError):
        agent.validate_sql("DELETE FROM orders")


def test_validate_sql_rejects_multiple_statements():
    agent = QuantitativeAgent(db_path=None)
    with pytest.raises(UnsafeSQLError):
        agent.validate_sql("SELECT 1; DROP TABLE orders;")


def test_validate_sql_allows_plain_select():
    agent = QuantitativeAgent(db_path=None)
    agent.validate_sql("SELECT * FROM orders")  # should not raise


def test_execute_returns_rows(tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    rows = agent.execute("SELECT * FROM orders WHERE region = 'APAC'")
    assert rows == [{"order_id": 1, "amount": 100.0, "region": "APAC"}]


@patch("agents.quantitative._client")
def test_answer_generates_sql_executes_and_summarizes(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    mock_client.models.generate_content.side_effect = [
        MagicMock(text="SELECT * FROM orders"),
        MagicMock(text="There are 2 orders totaling $300."),
    ]

    result = agent.answer("How many orders are there?")

    assert result.sql == "SELECT * FROM orders"
    assert len(result.rows) == 2
    assert "300" in result.answer


@patch("agents.quantitative._client")
def test_answer_handles_unsafe_sql_gracefully(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    mock_client.models.generate_content.return_value = MagicMock(text="DROP TABLE orders")

    result = agent.answer("Delete everything")

    assert result.error is not None
    assert "couldn't safely run" in result.answer