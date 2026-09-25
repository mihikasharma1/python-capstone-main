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


def make_unrelated_db(tmp_path):
    """A completely different schema, to prove safety checks are schema-agnostic
    rather than hardcoded to 'customers'/'orders'."""
    db_path = tmp_path / "other.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (id INTEGER, name TEXT, price REAL)")
    conn.executemany("INSERT INTO products VALUES (?, ?, ?)", [(1, "Widget", 9.99)])
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
    agent.validate_sql("SELECT * FROM orders")


def test_validate_sql_allows_cte():
    agent = QuantitativeAgent(db_path=None)
    agent.validate_sql(
        "WITH regional AS (SELECT region, COUNT(*) AS n FROM orders GROUP BY region) SELECT * FROM regional"
    )


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


@patch("agents.quantitative._client")
def test_answer_declines_when_sql_ignores_schema(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    mock_client.models.generate_content.return_value = MagicMock(text="SELECT 0")

    result = agent.answer("What's our employee headcount?")

    assert result.error == "no_matching_data_source"
    assert "don't have data" in result.answer


@patch("agents.quantitative._client")
def test_answer_declines_on_no_query_sentinel(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    mock_client.models.generate_content.return_value = MagicMock(text="NO_QUERY")

    result = agent.answer("How many employees took maternity leave?")

    assert result.error == "no_matching_data_source"


@patch("agents.quantitative._client")
def test_answer_declines_when_sql_is_a_literal_explanation(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_test_db(tmp_path))
    mock_client.models.generate_content.return_value = MagicMock(
        text="SELECT 'This question cannot be answered using the provided schema about customers.'"
    )

    result = agent.answer("How does employee satisfaction compare to industry standards?")

    assert result.error == "no_matching_data_source"


def test_describe_schema_reflects_whatever_db_is_passed(tmp_path):
    agent = QuantitativeAgent(db_path=make_unrelated_db(tmp_path))
    schema = agent._describe_schema()
    assert "products" in schema
    assert "price" in schema
    assert "orders" not in schema


def test_known_tables_reflects_whatever_db_is_passed(tmp_path):
    agent = QuantitativeAgent(db_path=make_unrelated_db(tmp_path))
    assert agent._known_tables() == {"products"}


@patch("agents.quantitative._client")
def test_declines_hallucinated_table_not_in_this_database(mock_client, tmp_path):
    agent = QuantitativeAgent(db_path=make_unrelated_db(tmp_path))
    mock_client.models.generate_content.return_value = MagicMock(text="SELECT * FROM customers")

    result = agent.answer("How many customers do we have?")

    assert result.error == "no_matching_data_source"