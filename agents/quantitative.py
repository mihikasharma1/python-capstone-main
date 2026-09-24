"""Quantitative agent: translates natural language into SQL, executes it, and summarizes results."""
import re
import sqlite3
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from agents.gemini_utils import call_with_progress

from config import GEMINI_API_KEY, GENERATION_MODEL, DB_PATH

_client = genai.Client(api_key=GEMINI_API_KEY)

SCHEMA_DESCRIPTION = """
Table: customers
  customer_id INTEGER PRIMARY KEY
  name TEXT
  region TEXT            -- one of: North America, EMEA, APAC, LATAM
  signup_date TEXT        -- ISO date, e.g. 2023-05-14
  status TEXT             -- 'active' or 'churned'

Table: orders
  order_id INTEGER PRIMARY KEY
  customer_id INTEGER     -- references customers.customer_id
  order_date TEXT         -- ISO date
  amount REAL             -- order value in USD
  region TEXT             -- denormalized copy of the customer's region at order time
  year INTEGER
  quarter TEXT            -- 'Q1'..'Q4'
"""

SQL_PROMPT = """You are a SQLite expert. Given the schema below, write ONE read-only SQL query
that answers the question. Output ONLY the SQL, no explanation, no markdown fences.
Only use SELECT statements against the tables/columns given. Never use INSERT, UPDATE,
DELETE, DROP, ALTER, or PRAGMA.

Schema:
{schema}

Question: {question}
SQL:"""

SUMMARY_PROMPT = """Question: {question}
SQL used: {sql}
Result rows (as a list of dicts, possibly truncated): {rows}

In 2-3 sentences, answer the question in plain language based on these results.
Mention concrete numbers from the rows. Do not invent data not present in the rows."""

_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", re.IGNORECASE)


class UnsafeSQLError(Exception):
    pass


@dataclass
class QuantitativeAnswer:
    answer: str
    sql: str
    rows: list = field(default_factory=list)
    error: str | None = None


class QuantitativeAgent:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def generate_sql(self, question: str) -> str:
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=SQL_PROMPT.format(schema=SCHEMA_DESCRIPTION, question=question),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                ),
            ),
            label="generating SQL for your question",
        )
        sql = (response.text or "").strip()
        if not sql:
            raise UnsafeSQLError("Model returned no SQL.")
        return re.sub(r"^```sql\s*|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    def validate_sql(self, sql: str) -> None:
        stripped = sql.strip().rstrip(";")
        if not stripped.lower().startswith("select"):
            raise UnsafeSQLError("Only SELECT statements are allowed.")
        if ";" in stripped:
            raise UnsafeSQLError("Multiple statements are not allowed.")
        if _FORBIDDEN.search(stripped):
            raise UnsafeSQLError("Query contains a disallowed keyword.")

    def execute(self, sql: str) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchmany(50)]
        finally:
            conn.close()
        return rows

    def answer(self, question: str) -> QuantitativeAnswer:
        sql = self.generate_sql(question)
        try:
            self.validate_sql(sql)
            rows = self.execute(sql)
        except (UnsafeSQLError, sqlite3.Error) as e:
            return QuantitativeAnswer(
                answer=f"I couldn't safely run a query for that question ({e}).",
                sql=sql, rows=[], error=str(e),
            )

        if not rows:
            return QuantitativeAnswer(answer="That query ran but returned no matching data.", sql=sql, rows=[])

        summary = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=SUMMARY_PROMPT.format(question=question, sql=sql, rows=rows[:10]),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=200,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                ),
            ),
            label="summarizing the query results",
        )
        return QuantitativeAnswer(answer=(summary.text or "I ran the query but couldn't summarize it.").strip(), sql=sql, rows=rows)