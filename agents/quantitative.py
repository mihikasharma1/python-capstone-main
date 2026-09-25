"""Quantitative agent: translates natural language into SQL, executes it, and summarizes results.
Schema-agnostic by design — the prompt and the safety checks both introspect the live database,
so adding/renaming/dropping tables or columns never requires touching this file."""
import re
import sqlite3
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from agents.gemini_utils import call_with_progress

from config import GEMINI_API_KEY, GENERATION_MODEL, DB_PATH

_client = genai.Client(api_key=GEMINI_API_KEY)

# Optional, purely additive semantic hints for columns whose meaning isn't obvious from
# name/type alone (e.g. enum-like TEXT columns). Safe to leave stale or incomplete — the
# schema description and safety checks work correctly with or without entries here.
COLUMN_NOTES = {
    ("customers", "region"): "one of: North America, EMEA, APAC, LATAM",
    ("customers", "status"): "one of: active, churned",
    ("orders", "quarter"): "one of: Q1, Q2, Q3, Q4",
}

SQL_PROMPT = """You are a SQLite expert. Given the schema below — the actual current tables and
columns of this database — write ONE read-only SQL query that computes whatever part of the
question CAN be answered using ONLY those tables and columns. Output ONLY the SQL, no
explanation, no markdown fences. Only use SELECT or WITH ... SELECT statements — never INSERT,
UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, or PRAGMA.

Some questions reference an external policy, target, threshold, or comparison that is NOT stored
in this schema. In that case, do not decline — write a query that computes the raw numbers the
question is actually about, using whatever tables/columns exist, so the comparison against that
external context can be made separately by someone who has it. You are only responsible for
producing data that IS available, never for judging it against information you don't have.

Only output exactly NO_QUERY if answering would require a table or column that does not exist
anywhere in the schema below. Before declining, check carefully whether the schema lets you
compute even a partial or approximate answer — if so, write that query instead.

Current schema:
{schema}

Example (illustrative only — uses placeholder names unrelated to your real schema above; apply
the same reasoning to whatever tables/columns are actually listed):
  Question: "Are we meeting our on-time delivery target, and which warehouses need review?"
  Reasoning: nothing above stores a "target" or "review threshold" — but if there's a
  shipments/warehouse table with a status or timestamp, compute the actual on-time rate per
  warehouse instead of declining.
  SQL: SELECT warehouse_id, AVG(delivered_on_time) AS on_time_rate FROM shipments GROUP BY warehouse_id

  Question: "How many open support tickets do we have?"
  Reasoning: nothing in the schema relates to support tickets at all.
  SQL: NO_QUERY

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

    def _known_tables(self) -> set[str]:
        """The real, current table names — always in sync with the actual database file."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def _describe_schema(self) -> str:
        """Build the schema text shown to the model by introspecting the live database,
        so it always reflects whatever tables/columns actually exist right now."""
        conn = sqlite3.connect(self.db_path)
        try:
            tables = sorted(self._known_tables())
            lines = []
            for table in tables:
                lines.append(f"Table: {table}")
                for col in conn.execute(f"PRAGMA table_info({table})").fetchall():
                    col_name, col_type = col[1], col[2]
                    note = COLUMN_NOTES.get((table, col_name))
                    lines.append(f"  {col_name} {col_type}" + (f"  -- {note}" if note else ""))
            return "\n".join(lines)
        finally:
            conn.close()

    def generate_sql(self, question: str) -> str:
        schema = self._describe_schema()
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=SQL_PROMPT.format(schema=schema, question=question),
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=400),
            ),
            label="generating SQL for your question",
        )
        sql = (response.text or "").strip()
        if not sql:
            raise UnsafeSQLError("Model returned no SQL.")
        return re.sub(r"^```sql\s*|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()

    def validate_sql(self, sql: str) -> None:
        stripped = sql.strip().rstrip(";")
        lowered = stripped.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise UnsafeSQLError("Only SELECT (or WITH ... SELECT) statements are allowed.")
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

        if sql.strip().upper() == "NO_QUERY":
            return QuantitativeAnswer(
                answer="I don't have data to answer that — it doesn't match any table in our current database.",
                sql=sql, rows=[], error="no_matching_data_source",
            )

        try:
            self.validate_sql(sql)
        except UnsafeSQLError as e:
            return QuantitativeAnswer(
                answer=f"I couldn't safely run a query for that question ({e}).",
                sql=sql, rows=[], error=str(e),
            )

        known_tables = self._known_tables()
        if known_tables:
            table_pattern = re.compile(
                r"\bfrom\s+(" + "|".join(re.escape(t) for t in known_tables) + r")\b",
                re.IGNORECASE,
            )
            if not table_pattern.search(sql):
                return QuantitativeAnswer(
                    answer="I don't have data to answer that — it doesn't match any table in our current database.",
                    sql=sql, rows=[], error="no_matching_data_source",
                )

        try:
            rows = self.execute(sql)
        except sqlite3.Error as e:
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
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=300),
            ),
            label="summarizing the query results",
        )
        return QuantitativeAnswer(
            answer=(summary.text or "I ran the query but couldn't summarize it.").strip(),
            sql=sql, rows=rows,
        )