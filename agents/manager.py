"""Manager agent: classifies a query and routes it to the right specialist agent(s)."""
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GENERATION_MODEL
from agents.qualitative import QualitativeAgent
from agents.quantitative import QuantitativeAgent
from agents.gemini_utils import call_with_progress
_client = genai.Client(api_key=GEMINI_API_KEY)

CLASSIFY_PROMPT = """Classify the user's question into exactly one category. Reply with only the category word, nothing else.

- qualitative: about policies, processes, documentation, "how do we...", "what is our...", explanations
- quantitative: about numbers, revenue, counts, comparisons, trends that need querying data
- complex: requires both document context and numeric data
- unsupported: anything unrelated to enterprise documentation or company data

Question: {question}
Category:"""


class QueryType(Enum):
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"
    COMPLEX = "complex"
    UNSUPPORTED = "unsupported"


@dataclass
class ManagerResponse:
    query_type: QueryType
    answer: str
    details: dict


class ManagerAgent:
    def __init__(self):
        self.qualitative_agent = QualitativeAgent()
        self.quantitative_agent = QuantitativeAgent()  # filled in for the silver milestone

    def classify(self, question: str) -> QueryType:
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=CLASSIFY_PROMPT.format(question=question),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                ),
            ),
            label="classifying your question",
        )
        print(f"  [debug] raw text={response.text!r} finish_reason={response.candidates[0].finish_reason if response.candidates else 'no candidates'}")
        raw = (response.text or "").strip().lower()
        if not raw:
            return QueryType.UNSUPPORTED
        for qt in QueryType:
            if qt.value in raw:
                return qt
        return QueryType.UNSUPPORTED

    def handle(self, question: str) -> ManagerResponse:
        query_type = self.classify(question)

        if query_type == QueryType.QUALITATIVE:
            result = self.qualitative_agent.answer(question)
            return ManagerResponse(query_type, result.answer, {"citations": result.citations})

        if query_type == QueryType.QUANTITATIVE:
            if self.quantitative_agent is None:
                return ManagerResponse(
                    query_type,
                    "The quantitative (SQL) agent isn't wired in yet in this build.",
                    {},
                )
            result = self.quantitative_agent.answer(question)
            return ManagerResponse(query_type, result.answer, {"sql": result.sql, "rows": result.rows})

        if query_type == QueryType.COMPLEX:
            qual = self.qualitative_agent.answer(question)
            parts = [f"[Qualitative] {qual.answer}"]
            if self.quantitative_agent is not None:
                quant = self.quantitative_agent.answer(question)
                parts.append(f"[Quantitative] {quant.answer}")
            return ManagerResponse(query_type, "\n\n".join(parts), {"citations": qual.citations})

        return ManagerResponse(
            QueryType.UNSUPPORTED,
            "I can only answer questions about company documentation or company data. "
            "Try something like 'What is our security policy?'",
            {},
        )