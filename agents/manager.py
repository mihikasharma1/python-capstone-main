"""Manager agent: classifies a query, decomposes complex ones, routes to the right specialist
agent(s), and synthesizes a single coherent answer when more than one agent contributed."""
import json
import time
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

DECOMPOSE_PROMPT = """The user asked a question that may require both document/policy context
and numeric data analysis to answer fully. Split it into two separate, self-contained
sub-questions:

1. qualitative_question: the part that should be answered from company policy/documentation
   (targets, thresholds, processes, rules). Phrase it as a clean standalone question. If the
   original question has no qualitative/document component, repeat the original question as-is.
2. quantitative_question: the part that should be answered by querying the database (counts,
   rates, comparisons, aggregates). Phrase it as a clean standalone question, including any
   numeric target/threshold mentioned in the original question if it's needed for a comparison.
   If the original question has no quantitative/data component, repeat the original question
   as-is.

IMPORTANT: keep every specific term, noun, and metric from the original question in BOTH
sub-questions, even if that makes the phrasing slightly redundant. Do not replace a specific word
(e.g. "churn") with a more general paraphrase (e.g. "customer attrition", "escalation criteria")
— the exact wording matters for finding the right documents. Only remove parts of the question
that are irrelevant to that sub-question; do not reword the parts you keep.

Original question: {question}"""

SYNTHESIS_PROMPT = """A user asked a question that required both document context and data
analysis. Below are two independent answers. Combine them into ONE coherent response that
directly connects the facts — for example, explicitly compare a computed number against a
stated policy target or threshold rather than just repeating both answers separately. Preserve
any citations from the qualitative answer, in their original [filename#chunk_index] format. If
the two answers genuinely don't relate to each other, say so honestly rather than forcing a
connection.

Original question: {question}

Qualitative answer: {qualitative_answer}

Quantitative answer: {quantitative_answer}

Unified answer:"""


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
        self.quantitative_agent = QuantitativeAgent()

    def classify(self, question: str) -> QueryType:
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=CLASSIFY_PROMPT.format(question=question),
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=500),
            ),
            label="classifying your question",
        )
        raw = (response.text or "").strip().lower()
        if not raw:
            return QueryType.UNSUPPORTED
        for qt in QueryType:
            if qt.value in raw:
                return qt
        return QueryType.UNSUPPORTED

    def decompose(self, question: str) -> tuple[str, str]:
        """Split a compound question into a clean sub-question per agent, so retrieval and SQL
        generation each work against a focused question instead of the full compound text."""
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=DECOMPOSE_PROMPT.format(question=question),
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=400,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "qualitative_question": {"type": "string"},
                            "quantitative_question": {"type": "string"},
                        },
                        "required": ["qualitative_question", "quantitative_question"],
                    },
                ),
            ),
            label="splitting your question for each agent",
        )
        try:
            parsed = json.loads(response.text or "{}")
        except json.JSONDecodeError:
            parsed = {}
        qual_q = parsed.get("qualitative_question") or question
        quant_q = parsed.get("quantitative_question") or question
        return qual_q, quant_q

    def synthesize(self, question: str, qualitative_answer: str, quantitative_answer: str) -> str:
        response = call_with_progress(
            lambda: _client.models.generate_content(
                model=GENERATION_MODEL,
                contents=SYNTHESIS_PROMPT.format(
                    question=question,
                    qualitative_answer=qualitative_answer,
                    quantitative_answer=quantitative_answer,
                ),
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=400),
            ),
            label="combining both agents' answers",
        )
        return (response.text or f"{qualitative_answer}\n\n{quantitative_answer}").strip()

    def handle(self, question: str) -> ManagerResponse:
        start = time.monotonic()
        query_type = self.classify(question)
        agents_used, sources, sql, error = [], None, None, None

        try:
            if query_type == QueryType.QUALITATIVE:
                result = self.qualitative_agent.answer(question)
                agents_used = ["qualitative"]
                sources = [f"{c.source}#{c.chunk_index}" for c in result.citations]
                response = ManagerResponse(query_type, result.answer, {"citations": result.citations})

            elif query_type == QueryType.QUANTITATIVE:
                if self.quantitative_agent is None:
                    response = ManagerResponse(
                        query_type, "The quantitative (SQL) agent isn't wired in yet in this build.", {}
                    )
                else:
                    result = self.quantitative_agent.answer(question)
                    agents_used = ["quantitative"]
                    sql = result.sql
                    response = ManagerResponse(query_type, result.answer, {"sql": result.sql, "rows": result.rows})

            elif query_type == QueryType.COMPLEX:
                qual_question, quant_question = self.decompose(question)
                qual = self.qualitative_agent.answer(qual_question)
                quant = self.quantitative_agent.answer(quant_question)
                unified = self.synthesize(question, qual.answer, quant.answer)
                answer = (
                    f"{unified}\n\n"
                    f"— Qualitative agent handled: {qual_question}\n"
                    f"— Quantitative agent handled: {quant_question}"
                )
                agents_used = ["qualitative", "quantitative"]
                sources = [f"{c.source}#{c.chunk_index}" for c in qual.citations]
                sql = quant.sql
                response = ManagerResponse(
                    query_type, answer, {"citations": qual.citations, "sql": quant.sql, "rows": quant.rows}
                )

            else:
                response = ManagerResponse(
                    QueryType.UNSUPPORTED,
                    "I can only answer questions about company documentation or company data. "
                    "Try something like 'What is our security policy?'",
                    {},
                )
            return response

        except Exception as e:
            error = str(e)
            raise
        # finally:
        #     log_query(
        #         question=question, query_type=query_type.value, agents_used=agents_used,
        #         sources=sources, sql=sql, duration_s=time.monotonic() - start, error=error,
        #     )