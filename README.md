# Capstone Project: Multi-Agent RAG System for Enterprise Documentation

In this capstone, you'll design and build a **multi-agent system in Python** that can answer enterprise documentation queries end-to-end. Your solution will combine different types of AI agents into a single workflow that routes user questions to the right tools.

You'll implement and integrate three agents:

-   **Manager Agent** -- Orchestrates queries, decides which agent to call, and assembles the final response.

-   **Qualitative RAG Agent** -- Uses a vector database and an LLM to retrieve and cite relevant document chunks.

-   **Quantitative NL-to-SQL Agent** -- Translates natural language into SQL, queries a database, and returns tabular or numeric answers.

By the end, your system will:

-   Accept queries through a **CLI interface**.

-   Handle both qualitative (text/document) and quantitative (data/SQL) queries.

-   Return responses with **citations, results, and clean formatting**.

-   Show foundational knowledge of **GenAI concepts** like hallucinations, prompt design, and agent-tool patterns.

**Technical expectations:**

-   Use **Python 3.8+** with pinned dependencies.

-   Follow **Pythonic coding practices** (modularity, naming, clarity).

-   Provide a **README with setup, architecture, and diagrams**.

-   Include **basic automated tests** (unit and/or integration).

-   Ensure the project runs cleanly from setup to query handling.

Stretch goals include full three-agent integration, robust testing, and a production-ready architecture.
