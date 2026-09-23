Unit 2 Capstone: Multi-Agent RAG System for Enterprise Documentation
===========================

* * * * *

### Project Overview

You will build a **Python CLI-based Retrieval-Augmented Generation (RAG) system** that uses specialized AI agents to answer both qualitative and quantitative questions about enterprise documentation. The system features a manager agent that classifies queries and routes them to a semantic search agent (for qualitative questions) or a SQL agent (for quantitative data).

Step 1: Review System Architecture
----------------------------------

-   **Manager Agent (CLI Interface):**

    -   Accepts user queries via the command line

    -   Classifies questions as qualitative or quantitative

    -   Routes queries to the correct agent(s) based appropiate querying

    -   Coordinates responses, including complex, multi-part answers

-   **Qualitative Data Agent:**

    -   Connects to a vector database (Chroma or Pinecone)

    -   Performs semantic search using embeddings (OpenAI Ada or Sentence Transformers)

    -   Retrieves and cites relevant documentation

    -   Uses an LLM (e.g., GPT, Claude, or local model) to generate clear, contextual responses

-   **Quantitative Data Agent:**

    -   Connects to a SQL database (e.g., SQLite)

    -   Converts natural language questions to SQL queries

    -   Executes queries, performs calculations, and generates insights/visualizations

Step 2: Build & Integrate Core Components
-----------------------------------------

-   Set up your Python project and required environments

-   Build each agent as described above

-   Integrate the agents so the Manager agent coordinates all workflows

-   Ensure your CLI supports all interaction flows

Step 3: Supported Query Types
-----------------------------

**Qualitative Queries (Semantic/Document):**

-   "What is our company's security policy?"

-   "Explain the code review process"

-   "How do we handle customer complaints?"

**Quantitative Queries (SQL/Numerical):**

-   "Show me monthly revenue trends"

-   "What's our customer churn rate?"

-   "Compare Q4 performance across regions"

**Complex Multi-Agent Queries:**

-   "How does our employee satisfaction compare to industry standards and what policies might impact this?"

-   "Analyze our sales performance and recommend policy changes based on our customer success strategies"

Step 4: Test & Document
-----------------------

-   Test each agent individually and together in multi-agent workflows

-   Document your system architecture, agent interactions, and sample queries

Must-Have Checklist
-------------------
>🥉 Bronze - complete all must-haves

-   Python 3.8+ codebase

-   Command-line interface for user interaction

-   2 out of 3 agents:
    -   Manager agent for query routing and response coordination

    -   Qualitative agent integrated with a vector database and LLM for semantic/document search
 
    -   Quantitative agent with SQL database integration and natural language to SQL translation

-   Proper source attribution for qualitative answers

-   Support for all query types (qualitative, quantitative, complex)

-   Unit tests for each agent

Stretch Goals
----------------------------
>🥈 Silver - complete all 3 agents

-   3 out of 3 agents:
    -   Manager agent for query routing and response coordination

    -   Qualitative agent integrated with a vector database and LLM for semantic/document search
 
    -   Quantitative agent with SQL database integration and natural language to SQL translation
 
> 🥇 Gold - complete all 3 agents and implement testing

-   Integration tests for multi-agent workflows

-   Database and LLM connection tests

-   CLI interface tests

- Retrieval quality

  - Return source citations for every qualitative answer.
  - Include retrieved document IDs and similarity scores.
  - Add a minimum relevance threshold.
  - Test questions whose answers are not in the knowledge base.
  - Evaluate retrieval with a small set of expected documents.
  - 
- API 

    - FastAPI/LangServe routes for each agent and the manager.
    - Pydantic request and response schemas.
    - /health endpoint.
    - OpenAPI documentation.
    - Environment-based configuration for API keys, database paths, and
        model names.

- Structured logs for:
- 
      - Incoming query
      - Selected agent
      - Retrieved sources
      - Generated SQL
      - Execution time
      - Errors
   
- Manager Agent

   - Support queries requiring both qualitative and quantitative
    agents.
  - Merge responses from both agents into one clearly labeled answer.
  - Explain which agent handled each part of the query.
  - Handle ambiguous queries by asking a clarification question.




Resources & Support
-------------------

-   Class labs, code samples, and documentation for Python, Chroma/Pinecone, SQLite, and LLM APIs

-   Instructors available for help if blocked for more than 30 minutes

Tips for Success
----------------

-   **Work in small steps:** Build, test, and document each agent before integrating

-   **Emphasize clear routing:** Make sure the manager agent reliably classifies and routes all queries

-   **Test for robustness:** Try edge cases and both simple and complex queries

-   **Comment and document:** Help reviewers understand your design

-   **Sqlite3 tip:**
    - If you are using sqlite as your db, we need to put these lines the top of the file that's using sqlite (or chroma), which tricks the system into thinking pysqlite3 (which doesn't require gdb) is sqlite3 (Reference: source: https://discuss.streamlit.io/t/issues-with-chroma-and-sqlite/47950/4):
 
```
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

Deliverables
------------

-   Python codebase meeting all requirements above

-   Working CLI that handles all supported query types

-   README with architecture diagram, setup, and usage instructions

-   Automated test suite covering agents and workflows
