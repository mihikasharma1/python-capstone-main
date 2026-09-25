# Enterprise Documentation Assistant — Multi-Agent RAG System

A CLI-based multi-agent system that answers enterprise questions by routing them to a
retrieval-augmented (RAG) agent for policy/documentation questions, a natural-language-to-SQL
agent for data questions, or both — synthesized into one answer — when a question needs both.

## Architecture



Two rules hold everywhere: **the model never executes anything unchecked** (SQL passes a
deterministic validator before running; RAG answers only use context that clears a similarity
threshold), and **every Gemini call is guarded against empty responses**, since some models
spend their whole token budget on internal reasoning and return no visible text.

## Components

- **Manager Agent** (`agents/manager.py`) — classifies queries, decomposes complex ones,
  routes to the right specialist agent(s), synthesizes multi-agent answers, logs each query.
- **Qualitative RAG Agent** (`agents/qualitative.py`) — Chroma + local MiniLM embeddings,
  paragraph-based chunking, cited answers, honest "not found" below the relevance threshold.
- **Quantitative NL-to-SQL Agent** (`agents/quantitative.py`) — schema-agnostic (introspects
  the live DB, no hardcoded table names), validated read-only SQL, honest decline when the
  schema has no data to answer with.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env            # add your GEMINI_API_KEY
python data/seed_database.py
python data/build_vector_store.py
```

## Usage 
```bash
python main.py
```
Type a question, exit/quit to leave

## Sample runs

## Testing

```bash
python -m pytest tests/ -v          # unit + integration tests
python data/eval_retrieval.py       # retrieval-quality check against known question/doc pairs
```

