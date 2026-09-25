# Enterprise Documentation Assistant — Multi-Agent RAG System

A CLI-based multi-agent system that answers enterprise questions by routing them to a
retrieval-augmented (RAG) agent for policy/documentation questions, a natural-language-to-SQL
agent for data questions, or both — synthesized into one answer — when a question needs both.

## Architecture
<img width="749" height="762" alt="image" src="https://github.com/user-attachments/assets/34d4f8a8-16aa-4567-b151-8d5bc2ce3206" />


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
<img width="1132" height="178" alt="Screenshot 2026-09-24 214954" src="https://github.com/user-attachments/assets/30d42f3f-5433-4282-a3b7-9187d9073d8d" />
<img width="1020" height="197" alt="Screenshot 2026-09-24 214925" src="https://github.com/user-attachments/assets/58a0a66c-707a-47cf-957a-1e700d779e8c" />
<img width="1442" height="446" alt="Screenshot 2026-09-24 214848" src="https://github.com/user-attachments/assets/e8590b56-3c80-4622-959c-d00a07046ea4" />
<img width="1040" height="113" alt="Screenshot 2026-09-24 215157" src="https://github.com/user-attachments/assets/9bbcff4f-524e-46ea-8b16-174411cd82e6" />
<img width="634" height="98" alt="Screenshot 2026-09-24 215122" src="https://github.com/user-attachments/assets/be9b07f4-d1b4-4176-9b63-ef8cf1174280" />

## Testing

```bash
python -m pytest tests/ -v          # unit + integration tests
python data/eval_retrieval.py       # retrieval-quality check against known question/doc pairs
```

