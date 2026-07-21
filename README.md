# AnswerVault

A retrieval-first interview prep assistant. Answers questions using **your own
memorized interview answers** when a close match exists in your documented
Q&A bank — returned verbatim, no LLM rewriting. Falls back to general
knowledge (via Gemini) only when nothing relevant is found. Every response is
labeled with its source.

Built as a learning project to understand RAG, embeddings, pgvector, tool
calling, and agentic AI from first principles — every line written and
understood, not copy-pasted. A second project, OpsMind AI, will be finished
afterward applying what's learned here.

## Why this exists

Most "chat with your docs" projects blend retrieval with generation and let
the LLM paraphrase everything, which risks drifting from answers you've
actually memorized for interviews. AnswerVault is deliberately architected so
that a **high-confidence match returns your stored answer directly from the
database, with no LLM call involved at all** — the only way to guarantee
verbatim output.

## Core behavior — three confidence tiers

```
User Question
     │
     ▼
Embed question (Gemini)
     │
     ▼
Cosine similarity search (pgvector)
     │
     ▼
     ├── HIGH similarity (> 0.90)   → return stored answer verbatim, no LLM
     ├── MEDIUM similarity (0.75–0.90) → ask "did you mean...?" confirmation
     └── LOW similarity (< 0.75)    → Gemini answers from general knowledge
     │
     ▼
Every response labeled: "From your documents" / "From general knowledge"
```

## Stack

- FastAPI — HTTP layer
- PostgreSQL + pgvector (Docker) — storage + vector similarity search
- Google Gemini API (free tier) — `gemini-3.1-flash-lite` for chat,
  `gemini-embedding-001` for embeddings (768 dims)
- SQLAlchemy — ORM
- `google-genai` SDK (current; the old `google.generativeai` is deprecated)

## Project structure

```
answervault/
  app/
    __init__.py
    config.py          # loads env vars
    db.py               # SQLAlchemy models + engine
    gemini_client.py     # all Gemini calls isolated here
    main.py              # FastAPI app, thin HTTP layer
  data/
    qa_seed.json          # structured Q&A entries, ready to ingest
    interview_qa.txt       # (planned) raw notes in TOPIC/Q/A text format
  scripts/
    ingest_qa.py            # embeds + stores qa_seed.json into pgvector
    parse_qa.py               # (planned) parses interview_qa.txt → qa_seed.json
  .env
  docker-compose.yml
```

## Setup

1. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no card required.
2. Copy `.env.example` to `.env`, fill in `GEMINI_API_KEY` and Postgres credentials.
3. Start Postgres + pgvector:
   ```bash
   docker compose up -d
   docker exec -it answervault_db psql -U postgres -d answervault -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```
4. Install dependencies:
   ```bash
   pip install fastapi uvicorn google-genai psycopg2-binary python-dotenv sqlalchemy pydantic pgvector
   ```
5. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
6. Ingest your Q&A data:
   ```bash
   python -m scripts.ingest_qa
   ```

## Current status

**Working:** conversation-memory chat (`/chat`, backed by Postgres),
embeddings pipeline, 49 Q&A entries stored in pgvector at 768 dimensions.

**Not yet built:** similarity search, confidence-tier branching, source
labeling, long-term memory summarization, tool calling, agentic loop.

See `PROJECT_LOG.md` for the detailed day-by-day build history, decisions,
and issues encountered.

## Data model

```python
class QAEntry(Base):
    __tablename__ = "qa_entries"
    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
```

Deliberately minimal — no `difficulty`, `company`, `tags`, etc. yet. Metadata
gets added only once it has an actual job to do (e.g. filtering retrieval by
topic), not speculatively.

```python
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)      # "user" or "model"
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=...)
```

## Roadmap

| Day | Goal | Status |
|---|---|---|
| 1 | FastAPI + Gemini chat, conversation memory (Postgres) | ✅ Done |
| 2 | Embeddings on Q&A docs, stored in pgvector | ✅ Done |
| 3 | Similarity search only — no LLM involved, prove retrieval works | ⏳ Next |
| 4 | Retrieve top-K + three-tier confidence branching | Planned |
| 5 | Full RAG — low-confidence path calls Gemini, source labeling | Planned |
| 6 | Long-term memory (summarization for long sessions) | Planned |
| 7 | Tool calling (log wrong answers, track weak topics) | Planned |
| 8 | Agentic loop ("quiz me on SQL" → multi-step reasoning) | Planned |

Note: similarity search (3) and RAG (5) were deliberately split into
separate days rather than combined, so retrieval correctness can be verified
in isolation before any LLM generation is layered on top.
