# AnswerVault

A retrieval-first interview prep assistant. Answers questions using **your own
memorized interview answers** when a close match exists in your documented
Q&A bank — returned verbatim, no LLM rewriting. Falls back to general
knowledge (via Gemini) only when nothing relevant is found. Every response is
labeled with its source.

Built as a learning project to understand RAG, embeddings, pgvector, tool
calling, and agentic AI from first principles.

## Core behavior — three confidence tiers

```
User Question
     │
     ▼
Embed question (Gemini, task_type="retrieval_query")
     │
     ▼
Cosine similarity search (pgvector, <=> operator)
     │
     ▼
     ├── HIGH similarity (> 0.75)     → return stored answer verbatim, no LLM
     ├── MEDIUM similarity (0.60–0.75) → ask "did you mean...?" confirmation
     └── LOW similarity (< 0.60)      → Gemini answers from general knowledge
     │
     ▼
Every response labeled: "From your documents" / "From general knowledge"
```

**Thresholds are evidence-based, not guessed.** Originally planned as
0.90/0.75, but real testing against the actual embedded data (Day 3) showed
even exact question matches only reach ~0.75–0.81 similarity — Gemini's
embedding space doesn't spread scores as wide as intuition suggests.
Thresholds were recalibrated using real query results rather than shipped
as originally assumed. See `PROJECT_LOG.md` Day 3 for the full test data.

## Stack

- FastAPI — HTTP layer
- PostgreSQL + pgvector (Docker) — storage + vector similarity search
- Google Gemini API (free tier) — `gemini-3.1-flash-lite` for chat,
  `gemini-embedding-001` for embeddings (768 dims)
- SQLAlchemy — ORM
- `google-genai` SDK (current; the old `google.generativeai` is deprecated)

## Project structure (current, actual state on disk)

```
answervault/
  app/
    __init__.py
    config.py          # loads env vars
    db.py               # engine, session, Base, AND all models (Message, QAEntry)
    gemini_client.py     # all Gemini calls isolated here
    main.py               # FastAPI app + /chat and any other endpoints, flat
  data/
    qa_seed.json          # 49 structured Q&A entries, already ingested
  scripts/
    ingest_qa.py            # embeds + stores qa_seed.json into pgvector
    search_qa.py             # similarity search against stored entries (Day 3)
  .env
  docker-compose.yml
```

Note: this is intentionally a flat structure (no routers/services/models
split). A production-style refactor was explored and reverted — see
`PROJECT_LOG.md` for what was learned from that detour and why it was undone.

## Setup

1. Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no card required.
2. Fill in `.env` with `GEMINI_API_KEY` and Postgres credentials.
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
6. Ingest Q&A data (already done once, safe to re-run if the table is empty):
   ```bash
   python -m scripts.ingest_qa
   ```

## Current status

**Working:** conversation-memory chat (`/chat`, backed by Postgres),
embeddings pipeline, 49 Q&A entries stored in pgvector at 768 dimensions,
cosine similarity search (`scripts/search_qa.py`) verified against real
queries with evidence-based confidence thresholds.

**Not yet built:** confidence-tier branching logic, source labeling, long-term
memory summarization, tool calling, agentic loop.

**Explored, then intentionally reverted:** a full document-ingestion pipeline
(PDF upload, PyMuPDF extraction, text cleaning, router/service/model
architecture). Real, working knowledge gained from this is preserved in
`PROJECT_LOG.md` under "Parked Knowledge" — it will be revisited as a
separate, deliberate learning track, not as part of AnswerVault's main path.

## Data model

```python
class QAEntry(Base):
    __tablename__ = "qa_entries"
    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    created_at = Column(DateTime, default=...)
```

## Roadmap

| Day | Goal | Status |
|---|---|---|
| 1 | FastAPI + Gemini chat, conversation memory (Postgres) | ✅ Done |
| 2 | Embeddings on Q&A docs, stored in pgvector | ✅ Done |
| 3 | Similarity search only — no LLM involved, prove retrieval works | ✅ Done |
| 4 | Retrieve top-K + three-tier confidence branching | ⏳ Next |
| 5 | Full RAG — low-confidence path calls Gemini, source labeling | Planned |
| 6 | Long-term memory (summarization for long sessions) | Planned |
| 7 | Tool calling (log wrong answers, track weak topics) | Planned |
| 8 | Agentic loop ("quiz me on SQL" → multi-step reasoning) | Planned |

Similarity search (3) and RAG (5) are deliberately separate days, so
retrieval correctness can be verified before any LLM generation is layered
on top.
