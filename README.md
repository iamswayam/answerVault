# AnswerVault

A retrieval-first interview prep assistant. Answers questions using **your own
memorized interview answers** when a close match exists in your documented
Q&A bank — returned verbatim, no LLM rewriting. Falls back to general
knowledge (via Gemini) only when nothing relevant is found. Every response is
labeled with its source.

Built as a learning project to understand RAG, embeddings, pgvector, tool
calling, and agentic AI from first principles.

**Status: actively being hardened, not yet production-quality.** The core
architecture (tiers, memory, query rewriting) works, but RAG reliability
and prompt quality still need real refinement — see "Known Issues" below
and `PROJECT_LOG.md` Day 5 for the full story.

## Core behavior — three confidence tiers

```
User Question
     │
     ▼
Rewrite ambiguous follow-ups using conversation history (if any)
     │
     ▼
Embed question (Gemini, task_type="retrieval_query")
     │
     ▼
Cosine similarity search (pgvector, <=> operator)
     │
     ▼
     ├── HIGH (> 0.75)    → return stored answer verbatim, no LLM
     │                        + "get general answer instead" escape button
     ├── MEDIUM (0.60–0.75) → show suggestion card, user confirms Yes/No
     │                          before the stored answer is ever revealed
     └── LOW (< 0.60)       → Gemini answers from general knowledge,
                                 short/plain interview-flashcard style
     │
     ▼
Every response labeled with source + confidence + similarity score
```

**Thresholds are evidence-based, not guessed.** Originally planned as
0.90/0.75, but real testing against the actual embedded data (Day 3) showed
even exact question matches only reach ~0.75–0.81 similarity. Thresholds
were recalibrated using real query results. See `PROJECT_LOG.md` Day 3.

**The verbatim guarantee is architectural, not just a prompt instruction.**
At every tier, a stored answer either passes straight through untouched or
never gets used at all — it is never passed as input to any Gemini call,
so it cannot be paraphrased or reworded, by design, not by request.

## Known issues (real, unresolved, tracked deliberately)

- **No cross-turn "mode" memory.** If a user escapes to general knowledge
  for a topic (via the HIGH-tier escape button), the very next message can
  re-trigger vector search from scratch and land back on the same stored
  match, forcing the user to escape again. `/ask` currently has no concept
  of a prior turn's tier decision carrying forward. Deliberately not
  patched yet — planned to be addressed properly once Day 6–8 (memory
  summarization, tool calling, agentic loop) are in place, so the fix can
  account for the full system rather than being a one-off patch.
- **Query rewriting doesn't help meta-comments.** Ambiguous *questions*
  ("I asked difference?") get correctly rewritten into standalone
  questions using history. Meta-comments about the conversation itself
  ("that answer is wrong") have no real question to recover and will still
  search back into the same topic. Solved for now via a UI escape button
  rather than fragile natural-language intent detection — see
  `PROJECT_LOG.md` Day 5.
- **Prompt/RAG quality needs more adversarial testing overall.** The
  architecture is proven to function; it has not yet been proven robust
  under varied real-world phrasing. Expect more issues to surface with
  continued use.

## Stack

- FastAPI — HTTP layer, serves both the API and the frontend (`StaticFiles`)
- PostgreSQL + pgvector (Docker) — storage + vector similarity search
- Google Gemini API (free tier) — `gemini-3.1-flash-lite` for chat,
  `gemini-embedding-001` for embeddings (768 dims)
- SQLAlchemy — ORM
- `google-genai` SDK (current; the old `google.generativeai` is deprecated)
- Vanilla HTML/CSS/JS frontend — no build step, no framework, same-origin

## Project structure (current, actual state on disk)

```
answervault/
  app/
    __init__.py
    config.py             # loads env vars
    db.py                  # engine, session, Base, AND all models (Message, QAEntry)
    gemini_client.py        # all Gemini calls: chat, embeddings, query rewriting,
    main.py                  # FastAPI app, includes routers, mounts static/ last
    routers/
      qa.py                    # /ask, /ask/general, /ask/confirm
    services/
      retrieval.py               # answer_query() -- the three-tier branch logic
  static/
    index.html               # dark-themed chat UI
  data/
    qa_seed.json                # 49 structured Q&A entries, already ingested
  scripts/
    ingest_qa.py                  # embeds + stores qa_seed.json into pgvector
    search_qa.py                    # raw similarity search script (Day 3)
  .env
  docker-compose.yml
```

Note: `app/db.py` stays flat (no separate `models/` folder) and `/chat`
uses manual `SessionLocal()` rather than `Depends(get_db)` — this reflects
a deliberate revert of an earlier production-style refactor. See
`PROJECT_LOG.md` "Detour" section for what was learned from that
architecture before it was undone. `routers/` and `services/` here were
added fresh after the revert, specifically because Day 4/5 needed them —
not a re-adoption of the reverted structure.

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
   pip install fastapi uvicorn google-genai psycopg2-binary python-dotenv sqlalchemy pydantic pgvector python-multipart
   ```
5. Run the API:
   ```bash
   uvicorn app.main:app --reload
   ```
6. Open `http://127.0.0.1:8000` for the chat UI, or `/docs` for the raw API.
7. Ingest Q&A data (already done once, safe to re-run if the table is empty):
   ```bash
   python -m scripts.ingest_qa
   ```

## Current status

**Working:**
- `/chat` — general conversation memory (Postgres-backed)
- `/ask`, `/ask/general`, `/ask/confirm` — full three-tier RAG with
  conversation memory, query rewriting for ambiguous follow-ups, MEDIUM
  suggestion/confirm flow, HIGH-tier escape hatch
- Dark-themed chat frontend at `/`
- 49 Q&A entries embedded and searchable in pgvector

**Not yet built:** long-term memory summarization, tool calling, agentic
loop, and a real fix for the cross-turn "mode memory" gap noted above.

**Explored, then intentionally reverted:** a full document-ingestion
pipeline (PDF upload, PyMuPDF extraction, text cleaning, router/service/
model architecture). Preserved in `PROJECT_LOG.md` under "Parked
Knowledge" as a separate future study track.

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

Both `/chat` and `/ask` share the same `messages` table, keyed by
`conversation_id` — one conversation can freely mix general chat and RAG
questions with continuous memory across both.

## Roadmap

| Day | Goal | Status |
|---|---|---|
| 1 | FastAPI + Gemini chat, conversation memory (Postgres) | ✅ Done |
| 2 | Embeddings on Q&A docs, stored in pgvector | ✅ Done |
| 3 | Similarity search only — no LLM involved, prove retrieval works | ✅ Done |
| 4 | Three-tier confidence branching, tested against 11 real queries | ✅ Done |
| 5 | Full RAG + frontend + conversation memory + query rewriting | 🔶 Functional, hardening in progress |
| 6 | Long-term memory (summarization) — likely also fixes the cross-turn mode gap | Planned |
| 7 | Tool calling (log wrong answers, track weak topics) | Planned |
| 8 | Agentic loop ("quiz me on SQL" → multi-step reasoning) | Planned |

Similarity search (3) and RAG (5) were deliberately kept as separate days
so retrieval correctness could be verified before any LLM generation was
layered on top — this paid off directly during Day 5, since several bugs
were clearly diagnosable as "search problem" vs. "generation problem"
rather than tangled together.