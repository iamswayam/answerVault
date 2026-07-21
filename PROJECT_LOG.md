# AnswerVault — Project Log

Detailed running history of decisions, issues, and implementation details.
Kept for personal reference and as real interview material ("tell me about a
challenge you solved" — several entries below answer that directly).

---

## Background / how this project was scoped

Originally planned a resume project (OpsMind AI style: RAG ops-copilot) to be
built in ~1 week at 6-8 hrs/day. After reflection, decided to build a smaller,
different-shaped project first — AnswerVault — purely to build deep,
first-principles understanding of RAG/embeddings/tool calling/agentic AI
before returning to finish OpsMind with real understanding instead of
copy-pasted structure.

**Key design decision, locked in early:** unlike typical RAG (retrieve →
always generate), AnswerVault uses a three-tier confidence branch:

- High similarity match → return the user's own stored answer **verbatim**,
  straight from the database, no LLM call at all
- Medium similarity → ask for confirmation ("did you mean...?") before
  returning a stored answer
- Low similarity → let Gemini answer freely from general knowledge, no
  restriction

This came from a real, specific requirement: the user has memorized answers
for real recurring interview questions and cannot risk the LLM subtly
rewording/"improving" them. LLMs can't be trusted to reproduce text verbatim
even under explicit instruction — so the only reliable fix is architectural:
skip the LLM entirely on a high-confidence match, don't just prompt it to
behave.

This is a meaningfully different retrieval philosophy from OpsMind (which
uses a single confidence gate: refuse to answer below a threshold).
AnswerVault instead branches to a different *source* of the answer depending
on confidence, never refusing outright.

---

## Day 1 — FastAPI + Gemini chat with conversation memory

**Goal:** a working `/chat` endpoint where conversation history persists in
Postgres and Gemini genuinely remembers prior turns within a conversation.

**Built:**
- `app/config.py` — loads `GEMINI_API_KEY`, `DATABASE_URL` from `.env`
- `app/db.py` — SQLAlchemy `Message` model (conversation_id, role, content,
  created_at) + `init_db()`
- `app/gemini_client.py` — `generate_reply(history)`, resends the full
  conversation on every call (LLMs are stateless between calls — this is the
  core mechanic of "memory")
- `app/main.py` — `/chat` endpoint: loads prior messages for the given
  `conversation_id` from Postgres, appends the new message, calls Gemini,
  saves both turns back to the DB

**Decision — Postgres from day 1, not an in-memory dict:**
Originally considered starting with an in-memory Python dict for simplicity,
then swapping to Postgres later. Skipped that step since Docker + Postgres
was already available and the user already knows Postgres well from Django.
Went straight to the DB-backed version.

**Decision — why history is rebuilt from the DB on every request:**
An in-memory Python variable only lives in that process's RAM — lost on
restart, and inconsistent across multiple server processes/workers. Postgres
is the single shared source of truth. (Same reason Django doesn't keep
session data in a bare Python variable.)

**Understanding check passed:** confirmed a brand-new `conversation_id`
correctly has zero memory (empty history from the DB query) — proved
conversation memory is scoped per-conversation, not global.

**Issue #1 — `gemini-2.5-flash` returned 404:**
```
google.api_core.exceptions.NotFound: 404 This model models/gemini-2.5-flash
is no longer available to new users.
```
Fix: switched to `gemini-3.1-flash-lite` (confirmed via web search to be a
real, current Gemini 3-series model — optimized for low-latency,
high-volume, cost-sensitive tasks). Noted for later: if answer quality feels
thin once we reach the free-generation tier (Day 5), consider comparing
against a non-lite Gemini 3 model.

**Result:** confirmed working — asked "My name is Swayam" then in a follow-up
message asked "what's my name," got correct recall. Tested a second, fresh
`conversation_id` and confirmed it correctly had no memory of the name.

---

## Day 2 — Embeddings + pgvector storage

**Goal:** convert interview Q&A content into vectors and store them in
Postgres via pgvector — storage only, no search yet.

**Infra setup:**
- Docker Compose with `pgvector/pgvector:pg17` image (official image; more
  current than initially suggested `ankane/pgvector`)
- `.env` uses `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` vars,
  referenced in `docker-compose.yml` via `${VAR}` substitution
- Caught and fixed a mismatch where `DATABASE_URL` had a different password
  than the actual container env vars — would have caused a silent-looking
  auth failure at connection time
- Enabled the extension: `CREATE EXTENSION IF NOT EXISTS vector;`

**Concept — what an embedding is:**
A list of numbers (a vector, 768 dimensions here) representing the *meaning*
of a piece of text. Semantically similar text produces vectors that are close
together in that 768-dimensional space; unrelated text produces vectors far
apart. No individual dimension is human-interpretable — only relative
distance/angle between vectors matters.

**Source data — Backend_Interview_QA.pdf:**
User provided an existing, well-structured interview prep PDF covering
Python, Django, DRF, FastAPI, SQL, and AWS (~49 Q&A pairs total).

**Decision — manual JSON conversion vs. building a parser first:**
Initially converted the PDF content to structured JSON by hand (all 49
entries, across 6 topics) to avoid learning two new things (parsing +
embeddings) in the same session. ChatGPT (consulted in parallel) pushed back:
argued that since the source format is already consistent (`Q:` / `A:`
blocks), a small parser is realistic to build immediately and is a genuine,
transferable AI engineering skill (every RAG system needs an ingestion step).

**Resolution:** agreed with ChatGPT's core point, with one adjustment — scope
the parser to *only* the one text format actually in use (`TOPIC:` / `Q:` /
`A:` plain text blocks), not a general multi-format (PDF/DOCX/Markdown/CSV)
ingestion system, which would be real scope creep for this stage. Parser
(`scripts/parse_qa.py`) written using `re.split` on `TOPIC:` markers, with
`re.DOTALL` on the answer group only (answers may span multiple lines),
skipping malformed blocks defensively rather than crashing. The 49 entries
already hand-converted remain valid; the parser is for future additions to
the notes.

**Schema decision:**
ChatGPT also flagged that the plan to add rich metadata (difficulty,
company, tags, etc.) was premature — nothing in the pipeline would use it
yet. Agreed: kept `QAEntry` to `topic`, `question`, `answer`, `embedding`
only. Metadata will be added later, only once a real feature (e.g. filtering
retrieval by topic or difficulty) needs it.

**Code — `app/db.py` addition:**
```python
from pgvector.sqlalchemy import Vector

class QAEntry(Base):
    __tablename__ = "qa_entries"
    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
```

**Code — `scripts/ingest_qa.py`:**
Loads `data/qa_seed.json`, embeds `question + answer` together (not question
alone — gives the vector more context, so a differently-phrased search query
still matches), stores each as a `QAEntry` row. Run via
`python -m scripts.ingest_qa` (module form, so `app` package imports resolve
correctly).

**Issue #2 — embedding model deprecated:**
Originally planned to use `text-embedding-004`. Web search confirmed it was
**shut down January 14, 2026**. Switched to `gemini-embedding-001`.

**Issue #3 — embedding dimension mismatch:**
`gemini-embedding-001` defaults to **3072** dimensions, not 768. Had to
explicitly pass `output_dimensionality=768` in the embed call to match the
`Vector(768)` column (768 chosen as the standard "good quality, reasonable
storage cost" tier per Google's own guidance, and matches what OpsMind AI's
build already established as sufficient).

**Result:** ingestion ran successfully — "Ingested 49 entries." confirmed in
Postgres.

**Issue #4 — SDK fully deprecated mid-build:**
Running the (working) ingestion script surfaced:
```
FutureWarning: All support for the `google.generativeai` package has ended.
Please switch to the `google.genai` package as soon as possible.
```
Investigated via web search — confirmed `google.generativeai` is fully
legacy (limited/no further updates). Migrated both `generate_reply()` and
`embed_text()` in `gemini_client.py` to the new `google.genai` SDK:

- Old: implicit global config via `genai.configure()`, stateful
  `GenerativeModel` + `start_chat()`/`send_message()` objects
- New: explicit `genai.Client(api_key=...)` object, fully stateless
  `client.models.generate_content(model=..., contents=[...])` — the entire
  conversation is passed as `contents` every call, no hidden chat-session
  state. This actually makes the "no memory between calls" fact from Day 1
  more explicit in the code, not less.
- Embedding response shape changed: `result["embedding"]` (dict-style, old
  SDK) → `result.embeddings[0].values` (typed object, list of embeddings,
  new SDK)

**Result after migration:** re-ran ingestion (clean, no warning), then
re-tested `/chat` — conversation memory still works correctly under the new
SDK. Verified with a genuine reasoning test (asked "which month does Swayam
celebrate his birthday" after stating a DOB earlier — correct answer,
confirms the model is using stored context, not just echoing).

---

## Day 3 — Similarity search (planned, not yet built)

**Scope decision (via ChatGPT review, adopted):** originally planned to
combine similarity search and full RAG generation into one day. Split into
two separate days instead:

- Day 3: prove vector search alone can retrieve the correct answer — print
  top match, its similarity score, and the raw stored answer. **No Gemini
  call involved at this stage.**
- Day 5: only then reintroduce Gemini for the low-confidence / general
  knowledge path.

Rationale: testing retrieval and generation together hides two independent
failure modes behind one pass/fail signal. Verifying the deterministic
retrieval layer first, before adding the probabilistic LLM layer on top,
makes debugging and understanding each piece far cleaner.

*(To be filled in once Day 3 is actually built.)*
