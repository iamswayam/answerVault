# AnswerVault — Project Log

Detailed running history of decisions, issues, and implementation details.

---

## Background / how this project was scoped

Built as a smaller, first-principles learning project (separate from OpsMind
AI) to deeply understand RAG, embeddings, pgvector, tool calling, and
agentic AI before returning to finish OpsMind with real understanding.

**Core design decision, locked in early:** a three-tier confidence branch —
high similarity match returns the user's own stored answer **verbatim**, no
LLM call at all; medium similarity asks for confirmation; low similarity
lets Gemini answer freely from general knowledge. This came from a specific
real requirement: the user has memorized real interview answers and cannot
risk an LLM subtly rewording them.

---

## Day 1 — FastAPI + Gemini chat with conversation memory

**Built:** `app/config.py` (env loading), `app/db.py` (`Message` model),
`app/gemini_client.py` (`generate_reply`, resends full conversation every
call since LLMs are stateless), `app/main.py` (`/chat` endpoint backed by
Postgres).

**Key decision:** Postgres from day 1, not an in-memory dict — an in-memory
Python variable is lost on restart and inconsistent across multiple server
processes; Postgres is the single shared source of truth.

**Issue — `gemini-2.5-flash` returned 404** ("no longer available to new
users"). Fixed by switching to `gemini-3.1-flash-lite`.

**Result:** confirmed working — multi-turn recall verified, and a fresh
`conversation_id` correctly showed no memory (proves per-conversation
scoping).

---

## Day 2 — Embeddings + pgvector storage

**Goal:** convert interview Q&A content into vectors and store them in
pgvector.

**Infra:** Docker Compose with `pgvector/pgvector:pg17`, `CREATE EXTENSION
IF NOT EXISTS vector;` run inside the container.

**Source data:** `Backend_Interview_QA.pdf` — ~49 Q&A pairs across Python,
Django, DRF, FastAPI, SQL, AWS. Manually converted to structured
`data/qa_seed.json` (topic, question, answer per entry).

**Schema (kept intentionally minimal):**
```python
class QAEntry(Base):
    __tablename__ = "qa_entries"
    id = Column(Integer, primary_key=True)
    topic = Column(String, nullable=False)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    embedding = Column(Vector(768), nullable=False)
```
No `difficulty`/`company`/`tags` yet — deferred until a real feature needs
them (e.g. filtering retrieval by topic).

**Ingestion (`scripts/ingest_qa.py`):** embeds `question + answer` together
(not question alone, so a differently-phrased search query still matches),
stores each as a `QAEntry` row.

**Issue — embedding model deprecated:** `text-embedding-004` was shut down
January 14, 2026. Switched to `gemini-embedding-001`.

**Issue — dimension mismatch:** `gemini-embedding-001` defaults to 3072
dimensions, not 768. Fixed with `output_dimensionality=768` in the embed
call.

**Issue — SDK fully deprecated mid-build:** `google.generativeai` was
retired in favor of `google.genai`. Migrated `generate_reply()` and
`embed_text()`. Old SDK used a stateful `start_chat()`/`send_message()`
pattern; new SDK is fully stateless — the whole conversation is passed as
`contents` every call, which makes the "AI has no memory between calls"
lesson from Day 1 explicit in the code itself.

**Result:** 49 entries ingested successfully, confirmed in pgvector. `/chat`
re-tested and confirmed working under the new SDK (correct multi-turn
reasoning, not just echoing).

**This is the current stable baseline.** Tagged in git as `day-2-complete`.

---

## Detour: Production-Pipeline Exploration (explored, then reverted)

After Day 2, the project's learning philosophy was temporarily redirected
toward building a full production-style document ingestion pipeline (PDF
upload → extraction → cleaning → chunking → embedding → storage →
retrieval), on the reasoning that the Day 2 approach (hand-converting a PDF
to JSON) was a shortcut that skipped real ingestion engineering.

This was later reverted via `git reset --hard` + `git clean -fd` back to
the `day-2-complete` tag, on the decision to keep AnswerVault's original
scope (learn RAG/agent concepts end-to-end without getting stuck for
multiple days on ingestion engineering specifically) and study the
production-ingestion pipeline as a **separate, dedicated learning track**
later, rather than folding it into this project's main path.

**The code no longer exists on disk**, but the concepts covered were real
and correctly understood before the revert. Recorded here so nothing is
lost and this can be picked up as standalone study material.

### What was built and verified during the detour

**1. Document upload endpoint**
A `POST /documents/upload` endpoint accepting a PDF via FastAPI's
`UploadFile`, saving it to local disk, and storing metadata (filename,
file path, size, status) in a new `Document` table — proven working
end-to-end against a real 172KB PDF.

Key lesson: databases are bad at storing large binary blobs efficiently;
real systems store files on disk or object storage (S3) and keep only a
*reference* in the database.

**2. PDF text extraction with PyMuPDF**
`fitz.open(file_path)` + `page.get_text()`, page markers inserted for
future source attribution. Chosen over `pypdf` for speed (C-based under the
hood) and better handling of complex layouts — a genuine production-grade
choice, not an arbitrary pick.

**Real, evidence-based finding:** extraction worked near-perfectly for
prose, section headers, bullet lists, and even multi-line Python code
blocks (indentation, comments, and nested brackets all survived intact).
It broke specifically on 2D ASCII box-drawing diagrams — PyMuPDF extracts
in reading-order (left-to-right, top-to-bottom), which has no way to
represent a diagram's actual 2D spatial structure. Also observed a code
block getting split across a page boundary, mid-function — a concrete,
evidence-based reason to chunk by content size/structure later, not by
page.

**3. Text cleaning — a real debugged heuristic**
Built `remove_diagrams()`, `fix_bullet_points()`, `normalize_whitespace()`.

The first version of diagram detection scored each line individually by
box-character density (`box_chars / line_length`). It failed silently on
lines with a diagram border character plus lots of interior padding/spaces
(e.g. a centered label inside a box) — the padding diluted the ratio below
threshold, so those lines were wrongly kept.

**The fix:** switched from a per-line score to a stateful approach —
enter "diagram mode" when a line has 3+ box-drawing characters, stay in
that mode (skipping every line) until a line with zero box characters and
4+ real letters appears (genuine prose), which signals the diagram has
ended. Verified against the real E-Z Auto PDF: the full architecture
diagram correctly collapsed to a single `[DIAGRAM REMOVED]` marker, with
prose resuming cleanly right after.

This is a genuinely good, real debugging story: a heuristic that looked
reasonable, failed on real data, was diagnosed correctly (padding diluting
a ratio-based score), and was fixed with a different algorithmic approach
(state machine instead of per-line scoring) rather than a patch.

**4. Architecture refactor — routers / services / models**
Restructured from a flat `main.py` into:
```
app/
  db.py            # engine, session, Base, get_db(), init_db() only
  models/           # Message, Document + DocumentStatus enum, one file each
  routers/           # chat.py, documents.py — HTTP layer only
  services/            # documents/upload.py — business logic, no FastAPI imports
```
Key lessons genuinely internalized, worth remembering even without the code:
- **Dependency injection for DB sessions** (`Depends(get_db)`) — a service
  receiving a session as a parameter (not creating its own) can be called
  from a test, script, cron job, or agent tool, not just a live HTTP
  request.
- **Routers vs. services separation** — a router only knows about HTTP; a
  service has zero FastAPI imports and doesn't know it's being called from
  an endpoint. This matters directly for later phases like tool calling and
  agentic loops, where the same logic needs to be called without going
  through HTTP at all.
- **`str, Enum` for status fields** (`class DocumentStatus(str, Enum)`) —
  prevents typos like `"uploded"` silently entering the database, and
  serializes cleanly as a plain string in JSON responses.
- This refactor was verified correct via a structured Codex review against
  an explicit target spec (10 checks: folder structure, DI consistency,
  no models in `db.py`, no manual `SessionLocal()` calls outside `db.py`,
  etc.) — all checks passed before the revert.

### Why it was reverted

Two compounding reasons, not one:
1. **Pacing** — going this deep on ingestion engineering (upload, PyMuPDF,
   cleaning heuristics, a full architectural refactor) was taking multiple
   real days for what was meant to be a supporting phase, not the main
   subject of AnswerVault.
2. **Chunking was next**, and doing it with the same exhaustive rigor
   (character vs. token-based, overlap tuning, comparing strategies)
   threatened to add several more days before AnswerVault's actual core
   goal (RAG confidence tiers, memory, tool calling, agentic loop) was even
   reached.

**Decision:** treat "production-grade document ingestion" (upload → extract
→ clean → chunk → embed) as its own **separate, dedicated learning topic**
to study later — not abandoned, just decoupled from AnswerVault's timeline.
AnswerVault resumes from the `day-2-complete` baseline and continues with
its original plan: similarity search → confidence tiers → RAG → memory →
tools → agentic loop, using the already-embedded 49 Q&A entries.

### Parked knowledge — to revisit as a separate study track later

- Chunking strategies: character-based vs. token-based, chunk size and
  overlap tuning, recursive/semantic chunking, why naive chunking splits
  meaning apart
- Full production ingestion pipeline for arbitrary file types (PDF, DOCX,
  HTML, Markdown), not just one clean PDF format
- Async ingestion / background jobs (Celery, task queues) for
  extraction+chunking+embedding at scale, instead of synchronous request
  handling
- Metadata filtering, hybrid search (vector + keyword), and re-ranking as
  retrieval quality improvements
- Revisiting whether/how to reintroduce the routers/services/models
  architecture once AnswerVault's core feature set (through agentic loop)
  is complete and the codebase has genuinely outgrown a flat structure

---

## Day 3 — Similarity search (resuming original plan)

**Scope decision (unchanged from before the detour):** similarity search
and full RAG generation kept as separate days. Day 3 proves vector search
alone can retrieve the correct answer — no Gemini generation call involved,
only embedding the query and searching.

**Built:**
- `embed_query()` in `app/gemini_client.py` — embeds the incoming question
  using `task_type="retrieval_query"`, deliberately different from the
  `"retrieval_document"` type used when the Q&A entries were originally
  embedded on Day 2. Gemini optimizes the vector differently depending on
  which side of a search it's playing; matching the type on both sides
  improves match quality.
- `scripts/search_qa.py` — embeds a query, runs a cosine-distance search
  against `qa_entries` via pgvector's `.cosine_distance()` method (wraps
  the underlying `<=>` SQL operator), returns top-K ranked results with
  similarity scores (`1 - distance`, for readable "higher = better"
  numbers).

**Issue — duplicate rows found:** first test run returned the exact same
entry twice with an identical similarity score. Root cause: `ingest_qa.py`
has no duplicate-check, and it had been run more than once across the
original build and the detour/revert cycle, doubling the table (49 → 98
rows). Fixed by truncating `qa_entries` and re-ingesting once
(`TRUNCATE TABLE qa_entries;` then `python -m scripts.ingest_qa`), restoring
exactly 49 rows. Noted as a real gap: production ingestion scripts need
either an upsert/dedupe strategy or a check for existing rows before
insert — not built here, since a single manual re-ingest was enough for
this project's needs.

**Threshold calibration — real evidence overturned the original guess.**
Ran five deliberately varied test queries against the real embedded data:

| Query | Top match | Similarity |
|---|---|---|
| "Iterators vs generators?" (exact stored question) | Iterators vs generators? | 0.772 |
| "Explain generators in Python" (paraphrase) | Iterators vs generators? | 0.741 |
| "select_related vs prefetch_related?" (paraphrase) | select_related vs prefetch_related? | 0.809 |
| "How do I center a div in CSS" (unrelated) | (best of unrelated junk) | 0.527 |
| "Tell me about the GIL" (casual phrasing) | What is the GIL? | 0.767 |

**Key finding:** even an exact, word-for-word question match only reached
0.772–0.809 similarity — nowhere near the originally planned 0.90 "high
confidence" threshold. Gemini's embedding space doesn't spread scores as
widely as intuition suggests; shipping the original threshold would have
meant the system could *never* return a verbatim stored answer, silently
defeating the project's core purpose.

**Also found:** a real near-miss case. Query 3's correct answer scored
0.809, but a related-but-wrong entry (N+1 query problem) scored 0.735 —
uncomfortably close to query 2's *correct* match score of 0.741. A single
global threshold cannot cleanly separate "right answer" from "closely
related wrong answer" in this zone. This is concrete evidence that the
medium-confidence "did you mean...?" tier is doing real, necessary work,
not just a nice-to-have safety net.

**Revised thresholds, set from evidence rather than guessed:**
```
HIGH   (> 0.75)          → return stored answer verbatim, no LLM
MEDIUM (0.60 – 0.75)      → ask "did you mean...?" confirmation
LOW    (< 0.60)            → Gemini answers from general knowledge
```
0.60 sits comfortably above the unrelated-content ceiling (0.527); 0.75
sits at the boundary where genuine correct matches cluster, correctly
routing the tricky 0.735 distractor into confirmation rather than falsely
auto-returning it as verbatim truth.

**Housekeeping:** temporary debugging script `scripts/check_duplicates.py`
(used only to confirm the duplicate-row count) deleted after the fix was
verified — not part of the ongoing project, per the standing rule that
one-off diagnostic scripts get removed once their job is done.
`scripts/search_qa.py` was kept, since it's a real, reusable project
capability (the foundation of Day 4's retrieval layer), not scaffolding.

**Result:** Day 3 complete. Real similarity search working against real
data, with thresholds grounded in actual evidence rather than assumption —
directly ready to feed into Day 4's confidence-tier branching logic.

---

## Day 4 — Confidence-tier branching (planned)

*(To be filled in once built.)*
