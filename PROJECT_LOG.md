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

**4. Architecture refactor — routers / services / models**
Restructured from a flat `main.py` into a routers/services/models split
with proper dependency injection (`Depends(get_db)`) and `str, Enum`
status fields. Verified correct via a structured Codex review against an
explicit target spec before the revert.

### Why it was reverted

Two compounding reasons: (1) pacing — ingestion engineering was taking
multiple real days for what was meant to be a supporting phase, and (2)
chunking was next, and doing it with the same exhaustive rigor threatened
to add several more days before AnswerVault's actual core goal (RAG
confidence tiers, memory, tool calling, agentic loop) was even reached.

**Decision:** treat "production-grade document ingestion" as its own
**separate, dedicated learning topic** to study later — not abandoned,
just decoupled from AnswerVault's timeline. AnswerVault resumed from the
`day-2-complete` baseline.

### Parked knowledge — to revisit as a separate study track later

- Chunking strategies: character-based vs. token-based, chunk size and
  overlap tuning, recursive/semantic chunking, why naive chunking splits
  meaning apart
- Full production ingestion pipeline for arbitrary file types (PDF, DOCX,
  HTML, Markdown), not just one clean PDF format
- Async ingestion / background jobs (Celery, task queues)
- Metadata filtering, hybrid search (vector + keyword), and re-ranking
- Revisiting the routers/services/models architecture once AnswerVault's
  core feature set is complete and the codebase has genuinely outgrown a
  flat structure

---

## Day 3 — Similarity search (resuming original plan)

**Built:**
- `embed_query()` in `app/gemini_client.py` — embeds the incoming question
  using `task_type="retrieval_query"`, deliberately different from the
  `"retrieval_document"` type used for stored entries.
- `scripts/search_qa.py` — cosine-distance search against `qa_entries` via
  pgvector's `.cosine_distance()`, returns top-K ranked results.

**Issue — duplicate rows found:** `ingest_qa.py` had been run more than
once across the original build and the detour/revert cycle, doubling the
table (49 → 98 rows). Fixed by truncating and re-ingesting once.

**Threshold calibration — real evidence overturned the original guess.**
Five test queries against real embedded data showed even an exact
word-for-word question match only reached 0.772–0.809 similarity — nowhere
near the originally planned 0.90 "high confidence" threshold. Also found a
real near-miss: a correct match (0.741) and a related-but-wrong match
(0.735) landed uncomfortably close together, concrete evidence the
medium-confidence tier is doing real, necessary work.

**Revised thresholds, set from evidence:**
```
HIGH   (> 0.75)   → return stored answer verbatim, no LLM
MEDIUM (0.60–0.75) → ask "did you mean...?" confirmation
LOW    (< 0.60)     → Gemini answers from general knowledge
```

**Result:** Day 3 complete. Tagged `day-3-complete`.

---

## Day 4 — Confidence-tier branching

**Built:** `answer_query()` (originally `retrieve_answer()`) in
`app/services/retrieval.py` — takes a query, searches, and branches into
HIGH/MEDIUM/LOW based on Day 3's calibrated thresholds.

**Testing — 11 real queries across all three tiers,** including two
deliberate stress tests:
- Re-tested the Day 3 near-miss ("What is the N+1 query problem?") —
  correctly resolved HIGH to its own entry (0.814), not the
  select_related distractor.
- Two genuinely ambiguous queries ("database query optimization
  techniques", "how does python manage memory") correctly triggered
  MEDIUM rather than falsely claiming HIGH confidence.

**Result:** Day 4 complete. Branching logic proven against real, varied
evidence, not just happy-path queries. Tagged `day-4-complete`.

---

## Day 5 — Full RAG: MEDIUM/LOW wiring, frontend, and conversation memory
### (in progress — functional but not yet reliable; see Known Issues)

This day expanded far beyond the original scope (wire LOW to Gemini, add
source labels) into a real, iteratively-debugged RAG interface. Documenting
the full arc, not just the end state, since the debugging *is* the
learning here.

### MEDIUM-tier design debate — verbatim guarantee vs. clean UX

Initial plan (ChatGPT-suggested): feed the stored answer into Gemini as
"evidence" and let it synthesize one grounded response for MEDIUM. Rejected
— this would let Gemini paraphrase a stored answer, directly violating the
project's founding requirement that memorized answers are never reworded
by an LLM, at any confidence tier.

**Resolution:** hardcoded (not Gemini-generated) framing sentence prepended
to the verbatim stored answer, e.g. "I'm not fully confident, but the
closest match is about X — here's that answer:". Verbatim guarantee stays
architectural (the stored answer text is only ever f-string concatenated,
never passed through any Gemini call) while still giving a coherent,
non-jarring response instead of two disconnected blocks.

### MEDIUM redesigned again — suggestion + explicit confirmation

The hardcoded-framing version still auto-displayed the full stored answer
even when the match was a genuine stretch (e.g. "how does python manage
memory" → Multithreading vs multiprocessing, 0.648). Redesigned so MEDIUM
returns a **suggestion only**; the frontend shows a "did you mean X?" card
with Yes/No buttons, and the full answer is only revealed if the user
confirms. `/ask/confirm` persists an accepted suggestion to conversation
history.

### Frontend built

`static/index.html`, served via FastAPI `StaticFiles` mounted at `/`
(after routers, so it doesn't shadow API routes). Plain HTML/CSS/JS, no
build step, no framework — same-origin with the API so no CORS setup
needed. Dark theme, chat-bubble UI, color-coded confidence tags.

New endpoints in `app/routers/qa.py`: `POST /ask` (wraps `answer_query()`),
`POST /ask/general` (LOW-style Gemini fallback, used both for genuine LOW
results and for the MEDIUM "No" button), `POST /ask/confirm` (persists a
confirmed MEDIUM suggestion to history).

### Bug — /ask had no conversation memory at all

`/ask` originally ran fully isolated per request — no `conversation_id`,
no persistence, unrelated to the `/chat` endpoint's memory. Follow-up
questions ("now show me one that measures time") only worked by accident
when they happened to repeat enough keywords from the original question.

**Fix:** `/ask` now takes a `conversation_id`, saves every exchange to the
same `messages` table `/chat` uses, and `generate_free_answer()` was
changed to accept full history (not a single isolated string) plus a
`system_instruction` steering it toward short, plain-English,
interview-flashcard-style answers instead of a generic verbose default.

### Bug — ambiguous follow-ups broke vector search specifically

Even with conversation memory added to *generation*, **vector search itself
still only ever embedded the current message alone.** Real failure
observed: after discussing "Django vs Flask vs FastAPI" (correctly HIGH,
0.793), asking "I asked difference?" embedded only those three words,
matched an unrelated stored entry (`is` vs `==`, because it contains the
word "difference") at HIGH confidence, and confidently returned the wrong
answer.

**Fix — query rewriting:** `rewrite_query_with_history()` added to
`gemini_client.py`. Before searching, if conversation history exists, an
ambiguous follow-up is rewritten into a standalone question using prior
turns (e.g. "I asked difference?" → "What is the difference between
Django, FastAPI, and Flask?") *before* embedding. The rewritten text is
used only for search; the original wording is what's saved to history, so
the conversation log still reads naturally.

**Known limitation of this fix, found immediately after:** query rewriting
only helps when a follow-up secretly contains a real question. It does
**not** help meta-comments or control messages ("Answer is wrong", "I want
general answer?") — these have no real question to recover, so rewriting
them still produces text that searches right back into the same topic
(e.g. "Is the FastAPI/Django/Flask answer wrong?" → still matches the
FastAPI entry). Attempting to pattern-match phrases like "that's wrong" in
code was explicitly rejected as fragile and unreliable (English has
infinite equivalent phrasings).

**Fix — UI escape hatch instead of text-intent detection:** every
HIGH-confidence response now has a "Not what you wanted? Get a general
answer" button. Clicking it re-sends the frontend's already-known
*original* question straight to `/ask/general`, bypassing search entirely.
Reliable by construction — no natural-language guessing involved.

### UI iteration

Light theme built first; later replaced with a dark theme (chat-bubble
layout, floating bottom input bar, color-coded uppercase tags) per
request. Functionality unchanged across the visual redesign.

### KNOWN ISSUE — unresolved, real, needs a proper fix

**The escape hatch fixes one bad answer, once, but the system has no
memory of that decision on the next turn.** Observed directly: after using
the "get general answer" button for the Django/FastAPI/Flask topic, the
very next message in the same conversation ("I want longer answer for all
3") hit `/ask` fresh, re-ran vector search from scratch, and landed right
back on the same stored FastAPI entry at HIGH confidence — forcing the user
to click the escape button again. `/ask` has no concept of "this
conversation has already decided this topic needs general knowledge, not
retrieval" — every single message re-searches independently regardless of
what was decided one turn earlier.

This is a real, unresolved design gap, not a one-off glitch. Explicitly
**not being patched immediately** — decision made to hold off on deeper
RAG/retrieval-vs-memory-interaction fixes until the remaining architecture
(Day 6 memory summarization, Day 7 tool calling, Day 8 agentic loop) is in
place, so the eventual fix can be designed with the full system in view
rather than patched piecemeal. Revisit explicitly before considering
AnswerVault's RAG behavior production-quality.

**Broader, explicit note on state:** prompt quality and RAG reliability
overall need significantly more adversarial testing before being
considered solid — this day proved the *architecture* (tiers, memory,
rewriting, escape hatches) all function, not that the system is tuned or
robust yet. Real interview-style usage will keep surfacing edge cases like
the ones above; that iteration is expected, not a sign something is wrong
with the approach.

**Result:** Day 5 functionally working end-to-end through a real UI, with
several real bugs found and fixed along the way — but explicitly **not**
marked complete. No tag issued for the frontend/memory work; commit made
as a work-in-progress checkpoint (`day-5-frontend-wip` in spirit, no tag
per the decision to only tag genuinely solid milestones).

---

## Day 6 — Long-term memory / summarization (planned)

*(To be filled in once built. Note: may need to incorporate a fix for the
Day 5 known issue — conversations "forgetting" a prior general-knowledge
decision — as part of this day's design, since summarization and
mode-tracking are related problems.)*