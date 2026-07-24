from enum import Enum
from app.db import SessionLocal, QAEntry
from app.gemini_client import embed_query, generate_free_answer

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.60

class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

MEDIUM_FRAMING = (
    "I'm not fully confident, but the closest match in your notes is about "
    "\"{matched_question}\" (similarity: {similarity}). Here's that answer, "
    "in case it's what you meant:"
)

def _search_top_match(query_text: str):
    query_vector = embed_query(query_text)
    db = SessionLocal()
    entry, distance = (
        db.query(QAEntry, QAEntry.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .first()
    )
    db.close()
    return entry, 1 - distance


def answer_query(query_text: str) -> dict:
    entry, similarity = _search_top_match(query_text)

    if similarity > HIGH_THRESHOLD:
        return {
            "source": "your_documents",
            "confidence": "high",
            "similarity": round(similarity, 3),
            "matched_question": entry.question,
            "answer": entry.answer,   # verbatim, untouched, no LLM involved
        }

    if similarity > MEDIUM_THRESHOLD:
        framing = MEDIUM_FRAMING.format(
            matched_question=entry.question,
            similarity=round(similarity, 3),
        )
        return {
            "source": "documents_plus_general",
            "confidence": "medium",
            "similarity": round(similarity, 3),
            "matched_question": entry.question,
            "answer": f"{framing}\n\n{entry.answer}",   # framing is hardcoded, answer is still verbatim
        }

    # LOW — no relevant match, fall back to Gemini entirely
    general_answer = generate_free_answer(query_text)
    return {
        "source": "general_knowledge",
        "confidence": "low",
        "similarity": round(similarity, 3),
        "matched_question": None,
        "answer": general_answer,
    }