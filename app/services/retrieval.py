from enum import Enum
from app.db import SessionLocal, QAEntry
from app.gemini_client import embed_query

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.60

class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

def retrieve_answer(query_text: str) -> dict:
    query_vector = embed_query(query_text)

    db = SessionLocal()
    entry, distance = (
        db.query(QAEntry, QAEntry.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .first()
    )
    db.close()

    similarity = 1 - distance

    if similarity > HIGH_THRESHOLD:
        tier = ConfidenceTier.HIGH
    elif similarity > MEDIUM_THRESHOLD:
        tier = ConfidenceTier.MEDIUM
    else:
        tier = ConfidenceTier.LOW

    return {
        "tier": tier,
        "similarity": round(similarity, 3),
        "matched_question": entry.question,
        "matched_answer": entry.answer,
        "topic": entry.topic,
    }