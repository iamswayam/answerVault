from app.db import SessionLocal, QAEntry, Message
from app.gemini_client import embed_query, generate_free_answer, rewrite_query_with_history

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.60


def _search_top_match(search_text: str):
    query_vector = embed_query(search_text)
    db = SessionLocal()
    entry, distance = (
        db.query(
            QAEntry,
            QAEntry.embedding.cosine_distance(query_vector).label("distance"),
        )
        .order_by("distance")
        .first()
    )
    db.close()
    return entry, 1 - distance


def _load_history(conversation_id: str) -> list[dict]:
    db = SessionLocal()
    past = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.id.asc())
        .all()
    )
    db.close()
    return [{"role": m.role, "parts": m.content} for m in past]


def _save_message(conversation_id: str, role: str, content: str):
    db = SessionLocal()
    db.add(Message(conversation_id=conversation_id, role=role, content=content))
    db.commit()
    db.close()


def answer_query(conversation_id: str, query_text: str) -> dict:
    # Load history BEFORE saving the current message, so it only contains
    # prior turns -- exactly what's needed to rewrite an ambiguous follow-up.
    prior_history = _load_history(conversation_id)
    search_text = rewrite_query_with_history(prior_history, query_text)

    entry, similarity = _search_top_match(search_text)

    _save_message(conversation_id, "user", query_text)  # store the ORIGINAL wording

    if similarity > HIGH_THRESHOLD:
        _save_message(conversation_id, "model", entry.answer)
        return {
            "source": "your_documents",
            "confidence": "high",
            "similarity": round(similarity, 3),
            "matched_question": entry.question,
            "answer": entry.answer,
            "resolved_query": search_text,  # useful for debugging what was actually searched
        }

    if similarity > MEDIUM_THRESHOLD:
        return {
            "source": "suggestion",
            "confidence": "medium",
            "similarity": round(similarity, 3),
            "matched_question": entry.question,
            "matched_answer": entry.answer,
            "original_query": query_text,
        }

    history_for_generation = prior_history + [{"role": "user", "parts": query_text}]
    general_answer = generate_free_answer(history_for_generation)
    _save_message(conversation_id, "model", general_answer)
    return {
        "source": "general_knowledge",
        "confidence": "low",
        "similarity": round(similarity, 3),
        "matched_question": None,
        "answer": general_answer,
    }


def confirm_suggestion(conversation_id: str, matched_answer: str) -> None:
    _save_message(conversation_id, "model", matched_answer)


def answer_general(conversation_id: str, query_text: str) -> str:
    history = _load_history(conversation_id)
    general_answer = generate_free_answer(history)
    _save_message(conversation_id, "model", general_answer)
    return general_answer