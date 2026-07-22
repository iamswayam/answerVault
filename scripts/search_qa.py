# scripts/search_qa.py
from app.db import SessionLocal, QAEntry
from app.gemini_client import embed_query

def search(query_text: str, top_k: int = 3):
    query_vector = embed_query(query_text)

    db = SessionLocal()
    results = (
        db.query(
            QAEntry,
            QAEntry.embedding.cosine_distance(query_vector).label("distance")
        )
        .order_by("distance")
        .limit(top_k)
        .all()
    )
    db.close()

    print(f"\nQuery: {query_text}")
    for entry, distance in results:
        similarity = 1 - distance
        print(f"  [{similarity:.3f}] ({entry.topic}) {entry.question}")

if __name__ == "__main__":
    test_queries = [
        "Iterators vs generators?",                      # exact match to a stored question
        "Explain generators in Python",                    # close paraphrase
        "What's the difference between select_related and prefetch_related?",  # paraphrase, different topic
        "How do I center a div in CSS",                       # totally unrelated, should score low
        "Tell me about the GIL",                                # short/casual phrasing of a real entry
    ]
    for q in test_queries:
        search(q)