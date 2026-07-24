# scripts/test_day5.py
from app.services.retrieval import answer_query

test_queries = [
    "Iterators vs generators?",                     # HIGH
    "database query optimization techniques",         # MEDIUM
    "how does python manage memory",                     # MEDIUM (genuine near-miss)
    "How do I center a div in CSS",                        # LOW
]

for q in test_queries:
    result = answer_query(q)
    print("=" * 50)
    print(f"Query      : {q}")
    print(f"Source     : {result['source']}")
    print(f"Confidence : {result['confidence']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("=" * 50 + "\n")