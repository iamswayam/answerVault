# scripts/test_retrieval.py
from app.services.retrieval import retrieve_answer

test_queries = [
    "Iterators vs generators?",
    "How do I center a div in CSS",
    "What's the deal with select_related?",
    "Have you heard of 4 pillars in python?",
    "Have you used S3 in your project?",
    "When to use FastAPI over Django or Flask",
    "Have you heard transaction in SQL? If yes, what is it?",
    # New — deliberately targeting MEDIUM and the Day 3 near-miss
    "What is the N+1 query problem?",              # near-miss re-test vs select_related
    "database query optimization techniques",         # broad, could partially match several
    "difference between joins in SQL",                  # oddly phrased vs stored "types of JOIN"
    "how does python manage memory",                       # loosely related, no exact stored entry
]

for q in test_queries:
    result = retrieve_answer(q)
    print("=" * 50)
    print(f"Query      : {q}")
    print(f"Similarity : {result['similarity']}")
    print(f"Tier       : {result['tier'].value.upper()}")
    if result["tier"].value != "low":
        print(f"\nMatched Question\n{'-' * 16}\n{result['matched_question']}")
        print(f"\nStored Answer\n{'-' * 13}\n{result['matched_answer']}")
    print("=" * 50 + "\n")