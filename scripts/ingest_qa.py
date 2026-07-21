# scripts/ingest_qa.py
import json
from app.db import SessionLocal, QAEntry, init_db
from app.gemini_client import embed_text  # we'll add this function next

def ingest(json_path: str):
    init_db()
    db = SessionLocal()

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    for entry in entries:
        text_to_embed = f"{entry['question']} {entry['answer']}"
        vector = embed_text(text_to_embed)

        db.add(QAEntry(
            topic=entry["topic"],
            question=entry["question"],
            answer=entry["answer"],
            embedding=vector,
        ))

    db.commit()
    db.close()
    print(f"Ingested {len(entries)} entries.")

if __name__ == "__main__":
    ingest("data/qa_seed.json")