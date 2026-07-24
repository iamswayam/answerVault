from fastapi import FastAPI
from pydantic import BaseModel
from app.db import SessionLocal, Message, init_db
from app.gemini_client import generate_reply

from fastapi.staticfiles import StaticFiles
from app.routers import qa
# ...

app = FastAPI()
init_db()  # creates the messages table on startup if it doesn't exist

app.include_router(qa.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

class ChatRequest(BaseModel):
    conversation_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    db = SessionLocal()

    # 1. Load prior turns for this conversation, oldest first
    past_messages = (
        db.query(Message)
        .filter(Message.conversation_id == req.conversation_id)
        .order_by(Message.id.asc())
        .all()
    )

    history = [{"role": m.role, "parts": m.content} for m in past_messages]

    # 2. Append the new user message
    history.append({"role": "user", "parts": req.message})

    # 3. Save the user's message to the DB
    db.add(Message(conversation_id=req.conversation_id, role="user", content=req.message))
    db.commit()

    # 4. Get Gemini's reply using full history
    reply = generate_reply(history)

    # 5. Save the model's reply to the DB
    db.add(Message(conversation_id=req.conversation_id, role="model", content=reply))
    db.commit()
    db.close()

    return {"reply": reply}