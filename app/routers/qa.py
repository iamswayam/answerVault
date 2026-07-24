from fastapi import APIRouter
from pydantic import BaseModel
from app.services.retrieval import answer_query, answer_general, confirm_suggestion

router = APIRouter(tags=["qa"])


class AskRequest(BaseModel):
    conversation_id: str
    message: str


class GeneralRequest(BaseModel):
    conversation_id: str
    message: str


class ConfirmRequest(BaseModel):
    conversation_id: str
    matched_answer: str


@router.post("/ask")
def ask(req: AskRequest):
    return answer_query(req.conversation_id, req.message)


@router.post("/ask/general")
def ask_general(req: GeneralRequest):
    answer = answer_general(req.conversation_id, req.message)
    return {
        "source": "general_knowledge",
        "confidence": "low",
        "answer": answer,
    }


@router.post("/ask/confirm")
def ask_confirm(req: ConfirmRequest):
    confirm_suggestion(req.conversation_id, req.matched_answer)
    return {"status": "saved"}