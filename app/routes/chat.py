from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.rag_app import rag_app

router = APIRouter()

class ChatQuestion(BaseModel):
    question: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    chat_id: str
    question: str
    answer: str

class ChatHistory(BaseModel):
    chat_id: str
    history: List[dict]

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_question: ChatQuestion):
    try:
        result = rag_app.get_answer(chat_question.question, chat_question.chat_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/{chat_id}", response_model=ChatHistory)
async def get_chat_history(chat_id: str):
    history = rag_app.get_chat_history(chat_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat history not found")
    return ChatHistory(chat_id=chat_id, history=history)