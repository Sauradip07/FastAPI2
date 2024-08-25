from fastapi import APIRouter, HTTPException
from app.models import Question
from app.rag_app import rag_app

router = APIRouter()

@router.post("/ask")
async def ask_question(question: Question):
    try:
        answer = rag_app.get_answer(question.question)
        return {"question": question.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))