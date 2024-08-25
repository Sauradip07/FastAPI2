from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from app.services.task_service import start_new_task, get_task_progress

router = APIRouter()

@router.post("/start-task")
async def start_task(request: Request):
    task_id = await start_new_task(request)
    return {"task_id": task_id}

@router.get("/task-progress/{task_id}")
async def task_progress(task_id: str, request: Request):
    return EventSourceResponse(get_task_progress(task_id, request))