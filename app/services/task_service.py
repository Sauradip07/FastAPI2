import asyncio
import uuid
from fastapi import Request

tasks = {}

async def start_new_task(request: Request):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"progress": 0, "status": "Started"}
    background_tasks = request.app.state.background_tasks
    background_tasks.add_task(run_task, task_id)
    return task_id

async def get_task_progress(task_id: str, request: Request):
    while True:
        if task_id not in tasks:
            yield {"event": "error", "data": "Task not found"}
            break
        
        progress = tasks[task_id]["progress"]
        status = tasks[task_id]["status"]
        
        if status == "Completed":
            yield {"event": "complete", "data": f"Task completed: {progress}%"}
            break
        
        yield {"event": "progress", "data": f"Progress: {progress}%"}
        
        if await request.is_disconnected():
            break
        
        await asyncio.sleep(1)

async def run_task(task_id: str):
    for i in range(1, 101):
        tasks[task_id]["progress"] = i
        await asyncio.sleep(0.5)  # Simulate work
    tasks[task_id]["status"] = "Completed"