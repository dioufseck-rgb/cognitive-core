"""
HITL governance task routes.

GET  /api/tasks/pending            — list pending approval tasks
POST /api/tasks/{task_id}/approve  — approve a governance gate
POST /api/tasks/{task_id}/reject   — reject a governance gate

IMPORTANT: coordinator.approve() / reject() are synchronous and trigger
LLM delegation chains that can run for minutes.  We run them in the
thread-pool executor so the async event loop stays free for UI polling.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import cognitive_core.api.deps as deps

router = APIRouter()


class ApproveRequest(BaseModel):
    approver: str = "analyst"
    notes: str = ""


class RejectRequest(BaseModel):
    rejector: str = "analyst"
    reason: str = ""


@router.get("/tasks/pending")
async def get_pending_tasks() -> list[dict[str, Any]]:
    """Return all pending governance approval tasks."""
    coord = deps.get_coordinator()
    executor = deps.get_executor()
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(executor, coord.list_pending_approvals)
    except Exception as exc:
        raise HTTPException(500, f"Could not fetch tasks: {exc}")


@router.post("/tasks/{task_id}/approve")
async def approve_task(task_id: str, req: ApproveRequest) -> dict[str, Any]:
    """
    Approve a governance gate.

    Runs coordinator.approve() in the thread pool so the event loop
    stays free for UI polling during the (potentially multi-minute)
    LLM delegation chain that fires after approval.
    """
    coord = deps.get_coordinator()
    executor = deps.get_executor()

    # Look up task to get instance_id (fast DB read, OK inline)
    task = coord.tasks.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")

    instance_id = task.callback.instance_id

    loop = asyncio.get_event_loop()
    # Fire-and-forget: submit to thread pool and return immediately.
    # The approval may trigger a multi-minute LLM chain; the UI polls for updates.
    loop.run_in_executor(executor, lambda: _run_approve(coord, instance_id, req.approver, req.notes))

    return {"instance_id": instance_id, "status": "processing"}


@router.post("/tasks/{task_id}/reject")
async def reject_task(task_id: str, req: RejectRequest) -> dict[str, Any]:
    """
    Reject a governance gate.  Runs in thread pool for same reasons as approve.
    """
    coord = deps.get_coordinator()
    executor = deps.get_executor()

    task = coord.tasks.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task not found: {task_id}")

    instance_id = task.callback.instance_id

    loop = asyncio.get_event_loop()
    # Fire-and-forget: return immediately, UI polls for state change.
    loop.run_in_executor(executor, lambda: _run_reject(coord, instance_id, req.rejector, req.reason))

    return {"instance_id": instance_id, "status": "processing"}


# ── Thread-worker helpers ─────────────────────────────────────────────
# These run inside ThreadPoolExecutor workers which have no asyncio event
# loop by default.  LangGraph / MCP internals call asyncio.get_event_loop(),
# so we create a dedicated loop for each worker invocation.

def _run_approve(coord, instance_id: str, approver: str, notes: str):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.approve(instance_id=instance_id, approver=approver, notes=notes)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _run_reject(coord, instance_id: str, rejector: str, reason: str):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.reject(instance_id=instance_id, rejector=rejector, reason=reason)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
