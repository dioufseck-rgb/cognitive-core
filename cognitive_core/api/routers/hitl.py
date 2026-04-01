"""
Cognitive Core — HITL & SSE endpoints (Sprint 1.1 + 1.2)

Three HITL endpoints:
  GET  /instances/{id}/workorder  — structured work order for a suspended instance
  POST /instances/{id}/decision   — inject human decision, resume workflow
  POST /instances/{id}/evidence   — supply missing evidence to suspended retrieve step

One SSE endpoint:
  GET  /instances/{id}/stream     — live action ledger stream (text/event-stream)

These form the machine interface of the governance loop.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import cognitive_core.api.deps as deps

router = APIRouter()


# ─── Request models ──────────────────────────────────────────────────

class DecisionRequest(BaseModel):
    decision: str                # e.g. "approve", "approve_modified", "deny", "refer"
    rationale: str = ""
    reviewer_id: str = "reviewer"


class EvidenceRequest(BaseModel):
    step_name: str               # the retrieve step that requested the evidence
    content: Any                 # the evidence payload
    content_type: str = "text"   # "text" | "json" | "document"


# ─── GET /instances/{id}/workorder ──────────────────────────────────

@router.get("/instances/{instance_id}/workorder")
async def get_workorder(instance_id: str) -> dict[str, Any]:
    """
    Return the structured work order for a suspended instance.

    Includes:
      - brief: one-paragraph summary for the reviewer
      - reasoning_trace: step-by-step decisions made so far
      - decision_options: list of valid decisions for this workflow
      - governance_tier, escalation_reason
    """
    coord = deps.get_coordinator()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")
    if instance.status.value != "suspended":
        raise HTTPException(400, f"Instance is {instance.status.value}, not suspended — work orders only exist for suspended instances")

    # Get the suspension record for the escalation brief
    suspension = coord.store.get_suspension(instance_id)
    if not suspension:
        raise HTTPException(404, f"No suspension record for instance {instance_id}")

    # Get the pending task payload (contains escalation brief + governance info)
    pending = coord.list_pending_approvals()
    task_payload: dict[str, Any] = {}
    task_id: str | None = None
    for p in pending:
        if p["instance_id"] == instance_id:
            task_payload = p
            task_id = p.get("task_id")
            break

    # Build reasoning trace from the action ledger
    ledger = coord.store.get_ledger(instance_id=instance_id)
    reasoning_trace = []
    for entry in ledger:
        details = entry.get("details", {})
        action_type = entry.get("action_type", "")
        # Include step executions in the trace
        if action_type in ("step_completed", "step_started", "step_failed", "governance_evaluated"):
            reasoning_trace.append({
                "action": action_type,
                "step": details.get("step_name", details.get("step", "")),
                "primitive": details.get("primitive", ""),
                "output": details.get("output", details.get("result", {})),
                "elapsed_ms": details.get("elapsed_ms", 0),
                "timestamp": entry.get("created_at", 0),
            })

    # Extract decision options from domain config if available
    decision_options = _get_decision_options(instance, task_payload)

    # Build brief from escalation_brief or fallback
    escalation_brief = task_payload.get("escalation_brief") or {}
    brief = (
        escalation_brief.get("summary")
        or task_payload.get("reason", "")
        or f"Instance {instance_id} is suspended pending human review."
    )

    return {
        "instance_id": instance_id,
        "task_id": task_id,
        "workflow_type": instance.workflow_type,
        "domain": instance.domain,
        "governance_tier": instance.governance_tier or task_payload.get("governance_tier", "GATE"),
        "status": instance.status.value,
        "suspended_at_step": suspension.suspended_at_step,
        "created_at": instance.created_at,
        "updated_at": instance.updated_at,
        "brief": brief,
        "escalation_reason": task_payload.get("reason", ""),
        "escalation_brief": escalation_brief,
        "reasoning_trace": reasoning_trace,
        "decision_options": decision_options,
        "result_summary": task_payload.get("result_summary", instance.result),
        "case_meta": instance.case_meta or {},
        "resume_nonce": suspension.resume_nonce,
    }


def _get_decision_options(instance, task_payload: dict) -> list[str]:
    """Derive valid decision options for this workflow type."""
    tier = (instance.governance_tier or task_payload.get("governance_tier", "")).upper()
    # GATE tier: full set of decisions
    if tier == "GATE":
        return ["approve", "approve_modified", "deny", "refer"]
    # SPOT_CHECK tier: approve or escalate
    if tier in ("SPOT_CHECK", "SPOT-CHECK"):
        return ["approve", "escalate"]
    # HOLD tier: release or escalate
    if tier == "HOLD":
        return ["release", "escalate", "deny"]
    # Default
    return ["approve", "deny"]


# ─── POST /instances/{id}/decision ──────────────────────────────────

@router.post("/instances/{instance_id}/decision")
async def post_decision(instance_id: str, req: DecisionRequest) -> dict[str, Any]:
    """
    Inject a human decision into a suspended instance and resume it.

    Body: { decision, rationale, reviewer_id }
    Effect: logs the decision, then triggers coordinator resume.
    """
    coord = deps.get_coordinator()
    executor = deps.get_executor()

    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")
    if instance.status.value != "suspended":
        raise HTTPException(400, f"Instance is {instance.status.value}, not suspended")
    if not req.decision:
        raise HTTPException(422, "decision field is required and must not be empty")
    if not req.reviewer_id:
        raise HTTPException(422, "reviewer_id field is required and must not be empty")

    # Log the human decision to the action ledger before resuming
    coord.store.log_action(
        instance_id=instance_id,
        correlation_id=instance.correlation_id,
        action_type="human_decision",
        details={
            "decision": req.decision,
            "rationale": req.rationale,
            "reviewer_id": req.reviewer_id,
            "timestamp": time.time(),
        },
        idempotency_key=f"decision:{instance_id}:{req.reviewer_id}:{req.decision}:{time.time()}",
    )

    # Fire-and-forget resume — LLM work may take minutes
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        lambda: _run_decision(coord, instance_id, req),
    )

    return {
        "instance_id": instance_id,
        "decision": req.decision,
        "reviewer_id": req.reviewer_id,
        "status": "processing",
    }


def _run_decision(coord, instance_id: str, req: DecisionRequest):
    """Run in thread pool. Map decision → approve/reject and resume."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Map semantic decision to coordinator method
        if req.decision in ("approve", "approve_modified", "release", "refer"):
            coord.approve(
                instance_id=instance_id,
                approver=req.reviewer_id,
                notes=f"{req.decision}: {req.rationale}",
            )
        elif req.decision in ("deny", "reject", "escalate"):
            coord.reject(
                instance_id=instance_id,
                rejector=req.reviewer_id,
                reason=f"{req.decision}: {req.rationale}",
            )
        else:
            # Unknown decision — treat as approval with the raw decision in notes
            coord.approve(
                instance_id=instance_id,
                approver=req.reviewer_id,
                notes=f"{req.decision}: {req.rationale}",
            )
    except Exception as exc:
        print(f"[hitl] Decision error ({instance_id}): {exc}", file=sys.stderr, flush=True)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# ─── POST /instances/{id}/evidence ──────────────────────────────────

@router.post("/instances/{instance_id}/evidence")
async def post_evidence(instance_id: str, req: EvidenceRequest) -> dict[str, Any]:
    """
    Supply missing evidence to a suspended retrieve step and resume.

    Body: { step_name, content, content_type }
    Effect: injects evidence into the workflow state, resumes from the correct step.
    """
    coord = deps.get_coordinator()
    executor = deps.get_executor()

    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")
    if instance.status.value != "suspended":
        raise HTTPException(400, f"Instance is {instance.status.value}, not suspended")
    if not req.step_name:
        raise HTTPException(422, "step_name field is required")
    if req.content is None:
        raise HTTPException(422, "content field is required")

    # Log evidence supply to the action ledger
    coord.store.log_action(
        instance_id=instance_id,
        correlation_id=instance.correlation_id,
        action_type="evidence_supplied",
        details={
            "step_name": req.step_name,
            "content_type": req.content_type,
            "content_preview": str(req.content)[:200],
            "timestamp": time.time(),
        },
    )

    # Fire-and-forget resume with evidence as external_input
    external_input = {
        "evidence": {
            "step_name": req.step_name,
            "content": req.content,
            "content_type": req.content_type,
        }
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        lambda: _run_evidence_resume(coord, instance_id, external_input),
    )

    return {
        "instance_id": instance_id,
        "step_name": req.step_name,
        "content_type": req.content_type,
        "status": "processing",
    }


def _run_evidence_resume(coord, instance_id: str, external_input: dict):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.resume(instance_id=instance_id, external_input=external_input)
    except Exception as exc:
        print(f"[hitl] Evidence resume error ({instance_id}): {exc}", file=sys.stderr, flush=True)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


# ─── GET /instances/{id}/stream  (SSE) ──────────────────────────────

# Map coordinator action_type values to SSE event names
_ACTION_TO_EVENT = {
    "step_started":           "step_started",
    "step_completed":         "step_completed",
    "step_failed":            "step_failed",
    "governance_evaluated":   "governance_decision",
    "governance_suspension":  "hitl_requested",
    "human_decision":         "hitl_resolved",
    "evidence_supplied":      "hitl_resolved",
    "workflow_completed":     "workflow_completed",
    "workflow_failed":        "workflow_failed",
    "workflow_started":       "step_started",
    "resume":                 "step_started",
}


def _ledger_entry_to_sse(entry: dict, instance_id: str) -> str:
    """Convert a ledger row into an SSE data line."""
    action_type = entry.get("action_type", "")
    details = entry.get("details", {})
    event_name = _ACTION_TO_EVENT.get(action_type, action_type)

    payload = {
        "event": event_name,
        "instance_id": instance_id,
        "ledger_id": entry.get("id"),
        "action_type": action_type,
        "step_name": details.get("step_name", details.get("step", "")),
        "primitive": details.get("primitive", ""),
        "output": details.get("output", details.get("result", {})),
        "elapsed_ms": details.get("elapsed_ms", 0),
        "timestamp": entry.get("created_at", time.time()),
    }
    # Merge any extra details the LLM steps store
    for key in ("decision", "rationale", "reviewer_id", "reason", "governance_tier"):
        if key in details:
            payload[key] = details[key]

    data = json.dumps(payload, default=str)
    return f"event: {event_name}\ndata: {data}\n\n"


@router.get("/instances/{instance_id}/stream")
async def stream_instance(instance_id: str, request: Request, since_id: int = 0):
    """
    Server-Sent Events stream for a workflow instance's action ledger.

    Events: step_started, step_completed, governance_decision,
            hitl_requested, hitl_resolved, workflow_completed, workflow_failed

    For completed instances: replays from the ledger (use since_id=0).
    For live instances: streams in real time, polls every 0.5s.

    Query params:
      since_id: only return entries with ledger id > since_id (default 0)
    """
    coord = deps.get_coordinator()

    # Validate instance exists
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    async def event_generator():
        last_id = since_id
        terminal_sent = False

        # Send a connection-established comment
        yield f": connected to {instance_id}\n\n"

        while True:
            # Check for client disconnect
            if await request.is_disconnected():
                break

            # Fetch new ledger entries since last seen
            try:
                all_entries = coord.store.get_ledger(instance_id=instance_id)
                new_entries = [e for e in all_entries if e.get("id", 0) > last_id]
            except Exception:
                new_entries = []

            for entry in new_entries:
                eid = entry.get("id", 0)
                if eid > last_id:
                    last_id = eid
                yield _ledger_entry_to_sse(entry, instance_id)

            # Check terminal state
            if not terminal_sent:
                inst = coord.store.get_instance(instance_id)
                if inst and inst.status.value in ("completed", "failed", "terminated"):
                    terminal_event = "workflow_completed" if inst.status.value == "completed" else "workflow_failed"
                    payload = json.dumps({
                        "event": terminal_event,
                        "instance_id": instance_id,
                        "status": inst.status.value,
                        "elapsed_seconds": inst.elapsed_seconds,
                        "step_count": inst.step_count,
                        "result": inst.result,
                        "error": inst.error,
                        "timestamp": time.time(),
                    }, default=str)
                    yield f"event: {terminal_event}\ndata: {payload}\n\n"
                    terminal_sent = True
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── GET /instances/{id}/verify  (Sprint 4.1) ────────────────────

@router.get("/instances/{instance_id}/verify")
async def verify_ledger(instance_id: str) -> dict:
    """
    Verify the hash chain integrity of the action ledger for an instance.

    Returns:
      { valid: bool, entry_count: int, first_invalid_entry: int | None }

    An unmodified ledger returns valid=True.
    Any tampering causes valid=False and identifies the first corrupted entry.
    """
    coord = deps.get_coordinator()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    result = coord.store.verify_ledger(instance_id)
    result["instance_id"] = instance_id
    return result
