"""
Instance chain routes.

GET  /api/instances/{instance_id}/chain     — full delegation chain for a case
POST /api/instances/{instance_id}/terminate — terminate an instance
POST /api/instances/{instance_id}/resume    — resume a suspended instance
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import cognitive_core.api.deps as deps

router = APIRouter()


class TerminateRequest(BaseModel):
    reason: str = ""


class ResumeRequest(BaseModel):
    external_input: dict[str, Any] = {}
    resume_nonce: str = ""


@router.get("/instances/{instance_id}/chain")
async def get_chain(instance_id: str) -> list[dict[str, Any]]:
    """
    Return the full delegation chain for a case, ordered oldest-first.

    Each entry is an enriched InstanceState with:
      - result       (step summaries + full case input for completed instances)
      - work_orders  (delegation work orders sent by this instance)
      - pending_task (pending HITL task for this instance, if any)
      - case_meta    (key case facts extracted at start time)
    """
    coord = deps.get_coordinator()

    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")

    correlation_id = instance.correlation_id
    chain_instances = coord.get_correlation_chain(correlation_id)
    chain_instances = list(reversed(chain_instances))

    try:
        pending = coord.list_pending_approvals()
        pending_by_instance = {p["instance_id"]: p for p in pending}
    except Exception:
        pending_by_instance: dict[str, Any] = {}

    result = []
    for inst in chain_instances:
        try:
            work_orders = [
                {
                    "work_order_id": wo.work_order_id,
                    "contract_name": wo.contract_name,
                    "handler_workflow_type": wo.handler_workflow_type,
                    "handler_domain": wo.handler_domain,
                    "handler_instance_id": wo.handler_instance_id,
                    "status": wo.status.value,
                    "mode": "wait_for_result" if wo.handler_instance_id else "fire_and_forget",
                    "created_at": wo.created_at,
                    "completed_at": wo.completed_at,
                }
                for wo in coord.get_work_orders(inst.instance_id)
            ]
        except Exception:
            work_orders = []

        # For suspended instances, checkpoint returns the full state snapshot
        # (which includes the case input under "input" key)
        case_input_data: dict[str, Any] = {}
        if inst.status.value == "suspended":
            try:
                snap = coord.checkpoint(inst.instance_id)
                case_input_data = snap.get("input", {})
            except Exception:
                pass
        elif inst.result and "input" in inst.result:
            case_input_data = inst.result.get("input", {})

        entry: dict[str, Any] = {
            "instance_id": inst.instance_id,
            "workflow_type": inst.workflow_type,
            "domain": inst.domain,
            "status": inst.status.value,
            "governance_tier": inst.governance_tier,
            "correlation_id": inst.correlation_id,
            "lineage": inst.lineage,
            "created_at": inst.created_at,
            "updated_at": inst.updated_at,
            "elapsed_seconds": inst.elapsed_seconds,
            "step_count": inst.step_count,
            "current_step": inst.current_step,
            "result": inst.result,
            "error": inst.error,
            "case_meta": inst.case_meta,
            "case_input": case_input_data,
            "work_orders": work_orders,
            "pending_task": pending_by_instance.get(inst.instance_id),
        }
        result.append(entry)

    return result


@router.post("/instances/{instance_id}/terminate")
async def terminate_instance(instance_id: str, req: TerminateRequest) -> dict[str, Any]:
    """Terminate a workflow instance immediately."""
    coord = deps.get_coordinator()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")
    try:
        coord.terminate(instance_id=instance_id, reason=req.reason or "Terminated by analyst")
    except Exception as exc:
        raise HTTPException(500, f"Terminate failed: {exc}")
    return {"instance_id": instance_id, "status": "terminated"}


@router.post("/instances/{instance_id}/resume")
async def resume_instance(instance_id: str, req: ResumeRequest) -> dict[str, Any]:
    """Resume a suspended workflow instance with optional external input."""
    coord = deps.get_coordinator()
    executor = deps.get_executor()
    instance = coord.store.get_instance(instance_id)
    if not instance:
        raise HTTPException(404, f"Instance not found: {instance_id}")
    if instance.status.value != "suspended":
        raise HTTPException(400, f"Instance is {instance.status.value}, not suspended")

    loop = asyncio.get_event_loop()
    # Fire-and-forget: resume may trigger more LLM work
    loop.run_in_executor(
        executor,
        lambda: _run_resume(coord, instance_id, req.external_input, req.resume_nonce),
    )
    return {"instance_id": instance_id, "status": "processing"}


def _run_resume(coord, instance_id: str, external_input: dict, resume_nonce: str):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.resume(
            instance_id=instance_id,
            external_input=external_input,
            resume_nonce=resume_nonce,
        )
    except Exception as exc:
        print(f"[api] Resume error ({instance_id}): {exc}", file=sys.stderr, flush=True)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
