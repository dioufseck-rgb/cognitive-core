"""
Case management routes.

GET  /api/backlog           — list all root workflow instances (one per case)
GET  /api/cases             — list available case files
POST /api/run               — submit a new case for processing
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import cognitive_core.api.deps as deps

router = APIRouter()

# Ordered list of available case files shown in the Submit modal
AVAILABLE_CASES = [
    "demos/fraud-operations/cases/app_scam_romance.json",
    "demos/fraud-operations/cases/app_scam_investment_crypto.json",
    "demos/fraud-operations/cases/app_scam_govt_impersonation.json",
    "demos/fraud-operations/cases/card_fraud_cnp.json",
    "demos/fraud-operations/cases/card_fraud_atm_skimmer.json",
    "demos/fraud-operations/cases/card_fraud_friendly_fraud.json",
    "demos/fraud-operations/cases/check_fraud_duplicate.json",
    "demos/fraud-operations/cases/check_fraud_altered_amount.json",
    "demos/fraud-operations/cases/check_fraud_counterfeit_payroll.json",
]

WORKFLOW_OPTIONS = [
    {"workflow": "fraud_triage", "domain": "fraud_triage", "label": "Fraud Triage (auto-route)"},
]


class RunRequest(BaseModel):
    case_file: str
    workflow: str = "fraud_triage"
    domain: str = "fraud_triage"


def _compute_stage(chain: list, pending_instance_ids: set | None = None) -> str:
    """Determine the current processing stage for a case from its full chain."""
    # Suspended with a pending governance gate = needs analyst review
    if any(i.status.value == "suspended" for i in chain):
        if pending_instance_ids and any(
            i.status.value == "suspended" and i.instance_id in pending_instance_ids
            for i in chain
        ):
            return "needs_review"
        # Suspended but no governance gate — waiting for delegations
        return "investigation"
    # Any instance actively running
    active = [i for i in chain if i.status.value in ("running", "created")]
    if active:
        running_workflows = [i.workflow_type for i in active]
        if any("specialty_investigation" in w or "regulatory" in w or "resolution" in w
               for w in running_workflows):
            return "investigation"
        return "triage"
    # All terminal — only "completed" if the root finished successfully
    root = chain[0] if chain else None
    if root and root.status.value == "completed":
        return "completed"
    return "failed"


@router.get("/backlog")
async def get_backlog() -> list[dict[str, Any]]:
    """
    Return one entry per unique case (correlation chain), enriched with:
      - stage: triage | investigation | needs_review | completed
      - case_meta: key case facts (member name, fraud type, amount, etc.)
    """
    coord = deps.get_coordinator()

    all_instances = coord.store.list_instances(limit=500)

    # Group by correlation_id
    from collections import defaultdict
    by_corr: dict[str, list] = defaultdict(list)
    for inst in all_instances:
        by_corr[inst.correlation_id].append(inst)

    # Root instances: no lineage, used as the case anchor
    root_instances = [i for i in all_instances if not i.lineage]

    try:
        pending = coord.list_pending_approvals()
        pending_corr_ids = {p["correlation_id"] for p in pending}
        pending_instance_ids = {p["instance_id"] for p in pending}
        pending_task_by_corr = {p["correlation_id"]: p for p in pending}
    except Exception:
        pending_corr_ids: set[str] = set()
        pending_instance_ids: set[str] = set()
        pending_task_by_corr: dict = {}

    result = []
    for inst in root_instances:
        chain = by_corr.get(inst.correlation_id, [inst])
        stage = _compute_stage(chain, pending_instance_ids)
        has_gate = inst.correlation_id in pending_corr_ids
        pending_task = pending_task_by_corr.get(inst.correlation_id)

        result.append({
            "instance_id": inst.instance_id,
            "workflow_type": inst.workflow_type,
            "domain": inst.domain,
            "status": inst.status.value,
            "governance_tier": inst.governance_tier,
            "correlation_id": inst.correlation_id,
            "created_at": inst.created_at,
            "elapsed_seconds": inst.elapsed_seconds,
            "step_count": inst.step_count,
            "has_pending_gate": has_gate,
            "pending_task": pending_task,
            "stage": stage,
            "case_meta": inst.case_meta,
        })
    return result


@router.get("/cases")
async def list_cases() -> list[dict[str, str]]:
    """List available case files for the Submit modal."""
    result = []
    for path in AVAILABLE_CASES:
        p = Path(path)
        if not p.exists():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            case_id = data.get("case_id", p.stem)
            description = data.get("description", "")
            fraud_type = data.get("fraud_type", "unknown")
        except Exception:
            case_id = p.stem
            description = ""
            fraud_type = "unknown"
        result.append({
            "path": path,
            "case_id": case_id,
            "fraud_type": fraud_type,
            "description": description,
        })
    return result


@router.post("/run")
async def run_case(req: RunRequest) -> dict[str, Any]:
    """
    Submit a new case for workflow processing.

    Validates the case file, pre-allocates a correlation_id, submits
    coordinator.start() to the thread pool (non-blocking), then polls
    the DB briefly to obtain the root instance_id before returning.

    The caller should then poll GET /api/instances/{instance_id}/chain
    to observe progression.
    """
    coord = deps.get_coordinator()
    executor = deps.get_executor()

    case_path = Path(req.case_file)
    if not case_path.exists():
        raise HTTPException(400, f"Case file not found: {req.case_file}")

    with open(case_path) as f:
        case_input = json.load(f)

    # Pre-allocate correlation_id so we can find the root instance before
    # start() returns (start() may block for minutes during LLM calls).
    # The coordinator passes this correlation_id to InstanceState.create(),
    # which saves it to the DB early — before any LLM calls.
    correlation_id = f"corr_{uuid.uuid4().hex[:12]}"

    # Submit to thread pool — non-blocking
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        lambda: _run_workflow(coord, req.workflow, req.domain, case_input, correlation_id),
    )

    # Poll the DB for the root instance (created at the very start of start())
    instance_id: str | None = None
    for _ in range(30):  # up to 6 seconds
        await asyncio.sleep(0.2)
        instances = coord.store.list_instances(correlation_id=correlation_id)
        roots = [i for i in instances if not i.lineage]
        if roots:
            instance_id = roots[0].instance_id
            break

    if not instance_id:
        raise HTTPException(500, "Workflow did not appear in DB within 6 seconds")

    return {"instance_id": instance_id, "correlation_id": correlation_id}


@router.post("/reset")
async def reset_all() -> dict:
    """
    Clear all workflow instances from the database.
    Terminates any running instances first, then wipes the tables.
    """
    coord = deps.get_coordinator()
    try:
        all_instances = coord.store.list_instances(limit=2000)
        for inst in all_instances:
            if inst.status.value in ("running", "created", "suspended"):
                try:
                    coord.terminate(instance_id=inst.instance_id, reason="Manual reset")
                except Exception:
                    pass
        for table in ("action_ledger", "suspensions", "work_orders", "instances", "task_queue"):
            coord.store.db.execute(f"DELETE FROM {table}")
        coord.store.db.commit()
        return {"status": "ok", "cleared": len(all_instances)}
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {e}")


def _run_workflow(coord, workflow: str, domain: str, case_input: dict, correlation_id: str):
    """
    Execute a workflow synchronously (runs in thread pool worker).

    Thread pool workers have no asyncio event loop.  LangGraph / MCP internals
    call asyncio.get_event_loop(), so we create a dedicated loop for this thread.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        coord.start(
            workflow_type=workflow,
            domain=domain,
            case_input=case_input,
            correlation_id=correlation_id,
        )
    except Exception as exc:
        print(f"[api] Workflow error ({workflow}/{domain}): {exc}", file=sys.stderr, flush=True)
    finally:
        loop.close()
        asyncio.set_event_loop(None)
