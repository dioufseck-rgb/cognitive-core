"""
Cognitive Core — API Server (S-001)

FastAPI application serving:
  POST /v1/cases              — submit a case for async execution
  GET  /v1/cases/{id}         — poll case status + result
  GET  /v1/cases/{id}/trail   — audit trail for a case
  GET  /v1/approvals          — list pending governance approvals
  POST /v1/approvals/{id}/approve — approve a suspended instance
  POST /v1/approvals/{id}/reject  — reject a suspended instance
  GET  /health                — liveness
  GET  /ready                 — readiness
  GET  /startup               — startup checks
  GET  /v1/stats              — coordinator statistics

Usage:
    uvicorn api.server:app --host 0.0.0.0 --port 8080

    # Development (no Redis)
    CC_WORKER_MODE=inline uvicorn api.server:app --reload

Requires: pip install fastapi uvicorn
Optional: pip install arq redis (for production worker)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("cognitive_core.api")


def create_app(project_root: str = ".") -> Any:
    """
    Create and configure the FastAPI application.

    Returns the app instance. Separated from module-level creation
    so tests can create fresh instances.
    """
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    from api.models import (
        CaseSubmission, CaseResponse, CaseStatus,
        ApprovalAction, ApprovalEntry,
    )
    from api.worker import create_backend, WorkerBackend
    from coordinator.runtime import Coordinator
    from coordinator.types import InstanceStatus

    app = FastAPI(
        title="Cognitive Core API",
        version="0.1.0",
        description="Composable AI Workflow Engine",
    )

    # ── State ────────────────────────────────────────────────

    _backend: WorkerBackend | None = None
    _coordinator: Coordinator | None = None

    def get_backend() -> WorkerBackend:
        nonlocal _backend
        if _backend is None:
            _backend = create_backend(project_root=project_root)
        return _backend

    def get_coordinator() -> Coordinator:
        nonlocal _coordinator
        if _coordinator is None:
            _coordinator = Coordinator(project_root=project_root, verbose=False)
        return _coordinator

    # ── Lifecycle ─────────────────────────────────────────────

    @app.on_event("shutdown")
    async def shutdown():
        if _backend:
            _backend.shutdown()

    # ── Case Submission ───────────────────────────────────────

    @app.post("/v1/cases", response_model=None)
    async def submit_case(request: Request):
        body = await request.json()

        submission = CaseSubmission(
            workflow=body.get("workflow", ""),
            domain=body.get("domain", ""),
            case_input=body.get("case_input", {}),
            model=body.get("model", "default"),
            temperature=body.get("temperature", 0.1),
            correlation_id=body.get("correlation_id", ""),
        )

        errors = submission.validate()
        if errors:
            return JSONResponse(
                status_code=422,
                content={"errors": errors},
            )

        # Generate instance ID and correlation ID upfront
        instance_id = f"wf_{uuid.uuid4().hex[:12]}"
        correlation_id = submission.correlation_id or instance_id

        # Enqueue for async execution
        backend = get_backend()
        job_id = backend.enqueue(
            instance_id=instance_id,
            workflow=submission.workflow,
            domain=submission.domain,
            case_input=submission.case_input,
            model=submission.model,
            temperature=submission.temperature,
            correlation_id=correlation_id,
        )

        response = CaseResponse(
            instance_id=instance_id,
            correlation_id=correlation_id,
            status="accepted",
            message=f"Case enqueued for processing (job: {job_id})",
        )
        return JSONResponse(status_code=202, content=response.to_dict())

    # ── Case Status ───────────────────────────────────────────

    @app.get("/v1/cases/{instance_id}")
    async def get_case_status(instance_id: str):
        coord = get_coordinator()
        inst = coord.get_instance(instance_id)

        if inst is None:
            # Check if it's a job that hasn't started yet
            backend = get_backend()
            job = backend.tracker.get_by_instance(instance_id)
            if job and job.status == "queued":
                return JSONResponse(content={
                    "instance_id": instance_id,
                    "status": "queued",
                    "message": "Case is queued for processing",
                })
            raise HTTPException(status_code=404, detail="Instance not found")

        status = CaseStatus(
            instance_id=inst.instance_id,
            workflow_type=inst.workflow_type,
            domain=inst.domain,
            status=inst.status.value if hasattr(inst.status, 'value') else str(inst.status),
            governance_tier=inst.governance_tier,
            correlation_id=inst.correlation_id,
            created_at=inst.created_at,
            updated_at=inst.updated_at,
            current_step=inst.current_step,
            step_count=inst.step_count,
            elapsed_seconds=inst.elapsed_seconds,
            result=inst.result,
            error=inst.error,
        )
        return JSONResponse(content=status.to_dict())

    # ── Audit Trail ───────────────────────────────────────────

    @app.get("/v1/cases/{instance_id}/trail")
    async def get_case_trail(instance_id: str):
        coord = get_coordinator()
        inst = coord.get_instance(instance_id)
        if inst is None:
            raise HTTPException(status_code=404, detail="Instance not found")

        ledger = coord.get_ledger(instance_id=instance_id)
        chain = coord.get_correlation_chain(inst.correlation_id)

        return JSONResponse(content={
            "instance_id": instance_id,
            "correlation_id": inst.correlation_id,
            "ledger_entries": len(ledger),
            "chain_length": len(chain),
            "ledger": ledger[:100],  # Cap at 100 entries
            "chain": [
                {
                    "instance_id": c.instance_id,
                    "workflow_type": c.workflow_type,
                    "domain": c.domain,
                    "status": c.status.value if hasattr(c.status, 'value') else str(c.status),
                }
                for c in chain
            ],
        })

    # ── Approvals ─────────────────────────────────────────────

    @app.get("/v1/approvals")
    async def list_approvals():
        coord = get_coordinator()
        pending = coord.list_pending_approvals()
        return JSONResponse(content={
            "count": len(pending),
            "approvals": pending,
        })

    @app.post("/v1/approvals/{instance_id}/approve")
    async def approve_instance(instance_id: str, request: Request):
        body = await request.json()
        action = ApprovalAction(
            approver=body.get("approver", ""),
            reason=body.get("reason", ""),
        )
        errors = action.validate()
        if errors:
            return JSONResponse(status_code=422, content={"errors": errors})

        coord = get_coordinator()
        inst = coord.get_instance(instance_id)
        if inst is None:
            raise HTTPException(status_code=404, detail="Instance not found")

        try:
            coord.approve(instance_id, approver=action.approver)
            return JSONResponse(content={
                "instance_id": instance_id,
                "action": "approved",
                "approver": action.approver,
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/v1/approvals/{instance_id}/reject")
    async def reject_instance(instance_id: str, request: Request):
        body = await request.json()
        action = ApprovalAction(
            approver=body.get("approver", ""),
            reason=body.get("reason", ""),
        )
        errors = action.validate()
        if errors:
            return JSONResponse(status_code=422, content={"errors": errors})

        coord = get_coordinator()
        inst = coord.get_instance(instance_id)
        if inst is None:
            raise HTTPException(status_code=404, detail="Instance not found")

        try:
            coord.reject(instance_id, reason=action.reason)
            return JSONResponse(content={
                "instance_id": instance_id,
                "action": "rejected",
                "reason": action.reason,
            })
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ── Stats ─────────────────────────────────────────────────

    @app.get("/v1/stats")
    async def get_stats():
        coord = get_coordinator()
        stats = coord.stats()
        backend = get_backend()
        stats["worker"] = backend.tracker.stats
        return JSONResponse(content=stats)

    # ── Health ────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return JSONResponse(content={
            "status": "ok",
            "timestamp": time.time(),
        })

    @app.get("/ready")
    async def ready():
        # Check coordinator store is accessible
        try:
            coord = get_coordinator()
            coord.stats()
            return JSONResponse(content={"status": "ok"})
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"status": "fail", "error": str(e)[:200]},
            )

    @app.get("/startup")
    async def startup():
        # Validate configs
        checks = {}
        try:
            from engine.validate import validate_all
            errors, warnings = validate_all(project_root)
            checks["spec_validation"] = {
                "status": "ok" if not errors else "fail",
                "errors": len(errors),
                "warnings": len(warnings),
            }
        except Exception as e:
            checks["spec_validation"] = {"status": "fail", "error": str(e)[:200]}

        overall = "ok" if all(c["status"] == "ok" for c in checks.values()) else "fail"
        code = 200 if overall == "ok" else 503
        return JSONResponse(status_code=code, content={"status": overall, "checks": checks})

    return app


# ── Module-level app for uvicorn ──────────────────────────────

try:
    app = create_app(project_root=os.environ.get("CC_PROJECT_ROOT", "."))
except ImportError:
    # FastAPI not installed — app creation deferred
    app = None
