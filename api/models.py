"""
Cognitive Core — API Models

Request/response dataclasses for the API server.
No FastAPI dependency — used by server, worker, and tests.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


class JobStatus(str, enum.Enum):
    """Status of an async job in the worker queue."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CaseSubmission:
    """POST /v1/cases request body."""
    workflow: str
    domain: str
    case_input: dict[str, Any]
    model: str = "default"
    temperature: float = 0.1
    correlation_id: str = ""

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if not self.workflow or not isinstance(self.workflow, str):
            errors.append("workflow is required and must be a string")
        if not self.domain or not isinstance(self.domain, str):
            errors.append("domain is required and must be a string")
        if not isinstance(self.case_input, dict):
            errors.append("case_input is required and must be an object")
        if self.temperature < 0 or self.temperature > 2.0:
            errors.append("temperature must be between 0 and 2.0")
        return errors


@dataclass
class CaseResponse:
    """POST /v1/cases response — returned immediately on submission."""
    instance_id: str
    correlation_id: str
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseStatus:
    """GET /v1/cases/{id} response — full instance status."""
    instance_id: str
    workflow_type: str
    domain: str
    status: str
    governance_tier: str
    correlation_id: str
    created_at: float
    updated_at: float
    current_step: str
    step_count: int
    elapsed_seconds: float
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ApprovalAction:
    """POST /v1/approvals/{id}/approve or /reject body."""
    approver: str
    reason: str = ""

    def validate(self) -> list[str]:
        errors = []
        if not self.approver or not isinstance(self.approver, str):
            errors.append("approver is required and must be a string")
        return errors


@dataclass
class ApprovalEntry:
    """GET /v1/approvals response item."""
    task_id: str
    instance_id: str
    workflow_type: str
    domain: str
    governance_tier: str
    correlation_id: str
    queue: str
    created_at: float
    sla_seconds: float
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
