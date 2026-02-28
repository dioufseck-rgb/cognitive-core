"""
Cognitive Core — Coordinator Type Definitions

All data structures for instance management, work orders,
governance tiers, delegation policies, and contracts.
"""

from __future__ import annotations

import enum
import uuid
import time
from dataclasses import dataclass, field
from typing import Any


# ─── Instance Management ────────────────────────────────────────────

class InstanceStatus(str, enum.Enum):
    """Lifecycle states for a workflow instance."""
    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class InstanceState:
    """
    Registry entry for a workflow instance.
    The coordinator tracks these; workflow state snapshots
    are stored separately.
    """
    instance_id: str
    workflow_type: str
    domain: str
    status: InstanceStatus
    created_at: float
    updated_at: float

    # Lineage: chain of instance IDs that led to this one
    lineage: list[str] = field(default_factory=list)
    # Correlation: root instance ID linking all related work
    correlation_id: str = ""

    # Governance tier (resolved from domain config)
    governance_tier: str = "gate"  # safe default

    # Execution tracking
    current_step: str = ""
    step_count: int = 0
    elapsed_seconds: float = 0.0

    # Suspension / delegation
    pending_work_orders: list[str] = field(default_factory=list)
    resume_nonce: str = ""

    # Result (populated on completion)
    result: dict[str, Any] | None = None
    error: str | None = None

    @staticmethod
    def create(
        workflow_type: str,
        domain: str,
        governance_tier: str = "gate",
        lineage: list[str] | None = None,
        correlation_id: str = "",
    ) -> InstanceState:
        now = time.time()
        iid = f"wf_{uuid.uuid4().hex[:12]}"
        cid = correlation_id or iid
        return InstanceState(
            instance_id=iid,
            workflow_type=workflow_type,
            domain=domain,
            status=InstanceStatus.CREATED,
            created_at=now,
            updated_at=now,
            lineage=lineage or [],
            correlation_id=cid,
            governance_tier=governance_tier,
        )


# ─── Work Orders (A2A Delegation) ───────────────────────────────────

class WorkOrderStatus(str, enum.Enum):
    """Lifecycle states for a work order."""
    CREATED = "created"
    QUEUED = "queued"          # waiting for resource capacity (backpressure)
    DISPATCHED = "dispatched"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class WorkOrder:
    """
    A delegation request routed through the coordinator.
    Created when a delegation policy fires or a need is matched.
    """
    work_order_id: str
    # Who requested this
    requester_instance_id: str
    correlation_id: str

    # What to do
    contract_name: str
    contract_version: int
    inputs: dict[str, Any]

    # Constraints
    sla_seconds: float | None = None
    urgency: str = "routine"  # routine, elevated, critical

    # Routing (filled by coordinator)
    handler_workflow_type: str = ""
    handler_domain: str = ""
    handler_instance_id: str = ""

    # Lifecycle
    status: WorkOrderStatus = WorkOrderStatus.CREATED
    created_at: float = 0.0
    dispatched_at: float | None = None
    completed_at: float | None = None

    # Result (populated on completion)
    result: WorkOrderResult | None = None

    @staticmethod
    def create(
        requester_instance_id: str,
        correlation_id: str,
        contract_name: str,
        contract_version: int = 1,
        inputs: dict[str, Any] | None = None,
        sla_seconds: float | None = None,
        urgency: str = "routine",
    ) -> WorkOrder:
        return WorkOrder(
            work_order_id=f"wo_{uuid.uuid4().hex[:12]}",
            requester_instance_id=requester_instance_id,
            correlation_id=correlation_id,
            contract_name=contract_name,
            contract_version=contract_version,
            inputs=inputs or {},
            sla_seconds=sla_seconds,
            urgency=urgency,
            created_at=time.time(),
        )


@dataclass
class WorkOrderResult:
    """Result of a completed work order."""
    work_order_id: str
    status: str  # "completed", "failed", "expired"
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: float = 0.0


# ─── Suspension / Resumption ─────────────────────────────────────────

@dataclass
class Suspension:
    """
    Everything needed to resume a suspended workflow instance.
    Created when an instance can't proceed without external input.
    """
    instance_id: str
    suspended_at_step: str
    state_snapshot: dict[str, Any]  # serialized WorkflowState
    unresolved_needs: list[dict[str, Any]] = field(default_factory=list)
    work_order_ids: list[str] = field(default_factory=list)
    deferred_needs: list[dict[str, Any]] = field(default_factory=list)
    wo_need_map: dict[str, str] = field(default_factory=dict)  # wo_id → need_name
    resume_nonce: str = ""
    suspended_at: float = 0.0

    @staticmethod
    def create(
        instance_id: str,
        suspended_at_step: str,
        state_snapshot: dict[str, Any],
        unresolved_needs: list[dict[str, Any]] | None = None,
        work_order_ids: list[str] | None = None,
    ) -> Suspension:
        return Suspension(
            instance_id=instance_id,
            suspended_at_step=suspended_at_step,
            state_snapshot=state_snapshot,
            unresolved_needs=unresolved_needs or [],
            work_order_ids=work_order_ids or [],
            resume_nonce=uuid.uuid4().hex[:16],
            suspended_at=time.time(),
        )


# ─── Governance Tiers ────────────────────────────────────────────────

class GovernanceTier(str, enum.Enum):
    """Risk-based governance posture, declared per domain."""
    AUTO = "auto"           # Low risk, fully autonomous
    SPOT_CHECK = "spot_check"  # Medium risk, sampled review
    GATE = "gate"           # High risk, mandatory pre-action review
    HOLD = "hold"           # Regulatory, mandatory expert sign-off


@dataclass
class GovernanceTierConfig:
    """Configuration for a governance tier."""
    tier: GovernanceTier
    hitl: str = "none"        # none, post_completion, before_act, before_finalize
    sample_rate: float = 0.0  # 0.0 = no sampling, 0.1 = 10%
    queue: str = ""           # human task queue name
    sla_seconds: float = 0.0  # SLA for human review


# ─── Delegation Policies ─────────────────────────────────────────────

@dataclass
class DelegationCondition:
    """
    When to trigger a delegation.
    Evaluated against workflow output using primitive-type selectors.
    """
    domain: str                          # source domain
    selector: str = "any_investigate"    # primitive-type selector
    field: str = ""                      # dot-path into step output
    operator: str = "exists"             # exists, eq, gte, contains_any
    value: Any = None                    # comparison value


@dataclass
class DelegationPolicy:
    """
    A rule that triggers cross-workflow delegation.
    Evaluated by the coordinator against completed workflow output.
    """
    name: str
    conditions: list[DelegationCondition]
    # What to spawn
    target_workflow: str
    target_domain: str
    contract_name: str
    contract_version: int = 1
    sla_seconds: float | None = None
    # Delegation mode
    #   fire_and_forget: source completes, delegated workflow runs independently
    #   wait_for_result: source suspends, resumes when delegated workflow completes
    mode: str = "fire_and_forget"
    # For wait_for_result: which step to resume at (default: re-run last step with enriched data)
    resume_at_step: str = ""
    # Input mapping: keys are target input fields,
    # values are ${source.xxx} references resolved against source output
    input_mapping: dict[str, str] = field(default_factory=dict)


# ─── Contracts ───────────────────────────────────────────────────────

@dataclass
class ContractField:
    """A field in a contract schema."""
    name: str
    type: str  # string, int, float, bool, object, list, enum
    required: bool = True
    enum_values: list[str] | None = None


@dataclass
class Contract:
    """
    Versioned interface schema for cross-workflow delegation.
    Neither sender nor receiver knows the other's identity.
    """
    name: str
    version: int
    request_fields: list[ContractField] = field(default_factory=list)
    response_fields: list[ContractField] = field(default_factory=list)

    def validate_request(self, inputs: dict[str, Any]) -> list[str]:
        """Validate inputs against contract request schema."""
        errors = []
        for f in self.request_fields:
            if f.required and f.name not in inputs:
                errors.append(f"Missing required field: {f.name}")
            if f.name in inputs and f.enum_values:
                if inputs[f.name] not in f.enum_values:
                    errors.append(
                        f"Field {f.name}: '{inputs[f.name]}' not in {f.enum_values}"
                    )
        return errors

    def validate_response(self, outputs: dict[str, Any]) -> list[str]:
        """Validate outputs against contract response schema."""
        errors = []
        for f in self.response_fields:
            if f.required and f.name not in outputs:
                errors.append(f"Missing required field: {f.name}")
        return errors


# ─── Capability Catalog ──────────────────────────────────────────────

@dataclass
class Capability:
    """
    Maps a need type to a provider (workflow, human queue, or service).
    """
    need_type: str
    provider_type: str  # "workflow", "human_task", "external_service"
    # For workflow providers
    workflow_type: str = ""
    domain: str = ""
    contract_name: str = ""
    # For human task providers
    queue: str = ""
    # For external services
    endpoint: str = ""
