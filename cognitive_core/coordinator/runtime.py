"""
Cognitive Core — Runtime Coordinator

The DEVS kernel. Manages workflow instance lifecycle, evaluates
governance policies, routes delegations, and handles suspension/resumption.

Phase 1: In-process. All workflows execute in the same process.
The coordinator interface (start/resume/checkpoint/terminate) is
topology-independent — same contract in-process or distributed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Any

from cognitive_core.coordinator.types import (
    InstanceState,
    InstanceStatus,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderResult,
    Suspension,
)
from cognitive_core.coordinator.store import CoordinatorStore
from cognitive_core.coordinator.policy import PolicyEngine, load_policy_engine, GovernanceDecision, DelegationDecision
from cognitive_core.coordinator.tasks import (
    TaskQueue, SQLiteTaskQueue, InMemoryTaskQueue,
    Task, TaskType, TaskStatus, TaskCallback, TaskResolution,
    create_task_queue,
)

# Dispatch optimization (optional — graceful degradation if not configured)
try:
    from cognitive_core.coordinator.optimizer import DispatchOptimizer
    from cognitive_core.coordinator.physics import OptimizationConfig, parse_optimization_config
    from cognitive_core.coordinator.ddd import ResourceRegistry
    _OPTIMIZER_AVAILABLE = True
except ImportError:
    _OPTIMIZER_AVAILABLE = False

# Resilience layer (optional — graceful degradation)
try:
    from cognitive_core.coordinator.resilience import (
        ResumeRevalidator, RevalidationGuard, StalenessVerdict,
        OscillationDetector, OscillationAction,
        CapacityRevocationManager, RevocationConfig,
        SagaCoordinator,
    )
    _RESILIENCE_AVAILABLE = True
except ImportError:
    _RESILIENCE_AVAILABLE = False

# Production hardening layer (optional)
try:
    from cognitive_core.coordinator.hardening import (
        build_ddr, DDREligibilityEntry, DDRCandidateScore,
        PartialFailureHandler, PartialFailurePolicy, FailureAction,
        ReservationEventLog, ReservationOp,
        LearningScopeEnforcer,
    )
    _HARDENING_AVAILABLE = True
except ImportError:
    _HARDENING_AVAILABLE = False

try:
    from cognitive_core.coordinator.federation import FederationSpoke, FederatedCapability
    _FEDERATION_AVAILABLE = True
except ImportError:
    _FEDERATION_AVAILABLE = False


# ─── Reliability Exceptions ──────────────────────────────────────────

class DelegationDepthExceeded(Exception):
    """Raised when delegation chain exceeds MAX_DELEGATION_DEPTH."""
    pass


class StuckInstanceError(Exception):
    """Raised when an instance is detected as stuck."""
    pass


class Coordinator:
    """
    Runtime Coordinator — the fourth architectural layer.

    Manages multi-workflow execution through four operations:
        start(workflow_type, domain, case_input) → instance_id
        resume(instance_id, resume_token, external_input) → result | suspended
        checkpoint(instance_id) → state_snapshot
        terminate(instance_id, reason) → confirmation

    And receives four event types from the workflow engine:
        on_completed(instance_id, result)
        on_suspended(instance_id, suspension)
        on_failed(instance_id, error)
        on_checkpoint(instance_id, state)
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        store: CoordinatorStore | None = None,
        task_queue: TaskQueue | None = None,
        db_path: str | Path = "coordinator.db",
        verbose: bool = True,
        overrides: dict[str, Any] | None = None,
    ):
        # Load coordinator config
        if config:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Apply caller overrides (e.g. absolute paths resolved in run.py)
        if overrides:
            self.config.update(overrides)

        # Build policy engine
        self.policy = load_policy_engine(self.config)

        # State store
        self.store = store or CoordinatorStore(db_path)

        # Task queue (shares DB connection with store for SQLite)
        if task_queue:
            self.tasks = task_queue
        else:
            self.tasks = create_task_queue(db=self.store.db)

        # Workflow/domain resolution paths
        # Resolve relative to config file directory
        if config_path:
            _base = Path(config_path).resolve().parent
        else:
            _base = Path.cwd()
        self.workflow_dir = _base / self.config.get("workflow_dir", "workflows")
        self.domain_dir = _base / self.config.get("domain_dir", "domains")
        self.case_dir = _base / self.config.get("case_dir", "cases")

        # Domain → governance tier mapping (loaded from domain configs)
        self._domain_tiers: dict[str, str] = {}

        # Dispatch optimizer (Spec v1.1 Sections 9, 14, 17.2, 17.4)
        self._optimizer: Any = None
        self._resource_registry: Any = None
        self._optimization_configs: dict[str, Any] = {}  # domain → OptimizationConfig
        self._ddr_log: list[dict[str, Any]] = []  # in-memory DDR log (production: action_ledger)
        if _OPTIMIZER_AVAILABLE:
            self._resource_registry = ResourceRegistry()
            # Build production hooks for optimizer
            ddr_cb = None
            res_log = None
            learn_enf = None
            if _HARDENING_AVAILABLE:
                res_log = ReservationEventLog()
                learn_enf = LearningScopeEnforcer()
                ddr_cb = self._persist_ddr
            self._optimizer = DispatchOptimizer(
                self._resource_registry,
                ddr_callback=ddr_cb,
                reservation_log=res_log,
                learning_enforcer=learn_enf,
            )

        # Resilience layer (four failure mode defenses)
        self._revalidator: Any = None
        self._oscillation_detector: Any = None
        self._revocation_manager: Any = None
        self._saga_coordinator: Any = None
        if _RESILIENCE_AVAILABLE:
            self._revalidator = ResumeRevalidator()
            self._oscillation_detector = OscillationDetector()
            self._revocation_manager = CapacityRevocationManager()
            self._saga_coordinator = SagaCoordinator()

        # Production hardening layer
        self._partial_failure_handler: Any = None
        self._reservation_log: Any = None
        self._learning_enforcer: Any = None
        if _HARDENING_AVAILABLE:
            self._partial_failure_handler = PartialFailureHandler()
            self._reservation_log = ReservationEventLog()
            self._learning_enforcer = LearningScopeEnforcer()

        self.verbose = verbose

        # Cached MCP provider for data tools (avoids spawning a new subprocess per workflow)
        self._mcp_provider: Any = None

        # Cached MCP client for execution tools — initialized from execution_mcp_cmd
        self._execution_mcp_client: Any = None
        # Parsed command info for per-call MCP sessions
        self._execution_mcp_cmd_info: dict | None = None

        # Persistent background event loop for MCP operations.
        # Thread pool workers create/close their own loops; we keep one
        # long-lived loop so the MCP session's asyncio transports are never
        # bound to a closed loop.
        self._mcp_bg_loop = None
        self._mcp_bg_thread = None

        # Resource backpressure queue: work orders waiting for capacity
        # Key: resource_id or capability_key → list of (wo_id, instance_id, req, capability)
        self._resource_wait_queue: dict[str, list[dict[str, Any]]] = {}

        # Delegation dispatch lock — prevents concurrent threads from entering
        # _evaluate_and_execute_delegations for the same instance simultaneously.
        import threading as _th
        # RLock (reentrant): same thread can acquire multiple times.
        # Needed because _execute_delegation → self.start() → _on_completed
        # → _evaluate_and_execute_delegations re-enters this lock from the
        # same thread. Without reentrancy this deadlocks.
        # Cross-thread protection still works: a different thread blocks
        # until the acquiring thread fully releases all its acquisitions.
        self._delegation_lock = _th.RLock()

        # Federation spoke (child coordinator in a federated hierarchy)
        self._federation_spoke: Any = None
        if _FEDERATION_AVAILABLE:
            cid = config.get("coordinator_id", f"coord_{id(self) % 100000}") \
                if isinstance(config, dict) else f"coord_{id(self) % 100000}"
            cname = config.get("coordinator_name", cid) \
                if isinstance(config, dict) else cid
            self._federation_spoke = FederationSpoke(
                child_id=cid,
                child_name=cname,
            )

    # ─── Four Operations ─────────────────────────────────────────────

    # Maximum delegation chain depth. Each delegation adds to lineage.
    # At depth 20, something is wrong — likely a policy loop.
    MAX_DELEGATION_DEPTH = 20

    def start(
        self,
        workflow_type: str,
        domain: str,
        case_input: dict[str, Any],
        lineage: list[str] | None = None,
        correlation_id: str = "",
        model: str = "default",
        temperature: float = 0.1,
        governance_tier_override: str = "",
    ) -> str:
        """
        Create and execute a new workflow instance.
        Returns the instance_id.

        The workflow may complete, be interrupted by a resource request
        (demand-driven delegation), or fail. All three outcomes are
        handled here.

        governance_tier_override: If set, overrides the domain's default
        governance tier. Used by delegation policies to control the
        handler's governance behavior from config.
        """
        # ── C1: Delegation depth guard ──
        effective_lineage = lineage or []
        if len(effective_lineage) >= self.MAX_DELEGATION_DEPTH:
            chain_tail = " → ".join(effective_lineage[-5:])
            raise DelegationDepthExceeded(
                f"Delegation chain depth {len(effective_lineage)} exceeds "
                f"limit {self.MAX_DELEGATION_DEPTH}. "
                f"Last 5: {chain_tail}. "
                f"Check delegation policies for circular triggers."
            )

        # ── C2: Case input validation (Sprint 2.1) ──
        # Validate at the boundary — clean errors here, not deep in execution
        try:
            from cognitive_core.engine.input_validation import validate_case_input
            validate_case_input(case_input)
        except Exception as _ve:
            # Re-raise as ValueError so callers get a clean, typed error
            raise ValueError(str(_ve)) from _ve

        # Resolve governance tier — override takes precedence
        tier = governance_tier_override or self._resolve_governance_tier(domain)
        tier_locked = bool(governance_tier_override)

        # Create instance
        instance = InstanceState.create(
            workflow_type=workflow_type,
            domain=domain,
            governance_tier=tier,
            governance_tier_locked=tier_locked,
            lineage=lineage or [],
            correlation_id=correlation_id,
        )
        instance.status = InstanceStatus.RUNNING
        instance.updated_at = time.time()
        # Extract key case facts for UI display (stable, doesn't change)
        if not lineage:
            ta = case_input.get("get_triggering_activity", {})
            amount = (
                ta.get("total_approved")
                or ta.get("total_amount")
                or sum(p.get("amount", 0) for p in ta.get("payments", []))
            )
            instance.case_meta = {
                "case_id": case_input.get("case_id", ""),
                "fraud_type": case_input.get("fraud_type", ""),
                "risk_score": case_input.get("risk_score"),
                "priority": case_input.get("get_alert", {}).get("priority", ""),
                "sla_deadline": case_input.get("get_alert", {}).get("sla_deadline", ""),
                "member_name": case_input.get("get_member", {}).get("full_name", ""),
                "member_id": case_input.get("get_member", {}).get("member_id", ""),
                "account_masked": case_input.get("get_account", {}).get("account_number_masked", ""),
                "average_balance": case_input.get("get_account", {}).get("average_balance_6mo"),
                "amount_at_risk": amount,
                "description": case_input.get("description", ""),
            }
        self.store.save_instance(instance)

        # Log to action ledger
        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="start",
            details={
                "workflow_type": workflow_type,
                "domain": domain,
                "governance_tier": tier,
                "lineage": instance.lineage,
            },
            idempotency_key=f"start:{instance.instance_id}:{instance.created_at}",
        )

        self._log(f"▶ START {instance.instance_id} "
                   f"({workflow_type}/{domain}) [tier={tier}]")

        # ── Start gates — kill switches, guardrails, eval gate ──
        try:
            from cognitive_core.engine.governance import get_governance
            _gov = get_governance()
            _gate = _gov.check_start_gates(workflow_type, domain, case_input)
            if _gate.get("blocked"):
                self._on_failed(instance, f"Start gate blocked: {_gate.get('reason', '')}")
                raise RuntimeError(f"Start gate blocked: {_gate.get('reason', '')}")
            if _gate.get("tier_escalation"):
                tier = self._resolve_governance_tier_upgrade(
                    tier, _gate["tier_escalation"]
                )
                instance.governance_tier = tier
                self.store.save_instance(instance)
        except RuntimeError:
            raise
        except Exception as _e:
            self._log(f"  ⚠ Start gates check failed (non-blocking): {_e}")

        # ── Spec manifest — capture config hashes at instance start ──
        try:
            from cognitive_core.engine.governance import get_governance
            _gov = get_governance()
            _wf_path = self.workflow_dir / f"{workflow_type}.yaml"
            _dom_path = self.domain_dir / f"{domain}.yaml"
            _manifest = _gov.capture_spec_manifest(
                workflow_path=str(_wf_path),
                domain_path=str(_dom_path),
                coordinator_path="",
            )
            if _manifest:
                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="spec_manifest",
                    details=_manifest,
                    idempotency_key=f"manifest:{instance.instance_id}",
                )
        except Exception as _e:
            self._log(f"  ⚠ Spec manifest capture failed (non-blocking): {_e}")

        # ── Analytics artifacts — load at workflow start, record provenance ──
        try:
            from cognitive_core.engine.governance import get_governance
            _gov = get_governance()
            _dom_cfg = self._load_domain_config(domain)
            _analytics_block = _dom_cfg.get("analytics", {})
            _artifact_names = [
                v["artifact"] for v in _analytics_block.values()
                if isinstance(v, dict) and "artifact" in v
            ]
            if _artifact_names:
                _gov.load_analytics_artifacts(domain, workflow_type, _artifact_names)
        except Exception as _e:
            self._log(f"  ⚠ Analytics artifact loading failed (non-blocking): {_e}")

        # When MCP is configured, strip pre-labeled fields so the LLM
        # cannot see the answer (fraud_type, description, etc.) before
        # it reasons. Child workflows triggered by delegation policies
        # already receive lean inputs (just identifiers).
        _mcp_active = (
            os.environ.get("DATA_MCP_URL", "")
            or os.environ.get("DATA_MCP_CMD", "")
            or self.config.get("data_mcp_cmd", "")
        )
        if _mcp_active and not lineage:
            # Only sanitize the root workflow input; delegation children
            # already receive lean inputs constructed by the policy engine.
            case_input = self._lean_case_input(case_input)
            self._log(f"  🔒 Lean case_input applied (MCP active) — keys: {list(case_input.keys())}")

        # Execute the workflow (may complete or interrupt)
        try:
            result = self._execute_workflow(
                instance, case_input, model, temperature
            )

            if isinstance(result, dict):
                # Workflow completed — normal path
                self._on_completed(instance, result)
            else:
                # Workflow was interrupted by a resource request
                # result is a StepInterrupt from cognitive_core.engine.stepper
                self._on_interrupted(instance, result, model, temperature)

        except Exception as e:
            from cognitive_core.engine.kill_switch import KillSwitchTripped
            if isinstance(e, KillSwitchTripped):
                self._on_kill_switch_suspended(instance, str(e))
            else:
                self._on_failed(instance, str(e))
                raise

        return instance.instance_id

    def resume(
        self,
        instance_id: str,
        external_input: dict[str, Any],
        resume_nonce: str = "",
        model: str = "default",
        temperature: float = 0.1,
    ) -> str:
        """
        Resume a suspended workflow instance with external input.
        Returns the instance_id.
        """
        instance = self.store.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")
        if instance.status != InstanceStatus.SUSPENDED:
            raise ValueError(
                f"Instance {instance_id} is {instance.status.value}, not suspended"
            )

        # Validate resume nonce
        suspension = self.store.get_suspension(instance_id)
        if not suspension:
            raise ValueError(f"No suspension record for {instance_id}")
        if resume_nonce and suspension.resume_nonce != resume_nonce:
            raise ValueError("Resume nonce mismatch")

        self._log(f"▶ RESUME {instance_id} at step '{suspension.suspended_at_step}'")

        # Inject external input into the state snapshot
        state_snapshot = suspension.state_snapshot
        if "delegation_results" not in state_snapshot:
            state_snapshot["delegation_results"] = {}
        state_snapshot["delegation_results"].update(external_input)

        # Update instance
        instance.status = InstanceStatus.RUNNING
        instance.updated_at = time.time()
        instance.pending_work_orders = []
        self.store.save_instance(instance)

        # Clean up suspension
        self.store.delete_suspension(instance_id)

        # Log
        self.store.log_action(
            instance_id=instance_id,
            correlation_id=instance.correlation_id,
            action_type="resume",
            details={
                "resumed_at_step": suspension.suspended_at_step,
                "external_input_keys": list(external_input.keys()),
            },
            idempotency_key=f"resume:{instance_id}:{suspension.resume_nonce}",
        )

        # Re-execute from suspended step using mid-graph entry
        try:
            result = self._execute_workflow_from_state(
                instance, state_snapshot,
                resume_step=suspension.suspended_at_step,
                model=model, temperature=temperature,
            )

            if isinstance(result, dict):
                # Workflow completed — normal path.
                # is_resume=True only when we resumed from a governance gate
                # (governance was already evaluated before that suspension).
                # For delegation-resumes (suspended_at_step != __governance_gate__),
                # governance has NOT yet evaluated — let it run now so the gate
                # fires after generate_final_report has the full enriched package.
                is_governance_resume = (
                    suspension.suspended_at_step == "__governance_gate__"
                )
                self._on_completed(instance, result, is_resume=is_governance_resume)
            else:
                # Workflow was interrupted again (recursive demand)
                self._on_interrupted(instance, result, model, temperature)

        except Exception as e:
            self._on_failed(instance, str(e))
            raise

        return instance_id

    def checkpoint(self, instance_id: str) -> dict[str, Any]:
        """Get the current state snapshot for an instance."""
        instance = self.store.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")
        suspension = self.store.get_suspension(instance_id)
        if suspension:
            return suspension.state_snapshot
        if instance.result:
            return instance.result
        return {}

    def terminate(self, instance_id: str, reason: str = "") -> dict[str, Any]:
        """Terminate an instance. Returns final state."""
        instance = self.store.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")

        instance.status = InstanceStatus.TERMINATED
        instance.updated_at = time.time()
        instance.error = reason or "Terminated by coordinator"
        self.store.save_instance(instance)

        # Clean up any suspension
        self.store.delete_suspension(instance_id)

        self.store.log_action(
            instance_id=instance_id,
            correlation_id=instance.correlation_id,
            action_type="terminate",
            details={"reason": reason},
        )

        self._log(f"■ TERMINATED {instance_id}: {reason}")
        return {"instance_id": instance_id, "status": "terminated", "reason": reason}

    def approve(
        self,
        instance_id: str,
        approver: str = "",
        notes: str = "",
        model: str = "default",
        temperature: float = 0.1,
    ) -> str:
        """
        Approve a governance-suspended instance.
        Marks the governance gate as passed, then proceeds to
        delegation evaluation and completion.
        """
        instance = self.store.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")
        if instance.status != InstanceStatus.SUSPENDED:
            raise ValueError(
                f"Instance {instance_id} is {instance.status.value}, not suspended"
            )

        suspension = self.store.get_suspension(instance_id)
        if not suspension:
            raise ValueError(f"No suspension record for {instance_id}")

        self._log(f"✓ APPROVED {instance_id} by '{approver or 'system'}'")

        # Log approval
        self.store.log_action(
            instance_id=instance_id,
            correlation_id=instance.correlation_id,
            action_type="governance_approved",
            details={
                "approver": approver,
                "notes": notes,
                "was_suspended_at": suspension.suspended_at_step,
            },
            idempotency_key=f"approve:{instance_id}:{suspension.resume_nonce}",
        )

        # Recover the full workflow state from the suspension
        final_state = suspension.state_snapshot

        # Clean up suspension
        self.store.delete_suspension(instance_id)

        # Mark the pending governance task as resolved. The CLI calls approve()
        # directly without going through resolve_task(), so the task stays PENDING
        # unless we close it here.
        pending_tasks = self.tasks.list_tasks(
            instance_id=instance_id, status=TaskStatus.PENDING
        )
        for task in pending_tasks:
            if task.task_type == TaskType.GOVERNANCE_APPROVAL:
                self.tasks.force_complete(task.task_id)
                break

        # Mark completed
        instance.status = InstanceStatus.COMPLETED
        instance.updated_at = time.time()
        instance.result = self._extract_result_summary(final_state)
        instance.step_count = len(final_state.get("steps", []))
        self.store.save_instance(instance)

        self._log(f"✓ COMPLETED {instance_id} (post-approval)")

        # Now proceed to delegation evaluation (skipped during suspension)
        self._evaluate_and_execute_delegations(instance, final_state)

        # ── Execution policies — dispatched after tier clearance ──
        self._evaluate_and_execute_executions(instance, final_state)

        # Check unresolved needs
        needs = self._extract_unresolved_needs(final_state)
        if needs:
            matches = self.policy.match_needs(needs)
            for match in matches:
                self._log(f"  need matched: {match.need_type} → "
                          f"{match.capability.provider_type}")

        # Check if this instance was a handler for a blocking delegation
        # If so, resume the requester
        self._check_delegation_completion(instance)

        return instance_id

    def reject(
        self,
        instance_id: str,
        rejector: str = "",
        reason: str = "",
    ) -> str:
        """Reject a governance-suspended instance. Terminates it."""
        instance = self.store.get_instance(instance_id)
        if not instance:
            raise ValueError(f"Instance not found: {instance_id}")
        if instance.status != InstanceStatus.SUSPENDED:
            raise ValueError(
                f"Instance {instance_id} is {instance.status.value}, not suspended"
            )

        self._log(f"✗ REJECTED {instance_id} by '{rejector or 'system'}': {reason}")

        self.store.log_action(
            instance_id=instance_id,
            correlation_id=instance.correlation_id,
            action_type="governance_rejected",
            details={
                "rejector": rejector,
                "reason": reason,
            },
        )

        self.store.delete_suspension(instance_id)
        return self.terminate(instance_id, f"Rejected by {rejector}: {reason}")["instance_id"]

    def list_pending_approvals(self) -> list[dict[str, Any]]:
        """List all tasks awaiting governance approval."""
        from cognitive_core.coordinator.contracts import assert_contract, PendingApproval

        tasks = self.tasks.list_tasks(status=TaskStatus.PENDING)
        approvals = []
        for task in tasks:
            if task.task_type == TaskType.GOVERNANCE_APPROVAL:
                entry = {
                    "task_id": task.task_id,
                    "instance_id": task.instance_id,
                    "workflow_type": task.workflow_type,
                    "domain": task.domain,
                    "governance_tier": task.payload.get("governance_tier", ""),
                    "correlation_id": task.correlation_id,
                    "queue": task.queue,
                    "created_at": task.created_at,
                    "sla_seconds": task.sla_seconds,
                    "expires_at": task.expires_at,
                    "priority": task.priority,
                    "callback_method": task.callback.method,
                    "resume_nonce": task.callback.resume_nonce,
                }
                assert_contract(entry, PendingApproval, "list_pending_approvals")
                approvals.append(entry)
        return approvals

    def list_queue_tasks(
        self,
        queue: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List tasks in the queue. Used by frontends and queue consumers
        to discover work items.
        """
        tasks = self.tasks.list_tasks(queue=queue, status=status)
        return [
            {
                "task_id": t.task_id,
                "task_type": t.task_type,
                "queue": t.queue,
                "instance_id": t.instance_id,
                "correlation_id": t.correlation_id,
                "workflow_type": t.workflow_type,
                "domain": t.domain,
                "status": t.status,
                "priority": t.priority,
                "created_at": t.created_at,
                "claimed_by": t.claimed_by,
                "sla_seconds": t.sla_seconds,
                "expires_at": t.expires_at,
                "payload": t.payload,
                "callback": {
                    "method": t.callback.method,
                    "instance_id": t.callback.instance_id,
                },
            }
            for t in tasks
        ]

    def claim_task(self, queue: str, claimed_by: str) -> dict[str, Any] | None:
        """
        Claim the next pending task from a queue.
        Returns the task payload + callback info, or None if empty.
        Used by queue consumers (API handlers, frontend polling, workers).
        """
        task = self.tasks.claim(queue, claimed_by)
        if not task:
            return None
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "instance_id": task.instance_id,
            "correlation_id": task.correlation_id,
            "workflow_type": task.workflow_type,
            "domain": task.domain,
            "payload": task.payload,
            "callback": {
                "method": task.callback.method,
                "instance_id": task.callback.instance_id,
                "resume_nonce": task.callback.resume_nonce,
            },
            "sla_seconds": task.sla_seconds,
            "expires_at": task.expires_at,
        }

    def resolve_task(
        self,
        task_id: str,
        action: str,
        resolved_by: str = "",
        notes: str = "",
        data: dict[str, Any] | None = None,
    ) -> str:
        """
        Resolve a claimed task. This is the primary API for external consumers.

        action: "approve" | "reject" | "defer" | "escalate"

        Calls the appropriate coordinator method based on the task's callback.
        Returns the instance_id.
        """
        task = self.tasks.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        if task.status != TaskStatus.CLAIMED:
            raise ValueError(
                f"Task {task_id} is {task.status}, not claimed"
            )

        # Resolve the task in the queue
        resolution = TaskResolution(
            task_id=task_id,
            action=action,
            resolved_by=resolved_by,
            notes=notes,
            data=data or {},
            resolved_at=time.time(),
        )
        self.tasks.resolve(task_id, resolution)

        # Execute the callback
        if action == "approve":
            return self.approve(
                instance_id=task.callback.instance_id,
                approver=resolved_by,
                notes=notes,
            )
        elif action == "reject":
            return self.reject(
                instance_id=task.callback.instance_id,
                rejector=resolved_by,
                reason=notes,
            )
        elif action == "defer":
            # Unclaim: set back to pending
            task_obj = self.tasks.get_task(task_id)
            if task_obj:
                self.tasks.unclaim(task_id)
            return task_obj.instance_id if task_obj else ""
        else:
            raise ValueError(f"Unknown action: {action}")

    def expire_overdue_tasks(self) -> int:
        """Expire tasks past their SLA. Call periodically."""
        return self.tasks.expire_overdue()

    def sweep_reservations(self) -> dict[str, Any]:
        """
        Periodic reservation sweep. Call alongside expire_overdue_tasks.

        1. Sweeps expired reservations in the resource registry (ddd.py)
        2. Evaluates expiring reservations through the revocation manager
        3. Logs reservation events for audit trail

        Returns summary dict with counts.
        """
        summary = {"expired": 0, "extended": 0, "revoked": 0}

        if not (self._resource_registry and _OPTIMIZER_AVAILABLE):
            return summary

        # Phase 1: Hard TTL expiry sweep via ddd.py
        expired_ids = self._resource_registry.sweep_expired_reservations()
        summary["expired"] = len(expired_ids)

        # Log expire events
        if _HARDENING_AVAILABLE and hasattr(self, '_reservation_log') and self._reservation_log:
            for rsv_id in expired_ids:
                self._reservation_log.record(
                    rsv_id, "", "", "expire",
                )

        # Phase 2: Graceful revocation for near-expiry reservations
        if self._revocation_manager:
            for res in self._resource_registry.list_resources():
                for rsv in getattr(res, '_reservations', {}).values():
                    if not hasattr(rsv, 'is_expired') or rsv.is_expired():
                        continue
                    ttl_remaining = 0.0
                    if hasattr(rsv, 'expires_at') and rsv.expires_at:
                        import time as _time
                        ttl_remaining = rsv.expires_at - _time.time()

                    # Estimate progress from work order status
                    progress = 0.5  # default estimate

                    signal = self._revocation_manager.evaluate_expiring_reservation(
                        work_order_id=getattr(rsv, 'work_order_id', ''),
                        reservation_ttl_remaining=ttl_remaining,
                        work_order_priority="routine",
                        work_order_progress=progress,
                        waiting_queue_depth=0,
                    )
                    if signal:
                        if signal.policy.value == "extend_ttl":
                            summary["extended"] += 1
                        else:
                            summary["revoked"] += 1
                            self._log(
                                f"  ⚠ REVOCATION: {signal.work_order_id} → "
                                f"{signal.policy.value}: {signal.reason}"
                            )

        # Phase 3: Drain backpressure queue — resources may now have capacity
        drained = self.drain_resource_queue()
        if drained:
            summary["drained"] = drained

        return summary

    # ─── Event Handlers ──────────────────────────────────────────────

    def _on_completed(self, instance: InstanceState, final_state: dict[str, Any],
                       is_resume: bool = False):
        """Handle workflow completion. Evaluate governance and delegations.

        Args:
            is_resume: If True, skip governance re-evaluation. Governance was
                       already evaluated before the suspension that led to this
                       resume. Re-evaluating risks a different decision if
                       policy config changed between suspension and resume.
        """
        instance.result = self._extract_result_summary(final_state)
        instance.step_count = len(final_state.get("steps", []))

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="execution_finished",
            details={
                "step_count": instance.step_count,
                "result_keys": list(instance.result.keys()) if instance.result else [],
            },
        )

        self._log(f"✓ EXECUTION FINISHED {instance.instance_id} "
                   f"({instance.step_count} steps)")

        # ── Step 0: Quality Gate — Fail Closed ──
        # Check all step confidences against thresholds. If any step
        # is below the floor, escalate to HITL regardless of domain tier.
        # Exception: if governance_tier was explicitly locked via delegation
        # governance_tier_override, the quality gate must not escalate —
        # the parent workflow's gate is the authoritative quality check.
        if not is_resume and not instance.governance_tier_locked:
            qg = self._evaluate_quality_gate(instance, final_state)
            if qg:
                self._log(f"  ⚠ QUALITY GATE FIRED: {qg['reason']}")
                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="quality_gate_fired",
                    details=qg,
                )
                # Override governance tier to force HITL review
                instance.governance_tier = qg["escalation_tier"]
                # Stash for escalation brief
                instance._quality_gate = qg
        elif not is_resume and instance.governance_tier_locked:
            self._log(f"  ℹ quality gate skipped — tier locked by delegation override "
                       f"({instance.governance_tier})")

        # ── Step 1a: Peek at wait_for_result delegations ──
        # Blocking delegations MUST fire before the governance gate so that
        # generate_final_report runs with the full enriched package.
        # If any new wait_for_result delegation would fire, skip governance
        # now — it fires on the delegation-resume completion instead.
        has_blocking_delegations = False
        if not is_resume:
            try:
                pending_deleg = self.policy.evaluate_delegations(
                    domain=instance.domain,
                    workflow_output=final_state,
                    lineage=instance.lineage,
                )
                # Dedup: skip already-fired policies
                already_fired_set: set[str] = set()
                try:
                    ledger_entries = self.store.get_ledger(instance_id=instance.instance_id)
                    for entry in ledger_entries:
                        if entry.get("action_type") == "delegation_dispatched":
                            details = entry.get("details", {})
                            if isinstance(details, str):
                                try:
                                    details = json.loads(details)
                                except (ValueError, TypeError):
                                    details = {}
                            pname = details.get("policy", "")
                            if pname:
                                already_fired_set.add(pname)
                except Exception:
                    pass
                new_blocking = [
                    d for d in pending_deleg
                    if d.mode == "wait_for_result"
                    and d.policy_name not in already_fired_set
                ]
                has_blocking_delegations = bool(new_blocking)
                if has_blocking_delegations:
                    self._log(
                        f"  governance: deferred — {len(new_blocking)} wait_for_result "
                        f"delegation(s) must run first (gate fires after final resume)"
                    )
            except Exception as e:
                self._log(f"  governance: delegation pre-check failed ({e}); proceeding normally")

        # ── Step 1b: Evaluate governance tier ──
        # Skip when:
        #   is_resume=True  — resumed from governance gate (already evaluated)
        #   has_blocking_delegations — wait_for_result delegations fire first;
        #                              gate deferred to delegation-resume completion
        if not is_resume and not has_blocking_delegations:

            # ── Honour govern primitive output if present ──
            # If the workflow included a govern step, its tier_applied and
            # disposition are the authoritative governance decision. Upgrade
            # the instance tier before policy.evaluate_governance() runs so
            # the coordinator honours what the primitive decided.
            govern_steps = [
                s for s in final_state.get("steps", [])
                if s.get("primitive") == "govern"
            ]
            if govern_steps:
                # Use the last govern step (most authoritative)
                gov_output = govern_steps[-1].get("output", {})
                prim_tier = gov_output.get("tier_applied")
                prim_disposition = gov_output.get("disposition")
                if prim_tier:
                    # Normalise enum to string: GovernanceTier.GATE → "gate"
                    tier_str = str(prim_tier).lower().replace("governancetier.", "")
                    # Tier escalation is strictly upward — never de-escalate
                    current = instance.governance_tier or "auto"
                    tier_order = {"auto": 0, "spot_check": 1, "gate": 2, "hold": 3}
                    if tier_order.get(tier_str, 0) > tier_order.get(current, 0):
                        instance.governance_tier = tier_str
                        self._log(f"  govern primitive → tier upgraded to {tier_str}")
                if prim_disposition in ("suspend", "escalate", "hold"):
                    # Force the instance tier to at least gate
                    if instance.governance_tier not in ("gate", "hold"):
                        instance.governance_tier = "gate"
                        self._log(f"  govern primitive → disposition={prim_disposition}, tier forced to gate")

            gov_decision = self.policy.evaluate_governance(
                domain=instance.domain,
                governance_tier=instance.governance_tier,
                workflow_result=final_state,
            )
            self._log(f"  governance: {gov_decision.tier} → {gov_decision.action} "
                       f"({gov_decision.reason})")

            self.store.log_action(
                instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                action_type="governance_evaluation",
                details={
                    "tier": gov_decision.tier,
                    "action": gov_decision.action,
                    "queue": gov_decision.queue,
                    "reason": gov_decision.reason,
                    "sampled": gov_decision.sampled,
                },
            )

            if gov_decision.action == "suspend_for_approval":
                self._suspend_for_governance(
                    instance, final_state, gov_decision
                )
                return

            if gov_decision.action == "queue_review":
                self._log(f"  → queued for post-completion review: {gov_decision.queue}")
                task = Task.create(
                    task_type=TaskType.SPOT_CHECK_REVIEW,
                    queue=gov_decision.queue,
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    workflow_type=instance.workflow_type,
                    domain=instance.domain,
                    payload={
                        "governance_tier": gov_decision.tier,
                        "reason": gov_decision.reason,
                        "sampled": gov_decision.sampled,
                        "step_count": instance.step_count,
                        "result_summary": instance.result,
                    },
                    callback=TaskCallback(
                        method="complete",
                        instance_id=instance.instance_id,
                    ),
                    priority=0,
                    sla_seconds=self._get_tier_sla(gov_decision.tier),
                )
                self.tasks.publish(task)
        elif is_resume:
            self._log(f"  governance: skipped (resume from governance gate — already evaluated)")
            self.store.log_action(
                instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                action_type="governance_evaluation",
                details={"skipped": True, "reason": "resume after governance approval"},
            )
        # else: has_blocking_delegations — governance deferred, logged above

        # Mark completed (only reached if governance allows)
        instance.status = InstanceStatus.COMPLETED
        instance.updated_at = time.time()
        self.store.save_instance(instance)

        self._log(f"✓ COMPLETED {instance.instance_id}")

        # ── Step 2: Evaluate delegation policies ──
        self._evaluate_and_execute_delegations(instance, final_state)

        # ── Step 2b: Evaluate execution policies ──
        self._evaluate_and_execute_executions(instance, final_state)

        # ── Step 3: Check for unresolved needs ──
        needs = self._extract_unresolved_needs(final_state)
        if needs:
            matches = self.policy.match_needs(needs)
            for match in matches:
                self._log(f"  need matched: {match.need_type} → "
                          f"{match.capability.provider_type}")

        # ── Step 4: Check if this was a handler for a blocking delegation ──
        self._check_delegation_completion(instance)

    def _suspend_for_governance(
        self,
        instance: InstanceState,
        final_state: dict[str, Any],
        gov_decision,
    ):
        """
        Truly suspend an instance pending governance approval.
        Saves state snapshot, publishes a task to the approval queue,
        and marks instance as suspended.

        The task includes an escalation brief — a structured summary
        that helps the human reviewer make a decision faster.
        """
        suspension = Suspension.create(
            instance_id=instance.instance_id,
            suspended_at_step="__governance_gate__",
            state_snapshot=self._compact_state_for_suspension(final_state),
        )
        self.store.save_suspension(suspension)

        instance.status = InstanceStatus.SUSPENDED
        instance.updated_at = time.time()
        instance.resume_nonce = suspension.resume_nonce
        self.store.save_instance(instance)

        # Build escalation brief for the human reviewer
        escalation_brief = None
        try:
            from cognitive_core.coordinator.escalation import build_escalation_brief
            escalation_brief = build_escalation_brief(
                workflow_type=instance.workflow_type,
                domain=instance.domain,
                final_state=final_state,
                escalation_reason=gov_decision.reason,
                quality_gate=getattr(instance, '_quality_gate', None),
            )
        except Exception as e:
            self._log(f"  ⚠ Failed to build escalation brief: {e}")

        # Publish task to approval queue
        task_payload = {
            "governance_tier": gov_decision.tier,
            "reason": gov_decision.reason,
            "step_count": instance.step_count,
            "result_summary": instance.result,
        }
        if escalation_brief:
            task_payload["escalation_brief"] = escalation_brief

        # Attach epistemic context to the task payload so the reviewer
        # sees structured uncertainty, not just a scalar confidence.
        try:
            ep_record = getattr(instance, "_epistemic_record", None)
            if ep_record:
                task_payload["epistemic"] = ep_record.reviewer_context()
                # Merge into escalation_brief if present
                if escalation_brief:
                    escalation_brief["epistemic"] = ep_record.reviewer_context()
        except Exception:
            pass

        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue=gov_decision.queue,
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            workflow_type=instance.workflow_type,
            domain=instance.domain,
            payload=task_payload,
            callback=TaskCallback(
                method="approve",
                instance_id=instance.instance_id,
                resume_nonce=suspension.resume_nonce,
            ),
            priority=2 if gov_decision.tier == "hold" else 1,
            sla_seconds=self._get_tier_sla(gov_decision.tier),
        )
        task_id = self.tasks.publish(task)

        # Record which backend handled this governance task
        try:
            from cognitive_core.engine.governance import get_governance
            _gov_tq = get_governance()
            _gov_tq._record_proof(
                "governance_task_published",
                step="__governance__",
                instance_id=instance.instance_id,
                task_id=task_id,
                queue_backend=self.tasks.backend_name,
                task_type=task.task_type,
                queue=gov_decision.queue,
            )
        except Exception:
            pass

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="suspended_for_approval",
            details={
                "tier": gov_decision.tier,
                "queue": gov_decision.queue,
                "reason": gov_decision.reason,
                "resume_nonce": suspension.resume_nonce,
                "task_id": task_id,
                "queue_backend": self.tasks.backend_name,
            },
        )

        self._log(f"  ⏸ SUSPENDED {instance.instance_id} "
                   f"→ task {task_id} published to '{gov_decision.queue}'")

    # ─── Quality Gate ────────────────────────────────────────────

    def _evaluate_quality_gate(
        self, instance: InstanceState, final_state: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Check workflow output against quality gate conditions.

        Supports two forms (both can coexist in coordinator_config.yaml):

        1. Scalar form (backward compatible):
           quality_gates:
             min_confidence: 0.5
             primitive_floors: {classify: 0.70, deliberate: 0.75}

        2. Multi-condition epistemic form (new):
           quality_gates:
             gate_triggers:
               - any_step.evidence_completeness < 0.70
               - verify.rule_coverage < 0.80
               - coherence_flag: VERIFY_DELIBERATE_TENSION
               - confidence_drop > 0.25
               - not_warranted

        gate_triggers are evaluated against the WorkflowEpistemicRecord
        accumulated during execution. Any trigger that fires causes
        escalation. Multiple fired triggers are all reported.

        Returns None if all gates pass, or a dict with details.
        """
        qg_config = self.policy.raw_config.get("quality_gates", {})
        if not qg_config:
            return None

        exempt = qg_config.get("exempt_domains", [])
        if instance.domain in exempt:
            return None

        escalation_tier = qg_config.get("escalation_tier", "gate")
        escalation_queue = qg_config.get("escalation_queue", "quality_review")

        # ── Form 2: gate_triggers DSL (epistemic-aware) ────────────
        gate_triggers = qg_config.get("gate_triggers", [])
        if gate_triggers:
            ep_record = getattr(instance, "_epistemic_record", None)
            if ep_record is not None:
                try:
                    from cognitive_core.engine.epistemic import evaluate_gate_triggers
                    fired = evaluate_gate_triggers(gate_triggers, ep_record)
                    if fired:
                        # Report all fired triggers, not just the first
                        reasons = " | ".join(t["description"] for t in fired)
                        return {
                            "reason": reasons,
                            "fired_triggers": fired,
                            "trigger_count": len(fired),
                            "escalation_tier": escalation_tier,
                            "escalation_queue": escalation_queue,
                            # Compat fields for existing escalation brief builder
                            "step_name": fired[0].get("step_name", ""),
                            "primitive": fired[0].get("primitive", ""),
                            "confidence": fired[0].get("value"),
                            "floor": fired[0].get("threshold"),
                        }
                except Exception as e:
                    self._log(f"  ⚠ gate_triggers evaluation failed (falling back to scalar): {e}")

        # ── Form 1: scalar confidence floors (backward compatible) ──
        global_min = qg_config.get("min_confidence", 0.5)
        primitive_floors = qg_config.get("primitive_floors", {})

        if not global_min and not primitive_floors:
            return None

        steps = final_state.get("steps", [])
        for step in steps:
            output = step.get("output", {})
            confidence = output.get("confidence")
            if confidence is None:
                continue

            primitive = step.get("primitive", "")
            step_name = step.get("step_name", "")
            floor = primitive_floors.get(primitive, global_min)

            if confidence < floor:
                return {
                    "reason": f"Step '{step_name}' ({primitive}) confidence {confidence} < floor {floor}",
                    "step_name": step_name,
                    "primitive": primitive,
                    "confidence": confidence,
                    "floor": floor,
                    "escalation_tier": escalation_tier,
                    "escalation_queue": escalation_queue,
                }

        return None

    def _get_tier_sla(self, tier: str) -> float | None:
        """Get SLA seconds for a governance tier from config."""
        tier_config = self.policy.governance_tiers.get(tier)
        if tier_config and tier_config.sla_seconds:
            return tier_config.sla_seconds
        return None

    def _call_execution_tool(self, tool_name: str, inputs: dict) -> dict:
        """
        Call an execution MCP tool.

        Opens a fresh stdio session per call to avoid anyio cancel-scope
        cross-task issues that occur with persistent sessions on background loops.
        Each call is fully self-contained: connect → call → disconnect.
        """
        import asyncio
        import json

        cmd_info = self._execution_mcp_cmd_info
        if not cmd_info:
            return {"error": "No execution MCP command configured"}

        async def _call_fresh():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=cmd_info["command"],
                args=cmd_info["args"],
                env=cmd_info.get("env"),
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=inputs)
                    contents = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            contents.append(block.text)
                        elif hasattr(block, "data"):
                            contents.append(str(block.data))
                    raw = " ".join(contents)
                    try:
                        return json.loads(raw)
                    except Exception:
                        return {"result": raw}

        return asyncio.run(_call_fresh())

    def _init_execution_mcp(self) -> None:
        """
        Initialize the execution MCP client from coordinator config.
        Called lazily on first execution policy dispatch.
        Reads execution_mcp_cmd from config or EXECUTION_MCP_CMD env var.
        """
        if self._execution_mcp_client is not None:
            return  # Already initialized

        import os
        execution_mcp_cmd = (
            os.environ.get("EXECUTION_MCP_CMD", "")
            or self.config.get("execution_mcp_cmd", "")
        )

        if not execution_mcp_cmd:
            self._log("  ⚠ No execution_mcp_cmd configured — execution ops will be logged as pending")
            return

        try:
            import re as _re, os as _os, asyncio

            # Parse optional KEY=VALUE env prefixes before the command.
            # e.g. "DRY_RUN=true python execution_mcp.py"
            # → env={"DRY_RUN": "true"}, command="python", args=["execution_mcp.py"]
            parts = execution_mcp_cmd.split()
            env_overrides = {}
            cmd_start = 0
            for i, part in enumerate(parts):
                if _re.match(r'^[A-Z_][A-Z0-9_]*=', part):
                    k, v = part.split("=", 1)
                    env_overrides[k] = v
                    cmd_start = i + 1
                else:
                    cmd_start = i
                    break

            cmd_parts = parts[cmd_start:]
            if not cmd_parts:
                raise ValueError(f"No command found in execution_mcp_cmd: {execution_mcp_cmd!r}")

            mcp_env = {**_os.environ, **env_overrides} if env_overrides else None

            # Store parsed command info — no persistent session needed.
            # _call_execution_tool opens a fresh connection per call to avoid
            # anyio cancel-scope cross-task errors with persistent sessions.
            self._execution_mcp_cmd_info = {
                "command": cmd_parts[0],
                "args": cmd_parts[1:],
                "env": mcp_env,
            }

            # Validate by doing a quick connect/list-tools/disconnect
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            async def _validate():
                server_params = StdioServerParameters(
                    command=cmd_parts[0],
                    args=cmd_parts[1:],
                    env=mcp_env,
                )
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        return [t.name for t in tools_result.tools]

            tool_names = asyncio.run(_validate())
            self._execution_mcp_client = True  # sentinel: initialized
            self._log(f"  ⚡ Execution MCP ready ({len(tool_names)} tools): "
                      f"{', '.join(tool_names[:5])}")

        except Exception as e:
            self._log(f"  ✗ Failed to connect execution MCP: {e}")
            self._execution_mcp_client = None
            self._execution_mcp_cmd_info = None

    def _evaluate_and_execute_executions(
        self,
        instance: "InstanceState",
        final_state: dict[str, Any],
    ) -> None:
        """
        Evaluate execution policies and dispatch any that fire.

        Execution policies are deterministic — no LLM calls.
        They fire after tier clearance (post-approval for GATE/HOLD,
        immediately for AUTO/SPOT_CHECK).

        Each policy maps to a list of ExecutionOps backed by MCP tools.
        For GATE/HOLD cases this is called post-approval so the tier
        has already been cleared by the human decision.

        Each dispatched op is logged to the action ledger with:
          - tool name and resolved inputs
          - reversible flag and reversal_tool
          - execution result or error

        Reversible ops are registered in the CompensationLedger so
        they can be unwound if the workflow is subsequently reversed.
        """
        # Initialize execution MCP client on first use
        self._init_execution_mcp()

        tier_str = str(instance.governance_tier).lower().replace("governancetier.", "")
        decisions = self.policy.evaluate_executions(
            domain=instance.domain,
            workflow_output=final_state,
            tier=tier_str,
        )

        if not decisions:
            return

        total_ops = sum(len(d.ops) for d in decisions)
        self._log(f"  executions: {len(self.policy.execution_policies)} policies evaluated, "
                  f"{len(decisions)} triggered, {total_ops} op(s)")

        for decision in decisions:
            for op in decision.ops:
                self._dispatch_execution_op(instance, decision.policy_name, op)

    def _dispatch_execution_op(
        self,
        instance: "InstanceState",
        policy_name: str,
        op: Any,
    ) -> None:
        """
        Dispatch a single execution op via MCP tool call.
        Logs the attempt and result to the action ledger.
        Registers reversible ops in the CompensationLedger.
        """
        import time as _time

        self._log(f"  ⚡ exec: {op.tool}")
        if op.description:
            self._log(f"     {op.description[:70]}")

        # Log the dispatch attempt
        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="execution_dispatched",
            details={
                "policy": policy_name,
                "tool": op.tool,
                "inputs": op.inputs,
                "reversible": op.reversible,
                "reversal_tool": op.reversal_tool,
            },
            idempotency_key=f"exec:{instance.instance_id}:{policy_name}:{op.tool}",
        )

        # Dispatch via MCP client if available
        result = None
        error = None
        t0 = _time.time()
        try:
            if self._execution_mcp_client:
                # Call tool directly via MCP session — bypasses ToolRegistry
                result = self._call_execution_tool(op.tool, op.inputs)
                self._log(f"     ✓ {op.tool} ({int((_time.time()-t0)*1000)}ms)")
            else:
                # No execution MCP client — log as pending for external pickup
                self._log(f"     ⚠ {op.tool}: no execution MCP client — logged for external dispatch")
                result = {"status": "pending_external_dispatch", "tool": op.tool,
                          "inputs": op.inputs}
        except Exception as e:
            error = str(e)
            self._log(f"     ✗ {op.tool}: {error[:80]}")

        # Log the result
        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="execution_completed" if not error else "execution_failed",
            details={
                "policy": policy_name,
                "tool": op.tool,
                "result": result,
                "error": error,
                "elapsed_ms": int((_time.time() - t0) * 1000),
            },
        )

        # Register reversible ops with CompensationLedger
        if op.reversible and op.reversal_tool and not error:
            try:
                from cognitive_core.engine.compensation import get_compensation_ledger
                ledger = get_compensation_ledger()
                if ledger:
                    ledger.register(
                        instance_id=instance.instance_id,
                        tool=op.reversal_tool,
                        inputs=op.inputs,
                        description=f"Reverse {op.tool}",
                    )
            except Exception:
                pass  # Compensation ledger is optional


    def _evaluate_and_execute_delegations(
        self,
        instance: InstanceState,
        final_state: dict[str, Any],
    ):
        """
        Evaluate delegation policies and execute any that fire.

        Session 6 duplication fix
        ─────────────────────────
        Root cause: two threads could both read the ledger (empty), both
        conclude no delegations had fired, both enter the RLock, and both
        pass the fired_recheck — because neither had written to the DB yet.
        The ledger write was in _execute_delegation, *after* the lock,
        so the recheck inside the lock saw stale data.

        Fix: write the idempotency claim INSIDE the lock, BEFORE executing
        the delegation. The claim uses log_action's UNIQUE idempotency_key
        constraint — the second thread's log_action returns False (duplicate),
        so it skips. First writer wins, atomically, even across threads.

        This also handles the recursive call case: when a handler's
        _on_completed calls back into this method, the claim is already
        in the DB, so the delegation is skipped without needing a separate
        recursive guard.
        """
        import threading as _thr
        _tid = _thr.current_thread().name

        delegations = self.policy.evaluate_delegations(
            domain=instance.domain,
            workflow_output=final_state,
            lineage=instance.lineage,
        )

        # ── Pre-filter: skip policies visibly already fired ──
        # This is an optimistic read — may have false negatives under concurrency,
        # but those are caught by the atomic claim inside the lock below.
        already_fired = set()
        ledger = self.store.get_ledger(instance_id=instance.instance_id)
        for entry in ledger:
            if entry["action_type"] == "delegation_dispatched":
                details = entry.get("details", {})
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except (json.JSONDecodeError, TypeError):
                        details = {}
                policy_name = details.get("policy", "")
                if policy_name:
                    already_fired.add(policy_name)

        new_delegations = [d for d in delegations if d.policy_name not in already_fired]
        skipped = len(delegations) - len(new_delegations)

        self._log(f"  delegations: {len(self.policy.delegation_policies)} policies evaluated, "
                   f"{len(delegations)} triggered"
                   f"{f', {skipped} skipped (already fired)' if skipped else ''}")
        self._log(f"  [DELEG-TRACE] thread={_tid} instance={instance.instance_id[:8]} "
                  f"candidates={[d.policy_name for d in new_delegations]}")

        # Log evidence flags from investigate steps for visibility
        for step in final_state.get("steps", []):
            if step.get("primitive") == "investigate":
                flags = step.get("output", {}).get("evidence_flags", [])
                if flags:
                    self._log(f"  evidence_flags ({step['step_name']}): {flags}")

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="delegation_evaluation",
            details={
                "policies_checked": len(self.policy.delegation_policies),
                "delegations_triggered": len(delegations),
                "delegations_new": len(new_delegations),
                "delegations_skipped": skipped,
                "delegation_names": [d.policy_name for d in new_delegations],
                "already_fired": list(already_fired) if already_fired else [],
            },
        )

        if not new_delegations:
            return

        # ── Partition by mode ──
        ff_delegations = [d for d in new_delegations if d.mode != "wait_for_result"]
        blocking_delegations = [d for d in new_delegations if d.mode == "wait_for_result"]

        with self._delegation_lock:
            # ── Atomic claim: write idempotency key BEFORE executing ──
            # log_action uses INSERT with a UNIQUE idempotency_key.
            # If two threads race here, the second INSERT returns False.
            # First writer executes the delegation; second writer skips.
            # This is the definitive fix for the duplication bug:
            # the claim is visible to all threads immediately after commit.
            claimed_ff = []
            for deleg in ff_delegations:
                claim_key = f"deleg:{instance.instance_id}:{deleg.policy_name}"
                claimed = self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="delegation_dispatched",
                    details={
                        "policy": deleg.policy_name,
                        "mode": deleg.mode,
                        "target_workflow": deleg.target_workflow,
                        "target_domain": deleg.target_domain,
                        "contract": deleg.contract_name,
                        "_claimed_only": True,  # work_order_id filled in _execute_delegation
                    },
                    idempotency_key=claim_key,
                )
                if claimed:
                    claimed_ff.append(deleg)
                    self._log(f"  [DELEG-TRACE] claimed {deleg.policy_name} (ff)")
                else:
                    self._log(f"  [DELEG-TRACE] {deleg.policy_name} already claimed by another thread — skipping")

            claimed_blocking = []
            for deleg in blocking_delegations:
                claim_key = f"deleg:{instance.instance_id}:{deleg.policy_name}"
                claimed = self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="delegation_dispatched",
                    details={
                        "policy": deleg.policy_name,
                        "mode": deleg.mode,
                        "target_workflow": deleg.target_workflow,
                        "target_domain": deleg.target_domain,
                        "contract": deleg.contract_name,
                        "_claimed_only": True,
                    },
                    idempotency_key=claim_key,
                )
                if claimed:
                    claimed_blocking.append(deleg)
                    self._log(f"  [DELEG-TRACE] claimed {deleg.policy_name} (blocking)")
                else:
                    self._log(f"  [DELEG-TRACE] {deleg.policy_name} already claimed — skipping")

            # ── Execute claimed delegations (lock still held for FF) ──
            # FF delegations: source is already COMPLETED, safe to dispatch inside lock.
            # Blocking delegations: source will be suspended; also safe inside lock
            # since suspension writes happen inside _execute_all_blocking_delegations.
            for deleg in claimed_ff:
                self._execute_claimed_delegation(instance, deleg, final_state)

            if claimed_blocking:
                self._execute_all_blocking_delegations(instance, claimed_blocking, final_state)


    def _execute_claimed_delegation(
        self,
        source: InstanceState,
        decision: "DelegationDecision",
        source_state: dict[str, Any],
    ):
        """
        Execute a fire-and-forget delegation that was already atomically claimed
        via log_action idempotency key in _evaluate_and_execute_delegations.
        Skips the redundant log_action write that _execute_delegation would do.
        """
        errors = self.policy.validate_work_order_inputs(
            decision.contract_name, decision.inputs
        )
        if errors:
            self._log(f"  ⚠ delegation {decision.policy_name} skipped: "
                       f"contract validation failed: {errors}")
            self.store.log_action(
                instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                action_type="delegation_skipped",
                details={"policy": decision.policy_name, "errors": errors},
            )
            return

        wo = WorkOrder.create(
            requester_instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            contract_name=decision.contract_name,
            contract_version=decision.contract_version,
            inputs=decision.inputs,
            sla_seconds=decision.sla_seconds,
        )
        wo.handler_workflow_type = decision.target_workflow
        wo.handler_domain = decision.target_domain
        wo.status = WorkOrderStatus.DISPATCHED
        wo.dispatched_at = time.time()
        self.store.save_work_order(wo)

        self.store.log_action(
            instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            action_type="delegation_work_order_created",
            details={
                "policy": decision.policy_name,
                "work_order_id": wo.work_order_id,
                "target_workflow": decision.target_workflow,
                "target_domain": decision.target_domain,
            },
        )

        self._log(f"  → DELEGATION [fire]: {decision.policy_name}")
        self._log(f"    work_order: {wo.work_order_id}")
        self._log(f"    target: {decision.target_workflow}/{decision.target_domain}")

        tool_keys = [k for k, v in decision.inputs.items()
                     if isinstance(v, (dict, list)) and k.startswith("get_")]
        scalar_keys = [k for k in decision.inputs.keys() if k not in tool_keys]
        self._log(f"    inputs: {scalar_keys}")
        if not tool_keys:
            self._log(f"    ⚠ NO tool data in inputs — handler retrieve may fail")

        self._execute_fire_and_forget_delegation(source, decision, wo)

    def _on_kill_switch_suspended(self, instance: InstanceState, reason: str):
        """Suspend workflow when a kill switch blocks Act execution."""
        suspension = Suspension.create(
            instance_id=instance.instance_id,
            suspended_at_step="__kill_switch__",
            state_snapshot={"kill_switch_reason": reason},
        )
        self.store.save_suspension(suspension)

        instance.status = InstanceStatus.SUSPENDED
        instance.updated_at = time.time()
        instance.resume_nonce = suspension.resume_nonce
        self.store.save_instance(instance)

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="kill_switch_suspended",
            details={"reason": reason},
        )
        self._log(f"  ⛔ SUSPENDED (kill switch) {instance.instance_id}: {reason}")

    def _resolve_governance_tier_upgrade(self, current_tier: str, upgrade_to: str) -> str:
        """Upgrade tier only — never downgrade."""
        order = {"auto": 0, "gate": 1, "hold": 2}
        if order.get(upgrade_to, 0) > order.get(current_tier, 0):
            return upgrade_to
        return current_tier

    def _on_failed(self, instance: InstanceState, error: str):
        """Handle workflow failure."""
        instance.status = InstanceStatus.FAILED
        instance.updated_at = time.time()
        instance.error = error
        self.store.save_instance(instance)

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="failed",
            details={"error": error},
        )

        self._log(f"✗ FAILED {instance.instance_id}: {error}")

    # ─── Demand-Driven Delegation (Backward Chaining) ────────────────

    def _on_interrupted(
        self,
        instance: InstanceState,
        interrupt,  # StepInterrupt from cognitive_core.engine.stepper
        model: str = "default",
        temperature: float = 0.1,
    ):
        """
        Handle a workflow that paused mid-execution because a step
        couldn't proceed without external input.

        The workflow was moving forward through its steps normally.
        At some step, the agent realized it needs something it doesn't
        have — a decision, analysis, data, authorization, anything.
        It produced a ResourceRequest and the stepper paused.

        The coordinator now:
        1. Looks up each need in the capability registry
        2. Dispatches providers (workflows or human tasks) to fulfill them
        3. Suspends the source workflow with its partial state
        4. When all providers complete, resumes the source forward
           from the step that paused, now with the resources available
        """
        from cognitive_core.engine.devs_executor import StepInterrupt

        step_name = interrupt.suspended_at_step
        requests = interrupt.resource_requests
        partial_state = interrupt.state_at_interrupt

        self._log(f"  ⏸ INTERRUPTED {instance.instance_id} at '{step_name}'")
        self._log(f"    {len(requests)} blocking resource request(s)")

        # ── Match needs to capabilities ──
        # Guard: check if this need was already dispatched & completed
        # by looking at delegation_results in the partial state.
        # After prepare_resume_state, delegation results live in
        # state.input.delegation (NOT at top-level delegation_results).
        existing_delegations = set(
            partial_state.get("input", {}).get("delegation", {}).keys()
        ) | set(
            partial_state.get("delegation_results", {}).keys()
        )

        matched = []
        unmatched = []
        for req in requests:
            need = req.get("need", "")

            if need in existing_delegations:
                self._log(f"    ⚠ '{need}' already fulfilled (result in delegation_results) — "
                          f"LLM is not using delegation results. Skipping re-dispatch.")
                # Don't add to unmatched — just skip it entirely
                continue

            # Look up in capability registry
            capability = self._find_capability(need)
            if capability:
                matched.append((req, capability))
                self._log(f"    ✓ '{need}' → {capability.provider_type}"
                           f":{capability.workflow_type}/{capability.domain}")
            else:
                unmatched.append(req)
                self._log(f"    ✗ '{need}' → no capability found")
        if not matched:
            # Check if ALL requests were already fulfilled (duplicate guard)
            all_skipped = all(
                req.get("need", "") in existing_delegations
                for req in requests
            )
            if all_skipped and existing_delegations:
                # All needs already fulfilled — LLM is not using delegation
                # results. The stepper callback should have caught this,
                # but if we got here as a fallback, just log and return.
                # The instance will be left in a confused state, but
                # that's better than marking it as failed.
                self._log(f"    ⚠ All {len(requests)} request(s) already fulfilled "
                          f"but stepper didn't filter them. This shouldn't happen.")
                self._on_failed(instance,
                    f"LLM re-requested already-fulfilled needs at '{step_name}'. "
                    f"Delegation results were present but LLM ignored them.")
                return

            # Truly unresolvable locally — try federation escalation
            need_names = [r.get("need", "?") for r in requests]

            # If we have a federation spoke with a parent, escalate
            if (self._federation_spoke
                    and self._federation_spoke.has_parent):
                self._log(f"    ⚠ No local capability — escalating to parent hub")
                escalated_any = False
                for req in requests:
                    need = req.get("need", "")
                    receipt = self._federation_spoke.escalate(
                        need_type=need,
                        work_order_id=f"esc_{instance.instance_id}_{need}",
                        context=req.get("context", {}),
                        correlation_id=instance.correlation_id,
                    )
                    if receipt:
                        escalated_any = True
                        self._log(f"    ↗ '{need}' escalated → {receipt.target_child_id} "
                                  f"(receipt={receipt.receipt_id})")
                        self.store.log_action(
                            instance_id=instance.instance_id,
                            correlation_id=instance.correlation_id,
                            action_type="federation_escalated",
                            details={
                                "need_type": need,
                                "receipt_id": receipt.receipt_id,
                                "target_child": receipt.target_child_id,
                            },
                        )
                    else:
                        self._log(f"    ✗ '{need}' → parent also cannot fulfill")
                if escalated_any:
                    # Don't fail — the parent is handling it.
                    # Instance stays SUSPENDED waiting for federated result.
                    return
                # Fall through to failure if parent couldn't help either

            error = (
                f"Workflow interrupted at '{step_name}' with unresolvable "
                f"resource requests: {need_names}. No matching capabilities "
                f"in registry."
            )
            self._on_failed(instance, error)
            return

        if unmatched:
            unmatched_names = [r.get("need", "?") for r in unmatched]
            self._log(f"    ⚠ {len(unmatched)} unmatched needs (proceeding with matched): "
                       f"{unmatched_names}")

        # ── Create work orders for each matched capability ──
        work_order_ids = []
        wo_need_map = {}  # wo_id → need_name
        for req, capability in matched:
            wo = WorkOrder.create(
                requester_instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                contract_name=capability.contract_name or req.get("contract", ""),
                contract_version=1,
                inputs=req.get("context", {}),
                sla_seconds=None,
                urgency=req.get("urgency", "routine"),
            )
            wo.handler_workflow_type = capability.workflow_type
            wo.handler_domain = capability.domain
            wo.status = WorkOrderStatus.DISPATCHED
            wo.dispatched_at = time.time()
            self.store.save_work_order(wo)
            work_order_ids.append(wo.work_order_id)
            wo_need_map[wo.work_order_id] = req.get("need", "")

            self._log(f"    → DISPATCH [{req.get('need', '?')}]: "
                       f"{wo.work_order_id} → "
                       f"{capability.workflow_type}/{capability.domain}")

        # ── Suspend the source ──
        # Resume at the same step that paused. It already ran and
        # produced the ResourceRequest, but on resume it will re-run
        # with the provider results now available in state.input.delegation.
        # Its prior output (with the ResourceRequest) is stripped by
        # prepare_resume_state.
        resume_step = step_name

        suspension = Suspension.create(
            instance_id=instance.instance_id,
            suspended_at_step=resume_step,
            state_snapshot=self._compact_state_for_suspension(partial_state),
            unresolved_needs=[r for r in requests],
            work_order_ids=work_order_ids,
        )
        suspension.wo_need_map = wo_need_map
        self.store.save_suspension(suspension)

        instance.status = InstanceStatus.SUSPENDED
        instance.updated_at = time.time()
        instance.resume_nonce = suspension.resume_nonce
        instance.pending_work_orders = work_order_ids
        self.store.save_instance(instance)

        self.store.log_action(
            instance_id=instance.instance_id,
            correlation_id=instance.correlation_id,
            action_type="interrupted_for_resources",
            details={
                "suspended_at_step": step_name,
                "matched_needs": [r.get("need") for r, _ in matched],
                "unmatched_needs": [r.get("need", "?") for r in unmatched],
                "work_order_ids": work_order_ids,
                "resume_nonce": suspension.resume_nonce,
            },
            idempotency_key=f"interrupt:{instance.instance_id}:{step_name}",
        )

        self._log(f"    source SUSPENDED at '{resume_step}', "
                   f"waiting for {len(work_order_ids)} provider(s)")

        # ── Dispatch providers: parallel for independent, staged for dependent ──
        lineage = instance.lineage + [
            f"{instance.workflow_type}:{instance.instance_id}"
        ]

        # Build dependency graph: which needs must complete before others
        need_to_wo = {}  # need_name → work_order_id
        for (req, capability), wo_id in zip(matched, work_order_ids):
            need_to_wo[req.get("need", "")] = wo_id

        # Partition into dispatchable now (no unmet deps) vs deferred
        dispatched_needs = set()
        deferred = []
        dispatch_now = []

        for (req, capability), wo_id in zip(matched, work_order_ids):
            deps = req.get("depends_on", [])
            unmet = [d for d in deps if d in need_to_wo and d not in dispatched_needs]
            if unmet:
                deferred.append((req, capability, wo_id, deps))
                self._log(f"    ⏳ '{req.get('need')}' deferred — "
                           f"depends on: {unmet}")
            else:
                dispatch_now.append((req, capability, wo_id))

        # Store deferred needs in suspension for staged dispatch
        # Only put dispatched work orders in work_order_ids — deferred
        # ones stay in deferred_needs until their deps are met
        dispatched_wo_ids = [wo_id for _, _, wo_id in dispatch_now]
        if deferred:
            suspension.deferred_needs = [
                {
                    "need": req.get("need"),
                    "context": req.get("context", {}),
                    "depends_on": deps,
                    "work_order_id": wo_id,
                    "capability_need_type": capability.need_type,
                }
                for req, capability, wo_id, deps in deferred
            ]
        # Update work_order_ids to only include dispatched ones
        suspension.work_order_ids = dispatched_wo_ids
        self.store.save_suspension(suspension)

        all_handlers_done = True

        # ── Optimizer consultation (Spec v1.1 Section 9) ──
        # If the optimizer is available AND we have a resource registry,
        # consult it for optimal provider selection. The optimizer
        # enhances the capability routing with resource-level optimization.
        # Without it, the existing direct dispatch path is preserved.
        optimizer_decisions = {}
        if self._optimizer and self._resource_registry and len(dispatch_now) > 1:
            opt_config = self._get_optimization_config(instance.domain)
            if opt_config:
                try:
                    from cognitive_core.coordinator.ddd import DDDWorkOrder
                    ddd_wos = []
                    for req, capability, wo_id in dispatch_now:
                        ddd_wo = DDDWorkOrder.create(
                            instance.workflow_type,
                            instance.correlation_id,
                            req.get("need", ""),
                            priority=req.get("urgency", "routine"),
                            sla_seconds=self._get_tier_sla(
                                self._resolve_governance_tier(instance.domain)
                            ) or 3600.0,
                        )
                        ddd_wos.append((ddd_wo, wo_id))

                    decisions = self._optimizer.dispatch(
                        [dwo for dwo, _ in ddd_wos],
                        instance.workflow_type,
                        instance.domain,
                        opt_config,
                    )
                    for (dwo, wo_id), decision in zip(ddd_wos, decisions):
                        optimizer_decisions[wo_id] = decision
                    self._log(
                        f"    optimizer consulted: {len(decisions)} decisions, "
                        f"{sum(1 for d in decisions if d.selected_resource_id)} assigned"
                    )
                except Exception as e:
                    self._log(f"    optimizer error (fallback to direct dispatch): {e}")

        for req, capability, wo_id in dispatch_now:
            wo = self.store.get_work_order(wo_id)
            if not wo:
                continue

            done = self._dispatch_provider(
                instance, wo, req, capability, lineage, model, temperature
            )
            # Log optimizer decision if available
            opt_decision = optimizer_decisions.get(wo_id)
            if opt_decision and opt_decision.selected_resource_id:
                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="optimizer_resource_selection",
                    details={
                        "work_order_id": wo_id,
                        "selected_resource": opt_decision.selected_resource_id,
                        "tier": opt_decision.tier,
                        "reservation_id": opt_decision.reservation_id,
                    },
                    idempotency_key=f"opt:{wo_id}",
                )
            if done:
                dispatched_needs.add(req.get("need", ""))
            else:
                all_handlers_done = False

        # ── If all dispatched providers completed, check deferred ──
        if all_handlers_done and work_order_ids:
            self._try_resume_after_all_providers(instance, suspension)

    def _dispatch_provider(
        self,
        instance: InstanceState,
        wo: WorkOrder,
        req: dict,
        capability,
        lineage: list[str],
        model: str = "default",
        temperature: float = 0.1,
    ) -> bool:
        """
        Dispatch a single provider for a resource request.
        Returns True if the provider completed synchronously.
        """
        if capability.provider_type == "workflow":
            try:
                handler_id = self.start(
                    workflow_type=capability.workflow_type,
                    domain=capability.domain,
                    case_input=req.get("context", {}),
                    lineage=lineage,
                    correlation_id=instance.correlation_id,
                    model=model,
                    temperature=temperature,
                )
                wo.handler_instance_id = handler_id
                handler = self.store.get_instance(handler_id)

                if handler and handler.status == InstanceStatus.COMPLETED:
                    wo.status = WorkOrderStatus.COMPLETED
                    wo.completed_at = time.time()
                    wo.result = WorkOrderResult(
                        work_order_id=wo.work_order_id,
                        status="completed",
                        outputs=handler.result or {},
                        completed_at=time.time(),
                    )
                    self.store.save_work_order(wo)
                    self._log(f"    provider {handler_id} → completed")
                    # Record for oscillation detection
                    self._record_work_order_completion(
                        instance, wo,
                        need_name=req.get("need", ""),
                        accepted=True,
                    )
                    # Saga: register side effects from completed work order
                    if self._saga_coordinator and handler and handler.result:
                        self._register_saga_side_effects(
                            instance.correlation_id,
                            wo.work_order_id,
                            handler.result,
                        )
                    return True
                elif handler and handler.status == InstanceStatus.SUSPENDED:
                    wo.status = WorkOrderStatus.RUNNING
                    self.store.save_work_order(wo)
                    self._log(f"    provider {handler_id} → suspended (waiting)")
                    return False
                else:
                    wo.status = WorkOrderStatus.FAILED
                    wo.result = WorkOrderResult(
                        work_order_id=wo.work_order_id,
                        status="failed",
                        error=f"Provider in state: {handler.status.value if handler else 'none'}",
                        completed_at=time.time(),
                    )
                    self.store.save_work_order(wo)
                    self._log(f"    provider {handler_id} → failed")
                    # Record for oscillation detection
                    self._record_work_order_completion(
                        instance, wo,
                        need_name=req.get("need", ""),
                        accepted=False,
                        rejection_reason=f"Provider in state: {handler.status.value if handler else 'none'}",
                    )
                    return True  # Failed is still "done" for fan-in purposes

            except Exception as e:
                wo.status = WorkOrderStatus.FAILED
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="failed",
                    error=str(e)[:500],
                    completed_at=time.time(),
                )
                self.store.save_work_order(wo)
                self._log(f"    provider FAILED: {e}")
                # Record for oscillation detection
                self._record_work_order_completion(
                    instance, wo,
                    need_name=req.get("need", ""),
                    accepted=False,
                    rejection_reason=str(e)[:200],
                )
                return True

        elif capability.provider_type == "human_task":
            task = Task.create(
                task_type=TaskType.RESOURCE_REQUEST,
                queue=capability.queue,
                instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                workflow_type=instance.workflow_type,
                domain=instance.domain,
                payload={
                    "need": req.get("need"),
                    "reason": req.get("reason"),
                    "context": req.get("context", {}),
                    "work_order_id": wo.work_order_id,
                },
                callback=TaskCallback(
                    method="fulfill_resource",
                    instance_id=instance.instance_id,
                ),
                priority=1 if req.get("urgency") == "critical" else 0,
            )
            self.tasks.publish(task)
            wo.status = WorkOrderStatus.RUNNING
            self.store.save_work_order(wo)
            self._log(f"    → human task published to '{capability.queue}'")
            return False

        return False

    def _find_capability(self, need: str):
        """Look up a capability by need type."""
        for cap in self.policy.capabilities:
            if cap.need_type == need:
                return cap
        return None

    def _try_resume_after_all_providers(
        self,
        instance: InstanceState,
        suspension: Suspension,
    ):
        """
        Check if all work orders for a suspended instance are done.

        Three cases:
        1. All work orders done, no deferred needs → resume source
        2. All dispatched work orders done, deferred needs remain →
           dispatch the next wave (needs whose deps are now met)
        3. Some work orders still running → do nothing, wait
        """
        # Build wo_id → need_name mapping
        wo_to_need = getattr(suspension, 'wo_need_map', {}) or {}

        completed_needs = set()
        all_dispatched_done = True
        external_input = {}

        for wo_id in suspension.work_order_ids:
            wo = self.store.get_work_order(wo_id)
            if not wo:
                continue
            if wo.status == WorkOrderStatus.COMPLETED and wo.result:
                external_input[wo_id] = wo.result.outputs
                # Also key by need name so LLM sees meaningful keys
                need_name = wo_to_need.get(wo_id, "")
                if need_name:
                    external_input[need_name] = wo.result.outputs
                    completed_needs.add(need_name)
            elif wo.status == WorkOrderStatus.FAILED and wo.result:
                external_input[wo_id] = {
                    "_error": wo.result.error,
                    "_status": "failed",
                }
                need_name = wo_to_need.get(wo_id, "")
                if need_name:
                    external_input[need_name] = {
                        "_error": wo.result.error,
                        "_status": "failed",
                    }
                    completed_needs.add(need_name)
            else:
                all_dispatched_done = False

        if not all_dispatched_done:
            return  # Still waiting

        # ── Check for deferred needs that can now be dispatched ──
        deferred = getattr(suspension, 'deferred_needs', []) or []
        if deferred:
            still_deferred = []
            dispatch_wave = []

            for d in deferred:
                deps = d.get("depends_on", [])
                unmet = [dep for dep in deps if dep not in completed_needs]
                if unmet:
                    still_deferred.append(d)
                else:
                    dispatch_wave.append(d)

            if dispatch_wave:
                self._log(f"  ▶ DISPATCHING NEXT WAVE: {len(dispatch_wave)} deferred need(s)")
                lineage = instance.lineage + [
                    f"{instance.workflow_type}:{instance.instance_id}"
                ]
                for d in dispatch_wave:
                    wo_id = d.get("work_order_id")
                    wo = self.store.get_work_order(wo_id) if wo_id else None
                    if not wo:
                        continue

                    # Enrich context with results from dependencies
                    enriched_context = dict(d.get("context", {}))
                    enriched_context["_dependency_results"] = external_input

                    capability = self._find_capability(d.get("capability_need_type", d.get("need", "")))
                    if capability:
                        req = {
                            "need": d.get("need"),
                            "context": enriched_context,
                            "urgency": d.get("urgency", "routine"),
                        }
                        done = self._dispatch_provider(
                            instance, wo, req, capability, lineage
                        )
                        # Track this work order and its need
                        suspension.work_order_ids.append(wo_id)
                        suspension.wo_need_map[wo_id] = d.get("need", "")
                        if done:
                            completed_needs.add(d.get("need", ""))

                # Update deferred list and re-check
                suspension.deferred_needs = still_deferred
                self.store.save_suspension(suspension)

                # If this wave also completed synchronously, recurse
                if not still_deferred:
                    self._try_resume_after_all_providers(instance, suspension)
                return

        # ── All done (no deferred remaining) → resume source ──

        # ── Resilience: Revalidation (Failure Mode 1) ──
        # Before resuming, verify the world hasn't changed in ways
        # that invalidate the original Need's assumptions.
        if self._revalidator:
            reval_result = self._revalidator.revalidate(
                instance.instance_id, external_input,
            )
            if reval_result.verdict.value == "invalidated":
                self._log(
                    f"  ⚠ REVALIDATION FAILED: {reval_result.reason} "
                    f"— escalating to HITL"
                )
                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="resume_revalidation_failed",
                    details={
                        "verdict": reval_result.verdict.value,
                        "reason": reval_result.reason,
                        "checks_run": reval_result.checks_run,
                    },
                )
                return  # Do NOT resume — escalate to HITL
            elif reval_result.verdict.value == "stale":
                self._log(
                    f"  ⚠ STALE CONTEXT: enriching with {len(reval_result.enrichment)} fields"
                )
                external_input.update(reval_result.enrichment)
                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="resume_revalidation_stale",
                    details={
                        "enrichment_keys": list(reval_result.enrichment.keys()),
                        "checks_run": reval_result.checks_run,
                    },
                )

        # ── Hardening: Partial Failure (Requirement 3) ──
        # Check if any work orders failed and resolve per-need policies.
        if self._partial_failure_handler:
            for wo_id in suspension.work_order_ids:
                wo = self.store.get_work_order(wo_id)
                if wo and wo.status == WorkOrderStatus.FAILED and wo.result:
                    need_name = wo_to_need.get(wo_id, "")
                    error_class = "retryable"  # default
                    if wo.result.error and "permanent" in str(wo.result.error).lower():
                        error_class = "permanent"
                    elif wo.result.error and "degraded" in str(wo.result.error).lower():
                        error_class = "degraded"

                    decision = self._partial_failure_handler.resolve(
                        wo.work_order_id, need_name,
                        instance.correlation_id, error_class,
                    )
                    self.store.log_action(
                        instance_id=instance.instance_id,
                        correlation_id=instance.correlation_id,
                        action_type="partial_failure_resolved",
                        details={
                            "work_order_id": wo_id,
                            "need_type": need_name,
                            "error_class": error_class,
                            "action": decision.action.value,
                            "reason": decision.reason,
                        },
                    )
                    if decision.action.value == "degrade" and decision.degraded_output:
                        # Inject degraded output so agent can proceed
                        external_input[wo_id] = decision.degraded_output
                        if need_name:
                            external_input[need_name] = decision.degraded_output
                    elif decision.action.value == "abort":
                        self._log(f"  ⚠ ABORT: partial failure policy aborted saga")
                        # Trigger saga compensation if available
                        if self._saga_coordinator:
                            self._saga_coordinator.compensate(
                                instance.correlation_id,
                                handler=None,  # escalate to HITL
                                failed_work_order_id=wo_id,
                            )
                        return  # Do NOT resume

        self._log(f"  ▶ ALL PROVIDERS DONE — resuming {instance.instance_id}")
        self.resume(
            instance_id=instance.instance_id,
            external_input=external_input,
            resume_nonce=suspension.resume_nonce,
        )

    # ─── Delegation Execution ────────────────────────────────────────

    def _execute_all_blocking_delegations(
        self,
        source: InstanceState,
        decisions: list[DelegationDecision],
        source_state: dict[str, Any],
    ):
        """
        Dispatch all wait_for_result delegations under a single suspension.

        Creates work orders for every decision, suspends the source once with
        all work_order_ids, then starts each handler. Because the suspension
        lists all WO IDs, _check_delegation_completion only resumes the source
        after every handler has finished — no intermediate runs of
        generate_final_report.

        If any handler ends up suspended (governance gate), the source stays
        suspended. When that handler is eventually approved, _check_delegation_
        completion picks up all sibling WOs and resumes the source with the
        combined results once everything is resolved.
        """
        # ── Build work orders ──────────────────────────────────────────
        valid_pairs: list[tuple[DelegationDecision, WorkOrder]] = []
        for decision in decisions:
            errors = self.policy.validate_work_order_inputs(
                decision.contract_name, decision.inputs
            )
            if errors:
                self._log(f"  ⚠ delegation {decision.policy_name} skipped: "
                           f"contract validation failed: {errors}")
                self.store.log_action(
                    instance_id=source.instance_id,
                    correlation_id=source.correlation_id,
                    action_type="delegation_skipped",
                    details={"policy": decision.policy_name, "errors": errors},
                )
                continue

            wo = WorkOrder.create(
                requester_instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                contract_name=decision.contract_name,
                contract_version=decision.contract_version,
                inputs=decision.inputs,
                sla_seconds=decision.sla_seconds,
            )
            wo.handler_workflow_type = decision.target_workflow
            wo.handler_domain = decision.target_domain
            wo.status = WorkOrderStatus.DISPATCHED
            wo.dispatched_at = time.time()
            self.store.save_work_order(wo)

            mode_label = "wait"
            self._log(f"  → DELEGATION [{mode_label}]: {decision.policy_name}")
            self._log(f"    work_order: {wo.work_order_id}")
            self._log(f"    target: {decision.target_workflow}/{decision.target_domain}")
            self._log(f"    contract: {decision.contract_name} v{decision.contract_version}")
            tool_keys = [k for k, v in decision.inputs.items()
                         if isinstance(v, (dict, list)) and k.startswith("get_")]
            scalar_keys = [k for k in decision.inputs.keys() if k not in tool_keys]
            self._log(f"    inputs: {scalar_keys}")
            if tool_keys:
                self._log(f"    tool data: [{', '.join(tool_keys)}]")
            else:
                self._log(f"    ⚠ NO tool data in inputs — handler retrieve may fail")
                self._log(f"      hint: add get_* entries to delegation inputs in config.yaml")

            self.store.log_action(
                instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                action_type="delegation_dispatched",
                details={
                    "work_order_id": wo.work_order_id,
                    "policy": decision.policy_name,
                    "mode": "wait_for_result",
                    "target_workflow": decision.target_workflow,
                    "target_domain": decision.target_domain,
                    "contract": decision.contract_name,
                },
                idempotency_key=f"deleg:{source.instance_id}:{decision.policy_name}",
            )
            valid_pairs.append((decision, wo))

        if not valid_pairs:
            return

        wo_ids = [wo.work_order_id for _, wo in valid_pairs]

        # Determine resume step (all decisions should share resume_at_step)
        resume_step = valid_pairs[0][0].resume_at_step
        if not resume_step:
            steps = source_state.get("steps", [])
            if steps:
                resume_step = steps[-1]["step_name"]

        # ── Single suspension for all work orders ──────────────────────
        suspension = Suspension.create(
            instance_id=source.instance_id,
            suspended_at_step=resume_step,
            state_snapshot=self._compact_state_for_suspension(source_state),
            work_order_ids=wo_ids,
        )
        self.store.save_suspension(suspension)

        source.status = InstanceStatus.SUSPENDED
        source.updated_at = time.time()
        source.resume_nonce = suspension.resume_nonce
        source.pending_work_orders = wo_ids
        self.store.save_instance(source)

        self._log(f"    source SUSPENDED at '{resume_step}', "
                   f"waiting for {len(wo_ids)} work order(s)")

        self.store.log_action(
            instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            action_type="suspended_for_delegation",
            details={
                "work_order_ids": wo_ids,
                "resume_step": resume_step,
                "resume_nonce": suspension.resume_nonce,
            },
        )

        # ── Dispatch all handlers ───────────────────────────────────────
        lineage = source.lineage + [f"{source.workflow_type}:{source.instance_id}"]
        any_suspended = False

        for decision, wo in valid_pairs:
            try:
                handler_id = self.start(
                    workflow_type=decision.target_workflow,
                    domain=decision.target_domain,
                    case_input=decision.inputs,
                    lineage=lineage,
                    correlation_id=source.correlation_id,
                    governance_tier_override=decision.governance_tier,
                )
                wo.handler_instance_id = handler_id
                handler = self.store.get_instance(handler_id)

                if handler and handler.status == InstanceStatus.COMPLETED:
                    wo.status = WorkOrderStatus.COMPLETED
                    wo.completed_at = time.time()
                    wo.result = WorkOrderResult(
                        work_order_id=wo.work_order_id,
                        status="completed",
                        outputs=handler.result or {},
                        completed_at=time.time(),
                    )
                elif handler and handler.status == InstanceStatus.SUSPENDED:
                    wo.status = WorkOrderStatus.RUNNING
                    any_suspended = True
                else:
                    wo.status = WorkOrderStatus.FAILED
                    wo.result = WorkOrderResult(
                        work_order_id=wo.work_order_id,
                        status="failed",
                        error=f"Handler in unexpected state: "
                              f"{handler.status.value if handler else 'none'}",
                        completed_at=time.time(),
                    )
                self.store.save_work_order(wo)
                self._log(f"    handler: {handler_id} → "
                           f"{handler.status.value if handler else 'unknown'}")

            except Exception as e:
                wo.status = WorkOrderStatus.FAILED
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="failed",
                    error=str(e),
                    completed_at=time.time(),
                )
                self.store.save_work_order(wo)
                self.store.log_action(
                    instance_id=source.instance_id,
                    correlation_id=source.correlation_id,
                    action_type="delegation_handler_failed",
                    details={
                        "policy": decision.policy_name,
                        "target": f"{decision.target_workflow}/{decision.target_domain}",
                        "mode": "wait_for_result",
                        "error": str(e)[:500],
                        "work_order_id": wo.work_order_id,
                    },
                )
                self._log(f"    handler FAILED: {e}")

        # If any handler is suspended (governance gate), source stays suspended.
        # _check_delegation_completion will resume source when all handlers resolve.
        if any_suspended:
            return

        # All handlers completed synchronously — resume source once with all results.
        combined_input = {}
        for _, wo in valid_pairs:
            result_key = wo.handler_workflow_type or wo.work_order_id
            if wo.status == WorkOrderStatus.COMPLETED and wo.result:
                combined_input[result_key] = wo.result.outputs
            elif wo.status == WorkOrderStatus.FAILED and wo.result:
                combined_input[result_key] = {
                    "_error": wo.result.error,
                    "_status": wo.result.status,
                }

        self._log(f"  ↩ all {len(valid_pairs)} handler(s) done → resuming {source.instance_id}")
        self.resume(
            instance_id=source.instance_id,
            external_input=combined_input,
            resume_nonce=suspension.resume_nonce,
        )

    def _execute_delegation(
        self,
        source: InstanceState,
        decision: DelegationDecision,
        source_state: dict[str, Any],
    ):
        """
        Execute a delegation in one of two modes:

        fire_and_forget: Source is already completed. Delegated workflow
            runs independently. Results enrich the case record via
            correlation chain but source doesn't wait.

        wait_for_result: Source suspends. Delegated workflow runs.
            When it completes, source resumes at the specified step
            with the delegation result injected into state.
        """

        # Validate inputs against contract
        errors = self.policy.validate_work_order_inputs(
            decision.contract_name, decision.inputs
        )
        if errors:
            self._log(f"  ⚠ delegation {decision.policy_name} skipped: "
                       f"contract validation failed: {errors}")
            self.store.log_action(
                instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                action_type="delegation_skipped",
                details={"policy": decision.policy_name, "errors": errors},
            )
            return

        # Create work order
        wo = WorkOrder.create(
            requester_instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            contract_name=decision.contract_name,
            contract_version=decision.contract_version,
            inputs=decision.inputs,
            sla_seconds=decision.sla_seconds,
        )
        wo.handler_workflow_type = decision.target_workflow
        wo.handler_domain = decision.target_domain
        wo.status = WorkOrderStatus.DISPATCHED
        wo.dispatched_at = time.time()
        self.store.save_work_order(wo)

        mode_label = "wait" if decision.mode == "wait_for_result" else "fire"
        self._log(f"  → DELEGATION [{mode_label}]: {decision.policy_name}")
        self._log(f"    work_order: {wo.work_order_id}")
        self._log(f"    target: {decision.target_workflow}/{decision.target_domain}")
        self._log(f"    contract: {decision.contract_name} v{decision.contract_version}")

        # ── Delegation input diagnostics ──
        tool_keys = [k for k, v in decision.inputs.items()
                     if isinstance(v, (dict, list)) and k.startswith("get_")]
        scalar_keys = [k for k in decision.inputs.keys() if k not in tool_keys]
        self._log(f"    inputs: {scalar_keys}")
        if tool_keys:
            self._log(f"    tool data: [{', '.join(tool_keys)}]")
        else:
            self._log(f"    ⚠ NO tool data in inputs — handler retrieve may fail")
            self._log(f"      hint: add get_* entries to delegation inputs in config.yaml")

        # Log the delegation dispatch — idempotency key prevents double dispatch.
        # If log_action returns False, this delegation already fired (concurrent thread).
        # Skip the handler start to prevent duplicate workflows.
        dispatched = self.store.log_action(
            instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            action_type="delegation_dispatched",
            details={
                "work_order_id": wo.work_order_id,
                "policy": decision.policy_name,
                "mode": decision.mode,
                "target_workflow": decision.target_workflow,
                "target_domain": decision.target_domain,
                "contract": decision.contract_name,
            },
            idempotency_key=f"deleg:{source.instance_id}:{decision.policy_name}",
        )

        if not dispatched:
            self._log(f"  ⚠ delegation {decision.policy_name} already dispatched (concurrent thread) — skipping")
            return

        if decision.mode == "wait_for_result":
            self._execute_blocking_delegation(
                source, decision, source_state, wo
            )
        else:
            self._execute_fire_and_forget_delegation(
                source, decision, wo
            )

    def _execute_fire_and_forget_delegation(
        self,
        source: InstanceState,
        decision: DelegationDecision,
        wo: WorkOrder,
    ):
        """Source already completed. Spawn handler independently."""
        lineage = source.lineage + [
            f"{source.workflow_type}:{source.instance_id}"
        ]

        # Guard: if handler already started (concurrent thread), skip.
        # Refresh work order from DB to get the latest handler_instance_id.
        fresh_wo = self.store.get_work_order(wo.work_order_id)
        if fresh_wo and fresh_wo.handler_instance_id:
            self._log(f"  ⚠ delegation {decision.policy_name} handler already started "
                      f"({fresh_wo.handler_instance_id}) — skipping duplicate start")
            return

        # Atomically claim the work order by setting a placeholder handler_id.
        # If another thread already set it, the UPDATE will affect 0 rows.
        claimed = self.store.claim_work_order_handler(wo.work_order_id)
        if not claimed:
            self._log(f"  ⚠ delegation {decision.policy_name} handler claimed by another thread — skipping")
            return

        try:
            handler_id = self.start(
                workflow_type=decision.target_workflow,
                domain=decision.target_domain,
                case_input=decision.inputs,
                lineage=lineage,
                correlation_id=source.correlation_id,
                governance_tier_override=decision.governance_tier,
            )

            wo.handler_instance_id = handler_id
            handler = self.store.get_instance(handler_id)
            if handler and handler.status == InstanceStatus.COMPLETED:
                wo.status = WorkOrderStatus.COMPLETED
                wo.completed_at = time.time()
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="completed",
                    outputs=handler.result or {},
                    completed_at=time.time(),
                )
            elif handler and handler.status == InstanceStatus.SUSPENDED:
                wo.status = WorkOrderStatus.RUNNING
            self.store.save_work_order(wo)
            self._log(f"    handler: {handler_id} → {handler.status.value if handler else 'unknown'}")

        except Exception as e:
            wo.status = WorkOrderStatus.FAILED
            wo.result = WorkOrderResult(
                work_order_id=wo.work_order_id,
                status="failed",
                error=str(e),
                completed_at=time.time(),
            )
            self.store.save_work_order(wo)
            self.store.log_action(
                instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                action_type="delegation_handler_failed",
                details={
                    "policy": decision.policy_name,
                    "target": f"{decision.target_workflow}/{decision.target_domain}",
                    "mode": "fire_and_forget",
                    "error": str(e)[:500],
                    "work_order_id": wo.work_order_id,
                },
            )
            self._log(f"    handler FAILED: {e}")

    def _execute_blocking_delegation(
        self,
        source: InstanceState,
        decision: DelegationDecision,
        source_state: dict[str, Any],
        wo: WorkOrder,
    ):
        """
        Source suspends. Spawn handler. When handler completes,
        coordinator resumes source with the result.

        In-process (Phase 2): handler runs synchronously, then
        coordinator resumes source immediately.

        Distributed (Phase 4): handler dispatched to queue,
        completion callback triggers resume.
        """
        # Determine resume step
        resume_step = decision.resume_at_step
        if not resume_step:
            # Default: resume at the last step that ran
            steps = source_state.get("steps", [])
            if steps:
                resume_step = steps[-1]["step_name"]

        # Suspend the source
        suspension = Suspension.create(
            instance_id=source.instance_id,
            suspended_at_step=resume_step,
            state_snapshot=self._compact_state_for_suspension(source_state),
            work_order_ids=[wo.work_order_id],
        )
        self.store.save_suspension(suspension)

        source.status = InstanceStatus.SUSPENDED
        source.updated_at = time.time()
        source.resume_nonce = suspension.resume_nonce
        source.pending_work_orders = [wo.work_order_id]
        self.store.save_instance(source)

        self._log(f"    source SUSPENDED at '{resume_step}', waiting for {wo.work_order_id}")

        self.store.log_action(
            instance_id=source.instance_id,
            correlation_id=source.correlation_id,
            action_type="suspended_for_delegation",
            details={
                "work_order_id": wo.work_order_id,
                "resume_step": resume_step,
                "resume_nonce": suspension.resume_nonce,
            },
        )

        # Execute the handler
        lineage = source.lineage + [
            f"{source.workflow_type}:{source.instance_id}"
        ]

        try:
            handler_id = self.start(
                workflow_type=decision.target_workflow,
                domain=decision.target_domain,
                case_input=decision.inputs,
                lineage=lineage,
                correlation_id=source.correlation_id,
                governance_tier_override=decision.governance_tier,
            )

            wo.handler_instance_id = handler_id
            handler = self.store.get_instance(handler_id)

            if handler and handler.status == InstanceStatus.COMPLETED:
                # Handler completed — resume the source
                wo.status = WorkOrderStatus.COMPLETED
                wo.completed_at = time.time()
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="completed",
                    outputs=handler.result or {},
                    completed_at=time.time(),
                )
                self.store.save_work_order(wo)

                self._log(f"    handler: {handler_id} → completed, resuming source")
                self._resume_after_delegation(source, wo, suspension)

            elif handler and handler.status == InstanceStatus.SUSPENDED:
                # Handler itself is suspended (e.g., governance hold)
                # Source stays suspended. When handler eventually completes
                # and its approve() is called, _on_delegation_completed
                # will be triggered to resume the source.
                wo.status = WorkOrderStatus.RUNNING
                self.store.save_work_order(wo)
                self._log(f"    handler: {handler_id} → suspended (source remains suspended)")

            else:
                wo.status = WorkOrderStatus.FAILED
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="failed",
                    error=f"Handler in unexpected state: {handler.status.value if handler else 'none'}",
                    completed_at=time.time(),
                )
                self.store.save_work_order(wo)
                # Resume source without the result — it continues with what it has
                self._log(f"    handler failed, resuming source without result")
                self._resume_after_delegation(source, wo, suspension)

        except Exception as e:
            wo.status = WorkOrderStatus.FAILED
            wo.result = WorkOrderResult(
                work_order_id=wo.work_order_id,
                status="failed",
                error=str(e),
                completed_at=time.time(),
            )
            self.store.save_work_order(wo)
            self.store.log_action(
                instance_id=source.instance_id,
                correlation_id=source.correlation_id,
                action_type="delegation_handler_failed",
                details={
                    "policy": decision.policy_name,
                    "target": f"{decision.target_workflow}/{decision.target_domain}",
                    "mode": "wait_for_result",
                    "error": str(e)[:500],
                    "work_order_id": wo.work_order_id,
                },
            )
            self._log(f"    handler FAILED: {e}, resuming source without result")
            self._resume_after_delegation(source, wo, suspension)

    def _resume_after_delegation(
        self,
        source: InstanceState,
        wo: WorkOrder,
        suspension: Suspension,
    ):
        """
        Resume a source workflow after a blocking delegation completes.
        Injects the delegation result into state and resumes at the
        suspended step.

        Results are keyed by handler_workflow_type so workflow templates
        can reference them: ${input.delegation.fraud_regulatory_review}
        """
        # Build external input keyed by handler workflow type
        external_input = {}
        result_key = wo.handler_workflow_type or wo.work_order_id
        if wo.result and wo.result.outputs:
            external_input[result_key] = wo.result.outputs
        elif wo.result and wo.result.error:
            external_input[result_key] = {
                "_error": wo.result.error,
                "_status": wo.result.status,
            }

        self._log(f"  ▶ RESUMING {source.instance_id} at '{suspension.suspended_at_step}'")

        self.resume(
            instance_id=source.instance_id,
            external_input=external_input,
            resume_nonce=suspension.resume_nonce,
        )

    def _check_delegation_completion(self, instance: InstanceState):
        """
        Check if this instance was a handler for a blocking delegation.
        If so, update the work order and check if the requester can resume.

        For single-WO suspensions (forward delegation), resumes immediately.
        For multi-WO suspensions (demand-driven fan-out), waits until ALL
        work orders are resolved before resuming the source.
        """
        # Find work orders where this instance is the handler
        all_wos = self.store.get_work_orders_for_requester_or_handler(
            instance.instance_id
        )

        for wo in all_wos:
            if wo.handler_instance_id != instance.instance_id:
                continue
            if wo.status in (WorkOrderStatus.COMPLETED, WorkOrderStatus.FAILED):
                continue  # Already resolved

            # Atomically mark work order completed — first thread wins.
            # If another thread already completed it, skip to avoid duplicate resume.
            import json as _json
            completed_at = time.time()
            result_outputs = instance.result or {}
            result_json = _json.dumps({
                "work_order_id": wo.work_order_id,
                "status": "completed",
                "outputs": result_outputs,
                "error": None,
                "completed_at": completed_at,
            })
            completed = self.store.complete_work_order(wo.work_order_id, result_json, completed_at)
            if not completed:
                self._log(f"  ⚠ work order {wo.work_order_id} already completed by another thread — skipping")
                continue

            # Refresh from DB to get consistent state
            wo = self.store.get_work_order(wo.work_order_id)
            if not wo:
                continue

            # Check if requester is suspended waiting
            requester = self.store.get_instance(wo.requester_instance_id)
            if not requester or requester.status != InstanceStatus.SUSPENDED:
                continue

            suspension = self.store.get_suspension(requester.instance_id)
            if not suspension or wo.work_order_id not in suspension.work_order_ids:
                continue

            # Check if ALL work orders for this suspension are resolved
            all_resolved = True
            combined_input = {}
            for sibling_wo_id in suspension.work_order_ids:
                sibling_wo = self.store.get_work_order(sibling_wo_id)
                if not sibling_wo:
                    continue
                result_key = sibling_wo.handler_workflow_type or sibling_wo_id
                if sibling_wo.status == WorkOrderStatus.COMPLETED and sibling_wo.result:
                    combined_input[result_key] = sibling_wo.result.outputs
                elif sibling_wo.status == WorkOrderStatus.FAILED and sibling_wo.result:
                    combined_input[result_key] = {
                        "_error": sibling_wo.result.error,
                        "_status": "failed",
                    }
                else:
                    all_resolved = False

            if all_resolved:
                self._log(f"  ↩ all {len(suspension.work_order_ids)} provider(s) done → "
                           f"resuming {requester.instance_id}")
                self.resume(
                    instance_id=requester.instance_id,
                    external_input=combined_input,
                    resume_nonce=suspension.resume_nonce,
                )
            else:
                pending = [
                    wid for wid in suspension.work_order_ids
                    if wid not in combined_input
                ]
                self._log(f"  ⏳ {instance.instance_id} done, but "
                           f"{len(pending)} provider(s) still pending for "
                           f"{requester.instance_id}")

    # ─── Workflow Execution ──────────────────────────────────────────

    def _execute_workflow(
        self,
        instance: InstanceState,
        case_input: dict[str, Any],
        model: str = "default",
        temperature: float = 0.1,
    ) -> dict[str, Any] | Any:
        """
        Execute a workflow using step-by-step execution.

        Returns either:
          - dict (WorkflowState): workflow completed normally
          - StepInterrupt: workflow was interrupted by a resource request

        The stepper runs each step, then calls our callback between steps.
        The callback checks for blocking ResourceRequests in step output.
        If found, the stepper pauses and returns a StepInterrupt.

        If LangGraph is not installed, falls back to simulated execution
        using the case fixture data and workflow step definitions.
        """
        from cognitive_core.engine.composer import load_three_layer
        from cognitive_core.engine.trace import set_trace
        from cognitive_core.engine.devs_executor import (
            step_execute, resource_request_callback,
            no_interrupt_callback, StepResult,
        )

        # Resolve workflow and domain file paths
        workflow_path = self._find_workflow(instance.workflow_type)
        domain_path = self._find_domain(instance.domain)

        # Load three-layer config
        config, _ = load_three_layer(workflow_path, domain_path)
        is_agentic = config.get("mode") == "agentic"

        # Build tool registry
        tool_registry = self._build_tool_registry(case_input)

        # ── Diagnostics: log available tools and input keys ──
        available_tools = tool_registry.list_tools() if tool_registry else []
        input_keys = [k for k in case_input.keys() if isinstance(case_input[k], (dict, list))]
        self._log(f"  tools: [{', '.join(available_tools)}]")
        if not available_tools:
            self._log(f"  ⚠ NO TOOLS AVAILABLE — retrieve steps will find nothing")
            self._log(f"    input keys: {list(case_input.keys())}")
            self._log(f"    hint: case_input needs get_* keys with dict values for case registry")

        # Check for tool/workflow mismatch: workflow specifies tools not in registry
        step_specs = config.get("steps", [])
        for step in step_specs:
            if step.get("primitive") == "retrieve":
                spec_text = step.get("params", {}).get("specification", "")
                # Quick parse: look for get_* or tool-like references
                import re
                referenced_tools = set(re.findall(r'\b(get_\w+)\b', spec_text))
                missing = referenced_tools - set(available_tools)
                if missing:
                    self._log(f"  ⚠ {step['name']}: references tools not in registry: {missing}")

        # ── Wire engine tracing into coordinator output ──
        if self.verbose:
            from cognitive_core.engine.trace import get_trace
            existing = get_trace()
            coord_trace = _CoordinatorTrace(self, instance.instance_id)
            # Compose: coordinator trace fires first (for [coord] output),
            # then delegate to any trace installed by the caller (e.g. run.py capture).
            set_trace(_CompositeTrace(coord_trace, existing))

        # Build action registry if needed
        action_registry = None
        if not is_agentic:
            has_act = any(
                s["primitive"] == "govern"
                for s in config.get("steps", [])
            )
            if has_act:
                action_registry = self._build_action_registry(case_input)
        else:
            # Agentic mode: build action_registry if govern is available
            if "govern" in config.get("available_primitives", []):
                action_registry = self._build_action_registry(case_input)

        # ── Check execution prerequisites ──
        # DEVS executor has no LangGraph dependency.
        # Only check LLM availability; fall back to simulation if not configured.
        _can_execute = True
        _fallback_reason = None
        try:
            from cognitive_core.engine.llm import create_llm
            create_llm(model="default", temperature=0.1)
        except Exception as e:
            _can_execute = False
            _fallback_reason = f"LLM not configured ({e})"
            import sys as _sys, traceback as _tb
            print(f"[coordinator] LLM test failed — falling back to simulation: {e}", file=_sys.stderr, flush=True)
            _tb.print_exc(file=_sys.stderr)

        if not _can_execute:
            self._log(f"  ⚠ {_fallback_reason} — running simulated execution")
            return self._execute_workflow_simulated(
                instance, config, case_input, tool_registry, action_registry,
            )

        # ── Shared step callback: epistemic state + ledger ────────────────
        # Used by both workflow and agentic execution modes.
        import time as _step_time
        _start_time = _step_time.time()

        from cognitive_core.engine.epistemic import (
            WorkflowEpistemicRecord, compute_step_epistemic_state,
        )
        _epistemic_record = WorkflowEpistemicRecord(
            instance_id=instance.instance_id
        )

        def _step_log_callback(step_name, step_output, state):
            try:
                prim = step_output.get("primitive", "")
                output = step_output.get("output", step_output.get("result", {}))
                elapsed_ms = int((_step_time.time() - _start_time) * 1000)

                prior_steps = state.get("steps", [])[:-1]
                ep_state = compute_step_epistemic_state(
                    step_output=step_output,
                    prior_steps=prior_steps,
                    open_gaps=list(_epistemic_record.open_gaps),
                )
                _epistemic_record.add_step(ep_state)

                if "epistemic" not in state:
                    state["epistemic"] = {}
                state["epistemic"][step_name] = ep_state.to_dict()
                state["epistemic"]["_record"] = _epistemic_record.reviewer_context()

                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="step_completed",
                    details={
                        "step_name": step_name,
                        "primitive": prim,
                        "output": output,
                        "elapsed_ms": elapsed_ms,
                        "confidence": step_output.get("confidence"),
                        "epistemic": ep_state.to_dict(),
                    },
                )
                # Compact epistemic summary — fires for every step at framework level
                def _fmt(v):
                    return f"{v:.2f}" if v is not None else "—"
                # Interpretation: derive operational state from warranted + flags + overall
                _flags = ep_state.coherence_flags or []
                _critical = [f for f in _flags
                             if hasattr(f, "value") and "tension" in f.value.lower()
                             or hasattr(f, "value") and "mismatch" in f.value.lower()]
                if not ep_state.warranted or _critical:
                    _interp = "INSUFFICIENT"
                elif _flags or (ep_state.overall is not None and ep_state.overall < 0.6):
                    _interp = "DEGRADED"
                else:
                    _interp = "SUPPORTED"
                ep_parts = [
                    f"{_interp}",
                    f"overall={_fmt(ep_state.overall)}",
                    f"warranted={'✓' if ep_state.warranted else '✗'}",
                ]
                if ep_state.reasoning_quality is not None:
                    ep_parts.append(f"rq={_fmt(ep_state.reasoning_quality)}")
                if ep_state.outcome_certainty is not None:
                    ep_parts.append(f"oc={_fmt(ep_state.outcome_certainty)}")
                if ep_state.alternative_separation is not None:
                    ep_parts.append(f"sep={_fmt(ep_state.alternative_separation)}")
                if ep_state.coherence_flags:
                    flag_names = [f.value if hasattr(f, "value") else str(f)
                                  for f in ep_state.coherence_flags]
                    ep_parts.append(f"⚠ flags={flag_names}")
                self._log_epistemic(f"     epistemic: {step_name}  {' · '.join(ep_parts)}")
            except Exception:
                pass
            return None

        instance._epistemic_record = _epistemic_record

        from cognitive_core.engine.devs_executor import (
            combined_callback, resource_request_callback, step_execute,
        )

        # Execute with step-level interception
        if is_agentic:
            # Agentic mode: DEVS-native executor.
            # The orchestrator selects the trajectory autonomously.
            # The substrate enforces governance on that trajectory identically
            # to workflow mode — same step callback, same epistemic engine,
            # same ledger recording.
            from cognitive_core.engine.agentic_devs import run_agentic_workflow_devs
            callback = combined_callback(_step_log_callback, resource_request_callback)

            def _decision_log_callback(decision: dict):
                """Log each orchestrator trajectory decision to the action ledger."""
                try:
                    self.store.log_action(
                        instance_id=instance.instance_id,
                        correlation_id=instance.correlation_id,
                        action_type="orchestrator_decision",
                        details={
                            "action": decision.get("action", "invoke"),
                            "primitive": decision.get("primitive", ""),
                            "step_name": decision.get("step_name", ""),
                            "params_key": decision.get("params_key", ""),
                            "reasoning": decision.get("reasoning", ""),
                        },
                    )
                except Exception:
                    pass

            result = run_agentic_workflow_devs(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
                action_registry=action_registry,
                step_callback=callback,
                decision_callback=_decision_log_callback,
            )
            if result.completed:
                return result.final_state
            else:
                return result.interrupt
        else:
            # Workflow mode: DEVS executor with shared step callback.
            callback = combined_callback(_step_log_callback, resource_request_callback)

            result = step_execute(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
                action_registry=action_registry,
                step_callback=callback,
            )

            if result.completed:
                return result.final_state
            else:
                # Return the interrupt for start() to handle
                return result.interrupt

    def _execute_workflow_simulated(
        self,
        instance: InstanceState,
        config: dict[str, Any],
        case_input: dict[str, Any],
        tool_registry,
        action_registry,
    ) -> dict[str, Any]:
        """
        Simulated workflow execution when LangGraph is not installed.

        Walks through each step in the workflow config, using tool registry
        data to populate retrieve steps and generating deterministic outputs
        for think/verify/generate/decide steps. Detects resource request
        patterns in step configurations to trigger realistic interrupts.

        This enables the coordinator state machine (suspend/resume/dispatch)
        to be tested end-to-end without LLM or LangGraph dependencies.
        """
        from cognitive_core.engine.devs_executor import StepInterrupt

        steps = config.get("steps", [])
        step_outputs = []
        step_names = [s["name"] for s in steps]

        # Build context from case_input for tool calls
        for i, step in enumerate(steps):
            step_name = step["name"]
            primitive = step["primitive"]
            params = step.get("params", {})

            self._log(f"  [sim] step {i+1}/{len(steps)}: {step_name} ({primitive})")

            output: dict[str, Any] = {
                "step_name": step_name,
                "primitive": primitive,
                "confidence": 0.85,
            }

            if primitive == "retrieve":
                # Use tool registry to get actual data
                data = {}
                if tool_registry:
                    for tool_name in tool_registry.list_tools():
                        try:
                            result = tool_registry.call(tool_name, case_input)
                            if result.success:
                                data[tool_name] = result.data
                        except Exception:
                            pass
                output["output"] = {"data": data}
                output["confidence"] = 0.95 if data else 0.50

            elif primitive == "deliberate":
                # Check if this step might need resources (coverage analysis pattern)
                spec = params.get("specification", "")
                resource_triggers = params.get("resource_request_triggers", [])

                if resource_triggers or "equipment" in spec.lower() or "schedule" in spec.lower():
                    # Check if we have the data already (from delegation results)
                    delegation_results = case_input.get("_delegation_results", {})
                    if not delegation_results:
                        # Simulate resource request
                        need_name = "scheduled_equipment_verification"
                        for trigger in resource_triggers:
                            if isinstance(trigger, dict):
                                need_name = trigger.get("need", need_name)
                                break

                        output["output"] = {
                            "analysis": f"Cannot complete {step_name} without external resource.",
                            "resource_requests": [{
                                "need": need_name,
                                "reason": f"Step {step_name} requires external data to proceed.",
                                "blocking": True,
                                "context": {"claim_id": case_input.get("claim_id", "")},
                            }],
                        }
                        output["confidence"] = 0.55

                        # Create interrupt
                        interrupt = StepInterrupt(
                            step_name=step_name,
                            resource_requests=output["output"]["resource_requests"],
                            partial_output=output,
                            state_at_interrupt={
                                "input": case_input,
                                "steps": step_outputs + [output],
                                "current_step": step_name,
                                "metadata": config.get("metadata", {}),
                                "loop_counts": {},
                                "routing_log": [],
                            },
                        )
                        return interrupt

                output["output"] = {"analysis": f"Analysis complete for {step_name}."}

            elif primitive == "generate":
                # Settlement recommendation — use calculate_settlement if available
                if tool_registry and "calculate_settlement" in tool_registry.list_tools():
                    try:
                        calc_result = tool_registry.call("calculate_settlement", case_input)
                        if calc_result.success:
                            output["output"] = {"artifact": calc_result.data}
                            output["confidence"] = 0.94
                        else:
                            output["output"] = {"artifact": {"summary": f"Generated output for {step_name}"}}
                    except Exception:
                        output["output"] = {"artifact": {"summary": f"Generated output for {step_name}"}}
                else:
                    output["output"] = {"artifact": {"summary": f"Generated output for {step_name}"}}

            elif primitive == "verify":
                output["output"] = {"conforms": True, "findings": []}
                output["confidence"] = 0.90

            elif primitive == "decide":
                output["output"] = {"decision": "approve", "reasoning": "Simulated decision."}

            elif primitive == "govern":
                output["output"] = {"tier_applied": "auto", "disposition": "proceed", "tier_rationale": "Simulated governance."}

            elif primitive == "delegate":
                # Explicit delegation step
                output["output"] = {"delegated": True}

            else:
                output["output"] = {"result": f"Simulated {primitive} output."}

            step_outputs.append(output)

        # Workflow completed
        final_state = {
            "input": case_input,
            "steps": step_outputs,
            "current_step": step_names[-1] if step_names else "",
            "metadata": config.get("metadata", {}),
            "loop_counts": {},
            "routing_log": [],
        }
        return final_state

    def _execute_workflow_from_state(
        self,
        instance: InstanceState,
        state_snapshot: dict[str, Any],
        resume_step: str,
        model: str = "default",
        temperature: float = 0.1,
    ) -> dict[str, Any] | Any:
        """
        Resume a workflow from a saved state snapshot at the given step.
        Uses step-by-step execution so resumed workflows can also
        produce ResourceRequests and be interrupted again.

        Returns either:
          - dict: workflow completed
          - StepInterrupt: workflow interrupted again
        """
        from cognitive_core.engine.composer import load_three_layer
        from cognitive_core.engine.devs_executor import (
            step_resume, resource_request_callback, StepResult,
        )

        workflow_path = self._find_workflow(instance.workflow_type)
        domain_path = self._find_domain(instance.domain)
        config, _ = load_three_layer(workflow_path, domain_path)

        tool_registry = self._build_tool_registry(
            state_snapshot.get("input", {})
        )
        action_registry = None
        if config.get("mode") != "agentic":
            has_act = any(
                s["primitive"] == "govern"
                for s in config.get("steps", [])
            )
            if has_act:
                action_registry = self._build_action_registry(
                    state_snapshot.get("input", {})
                )

        # ── Check execution prerequisites ──
        # DEVS executor has no LangGraph dependency.
        _can_execute = True
        try:
            from cognitive_core.engine.llm import create_llm
            create_llm(model="default", temperature=0.1)
        except Exception:
            _can_execute = False

        if not _can_execute:
            self._log(f"  ⚠ LLM not available — simulated resume at '{resume_step}'")
            case_input = dict(state_snapshot.get("input", {}))
            case_input["_delegation_results"] = state_snapshot.get("delegation_results", {})
            return self._execute_workflow_simulated(
                instance, config, case_input, tool_registry, action_registry,
            )

        import time as _step_time_r
        _start_time_r = _step_time_r.time()

        from cognitive_core.engine.epistemic import (
            WorkflowEpistemicRecord, compute_step_epistemic_state,
        )
        # Reconstruct epistemic record from prior step ledger entries
        _epistemic_record_r = WorkflowEpistemicRecord(instance_id=instance.instance_id)
        try:
            for entry in self.store.get_ledger(instance_id=instance.instance_id):
                if entry.get("action_type") == "step_completed":
                    ep = entry.get("details", {}).get("epistemic")
                    if ep:
                        from cognitive_core.engine.epistemic import (
                            StepEpistemicState, CoherenceFlag,
                        )
                        _epistemic_record_r.open_gaps = list(ep.get("inherited_gaps", []))
        except Exception:
            pass

        def _step_log_callback_r(step_name, step_output, state):
            try:
                prim = step_output.get("primitive", "")
                output = step_output.get("output", step_output.get("result", {}))
                elapsed_ms = int((_step_time_r.time() - _start_time_r) * 1000)

                prior_steps = state.get("steps", [])[:-1]
                ep_state = compute_step_epistemic_state(
                    step_output=step_output,
                    prior_steps=prior_steps,
                    open_gaps=list(_epistemic_record_r.open_gaps),
                )
                _epistemic_record_r.add_step(ep_state)

                if "epistemic" not in state:
                    state["epistemic"] = {}
                state["epistemic"][step_name] = ep_state.to_dict()
                state["epistemic"]["_record"] = _epistemic_record_r.reviewer_context()

                self.store.log_action(
                    instance_id=instance.instance_id,
                    correlation_id=instance.correlation_id,
                    action_type="step_completed",
                    details={
                        "step_name": step_name,
                        "primitive": prim,
                        "output": output,
                        "elapsed_ms": elapsed_ms,
                        "confidence": step_output.get("confidence"),
                        "epistemic": ep_state.to_dict(),
                    },
                )
                # Compact epistemic summary — fires for every step at framework level
                def _fmt(v):
                    return f"{v:.2f}" if v is not None else "—"
                # Interpretation: derive operational state from warranted + flags + overall
                _flags = ep_state.coherence_flags or []
                _critical = [f for f in _flags
                             if hasattr(f, "value") and "tension" in f.value.lower()
                             or hasattr(f, "value") and "mismatch" in f.value.lower()]
                if not ep_state.warranted or _critical:
                    _interp = "INSUFFICIENT"
                elif _flags or (ep_state.overall is not None and ep_state.overall < 0.6):
                    _interp = "DEGRADED"
                else:
                    _interp = "SUPPORTED"
                ep_parts = [
                    f"{_interp}",
                    f"overall={_fmt(ep_state.overall)}",
                    f"warranted={'✓' if ep_state.warranted else '✗'}",
                ]
                if ep_state.reasoning_quality is not None:
                    ep_parts.append(f"rq={_fmt(ep_state.reasoning_quality)}")
                if ep_state.outcome_certainty is not None:
                    ep_parts.append(f"oc={_fmt(ep_state.outcome_certainty)}")
                if ep_state.alternative_separation is not None:
                    ep_parts.append(f"sep={_fmt(ep_state.alternative_separation)}")
                if ep_state.coherence_flags:
                    flag_names = [f.value if hasattr(f, "value") else str(f)
                                  for f in ep_state.coherence_flags]
                    ep_parts.append(f"⚠ flags={flag_names}")
                self._log_epistemic(f"     epistemic: {step_name}  {' · '.join(ep_parts)}")
            except Exception:
                pass
            return None

        instance._epistemic_record = _epistemic_record_r

        from cognitive_core.engine.devs_executor import combined_callback
        resume_callback = combined_callback(_step_log_callback_r, resource_request_callback)

        result = step_resume(
            config, state_snapshot, resume_step,
            model, temperature,
            tool_registry=tool_registry,
            action_registry=action_registry,
            step_callback=resume_callback,
        )

        if result.completed:
            return result.final_state
        else:
            return result.interrupt

    # ─── Result Extraction ───────────────────────────────────────────

    def _extract_result_summary(self, final_state: dict[str, Any]) -> dict[str, Any]:
        """Extract a structured summary from workflow final state."""
        steps = final_state.get("steps", [])
        summary: dict[str, Any] = {
            "step_count": len(steps),
            "steps": [],
            "input": final_state.get("input", {}),
        }

        for step in steps:
            prim = step.get("primitive", "")
            output = step.get("output", {})
            step_summary: dict[str, Any] = {
                "step_name": step.get("step_name", ""),
                "primitive": prim,
            }

            if prim == "classify":
                step_summary["category"] = output.get("category")
                step_summary["confidence"] = output.get("confidence")
            elif prim == "deliberate":
                step_summary["recommended_action"] = output.get("recommended_action")
                step_summary["warrant"] = str(output.get("warrant", ""))[:500]
                step_summary["situation_summary"] = str(output.get("situation_summary", ""))[:500]
                step_summary["confidence_basis"] = str(output.get("confidence_basis", ""))[:500]
                step_summary["options_considered"] = output.get("options_considered", [])
                step_summary["confidence"] = output.get("confidence")
            elif prim == "govern":
                step_summary["tier_applied"] = output.get("tier_applied")
                step_summary["disposition"] = output.get("disposition")
                step_summary["tier_rationale"] = str(output.get("tier_rationale", ""))[:300]
                step_summary["has_work_order"] = output.get("work_order") is not None
                step_summary["escalation_target"] = output.get("escalation_target")
                step_summary["confidence"] = output.get("confidence")
            elif prim == "investigate":
                step_summary["finding"] = str(output.get("finding", ""))[:500]
                step_summary["confidence"] = output.get("confidence")
                step_summary["evidence_flags"] = output.get("evidence_flags", [])
                step_summary["missing_evidence"] = output.get("missing_evidence", [])
            elif prim == "generate":
                step_summary["artifact_preview"] = str(output.get("artifact", ""))[:300]
                step_summary["artifact"] = output.get("artifact")
                step_summary["confidence"] = output.get("confidence")
            elif prim == "challenge":
                step_summary["survives"] = output.get("survives")
                step_summary["vulnerabilities"] = len(output.get("vulnerabilities", []))
            elif prim == "verify":
                step_summary["conforms"] = output.get("conforms")
                step_summary["violations"] = len(output.get("violations", []))
            elif prim == "retrieve":
                step_summary["sources"] = list(output.get("data", {}).keys())

            summary["steps"].append(step_summary)

        return summary

    # ─── C3: State Compaction ────────────────────────────────────────

    MAX_SNAPSHOT_BYTES = 512 * 1024  # 512KB warn threshold

    def _compact_state_for_suspension(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Strip bulky debug data before persisting a state snapshot.

        The state snapshot must contain everything needed to resume the
        workflow at a given step. What IS needed:
          - input: the original case input (tool data for registry rebuild)
          - steps[].output: parsed LLM output (used by build_context_from_state)
          - steps[].step_name, steps[].primitive: routing context
          - current_step, metadata, loop_counts, routing_log

        What is NOT needed (and regenerated on resume):
          - steps[].raw_response: full LLM text (5-15KB per step)
          - steps[].prompt_used: full rendered prompt (2-10KB per step)
          - Large retrieve data payloads (available via tool registry)

        At 5 steps, this typically reduces snapshot from ~80KB to ~15KB.
        At 20 delegation levels, the difference is 1.6MB vs 300KB.
        """
        import copy
        compact = copy.deepcopy(state)

        for step in compact.get("steps", []):
            # Strip raw LLM response — only used for debugging, never for resume
            step.pop("raw_response", None)

            # Strip rendered prompt — rebuilt from workflow config on resume
            step.pop("prompt_used", None)

            # For retrieve steps: truncate large data payloads
            # The tool registry reconstructs this data from case_input
            output = step.get("output", {})
            if step.get("primitive") == "retrieve" and "data" in output:
                data = output["data"]
                for key, val in list(data.items()):
                    serialized = json.dumps(val, default=str)
                    if len(serialized) > 2000:
                        data[key] = {"_truncated": True, "_keys": list(val.keys()) if isinstance(val, dict) else None, "_size": len(serialized)}

            # For generate steps: truncate very large artifacts
            if step.get("primitive") == "generate" and "artifact" in output:
                artifact = str(output["artifact"])
                if len(artifact) > 5000:
                    output["artifact"] = artifact[:5000] + "\n... [truncated for suspension]"

        # Log compaction stats
        raw_size = len(json.dumps(state, default=str))
        compact_size = len(json.dumps(compact, default=str))
        if raw_size > 0:
            ratio = (1 - compact_size / raw_size) * 100
            self._log(f"    snapshot compacted: {raw_size:,}B → {compact_size:,}B ({ratio:.0f}% reduction)")
        if compact_size > self.MAX_SNAPSHOT_BYTES:
            self._log(f"    ⚠ snapshot still large: {compact_size:,}B > {self.MAX_SNAPSHOT_BYTES:,}B threshold")

        return compact

    def _extract_unresolved_needs(
        self, final_state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract unresolved needs from investigate steps."""
        needs = []
        for step in final_state.get("steps", []):
            if step.get("primitive") == "investigate":
                output = step.get("output", {})
                for need in output.get("missing_evidence", []):
                    if isinstance(need, dict):
                        needs.append(need)
                    elif isinstance(need, str):
                        needs.append({"type": "unknown", "description": need})
        return needs

    # ─── File Resolution ─────────────────────────────────────────────

    def _find_workflow(self, workflow_type: str) -> Path:
        """Find workflow YAML by type name."""
        for suffix in ["", ".yaml", ".yml"]:
            p = self.workflow_dir / f"{workflow_type}{suffix}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Workflow not found: {workflow_type} (searched {self.workflow_dir})"
        )

    def _find_domain(self, domain: str) -> Path:
        """Find domain YAML by name."""
        for suffix in ["", ".yaml", ".yml"]:
            p = self.domain_dir / f"{domain}{suffix}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Domain not found: {domain} (searched {self.domain_dir})"
        )

    def _load_domain_config(self, domain: str) -> dict[str, Any]:
        """Load full domain YAML config. Returns empty dict on error."""
        try:
            domain_path = self._find_domain(domain)
            with open(domain_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Failed to load domain config for '%s': %s", domain, e)
            return {}

    def _resolve_governance_tier(self, domain: str) -> str:
        """Read governance tier from domain config. Cached."""
        if domain in self._domain_tiers:
            return self._domain_tiers[domain]

        try:
            domain_path = self._find_domain(domain)
            with open(domain_path) as f:
                domain_config = yaml.safe_load(f)
            tier = domain_config.get("governance", "gate")
            # v2 domains have governance as a dict: {"tier": "auto", ...}
            # v1 domains have governance as a string: "gate"
            if isinstance(tier, dict):
                tier = tier.get("tier", "gate")
        except FileNotFoundError:
            tier = "gate"  # safe default

        self._domain_tiers[domain] = tier
        return tier

    def _get_optimization_config(self, domain: str) -> Any:
        """
        Load optimization config from domain YAML. Cached.

        Returns OptimizationConfig if available, None if optimizer not loaded
        or domain has no optimization section (backward compatible).
        """
        if not _OPTIMIZER_AVAILABLE:
            return None

        if domain in self._optimization_configs:
            return self._optimization_configs[domain]

        opt_config = None
        try:
            domain_path = self._find_domain(domain)
            with open(domain_path) as f:
                domain_yaml = yaml.safe_load(f) or {}
            opt_section = domain_yaml.get("optimization")
            if opt_section:
                opt_config = parse_optimization_config(opt_section)
            else:
                # Use coordinator-level defaults if present
                defaults = self.config.get("optimization_defaults")
                if defaults:
                    opt_config = parse_optimization_config(defaults)
        except (FileNotFoundError, Exception):
            pass

        self._optimization_configs[domain] = opt_config
        return opt_config

    # ─── Resource Backpressure Queue ────────────────────────────────

    def _enqueue_for_resource(
        self,
        resource_key: str,
        work_order_id: str,
        instance_id: str,
        req: dict,
        capability: Any,
        lineage: list[str],
    ) -> None:
        """
        Queue a work order for dispatch when a resource becomes available.

        Called when the optimizer returns no_eligible_resources because
        all resources are at capacity (e.g., batch in EXECUTING state).
        The work order transitions to QUEUED status and waits.
        """
        entry = {
            "work_order_id": work_order_id,
            "instance_id": instance_id,
            "req": req,
            "capability_need_type": getattr(capability, 'need_type', ''),
            "lineage": lineage,
            "queued_at": time.time(),
        }
        self._resource_wait_queue.setdefault(resource_key, []).append(entry)
        self._log(f"    ⏳ QUEUED: {work_order_id} waiting for {resource_key}")

        self.store.log_action(
            instance_id=instance_id,
            correlation_id="",
            action_type="work_order_queued_for_resource",
            details={
                "work_order_id": work_order_id,
                "resource_key": resource_key,
                "queue_depth": len(self._resource_wait_queue.get(resource_key, [])),
            },
        )

    def drain_resource_queue(self, resource_key: str | None = None) -> int:
        """
        Attempt to dispatch queued work orders whose resources are now available.

        Called:
        - After batch complete_batch() (batch resource returns to COLLECTING)
        - After sweep_reservations() releases expired reservations
        - After any resource capacity is freed

        Returns number of work orders successfully drained.
        """
        drained = 0
        keys = [resource_key] if resource_key else list(self._resource_wait_queue.keys())

        for rk in keys:
            queue = self._resource_wait_queue.get(rk, [])
            if not queue:
                continue

            still_waiting = []
            for entry in queue:
                wo_id = entry["work_order_id"]
                wo = self.store.get_work_order(wo_id)
                if not wo or wo.status != WorkOrderStatus.QUEUED:
                    continue  # already handled or cancelled

                # Check if resource can now accept
                can_dispatch = False
                if self._resource_registry:
                    res = self._resource_registry.get(rk)
                    if res and res.capacity.can_accept():
                        can_dispatch = True

                if can_dispatch:
                    # Mark as DISPATCHED — resource has capacity
                    wo.status = WorkOrderStatus.DISPATCHED
                    self.store.save_work_order(wo)

                    # Try full dispatch if instance and capability available
                    instance = self.store.get_instance(entry["instance_id"])
                    capability = self._find_capability(entry.get("capability_need_type", ""))
                    if instance and capability:
                        done = self._dispatch_provider(
                            instance, wo, entry["req"], capability,
                            entry.get("lineage", []),
                        )
                    else:
                        # Capability routing not available (e.g., test environment)
                        # WO is DISPATCHED and will be picked up by the next
                        # sweep or poll cycle. This is safe: the WO is no longer
                        # QUEUED, so it won't be drained again.
                        pass

                    drained += 1
                    wait_time = time.time() - entry.get("queued_at", 0)
                    self._log(
                        f"    ▶ DRAINED: {wo_id} from queue for {rk} "
                        f"(waited {wait_time:.1f}s)"
                    )
                    self.store.log_action(
                        instance_id=entry["instance_id"],
                        correlation_id="",
                        action_type="work_order_drained_from_queue",
                        details={
                            "work_order_id": wo_id,
                            "resource_key": rk,
                            "waited_seconds": wait_time,
                        },
                    )
                else:
                    still_waiting.append(entry)

            if still_waiting:
                self._resource_wait_queue[rk] = still_waiting
            else:
                self._resource_wait_queue.pop(rk, None)

        return drained

    @property
    def queued_work_order_count(self) -> int:
        """Total work orders waiting for resource capacity."""
        return sum(len(q) for q in self._resource_wait_queue.values())

    def get_queue_depth(self, resource_key: str) -> int:
        """Queue depth for a specific resource."""
        return len(self._resource_wait_queue.get(resource_key, []))

    # ─── Production Hardening Hooks ──────────────────────────────────

    def _persist_ddr(
        self,
        decision: Any,   # DispatchDecision from ddd.py
        eligible: list,   # eligible ResourceRegistration list
        eligibility_audit: list,  # EligibilityResult list
        config: Any,      # OptimizationConfig
        solution: Any,    # ArchetypeSolution (may be None)
    ) -> None:
        """
        DDR callback — called by the optimizer for every dispatch decision.

        Builds a DispatchDecisionRecord and persists it to the action_ledger
        so every dispatch is auditable.
        """
        if not _HARDENING_AVAILABLE:
            return

        try:
            # Build eligibility entries for DDR
            eligible_entries = []
            excluded_entries = []
            for er in eligibility_audit:
                entry = DDREligibilityEntry(
                    resource_id=getattr(er, 'resource_id', ''),
                    eligible=getattr(er, 'eligible', False),
                    constraint_name=getattr(er, 'failed_constraint', ''),
                    constraint_reason=getattr(er, 'audit_reason', ''),
                )
                if entry.eligible:
                    eligible_entries.append(entry)
                else:
                    excluded_entries.append(entry)

            # Build candidate scores
            candidate_scores = []
            for i, rs in enumerate(getattr(decision, 'ranking_scores', [])):
                candidate_scores.append(DDRCandidateScore(
                    resource_id=getattr(rs, 'resource_id', ''),
                    total_score=getattr(rs, 'total_score', 0.0),
                    rank=i + 1,
                    feature_scores=getattr(rs, 'feature_scores', {}),
                ))

            solver_name = ""
            solver_seed = 0
            if solution:
                solver_name = getattr(solution, 'solver_name', '')
                solver_seed = getattr(solution, 'solver_seed', 0)

            ddr = build_ddr(
                work_order_id=getattr(decision, 'work_order_id', ''),
                correlation_id="",  # filled by coordinator context
                case_id="",
                trace_id="",
                policy_version="1.0.0",
                policy_mode="static",
                solver_name=solver_name,
                solver_version="1.0",
                solver_seed=solver_seed,
                eligible_entries=eligible_entries,
                excluded_entries=excluded_entries,
                objective_weights=getattr(config, 'objectives', {}),
                candidate_scores=candidate_scores,
                selected_resource_id=getattr(decision, 'selected_resource_id', None),
                selection_tier=getattr(decision, 'tier', ''),
                reservation_id=getattr(decision, 'reservation_id', None),
                active_constraints=[],
            )

            # Persist to in-memory log (production: action_ledger table)
            self._ddr_log.append(ddr.to_ledger_entry())

            # Also persist to action_ledger if store is available
            if hasattr(self, 'store') and self.store:
                self.store.log_action(
                    instance_id="",
                    correlation_id="",
                    action_type="dispatch_decision_record",
                    details=ddr.to_ledger_entry(),
                    idempotency_key=f"ddr:{ddr.ddr_id}",
                )
        except Exception as e:
            # DDR persistence failure must never break dispatch
            if self.verbose:
                self._log(f"  ⚠ DDR persistence error (non-fatal): {e}")

    # ─── Production Hardening Hooks ────────────────────────────────

    def _persist_ddr(
        self,
        decision: Any,
        eligible: list,
        eligibility_audit: list,
        config: Any,
        solution: Any,
    ) -> None:
        """
        DDR callback — called by optimizer for every dispatch decision.
        Builds and persists a DispatchDecisionRecord to the action_ledger.
        """
        if not _HARDENING_AVAILABLE:
            return

        # Build eligible/excluded entries
        eligible_entries = []
        excluded_entries = []
        for e in eligibility_audit:
            entry = DDREligibilityEntry(
                resource_id=e.resource_id,
                eligible=e.eligible,
                constraint_name=e.failed_constraint or "",
                constraint_reason=e.audit_reason or "",
            )
            if e.eligible:
                eligible_entries.append(entry)
            else:
                excluded_entries.append(entry)

        # Build candidate scores from ranking_scores
        candidate_scores = []
        for rank, rs in enumerate(decision.ranking_scores or [], 1):
            candidate_scores.append(DDRCandidateScore(
                resource_id=rs.resource_id,
                total_score=rs.total_score,
                rank=rank,
                feature_scores=rs.feature_scores,
            ))

        # Build reason codes
        reason_codes = [decision.tier]
        if not decision.selected_resource_id:
            reason_codes.append("no_assignment")

        # Solver info
        solver_name = ""
        solver_version = "1.0"
        solver_seed = 0
        if solution:
            solver_name = getattr(solution, 'solver_name', '')
            solver_seed = getattr(solution, 'solver_seed', 0)

        ddr = build_ddr(
            work_order_id=decision.work_order_id,
            correlation_id="",  # filled by caller if available
            case_id="",
            trace_id="",
            policy_version="1.0.0",  # from PolicyManager when wired
            policy_mode="static",
            solver_name=solver_name,
            solver_version=solver_version,
            solver_seed=solver_seed,
            eligible_entries=eligible_entries,
            excluded_entries=excluded_entries,
            objective_weights=config.objectives if config else {},
            candidate_scores=candidate_scores,
            selected_resource_id=decision.selected_resource_id,
            selection_tier=decision.tier,
            reservation_id=decision.reservation_id,
            reason_codes=reason_codes,
        )

        # Persist to action_ledger
        self.store.log_action(
            instance_id="",
            correlation_id="",
            action_type="dispatch_decision_record",
            details=ddr.to_ledger_entry(),
            idempotency_key=f"ddr:{ddr.ddr_id}",
        )
        self._ddr_log.append(ddr.to_ledger_entry())

    def _record_work_order_completion(
        self,
        instance: Any,
        wo: Any,
        need_name: str,
        accepted: bool,
        rejection_reason: str = "",
    ) -> None:
        """
        Record work order completion for oscillation detection.
        Called when a provider completes and the requestor evaluates the result.
        """
        if not self._oscillation_detector:
            return

        provider_id = getattr(wo, 'handler_instance_id', '') or ''
        verdict = self._oscillation_detector.record_attempt(
            need_type=need_name,
            case_id=instance.correlation_id,
            correlation_id=instance.correlation_id,
            provider_resource_id=provider_id,
            work_order_id=wo.work_order_id,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

        if verdict.action.value == "escalate":
            self._log(
                f"  ⚠ OSCILLATION: {need_name} for {instance.correlation_id} "
                f"— {verdict.reason}"
            )
            self.store.log_action(
                instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                action_type="semantic_oscillation_detected",
                details={
                    "need_type": need_name,
                    "action": verdict.action.value,
                    "reason": verdict.reason,
                },
            )
        elif verdict.action.value == "retry_different":
            self._log(
                f"  ↻ OSCILLATION: retrying {need_name} with different provider "
                f"(excluding {verdict.exclude_providers})"
            )

    def _register_saga_side_effects(
        self,
        correlation_id: str,
        work_order_id: str,
        result: dict[str, Any] | Any,
    ) -> None:
        """
        Register side effects from a completed work order in the saga coordinator.

        Side effects are detected by looking for `_side_effects` in the
        handler's result output. Each side effect must include:
        - action: what was done ("send_email", "update_database")
        - inverse: how to undo it (serialized compensating transaction)
        - step_name: which workflow step produced it

        This is also called when the engine's CompensationLedger has
        confirmed entries — those are promoted to the saga level.
        """
        if not self._saga_coordinator:
            return

        # Check for explicitly declared side effects in result
        result_dict = result if isinstance(result, dict) else {}
        side_effects = result_dict.get("_side_effects", [])
        for se in side_effects:
            if isinstance(se, dict) and "action" in se:
                entry_id = self._saga_coordinator.register(
                    saga_id=correlation_id,
                    work_order_id=work_order_id,
                    step_name=se.get("step_name", "unknown"),
                    action_description=se.get("action", ""),
                    inverse_action=se.get("inverse", {}),
                )
                # Auto-confirm since the work order already completed
                self._saga_coordinator.confirm(entry_id)

    def _lean_case_input(self, case_input: dict[str, Any]) -> dict[str, Any]:
        """
        Strip pre-analysis labels and embedded tool data from case_input.

        Used when MCP is active: the workflow receives only identifiers
        (case_id, alert_id) and fetches all data via tool calls.
        Prevents pre-labeled fields (fraud_type, description, etc.) from
        leaking into LLM context before the model has a chance to reason.
        """
        _LEAN_EXCLUDE = {
            "_meta", "description", "fraud_type", "alert_type", "risk_score",
        }
        return {
            k: v for k, v in case_input.items()
            if k not in _LEAN_EXCLUDE and not k.startswith("get_")
        }

    def _get_mcp_loop(self):
        """
        Return a persistent background event loop for MCP operations.

        Creates a daemon thread running loop.run_forever() on first call.
        All MCP connect/disconnect/tool-call operations are scheduled on this
        loop via run_coroutine_threadsafe(), so the MCP session's asyncio
        transports are always bound to a live loop regardless of how many
        thread-pool workers have come and gone.
        """
        import asyncio
        import threading
        if self._mcp_bg_loop is None or self._mcp_bg_loop.is_closed():
            self._mcp_bg_loop = asyncio.new_event_loop()
            self._mcp_bg_thread = threading.Thread(
                target=self._mcp_bg_loop.run_forever,
                daemon=True,
                name="mcp-event-loop",
            )
            self._mcp_bg_thread.start()
        return self._mcp_bg_loop

    def _build_tool_registry(self, case_input: dict[str, Any]):
        """
        Build tool registry with correct priority:
          1. MCP server (env var DATA_MCP_URL / DATA_MCP_CMD, or config data_mcp_cmd)
          2. Case JSON tools (specific to this run — always take priority)
          3. Fixtures DB (fills in gaps for tools not in case JSON)

        Case JSON keys that look like tool data (dicts/lists) become tools.
        If a case JSON provides tools, it's the primary source — the LLM
        should call the tools the workflow specifies, not everything available.
        """
        data_mcp_url = os.environ.get("DATA_MCP_URL", "")
        # env var takes precedence; fall back to coordinator config
        data_mcp_cmd = (
            os.environ.get("DATA_MCP_CMD", "")
            or self.config.get("data_mcp_cmd", "")
        )

        if data_mcp_url or data_mcp_cmd:
            import asyncio
            from cognitive_core.engine.providers import MCPProvider
            from cognitive_core.engine.tools import ToolRegistry

            # Reuse cached provider — avoids spawning a subprocess per workflow
            if self._mcp_provider is None:
                if data_mcp_url:
                    self._mcp_provider = MCPProvider(transport="http", url=data_mcp_url)
                else:
                    parts = data_mcp_cmd.split()
                    self._mcp_provider = MCPProvider(
                        transport="stdio", command=parts[0], args=parts[1:],
                    )
                async def _connect():
                    await self._mcp_provider.connect()
                mcp_loop = self._get_mcp_loop()
                future = asyncio.run_coroutine_threadsafe(_connect(), mcp_loop)
                future.result(timeout=30)
                self._log(f"  📡 MCP provider connected ({len(self._mcp_provider.tools)} tools)")

            registry = ToolRegistry()
            self._mcp_provider.register_all(registry)
            return registry

        # Check if case_input has tool-shaped data (dict values with tool-like keys)
        has_case_tools = any(
            isinstance(v, (dict, list)) and k.startswith("get_")
            for k, v in case_input.items()
        )

        if has_case_tools:
            # Case JSON is the primary source for this run
            from cognitive_core.engine.tools import create_case_registry
            return create_case_registry(case_input)

        # Fallback: try fixtures DB, then case registry
        try:
            from fixtures.api import create_service_registry
            from fixtures.db import DB_PATH
            if DB_PATH.exists():
                return create_service_registry()
        except (ImportError, FileNotFoundError):
            pass

        from cognitive_core.engine.tools import create_case_registry
        return create_case_registry(case_input, fixtures_dir=str(self.case_dir))

    def _build_action_registry(self, case_input: dict[str, Any]):
        """Build action registry."""
        from cognitive_core.engine.actions import create_simulation_registry
        return create_simulation_registry()

    # ─── Query API ───────────────────────────────────────────────────

    def get_instance(self, instance_id: str) -> InstanceState | None:
        return self.store.get_instance(instance_id)

    def get_correlation_chain(self, correlation_id: str) -> list[InstanceState]:
        return self.store.list_instances(correlation_id=correlation_id)

    def get_work_orders(self, instance_id: str) -> list[WorkOrder]:
        return self.store.get_work_orders_for_instance(instance_id)

    def get_ledger(
        self,
        instance_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.get_ledger(instance_id, correlation_id)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    # ─── Logging ─────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [coord] {msg}", file=sys.stderr, flush=True)

    def _log_epistemic(self, msg: str):
        """Always print epistemic state — not gated by verbose."""
        print(f"  [coord] {msg}", file=sys.stderr, flush=True)


class _CompositeTrace:
    """
    Chains two trace callbacks. Primary fires first, then delegate.
    Used so the coordinator's [coord] logging and a caller-installed
    capture trace can both receive step events.
    """
    def __init__(self, primary, delegate):
        self._primary = primary
        self._delegate = delegate

    def on_step_start(self, *a, **kw):
        self._primary.on_step_start(*a, **kw)
        self._delegate.on_step_start(*a, **kw)

    def on_llm_start(self, *a, **kw):
        self._primary.on_llm_start(*a, **kw)
        self._delegate.on_llm_start(*a, **kw)

    def on_llm_end(self, *a, **kw):
        self._primary.on_llm_end(*a, **kw)
        self._delegate.on_llm_end(*a, **kw)

    def on_parse_result(self, *a, **kw):
        self._primary.on_parse_result(*a, **kw)
        self._delegate.on_parse_result(*a, **kw)

    def on_parse_error(self, *a, **kw):
        self._primary.on_parse_error(*a, **kw)
        self._delegate.on_parse_error(*a, **kw)

    def on_route_decision(self, *a, **kw):
        self._primary.on_route_decision(*a, **kw)
        self._delegate.on_route_decision(*a, **kw)

    def on_retrieve_start(self, *a, **kw):
        self._primary.on_retrieve_start(*a, **kw)
        self._delegate.on_retrieve_start(*a, **kw)

    def on_retrieve_end(self, *a, **kw):
        self._primary.on_retrieve_end(*a, **kw)
        self._delegate.on_retrieve_end(*a, **kw)


class _CoordinatorTrace:
    """
    Bridges engine trace events into coordinator [coord] logging.
    Provides step-level visibility when running workflows through
    the coordinator, matching the ConsoleTrace format from runner.py.
    """

    PRIM_ICONS = {
        "classify": "🏷️ ",
        "investigate": "🔍",
        "verify": "✅",
        "generate": "📝",
        "challenge": "⚔️ ",
        "retrieve": "📡",
        "deliberate": "🤔",
        "govern": "⚖️ ",
    }

    def __init__(self, coord, instance_id: str):
        self.coord = coord
        self.instance_id = instance_id
        self.step_start = None

    def _log(self, msg: str):
        self.coord._log(msg)

    def on_step_start(self, step_name, primitive, loop_iteration):
        self.step_start = time.time()
        icon = self.PRIM_ICONS.get(primitive, "  ")
        iter_label = f" (iter {loop_iteration})" if loop_iteration > 1 else ""
        self._log(f"  {icon} {step_name}{iter_label}")

    def on_llm_start(self, step_name, prompt_chars):
        self._log(f"    ↳ LLM call ({prompt_chars:,} chars)...")

    def on_llm_end(self, step_name, response_chars, elapsed):
        self._log(f"    ↳ LLM response ({response_chars:,} chars, {elapsed:.1f}s)")

    def on_parse_result(self, step_name, primitive, output):
        if primitive == "classify":
            self._log(f"    → {output.get('category', '?')} "
                       f"({output.get('confidence', 0):.2f})")
            alts = output.get("alternative_categories", [])
            if alts:
                alt_str = ", ".join(
                    f"{a.get('category','?')} ({a.get('confidence',0):.2f})"
                    for a in alts[:3]
                )
                self._log(f"    → alternatives: {alt_str}")
            reasoning = str(output.get("reasoning", ""))
            if reasoning:
                for line in reasoning[:400].splitlines():
                    if line.strip():
                        self._log(f"    · {line.strip()}")
        elif primitive == "investigate":
            finding = str(output.get("finding", "?"))
            flags = output.get("evidence_flags", [])
            missing = output.get("evidence_missing", [])
            # Print finding across multiple lines if needed
            for i, chunk in enumerate([finding[j:j+120] for j in range(0, min(len(finding), 480), 120)]):
                prefix = "    →" if i == 0 else "      "
                self._log(f"{prefix} {chunk}")
            if flags:
                self._log(f"    → flags: {flags}")
            if missing:
                for m in missing[:2]:
                    desc = m.get("description", "") if isinstance(m, dict) else str(m)
                    self._log(f"    → missing: {str(desc)[:120]}")
            conf = output.get("confidence")
            if conf is not None:
                self._log(f"    → confidence: {conf:.2f}")
        elif primitive == "retrieve":
            data = output.get("data", {})
            sources = output.get("sources_queried", [])
            n_ok = sum(1 for s in sources if s.get("status") == "success")
            n_empty = sum(1 for s in sources
                         if s.get("status") == "success"
                         and s.get("record_count") is not None
                         and s.get("record_count") == 0)
            n_untracked = sum(1 for s in sources
                             if s.get("status") == "success"
                             and s.get("record_count") is None)
            parts = [f"{n_ok}/{len(sources)} sources", f"{len(data)} data keys"]
            if n_empty:
                parts.append(f"{n_empty} EMPTY")
            if n_untracked:
                parts.append(f"{n_untracked} record_count n/a")
            self._log(f"    → {', '.join(parts)}")
        elif primitive == "generate":
            artifact = str(output.get("artifact", ""))
            self._log(f"    → generated {len(artifact):,} chars")
            # Show first 300 chars of artifact
            preview = artifact[:300].replace("\n", " ")
            if preview:
                self._log(f"    · {preview}{'...' if len(artifact) > 300 else ''}")
        elif primitive == "deliberate":
            action = str(output.get("recommended_action") or "")
            conf = output.get("confidence", 0)
            rq = output.get("reasoning_quality")
            oc = output.get("outcome_certainty")
            # Print full action across multiple lines
            for i, chunk in enumerate([action[j:j+120] for j in range(0, min(len(action), 360), 120)]):
                prefix = "    →" if i == 0 else "      "
                self._log(f"{prefix} {chunk}")
            warrant = str(output.get("warrant") or "")
            if warrant:
                for i, chunk in enumerate([warrant[j:j+120] for j in range(0, min(len(warrant), 360), 120)]):
                    prefix = "    · warrant:" if i == 0 else "             "
                    self._log(f"{prefix} {chunk}")
            self._log(f"    · confidence: {conf:.2f}" +
                      (f"  rq={rq:.2f}" if rq is not None else "") +
                      (f"  oc={oc:.2f}" if oc is not None else ""))
            situation = str(output.get("situation_summary") or "")
            if situation:
                self._log(f"    · situation: {situation[:200]}")
        elif primitive == "govern":
            tier = output.get("tier_applied", "?")
            disposition = output.get("disposition", "?")
            rationale = str(output.get("tier_rationale") or "")
            self._log(f"    → tier={tier} disposition={disposition}")
            if rationale:
                for i, chunk in enumerate([rationale[j:j+120] for j in range(0, min(len(rationale), 480), 120)]):
                    prefix = "    · rationale:" if i == 0 else "              "
                    self._log(f"{prefix} {chunk}")
            wo = output.get("work_order") or {}
            if wo.get("instructions"):
                instr = str(wo["instructions"])
                for i, chunk in enumerate([instr[j:j+120] for j in range(0, min(len(instr), 360), 120)]):
                    prefix = "    · instructions:" if i == 0 else "                 "
                    self._log(f"{prefix} {chunk}")
        elif primitive == "verify":
            conforms = output.get("conforms", "?")
            violations = output.get("violations", [])
            rules = output.get("rules_checked", [])
            self._log(f"    → conforms: {conforms}  rules_checked: {len(rules)}")
            for v in violations:
                rule = v.get("rule", "") if isinstance(v, dict) else str(v)
                desc = v.get("description", "") if isinstance(v, dict) else ""
                sev = v.get("severity", "") if isinstance(v, dict) else ""
                self._log(f"    ✗ [{sev}] {rule}: {str(desc)[:120]}")
            if conforms and not violations:
                self._log(f"    ✓ all {len(rules)} rules passed")

    def on_parse_error(self, step_name, error):
        # Show full error, split into multiple lines for tracebacks
        for line in str(error).split("\n"):
            line = line.rstrip()
            if line:
                self._log(f"    ⚠ PARSE ERROR: {line}")

    def on_route_decision(self, from_step, to_step, decision_type, reason):
        target = "END" if to_step == "__end__" else to_step
        self._log(f"    → route → {target} ({decision_type})")

    def on_retrieve_start(self, step_name, source_name):
        self._log(f"    📡 {source_name}...")

    def on_retrieve_end(self, step_name, source_name, status, latency_ms):
        icon = "✓" if status == "success" else "✗"
        self._log(f"    📡 {source_name}: {icon} ({latency_ms:.0f}ms)")