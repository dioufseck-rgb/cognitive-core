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

from coordinator.types import (
    InstanceState,
    InstanceStatus,
    WorkOrder,
    WorkOrderStatus,
    WorkOrderResult,
    Suspension,
)
from coordinator.store import CoordinatorStore
from coordinator.policy import PolicyEngine, load_policy_engine, GovernanceDecision, DelegationDecision
from coordinator.tasks import (
    TaskQueue, SQLiteTaskQueue, InMemoryTaskQueue,
    Task, TaskType, TaskStatus, TaskCallback, TaskResolution,
)


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
    ):
        # Load coordinator config
        if config:
            self.config = config
        elif config_path:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Build policy engine
        self.policy = load_policy_engine(self.config)

        # State store
        self.store = store or CoordinatorStore(db_path)

        # Task queue (shares DB connection with store for SQLite)
        if task_queue:
            self.tasks = task_queue
        else:
            self.tasks = SQLiteTaskQueue(self.store.conn)

        # Workflow/domain resolution paths
        # Resolve relative to config file directory (not cwd)
        if config_path:
            _base = Path(config_path).resolve().parent.parent
        else:
            _base = Path.cwd()
        self.workflow_dir = _base / self.config.get("workflow_dir", "workflows")
        self.domain_dir = _base / self.config.get("domain_dir", "domains")
        self.case_dir = _base / self.config.get("case_dir", "cases")

        # Domain → governance tier mapping (loaded from domain configs)
        self._domain_tiers: dict[str, str] = {}

        self.verbose = verbose

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
    ) -> str:
        """
        Create and execute a new workflow instance.
        Returns the instance_id.
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

        # Resolve governance tier from domain config
        tier = self._resolve_governance_tier(domain)

        # Create instance
        instance = InstanceState.create(
            workflow_type=workflow_type,
            domain=domain,
            governance_tier=tier,
            lineage=lineage or [],
            correlation_id=correlation_id,
        )
        instance.status = InstanceStatus.RUNNING
        instance.updated_at = time.time()
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

        # Execute the workflow
        try:
            final_state = self._execute_workflow(
                instance, case_input, model, temperature
            )
            self._on_completed(instance, final_state)
        except Exception as e:
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
            final_state = self._execute_workflow_from_state(
                instance, state_snapshot,
                resume_step=suspension.suspended_at_step,
                model=model, temperature=temperature,
            )
            self._on_completed(instance, final_state, is_resume=True)
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

        # Mark completed
        instance.status = InstanceStatus.COMPLETED
        instance.updated_at = time.time()
        instance.result = self._extract_result_summary(final_state)
        instance.step_count = len(final_state.get("steps", []))
        self.store.save_instance(instance)

        self._log(f"✓ COMPLETED {instance_id} (post-approval)")

        # Now proceed to delegation evaluation (skipped during suspension)
        self._evaluate_and_execute_delegations(instance, final_state)

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
        from coordinator.contracts import assert_contract, PendingApproval

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
                task_obj.status = TaskStatus.PENDING
                task_obj.claimed_at = None
                task_obj.claimed_by = ""
                # Update in-place (SQLite uses INSERT OR REPLACE, InMemory updates dict)
                if isinstance(self.tasks, SQLiteTaskQueue):
                    self.tasks.conn.execute("""
                        UPDATE task_queue SET status = 'pending',
                            claimed_at = NULL, claimed_by = ''
                        WHERE task_id = ?
                    """, (task_id,))
                    self.tasks.conn.commit()
                elif isinstance(self.tasks, InMemoryTaskQueue):
                    self.tasks._tasks[task_id] = task_obj
            return task_obj.instance_id if task_obj else ""
        else:
            raise ValueError(f"Unknown action: {action}")

    def expire_overdue_tasks(self) -> int:
        """Expire tasks past their SLA. Call periodically."""
        return self.tasks.expire_overdue()

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
        if not is_resume:
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

        # ── Step 1: Evaluate governance tier ──
        # Skip on resume: governance was already evaluated before suspension.
        if not is_resume:
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
        else:
            self._log(f"  governance: skipped (resume — already evaluated before suspension)")
            self.store.log_action(
                instance_id=instance.instance_id,
                correlation_id=instance.correlation_id,
                action_type="governance_evaluation",
                details={"skipped": True, "reason": "resume after delegation/approval"},
            )

        # Mark completed (only reached if governance allows)
        instance.status = InstanceStatus.COMPLETED
        instance.updated_at = time.time()
        self.store.save_instance(instance)

        self._log(f"✓ COMPLETED {instance.instance_id}")

        # ── Step 2: Evaluate delegation policies ──
        self._evaluate_and_execute_delegations(instance, final_state)

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
            from coordinator.escalation import build_escalation_brief
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
            },
        )

        self._log(f"  ⏸ SUSPENDED {instance.instance_id} "
                   f"→ task {task_id} published to '{gov_decision.queue}'")

    # ─── Quality Gate ────────────────────────────────────────────

    def _evaluate_quality_gate(
        self, instance: InstanceState, final_state: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Check workflow output against quality gate thresholds.

        Returns None if all gates pass, or a dict with details
        about the gate that fired. When a gate fires, the coordinator
        should escalate to HITL regardless of the domain's declared tier.
        """
        qg_config = self.policy.raw_config.get("quality_gates", {})
        if not qg_config:
            return None

        exempt = qg_config.get("exempt_domains", [])
        if instance.domain in exempt:
            return None

        global_min = qg_config.get("min_confidence", 0.5)
        primitive_floors = qg_config.get("primitive_floors", {})
        escalation_tier = qg_config.get("escalation_tier", "gate")

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
                    "escalation_queue": qg_config.get("escalation_queue", "quality_review"),
                }

        return None

    def _get_tier_sla(self, tier: str) -> float | None:
        """Get SLA seconds for a governance tier from config."""
        tier_config = self.policy.governance_tiers.get(tier)
        if tier_config and tier_config.sla_seconds:
            return tier_config.sla_seconds
        return None

    def _evaluate_and_execute_delegations(
        self,
        instance: InstanceState,
        final_state: dict[str, Any],
    ):
        """Evaluate delegation policies and execute any that fire."""
        delegations = self.policy.evaluate_delegations(
            domain=instance.domain,
            workflow_output=final_state,
            lineage=instance.lineage,
        )

        # ── Deduplicate: skip policies that already fired for this instance ──
        # This prevents infinite loops when a resumed workflow re-completes
        # and the same evidence flags trigger the same delegation again.
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

        # Execute fire-and-forget first (source stays completed),
        # then blocking (source suspends). This ensures F&F delegations
        # dispatch before the source is suspended by a blocking delegation.
        #
        # MULTIPLE BLOCKING DELEGATIONS execute serially, not in parallel:
        #   1. First blocking delegation suspends source, runs handler
        #   2. When handler completes → source resumes → _on_completed
        #   3. Dedup guard skips first delegation (already fired)
        #   4. Second blocking delegation fires, suspends source again
        #   5. Repeat until all blocking delegations have executed
        #
        # This is BY DESIGN: each blocking delegation gets the enriched
        # state from the previous one. If you need parallel execution,
        # use fire_and_forget mode with a final gather step.
        #
        # If any handler gets governance-held (doesn't complete immediately),
        # remaining blocking delegations are deferred until the source
        # resumes after that handler completes.
        ff_delegations = [d for d in new_delegations if d.mode != "wait_for_result"]
        blocking_delegations = [d for d in new_delegations if d.mode == "wait_for_result"]

        if len(blocking_delegations) > 1:
            self._log(f"  ⚠ {len(blocking_delegations)} blocking delegations — "
                       f"will execute serially via suspend/resume chain")

        for deleg in ff_delegations:
            self._execute_delegation(instance, deleg, final_state)
        for deleg in blocking_delegations:
            self._execute_delegation(instance, deleg, final_state)

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

    # ─── Delegation Execution ────────────────────────────────────────

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

        self.store.log_action(
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

        try:
            handler_id = self.start(
                workflow_type=decision.target_workflow,
                domain=decision.target_domain,
                case_input=decision.inputs,
                lineage=lineage,
                correlation_id=source.correlation_id,
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
        """
        # Build external input from work order result
        external_input = {}
        if wo.result and wo.result.outputs:
            external_input[wo.work_order_id] = wo.result.outputs
        elif wo.result and wo.result.error:
            external_input[wo.work_order_id] = {
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
        If so, update the work order and resume the requester.

        This is the cascade mechanism: when a handler completes
        (either directly or via governance approval), the coordinator
        checks if any work order references this instance as its handler,
        and if the requester is suspended waiting for it.
        """
        # Find work orders where this instance is the handler
        # and the requester is suspended
        all_wos = self.store.get_work_orders_for_requester_or_handler(
            instance.instance_id
        )

        for wo in all_wos:
            if wo.handler_instance_id != instance.instance_id:
                continue
            if wo.status in (WorkOrderStatus.COMPLETED, WorkOrderStatus.FAILED):
                continue  # Already resolved

            # Update work order
            wo.status = WorkOrderStatus.COMPLETED
            wo.completed_at = time.time()
            wo.result = WorkOrderResult(
                work_order_id=wo.work_order_id,
                status="completed",
                outputs=instance.result or {},
                completed_at=time.time(),
            )
            self.store.save_work_order(wo)

            # Check if requester is suspended waiting
            requester = self.store.get_instance(wo.requester_instance_id)
            if requester and requester.status == InstanceStatus.SUSPENDED:
                suspension = self.store.get_suspension(requester.instance_id)
                if suspension and wo.work_order_id in suspension.work_order_ids:
                    self._log(f"  ↩ delegation cascade: {instance.instance_id} → "
                               f"resuming {requester.instance_id}")
                    self._resume_after_delegation(requester, wo, suspension)

    # ─── Workflow Execution ──────────────────────────────────────────

    def _execute_workflow(
        self,
        instance: InstanceState,
        case_input: dict[str, Any],
        model: str = "default",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Execute a workflow using the existing engine."""
        from engine.composer import load_three_layer, run_workflow
        from engine.agentic import run_agentic_workflow
        from engine.nodes import set_trace

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
            set_trace(_CoordinatorTrace(self, instance.instance_id))

        # Build action registry if needed
        action_registry = None
        if not is_agentic:
            has_act = any(
                s["primitive"] == "act"
                for s in config.get("steps", [])
            )
            if has_act:
                action_registry = self._build_action_registry(case_input)

        # Execute
        if is_agentic:
            return run_agentic_workflow(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
            )
        else:
            return run_workflow(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
                action_registry=action_registry,
            )

    def _execute_workflow_from_state(
        self,
        instance: InstanceState,
        state_snapshot: dict[str, Any],
        resume_step: str,
        model: str = "default",
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Resume a workflow from a saved state snapshot at the given step.
        Uses mid-graph entry: builds a subgraph from resume_step forward,
        feeds it the pre-populated state with prior step outputs intact.
        """
        from engine.composer import load_three_layer, run_workflow_from_step

        workflow_path = self._find_workflow(instance.workflow_type)
        domain_path = self._find_domain(instance.domain)
        config, _ = load_three_layer(workflow_path, domain_path)

        tool_registry = self._build_tool_registry(
            state_snapshot.get("input", {})
        )
        action_registry = None
        if config.get("mode") != "agentic":
            has_act = any(
                s["primitive"] == "act"
                for s in config.get("steps", [])
            )
            if has_act:
                action_registry = self._build_action_registry(
                    state_snapshot.get("input", {})
                )

        return run_workflow_from_step(
            config, state_snapshot, resume_step,
            model, temperature,
            tool_registry=tool_registry,
            action_registry=action_registry,
        )

    # ─── Result Extraction ───────────────────────────────────────────

    def _extract_result_summary(self, final_state: dict[str, Any]) -> dict[str, Any]:
        """Extract a structured summary from workflow final state."""
        steps = final_state.get("steps", [])
        summary: dict[str, Any] = {
            "step_count": len(steps),
            "steps": [],
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
            elif prim == "think":
                step_summary["risk_score"] = output.get("risk_score")
                step_summary["decision"] = output.get("decision")
                # Pull recommendation from dedicated field first, fall back to decision
                step_summary["recommendation"] = (
                    output.get("recommendation") or output.get("decision")
                )
                step_summary["reasoning"] = str(output.get("reasoning", ""))[:500]
                step_summary["thought"] = str(output.get("thought", ""))[:500]
                step_summary["confidence"] = output.get("confidence")
                step_summary["conclusions"] = output.get("conclusions", [])
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

    def _build_tool_registry(self, case_input: dict[str, Any]):
        """
        Build tool registry with correct priority:
          1. MCP server (production)
          2. Case JSON tools (specific to this run — always take priority)
          3. Fixtures DB (fills in gaps for tools not in case JSON)

        Case JSON keys that look like tool data (dicts/lists) become tools.
        If a case JSON provides tools, it's the primary source — the LLM
        should call the tools the workflow specifies, not everything available.
        """
        data_mcp_url = os.environ.get("DATA_MCP_URL", "")
        data_mcp_cmd = os.environ.get("DATA_MCP_CMD", "")

        if data_mcp_url or data_mcp_cmd:
            import asyncio
            from engine.providers import MCPProvider
            from engine.tools import ToolRegistry
            registry = ToolRegistry()
            if data_mcp_url:
                provider = MCPProvider(transport="http", url=data_mcp_url)
            else:
                parts = data_mcp_cmd.split()
                provider = MCPProvider(
                    transport="stdio", command=parts[0], args=parts[1:],
                )
            async def _connect():
                await provider.connect()
                provider.register_all(registry)
            asyncio.get_event_loop().run_until_complete(_connect())
            return registry

        # Check if case_input has tool-shaped data (dict values with tool-like keys)
        has_case_tools = any(
            isinstance(v, (dict, list)) and k.startswith("get_")
            for k, v in case_input.items()
        )

        if has_case_tools:
            # Case JSON is the primary source for this run
            from engine.tools import create_case_registry
            return create_case_registry(case_input)

        # Fallback: try fixtures DB, then case registry
        try:
            from fixtures.api import create_service_registry
            from fixtures.db import DB_PATH
            if DB_PATH.exists():
                return create_service_registry()
        except (ImportError, FileNotFoundError):
            pass

        from engine.tools import create_case_registry
        return create_case_registry(case_input)

    def _build_action_registry(self, case_input: dict[str, Any]):
        """Build action registry."""
        from engine.actions import create_simulation_registry
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
        "think": "💭",
        "act": "⚡",
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
        elif primitive == "investigate":
            finding = output.get("finding", "?")[:70]
            flags = output.get("evidence_flags", [])
            self._log(f"    → {finding}...")
            if flags:
                self._log(f"    → flags: {flags}")
        elif primitive == "retrieve":
            data = output.get("data", {})
            sources = output.get("sources_queried", [])
            n_ok = sum(1 for s in sources if s.get("status") == "success")
            # Only count as EMPTY when record_count is explicitly 0,
            # not when it's absent/None (which means "not tracked")
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
            self._log(f"    → generated {len(str(output.get('artifact', ''))):,} chars")
        elif primitive == "think":
            decision = output.get("decision", "")[:70]
            self._log(f"    → {decision}")
        elif primitive == "verify":
            conforms = output.get("conforms", "?")
            n_viol = len(output.get("violations", []))
            self._log(f"    → conforms: {conforms}"
                       f"{f' ({n_viol} violations)' if n_viol else ''}")

    def on_parse_error(self, step_name, error):
        self._log(f"    ⚠ PARSE ERROR: {error[:120]}")

    def on_route_decision(self, from_step, to_step, decision_type, reason):
        target = "END" if to_step == "__end__" else to_step
        self._log(f"    → route → {target} ({decision_type})")

    def on_retrieve_start(self, step_name, source_name):
        self._log(f"    📡 {source_name}...")

    def on_retrieve_end(self, step_name, source_name, status, latency_ms):
        icon = "✓" if status == "success" else "✗"
        self._log(f"    📡 {source_name}: {icon} ({latency_ms:.0f}ms)")
