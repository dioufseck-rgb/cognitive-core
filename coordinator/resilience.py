"""
Cognitive Core — DDD Resilience Layer

Addresses four production failure modes identified in review:

  1. Observer-State Divergence (Resume Revalidation)
     The world changes while an agent sleeps. When resuming,
     the platform must verify that the assumptions underlying
     the original Need are still valid.

  2. Subjective Acceptance Loop (Semantic Oscillation Detection)
     A requestor's quality bar is unachievable by any provider.
     The Need re-issues indefinitely, creating a shadow queue
     invisible to the solver.

  3. Stochastic Capacity Erosion (Graceful Revocation)
     LLM service times have high variance. A reservation TTL
     expires while work is 90% done. The system needs checkpoint-
     and-exit, not just hard kill.

  4. Saga of Side Effects (Cross-Workflow Compensation)
     Work Order 1 sends an email, Work Order 2 fails. The system
     is inconsistent. The DDD framework needs first-class
     compensating transactions across the delegation boundary.

Design principle: Platform owns state (stateless agent).
The agent receives a fully reconstructed context on every resume.
This solves 90% of auditability and recovery but pays the
context-loading tax. We accept that tax because:
  - Resume is rare (suspensions, not every step)
  - Context loading is O(steps × avg_output_size), bounded by compaction
  - The alternative (agent-owned state) makes audit impossible
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

log = logging.getLogger("cognitive_core.resilience")


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 1: OBSERVER-STATE DIVERGENCE
# Resume Revalidation Guards
# ═══════════════════════════════════════════════════════════════════

class StalenessVerdict(str, Enum):
    """Result of a resume revalidation check."""
    VALID = "valid"              # Original assumptions still hold
    STALE = "stale"              # Context changed, needs re-evaluation
    INVALIDATED = "invalidated"  # Original Need is no longer valid
    ERROR = "error"              # Revalidation check itself failed


@dataclass
class RevalidationResult:
    """
    Output of resume revalidation.

    When verdict is STALE, the enrichment dict carries updated context
    that the agent receives on resume. The agent re-runs its step with
    fresh data and may produce a different decision.

    When verdict is INVALIDATED, the workflow should not resume at the
    same step. Instead it either re-evaluates from a checkpoint or
    escalates to HITL.
    """
    verdict: StalenessVerdict
    reason: str = ""
    enrichment: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    checks_run: list[str] = field(default_factory=list)


@dataclass
class RevalidationGuard:
    """
    A single revalidation check registered for a suspension.

    Guards are domain-specific. Examples:
    - "case_status_active": verify case not canceled while suspended
    - "entity_version_match": verify entity hasn't been modified
    - "sla_not_breached": verify SLA hasn't expired
    - "regulatory_hold_check": verify no regulatory freeze imposed
    """
    name: str
    check_fn: Callable[[dict[str, Any]], StalenessVerdict] | None = None
    # For declarative guards (no custom function):
    entity_type: str = ""       # e.g., "case", "claim", "account"
    field_path: str = ""        # dot-path to check, e.g., "status"
    expected_value: Any = None  # value at suspension time
    # Metadata
    registered_at: float = 0.0


class ResumeRevalidator:
    """
    Failure Mode 1: Observer-State Divergence defense.

    Before an agent resumes, the platform runs all registered guards
    against the CURRENT state of the world (not the snapshot).

    Integration: Called by Coordinator._try_resume_after_all_providers
    BEFORE injecting delegation results and calling resume().

    The agent is stateless — it receives whatever context the platform
    provides. If the platform detects staleness, it enriches the
    context with fresh data so the agent makes a correct decision.
    """

    def __init__(self):
        # instance_id → list of guards
        self._guards: dict[str, list[RevalidationGuard]] = {}
        # instance_id → entity version at suspension time
        self._snapshots: dict[str, dict[str, Any]] = {}

    def register_guard(
        self,
        instance_id: str,
        guard: RevalidationGuard,
    ) -> None:
        """Register a revalidation guard for a suspended instance."""
        guard.registered_at = time.time()
        self._guards.setdefault(instance_id, []).append(guard)

    def snapshot_entities(
        self,
        instance_id: str,
        entities: dict[str, Any],
    ) -> None:
        """
        Capture entity state at suspension time.
        Keys are entity paths, values are their current state.
        """
        self._snapshots[instance_id] = {
            k: v for k, v in entities.items()
        }

    def revalidate(
        self,
        instance_id: str,
        current_state: dict[str, Any],
        entity_loader: Callable[[str, str], Any] | None = None,
    ) -> RevalidationResult:
        """
        Run all guards for a suspended instance.

        current_state: the work order results / delegation results
        entity_loader: function(entity_type, entity_id) → current entity
                       Used to fetch fresh entity state for comparison.

        Returns RevalidationResult with verdict and any enrichment.
        """
        guards = self._guards.get(instance_id, [])
        if not guards:
            # No guards registered → valid by default
            return RevalidationResult(
                verdict=StalenessVerdict.VALID,
                reason="No revalidation guards registered",
            )

        start = time.time()
        checks_run = []
        enrichment = {}
        worst_verdict = StalenessVerdict.VALID

        for guard in guards:
            check_name = guard.name
            checks_run.append(check_name)

            try:
                if guard.check_fn:
                    # Custom function guard
                    verdict = guard.check_fn(current_state)
                elif guard.entity_type and guard.field_path and entity_loader:
                    # Declarative entity guard
                    snapshot = self._snapshots.get(instance_id, {})
                    entity_key = f"{guard.entity_type}.{guard.field_path}"
                    expected = guard.expected_value or snapshot.get(entity_key)

                    # Load current value
                    current = entity_loader(guard.entity_type, guard.field_path)
                    if current != expected:
                        verdict = StalenessVerdict.STALE
                        enrichment[f"_revalidation.{check_name}"] = {
                            "expected": expected,
                            "actual": current,
                            "entity_type": guard.entity_type,
                            "field_path": guard.field_path,
                        }
                        log.warning(
                            "Revalidation STALE: %s — %s changed from %s to %s",
                            instance_id, entity_key, expected, current,
                        )
                    else:
                        verdict = StalenessVerdict.VALID
                else:
                    verdict = StalenessVerdict.VALID  # unrunnable guard = pass

                # Track worst verdict
                if verdict == StalenessVerdict.INVALIDATED:
                    worst_verdict = StalenessVerdict.INVALIDATED
                elif verdict == StalenessVerdict.STALE and worst_verdict == StalenessVerdict.VALID:
                    worst_verdict = StalenessVerdict.STALE

            except Exception as e:
                log.error("Revalidation guard %s failed: %s", check_name, e)
                checks_run[-1] = f"{check_name}:ERROR"
                if worst_verdict == StalenessVerdict.VALID:
                    worst_verdict = StalenessVerdict.ERROR

        elapsed = time.time() - start

        reason = ""
        if worst_verdict == StalenessVerdict.STALE:
            stale_checks = [c for c in checks_run if not c.endswith(":ERROR")]
            reason = f"Context stale — enrichment provided for {len(enrichment)} fields"
        elif worst_verdict == StalenessVerdict.INVALIDATED:
            reason = "Original need invalidated — do not resume at same step"

        return RevalidationResult(
            verdict=worst_verdict,
            reason=reason,
            enrichment=enrichment,
            elapsed_seconds=elapsed,
            checks_run=checks_run,
        )

    def cleanup(self, instance_id: str) -> None:
        """Remove guards and snapshots after instance completes."""
        self._guards.pop(instance_id, None)
        self._snapshots.pop(instance_id, None)


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 2: SUBJECTIVE ACCEPTANCE LOOP
# Semantic Oscillation Detection
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NeedAttempt:
    """Record of a single attempt to fulfill a Need."""
    attempt_number: int
    provider_resource_id: str
    work_order_id: str
    completed_at: float
    accepted: bool
    rejection_reason: str = ""


@dataclass
class OscillationState:
    """Tracking state for a single Need across attempts."""
    need_type: str
    case_id: str
    correlation_id: str
    attempts: list[NeedAttempt] = field(default_factory=list)
    created_at: float = 0.0
    resolved: bool = False
    resolution: str = ""  # "accepted", "exhausted", "escalated"

    @property
    def rejection_count(self) -> int:
        return sum(1 for a in self.attempts if not a.accepted)

    @property
    def distinct_providers_tried(self) -> set[str]:
        return {a.provider_resource_id for a in self.attempts}


class OscillationDetector:
    """
    Failure Mode 2: Subjective Acceptance Loop defense.

    Detects when a Need is rejected repeatedly, either by the same
    provider (quality mismatch) or by cycling through all eligible
    providers (unsatisfiable criteria).

    Triggers:
    - After N rejections of the same Need → flag as "semantic oscillation"
    - After all eligible providers tried → flag as "unsatisfiable"
    - When distinct rejection reasons cluster → flag as "criteria conflict"

    Resolution: Escalation to HITL with full oscillation audit trail.
    The solver cannot fix subjective quality disagreements.
    """

    def __init__(
        self,
        max_rejections: int = 3,
        max_same_provider_rejections: int = 2,
    ):
        self.max_rejections = max_rejections
        self.max_same_provider = max_same_provider_rejections
        # (need_type, case_id) → OscillationState
        self._tracking: dict[tuple[str, str], OscillationState] = {}

    def record_attempt(
        self,
        need_type: str,
        case_id: str,
        correlation_id: str,
        provider_resource_id: str,
        work_order_id: str,
        accepted: bool,
        rejection_reason: str = "",
    ) -> OscillationVerdict:
        """
        Record an attempt to fulfill a Need and check for oscillation.

        Returns OscillationVerdict indicating whether to proceed,
        retry with different provider, or escalate.
        """
        key = (need_type, case_id)
        state = self._tracking.get(key)
        if not state:
            state = OscillationState(
                need_type=need_type,
                case_id=case_id,
                correlation_id=correlation_id,
                created_at=time.time(),
            )
            self._tracking[key] = state

        attempt = NeedAttempt(
            attempt_number=len(state.attempts) + 1,
            provider_resource_id=provider_resource_id,
            work_order_id=work_order_id,
            completed_at=time.time(),
            accepted=accepted,
            rejection_reason=rejection_reason,
        )
        state.attempts.append(attempt)

        if accepted:
            state.resolved = True
            state.resolution = "accepted"
            return OscillationVerdict(
                action=OscillationAction.PROCEED,
                reason="Need accepted",
            )

        # Check oscillation conditions
        return self._evaluate(state, provider_resource_id)

    def _evaluate(
        self,
        state: OscillationState,
        last_provider: str,
    ) -> OscillationVerdict:
        """Evaluate whether oscillation thresholds are breached."""
        # Condition 1: Too many total rejections
        if state.rejection_count >= self.max_rejections:
            state.resolved = True
            state.resolution = "exhausted"
            log.warning(
                "Oscillation detected: need=%s case=%s rejections=%d → escalating",
                state.need_type, state.case_id, state.rejection_count,
            )
            return OscillationVerdict(
                action=OscillationAction.ESCALATE,
                reason=f"Need rejected {state.rejection_count} times "
                       f"(threshold: {self.max_rejections})",
                oscillation_state=state,
            )

        # Condition 2: Same provider rejected multiple times
        same_provider_rejects = sum(
            1 for a in state.attempts
            if a.provider_resource_id == last_provider and not a.accepted
        )
        if same_provider_rejects >= self.max_same_provider:
            return OscillationVerdict(
                action=OscillationAction.RETRY_DIFFERENT_PROVIDER,
                reason=f"Provider {last_provider} rejected "
                       f"{same_provider_rejects} times — try another",
                exclude_providers={last_provider},
            )

        # Condition 3: Check if rejection reasons cluster (all same reason)
        reasons = [a.rejection_reason for a in state.attempts if not a.accepted and a.rejection_reason]
        if len(reasons) >= 2 and len(set(reasons)) == 1:
            return OscillationVerdict(
                action=OscillationAction.ESCALATE,
                reason=f"All rejections cite same reason: '{reasons[0]}' — "
                       f"criteria may be unsatisfiable",
                oscillation_state=state,
            )

        # Under threshold — retry
        return OscillationVerdict(
            action=OscillationAction.RETRY,
            reason=f"Rejection {state.rejection_count}/{self.max_rejections} — retrying",
        )

    def get_state(self, need_type: str, case_id: str) -> OscillationState | None:
        return self._tracking.get((need_type, case_id))

    def cleanup(self, case_id: str) -> None:
        """Remove all tracking for a case."""
        keys = [k for k in self._tracking if k[1] == case_id]
        for k in keys:
            del self._tracking[k]


class OscillationAction(str, Enum):
    PROCEED = "proceed"                        # Need accepted, continue
    RETRY = "retry"                            # Retry with any provider
    RETRY_DIFFERENT_PROVIDER = "retry_different"  # Retry, exclude provider
    ESCALATE = "escalate"                      # Circuit-break → HITL


@dataclass
class OscillationVerdict:
    """Decision from the oscillation detector."""
    action: OscillationAction
    reason: str
    exclude_providers: set[str] = field(default_factory=set)
    oscillation_state: OscillationState | None = None


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 3: STOCHASTIC CAPACITY EROSION
# Graceful Revocation Protocol
# ═══════════════════════════════════════════════════════════════════

class RevocationPolicy(str, Enum):
    """How to reclaim a slot from a running work order."""
    HARD_KILL = "hard_kill"          # Immediate termination, work lost
    CHECKPOINT_EXIT = "checkpoint_exit"  # Signal to save state and yield
    EXTEND_TTL = "extend_ttl"        # Grant more time (up to max)
    PREEMPT_QUEUE = "preempt_queue"  # Queue the preemption for next safe point


@dataclass
class RevocationSignal:
    """
    Signal sent to a running work order that its capacity is being reclaimed.

    The agent checks this signal at safe checkpoints (between tool calls,
    between LLM inference rounds). If present, the agent should:
    1. Serialize its current partial state
    2. Return a CHECKPOINT result (not COMPLETED or FAILED)
    3. The coordinator re-queues the work order with the partial state
    """
    work_order_id: str
    reason: str
    policy: RevocationPolicy
    issued_at: float
    deadline_seconds: float = 30.0    # time to checkpoint before hard kill
    preemptor_work_order_id: str = ""  # who's taking the slot


@dataclass
class RevocationConfig:
    """Per-domain revocation policy from YAML."""
    default_policy: RevocationPolicy = RevocationPolicy.CHECKPOINT_EXIT
    max_ttl_extensions: int = 2
    extension_seconds: float = 30.0
    max_total_ttl_seconds: float = 300.0
    checkpoint_grace_seconds: float = 30.0
    # Priority-based: higher-priority work can preempt lower
    priority_preemption: bool = True
    priority_order: list[str] = field(
        default_factory=lambda: ["critical", "high", "routine"]
    )


class CapacityRevocationManager:
    """
    Failure Mode 3: Stochastic Capacity Erosion defense.

    When a reservation TTL is about to expire while work is in progress,
    this manager decides whether to extend, checkpoint, or kill.

    Key insight: LLM service times have high variance (TTFT degradation,
    long chain-of-thought). Hard kills lose partial work. Soft signals
    risk SLA violations on waiting work. The policy navigates this trade-off.

    Integration: Called by the Coordinator's TTL sweep loop.
    """

    def __init__(self, config: RevocationConfig | None = None):
        self.config = config or RevocationConfig()
        # work_order_id → RevocationSignal
        self._active_signals: dict[str, RevocationSignal] = {}
        # work_order_id → number of TTL extensions granted
        self._extensions: dict[str, int] = {}

    def evaluate_expiring_reservation(
        self,
        work_order_id: str,
        reservation_ttl_remaining: float,
        work_order_priority: str,
        work_order_progress: float,  # 0.0–1.0 estimated
        waiting_queue_depth: int,
        highest_waiter_priority: str | None = None,
    ) -> RevocationSignal | None:
        """
        Decide what to do about an expiring reservation.

        Returns None if no action needed, or a RevocationSignal.
        """
        # If TTL still has > 5s remaining, no action yet
        if reservation_ttl_remaining > 5.0:
            return None

        # Check if we can extend
        extensions_used = self._extensions.get(work_order_id, 0)
        total_ttl = (extensions_used + 1) * self.config.extension_seconds

        can_extend = (
            extensions_used < self.config.max_ttl_extensions
            and total_ttl < self.config.max_total_ttl_seconds
        )

        # Decision tree
        if can_extend and not self._should_preempt(
            work_order_priority, highest_waiter_priority
        ):
            # Extend TTL — work is in progress, no higher-priority waiter
            self._extensions[work_order_id] = extensions_used + 1
            log.info(
                "TTL extension %d/%d granted for %s (%.0f%% complete)",
                extensions_used + 1, self.config.max_ttl_extensions,
                work_order_id, work_order_progress * 100,
            )
            return RevocationSignal(
                work_order_id=work_order_id,
                reason=f"TTL extension {extensions_used + 1}",
                policy=RevocationPolicy.EXTEND_TTL,
                issued_at=time.time(),
                deadline_seconds=self.config.extension_seconds,
            )

        # Must revoke — but how?
        if work_order_progress > 0.5:
            # More than half done → checkpoint
            policy = RevocationPolicy.CHECKPOINT_EXIT
            reason = (
                f"TTL expired, {work_order_progress:.0%} complete — "
                f"checkpoint requested"
            )
        elif self._should_preempt(work_order_priority, highest_waiter_priority):
            # Higher priority is waiting → preempt
            policy = RevocationPolicy.PREEMPT_QUEUE
            reason = (
                f"Preempted by {highest_waiter_priority}-priority work "
                f"(current: {work_order_priority})"
            )
        else:
            # Use configured default
            policy = self.config.default_policy
            reason = f"TTL expired, no extensions remaining"

        signal = RevocationSignal(
            work_order_id=work_order_id,
            reason=reason,
            policy=policy,
            issued_at=time.time(),
            deadline_seconds=self.config.checkpoint_grace_seconds,
        )
        self._active_signals[work_order_id] = signal
        log.warning(
            "Revocation signal: %s → %s (%s)",
            work_order_id, policy.value, reason,
        )
        return signal

    def check_signal(self, work_order_id: str) -> RevocationSignal | None:
        """
        Check if a revocation signal is pending for a work order.

        Called by the agent at safe checkpoints (between tool calls).
        If a signal exists, the agent should honor it.
        """
        return self._active_signals.get(work_order_id)

    def acknowledge(self, work_order_id: str) -> None:
        """Agent acknowledged the revocation signal."""
        self._active_signals.pop(work_order_id, None)

    def cleanup(self, work_order_id: str) -> None:
        """Remove all tracking for a completed work order."""
        self._active_signals.pop(work_order_id, None)
        self._extensions.pop(work_order_id, None)

    def _should_preempt(
        self,
        current_priority: str,
        waiter_priority: str | None,
    ) -> bool:
        """Check if waiter has higher priority than current work."""
        if not self.config.priority_preemption or not waiter_priority:
            return False
        order = self.config.priority_order
        if current_priority in order and waiter_priority in order:
            return order.index(waiter_priority) < order.index(current_priority)
        return False


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 4: SAGA OF SIDE EFFECTS
# Cross-Workflow Compensation
# ═══════════════════════════════════════════════════════════════════

class CompensationScope(str, Enum):
    """Scope of a compensation entry."""
    STEP = "step"              # Single step within a workflow
    WORK_ORDER = "work_order"  # Entire work order (delegation boundary)
    SAGA = "saga"              # Cross-workflow saga spanning multiple WOs


@dataclass
class SagaCompensationEntry:
    """
    A compensating transaction registered at the delegation boundary.

    When an agent executes a side-effecting action (send_email,
    update_database, submit_form), it registers the inverse action
    here BEFORE executing the forward action.

    Key difference from engine/compensation.py: that module handles
    single-workflow compensation. This handles CROSS-WORKFLOW sagas
    where Work Order 1 succeeded and Work Order 2 failed.
    """
    entry_id: str
    saga_id: str              # correlation_id — ties the saga together
    work_order_id: str
    step_name: str
    action_description: str
    inverse_action: dict[str, Any]  # serialized compensating transaction
    scope: CompensationScope = CompensationScope.WORK_ORDER
    status: str = "pending"   # pending | confirmed | compensated | failed | skipped
    created_at: float = 0.0
    executed_at: float | None = None
    compensated_at: float | None = None
    error: str = ""


class SagaCoordinator:
    """
    Failure Mode 4: Saga of Side Effects defense.

    Manages compensating transactions across the delegation boundary.

    When a parent workflow spawns multiple work orders and one fails,
    the SagaCoordinator walks back the CONFIRMED entries of the
    COMPLETED work orders in reverse chronological order.

    The saga is identified by correlation_id — the same ID that
    threads through all delegated work orders.

    Integration:
    - On Work Order side-effect: saga.register(correlation_id, wo_id, inverse)
    - On Work Order success: saga.confirm(entry_id)
    - On downstream failure: saga.compensate(correlation_id, handler)
    """

    def __init__(self):
        # saga_id (correlation_id) → ordered list of entries
        self._sagas: dict[str, list[SagaCompensationEntry]] = {}
        self._entry_counter = 0

    def register(
        self,
        saga_id: str,
        work_order_id: str,
        step_name: str,
        action_description: str,
        inverse_action: dict[str, Any],
        scope: CompensationScope = CompensationScope.WORK_ORDER,
    ) -> str:
        """
        Register a compensating transaction BEFORE the forward action.

        Returns the entry_id for later confirm/compensate.
        """
        self._entry_counter += 1
        entry_id = f"saga_{self._entry_counter}_{hashlib.sha256(f'{saga_id}:{work_order_id}:{step_name}'.encode()).hexdigest()[:8]}"

        entry = SagaCompensationEntry(
            entry_id=entry_id,
            saga_id=saga_id,
            work_order_id=work_order_id,
            step_name=step_name,
            action_description=action_description,
            inverse_action=inverse_action,
            scope=scope,
            status="pending",
            created_at=time.time(),
        )
        self._sagas.setdefault(saga_id, []).append(entry)
        log.info(
            "Saga compensation registered: saga=%s wo=%s action=%s",
            saga_id, work_order_id, action_description,
        )
        return entry_id

    def confirm(self, entry_id: str) -> bool:
        """Mark forward action as executed. Now eligible for compensation."""
        for entries in self._sagas.values():
            for entry in entries:
                if entry.entry_id == entry_id:
                    if entry.status == "pending":
                        entry.status = "confirmed"
                        entry.executed_at = time.time()
                        return True
                    return False  # already confirmed or compensated
        return False  # not found

    def compensate(
        self,
        saga_id: str,
        handler: Callable[[SagaCompensationEntry], bool] | None = None,
        failed_work_order_id: str = "",
    ) -> list[SagaCompensationEntry]:
        """
        Execute compensations for a saga in reverse chronological order.

        Only CONFIRMED entries are compensated (pending = action never ran).

        Compensates entries from ALL work orders in the saga EXCEPT
        the one that failed (which never completed successfully).

        Args:
            saga_id: The correlation_id of the saga
            handler: Function that takes entry and returns True if compensation
                     succeeded. If None, entries are marked SKIPPED (→ HITL).
            failed_work_order_id: The work order that triggered compensation.
                                  Its entries are skipped (action never completed).

        Returns list of entries with updated statuses.
        """
        entries = self._sagas.get(saga_id, [])
        if not entries:
            return []

        # Filter to confirmed entries NOT from the failed work order
        to_compensate = [
            e for e in entries
            if e.status == "confirmed"
            and e.work_order_id != failed_work_order_id
        ]

        # Reverse chronological order — last action compensated first
        to_compensate.sort(key=lambda e: e.executed_at or 0, reverse=True)

        results = []
        for entry in to_compensate:
            if handler is None:
                entry.status = "skipped"
                entry.error = "No compensation handler — requires human intervention"
                log.warning(
                    "Saga compensation SKIPPED: saga=%s wo=%s step=%s",
                    saga_id, entry.work_order_id, entry.step_name,
                )
            else:
                try:
                    success = handler(entry)
                    if success:
                        entry.status = "compensated"
                        entry.compensated_at = time.time()
                        log.info(
                            "Saga compensation SUCCESS: saga=%s wo=%s step=%s",
                            saga_id, entry.work_order_id, entry.step_name,
                        )
                    else:
                        entry.status = "failed"
                        entry.error = "Handler returned False"
                        log.error(
                            "Saga compensation FAILED: saga=%s wo=%s step=%s",
                            saga_id, entry.work_order_id, entry.step_name,
                        )
                except Exception as e:
                    entry.status = "failed"
                    entry.error = str(e)[:500]
                    log.error(
                        "Saga compensation ERROR: saga=%s wo=%s step=%s error=%s",
                        saga_id, entry.work_order_id, entry.step_name, e,
                    )
            results.append(entry)

        return results

    def get_saga_entries(self, saga_id: str) -> list[SagaCompensationEntry]:
        """Get all entries for a saga (for audit trail)."""
        return list(self._sagas.get(saga_id, []))

    def pending_compensations(self, saga_id: str) -> int:
        """Count entries that would need compensation if saga failed now."""
        return sum(
            1 for e in self._sagas.get(saga_id, [])
            if e.status == "confirmed"
        )

    def cleanup(self, saga_id: str) -> None:
        """Remove saga after all entries are final."""
        self._sagas.pop(saga_id, None)
