"""
Cognitive Core — Production Hardening Layer

Five production-readiness requirements that bridge the gap between
"working state machines" and "resilient production system":

  1. Dispatch Decision Record (DDR) — Persisted artifact per work order
  2. Policy Versioning + Rollout Modes — Static/MPC/ADP with shadow/canary
  3. Explicit Partial-Failure Semantics — retry/escalate/degrade/abort
  4. Reservation Protocol Specification — acquire/commit/release/expire/crash
  5. Learning Scope Constraints — Hard-coded guardrails on what learning can touch

These are CONTRACTS, not implementations of the underlying mechanisms
(which live in ddd.py, optimizer.py, resilience.py). This module
defines the schemas, invariants, and governance that make those
mechanisms auditable and safe in production.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════
# 1. DISPATCH DECISION RECORD (DDR)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DispatchDecisionRecord:
    """
    Persisted, immutable audit artifact for every dispatch decision.

    One DDR per work order. Created at dispatch time, never modified.
    Stored in the action_ledger with action_type = "dispatch_decision_record".

    This is the "receipt" that answers: Why was this work order assigned
    to this resource? What constraints were active? What alternatives
    were considered? What policy version made the decision?

    DDRs are queryable for:
    - Compliance audits (who decided, why, when)
    - Solver diagnostics (cost matrix snapshot, objective terms)
    - Dispute resolution (show the eligible set and scores)
    - Policy A/B testing (compare decisions across versions)
    """
    # Identity
    ddr_id: str = ""
    work_order_id: str = ""
    correlation_id: str = ""
    case_id: str = ""
    trace_id: str = ""

    # Timing
    created_at: float = 0.0

    # Policy context
    policy_version: str = ""          # semver of the dispatch policy
    policy_mode: str = "static"       # static | mpc | adp
    solver_name: str = ""
    solver_version: str = ""
    solver_seed: int = 0

    # Eligible set (FULL — every resource considered)
    eligible_set: list[DDREligibilityEntry] = field(default_factory=list)
    excluded_set: list[DDREligibilityEntry] = field(default_factory=list)

    # Objective terms (what the solver optimized)
    objective_weights: dict[str, float] = field(default_factory=dict)
    # Per-resource scores for all eligible resources
    candidate_scores: list[DDRCandidateScore] = field(default_factory=list)

    # Decision
    selected_resource_id: str | None = None
    selection_tier: str = ""          # optimal | fallback | exploration
    reservation_id: str | None = None

    # Reason codes (machine-readable)
    reason_codes: list[str] = field(default_factory=list)
    # Human-readable narrative
    reason_narrative: str = ""

    # Constraints active at decision time
    active_constraints: list[str] = field(default_factory=list)
    # Adaptive adjustments applied (CFA/VFA)
    adaptive_adjustments: dict[str, float] = field(default_factory=dict)

    # Hash of inputs (for determinism verification)
    input_hash: str = ""

    def to_ledger_entry(self) -> dict[str, Any]:
        """Serialize for storage in action_ledger."""
        return {
            "ddr_id": self.ddr_id,
            "work_order_id": self.work_order_id,
            "correlation_id": self.correlation_id,
            "case_id": self.case_id,
            "trace_id": self.trace_id,
            "policy_version": self.policy_version,
            "policy_mode": self.policy_mode,
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "solver_seed": self.solver_seed,
            "eligible_count": len(self.eligible_set),
            "excluded_count": len(self.excluded_set),
            "eligible_set": [e.to_dict() for e in self.eligible_set],
            "excluded_set": [e.to_dict() for e in self.excluded_set],
            "objective_weights": self.objective_weights,
            "candidate_scores": [c.to_dict() for c in self.candidate_scores],
            "selected_resource_id": self.selected_resource_id,
            "selection_tier": self.selection_tier,
            "reservation_id": self.reservation_id,
            "reason_codes": self.reason_codes,
            "reason_narrative": self.reason_narrative,
            "active_constraints": self.active_constraints,
            "adaptive_adjustments": self.adaptive_adjustments,
            "input_hash": self.input_hash,
        }

    @staticmethod
    def compute_input_hash(
        work_order_id: str,
        resource_ids: list[str],
        objective_weights: dict[str, float],
    ) -> str:
        """Deterministic hash of dispatch inputs (Invariant 9)."""
        h = hashlib.sha256()
        h.update(work_order_id.encode())
        for rid in sorted(resource_ids):
            h.update(rid.encode())
        h.update(json.dumps(objective_weights, sort_keys=True).encode())
        return h.hexdigest()[:16]


@dataclass
class DDREligibilityEntry:
    """One row in the eligible/excluded set of a DDR."""
    resource_id: str
    eligible: bool
    constraint_name: str = ""    # which constraint excluded it (if excluded)
    constraint_reason: str = ""  # human-readable reason
    capacity_at_decision: int = 0
    circuit_breaker_status: str = "closed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "eligible": self.eligible,
            "constraint_name": self.constraint_name,
            "constraint_reason": self.constraint_reason,
            "capacity_at_decision": self.capacity_at_decision,
            "circuit_breaker_status": self.circuit_breaker_status,
        }


@dataclass
class DDRCandidateScore:
    """Score breakdown for one candidate resource in a DDR."""
    resource_id: str
    total_score: float
    rank: int                       # 1 = best
    feature_scores: dict[str, float] = field(default_factory=dict)
    adaptive_adjustment: float = 0.0  # CFA delta applied

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "total_score": self.total_score,
            "rank": self.rank,
            "feature_scores": self.feature_scores,
            "adaptive_adjustment": self.adaptive_adjustment,
        }


def build_ddr(
    work_order_id: str,
    correlation_id: str,
    case_id: str,
    trace_id: str,
    policy_version: str,
    policy_mode: str,
    solver_name: str,
    solver_version: str,
    solver_seed: int,
    eligible_entries: list[DDREligibilityEntry],
    excluded_entries: list[DDREligibilityEntry],
    objective_weights: dict[str, float],
    candidate_scores: list[DDRCandidateScore],
    selected_resource_id: str | None,
    selection_tier: str,
    reservation_id: str | None = None,
    reason_codes: list[str] | None = None,
    active_constraints: list[str] | None = None,
    adaptive_adjustments: dict[str, float] | None = None,
) -> DispatchDecisionRecord:
    """Factory function for building a DDR with auto-computed fields."""
    input_hash = DispatchDecisionRecord.compute_input_hash(
        work_order_id,
        [e.resource_id for e in eligible_entries],
        objective_weights,
    )

    reason_narrative = _build_narrative(
        selected_resource_id, selection_tier,
        eligible_entries, excluded_entries, candidate_scores,
    )

    return DispatchDecisionRecord(
        ddr_id=f"ddr_{hashlib.sha256(f'{work_order_id}:{time.time()}'.encode()).hexdigest()[:12]}",
        work_order_id=work_order_id,
        correlation_id=correlation_id,
        case_id=case_id,
        trace_id=trace_id,
        created_at=time.time(),
        policy_version=policy_version,
        policy_mode=policy_mode,
        solver_name=solver_name,
        solver_version=solver_version,
        solver_seed=solver_seed,
        eligible_set=eligible_entries,
        excluded_set=excluded_entries,
        objective_weights=objective_weights,
        candidate_scores=candidate_scores,
        selected_resource_id=selected_resource_id,
        selection_tier=selection_tier,
        reservation_id=reservation_id,
        reason_codes=reason_codes or [],
        reason_narrative=reason_narrative,
        active_constraints=active_constraints or [],
        adaptive_adjustments=adaptive_adjustments or {},
        input_hash=input_hash,
    )


def _build_narrative(
    selected: str | None,
    tier: str,
    eligible: list[DDREligibilityEntry],
    excluded: list[DDREligibilityEntry],
    scores: list[DDRCandidateScore],
) -> str:
    """Auto-generate human-readable narrative for DDR."""
    parts = []
    parts.append(f"{len(eligible)} eligible, {len(excluded)} excluded.")
    if selected:
        rank_entry = next((s for s in scores if s.resource_id == selected), None)
        rank_str = f" (rank {rank_entry.rank}/{len(scores)})" if rank_entry else ""
        parts.append(f"Selected: {selected}{rank_str} via {tier}.")
    else:
        parts.append(f"No assignment — tier: {tier}.")
    if excluded:
        reasons = set(e.constraint_name for e in excluded if e.constraint_name)
        if reasons:
            parts.append(f"Exclusions: {', '.join(sorted(reasons))}.")
    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 2. POLICY VERSIONING + ROLLOUT MODES
# ═══════════════════════════════════════════════════════════════════

class PolicyMode(str, enum.Enum):
    """Dispatch policy operating mode (Spec Section 15)."""
    STATIC = "static"     # Baseline: YAML weights, no learning
    MPC = "mpc"           # Model Predictive Control: rolling-horizon
    ADP = "adp"           # Approximate Dynamic Programming: value estimates


class RolloutStage(str, enum.Enum):
    """Policy deployment stage."""
    TRAINING = "training"       # Offline, not serving decisions
    VALIDATION = "validation"   # Backtesting against historical data
    SHADOW = "shadow"           # Parallel execution, logging only
    CANARY = "canary"           # Serving 5% of traffic
    PRODUCTION = "production"   # Serving 100% of traffic
    ROLLBACK = "rollback"       # Reverted to prior version


@dataclass
class PolicyVersion:
    """
    Immutable policy version record.

    Every change to dispatch policy (objective weights, constraints,
    solver config, adaptive model) creates a new version. The old
    version remains available for rollback.
    """
    version_id: str               # semver: "1.2.3"
    mode: PolicyMode = PolicyMode.STATIC
    stage: RolloutStage = RolloutStage.TRAINING
    created_at: float = 0.0

    # Configuration snapshot
    objective_weights: dict[str, float] = field(default_factory=dict)
    solver_config: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    adaptive_model_id: str = ""   # reference to trained model artifact

    # Rollout tracking
    traffic_pct: float = 0.0      # 0.0–1.0, what % of traffic this serves
    promoted_at: float | None = None
    rolled_back_at: float | None = None
    rollback_reason: str = ""

    # Performance metrics (populated during canary/production)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyRolloutConfig:
    """Configuration for the policy rollout pipeline."""
    shadow_duration_hours: float = 24.0
    canary_traffic_pct: float = 0.05
    canary_duration_hours: float = 4.0
    # Auto-rollback triggers
    auto_rollback_sla_miss_rate: float = 0.05  # +5% SLA miss rate
    auto_rollback_window_hours: float = 4.0
    auto_rollback_error_rate: float = 0.10     # +10% error rate
    # MRM approval required for ADP mode
    require_mrm_approval_for_adp: bool = True


class PolicyManager:
    """
    Manages policy versions and rollout lifecycle.

    Invariant: Exactly one policy version is in PRODUCTION at any time.
    Shadow and canary versions run in parallel but don't affect dispatch
    decisions (shadow) or affect only a traffic slice (canary).
    """

    def __init__(self, config: PolicyRolloutConfig | None = None):
        self.config = config or PolicyRolloutConfig()
        self._versions: dict[str, PolicyVersion] = {}
        self._production_version: str | None = None

    def create_version(
        self,
        version_id: str,
        mode: PolicyMode = PolicyMode.STATIC,
        objective_weights: dict[str, float] | None = None,
        solver_config: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        adaptive_model_id: str = "",
    ) -> PolicyVersion:
        """Create a new policy version in TRAINING stage."""
        pv = PolicyVersion(
            version_id=version_id,
            mode=mode,
            stage=RolloutStage.TRAINING,
            created_at=time.time(),
            objective_weights=objective_weights or {},
            solver_config=solver_config or {},
            constraints=constraints or {},
            adaptive_model_id=adaptive_model_id,
        )
        self._versions[version_id] = pv
        return pv

    def promote(self, version_id: str, to_stage: RolloutStage) -> bool:
        """
        Promote a policy version to the next stage.

        Allowed transitions:
          TRAINING → VALIDATION → SHADOW → CANARY → PRODUCTION
        """
        pv = self._versions.get(version_id)
        if not pv:
            return False

        allowed = {
            RolloutStage.TRAINING: RolloutStage.VALIDATION,
            RolloutStage.VALIDATION: RolloutStage.SHADOW,
            RolloutStage.SHADOW: RolloutStage.CANARY,
            RolloutStage.CANARY: RolloutStage.PRODUCTION,
        }
        if allowed.get(pv.stage) != to_stage:
            return False

        # ADP mode requires MRM approval for production
        if (to_stage == RolloutStage.PRODUCTION
                and pv.mode == PolicyMode.ADP
                and self.config.require_mrm_approval_for_adp):
            if not pv.metrics.get("mrm_approved"):
                return False

        pv.stage = to_stage
        pv.promoted_at = time.time()

        if to_stage == RolloutStage.CANARY:
            pv.traffic_pct = self.config.canary_traffic_pct
        elif to_stage == RolloutStage.PRODUCTION:
            # Demote current production version
            if self._production_version:
                old = self._versions.get(self._production_version)
                if old and old.stage == RolloutStage.PRODUCTION:
                    old.stage = RolloutStage.ROLLBACK
                    old.rolled_back_at = time.time()
                    old.rollback_reason = f"Replaced by {version_id}"
            pv.traffic_pct = 1.0
            self._production_version = version_id

        return True

    def rollback(self, version_id: str, reason: str = "") -> bool:
        """Roll back a version. Restores previous production version."""
        pv = self._versions.get(version_id)
        if not pv:
            return False

        pv.stage = RolloutStage.ROLLBACK
        pv.rolled_back_at = time.time()
        pv.rollback_reason = reason
        pv.traffic_pct = 0.0

        # Find most recent non-rolled-back version to restore
        if self._production_version == version_id:
            candidates = [
                v for v in self._versions.values()
                if v.version_id != version_id
                and v.stage not in (RolloutStage.ROLLBACK, RolloutStage.TRAINING)
            ]
            candidates.sort(key=lambda v: v.promoted_at or 0, reverse=True)
            if candidates:
                restore = candidates[0]
                restore.stage = RolloutStage.PRODUCTION
                restore.traffic_pct = 1.0
                self._production_version = restore.version_id
            else:
                self._production_version = None

        return True

    def get_active_version(self) -> PolicyVersion | None:
        """Get the current production policy version."""
        if self._production_version:
            return self._versions.get(self._production_version)
        return None

    def get_version(self, version_id: str) -> PolicyVersion | None:
        return self._versions.get(version_id)

    def should_auto_rollback(self, version_id: str) -> bool:
        """Check if metrics warrant automatic rollback."""
        pv = self._versions.get(version_id)
        if not pv or pv.stage not in (RolloutStage.CANARY, RolloutStage.PRODUCTION):
            return False

        sla_miss = pv.metrics.get("sla_miss_rate", 0.0)
        error_rate = pv.metrics.get("error_rate", 0.0)

        return (
            sla_miss > self.config.auto_rollback_sla_miss_rate
            or error_rate > self.config.auto_rollback_error_rate
        )


# ═══════════════════════════════════════════════════════════════════
# 3. EXPLICIT PARTIAL-FAILURE SEMANTICS
# ═══════════════════════════════════════════════════════════════════

class FailureAction(str, enum.Enum):
    """What to do when a dependency fails."""
    RETRY = "retry"             # Re-dispatch to same or different provider
    ESCALATE = "escalate"       # Escalate to HITL
    DEGRADE = "degrade"         # Continue with degraded output
    ABORT = "abort"             # Abort the entire saga


@dataclass
class PartialFailurePolicy:
    """
    Per-need failure handling policy.

    Configured in domain YAML under each need's `on_failure:` section.
    This is the "sequential control meets real operations" contract.
    """
    need_type: str

    # Primary action
    action: FailureAction = FailureAction.RETRY

    # Retry config (when action=RETRY)
    max_retries: int = 2
    retry_different_provider: bool = True
    retry_backoff_seconds: float = 30.0

    # Degradation config (when action=DEGRADE)
    degraded_output_template: dict[str, Any] = field(default_factory=dict)
    degraded_quality_flag: str = "degraded"

    # Escalation config (when action=ESCALATE)
    escalation_queue: str = ""
    escalation_sla_seconds: float = 3600.0

    # Abort config (when action=ABORT)
    compensate_on_abort: bool = True

    # Conditional: different actions based on error class
    on_retryable: FailureAction = FailureAction.RETRY
    on_permanent: FailureAction = FailureAction.ESCALATE
    on_degraded: FailureAction = FailureAction.DEGRADE


@dataclass
class PartialFailureDecision:
    """The resolved decision for a specific failure event."""
    work_order_id: str
    need_type: str
    error_class: str           # retryable | permanent | degraded
    action: FailureAction
    reason: str
    retry_count: int = 0       # how many retries already attempted
    degraded_output: dict[str, Any] | None = None
    escalation_details: dict[str, Any] | None = None


class PartialFailureHandler:
    """
    Resolves failure handling based on per-need policies.

    When a work order fails, this handler determines the correct
    action based on the need type's failure policy and the error class.

    Integration: Called by Coordinator when a work order result
    arrives with status=FAILED.
    """

    def __init__(self):
        # need_type → PartialFailurePolicy
        self._policies: dict[str, PartialFailurePolicy] = {}
        # (need_type, case_id) → retry count
        self._retry_counts: dict[tuple[str, str], int] = {}

    def register_policy(self, policy: PartialFailurePolicy) -> None:
        self._policies[policy.need_type] = policy

    def resolve(
        self,
        work_order_id: str,
        need_type: str,
        case_id: str,
        error_class: str,
    ) -> PartialFailureDecision:
        """
        Determine what to do when a dependency fails.

        Error class drives the initial action selection:
        - retryable → policy.on_retryable (default: RETRY)
        - permanent → policy.on_permanent (default: ESCALATE)
        - degraded  → policy.on_degraded  (default: DEGRADE)

        Retry count is then checked: if max retries exceeded,
        RETRY escalates to ESCALATE.
        """
        policy = self._policies.get(need_type, PartialFailurePolicy(need_type=need_type))
        key = (need_type, case_id)
        retries = self._retry_counts.get(key, 0)

        # Select action based on error class
        if error_class == "retryable":
            action = policy.on_retryable
        elif error_class == "permanent":
            action = policy.on_permanent
        elif error_class == "degraded":
            action = policy.on_degraded
        else:
            action = policy.action

        reason = f"Error class: {error_class}"

        # Check retry budget
        if action == FailureAction.RETRY:
            if retries >= policy.max_retries:
                action = FailureAction.ESCALATE
                reason = f"Retry budget exhausted ({retries}/{policy.max_retries})"
            else:
                self._retry_counts[key] = retries + 1
                reason = f"Retry {retries + 1}/{policy.max_retries}"
                if policy.retry_different_provider:
                    reason += " (different provider)"

        # Build decision
        decision = PartialFailureDecision(
            work_order_id=work_order_id,
            need_type=need_type,
            error_class=error_class,
            action=action,
            reason=reason,
            retry_count=retries,
        )

        if action == FailureAction.DEGRADE:
            decision.degraded_output = dict(policy.degraded_output_template)
            decision.degraded_output["_quality_flag"] = policy.degraded_quality_flag

        if action == FailureAction.ESCALATE:
            decision.escalation_details = {
                "queue": policy.escalation_queue,
                "sla_seconds": policy.escalation_sla_seconds,
                "error_class": error_class,
                "retries_attempted": retries,
            }

        return decision

    def cleanup(self, case_id: str) -> None:
        """Remove retry tracking for a case."""
        keys = [k for k in self._retry_counts if k[1] == case_id]
        for k in keys:
            del self._retry_counts[k]


# ═══════════════════════════════════════════════════════════════════
# 4. RESERVATION PROTOCOL SPECIFICATION
# ═══════════════════════════════════════════════════════════════════

class ReservationOp(str, enum.Enum):
    """Reservation protocol operations."""
    ACQUIRE = "acquire"
    COMMIT = "commit"
    RELEASE = "release"
    EXPIRE = "expire"
    RECLAIM = "reclaim"          # forced reclaim by revocation manager
    CRASH_RECOVER = "crash_recover"  # recovery after process crash


@dataclass
class ReservationEvent:
    """
    Immutable event record for the reservation protocol.

    Every reservation state change produces an event. Events are
    append-only and form the authoritative record of capacity state.
    The in-memory CapacityReservation objects in ddd.py are projections
    of this event stream.
    """
    event_id: str = ""
    reservation_id: str = ""
    resource_id: str = ""
    work_order_id: str = ""
    operation: ReservationOp = ReservationOp.ACQUIRE
    amount: float = 0.0
    ttl_seconds: float = 30.0
    timestamp: float = 0.0
    # For CRASH_RECOVER: what happened to the orphan
    recovery_action: str = ""  # released | recommitted | orphaned


@dataclass
class ReservationProtocolSpec:
    """
    Formal specification of reservation behavior.

    This is a CONTRACT — the ddd.py implementation must conform to it.
    The spec is the authority; the code is the implementation.
    """
    # TTL defaults
    default_ttl_seconds: float = 30.0
    max_ttl_seconds: float = 300.0

    # Crash recovery
    crash_recovery_enabled: bool = True
    orphan_scan_interval_seconds: float = 60.0
    orphan_max_age_seconds: float = 120.0  # orphans older than this are released

    # Reclaim (from revocation manager)
    reclaim_grace_seconds: float = 15.0    # time between signal and hard reclaim

    # Idempotency
    commit_idempotent: bool = True   # re-commit of committed reservation is no-op
    release_idempotent: bool = True  # re-release is no-op
    acquire_idempotent: bool = False  # re-acquire of same (resource, wo) creates new reservation

    # Atomicity guarantees
    acquire_atomic: bool = True   # check + deduct in single operation
    release_restores_capacity: bool = True  # released amount returns to pool

    # Ordering
    fifo_within_priority: bool = True  # same-priority reservations served FIFO


class ReservationEventLog:
    """
    Append-only log of reservation events.

    In production, this backs into the action_ledger table.
    Here it's in-memory for the protocol specification tests.
    """

    def __init__(self):
        self._events: list[ReservationEvent] = []
        self._counter = 0

    def record(
        self,
        reservation_id: str,
        resource_id: str,
        work_order_id: str,
        operation: ReservationOp,
        amount: float = 0.0,
        ttl_seconds: float = 30.0,
        recovery_action: str = "",
    ) -> ReservationEvent:
        self._counter += 1
        event = ReservationEvent(
            event_id=f"rev_{self._counter}",
            reservation_id=reservation_id,
            resource_id=resource_id,
            work_order_id=work_order_id,
            operation=operation,
            amount=amount,
            ttl_seconds=ttl_seconds,
            timestamp=time.time(),
            recovery_action=recovery_action,
        )
        self._events.append(event)
        return event

    def get_events(
        self,
        reservation_id: str | None = None,
        resource_id: str | None = None,
        operation: ReservationOp | None = None,
    ) -> list[ReservationEvent]:
        results = self._events
        if reservation_id:
            results = [e for e in results if e.reservation_id == reservation_id]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        if operation:
            results = [e for e in results if e.operation == operation]
        return results

    def orphan_scan(self, max_age_seconds: float = 120.0) -> list[ReservationEvent]:
        """
        Find ACQUIRE events with no matching COMMIT, RELEASE, or EXPIRE.
        These are potential crash orphans.
        """
        now = time.time()
        acquired = {}
        resolved = set()

        for e in self._events:
            if e.operation == ReservationOp.ACQUIRE:
                acquired[e.reservation_id] = e
            elif e.operation in (
                ReservationOp.COMMIT,
                ReservationOp.RELEASE,
                ReservationOp.EXPIRE,
                ReservationOp.RECLAIM,
                ReservationOp.CRASH_RECOVER,
            ):
                resolved.add(e.reservation_id)

        orphans = []
        for rid, event in acquired.items():
            if rid not in resolved and now - event.timestamp > max_age_seconds:
                orphans.append(event)

        return orphans


# ═══════════════════════════════════════════════════════════════════
# 5. LEARNING SCOPE CONSTRAINTS (HARD-CODED GUARDRAILS)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LearningConstraint:
    """
    A single hard-coded guardrail on what the adaptive layer can modify.

    These are INVARIANTS, not configuration. They cannot be changed by
    YAML, by learned models, or by any runtime mechanism. They are
    compiled into the system and verified at every dispatch cycle.
    """
    name: str
    description: str
    check_fn: Callable[..., bool] | None = None
    # For declarative constraints:
    target: str = ""          # what the constraint governs
    bound_type: str = ""      # "max_delta" | "range" | "never_modify" | "never_bypass"
    bound_value: Any = None


class LearningScopeEnforcer:
    """
    Hard-coded guardrails on what the adaptive layer can touch.

    INVARIANT: Learning may ONLY modify objective weights and scoring
    features, within bounded envelopes. Learning may NEVER:
    - Modify eligibility constraints
    - Modify capacity values
    - Bypass governance gates
    - Bypass circuit breakers
    - Override SLA commitments
    - Modify reservation TTLs

    These constraints are checked at dispatch time, AFTER the adaptive
    layer proposes adjustments and BEFORE the solver runs. If any
    constraint is violated, the adjustment is rejected and the dispatch
    proceeds with the unadjusted policy.

    This is the "Constitution" — the adaptive layer operates within it.
    """

    def __init__(self):
        self._constraints: list[LearningConstraint] = self._build_core_constraints()

    def _build_core_constraints(self) -> list[LearningConstraint]:
        """The non-negotiable guardrails. Hard-coded, not configurable."""
        return [
            LearningConstraint(
                name="eligibility_immutable",
                description="Learning CANNOT modify eligibility constraints. "
                            "Eligibility is boolean and domain-defined.",
                target="eligibility_constraints",
                bound_type="never_modify",
            ),
            LearningConstraint(
                name="capacity_immutable",
                description="Learning CANNOT modify resource capacity values. "
                            "Capacity is a physical property of the resource.",
                target="capacity",
                bound_type="never_modify",
            ),
            LearningConstraint(
                name="governance_gate_immutable",
                description="Learning CANNOT bypass governance gates (HITL). "
                            "Gates are regulatory requirements.",
                target="governance_gates",
                bound_type="never_bypass",
            ),
            LearningConstraint(
                name="circuit_breaker_immutable",
                description="Learning CANNOT override circuit breakers. "
                            "Breakers are safety mechanisms.",
                target="circuit_breaker",
                bound_type="never_bypass",
            ),
            LearningConstraint(
                name="sla_immutable",
                description="Learning CANNOT override SLA commitments. "
                            "SLAs are contractual obligations.",
                target="sla_seconds",
                bound_type="never_modify",
            ),
            LearningConstraint(
                name="reservation_ttl_immutable",
                description="Learning CANNOT modify reservation TTLs. "
                            "TTLs are protocol invariants.",
                target="reservation_ttl",
                bound_type="never_modify",
            ),
            LearningConstraint(
                name="cost_weight_bounded",
                description="Learning may adjust cost weight by at most ±30%.",
                target="objective.minimize_cost",
                bound_type="max_delta",
                bound_value=0.30,
            ),
            LearningConstraint(
                name="wait_weight_bounded",
                description="Learning may adjust wait weight by at most ±25%.",
                target="objective.minimize_wait_time",
                bound_type="max_delta",
                bound_value=0.25,
            ),
            LearningConstraint(
                name="sla_weight_bounded",
                description="Learning may adjust SLA risk weight by at most ±20%.",
                target="objective.minimize_sla_risk",
                bound_type="max_delta",
                bound_value=0.20,
            ),
        ]

    def validate_adjustments(
        self,
        base_weights: dict[str, float],
        proposed_weights: dict[str, float],
        proposed_modifications: dict[str, Any] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        Validate proposed adaptive adjustments against constraints.

        Returns (valid, violations) where violations is empty if valid.
        """
        violations = []
        modifications = proposed_modifications or {}

        for constraint in self._constraints:
            if constraint.bound_type == "never_modify":
                if constraint.target in modifications:
                    violations.append(
                        f"VIOLATION: {constraint.name} — "
                        f"attempted to modify {constraint.target}. "
                        f"{constraint.description}"
                    )

            elif constraint.bound_type == "never_bypass":
                if modifications.get(f"bypass_{constraint.target}"):
                    violations.append(
                        f"VIOLATION: {constraint.name} — "
                        f"attempted to bypass {constraint.target}. "
                        f"{constraint.description}"
                    )

            elif constraint.bound_type == "max_delta":
                # Extract the objective name from target
                parts = constraint.target.split(".")
                if len(parts) == 2 and parts[0] == "objective":
                    key = parts[1]
                    base_val = base_weights.get(key, 0.0)
                    proposed_val = proposed_weights.get(key, base_val)
                    if base_val > 0:
                        delta = abs(proposed_val - base_val) / base_val
                        if delta > constraint.bound_value:
                            violations.append(
                                f"VIOLATION: {constraint.name} — "
                                f"{key} adjusted by {delta:.1%} "
                                f"(max: ±{constraint.bound_value:.0%}). "
                                f"{constraint.description}"
                            )

        return len(violations) == 0, violations

    def enforce(
        self,
        base_weights: dict[str, float],
        proposed_weights: dict[str, float],
        proposed_modifications: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Enforce constraints. Returns safe weights.

        If valid: returns proposed_weights unchanged.
        If invalid: returns base_weights (rejects all adjustments).
        """
        valid, violations = self.validate_adjustments(
            base_weights, proposed_weights, proposed_modifications,
        )
        if valid:
            return dict(proposed_weights)
        else:
            return dict(base_weights)

    @property
    def constraints(self) -> list[LearningConstraint]:
        """Read-only access to the constraint set."""
        return list(self._constraints)
