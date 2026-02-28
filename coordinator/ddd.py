"""
Cognitive Core — DDD State Machines

Implements the formal state machines from the DDD Unified Specification v1.1:

  Section 3:  Work Order Lifecycle (8-state machine with failure semantics)
  Section 6:  Capacity Models (Slot, Volume, Batch)
  Section 8:  Capacity Reservation Protocol (reserve/commit/release with TTL)
  Section 9:  Eligibility vs. Ranking separation
  Section 17: Production Hardening (Circuit Breaker, Batch Reaper)

These are the *state machines* — pure logic, no I/O. The coordinator
runtime wires them to persistence (store.py) and execution (runtime.py).
"""

from __future__ import annotations

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — WORK ORDER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

class WOStatus(str, enum.Enum):
    """Work order states per Spec v1.1, Section 3."""
    CREATED     = "created"
    DISPATCHED  = "dispatched"
    EXPIRED     = "expired"
    CLAIMED     = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    CANCELED    = "canceled"


class ErrorClass(str, enum.Enum):
    """Failure classification per Spec v1.1, Section 3."""
    RETRYABLE  = "retryable"
    PERMANENT  = "permanent"
    DEGRADED   = "degraded"


class ResumePolicy(str, enum.Enum):
    """Parent workflow resume behavior on partial failure."""
    ALL_OR_ABORT = "all_or_abort"
    BEST_EFFORT  = "best_effort"
    QUORUM       = "quorum"


# Allowed transitions: from_state → set of valid to_states
_WO_TRANSITIONS: dict[WOStatus, set[WOStatus]] = {
    WOStatus.CREATED:     {WOStatus.DISPATCHED},
    WOStatus.DISPATCHED:  {WOStatus.CLAIMED, WOStatus.EXPIRED, WOStatus.CANCELED},
    WOStatus.CLAIMED:     {WOStatus.IN_PROGRESS, WOStatus.CANCELED},
    WOStatus.IN_PROGRESS: {WOStatus.COMPLETED, WOStatus.FAILED, WOStatus.CANCELED},
}

_TERMINAL_STATES = {WOStatus.COMPLETED, WOStatus.FAILED, WOStatus.CANCELED, WOStatus.EXPIRED}


@dataclass
class WorkOrderError:
    """Typed error payload per Spec v1.1, Section 3."""
    error_code: str
    error_class: ErrorClass
    message: str
    resource_reported: bool = True
    retry_hint_seconds: float | None = None


@dataclass
class RetryPolicy:
    """Retry configuration per capability contract."""
    max_attempts: int = 3
    backoff: str = "exponential"   # constant | linear | exponential
    base_delay_seconds: float = 30.0
    max_delay_seconds: float = 600.0
    retryable_errors: list[str] = field(default_factory=lambda: [
        "resource_timeout", "transient_api_failure", "stale_data",
    ])
    non_retryable_errors: list[str] = field(default_factory=lambda: [
        "schema_violation", "authorization_denied", "data_not_found",
    ])

    def compute_delay(self, attempt: int) -> float:
        """Compute backoff delay for the given attempt number (1-indexed)."""
        if self.backoff == "constant":
            return self.base_delay_seconds
        elif self.backoff == "linear":
            return min(self.base_delay_seconds * attempt, self.max_delay_seconds)
        else:  # exponential
            return min(self.base_delay_seconds * (2 ** (attempt - 1)), self.max_delay_seconds)

    def is_retryable(self, error: WorkOrderError) -> bool:
        """Determine if an error should trigger retry."""
        if error.error_code in self.non_retryable_errors:
            return False
        if error.error_class == ErrorClass.PERMANENT:
            return False
        if error.error_class == ErrorClass.RETRYABLE:
            return True
        if error.error_code in self.retryable_errors:
            return True
        return False


@dataclass
class DDDWorkOrder:
    """
    Full work order per Spec v1.1, Section 3.

    Extends the existing coordinator WorkOrder with the formal state
    machine, typed errors, retry tracking, and reservation binding.
    """
    work_order_id: str
    request_id: str              # idempotency key from ResourceRequest
    case_id: str
    trace_id: str
    requester_instance_id: str
    correlation_id: str

    # Contract binding
    contract_name: str
    contract_version: str        # semver
    inputs: dict[str, Any]

    # Constraints
    priority: str = "routine"    # critical | high | routine
    sla_seconds: float | None = None
    depends_on: list[str] = field(default_factory=list)

    # Assignment
    resource_id: str = ""
    reservation_id: str = ""

    # Lifecycle
    status: WOStatus = WOStatus.CREATED
    created_at: float = 0.0
    dispatched_at: float | None = None
    claimed_at: float | None = None
    completed_at: float | None = None
    claim_ttl_seconds: float = 60.0

    # Result / Error
    result: dict[str, Any] | None = None
    error: WorkOrderError | None = None
    quality_flag: str = "normal"   # normal | degraded

    # Retry tracking
    attempt: int = 1
    max_attempts: int = 3

    @staticmethod
    def create(
        requester_instance_id: str,
        correlation_id: str,
        contract_name: str,
        contract_version: str = "1.0.0",
        inputs: dict[str, Any] | None = None,
        priority: str = "routine",
        sla_seconds: float | None = None,
        case_id: str = "",
        depends_on: list[str] | None = None,
    ) -> DDDWorkOrder:
        rid = uuid.uuid4().hex[:16]
        return DDDWorkOrder(
            work_order_id=f"wo_{uuid.uuid4().hex[:12]}",
            request_id=rid,
            case_id=case_id,
            trace_id=f"tr_{uuid.uuid4().hex[:12]}",
            requester_instance_id=requester_instance_id,
            correlation_id=correlation_id,
            contract_name=contract_name,
            contract_version=contract_version,
            inputs=inputs or {},
            priority=priority,
            sla_seconds=sla_seconds,
            depends_on=depends_on or [],
            created_at=time.time(),
        )

    def transition(self, to: WOStatus, now: float | None = None) -> None:
        """
        Enforce the work order state machine.
        Raises InvalidTransition if the transition is not allowed.
        """
        now = now or time.time()
        allowed = _WO_TRANSITIONS.get(self.status, set())
        if to not in allowed:
            raise InvalidTransition(
                f"Work order {self.work_order_id}: "
                f"{self.status.value} → {to.value} is not allowed. "
                f"Valid transitions: {sorted(s.value for s in allowed)}"
            )
        self.status = to
        if to == WOStatus.DISPATCHED:
            self.dispatched_at = now
        elif to == WOStatus.CLAIMED:
            self.claimed_at = now
        elif to in _TERMINAL_STATES:
            self.completed_at = now

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATES

    @property
    def is_claim_expired(self) -> bool:
        """Check if dispatch has exceeded claim TTL without being claimed."""
        if self.status != WOStatus.DISPATCHED or not self.dispatched_at:
            return False
        return time.time() - self.dispatched_at > self.claim_ttl_seconds


class InvalidTransition(Exception):
    """Raised when a state machine transition is not allowed."""
    pass


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — CAPACITY MODELS
# ═══════════════════════════════════════════════════════════════════

class CapacityModel(str, enum.Enum):
    SLOT   = "slot"
    VOLUME = "volume"
    BATCH  = "batch"


class BatchStatus(str, enum.Enum):
    COLLECTING = "collecting"
    EXECUTING  = "executing"
    COMPLETED  = "completed"
    REAPED     = "reaped"        # Section 17.1


class ReaperAction(str, enum.Enum):
    FAIL          = "fail"
    RETRY_ONCE    = "retry_once"
    REDISTRIBUTE  = "redistribute"


@dataclass
class CapacityState:
    """
    Unified capacity state for all three models.
    The model field determines which fields are active.
    """
    model: CapacityModel

    # Slot model
    max_concurrent: int = 0
    current_load: int = 0

    # Volume model
    max_volume: float = 0.0
    current_volume: float = 0.0
    unit: str = "items"

    # Batch model
    batch_threshold: int = 0
    batch_timeout_seconds: float = 0.0
    batch_items: int = 0
    batch_status: BatchStatus = BatchStatus.COLLECTING
    batch_execution_start: float | None = None
    batch_collecting_since: float | None = None  # when first item arrived
    max_execution_duration_seconds: float = 14400.0  # 4h default
    reaper_action: ReaperAction = ReaperAction.FAIL
    reaper_retry_count: int = 0

    def can_accept(self, amount: float = 1.0) -> bool:
        """Optimizer question: can this resource accept work?"""
        if self.model == CapacityModel.SLOT:
            return self.current_load < self.max_concurrent
        elif self.model == CapacityModel.VOLUME:
            return self.current_volume + amount <= self.max_volume
        elif self.model == CapacityModel.BATCH:
            return self.batch_status == BatchStatus.COLLECTING
        return False

    def on_assign(self, amount: float = 1.0) -> None:
        """Update state when work is assigned."""
        if self.model == CapacityModel.SLOT:
            self.current_load += 1
        elif self.model == CapacityModel.VOLUME:
            self.current_volume += amount
        elif self.model == CapacityModel.BATCH:
            self.batch_items += 1
            if self.batch_items == 1 and self.batch_collecting_since is None:
                self.batch_collecting_since = time.time()

    def on_release(self, amount: float = 1.0) -> None:
        """Update state when work completes."""
        if self.model == CapacityModel.SLOT:
            self.current_load = max(0, self.current_load - 1)
        elif self.model == CapacityModel.VOLUME:
            self.current_volume = max(0.0, self.current_volume - amount)
        elif self.model == CapacityModel.BATCH:
            pass  # batch releases all at once on completion

    def check_batch_trigger(self, now: float | None = None) -> bool:
        """
        Check if batch should transition to EXECUTING.

        Two triggers (OR logic):
        1. Item threshold: batch_items >= batch_threshold
        2. Time window: time since first item >= batch_timeout_seconds
           (only if batch_timeout_seconds > 0 and at least 1 item collected)
        """
        if self.model != CapacityModel.BATCH:
            return False
        if self.batch_status != BatchStatus.COLLECTING:
            return False
        if self.batch_items <= 0:
            return False
        # Threshold trigger
        if self.batch_items >= self.batch_threshold:
            return True
        # Time window trigger
        if (self.batch_timeout_seconds > 0
                and self.batch_collecting_since is not None):
            now = now or time.time()
            elapsed = now - self.batch_collecting_since
            if elapsed >= self.batch_timeout_seconds:
                return True
        return False

    def trigger_batch(self, now: float | None = None) -> None:
        """Transition batch to EXECUTING."""
        now = now or time.time()
        self.batch_status = BatchStatus.EXECUTING
        self.batch_execution_start = now

    def complete_batch(self) -> None:
        """Transition batch to COMPLETED, reset for next cycle."""
        self.batch_status = BatchStatus.COLLECTING
        self.batch_items = 0
        self.batch_execution_start = None
        self.batch_collecting_since = None

    def check_reaper(self, now: float | None = None) -> bool:
        """Section 17.1: Check if batch execution has timed out."""
        if self.model != CapacityModel.BATCH:
            return False
        if self.batch_status != BatchStatus.EXECUTING:
            return False
        if not self.batch_execution_start:
            return False
        now = now or time.time()
        return now - self.batch_execution_start > self.max_execution_duration_seconds

    @property
    def utilization_pct(self) -> float:
        if self.model == CapacityModel.SLOT:
            return self.current_load / max(1, self.max_concurrent)
        elif self.model == CapacityModel.VOLUME:
            return self.current_volume / max(1.0, self.max_volume)
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — CAPACITY RESERVATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════

class ReservationStatus(str, enum.Enum):
    HELD      = "held"
    COMMITTED = "committed"
    RELEASED  = "released"
    EXPIRED   = "expired"


@dataclass
class CapacityReservation:
    """
    Short-lived capacity claim per Spec v1.1, Section 8.
    
    Lifecycle: HELD → COMMITTED (dispatch succeeds)
               HELD → RELEASED (dispatch fails or cancelled)
               HELD → EXPIRED  (TTL exceeded)
    """
    reservation_id: str
    resource_id: str
    work_order_id: str
    amount: float
    status: ReservationStatus = ReservationStatus.HELD
    created_at: float = 0.0
    ttl_seconds: float = 30.0
    committed_at: float | None = None
    released_at: float | None = None

    @staticmethod
    def create(
        resource_id: str,
        work_order_id: str,
        amount: float = 1.0,
        ttl_seconds: float = 30.0,
    ) -> CapacityReservation:
        return CapacityReservation(
            reservation_id=f"rsv_{uuid.uuid4().hex[:12]}",
            resource_id=resource_id,
            work_order_id=work_order_id,
            amount=amount,
            ttl_seconds=ttl_seconds,
            created_at=time.time(),
        )

    def commit(self, now: float | None = None) -> None:
        """Mark reservation permanent. Idempotent."""
        if self.status == ReservationStatus.COMMITTED:
            return  # idempotent
        if self.status != ReservationStatus.HELD:
            raise InvalidTransition(
                f"Reservation {self.reservation_id}: "
                f"cannot commit from {self.status.value}"
            )
        self.status = ReservationStatus.COMMITTED
        self.committed_at = now or time.time()

    def release(self, now: float | None = None) -> float:
        """Release capacity. Idempotent. Returns amount released."""
        if self.status in (ReservationStatus.RELEASED, ReservationStatus.EXPIRED):
            return 0.0  # idempotent
        self.status = ReservationStatus.RELEASED
        self.released_at = now or time.time()
        return self.amount

    @property
    def is_expired(self) -> bool:
        if self.status != ReservationStatus.HELD:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def expire(self, now: float | None = None) -> float:
        """Expire a held reservation. Returns amount released."""
        if self.status != ReservationStatus.HELD:
            return 0.0
        self.status = ReservationStatus.EXPIRED
        self.released_at = now or time.time()
        return self.amount


# ═══════════════════════════════════════════════════════════════════
# SECTION 5/9 — RESOURCE REGISTRY WITH ELIGIBILITY
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EligibilityConstraint:
    """Boolean predicate for hard resource filtering (Section 9)."""
    name: str                   # "licensing", "conflict_of_interest", etc.
    predicate: str              # expression evaluated against resource + work order
    audit_reason_template: str  # human-readable exclusion reason


@dataclass
class CircuitBreakerState:
    """Per-resource circuit breaker state (Section 17.3)."""
    status: str = "closed"      # closed | open | half_open
    window: list[bool] = field(default_factory=list)  # True=success, False=failure
    window_size: int = 20
    open_threshold: float = 0.50
    cooldown_seconds: float = 900.0   # 15 min
    backoff_multiplier: float = 2.0
    max_cooldown_seconds: float = 14400.0  # 4h
    opened_at: float | None = None
    current_cooldown: float | None = None  # defaults to cooldown_seconds

    def __post_init__(self):
        if self.current_cooldown is None:
            self.current_cooldown = self.cooldown_seconds

    def record_outcome(self, success: bool, now: float | None = None) -> str | None:
        """
        Record a work order outcome. Returns the new status if it changed.
        """
        now = now or time.time()
        if self.status == "open":
            return None  # not receiving work, no outcomes to record

        self.window.append(success)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        if self.status == "half_open":
            if success:
                self.status = "closed"
                self.current_cooldown = self.cooldown_seconds  # reset backoff
                return "closed"
            else:
                self.status = "open"
                self.opened_at = now
                self.current_cooldown = min(
                    self.current_cooldown * self.backoff_multiplier,
                    self.max_cooldown_seconds,
                )
                return "open"

        # closed: check failure rate
        if len(self.window) >= self.window_size:
            failures = sum(1 for x in self.window if not x)
            failure_rate = failures / len(self.window)
            if failure_rate >= self.open_threshold:
                self.status = "open"
                self.opened_at = now
                return "open"

        return None

    def check_cooldown(self, now: float | None = None) -> bool:
        """Check if cooldown has elapsed and transition to half_open."""
        now = now or time.time()
        if self.status != "open" or not self.opened_at:
            return False
        if now - self.opened_at >= self.current_cooldown:
            self.status = "half_open"
            return True
        return False

    @property
    def is_eligible(self) -> bool:
        return self.status != "open"


@dataclass
class ResourceRegistration:
    """
    Full resource registration per Spec v1.1, Sections 5/6/9.
    """
    resource_id: str
    resource_type: str           # human | system | composite

    # Capabilities: list of (workflow, domain) pairs
    capabilities: list[tuple[str, str]] = field(default_factory=list)

    # Capacity
    capacity: CapacityState = field(default_factory=lambda: CapacityState(model=CapacityModel.SLOT))

    # Attributes (used for ranking, not eligibility)
    attributes: dict[str, Any] = field(default_factory=dict)

    # Heartbeat
    heartbeat_interval_seconds: float = 300.0
    stale_after_seconds: float = 900.0
    last_heartbeat: float = 0.0

    # Circuit breaker
    circuit_breaker: CircuitBreakerState = field(default_factory=CircuitBreakerState)

    # Eligibility constraints (populated from domain config)
    eligibility_constraints: list[EligibilityConstraint] = field(default_factory=list)

    # Exploration tracking (Section 17.4)
    completed_work_orders: int = 0

    # Timestamps
    registered_at: float = 0.0
    updated_at: float = 0.0

    @staticmethod
    def create(
        resource_id: str,
        resource_type: str,
        capabilities: list[tuple[str, str]],
        capacity_model: CapacityModel = CapacityModel.SLOT,
        max_capacity: int = 10,
        **attributes: Any,
    ) -> ResourceRegistration:
        now = time.time()
        cap = CapacityState(model=capacity_model, max_concurrent=max_capacity)
        if capacity_model == CapacityModel.VOLUME:
            cap.max_volume = float(max_capacity)
        elif capacity_model == CapacityModel.BATCH:
            cap.batch_threshold = max_capacity
        return ResourceRegistration(
            resource_id=resource_id,
            resource_type=resource_type,
            capabilities=capabilities,
            capacity=cap,
            attributes=attributes,
            registered_at=now,
            updated_at=now,
            last_heartbeat=now,
        )

    @property
    def is_healthy(self) -> bool:
        """Check heartbeat liveness (Invariant 6) and circuit breaker."""
        if self.capacity.model == CapacityModel.VOLUME:
            if time.time() - self.last_heartbeat > self.stale_after_seconds:
                return False
        return self.circuit_breaker.is_eligible


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — DISPATCH OPTIMIZER (Eligibility + Ranking)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EligibilityResult:
    """Audit record for an eligibility evaluation."""
    resource_id: str
    eligible: bool
    failed_constraint: str | None = None
    audit_reason: str | None = None


@dataclass
class RankingScore:
    """Per-resource score breakdown for audit trail."""
    resource_id: str
    total_score: float
    feature_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class DispatchDecision:
    """Complete dispatch decision with full audit trail."""
    work_order_id: str
    selected_resource_id: str | None
    reservation_id: str | None = None
    tier: str = "optimal"        # optimal | fallback | exploration
    eligibility_results: list[EligibilityResult] = field(default_factory=list)
    ranking_scores: list[RankingScore] = field(default_factory=list)
    timestamp: float = 0.0


class ResourceRegistry:
    """
    In-memory resource registry with eligibility filtering,
    capacity reservation, and circuit breaker management.
    """

    def __init__(self):
        self._resources: dict[str, ResourceRegistration] = {}
        self._reservations: dict[str, CapacityReservation] = {}

    # ─── Registration ────────────────────────────────────────────

    def register(self, resource: ResourceRegistration) -> None:
        self._resources[resource.resource_id] = resource

    def unregister(self, resource_id: str) -> None:
        self._resources.pop(resource_id, None)

    def get(self, resource_id: str) -> ResourceRegistration | None:
        return self._resources.get(resource_id)

    def heartbeat(self, resource_id: str, current_volume: float | None = None) -> bool:
        """Update heartbeat. Optionally reconcile volume state."""
        res = self._resources.get(resource_id)
        if not res:
            return False
        res.last_heartbeat = time.time()
        res.updated_at = res.last_heartbeat
        if current_volume is not None and res.capacity.model == CapacityModel.VOLUME:
            res.capacity.current_volume = current_volume
        return True

    # ─── Eligibility Filter (Section 9, Phase 1) ─────────────────

    def filter_eligible(
        self,
        workflow: str,
        domain: str,
        work_order: DDDWorkOrder | None = None,
    ) -> tuple[list[ResourceRegistration], list[EligibilityResult]]:
        """
        Return eligible resources and full audit trail of exclusions.
        """
        eligible = []
        audit = []

        for res in self._resources.values():
            # Constraint 1: capability match
            if (workflow, domain) not in res.capabilities:
                audit.append(EligibilityResult(
                    resource_id=res.resource_id,
                    eligible=False,
                    failed_constraint="capability_match",
                    audit_reason=f"Not registered for ({workflow}, {domain})",
                ))
                continue

            # Constraint 2: circuit breaker
            if not res.circuit_breaker.is_eligible:
                audit.append(EligibilityResult(
                    resource_id=res.resource_id,
                    eligible=False,
                    failed_constraint="circuit_breaker",
                    audit_reason=f"Circuit breaker OPEN since {res.circuit_breaker.opened_at}",
                ))
                continue

            # Constraint 3: heartbeat liveness (volume model only)
            if not res.is_healthy:
                audit.append(EligibilityResult(
                    resource_id=res.resource_id,
                    eligible=False,
                    failed_constraint="heartbeat_liveness",
                    audit_reason=f"Stale heartbeat: last seen {res.last_heartbeat}",
                ))
                continue

            # Constraint 4: domain-specific eligibility constraints
            excluded = False
            for constraint in res.eligibility_constraints:
                if not self._evaluate_constraint(constraint, res, work_order):
                    audit.append(EligibilityResult(
                        resource_id=res.resource_id,
                        eligible=False,
                        failed_constraint=constraint.name,
                        audit_reason=constraint.audit_reason_template,
                    ))
                    excluded = True
                    break

            if excluded:
                continue

            # Passed all eligibility checks
            audit.append(EligibilityResult(
                resource_id=res.resource_id,
                eligible=True,
            ))
            eligible.append(res)

        return eligible, audit

    def _evaluate_constraint(
        self,
        constraint: EligibilityConstraint,
        resource: ResourceRegistration,
        work_order: DDDWorkOrder | None,
    ) -> bool:
        """
        Evaluate a single eligibility constraint.
        Returns True if the resource passes (is eligible).

        In production, this evaluates expressions against resource
        attributes and work order context. For now, supports
        simple attribute-based checks.
        """
        # Simple built-in checks
        if constraint.name == "licensing":
            required = (work_order.inputs.get("required_certifications", [])
                       if work_order else [])
            held = resource.attributes.get("certifications", [])
            return all(r in held for r in required)

        if constraint.name == "geographic_limit":
            limit = resource.attributes.get("geographic_limit_miles")
            distance = resource.attributes.get("distance_to_work")
            if limit and distance:
                return distance <= limit

        # Default: pass (unknown constraints don't block)
        return True

    # ─── Capacity Reservation (Section 8) ─────────────────────────

    def reserve(
        self,
        resource_id: str,
        work_order_id: str,
        amount: float = 1.0,
        ttl_seconds: float = 30.0,
    ) -> CapacityReservation | None:
        """
        Attempt to reserve capacity. Returns reservation or None if denied.
        Atomic: checks capacity and deducts in one step.
        """
        res = self._resources.get(resource_id)
        if not res:
            return None
        if not res.capacity.can_accept(amount):
            return None

        # Deduct capacity and create reservation
        res.capacity.on_assign(amount)
        reservation = CapacityReservation.create(
            resource_id=resource_id,
            work_order_id=work_order_id,
            amount=amount,
            ttl_seconds=ttl_seconds,
        )
        self._reservations[reservation.reservation_id] = reservation
        return reservation

    def commit_reservation(self, reservation_id: str) -> bool:
        """Commit a held reservation. Idempotent."""
        rsv = self._reservations.get(reservation_id)
        if not rsv:
            return False
        rsv.commit()
        return True

    def release_reservation(self, reservation_id: str) -> bool:
        """Release a reservation and return capacity. Idempotent."""
        rsv = self._reservations.get(reservation_id)
        if not rsv:
            return False
        amount = rsv.release()
        if amount > 0:
            res = self._resources.get(rsv.resource_id)
            if res:
                res.capacity.on_release(amount)
        return True

    def sweep_expired_reservations(self, now: float | None = None) -> list[str]:
        """
        TTL sweep: expire held reservations and return capacity.
        Returns list of expired reservation IDs. Idempotent.
        """
        now = now or time.time()
        expired = []
        for rsv in list(self._reservations.values()):
            if rsv.status == ReservationStatus.HELD:
                if now - rsv.created_at > rsv.ttl_seconds:
                    amount = rsv.expire(now)
                    if amount > 0:
                        res = self._resources.get(rsv.resource_id)
                        if res:
                            res.capacity.on_release(amount)
                    expired.append(rsv.reservation_id)
        return expired

    # ─── Batch Reaper (Section 17.1) ──────────────────────────────

    def sweep_stale_batches(self, now: float | None = None) -> list[str]:
        """
        Batch reaper sweep. Returns list of reaped resource IDs.
        Idempotent: running twice on the same stale batch is a no-op.
        """
        now = now or time.time()
        reaped = []
        for res in self._resources.values():
            if res.capacity.check_reaper(now):
                action = res.capacity.reaper_action
                if action == ReaperAction.FAIL:
                    res.capacity.batch_status = BatchStatus.REAPED
                    reaped.append(res.resource_id)
                elif action == ReaperAction.RETRY_ONCE:
                    if res.capacity.reaper_retry_count < 1:
                        res.capacity.reaper_retry_count += 1
                        res.capacity.batch_status = BatchStatus.COLLECTING
                        res.capacity.batch_execution_start = None
                    else:
                        res.capacity.batch_status = BatchStatus.REAPED
                        reaped.append(res.resource_id)
                elif action == ReaperAction.REDISTRIBUTE:
                    res.capacity.batch_status = BatchStatus.REAPED
                    reaped.append(res.resource_id)
        return reaped

    # ─── Circuit Breaker Management ───────────────────────────────

    def record_outcome(
        self,
        resource_id: str,
        success: bool,
    ) -> str | None:
        """
        Record a work order outcome for circuit breaker tracking.
        Returns new circuit breaker status if it changed, else None.
        """
        res = self._resources.get(resource_id)
        if not res:
            return None
        transition = res.circuit_breaker.record_outcome(success)
        if success:
            res.completed_work_orders += 1
        return transition

    def check_circuit_breaker_cooldowns(self, now: float | None = None) -> list[str]:
        """
        Sweep all open circuit breakers for cooldown expiry.
        Returns list of resource IDs transitioned to half_open.
        """
        transitioned = []
        for res in self._resources.values():
            if res.circuit_breaker.check_cooldown(now):
                transitioned.append(res.resource_id)
        return transitioned

    # ─── Exploration (Section 17.4) ───────────────────────────────

    def partition_by_maturity(
        self,
        candidates: list[ResourceRegistration],
        maturity_threshold: int = 10,
    ) -> tuple[list[ResourceRegistration], list[ResourceRegistration]]:
        """Split candidates into proven and unproven for exploration policy."""
        proven = [r for r in candidates if r.completed_work_orders >= maturity_threshold]
        unproven = [r for r in candidates if r.completed_work_orders < maturity_threshold]
        return proven, unproven

    # ─── Queries ──────────────────────────────────────────────────

    def list_resources(
        self,
        workflow: str | None = None,
        domain: str | None = None,
    ) -> list[ResourceRegistration]:
        results = []
        for res in self._resources.values():
            if workflow and domain:
                if (workflow, domain) not in res.capabilities:
                    continue
            results.append(res)
        return results

    @property
    def resource_count(self) -> int:
        return len(self._resources)

    @property
    def reservation_count(self) -> int:
        return sum(1 for r in self._reservations.values()
                   if r.status == ReservationStatus.HELD)
