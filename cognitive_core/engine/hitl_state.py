"""
Cognitive Core — HITL State Machine (H-011, H-012, H-013)

Explicit state machine for human-in-the-loop review lifecycle.
Every transition is validated and audited. SLA enforcement with
timeout escalation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("cognitive_core.hitl_state")


class HITLState(str, Enum):
    SUSPENDED = "suspended"
    PENDING_REVIEW = "pending_review"
    ASSIGNED = "assigned"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    RESUMED = "resumed"
    TERMINATED = "terminated"


class ReviewerAction(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REASSIGNED = "reassigned"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


class IllegalStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


# Valid transitions: {from_state: [valid_to_states]}
VALID_TRANSITIONS = {
    HITLState.SUSPENDED: [HITLState.PENDING_REVIEW],
    HITLState.PENDING_REVIEW: [HITLState.ASSIGNED, HITLState.TIMED_OUT],
    HITLState.ASSIGNED: [HITLState.UNDER_REVIEW, HITLState.PENDING_REVIEW, HITLState.TIMED_OUT],
    HITLState.UNDER_REVIEW: [HITLState.APPROVED, HITLState.REJECTED, HITLState.TIMED_OUT, HITLState.PENDING_REVIEW],
    HITLState.APPROVED: [HITLState.RESUMED],
    HITLState.REJECTED: [HITLState.TERMINATED],
    HITLState.TIMED_OUT: [HITLState.PENDING_REVIEW, HITLState.TERMINATED],
    HITLState.RESUMED: [],  # Terminal
    HITLState.TERMINATED: [],  # Terminal
}


@dataclass
class HITLTransitionRecord:
    """Immutable record of a state transition."""
    instance_id: str
    from_state: HITLState
    to_state: HITLState
    actor: str
    reason: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewSLA:
    """SLA timer for a review assignment."""
    instance_id: str
    assigned_to: str
    assigned_at: float
    sla_seconds: float
    escalation_target: str = ""
    queue: str = ""

    @property
    def deadline(self) -> float:
        return self.assigned_at + self.sla_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.deadline

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.deadline - time.time())


class HITLStateMachine:
    """
    Explicit state machine for HITL review lifecycle.

    States: suspended → pending_review → assigned → under_review
            → approved | rejected | timed_out → resumed | terminated

    Every transition is validated, logged, and optionally audited
    to the hash-chained audit trail.
    """

    def __init__(self, audit_trail: Any = None):
        self._states: dict[str, HITLState] = {}
        self._history: dict[str, list[HITLTransitionRecord]] = {}
        self._slas: dict[str, ReviewSLA] = {}
        self._audit = audit_trail
        self._lock = threading.RLock()

    def get_state(self, instance_id: str) -> HITLState | None:
        return self._states.get(instance_id)

    def initialize(self, instance_id: str) -> HITLState:
        """Initialize a new instance in SUSPENDED state."""
        with self._lock:
            self._states[instance_id] = HITLState.SUSPENDED
            self._history[instance_id] = []
            return HITLState.SUSPENDED

    def transition(
        self,
        instance_id: str,
        to_state: HITLState,
        actor: str = "system",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> HITLTransitionRecord:
        """
        Perform a state transition with validation.

        Raises:
            IllegalStateTransition: If the transition is not allowed
            KeyError: If instance not initialized
        """
        with self._lock:
            current = self._states.get(instance_id)
            if current is None:
                raise KeyError(f"Instance {instance_id!r} not in HITL state machine")

            valid_targets = VALID_TRANSITIONS.get(current, [])
            if to_state not in valid_targets:
                raise IllegalStateTransition(
                    f"Cannot transition {instance_id!r} from {current.value} to {to_state.value}. "
                    f"Valid targets: {[s.value for s in valid_targets]}"
                )

            record = HITLTransitionRecord(
                instance_id=instance_id,
                from_state=current,
                to_state=to_state,
                actor=actor,
                reason=reason,
                metadata=metadata or {},
            )

            self._states[instance_id] = to_state
            self._history.setdefault(instance_id, []).append(record)

            logger.info(
                "HITL transition: instance=%s %s→%s actor=%s reason=%s",
                instance_id, current.value, to_state.value, actor, reason,
                extra={
                    "event_type": "hitl_transition",
                    "instance_id": instance_id,
                    "from_state": current.value,
                    "to_state": to_state.value,
                    "actor": actor,
                },
            )

            # H-012: Record to audit trail
            if self._audit:
                try:
                    self._audit.record(
                        trace_id=instance_id,
                        event_type="reviewer_action" if actor != "system" else "hitl_transition",
                        payload={
                            "instance_id": instance_id,
                            "from_state": current.value,
                            "to_state": to_state.value,
                            "actor": actor,
                            "reason": reason,
                            "metadata": metadata or {},
                        },
                    )
                except Exception as e:
                    logger.warning("Failed to record HITL transition to audit: %s", e)

            return record

    def get_history(self, instance_id: str) -> list[HITLTransitionRecord]:
        """Get full transition history for an instance."""
        return list(self._history.get(instance_id, []))

    # ─── Convenience methods ─────────────────────────────────────

    def suspend(self, instance_id: str, reason: str = "") -> HITLTransitionRecord:
        """Initialize and move to pending_review."""
        self.initialize(instance_id)
        return self.transition(instance_id, HITLState.PENDING_REVIEW, "system", reason)

    def assign(self, instance_id: str, reviewer: str, sla_seconds: float = 3600, escalation_target: str = "", queue: str = "") -> HITLTransitionRecord:
        """Assign to a reviewer with SLA."""
        record = self.transition(instance_id, HITLState.ASSIGNED, "system", f"Assigned to {reviewer}")
        self._slas[instance_id] = ReviewSLA(
            instance_id=instance_id,
            assigned_to=reviewer,
            assigned_at=time.time(),
            sla_seconds=sla_seconds,
            escalation_target=escalation_target,
            queue=queue,
        )
        return record

    def start_review(self, instance_id: str, reviewer: str) -> HITLTransitionRecord:
        return self.transition(instance_id, HITLState.UNDER_REVIEW, reviewer, "Review started")

    def approve(self, instance_id: str, reviewer: str, rationale: str) -> HITLTransitionRecord:
        return self.transition(
            instance_id, HITLState.APPROVED, reviewer, rationale,
            metadata={"action": ReviewerAction.APPROVED.value},
        )

    def reject(self, instance_id: str, reviewer: str, rationale: str) -> HITLTransitionRecord:
        return self.transition(
            instance_id, HITLState.REJECTED, reviewer, rationale,
            metadata={"action": ReviewerAction.REJECTED.value},
        )

    def resume(self, instance_id: str) -> HITLTransitionRecord:
        return self.transition(instance_id, HITLState.RESUMED, "system", "Resumed after approval")

    def terminate(self, instance_id: str, reason: str = "") -> HITLTransitionRecord:
        return self.transition(instance_id, HITLState.TERMINATED, "system", reason)

    # ─── H-013: SLA Enforcement ──────────────────────────────────

    def get_sla(self, instance_id: str) -> ReviewSLA | None:
        return self._slas.get(instance_id)

    def sweep_expired_slas(
        self,
        on_timeout: str = "reassign",
    ) -> list[HITLTransitionRecord]:
        """
        Check all SLAs and handle expired ones.

        Args:
            on_timeout: "reassign" (back to pending_review) or "terminate"

        Returns:
            List of transition records for timed-out instances
        """
        results = []
        with self._lock:
            expired = [
                (iid, sla) for iid, sla in self._slas.items()
                if sla.is_expired and self._states.get(iid) in (
                    HITLState.ASSIGNED, HITLState.UNDER_REVIEW,
                )
            ]

            for instance_id, sla in expired:
                try:
                    record = self.transition(
                        instance_id, HITLState.TIMED_OUT, "system",
                        f"SLA expired after {sla.sla_seconds}s (assigned to {sla.assigned_to})",
                        metadata={"action": ReviewerAction.TIMED_OUT.value, "sla": sla.sla_seconds},
                    )
                    results.append(record)

                    if on_timeout == "reassign":
                        reassign = self.transition(
                            instance_id, HITLState.PENDING_REVIEW, "system",
                            f"Reassigned after SLA timeout (was: {sla.assigned_to})",
                        )
                        results.append(reassign)
                    else:
                        term = self.transition(
                            instance_id, HITLState.TERMINATED, "system",
                            "Terminated after SLA timeout",
                        )
                        results.append(term)

                    del self._slas[instance_id]
                except Exception as e:
                    logger.error("SLA sweep error for %s: %s", instance_id, e)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Statistics for monitoring."""
        state_counts: dict[str, int] = {}
        for s in self._states.values():
            state_counts[s.value] = state_counts.get(s.value, 0) + 1
        return {
            "instances_tracked": len(self._states),
            "state_distribution": state_counts,
            "active_slas": len(self._slas),
            "expired_slas": sum(1 for s in self._slas.values() if s.is_expired),
        }
