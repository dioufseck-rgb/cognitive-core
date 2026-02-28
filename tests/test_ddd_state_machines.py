"""
Cognitive Core — DDD State Machine Tests

Tests the formal state machines from the DDD Unified Specification v1.1:
  - Work Order Lifecycle (Section 3)
  - Capacity Models (Section 6)
  - Capacity Reservation Protocol (Section 8)
  - Eligibility vs. Ranking (Section 9)
  - Circuit Breaker (Section 17.3)
  - Batch Reaper (Section 17.1)
  - Exploration partitioning (Section 17.4)
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.ddd import (
    # Work Order
    DDDWorkOrder, WOStatus, WorkOrderError, ErrorClass,
    RetryPolicy, InvalidTransition, ResumePolicy,
    # Capacity
    CapacityModel, CapacityState, BatchStatus, ReaperAction,
    # Reservation
    CapacityReservation, ReservationStatus,
    # Resource Registry
    ResourceRegistration, ResourceRegistry,
    EligibilityConstraint, EligibilityResult,
    CircuitBreakerState,
)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — WORK ORDER LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

class TestWorkOrderStateMachine(unittest.TestCase):
    """Spec Section 3: Work order state transitions."""

    def _make_wo(self) -> DDDWorkOrder:
        return DDDWorkOrder.create(
            requester_instance_id="wf_test123",
            correlation_id="cor_test",
            contract_name="forensic_review",
            contract_version="1.0.0",
            inputs={"case_id": "CLM-001"},
            priority="high",
            sla_seconds=3600,
        )

    def test_happy_path(self):
        """CREATED → DISPATCHED → CLAIMED → IN_PROGRESS → COMPLETED."""
        wo = self._make_wo()
        self.assertEqual(wo.status, WOStatus.CREATED)
        self.assertFalse(wo.is_terminal)

        wo.transition(WOStatus.DISPATCHED)
        self.assertEqual(wo.status, WOStatus.DISPATCHED)
        self.assertIsNotNone(wo.dispatched_at)

        wo.transition(WOStatus.CLAIMED)
        self.assertEqual(wo.status, WOStatus.CLAIMED)
        self.assertIsNotNone(wo.claimed_at)

        wo.transition(WOStatus.IN_PROGRESS)
        self.assertEqual(wo.status, WOStatus.IN_PROGRESS)

        wo.transition(WOStatus.COMPLETED)
        self.assertEqual(wo.status, WOStatus.COMPLETED)
        self.assertTrue(wo.is_terminal)
        self.assertIsNotNone(wo.completed_at)

    def test_dispatch_to_expired(self):
        """DISPATCHED → EXPIRED when claim_ttl exceeded."""
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.EXPIRED)
        self.assertTrue(wo.is_terminal)

    def test_in_progress_to_failed(self):
        """IN_PROGRESS → FAILED on resource error."""
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.CLAIMED)
        wo.transition(WOStatus.IN_PROGRESS)
        wo.transition(WOStatus.FAILED)
        self.assertTrue(wo.is_terminal)

    def test_cancel_from_dispatched(self):
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.CANCELED)
        self.assertTrue(wo.is_terminal)

    def test_cancel_from_claimed(self):
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.CLAIMED)
        wo.transition(WOStatus.CANCELED)
        self.assertTrue(wo.is_terminal)

    def test_cancel_from_in_progress(self):
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.CLAIMED)
        wo.transition(WOStatus.IN_PROGRESS)
        wo.transition(WOStatus.CANCELED)
        self.assertTrue(wo.is_terminal)

    def test_invalid_transition_created_to_completed(self):
        """Cannot skip states: CREATED → COMPLETED is illegal."""
        wo = self._make_wo()
        with self.assertRaises(InvalidTransition):
            wo.transition(WOStatus.COMPLETED)

    def test_invalid_transition_dispatched_to_in_progress(self):
        """Must CLAIM before IN_PROGRESS."""
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        with self.assertRaises(InvalidTransition):
            wo.transition(WOStatus.IN_PROGRESS)

    def test_invalid_transition_from_terminal(self):
        """No transitions from terminal states."""
        wo = self._make_wo()
        wo.transition(WOStatus.DISPATCHED)
        wo.transition(WOStatus.EXPIRED)
        with self.assertRaises(InvalidTransition):
            wo.transition(WOStatus.DISPATCHED)

    def test_claim_ttl_expiry_check(self):
        """is_claim_expired detects stale dispatches."""
        wo = self._make_wo()
        wo.claim_ttl_seconds = 0.01  # 10ms
        wo.transition(WOStatus.DISPATCHED)
        time.sleep(0.02)
        self.assertTrue(wo.is_claim_expired)

    def test_claim_ttl_not_expired(self):
        wo = self._make_wo()
        wo.claim_ttl_seconds = 60
        wo.transition(WOStatus.DISPATCHED)
        self.assertFalse(wo.is_claim_expired)


class TestWorkOrderError(unittest.TestCase):
    """Spec Section 3: Failure classification."""

    def test_retryable_error(self):
        err = WorkOrderError(
            error_code="resource_timeout",
            error_class=ErrorClass.RETRYABLE,
            message="Resource did not respond within SLA",
        )
        policy = RetryPolicy()
        self.assertTrue(policy.is_retryable(err))

    def test_permanent_error(self):
        err = WorkOrderError(
            error_code="authorization_denied",
            error_class=ErrorClass.PERMANENT,
            message="Resource not authorized",
        )
        policy = RetryPolicy()
        self.assertFalse(policy.is_retryable(err))

    def test_degraded_error(self):
        err = WorkOrderError(
            error_code="partial_result",
            error_class=ErrorClass.DEGRADED,
            message="Partial result available",
        )
        self.assertEqual(err.error_class, ErrorClass.DEGRADED)

    def test_exponential_backoff(self):
        policy = RetryPolicy(base_delay_seconds=10.0, max_delay_seconds=300.0)
        self.assertEqual(policy.compute_delay(1), 10.0)
        self.assertEqual(policy.compute_delay(2), 20.0)
        self.assertEqual(policy.compute_delay(3), 40.0)
        self.assertEqual(policy.compute_delay(6), 300.0)  # capped

    def test_non_retryable_overrides_class(self):
        """Non-retryable error code overrides retryable class."""
        err = WorkOrderError(
            error_code="schema_violation",
            error_class=ErrorClass.RETRYABLE,
            message="Schema mismatch",
        )
        policy = RetryPolicy()
        self.assertFalse(policy.is_retryable(err))


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — CAPACITY MODELS
# ═══════════════════════════════════════════════════════════════════

class TestSlotCapacity(unittest.TestCase):
    """Spec Section 6, Model 1: Slot (The Plane)."""

    def test_accept_and_release(self):
        cap = CapacityState(model=CapacityModel.SLOT, max_concurrent=3)
        self.assertTrue(cap.can_accept())
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        self.assertFalse(cap.can_accept())
        cap.on_release()
        self.assertTrue(cap.can_accept())

    def test_utilization(self):
        cap = CapacityState(model=CapacityModel.SLOT, max_concurrent=4, current_load=2)
        self.assertAlmostEqual(cap.utilization_pct, 0.5)


class TestVolumeCapacity(unittest.TestCase):
    """Spec Section 6, Model 2: Volume (The Train)."""

    def test_variable_amounts(self):
        cap = CapacityState(model=CapacityModel.VOLUME, max_volume=100.0)
        self.assertTrue(cap.can_accept(30.0))
        cap.on_assign(30.0)
        self.assertTrue(cap.can_accept(70.0))
        self.assertFalse(cap.can_accept(71.0))
        cap.on_release(30.0)
        self.assertTrue(cap.can_accept(71.0))

    def test_release_never_negative(self):
        cap = CapacityState(model=CapacityModel.VOLUME, max_volume=100.0, current_volume=5.0)
        cap.on_release(10.0)
        self.assertEqual(cap.current_volume, 0.0)


class TestBatchCapacity(unittest.TestCase):
    """Spec Section 6, Model 3: Batch (The Charter Flight)."""

    def test_batch_lifecycle(self):
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=3,
            batch_timeout_seconds=3600,
        )
        self.assertTrue(cap.can_accept())
        cap.on_assign()
        cap.on_assign()
        self.assertFalse(cap.check_batch_trigger())
        cap.on_assign()
        self.assertTrue(cap.check_batch_trigger())

        cap.trigger_batch()
        self.assertEqual(cap.batch_status, BatchStatus.EXECUTING)
        self.assertFalse(cap.can_accept())

        cap.complete_batch()
        self.assertEqual(cap.batch_status, BatchStatus.COLLECTING)
        self.assertEqual(cap.batch_items, 0)
        self.assertTrue(cap.can_accept())

    def test_reaper_detection(self):
        """Section 17.1: Batch stuck in EXECUTING detected by reaper."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=1,
            max_execution_duration_seconds=0.01,  # 10ms
        )
        cap.on_assign()
        cap.trigger_batch()
        time.sleep(0.02)
        self.assertTrue(cap.check_reaper())


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — CAPACITY RESERVATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════

class TestCapacityReservation(unittest.TestCase):
    """Spec Section 8: Reserve/commit/release with TTL."""

    def test_reservation_lifecycle_commit(self):
        rsv = CapacityReservation.create("res_1", "wo_1", amount=1.0)
        self.assertEqual(rsv.status, ReservationStatus.HELD)
        rsv.commit()
        self.assertEqual(rsv.status, ReservationStatus.COMMITTED)

    def test_reservation_lifecycle_release(self):
        rsv = CapacityReservation.create("res_1", "wo_1", amount=5.0)
        amount = rsv.release()
        self.assertEqual(amount, 5.0)
        self.assertEqual(rsv.status, ReservationStatus.RELEASED)

    def test_commit_idempotent(self):
        rsv = CapacityReservation.create("res_1", "wo_1")
        rsv.commit()
        rsv.commit()  # should not raise
        self.assertEqual(rsv.status, ReservationStatus.COMMITTED)

    def test_release_idempotent(self):
        rsv = CapacityReservation.create("res_1", "wo_1", amount=3.0)
        self.assertEqual(rsv.release(), 3.0)
        self.assertEqual(rsv.release(), 0.0)  # idempotent

    def test_ttl_expiry(self):
        rsv = CapacityReservation.create("res_1", "wo_1", ttl_seconds=0.01)
        time.sleep(0.02)
        self.assertTrue(rsv.is_expired)
        amount = rsv.expire()
        self.assertEqual(rsv.status, ReservationStatus.EXPIRED)
        self.assertGreater(amount, 0)

    def test_committed_not_expirable(self):
        rsv = CapacityReservation.create("res_1", "wo_1", ttl_seconds=0.01)
        rsv.commit()
        time.sleep(0.02)
        self.assertFalse(rsv.is_expired)  # committed, not held

    def test_cannot_commit_released(self):
        rsv = CapacityReservation.create("res_1", "wo_1")
        rsv.release()
        with self.assertRaises(InvalidTransition):
            rsv.commit()


class TestResourceRegistryReservation(unittest.TestCase):
    """Spec Section 8: Registry-level reserve/commit/release."""

    def setUp(self):
        self.registry = ResourceRegistry()
        self.res = ResourceRegistration.create(
            resource_id="adj_001",
            resource_type="human",
            capabilities=[("claim_adjudication", "commercial_property")],
            capacity_model=CapacityModel.SLOT,
            max_capacity=2,
        )
        self.registry.register(self.res)

    def test_reserve_and_commit(self):
        rsv = self.registry.reserve("adj_001", "wo_1")
        self.assertIsNotNone(rsv)
        self.assertEqual(self.res.capacity.current_load, 1)
        self.registry.commit_reservation(rsv.reservation_id)

    def test_reserve_denied_at_capacity(self):
        self.registry.reserve("adj_001", "wo_1")
        self.registry.reserve("adj_001", "wo_2")
        rsv3 = self.registry.reserve("adj_001", "wo_3")
        self.assertIsNone(rsv3)

    def test_release_returns_capacity(self):
        rsv = self.registry.reserve("adj_001", "wo_1")
        self.assertEqual(self.res.capacity.current_load, 1)
        self.registry.release_reservation(rsv.reservation_id)
        self.assertEqual(self.res.capacity.current_load, 0)

    def test_ttl_sweep(self):
        rsv = self.registry.reserve("adj_001", "wo_1")
        rsv.ttl_seconds = 0.01
        time.sleep(0.02)
        expired = self.registry.sweep_expired_reservations()
        self.assertEqual(len(expired), 1)
        self.assertEqual(self.res.capacity.current_load, 0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — ELIGIBILITY VS. RANKING
# ═══════════════════════════════════════════════════════════════════

class TestEligibilityFilter(unittest.TestCase):
    """Spec Section 9: Hard eligibility filters with audit trail."""

    def setUp(self):
        self.registry = ResourceRegistry()

    def test_capability_match(self):
        r1 = ResourceRegistration.create(
            "adj_1", "human",
            capabilities=[("claim_adjudication", "commercial_property")],
        )
        r2 = ResourceRegistration.create(
            "adj_2", "human",
            capabilities=[("claim_adjudication", "residential_property")],
        )
        self.registry.register(r1)
        self.registry.register(r2)

        eligible, audit = self.registry.filter_eligible(
            "claim_adjudication", "commercial_property"
        )
        self.assertEqual(len(eligible), 1)
        self.assertEqual(eligible[0].resource_id, "adj_1")

        # Verify audit trail
        excluded = [a for a in audit if not a.eligible]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].failed_constraint, "capability_match")

    def test_circuit_breaker_excludes(self):
        r1 = ResourceRegistration.create(
            "adj_1", "human",
            capabilities=[("review", "fraud")],
        )
        r1.circuit_breaker.status = "open"
        r1.circuit_breaker.opened_at = time.time()
        self.registry.register(r1)

        eligible, audit = self.registry.filter_eligible("review", "fraud")
        self.assertEqual(len(eligible), 0)
        excluded = [a for a in audit if not a.eligible]
        self.assertEqual(excluded[0].failed_constraint, "circuit_breaker")

    def test_stale_heartbeat_excludes_volume(self):
        r1 = ResourceRegistration.create(
            "truck_1", "system",
            capabilities=[("delivery", "packages")],
            capacity_model=CapacityModel.VOLUME,
            max_capacity=100,
        )
        r1.last_heartbeat = time.time() - 10000  # very stale
        r1.stale_after_seconds = 900
        self.registry.register(r1)

        eligible, audit = self.registry.filter_eligible("delivery", "packages")
        self.assertEqual(len(eligible), 0)

    def test_slot_model_no_heartbeat_check(self):
        """Slot model resources don't need heartbeats."""
        r1 = ResourceRegistration.create(
            "adj_1", "human",
            capabilities=[("review", "claims")],
            capacity_model=CapacityModel.SLOT,
        )
        r1.last_heartbeat = time.time() - 10000  # stale, but irrelevant for slot
        self.registry.register(r1)

        eligible, _ = self.registry.filter_eligible("review", "claims")
        self.assertEqual(len(eligible), 1)


# ═══════════════════════════════════════════════════════════════════
# SECTION 17.1 — BATCH REAPER
# ═══════════════════════════════════════════════════════════════════

class TestBatchReaper(unittest.TestCase):
    """Spec Section 17.1: Reaper catches stuck batches."""

    def test_reaper_fail_action(self):
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "etl_1", "system",
            capabilities=[("etl_load", "warehouse")],
            capacity_model=CapacityModel.BATCH,
            max_capacity=50,
        )
        r.capacity.max_execution_duration_seconds = 0.01
        r.capacity.reaper_action = ReaperAction.FAIL
        registry.register(r)

        # Trigger batch
        r.capacity.on_assign()
        r.capacity.batch_threshold = 1
        r.capacity.trigger_batch()
        time.sleep(0.02)

        reaped = registry.sweep_stale_batches()
        self.assertEqual(reaped, ["etl_1"])
        self.assertEqual(r.capacity.batch_status, BatchStatus.REAPED)

    def test_reaper_retry_once(self):
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "etl_1", "system",
            capabilities=[("etl_load", "warehouse")],
            capacity_model=CapacityModel.BATCH,
            max_capacity=50,
        )
        r.capacity.max_execution_duration_seconds = 0.01
        r.capacity.reaper_action = ReaperAction.RETRY_ONCE
        r.capacity.batch_threshold = 1
        registry.register(r)

        # First timeout: retry
        r.capacity.on_assign()
        r.capacity.trigger_batch()
        time.sleep(0.02)
        reaped = registry.sweep_stale_batches()
        self.assertEqual(reaped, [])  # retried, not reaped
        self.assertEqual(r.capacity.batch_status, BatchStatus.COLLECTING)

        # Second timeout: fail
        r.capacity.trigger_batch()
        time.sleep(0.02)
        reaped = registry.sweep_stale_batches()
        self.assertEqual(reaped, ["etl_1"])

    def test_reaper_idempotent(self):
        """Invariant 11: Running sweep twice is idempotent."""
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "etl_1", "system",
            capabilities=[("etl_load", "warehouse")],
            capacity_model=CapacityModel.BATCH,
            max_capacity=50,
        )
        r.capacity.max_execution_duration_seconds = 0.01
        r.capacity.batch_threshold = 1
        registry.register(r)

        r.capacity.on_assign()
        r.capacity.trigger_batch()
        time.sleep(0.02)

        reaped1 = registry.sweep_stale_batches()
        reaped2 = registry.sweep_stale_batches()
        self.assertEqual(len(reaped1), 1)
        self.assertEqual(len(reaped2), 0)  # already reaped


# ═══════════════════════════════════════════════════════════════════
# SECTION 17.3 — CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

class TestCircuitBreaker(unittest.TestCase):
    """Spec Section 17.3: Per-resource circuit breaker."""

    def test_stays_closed_on_success(self):
        cb = CircuitBreakerState(window_size=5, open_threshold=0.50)
        for _ in range(20):
            result = cb.record_outcome(True)
        self.assertEqual(cb.status, "closed")

    def test_opens_on_failure_threshold(self):
        cb = CircuitBreakerState(window_size=4, open_threshold=0.50)
        cb.record_outcome(True)
        cb.record_outcome(True)
        cb.record_outcome(False)
        result = cb.record_outcome(False)  # 2/4 = 50% → opens
        self.assertEqual(result, "open")
        self.assertEqual(cb.status, "open")
        self.assertFalse(cb.is_eligible)

    def test_half_open_after_cooldown(self):
        cb = CircuitBreakerState(
            window_size=4,
            open_threshold=0.50,
            cooldown_seconds=10.0,
        )
        # Trip it at a known time
        base = time.time()
        for _ in range(4):
            cb.record_outcome(False, now=base)
        self.assertEqual(cb.status, "open")

        # Not yet cooled down
        self.assertFalse(cb.check_cooldown(now=base + 5.0))
        # After cooldown
        self.assertTrue(cb.check_cooldown(now=base + 11.0))
        self.assertEqual(cb.status, "half_open")

    def test_half_open_success_closes(self):
        cb = CircuitBreakerState(window_size=4, open_threshold=0.50)
        cb.status = "half_open"
        result = cb.record_outcome(True)
        self.assertEqual(result, "closed")
        self.assertTrue(cb.is_eligible)

    def test_half_open_failure_reopens_with_backoff(self):
        cb = CircuitBreakerState(
            window_size=4,
            open_threshold=0.50,
            cooldown_seconds=100.0,
            backoff_multiplier=2.0,
        )
        cb.status = "half_open"
        cb.current_cooldown = 100.0
        result = cb.record_outcome(False)
        self.assertEqual(result, "open")
        self.assertEqual(cb.current_cooldown, 200.0)  # doubled

    def test_registry_integration(self):
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "adj_1", "human",
            capabilities=[("review", "claims")],
        )
        r.circuit_breaker.window_size = 4
        r.circuit_breaker.open_threshold = 0.50
        registry.register(r)

        # Record failures until circuit opens
        registry.record_outcome("adj_1", False)
        registry.record_outcome("adj_1", False)
        registry.record_outcome("adj_1", False)
        transition = registry.record_outcome("adj_1", False)
        self.assertEqual(transition, "open")

        # Verify excluded from eligibility
        eligible, _ = registry.filter_eligible("review", "claims")
        self.assertEqual(len(eligible), 0)


# ═══════════════════════════════════════════════════════════════════
# SECTION 17.4 — EXPLORATION POLICY
# ═══════════════════════════════════════════════════════════════════

class TestExplorationPartitioning(unittest.TestCase):
    """Spec Section 17.4: Proven vs. unproven resource partitioning."""

    def test_partition_by_maturity(self):
        registry = ResourceRegistry()
        proven = ResourceRegistration.create(
            "adj_1", "human", capabilities=[("review", "claims")]
        )
        proven.completed_work_orders = 15

        unproven = ResourceRegistration.create(
            "adj_2", "human", capabilities=[("review", "claims")]
        )
        unproven.completed_work_orders = 3

        registry.register(proven)
        registry.register(unproven)

        eligible, _ = registry.filter_eligible("review", "claims")
        p, u = registry.partition_by_maturity(eligible, maturity_threshold=10)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].resource_id, "adj_1")
        self.assertEqual(len(u), 1)
        self.assertEqual(u[0].resource_id, "adj_2")

    def test_outcome_increments_completions(self):
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "adj_1", "human", capabilities=[("review", "claims")]
        )
        registry.register(r)
        self.assertEqual(r.completed_work_orders, 0)

        registry.record_outcome("adj_1", True)
        registry.record_outcome("adj_1", True)
        self.assertEqual(r.completed_work_orders, 2)


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: FULL DISPATCH FLOW
# ═══════════════════════════════════════════════════════════════════

class TestFullDispatchFlow(unittest.TestCase):
    """End-to-end: create work order → reserve → dispatch → complete."""

    def test_full_lifecycle(self):
        registry = ResourceRegistry()
        r = ResourceRegistration.create(
            "adj_001", "human",
            capabilities=[("claim_adjudication", "commercial_property")],
            capacity_model=CapacityModel.SLOT,
            max_capacity=5,
        )
        registry.register(r)

        # 1. Create work order
        wo = DDDWorkOrder.create(
            requester_instance_id="wf_parent",
            correlation_id="cor_001",
            contract_name="adjudication_v1",
            priority="high",
            sla_seconds=3600,
        )

        # 2. Filter eligible
        eligible, audit = registry.filter_eligible(
            "claim_adjudication", "commercial_property"
        )
        self.assertEqual(len(eligible), 1)

        # 3. Reserve capacity
        rsv = registry.reserve("adj_001", wo.work_order_id)
        self.assertIsNotNone(rsv)
        self.assertEqual(r.capacity.current_load, 1)

        # 4. Dispatch
        wo.resource_id = "adj_001"
        wo.reservation_id = rsv.reservation_id
        wo.transition(WOStatus.DISPATCHED)
        registry.commit_reservation(rsv.reservation_id)

        # 5. Claim
        wo.transition(WOStatus.CLAIMED)

        # 6. Execute
        wo.transition(WOStatus.IN_PROGRESS)

        # 7. Complete
        wo.transition(WOStatus.COMPLETED)
        wo.result = {"decision": "approved", "amount": 27800.00}

        # 8. Release capacity
        registry.release_reservation(rsv.reservation_id)
        self.assertEqual(r.capacity.current_load, 0)

        # 9. Record outcome for circuit breaker
        registry.record_outcome("adj_001", True)
        self.assertEqual(r.completed_work_orders, 1)

    def test_failure_with_retry(self):
        """Work order fails with retryable error, gets re-dispatched."""
        registry = ResourceRegistry()
        r1 = ResourceRegistration.create(
            "adj_001", "human",
            capabilities=[("review", "fraud")],
            capacity_model=CapacityModel.SLOT,
            max_capacity=5,
        )
        registry.register(r1)

        wo = DDDWorkOrder.create(
            requester_instance_id="wf_parent",
            correlation_id="cor_001",
            contract_name="fraud_review_v1",
        )

        # First attempt: dispatch → claim → execute → fail
        rsv = registry.reserve("adj_001", wo.work_order_id)
        wo.transition(WOStatus.DISPATCHED)
        registry.commit_reservation(rsv.reservation_id)
        wo.transition(WOStatus.CLAIMED)
        wo.transition(WOStatus.IN_PROGRESS)

        # Fail with retryable error
        wo.transition(WOStatus.FAILED)
        wo.error = WorkOrderError(
            error_code="resource_timeout",
            error_class=ErrorClass.RETRYABLE,
            message="Timeout waiting for response",
        )
        registry.release_reservation(rsv.reservation_id)
        registry.record_outcome("adj_001", False)

        # Retry policy says: retryable
        policy = RetryPolicy()
        self.assertTrue(policy.is_retryable(wo.error))

        # Second attempt: new work order (same request_id for idempotency)
        wo2 = DDDWorkOrder.create(
            requester_instance_id="wf_parent",
            correlation_id="cor_001",
            contract_name="fraud_review_v1",
        )
        wo2.attempt = 2

        rsv2 = registry.reserve("adj_001", wo2.work_order_id)
        self.assertIsNotNone(rsv2)
        wo2.transition(WOStatus.DISPATCHED)
        wo2.transition(WOStatus.CLAIMED)
        wo2.transition(WOStatus.IN_PROGRESS)
        wo2.transition(WOStatus.COMPLETED)
        registry.release_reservation(rsv2.reservation_id)
        registry.record_outcome("adj_001", True)

        self.assertTrue(wo2.is_terminal)
        self.assertEqual(wo2.status, WOStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
