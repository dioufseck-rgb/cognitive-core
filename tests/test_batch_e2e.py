"""
Cognitive Core — Batch Mode End-to-End Test

Tests a single resource operating in batch mode with a time window:

  Scenario: A "nightly SAR batch processor" collects suspicious activity
  reports during the day. It fires either when 5 reports accumulate
  (threshold trigger) OR when 60 seconds pass since the first report
  (time window trigger), whichever comes first.

  This test proves:
  1. Batch capacity model accepts items during COLLECTING phase
  2. Time window trigger fires batch before threshold is reached
  3. Threshold trigger fires batch when items accumulate fast
  4. Optimizer dispatches to batch resource with correct cost matrix
  5. DDR records the batch dispatch decision
  6. Reservation events log acquire/commit per work order
  7. Reaper detects stuck batches
  8. Complete lifecycle: collect → trigger → execute → complete → reset
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.ddd import (
    DDDWorkOrder,
    ResourceRegistration,
    ResourceRegistry,
    CapacityModel,
    CapacityState,
    BatchStatus,
    ReaperAction,
)
from coordinator.optimizer import DispatchOptimizer
from coordinator.physics import OptimizationConfig, DefaultAssignmentPhysics
from coordinator.archetypes import AssignmentArchetype, AssignmentParams
from coordinator.hardening import ReservationEventLog, LearningScopeEnforcer


def make_batch_resource(
    rid: str = "sar_batch_processor",
    threshold: int = 5,
    timeout_seconds: float = 60.0,
    max_execution: float = 3600.0,
) -> ResourceRegistration:
    """Create a batch-mode resource with time window."""
    cap = CapacityState(
        model=CapacityModel.BATCH,
        batch_threshold=threshold,
        batch_timeout_seconds=timeout_seconds,
        max_execution_duration_seconds=max_execution,
        reaper_action=ReaperAction.RETRY_ONCE,
    )
    return ResourceRegistration(
        resource_id=rid,
        resource_type="automated",
        capabilities=[("investigate", "sar"), ("batch_review", "sar")],
        capacity=cap,
        attributes={"cost_rate": 10.0, "quality_score": 0.95},
        completed_work_orders=50,
    )


def make_sar_wo(case_id: str = "SAR-001") -> DDDWorkOrder:
    """Create a SAR work order."""
    return DDDWorkOrder.create(
        "sar_investigation", "cor_sar",
        "investigate",
        priority="high",
        sla_seconds=86400.0,  # 24h SLA
        case_id=case_id,
    )


# ═══════════════════════════════════════════════════════════════════
# BATCH CAPACITY MODEL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestBatchTimeWindow(unittest.TestCase):
    """Core batch model: time window trigger."""

    def test_time_window_trigger(self):
        """Batch fires on time window even below threshold."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=10,          # high threshold
            batch_timeout_seconds=60.0,  # 60-second window
        )
        # Add 3 items (well below threshold of 10)
        t0 = 1000000.0
        cap.batch_collecting_since = t0
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        self.assertEqual(cap.batch_items, 3)

        # Not triggered yet — only 30s elapsed
        self.assertFalse(cap.check_batch_trigger(now=t0 + 30))

        # Triggered at 60s — time window expired
        self.assertTrue(cap.check_batch_trigger(now=t0 + 60))

    def test_threshold_trigger_before_window(self):
        """Batch fires on threshold even if time window hasn't expired."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=3,
            batch_timeout_seconds=3600.0,  # 1-hour window
        )
        cap.on_assign()
        cap.on_assign()
        self.assertFalse(cap.check_batch_trigger())

        cap.on_assign()  # 3rd item → threshold reached
        self.assertTrue(cap.check_batch_trigger())

    def test_collecting_since_set_on_first_item(self):
        """batch_collecting_since is set when the first item arrives."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=5,
            batch_timeout_seconds=60.0,
        )
        self.assertIsNone(cap.batch_collecting_since)
        cap.on_assign()
        self.assertIsNotNone(cap.batch_collecting_since)

    def test_collecting_since_not_reset_on_subsequent(self):
        """batch_collecting_since stays at first-item time."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=5,
            batch_timeout_seconds=60.0,
        )
        cap.on_assign()
        first_time = cap.batch_collecting_since
        time.sleep(0.01)
        cap.on_assign()
        self.assertEqual(cap.batch_collecting_since, first_time)

    def test_complete_batch_resets_collecting_since(self):
        """complete_batch() resets collecting_since for next cycle."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=3,
            batch_timeout_seconds=60.0,
        )
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        cap.trigger_batch()
        cap.complete_batch()
        self.assertIsNone(cap.batch_collecting_since)
        self.assertEqual(cap.batch_items, 0)
        self.assertEqual(cap.batch_status, BatchStatus.COLLECTING)

    def test_no_trigger_with_zero_items(self):
        """Empty batch never triggers, even past time window."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=5,
            batch_timeout_seconds=60.0,
        )
        self.assertFalse(cap.check_batch_trigger(now=time.time() + 9999))

    def test_no_trigger_when_executing(self):
        """Batch in EXECUTING state doesn't re-trigger."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=3,
            batch_timeout_seconds=60.0,
        )
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        cap.trigger_batch()
        self.assertFalse(cap.check_batch_trigger())


# ═══════════════════════════════════════════════════════════════════
# FULL LIFECYCLE THROUGH OPTIMIZER
# ═══════════════════════════════════════════════════════════════════

class TestBatchDispatchLifecycle(unittest.TestCase):
    """Full batch lifecycle through the optimizer pipeline."""

    def setUp(self):
        self.registry = ResourceRegistry()
        self.resource = make_batch_resource(
            threshold=5, timeout_seconds=60.0,
        )
        self.registry.register(self.resource)

    def test_batch_collects_until_threshold(self):
        """Work orders accumulate in batch until threshold fires."""
        res_log = ReservationEventLog()
        ddr_log = []

        def capture_ddr(decision, eligible, audit, config, solution):
            ddr_log.append({
                "wo": decision.work_order_id,
                "res": decision.selected_resource_id,
                "tier": decision.tier,
            })

        optimizer = DispatchOptimizer(
            self.registry,
            ddr_callback=capture_ddr,
            reservation_log=res_log,
        )
        config = OptimizationConfig(exploration_enabled=False)

        # Dispatch 5 work orders — one at a time, simulating arrival
        for i in range(5):
            wo = make_sar_wo(f"SAR-{i:03d}")
            decisions = optimizer.dispatch(
                [wo], "investigate", "sar", config,
            )
            self.assertEqual(len(decisions), 1)
            if decisions[0].selected_resource_id:
                self.assertEqual(
                    decisions[0].selected_resource_id,
                    "sar_batch_processor",
                )

        # Batch resource should now have 5 items
        res = self.registry.get("sar_batch_processor")
        # Items were dispatched individually through optimizer which
        # reserves capacity — check the batch item count matches assigns
        self.assertEqual(len(ddr_log), 5)
        self.assertTrue(all(d["res"] == "sar_batch_processor" for d in ddr_log))

        # Reservation log should have 5 acquire + 5 commit events
        events = res_log.get_events()
        acquires = [e for e in events if e.operation == "acquire"]
        commits = [e for e in events if e.operation == "commit"]
        self.assertEqual(len(acquires), 5)
        self.assertEqual(len(commits), 5)

    def test_time_window_fires_below_threshold(self):
        """Batch fires on time window even with only 2 items."""
        resource = make_batch_resource(threshold=10, timeout_seconds=60.0)
        cap = resource.capacity

        # Simulate: 2 items arrive at t=0
        t0 = 1000000.0
        cap.on_assign()
        cap.batch_collecting_since = t0  # override auto-set time for test control
        cap.on_assign()

        # At t+30: not triggered
        self.assertFalse(cap.check_batch_trigger(now=t0 + 30))
        self.assertEqual(cap.batch_items, 2)

        # At t+60: time window fires!
        self.assertTrue(cap.check_batch_trigger(now=t0 + 60))

        # Trigger and execute
        cap.trigger_batch(now=t0 + 60)
        self.assertEqual(cap.batch_status, BatchStatus.EXECUTING)
        self.assertEqual(cap.batch_items, 2)

        # Cannot accept new items while executing
        self.assertFalse(cap.can_accept())

        # Complete the batch
        cap.complete_batch()
        self.assertEqual(cap.batch_status, BatchStatus.COLLECTING)
        self.assertEqual(cap.batch_items, 0)
        self.assertTrue(cap.can_accept())

    def test_reaper_detects_stuck_batch(self):
        """Reaper fires when batch execution exceeds max duration."""
        resource = make_batch_resource(
            threshold=3, timeout_seconds=60.0,
        )
        resource.capacity.max_execution_duration_seconds = 120.0
        cap = resource.capacity

        # Collect and trigger
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        t0 = 1000000.0
        cap.trigger_batch(now=t0)

        # At t+60: not reaped yet
        self.assertFalse(cap.check_reaper(now=t0 + 60))

        # At t+121: reaped!
        self.assertTrue(cap.check_reaper(now=t0 + 121))

    def test_optimizer_cost_matrix_for_batch(self):
        """Physics extraction produces valid cost matrix for batch resource."""
        physics = DefaultAssignmentPhysics()
        config = OptimizationConfig()
        wos = [make_sar_wo(f"SAR-{i}") for i in range(3)]

        params = physics.extract_parameters(
            wos, [self.resource], config,
        )
        # 3 work orders × 1 resource
        self.assertEqual(params.num_work_orders, 3)
        self.assertEqual(params.num_resources, 1)
        # Batch capacity: should show accepting (COLLECTING)
        self.assertTrue(params.capacities[0] > 0)
        # Cost matrix should have values
        for row in params.cost_matrix:
            self.assertTrue(all(c > 0 for c in row))

    def test_batch_rejects_when_executing(self):
        """Optimizer cannot dispatch to batch resource in EXECUTING state."""
        # Put resource into EXECUTING
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.batch_items = self.resource.capacity.batch_threshold
        self.resource.capacity.trigger_batch()

        optimizer = DispatchOptimizer(self.registry)
        config = OptimizationConfig(exploration_enabled=False)
        wo = make_sar_wo()
        decisions = optimizer.dispatch([wo], "investigate", "sar", config)

        # Should not be assigned — batch is executing
        self.assertIsNone(decisions[0].selected_resource_id)


class TestBatchMultiCycle(unittest.TestCase):
    """Multiple batch cycles: collect → fire → execute → complete → repeat."""

    def test_two_full_cycles(self):
        """Resource completes two full batch cycles."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=3,
            batch_timeout_seconds=60.0,
        )

        # Cycle 1
        cap.on_assign()
        cap.on_assign()
        cap.on_assign()
        self.assertTrue(cap.check_batch_trigger())
        cap.trigger_batch()
        self.assertEqual(cap.batch_status, BatchStatus.EXECUTING)
        self.assertFalse(cap.can_accept())

        cap.complete_batch()
        self.assertEqual(cap.batch_status, BatchStatus.COLLECTING)
        self.assertTrue(cap.can_accept())
        self.assertIsNone(cap.batch_collecting_since)

        # Cycle 2
        cap.on_assign()
        self.assertIsNotNone(cap.batch_collecting_since)
        cap.on_assign()
        cap.on_assign()
        self.assertTrue(cap.check_batch_trigger())
        cap.trigger_batch()
        cap.complete_batch()
        self.assertEqual(cap.batch_items, 0)

    def test_time_window_cycle_then_threshold_cycle(self):
        """First cycle fires on time window, second on threshold."""
        cap = CapacityState(
            model=CapacityModel.BATCH,
            batch_threshold=5,
            batch_timeout_seconds=30.0,
        )

        # Cycle 1: 2 items, time window fires at t+30
        t0 = 1000000.0
        cap.on_assign()
        cap.batch_collecting_since = t0
        cap.on_assign()
        self.assertFalse(cap.check_batch_trigger(now=t0 + 15))
        self.assertTrue(cap.check_batch_trigger(now=t0 + 30))
        cap.trigger_batch(now=t0 + 30)
        cap.complete_batch()

        # Cycle 2: 5 items, threshold fires immediately
        for _ in range(5):
            cap.on_assign()
        self.assertTrue(cap.check_batch_trigger())


class TestBatchWithFullAuditTrail(unittest.TestCase):
    """Batch dispatch produces complete audit trail (DDR + reservation log)."""

    def test_audit_trail_completeness(self):
        """Every batch work order has DDR and reservation events."""
        registry = ResourceRegistry()
        resource = make_batch_resource(threshold=5, timeout_seconds=60.0)
        registry.register(resource)

        res_log = ReservationEventLog()
        ddr_entries = []

        def capture_ddr(decision, eligible, audit, config, solution):
            ddr_entries.append({
                "wo_id": decision.work_order_id,
                "resource": decision.selected_resource_id,
                "tier": decision.tier,
                "reservation": decision.reservation_id,
                "eligibility_count": len(decision.eligibility_results),
                "ranking_count": len(decision.ranking_scores),
            })

        optimizer = DispatchOptimizer(
            registry,
            ddr_callback=capture_ddr,
            reservation_log=res_log,
            learning_enforcer=LearningScopeEnforcer(),
        )
        config = OptimizationConfig(exploration_enabled=False)

        # Dispatch 3 work orders as a batch
        wos = [make_sar_wo(f"SAR-{i:03d}") for i in range(3)]
        decisions = optimizer.dispatch(wos, "investigate", "sar", config)

        # All assigned
        assigned = [d for d in decisions if d.selected_resource_id]
        self.assertEqual(len(assigned), 3)

        # DDR for each
        self.assertEqual(len(ddr_entries), 3)
        for entry in ddr_entries:
            self.assertEqual(entry["resource"], "sar_batch_processor")
            self.assertIsNotNone(entry["reservation"])
            self.assertGreater(entry["ranking_count"], 0)

        # Reservation events: 3 acquires + 3 commits
        all_events = res_log.get_events()
        self.assertEqual(len(all_events), 6)
        acquires = [e for e in all_events if e.operation == "acquire"]
        commits = [e for e in all_events if e.operation == "commit"]
        self.assertEqual(len(acquires), 3)
        self.assertEqual(len(commits), 3)
        # All for same resource
        self.assertTrue(all(e.resource_id == "sar_batch_processor" for e in all_events))


if __name__ == "__main__":
    unittest.main()
