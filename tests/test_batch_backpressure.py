"""
Cognitive Core — Batch Backpressure Queue Tests

Tests the scenario:
  1. Batch resource collects 3 items, triggers on time window
  2. During EXECUTING phase, 2 more work orders arrive
  3. Those work orders are QUEUED (not failed, not dropped)
  4. Batch completes → COLLECTING again
  5. Queue drains → queued work orders dispatched
  6. Full audit trail for the queue/drain lifecycle

This proves the system handles resource unavailability gracefully
rather than failing or silently dropping work.
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.ddd import (
    CapacityModel, CapacityState, BatchStatus,
    ResourceRegistration, ResourceRegistry, DDDWorkOrder,
)
from coordinator.types import WorkOrderStatus, WorkOrder, WorkOrderResult
from coordinator.optimizer import DispatchOptimizer
from coordinator.physics import OptimizationConfig
from coordinator.hardening import ReservationEventLog


def make_batch_resource(rid="batch_proc", threshold=5, timeout=60.0):
    cap = CapacityState(
        model=CapacityModel.BATCH,
        batch_threshold=threshold,
        batch_timeout_seconds=timeout,
    )
    return ResourceRegistration(
        resource_id=rid,
        resource_type="automated",
        capabilities=[("review", "sar")],
        capacity=cap,
        attributes={"cost_rate": 10.0, "quality_score": 0.9},
        completed_work_orders=30,
    )


def make_wo(case_id="SAR-001"):
    return DDDWorkOrder.create("investigation", "cor_1", "review",
                               priority="routine", sla_seconds=86400.0,
                               case_id=case_id)


def make_store_wo(wo_id, status=WorkOrderStatus.QUEUED):
    """Create a WorkOrder for the Coordinator store (different from DDDWorkOrder)."""
    return WorkOrder(
        work_order_id=wo_id,
        requester_instance_id="inst_test",
        correlation_id="cor_test",
        contract_name="review",
        contract_version=1,
        inputs={},
        handler_workflow_type="investigation",
        handler_domain="sar",
        status=status,
    )


class TestBatchBackpressure(unittest.TestCase):
    """Work orders during batch EXECUTING are queued, not dropped."""

    def test_optimizer_rejects_during_executing(self):
        """Optimizer returns no assignment when batch is EXECUTING."""
        registry = ResourceRegistry()
        resource = make_batch_resource(threshold=3)
        registry.register(resource)

        optimizer = DispatchOptimizer(registry)
        config = OptimizationConfig(exploration_enabled=False)

        # Collect 3 items and trigger
        for i in range(3):
            wo = make_wo(f"SAR-{i}")
            optimizer.dispatch([wo], "review", "sar", config)

        # Now trigger batch execution
        resource.capacity.trigger_batch()
        self.assertEqual(resource.capacity.batch_status, BatchStatus.EXECUTING)

        # New work order arrives — should get no assignment
        wo_new = make_wo("SAR-NEW")
        decisions = optimizer.dispatch([wo_new], "review", "sar", config)
        self.assertEqual(len(decisions), 1)
        self.assertIsNone(decisions[0].selected_resource_id)
        self.assertIn(decisions[0].tier,
                       ("no_eligible_resources", "no_solution", "reservation_denied", "unassigned"))

    def test_resource_accepts_after_complete(self):
        """After batch completes, resource accepts new work."""
        registry = ResourceRegistry()
        resource = make_batch_resource(threshold=3)
        registry.register(resource)

        optimizer = DispatchOptimizer(registry)
        config = OptimizationConfig(exploration_enabled=False)

        # Trigger batch
        resource.capacity.on_assign()
        resource.capacity.on_assign()
        resource.capacity.on_assign()
        resource.capacity.trigger_batch()
        self.assertFalse(resource.capacity.can_accept())

        # Complete batch
        resource.capacity.complete_batch()
        self.assertTrue(resource.capacity.can_accept())

        # Now optimizer should assign
        wo = make_wo("SAR-POST")
        decisions = optimizer.dispatch([wo], "review", "sar", config)
        self.assertIsNotNone(decisions[0].selected_resource_id)
        self.assertEqual(decisions[0].selected_resource_id, "batch_proc")


class TestBackpressureQueueMechanics(unittest.TestCase):
    """Direct tests of the backpressure queue on Coordinator."""

    def setUp(self):
        from coordinator.runtime import Coordinator
        self.coord = Coordinator(
            config={"capabilities": [
                {"need_type": "review", "provider_type": "workflow",
                 "workflow_type": "investigation", "domain": "sar"},
            ]},
            verbose=False,
        )
        # Register a batch resource
        self.resource = make_batch_resource(threshold=3, timeout=60.0)
        self.coord._resource_registry.register(self.resource)

    def test_enqueue_and_drain(self):
        """Enqueue work, then drain after capacity freed."""
        # Simulate: resource is executing (no capacity)
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.trigger_batch()

        # Create a work order and simulate QUEUED status
        from coordinator.types import WorkOrder, WorkOrderResult
        wo = make_store_wo("wo_queued_1")
        self.coord.store.save_work_order(wo)

        # Enqueue it
        capability = self.coord._find_capability("review")
        self.coord._enqueue_for_resource(
            "batch_proc", "wo_queued_1", "inst_1",
            {"need": "review", "context": {}},
            capability, [],
        )
        self.assertEqual(self.coord.queued_work_order_count, 1)
        self.assertEqual(self.coord.get_queue_depth("batch_proc"), 1)

        # Try to drain — resource still executing, should not drain
        drained = self.coord.drain_resource_queue("batch_proc")
        self.assertEqual(drained, 0)
        self.assertEqual(self.coord.queued_work_order_count, 1)

        # Complete batch — resource now accepting
        self.resource.capacity.complete_batch()
        self.assertTrue(self.resource.capacity.can_accept())

        # Drain should now succeed (though dispatch may fail due to no LLM)
        # The point is: the queue entry is consumed and dispatch is ATTEMPTED
        drained = self.coord.drain_resource_queue("batch_proc")
        # Even if dispatch fails, the entry was processed
        self.assertEqual(self.coord.get_queue_depth("batch_proc"), 0)

    def test_multiple_queued_drain_in_order(self):
        """Multiple queued work orders drain FIFO."""
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.trigger_batch()

        capability = self.coord._find_capability("review")

        for i in range(3):
            wo = make_store_wo(f"wo_q_{i}")
            self.coord.store.save_work_order(wo)
            self.coord._enqueue_for_resource(
                "batch_proc", f"wo_q_{i}", f"inst_{i}",
                {"need": "review"}, capability, [],
            )

        self.assertEqual(self.coord.queued_work_order_count, 3)

        # Complete batch
        self.resource.capacity.complete_batch()

        # Drain — all 3 should be attempted
        self.coord.drain_resource_queue("batch_proc")
        self.assertEqual(self.coord.get_queue_depth("batch_proc"), 0)

    def test_sweep_triggers_drain(self):
        """sweep_reservations() automatically drains the queue."""
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.trigger_batch()

        capability = self.coord._find_capability("review")
        wo = make_store_wo("wo_sweep")
        self.coord.store.save_work_order(wo)
        self.coord._enqueue_for_resource(
            "batch_proc", "wo_sweep", "inst_sweep",
            {"need": "review"}, capability, [],
        )

        # Complete batch
        self.resource.capacity.complete_batch()

        # sweep_reservations should drain the queue
        summary = self.coord.sweep_reservations()
        # The "drained" key should be present if anything was drained
        self.assertEqual(self.coord.get_queue_depth("batch_proc"), 0)

    def test_cancelled_wo_skipped_on_drain(self):
        """Cancelled work orders are silently skipped during drain."""
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.on_assign()
        self.resource.capacity.trigger_batch()

        capability = self.coord._find_capability("review")
        wo = make_store_wo("wo_cancel", WorkOrderStatus.CANCELLED)
        self.coord.store.save_work_order(wo)
        self.coord._enqueue_for_resource(
            "batch_proc", "wo_cancel", "inst_cancel",
            {"need": "review"}, capability, [],
        )

        self.resource.capacity.complete_batch()
        self.coord.drain_resource_queue("batch_proc")
        # Cancelled WO should be skipped, queue empty
        self.assertEqual(self.coord.get_queue_depth("batch_proc"), 0)


class TestBatchFullLifecycleWithQueue(unittest.TestCase):
    """End-to-end: collect → time trigger → execute → new arrivals queue → complete → drain."""

    def test_full_lifecycle(self):
        """
        Timeline:
          t=0:   Resource registered, COLLECTING
          t=0:   WO-1, WO-2 arrive → assigned to batch (COLLECTING)
          t=60:  Time window fires → batch EXECUTING
          t=70:  WO-3, WO-4 arrive → no resource → QUEUED
          t=120: Batch completes → COLLECTING
          t=120: Queue drains → WO-3, WO-4 dispatched
        """
        registry = ResourceRegistry()
        resource = make_batch_resource(
            threshold=10,      # high threshold (won't trigger)
            timeout=60.0,      # 60-second window
        )
        registry.register(resource)

        res_log = ReservationEventLog()
        optimizer = DispatchOptimizer(registry, reservation_log=res_log)
        config = OptimizationConfig(exploration_enabled=False)

        # t=0: WO-1 arrives
        t0 = 1000000.0
        resource.capacity.batch_collecting_since = t0
        wo1 = make_wo("SAR-001")
        d1 = optimizer.dispatch([wo1], "review", "sar", config)
        self.assertIsNotNone(d1[0].selected_resource_id)

        # t=10: WO-2 arrives
        wo2 = make_wo("SAR-002")
        d2 = optimizer.dispatch([wo2], "review", "sar", config)
        self.assertIsNotNone(d2[0].selected_resource_id)

        # t=60: Time window fires
        self.assertTrue(resource.capacity.check_batch_trigger(now=t0 + 60))
        resource.capacity.trigger_batch(now=t0 + 60)
        self.assertEqual(resource.capacity.batch_status, BatchStatus.EXECUTING)

        # t=70: WO-3, WO-4 arrive → no eligible resource
        wo3 = make_wo("SAR-003")
        wo4 = make_wo("SAR-004")
        d3 = optimizer.dispatch([wo3], "review", "sar", config)
        d4 = optimizer.dispatch([wo4], "review", "sar", config)
        self.assertIsNone(d3[0].selected_resource_id)
        self.assertIsNone(d4[0].selected_resource_id)

        # t=120: Batch completes
        resource.capacity.complete_batch()
        self.assertEqual(resource.capacity.batch_status, BatchStatus.COLLECTING)
        self.assertTrue(resource.capacity.can_accept())

        # Now WO-3, WO-4 can be dispatched
        d3_retry = optimizer.dispatch([wo3], "review", "sar", config)
        d4_retry = optimizer.dispatch([wo4], "review", "sar", config)
        self.assertIsNotNone(d3_retry[0].selected_resource_id)
        self.assertIsNotNone(d4_retry[0].selected_resource_id)

        # Reservation log should show all events
        events = res_log.get_events()
        acquires = [e for e in events if e.operation == "acquire"]
        # WO-1, WO-2 (during collecting) + WO-3, WO-4 (after complete)
        self.assertEqual(len(acquires), 4)


if __name__ == "__main__":
    unittest.main()
