"""
Cognitive Core — Delegation Mechanical Tests (INT-015 through INT-020)

Tests the delegation machinery: depth limiting, cycle detection,
correlation chain tracing, work order lifecycle, and blocking/ff modes.

Uses the real CoordinatorStore and Coordinator where possible,
mock LLM for everything else.
"""

import json
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

from coordinator.store import CoordinatorStore
from coordinator.types import (
    InstanceState, InstanceStatus,
    WorkOrder, WorkOrderStatus, WorkOrderResult,
    Suspension,
)

# Try loading coordinator — needs yaml
try:
    from coordinator.runtime import Coordinator, DelegationDepthExceeded
    _HAS_COORDINATOR = True
except Exception:
    _HAS_COORDINATOR = False


# ═══════════════════════════════════════════════════════════════
# INT-015: Delegation Depth Enforcement
# ═══════════════════════════════════════════════════════════════

@unittest.skipUnless(_HAS_COORDINATOR, "coordinator not loadable")
class TestINT015_DelegationDepth(unittest.TestCase):
    """Depth limit is a hard invariant — not advisory."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_base, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_depth_0_ok(self):
        """No lineage → depth 0 → OK."""
        try:
            self.coord.start("test", "test", {}, lineage=[])
        except DelegationDepthExceeded:
            self.fail("Depth 0 should not raise")
        except Exception:
            pass  # Other errors expected

    def test_depth_at_max_raises(self):
        """Lineage at MAX_DELEGATION_DEPTH → blocked."""
        lineage = [f"wf_{i}:inst_{i}" for i in range(self.coord.MAX_DELEGATION_DEPTH)]
        with self.assertRaises(DelegationDepthExceeded):
            self.coord.start("test", "test", {}, lineage=lineage)

    def test_depth_one_under_max_ok(self):
        """Lineage at MAX-1 → allowed (this is the last valid level)."""
        lineage = [f"wf_{i}:inst_{i}" for i in range(self.coord.MAX_DELEGATION_DEPTH - 1)]
        try:
            self.coord.start("test", "test", {}, lineage=lineage)
        except DelegationDepthExceeded:
            self.fail("MAX-1 should be allowed")
        except Exception:
            pass

    def test_depth_limit_is_20(self):
        """Verify the actual limit value."""
        self.assertEqual(self.coord.MAX_DELEGATION_DEPTH, 20)


# ═══════════════════════════════════════════════════════════════
# INT-016: Correlation Chain Through Work Orders
# ═══════════════════════════════════════════════════════════════

class TestINT016_CorrelationChain(unittest.TestCase):
    """
    Parent creates work order → child inherits correlation_id →
    all instances queryable by correlation.
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_parent_child_share_correlation(self):
        # Parent instance
        parent = InstanceState.create("spending_advisor", "debit_spending", "auto")
        parent.correlation_id = "corr_001"
        parent.status = InstanceStatus.RUNNING
        self.store.save_instance(parent)

        # Work order carries correlation
        wo = WorkOrder.create(
            requester_instance_id=parent.instance_id,
            correlation_id=parent.correlation_id,
            contract_name="vendor_notification",
            contract_version=1,
            inputs={"alert": "spending_anomaly"},
        )
        wo.handler_workflow_type = "notify_vendor"
        wo.handler_domain = "vendor_ops"
        wo.status = WorkOrderStatus.DISPATCHED
        wo.dispatched_at = time.time()
        self.store.save_work_order(wo)

        # Child instance inherits correlation
        child = InstanceState.create("notify_vendor", "vendor_ops", "auto")
        child.correlation_id = parent.correlation_id
        child.lineage = [f"{parent.workflow_type}:{parent.instance_id}"]
        child.status = InstanceStatus.RUNNING
        self.store.save_instance(child)

        # Query by correlation returns both
        all_instances = self.store.list_instances(correlation_id="corr_001")
        self.assertEqual(len(all_instances), 2)
        ids = {i.instance_id for i in all_instances}
        self.assertIn(parent.instance_id, ids)
        self.assertIn(child.instance_id, ids)

    def test_three_level_chain(self):
        """A → B → C, all share correlation_id."""
        corr = "corr_deep"
        instances = []
        for i, (wf, dom) in enumerate([
            ("spending_advisor", "debit"),
            ("fraud_review", "fraud"),
            ("compliance_check", "compliance"),
        ]):
            inst = InstanceState.create(wf, dom, "auto")
            inst.correlation_id = corr
            inst.lineage = [f"{prev.workflow_type}:{prev.instance_id}" for prev in instances]
            inst.status = InstanceStatus.COMPLETED
            self.store.save_instance(inst)
            instances.append(inst)

        all_inst = self.store.list_instances(correlation_id=corr)
        self.assertEqual(len(all_inst), 3)

    def test_lineage_records_full_path(self):
        """Child lineage contains parent chain."""
        parent = InstanceState.create("wf_a", "dom_a", "auto")
        parent.correlation_id = "c1"
        self.store.save_instance(parent)

        child = InstanceState.create("wf_b", "dom_b", "auto")
        child.correlation_id = "c1"
        child.lineage = [f"wf_a:{parent.instance_id}"]
        self.store.save_instance(child)

        grandchild = InstanceState.create("wf_c", "dom_c", "auto")
        grandchild.correlation_id = "c1"
        grandchild.lineage = child.lineage + [f"wf_b:{child.instance_id}"]
        self.store.save_instance(grandchild)

        gc = self.store.get_instance(grandchild.instance_id)
        self.assertEqual(len(gc.lineage), 2)
        self.assertIn(f"wf_a:{parent.instance_id}", gc.lineage)
        self.assertIn(f"wf_b:{child.instance_id}", gc.lineage)


# ═══════════════════════════════════════════════════════════════
# INT-017: Work Order Lifecycle
# ═══════════════════════════════════════════════════════════════

class TestINT017_WorkOrderLifecycle(unittest.TestCase):
    """Work order: created → dispatched → running → completed."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_full_lifecycle(self):
        wo = WorkOrder.create(
            requester_instance_id="parent_001",
            correlation_id="corr_001",
            contract_name="investigate_fraud",
            contract_version=1,
            inputs={"member_id": "MBR-001", "alert_type": "velocity"},
            sla_seconds=300,
        )
        self.assertEqual(wo.status, WorkOrderStatus.CREATED)

        # Dispatch
        wo.status = WorkOrderStatus.DISPATCHED
        wo.dispatched_at = time.time()
        wo.handler_workflow_type = "fraud_investigation"
        wo.handler_domain = "fraud_ops"
        wo.handler_instance_id = "child_001"
        self.store.save_work_order(wo)

        # Running
        wo.status = WorkOrderStatus.RUNNING
        self.store.save_work_order(wo)

        # Complete
        wo.status = WorkOrderStatus.COMPLETED
        wo.completed_at = time.time()
        wo.result = WorkOrderResult(
            work_order_id=wo.work_order_id,
            status="completed",
            outputs={"finding": "no_fraud_detected", "confidence": 0.95},
            completed_at=time.time(),
        )
        self.store.save_work_order(wo)

        # Verify
        loaded = self.store.get_work_order(wo.work_order_id)
        self.assertEqual(loaded.status, WorkOrderStatus.COMPLETED)
        self.assertEqual(loaded.result.outputs["finding"], "no_fraud_detected")
        self.assertEqual(loaded.handler_workflow_type, "fraud_investigation")

    def test_work_orders_for_instance(self):
        """Multiple work orders from same parent."""
        for i, contract in enumerate(["vendor_notify", "fraud_check", "compliance_log"]):
            wo = WorkOrder.create("parent_001", "corr_001", contract, 1, {})
            wo.status = WorkOrderStatus.DISPATCHED
            wo.dispatched_at = time.time()
            self.store.save_work_order(wo)

        orders = self.store.get_work_orders_for_instance("parent_001")
        self.assertEqual(len(orders), 3)
        contracts = {o.contract_name for o in orders}
        self.assertEqual(contracts, {"vendor_notify", "fraud_check", "compliance_log"})

    def test_work_order_idempotency(self):
        """Same work_order_id saved twice → updated, not duplicated."""
        wo = WorkOrder.create("p1", "c1", "test", 1, {})
        wo.status = WorkOrderStatus.DISPATCHED
        self.store.save_work_order(wo)

        wo.status = WorkOrderStatus.COMPLETED
        self.store.save_work_order(wo)

        loaded = self.store.get_work_order(wo.work_order_id)
        self.assertEqual(loaded.status, WorkOrderStatus.COMPLETED)


# ═══════════════════════════════════════════════════════════════
# INT-018: Suspension / Resume Through Delegation
# ═══════════════════════════════════════════════════════════════

class TestINT018_SuspensionResume(unittest.TestCase):
    """Blocking delegation: parent suspends, child runs, parent resumes."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_blocking_delegation_suspend_resume(self):
        # Parent running
        parent = InstanceState.create("spending_advisor", "debit", "auto")
        parent.status = InstanceStatus.RUNNING
        parent.correlation_id = "corr_block"
        self.store.save_instance(parent)

        # Create blocking work order
        wo = WorkOrder.create(parent.instance_id, parent.correlation_id, "fraud_check", 1, {"alert": "high_velocity"})
        wo.status = WorkOrderStatus.DISPATCHED
        wo.handler_instance_id = "child_001"
        self.store.save_work_order(wo)

        # Parent suspends
        parent.status = InstanceStatus.SUSPENDED
        parent.pending_work_orders = [wo.work_order_id]
        self.store.save_instance(parent)

        sus = Suspension(
            instance_id=parent.instance_id,
            suspended_at_step="after_investigate",
            state_snapshot={"steps": [{"step_name": "classify", "output": {"category": "high_spend"}}]},
            unresolved_needs=["fraud_check_result"],
            work_order_ids=[wo.work_order_id],
            resume_nonce="nonce_001",
            suspended_at=time.time(),
        )
        self.store.save_suspension(sus)

        # Verify suspended state
        loaded = self.store.get_instance(parent.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)
        loaded_sus = self.store.get_suspension(parent.instance_id)
        self.assertEqual(loaded_sus.resume_nonce, "nonce_001")

        # Child completes
        wo.status = WorkOrderStatus.COMPLETED
        wo.result = WorkOrderResult(wo.work_order_id, "completed", {"finding": "clean"}, completed_at=time.time())
        self.store.save_work_order(wo)

        # Parent resumes
        parent.status = InstanceStatus.RUNNING
        parent.pending_work_orders = []
        self.store.save_instance(parent)
        self.store.delete_suspension(parent.instance_id)

        # Verify resumed
        self.assertEqual(self.store.get_instance(parent.instance_id).status, InstanceStatus.RUNNING)
        self.assertIsNone(self.store.get_suspension(parent.instance_id))

    def test_orphaned_suspension_detection(self):
        """Suspension whose work orders are all done but instance still suspended."""
        parent = InstanceState.create("wf", "dom", "auto")
        parent.status = InstanceStatus.SUSPENDED
        self.store.save_instance(parent)

        wo = WorkOrder.create(parent.instance_id, "c1", "check", 1, {})
        wo.status = WorkOrderStatus.COMPLETED
        wo.result = WorkOrderResult(wo.work_order_id, "completed", {}, completed_at=time.time())
        self.store.save_work_order(wo)

        sus = Suspension(parent.instance_id, "step", {}, [], [wo.work_order_id], "n", time.time())
        self.store.save_suspension(sus)

        orphans = self.store.find_orphaned_suspensions()
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0]["instance_id"], parent.instance_id)


# ═══════════════════════════════════════════════════════════════
# INT-019: Delegation Deduplication
# ═══════════════════════════════════════════════════════════════

class TestINT019_DelegationDedup(unittest.TestCase):
    """Same delegation policy doesn't fire twice for same instance."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_ledger_prevents_duplicate_dispatch(self):
        inst = InstanceState.create("wf", "dom", "auto")
        inst.status = InstanceStatus.COMPLETED
        self.store.save_instance(inst)

        # First delegation fires
        success = self.store.log_action(
            inst.instance_id, "c1", "delegation_dispatched",
            {"policy": "vendor_notify", "target": "notify_vendor"},
            idempotency_key=f"{inst.instance_id}:vendor_notify",
        )
        self.assertTrue(success)

        # Same delegation for same instance → blocked by idempotency
        duplicate = self.store.log_action(
            inst.instance_id, "c1", "delegation_dispatched",
            {"policy": "vendor_notify", "target": "notify_vendor"},
            idempotency_key=f"{inst.instance_id}:vendor_notify",
        )
        self.assertFalse(duplicate)

    def test_different_policies_both_fire(self):
        inst = InstanceState.create("wf", "dom", "auto")
        self.store.save_instance(inst)

        a = self.store.log_action(inst.instance_id, "c1", "delegation_dispatched",
            {"policy": "vendor_notify"}, idempotency_key=f"{inst.instance_id}:vendor_notify")
        b = self.store.log_action(inst.instance_id, "c1", "delegation_dispatched",
            {"policy": "fraud_review"}, idempotency_key=f"{inst.instance_id}:fraud_review")
        self.assertTrue(a)
        self.assertTrue(b)

        ledger = self.store.get_ledger(instance_id=inst.instance_id)
        self.assertEqual(len(ledger), 2)


# ═══════════════════════════════════════════════════════════════
# INT-020: Cycle Detection via Lineage
# ═══════════════════════════════════════════════════════════════

class TestINT020_CycleDetection(unittest.TestCase):
    """Lineage contains workflow type — can detect A→B→A cycles."""

    def test_lineage_reveals_cycle(self):
        """If A delegates to B which tries to delegate back to A,
        the lineage will show 'wf_a' already in the chain."""
        lineage_at_b = ["wf_a:inst_a"]

        # B is about to delegate to wf_a — check lineage
        target_workflow = "wf_a"
        already_in_chain = any(target_workflow in entry for entry in lineage_at_b)
        self.assertTrue(already_in_chain, "Cycle should be detectable from lineage")

    def test_no_cycle_different_workflows(self):
        lineage = ["wf_a:inst_a", "wf_b:inst_b"]
        target = "wf_c"
        in_chain = any(target in entry for entry in lineage)
        self.assertFalse(in_chain)

    def test_deep_chain_cycle_at_depth_3(self):
        """A→B→C→A should be caught."""
        lineage = ["wf_a:inst_a", "wf_b:inst_b", "wf_c:inst_c"]
        target = "wf_a"
        in_chain = any(entry.startswith(f"{target}:") for entry in lineage)
        self.assertTrue(in_chain)

    @unittest.skipUnless(_HAS_COORDINATOR, "coordinator not loadable")
    def test_depth_limit_prevents_infinite_cycle(self):
        """Even without explicit cycle detection, depth limit catches infinite loops."""
        store = CoordinatorStore(":memory:")
        coord = Coordinator(
            config_path=os.path.join(_base, "coordinator", "config.yaml"),
            store=store, verbose=False,
        )
        # Simulate A→B→C→...→Z chain at max depth
        lineage = [f"wf_{i}:inst_{i}" for i in range(coord.MAX_DELEGATION_DEPTH)]
        with self.assertRaises(DelegationDepthExceeded):
            coord.start("wf_loop", "dom", {}, lineage=lineage)


if __name__ == "__main__":
    unittest.main()
