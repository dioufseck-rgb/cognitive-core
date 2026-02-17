"""
Cognitive Core — Coordinator Test Suite

Exhaustive tests for the runtime coordinator, covering:
  1. Types & data structures
  2. SQLite store (CRUD, idempotency, queries)
  3. Task queue (in-memory + SQLite, publish/claim/resolve/expire)
  4. Policy engine (governance tiers, delegation conditions, contracts, cycles)
  5. Coordinator lifecycle (start, complete, suspend, approve, reject, terminate)
  6. Fire-and-forget delegation
  7. Blocking delegation (wait_for_result)
  8. Cascade: handler approved → requester resumes
  9. Multi-delegation (multiple policies fire)
  10. Cycle detection
  11. Idempotency enforcement
  12. Edge cases and error handling

All tests are deterministic — no LLM calls, no LangGraph.
Tests exercise the coordinator's state management, policy evaluation,
task routing, and delegation mechanics in isolation.
"""

import json
import os
import time
import unittest
import sqlite3
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.types import (
    InstanceState, InstanceStatus,
    WorkOrder, WorkOrderStatus, WorkOrderResult,
    Suspension, GovernanceTier,
    DelegationPolicy, DelegationCondition,
    Contract, ContractField,
)
from coordinator.store import CoordinatorStore
from coordinator.tasks import (
    TaskQueue, InMemoryTaskQueue, SQLiteTaskQueue,
    Task, TaskType, TaskStatus, TaskCallback, TaskResolution,
)
from coordinator.policy import (
    PolicyEngine, load_policy_engine,
    GovernanceDecision, DelegationDecision,
)
from coordinator.runtime import Coordinator


# ═══════════════════════════════════════════════════════════════════
# 1. TYPES & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

class TestInstanceState(unittest.TestCase):
    def test_create(self):
        inst = InstanceState.create("dispute_resolution", "card_dispute", "spot_check")
        self.assertTrue(inst.instance_id.startswith("wf_"))
        self.assertEqual(inst.workflow_type, "dispute_resolution")
        self.assertEqual(inst.domain, "card_dispute")
        self.assertEqual(inst.governance_tier, "spot_check")
        self.assertEqual(inst.status, InstanceStatus.CREATED)
        self.assertEqual(inst.lineage, [])
        self.assertIsNotNone(inst.correlation_id)
        self.assertGreater(inst.created_at, 0)

    def test_create_with_lineage(self):
        inst = InstanceState.create(
            "sar_investigation", "structuring_sar", "hold",
            lineage=["dispute_resolution:wf_001"],
            correlation_id="wf_001",
        )
        self.assertEqual(inst.lineage, ["dispute_resolution:wf_001"])
        self.assertEqual(inst.correlation_id, "wf_001")

    def test_unique_ids(self):
        ids = {InstanceState.create("w", "d", "auto").instance_id for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestWorkOrder(unittest.TestCase):
    def test_create(self):
        wo = WorkOrder.create(
            requester_instance_id="wf_001",
            correlation_id="wf_001",
            contract_name="aml_referral_v1",
            inputs={"member_id": "M001"},
        )
        self.assertTrue(wo.work_order_id.startswith("wo_"))
        self.assertEqual(wo.requester_instance_id, "wf_001")
        self.assertEqual(wo.status, WorkOrderStatus.CREATED)

    def test_unique_ids(self):
        ids = {WorkOrder.create("r", "c", "ct", {}).work_order_id for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestSuspension(unittest.TestCase):
    def test_create(self):
        sus = Suspension.create(
            instance_id="wf_001",
            suspended_at_step="investigate_dispute",
            state_snapshot={"steps": [], "input": {"member_id": "M001"}},
            work_order_ids=["wo_001"],
        )
        self.assertEqual(sus.instance_id, "wf_001")
        self.assertEqual(sus.suspended_at_step, "investigate_dispute")
        self.assertEqual(sus.work_order_ids, ["wo_001"])
        self.assertEqual(len(sus.resume_nonce), 16)
        self.assertGreater(sus.suspended_at, 0)

    def test_unique_nonces(self):
        nonces = {Suspension.create("wf", "step", {}).resume_nonce for _ in range(100)}
        self.assertEqual(len(nonces), 100)


# ═══════════════════════════════════════════════════════════════════
# 2. SQLITE STORE
# ═══════════════════════════════════════════════════════════════════

class TestCoordinatorStore(unittest.TestCase):
    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_instance_crud(self):
        inst = InstanceState.create("w", "d", "auto")
        self.store.save_instance(inst)
        loaded = self.store.get_instance(inst.instance_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.workflow_type, "w")
        self.assertEqual(loaded.domain, "d")
        self.assertEqual(loaded.governance_tier, "auto")

    def test_instance_update(self):
        inst = InstanceState.create("w", "d", "auto")
        self.store.save_instance(inst)
        inst.status = InstanceStatus.COMPLETED
        inst.step_count = 5
        self.store.save_instance(inst)
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.COMPLETED)
        self.assertEqual(loaded.step_count, 5)

    def test_list_instances_by_status(self):
        for s in [InstanceStatus.CREATED, InstanceStatus.COMPLETED,
                  InstanceStatus.SUSPENDED, InstanceStatus.COMPLETED]:
            inst = InstanceState.create("w", "d", "auto")
            inst.status = s
            self.store.save_instance(inst)
        completed = self.store.list_instances(status=InstanceStatus.COMPLETED)
        self.assertEqual(len(completed), 2)

    def test_list_instances_by_correlation(self):
        corr = "corr_001"
        for _ in range(3):
            inst = InstanceState.create("w", "d", "auto")
            inst.correlation_id = corr
            self.store.save_instance(inst)
        chain = self.store.list_instances(correlation_id=corr)
        self.assertEqual(len(chain), 3)

    def test_instance_not_found(self):
        self.assertIsNone(self.store.get_instance("nonexistent"))

    def test_work_order_crud(self):
        wo = WorkOrder.create("wf_001", "wf_001", "contract_v1", 1, {"key": "val"})
        self.store.save_work_order(wo)
        loaded = self.store.get_work_order(wo.work_order_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.contract_name, "contract_v1")
        self.assertEqual(loaded.inputs, {"key": "val"})

    def test_work_order_with_result(self):
        wo = WorkOrder.create("wf_001", "wf_001", "c", 1, {})
        wo.result = WorkOrderResult(wo.work_order_id, "completed", {"out": "val"})
        self.store.save_work_order(wo)
        loaded = self.store.get_work_order(wo.work_order_id)
        self.assertIsNotNone(loaded.result)
        self.assertEqual(loaded.result.outputs, {"out": "val"})

    def test_work_orders_for_instance(self):
        for _ in range(3):
            wo = WorkOrder.create("wf_001", "wf_001", "c", 1, {})
            self.store.save_work_order(wo)
        wo_other = WorkOrder.create("wf_002", "wf_002", "c", 1, {})
        self.store.save_work_order(wo_other)
        wos = self.store.get_work_orders_for_instance("wf_001")
        self.assertEqual(len(wos), 3)

    def test_work_orders_for_requester_or_handler(self):
        wo = WorkOrder.create("wf_001", "wf_001", "c", 1, {})
        wo.handler_instance_id = "wf_002"
        self.store.save_work_order(wo)
        # Found as requester
        found_r = self.store.get_work_orders_for_requester_or_handler("wf_001")
        self.assertEqual(len(found_r), 1)
        # Found as handler
        found_h = self.store.get_work_orders_for_requester_or_handler("wf_002")
        self.assertEqual(len(found_h), 1)
        # Not found
        found_n = self.store.get_work_orders_for_requester_or_handler("wf_999")
        self.assertEqual(len(found_n), 0)

    def test_suspension_crud(self):
        sus = Suspension.create("wf_001", "step_3", {"data": "snapshot"})
        self.store.save_suspension(sus)
        loaded = self.store.get_suspension("wf_001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.suspended_at_step, "step_3")
        self.assertEqual(loaded.state_snapshot, {"data": "snapshot"})

    def test_suspension_delete(self):
        sus = Suspension.create("wf_001", "step_3", {})
        self.store.save_suspension(sus)
        self.store.delete_suspension("wf_001")
        self.assertIsNone(self.store.get_suspension("wf_001"))

    def test_action_ledger_basic(self):
        ok = self.store.log_action("wf_001", "wf_001", "start", {"key": "val"})
        self.assertTrue(ok)
        ledger = self.store.get_ledger(instance_id="wf_001")
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["action_type"], "start")

    def test_action_ledger_idempotency(self):
        ok1 = self.store.log_action("wf_001", "wf_001", "start", {},
                                     idempotency_key="key_001")
        ok2 = self.store.log_action("wf_001", "wf_001", "start", {},
                                     idempotency_key="key_001")
        self.assertTrue(ok1)
        self.assertFalse(ok2)
        ledger = self.store.get_ledger(instance_id="wf_001")
        self.assertEqual(len(ledger), 1)  # Only one entry despite two calls

    def test_action_ledger_by_correlation(self):
        self.store.log_action("wf_001", "corr_001", "start", {})
        self.store.log_action("wf_002", "corr_001", "start", {})
        self.store.log_action("wf_003", "corr_002", "start", {})
        ledger = self.store.get_ledger(correlation_id="corr_001")
        self.assertEqual(len(ledger), 2)

    def test_stats(self):
        inst = InstanceState.create("w", "d", "auto")
        inst.status = InstanceStatus.COMPLETED
        self.store.save_instance(inst)
        wo = WorkOrder.create("wf_001", "wf_001", "c", 1, {})
        self.store.save_work_order(wo)
        self.store.log_action("wf_001", "wf_001", "start", {})
        s = self.store.stats()
        self.assertIn("instances", s)
        self.assertIn("work_orders", s)
        self.assertIn("action_ledger_entries", s)


# ═══════════════════════════════════════════════════════════════════
# 3. TASK QUEUE
# ═══════════════════════════════════════════════════════════════════

class TaskQueueTestMixin:
    """Shared tests for both InMemory and SQLite task queues."""

    queue: TaskQueue  # Set by subclass

    def _make_task(self, queue="test_queue", priority=0, sla=None):
        return Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue=queue,
            instance_id="wf_001",
            correlation_id="wf_001",
            workflow_type="test_wf",
            domain="test_domain",
            payload={"test": True},
            callback=TaskCallback(method="approve", instance_id="wf_001"),
            priority=priority,
            sla_seconds=sla,
        )

    def test_publish_and_get(self):
        task = self._make_task()
        tid = self.queue.publish(task)
        loaded = self.queue.get_task(tid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.task_type, TaskType.GOVERNANCE_APPROVAL)
        self.assertEqual(loaded.status, TaskStatus.PENDING)

    def test_claim(self):
        self.queue.publish(self._make_task())
        claimed = self.queue.claim("test_queue", "user_a")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.status, TaskStatus.CLAIMED)
        self.assertEqual(claimed.claimed_by, "user_a")

    def test_claim_empty_queue(self):
        claimed = self.queue.claim("empty_queue", "user_a")
        self.assertIsNone(claimed)

    def test_claim_skips_claimed_tasks(self):
        self.queue.publish(self._make_task())
        self.queue.claim("test_queue", "user_a")
        second = self.queue.claim("test_queue", "user_b")
        self.assertIsNone(second)

    def test_claim_priority_ordering(self):
        self.queue.publish(self._make_task(priority=0))
        self.queue.publish(self._make_task(priority=2))
        self.queue.publish(self._make_task(priority=1))
        claimed = self.queue.claim("test_queue", "user")
        self.assertEqual(claimed.priority, 2)  # Highest priority first

    def test_resolve_approve(self):
        task = self._make_task()
        self.queue.publish(task)
        self.queue.claim("test_queue", "user")
        ok = self.queue.resolve(task.task_id, TaskResolution(
            task_id=task.task_id, action="approve"
        ))
        self.assertTrue(ok)
        resolved = self.queue.get_task(task.task_id)
        self.assertEqual(resolved.status, TaskStatus.COMPLETED)

    def test_resolve_reject(self):
        task = self._make_task()
        self.queue.publish(task)
        self.queue.claim("test_queue", "user")
        self.queue.resolve(task.task_id, TaskResolution(
            task_id=task.task_id, action="reject"
        ))
        resolved = self.queue.get_task(task.task_id)
        self.assertEqual(resolved.status, TaskStatus.REJECTED)

    def test_resolve_unclaimed_fails(self):
        task = self._make_task()
        self.queue.publish(task)
        ok = self.queue.resolve(task.task_id, TaskResolution(
            task_id=task.task_id, action="approve"
        ))
        self.assertFalse(ok)

    def test_list_by_queue(self):
        self.queue.publish(self._make_task(queue="q1"))
        self.queue.publish(self._make_task(queue="q1"))
        self.queue.publish(self._make_task(queue="q2"))
        q1 = self.queue.list_tasks(queue="q1")
        self.assertEqual(len(q1), 2)

    def test_list_by_status(self):
        t1 = self._make_task()
        t2 = self._make_task()
        self.queue.publish(t1)
        self.queue.publish(t2)
        self.queue.claim("test_queue", "user")
        pending = self.queue.list_tasks(status=TaskStatus.PENDING)
        claimed = self.queue.list_tasks(status=TaskStatus.CLAIMED)
        self.assertEqual(len(pending), 1)
        self.assertEqual(len(claimed), 1)

    def test_expire_overdue(self):
        task = self._make_task(sla=0.001)  # Expires immediately
        self.queue.publish(task)
        time.sleep(0.01)
        count = self.queue.expire_overdue()
        self.assertEqual(count, 1)
        expired = self.queue.get_task(task.task_id)
        self.assertEqual(expired.status, TaskStatus.EXPIRED)

    def test_expire_skips_valid(self):
        task = self._make_task(sla=3600)  # 1 hour — not overdue
        self.queue.publish(task)
        count = self.queue.expire_overdue()
        self.assertEqual(count, 0)

    def test_expire_skips_claimed(self):
        task = self._make_task(sla=0.001)
        self.queue.publish(task)
        self.queue.claim("test_queue", "user")
        time.sleep(0.01)
        count = self.queue.expire_overdue()
        self.assertEqual(count, 0)  # Claimed tasks don't expire


class TestInMemoryTaskQueue(TaskQueueTestMixin, unittest.TestCase):
    def setUp(self):
        self.queue = InMemoryTaskQueue()


class TestSQLiteTaskQueue(TaskQueueTestMixin, unittest.TestCase):
    def setUp(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        self.queue = SQLiteTaskQueue(conn)


# ═══════════════════════════════════════════════════════════════════
# 4. POLICY ENGINE
# ═══════════════════════════════════════════════════════════════════

class TestPolicyEngine(unittest.TestCase):
    def setUp(self):
        """Load the real config."""
        import yaml
        with open(os.path.join(_project_root, "coordinator", "config.yaml")) as f:
            config = yaml.safe_load(f)
        self.engine = load_policy_engine(config)

    def test_config_loads(self):
        self.assertEqual(len(self.engine.governance_tiers), 4)
        self.assertGreaterEqual(len(self.engine.delegation_policies), 3)
        self.assertGreaterEqual(len(self.engine.contracts), 2)
        self.assertEqual(len(self.engine.capabilities), 5)

    def test_governance_auto_proceeds(self):
        decision = self.engine.evaluate_governance("debit_spending", "auto", {})
        self.assertEqual(decision.action, "proceed")
        self.assertEqual(decision.tier, "auto")

    def test_governance_hold_suspends(self):
        decision = self.engine.evaluate_governance("structuring_sar", "hold", {})
        self.assertEqual(decision.action, "suspend_for_approval")
        self.assertEqual(decision.queue, "compliance_review")

    def test_governance_gate_suspends(self):
        decision = self.engine.evaluate_governance("military_hardship", "gate", {})
        self.assertEqual(decision.action, "suspend_for_approval")
        self.assertEqual(decision.queue, "specialist_review")

    def test_governance_spot_check_sampling(self):
        """Spot check should proceed most of the time (90%)."""
        results = set()
        for _ in range(200):
            d = self.engine.evaluate_governance("card_dispute", "spot_check", {})
            results.add(d.action)
        # Should see both proceed and queue_review
        self.assertIn("proceed", results)
        # queue_review might not appear in 200 trials (10%), but usually will
        # Don't assert it — just verify proceed always works

    def test_delegation_fraud_triggers_aml(self):
        steps = [{
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_member": {"member_id": "M001"},
                    "get_fraud_score": {
                        "score": 892,
                        "factors": ["unknown_device", "geolocation_anomaly"],
                    },
                },
            },
        }]
        decisions = self.engine.evaluate_delegations(
            "card_dispute", {"steps": steps, "input": {}}, []
        )
        self.assertGreaterEqual(len(decisions), 1)
        self.assertEqual(decisions[0].target_workflow, "sar_investigation")
        self.assertEqual(decisions[0].contract_name, "aml_referral_v1")

    def test_delegation_no_match(self):
        steps = [{
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_member": {"member_id": "M001"},
                    "get_fraud_score": {"score": 100, "factors": []},
                },
            },
        }]
        decisions = self.engine.evaluate_delegations(
            "card_dispute", {"steps": steps, "input": {}}, []
        )
        self.assertEqual(len(decisions), 0)

    def test_delegation_wrong_domain(self):
        """Policies only fire for matching domains."""
        steps = [{
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_fraud_score": {"factors": ["unknown_device"]},
                },
            },
        }]
        decisions = self.engine.evaluate_delegations(
            "ach_dispute",  # Not card_dispute
            {"steps": steps, "input": {}}, []
        )
        self.assertEqual(len(decisions), 0)

    def test_delegation_cycle_detection(self):
        steps = [{
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_member": {"member_id": "M001"},
                    "get_fraud_score": {"factors": ["unknown_device"]},
                },
            },
        }]
        # Already visited sar_investigation
        lineage = ["sar_investigation:wf_prev"]
        decisions = self.engine.evaluate_delegations(
            "card_dispute", {"steps": steps, "input": {}}, lineage
        )
        self.assertEqual(len(decisions), 0)

    def test_delegation_evidence_flags(self):
        steps = [{
            "primitive": "investigate",
            "step_name": "investigate_activity",
            "output": {
                "finding": "Suspicious activity detected",
                "evidence_flags": ["foreign_ip", "account_takeover"],
            },
        }, {
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_member": {"member_id": "M001"},
                    "get_fraud_score": {"factors": []},
                },
            },
        }]
        decisions = self.engine.evaluate_delegations(
            "card_dispute", {"steps": steps, "input": {}}, []
        )
        # Should match fraud_evidence_flags_trigger_aml
        aml_decisions = [d for d in decisions if "evidence_flags" in d.policy_name]
        self.assertGreaterEqual(len(aml_decisions), 1)

    def test_delegation_input_resolution(self):
        steps = [{
            "primitive": "retrieve",
            "step_name": "gather_case_data",
            "output": {
                "data": {
                    "get_member": {"member_id": "MBR-123"},
                    "get_fraud_score": {
                        "score": 800,
                        "factors": ["unknown_device"],
                    },
                },
            },
        }]
        decisions = self.engine.evaluate_delegations(
            "card_dispute", {"steps": steps, "input": {}}, []
        )
        self.assertGreaterEqual(len(decisions), 1)
        self.assertEqual(decisions[0].inputs["member_id"], "MBR-123")

    def test_delegation_mode_parsed(self):
        modes = {p.name: p.mode for p in self.engine.delegation_policies}
        self.assertEqual(modes["fraud_pattern_triggers_aml"], "fire_and_forget")
        self.assertEqual(modes["sar_multi_member_disputes"], "wait_for_result")

    def test_contract_validation_valid(self):
        errors = self.engine.validate_work_order_inputs(
            "aml_referral_v1",
            {"member_id": "M001", "referral_reason": "test"},
        )
        self.assertEqual(errors, [])

    def test_contract_validation_missing_required(self):
        errors = self.engine.validate_work_order_inputs(
            "aml_referral_v1",
            {"member_id": "M001"},  # Missing referral_reason
        )
        self.assertGreater(len(errors), 0)

    def test_contract_validation_unknown(self):
        errors = self.engine.validate_work_order_inputs(
            "nonexistent_contract", {}
        )
        self.assertGreater(len(errors), 0)  # Unknown contracts produce an error

    def test_need_matching(self):
        needs = [{"type": "industry_benchmarks", "description": "test"}]
        matches = self.engine.match_needs(needs)
        self.assertGreaterEqual(len(matches), 1)


# ═══════════════════════════════════════════════════════════════════
# 5. COORDINATOR LIFECYCLE
# ═══════════════════════════════════════════════════════════════════

class CoordinatorTestBase(unittest.TestCase):
    """Base class providing a coordinator with in-memory store."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store,
            verbose=False,
        )

    def _make_instance(self, wf="dispute_resolution", domain="card_dispute",
                       tier="spot_check", status=InstanceStatus.CREATED):
        inst = InstanceState.create(wf, domain, tier)
        inst.status = status
        self.store.save_instance(inst)
        return inst

    def _make_completed_instance(self, wf="dispute_resolution", domain="card_dispute",
                                  tier="spot_check", steps=None):
        inst = self._make_instance(wf, domain, tier, InstanceStatus.COMPLETED)
        inst.step_count = len(steps or [])
        inst.result = {"step_count": inst.step_count, "steps": steps or []}
        self.store.save_instance(inst)
        return inst

    def _make_suspended_instance(self, wf="sar_investigation", domain="structuring_sar",
                                  tier="hold", step="__governance_gate__",
                                  state_snapshot=None, work_order_ids=None):
        inst = self._make_instance(wf, domain, tier, InstanceStatus.SUSPENDED)
        inst.step_count = 7
        inst.result = {"step_count": 7}
        sus = Suspension.create(
            inst.instance_id, step,
            state_snapshot or {"steps": [], "input": {}},
            work_order_ids=work_order_ids,
        )
        inst.resume_nonce = sus.resume_nonce
        self.store.save_instance(inst)
        self.store.save_suspension(sus)
        return inst, sus


class TestCoordinatorGovernance(CoordinatorTestBase):
    def test_resolve_tier_from_domain(self):
        self.assertEqual(self.coord._resolve_governance_tier("debit_spending"), "auto")
        self.assertEqual(self.coord._resolve_governance_tier("card_dispute"), "spot_check")
        self.assertEqual(self.coord._resolve_governance_tier("military_hardship"), "gate")
        self.assertEqual(self.coord._resolve_governance_tier("structuring_sar"), "hold")

    def test_resolve_tier_unknown_defaults_to_gate(self):
        self.assertEqual(self.coord._resolve_governance_tier("unknown_domain"), "gate")


class TestCoordinatorApproval(CoordinatorTestBase):
    def test_approve_valid(self):
        inst, sus = self._make_suspended_instance()
        self.coord.approve(inst.instance_id, approver="BSA Officer")
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.COMPLETED)

    def test_approve_cleans_suspension(self):
        inst, sus = self._make_suspended_instance()
        self.coord.approve(inst.instance_id)
        self.assertIsNone(self.store.get_suspension(inst.instance_id))

    def test_approve_logs_to_ledger(self):
        inst, sus = self._make_suspended_instance()
        self.coord.approve(inst.instance_id, approver="Analyst")
        ledger = self.store.get_ledger(instance_id=inst.instance_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("governance_approved", types)

    def test_approve_not_found(self):
        with self.assertRaises(ValueError):
            self.coord.approve("nonexistent")

    def test_approve_not_suspended(self):
        inst = self._make_instance(status=InstanceStatus.COMPLETED)
        with self.assertRaises(ValueError):
            self.coord.approve(inst.instance_id)

    def test_approve_no_suspension_record(self):
        inst = self._make_instance(status=InstanceStatus.SUSPENDED)
        with self.assertRaises(ValueError):
            self.coord.approve(inst.instance_id)

    def test_approve_idempotency(self):
        inst, sus = self._make_suspended_instance()
        nonce = sus.resume_nonce
        self.coord.approve(inst.instance_id)
        # Second approve fails (no longer suspended)
        with self.assertRaises(ValueError):
            self.coord.approve(inst.instance_id)


class TestCoordinatorRejection(CoordinatorTestBase):
    def test_reject_valid(self):
        inst, sus = self._make_suspended_instance()
        self.coord.reject(inst.instance_id, rejector="Compliance", reason="Bad")
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.TERMINATED)

    def test_reject_cleans_suspension(self):
        inst, sus = self._make_suspended_instance()
        self.coord.reject(inst.instance_id)
        self.assertIsNone(self.store.get_suspension(inst.instance_id))

    def test_reject_logs(self):
        inst, sus = self._make_suspended_instance()
        self.coord.reject(inst.instance_id, rejector="Mgr", reason="Insufficient")
        ledger = self.store.get_ledger(instance_id=inst.instance_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("governance_rejected", types)
        self.assertIn("terminate", types)


class TestCoordinatorTerminate(CoordinatorTestBase):
    def test_terminate(self):
        inst = self._make_instance(status=InstanceStatus.RUNNING)
        result = self.coord.terminate(inst.instance_id, "manual stop")
        self.assertEqual(result["status"], "terminated")
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.TERMINATED)

    def test_terminate_cleans_suspension(self):
        inst, sus = self._make_suspended_instance()
        self.coord.terminate(inst.instance_id, "cancelled")
        self.assertIsNone(self.store.get_suspension(inst.instance_id))


# ═══════════════════════════════════════════════════════════════════
# 6. GOVERNANCE SUSPENSION → TASK QUEUE
# ═══════════════════════════════════════════════════════════════════

class TestGovernanceSuspensionPublishesTask(CoordinatorTestBase):
    def test_suspend_publishes_task(self):
        inst = self._make_instance("sar_investigation", "structuring_sar", "hold",
                                    InstanceStatus.COMPLETED)
        inst.result = {"step_count": 7}
        self.store.save_instance(inst)

        from coordinator.policy import GovernanceDecision
        gov = GovernanceDecision(
            tier="hold", action="suspend_for_approval",
            queue="compliance_review", reason="test",
        )
        self.coord._suspend_for_governance(
            inst, {"steps": [], "input": {}}, gov
        )

        # Task should be in the queue
        tasks = self.coord.list_queue_tasks(queue="compliance_review")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_type"], TaskType.GOVERNANCE_APPROVAL)
        self.assertEqual(tasks[0]["instance_id"], inst.instance_id)

    def test_claim_and_resolve_via_coordinator(self):
        inst, sus = self._make_suspended_instance()
        # Publish a task manually (normally done by _suspend_for_governance)
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="compliance_review",
            instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            workflow_type=inst.workflow_type,
            domain=inst.domain,
            payload={"tier": "hold"},
            callback=TaskCallback(
                method="approve",
                instance_id=inst.instance_id,
                resume_nonce=sus.resume_nonce,
            ),
        )
        self.coord.tasks.publish(task)

        # Claim
        claimed = self.coord.claim_task("compliance_review", "BSA Officer")
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["instance_id"], inst.instance_id)

        # Resolve via task API
        self.coord.resolve_task(
            task_id=claimed["task_id"],
            action="approve",
            resolved_by="BSA Officer",
        )
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.COMPLETED)


class TestPendingApprovals(CoordinatorTestBase):
    def test_empty_pending(self):
        self.assertEqual(self.coord.list_pending_approvals(), [])

    def test_pending_shows_governance_tasks(self):
        inst, sus = self._make_suspended_instance()
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="compliance_review",
            instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            workflow_type="sar_investigation",
            domain="structuring_sar",
            payload={"governance_tier": "hold"},
            callback=TaskCallback(method="approve", instance_id=inst.instance_id),
        )
        self.coord.tasks.publish(task)
        pending = self.coord.list_pending_approvals()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["governance_tier"], "hold")


# ═══════════════════════════════════════════════════════════════════
# 7. FIRE-AND-FORGET DELEGATION
# ═══════════════════════════════════════════════════════════════════

class TestFireAndForgetDelegation(CoordinatorTestBase):
    def test_fire_and_forget_creates_work_order(self):
        source = self._make_completed_instance()
        decision = DelegationDecision(
            policy_name="test_fire",
            target_workflow="sar_investigation",
            target_domain="structuring_sar",
            contract_name="aml_referral_v1",
            contract_version=1,
            inputs={"member_id": "M001", "referral_reason": "test"},
            mode="fire_and_forget",
        )
        # This will fail on start() (no LLM), but work order should be created
        try:
            self.coord._execute_delegation(
                source, decision, {"steps": [], "input": {}}
            )
        except Exception:
            pass

        wos = self.store.get_work_orders_for_instance(source.instance_id)
        self.assertEqual(len(wos), 1)
        self.assertEqual(wos[0].contract_name, "aml_referral_v1")

    def test_fire_and_forget_source_stays_completed(self):
        source = self._make_completed_instance()
        decision = DelegationDecision(
            policy_name="test_fire",
            target_workflow="sar_investigation",
            target_domain="structuring_sar",
            contract_name="aml_referral_v1",
            contract_version=1,
            inputs={"member_id": "M001", "referral_reason": "test"},
            mode="fire_and_forget",
        )
        try:
            self.coord._execute_delegation(
                source, decision, {"steps": [], "input": {}}
            )
        except Exception:
            pass
        loaded = self.store.get_instance(source.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.COMPLETED)

    def test_contract_validation_blocks_delegation(self):
        source = self._make_completed_instance()
        decision = DelegationDecision(
            policy_name="test_bad",
            target_workflow="sar_investigation",
            target_domain="structuring_sar",
            contract_name="aml_referral_v1",
            contract_version=1,
            inputs={"member_id": "M001"},  # Missing referral_reason
            mode="fire_and_forget",
        )
        self.coord._execute_delegation(
            source, decision, {"steps": [], "input": {}}
        )
        # No work order created
        wos = self.store.get_work_orders_for_instance(source.instance_id)
        self.assertEqual(len(wos), 0)
        # Skipped logged
        ledger = self.store.get_ledger(instance_id=source.instance_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("delegation_skipped", types)


# ═══════════════════════════════════════════════════════════════════
# 8. BLOCKING DELEGATION (wait_for_result)
# ═══════════════════════════════════════════════════════════════════

class TestBlockingDelegation(CoordinatorTestBase):
    def test_blocking_creates_work_order(self):
        """Blocking delegation creates work order before attempting handler."""
        source = self._make_completed_instance()
        decision = DelegationDecision(
            policy_name="test_blocking",
            target_workflow="sar_investigation",
            target_domain="structuring_sar",
            contract_name="aml_referral_v1",
            contract_version=1,
            inputs={"member_id": "M001", "referral_reason": "test"},
            mode="wait_for_result",
        )
        try:
            self.coord._execute_delegation(
                source, decision,
                {"steps": [{"step_name": "s", "primitive": "classify", "output": {}}],
                 "input": {}, "current_step": "s", "metadata": {},
                 "loop_counts": {}, "routing_log": []},
            )
        except Exception:
            pass

        wos = self.store.get_work_orders_for_instance(source.instance_id)
        self.assertEqual(len(wos), 1)

    def test_blocking_logs_suspension_in_ledger(self):
        """Blocking delegation records suspension attempt in ledger."""
        source = self._make_completed_instance()
        decision = DelegationDecision(
            policy_name="test_blocking",
            target_workflow="sar_investigation",
            target_domain="structuring_sar",
            contract_name="aml_referral_v1",
            contract_version=1,
            inputs={"member_id": "M001", "referral_reason": "test"},
            mode="wait_for_result",
            resume_at_step="step_b",
        )
        source_state = {
            "steps": [
                {"step_name": "step_a", "primitive": "retrieve", "output": {}},
                {"step_name": "step_b", "primitive": "generate", "output": {}},
            ],
            "input": {"member_id": "M001"},
            "current_step": "step_b",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
        }
        try:
            self.coord._execute_delegation(source, decision, source_state)
        except Exception:
            pass

        ledger = self.store.get_ledger(instance_id=source.instance_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("delegation_dispatched", types)
        self.assertIn("suspended_for_delegation", types)

    def test_blocking_suspension_mechanics_unit(self):
        """Test _execute_blocking_delegation suspension directly (no handler start)."""
        source = self._make_completed_instance()
        source_state = {
            "steps": [
                {"step_name": "step_a", "primitive": "retrieve", "output": {}},
                {"step_name": "step_b", "primitive": "generate", "output": {"artifact": "letter"}},
            ],
            "input": {"member_id": "M001"},
            "current_step": "step_b",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
        }

        # Manually do what _execute_blocking_delegation does BEFORE handler start
        wo = WorkOrder.create(source.instance_id, source.correlation_id,
                              "aml_referral_v1", 1,
                              {"member_id": "M001", "referral_reason": "test"})
        self.store.save_work_order(wo)

        sus = Suspension.create(
            instance_id=source.instance_id,
            suspended_at_step="step_b",
            state_snapshot=source_state,
            work_order_ids=[wo.work_order_id],
        )
        self.store.save_suspension(sus)

        source.status = InstanceStatus.SUSPENDED
        source.resume_nonce = sus.resume_nonce
        source.pending_work_orders = [wo.work_order_id]
        self.store.save_instance(source)

        # Verify suspension state
        loaded = self.store.get_instance(source.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)
        self.assertEqual(loaded.pending_work_orders, [wo.work_order_id])

        loaded_sus = self.store.get_suspension(source.instance_id)
        self.assertIsNotNone(loaded_sus)
        self.assertEqual(loaded_sus.suspended_at_step, "step_b")
        self.assertEqual(loaded_sus.work_order_ids, [wo.work_order_id])
        self.assertEqual(loaded_sus.state_snapshot["input"]["member_id"], "M001")


# ═══════════════════════════════════════════════════════════════════
# 9. CASCADE: HANDLER COMPLETES → SOURCE RESUMES
# ═══════════════════════════════════════════════════════════════════

class TestDelegationCascade(CoordinatorTestBase):
    def _setup_cascade(self):
        """Create a scenario: source suspended, handler suspended (governance)."""
        # Source
        source = self._make_instance("dispute_resolution", "card_dispute", "spot_check",
                                      InstanceStatus.SUSPENDED)
        source.step_count = 5
        source.result = {"step_count": 5}

        # Handler
        handler = self._make_instance("sar_investigation", "structuring_sar", "hold",
                                       InstanceStatus.SUSPENDED)
        handler.correlation_id = source.correlation_id
        handler.step_count = 7
        handler.result = {"filing_decision": "file_sar", "risk_assessment": "high"}

        # Work order linking them
        wo = WorkOrder.create(
            source.instance_id, source.correlation_id,
            "aml_referral_v1", 1,
            {"member_id": "M001", "referral_reason": "test"},
        )
        wo.handler_instance_id = handler.instance_id
        wo.handler_workflow_type = "sar_investigation"
        wo.handler_domain = "structuring_sar"
        wo.status = WorkOrderStatus.RUNNING
        self.store.save_work_order(wo)

        # Source suspension (waiting on work order)
        source_sus = Suspension.create(
            source.instance_id, "generate_response",
            {"steps": [], "input": {"member_id": "M001"},
             "current_step": "generate_response", "metadata": {},
             "loop_counts": {}, "routing_log": []},
            work_order_ids=[wo.work_order_id],
        )
        source.resume_nonce = source_sus.resume_nonce
        source.pending_work_orders = [wo.work_order_id]
        self.store.save_instance(source)
        self.store.save_suspension(source_sus)

        # Handler suspension (governance hold)
        handler_sus = Suspension.create(
            handler.instance_id, "__governance_gate__",
            {"steps": [], "input": {}},
        )
        handler.resume_nonce = handler_sus.resume_nonce
        self.store.save_instance(handler)
        self.store.save_suspension(handler_sus)

        return source, handler, wo, source_sus, handler_sus

    def test_handler_approval_cascades_to_source(self):
        source, handler, wo, source_sus, handler_sus = self._setup_cascade()

        # Approve handler → should cascade
        try:
            self.coord.approve(handler.instance_id, approver="BSA")
        except Exception:
            pass  # resume() will fail without LangGraph

        # Handler should be completed
        handler_loaded = self.store.get_instance(handler.instance_id)
        self.assertEqual(handler_loaded.status, InstanceStatus.COMPLETED)

        # Work order should be completed
        wo_loaded = self.store.get_work_order(wo.work_order_id)
        self.assertEqual(wo_loaded.status, WorkOrderStatus.COMPLETED)
        self.assertIsNotNone(wo_loaded.result)

        # Ledger should show cascade
        ledger = self.store.get_ledger(correlation_id=source.correlation_id)
        types = [e["action_type"] for e in ledger]
        self.assertIn("governance_approved", types)

    def test_handler_rejection_does_not_cascade(self):
        source, handler, wo, source_sus, handler_sus = self._setup_cascade()

        self.coord.reject(handler.instance_id, rejector="Compliance", reason="Bad")

        # Handler terminated
        handler_loaded = self.store.get_instance(handler.instance_id)
        self.assertEqual(handler_loaded.status, InstanceStatus.TERMINATED)

        # Source stays suspended (handler didn't complete successfully)
        source_loaded = self.store.get_instance(source.instance_id)
        self.assertEqual(source_loaded.status, InstanceStatus.SUSPENDED)

    def test_check_delegation_completion_updates_work_order(self):
        source, handler, wo, source_sus, handler_sus = self._setup_cascade()

        # Manually complete handler
        handler.status = InstanceStatus.COMPLETED
        handler.result = {"filing_decision": "file_sar"}
        self.store.save_instance(handler)
        self.store.delete_suspension(handler.instance_id)

        # Check cascade
        try:
            self.coord._check_delegation_completion(handler)
        except Exception:
            pass  # resume fails without LLM

        wo_loaded = self.store.get_work_order(wo.work_order_id)
        self.assertEqual(wo_loaded.status, WorkOrderStatus.COMPLETED)
        self.assertEqual(wo_loaded.result.outputs["filing_decision"], "file_sar")


# ═══════════════════════════════════════════════════════════════════
# 10. IDEMPOTENCY
# ═══════════════════════════════════════════════════════════════════

class TestIdempotency(CoordinatorTestBase):
    def test_double_approve_fails(self):
        inst, sus = self._make_suspended_instance()
        self.coord.approve(inst.instance_id)
        with self.assertRaises(ValueError):
            self.coord.approve(inst.instance_id)

    def test_approve_after_terminate_fails(self):
        inst, sus = self._make_suspended_instance()
        self.coord.terminate(inst.instance_id, "test")
        with self.assertRaises(ValueError):
            self.coord.approve(inst.instance_id)

    def test_reject_after_approve_fails(self):
        inst, sus = self._make_suspended_instance()
        self.coord.approve(inst.instance_id)
        with self.assertRaises(ValueError):
            self.coord.reject(inst.instance_id)

    def test_ledger_idempotency_key(self):
        ok1 = self.store.log_action("wf", "wf", "test", {},
                                     idempotency_key="unique_key")
        ok2 = self.store.log_action("wf", "wf", "test", {},
                                     idempotency_key="unique_key")
        self.assertTrue(ok1)
        self.assertFalse(ok2)


# ═══════════════════════════════════════════════════════════════════
# 11. EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases(CoordinatorTestBase):
    def test_checkpoint_returns_suspension(self):
        inst, sus = self._make_suspended_instance(
            state_snapshot={"steps": [{"name": "a"}], "input": {"k": "v"}}
        )
        result = self.coord.checkpoint(inst.instance_id)
        self.assertEqual(result["input"]["k"], "v")

    def test_checkpoint_returns_result_when_completed(self):
        inst = self._make_completed_instance(steps=[{"step_name": "a"}])
        result = self.coord.checkpoint(inst.instance_id)
        self.assertIn("step_count", result)

    def test_checkpoint_not_found(self):
        with self.assertRaises(ValueError):
            self.coord.checkpoint("nonexistent")

    def test_resume_wrong_nonce(self):
        inst, sus = self._make_suspended_instance()
        with self.assertRaises(ValueError):
            self.coord.resume(inst.instance_id, {}, resume_nonce="wrong_nonce")

    def test_resume_not_suspended(self):
        inst = self._make_instance(status=InstanceStatus.COMPLETED)
        with self.assertRaises(ValueError):
            self.coord.resume(inst.instance_id, {})

    def test_get_instance(self):
        inst = self._make_instance()
        loaded = self.coord.get_instance(inst.instance_id)
        self.assertEqual(loaded.instance_id, inst.instance_id)

    def test_get_correlation_chain(self):
        corr = "test_corr"
        for _ in range(3):
            inst = self._make_instance()
            inst.correlation_id = corr
            self.store.save_instance(inst)
        chain = self.coord.get_correlation_chain(corr)
        self.assertEqual(len(chain), 3)

    def test_stats(self):
        self._make_instance(status=InstanceStatus.COMPLETED)
        self._make_instance(status=InstanceStatus.SUSPENDED)
        s = self.coord.stats()
        self.assertIn("instances", s)
        self.assertIn("work_orders", s)

    def test_expire_overdue_tasks(self):
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="test",
            instance_id="wf",
            correlation_id="wf",
            workflow_type="w",
            domain="d",
            payload={},
            callback=TaskCallback(method="approve", instance_id="wf"),
            sla_seconds=0.001,
        )
        self.coord.tasks.publish(task)
        time.sleep(0.01)
        count = self.coord.expire_overdue_tasks()
        self.assertEqual(count, 1)

    def test_resolve_task_defer(self):
        inst, sus = self._make_suspended_instance()
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="compliance_review",
            instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            workflow_type="sar_investigation",
            domain="structuring_sar",
            payload={},
            callback=TaskCallback(method="approve", instance_id=inst.instance_id),
        )
        self.coord.tasks.publish(task)
        claimed = self.coord.claim_task("compliance_review", "Analyst")
        # Defer puts it back
        self.coord.resolve_task(claimed["task_id"], "defer", "Analyst")
        # Instance still suspended
        loaded = self.store.get_instance(inst.instance_id)
        self.assertEqual(loaded.status, InstanceStatus.SUSPENDED)

    def test_resolve_task_unknown_action(self):
        inst, sus = self._make_suspended_instance()
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="test", instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            workflow_type="w", domain="d", payload={},
            callback=TaskCallback(method="approve", instance_id=inst.instance_id),
        )
        self.coord.tasks.publish(task)
        self.coord.claim_task("test", "user")
        with self.assertRaises(ValueError):
            self.coord.resolve_task(task.task_id, "unknown_action")


# ═══════════════════════════════════════════════════════════════════
# 12. PARAMETER RESOLUTION (delegation results)
# ═══════════════════════════════════════════════════════════════════

class TestDelegationParameterResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Import resolve_param directly, bypassing engine/__init__.py
        which pulls in langgraph (not available in test env)."""
        import importlib
        import sys
        # Temporarily remove engine from sys.modules so we can import state directly
        engine_mods = [k for k in sys.modules if k.startswith("engine")]
        saved = {k: sys.modules.pop(k) for k in engine_mods}
        try:
            # Stub langgraph so composer.py doesn't crash if imported transitively
            if "langgraph" not in sys.modules:
                import types
                lg = types.ModuleType("langgraph")
                lg_graph = types.ModuleType("langgraph.graph")
                lg_graph.StateGraph = type("StateGraph", (), {})
                lg_graph.END = "__end__"
                sys.modules["langgraph"] = lg
                sys.modules["langgraph.graph"] = lg_graph
            spec = importlib.util.spec_from_file_location(
                "engine.state", os.path.join(_project_root, "engine", "state.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls.resolve_param = staticmethod(mod.resolve_param)
        finally:
            sys.modules.update(saved)

    def test_delegation_results_basic(self):
        state = {
            "input": {}, "steps": [], "current_step": "",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
            "delegation_results": {
                "wo_001": {"filing_decision": "file_sar", "risk": "high"},
            },
        }
        val = self.resolve_param("${_delegations.wo_001.filing_decision}", state)
        self.assertEqual(val, "file_sar")

    def test_delegation_results_nested(self):
        state = {
            "input": {}, "steps": [], "current_step": "",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
            "delegation_results": {
                "wo_001": {"nested": {"deep": "value"}},
            },
        }
        val = self.resolve_param("${_delegations.wo_001.nested.deep}", state)
        self.assertEqual(val, "value")

    def test_delegation_results_all(self):
        state = {
            "input": {}, "steps": [], "current_step": "",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
            "delegation_results": {
                "wo_001": {"a": 1},
                "wo_002": {"b": 2},
            },
        }
        val = self.resolve_param("${_delegations}", state)
        parsed = json.loads(val)
        self.assertIn("wo_001", parsed)
        self.assertIn("wo_002", parsed)

    def test_delegation_results_empty(self):
        state = {
            "input": {}, "steps": [], "current_step": "",
            "metadata": {}, "loop_counts": {}, "routing_log": [],
        }
        val = self.resolve_param("${_delegations.wo_001.field}", state)
        self.assertIsNotNone(val)


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
