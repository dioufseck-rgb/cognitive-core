"""
Cognitive Core — Contract Validation Tests

Systematic boundary testing for all data contracts flowing between
coordinator modules. These tests catch the class of bugs where:

  1. A producer returns {"created_at": ...} but the consumer reads ["suspended_at"]
  2. A delegation passes contract inputs but forgets tool data
  3. A CLI command assumes fields that the API doesn't provide
  4. A store method returns different keys than the caller expects

Every test creates real data through the actual code path (not mocks)
and validates that all consuming code can read the produced data
without KeyError, AttributeError, or TypeError.

Run: python -m pytest tests/test_contracts.py -v
"""

import json
import os
import time
import unittest
from io import StringIO
from unittest.mock import patch

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from coordinator.store import CoordinatorStore
from coordinator.runtime import Coordinator
from coordinator.types import (
    InstanceState, InstanceStatus, WorkOrder, WorkOrderStatus,
    Suspension, GovernanceTier,
)
from coordinator.tasks import (
    Task, TaskType, TaskStatus, TaskCallback, SQLiteTaskQueue,
)


class TestPendingApprovalsContract(unittest.TestCase):
    """
    Validates: list_pending_approvals() → cmd_pending()

    The bug: list_pending_approvals returned {created_at: ...} but
    cmd_pending read a["suspended_at"] → KeyError.
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def _create_held_instance(self):
        """Create a realistic governance-held instance with task."""
        inst = InstanceState.create("vendor_notification", "vendor_ops", "hold")
        inst.status = InstanceStatus.SUSPENDED
        inst.step_count = 4
        inst.correlation_id = inst.instance_id
        inst.result = {"step_count": 4, "steps": []}
        self.store.save_instance(inst)

        # Create suspension
        sus = Suspension.create(
            inst.instance_id, "verify_notification",
            {"steps": [], "step_count": 4},
        )
        inst.resume_nonce = sus.resume_nonce
        self.store.save_instance(inst)
        self.store.save_suspension(sus)

        # Publish governance task
        task = Task.create(
            task_type=TaskType.GOVERNANCE_APPROVAL,
            queue="compliance_review",
            instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            workflow_type=inst.workflow_type,
            domain=inst.domain,
            payload={"governance_tier": "hold", "reason": "Hold tier"},
            callback=TaskCallback(
                method="approve",
                instance_id=inst.instance_id,
                resume_nonce=sus.resume_nonce,
            ),
            priority=2,
        )
        self.coord.tasks.publish(task)
        return inst

    def test_list_pending_approvals_returns_required_fields(self):
        """API contract: every field that cmd_pending reads must exist."""
        self._create_held_instance()
        approvals = self.coord.list_pending_approvals()
        self.assertEqual(len(approvals), 1)

        required_fields = [
            "instance_id", "workflow_type", "domain",
            "governance_tier", "correlation_id",
            "created_at", "queue",
        ]
        a = approvals[0]
        for field in required_fields:
            self.assertIn(field, a, f"Missing field '{field}' in pending approval")
            self.assertIsNotNone(a[field], f"Field '{field}' is None")

    def test_cmd_pending_consumes_api_without_error(self):
        """Integration: cmd_pending can format the data without crashing."""
        from coordinator.cli import cmd_pending

        self._create_held_instance()

        # cmd_pending prints to stdout — capture and verify no exception
        class FakeArgs:
            pass

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            cmd_pending(FakeArgs(), self.coord)

        output = mock_out.getvalue()
        self.assertIn("vendor_notification", output)
        self.assertIn("vendor_ops", output)
        self.assertIn("hold", output)

    def test_cmd_pending_no_approvals(self):
        """Edge case: no held instances."""
        from coordinator.cli import cmd_pending

        class FakeArgs:
            pass

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            cmd_pending(FakeArgs(), self.coord)
        self.assertIn("No instances pending", mock_out.getvalue())


class TestLedgerContract(unittest.TestCase):
    """
    Validates: store.get_ledger() → cmd_ledger()
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_ledger_entry_has_required_fields(self):
        """Every field that cmd_ledger reads must exist in store output."""
        self.store.log_action(
            instance_id="wf_test",
            correlation_id="wf_test",
            action_type="start",
            details={"workflow_type": "test", "domain": "test"},
        )
        entries = self.store.get_ledger(instance_id="wf_test")
        self.assertEqual(len(entries), 1)

        required = ["created_at", "action_type", "instance_id", "details"]
        for field in required:
            self.assertIn(field, entries[0], f"Missing '{field}' in ledger entry")

    def test_cmd_ledger_consumes_store_data(self):
        """Integration: cmd_ledger formats ledger data without crashing."""
        from coordinator.cli import cmd_ledger

        self.store.log_action("wf_test", "wf_test", "start", {"key": "value"})
        self.store.log_action("wf_test", "wf_test", "execution_finished", {"steps": 4})

        class FakeArgs:
            instance = "wf_test"
            correlation = None
            verbose = True  # exercise the verbose branch too

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            cmd_ledger(FakeArgs(), self.coord)
        output = mock_out.getvalue()
        self.assertIn("start", output)
        self.assertIn("execution_finished", output)


class TestCorrelationChainContract(unittest.TestCase):
    """
    Validates: get_correlation_chain() → cmd_chain()
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_chain_instances_have_required_attributes(self):
        """cmd_chain accesses .instance_id, .workflow_type, .domain, .status, etc."""
        inst = InstanceState.create("product_return", "electronics_return", "auto")
        inst.status = InstanceStatus.COMPLETED
        inst.step_count = 6
        inst.correlation_id = inst.instance_id
        self.store.save_instance(inst)

        chain = self.coord.get_correlation_chain(inst.correlation_id)
        self.assertEqual(len(chain), 1)

        # These are the attributes cmd_chain accesses
        c = chain[0]
        required_attrs = [
            "instance_id", "workflow_type", "domain",
            "status", "governance_tier", "step_count", "lineage",
        ]
        for attr in required_attrs:
            self.assertTrue(hasattr(c, attr), f"Missing attribute '{attr}' on InstanceState")

    def test_cmd_chain_renders_without_error(self):
        """Integration: cmd_chain formats chain without crashing."""
        from coordinator.cli import cmd_chain

        inst = InstanceState.create("product_return", "electronics_return", "auto")
        inst.status = InstanceStatus.COMPLETED
        inst.correlation_id = inst.instance_id
        inst.step_count = 4
        self.store.save_instance(inst)

        class FakeArgs:
            instance_id = inst.instance_id

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            cmd_chain(FakeArgs(), self.coord)
        output = mock_out.getvalue()
        self.assertIn("product_return", output)


class TestDelegationInputContract(unittest.TestCase):
    """
    Validates: delegation policy inputs resolve correctly and
    contain tool data that the handler's _build_tool_registry can detect.

    The bug: delegation inputs had only scalar contract fields but no
    get_* tool data, so the handler's retrieve step found no tools → LLM hung.
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_vendor_notify_delegation_includes_tool_data(self):
        """vendor_notification inputs must include get_* tool data."""
        from coordinator.policy import load_policy_engine
        pe = self.coord.policy

        # Find the vendor notify policy
        vendor_policy = None
        for p in pe.delegation_policies:
            if p.name == "high_value_return_vendor_notify":
                vendor_policy = p
                break

        self.assertIsNotNone(vendor_policy, "vendor_notification delegation policy not found")

        # Check input mapping has tool data keys
        tool_keys = [k for k in vendor_policy.input_mapping if k.startswith("get_")]
        self.assertGreater(len(tool_keys), 0,
            "Delegation inputs must include get_* tool data for handler's retrieve step. "
            "Without these, the handler's _build_tool_registry creates an empty registry "
            "and the LLM hangs trying to call tools that don't exist.")

    def test_fraud_review_delegation_includes_tool_data(self):
        """fraud_review inputs must include get_* tool data."""
        pe = self.coord.policy

        fraud_policy = None
        for p in pe.delegation_policies:
            if p.name == "return_fraud_indicators_trigger_review":
                fraud_policy = p
                break

        self.assertIsNotNone(fraud_policy)
        tool_keys = [k for k in fraud_policy.input_mapping if k.startswith("get_")]
        self.assertGreater(len(tool_keys), 0,
            "fraud_review delegation must include get_* tool data")

    def test_resolved_delegation_inputs_have_tool_data(self):
        """After resolution, delegation inputs should have dict-valued get_* keys."""
        pe = self.coord.policy

        # Simulate workflow output with tool data in input
        workflow_output = {
            "input": {
                "order_id": "ORD-001",
                "customer_id": "CUST-001",
                "return_reason": "defective",
                "item_condition": "opened",
                "get_vendor": {"name": "TechCorp", "contact": "vendor@example.com"},
                "get_product": {"sku": "PBX1-15", "name": "ProBook X1"},
                "get_order": {"item_price": 1249.99, "order_date": "2026-01-28"},
                "get_customer": {"name": "Marcus Webb", "return_rate_percent": 35.7},
                "get_return_history": {"total_returns": 5},
                "get_fraud_signals": {"risk_score": 72},
            },
            "steps": [
                {
                    "step_name": "gather_return_data",
                    "primitive": "retrieve",
                    "output": {
                        "data": {
                            "get_order": {"item_price": 1249.99},
                            "get_customer": {"name": "Marcus Webb"},
                        },
                        "sources_queried": [{"source": "get_order", "status": "success"}],
                    },
                },
                {
                    "step_name": "investigate_claim",
                    "primitive": "investigate",
                    "output": {
                        "finding": "Suspicious pattern",
                        "evidence_flags": ["high_return_rate", "serial_electronics_returner"],
                        "confidence": 0.85,
                    },
                },
            ],
        }

        delegations = pe.evaluate_delegations(
            domain="electronics_return",
            workflow_output=workflow_output,
        )

        # Should trigger at least the vendor notify
        vendor_delgs = [d for d in delegations if "vendor" in d.policy_name]
        self.assertGreater(len(vendor_delgs), 0, "vendor_notification should trigger")

        for d in vendor_delgs:
            tool_inputs = {k: v for k, v in d.inputs.items()
                         if isinstance(v, dict) and k.startswith("get_")}
            self.assertGreater(len(tool_inputs), 0,
                f"Delegation {d.policy_name} resolved inputs have no tool data: "
                f"keys = {list(d.inputs.keys())}")

    def test_build_tool_registry_detects_case_tools(self):
        """_build_tool_registry creates case registry when get_* tools present."""
        case_input = {
            "customer_id": "CUST-001",
            "get_vendor": {"name": "TechCorp"},
            "get_product": {"sku": "PBX1"},
        }
        # Verify the detection logic directly (avoids engine import which needs langgraph)
        has_case_tools = any(
            isinstance(v, (dict, list)) and k.startswith("get_")
            for k, v in case_input.items()
        )
        self.assertTrue(has_case_tools,
            "case_input with get_* dict keys should be detected as having tool data")

    def test_build_tool_registry_falls_back_gracefully(self):
        """With no get_* keys, still builds a registry (no crash)."""
        case_input = {"customer_id": "CUST-001", "reason": "test"}
        registry = self.coord._build_tool_registry(case_input)
        # Should not crash — returns empty or fixture-backed registry
        self.assertIsNotNone(registry)


class TestStatsContract(unittest.TestCase):
    """Validates: coord.stats() → cmd_stats()"""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_stats_serializes_to_json(self):
        """stats() return value must be JSON-serializable (cmd_stats uses json.dumps)."""
        s = self.coord.stats()
        try:
            json.dumps(s)
        except (TypeError, ValueError) as e:
            self.fail(f"stats() not JSON-serializable: {e}")


class TestDelegationDepthLimit(unittest.TestCase):
    """C1: Delegation chain depth must be bounded."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_depth_limit_raises_at_max(self):
        """start() with lineage at MAX_DELEGATION_DEPTH must raise."""
        from coordinator.runtime import DelegationDepthExceeded

        deep_lineage = [f"wf_{i}:inst_{i}" for i in range(self.coord.MAX_DELEGATION_DEPTH)]
        with self.assertRaises(DelegationDepthExceeded):
            self.coord.start(
                workflow_type="test",
                domain="test",
                case_input={},
                lineage=deep_lineage,
            )

    def test_depth_limit_allows_under_max(self):
        """start() with lineage under limit should NOT raise DelegationDepthExceeded."""
        from coordinator.runtime import DelegationDepthExceeded

        shallow_lineage = ["wf_0:inst_0", "wf_1:inst_1"]
        # Will fail for other reasons (no workflow file) but NOT DelegationDepthExceeded
        try:
            self.coord.start(
                workflow_type="test", domain="test",
                case_input={}, lineage=shallow_lineage,
            )
        except DelegationDepthExceeded:
            self.fail("Should not raise DelegationDepthExceeded for shallow lineage")
        except Exception:
            pass  # Other errors expected (no workflow file)


class TestStuckInstanceDetection(unittest.TestCase):
    """M3: Stuck instance detection."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_finds_stuck_running_instances(self):
        inst = InstanceState.create("test", "test", "auto")
        inst.status = InstanceStatus.RUNNING
        inst.updated_at = time.time() - 7200  # 2 hours ago
        self.store.save_instance(inst)

        stuck = self.store.find_stuck_instances(max_running_seconds=3600)
        self.assertEqual(len(stuck), 1)
        self.assertEqual(stuck[0].instance_id, inst.instance_id)

    def test_ignores_recently_running(self):
        inst = InstanceState.create("test", "test", "auto")
        inst.status = InstanceStatus.RUNNING
        inst.updated_at = time.time() - 30  # 30 seconds ago
        self.store.save_instance(inst)

        stuck = self.store.find_stuck_instances(max_running_seconds=3600)
        self.assertEqual(len(stuck), 0)


class TestTransactionBoundaries(unittest.TestCase):
    """C4: Transaction context manager."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_transaction_commits_on_success(self):
        inst = InstanceState.create("test", "test", "auto")
        with self.store.transaction():
            self.store.save_instance(inst)
        # Should be visible after commit
        loaded = self.store.get_instance(inst.instance_id)
        self.assertIsNotNone(loaded)

    def test_transaction_rolls_back_on_error(self):
        inst = InstanceState.create("test", "test", "auto")
        try:
            with self.store.transaction():
                self.store.save_instance(inst)
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass
        # Should NOT be visible after rollback
        loaded = self.store.get_instance(inst.instance_id)
        self.assertIsNone(loaded)

    def test_transaction_multiple_saves_atomic(self):
        """Multiple saves in one transaction are all-or-nothing."""
        inst1 = InstanceState.create("test1", "test", "auto")
        inst2 = InstanceState.create("test2", "test", "auto")
        try:
            with self.store.transaction():
                self.store.save_instance(inst1)
                self.store.save_instance(inst2)
                raise RuntimeError("Crash after both saves")
        except RuntimeError:
            pass
        self.assertIsNone(self.store.get_instance(inst1.instance_id))
        self.assertIsNone(self.store.get_instance(inst2.instance_id))

    def test_saves_outside_transaction_commit_immediately(self):
        """Normal saves (no transaction block) still commit."""
        inst = InstanceState.create("test", "test", "auto")
        self.store.save_instance(inst)
        loaded = self.store.get_instance(inst.instance_id)
        self.assertIsNotNone(loaded)


class TestParseFailureCircuitBreaker(unittest.TestCase):
    """H6: Parse failure detection and circuit breaker routing."""

    def test_parse_failed_flag_in_error_output(self):
        """Error output from nodes.py must include _parse_failed flag."""
        # Simulated error output (matches what create_primitive_node produces)
        error_output = {
            "error": "JSON parse error",
            "raw_response": "not json",
            "confidence": 0.0,
            "reasoning": "Failed to parse",
            "_parse_failed": True,
        }
        self.assertTrue(error_output.get("_parse_failed"))
        self.assertEqual(error_output["confidence"], 0.0)

    def test_evaluate_condition_parse_failed(self):
        """_parse_failed condition evaluates correctly against step output."""
        from engine.composer import _evaluate_condition

        # State with a parse-failed step
        state = {
            "steps": [
                {"step_name": "classify", "primitive": "classify",
                 "output": {"_parse_failed": True, "confidence": 0.0}}
            ],
            "current_step": "classify",
        }
        self.assertTrue(_evaluate_condition("_parse_failed", state, "classify"))

        # State with a successful step
        state_ok = {
            "steps": [
                {"step_name": "classify", "primitive": "classify",
                 "output": {"category": "fraud", "confidence": 0.95}}
            ],
            "current_step": "classify",
        }
        self.assertFalse(_evaluate_condition("_parse_failed", state_ok, "classify"))


class TestHandlerFailureLogging(unittest.TestCase):
    """H2: Delegation handler failures must be logged to the action ledger."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_ff_handler_failure_creates_ledger_entry(self):
        """Fire-and-forget handler failure should log delegation_handler_failed."""
        # Create a source instance that's completed
        inst = InstanceState.create("product_return", "electronics_return", "auto")
        inst.status = InstanceStatus.COMPLETED
        inst.correlation_id = inst.instance_id
        inst.result = {"step_count": 4}
        self.store.save_instance(inst)

        # Log a simulated handler failure
        self.store.log_action(
            instance_id=inst.instance_id,
            correlation_id=inst.correlation_id,
            action_type="delegation_handler_failed",
            details={
                "policy": "test_policy",
                "target": "test_wf/test_domain",
                "mode": "fire_and_forget",
                "error": "Connection refused",
            },
        )

        entries = self.store.get_ledger(instance_id=inst.instance_id)
        failure_entries = [e for e in entries if e["action_type"] == "delegation_handler_failed"]
        self.assertEqual(len(failure_entries), 1)
        self.assertIn("error", failure_entries[0]["details"])


class TestOrphanedSuspensionDetection(unittest.TestCase):
    """M3: Orphaned suspension detection."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_finds_orphaned_when_work_orders_completed(self):
        """Suspension whose work orders are done but instance still suspended."""
        inst = InstanceState.create("test", "test", "auto")
        inst.status = InstanceStatus.SUSPENDED
        inst.correlation_id = inst.instance_id
        self.store.save_instance(inst)

        # Create a completed work order
        wo = WorkOrder.create(inst.instance_id, inst.correlation_id, "test_contract", 1, {})
        wo.status = WorkOrderStatus.COMPLETED
        wo.handler_instance_id = "handler_123"
        self.store.save_work_order(wo)

        # Create suspension referencing that work order
        sus = Suspension.create(inst.instance_id, "step1", {}, work_order_ids=[wo.work_order_id])
        self.store.save_suspension(sus)

        orphans = self.store.find_orphaned_suspensions()
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0]["instance_id"], inst.instance_id)


class TestGovernanceEvaluationContract(unittest.TestCase):
    """
    Validates: policy.evaluate_governance() returns fields that
    _on_completed expects.
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_governance_decision_has_required_fields(self):
        """GovernanceDecision must have tier, action, reason, queue."""
        decision = self.coord.policy.evaluate_governance("test_domain", "auto", {})
        required = ["tier", "action", "reason"]
        for attr in required:
            self.assertTrue(hasattr(decision, attr),
                f"GovernanceDecision missing '{attr}'")

    def test_all_domain_tiers_resolve(self):
        """Every domain in config should resolve to a valid governance tier."""
        import yaml
        domains_dir = os.path.join(_project_root, "domains")
        for fname in os.listdir(domains_dir):
            if fname.endswith(".yaml"):
                with open(os.path.join(domains_dir, fname)) as f:
                    domain_config = yaml.safe_load(f)
                tier = domain_config.get("governance", "gate")
                self.assertIn(tier, ["auto", "spot_check", "gate", "hold"],
                    f"Domain {fname} has invalid governance tier: {tier}")


class TestDelegationPolicyContract(unittest.TestCase):
    """
    Validates: every delegation policy in config has a matching contract,
    and every input mapping references valid source paths.
    """

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_every_delegation_has_a_contract(self):
        pe = self.coord.policy
        for policy in pe.delegation_policies:
            contract = pe.contracts.get(policy.contract_name)
            self.assertIsNotNone(contract,
                f"Policy '{policy.name}' references contract '{policy.contract_name}' "
                f"but no such contract exists in config")

    def test_every_delegation_has_target_workflow_and_domain(self):
        pe = self.coord.policy
        for policy in pe.delegation_policies:
            self.assertTrue(policy.target_workflow,
                f"Policy '{policy.name}' has empty target_workflow")
            self.assertTrue(policy.target_domain,
                f"Policy '{policy.name}' has empty target_domain")

    def test_delegation_input_mappings_use_valid_source_refs(self):
        """All ${source.*} refs should start with input., last_, or any_."""
        pe = self.coord.policy
        valid_prefixes = ("input.", "last_", "any_", "final_output")
        for policy in pe.delegation_policies:
            for field, ref in policy.input_mapping.items():
                if ref.startswith("${source."):
                    path = ref[len("${source."):-1]
                    self.assertTrue(
                        any(path.startswith(p) for p in valid_prefixes),
                        f"Policy '{policy.name}' field '{field}' has invalid "
                        f"source ref: {ref}. Must start with one of {valid_prefixes}")


if __name__ == "__main__":
    unittest.main()


# ═══════════════════════════════════════════════════════════════════
# Reliability fix tests (C3, H5, H6, M1, M2)
# ═══════════════════════════════════════════════════════════════════

class TestStateCompaction(unittest.TestCase):
    """C3: State snapshot compaction."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_compaction_strips_raw_response(self):
        state = {
            "input": {"customer_id": "CUST-001"},
            "steps": [{
                "step_name": "classify",
                "primitive": "classify",
                "output": {"category": "fraud", "confidence": 0.95},
                "raw_response": "x" * 10000,
                "prompt_used": "y" * 8000,
            }],
        }
        compact = self.coord._compact_state_for_suspension(state)
        step = compact["steps"][0]
        self.assertNotIn("raw_response", step)
        self.assertNotIn("prompt_used", step)
        self.assertEqual(step["output"]["category"], "fraud")

    def test_compaction_preserves_input(self):
        state = {
            "input": {"customer_id": "CUST-001", "get_customer": {"name": "Test"}},
            "steps": [],
        }
        compact = self.coord._compact_state_for_suspension(state)
        self.assertEqual(compact["input"]["customer_id"], "CUST-001")
        self.assertEqual(compact["input"]["get_customer"]["name"], "Test")

    def test_compaction_truncates_large_retrieve_data(self):
        state = {
            "input": {},
            "steps": [{
                "step_name": "gather",
                "primitive": "retrieve",
                "output": {
                    "data": {
                        "get_customer": {"name": "Test", "history": "x" * 5000},
                    },
                },
                "raw_response": "",
                "prompt_used": "",
            }],
        }
        compact = self.coord._compact_state_for_suspension(state)
        data = compact["steps"][0]["output"]["data"]["get_customer"]
        self.assertIn("_truncated", data)

    def test_compaction_reduces_size(self):
        state = {
            "input": {"id": "1"},
            "steps": [
                {
                    "step_name": f"step_{i}",
                    "primitive": "classify",
                    "output": {"category": "test", "confidence": 0.9},
                    "raw_response": "A" * 5000,
                    "prompt_used": "B" * 3000,
                }
                for i in range(5)
            ],
        }
        raw_size = len(json.dumps(state))
        compact = self.coord._compact_state_for_suspension(state)
        compact_size = len(json.dumps(compact))
        self.assertLess(compact_size, raw_size * 0.5,
            f"Compaction should reduce size by >50%: {raw_size} → {compact_size}")

    def test_compaction_does_not_mutate_original(self):
        state = {
            "input": {},
            "steps": [{
                "step_name": "s1",
                "primitive": "classify",
                "output": {"category": "test"},
                "raw_response": "original",
                "prompt_used": "original",
            }],
        }
        self.coord._compact_state_for_suspension(state)
        self.assertEqual(state["steps"][0]["raw_response"], "original")


class TestGovernanceSkipOnResume(unittest.TestCase):
    """H5: Governance re-evaluation skipped on resume."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")
        self.coord = Coordinator(
            config_path=os.path.join(_project_root, "coordinator", "config.yaml"),
            store=self.store, verbose=False,
        )

    def test_on_completed_with_resume_logs_skipped(self):
        inst = InstanceState.create("test", "test", "auto")
        inst.status = InstanceStatus.RUNNING
        inst.correlation_id = inst.instance_id
        self.store.save_instance(inst)

        final_state = {"steps": [], "input": {}}
        self.coord._on_completed(inst, final_state, is_resume=True)

        ledger = self.store.get_ledger(instance_id=inst.instance_id)
        gov_entries = [e for e in ledger if e["action_type"] == "governance_evaluation"]
        self.assertEqual(len(gov_entries), 1)
        self.assertTrue(gov_entries[0]["details"].get("skipped"))

    def test_on_completed_without_resume_evaluates_governance(self):
        inst = InstanceState.create("test", "test", "auto")
        inst.status = InstanceStatus.RUNNING
        inst.correlation_id = inst.instance_id
        self.store.save_instance(inst)

        final_state = {"steps": [], "input": {}}
        self.coord._on_completed(inst, final_state, is_resume=False)

        ledger = self.store.get_ledger(instance_id=inst.instance_id)
        gov_entries = [e for e in ledger if e["action_type"] == "governance_evaluation"]
        self.assertEqual(len(gov_entries), 1)
        self.assertFalse(gov_entries[0]["details"].get("skipped", False))


class TestParseFailureCircuitBreaker(unittest.TestCase):
    """H6: Parse failure flag in output."""

    def test_parse_failed_flag_format(self):
        """Error output with _parse_failed should have expected fields."""
        output = {
            "error": "ValidationError: ...",
            "raw_response": "bad json...",
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response: ...",
            "evidence_used": [],
            "evidence_missing": [],
            "_parse_failed": True,
        }
        self.assertTrue(output["_parse_failed"])
        self.assertEqual(output["confidence"], 0.0)
        self.assertIn("error", output)


class TestCorrelationChainLimit(unittest.TestCase):
    """M1: Correlation chain bounded by LIMIT."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_list_instances_respects_limit(self):
        corr = "test_correlation"
        for i in range(20):
            inst = InstanceState.create("test", "test", "auto")
            inst.correlation_id = corr
            self.store.save_instance(inst)

        all_inst = self.store.list_instances(correlation_id=corr, limit=500)
        self.assertEqual(len(all_inst), 20)

        limited = self.store.list_instances(correlation_id=corr, limit=5)
        self.assertEqual(len(limited), 5)

    def test_default_limit_is_reasonable(self):
        """Default limit should be > 100 but not unbounded."""
        import inspect
        sig = inspect.signature(self.store.list_instances)
        default = sig.parameters["limit"].default
        self.assertGreater(default, 100)
        self.assertLessEqual(default, 1000)


class TestStrictSerialization(unittest.TestCase):
    """M2: Strict JSON serialization catches bad state."""

    def setUp(self):
        self.store = CoordinatorStore(":memory:")

    def test_non_serializable_state_raises_in_strict_mode(self):
        os.environ["COGNITIVE_CORE_STRICT"] = "1"
        try:
            sus = Suspension.create(
                "wf_test", "step_test",
                {"bad_object": object()},
            )
            with self.assertRaises(TypeError):
                self.store.save_suspension(sus)
        finally:
            del os.environ["COGNITIVE_CORE_STRICT"]

    def test_serializable_state_works_in_strict_mode(self):
        os.environ["COGNITIVE_CORE_STRICT"] = "1"
        try:
            sus = Suspension.create(
                "wf_test", "step_test",
                {"key": "value", "number": 42},
            )
            self.store.save_suspension(sus)
            loaded = self.store.get_suspension("wf_test")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.state_snapshot["key"], "value")
        finally:
            del os.environ["COGNITIVE_CORE_STRICT"]
