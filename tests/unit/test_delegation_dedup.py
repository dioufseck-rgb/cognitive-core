"""
Tests: Delegation deduplication — atomic claim prevents duplicate handler dispatch

Verifies:
- Two concurrent threads calling _evaluate_and_execute_delegations for the
  same instance result in exactly one delegation being fired, not two.
- The atomic log_action claim (idempotency key) is the gate, not the
  RLock + ledger recheck (which had a visibility race).
- Recursive call from a handler's _on_completed does not re-fire a
  delegation that was already claimed.
- A genuinely new delegation (different policy_name) fires correctly after
  an earlier one was claimed.
"""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_coordinator(tmp_path: str | None = None):
    """
    Create a real Coordinator with in-memory SQLite, minimal config,
    and LLM execution disabled (simulated mode).
    """
    import tempfile, os
    if tmp_path is None:
        tmp_path = tempfile.mktemp(suffix=".db")

    from cognitive_core.coordinator.runtime import Coordinator
    config = {
        "governance_tiers": {
            "auto": {"hitl": "none", "sample_rate": 0},
        },
        "delegations": [
            {
                "name": "test_delegation_alpha",
                "conditions": [
                    {
                        "domain": "test_domain",
                        "selector": "final_output",
                        "field": "trigger",
                        "operator": "eq",
                        "value": True,
                    }
                ],
                "target_workflow": "handler_workflow",
                "target_domain": "handler_domain",
                "contract": "",
                "mode": "fire_and_forget",
                "inputs": {},
            }
        ],
    }
    coord = Coordinator(config=config, db_path=tmp_path, verbose=False)
    return coord


def _make_instance(coord, domain="test_domain"):
    """Create and save a COMPLETED instance for testing delegation evaluation."""
    from cognitive_core.coordinator.types import InstanceState, InstanceStatus
    inst = InstanceState.create(
        workflow_type="test_workflow",
        domain=domain,
        governance_tier="auto",
    )
    inst.status = InstanceStatus.COMPLETED
    inst.updated_at = time.time()
    coord.store.save_instance(inst)
    return inst


def _final_state_with_trigger(trigger_value=True):
    """Workflow state that satisfies the test_delegation_alpha condition."""
    return {
        "input": {},
        "steps": [
            {
                "step_name": "generate_output",
                "primitive": "generate",
                "output": {"trigger": trigger_value},
                "confidence": 0.9,
            }
        ],
    }


# ── Tests ────────────────────────────────────────────────────────────────────

class TestDelegationAtomicClaim(unittest.TestCase):
    """Atomic idempotency claim prevents duplicate delegation dispatch."""

    def setUp(self):
        import tempfile
        self.db_path = tempfile.mktemp(suffix=".db")
        self.coord = _make_coordinator(self.db_path)

    def test_single_call_fires_once(self):
        """Single call to _evaluate_and_execute_delegations fires exactly one delegation."""
        inst = _make_instance(self.coord)
        final_state = _final_state_with_trigger(True)

        dispatched = []

        # Patch _execute_fire_and_forget_delegation to capture calls
        original = self.coord._execute_fire_and_forget_delegation
        def _capture(source, decision, wo):
            dispatched.append(decision.policy_name)
            # Don't actually start a handler workflow
        self.coord._execute_fire_and_forget_delegation = _capture

        self.coord._evaluate_and_execute_delegations(inst, final_state)

        self.assertEqual(dispatched, ["test_delegation_alpha"],
                         f"Expected exactly one dispatch, got: {dispatched}")

    def test_concurrent_threads_fire_exactly_once(self):
        """
        Two threads calling _evaluate_and_execute_delegations simultaneously
        must result in exactly one delegation dispatch, not two.

        This is the regression test for the duplication bug.
        """
        inst = _make_instance(self.coord)
        final_state = _final_state_with_trigger(True)

        dispatched = []
        lock = threading.Lock()
        barrier = threading.Barrier(2)  # synchronise both threads at the same point

        original = self.coord._execute_fire_and_forget_delegation
        def _capture(source, decision, wo):
            with lock:
                dispatched.append(decision.policy_name)
        self.coord._execute_fire_and_forget_delegation = _capture

        errors = []

        def _run():
            try:
                barrier.wait()  # both threads start at the same moment
                self.coord._evaluate_and_execute_delegations(inst, final_state)
            except Exception as e:
                errors.append(str(e))

        t1 = threading.Thread(target=_run)
        t2 = threading.Thread(target=_run)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertFalse(errors, f"Thread errors: {errors}")
        self.assertEqual(len(dispatched), 1,
                         f"Expected exactly 1 dispatch across 2 threads, got {len(dispatched)}: {dispatched}")

    def test_second_call_skipped_after_first_claimed(self):
        """
        A second call to _evaluate_and_execute_delegations after the first has
        claimed the delegation must skip it (idempotency via DB).
        """
        inst = _make_instance(self.coord)
        final_state = _final_state_with_trigger(True)

        dispatched = []

        def _capture(source, decision, wo):
            dispatched.append(decision.policy_name)
        self.coord._execute_fire_and_forget_delegation = _capture

        # First call — should fire
        self.coord._evaluate_and_execute_delegations(inst, final_state)
        # Second call (simulating recursive _on_completed) — should skip
        self.coord._evaluate_and_execute_delegations(inst, final_state)

        self.assertEqual(len(dispatched), 1,
                         f"Second call should be skipped, but got {len(dispatched)} dispatches")

    def test_no_trigger_no_dispatch(self):
        """Delegation does not fire when condition is not met."""
        inst = _make_instance(self.coord)
        final_state = _final_state_with_trigger(False)  # trigger=False, condition requires True

        dispatched = []
        def _capture(source, decision, wo):
            dispatched.append(decision.policy_name)
        self.coord._execute_fire_and_forget_delegation = _capture

        self.coord._evaluate_and_execute_delegations(inst, final_state)

        self.assertEqual(dispatched, [],
                         f"No delegation should fire when condition is False, got: {dispatched}")

    def test_claim_written_to_ledger_before_execution(self):
        """
        The idempotency key must be present in the ledger before the
        delegation handler is actually executed. This is what makes the
        concurrent thread protection work — the second thread sees the
        claim before the first thread finishes.
        """
        inst = _make_instance(self.coord)
        final_state = _final_state_with_trigger(True)

        ledger_state_at_execution = {}

        def _capture(source, decision, wo):
            # At this point, the claim should already be in the ledger
            ledger = self.coord.store.get_ledger(instance_id=source.instance_id)
            claim_entries = [
                e for e in ledger
                if e["action_type"] == "delegation_dispatched"
                and e.get("details", {}).get("policy") == decision.policy_name
            ]
            ledger_state_at_execution[decision.policy_name] = len(claim_entries)

        self.coord._execute_fire_and_forget_delegation = _capture
        self.coord._evaluate_and_execute_delegations(inst, final_state)

        self.assertEqual(
            ledger_state_at_execution.get("test_delegation_alpha", 0), 1,
            "Idempotency claim must be written to ledger BEFORE handler executes"
        )

    def test_different_instances_independent(self):
        """Two different instances can each fire the same delegation independently."""
        inst1 = _make_instance(self.coord)
        inst2 = _make_instance(self.coord)
        final_state = _final_state_with_trigger(True)

        dispatched = []
        def _capture(source, decision, wo):
            dispatched.append((source.instance_id, decision.policy_name))
        self.coord._execute_fire_and_forget_delegation = _capture

        self.coord._evaluate_and_execute_delegations(inst1, final_state)
        self.coord._evaluate_and_execute_delegations(inst2, final_state)

        self.assertEqual(len(dispatched), 2,
                         f"Each instance should fire independently, got: {dispatched}")
        instance_ids = {d[0] for d in dispatched}
        self.assertEqual(instance_ids, {inst1.instance_id, inst2.instance_id})


class TestDelegationLedgerIdempotency(unittest.TestCase):
    """log_action idempotency key is the correct atomic primitive."""

    def setUp(self):
        import tempfile
        self.db_path = tempfile.mktemp(suffix=".db")
        self.coord = _make_coordinator(self.db_path)
        self.inst = _make_instance(self.coord)

    def test_log_action_returns_false_on_duplicate_key(self):
        """log_action returns True on first write, False on duplicate key."""
        key = f"deleg:{self.inst.instance_id}:test_policy"

        result1 = self.coord.store.log_action(
            instance_id=self.inst.instance_id,
            correlation_id=self.inst.correlation_id,
            action_type="delegation_dispatched",
            details={"policy": "test_policy"},
            idempotency_key=key,
        )
        result2 = self.coord.store.log_action(
            instance_id=self.inst.instance_id,
            correlation_id=self.inst.correlation_id,
            action_type="delegation_dispatched",
            details={"policy": "test_policy"},
            idempotency_key=key,
        )

        self.assertTrue(result1, "First write should return True")
        self.assertFalse(result2, "Duplicate key write should return False")

    def test_concurrent_log_action_only_one_succeeds(self):
        """Under concurrent writes with the same key, exactly one returns True."""
        key = f"deleg:{self.inst.instance_id}:concurrent_policy"
        results = []
        lock = threading.Lock()
        barrier = threading.Barrier(5)

        def _write():
            barrier.wait()
            r = self.coord.store.log_action(
                instance_id=self.inst.instance_id,
                correlation_id=self.inst.correlation_id,
                action_type="delegation_dispatched",
                details={"policy": "concurrent_policy"},
                idempotency_key=key,
            )
            with lock:
                results.append(r)

        threads = [threading.Thread(target=_write) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        true_count = sum(1 for r in results if r)
        self.assertEqual(true_count, 1,
                         f"Exactly 1 of 5 concurrent writes should succeed, got {true_count}: {results}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
