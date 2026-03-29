"""
TASK 3 — Tests: Enforce eval gate at coordinator startup

Verifies:
- Coordinator check_start_gates() calls the eval gate module
- An unregistered model ID causes the gate to block (EvalGateNotPassedError)
- A registered (approved) model passes normally
- CC_EVAL_GATE_ENFORCED=false bypasses the check with a warning (never silently)
- Proof ledger records eval_gate.checked event at every startup call
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest


def _fresh_governance():
    from cognitive_core.engine.governance import GovernancePipeline
    gov = GovernancePipeline()
    gov.initialize()
    return gov


class TestEvalGateModule(unittest.TestCase):
    """EvalGate.is_model_approved() checks the baselines directory."""

    def test_unregistered_model_returns_false(self):
        from cognitive_core.engine.eval_gate import EvalGate
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = EvalGate(baselines_dir=tmpdir)
            self.assertFalse(gate.is_model_approved("unknown-model-xyz"))

    def test_registered_model_returns_true(self):
        from cognitive_core.engine.eval_gate import EvalGate
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a fake baseline for "gemini-2.0-flash"
            baseline = {
                "pack_name": "fraud_triage",
                "model_version": "gemini-2.0-flash",
                "timestamp": 1700000000.0,
                "gate_scores": {"accuracy": 95.0},
                "total_cases": 100,
                "passed_cases": 95,
            }
            with open(os.path.join(tmpdir, "fraud_triage.json"), "w") as f:
                json.dump(baseline, f)

            gate = EvalGate(baselines_dir=tmpdir)
            self.assertTrue(gate.is_model_approved("gemini-2.0-flash"))

    def test_different_model_not_matched(self):
        from cognitive_core.engine.eval_gate import EvalGate
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = {
                "pack_name": "fraud_triage",
                "model_version": "gemini-2.0-flash",
                "timestamp": 1700000000.0,
                "gate_scores": {},
                "total_cases": 10,
                "passed_cases": 10,
            }
            with open(os.path.join(tmpdir, "fraud_triage.json"), "w") as f:
                json.dump(baseline, f)

            gate = EvalGate(baselines_dir=tmpdir)
            self.assertFalse(gate.is_model_approved("gpt-4o"))

    def test_empty_baselines_dir_returns_false(self):
        from cognitive_core.engine.eval_gate import EvalGate
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = EvalGate(baselines_dir=tmpdir)
            self.assertFalse(gate.is_model_approved("any-model"))

    def test_nonexistent_baselines_dir_returns_false(self):
        from cognitive_core.engine.eval_gate import EvalGate
        gate = EvalGate(baselines_dir="/tmp/nonexistent_evals_dir_xyz123")
        self.assertFalse(gate.is_model_approved("any-model"))


class TestEvalGateNotPassedError(unittest.TestCase):
    """EvalGateNotPassedError has correct model_id attribute."""

    def test_error_carries_model_id(self):
        from cognitive_core.engine.eval_gate import EvalGateNotPassedError
        err = EvalGateNotPassedError("gemini-unknown")
        self.assertEqual(err.model_id, "gemini-unknown")
        self.assertIn("gemini-unknown", str(err))


class TestEvalGateInCheckStartGates(unittest.TestCase):
    """check_start_gates() enforces eval gate when LLM_MODEL_ID is set."""

    def setUp(self):
        self._orig_model = os.environ.get("LLM_MODEL_ID")
        self._orig_enforced = os.environ.get("CC_EVAL_GATE_ENFORCED")

    def tearDown(self):
        if self._orig_model is None:
            os.environ.pop("LLM_MODEL_ID", None)
        else:
            os.environ["LLM_MODEL_ID"] = self._orig_model

        if self._orig_enforced is None:
            os.environ.pop("CC_EVAL_GATE_ENFORCED", None)
        else:
            os.environ["CC_EVAL_GATE_ENFORCED"] = self._orig_enforced

    def test_eval_gate_proof_event_always_recorded(self):
        """eval_gate.checked event fires even when no model ID is set."""
        os.environ.pop("LLM_MODEL_ID", None)
        os.environ["CC_EVAL_GATE_ENFORCED"] = "true"

        gov = _fresh_governance()
        gov.check_start_gates("test_workflow", "test_domain", {})

        gate_events = [e for e in gov._proof_ledger if "eval_gate" in e["event"]]
        self.assertGreaterEqual(len(gate_events), 1,
                                f"eval_gate event missing; ledger: {gov._proof_ledger}")

    def test_unregistered_model_blocks_start(self):
        """An unregistered model ID causes check_start_gates to return blocked=True."""
        os.environ["LLM_MODEL_ID"] = "unregistered-model-abc123"
        os.environ["CC_EVAL_GATE_ENFORCED"] = "true"

        gov = _fresh_governance()
        result = gov.check_start_gates("fraud_triage", "fraud", {})

        self.assertTrue(
            result["blocked"],
            "check_start_gates must block when model has no passing eval baseline",
        )

    def test_enforced_false_does_not_block(self):
        """CC_EVAL_GATE_ENFORCED=false bypasses the check (never silently)."""
        os.environ["LLM_MODEL_ID"] = "unregistered-model-abc123"
        os.environ["CC_EVAL_GATE_ENFORCED"] = "false"

        import logging
        with self.assertLogs("cognitive_core.governance", level="WARNING") as log_cm:
            gov = _fresh_governance()
            result = gov.check_start_gates("fraud_triage", "fraud", {})

        # Must NOT be blocked
        self.assertFalse(
            result["blocked"],
            "Eval gate bypass must not block when CC_EVAL_GATE_ENFORCED=false",
        )
        # Must log a warning (never silent bypass)
        warning_msgs = " ".join(log_cm.output)
        self.assertIn(
            "CC_EVAL_GATE_ENFORCED", warning_msgs,
            "A warning about CC_EVAL_GATE_ENFORCED=false must be logged",
        )

    def test_registered_model_passes(self):
        """A model with a passing baseline passes the eval gate."""
        from cognitive_core.engine.eval_gate import EvalGate
        from unittest.mock import patch

        os.environ["LLM_MODEL_ID"] = "approved-model-v1"
        os.environ["CC_EVAL_GATE_ENFORCED"] = "true"

        with patch.object(EvalGate, "is_model_approved", return_value=True):
            gov = _fresh_governance()
            result = gov.check_start_gates("fraud_triage", "fraud", {})

        self.assertFalse(
            result["blocked"],
            "check_start_gates must not block when model is approved",
        )
        passed_events = [e for e in gov._proof_ledger if e["event"] == "eval_gate.passed"]
        self.assertGreaterEqual(len(passed_events), 1)


if __name__ == "__main__":
    unittest.main()
