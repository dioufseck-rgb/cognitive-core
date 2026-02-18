"""
Cognitive Core — P-010: Eval-Gated Deployment Tests
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from typing import Any

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "eval_gate.py")
_spec = importlib.util.spec_from_file_location("engine.eval_gate", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.eval_gate"] = _mod
_spec.loader.exec_module(_mod)

EvalGate = _mod.EvalGate
Baseline = _mod.Baseline
GateVerdict = _mod.GateVerdict
RegressionDetail = _mod.RegressionDetail


# ═══════════════════════════════════════════════════════════════════
# Mock EvalResult that mimics the real one's interface
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MockEvalResult:
    pack_name: str = "product_return"
    workflow: str = "product_return"
    domain: str = "electronics_return"
    total: int = 25
    passed: int = 22
    _gate_results: dict = field(default_factory=dict)
    _all_pass: bool = True

    def gate_results(self) -> dict[str, dict[str, Any]]:
        return self._gate_results

    @property
    def all_gates_pass(self) -> bool:
        return self._all_pass


def make_passing_result(scores=None) -> MockEvalResult:
    """Create a MockEvalResult where all gates pass."""
    s = scores or {
        "schema_valid": 100.0,
        "classification_accuracy": 95.0,
        "investigation_quality": 80.0,
        "confidence_calibration": 85.0,
        "fail_closed": 100.0,
        "generate_compliance": 90.0,
    }
    gates = {
        name: {"actual": val, "threshold": val - 10, "passed": True}
        for name, val in s.items()
    }
    return MockEvalResult(
        _gate_results=gates,
        _all_pass=True,
        passed=22,
    )


def make_failing_result(failing_gates=None) -> MockEvalResult:
    """Create a MockEvalResult where specified gates fail."""
    failing = failing_gates or ["investigation_quality"]
    scores = {
        "schema_valid": 100.0,
        "classification_accuracy": 95.0,
        "investigation_quality": 80.0,
        "confidence_calibration": 85.0,
        "fail_closed": 100.0,
        "generate_compliance": 90.0,
    }
    gates = {}
    for name, val in scores.items():
        passed = name not in failing
        threshold = val - 10 if passed else val + 10
        gates[name] = {"actual": val, "threshold": threshold, "passed": passed}

    return MockEvalResult(
        _gate_results=gates,
        _all_pass=False,
        passed=18,
    )


class _GateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.gate = EvalGate(baselines_dir=self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir)


class TestEvalGatePass(_GateTestCase):
    """test_eval_gate_pass — all gates pass, no regression → exit 0"""

    def test_pass_no_baseline(self):
        result = make_passing_result()
        verdict = self.gate.evaluate(result, model_version="gemini-2.0-flash")

        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.exit_code, 0)
        self.assertTrue(verdict.gates_passed)
        self.assertFalse(verdict.baseline_exists)

    def test_pass_with_baseline(self):
        # Create baseline first
        result1 = make_passing_result()
        self.gate.evaluate(result1, model_version="gemini-2.0-flash")

        # Run again with same scores
        result2 = make_passing_result()
        verdict = self.gate.evaluate(result2, model_version="gemini-2.0-flash-002")

        self.assertTrue(verdict.approved)
        self.assertTrue(verdict.baseline_exists)
        self.assertTrue(verdict.regression_passed)


class TestEvalGateFailThreshold(_GateTestCase):
    """test_eval_gate_fail_threshold — gate below threshold → exit 1"""

    def test_fail_threshold(self):
        result = make_failing_result(["investigation_quality"])
        verdict = self.gate.evaluate(result, model_version="gemini-2.0-flash")

        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.exit_code, 1)
        self.assertFalse(verdict.gates_passed)
        self.assertTrue(any("investigation_quality" in r for r in verdict.reasons))


class TestEvalGateFailRegression(_GateTestCase):
    """test_eval_gate_fail_regression — passes threshold but dropped >5% from baseline → exit 1"""

    def test_regression_detected(self):
        # Establish high baseline
        high_scores = {
            "schema_valid": 100.0,
            "classification_accuracy": 95.0,
            "investigation_quality": 90.0,
            "confidence_calibration": 85.0,
            "fail_closed": 100.0,
            "generate_compliance": 95.0,
        }
        baseline_result = make_passing_result(high_scores)
        self.gate.evaluate(baseline_result, model_version="v1")

        # Run with dropped score (still passes absolute threshold but regressed >5%)
        dropped_scores = {
            "schema_valid": 100.0,
            "classification_accuracy": 95.0,
            "investigation_quality": 82.0,  # Dropped 8% from 90%
            "confidence_calibration": 85.0,
            "fail_closed": 100.0,
            "generate_compliance": 95.0,
        }
        dropped_result = make_passing_result(dropped_scores)
        verdict = self.gate.evaluate(dropped_result, model_version="v2")

        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.exit_code, 1)
        self.assertTrue(verdict.gates_passed)  # Absolute thresholds pass
        self.assertFalse(verdict.regression_passed)  # But regression fails

        # Check regression detail
        reg = [r for r in verdict.regressions if r.gate == "investigation_quality"]
        self.assertEqual(len(reg), 1)
        self.assertTrue(reg[0].regressed)
        self.assertAlmostEqual(reg[0].delta, -8.0, places=0)


class TestBaselinePersisted(_GateTestCase):
    """test_baseline_persisted — passing run saves baseline JSON."""

    def test_baseline_saved(self):
        result = make_passing_result()
        self.gate.evaluate(result, model_version="gemini-2.0-flash")

        baseline = self.gate.load_baseline("product_return")
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.model_version, "gemini-2.0-flash")
        self.assertEqual(baseline.pack_name, "product_return")
        self.assertIn("schema_valid", baseline.gate_scores)

    def test_baseline_not_saved_on_failure(self):
        result = make_failing_result()
        self.gate.evaluate(result, model_version="bad_model")

        baseline = self.gate.load_baseline("product_return")
        self.assertIsNone(baseline)

    def test_baseline_file_is_valid_json(self):
        result = make_passing_result()
        self.gate.evaluate(result, model_version="v1")

        path = self.gate._baseline_path("product_return")
        with open(path) as f:
            data = json.load(f)

        self.assertIn("pack_name", data)
        self.assertIn("gate_scores", data)
        self.assertIn("model_version", data)
        self.assertIn("timestamp", data)


class TestBaselineComparison(_GateTestCase):
    """test_baseline_comparison — new run compared against stored baseline."""

    def test_improvement_passes(self):
        # Baseline
        scores_v1 = {
            "schema_valid": 95.0,
            "classification_accuracy": 85.0,
            "investigation_quality": 75.0,
            "confidence_calibration": 80.0,
            "fail_closed": 100.0,
            "generate_compliance": 85.0,
        }
        self.gate.evaluate(make_passing_result(scores_v1), model_version="v1")

        # Improved scores
        scores_v2 = {
            "schema_valid": 98.0,
            "classification_accuracy": 90.0,
            "investigation_quality": 80.0,
            "confidence_calibration": 85.0,
            "fail_closed": 100.0,
            "generate_compliance": 90.0,
        }
        verdict = self.gate.evaluate(make_passing_result(scores_v2), model_version="v2")

        self.assertTrue(verdict.approved)
        self.assertTrue(verdict.regression_passed)

    def test_small_drop_passes(self):
        """Drop within 5% tolerance passes."""
        scores_v1 = {
            "schema_valid": 100.0,
            "classification_accuracy": 95.0,
            "investigation_quality": 80.0,
            "confidence_calibration": 85.0,
            "fail_closed": 100.0,
            "generate_compliance": 90.0,
        }
        self.gate.evaluate(make_passing_result(scores_v1), model_version="v1")

        # 3% drop — within tolerance
        scores_v2 = dict(scores_v1)
        scores_v2["investigation_quality"] = 77.0  # -3%
        verdict = self.gate.evaluate(make_passing_result(scores_v2), model_version="v2")

        self.assertTrue(verdict.approved)
        self.assertTrue(verdict.regression_passed)


class TestModelVersionInVerdict(_GateTestCase):
    """Model version tracked in verdict."""

    def test_version_in_verdict(self):
        result = make_passing_result()
        verdict = self.gate.evaluate(result, model_version="gemini-2.0-flash-002")
        self.assertEqual(verdict.model_version, "gemini-2.0-flash-002")


class TestVerdictSummary(_GateTestCase):
    """Summary string is readable."""

    def test_summary_contains_key_info(self):
        result = make_passing_result()
        verdict = self.gate.evaluate(result, model_version="v1")
        summary = verdict.summary()

        self.assertIn("APPROVED", summary)
        self.assertIn("v1", summary)
        self.assertIn("schema_valid", summary)

    def test_rejection_summary(self):
        result = make_failing_result()
        verdict = self.gate.evaluate(result, model_version="bad")
        summary = verdict.summary()

        self.assertIn("REJECTED", summary)


class TestBaselineOverwrite(_GateTestCase):
    """New passing run overwrites old baseline."""

    def test_overwrite(self):
        self.gate.evaluate(make_passing_result(), model_version="v1")
        b1 = self.gate.load_baseline("product_return")
        self.assertEqual(b1.model_version, "v1")

        self.gate.evaluate(make_passing_result(), model_version="v2")
        b2 = self.gate.load_baseline("product_return")
        self.assertEqual(b2.model_version, "v2")
        self.assertGreater(b2.timestamp, b1.timestamp)


if __name__ == "__main__":
    unittest.main()
