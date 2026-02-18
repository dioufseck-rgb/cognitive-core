"""
Cognitive Core — P-011: Cost Tracking Tests
"""

import os
import sys
import unittest

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "cost.py")
_spec = importlib.util.spec_from_file_location("engine.cost", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.cost"] = _mod
_spec.loader.exec_module(_mod)

CostTracker = _mod.CostTracker
ModelPricing = _mod.ModelPricing
load_pricing = _mod.load_pricing


class TestTokensCounted(unittest.TestCase):
    """test_tokens_counted — every call records input and output tokens."""

    def test_tokens_recorded(self):
        tracker = CostTracker()
        record = tracker.record_call(
            model="gemini-2.0-flash",
            input_tokens=1500,
            output_tokens=500,
            step_name="classify",
        )
        self.assertEqual(record.input_tokens, 1500)
        self.assertEqual(record.output_tokens, 500)
        self.assertEqual(record.step_name, "classify")

    def test_multiple_calls(self):
        tracker = CostTracker()
        tracker.record_call("gemini-2.0-flash", 1000, 300, "classify")
        tracker.record_call("gemini-2.0-flash", 2000, 800, "investigate")
        tracker.record_call("gemini-2.0-flash", 500, 200, "generate")

        summary = tracker.summary()
        self.assertEqual(summary["total_calls"], 3)
        self.assertEqual(summary["total_input_tokens"], 3500)
        self.assertEqual(summary["total_output_tokens"], 1300)


class TestCostCalculated(unittest.TestCase):
    """test_cost_calculated — tokens × pricing = correct cost."""

    def test_gemini_flash_cost(self):
        pricing = ModelPricing(
            model="gemini-2.0-flash",
            input_per_million=0.10,
            output_per_million=0.40,
        )
        # 1M input tokens = $0.10, 1M output tokens = $0.40
        cost = pricing.cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 0.50, places=2)

    def test_small_call_cost(self):
        pricing = ModelPricing(
            model="gemini-2.0-flash",
            input_per_million=0.10,
            output_per_million=0.40,
        )
        # 1500 input, 500 output
        cost = pricing.cost(1500, 500)
        expected = (1500 / 1e6) * 0.10 + (500 / 1e6) * 0.40
        self.assertAlmostEqual(cost, expected, places=8)

    def test_gpt4o_cost(self):
        pricing = ModelPricing(
            model="gpt-4o",
            input_per_million=2.50,
            output_per_million=10.00,
        )
        cost = pricing.cost(10000, 5000)
        expected = (10000 / 1e6) * 2.50 + (5000 / 1e6) * 10.00
        self.assertAlmostEqual(cost, expected, places=6)


class TestWorkflowCostAggregated(unittest.TestCase):
    """test_workflow_cost_aggregated — sum of step costs = total."""

    def test_sum_matches_total(self):
        tracker = CostTracker()
        tracker.record_call("gemini-2.0-flash", 1000, 300, "classify")
        tracker.record_call("gemini-2.0-flash", 2000, 800, "investigate")
        tracker.record_call("gemini-2.0-flash", 500, 200, "generate")

        summary = tracker.summary()
        step_total = sum(s["cost_usd"] for s in summary["by_step"].values())
        self.assertAlmostEqual(step_total, summary["total_cost_usd"], places=6)


class TestCostByStep(unittest.TestCase):
    """Cost breakdown by step."""

    def test_by_step(self):
        tracker = CostTracker()
        tracker.record_call("gemini-2.0-flash", 1000, 300, "classify")
        tracker.record_call("gemini-2.0-flash", 2000, 800, "investigate")
        tracker.record_call("gemini-2.0-flash", 1500, 600, "investigate")

        summary = tracker.summary()
        self.assertIn("classify", summary["by_step"])
        self.assertIn("investigate", summary["by_step"])
        self.assertEqual(summary["by_step"]["classify"]["calls"], 1)
        self.assertEqual(summary["by_step"]["investigate"]["calls"], 2)
        self.assertEqual(summary["by_step"]["investigate"]["input_tokens"], 3500)


class TestCostByModel(unittest.TestCase):
    """Cost breakdown by model."""

    def test_by_model(self):
        tracker = CostTracker()
        tracker.record_call("gemini-2.0-flash", 1000, 300, "classify")
        tracker.record_call("gpt-4o", 1000, 300, "investigate")

        summary = tracker.summary()
        self.assertIn("gemini-2.0-flash", summary["by_model"])
        self.assertIn("gpt-4o", summary["by_model"])


class TestCostInLog(unittest.TestCase):
    """CallRecord has cost_usd field."""

    def test_record_has_cost(self):
        tracker = CostTracker()
        record = tracker.record_call("gemini-2.0-flash", 1000, 500, "classify")
        self.assertIsInstance(record.cost_usd, float)
        self.assertGreaterEqual(record.cost_usd, 0.0)


class TestPricingLoadsFromConfig(unittest.TestCase):
    """Pricing loads from actual llm_config.yaml."""

    def test_pricing_loaded(self):
        pricing = load_pricing()
        self.assertIn("gemini-2.0-flash", pricing)
        self.assertIn("gpt-4o", pricing)

        gf = pricing["gemini-2.0-flash"]
        self.assertAlmostEqual(gf.input_per_million, 0.10, places=2)
        self.assertAlmostEqual(gf.output_per_million, 0.40, places=2)

    def test_unknown_model_zero_cost(self):
        tracker = CostTracker()
        record = tracker.record_call("unknown-model-xyz", 1000, 500, "test")
        self.assertEqual(record.cost_usd, 0.0)


class TestTotalCostProperty(unittest.TestCase):
    """total_cost property works."""

    def test_total_cost(self):
        tracker = CostTracker()
        tracker.record_call("gemini-2.0-flash", 1000, 300, "s1")
        tracker.record_call("gemini-2.0-flash", 2000, 600, "s2")
        self.assertGreater(tracker.total_cost, 0.0)
        self.assertEqual(tracker.call_count, 2)


class TestEmptyTracker(unittest.TestCase):
    """Empty tracker returns zeros."""

    def test_empty(self):
        tracker = CostTracker()
        summary = tracker.summary()
        self.assertEqual(summary["total_calls"], 0)
        self.assertEqual(summary["total_cost_usd"], 0.0)
        self.assertEqual(summary["total_input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
