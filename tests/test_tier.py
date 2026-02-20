"""
Cognitive Core — Tier Escalation Invariant Tests (H-001)

Exhaustive: 16 combinations (4 declared × 4 override) plus edge cases.
"""

import os
import sys
import unittest

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "tier.py")
_spec = importlib.util.spec_from_file_location("engine.tier", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.tier"] = _mod
_spec.loader.exec_module(_mod)

resolve_effective_tier = _mod.resolve_effective_tier
TierInvariantViolation = _mod.TierInvariantViolation
validate_tier = _mod.validate_tier
tier_at_least = _mod.tier_at_least
TIER_ORDER = _mod.TIER_ORDER


class TestExhaustiveCombinations(unittest.TestCase):
    """16 test cases: every declared × override combination."""

    TIERS = ["auto", "spot_check", "gate", "hold"]

    def test_all_combinations_upward_only(self):
        """For every (declared, override) pair, effective >= declared."""
        for declared in self.TIERS:
            for override in self.TIERS:
                with self.subTest(declared=declared, override=override):
                    effective, source = resolve_effective_tier(declared, override)
                    self.assertGreaterEqual(
                        TIER_ORDER[effective], TIER_ORDER[declared],
                        f"Downgrade! declared={declared}, override={override}, effective={effective}"
                    )

    def test_all_combinations_max_wins(self):
        """Effective tier is always the max of declared and override."""
        for declared in self.TIERS:
            for override in self.TIERS:
                with self.subTest(declared=declared, override=override):
                    effective, _ = resolve_effective_tier(declared, override)
                    expected_val = max(TIER_ORDER[declared], TIER_ORDER[override])
                    self.assertEqual(
                        TIER_ORDER[effective], expected_val,
                        f"declared={declared}, override={override}, effective={effective}"
                    )


class TestNoOverride(unittest.TestCase):
    """When no overrides, effective == declared."""

    def test_auto_stays_auto(self):
        eff, src = resolve_effective_tier("auto")
        self.assertEqual(eff, "auto")
        self.assertEqual(src, "declared")

    def test_hold_stays_hold(self):
        eff, src = resolve_effective_tier("hold")
        self.assertEqual(eff, "hold")
        self.assertEqual(src, "declared")


class TestMultipleOverrides(unittest.TestCase):
    """Multiple override candidates — highest wins."""

    def test_highest_wins(self):
        eff, src = resolve_effective_tier(
            "auto", "spot_check", "gate", "spot_check",
            source_labels=["breaker", "kill_switch", "eval_gate"],
        )
        self.assertEqual(eff, "gate")
        self.assertEqual(src, "kill_switch")

    def test_declared_wins_when_highest(self):
        eff, src = resolve_effective_tier(
            "hold", "auto", "spot_check",
            source_labels=["breaker", "config"],
        )
        self.assertEqual(eff, "hold")
        self.assertEqual(src, "declared")

    def test_all_none_overrides_ignored(self):
        eff, src = resolve_effective_tier("gate", None, None, "")
        self.assertEqual(eff, "gate")
        self.assertEqual(src, "declared")


class TestSourceLabels(unittest.TestCase):
    """Override source tracking."""

    def test_circuit_breaker_source(self):
        eff, src = resolve_effective_tier(
            "auto", "gate",
            source_labels=["circuit_breaker"],
        )
        self.assertEqual(src, "circuit_breaker")

    def test_default_label_when_none(self):
        eff, src = resolve_effective_tier("auto", "gate")
        self.assertEqual(src, "override_0")


class TestInvalidInputs(unittest.TestCase):
    """Edge cases and invalid inputs."""

    def test_invalid_declared_raises(self):
        with self.assertRaises(ValueError):
            resolve_effective_tier("invalid_tier")

    def test_invalid_override_ignored(self):
        eff, src = resolve_effective_tier("auto", "bogus_tier")
        self.assertEqual(eff, "auto")
        self.assertEqual(src, "declared")

    def test_empty_string_override_ignored(self):
        eff, src = resolve_effective_tier("spot_check", "")
        self.assertEqual(eff, "spot_check")


class TestConfigReloadCannotDowngrade(unittest.TestCase):
    """Simulates config reload attempting to lower tier while breaker is active."""

    def test_config_reload_with_active_breaker(self):
        """Domain YAML reloaded with 'auto', but circuit breaker is at 'gate'."""
        # New config says auto, but breaker says gate
        eff, src = resolve_effective_tier(
            "auto", "gate",
            source_labels=["circuit_breaker"],
        )
        self.assertEqual(eff, "gate")
        self.assertEqual(src, "circuit_breaker")

    def test_config_reload_cannot_lower_below_declared(self):
        """Even if all overrides try 'auto', declared 'gate' holds."""
        eff, src = resolve_effective_tier(
            "gate", "auto", "auto", "auto",
            source_labels=["reload", "breaker", "eval"],
        )
        self.assertEqual(eff, "gate")
        self.assertEqual(src, "declared")


class TestHelpers(unittest.TestCase):
    """validate_tier and tier_at_least utilities."""

    def test_validate_valid(self):
        for t in ["auto", "spot_check", "gate", "hold"]:
            self.assertTrue(validate_tier(t))

    def test_validate_invalid(self):
        self.assertFalse(validate_tier("bogus"))
        self.assertFalse(validate_tier(""))

    def test_tier_at_least(self):
        self.assertTrue(tier_at_least("gate", "auto"))
        self.assertTrue(tier_at_least("gate", "gate"))
        self.assertFalse(tier_at_least("auto", "gate"))
        self.assertTrue(tier_at_least("hold", "spot_check"))


if __name__ == "__main__":
    unittest.main()
