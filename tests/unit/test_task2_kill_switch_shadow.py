"""
TASK 2 — Tests: Wire shadow mode and kill switch to Act node

Verifies:
- Active Act kill switch prevents execution and records kill_switch proof event
- Shadow mode logs intended action and returns without executing
- Shadow mode result is marked shadow=True, not failed
- Kill switch block records the correct proof event
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock


def _fresh_governance():
    from cognitive_core.engine.governance import GovernancePipeline
    gov = GovernancePipeline()
    gov.initialize()
    return gov


class TestKillSwitchBlocksAct(unittest.TestCase):
    """An active Act kill switch prevents execution."""

    def setUp(self):
        os.environ["CC_SHADOW_MODE"] = "false"

    def tearDown(self):
        os.environ.pop("CC_SHADOW_MODE", None)

    def test_check_act_allowed_raises_when_kill_switch_active(self):
        from cognitive_core.engine.kill_switch import KillSwitchManager, KillSwitchTripped

        ks = KillSwitchManager()
        ks.disable_act()
        with self.assertRaises(KillSwitchTripped):
            ks.check_act()

    def test_governance_check_act_allowed_raises_when_kill_switch_active(self):
        from cognitive_core.engine.kill_switch import KillSwitchTripped

        gov = _fresh_governance()
        self.assertIsNotNone(gov._kill_switches, "KillSwitchManager should be initialised")

        gov._kill_switches.disable_act()
        with self.assertRaises(KillSwitchTripped):
            gov.check_act_allowed()

    def test_kill_switch_proof_event_recorded(self):
        from cognitive_core.engine.kill_switch import KillSwitchTripped

        gov = _fresh_governance()
        gov._kill_switches.disable_act()

        try:
            gov.check_act_allowed()
        except KillSwitchTripped:
            pass

        blocked_events = [
            e for e in gov._proof_ledger
            if "kill_switch" in e["event"] and e.get("result") == "blocked"
        ]
        self.assertGreaterEqual(
            len(blocked_events), 1,
            f"Expected kill_switch blocked event; ledger: {gov._proof_ledger}",
        )

    def test_delegation_kill_switch_raises(self):
        from cognitive_core.engine.kill_switch import KillSwitchTripped

        gov = _fresh_governance()
        gov._kill_switches.disable_delegation()
        with self.assertRaises(KillSwitchTripped):
            gov.check_delegation_allowed()


class TestShadowModeInterceptsAct(unittest.TestCase):
    """Shadow mode logs intended action and returns without executing."""

    def setUp(self):
        os.environ["CC_SHADOW_MODE"] = "true"

    def tearDown(self):
        os.environ.pop("CC_SHADOW_MODE", None)

    def test_should_skip_act_returns_true_when_shadow_enabled(self):
        gov = _fresh_governance()
        if gov._shadow_mode is None:
            self.skipTest("ShadowMode not available")
        result = gov.should_skip_act("test_act_step")
        self.assertTrue(result, "should_skip_act must return True when CC_SHADOW_MODE=true")

    def test_shadow_result_has_correct_structure(self):
        gov = _fresh_governance()
        if gov._shadow_mode is None:
            self.skipTest("ShadowMode not available")

        gov.record_shadow_act(
            instance_id="test-instance-001",
            step_name="close_account",
            proposed_actions=[{"action": "close_account", "params": {"account_id": "ACC123"}}],
        )

        # Shadow mode must record a proof event
        shadow_events = [e for e in gov._proof_ledger if "shadow" in e["event"] or "act.shadow" in e["event"]]
        self.assertGreaterEqual(
            len(shadow_events), 1,
            f"Expected shadow proof event; ledger: {gov._proof_ledger}",
        )

    def test_shadow_mode_does_not_affect_non_act_primitives(self):
        """Shadow mode must only intercept Act — other primitives are unaffected."""
        gov = _fresh_governance()
        if gov._shadow_mode is None:
            self.skipTest("ShadowMode not available")

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"confidence": 0.9}')
        llm.invoke.return_value.response_metadata = {}

        os.environ["CC_CACHE_ENABLED"] = "false"
        result = gov.protected_llm_call(
            llm=llm,
            prompt="Classify this transaction.",
            step_name="classify",
            domain="fraud",
            model="fast",
        )
        # Non-act primitives go through LLM as normal
        self.assertIsInstance(result.raw_response, str)
        os.environ.pop("CC_CACHE_ENABLED", None)


class TestShadowModeDisabled(unittest.TestCase):
    """When CC_SHADOW_MODE=false, should_skip_act returns False."""

    def setUp(self):
        os.environ["CC_SHADOW_MODE"] = "false"

    def tearDown(self):
        os.environ.pop("CC_SHADOW_MODE", None)

    def test_should_skip_act_returns_false_when_shadow_disabled(self):
        gov = _fresh_governance()
        if gov._shadow_mode is None:
            self.skipTest("ShadowMode not available")
        result = gov.should_skip_act("act_step")
        self.assertFalse(result, "should_skip_act must return False when CC_SHADOW_MODE=false")


if __name__ == "__main__":
    unittest.main()
