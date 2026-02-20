"""Tests for Shadow Mode (H-004)."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mod_name, fname in [("engine.shadow", "engine/shadow.py")]:
    p = os.path.join(_base, fname)
    s = importlib.util.spec_from_file_location(mod_name, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[mod_name] = m
    s.loader.exec_module(m)

from engine.shadow import ShadowMode, ShadowActRecord


class TestShadowModeBasics(unittest.TestCase):

    def test_disabled_by_default(self):
        sm = ShadowMode()
        self.assertFalse(sm.enabled)

    def test_enabled_via_constructor(self):
        sm = ShadowMode(enabled=True)
        self.assertTrue(sm.enabled)

    def test_enabled_via_env(self):
        os.environ["CC_SHADOW_MODE"] = "true"
        sm = ShadowMode()
        self.assertTrue(sm.enabled)
        del os.environ["CC_SHADOW_MODE"]

    def test_should_skip_act_when_enabled(self):
        sm = ShadowMode(enabled=True)
        self.assertTrue(sm.should_skip_act("act"))

    def test_should_not_skip_non_act(self):
        sm = ShadowMode(enabled=True)
        self.assertFalse(sm.should_skip_act("classify"))
        self.assertFalse(sm.should_skip_act("investigate"))
        self.assertFalse(sm.should_skip_act("retrieve"))

    def test_should_not_skip_act_when_disabled(self):
        sm = ShadowMode(enabled=False)
        self.assertFalse(sm.should_skip_act("act"))


class TestShadowRecording(unittest.TestCase):

    def setUp(self):
        self.sm = ShadowMode(enabled=True)

    def test_record_shadow_act(self):
        record = self.sm.record_shadow_act(
            instance_id="inst_1",
            step_name="execute_transfer",
            proposed_action={"action": "transfer", "amount": 500, "to": "ACC_123"},
        )
        self.assertIsInstance(record, ShadowActRecord)
        self.assertEqual(record.instance_id, "inst_1")
        self.assertEqual(record.proposed_action["amount"], 500)
        self.assertTrue(record.shadow)

    def test_multiple_records(self):
        self.sm.record_shadow_act("i1", "step_a", {"a": 1})
        self.sm.record_shadow_act("i2", "step_b", {"b": 2})
        self.assertEqual(self.sm.shadow_count, 2)
        self.assertEqual(len(self.sm.shadow_records), 2)

    def test_get_shadow_result(self):
        result = self.sm.get_shadow_result("execute_transfer")
        self.assertEqual(result["action_taken"], "SHADOW_MODE_NO_ACTION")
        self.assertTrue(result["shadow"])
        self.assertEqual(result["step"], "execute_transfer")

    def test_record_with_audit_trail(self):
        mock_audit = MagicMock()
        sm = ShadowMode(enabled=True, audit_trail=mock_audit)
        sm.record_shadow_act("i1", "step", {"x": 1}, trace_id="trace_1")
        mock_audit.record.assert_called_once()
        call_kwargs = mock_audit.record.call_args
        self.assertEqual(call_kwargs.kwargs.get("event_type") or call_kwargs[1].get("event_type", call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None), "shadow_act")

    def test_record_without_audit_ok(self):
        sm = ShadowMode(enabled=True)
        record = sm.record_shadow_act("i1", "step", {"x": 1})
        self.assertIsNotNone(record)

    def test_stats(self):
        self.sm.record_shadow_act("i1", "s1", {})
        self.sm.record_shadow_act("i1", "s2", {})
        self.sm.record_shadow_act("i2", "s1", {})
        stats = self.sm.get_stats()
        self.assertTrue(stats["shadow_mode_enabled"])
        self.assertEqual(stats["shadow_acts_recorded"], 3)
        self.assertIn("i1", stats["instances"])
        self.assertIn("i2", stats["instances"])


if __name__ == "__main__":
    unittest.main()
