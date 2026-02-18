"""
Cognitive Core — S-007: Kill Switch Tests
"""

import importlib.util
import os
import sys
import threading
import time
import unittest

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_ks_path = os.path.join(_base, "engine", "kill_switch.py")
_spec = importlib.util.spec_from_file_location("engine.kill_switch", _ks_path)
_ks_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.kill_switch"] = _ks_mod
_spec.loader.exec_module(_ks_mod)

KillSwitchManager = _ks_mod.KillSwitchManager
KillSwitchTripped = _ks_mod.KillSwitchTripped
get_kill_switches = _ks_mod.get_kill_switches
reset_kill_switches = _ks_mod.reset_kill_switches


class TestActSwitch(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_act_disabled(self):
        self.assertFalse(self.ks.is_act_disabled())
        self.ks.disable_act("incident #1234", by="ops_admin")
        self.assertTrue(self.ks.is_act_disabled())

    def test_act_reenabled(self):
        self.ks.disable_act("test")
        self.ks.enable_act(by="ops_admin")
        self.assertFalse(self.ks.is_act_disabled())

    def test_check_act_raises(self):
        self.ks.disable_act("test reason")
        with self.assertRaises(KillSwitchTripped) as ctx:
            self.ks.check_act()
        self.assertIn("act_disabled", str(ctx.exception))
        self.assertIn("test reason", str(ctx.exception))

    def test_check_act_passes_when_enabled(self):
        self.ks.check_act()  # Should not raise


class TestDelegationSwitch(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_delegation_disabled(self):
        self.ks.disable_delegation("runaway delegation chain")
        self.assertTrue(self.ks.is_delegation_disabled())

    def test_check_delegation_raises(self):
        self.ks.disable_delegation("loop detected")
        with self.assertRaises(KillSwitchTripped):
            self.ks.check_delegation()


class TestDomainSwitch(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_disable_specific_domain(self):
        self.ks.disable_domain("card_dispute", "SME review in progress")
        self.assertTrue(self.ks.is_domain_disabled("card_dispute"))
        self.assertFalse(self.ks.is_domain_disabled("electronics_return"))

    def test_reenable_domain(self):
        self.ks.disable_domain("card_dispute")
        self.ks.enable_domain("card_dispute")
        self.assertFalse(self.ks.is_domain_disabled("card_dispute"))

    def test_check_domain_raises(self):
        self.ks.disable_domain("card_dispute", "bad classification rate")
        with self.assertRaises(KillSwitchTripped) as ctx:
            self.ks.check_domain("card_dispute")
        self.assertIn("card_dispute", str(ctx.exception))

    def test_check_unregistered_domain_passes(self):
        self.ks.check_domain("never_registered")  # Should not raise


class TestWorkflowSwitch(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_disable_workflow(self):
        self.ks.disable_workflow("sar_investigation", "pending audit")
        self.assertTrue(self.ks.is_workflow_disabled("sar_investigation"))
        self.assertFalse(self.ks.is_workflow_disabled("product_return"))

    def test_check_workflow_raises(self):
        self.ks.disable_workflow("sar_investigation")
        with self.assertRaises(KillSwitchTripped):
            self.ks.check_workflow("sar_investigation")


class TestPolicySwitch(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_disable_policy(self):
        self.ks.disable_policy("fraud_pattern_triggers_aml", "false positive rate too high")
        self.assertTrue(self.ks.is_policy_disabled("fraud_pattern_triggers_aml"))

    def test_check_policy_raises(self):
        self.ks.disable_policy("fraud_pattern_triggers_aml")
        with self.assertRaises(KillSwitchTripped):
            self.ks.check_policy("fraud_pattern_triggers_aml")


class TestPreflightCheck(unittest.TestCase):
    def setUp(self):
        self.ks = KillSwitchManager()

    def test_all_clear(self):
        self.ks.preflight_check(
            workflow="product_return", domain="electronics_return",
            has_act=True, has_delegation=True,
            policies=["return_fraud_indicators"],
        )  # Should not raise

    def test_act_blocked(self):
        self.ks.disable_act("test")
        with self.assertRaises(KillSwitchTripped):
            self.ks.preflight_check(has_act=True)

    def test_act_not_checked_when_not_needed(self):
        self.ks.disable_act("test")
        # has_act=False, so Act switch is not checked
        self.ks.preflight_check(has_act=False, workflow="w", domain="d")

    def test_delegation_blocked(self):
        self.ks.disable_delegation("test")
        with self.assertRaises(KillSwitchTripped):
            self.ks.preflight_check(has_delegation=True)

    def test_domain_blocked(self):
        self.ks.disable_domain("bad_domain")
        with self.assertRaises(KillSwitchTripped):
            self.ks.preflight_check(domain="bad_domain")

    def test_workflow_blocked(self):
        self.ks.disable_workflow("bad_workflow")
        with self.assertRaises(KillSwitchTripped):
            self.ks.preflight_check(workflow="bad_workflow")

    def test_policy_blocked(self):
        self.ks.disable_policy("bad_policy")
        with self.assertRaises(KillSwitchTripped):
            self.ks.preflight_check(policies=["ok_policy", "bad_policy"])


class TestStatus(unittest.TestCase):
    def test_empty_status(self):
        ks = KillSwitchManager()
        s = ks.status()
        self.assertFalse(s["any_active"])
        self.assertFalse(s["act_disabled"]["enabled"])
        self.assertEqual(s["disabled_domains"], {})

    def test_active_status(self):
        ks = KillSwitchManager()
        ks.disable_act("reason_a", by="admin")
        ks.disable_domain("card_dispute", "reason_d")
        s = ks.status()
        self.assertTrue(s["any_active"])
        self.assertTrue(s["act_disabled"]["enabled"])
        self.assertEqual(s["act_disabled"]["reason"], "reason_a")
        self.assertEqual(s["act_disabled"]["toggled_by"], "admin")
        self.assertIn("card_dispute", s["disabled_domains"])

    def test_disabled_domains_only_shows_active(self):
        ks = KillSwitchManager()
        ks.disable_domain("d1")
        ks.disable_domain("d2")
        ks.enable_domain("d1")
        s = ks.status()
        self.assertNotIn("d1", s["disabled_domains"])
        self.assertIn("d2", s["disabled_domains"])


class TestResetAll(unittest.TestCase):
    def test_reset_clears_everything(self):
        ks = KillSwitchManager()
        ks.disable_act("a")
        ks.disable_delegation("b")
        ks.disable_domain("d1", "c")
        ks.disable_workflow("w1", "d")
        ks.disable_policy("p1", "e")

        ks.reset_all(by="admin")

        self.assertFalse(ks.is_act_disabled())
        self.assertFalse(ks.is_delegation_disabled())
        self.assertFalse(ks.is_domain_disabled("d1"))
        self.assertFalse(ks.is_workflow_disabled("w1"))
        self.assertFalse(ks.is_policy_disabled("p1"))
        self.assertFalse(ks.status()["any_active"])


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_toggle(self):
        ks = KillSwitchManager()
        errors = []

        def toggle_act():
            try:
                for _ in range(100):
                    ks.disable_act("test")
                    ks.is_act_disabled()
                    ks.enable_act()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=toggle_act) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestSingleton(unittest.TestCase):
    def setUp(self):
        reset_kill_switches()

    def test_singleton(self):
        a = get_kill_switches()
        b = get_kill_switches()
        self.assertIs(a, b)

    def test_reset(self):
        a = get_kill_switches()
        reset_kill_switches()
        b = get_kill_switches()
        self.assertIsNot(a, b)


class TestExceptionAttributes(unittest.TestCase):
    def test_exception_has_switch_name(self):
        e = KillSwitchTripped("act_disabled", "some reason")
        self.assertEqual(e.switch_name, "act_disabled")
        self.assertEqual(e.reason, "some reason")

    def test_exception_str(self):
        e = KillSwitchTripped("act_disabled", "reason")
        self.assertIn("act_disabled", str(e))
        self.assertIn("reason", str(e))


if __name__ == "__main__":
    unittest.main()
