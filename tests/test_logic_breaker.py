"""Tests for S-010: Logic Circuit Breakers."""

import os
import sys
import sqlite3
import tempfile
import threading
import types
import unittest

# Load module using exec to avoid dataclass + importlib issue
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _base)

_mod = types.ModuleType("engine.logic_breaker")
_mod.__file__ = os.path.join(_base, "engine", "logic_breaker.py")
sys.modules["engine.logic_breaker"] = _mod
with open(_mod.__file__) as _f:
    exec(_f.read(), _mod.__dict__)

LogicCircuitBreaker = _mod.LogicCircuitBreaker


class TestBasicRecording(unittest.TestCase):
    def test_normal_confidence_no_override(self):
        b = LogicCircuitBreaker(window_size=10)
        b.record("d", "p", confidence=0.9)
        self.assertIsNone(b.get_tier_override("d"))

    def test_single_low_no_override(self):
        b = LogicCircuitBreaker(window_size=10)
        b.record("d", "p", confidence=0.2)
        self.assertIsNone(b.get_tier_override("d"))

    def test_get_state(self):
        b = LogicCircuitBreaker(window_size=10)
        b.record("d", "p", confidence=0.3)
        s = b.get_state("d", "p")
        self.assertEqual(s["window_fill"], 1)

    def test_nonexistent_state(self):
        b = LogicCircuitBreaker(window_size=10)
        self.assertIsNone(b.get_state("x", "y"))


class TestSlidingWindow(unittest.TestCase):
    def test_window_caps_at_size(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(15):
            b.record("d", "p", confidence=0.9)
        self.assertEqual(b.get_state("d", "p")["window_fill"], 10)

    def test_window_slides(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10):
            b.record("d", "p", confidence=0.1)
        self.assertAlmostEqual(b.get_state("d", "p")["low_confidence_rate"], 1.0)
        for _ in range(10):
            b.record("d", "p", confidence=0.9)
        self.assertAlmostEqual(b.get_state("d", "p")["low_confidence_rate"], 0.0)


class TestGraduatedResponse(unittest.TestCase):
    def setUp(self):
        self.b = LogicCircuitBreaker(window_size=10, spot_check_threshold=0.50, gate_threshold=0.80)

    def test_below_spot_check(self):
        for _ in range(6): self.b.record("d", "p", confidence=0.9)
        for _ in range(4): self.b.record("d", "p", confidence=0.1)
        self.assertIsNone(self.b.get_tier_override("d"))

    def test_spot_check(self):
        for _ in range(4): self.b.record("d", "p", confidence=0.9)
        for _ in range(6): self.b.record("d", "p", confidence=0.1)
        self.assertEqual(self.b.get_tier_override("d"), "spot_check")

    def test_gate(self):
        for _ in range(1): self.b.record("d", "p", confidence=0.9)
        for _ in range(9): self.b.record("d", "p", confidence=0.1)
        self.assertEqual(self.b.get_tier_override("d"), "gate")

    def test_all_low_is_gate(self):
        for _ in range(10): self.b.record("d", "p", confidence=0.1)
        self.assertEqual(self.b.get_tier_override("d"), "gate")

    def test_exact_spot_check_boundary(self):
        for _ in range(5): self.b.record("d", "p", confidence=0.9)
        for _ in range(5): self.b.record("d", "p", confidence=0.1)
        self.assertEqual(self.b.get_tier_override("d"), "spot_check")

    def test_exact_gate_boundary(self):
        for _ in range(2): self.b.record("d", "p", confidence=0.9)
        for _ in range(8): self.b.record("d", "p", confidence=0.1)
        self.assertEqual(self.b.get_tier_override("d"), "gate")


class TestAutoRecovery(unittest.TestCase):
    def test_recovers_from_spot_check(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(6): b.record("d", "p", confidence=0.1)
        for _ in range(4): b.record("d", "p", confidence=0.9)
        self.assertEqual(b.get_tier_override("d"), "spot_check")
        for _ in range(10): b.record("d", "p", confidence=0.9)
        self.assertIsNone(b.get_tier_override("d"))

    def test_recovers_from_gate(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("d", "p", confidence=0.1)
        self.assertEqual(b.get_tier_override("d"), "gate")
        for _ in range(10): b.record("d", "p", confidence=0.9)
        self.assertIsNone(b.get_tier_override("d"))

    def test_gradual_recovery(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("d", "p", confidence=0.1)
        self.assertEqual(b.get_tier_override("d"), "gate")
        for _ in range(5): b.record("d", "p", confidence=0.9)
        self.assertEqual(b.get_tier_override("d"), "spot_check")


class TestMultiplePrimitives(unittest.TestCase):
    def test_independent(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("d", "classify", confidence=0.1)
        for _ in range(10): b.record("d", "investigate", confidence=0.9)
        self.assertAlmostEqual(b.get_state("d", "classify")["low_confidence_rate"], 1.0)
        self.assertAlmostEqual(b.get_state("d", "investigate")["low_confidence_rate"], 0.0)

    def test_domain_takes_worst(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("d", "classify", confidence=0.1)  # gate
        for _ in range(10): b.record("d", "investigate", confidence=0.9)  # clear
        self.assertEqual(b.get_tier_override("d"), "gate")

    def test_multiple_domains_independent(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("a", "p", confidence=0.1)
        for _ in range(10): b.record("b", "p", confidence=0.9)
        self.assertEqual(b.get_tier_override("a"), "gate")
        self.assertIsNone(b.get_tier_override("b"))


class TestCustomFloor(unittest.TestCase):
    def test_default_floor(self):
        b = LogicCircuitBreaker(window_size=10, confidence_floor=0.5)
        b.record("d", "p", confidence=0.6)
        self.assertAlmostEqual(b.get_state("d", "p")["low_confidence_rate"], 0.0)

    def test_override_floor(self):
        b = LogicCircuitBreaker(window_size=10, confidence_floor=0.5)
        b.record("d", "p", confidence=0.6, floor=0.7)
        self.assertAlmostEqual(b.get_state("d", "p")["low_confidence_rate"], 1.0)


class TestTripTracking(unittest.TestCase):
    def test_trip_count(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(6): b.record("d", "p", confidence=0.1)
        for _ in range(4): b.record("d", "p", confidence=0.9)
        s = b.get_state("d", "p")
        self.assertGreaterEqual(s["trip_count"], 1)

    def test_record_returns_change(self):
        b = LogicCircuitBreaker(window_size=10)
        results = []
        for _ in range(10):
            r = b.record("d", "p", confidence=0.1)
            if r is not None:
                results.append(r)
        self.assertGreater(len(results), 0)


class TestReset(unittest.TestCase):
    def test_reset_specific(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10): b.record("d", "classify", confidence=0.1)
        for _ in range(10): b.record("d", "investigate", confidence=0.1)
        b.reset("d", "classify")
        self.assertIsNone(b.get_state("d", "classify"))
        self.assertIsNotNone(b.get_state("d", "investigate"))

    def test_reset_domain(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10):
            b.record("d", "classify", confidence=0.1)
            b.record("d", "investigate", confidence=0.1)
        b.reset("d")
        self.assertIsNone(b.get_state("d", "classify"))
        self.assertIsNone(b.get_state("d", "investigate"))

    def test_reset_all(self):
        b = LogicCircuitBreaker(window_size=10)
        for _ in range(10):
            b.record("a", "p", confidence=0.1)
            b.record("b", "p", confidence=0.1)
        b.reset_all()
        self.assertEqual(b.get_all_states(), [])


class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b1 = LogicCircuitBreaker(window_size=10, db_path=db_path)
            for _ in range(8): b1.record("d", "p", confidence=0.1)
            for _ in range(2): b1.record("d", "p", confidence=0.9)
            s1 = b1.get_state("d", "p")

            b2 = LogicCircuitBreaker(window_size=10, db_path=db_path)
            s2 = b2.get_state("d", "p")
            self.assertIsNotNone(s2)
            self.assertEqual(s2["window_fill"], s1["window_fill"])
            self.assertAlmostEqual(s2["low_confidence_rate"], s1["low_confidence_rate"], places=2)
        finally:
            os.unlink(db_path)

    def test_reset_clears_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            b1 = LogicCircuitBreaker(window_size=10, db_path=db_path)
            for _ in range(10): b1.record("d", "p", confidence=0.1)
            b1.reset_all()
            b2 = LogicCircuitBreaker(window_size=10, db_path=db_path)
            self.assertEqual(b2.get_all_states(), [])
        finally:
            os.unlink(db_path)


class TestThreadSafety(unittest.TestCase):
    def test_concurrent_records(self):
        b = LogicCircuitBreaker(window_size=100)
        errors = []
        def work(domain, conf):
            try:
                for _ in range(50): b.record(domain, "p", confidence=conf)
            except Exception as e:
                errors.append(str(e))
        threads = [
            threading.Thread(target=work, args=("d1", 0.1)),
            threading.Thread(target=work, args=("d1", 0.9)),
            threading.Thread(target=work, args=("d2", 0.5)),
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_read_write(self):
        b = LogicCircuitBreaker(window_size=50)
        errors = []
        def writer():
            try:
                for _ in range(50): b.record("d", "p", confidence=0.3)
            except Exception as e: errors.append(str(e))
        def reader():
            try:
                for _ in range(50):
                    b.get_tier_override("d")
                    b.get_all_states()
            except Exception as e: errors.append(str(e))
        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start(); t2.start()
        t1.join(); t2.join()
        self.assertEqual(len(errors), 0)


class TestGetAllStates(unittest.TestCase):
    def test_all_states(self):
        b = LogicCircuitBreaker(window_size=10)
        b.record("d1", "classify", confidence=0.1)
        b.record("d2", "investigate", confidence=0.9)
        states = b.get_all_states()
        self.assertEqual(len(states), 2)
        domains = {s["domain"] for s in states}
        self.assertEqual(domains, {"d1", "d2"})


if __name__ == "__main__":
    unittest.main()
