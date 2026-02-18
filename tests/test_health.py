"""
Cognitive Core — P-008: Health Check Tests
"""

import json
import os
import sys
import time
import unittest
import urllib.request

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_health_path = os.path.join(_base, "engine", "health.py")
_spec = importlib.util.spec_from_file_location("engine.health", _health_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.health"] = _mod
_spec.loader.exec_module(_mod)

HealthChecker = _mod.HealthChecker
CheckResult = _mod.CheckResult
run_health_server = _mod.run_health_server


class TestHealthReturns200(unittest.TestCase):
    """test_health_returns_200 — always returns 200 if process is alive"""

    def test_health_always_ok(self):
        checker = HealthChecker()
        result = checker.check_health()
        self.assertEqual(result["status"], "ok")
        self.assertIn("timestamp", result)


class TestReadyChecksSubsystems(unittest.TestCase):
    """Readiness checks run all registered checks."""

    def test_ready_all_pass(self):
        checker = HealthChecker()
        checker.register("db", lambda: (True, "connected"))
        checker.register("llm", lambda: (True, "provider ready"))

        result = checker.check_ready()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["checks"]["db"]["status"], "ok")
        self.assertEqual(result["checks"]["llm"]["status"], "ok")

    def test_ready_llm_fails(self):
        checker = HealthChecker()
        checker.register("db", lambda: (True, "connected"))
        checker.register("llm", lambda: (False, "timeout"))

        result = checker.check_ready()
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checks"]["llm"]["status"], "fail")
        self.assertEqual(result["checks"]["db"]["status"], "ok")

    def test_ready_db_fails(self):
        checker = HealthChecker()
        checker.register("db", lambda: (False, "connection refused"))

        result = checker.check_ready()
        self.assertEqual(result["status"], "fail")
        self.assertIn("connection refused", result["checks"]["db"]["detail"])

    def test_ready_check_exception(self):
        """Check that raises exception is caught and reported as fail."""
        checker = HealthChecker()
        def bad_check():
            raise ConnectionError("host unreachable")
        checker.register("service", bad_check)

        result = checker.check_ready()
        self.assertEqual(result["status"], "fail")
        self.assertIn("host unreachable", result["checks"]["service"]["error"])

    def test_ready_no_checks_is_ok(self):
        checker = HealthChecker()
        result = checker.check_ready()
        self.assertEqual(result["status"], "ok")


class TestStartupChecks(unittest.TestCase):
    """Startup checks validate one-time conditions."""

    def test_startup_config_valid(self):
        checker = HealthChecker()
        checker.register_startup("config", lambda: (True, "valid"))

        result = checker.check_startup()
        self.assertEqual(result["status"], "ok")

    def test_startup_config_invalid(self):
        checker = HealthChecker()
        checker.register_startup("config", lambda: (False, "missing API key"))

        result = checker.check_startup()
        self.assertEqual(result["status"], "fail")
        self.assertIn("missing API key", result["checks"]["config"]["detail"])


class TestCheckLatency(unittest.TestCase):
    """Checks record latency."""

    def test_latency_recorded(self):
        checker = HealthChecker()
        def slow_check():
            time.sleep(0.05)
            return True, "ok"
        checker.register("slow", slow_check)

        result = checker.check_ready()
        latency = result["checks"]["slow"]["latency_ms"]
        self.assertGreater(latency, 40)
        self.assertLess(latency, 200)


class TestCheckResult(unittest.TestCase):
    """CheckResult serializes correctly."""

    def test_to_dict_ok(self):
        r = CheckResult(name="db", status="ok", latency_ms=2.3, detail="connected")
        d = r.to_dict()
        self.assertEqual(d["status"], "ok")
        self.assertEqual(d["latency_ms"], 2.3)
        self.assertEqual(d["detail"], "connected")
        self.assertNotIn("error", d)

    def test_to_dict_fail(self):
        r = CheckResult(name="llm", status="fail", latency_ms=5001.0, error="timeout")
        d = r.to_dict()
        self.assertEqual(d["status"], "fail")
        self.assertIn("error", d)


class TestHttpEndpoints(unittest.TestCase):
    """HTTP server exposes health, ready, startup endpoints."""

    @classmethod
    def setUpClass(cls):
        cls.checker = HealthChecker()
        cls.checker.register("test_service", lambda: (True, "ok"))
        cls.checker.register_startup("test_config", lambda: (True, "valid"))
        cls.server = run_health_server(cls.checker, port=18765, daemon=True)
        time.sleep(0.1)  # Give server time to start

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _get(self, path):
        url = f"http://localhost:18765{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_endpoint(self):
        code, body = self._get("/health")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")

    def test_ready_endpoint(self):
        code, body = self._get("/ready")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")
        self.assertIn("test_service", body["checks"])

    def test_startup_endpoint(self):
        code, body = self._get("/startup")
        self.assertEqual(code, 200)
        self.assertEqual(body["status"], "ok")

    def test_not_found(self):
        code, body = self._get("/nonexistent")
        self.assertEqual(code, 404)

    def test_ready_returns_503_on_failure(self):
        """When a check fails, /ready returns 503."""
        self.checker.register("failing", lambda: (False, "down"))
        code, body = self._get("/ready")
        self.assertEqual(code, 503)
        self.assertEqual(body["status"], "fail")
        # Clean up
        del self.checker._checks["failing"]


if __name__ == "__main__":
    unittest.main()
