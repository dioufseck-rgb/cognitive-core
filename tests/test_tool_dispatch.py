"""
Tests for Tool Dispatch + Act Safety (H-005, H-006, H-007, H-008).
"""

import os
import sys
import time
import json
import hashlib
import unittest
from unittest.mock import MagicMock

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "tool_dispatch.py")
_spec = importlib.util.spec_from_file_location("engine.tool_dispatch", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.tool_dispatch"] = _mod
_spec.loader.exec_module(_mod)

ToolDispatcher = _mod.ToolDispatcher
ToolMode = _mod.ToolMode
WriteBoundaryViolation = _mod.WriteBoundaryViolation
UnknownToolError = _mod.UnknownToolError
StepTimeoutError = _mod.StepTimeoutError
run_with_timeout = _mod.run_with_timeout
IntegrityChecker = _mod.IntegrityChecker
IdempotencyManager = _mod.IdempotencyManager
IdempotencyStatus = _mod.IdempotencyStatus


# ═══════════════════════════════════════════════════════════════
# H-005: Tool Registry + Write Boundary
# ═══════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.td = ToolDispatcher()
        self.td.register("fetch_data", lambda source="": {"data": source}, ToolMode.READ)
        self.td.register("write_db", lambda record="": {"written": record}, ToolMode.WRITE)

    def test_register_and_list(self):
        tools = self.td.list_tools()
        self.assertEqual(len(tools), 2)

    def test_list_by_mode(self):
        reads = self.td.list_tools(ToolMode.READ)
        writes = self.td.list_tools(ToolMode.WRITE)
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(writes), 1)

    def test_get_tool(self):
        t = self.td.get_tool("fetch_data")
        self.assertIsNotNone(t)
        self.assertEqual(t.mode, ToolMode.READ)

    def test_get_unknown_tool(self):
        self.assertIsNone(self.td.get_tool("nonexistent"))


class TestWriteBoundary(unittest.TestCase):

    def setUp(self):
        self.td = ToolDispatcher()
        self.td.register("fetch", lambda: {"ok": True}, ToolMode.READ)
        self.td.register("update", lambda: {"ok": True}, ToolMode.WRITE)

    def test_read_from_classify_ok(self):
        result = self.td.dispatch("fetch", {}, "classify")
        self.assertTrue(result["ok"])

    def test_read_from_investigate_ok(self):
        result = self.td.dispatch("fetch", {}, "investigate")
        self.assertTrue(result["ok"])

    def test_read_from_act_ok(self):
        result = self.td.dispatch("fetch", {}, "act")
        self.assertTrue(result["ok"])

    def test_write_from_act_ok(self):
        result = self.td.dispatch("update", {}, "act")
        self.assertTrue(result["ok"])

    def test_write_from_classify_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "classify")

    def test_write_from_investigate_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "investigate")

    def test_write_from_retrieve_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "retrieve")

    def test_write_from_think_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "think")

    def test_write_from_verify_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "verify")

    def test_write_from_generate_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "generate")

    def test_write_from_challenge_blocked(self):
        with self.assertRaises(WriteBoundaryViolation):
            self.td.dispatch("update", {}, "challenge")

    def test_unknown_tool_raises(self):
        with self.assertRaises(UnknownToolError):
            self.td.dispatch("nonexistent", {}, "act")

    def test_call_log(self):
        self.td.dispatch("fetch", {}, "classify")
        log = self.td.call_log
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["tool"], "fetch")
        self.assertTrue(log[0]["success"])

    def test_call_log_on_error(self):
        try:
            self.td.dispatch("update", {}, "classify")
        except WriteBoundaryViolation:
            pass
        # Boundary violations don't make it to the call log (raised before call)
        # Only actual function call failures do

    def test_function_error_logged(self):
        self.td.register("bad_tool", lambda: 1/0, ToolMode.READ)
        with self.assertRaises(ZeroDivisionError):
            self.td.dispatch("bad_tool", {}, "retrieve")
        log = self.td.call_log
        self.assertEqual(len(log), 1)
        self.assertFalse(log[0]["success"])


# ═══════════════════════════════════════════════════════════════
# H-006: Step-Level Hard Timeouts
# ═══════════════════════════════════════════════════════════════

class TestStepTimeouts(unittest.TestCase):

    def test_fast_function_ok(self):
        result = run_with_timeout(lambda: 42, timeout_seconds=5, step_name="fast")
        self.assertEqual(result, 42)

    def test_slow_function_times_out(self):
        with self.assertRaises(StepTimeoutError) as ctx:
            run_with_timeout(
                lambda: time.sleep(10),
                timeout_seconds=0.2,
                step_name="slow_step",
            )
        self.assertEqual(ctx.exception.step_name, "slow_step")
        self.assertAlmostEqual(ctx.exception.timeout_seconds, 0.2)

    def test_function_exception_propagated(self):
        def bad():
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            run_with_timeout(bad, timeout_seconds=5, step_name="bad")

    def test_timeout_error_fields(self):
        try:
            run_with_timeout(lambda: time.sleep(5), timeout_seconds=0.1, step_name="test_step")
        except StepTimeoutError as e:
            self.assertEqual(e.step_name, "test_step")
            self.assertGreater(e.elapsed, 0.09)

    def test_function_with_args(self):
        result = run_with_timeout(
            lambda x, y: x + y,
            args=(3, 4),
            timeout_seconds=5,
        )
        self.assertEqual(result, 7)

    def test_function_with_kwargs(self):
        result = run_with_timeout(
            lambda x=0, y=0: x * y,
            kwargs={"x": 5, "y": 6},
            timeout_seconds=5,
        )
        self.assertEqual(result, 30)


# ═══════════════════════════════════════════════════════════════
# H-007: Input Integrity Checksums
# ═══════════════════════════════════════════════════════════════

class TestIntegrityChecker(unittest.TestCase):

    def setUp(self):
        self.checker = IntegrityChecker()

    def test_hash_content(self):
        content = b"Hello, this is a PDF document"
        record = self.checker.hash_content(content, "test.pdf")
        expected = hashlib.sha256(content).hexdigest()
        self.assertEqual(record.content_hash, expected)
        self.assertEqual(record.byte_count, len(content))
        self.assertEqual(record.source_name, "test.pdf")

    def test_verify_match(self):
        content = b"document bytes"
        h = hashlib.sha256(content).hexdigest()
        self.assertTrue(self.checker.verify(content, h))

    def test_verify_mismatch(self):
        self.assertFalse(self.checker.verify(b"original", hashlib.sha256(b"tampered").hexdigest()))

    def test_multiple_documents(self):
        self.checker.hash_content(b"doc1", "file1.pdf")
        self.checker.hash_content(b"doc2", "file2.csv")
        self.checker.hash_content(b"doc3", "file1.pdf")  # same name, different content
        records = self.checker.get_records()
        self.assertEqual(len(records), 3)
        by_name = self.checker.get_records("file1.pdf")
        self.assertEqual(len(by_name), 2)

    def test_with_audit_trail(self):
        mock_audit = MagicMock()
        checker = IntegrityChecker(audit_trail=mock_audit)
        checker.hash_content(b"data", "file.pdf", trace_id="t1")
        mock_audit.record.assert_called_once()

    def test_empty_content(self):
        record = self.checker.hash_content(b"", "empty.txt")
        self.assertEqual(record.byte_count, 0)
        self.assertEqual(record.content_hash, hashlib.sha256(b"").hexdigest())


# ═══════════════════════════════════════════════════════════════
# H-008: Idempotency Key Management
# ═══════════════════════════════════════════════════════════════

class TestIdempotencyManager(unittest.TestCase):

    def setUp(self):
        self.mgr = IdempotencyManager(":memory:")

    def tearDown(self):
        self.mgr.close()

    def test_compute_key_deterministic(self):
        k1 = IdempotencyManager.compute_key("inst_1", "step_a", {"x": 1})
        k2 = IdempotencyManager.compute_key("inst_1", "step_a", {"x": 1})
        self.assertEqual(k1, k2)

    def test_compute_key_different_inputs(self):
        k1 = IdempotencyManager.compute_key("inst_1", "step_a", {"x": 1})
        k2 = IdempotencyManager.compute_key("inst_1", "step_a", {"x": 2})
        self.assertNotEqual(k1, k2)

    def test_acquire_new_key(self):
        key = "test_key_1"
        result = self.mgr.acquire(key, "inst_1", "step_a")
        self.assertIsNone(result)  # Acquired — proceed

    def test_acquire_existing_key(self):
        key = "test_key_2"
        self.mgr.acquire(key, "inst_1", "step_a")
        result = self.mgr.acquire(key, "inst_1", "step_a")
        self.assertIsNotNone(result)  # Already exists
        self.assertEqual(result.status, IdempotencyStatus.IN_PROGRESS)

    def test_complete_and_retrieve(self):
        key = "test_key_3"
        self.mgr.acquire(key, "inst_1", "step_a")
        self.mgr.complete(key, {"transferred": True, "amount": 500})

        record = self.mgr.check(key)
        self.assertEqual(record.status, IdempotencyStatus.COMPLETED)
        self.assertTrue(record.result["transferred"])

    def test_duplicate_returns_cached(self):
        key = "test_key_4"
        self.mgr.acquire(key, "inst_1", "step_a")
        self.mgr.complete(key, {"result": "done"})

        # Second attempt returns cached
        existing = self.mgr.acquire(key, "inst_1", "step_a")
        self.assertEqual(existing.status, IdempotencyStatus.COMPLETED)
        self.assertEqual(existing.result["result"], "done")

    def test_fail_and_release(self):
        key = "test_key_5"
        self.mgr.acquire(key, "inst_1", "step_a")
        self.mgr.fail(key, "network error")

        record = self.mgr.check(key)
        self.assertEqual(record.status, IdempotencyStatus.FAILED)

        # Release allows retry
        self.mgr.release(key)
        self.assertIsNone(self.mgr.check(key))

    def test_get_executions(self):
        self.mgr.acquire("k1", "inst_1", "step_a")
        self.mgr.complete("k1", {"r": 1})
        self.mgr.acquire("k2", "inst_1", "step_b")
        self.mgr.complete("k2", {"r": 2})
        self.mgr.acquire("k3", "inst_2", "step_a")

        execs = self.mgr.get_executions("inst_1")
        self.assertEqual(len(execs), 2)

    def test_check_nonexistent(self):
        self.assertIsNone(self.mgr.check("no_such_key"))


if __name__ == "__main__":
    unittest.main()
