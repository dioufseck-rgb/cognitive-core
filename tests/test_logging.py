"""
Cognitive Core — P-002: Structured Logging Tests

Tests:
  - test_trace_id_generated — every logger gets unique trace_id
  - test_trace_id_propagates_to_delegation — child inherits parent_trace_id
  - test_log_entry_schema — every entry has required fields
  - test_llm_call_logged — LLM calls produce entries with latency
  - test_governance_decision_logged — tier evaluation produces entry
  - test_log_level_filtering — DEBUG shows prompts, INFO hides them
  - test_json_parseable — every log line is valid JSON
  - test_step_start_creates_span — span_id generated per step
  - test_parse_error_logged — parse failures produce WARNING
  - test_route_decision_logged — routing produces entry
  - test_child_logger_different_trace — child has new trace_id
  - test_workflow_lifecycle — start/end logged
"""

import io
import json
import logging
import os
import sys
import unittest

# Import logging module directly
import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_log_path = os.path.join(_base, "engine", "logging.py")
_spec = importlib.util.spec_from_file_location("engine.logging", _log_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.logging"] = _mod
_spec.loader.exec_module(_mod)

StructuredLogger = _mod.StructuredLogger
JSONFormatter = _mod.JSONFormatter
configure_logging = _mod.configure_logging
generate_trace_id = _mod.generate_trace_id
generate_span_id = _mod.generate_span_id
get_logger = _mod.get_logger


def _capture_logs(logger_instance, level="DEBUG"):
    """Set up a StringIO capture on the cognitive_core logger."""
    buf = io.StringIO()
    root_logger = configure_logging(level=level, stream=buf)
    return buf, root_logger


def _parse_log_lines(buf):
    """Parse all JSON lines from buffer."""
    buf.seek(0)
    lines = []
    for line in buf.readlines():
        line = line.strip()
        if line:
            lines.append(json.loads(line))
    return lines


class TestTraceIdGenerated(unittest.TestCase):
    """test_trace_id_generated — every logger gets unique trace_id"""

    def test_unique_trace_ids(self):
        ids = {generate_trace_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_logger_has_trace_id(self):
        logger = StructuredLogger(workflow="test", domain="test")
        self.assertIsNotNone(logger.trace_id)
        self.assertEqual(len(logger.trace_id), 32)  # 32 hex chars (OTel format)

    def test_two_loggers_different_trace(self):
        l1 = StructuredLogger()
        l2 = StructuredLogger()
        self.assertNotEqual(l1.trace_id, l2.trace_id)


class TestTraceIdPropagates(unittest.TestCase):
    """test_trace_id_propagates_to_delegation — child inherits parent_trace_id"""

    def test_child_has_parent_trace_id(self):
        parent = StructuredLogger(workflow="product_return", domain="electronics_return")
        child = parent.child(workflow="sar_investigation")

        self.assertNotEqual(parent.trace_id, child.trace_id)
        self.assertEqual(child.parent_trace_id, parent.trace_id)
        self.assertIsNone(parent.parent_trace_id)

    def test_child_inherits_domain(self):
        parent = StructuredLogger(workflow="product_return", domain="electronics_return")
        child = parent.child(workflow="sar_investigation")
        self.assertEqual(child.domain, "electronics_return")

    def test_child_overrides_domain(self):
        parent = StructuredLogger(workflow="product_return", domain="electronics_return")
        child = parent.child(workflow="sar", domain="aml")
        self.assertEqual(child.domain, "aml")


class TestLogEntrySchema(unittest.TestCase):
    """test_log_entry_schema — every entry has required fields"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "DEBUG")
        self.logger = StructuredLogger(workflow="test_wf", domain="test_dom")

    def test_required_fields_present(self):
        self.logger.on_step_start("classify", "classify", 1)
        entries = _parse_log_lines(self.buf)

        self.assertGreater(len(entries), 0)
        entry = entries[0]

        # Required fields
        self.assertIn("timestamp", entry)
        self.assertIn("level", entry)
        self.assertIn("trace_id", entry)
        self.assertIn("action", entry)
        self.assertIn("service.name", entry)
        self.assertIn("workflow", entry)
        self.assertIn("domain", entry)

        # Values correct
        self.assertEqual(entry["trace_id"], self.logger.trace_id)
        self.assertEqual(entry["workflow"], "test_wf")
        self.assertEqual(entry["domain"], "test_dom")
        self.assertEqual(entry["service.name"], "cognitive_core")
        self.assertEqual(entry["action"], "step_start")

    def test_otel_compatible_trace_id(self):
        """trace_id should be 32 hex chars (OTel W3C format)."""
        self.logger.on_step_start("test", "classify", 1)
        entries = _parse_log_lines(self.buf)
        trace_id = entries[0]["trace_id"]
        self.assertEqual(len(trace_id), 32)
        int(trace_id, 16)  # Should be valid hex


class TestLlmCallLogged(unittest.TestCase):
    """test_llm_call_logged — LLM calls produce entries with latency"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "DEBUG")
        self.logger = StructuredLogger(workflow="test_wf", domain="test_dom")

    def test_llm_start_and_end(self):
        self.logger.on_step_start("classify", "classify", 1)
        self.logger.on_llm_start("classify", 5000)
        self.logger.on_llm_end("classify", 2000, 1.5)

        entries = _parse_log_lines(self.buf)
        actions = [e["action"] for e in entries]

        self.assertIn("llm_start", actions)
        self.assertIn("llm_end", actions)

        # llm_start at DEBUG level has prompt_chars
        llm_start = [e for e in entries if e["action"] == "llm_start"][0]
        self.assertEqual(llm_start["prompt_chars"], 5000)
        self.assertEqual(llm_start["step_name"], "classify")

        # llm_end has latency
        llm_end = [e for e in entries if e["action"] == "llm_end"][0]
        self.assertEqual(llm_end["response_chars"], 2000)
        self.assertEqual(llm_end["latency_ms"], 1500.0)
        self.assertEqual(llm_end["step_name"], "classify")


class TestGovernanceDecisionLogged(unittest.TestCase):
    """test_governance_decision_logged — tier evaluation produces entry"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "INFO")
        self.logger = StructuredLogger(workflow="product_return", domain="electronics_return")

    def test_governance_decision(self):
        self.logger.on_governance_decision(
            domain="electronics_return",
            declared_tier="auto",
            applied_tier="gate",
            quality_gate_result="fail",
            reason="confidence below floor",
        )

        entries = _parse_log_lines(self.buf)
        gov = [e for e in entries if e["action"] == "governance_decision"]
        self.assertEqual(len(gov), 1)
        self.assertEqual(gov[0]["declared_tier"], "auto")
        self.assertEqual(gov[0]["applied_tier"], "gate")
        self.assertEqual(gov[0]["quality_gate_result"], "fail")
        self.assertEqual(gov[0]["reason"], "confidence below floor")


class TestLogLevelFiltering(unittest.TestCase):
    """test_log_level_filtering — DEBUG shows prompts, INFO hides them"""

    def test_debug_shows_llm_start(self):
        buf, _ = _capture_logs(None, "DEBUG")
        logger = StructuredLogger(workflow="test")
        logger.on_llm_start("classify", 5000)
        entries = _parse_log_lines(buf)
        actions = [e["action"] for e in entries]
        self.assertIn("llm_start", actions)

    def test_info_hides_llm_start(self):
        buf, _ = _capture_logs(None, "INFO")
        logger = StructuredLogger(workflow="test")
        logger.on_llm_start("classify", 5000)
        entries = _parse_log_lines(buf)
        # llm_start is DEBUG level — should not appear at INFO
        actions = [e["action"] for e in entries]
        self.assertNotIn("llm_start", actions)

    def test_info_shows_step_start(self):
        buf, _ = _capture_logs(None, "INFO")
        logger = StructuredLogger(workflow="test")
        logger.on_step_start("classify", "classify", 1)
        entries = _parse_log_lines(buf)
        actions = [e["action"] for e in entries]
        self.assertIn("step_start", actions)

    def test_warning_shows_only_errors(self):
        buf, _ = _capture_logs(None, "WARNING")
        logger = StructuredLogger(workflow="test")
        logger.on_step_start("classify", "classify", 1)
        logger.on_llm_start("classify", 5000)
        logger.on_llm_end("classify", 2000, 1.5)
        logger.on_parse_error("classify", "bad json")

        entries = _parse_log_lines(buf)
        actions = [e["action"] for e in entries]

        # Only parse_error (WARNING) should appear
        self.assertEqual(actions, ["parse_error"])


class TestJsonParseable(unittest.TestCase):
    """test_json_parseable — every log line is valid JSON"""

    def test_all_lines_valid_json(self):
        buf, _ = _capture_logs(None, "DEBUG")
        logger = StructuredLogger(workflow="test", domain="test")

        # Emit every type of event
        logger.on_workflow_start()
        logger.on_step_start("classify", "classify", 1)
        logger.on_llm_start("classify", 5000)
        logger.on_llm_end("classify", 2000, 1.5)
        logger.on_parse_result("classify", "classify", {"confidence": 0.95, "category": "defective"})
        logger.on_parse_error("classify", "test error")
        logger.on_route_decision("classify", "investigate", "conditional", "confidence > 0.7")
        logger.on_retrieve_start("gather", "get_order")
        logger.on_retrieve_end("gather", "get_order", "success", 45.2)
        logger.on_governance_decision("test", "auto", "gate", "fail", "low confidence")
        logger.on_delegation_start("product_return", "sar", "policy1", "blocking", "child123")
        logger.on_delegation_complete("child123", "success", 12.5)
        logger.on_workflow_end("success", 45.0, 5)

        buf.seek(0)
        for i, line in enumerate(buf.readlines()):
            line = line.strip()
            if line:
                try:
                    parsed = json.loads(line)
                    self.assertIsInstance(parsed, dict)
                except json.JSONDecodeError:
                    self.fail(f"Line {i} is not valid JSON: {line[:200]}")


class TestStepStartCreatesSpan(unittest.TestCase):
    """test_step_start_creates_span — span_id generated per step"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "DEBUG")
        self.logger = StructuredLogger(workflow="test")

    def test_span_id_in_step_events(self):
        self.logger.on_step_start("classify", "classify", 1)
        self.logger.on_llm_start("classify", 1000)
        self.logger.on_llm_end("classify", 500, 0.5)

        entries = _parse_log_lines(self.buf)
        step_entry = [e for e in entries if e["action"] == "step_start"][0]
        llm_end = [e for e in entries if e["action"] == "llm_end"][0]

        self.assertIn("span_id", step_entry)
        self.assertIn("span_id", llm_end)
        self.assertEqual(step_entry["span_id"], llm_end["span_id"])

    def test_different_steps_different_spans(self):
        self.logger.on_step_start("classify", "classify", 1)
        self.logger.on_step_start("investigate", "investigate", 1)

        entries = _parse_log_lines(self.buf)
        spans = [e["span_id"] for e in entries if e["action"] == "step_start"]
        self.assertEqual(len(spans), 2)
        self.assertNotEqual(spans[0], spans[1])

    def test_span_id_is_16_hex(self):
        """span_id should be 16 hex chars (OTel format)."""
        self.logger.on_step_start("test", "classify", 1)
        entries = _parse_log_lines(self.buf)
        span_id = entries[0]["span_id"]
        self.assertEqual(len(span_id), 16)
        int(span_id, 16)  # Should be valid hex


class TestParseErrorLogged(unittest.TestCase):
    """test_parse_error_logged — parse failures produce WARNING"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "WARNING")
        self.logger = StructuredLogger(workflow="test")

    def test_parse_error_at_warning(self):
        self.logger.on_parse_error("classify", "No JSON object found")
        entries = _parse_log_lines(self.buf)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["action"], "parse_error")
        self.assertEqual(entries[0]["level"], "WARNING")
        self.assertIn("No JSON object found", entries[0]["error"])


class TestRouteDecisionLogged(unittest.TestCase):
    """test_route_decision_logged — routing produces entry"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "INFO")
        self.logger = StructuredLogger(workflow="test")

    def test_route_decision(self):
        self.logger.on_route_decision(
            "classify_return_type", "investigate_claim",
            "conditional", "confidence > 0.7"
        )
        entries = _parse_log_lines(self.buf)
        route = [e for e in entries if e["action"] == "route_decision"]
        self.assertEqual(len(route), 1)
        self.assertEqual(route[0]["from_step"], "classify_return_type")
        self.assertEqual(route[0]["to_step"], "investigate_claim")
        self.assertEqual(route[0]["decision_type"], "conditional")


class TestChildLoggerDifferentTrace(unittest.TestCase):
    """test_child_logger_different_trace — child has new trace_id"""

    def test_child_different_trace(self):
        parent = StructuredLogger(workflow="parent")
        child = parent.child(workflow="child")

        self.assertNotEqual(parent.trace_id, child.trace_id)
        self.assertEqual(child.parent_trace_id, parent.trace_id)

    def test_grandchild_chain(self):
        parent = StructuredLogger(workflow="p")
        child = parent.child(workflow="c")
        grandchild = child.child(workflow="gc")

        self.assertEqual(grandchild.parent_trace_id, child.trace_id)
        self.assertNotEqual(grandchild.trace_id, child.trace_id)
        self.assertNotEqual(grandchild.trace_id, parent.trace_id)


class TestWorkflowLifecycle(unittest.TestCase):
    """test_workflow_lifecycle — start/end logged"""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "INFO")
        self.logger = StructuredLogger(workflow="product_return", domain="electronics_return")

    def test_workflow_start_end(self):
        self.logger.on_workflow_start()
        self.logger.on_workflow_end("success", 45.0, 5)

        entries = _parse_log_lines(self.buf)
        actions = [e["action"] for e in entries]

        self.assertIn("workflow_start", actions)
        self.assertIn("workflow_end", actions)

        end = [e for e in entries if e["action"] == "workflow_end"][0]
        self.assertEqual(end["status"], "success")
        self.assertEqual(end["elapsed_s"], 45.0)
        self.assertEqual(end["steps_completed"], 5)


class TestDelegationEvents(unittest.TestCase):
    """Delegation start/complete events logged with child trace."""

    def setUp(self):
        self.buf, _ = _capture_logs(None, "INFO")
        self.logger = StructuredLogger(workflow="product_return")

    def test_delegation_start_complete(self):
        child = self.logger.child(workflow="sar_investigation")

        self.logger.on_delegation_start(
            "product_return", "sar_investigation",
            "fraud_triggers_sar", "blocking",
            child.trace_id,
        )
        self.logger.on_delegation_complete(child.trace_id, "success", 30.5)

        entries = _parse_log_lines(self.buf)
        start = [e for e in entries if e["action"] == "delegation_start"]
        complete = [e for e in entries if e["action"] == "delegation_complete"]

        self.assertEqual(len(start), 1)
        self.assertEqual(start[0]["child_trace_id"], child.trace_id)
        self.assertEqual(start[0]["policy"], "fraud_triggers_sar")
        self.assertEqual(start[0]["mode"], "blocking")

        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["child_trace_id"], child.trace_id)
        self.assertEqual(complete[0]["status"], "success")


class TestParseResultMetrics(unittest.TestCase):
    """Parse results extract key metrics at INFO, full output at DEBUG."""

    def test_info_has_confidence(self):
        buf, _ = _capture_logs(None, "INFO")
        logger = StructuredLogger(workflow="test")
        logger.on_parse_result("classify", "classify", {
            "confidence": 0.95,
            "category": "defective_product",
            "reasoning": "long text...",
        })
        entries = _parse_log_lines(buf)
        result = [e for e in entries if e["action"] == "parse_result"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["confidence"], 0.95)
        self.assertEqual(result[0]["category"], "defective_product")
        # Full output NOT at INFO
        self.assertNotIn("output", result[0])

    def test_debug_has_full_output(self):
        buf, _ = _capture_logs(None, "DEBUG")
        logger = StructuredLogger(workflow="test")
        logger.on_parse_result("classify", "classify", {
            "confidence": 0.95,
            "category": "defective_product",
        })
        entries = _parse_log_lines(buf)
        full = [e for e in entries if e["action"] == "parse_result_full"]
        self.assertEqual(len(full), 1)
        self.assertIn("output", full[0])


if __name__ == "__main__":
    unittest.main()
