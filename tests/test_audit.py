"""
Cognitive Core — P-009: Immutable Audit Trail Tests
"""

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(_base, "engine", "audit.py")
_spec = importlib.util.spec_from_file_location("engine.audit", _path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["engine.audit"] = _mod
_spec.loader.exec_module(_mod)

AuditTrail = _mod.AuditTrail
AuditEvent = _mod.AuditEvent
GENESIS_HASH = _mod.GENESIS_HASH
compute_event_hash = _mod.compute_event_hash


class _AuditTestCase(unittest.TestCase):
    """Base class that creates a temp audit DB per test."""
    def setUp(self):
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.trail = AuditTrail(self._tmpfile.name)

    def tearDown(self):
        self.trail.close()
        os.unlink(self._tmpfile.name)


class TestEveryPrimitiveAudited(_AuditTestCase):
    """test_every_primitive_audited — run workflow steps → audit has entry for each"""

    def test_primitive_events_recorded(self):
        trace = "trace_abc123"
        self.trail.record_primitive(
            trace, "classify_return_type", "classify",
            output_hash="sha256:aaa", model_version="gemini-2.0-flash",
            prompt_hash="sha256:bbb", confidence=0.92,
        )
        self.trail.record_primitive(
            trace, "investigate_claim", "investigate",
            output_hash="sha256:ccc", model_version="gemini-2.0-flash",
            prompt_hash="sha256:ddd", confidence=0.85,
        )
        self.trail.record_primitive(
            trace, "generate_response", "generate",
            output_hash="sha256:eee", model_version="gemini-2.0-flash",
            prompt_hash="sha256:fff",
        )

        events = self.trail.get_trail(trace)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].payload["step_name"], "classify_return_type")
        self.assertEqual(events[1].payload["step_name"], "investigate_claim")
        self.assertEqual(events[2].payload["step_name"], "generate_response")
        self.assertEqual(events[0].payload["confidence"], 0.92)


class TestAuditImmutable(_AuditTestCase):
    """test_audit_immutable — no UPDATE or DELETE methods exposed."""

    def test_no_update_method(self):
        self.assertFalse(hasattr(self.trail, 'update_event'))
        self.assertFalse(hasattr(self.trail, 'delete_event'))

    def test_direct_sql_update_breaks_chain(self):
        """Even if someone bypasses the API and does SQL UPDATE, chain detects it."""
        trace = "trace_tamper"
        self.trail.record_primitive(
            trace, "step1", "classify",
            output_hash="aaa", model_version="v1", prompt_hash="bbb",
        )
        self.trail.record_primitive(
            trace, "step2", "investigate",
            output_hash="ccc", model_version="v1", prompt_hash="ddd",
        )

        # Verify chain is good
        valid, _ = self.trail.verify_chain()
        self.assertTrue(valid)

        # Tamper via raw SQL
        self.trail._conn.execute(
            "UPDATE audit_events SET payload = ? WHERE id = 1",
            ('{"step_name": "TAMPERED", "primitive": "classify", "output_hash": "xxx", "model_version": "v1", "prompt_hash": "yyy"}',)
        )
        self.trail._conn.commit()

        # Chain should now be broken
        valid, msg = self.trail.verify_chain()
        self.assertFalse(valid)
        self.assertIn("Tampered", msg)


class TestHashChainIntegrity(_AuditTestCase):
    """test_hash_chain_integrity — verify SHA-256 chain across 100 events."""

    def test_chain_100_events(self):
        for i in range(100):
            self.trail.record_primitive(
                f"trace_{i % 10}", f"step_{i}", "classify",
                output_hash=f"hash_{i}", model_version="v1",
                prompt_hash=f"phash_{i}", confidence=0.9,
            )

        valid, msg = self.trail.verify_chain()
        self.assertTrue(valid)
        self.assertIn("100 events", msg)

    def test_first_event_links_to_genesis(self):
        self.trail.record_primitive(
            "trace1", "step1", "classify",
            output_hash="a", model_version="v1", prompt_hash="b",
        )
        events = self.trail.get_trail("trace1")
        self.assertEqual(events[0].previous_hash, GENESIS_HASH)

    def test_chain_linkage(self):
        self.trail.record_primitive(
            "t1", "s1", "classify", output_hash="a",
            model_version="v1", prompt_hash="b",
        )
        self.trail.record_primitive(
            "t1", "s2", "investigate", output_hash="c",
            model_version="v1", prompt_hash="d",
        )

        events = self.trail.get_trail("t1")
        self.assertEqual(events[1].previous_hash, events[0].event_hash)


class TestTamperDetection(_AuditTestCase):
    """test_tamper_detection — modify one event → chain verification fails."""

    def test_modify_payload_detected(self):
        for i in range(5):
            self.trail.record_primitive(
                "trace_x", f"step_{i}", "classify",
                output_hash=f"h{i}", model_version="v1", prompt_hash=f"p{i}",
            )

        # Tamper event 3
        self.trail._conn.execute(
            "UPDATE audit_events SET payload = ? WHERE id = 3",
            ('{"step_name": "EVIL", "primitive": "classify", "output_hash": "bad", "model_version": "v1", "prompt_hash": "bad"}',)
        )
        self.trail._conn.commit()

        valid, msg = self.trail.verify_chain()
        self.assertFalse(valid)
        self.assertIn("event 3", msg)

    def test_modify_hash_detected(self):
        self.trail.record_primitive(
            "t", "s1", "classify", output_hash="a",
            model_version="v1", prompt_hash="b",
        )
        self.trail.record_primitive(
            "t", "s2", "investigate", output_hash="c",
            model_version="v1", prompt_hash="d",
        )

        # Tamper the stored hash of event 1
        self.trail._conn.execute(
            "UPDATE audit_events SET event_hash = 'fakehash' WHERE id = 1"
        )
        self.trail._conn.commit()

        valid, msg = self.trail.verify_chain()
        self.assertFalse(valid)


class TestQueryByTraceId(_AuditTestCase):
    """test_query_by_trace_id — full decision trail returned in order."""

    def test_filter_by_trace(self):
        self.trail.record_primitive(
            "trace_A", "s1", "classify",
            output_hash="a", model_version="v1", prompt_hash="b",
        )
        self.trail.record_primitive(
            "trace_B", "s1", "classify",
            output_hash="c", model_version="v1", prompt_hash="d",
        )
        self.trail.record_primitive(
            "trace_A", "s2", "investigate",
            output_hash="e", model_version="v1", prompt_hash="f",
        )

        trail_a = self.trail.get_trail("trace_A")
        trail_b = self.trail.get_trail("trace_B")

        self.assertEqual(len(trail_a), 2)
        self.assertEqual(len(trail_b), 1)
        self.assertEqual(trail_a[0].payload["step_name"], "s1")
        self.assertEqual(trail_a[1].payload["step_name"], "s2")

    def test_empty_trace(self):
        events = self.trail.get_trail("nonexistent")
        self.assertEqual(events, [])


class TestGovernanceEventsIncluded(_AuditTestCase):
    """test_governance_events_included — tier and gate results in trail."""

    def test_governance_recorded(self):
        trace = "trace_gov"
        self.trail.record_governance(
            trace, domain="electronics_return",
            declared_tier="auto", applied_tier="gate",
            quality_gate_result="fail",
            reason="confidence below floor",
        )

        events = self.trail.get_trail(trace)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "governance_decision")
        self.assertEqual(events[0].payload["declared_tier"], "auto")
        self.assertEqual(events[0].payload["applied_tier"], "gate")
        self.assertEqual(events[0].payload["quality_gate_result"], "fail")


class TestEscalationEvents(_AuditTestCase):
    """Escalation events recorded with trigger and reason."""

    def test_escalation_recorded(self):
        trace = "trace_esc"
        self.trail.record_escalation(
            trace, step_name="investigate_claim",
            trigger="serial_number_mismatch",
            reason="Product SN doesn't match order SN",
            routed_to="escalate_to_manager",
        )

        events = self.trail.get_trail(trace)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "escalation")
        self.assertEqual(events[0].payload["trigger"], "serial_number_mismatch")


class TestDelegationEvents(_AuditTestCase):
    """Delegation events link parent and child traces."""

    def test_delegation_recorded(self):
        self.trail.record_delegation(
            trace_id="parent_trace",
            child_trace_id="child_trace",
            parent_workflow="product_return",
            child_workflow="sar_investigation",
            policy="fraud_triggers_sar",
            status="dispatched",
        )

        events = self.trail.get_trail("parent_trace")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["child_trace_id"], "child_trace")


class TestEventCount(_AuditTestCase):
    """Event count works."""

    def test_count(self):
        self.assertEqual(self.trail.count_events(), 0)
        self.trail.record_primitive(
            "t", "s", "classify", output_hash="a",
            model_version="v1", prompt_hash="b",
        )
        self.assertEqual(self.trail.count_events(), 1)


class TestEventSerialization(_AuditTestCase):
    """AuditEvent.to_dict produces clean JSON."""

    def test_to_dict(self):
        self.trail.record_primitive(
            "t1", "s1", "classify", output_hash="a",
            model_version="v1", prompt_hash="b", confidence=0.9,
        )
        event = self.trail.get_trail("t1")[0]
        d = event.to_dict()

        self.assertIn("id", d)
        self.assertIn("trace_id", d)
        self.assertIn("event_type", d)
        self.assertIn("payload", d)
        self.assertIn("event_hash", d)
        # Verify it's JSON-serializable
        json.dumps(d)


class TestEmptyChainVerification(_AuditTestCase):
    """Empty trail verifies successfully."""

    def test_empty_chain(self):
        valid, msg = self.trail.verify_chain()
        self.assertTrue(valid)
        self.assertIn("Empty", msg)


if __name__ == "__main__":
    unittest.main()
