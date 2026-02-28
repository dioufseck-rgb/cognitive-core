"""
Cognitive Core — Resilience Layer Tests

Tests the four production failure modes:
  1. Observer-State Divergence (Resume Revalidation)
  2. Subjective Acceptance Loop (Semantic Oscillation)
  3. Stochastic Capacity Erosion (Graceful Revocation)
  4. Saga of Side Effects (Cross-Workflow Compensation)
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.resilience import (
    # Failure Mode 1
    ResumeRevalidator,
    RevalidationGuard,
    RevalidationResult,
    StalenessVerdict,
    # Failure Mode 2
    OscillationDetector,
    OscillationAction,
    OscillationVerdict,
    OscillationState,
    # Failure Mode 3
    CapacityRevocationManager,
    RevocationConfig,
    RevocationPolicy,
    RevocationSignal,
    # Failure Mode 4
    SagaCoordinator,
    SagaCompensationEntry,
    CompensationScope,
)


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 1: OBSERVER-STATE DIVERGENCE
# ═══════════════════════════════════════════════════════════════════

class TestResumeRevalidation(unittest.TestCase):
    """Failure Mode 1: Observer-State Divergence defense."""

    def setUp(self):
        self.revalidator = ResumeRevalidator()

    def test_no_guards_returns_valid(self):
        """No registered guards → valid by default."""
        result = self.revalidator.revalidate("wf_123", {})
        self.assertEqual(result.verdict, StalenessVerdict.VALID)

    def test_custom_guard_valid(self):
        """Custom guard function returns VALID."""
        def check_active(state):
            return StalenessVerdict.VALID

        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="case_active", check_fn=check_active,
        ))
        result = self.revalidator.revalidate("wf_123", {})
        self.assertEqual(result.verdict, StalenessVerdict.VALID)
        self.assertIn("case_active", result.checks_run)

    def test_custom_guard_stale(self):
        """Custom guard detects stale context."""
        def check_status(state):
            if state.get("case_status") == "canceled":
                return StalenessVerdict.STALE
            return StalenessVerdict.VALID

        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="case_status_check", check_fn=check_status,
        ))
        result = self.revalidator.revalidate("wf_123", {"case_status": "canceled"})
        self.assertEqual(result.verdict, StalenessVerdict.STALE)

    def test_custom_guard_invalidated(self):
        """Custom guard detects invalidation (worse than stale)."""
        def check_deleted(state):
            if state.get("entity_deleted"):
                return StalenessVerdict.INVALIDATED
            return StalenessVerdict.VALID

        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="entity_exists", check_fn=check_deleted,
        ))
        result = self.revalidator.revalidate("wf_123", {"entity_deleted": True})
        self.assertEqual(result.verdict, StalenessVerdict.INVALIDATED)

    def test_declarative_entity_guard(self):
        """Declarative guard compares entity field against expected value."""
        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="case_status",
            entity_type="case",
            field_path="status",
            expected_value="active",
        ))

        # Entity loader returns current state
        def loader(entity_type, field_path):
            return "canceled"  # changed since suspension!

        result = self.revalidator.revalidate("wf_123", {}, entity_loader=loader)
        self.assertEqual(result.verdict, StalenessVerdict.STALE)
        self.assertIn("_revalidation.case_status", result.enrichment)
        enrichment = result.enrichment["_revalidation.case_status"]
        self.assertEqual(enrichment["expected"], "active")
        self.assertEqual(enrichment["actual"], "canceled")

    def test_multiple_guards_worst_wins(self):
        """Multiple guards — worst verdict propagates."""
        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="guard_valid",
            check_fn=lambda s: StalenessVerdict.VALID,
        ))
        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="guard_stale",
            check_fn=lambda s: StalenessVerdict.STALE,
        ))
        result = self.revalidator.revalidate("wf_123", {})
        self.assertEqual(result.verdict, StalenessVerdict.STALE)
        self.assertEqual(len(result.checks_run), 2)

    def test_guard_exception_returns_error(self):
        """Guard that throws exception → ERROR verdict."""
        def broken_guard(state):
            raise RuntimeError("DB connection failed")

        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="broken", check_fn=broken_guard,
        ))
        result = self.revalidator.revalidate("wf_123", {})
        self.assertEqual(result.verdict, StalenessVerdict.ERROR)

    def test_cleanup_removes_guards(self):
        """Cleanup removes all guards for an instance."""
        self.revalidator.register_guard("wf_123", RevalidationGuard(
            name="test", check_fn=lambda s: StalenessVerdict.VALID,
        ))
        self.revalidator.cleanup("wf_123")
        result = self.revalidator.revalidate("wf_123", {})
        self.assertEqual(result.verdict, StalenessVerdict.VALID)
        self.assertEqual(len(result.checks_run), 0)


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 2: SUBJECTIVE ACCEPTANCE LOOP
# ═══════════════════════════════════════════════════════════════════

class TestOscillationDetection(unittest.TestCase):
    """Failure Mode 2: Semantic Oscillation defense."""

    def setUp(self):
        self.detector = OscillationDetector(
            max_rejections=3,
            max_same_provider_rejections=2,
        )

    def test_accepted_proceeds(self):
        """Accepted result → PROCEED."""
        verdict = self.detector.record_attempt(
            "forensic_review", "CLM-001", "cor_1",
            "r_1", "wo_1", accepted=True,
        )
        self.assertEqual(verdict.action, OscillationAction.PROCEED)

    def test_first_rejection_retries(self):
        """First rejection → RETRY."""
        verdict = self.detector.record_attempt(
            "forensic_review", "CLM-001", "cor_1",
            "r_1", "wo_1", accepted=False,
            rejection_reason="insufficient detail",
        )
        self.assertEqual(verdict.action, OscillationAction.RETRY)

    def test_max_rejections_escalates(self):
        """After max rejections → ESCALATE."""
        for i in range(3):
            verdict = self.detector.record_attempt(
                "forensic_review", "CLM-001", "cor_1",
                f"r_{i}", f"wo_{i}", accepted=False,
                rejection_reason=f"reason_{i}",
            )
        self.assertEqual(verdict.action, OscillationAction.ESCALATE)
        self.assertIsNotNone(verdict.oscillation_state)
        self.assertEqual(verdict.oscillation_state.rejection_count, 3)

    def test_same_provider_rejected_twice_switches(self):
        """Same provider rejected twice → RETRY_DIFFERENT_PROVIDER."""
        self.detector.record_attempt(
            "forensic_review", "CLM-001", "cor_1",
            "r_1", "wo_1", accepted=False,
        )
        verdict = self.detector.record_attempt(
            "forensic_review", "CLM-001", "cor_1",
            "r_1", "wo_2", accepted=False,
        )
        self.assertEqual(verdict.action, OscillationAction.RETRY_DIFFERENT_PROVIDER)
        self.assertIn("r_1", verdict.exclude_providers)

    def test_same_reason_cluster_escalates(self):
        """All rejections cite same reason → ESCALATE (criteria conflict)."""
        self.detector = OscillationDetector(max_rejections=5)  # higher threshold
        self.detector.record_attempt(
            "review", "CLM-002", "cor_2",
            "r_1", "wo_1", accepted=False,
            rejection_reason="format does not meet standard",
        )
        verdict = self.detector.record_attempt(
            "review", "CLM-002", "cor_2",
            "r_2", "wo_2", accepted=False,
            rejection_reason="format does not meet standard",
        )
        self.assertEqual(verdict.action, OscillationAction.ESCALATE)
        self.assertIn("unsatisfiable", verdict.reason)

    def test_distinct_providers_tracked(self):
        """Track distinct providers tried."""
        self.detector.record_attempt("review", "CLM-003", "cor_3", "r_1", "wo_1", False)
        self.detector.record_attempt("review", "CLM-003", "cor_3", "r_2", "wo_2", False)
        state = self.detector.get_state("review", "CLM-003")
        self.assertEqual(state.distinct_providers_tried, {"r_1", "r_2"})

    def test_acceptance_after_rejections(self):
        """Acceptance after rejections resolves the oscillation."""
        self.detector.record_attempt("review", "CLM-004", "cor_4", "r_1", "wo_1", False)
        verdict = self.detector.record_attempt(
            "review", "CLM-004", "cor_4", "r_2", "wo_2", True,
        )
        self.assertEqual(verdict.action, OscillationAction.PROCEED)
        state = self.detector.get_state("review", "CLM-004")
        self.assertTrue(state.resolved)
        self.assertEqual(state.resolution, "accepted")

    def test_cleanup(self):
        """Cleanup removes tracking for a case."""
        self.detector.record_attempt("review", "CLM-005", "cor_5", "r_1", "wo_1", False)
        self.detector.cleanup("CLM-005")
        state = self.detector.get_state("review", "CLM-005")
        self.assertIsNone(state)


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 3: STOCHASTIC CAPACITY EROSION
# ═══════════════════════════════════════════════════════════════════

class TestGracefulRevocation(unittest.TestCase):
    """Failure Mode 3: Graceful Revocation defense."""

    def setUp(self):
        self.config = RevocationConfig(
            max_ttl_extensions=2,
            extension_seconds=30.0,
            max_total_ttl_seconds=120.0,
            checkpoint_grace_seconds=15.0,
            priority_preemption=True,
        )
        self.manager = CapacityRevocationManager(self.config)

    def test_no_action_with_remaining_ttl(self):
        """No signal when TTL has > 5s remaining."""
        signal = self.manager.evaluate_expiring_reservation(
            "wo_1", reservation_ttl_remaining=10.0,
            work_order_priority="routine",
            work_order_progress=0.5,
            waiting_queue_depth=0,
        )
        self.assertIsNone(signal)

    def test_first_extension_granted(self):
        """First TTL extension granted when no preemption needed."""
        signal = self.manager.evaluate_expiring_reservation(
            "wo_1", reservation_ttl_remaining=3.0,
            work_order_priority="routine",
            work_order_progress=0.7,
            waiting_queue_depth=0,
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.policy, RevocationPolicy.EXTEND_TTL)

    def test_max_extensions_exceeded(self):
        """After max extensions, revocation happens."""
        # Use first two extensions
        for _ in range(2):
            self.manager.evaluate_expiring_reservation(
                "wo_1", 3.0, "routine", 0.5, 0,
            )
        # Third request → no more extensions
        signal = self.manager.evaluate_expiring_reservation(
            "wo_1", 3.0, "routine", 0.7, 0,
        )
        self.assertIsNotNone(signal)
        self.assertNotEqual(signal.policy, RevocationPolicy.EXTEND_TTL)

    def test_checkpoint_when_mostly_done(self):
        """Work > 50% done → CHECKPOINT_EXIT (not hard kill)."""
        # Exhaust extensions first
        for _ in range(2):
            self.manager.evaluate_expiring_reservation(
                "wo_1", 3.0, "routine", 0.5, 0,
            )
        signal = self.manager.evaluate_expiring_reservation(
            "wo_1", 3.0, "routine", 0.8, 0,
        )
        self.assertEqual(signal.policy, RevocationPolicy.CHECKPOINT_EXIT)

    def test_priority_preemption(self):
        """Critical work preempts routine work."""
        signal = self.manager.evaluate_expiring_reservation(
            "wo_routine", reservation_ttl_remaining=3.0,
            work_order_priority="routine",
            work_order_progress=0.3,
            waiting_queue_depth=1,
            highest_waiter_priority="critical",
        )
        self.assertIsNotNone(signal)
        self.assertEqual(signal.policy, RevocationPolicy.PREEMPT_QUEUE)

    def test_no_preemption_same_priority(self):
        """Same priority → extend, don't preempt."""
        signal = self.manager.evaluate_expiring_reservation(
            "wo_1", 3.0, "routine", 0.5, 1, "routine",
        )
        self.assertEqual(signal.policy, RevocationPolicy.EXTEND_TTL)

    def test_check_signal(self):
        """Agent can check for pending revocation signal."""
        # No signal initially
        self.assertIsNone(self.manager.check_signal("wo_1"))

        # Trigger a revocation (exhaust extensions, then trigger)
        for _ in range(2):
            self.manager.evaluate_expiring_reservation("wo_1", 3.0, "routine", 0.5, 0)
        self.manager.evaluate_expiring_reservation("wo_1", 3.0, "routine", 0.8, 0)

        # Now signal should be pending
        signal = self.manager.check_signal("wo_1")
        self.assertIsNotNone(signal)

    def test_acknowledge_clears_signal(self):
        """Acknowledging a signal removes it."""
        for _ in range(2):
            self.manager.evaluate_expiring_reservation("wo_1", 3.0, "routine", 0.5, 0)
        self.manager.evaluate_expiring_reservation("wo_1", 3.0, "routine", 0.8, 0)
        self.manager.acknowledge("wo_1")
        self.assertIsNone(self.manager.check_signal("wo_1"))


# ═══════════════════════════════════════════════════════════════════
# FAILURE MODE 4: SAGA OF SIDE EFFECTS
# ═══════════════════════════════════════════════════════════════════

class TestSagaCompensation(unittest.TestCase):
    """Failure Mode 4: Cross-workflow Saga compensation."""

    def setUp(self):
        self.saga = SagaCoordinator()

    def test_register_and_confirm(self):
        """Register → confirm → entry is ready for compensation."""
        entry_id = self.saga.register(
            "saga_1", "wo_1", "send_email",
            "Sent notification email to customer",
            {"action": "unsend_email", "email_id": "em_123"},
        )
        self.assertTrue(self.saga.confirm(entry_id))
        entries = self.saga.get_saga_entries("saga_1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].status, "confirmed")

    def test_compensate_reverses_in_order(self):
        """Compensation walks entries in reverse chronological order."""
        compensated_order = []

        # Register three actions in order
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "Action 1", {"undo": 1})
        self.saga.confirm(id1)
        time.sleep(0.01)  # ensure ordering

        id2 = self.saga.register("saga_1", "wo_1", "step_2", "Action 2", {"undo": 2})
        self.saga.confirm(id2)
        time.sleep(0.01)

        id3 = self.saga.register("saga_1", "wo_2", "step_3", "Action 3", {"undo": 3})
        self.saga.confirm(id3)

        # Compensate all (no failed_work_order)
        def handler(entry):
            compensated_order.append(entry.step_name)
            return True

        results = self.saga.compensate("saga_1", handler=handler)
        self.assertEqual(len(results), 3)
        # Should be reverse order: step_3, step_2, step_1
        self.assertEqual(compensated_order, ["step_3", "step_2", "step_1"])

    def test_failed_wo_entries_skipped(self):
        """Entries from the failed work order are not compensated."""
        id1 = self.saga.register("saga_1", "wo_1", "send_email", "Email", {"unsend": True})
        self.saga.confirm(id1)

        id2 = self.saga.register("saga_1", "wo_2", "update_db", "DB update", {"delete": True})
        # wo_2 failed — don't confirm (pending entries aren't compensated either)

        compensated = []
        results = self.saga.compensate(
            "saga_1",
            handler=lambda e: (compensated.append(e.step_name), True)[1],
            failed_work_order_id="wo_2",
        )
        # Only wo_1's confirmed entry should be compensated
        self.assertEqual(compensated, ["send_email"])

    def test_no_handler_skips_to_hitl(self):
        """No handler → entries marked SKIPPED (escalate to human)."""
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "Action", {"undo": 1})
        self.saga.confirm(id1)

        results = self.saga.compensate("saga_1", handler=None)
        self.assertEqual(results[0].status, "skipped")
        self.assertIn("human intervention", results[0].error)

    def test_handler_failure_marks_failed(self):
        """Handler returning False → entry marked FAILED."""
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "Action", {"undo": 1})
        self.saga.confirm(id1)

        results = self.saga.compensate("saga_1", handler=lambda e: False)
        self.assertEqual(results[0].status, "failed")

    def test_handler_exception_marks_failed(self):
        """Handler throwing exception → entry marked FAILED."""
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "Action", {"undo": 1})
        self.saga.confirm(id1)

        def bad_handler(entry):
            raise RuntimeError("Compensation API down")

        results = self.saga.compensate("saga_1", handler=bad_handler)
        self.assertEqual(results[0].status, "failed")
        self.assertIn("API down", results[0].error)

    def test_pending_entries_not_compensated(self):
        """Pending entries (action never executed) are not compensated."""
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "Action 1", {"undo": 1})
        self.saga.confirm(id1)

        # step_2 registered but never confirmed (action never ran)
        self.saga.register("saga_1", "wo_1", "step_2", "Action 2", {"undo": 2})

        compensated = []
        self.saga.compensate(
            "saga_1",
            handler=lambda e: (compensated.append(e.step_name), True)[1],
        )
        # Only confirmed entry compensated
        self.assertEqual(compensated, ["step_1"])

    def test_pending_compensations_count(self):
        """Count of at-risk entries for saga health monitoring."""
        id1 = self.saga.register("saga_1", "wo_1", "step_1", "A1", {})
        self.saga.confirm(id1)
        id2 = self.saga.register("saga_1", "wo_2", "step_2", "A2", {})
        self.saga.confirm(id2)
        self.saga.register("saga_1", "wo_3", "step_3", "A3", {})  # pending

        self.assertEqual(self.saga.pending_compensations("saga_1"), 2)

    def test_cleanup(self):
        """Cleanup removes saga entirely."""
        self.saga.register("saga_1", "wo_1", "step_1", "A", {})
        self.saga.cleanup("saga_1")
        self.assertEqual(self.saga.get_saga_entries("saga_1"), [])

    def test_multi_saga_isolation(self):
        """Different sagas are isolated from each other."""
        id1 = self.saga.register("saga_A", "wo_1", "step_1", "A", {})
        self.saga.confirm(id1)
        id2 = self.saga.register("saga_B", "wo_2", "step_1", "B", {})
        self.saga.confirm(id2)

        compensated = []
        self.saga.compensate(
            "saga_A",
            handler=lambda e: (compensated.append(f"A:{e.step_name}"), True)[1],
        )
        # Only saga_A's entries compensated
        self.assertEqual(compensated, ["A:step_1"])
        # saga_B untouched
        entries = self.saga.get_saga_entries("saga_B")
        self.assertEqual(entries[0].status, "confirmed")


if __name__ == "__main__":
    unittest.main()
