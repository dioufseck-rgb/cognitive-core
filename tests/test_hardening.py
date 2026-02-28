"""
Cognitive Core — Production Hardening Tests

Tests the five production-readiness requirements:
  1. Dispatch Decision Record (DDR)
  2. Policy Versioning + Rollout Modes
  3. Explicit Partial-Failure Semantics
  4. Reservation Protocol Specification
  5. Learning Scope Constraints
"""

import os
import sys
import time
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.hardening import (
    # 1. DDR
    DispatchDecisionRecord, DDREligibilityEntry, DDRCandidateScore,
    build_ddr,
    # 2. Policy Versioning
    PolicyManager, PolicyVersion, PolicyMode, RolloutStage,
    PolicyRolloutConfig,
    # 3. Partial Failure
    PartialFailureHandler, PartialFailurePolicy, FailureAction,
    PartialFailureDecision,
    # 4. Reservation Protocol
    ReservationEventLog, ReservationOp, ReservationEvent,
    ReservationProtocolSpec,
    # 5. Learning Constraints
    LearningScopeEnforcer, LearningConstraint,
)


# ═══════════════════════════════════════════════════════════════════
# 1. DDR TESTS
# ═══════════════════════════════════════════════════════════════════

class TestDispatchDecisionRecord(unittest.TestCase):
    """Requirement 1: DDR per work order."""

    def test_build_ddr(self):
        """build_ddr produces a complete record."""
        eligible = [DDREligibilityEntry("r_1", True, capacity_at_decision=3)]
        excluded = [DDREligibilityEntry("r_2", False, "circuit_breaker", "OPEN")]
        scores = [DDRCandidateScore("r_1", 0.75, 1, {"cost": 0.3, "load": 0.45})]

        ddr = build_ddr(
            work_order_id="wo_1", correlation_id="cor_1",
            case_id="CLM-001", trace_id="tr_1",
            policy_version="1.0.0", policy_mode="static",
            solver_name="hungarian_builtin", solver_version="1.0",
            solver_seed=42,
            eligible_entries=eligible, excluded_entries=excluded,
            objective_weights={"minimize_cost": 0.3, "minimize_wait_time": 0.5},
            candidate_scores=scores,
            selected_resource_id="r_1", selection_tier="optimal",
            reservation_id="rsv_1",
            reason_codes=["optimal_assignment"],
            active_constraints=["capability_match", "circuit_breaker"],
        )
        self.assertTrue(ddr.ddr_id.startswith("ddr_"))
        self.assertEqual(ddr.work_order_id, "wo_1")
        self.assertEqual(ddr.selected_resource_id, "r_1")
        self.assertEqual(len(ddr.eligible_set), 1)
        self.assertEqual(len(ddr.excluded_set), 1)
        self.assertTrue(len(ddr.input_hash) > 0)
        self.assertTrue(len(ddr.reason_narrative) > 0)

    def test_ddr_serializes_for_ledger(self):
        """DDR serializes to dict for action_ledger storage."""
        ddr = build_ddr(
            "wo_1", "cor_1", "CLM-001", "tr_1",
            "1.0.0", "static", "greedy", "1.0", 0,
            [], [], {}, [], None, "no_eligible_resources",
        )
        entry = ddr.to_ledger_entry()
        self.assertIsInstance(entry, dict)
        self.assertEqual(entry["work_order_id"], "wo_1")
        self.assertIn("eligible_count", entry)
        self.assertIn("input_hash", entry)

    def test_input_hash_deterministic(self):
        """Same inputs produce same hash."""
        h1 = DispatchDecisionRecord.compute_input_hash(
            "wo_1", ["r_1", "r_2"], {"cost": 0.3},
        )
        h2 = DispatchDecisionRecord.compute_input_hash(
            "wo_1", ["r_2", "r_1"], {"cost": 0.3},  # order doesn't matter
        )
        self.assertEqual(h1, h2)

    def test_narrative_generation(self):
        """Auto-generated narrative is human-readable."""
        ddr = build_ddr(
            "wo_1", "cor_1", "CLM-001", "tr_1",
            "1.0.0", "static", "hungarian", "1.0", 42,
            [DDREligibilityEntry("r_1", True)],
            [DDREligibilityEntry("r_2", False, "circuit_breaker", "OPEN")],
            {"minimize_cost": 0.5},
            [DDRCandidateScore("r_1", 0.5, 1)],
            "r_1", "optimal",
        )
        self.assertIn("1 eligible", ddr.reason_narrative)
        self.assertIn("1 excluded", ddr.reason_narrative)
        self.assertIn("r_1", ddr.reason_narrative)
        self.assertIn("circuit_breaker", ddr.reason_narrative)


# ═══════════════════════════════════════════════════════════════════
# 2. POLICY VERSIONING TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPolicyVersioning(unittest.TestCase):
    """Requirement 2: Policy versioning + rollout modes."""

    def setUp(self):
        self.pm = PolicyManager()

    def test_create_version(self):
        """New version starts in TRAINING."""
        pv = self.pm.create_version("1.0.0", PolicyMode.STATIC)
        self.assertEqual(pv.stage, RolloutStage.TRAINING)
        self.assertEqual(pv.mode, PolicyMode.STATIC)

    def test_full_promotion_pipeline(self):
        """TRAINING → VALIDATION → SHADOW → CANARY → PRODUCTION."""
        self.pm.create_version("1.0.0")
        self.assertTrue(self.pm.promote("1.0.0", RolloutStage.VALIDATION))
        self.assertTrue(self.pm.promote("1.0.0", RolloutStage.SHADOW))
        self.assertTrue(self.pm.promote("1.0.0", RolloutStage.CANARY))
        pv = self.pm.get_version("1.0.0")
        self.assertAlmostEqual(pv.traffic_pct, 0.05)
        self.assertTrue(self.pm.promote("1.0.0", RolloutStage.PRODUCTION))
        pv = self.pm.get_version("1.0.0")
        self.assertEqual(pv.stage, RolloutStage.PRODUCTION)
        self.assertAlmostEqual(pv.traffic_pct, 1.0)

    def test_skip_stage_rejected(self):
        """Cannot skip stages: TRAINING → CANARY is rejected."""
        self.pm.create_version("1.0.0")
        self.assertFalse(self.pm.promote("1.0.0", RolloutStage.CANARY))

    def test_new_production_demotes_old(self):
        """Promoting new version to PRODUCTION demotes the old one."""
        self.pm.create_version("1.0.0")
        for stage in [RolloutStage.VALIDATION, RolloutStage.SHADOW,
                      RolloutStage.CANARY, RolloutStage.PRODUCTION]:
            self.pm.promote("1.0.0", stage)

        self.pm.create_version("2.0.0")
        for stage in [RolloutStage.VALIDATION, RolloutStage.SHADOW,
                      RolloutStage.CANARY, RolloutStage.PRODUCTION]:
            self.pm.promote("2.0.0", stage)

        old = self.pm.get_version("1.0.0")
        self.assertEqual(old.stage, RolloutStage.ROLLBACK)
        self.assertEqual(self.pm.get_active_version().version_id, "2.0.0")

    def test_rollback(self):
        """Rollback reverts to prior version."""
        self.pm.create_version("1.0.0")
        for s in [RolloutStage.VALIDATION, RolloutStage.SHADOW,
                  RolloutStage.CANARY, RolloutStage.PRODUCTION]:
            self.pm.promote("1.0.0", s)

        self.pm.create_version("2.0.0")
        for s in [RolloutStage.VALIDATION, RolloutStage.SHADOW,
                  RolloutStage.CANARY, RolloutStage.PRODUCTION]:
            self.pm.promote("2.0.0", s)

        self.assertTrue(self.pm.rollback("2.0.0", "SLA regression"))
        rolled = self.pm.get_version("2.0.0")
        self.assertEqual(rolled.stage, RolloutStage.ROLLBACK)
        self.assertIn("SLA", rolled.rollback_reason)

    def test_adp_requires_mrm_approval(self):
        """ADP mode cannot reach PRODUCTION without MRM approval."""
        config = PolicyRolloutConfig(require_mrm_approval_for_adp=True)
        pm = PolicyManager(config)
        pm.create_version("1.0.0", PolicyMode.ADP)
        for s in [RolloutStage.VALIDATION, RolloutStage.SHADOW, RolloutStage.CANARY]:
            pm.promote("1.0.0", s)
        # No MRM approval → blocked
        self.assertFalse(pm.promote("1.0.0", RolloutStage.PRODUCTION))
        # Add approval
        pm.get_version("1.0.0").metrics["mrm_approved"] = True
        self.assertTrue(pm.promote("1.0.0", RolloutStage.PRODUCTION))

    def test_auto_rollback_trigger(self):
        """Auto-rollback triggers on SLA miss rate."""
        self.pm.create_version("1.0.0")
        for s in [RolloutStage.VALIDATION, RolloutStage.SHADOW,
                  RolloutStage.CANARY, RolloutStage.PRODUCTION]:
            self.pm.promote("1.0.0", s)
        pv = self.pm.get_version("1.0.0")
        pv.metrics["sla_miss_rate"] = 0.08  # above 5% threshold
        self.assertTrue(self.pm.should_auto_rollback("1.0.0"))


# ═══════════════════════════════════════════════════════════════════
# 3. PARTIAL FAILURE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestPartialFailure(unittest.TestCase):
    """Requirement 3: Explicit partial-failure semantics."""

    def setUp(self):
        self.handler = PartialFailureHandler()

    def test_retryable_error_retries(self):
        """Retryable error → RETRY action."""
        self.handler.register_policy(PartialFailurePolicy(
            need_type="forensic_review", max_retries=2,
        ))
        d = self.handler.resolve("wo_1", "forensic_review", "CLM-001", "retryable")
        self.assertEqual(d.action, FailureAction.RETRY)
        self.assertIn("1/2", d.reason)

    def test_retry_budget_exhausted_escalates(self):
        """After max retries → ESCALATE."""
        self.handler.register_policy(PartialFailurePolicy(
            need_type="review", max_retries=2,
        ))
        self.handler.resolve("wo_1", "review", "CLM-001", "retryable")
        self.handler.resolve("wo_2", "review", "CLM-001", "retryable")
        d = self.handler.resolve("wo_3", "review", "CLM-001", "retryable")
        self.assertEqual(d.action, FailureAction.ESCALATE)
        self.assertIn("exhausted", d.reason)

    def test_permanent_error_escalates(self):
        """Permanent error → ESCALATE (no retry)."""
        self.handler.register_policy(PartialFailurePolicy(
            need_type="review",
            on_permanent=FailureAction.ESCALATE,
            escalation_queue="compliance",
        ))
        d = self.handler.resolve("wo_1", "review", "CLM-001", "permanent")
        self.assertEqual(d.action, FailureAction.ESCALATE)
        self.assertIsNotNone(d.escalation_details)
        self.assertEqual(d.escalation_details["queue"], "compliance")

    def test_degraded_error_degrades(self):
        """Degraded error → DEGRADE with template output."""
        self.handler.register_policy(PartialFailurePolicy(
            need_type="scoring",
            on_degraded=FailureAction.DEGRADE,
            degraded_output_template={"score": 0.5, "source": "fallback"},
            degraded_quality_flag="degraded",
        ))
        d = self.handler.resolve("wo_1", "scoring", "CLM-001", "degraded")
        self.assertEqual(d.action, FailureAction.DEGRADE)
        self.assertIsNotNone(d.degraded_output)
        self.assertEqual(d.degraded_output["score"], 0.5)
        self.assertEqual(d.degraded_output["_quality_flag"], "degraded")

    def test_default_policy_for_unknown_need(self):
        """Unknown need type gets default retry policy."""
        d = self.handler.resolve("wo_1", "unknown_need", "CLM-001", "retryable")
        self.assertEqual(d.action, FailureAction.RETRY)

    def test_abort_action(self):
        """Abort action is available for critical needs."""
        self.handler.register_policy(PartialFailurePolicy(
            need_type="payment",
            on_permanent=FailureAction.ABORT,
            compensate_on_abort=True,
        ))
        d = self.handler.resolve("wo_1", "payment", "CLM-001", "permanent")
        self.assertEqual(d.action, FailureAction.ABORT)


# ═══════════════════════════════════════════════════════════════════
# 4. RESERVATION PROTOCOL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestReservationProtocol(unittest.TestCase):
    """Requirement 4: Reservation protocol fully specified."""

    def setUp(self):
        self.log = ReservationEventLog()

    def test_acquire_commit_lifecycle(self):
        """Normal lifecycle: ACQUIRE → COMMIT."""
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0)
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.COMMIT)
        events = self.log.get_events(reservation_id="rsv_1")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].operation, ReservationOp.ACQUIRE)
        self.assertEqual(events[1].operation, ReservationOp.COMMIT)

    def test_acquire_release_lifecycle(self):
        """Cancel lifecycle: ACQUIRE → RELEASE."""
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0)
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.RELEASE)
        events = self.log.get_events(reservation_id="rsv_1")
        ops = [e.operation for e in events]
        self.assertEqual(ops, [ReservationOp.ACQUIRE, ReservationOp.RELEASE])

    def test_acquire_expire_lifecycle(self):
        """TTL expiry lifecycle: ACQUIRE → EXPIRE."""
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0, ttl_seconds=30.0)
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.EXPIRE)
        expires = self.log.get_events(operation=ReservationOp.EXPIRE)
        self.assertEqual(len(expires), 1)

    def test_reclaim_event(self):
        """Revocation manager reclaim produces RECLAIM event."""
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0)
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.RECLAIM)
        events = self.log.get_events(reservation_id="rsv_1")
        self.assertEqual(events[1].operation, ReservationOp.RECLAIM)

    def test_orphan_scan(self):
        """Orphan scan finds ACQUIRE events with no resolution."""
        # Orphan: acquired but never committed/released/expired
        self.log.record("rsv_orphan", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0)
        self.log._events[0].timestamp = time.time() - 200  # old enough

        # Healthy: acquired and committed
        self.log.record("rsv_healthy", "r_2", "wo_2", ReservationOp.ACQUIRE, 1.0)
        self.log.record("rsv_healthy", "r_2", "wo_2", ReservationOp.COMMIT)

        orphans = self.log.orphan_scan(max_age_seconds=120.0)
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0].reservation_id, "rsv_orphan")

    def test_crash_recovery_event(self):
        """Crash recovery produces CRASH_RECOVER event."""
        self.log.record("rsv_1", "r_1", "wo_1", ReservationOp.ACQUIRE, 1.0)
        self.log.record(
            "rsv_1", "r_1", "wo_1", ReservationOp.CRASH_RECOVER,
            recovery_action="released",
        )
        events = self.log.get_events(operation=ReservationOp.CRASH_RECOVER)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].recovery_action, "released")

    def test_protocol_spec_defaults(self):
        """Protocol spec has correct defaults."""
        spec = ReservationProtocolSpec()
        self.assertEqual(spec.default_ttl_seconds, 30.0)
        self.assertTrue(spec.commit_idempotent)
        self.assertTrue(spec.release_idempotent)
        self.assertFalse(spec.acquire_idempotent)
        self.assertTrue(spec.acquire_atomic)


# ═══════════════════════════════════════════════════════════════════
# 5. LEARNING SCOPE CONSTRAINTS TESTS
# ═══════════════════════════════════════════════════════════════════

class TestLearningScopeConstraints(unittest.TestCase):
    """Requirement 5: Hard-coded guardrails on learning."""

    def setUp(self):
        self.enforcer = LearningScopeEnforcer()

    def test_valid_adjustment_passes(self):
        """Small weight adjustment within bounds passes."""
        base = {"minimize_cost": 0.30, "minimize_wait_time": 0.50}
        proposed = {"minimize_cost": 0.35, "minimize_wait_time": 0.45}
        valid, violations = self.enforcer.validate_adjustments(base, proposed)
        self.assertTrue(valid)
        self.assertEqual(len(violations), 0)

    def test_excessive_adjustment_blocked(self):
        """Weight adjustment exceeding ±30% is blocked."""
        base = {"minimize_cost": 0.30}
        proposed = {"minimize_cost": 0.50}  # 67% increase, > 30%
        valid, violations = self.enforcer.validate_adjustments(base, proposed)
        self.assertFalse(valid)
        self.assertEqual(len(violations), 1)
        self.assertIn("cost_weight_bounded", violations[0])

    def test_eligibility_modification_blocked(self):
        """Attempting to modify eligibility constraints is blocked."""
        valid, violations = self.enforcer.validate_adjustments(
            {}, {},
            proposed_modifications={"eligibility_constraints": {"remove": "licensing"}},
        )
        self.assertFalse(valid)
        self.assertIn("eligibility_immutable", violations[0])

    def test_capacity_modification_blocked(self):
        """Attempting to modify capacity is blocked."""
        valid, violations = self.enforcer.validate_adjustments(
            {}, {},
            proposed_modifications={"capacity": {"max_concurrent": 100}},
        )
        self.assertFalse(valid)
        self.assertIn("capacity_immutable", violations[0])

    def test_governance_bypass_blocked(self):
        """Attempting to bypass governance gates is blocked."""
        valid, violations = self.enforcer.validate_adjustments(
            {}, {},
            proposed_modifications={"bypass_governance_gates": True},
        )
        self.assertFalse(valid)
        self.assertIn("governance_gate_immutable", violations[0])

    def test_circuit_breaker_bypass_blocked(self):
        """Attempting to bypass circuit breaker is blocked."""
        valid, violations = self.enforcer.validate_adjustments(
            {}, {},
            proposed_modifications={"bypass_circuit_breaker": True},
        )
        self.assertFalse(valid)
        self.assertIn("circuit_breaker_immutable", violations[0])

    def test_enforce_returns_safe_weights(self):
        """enforce() returns base weights when adjustment is invalid."""
        base = {"minimize_cost": 0.30}
        proposed = {"minimize_cost": 0.60}  # way out of bounds
        safe = self.enforcer.enforce(base, proposed)
        self.assertEqual(safe["minimize_cost"], 0.30)  # reverted to base

    def test_enforce_returns_proposed_when_valid(self):
        """enforce() returns proposed weights when valid."""
        base = {"minimize_cost": 0.30}
        proposed = {"minimize_cost": 0.33}  # 10% increase, within 30%
        safe = self.enforcer.enforce(base, proposed)
        self.assertEqual(safe["minimize_cost"], 0.33)

    def test_core_constraints_count(self):
        """All 9 core constraints are present."""
        self.assertEqual(len(self.enforcer.constraints), 9)

    def test_multiple_violations_all_reported(self):
        """Multiple simultaneous violations are all reported."""
        valid, violations = self.enforcer.validate_adjustments(
            {"minimize_cost": 0.30, "minimize_wait_time": 0.50},
            {"minimize_cost": 0.60, "minimize_wait_time": 0.90},  # both out of bounds
            proposed_modifications={"eligibility_constraints": "modify"},
        )
        self.assertFalse(valid)
        self.assertGreaterEqual(len(violations), 3)


if __name__ == "__main__":
    unittest.main()
