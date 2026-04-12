"""
Tests: Epistemic state computation — mechanical and coherence layers

Verifies:
- Mechanical metrics (evidence_completeness, rule_coverage, citation_rate)
  are computed correctly from existing output fields
- Coherence flags fire under the correct cross-step conditions
- Overall score and warranted flag derive correctly
- WorkflowEpistemicRecord accumulates correctly across steps
- reviewer_context() produces the right structure for the escalation brief
- The VERIFY_DELIBERATE_TENSION flag catches the permit review pattern:
  verify finds violations but deliberate recommends approval
"""

from __future__ import annotations

import unittest
from cognitive_core.engine.epistemic import (
    CoherenceFlag,
    StepEpistemicState,
    WorkflowEpistemicRecord,
    compute_mechanical,
    compute_coherence_flags,
    compute_overall,
    compute_step_epistemic_state,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _retrieve_step(sources: list[dict], confidence=0.95) -> dict:
    return {
        "step_name": "retrieve_data",
        "primitive": "retrieve",
        "output": {
            "sources_queried": sources,
            "data": {s["source"]: {} for s in sources if s.get("status") == "success"},
            "confidence": confidence,
        },
    }


def _verify_step(conforms: bool, violations: list, rules_checked: list, confidence=0.88) -> dict:
    return {
        "step_name": "verify_compliance",
        "primitive": "verify",
        "output": {
            "conforms": conforms,
            "violations": violations,
            "rules_checked": rules_checked,
            "confidence": confidence,
        },
    }


def _deliberate_step(action: str, warrant: str = "", evidence_used: list = None, confidence=0.90) -> dict:
    return {
        "step_name": "deliberate_determination",
        "primitive": "deliberate",
        "output": {
            "recommended_action": action,
            "warrant": warrant,
            "evidence_used": evidence_used or [],
            "confidence": confidence,
        },
    }


def _classify_step(category: str, confidence=0.85) -> dict:
    return {
        "step_name": "classify_type",
        "primitive": "classify",
        "output": {
            "category": category,
            "confidence": confidence,
            "evidence_used": [{"source": "data", "description": "classification basis"}],
        },
    }


# ── Mechanical computation tests ─────────────────────────────────────────────

class TestMechanicalComputation(unittest.TestCase):

    def test_retrieve_completeness_all_sources_succeed(self):
        sources = [
            {"source": "get_application", "status": "success", "record_count": None},
            {"source": "get_parcel_data", "status": "success", "record_count": None},
            {"source": "get_ceqa_text", "status": "success", "record_count": None},
        ]
        ec, rc, cr = compute_mechanical("retrieve", {
            "sources_queried": sources,
            "data": {"get_application": {}, "get_parcel_data": {}, "get_ceqa_text": {}},
        })
        self.assertEqual(ec, 1.0)
        self.assertIsNone(rc)
        self.assertIsNone(cr)

    def test_retrieve_completeness_partial_success(self):
        sources = [
            {"source": "get_application", "status": "success", "record_count": None},
            {"source": "get_parcel_data", "status": "failed", "record_count": None},
            {"source": "get_ceqa_text", "status": "success", "record_count": None},
        ]
        ec, _, _ = compute_mechanical("retrieve", {"sources_queried": sources, "data": {}})
        # 2 of 3 succeeded (record_count=None → assumed present for successful ones)
        self.assertAlmostEqual(ec, 2/3, places=2)

    def test_retrieve_completeness_empty_record_counts_reduce(self):
        sources = [
            {"source": "s1", "status": "success", "record_count": 5},
            {"source": "s2", "status": "success", "record_count": 0},  # empty
            {"source": "s3", "status": "success", "record_count": 3},
        ]
        ec, _, _ = compute_mechanical("retrieve", {"sources_queried": sources, "data": {}})
        # 2 of 3 have data (record_count > 0)
        self.assertAlmostEqual(ec, 2/3, places=2)

    def test_verify_full_coverage(self):
        _, rc, _ = compute_mechanical("verify", {
            "rules_checked": ["CEQA §15332", "CEQA §15070", "Housing Element"],
            "violations": [],
            "conforms": True,
        })
        self.assertEqual(rc, 1.0)

    def test_verify_no_rules_checked_is_zero(self):
        _, rc, _ = compute_mechanical("verify", {
            "rules_checked": [],
            "violations": [{"rule": "some rule", "severity": "major"}],
            "conforms": False,
        })
        self.assertEqual(rc, 0.0)

    def test_deliberate_citation_rate_with_evidence(self):
        _, _, cr = compute_mechanical("deliberate", {
            "recommended_action": "APPROVE",
            "warrant": "The project is consistent. The zoning allows it. Evidence supports approval.",
            "evidence_used": [
                {"source": "retrieve", "description": "parcel data"},
                {"source": "verify", "description": "compliance check"},
            ],
            "confidence": 0.90,
        })
        self.assertIsNotNone(cr)
        self.assertGreater(cr, 0.0)

    def test_deliberate_no_warrant_no_evidence_is_none(self):
        """Empty evidence_used and no warrant → no measurement (None), not a low score.
        We don't penalise absent evidence_used — it may be prompt non-compliance."""
        _, _, cr = compute_mechanical("deliberate", {
            "recommended_action": "APPROVE",
            "warrant": "",
            "evidence_used": [],
            "confidence": 0.75,
        })
        self.assertIsNone(cr)


# ── Coherence flag tests ─────────────────────────────────────────────────────

class TestCoherenceFlags(unittest.TestCase):

    def test_verify_deliberate_tension_fires(self):
        """
        Core permit review pattern: verify found violations,
        deliberate still recommended approval.
        """
        verify = _verify_step(
            conforms=False,
            violations=[{"rule": "CEQA exemption", "description": "not applicable", "severity": "major"}],
            rules_checked=["CEQA §15332"],
        )
        deliberate = _deliberate_step("APPROVE WITH CONDITIONS", warrant="project meets housing goals")

        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_verify_deliberate_tension_does_not_fire_when_conforms(self):
        """No tension when verify passes."""
        verify = _verify_step(conforms=True, violations=[], rules_checked=["rule1"])
        deliberate = _deliberate_step("APPROVE", warrant="all checks pass")
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertNotIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_classify_deliberate_mismatch_fires_for_prohibited_approve(self):
        classify = _classify_step("prohibited_use")
        deliberate = _deliberate_step("APPROVE the application", warrant="some warrant")
        flags, _, _ = compute_coherence_flags(deliberate, [classify], [])
        self.assertIn(CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH, flags)

    def test_classify_deliberate_no_mismatch_for_exempt_approve(self):
        classify = _classify_step("categorical_exempt")
        deliberate = _deliberate_step("APPROVE - exempt from CEQA", warrant="exempt per 15332")
        flags, _, _ = compute_coherence_flags(deliberate, [classify], [])
        self.assertNotIn(CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH, flags)

    def test_confidence_drop_fires(self):
        prior = _verify_step(conforms=True, violations=[], rules_checked=["r1"], confidence=0.95)
        deliberate = _deliberate_step("APPROVE", warrant="ok", confidence=0.60)  # drops 0.35
        flags, _, _ = compute_coherence_flags(deliberate, [prior], [])
        self.assertIn(CoherenceFlag.CONFIDENCE_DROP, flags)

    def test_confidence_drop_does_not_fire_for_small_drop(self):
        prior = _verify_step(conforms=True, violations=[], rules_checked=["r1"], confidence=0.90)
        deliberate = _deliberate_step("APPROVE", warrant="ok", confidence=0.80)  # drops 0.10
        flags, _, _ = compute_coherence_flags(deliberate, [prior], [])
        self.assertNotIn(CoherenceFlag.CONFIDENCE_DROP, flags)

    def test_unwarranted_recommendation_fires_when_no_warrant(self):
        deliberate = _deliberate_step("DENY", warrant="")
        flags, _, _ = compute_coherence_flags(deliberate, [], [])
        self.assertIn(CoherenceFlag.UNWARRANTED_RECOMMENDATION, flags)

    def test_unwarranted_recommendation_does_not_fire_with_warrant(self):
        deliberate = _deliberate_step("DENY", warrant="The application violates code section 3.2.")
        flags, _, _ = compute_coherence_flags(deliberate, [], [])
        self.assertNotIn(CoherenceFlag.UNWARRANTED_RECOMMENDATION, flags)

    def test_no_flags_on_clean_workflow(self):
        verify = _verify_step(conforms=True, violations=[], rules_checked=["r1", "r2"])
        deliberate = _deliberate_step("APPROVE", warrant="All rules satisfied. Evidence supports.")
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        # May have UNWARRANTED if warrant is short — check no critical flags
        critical = {CoherenceFlag.VERIFY_DELIBERATE_TENSION, CoherenceFlag.CLASSIFY_DELIBERATE_MISMATCH}
        self.assertFalse(any(f in critical for f in flags))


# ── Overall score and warranted tests ────────────────────────────────────────

class TestOverallAndWarranted(unittest.TestCase):

    def test_no_flags_no_mechanical_is_warranted(self):
        overall, warranted = compute_overall([], [], [])
        self.assertEqual(overall, 1.0)
        self.assertTrue(warranted)

    def test_good_mechanical_no_flags_high_overall(self):
        overall, warranted = compute_overall([0.9, 1.0, 0.85], [], [])
        self.assertGreater(overall, 0.8)
        self.assertTrue(warranted)

    def test_critical_flag_makes_not_warranted(self):
        _, warranted = compute_overall(
            [0.9, 1.0],
            [],
            [CoherenceFlag.VERIFY_DELIBERATE_TENSION],
        )
        self.assertFalse(warranted)

    def test_critical_flag_reduces_overall(self):
        without_flag, _ = compute_overall([0.9], [], [])
        with_flag, _ = compute_overall([0.9], [], [CoherenceFlag.VERIFY_DELIBERATE_TENSION])
        self.assertLess(with_flag, without_flag)

    def test_multiple_flags_floor_at_0_3(self):
        all_flags = list(CoherenceFlag)
        overall, _ = compute_overall([1.0], [], all_flags)
        self.assertGreaterEqual(overall, 0.3)

    def test_low_mechanical_low_overall(self):
        overall, warranted = compute_overall([0.2, 0.3], [], [])
        self.assertLess(overall, 0.5)
        self.assertFalse(warranted)


# ── Full pipeline tests ───────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):
    """End-to-end tests matching the actual permit review conditional case."""

    def _run_permit_review_conditional(self):
        """Simulate the PMT-2026-00318 conditional case steps."""
        record = WorkflowEpistemicRecord(instance_id="wf_test")
        steps = []

        # retrieve
        s1 = _retrieve_step([
            {"source": f"get_{t}", "status": "success", "record_count": None}
            for t in ["application", "parcel_data", "sensitive_area_overlay",
                      "ceqa_exemption_text", "municipal_code", "prior_determinations"]
        ])
        ep1 = compute_step_epistemic_state(s1, [], record.open_gaps)
        record.add_step(ep1); steps.append(s1)

        # classify: conditional
        s2 = _classify_step("conditional", confidence=0.85)
        ep2 = compute_step_epistemic_state(s2, steps, record.open_gaps)
        record.add_step(ep2); steps.append(s2)

        # verify: VIOLATIONS
        s3 = _verify_step(
            conforms=False,
            violations=[{"rule": "CEQA §15332", "description": "not categorical", "severity": "major"}],
            rules_checked=["CEQA §15332", "CEQA §15070", "Housing Element"],
            confidence=0.88,
        )
        ep3 = compute_step_epistemic_state(s3, steps, record.open_gaps)
        record.add_step(ep3); steps.append(s3)

        # deliberate: APPROVE WITH CONDITIONS despite violations
        s4 = _deliberate_step(
            "APPROVE WITH CONDITIONS. CEQA Determination: Mitigated Negative Declaration.",
            # Warrant deliberately omits MND/mitigation language to keep the
            # VERIFY_DELIBERATE_TENSION flag firing in the test — this tests
            # the flag detection mechanism. Real cases with a proper MND warrant
            # will not fire (see test_clean_workflow_no_flags and the resolution
            # indicator tests in TestCoherenceFlags).
            warrant="Project consistent with MU-T zoning and Housing Element.",
            evidence_used=[{"source": "verify", "description": "violations noted"}],
            confidence=0.90,
        )
        ep4 = compute_step_epistemic_state(s4, steps, record.open_gaps)
        record.add_step(ep4); steps.append(s4)

        return record, [ep1, ep2, ep3, ep4]

    def test_verify_deliberate_tension_detected_in_pipeline(self):
        record, steps = self._run_permit_review_conditional()
        ep_deliberate = steps[3]
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, ep_deliberate.coherence_flags)

    def test_warranted_is_false_due_to_tension(self):
        record, steps = self._run_permit_review_conditional()
        ep_deliberate = steps[3]
        self.assertFalse(ep_deliberate.warranted)

    def test_workflow_record_has_flag(self):
        record, _ = self._run_permit_review_conditional()
        flag_names = [f for _, f in record.all_flags]
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flag_names)

    def test_reviewer_context_surfaces_tension(self):
        record, _ = self._run_permit_review_conditional()
        ctx = record.reviewer_context()
        self.assertFalse(ctx["warranted"])
        self.assertTrue(any("VERIFY_DELIBERATE_TENSION" in f for f in ctx["coherence_flags"]))
        self.assertTrue(any(
            s["step"] == "deliberate_determination"
            for s in ctx["low_confidence_steps"]
        ))

    def test_clean_workflow_no_flags(self):
        """A clean approve-all-pass workflow has no coherence flags."""
        record = WorkflowEpistemicRecord(instance_id="wf_clean")
        steps = []

        s1 = _retrieve_step([
            {"source": "get_data", "status": "success", "record_count": None}
        ])
        ep1 = compute_step_epistemic_state(s1, [], record.open_gaps)
        record.add_step(ep1); steps.append(s1)

        s2 = _verify_step(conforms=True, violations=[], rules_checked=["r1", "r2"])
        ep2 = compute_step_epistemic_state(s2, steps, record.open_gaps)
        record.add_step(ep2); steps.append(s2)

        s3 = _deliberate_step("APPROVE", warrant="All rules satisfied. Evidence supports.")
        ep3 = compute_step_epistemic_state(s3, steps, record.open_gaps)
        record.add_step(ep3)

        ctx = record.reviewer_context()
        critical = {"VERIFY_DELIBERATE_TENSION", "CLASSIFY_DELIBERATE_MISMATCH"}
        self.assertFalse(any(
            any(c in f for c in critical)
            for f in ctx["coherence_flags"]
        ))

    def test_record_to_dict_is_serializable(self):
        """WorkflowEpistemicRecord.to_dict() must be JSON-serializable."""
        import json
        record, _ = self._run_permit_review_conditional()
        d = record.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        self.assertIn("VERIFY_DELIBERATE_TENSION", serialized)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── Gate trigger DSL tests ────────────────────────────────────────────────────

class TestGateTriggerDSL(unittest.TestCase):
    """Gate trigger DSL evaluates correctly against WorkflowEpistemicRecord."""

    def _permit_review_record(self):
        """Build the conditional permit review record (verify violations + approve)."""
        record = WorkflowEpistemicRecord(instance_id="wf_dsl_test")
        steps = []

        s1 = _retrieve_step([
            {"source": "get_app", "status": "success", "record_count": None}
        ], confidence=0.95)
        ep1 = compute_step_epistemic_state(s1, [], record.open_gaps)
        record.add_step(ep1); steps.append(s1)

        s2 = _verify_step(
            conforms=False,
            violations=[{"rule": "r1", "severity": "major", "description": "x"}],
            rules_checked=["r1", "r2"],
            confidence=0.88,
        )
        ep2 = compute_step_epistemic_state(s2, steps, record.open_gaps)
        record.add_step(ep2); steps.append(s2)

        s3 = _deliberate_step(
            "APPROVE WITH CONDITIONS",
            warrant="Consistent with zoning.",
            evidence_used=[{"source": "verify", "description": "ok"}],
            confidence=0.90,
        )
        ep3 = compute_step_epistemic_state(s3, steps, record.open_gaps)
        record.add_step(ep3)
        return record

    def test_coherence_flag_trigger_fires(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = self._permit_review_record()
        fired = evaluate_gate_triggers(
            ["coherence_flag: VERIFY_DELIBERATE_TENSION"], record
        )
        self.assertEqual(len(fired), 1)
        self.assertIn("VERIFY_DELIBERATE_TENSION", fired[0]["description"])

    def test_not_warranted_trigger_fires(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = self._permit_review_record()
        fired = evaluate_gate_triggers(["not_warranted"], record)
        self.assertEqual(len(fired), 1)

    def test_any_step_metric_trigger_fires(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = self._permit_review_record()
        # deliberate overall is 0.75 (penalised by VERIFY_DELIBERATE_TENSION)
        fired = evaluate_gate_triggers(["any_step.overall < 0.80"], record)
        self.assertGreater(len(fired), 0)
        self.assertEqual(fired[0]["metric"], "overall")

    def test_primitive_scoped_metric_trigger(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = self._permit_review_record()
        # verify.rule_coverage is 1.0 — this should NOT fire
        fired = evaluate_gate_triggers(["verify.rule_coverage < 0.50"], record)
        self.assertEqual(len(fired), 0)

    def test_confidence_drop_trigger_fires(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = WorkflowEpistemicRecord(instance_id="wf_drop")
        steps = []
        s1 = _retrieve_step([{"source": "s1", "status": "success", "record_count": None}], confidence=0.95)
        ep1 = compute_step_epistemic_state(s1, [], record.open_gaps)
        record.add_step(ep1); steps.append(s1)
        s2 = _deliberate_step("DENY", warrant="w", confidence=0.55)  # drops 0.40
        ep2 = compute_step_epistemic_state(s2, steps, record.open_gaps)
        record.add_step(ep2)
        fired = evaluate_gate_triggers(["confidence_drop > 0.25"], record)
        self.assertEqual(len(fired), 1)
        self.assertGreater(fired[0]["value"], 0.25)

    def test_multiple_triggers_all_reported(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = self._permit_review_record()
        fired = evaluate_gate_triggers([
            "coherence_flag: VERIFY_DELIBERATE_TENSION",
            "not_warranted",
            "any_step.overall < 0.80",
        ], record)
        self.assertGreaterEqual(len(fired), 2)

    def test_no_triggers_fire_on_clean_workflow(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = WorkflowEpistemicRecord(instance_id="wf_clean")
        steps = []
        s1 = _retrieve_step([{"source": "s1", "status": "success", "record_count": None}], confidence=0.95)
        ep1 = compute_step_epistemic_state(s1, [], record.open_gaps)
        record.add_step(ep1); steps.append(s1)
        s2 = _verify_step(conforms=True, violations=[], rules_checked=["r1"], confidence=0.90)
        ep2 = compute_step_epistemic_state(s2, steps, record.open_gaps)
        record.add_step(ep2); steps.append(s2)
        s3 = _deliberate_step("APPROVE", warrant="All checks pass. Evidence solid.", evidence_used=[
            {"source": "verify", "description": "passed"}], confidence=0.92)
        ep3 = compute_step_epistemic_state(s3, steps, record.open_gaps)
        record.add_step(ep3)
        fired = evaluate_gate_triggers([
            "coherence_flag: VERIFY_DELIBERATE_TENSION",
            "coherence_flag: CLASSIFY_DELIBERATE_MISMATCH",
            "not_warranted",
            "confidence_drop > 0.25",
        ], record)
        self.assertEqual(len(fired), 0)

    def test_unrecognised_trigger_does_not_raise(self):
        from cognitive_core.engine.epistemic import evaluate_gate_triggers
        record = WorkflowEpistemicRecord(instance_id="wf_x")
        # Should not raise, just return nothing fired
        fired = evaluate_gate_triggers(["this_is_not_valid_syntax"], record)
        self.assertEqual(fired, [])


class TestVerifyDeliberateTensionResolution(unittest.TestCase):
    """
    VERIFY_DELIBERATE_TENSION uses warrant resolution indicators to
    distinguish genuine unresolved tension from correctly-handled cases
    where a verify violation leads to a conditional approval pathway.

    The canonical case: CEQA categorical exemption fails (verify violation),
    but deliberate correctly recommends MND pathway (approve with conditions).
    The warrant should acknowledge the MND pathway — if it does, no flag.
    If it approves without addressing the violation, flag fires.
    """

    def _verify_with_violation(self):
        return _verify_step(
            conforms=False,
            violations=[{"rule": "CEQA §15332", "severity": "major", "description": "cond d fails"}],
            rules_checked=["§15332(a)", "§15332(b)", "§15332(c)", "§15332(d)"],
        )

    def test_mnd_warrant_resolves_tension(self):
        verify = self._verify_with_violation()
        deliberate = _deliberate_step(
            "APPROVE WITH CONDITIONS",
            warrant="Project requires MND. Mitigation measures address all impacts.",
        )
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertNotIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_mitigated_negative_declaration_resolves(self):
        verify = self._verify_with_violation()
        deliberate = _deliberate_step(
            "APPROVE WITH CONDITIONS. Adopt Mitigated Negative Declaration per §15070.",
            warrant="Mitigated Negative Declaration is the correct pathway.",
        )
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertNotIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_conditions_of_approval_resolves(self):
        verify = self._verify_with_violation()
        deliberate = _deliberate_step(
            "APPROVE WITH CONDITIONS",
            warrant="Conditions of approval address all violations identified.",
        )
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertNotIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_approve_without_resolution_flags(self):
        """Approving with no acknowledgment of violation is genuine tension."""
        verify = self._verify_with_violation()
        deliberate = _deliberate_step(
            "APPROVE the application.",
            warrant="The project is consistent with the General Plan.",
        )
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_approve_with_conditions_in_action_but_not_warrant_still_flags(self):
        """Resolution must be in the WARRANT, not just the recommended action."""
        verify = self._verify_with_violation()
        deliberate = _deliberate_step(
            "APPROVE WITH CONDITIONS",  # 'conditions' in action but not warrant
            warrant="The project meets all applicable development standards.",
        )
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_empty_warrant_flags(self):
        verify = self._verify_with_violation()
        deliberate = _deliberate_step("APPROVE", warrant="")
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)

    def test_no_tension_when_conforms_true(self):
        """No flag when verify passes regardless of warrant."""
        verify = _verify_step(conforms=True, violations=[], rules_checked=["r1"])
        deliberate = _deliberate_step("APPROVE", warrant="All rules satisfied.")
        flags, _, _ = compute_coherence_flags(deliberate, [verify], [])
        self.assertNotIn(CoherenceFlag.VERIFY_DELIBERATE_TENSION, flags)