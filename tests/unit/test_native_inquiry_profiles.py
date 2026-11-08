"""Native inquiry artifact/provenance contracts, without model or tool execution."""

from copy import deepcopy
import json
import unittest

from pydantic import ValidationError

from cognitive_core.inquiry.models import RetrievalPlan
from cognitive_core.inquiry.native_models import NativeStudy, StudyValidationError
from cognitive_core.inquiry.native_profiles import (
    apply_output,
    get_profile,
    ingest_research_result,
    initial_study,
    record_clarification_answer,
    review_basis,
    study_context,
    validate_decision_target,
    validate_retrieval_options,
    validate_retrieval_plan,
)
from cognitive_core.primitives.schemas import GovernOutput


STUDY = {"kind": "study"}
Q1 = {"kind": "question", "question_id": "q1"}
REPORT = {"kind": "report"}
URL = "https://example.com/policy"
PASSAGE = "The system searches competitor documents and drafts policies."


def envelope(**values):
    return {"confidence": 0.7, "reasoning": "A bounded assessment of the supplied material.", **values}


def frame_output(dependent=False, clarification=False):
    questions = [{"id": "q1", "question": "What does the evidence establish?",
                  "contribution": "Identify the relevant comparison.", "depends_on": [],
                  "evidence_needs": ["Comparable task, baseline, and outcome."]}]
    if dependent:
        questions.append({"id": "q2", "question": "What follows for the pilot?",
                          "contribution": "Apply the qualified evidence.", "depends_on": ["q1"],
                          "evidence_needs": []})
    return envelope(
        working_question="When does adaptive retrieval help answer internal policy questions?",
        purpose="Define criteria for a read-only pilot using public research sources.",
        subquestions=questions, answer_criteria=["Distinguish mechanisms from measured comparisons."],
        scope=["Internal policy answers; public sources provide research evidence."],
        assumptions=[], clarifications=[{
            "id": "c1", "question": "Which pilot outcome matters?",
            "reason": "The outcome changes the relevant architecture comparison.", "blocking": True,
        }] if clarification else [],
    )


def framed(**kwargs):
    return apply_output(initial_study("When does adaptive retrieval help?"), "frame",
                        frame_output(**kwargs), STUDY, "frame-1")


def with_source(study=None, text=PASSAGE):
    study = study or framed()
    study = ingest_research_result(study, "web_search", {"query": "policy retrieval comparison"},
        {"results": [{"url": URL, "title": "A policy authoring case", "snippet": "A discovery lead."}]},
        "search-1", Q1)
    return ingest_research_result(study, "read_source", {"url": URL}, {
        "url": URL, "requested_url": URL, "title": "A policy authoring case", "text": text,
        "content_type": "text/plain", "truncated": False, "retrieved_at": "2026-11-08T12:00:00Z",
    }, "read-1", Q1)


def source_claim():
    return {"text": "The case describes policy authoring from competitor documents.",
            "basis": "source", "evidence": [{"passage_id": "S1:P1", "quote": PASSAGE}],
            "qualification": "It is not a controlled comparison of policy question answering.",
            "support_assessment": "The passage describes the source task; no fixed-versus-adaptive outcome is reported."}


def finding_output(claim=None, disposition="conditional"):
    return envelope(finding="The example is adjacent evidence, not the requested comparison.",
                    claims=[claim or source_claim()], question_disposition=disposition,
                    unresolved_issues=["Comparative answer-quality evidence remains absent."],
                    material_change="Narrowed what the case study can establish.")


def synthesis_output():
    return envelope(situation_summary="The available case concerns an adjacent task.",
                    conclusion="The comparison remains evidence-limited.", claims=[source_claim()],
                    limitations=["No matched comparison was supplied."], open_questions=[],
                    research_outcome="evidence_limited",
                    criteria_coverage={"Distinguish mechanisms from measured comparisons.": "Evidence-limited."})


def challenge_output(survives=True):
    return envelope(survives=survives, vulnerabilities=[], strengths=["The task mismatch is explicit."],
                    overall_assessment="The report accurately represents its limited evidence.")


def report_ready():
    study = with_source()
    study = apply_output(study, "investigate", finding_output(), Q1, "finding-1")
    study = apply_output(study, "deliberate", synthesis_output(), STUDY, "synthesis-1")
    return apply_output(study, "generate", envelope(
        artifact="The case concerns policy authoring [S1:P1]. The architecture comparison remains unresolved.",
        format="markdown"), REPORT, "report-1")


class NativeInquiryProfilesTests(unittest.TestCase):
    def test_profiles_bind_native_operations_and_preserve_govern_contract(self):
        self.assertIs(get_profile("inquiry.govern.v1").schema, GovernOutput)
        self.assertIs(get_profile("inquiry.retrieve.v1").schema, RetrievalPlan)
        self.assertEqual(get_profile("inquiry.finding.v1").primitive, "investigate")
        with self.assertRaises(StudyValidationError):
            get_profile("arbitrary.module:Class")

    def test_initial_state_is_serializable_and_runtime_owns_counters(self):
        initial = initial_study("Read " + URL, limits={"max_steps": 9})
        self.assertEqual(initial["limits"]["max_steps"], 9)
        self.assertEqual(initial["usage"], {"model_calls": 0, "searches": 0, "reads": 0, "steps": 0})
        self.assertEqual(initial["sources"]["S1"]["status"], "discovered")
        study = apply_output(initial, "frame", frame_output(), STUDY, "frame-1")
        self.assertEqual(study["frame"], study["frames"][-1])
        self.assertEqual(study["usage"], initial["usage"])
        self.assertIsNone(initial["frame"])
        NativeStudy.model_validate(json.loads(json.dumps(study)))

    def test_search_snippets_cannot_be_cited_as_read_evidence(self):
        study = framed()
        study = ingest_research_result(study, "web_search", {"query": "policy comparison"},
            {"results": [{"url": URL, "snippet": PASSAGE}]}, "search", Q1)
        before = deepcopy(study)
        with self.assertRaisesRegex(StudyValidationError, "not read and supplied"):
            apply_output(study, "investigate", finding_output(), Q1, "bad-finding")
        self.assertEqual(study, before)

    def test_source_claim_requires_explicit_support_assessment(self):
        claim = source_claim()
        del claim["support_assessment"]
        with self.assertRaises(ValidationError):
            get_profile("inquiry.finding.v1").schema.model_validate(finding_output(claim))

    def test_claim_quote_must_occur_in_the_exact_cited_passage(self):
        study = with_source(text=PASSAGE + " " * 2400 + "A separate sentence.")
        claim = source_claim()
        claim["evidence"][0]["passage_id"] = "S1:P2"
        with self.assertRaisesRegex(StudyValidationError, "does not occur"):
            apply_output(study, "investigate", finding_output(claim), Q1, "wrong-passage")
        accepted = apply_output(study, "investigate", finding_output(), Q1, "finding")
        self.assertEqual(accepted["questions"]["q1"]["status"], "conditional")
        self.assertEqual(accepted["claims"]["C1"]["support_assessment"], source_claim()["support_assessment"])

    def test_unknown_read_url_is_rejected_before_and_after_execution(self):
        study = framed()
        plan = {"calls": [{"tool": "read_source", "arguments": {"url": URL},
                           "purpose": "Read a useful report."}], "rationale": "Need direct evidence."}
        with self.assertRaisesRegex(StudyValidationError, "discovered or user-supplied"):
            validate_retrieval_plan(study, plan)
        with self.assertRaises(StudyValidationError):
            ingest_research_result(study, "read_source", {"url": URL}, {"url": URL, "text": PASSAGE}, "read", Q1)

    def test_retrieval_policy_is_optional_and_strict_when_configured(self):
        self.assertEqual(validate_retrieval_options({}), {
            "allowed_tools": ["web_search", "read_source"], "require_new_tool_call": False,
        })
        configured = {"allowed_tools": ["read_source"], "require_new_tool_call": True,
                      "instructions": "Read the discovered comparison."}
        options = validate_retrieval_options(configured)
        self.assertEqual(options, {"allowed_tools": ["read_source"], "require_new_tool_call": True})
        options["allowed_tools"].append("web_search")
        self.assertEqual(configured["allowed_tools"], ["read_source"])
        for invalid in (
            {"allowed_tools": []}, {"allowed_tools": "read_source"}, {"allowed_tools": None},
            {"allowed_tools": ["read_source", "read_source"]}, {"allowed_tools": ["execute"]},
            {"allowed_tools": [["read_source"]]}, {"require_new_tool_call": "true"},
            {"require_new_tool_call": 1}, {"require_new_tool_call": None},
        ):
            with self.subTest(params=invalid), self.assertRaises(StudyValidationError):
                validate_retrieval_options(invalid)

    def test_new_source_version_preserves_existing_passage_identity(self):
        study = with_source()
        original = deepcopy(study["sources"]["S1"])
        updated = ingest_research_result(study, "read_source", {"url": URL},
            {"url": URL, "text": "A changed page with a different conclusion."}, "read-2", Q1)
        self.assertEqual(updated["sources"]["S1"], original)
        self.assertEqual(updated["sources"]["S2"]["supersedes"], "S1")
        self.assertEqual(updated["sources"]["S2"]["passages"][0]["id"], "S2:P1")

    def test_tool_result_replay_is_idempotent_and_conflict_is_rejected(self):
        study = framed()
        payload = {"results": [{"url": URL}]}
        once = ingest_research_result(study, "web_search", {"query": "research"}, payload, "call", Q1)
        twice = ingest_research_result(once, "web_search", {"query": "research"}, payload, "call", Q1)
        self.assertEqual(once, twice)
        with self.assertRaisesRegex(StudyValidationError, "identity was reused"):
            ingest_research_result(once, "web_search", {"query": "research"}, {"results": []}, "call", Q1)

    def test_targets_are_typed_and_revisions_cannot_be_forged(self):
        study = framed()
        for target in ({"kind": "question", "question_id": "missing"},
                       {"kind": "question", "question_id": "q1", "expected_study_revision": 0},
                       {"kind": "question", "question_id": "q1", "expected_study_revision": True},
                       {"kind": "question", "question_id": "q1", "invented": "field"}):
            with self.assertRaises(StudyValidationError):
                validate_decision_target(study, target, "investigate")
        with self.assertRaises(StudyValidationError):
            validate_decision_target(study, STUDY, "investigate")

    def test_revising_a_prerequisite_invalidates_dependent_findings(self):
        study = with_source(framed(dependent=True))
        study = apply_output(study, "investigate", finding_output(), Q1, "first")
        inferred = {"text": "Pilot criteria remain conditional on a matched comparison.", "basis": "inference",
                    "premises": ["The comparison is unresolved."]}
        study = apply_output(study, "investigate", finding_output(inferred),
                             {"kind": "question", "question_id": "q2"}, "dependent")
        revised = apply_output(study, "investigate", finding_output(), Q1, "revised")
        self.assertEqual(revised["questions"]["q2"]["status"], "active")
        self.assertIsNone(revised["questions"]["q2"]["current_finding_id"])
        self.assertTrue(next(f for f in revised["findings"] if f["invocation_id"] == "dependent")["stale"])

    def test_unresolved_prerequisite_permits_only_qualified_dependent_findings(self):
        study = framed(dependent=True)
        target = {"kind": "question", "question_id": "q2"}
        validate_decision_target(study, target, "investigate")
        claim = {"text": "The recommendation depends on evidence not yet examined.", "basis": "inference"}
        with self.assertRaisesRegex(StudyValidationError, "unresolved prerequisites"):
            apply_output(study, "investigate", finding_output(claim, disposition="answered"), target, "too-strong")
        conditional = apply_output(study, "investigate", finding_output(claim), target, "conditional")
        self.assertEqual(conditional["findings"][-1]["unresolved_dependencies"], ["q1"])
        self.assertEqual(study_context(conditional, target)["unresolved_dependencies"], ["q1"])

    def test_reframe_keeps_history_and_invalidates_changed_scope(self):
        study = with_source()
        study = apply_output(study, "investigate", finding_output(), Q1, "finding")
        revised_frame = frame_output()
        revised_frame["scope"] = ["A newly specified application boundary."]
        revised = apply_output(study, "frame", revised_frame, STUDY, "reframe")
        self.assertEqual(len(revised["frames"]), 2)
        self.assertTrue(revised["findings"][0]["stale"])
        self.assertEqual(len(revised["archived_questions"]), 1)
        self.assertEqual(revised["sources"], study["sources"])

    def test_context_supplies_actual_passages_and_complete_target(self):
        study = with_source(text=PASSAGE + "\n" + "x" * 14000)
        view = study_context({"study": study}, Q1)
        self.assertEqual(view["active_question"]["id"], "q1")
        self.assertEqual(view["evidence"][0]["text"], study["sources"]["S1"]["passages"][0]["text"])
        self.assertGreater(len(view["evidence"][0]["text"]), 120)
        self.assertLessEqual(len(json.dumps(view, ensure_ascii=False)), study["limits"]["max_context_chars"])
        self.assertEqual("".join(p["text"] for p in study["sources"]["S1"]["passages"]), PASSAGE + "\n" + "x" * 14000)

    def test_minimum_context_preserves_a_usable_read_url_and_existing_evidence(self):
        for scenario in ("discovered", "failed_fallback", "read_and_unread"):
            with self.subTest(scenario=scenario):
                study = with_source() if scenario == "read_and_unread" else framed()
                study["limits"]["max_context_chars"] = 4000
                next_url = URL + "/next-comparison"
                study = ingest_research_result(study, "web_search", {"query": "next comparison"}, {
                    "results": [{"url": next_url, "title": "An unread comparison", "snippet": "Discovery only."}],
                }, "next-search", Q1)
                if scenario == "failed_fallback":
                    study = ingest_research_result(study, "read_source", {"url": next_url},
                                                  {"error": "Temporary failure."}, "failed-read", Q1)
                before = deepcopy(study)
                view = study_context(study, Q1)
                retained = [source for source in view["source_catalog"] if source["url"] == next_url]
                self.assertEqual(len(retained), 1)
                self.assertEqual(retained[0]["status"], "failed" if scenario == "failed_fallback" else "discovered")
                self.assertLessEqual(len(json.dumps(view, ensure_ascii=False)), 4000)
                if scenario == "read_and_unread":
                    self.assertTrue(any(passage["text"] == PASSAGE for passage in view["evidence"]))
                self.assertEqual(study, before)

    def test_progress_counts_reads_leads_and_failures_without_promoting_discovery(self):
        study = with_source()
        unread_url, failed_url = URL + "/unread", URL + "/unavailable"
        study = ingest_research_result(study, "web_search", {"query": "another comparison"}, {
            "results": [{"url": unread_url}, {"url": failed_url}],
        }, "search-2", Q1)
        study = ingest_research_result(study, "read_source", {"url": failed_url},
                                      {"error": "Source returned an access error."}, "failed-read", Q1)
        study["usage"].update(searches=2, reads=2)
        study = apply_output(study, "retrieve", {
            "plan": {"calls": [{"tool": "web_search", "arguments": {"query": "another comparison"},
                                "purpose": "Discover a comparison."}], "rationale": "Find relevant evidence."},
            "sources_queried": [{"source": "web_search", "status": "success"}],
        }, Q1, "search-2")
        before = deepcopy(study)
        progress = study_context(study, STUDY)["research_progress"]
        self.assertEqual(progress["read_sources"], 1)
        self.assertEqual(progress["unread_leads"], 1)
        self.assertEqual(progress["failed_sources"], 1)
        self.assertEqual(progress["unread_source_ids"], ["S2"])
        self.assertEqual(progress["remaining_tool_calls"], {
            "web_search": study["limits"]["max_searches"] - 2,
            "read_source": study["limits"]["max_reads"] - 2,
        })
        latest = progress["recent_retrievals"][-1]
        self.assertEqual(latest["outcome"], "discovery_only")
        self.assertTrue(latest["no_new_read_evidence"])
        self.assertEqual(study, before)

    def test_progress_exposes_empty_cached_and_failed_retrieval_rationales(self):
        study = with_source()
        read_plan = {"calls": [{"tool": "read_source", "arguments": {"url": URL},
                                "purpose": "Read the comparison."}], "rationale": "Inspect the actual text."}
        study = apply_output(study, "retrieve", {
            "plan": read_plan, "sources_queried": [{"source": "read_source", "status": "success"}],
        }, Q1, "read-1")
        first = study_context(study, STUDY)["research_progress"]["recent_retrievals"][-1]
        self.assertEqual(first["new_read_source_ids"], ["S1"])
        self.assertFalse(first["no_new_read_evidence"])
        self.assertEqual(first["outcome"], "source_text_read")
        study = apply_output(study, "retrieve", {
            "plan": {"calls": [], "rationale": "Use the existing evidence to reason."}, "sources_queried": [],
        }, Q1, "empty")
        study = apply_output(study, "retrieve", {
            "plan": read_plan, "sources_queried": [{"source": "read_source", "status": "success", "cached": True}],
        }, Q1, "cached")
        study = apply_output(study, "retrieve", {
            "plan": read_plan, "sources_queried": [{"source": "read_source", "status": "error", "error": "Timed out."}],
        }, Q1, "failed")
        before = deepcopy(study)
        progress = study_context(study, Q1)["research_progress"]
        recent = progress["recent_retrievals"]
        self.assertEqual([record["outcome"] for record in recent], ["empty_plan", "cache_only", "tool_failure"])
        self.assertEqual(recent[0]["rationale"], "Use the existing evidence to reason.")
        self.assertEqual(recent[1]["cached_calls"], 1)
        self.assertEqual(recent[1]["actual_calls"], 0)
        self.assertEqual(recent[2]["errors"], ["Timed out."])
        self.assertTrue(all(record["no_new_read_evidence"] for record in recent))
        self.assertEqual(progress["retrievals_omitted"], 1)
        self.assertEqual(study, before)

    def test_long_retrieval_history_is_bounded_without_truncating_stored_rationales(self):
        study = framed()
        study["limits"]["max_context_chars"] = 4000
        rationale = ("No additional useful retrieval is available. " * 300).strip()
        for index in range(8):
            study = apply_output(study, "retrieve", {
                "plan": {"calls": [], "rationale": rationale}, "sources_queried": [],
            }, Q1, f"empty-{index}")
        view = study_context(study, Q1)
        self.assertLessEqual(len(json.dumps(view, ensure_ascii=False)), 4000)
        self.assertLessEqual(len(view["research_progress"]["recent_retrievals"]), 3)
        self.assertTrue(all(record["plan"]["rationale"] == rationale for record in study["retrievals"]))

    def test_final_review_does_not_invalidate_itself_or_change_research_outcome(self):
        study = report_ready()
        basis = review_basis(study)
        reviewed = apply_output(study, "challenge", challenge_output(), REPORT, "review")
        self.assertGreater(reviewed["revision"], study["revision"])
        self.assertEqual(reviewed["final_review"]["basis"], basis)
        self.assertEqual(review_basis(reviewed), basis)
        reviewed["usage"]["model_calls"] += 1
        self.assertEqual(review_basis(reviewed), basis)
        governed = apply_output(reviewed, "govern", envelope(
            tier_applied="auto", tier_rationale="The limited report is reviewed.", disposition="proceed"),
            REPORT, "govern")
        self.assertEqual(governed["outcome"], "evidence_limited")
        self.assertEqual(review_basis(governed), basis)
        with self.assertRaisesRegex(StudyValidationError, "terminal"):
            validate_decision_target(governed, Q1, "investigate")

    def test_orchestrator_overview_exposes_report_and_current_review(self):
        study = apply_output(report_ready(), "challenge", challenge_output(), REPORT, "review")
        overview = study_context(study, STUDY)
        self.assertEqual(overview["report"]["revision"], 1)
        self.assertNotIn("artifact", overview["report"])
        self.assertTrue(overview["final_review"]["current"])
        self.assertTrue(overview["final_review"]["survives"])
        self.assertEqual(overview["outcome"], "evidence_limited")
        self.assertEqual(study_context(study, REPORT)["report"]["artifact"], study["report"]["artifact"])

    def test_branch_pass_cannot_authorize_final_governance(self):
        study = report_ready()
        branch_review = apply_output(study, "challenge", challenge_output(), Q1, "branch-review")
        self.assertIsNone(branch_review["final_review"])
        with self.assertRaisesRegex(StudyValidationError, "final report review"):
            validate_decision_target(branch_review, REPORT, "govern")

    def test_new_evidence_invalidates_review_but_preserves_old_report(self):
        study = apply_output(report_ready(), "challenge", challenge_output(), REPORT, "review")
        changed = ingest_research_result(study, "read_source", {"url": URL},
            {"url": URL, "text": "A newly reported matched comparison."}, "new-read", Q1)
        self.assertTrue(changed["report"]["stale"])
        self.assertFalse(changed["final_review"]["valid"])
        self.assertEqual(changed["reports"][0]["artifact"], study["report"]["artifact"])
        with self.assertRaises(StudyValidationError):
            validate_decision_target(changed, REPORT, "govern")

    def test_report_cannot_invent_citations_or_drop_source_support(self):
        study = report_ready()
        for artifact in ("An uncited result.", "A fabricated passage [S99:P1].", "An invalid ID [S0:P1].",
                         "A source without a passage [S1]."):
            with self.assertRaises(StudyValidationError):
                apply_output(study, "generate", envelope(artifact=artifact, format="markdown"),
                             REPORT, "bad-report")

    def test_output_replay_does_not_duplicate_a_finding(self):
        study = with_source()
        target = {**Q1, "expected_study_revision": study["revision"]}
        once = apply_output(study, "investigate", finding_output(), target, "finding")
        twice = apply_output(once, "investigate", finding_output(), target, "finding")
        self.assertEqual(once, twice)
        different = finding_output(disposition="active")
        with self.assertRaisesRegex(StudyValidationError, "identity was reused"):
            apply_output(once, "investigate", different, target, "finding")

    def test_clarification_is_versioned_idempotent_and_not_governance_approval(self):
        study = framed(clarification=True)
        request = study["clarifications"]["c1"]
        self.assertEqual(request["requested_revision"], study["revision"])
        answered = record_clarification_answer(study, "c1", "Accuracy matters; use " + URL,
                                               expected_revision=study["revision"])
        self.assertEqual(answered["clarifications"]["c1"]["answer"], "Accuracy matters; use " + URL)
        self.assertIsNone(answered["governance"])
        self.assertEqual(answered["sources"]["S1"]["url"], URL)
        self.assertEqual(record_clarification_answer(answered, "c1", "Accuracy matters; use " + URL), answered)
        with self.assertRaises(StudyValidationError):
            record_clarification_answer(answered, "c1", "A different answer")

    def test_reflection_is_scoped_advice_and_cannot_rewrite_findings(self):
        study = with_source()
        study = apply_output(study, "investigate", finding_output(), Q1, "finding")
        reflected = apply_output(study, "reflect", envelope(
            what_changed="The source concerns a different task.", trajectory="revise",
            revision_target="retrieve", next_question="Is there a matched comparison?"), Q1, "reflect")
        self.assertEqual(reflected["findings"], study["findings"])
        self.assertTrue(reflected["reflections"][0]["advisory"])
        self.assertEqual(reflected["reflections"][0]["target"], Q1)
        self.assertEqual(study_context(reflected, STUDY)["reflections"], [])
        self.assertEqual(len(study_context(reflected, Q1)["reflections"]), 1)
        revised = apply_output(reflected, "investigate", finding_output(), Q1, "revision")
        self.assertEqual(revised["reflections"][0]["consumed_by"], "revision")
        self.assertEqual(study_context(revised, Q1)["reflections"], [])

    def test_authoritative_retrieve_result_does_not_claim_to_have_assessed_evidence(self):
        study = with_source()
        output = {"plan": {"calls": [], "rationale": "Reason conditionally with the evidence gap explicit."},
                  "sources_queried": [{"source": "read_source", "status": "success"}]}
        result = apply_output(study, "retrieve", output, Q1, "retrieve")
        self.assertEqual(result["retrievals"][0]["sources_queried"], output["sources_queried"])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["usage"], study["usage"])


if __name__ == "__main__":
    unittest.main()
