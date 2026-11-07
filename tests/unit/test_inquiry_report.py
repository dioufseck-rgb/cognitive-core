"""Reports preserve claim provenance and distinguish pending from completed review."""

import unittest

from cognitive_core.inquiry.demo import (
    DEMO_URL, demo_challenge, demo_finding, demo_frame, demo_reflection,
)
from cognitive_core.inquiry.models import (
    FindingOutput, InquiryEvent, InquiryReflection, InquiryState, Passage, QuestionRecord,
    SourceRecord,
)
from cognitive_core.inquiry.report import render_report
from cognitive_core.primitives.schemas import ChallengeOutput, FrameOutput


class InquiryReportTests(unittest.TestCase):
    def setUp(self):
        frame = FrameOutput.model_validate(demo_frame())
        self.question = QuestionRecord(specification=frame.subquestions[0], frame_version=1)
        self.state = InquiryState(
            original_question=frame.working_question,
            frames=[frame], questions={self.question.specification.id: self.question},
            status="budget_exhausted", phase="work",
        )

    def add_finding(self, claims=None):
        data = demo_finding(revised=True)
        if claims is not None:
            data["claims"] = claims
        self.question.findings.append(FindingOutput.model_validate(data))

    def add_read_source(self):
        self.state.sources["S1"] = SourceRecord(
            id="S1", url=DEMO_URL, title="A synthetic fixture", status="read",
            passages=[Passage(id="S1:P1", text="In this example, checking a supplied account ID follows an exact rule.")],
        )

    def test_inference_is_not_presented_as_source_evidence_when_other_sources_were_read(self):
        self.add_read_source()
        self.add_finding([
            {"text": "The design may improve efficiency.", "basis": "inference"},
            {"text": "Requests remain similar over time.", "basis": "assumption"},
            {"text": "The pilot only drafts answers.", "basis": "user"},
        ])

        report = render_report(self.state)

        self.assertIn("findings below are provisional", report)
        self.assertIn("Read sources: 1", report)
        self.assertIn("No claim in the latest finding cites a read passage.", report)
        self.assertIn("**Inference:** The design may improve efficiency.", report)
        self.assertIn("**Assumption:** Requests remain similar over time.", report)
        self.assertIn("**User-supplied claim:** The pilot only drafts answers.", report)
        self.assertNotIn("[S1:P1](", report)

    def test_source_claim_preserves_its_passage_link_and_qualification(self):
        self.add_read_source()
        self.add_finding()

        report = render_report(self.state)

        self.assertIn("Read passages are cited for 1 of 1 claim.", report)
        self.assertIn("**Source claim:**", report)
        self.assertIn(f"[S1:P1](<{DEMO_URL}>)", report)
        self.assertIn("Synthetic example; it establishes no real-world performance advantage.", report)
        self.assertIn("supplied by the model", report)

    def test_missing_unread_or_mismatched_evidence_is_never_rendered_as_a_passage_link(self):
        for source_status, passage_text in (
            (None, None),
            ("discovered", "checking a supplied account ID follows an exact rule"),
            ("read", "This passage does not contain the claimed quotation."),
        ):
            with self.subTest(source_status=source_status):
                self.state.sources.clear()
                self.question.findings.clear()
                if source_status is not None:
                    self.state.sources["S1"] = SourceRecord(
                        id="S1", url=DEMO_URL, status=source_status,
                        passages=[Passage(id="S1:P1", text=passage_text)],
                    )
                self.add_finding()

                report = render_report(self.state)

                self.assertNotIn("[S1:P1](", report)
                self.assertIn("Unavailable evidence references: S1:P1.", report)
                self.assertIn("No claim in the latest finding cites a read passage.", report)

    def test_new_finding_does_not_inherit_previous_review_verdict(self):
        self.add_finding()
        self.question.challenges.append(ChallengeOutput.model_validate(demo_challenge(False)))
        self.question.reflections.append(InquiryReflection.model_validate(demo_reflection("revise")))
        self.add_finding([{"text": "The result is only a conditional argument.", "basis": "inference"}])

        report = render_report(self.state)

        self.assertIn("The latest finding has not yet been challenged.", report)
        self.assertIn("Latest critique of an earlier finding", report)
        self.assertIn("findings drafted 2; critiques 1; review decisions 1.", report)
        self.assertIn("Suggested remedy: Remove the performance claim", report)

    def test_new_critique_does_not_inherit_an_older_objection_response(self):
        self.add_finding()
        self.question.challenges.append(ChallengeOutput.model_validate(demo_challenge()))
        first_reflection = demo_reflection()
        first_reflection["objection_response"] = "An old response that must not be attached to a new critique."
        self.question.reflections.append(InquiryReflection.model_validate(first_reflection))
        self.add_finding()
        self.question.challenges.append(ChallengeOutput.model_validate(demo_challenge(False)))

        report = render_report(self.state)

        self.assertIn("Latest critique (model assessment):** does not survive.", report)
        self.assertNotIn(first_reflection["objection_response"], report)
        self.assertNotIn("Last review decision (model):", report)

    def test_acceptance_after_an_objection_includes_the_response_and_retains_open_issue(self):
        self.add_finding()
        self.question.status = "conditional"
        self.question.challenges.append(ChallengeOutput.model_validate(demo_challenge(False)))
        reflection = demo_reflection()
        reflection["objection_response"] = "The finding expressly treats the benefit as an untested inference."
        self.question.reflections.append(InquiryReflection.model_validate(reflection))

        report = render_report(self.state)

        self.assertIn("Last review decision (model):** accept (conditional).", report)
        self.assertIn("Response to critique: " + reflection["objection_response"], report)
        self.assertIn("No empirical evidence about accuracy, latency, cost, or outcomes.", report)

    def test_progress_reports_read_unread_failed_sources_and_undrafted_branches(self):
        self.add_read_source()
        self.state.sources["S2"] = SourceRecord(id="S2", url="https://example.org/unread")
        self.state.sources["S3"] = SourceRecord(id="S3", url="https://example.org/failed", status="failed")

        report = render_report(self.state)

        self.assertIn("Read sources: 1; unread leads: 1; failed sources: 1.", report)
        self.assertIn("No finding has been drafted.", report)

    def test_final_answer_citation_resolution_is_preserved_without_dumping_reasoning(self):
        self.add_read_source()
        self.add_finding()
        self.state.status = "completed"
        self.state.phase = "done"
        self.state.answer = "The fixture illustrates an exact check. [S1:P1]"
        self.question.findings[-1].reasoning = "A private assessment that does not belong in the report."

        report = render_report(self.state)

        self.assertIn(f"## Answer\n\nThe fixture illustrates an exact check. [S1:P1](<{DEMO_URL}>)", report)
        self.assertNotIn(self.question.findings[-1].reasoning, report)
        self.assertNotIn("This is an unfinished inquiry", report)

    def test_model_operation_counts_exclude_started_cached_tool_and_controller_events(self):
        other_events = ["step_started", "cache_hit", "framed", "budget_exhausted",
                        "web_search", "read_source", "recovery"]
        self.state.events = [InquiryEvent(
            sequence=i, step=i, kind=kind, summary="retrieve",
            details={"operation": "retrieve"},
        ) for i, kind in enumerate(other_events)]
        self.assertNotIn("Completed model operations:", render_report(self.state))
        for kind in ("frame", "retrieve", "retrieve", "investigate", "challenge",
                     "reflect", "deliberate", "generate"):
            self.state.events.append(InquiryEvent(
                sequence=len(self.state.events), step=len(self.state.events), kind=kind,
                summary="Operation completed",
            ))

        report = render_report(self.state)

        operation_line = next(line for line in report.splitlines()
                              if line.startswith("Completed model operations:"))
        self.assertEqual(operation_line,
                         "Completed model operations: frame: 1; retrieval planning: 2; "
                         "investigate: 1; challenge: 1; reflect: 1; deliberate: 1; generate: 1.")


if __name__ == "__main__":
    unittest.main()
