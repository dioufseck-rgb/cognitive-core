"""Regression tests from inquiry lifecycle and evidence-context review."""

import json
from pathlib import Path
import tempfile
import unittest

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.models import (
    Claim,
    EvidenceReference,
    FindingOutput,
    InquiryLimits,
    InquiryState,
    Passage,
    QuestionRecord,
    ReadArguments,
    ReadCall,
    SearchRecord,
    SourceRecord,
    Usage,
    UserClarification,
)
from cognitive_core.inquiry.runner import InquiryRunner
from cognitive_core.inquiry.store import InquiryStore
from cognitive_core.primitives.schemas import FrameOutput, FrameQuestion


def frame(question_count=2, *, answer_criteria=None):
    return FrameOutput(
        working_question="When does a reusable capability help delivery?",
        purpose="Select a useful architecture boundary.",
        subquestions=[
            FrameQuestion(
                id=f"q{i}",
                question=f"What evidence informs architecture dimension {i}?",
                contribution=f"Assess dimension {i} of the architecture choice.",
            )
            for i in range(1, question_count + 1)
        ],
        answer_criteria=answer_criteria or ["Explain benefits and coordination costs."],
        confidence=0.8,
        reasoning="Separate consequential dimensions of the design choice.",
    )


def inquiry(plan=None, **kwargs):
    plan = plan or frame()
    return InquiryState(
        original_question="When does a reusable capability help delivery?",
        frames=[plan],
        questions={
            q.id: QuestionRecord(specification=q, frame_version=1)
            for q in plan.subquestions
        },
        **kwargs,
    )


def finding(claims=None):
    return FindingOutput(
        finding="The appropriate boundary depends on evidence about reuse and coordination.",
        claims=claims or [{"text": "The appropriate boundary depends on the design objective.", "basis": "inference"}],
        confidence=0.7,
        reasoning="The available evidence supports a qualified conclusion.",
    )


class NoModelCalls:
    def invoke(self, *args, **kwargs):
        raise AssertionError("This regression must not spend a model call")


class OneFrame:
    def __init__(self, output):
        self.output = output

    def invoke(self, primitive, **kwargs):
        if primitive != "frame":
            raise AssertionError("Only the frame operation is expected")
        return self.output


class TestInquiryReview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = InquiryStore(Path(self.directory.name) / "inquiry")
        self.runner = InquiryRunner(NoModelCalls(), ToolRegistry(), self.store)

    def test_new_search_results_after_thirty_sources_remain_available_to_active_question(self):
        state = inquiry()
        for i in range(1, 51):
            state.sources[f"S{i}"] = SourceRecord(
                id=f"S{i}",
                url=f"https://example.com/report/{i}",
                title=f"Architecture research source {i}",
                snippet=("Context for comparing architecture approaches. " * 18)[:700],
                question_ids=["q2" if i == 50 else "q1"],
            )
        state.searches.append(SearchRecord(
            query="architecture dimension two evidence",
            num=1,
            question_id="q2",
            purpose="Resolve the active question.",
            source_ids=["S50"],
        ))

        context = self.runner._context(state, state.questions["q2"])

        available_urls = {source["url"] for source in context["source_catalog"]}
        self.assertIn("https://example.com/report/50", available_urls)
        self.assertLessEqual(len(json.dumps(context)), state.limits.max_context_chars)

    def test_context_bound_reserves_room_for_cited_passages(self):
        state = inquiry(plan=frame(8))
        text = "Reuse benefits depend on whether the capability recurs across use cases."
        state.sources["S1"] = SourceRecord(
            id="S1", url="https://example.com/evidence", title="Reuse evidence",
            status="read", question_ids=["q1"], passages=[Passage(id="S1:P1", text=text)],
        )
        verbose_claim = "A qualified architecture finding with relevant details. " * 24
        for record in state.questions.values():
            record.findings.append(finding([
                Claim(
                    text=f"Finding {i}: {verbose_claim}",
                    basis="source",
                    evidence=[EvidenceReference(passage_id="S1:P1", quote=text)],
                )
                for i in range(10)
            ]))

        context = self.runner._context(state, state.questions["q1"])

        self.assertLessEqual(len(json.dumps(context)), state.limits.max_context_chars)
        passages = {p["id"]: p["text"] for p in context["evidence"]}
        self.assertEqual(passages.get("S1:P1"), text)

    def test_clarification_answer_registers_user_supplied_source_url(self):
        state = inquiry(
            phase="work", status="awaiting_user",
            limits=InquiryLimits(max_steps=1), usage=Usage(steps=1),
            clarifications=[UserClarification(
                id="C1", question="Which report should inform the comparison?",
                reason="The report defines the user's relevant evidence.", question_id="q1",
            )],
        )
        self.store.save(state)

        resumed = self.runner.resume(answers={
            "C1": "Use the report at https://example.com/new-report for this comparison.",
        })

        source = self.runner._find_source(resumed, "https://example.com/new-report")
        self.assertIsNotNone(source)
        self.assertEqual(source.status, "discovered")
        self.assertIn("q1", source.question_ids)
        self.assertEqual(resumed.clarifications[0].answer,
                         "Use the report at https://example.com/new-report for this comparison.")

    def test_fresh_source_read_is_associated_with_the_reading_question(self):
        state = inquiry(phase="work", status="running")
        source = self.runner._source(
            state, "https://example.com/user-report", "User report", "", "user",
        )
        tools = ToolRegistry()
        tools.register("read_source", lambda args: {
            "url": args["url"], "title": "User report", "text": "The report contains relevant evidence.",
            "content_type": "text/plain", "retrieved_at": "2026-11-07T12:00:00+00:00",
            "truncated": False,
        })
        runner = InquiryRunner(NoModelCalls(), tools, self.store)

        runner._retrieve(state, state.questions["q1"], ReadCall(
            tool="read_source", arguments=ReadArguments(url=source.url),
            purpose="Read the user-supplied evidence for the active question.",
        ))

        self.assertEqual(source.status, "read")
        self.assertIn("q1", source.question_ids)
        self.assertTrue(source.passages)

    def test_global_clarification_requires_frame_regardless_of_answer_order(self):
        for keys in (("C1", "C2"), ("C2", "C1")):
            with self.subTest(answer_order=keys):
                state = inquiry(
                    phase="work", status="awaiting_user",
                    limits=InquiryLimits(max_steps=1), usage=Usage(steps=1),
                    clarifications=[
                        UserClarification(
                            id="C1", question="Which decision must this study support?",
                            reason="Changes the inquiry's purpose.", question_id=None,
                        ),
                        UserClarification(
                            id="C2", question="Which team owns the capability?",
                            reason="Changes a branch's operating assumptions.", question_id="q1",
                        ),
                    ],
                )
                for record in state.questions.values():
                    record.status = "answered"
                    record.phase = "done"
                    record.findings.append(finding())
                self.store.save(state)
                values = {"C1": "Prioritize delivery risk.", "C2": "A shared platform team."}

                resumed = self.runner.resume(answers={key: values[key] for key in keys})

                self.assertEqual(resumed.phase, "frame")
                self.assertTrue(all(not q.findings for q in resumed.questions.values()))

    def test_changed_answer_criteria_invalidate_prior_accepted_findings(self):
        state = inquiry(phase="frame", status="running")
        for record in state.questions.values():
            record.status = "answered"
            record.phase = "done"
            record.findings.append(finding())
        revised = frame(answer_criteria=[
            "Explain benefits and coordination costs.",
            "Evaluate the consequences of failure and alternatives to shared ownership.",
        ])
        runner = InquiryRunner(OneFrame(revised), ToolRegistry(), self.store)

        runner._frame(state)

        self.assertEqual(len(state.frames), 2)
        self.assertEqual(state.frame.answer_criteria, revised.answer_criteria)
        self.assertTrue(all(q.status == "pending" for q in state.questions.values()))
        self.assertTrue(all(not q.findings for q in state.questions.values()))
        self.assertEqual(len(state.archived_questions), 2)


if __name__ == "__main__":
    unittest.main()
