"""Frame contracts: useful plans, valid dependencies, and runnable prompts."""

from copy import deepcopy
import json
import unittest

from pydantic import ValidationError

from cognitive_core.primitives.registry import (
    get_schema_class,
    list_primitives,
    render_prompt,
)
from cognitive_core.primitives.schemas import (
    BaseOutput,
    FrameClarification,
    FrameOutput,
    FrameQuestion,
    get_schema,
)


def frame_payload():
    return {
        "working_question": "When does a reusable agent help delivery?",
        "purpose": "Choose which capabilities to make reusable.",
        "subquestions": [
            {
                "id": "q1",
                "question": "Which capabilities recur across the use cases?",
                "contribution": "Establish the opportunity for reuse.",
                "evidence_needs": ["The use cases and their capability needs."],
            },
            {
                "id": "q2",
                "question": "When is the cost of coordination justified?",
                "contribution": "Determine the conditions for a useful shared boundary.",
                "depends_on": ["q1"],
            },
        ],
        "answer_criteria": ["Explain the benefits, costs, and boundary conditions."],
        "confidence": 0.8,
        "reasoning": "Separate the opportunity for reuse from its coordination costs.",
    }


class TestFrameContract(unittest.TestCase):
    def test_valid_plan_preserves_dependency_and_evidence_needs(self):
        frame = FrameOutput.model_validate(frame_payload())
        self.assertIsInstance(frame, BaseOutput)
        self.assertEqual(frame.subquestions[1].depends_on, ["q1"])
        self.assertEqual(
            frame.subquestions[0].evidence_needs,
            ["The use cases and their capability needs."],
        )
        self.assertEqual(frame.clarifications, [])

    def test_forward_reference_is_valid_without_ordering_the_schedule(self):
        payload = frame_payload()
        payload["subquestions"].reverse()
        frame = FrameOutput.model_validate(payload)
        self.assertEqual([item.id for item in frame.subquestions], ["q2", "q1"])

    def test_whitespace_is_normalized(self):
        payload = frame_payload()
        payload["working_question"] = "  A meaningful question? \n"
        payload["subquestions"][0]["question"] = "  A useful subquestion?  "
        payload["scope"] = ["  Publicly available evidence.  "]
        frame = FrameOutput.model_validate(payload)
        self.assertEqual(frame.working_question, "A meaningful question?")
        self.assertEqual(frame.subquestions[0].question, "A useful subquestion?")
        self.assertEqual(frame.scope, ["Publicly available evidence."])

    def test_blank_plan_text_is_rejected(self):
        for field in ("working_question", "purpose", "reasoning"):
            with self.subTest(field=field):
                payload = frame_payload()
                payload[field] = " \n\t "
                with self.assertRaises(ValidationError):
                    FrameOutput.model_validate(payload)

    def test_blank_list_entries_are_rejected(self):
        for field in ("answer_criteria", "scope", "assumptions"):
            with self.subTest(field=field):
                payload = frame_payload()
                payload[field] = [" "]
                with self.assertRaises(ValidationError):
                    FrameOutput.model_validate(payload)

    def test_empty_answer_criteria_are_rejected(self):
        payload = frame_payload()
        payload["answer_criteria"] = []
        with self.assertRaises(ValidationError):
            FrameOutput.model_validate(payload)

    def test_plan_requires_one_to_eight_questions(self):
        for count in (0, 9):
            with self.subTest(count=count):
                payload = frame_payload()
                template = payload["subquestions"][0]
                payload["subquestions"] = [
                    {**deepcopy(template), "id": f"q{i}"} for i in range(count)
                ]
                with self.assertRaises(ValidationError):
                    FrameOutput.model_validate(payload)

    def test_simple_question_can_remain_one_question(self):
        payload = frame_payload()
        payload["subquestions"] = payload["subquestions"][:1]
        self.assertEqual(len(FrameOutput.model_validate(payload).subquestions), 1)

    def test_duplicate_question_ids_are_rejected(self):
        payload = frame_payload()
        payload["subquestions"][1]["id"] = "q1"
        payload["subquestions"][1]["depends_on"] = []
        with self.assertRaisesRegex(ValidationError, "unique"):
            FrameOutput.model_validate(payload)

    def test_unknown_dependencies_are_rejected(self):
        payload = frame_payload()
        payload["subquestions"][1]["depends_on"] = ["not_in_plan"]
        with self.assertRaisesRegex(ValidationError, "unknown dependencies"):
            FrameOutput.model_validate(payload)

    def test_dependency_cycle_is_rejected(self):
        payload = frame_payload()
        payload["subquestions"][0]["depends_on"] = ["q2"]
        with self.assertRaisesRegex(ValidationError, "acyclic"):
            FrameOutput.model_validate(payload)

    def test_cycle_in_disconnected_component_is_rejected(self):
        payload = frame_payload()
        payload["subquestions"].extend([
            {"id": "q3", "question": "Third?", "contribution": "Third.", "depends_on": ["q4"]},
            {"id": "q4", "question": "Fourth?", "contribution": "Fourth.", "depends_on": ["q3"]},
        ])
        with self.assertRaisesRegex(ValidationError, "acyclic"):
            FrameOutput.model_validate(payload)

    def test_duplicate_clarification_ids_are_rejected(self):
        payload = frame_payload()
        clarification = {
            "id": "c1", "question": "Which use cases?", "reason": "They define the comparison.",
        }
        payload["clarifications"] = [clarification, deepcopy(clarification)]
        with self.assertRaisesRegex(ValidationError, "Clarification IDs must be unique"):
            FrameOutput.model_validate(payload)

    def test_unknown_fields_do_not_disappear_silently(self):
        payload = frame_payload()
        payload["assumption"] = ["Misspelled field should not disappear."]
        with self.assertRaises(ValidationError):
            FrameOutput.model_validate(payload)

    def test_save_load_roundtrip_preserves_plan(self):
        frame = FrameOutput.model_validate(frame_payload())
        restored = FrameOutput.model_validate_json(frame.model_dump_json())
        self.assertEqual(frame, restored)

    def test_mutable_defaults_are_not_shared(self):
        first = FrameOutput.model_validate(frame_payload())
        second = FrameOutput.model_validate(frame_payload())
        first.assumptions.append("Specific to first frame.")
        first.subquestions[0].depends_on.append("changed_after_validation")
        self.assertEqual(second.assumptions, [])
        self.assertEqual(second.subquestions[0].depends_on, [])


class TestFrameQuestion(unittest.TestCase):
    def test_id_must_be_a_simple_identifier(self):
        for identifier in ("", "q 1", "1q", "../q", "q.1"):
            with self.subTest(identifier=identifier):
                with self.assertRaises(ValidationError):
                    FrameQuestion(id=identifier, question="What?", contribution="Useful.")

    def test_question_and_contribution_must_not_be_blank(self):
        for field in ("question", "contribution"):
            with self.subTest(field=field):
                value = {"id": "q1", "question": "What?", "contribution": "Useful."}
                value[field] = " "
                with self.assertRaises(ValidationError):
                    FrameQuestion.model_validate(value)

    def test_self_dependency_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "itself"):
            FrameQuestion(id="q1", question="What?", contribution="Useful.", depends_on=["q1"])

    def test_duplicate_dependencies_are_rejected(self):
        with self.assertRaisesRegex(ValidationError, "duplicate dependencies"):
            FrameQuestion(id="q2", question="What?", contribution="Useful.", depends_on=["q1", "q1"])

    def test_blank_evidence_need_is_rejected(self):
        with self.assertRaises(ValidationError):
            FrameQuestion(id="q1", question="What?", contribution="Useful.", evidence_needs=[" "])


class TestFrameClarification(unittest.TestCase):
    def test_blocking_and_optional_clarifications_are_supported(self):
        value = {"id": "c1", "question": "Which use cases?", "reason": "Defines scope."}
        self.assertTrue(FrameClarification.model_validate(value).blocking)
        self.assertFalse(FrameClarification.model_validate({**value, "blocking": False}).blocking)

    def test_clarification_requires_meaningful_question_and_reason(self):
        for field in ("question", "reason"):
            with self.subTest(field=field):
                value = {"id": "c1", "question": "Which use cases?", "reason": "Defines scope."}
                value[field] = " "
                with self.assertRaises(ValidationError):
                    FrameClarification.model_validate(value)


class TestFrameRegistryAndPrompt(unittest.TestCase):
    def test_frame_is_available_through_both_registry_apis(self):
        self.assertIn("frame", list_primitives())
        self.assertIs(get_schema_class("frame"), FrameOutput)
        self.assertIs(get_schema("FRAME"), FrameOutput)

    def test_missing_question_is_an_actionable_configuration_error(self):
        with self.assertRaisesRegex(ValueError, "question"):
            render_prompt("frame", {})

    def test_question_alone_renders_with_defaults(self):
        prompt = render_prompt("frame", {"question": "When should a capability be reusable?"})
        self.assertIn("When should a capability be reusable?", prompt)
        self.assertIn("No additional context provided.", prompt)
        self.assertNotIn("{question}", prompt)

    def test_explicit_context_and_instructions_are_preserved(self):
        context = '{"clarification": "compare two architectures"}'
        prompt = render_prompt("frame", {
            "question": "Which architecture?",
            "purpose": "Select an implementation.",
            "context": context,
            "additional_instructions": "Keep the inquiry to two subquestions.",
        })
        self.assertIn(context, prompt)
        self.assertIn("Select an implementation.", prompt)
        self.assertIn("Keep the inquiry to two subquestions.", prompt)

    def test_rendered_prompt_example_is_valid_contract_json(self):
        prompt = render_prompt("frame", {"question": "What should we investigate?"})
        start = prompt.index('{\n  "working_question"')
        payload, _ = json.JSONDecoder().raw_decode(prompt[start:])
        frame = FrameOutput.model_validate(payload)
        self.assertEqual(frame.subquestions[0].id, "q1")


if __name__ == "__main__":
    unittest.main()
