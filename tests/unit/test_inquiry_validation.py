"""Useful repair diagnostics must never echo model-controlled error content."""

import json
from typing import Annotated, Union
import unittest

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from cognitive_core.inquiry.models import RetrievalPlan
from cognitive_core.inquiry.validation import validation_issues


class InquiryValidationTests(unittest.TestCase):
    def issues(self, value, model=RetrievalPlan):
        with self.assertRaises(ValidationError) as caught:
            model.model_validate(value)
        return validation_issues(caught.exception, model)

    def plan(self, arguments=None, **overrides):
        call = {"tool": "read_source", "arguments": arguments or {}, "purpose": "Inspect a source."}
        call.update(overrides)
        return {"calls": [call], "rationale": "Find evidence."}

    def test_missing_nested_read_argument_uses_schema_path(self):
        issues = self.issues(self.plan())
        self.assertEqual(issues, [{"path": "calls[0].arguments.url", "code": "missing",
                                   "message": "This field is required."}])

    def test_five_calls_reports_trusted_four_item_bound(self):
        plan = self.plan({"url": "https://example.org"})
        plan["calls"] *= 5
        issues = self.issues(plan)
        self.assertEqual(issues[0]["path"], "calls")
        self.assertEqual(issues[0]["code"], "too_long")
        self.assertIn("at most 4", issues[0]["message"])
        self.assertNotIn("not 5", issues[0]["message"])

    def test_nested_search_bound_comes_from_selected_union_branch(self):
        plan = self.plan({"query": "comparison", "num": 50}, tool="web_search")
        issues = self.issues(plan)
        self.assertEqual(issues, [{"path": "calls[0].arguments.num", "code": "less_than_equal",
                                   "message": "Use a number less than or equal to 10."}])

    def test_bound_is_taken_from_schema_even_when_error_context_claims_another(self):
        class ConflictingError(BaseModel):
            calls: list[int] = Field(max_length=4)

            @field_validator("calls")
            @classmethod
            def invalid(cls, value):
                raise PydanticCustomError("too_long", "{secret}: maximum {max_length}",
                                          {"secret": "private-context-sentinel", "max_length": 999})

        issues = self.issues({"calls": [1]}, ConflictingError)
        self.assertEqual(issues, [{"path": "calls", "code": "too_long",
                                   "message": "Must contain at most 4 items."}])

    def test_unknown_union_tag_is_omitted_but_allowed_tools_are_given(self):
        issues = self.issues(self.plan({"url": "https://example.org"}, tool="secret-tag-sentinel"))
        self.assertEqual(issues[0]["path"], "calls[0].tool")
        self.assertEqual(issues[0]["code"], "union_tag_invalid")
        rendered = json.dumps(issues)
        self.assertNotIn("secret-tag-sentinel", rendered)
        self.assertIn("web_search", rendered)
        self.assertIn("read_source", rendered)

    def test_missing_discriminator_identifies_tool(self):
        plan = self.plan({"url": "https://example.org"})
        del plan["calls"][0]["tool"]
        issues = self.issues(plan)
        self.assertEqual(issues[0]["path"], "calls[0].tool")
        self.assertEqual(issues[0]["code"], "union_tag_not_found")

    def test_extra_field_names_and_raw_values_are_never_echoed(self):
        plan = self.plan({"url": 42, "nested-secret-key-sentinel": "private-value-sentinel"})
        plan["root-secret-key-sentinel"] = "another-value-sentinel"
        issues = self.issues(plan)
        self.assertIn("calls[0].arguments.url", [issue["path"] for issue in issues])
        self.assertIn("calls[0].arguments", [issue["path"] for issue in issues])
        self.assertIn("$", [issue["path"] for issue in issues])
        self.assertNotIn("sentinel", json.dumps(issues))

    def test_custom_validator_messages_contexts_and_codes_are_not_reused(self):
        class SensitiveValidation(BaseModel):
            text: str

            @field_validator("text")
            @classmethod
            def invalid(cls, value):
                raise PydanticCustomError("secret-code-sentinel", "Do not echo {secret}", {"secret": value})

        issues = self.issues({"text": "private-value-sentinel"}, SensitiveValidation)
        self.assertEqual(issues, [{"path": "text", "code": "invalid",
                                   "message": "The value does not satisfy the response schema."}])

    def test_optional_nested_model_and_dynamic_mapping_keys_are_safe(self):
        class Entry(BaseModel):
            count: int = Field(ge=1)

        class Container(BaseModel):
            maybe: Entry | None = None
            entries: dict[str, Entry]

        issues = self.issues({"maybe": {"count": 0}, "entries": {"secret-key-sentinel": {"count": "raw-value-sentinel"}}}, Container)
        self.assertEqual([issue["path"] for issue in issues], ["maybe.count", "entries[*].count"])
        self.assertNotIn("sentinel", json.dumps(issues))

    def test_non_discriminated_model_union_uses_field_paths(self):
        class Left(BaseModel):
            left: int

        class Right(BaseModel):
            right: str

        class Container(BaseModel):
            value: Union[Left, Right]

        issues = self.issues({"value": {}}, Container)
        self.assertEqual({issue["path"] for issue in issues}, {"value.left", "value.right"})

    def test_feedback_count_is_bounded_and_repeated_unknown_fields_are_deduplicated(self):
        class ManyFields(BaseModel):
            model_config = ConfigDict(extra="forbid")
            values: list[Annotated[int, Field(ge=0)]]

        issues = self.issues({"values": [-1] * 100}, ManyFields)
        self.assertEqual(len(issues), 8)
        self.assertTrue(all(len(issue["message"]) <= 300 and len(issue["path"]) <= 223 for issue in issues))
        self.assertTrue(all(isinstance(value, str) for issue in issues for value in issue.values()))
        issues = self.issues({"values": [], **{f"secret-key-sentinel-{i}": "secret" for i in range(30)}}, ManyFields)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["path"], "$")
        self.assertNotIn("sentinel", json.dumps(issues))


if __name__ == "__main__":
    unittest.main()
