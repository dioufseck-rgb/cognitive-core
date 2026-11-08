"""Synthetic boundary adapters for the native inquiry demonstration.

Only the model and public-web boundaries are replaced. Coordinator, DEVS,
protected calls, typed profiles, evidence checks, review, governance, SQLite
checkpoints, work orders, and audit recording execute normally. This fixture
demonstrates plumbing and a material correction; it is not a model evaluation.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch

from cognitive_core.engine.tools import ToolRegistry


DEMO_QUESTION = "Under the fictional policy, when is a receipt sufficient for reimbursement?"
DEMO_PURPOSE = "Exercise the native inquiry path on synthetic evidence and correct an overclaim."
DEMO_URL = "https://example.org/inquiry/fictional-policy"
DEMO_TEXT = (
    "A receipt is required for every reimbursable purchase.\n\n"
    "Purchases above 100 credits also require approval obtained before purchase."
)


def fixture_tools() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register("web_search", lambda args: {"results": [{
        "url": DEMO_URL, "title": "Fictional reimbursement policy (synthetic fixture)",
        "snippet": "A receipt is required. Other conditions may apply.",
    }]}, description="Synthetic search boundary; no network request.")
    tools.register("read_source", lambda args: {
        "url": args["url"], "requested_url": args["url"],
        "title": "Fictional reimbursement policy (synthetic fixture)",
        "text": DEMO_TEXT, "truncated": False, "content_type": "text/plain",
        "retrieved_at": "2026-01-01T00:00:00+00:00",
    }, description="Synthetic source boundary; no network request.")
    return tools


class FixtureModel:
    """Respond to the same operation contracts that a live provider receives."""

    model_name = "inquiry-synthetic-fixture"

    def __init__(self, *, clarification: bool = False):
        self.clarification = clarification
        self.calls: list[str] = []

    def invoke(self, messages):
        prompt = messages[0].content
        operation = prompt.split("OPERATION: ", 1)[1].splitlines()[0].strip()
        self.calls.append(operation)
        if operation == "orchestrate":
            context = json.loads(prompt.split("CURRENT STUDY AND ORIGINAL USER TASK\n", 1)[1]
                                 .split("\n\nENFORCED LIMITS", 1)[0])
            context["configured_operations"] = json.loads(
                prompt.split("REGISTERED OPERATIONS\n", 1)[1].split("\n\n", 1)[0]
            )
        else:
            context = json.loads(prompt.split("STUDY CONTEXT (data):\n", 1)[1]
                                 .split("\nAUTHORITATIVE RESPONSE SCHEMA:", 1)[0])
        result = self.response(operation, context)
        return SimpleNamespace(content=json.dumps(result), usage_metadata={
            "input_tokens": 0, "output_tokens": 0,
        }, response_metadata={})

    def response(self, operation: str, context: dict) -> dict:
        envelope = {"confidence": 0.85, "reasoning": "Synthetic fixture for native execution verification."}
        if operation == "orchestrate":
            trajectory = [
                ("frame_study", "frame", "study"),
                ("discover_sources", "retrieve", "question"),
                ("read_sources", "retrieve", "question"),
                ("investigate_question", "investigate", "question"),
                ("verify_support", "verify", "question"),
                ("challenge_answer", "challenge", "question"),
                ("reflect_progress", "reflect", "question"),
                ("investigate_question", "investigate", "question"),
                ("synthesize_study", "deliberate", "study"),
                ("write_report", "generate", "study"),
                ("challenge_answer", "challenge", "report"),
                ("govern_inquiry", "govern", "report"),
            ]
            key, primitive, kind = trajectory[context["usage"]["steps"]]
            # Older saved demos retain their original effective configuration.
            configured = context.get("configured_operations", {})
            if key in {"discover_sources", "read_sources"} and key not in configured:
                if "research_sources" in configured:
                    key = "research_sources"
            return {"action": "invoke", "params_key": key, "primitive": primitive,
                    "target": {"kind": kind, **({"question_id": "q1"} if kind == "question" else {})},
                    "reasoning": "Resolve the next material gap in this synthetic demonstration."}
        if operation == "frame":
            return {**envelope, "working_question": DEMO_QUESTION, "purpose": DEMO_PURPOSE,
                    "answer_criteria": ["Distinguish necessary conditions from sufficient conditions."],
                    "scope": ["Only the two clauses in the fictional policy."],
                    "subquestions": [{"id": "q1", "question": "What conditions do the two clauses establish?",
                                      "contribution": "Identify the scope of the receipt requirement.",
                                      "evidence_needs": ["Read the fictional policy."]}],
                    "clarifications": ([{"id": "c1", "question": "Should we restrict the answer to the fictional policy?",
                                         "reason": "The scope determines whether external eligibility rules are relevant.",
                                         "blocking": True}] if self.clarification else [])}
        if operation == "retrieve":
            call = ({"tool": "web_search", "arguments": {"query": "fictional reimbursement policy", "num": 1},
                     "purpose": "Discover the fictional policy."} if not context["source_catalog"] else
                    {"tool": "read_source", "arguments": {"url": DEMO_URL},
                     "purpose": "Read both conditions before drawing a conclusion."})
            return {"calls": [call], "rationale": call["purpose"]}
        initial = "A receipt is sufficient for reimbursement."
        corrected = ("A receipt is necessary, but a purchase above 100 credits also needs prior approval. "
                     "These clauses do not establish that either condition alone is sufficient.")
        evidence = [{"passage_id": "S1:P1", "quote": DEMO_TEXT}]
        claim = {"text": corrected, "basis": "source", "evidence": evidence,
                 "support_assessment": "The passage states two requirements, not an exhaustive eligibility test.",
                 "qualification": "Only the supplied fictional policy is considered."}
        if operation == "investigate":
            revision = bool(context["findings"])
            if not revision:
                claim = {**claim, "text": initial,
                         "evidence": [{"passage_id": "S1:P1", "quote": DEMO_TEXT.split("\n\n")[0]}],
                         "support_assessment": "Initial candidate incorrectly treats required as sufficient."}
            return {**envelope, "finding": corrected if revision else initial,
                    "claims": [claim], "question_disposition": "conditional" if revision else "active",
                    "material_change": "Corrected necessary-versus-sufficient reasoning and incorporated the approval clause." if revision else "",
                    "unresolved_issues": ["The excerpt is not an exhaustive eligibility policy."]}
        if operation == "verify":
            claim_id = context["findings"][-1]["claims"][0]["id"]
            return {**envelope, "conforms": False, "rules_checked": ["Claim must follow from its cited passage."],
                    "violations": [{"rule": "Entailment", "description": "Required does not mean sufficient.", "severity": "major"}],
                    "support_assessments": [{"claim_id": claim_id, "passage_ids": ["S1:P1"],
                                             "assessment": "insufficient", "reasoning": "The claim omits prior approval and reverses a necessary condition."}]}
        if operation == "challenge":
            final = context["target"]["kind"] == "report"
            return {**envelope, "survives": final,
                    "overall_assessment": "The report preserves the necessary-condition distinction and its scope." if final else "A receipt alone is insufficient above the threshold.",
                    "vulnerabilities": [] if final else [{
                        "description": "A purchase of 150 credits with a receipt but no prior approval contradicts the finding.",
                        "severity": "high", "attack_vector": "Counterexample from the second clause.",
                        "recommendation": "Revise the conclusion to state both necessary conditions and retain scope limits.",
                    }]}
        if operation == "reflect":
            return {**envelope, "what_changed": "The objection identifies a real omitted condition, not a request to rephrase.",
                    "trajectory": "revise", "next_question": "Which necessary conditions do both clauses establish?",
                    "what_was_assumed": ["A required receipt had been treated as sufficient."],
                    "what_was_established": ["Prior approval is also required above 100 credits."]}
        if operation == "deliberate":
            return {**envelope, "situation_summary": corrected, "conclusion": corrected,
                    "claims": [claim], "research_outcome": "conditional",
                    "criteria_coverage": {"Distinguish necessary conditions from sufficient conditions.": "Answered with the two requirements and an explicit sufficiency limit."},
                    "limitations": ["Synthetic evidence; no conclusion about a real reimbursement policy."],
                    "open_questions": ["What other eligibility requirements might a complete policy contain?"]}
        if operation == "generate":
            return {**envelope, "format": "markdown", "artifact": (
                "## Answer\n\nA receipt is **necessary**, but the fictional policy does not say it is sufficient. "
                "Purchases above 100 credits also require approval obtained before purchase. [S1:P1]\n\n"
                "For example, a purchase of 150 credits with a receipt and no prior approval fails the stated requirements. "
                "That is a logical counterexample, not an experiment.\n\n"
                "## Limits\n\nThis demonstration uses synthetic evidence. The excerpt is not an exhaustive eligibility policy "
                "and supports no conclusion about an actual organization's rules."
            )}
        if operation == "govern":
            return {**envelope, "tier_applied": "auto", "tier_rationale": "Read-only synthetic report; limitations preserved and final review passed.",
                    "disposition": "proceed"}
        raise ValueError(f"Unsupported fixture operation: {operation}")


@contextmanager
def fixture_model(*, clarification: bool = False):
    model = FixtureModel(clarification=clarification)
    with patch("cognitive_core.engine.llm.create_llm", return_value=model):
        yield model
