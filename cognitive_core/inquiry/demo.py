"""An explicitly synthetic, offline demonstration of the complete inquiry loop."""

from __future__ import annotations

from cognitive_core.engine.tools import ToolRegistry
from cognitive_core.inquiry.runner import InquiryRunner
from cognitive_core.inquiry.store import InquiryStore


DEMO_QUESTION = "Should every step of a business workflow be implemented as an autonomous agent?"
DEMO_URL = "https://example.org/synthetic-workflow-study"
DEMO_TEXT = (
    "SYNTHETIC DEMO FIXTURE — this is not an empirical study. "
    "In this fictional workflow, checking a supplied account ID follows an exact rule. "
    "Resolving an ambiguous request requires interpreting incomplete context. "
    "The example does not measure accuracy, latency, cost, or business outcomes."
)


def _base(**fields):
    return {"confidence": 0.65, "reasoning": "Brief rationale for an offline test fixture.", **fields}


def demo_frame():
    return _base(
        working_question=DEMO_QUESTION,
        purpose="Identify where autonomous reasoning adds value and where a fixed operation suffices.",
        subquestions=[{
            "id": "q1", "question": "Which workflow steps benefit from autonomous reasoning?",
            "contribution": "Distinguish predictable operations from tasks requiring judgment.",
            "evidence_needs": ["Examples of fixed-rule and ambiguous tasks"],
        }],
        answer_criteria=["Separate conceptual reasoning from empirical performance claims"],
        assumptions=["The demonstration uses a fictional workflow."],
    )


def demo_claim():
    return {
        "text": "The fictional example contains both a fixed-rule check and an ambiguous request.",
        "basis": "source",
        "evidence": [{"passage_id": "S1:P1", "quote": "checking a supplied account ID follows an exact rule"}],
        "qualification": "Synthetic example; it establishes no real-world performance advantage.",
    }


def demo_finding(revised=False):
    return _base(
        finding=(
            "Use autonomous reasoning where interpreting incomplete context changes the next step. "
            "Keep exact checks explicit. This is a conceptual allocation rule; the fixture cannot establish performance."
            if revised else
            "Use agents for ambiguous requests and fixed operations for exact checks; this should improve efficiency."
        ),
        claims=[demo_claim()],
        assumptions=["The task distinction is meaningful for the user's workflow."],
        unresolved_issues=["No empirical evidence about accuracy, latency, cost, or outcomes."],
    )


def demo_challenge(survives=True):
    return _base(
        survives=survives,
        vulnerabilities=[] if survives else [{
            "description": "The efficiency claim is unsupported by the fictional source.",
            "severity": "high", "attack_vector": "The fixture supplies no performance measurement.",
            "recommendation": "Remove the performance claim or qualify it as an untested hypothesis.",
        }],
        overall_assessment="The qualified argument is coherent." if survives else "Remove the empirical overclaim.",
    )


def demo_reflection(action="accept"):
    return _base(
        what_changed="An empirical overclaim was identified and must be removed." if action == "revise" else "The finding now preserves its evidential limits.",
        trajectory="revise" if action == "revise" else "continue",
        next_action=action,
        resolution="conditional",
        next_question="What follows conceptually without asserting measured benefits?" if action == "revise" else None,
        template_guidance="Keep the synthetic example explicitly labeled.",
    )


def demo_synthesis():
    return _base(
        situation_summary="The synthetic example illustrates two kinds of work.",
        conclusion="Choose autonomy where interpretation and adaptive selection of steps are useful.",
        claims=[demo_claim()],
        limitations=["This is an offline synthetic demonstration, not evidence of actual performance."],
        open_questions=["How do candidate designs compare on real workflow cases?"],
    )


def demo_script():
    return [
        ("frame", demo_frame()),
        ("retrieve", {"calls": [{"tool": "web_search", "arguments": {"query": "synthetic workflow example"}, "purpose": "Discover the demo source"}], "rationale": "Find the explicitly synthetic example."}),
        ("retrieve", {"calls": [{"tool": "read_source", "arguments": {"url": DEMO_URL}, "purpose": "Read the example before citing it"}], "rationale": "Search snippets alone are not evidence."}),
        ("retrieve", {"calls": [], "rationale": "The source is read; proceed with a qualified conceptual argument."}),
        ("investigate", demo_finding()),
        ("challenge", demo_challenge(False)),
        ("reflect", demo_reflection("revise")),
        ("investigate", demo_finding(True)),
        ("challenge", demo_challenge()),
        ("reflect", demo_reflection()),
        ("deliberate", demo_synthesis()),
        ("challenge", demo_challenge()),
        ("reflect", demo_reflection()),
        ("generate", _base(
            artifact=(
                "**Offline synthetic demonstration.**\n\n"
                "Autonomous reasoning is useful when interpreting incomplete context changes what happens next. "
                "An exact rule can remain an explicit operation. The fictional source illustrates this distinction. [S1:P1]\n\n"
                "This is a design argument. The example supplies no measured accuracy, speed, cost, or business benefit. "
                "The initial efficiency claim was removed after challenge."
            ), format="markdown",
        )),
    ]


class ScriptedReasoner:
    """Deterministic test double; it does not perform real model reasoning."""

    def __init__(self, script):
        self.script = list(script)
        self.contexts = []

    def invoke(self, primitive, *, context, instructions, response_model):
        self.contexts.append((primitive, context, instructions))
        if not self.script:
            raise AssertionError("The offline script was exhausted")
        expected, data = self.script.pop(0)
        if primitive != expected:
            raise AssertionError(f"Expected {expected}, got {primitive}")
        return response_model.model_validate(data)


def demo_tools():
    registry = ToolRegistry()
    registry.register("web_search", lambda args: {"results": [{
        "url": DEMO_URL, "title": "Synthetic workflow example", "snippet": "A fictional workflow example.",
    }]})
    registry.register("read_source", lambda args: {
        "url": DEMO_URL, "title": "Synthetic workflow example", "text": DEMO_TEXT,
        "content_type": "text/plain", "truncated": False,
    })
    return registry


def run_demo(directory, on_event=None):
    runner = InquiryRunner(ScriptedReasoner(demo_script()), demo_tools(), InquiryStore(directory), on_event)
    return runner.start(DEMO_QUESTION, purpose="Exercise the complete loop without network access or API keys.",
                        context="All sources and responses in this demonstration are synthetic fixtures.")
