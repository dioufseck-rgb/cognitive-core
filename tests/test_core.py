"""
Cognitive Core — Core Engine Test Suite

Tests for the engine components that DON'T require LangGraph or LLM calls:
  1. State management (resolve_param, step outputs, context building)
  2. Schemas (Pydantic validation, schema registry, JSON spec generation)
  3. Primitive registry (listing, prompt templates, config specs, step validation)
  4. Tool registry (register, call, call_many, case registry, describe)
  5. Composer helpers (config validation, condition evaluation, reachable steps)
  6. Actions registry (register, execute, simulate, dry_run, rollback)

All tests are deterministic — no LLM, no LangGraph graph compilation.
"""

import json
import sys
import types
import os
import unittest
import importlib
import importlib.util
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ─── Stub missing deps so engine modules can be imported ───────────
# This test environment may not have langgraph or pydantic installed.
# We stub the minimum needed to import and test the non-LLM parts.

if "pydantic" not in sys.modules:
    # Minimal pydantic stub
    pydantic = types.ModuleType("pydantic")

    class _Field:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __repr__(self):
            return f"Field({self.kwargs})"

    class _BaseModelMeta(type):
        def __new__(mcs, name, bases, namespace):
            cls = super().__new__(mcs, name, bases, namespace)
            # Collect annotations for model_json_schema
            annotations = {}
            for base in reversed(cls.__mro__):
                annotations.update(getattr(base, '__annotations__', {}))
            cls.__all_annotations__ = annotations
            return cls

    class _BaseModel(metaclass=_BaseModelMeta):
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            # Apply defaults from annotations
            for key in getattr(self.__class__, '__all_annotations__', {}):
                if key not in kwargs:
                    default = getattr(self.__class__, key, None)
                    if isinstance(default, _Field):
                        if 'default' in default.kwargs:
                            setattr(self, key, default.kwargs['default'])
                        elif 'default_factory' in default.kwargs:
                            setattr(self, key, default.kwargs['default_factory']())
                    elif default is not None:
                        setattr(self, key, default)
            # Validate confidence bounds if present
            if hasattr(self, 'confidence'):
                conf = self.confidence
                ann = getattr(self.__class__, '__all_annotations__', {})
                if 'confidence' in ann and isinstance(conf, (int, float)):
                    if conf < 0 or conf > 1:
                        raise ValueError(f"confidence must be between 0 and 1, got {conf}")

        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {
                k: {"type": "string", "description": ""}
                for k in getattr(cls, '__all_annotations__', {})
            }}

    pydantic.BaseModel = _BaseModel
    pydantic.Field = lambda **kw: _Field(**kw)
    sys.modules["pydantic"] = pydantic

# Stub langchain_core as a proper package tree
def _stub_module_tree(paths_and_attrs):
    """Create module stubs with proper parent-child relationships."""
    for path, attrs in paths_and_attrs.items():
        if path in sys.modules:
            continue
        parts = path.split(".")
        # Ensure all parent modules exist
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                m = types.ModuleType(parent)
                m.__path__ = []  # Make it a package
                sys.modules[parent] = m
        m = types.ModuleType(path)
        if not path.endswith(parts[-1]) or len(parts) > 1:
            m.__path__ = []  # Make intermediate modules packages too
        for attr_name, attr_val in attrs.items():
            setattr(m, attr_name, attr_val)
        sys.modules[path] = m
        # Attach to parent
        if len(parts) > 1:
            parent_path = ".".join(parts[:-1])
            if parent_path in sys.modules:
                setattr(sys.modules[parent_path], parts[-1], m)

_DummyClass = lambda name: type(name, (), {"__init__": lambda self, *a, **kw: None})

class _FakeStateGraph:
    def __init__(self, *a, **kw): pass
    def add_node(self, *a, **kw): pass
    def add_edge(self, *a, **kw): pass
    def add_conditional_edges(self, *a, **kw): pass
    def set_entry_point(self, *a, **kw): pass
    def compile(self, *a, **kw): return self

_stub_module_tree({
    "langchain_core": {},
    "langchain_core.messages": {"HumanMessage": _DummyClass("HumanMessage")},
    "langchain_core.language_models": {},
    "langchain_core.language_models.chat_models": {"BaseChatModel": _DummyClass("BaseChatModel")},
    "langchain_anthropic": {"ChatAnthropic": _DummyClass("ChatAnthropic")},
    "langchain_openai": {
        "ChatOpenAI": _DummyClass("ChatOpenAI"),
        "AzureChatOpenAI": _DummyClass("AzureChatOpenAI"),
    },
    "langchain_google_genai": {"ChatGoogleGenerativeAI": _DummyClass("ChatGoogleGenerativeAI")},
    "langgraph": {},
    "langgraph.graph": {"StateGraph": _FakeStateGraph, "END": "__end__"},
})

from engine.state import (
    WorkflowState, StepResult,
    get_step_output, get_all_step_outputs, get_latest_output,
    get_latest_step, get_loop_count,
    resolve_param, build_context_from_state,
    _get_last_primitive_output,
)
from engine.tools import ToolRegistry, ToolResult, create_case_registry
from engine.actions import ActionRegistry
from engine.composer import (
    validate_use_case, _evaluate_condition, _find_reachable_steps,
)
from registry.schemas import (
    SCHEMA_REGISTRY, get_schema, schema_to_json_spec,
    ClassifyOutput, InvestigateOutput, VerifyOutput,
    GenerateOutput, ChallengeOutput, RetrieveOutput,
    ThinkOutput, ActOutput, BaseOutput,
)
from registry.primitives import (
    list_primitives, get_prompt_template, get_schema_class,
    get_config_spec, render_prompt, validate_use_case_step,
    PRIMITIVE_CONFIGS,
)


# ═══════════════════════════════════════════════════════════════════
# 1. STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

def _make_state(**overrides) -> dict:
    """Helper to build a WorkflowState dict."""
    base = {
        "input": {"member_id": "M001", "account_id": "A100"},
        "steps": [],
        "current_step": "",
        "metadata": {"use_case": "test"},
        "loop_counts": {},
        "routing_log": [],
    }
    base.update(overrides)
    return base


def _make_step(name, primitive, output, raw="", prompt=""):
    return {
        "step_name": name,
        "primitive": primitive,
        "output": output,
        "raw_response": raw,
        "prompt_used": prompt,
    }


class TestGetStepOutput(unittest.TestCase):
    def test_found(self):
        state = _make_state(steps=[
            _make_step("classify", "classify", {"category": "fraud", "confidence": 0.9}),
        ])
        out = get_step_output(state, "classify")
        self.assertEqual(out["category"], "fraud")

    def test_not_found(self):
        state = _make_state()
        self.assertIsNone(get_step_output(state, "missing"))

    def test_returns_latest_on_duplicate(self):
        state = _make_state(steps=[
            _make_step("investigate", "investigate", {"finding": "first"}),
            _make_step("investigate", "investigate", {"finding": "second"}),
        ])
        out = get_step_output(state, "investigate")
        self.assertEqual(out["finding"], "second")


class TestGetAllStepOutputs(unittest.TestCase):
    def test_multiple(self):
        state = _make_state(steps=[
            _make_step("investigate", "investigate", {"finding": "first"}),
            _make_step("classify", "classify", {"category": "fraud"}),
            _make_step("investigate", "investigate", {"finding": "second"}),
        ])
        outs = get_all_step_outputs(state, "investigate")
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0]["finding"], "first")
        self.assertEqual(outs[1]["finding"], "second")

    def test_empty(self):
        state = _make_state()
        self.assertEqual(get_all_step_outputs(state, "missing"), [])


class TestGetLatestOutput(unittest.TestCase):
    def test_has_steps(self):
        state = _make_state(steps=[
            _make_step("a", "classify", {"category": "A"}),
            _make_step("b", "generate", {"artifact": "letter"}),
        ])
        self.assertEqual(get_latest_output(state)["artifact"], "letter")

    def test_no_steps(self):
        self.assertIsNone(get_latest_output(_make_state()))


class TestGetLatestStep(unittest.TestCase):
    def test_returns_full_step(self):
        state = _make_state(steps=[
            _make_step("a", "classify", {"category": "A"}),
        ])
        step = get_latest_step(state)
        self.assertEqual(step["step_name"], "a")
        self.assertEqual(step["primitive"], "classify")


class TestGetLoopCount(unittest.TestCase):
    def test_found(self):
        state = _make_state(loop_counts={"investigate": 3})
        self.assertEqual(get_loop_count(state, "investigate"), 3)

    def test_missing_defaults_zero(self):
        state = _make_state()
        self.assertEqual(get_loop_count(state, "investigate"), 0)


class TestGetLastPrimitiveOutput(unittest.TestCase):
    def test_found(self):
        state = _make_state(steps=[
            _make_step("gen1", "generate", {"artifact": "draft"}),
            _make_step("challenge1", "challenge", {"survives": True}),
            _make_step("gen2", "generate", {"artifact": "final"}),
        ])
        self.assertEqual(
            _get_last_primitive_output(state, "generate")["artifact"], "final"
        )

    def test_not_found(self):
        state = _make_state()
        self.assertIsNone(_get_last_primitive_output(state, "classify"))


# ═══════════════════════════════════════════════════════════════════
# 1b. PARAMETER RESOLUTION
# ═══════════════════════════════════════════════════════════════════

class TestResolveParam(unittest.TestCase):
    def test_no_refs(self):
        self.assertEqual(resolve_param("plain text", _make_state()), "plain text")

    def test_input_ref(self):
        state = _make_state()
        self.assertEqual(resolve_param("${input.member_id}", state), "M001")

    def test_input_nested(self):
        state = _make_state(input={"member": {"name": "Alice", "id": 42}})
        self.assertEqual(resolve_param("${input.member.name}", state), "Alice")

    def test_step_ref(self):
        state = _make_state(steps=[
            _make_step("classify", "classify", {"category": "fraud", "confidence": 0.9}),
        ])
        self.assertEqual(resolve_param("${classify.category}", state), "fraud")

    def test_previous_ref(self):
        state = _make_state(steps=[
            _make_step("classify", "classify", {"category": "fraud"}),
        ])
        self.assertEqual(resolve_param("${previous.category}", state), "fraud")

    def test_previous_no_steps(self):
        val = resolve_param("${previous.field}", _make_state())
        self.assertIn("no previous", val)

    def test_step_not_found(self):
        val = resolve_param("${missing_step.field}", _make_state())
        self.assertIn("not found", val)

    def test_field_not_found(self):
        state = _make_state(steps=[
            _make_step("classify", "classify", {"category": "fraud"}),
        ])
        val = resolve_param("${classify.missing_field}", state)
        self.assertIn("not found", val)

    def test_loop_count(self):
        state = _make_state(loop_counts={"investigate": 2}, current_step="investigate")
        self.assertEqual(resolve_param("${_loop_count.investigate}", state), "2")

    def test_loop_count_current(self):
        state = _make_state(loop_counts={"investigate": 3}, current_step="investigate")
        self.assertEqual(resolve_param("${_loop_count}", state), "3")

    def test_last_primitive(self):
        state = _make_state(steps=[
            _make_step("gen1", "generate", {"artifact": "draft"}),
            _make_step("gen2", "generate", {"artifact": "final"}),
        ])
        self.assertEqual(resolve_param("${_last_generate.artifact}", state), "final")

    def test_last_primitive_not_found(self):
        val = resolve_param("${_last_verify.conforms}", _make_state())
        self.assertIn("no completed", val)

    def test_dict_output_as_json(self):
        state = _make_state(steps=[
            _make_step("retrieve", "retrieve", {"data": {"member": {"name": "Alice"}}}),
        ])
        val = resolve_param("${retrieve.data}", state)
        parsed = json.loads(val)
        self.assertEqual(parsed["member"]["name"], "Alice")

    def test_list_index(self):
        state = _make_state(steps=[
            _make_step("investigate", "investigate", {
                "hypotheses_tested": [
                    {"hypothesis": "H1", "status": "supported"},
                    {"hypothesis": "H2", "status": "rejected"},
                ]
            }),
        ])
        val = resolve_param("${investigate.hypotheses_tested.1.hypothesis}", state)
        self.assertEqual(val, "H2")

    def test_multiple_refs(self):
        state = _make_state()
        val = resolve_param("Member ${input.member_id} account ${input.account_id}", state)
        self.assertEqual(val, "Member M001 account A100")

    def test_delegation_results(self):
        state = _make_state()
        state["delegation_results"] = {"wo_001": {"decision": "approve"}}
        self.assertEqual(
            resolve_param("${_delegations.wo_001.decision}", state), "approve"
        )

    def test_delegation_results_missing(self):
        val = resolve_param("${_delegations.wo_001.field}", _make_state())
        self.assertIsNotNone(val)  # Should not crash


# ═══════════════════════════════════════════════════════════════════
# 1c. CONTEXT BUILDING
# ═══════════════════════════════════════════════════════════════════

class TestBuildContext(unittest.TestCase):
    def test_no_steps(self):
        ctx = build_context_from_state(_make_state())
        self.assertIn("No prior steps", ctx)

    def test_classify_step(self):
        state = _make_state(steps=[
            _make_step("classify", "classify", {"category": "fraud", "confidence": 0.9, "reasoning": "Pattern match"}),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Category: fraud", ctx)
        self.assertIn("Confidence: 0.9", ctx)

    def test_investigate_step(self):
        state = _make_state(steps=[
            _make_step("investigate", "investigate", {
                "finding": "Suspicious pattern",
                "confidence": 0.8,
                "reasoning": "Analysis",
                "hypotheses_tested": [{"hypothesis": "H1", "status": "supported"}],
            }),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Suspicious pattern", ctx)
        self.assertIn("H1", ctx)

    def test_generate_step(self):
        state = _make_state(steps=[
            _make_step("gen", "generate", {"artifact": "Dear member, ...", "confidence": 0.85}),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Dear member", ctx)

    def test_challenge_step(self):
        state = _make_state(steps=[
            _make_step("challenge", "challenge", {
                "survives": True,
                "overall_assessment": "Robust",
                "vulnerabilities": [{"severity": "low", "description": "Minor gap"}],
            }),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Survives: True", ctx)
        self.assertIn("Minor gap", ctx)

    def test_verify_step(self):
        state = _make_state(steps=[
            _make_step("verify", "verify", {
                "conforms": False,
                "reasoning": "Failed rule",
                "violations": [{"severity": "major", "description": "Missing signature"}],
            }),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Conforms: False", ctx)
        self.assertIn("Missing signature", ctx)

    def test_retrieve_step(self):
        state = _make_state(steps=[
            _make_step("retrieve", "retrieve", {"data": {"member": {"name": "Alice"}}}),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Sources retrieved", ctx)
        self.assertIn("Alice", ctx)

    def test_think_step(self):
        state = _make_state(steps=[
            _make_step("think", "think", {
                "thought": "Considering all factors...",
                "confidence": 0.7,
                "conclusions": ["A", "B"],
                "decision": "Proceed",
            }),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("Considering all factors", ctx)
        self.assertIn("Proceed", ctx)

    def test_iteration_label(self):
        state = _make_state(steps=[
            _make_step("investigate", "investigate", {"finding": "F1", "confidence": 0.5, "reasoning": "R1"}),
            _make_step("investigate", "investigate", {"finding": "F2", "confidence": 0.8, "reasoning": "R2"}),
        ])
        ctx = build_context_from_state(state)
        self.assertIn("iteration 2", ctx)

    def test_routing_log(self):
        state = _make_state(routing_log=[
            {"from_step": "classify", "to_step": "investigate",
             "decision_type": "deterministic", "reason": "Low confidence",
             "agent_reasoning": ""},
        ])
        state["steps"] = [_make_step("classify", "classify", {"category": "x", "confidence": 0.3, "reasoning": "y"})]
        ctx = build_context_from_state(state)
        self.assertIn("classify -> investigate", ctx)


# ═══════════════════════════════════════════════════════════════════
# 2. SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class TestSchemaRegistry(unittest.TestCase):
    def test_all_primitives_registered(self):
        expected = {"classify", "investigate", "verify", "generate",
                    "challenge", "retrieve", "think", "act"}
        self.assertEqual(set(SCHEMA_REGISTRY.keys()), expected)

    def test_get_schema(self):
        self.assertEqual(get_schema("classify"), ClassifyOutput)
        self.assertEqual(get_schema("investigate"), InvestigateOutput)
        self.assertEqual(get_schema("act"), ActOutput)

    def test_get_schema_unknown(self):
        with self.assertRaises(ValueError):
            get_schema("nonexistent")

    def test_json_spec_generates(self):
        for name in SCHEMA_REGISTRY:
            spec = schema_to_json_spec(name)
            self.assertIsInstance(spec, str)
            self.assertGreater(len(spec), 0)


class TestSchemaValidation(unittest.TestCase):
    def test_classify_valid(self):
        out = ClassifyOutput(
            confidence=0.9, reasoning="test",
            category="fraud",
        )
        self.assertEqual(out.category, "fraud")

    def test_classify_confidence_bounds(self):
        with self.assertRaises(Exception):
            ClassifyOutput(confidence=1.5, reasoning="test", category="x")
        with self.assertRaises(Exception):
            ClassifyOutput(confidence=-0.1, reasoning="test", category="x")

    def test_investigate_with_flags(self):
        out = InvestigateOutput(
            confidence=0.8, reasoning="test",
            finding="Suspicious",
            evidence_flags=["foreign_ip", "unknown_device"],
            missing_evidence=["device_fingerprint"],
        )
        self.assertEqual(out.evidence_flags, ["foreign_ip", "unknown_device"])

    def test_verify_output(self):
        out = VerifyOutput(
            confidence=0.95, reasoning="test",
            conforms=True, rules_checked=["R1", "R2"],
        )
        self.assertTrue(out.conforms)

    def test_generate_output(self):
        out = GenerateOutput(
            confidence=0.85, reasoning="test",
            artifact="Dear member, your dispute has been resolved.",
        )
        self.assertIn("Dear member", out.artifact)

    def test_challenge_output(self):
        out = ChallengeOutput(
            confidence=0.7, reasoning="test",
            survives=True, overall_assessment="Robust",
        )
        self.assertTrue(out.survives)

    def test_act_output(self):
        out = ActOutput(
            confidence=0.99, reasoning="test",
            mode="dry_run",
        )
        self.assertEqual(out.mode, "dry_run")
        self.assertFalse(out.requires_human_approval)

    def test_think_output(self):
        out = ThinkOutput(
            confidence=0.6, reasoning="test",
            thought="Considering...",
            conclusions=["A"],
            decision="Proceed",
        )
        self.assertEqual(out.decision, "Proceed")

    def test_retrieve_output(self):
        out = RetrieveOutput(
            confidence=0.95, reasoning="test",
            data={"member": {"name": "Alice"}},
        )
        self.assertEqual(out.data["member"]["name"], "Alice")


# ═══════════════════════════════════════════════════════════════════
# 3. PRIMITIVE REGISTRY
# ═══════════════════════════════════════════════════════════════════

class TestPrimitiveRegistry(unittest.TestCase):
    def test_list_primitives(self):
        prims = list_primitives()
        expected = {"classify", "investigate", "verify", "generate",
                    "challenge", "retrieve", "think", "act"}
        self.assertEqual(set(prims), expected)

    def test_get_prompt_template(self):
        for name in list_primitives():
            template = get_prompt_template(name)
            self.assertIsInstance(template, str)
            self.assertGreater(len(template), 50)

    def test_get_prompt_template_unknown(self):
        with self.assertRaises(ValueError):
            get_prompt_template("nonexistent")

    def test_get_schema_class(self):
        self.assertEqual(get_schema_class("classify"), ClassifyOutput)
        self.assertEqual(get_schema_class("act"), ActOutput)

    def test_get_config_spec(self):
        spec = get_config_spec("classify")
        self.assertIn("required_params", spec)
        self.assertIn("categories", spec["required_params"])
        self.assertIn("criteria", spec["required_params"])

    def test_get_config_spec_unknown(self):
        with self.assertRaises(ValueError):
            get_config_spec("nonexistent")


class TestRenderPrompt(unittest.TestCase):
    def test_classify_renders(self):
        prompt = render_prompt("classify", {
            "categories": "fraud, billing_error, merchant_dispute",
            "criteria": "Match based on transaction pattern",
            "context": "Transaction: $500 at unknown merchant",
            "input": "Member reports unauthorized charge of $500",
        })
        self.assertIn("fraud", prompt)

    def test_missing_required_param(self):
        with self.assertRaises(ValueError):
            render_prompt("classify", {"categories": "fraud", "input": "test"})
            # Missing "criteria"

    def test_defaults_applied(self):
        prompt = render_prompt("classify", {
            "categories": "A, B",
            "criteria": "test",
            "input": "Test input",
        })
        # Should not crash — defaults fill in context, confidence_threshold, etc.
        self.assertIsInstance(prompt, str)

    def test_investigate_renders(self):
        prompt = render_prompt("investigate", {
            "question": "Is this transaction fraudulent?",
            "scope": "Review transaction patterns and device info",
            "context": "Member reported unauthorized charge",
            "input": "Transaction $500 at unknown merchant",
        })
        self.assertIn("fraudulent", prompt)

    def test_generate_renders(self):
        prompt = render_prompt("generate", {
            "requirements": "Write a dispute resolution letter",
            "format": "formal letter",
            "constraints": "Must cite Reg E within 10 days",
            "context": "Fraud dispute for $500",
            "input": "Case data for member M001",
        })
        self.assertIn("dispute resolution", prompt)
        self.assertIn("Reg E", prompt)


class TestValidateStep(unittest.TestCase):
    def test_valid_classify(self):
        errors = validate_use_case_step({
            "name": "classify_dispute",
            "primitive": "classify",
            "params": {
                "categories": "fraud, billing",
                "criteria": "test",
            },
        })
        self.assertEqual(errors, [])

    def test_missing_primitive(self):
        errors = validate_use_case_step({"name": "bad"})
        self.assertGreater(len(errors), 0)

    def test_unknown_primitive(self):
        errors = validate_use_case_step({"name": "bad", "primitive": "nonexistent"})
        self.assertGreater(len(errors), 0)

    def test_missing_required_param(self):
        errors = validate_use_case_step({
            "name": "classify_dispute",
            "primitive": "classify",
            "params": {"categories": "fraud"},
            # Missing "criteria"
        })
        self.assertGreater(len(errors), 0)
        self.assertIn("criteria", errors[0])


# ═══════════════════════════════════════════════════════════════════
# 4. TOOL REGISTRY
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def test_register_and_list(self):
        self.reg.register("member", lambda ctx: {"name": "Alice"})
        self.reg.register("account", lambda ctx: {"balance": 1000})
        self.assertEqual(set(self.reg.list_tools()), {"member", "account"})

    def test_call_success(self):
        self.reg.register("member", lambda ctx: {"name": "Alice"})
        result = self.reg.call("member", {"member_id": "M001"})
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["name"], "Alice")
        self.assertGreaterEqual(result.latency_ms, 0)

    def test_call_not_registered(self):
        result = self.reg.call("missing", {})
        self.assertEqual(result.status, "failed")
        self.assertIn("not registered", result.error)

    def test_call_tool_raises(self):
        def bad_tool(ctx):
            raise RuntimeError("Connection failed")
        self.reg.register("bad", bad_tool)
        result = self.reg.call("bad", {})
        self.assertEqual(result.status, "failed")
        self.assertIn("Connection failed", result.error)

    def test_call_many(self):
        self.reg.register("a", lambda ctx: {"val": 1})
        self.reg.register("b", lambda ctx: {"val": 2})
        results = self.reg.call_many(["a", "b"], {})
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].data["val"], 1)
        self.assertEqual(results[1].data["val"], 2)

    def test_get_spec(self):
        self.reg.register("member", lambda ctx: {}, description="Member data", required=True)
        spec = self.reg.get("member")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.description, "Member data")
        self.assertTrue(spec.required)

    def test_get_not_found(self):
        self.assertIsNone(self.reg.get("missing"))

    def test_describe(self):
        self.reg.register("member", lambda ctx: {}, description="Member profile", required=True)
        self.reg.register("txn", lambda ctx: {}, description="Transaction detail", latency_hint_ms=80)
        desc = self.reg.describe()
        self.assertIn("member", desc)
        self.assertIn("REQUIRED", desc)
        self.assertIn("80ms", desc)

    def test_describe_empty(self):
        self.assertIn("No data sources", self.reg.describe())


class TestCaseRegistry(unittest.TestCase):
    def test_creates_tools_from_case(self):
        case = {"member_id": "M001", "complaint": {"type": "fraud"}}
        reg = create_case_registry(case)
        tools = reg.list_tools()
        self.assertIn("member_id", tools)
        self.assertIn("complaint", tools)

    def test_case_tool_returns_data(self):
        case = {"complaint": {"type": "fraud", "amount": 500}}
        reg = create_case_registry(case)
        result = reg.call("complaint", {})
        self.assertEqual(result.status, "success")
        self.assertEqual(result.data["type"], "fraud")

    def test_scalar_wrapped_in_dict(self):
        case = {"member_id": "M001"}
        reg = create_case_registry(case)
        result = reg.call("member_id", {})
        self.assertEqual(result.data["value"], "M001")


# ═══════════════════════════════════════════════════════════════════
# 5. ACTION REGISTRY
# ═══════════════════════════════════════════════════════════════════

class TestActionRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ActionRegistry()

    def test_register_and_list(self):
        self.reg.register("send_email", lambda params: {"sent": True}, "Send email")
        self.reg.register("create_ticket", lambda params: {"ticket_id": "T1"}, "Create ticket")
        self.assertEqual(set(self.reg.list_actions()), {"send_email", "create_ticket"})

    def test_execute_live(self):
        self.reg.register("send_email", lambda params: {"sent": True, "to": params.get("to")})
        result = self.reg.execute("send_email", {"to": "user@example.com"}, dry_run=False)
        self.assertEqual(result.status, "executed")
        self.assertEqual(result.response_data["to"], "user@example.com")

    def test_execute_not_registered(self):
        result = self.reg.execute("missing", {})
        self.assertEqual(result.status, "failed")
        self.assertIn("not registered", result.error)

    def test_execute_raises(self):
        def bad_fn(params):
            raise RuntimeError("Action failed")
        self.reg.register("bad", bad_fn)
        result = self.reg.execute("bad", {}, dry_run=False)
        self.assertEqual(result.status, "failed")
        self.assertIn("Action failed", result.error)

    def test_dry_run(self):
        self.reg.register("send_email", lambda p: {"sent": True})
        result = self.reg.execute("send_email", {"to": "x"}, dry_run=True)
        self.assertEqual(result.status, "simulated")
        self.assertTrue(result.response_data["dry_run"])

    def test_describe(self):
        self.reg.register("send_email", lambda p: {}, "Send notification email")
        desc = self.reg.describe()
        self.assertIn("send_email", desc)
        self.assertIn("Send notification", desc)

    def test_rollback_no_fn(self):
        self.reg.register("send_email", lambda p: {})
        result = self.reg.rollback("send_email", "handle123")
        self.assertEqual(result.status, "failed")
        self.assertIn("No rollback", result.error)

    def test_rollback_not_registered(self):
        result = self.reg.rollback("missing", "handle123")
        self.assertEqual(result.status, "failed")

    def test_rollback_success(self):
        self.reg.register(
            "credit", lambda p: {"confirmation_id": "C1"},
            rollback_fn=lambda p: {"reversed": True},
        )
        result = self.reg.rollback("credit", "handle123")
        self.assertEqual(result.status, "executed")

    def test_rollback_not_reversible(self):
        self.reg.register("irreversible", lambda p: {}, reversible=False)
        result = self.reg.rollback("irreversible", "h")
        self.assertEqual(result.status, "failed")
        self.assertIn("not reversible", result.error)

    def test_execution_log(self):
        self.reg.register("a", lambda p: {"ok": True})
        self.reg.execute("a", {}, dry_run=True)
        self.reg.execute("a", {}, dry_run=False)
        log = self.reg.get_execution_log()
        self.assertEqual(len(log), 2)
        self.assertEqual(log[0].status, "simulated")
        self.assertEqual(log[1].status, "executed")


# ═══════════════════════════════════════════════════════════════════
# 6. COMPOSER: CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════

class TestValidateUseCase(unittest.TestCase):
    def test_valid_config(self):
        config = {
            "name": "test",
            "steps": [
                {"name": "classify", "primitive": "classify",
                 "params": {"categories": "A,B", "criteria": "test"}},
                {"name": "generate", "primitive": "generate",
                 "params": {"requirements": "write", "format": "text", "constraints": "none"}},
            ],
        }
        errors = validate_use_case(config)
        self.assertEqual(errors, [])

    def test_missing_name(self):
        errors = validate_use_case({"steps": []})
        self.assertGreater(len(errors), 0)

    def test_missing_steps(self):
        errors = validate_use_case({"name": "test"})
        self.assertGreater(len(errors), 0)

    def test_empty_steps(self):
        errors = validate_use_case({"name": "test", "steps": []})
        self.assertGreater(len(errors), 0)

    def test_duplicate_step_names(self):
        config = {
            "name": "test",
            "steps": [
                {"name": "step_a", "primitive": "classify",
                 "params": {"categories": "A", "criteria": "test"}},
                {"name": "step_a", "primitive": "generate",
                 "params": {"requirements": "x", "format": "y", "constraints": "z"}},
            ],
        }
        errors = validate_use_case(config)
        self.assertGreater(len(errors), 0)

    def test_invalid_transition_target(self):
        config = {
            "name": "test",
            "steps": [
                {"name": "step_a", "primitive": "classify",
                 "params": {"categories": "A", "criteria": "test"},
                 "transitions": [{"when": "output.confidence >= 0.9", "goto": "nonexistent"}]},
            ],
        }
        errors = validate_use_case(config)
        self.assertGreater(len(errors), 0)


# ═══════════════════════════════════════════════════════════════════
# 7. COMPOSER: CONDITION EVALUATION
# ═══════════════════════════════════════════════════════════════════

class TestEvaluateCondition(unittest.TestCase):
    def _state_with_output(self, step_name, output):
        return _make_state(steps=[_make_step(step_name, "classify", output)])

    def test_equals_string(self):
        state = self._state_with_output("classify", {"category": "fraud"})
        self.assertTrue(_evaluate_condition("output.category == fraud", state, "classify"))
        self.assertFalse(_evaluate_condition("output.category == billing", state, "classify"))

    def test_equals_number(self):
        state = self._state_with_output("classify", {"confidence": 0.9})
        self.assertTrue(_evaluate_condition("output.confidence == 0.9", state, "classify"))

    def test_greater_than(self):
        state = self._state_with_output("classify", {"confidence": 0.95})
        self.assertTrue(_evaluate_condition("output.confidence >= 0.9", state, "classify"))
        self.assertFalse(_evaluate_condition("output.confidence > 0.95", state, "classify"))

    def test_less_than(self):
        state = self._state_with_output("classify", {"confidence": 0.3})
        self.assertTrue(_evaluate_condition("output.confidence < 0.5", state, "classify"))

    def test_not_equals(self):
        state = self._state_with_output("classify", {"category": "fraud"})
        self.assertTrue(_evaluate_condition("output.category != billing", state, "classify"))

    def test_boolean(self):
        state = self._state_with_output("challenge", {"survives": True})
        self.assertTrue(_evaluate_condition("output.survives == true", state, "challenge"))
        state2 = self._state_with_output("challenge", {"survives": False})
        self.assertTrue(_evaluate_condition("output.survives == false", state2, "challenge"))

    def test_nested_field(self):
        state = self._state_with_output("retrieve", {"data": {"fraud": {"score": 800}}})
        self.assertTrue(_evaluate_condition("output.data.fraud.score > 700", state, "retrieve"))

    def test_missing_output(self):
        self.assertFalse(_evaluate_condition("output.category == fraud", _make_state(), "missing"))

    def test_loop_count(self):
        state = _make_state(loop_counts={"investigate": 3})
        state["steps"] = [_make_step("investigate", "investigate", {})]
        self.assertTrue(_evaluate_condition("_loop_count >= 3", state, "investigate"))
        self.assertFalse(_evaluate_condition("_loop_count > 3", state, "investigate"))


# ═══════════════════════════════════════════════════════════════════
# 8. COMPOSER: REACHABLE STEPS
# ═══════════════════════════════════════════════════════════════════

class TestFindReachableSteps(unittest.TestCase):
    def test_linear(self):
        steps = [
            {"name": "a", "primitive": "retrieve"},
            {"name": "b", "primitive": "classify"},
            {"name": "c", "primitive": "generate"},
        ]
        self.assertEqual(_find_reachable_steps(steps, 0), {"a", "b", "c"})
        self.assertEqual(_find_reachable_steps(steps, 1), {"b", "c"})
        self.assertEqual(_find_reachable_steps(steps, 2), {"c"})

    def test_with_loop(self):
        steps = [
            {"name": "a", "primitive": "retrieve"},
            {"name": "b", "primitive": "generate"},
            {"name": "c", "primitive": "challenge",
             "transitions": [
                 {"when": "output.survives == false", "goto": "b"},
                 {"default": "__end__"},
             ]},
        ]
        reachable = _find_reachable_steps(steps, 1)
        self.assertEqual(reachable, {"b", "c"})

    def test_with_branch(self):
        steps = [
            {"name": "classify", "primitive": "classify",
             "transitions": [
                 {"when": "output.confidence >= 0.9", "goto": "generate"},
                 {"default": "investigate"},
             ]},
            {"name": "investigate", "primitive": "investigate"},
            {"name": "generate", "primitive": "generate"},
        ]
        reachable = _find_reachable_steps(steps, 0)
        self.assertEqual(reachable, {"classify", "investigate", "generate"})

    def test_agent_decide(self):
        steps = [
            {"name": "think", "primitive": "think",
             "transitions": [
                 {"agent_decide": {"options": [
                     {"step": "investigate"},
                     {"step": "generate"},
                     {"step": "__end__"},
                 ]}},
             ]},
            {"name": "investigate", "primitive": "investigate"},
            {"name": "generate", "primitive": "generate"},
        ]
        reachable = _find_reachable_steps(steps, 0)
        self.assertEqual(reachable, {"think", "investigate", "generate"})


# ═══════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
