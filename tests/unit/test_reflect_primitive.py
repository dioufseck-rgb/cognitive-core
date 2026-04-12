"""
Tests: Reflect primitive — mechanistic verification

Verifies the reflect primitive's mechanics at every layer:

1. SCHEMA — ReflectOutput fields, validation, trajectory enum values
2. REGISTRY — reflect registered, prompt loads, render works
3. PROMPT — all required sections present, output contract complete
4. DYNAMIC SPEC INJECTION — _apply_dynamic_spec correctly augments params
5. REFLECT LOOKUP — _get_last_reflect_output finds most recent reflect step
6. END-TO-END WIRING — _build_step_cfg applies dynamic spec to next primitive
7. REFLECT ISOLATION — reflect output does NOT inject into govern or reflect itself
8. TRAJECTORY SEMANTICS — continue/revise/escalate each produce correct behavior
9. ESTABLISHED FACTS — skip list prevents redundant re-examination
10. IDEMPOTENCY — applying dynamic spec twice does not compound
11. NO REFLECT IN STATE — _get_last_reflect_output returns None cleanly
12. SCHEMA REGISTRY — "reflect" in SCHEMA_REGISTRY, correct class

These are unit tests — no LLM calls required.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Repo root on path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reflect_output(
    trajectory: str = "continue",
    next_question: str = "What does ¶50 say about the reason for resource diversion?",
    hypothesis: str = "Resource diversion was voluntary, not compelled",
    template_guidance: str = "Focus only on ¶48-55. Do not broaden scope.",
    established_facts: list | None = None,
    domain_keys: list | None = None,
    revision_target: str | None = None,
    escalation_reason: str | None = None,
) -> dict:
    """Build a synthetic reflect step output dict."""
    return {
        "step_name": "reflect_after_challenge",
        "primitive": "reflect",
        "output": {
            "what_was_established": [
                "Nine violations catalogued with statutory citations",
                "Remediation delays of 16-26 months per issue",
                "Self-identification dates: ¶15, ¶23, ¶105",
            ],
            "what_was_assumed": [
                "Self-identification credit survives remediation delay scrutiny",
                "Good faith argument is available",
            ],
            "what_changed": "Challenge found self-identification argument weakened by delays",
            "sensitivity": {
                "load_bearing_fact": "¶50 resource diversion characterization",
                "current_value": "redirected resources away from dispute investigations altogether",
                "alternative_value": "redirected due to mandatory regulatory directive",
                "what_would_change": "Entire defense posture — good faith becomes viable",
            },
            "open_questions": [
                "Was resource diversion compelled or voluntary?",
                "What was the competing regulatory matter in ¶50?",
            ],
            "trajectory": trajectory,
            "revision_target": revision_target,
            "escalation_reason": escalation_reason,
            "next_question": next_question,
            "domain_keys_relevant": (
                domain_keys if domain_keys is not None else ["legal_standards.abusive_acts"]
            ),
            "established_facts_to_skip": (
                established_facts if established_facts is not None else [
                    "Violation inventory complete",
                    "Statutory citations verified",
                    "Penalty amount: $20M",
                ]
            ),
            "hypothesis": hypothesis,
            "template_guidance": template_guidance,
            "confidence": 0.88,
            "reasoning": "Challenge failed — targeted re-investigation needed",
            "reasoning_quality": 0.90,
            "outcome_certainty": 0.85,
            "evidence_used": [],
            "evidence_missing": [],
        },
    }


def _workflow_state_with_reflect(
    reflect_output: dict | None = None,
    additional_steps: list | None = None,
) -> dict:
    """Build a minimal workflow state containing a reflect step."""
    steps = []
    if additional_steps:
        steps.extend(additional_steps)
    if reflect_output:
        steps.append(reflect_output)
    return {
        "input": {"document_text": "test document"},
        "steps": steps,
        "routing_log": [],
        "loop_counts": {},
        "metadata": {"mode": "agentic"},
    }


def _make_orchestrator_step(workflow_state: dict, config: dict | None = None) -> object:
    """
    Build an OrchestratorStep with a real workflow_state_ref.
    Uses minimal config — no LLM calls.
    """
    from cognitive_core.engine.agentic_devs import OrchestratorStep

    cfg = config or {
        "name": "test_workflow",
        "mode": "agentic",
        "goal": "test goal",
        "available_primitives": ["retrieve", "investigate", "reflect", "govern"],
        "constraints": {
            "max_steps": 10,
            "max_repeat": 1,
            "must_include": ["retrieve", "govern"],
            "must_end_with": "govern",
            "challenge_must_pass": False,
        },
        "orchestrator": {"strategy": "test strategy"},
        "primitive_configs": {
            "retrieve_doc": {
                "primitive": "retrieve",
                "params": {
                    "specification": "Find all violations",
                    "strategy": "deterministic",
                },
            },
            "investigate_predicates": {
                "primitive": "investigate",
                "params": {
                    "question": "What factual predicates support the defense?",
                    "scope": "Examine the full record",
                },
            },
            "reflect_after_challenge": {
                "primitive": "reflect",
                "params": {
                    "scope": "Examine reasoning after challenge failure",
                },
            },
            "govern_outcome": {
                "primitive": "govern",
                "params": {
                    "workflow_state": "${accumulated}",
                    "governance_context": "Always GATE for legal review",
                },
            },
        },
    }

    step = OrchestratorStep.__new__(OrchestratorStep)
    # Set required attributes manually — avoid calling __init__ which needs simulator
    from cognitive_core.engine.devs import INFINITY
    step.name = "orchestrator"
    step.max_steps = cfg["constraints"]["max_steps"]
    step.max_repeat = cfg["constraints"]["max_repeat"]
    step.must_include = set(cfg["constraints"]["must_include"])
    step.must_end_with = cfg["constraints"]["must_end_with"]
    step.challenge_must_pass = cfg["constraints"]["challenge_must_pass"]
    step.goal = cfg["goal"]
    step.strategy = cfg["orchestrator"]["strategy"]
    step.primitive_configs = cfg["primitive_configs"]
    step.available_primitives = cfg["available_primitives"]
    step.model = "default"
    step.temperature = 0.1
    step.config = cfg
    step.workflow_state_ref = workflow_state
    step._simulator = None
    step._terminal_invoked = False
    step._decision_callback = None
    return step


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestReflectSchema(unittest.TestCase):
    """Verify ReflectOutput schema is correctly defined."""

    def setUp(self):
        from cognitive_core.primitives.schemas import ReflectOutput
        self.ReflectOutput = ReflectOutput

    def test_reflect_output_importable(self):
        """ReflectOutput class exists and is importable."""
        self.assertIsNotNone(self.ReflectOutput)

    def test_required_fields_present(self):
        """All required output fields are defined on ReflectOutput."""
        required = [
            "what_was_established",
            "what_was_assumed",
            "what_changed",
            "sensitivity",
            "open_questions",
            "trajectory",
            "revision_target",
            "escalation_reason",
            "next_question",
            "domain_keys_relevant",
            "established_facts_to_skip",
            "hypothesis",
            "template_guidance",
            "confidence",
            "reasoning",
            "reasoning_quality",
            "outcome_certainty",
            "evidence_used",
            "evidence_missing",
        ]
        # Collect field names from all annotations in MRO
        field_names = set()
        for cls in type.mro(self.ReflectOutput):
            field_names.update(getattr(cls, "__annotations__", {}).keys())
        # Also try pydantic model_fields if available
        try:
            field_names.update(self.ReflectOutput.model_fields.keys())
        except AttributeError:
            pass
        for field in required:
            self.assertIn(field, field_names,
                f"Required field '{field}' missing from ReflectOutput")

    def test_trajectory_accepts_valid_values(self):
        """trajectory field accepts continue, revise, escalate — documented in prompt."""
        # Verify in the prompt template rather than schema introspection
        from cognitive_core.primitives.registry import get_prompt_template
        template = get_prompt_template("reflect")
        for val in ["continue", "revise", "escalate"]:
            self.assertIn(val, template,
                f"trajectory value '{val}' not documented in reflect.txt")

    def test_sensitivity_is_optional(self):
        """sensitivity field is optional — reflect output valid without it."""
        # If sensitivity is optional, we can instantiate without it
        try:
            out = self.ReflectOutput(
                confidence=0.8,
                reasoning="test",
                trajectory="continue",
                what_changed="test",
                # sensitivity omitted
            )
            # Should not raise
            self.assertIsNotNone(out)
        except Exception as e:
            self.fail(f"ReflectOutput should be instantiable without sensitivity: {e}")

    def test_dynamic_spec_fields_are_optional(self):
        """next_question, hypothesis, template_guidance are all optional."""
        # If optional, we can instantiate without them
        try:
            out = self.ReflectOutput(
                confidence=0.8,
                reasoning="test",
                trajectory="continue",
                what_changed="test",
                # all dynamic spec fields omitted
            )
            self.assertIsNotNone(out)
        except Exception as e:
            self.fail(f"Dynamic spec fields should be optional: {e}")

    def test_schema_in_registry(self):
        """reflect is registered in SCHEMA_REGISTRY."""
        from cognitive_core.primitives.schemas import SCHEMA_REGISTRY, ReflectOutput
        self.assertIn("reflect", SCHEMA_REGISTRY)
        self.assertIs(SCHEMA_REGISTRY["reflect"], ReflectOutput)

    def test_inherits_from_base_output(self):
        """ReflectOutput inherits from BaseOutput."""
        from cognitive_core.primitives.schemas import BaseOutput, ReflectOutput
        self.assertTrue(issubclass(ReflectOutput, BaseOutput))


class TestReflectRegistry(unittest.TestCase):
    """Verify reflect is correctly registered and prompt loads."""

    def setUp(self):
        from cognitive_core.primitives.registry import (
            PRIMITIVE_CONFIGS, get_prompt_template, render_prompt
        )
        self.PRIMITIVE_CONFIGS = PRIMITIVE_CONFIGS
        self.get_prompt_template = get_prompt_template
        self.render_prompt = render_prompt

    def test_reflect_in_primitive_configs(self):
        """reflect is registered in PRIMITIVE_CONFIGS."""
        self.assertIn("reflect", self.PRIMITIVE_CONFIGS)

    def test_reflect_has_required_params(self):
        """reflect requires 'scope' parameter."""
        cfg = self.PRIMITIVE_CONFIGS["reflect"]
        self.assertIn("scope", cfg["required_params"])

    def test_reflect_has_domain_index_optional(self):
        """reflect has domain_index as optional parameter."""
        cfg = self.PRIMITIVE_CONFIGS["reflect"]
        self.assertIn("domain_index", cfg.get("optional_params", []))

    def test_reflect_prompt_loads(self):
        """reflect prompt file exists and loads without error."""
        template = self.get_prompt_template("reflect")
        self.assertIsInstance(template, str)
        self.assertGreater(len(template), 100)

    def test_reflect_prompt_has_required_sections(self):
        """Reflect prompt contains all required structural sections."""
        template = self.get_prompt_template("reflect")
        required_sections = [
            "what_was_established",
            "what_was_assumed",
            "what_changed",
            "sensitivity",
            "open_questions",
            "trajectory",
            "next_question",
            "domain_keys_relevant",
            "established_facts_to_skip",
            "hypothesis",
            "template_guidance",
            "reasoning_quality",
            "outcome_certainty",
        ]
        for section in required_sections:
            self.assertIn(section, template,
                f"Required section '{section}' missing from reflect.txt")

    def test_reflect_prompt_has_trajectory_options(self):
        """Prompt explicitly names continue, revise, escalate."""
        template = self.get_prompt_template("reflect")
        for option in ["continue", "revise", "escalate"]:
            self.assertIn(option, template,
                f"Trajectory option '{option}' missing from reflect.txt")

    def test_reflect_prompt_renders(self):
        """render_prompt produces a non-empty string for reflect."""
        rendered = self.render_prompt("reflect", {
            "scope": "Examine reasoning after challenge failure",
            "context": "Prior steps: retrieve, classify, investigate",
        })
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 500)
        self.assertIn("what_was_established", rendered)

    def test_reflect_prompt_renders_with_domain_index(self):
        """render_prompt works with domain_index provided."""
        rendered = self.render_prompt("reflect", {
            "scope": "test scope",
            "context": "test context",
            "domain_index": "  legal_standards.abusive_acts: ['citation', 'standard']",
        })
        self.assertIn("legal_standards.abusive_acts", rendered)

    def test_reflect_schema_class_correct(self):
        """PRIMITIVE_CONFIGS['reflect']['schema'] is ReflectOutput."""
        from cognitive_core.primitives.schemas import ReflectOutput
        cfg = self.PRIMITIVE_CONFIGS["reflect"]
        self.assertIs(cfg["schema"], ReflectOutput)


class TestGetLastReflectOutput(unittest.TestCase):
    """Verify _get_last_reflect_output finds the right step."""

    def _make_step(self, primitive: str, name: str, output: dict) -> dict:
        return {"step_name": name, "primitive": primitive, "output": output}

    def test_returns_none_when_no_steps(self):
        """Returns None when workflow has no steps."""
        state = _workflow_state_with_reflect()
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        self.assertIsNone(result)

    def test_returns_none_when_no_reflect_step(self):
        """Returns None when steps exist but none are reflect."""
        steps = [
            self._make_step("retrieve", "retrieve_doc", {"data": {}}),
            self._make_step("classify", "classify_violations", {"category": "systemic"}),
        ]
        state = _workflow_state_with_reflect(additional_steps=steps)
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        self.assertIsNone(result)

    def test_returns_reflect_output_when_present(self):
        """Returns the reflect step output when one exists."""
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        self.assertIsNotNone(result)
        self.assertIn("trajectory", result)
        self.assertEqual(result["trajectory"], "continue")

    def test_returns_most_recent_reflect(self):
        """Returns the LAST reflect step when multiple exist."""
        first_reflect = {
            "step_name": "reflect_1",
            "primitive": "reflect",
            "output": {"trajectory": "revise", "next_question": "first question"},
        }
        second_reflect = {
            "step_name": "reflect_2",
            "primitive": "reflect",
            "output": {"trajectory": "continue", "next_question": "second question"},
        }
        state = _workflow_state_with_reflect(
            additional_steps=[first_reflect, second_reflect]
        )
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        self.assertEqual(result["next_question"], "second question")

    def test_ignores_steps_after_reflect(self):
        """Returns reflect output even when later non-reflect steps exist."""
        reflect = _reflect_output()
        post_reflect = {
            "step_name": "investigate_targeted",
            "primitive": "investigate",
            "output": {"finding": "Resource diversion was voluntary"},
        }
        state = _workflow_state_with_reflect(
            reflect_output=reflect,
            additional_steps=[post_reflect],
        )
        # reflect comes before post_reflect in the state
        # _get_last_reflect_output should still find it
        orch = _make_orchestrator_step(state)
        # Manually order: reflect first, then investigate
        orch.workflow_state_ref["steps"] = [reflect, post_reflect]
        result = orch._get_last_reflect_output()
        self.assertIsNotNone(result)
        self.assertEqual(result["trajectory"], "continue")

    def test_handles_malformed_reflect_output(self):
        """Returns None gracefully if reflect output is not a dict."""
        bad_reflect = {
            "step_name": "reflect_bad",
            "primitive": "reflect",
            "output": "not a dict",
        }
        state = _workflow_state_with_reflect(additional_steps=[bad_reflect])
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        self.assertIsNone(result)


class TestApplyDynamicSpec(unittest.TestCase):
    """Verify _apply_dynamic_spec correctly augments primitive params."""

    def setUp(self):
        state = _workflow_state_with_reflect()
        self.orch = _make_orchestrator_step(state)

    def _apply(self, params: dict, primitive: str, reflect: dict) -> dict:
        """Helper: apply dynamic spec and return modified params."""
        self.orch._apply_dynamic_spec(params, primitive, reflect)
        return params

    def test_next_question_overrides_investigate_question(self):
        """next_question overrides the question param for investigate."""
        params = {"question": "Original static question", "scope": "Original scope"}
        reflect = _reflect_output(
            next_question="Was diversion compelled or voluntary?"
        )
        self._apply(params, "investigate", reflect["output"])
        self.assertIn("Was diversion compelled or voluntary?", params["question"])
        self.assertNotIn("Original static question", params["question"])

    def test_next_question_in_additional_instructions_for_non_investigate(self):
        """next_question goes into additional_instructions for non-investigate primitives."""
        params = {}
        reflect = _reflect_output(next_question="What does the warrant need to address?")
        self._apply(params, "deliberate", reflect["output"])
        instr = params.get("additional_instructions", "")
        self.assertIn("What does the warrant need to address?", instr)

    def test_hypothesis_prepended_to_scope(self):
        """hypothesis is prepended to scope for investigate."""
        params = {"question": "q", "scope": "Original scope"}
        reflect = _reflect_output(hypothesis="Diversion was voluntary")
        self._apply(params, "investigate", reflect["output"])
        self.assertIn("Diversion was voluntary", params["scope"])
        self.assertIn("Original scope", params["scope"])
        # Hypothesis should come BEFORE base scope
        self.assertLess(
            params["scope"].index("Diversion was voluntary"),
            params["scope"].index("Original scope"),
        )

    def test_template_guidance_in_additional_instructions(self):
        """template_guidance appears in additional_instructions."""
        params = {}
        reflect = _reflect_output(template_guidance="Focus only on ¶48-55")
        self._apply(params, "investigate", reflect["output"])
        instr = params.get("additional_instructions", "")
        self.assertIn("Focus only on ¶48-55", instr)

    def test_established_facts_in_additional_instructions(self):
        """established_facts_to_skip appears in additional_instructions."""
        params = {}
        reflect = _reflect_output(established_facts=["Violations catalogued", "Penalty: $20M"])
        self._apply(params, "investigate", reflect["output"])
        instr = params.get("additional_instructions", "")
        self.assertIn("Violations catalogued", instr)
        self.assertIn("Penalty: $20M", instr)
        self.assertIn("DO NOT RE-EXAMINE", instr.upper())

    def test_domain_keys_in_additional_instructions(self):
        """domain_keys_relevant appears in additional_instructions."""
        params = {}
        reflect = _reflect_output(domain_keys=["legal_standards.abusive_acts", "arguments.mitigation"])
        self._apply(params, "investigate", reflect["output"])
        instr = params.get("additional_instructions", "")
        self.assertIn("legal_standards.abusive_acts", instr)
        self.assertIn("arguments.mitigation", instr)

    def test_existing_additional_instructions_preserved(self):
        """Existing additional_instructions are not overwritten — dynamic spec appends."""
        params = {"additional_instructions": "EXISTING INSTRUCTIONS"}
        reflect = _reflect_output(template_guidance="New guidance")
        self._apply(params, "deliberate", reflect["output"])
        instr = params["additional_instructions"]
        self.assertIn("EXISTING INSTRUCTIONS", instr)
        self.assertIn("New guidance", instr)

    def test_dynamic_spec_marker_present(self):
        """Dynamic spec block is clearly marked in additional_instructions."""
        params = {}
        reflect = _reflect_output()
        self._apply(params, "investigate", reflect["output"])
        instr = params.get("additional_instructions", "")
        self.assertIn("DYNAMIC SPEC FROM REFLECT", instr)

    def test_empty_reflect_output_no_crash(self):
        """Empty reflect output does not crash or add noise."""
        params = {"question": "original"}
        self.orch._apply_dynamic_spec(params, "investigate", {})
        # No dynamic spec injected — params should be essentially unchanged
        self.assertEqual(params.get("question"), "original")
        self.assertEqual(params.get("additional_instructions", ""), "")

    def test_none_fields_not_injected(self):
        """None values in reflect output are not injected as strings."""
        params = {}
        reflect_out = {
            "trajectory": "continue",
            "next_question": None,
            "hypothesis": None,
            "template_guidance": None,
            "established_facts_to_skip": [],
            "domain_keys_relevant": [],
        }
        self.orch._apply_dynamic_spec(params, "investigate", reflect_out)
        instr = params.get("additional_instructions", "")
        self.assertNotIn("None", instr)


class TestReflectIsolation(unittest.TestCase):
    """Verify reflect output does NOT inject into govern or reflect itself."""

    def setUp(self):
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        self.orch = _make_orchestrator_step(state)

    def test_govern_not_affected_by_reflect(self):
        """_build_step_cfg does not apply dynamic spec to govern."""
        decision = {
            "action": "invoke",
            "primitive": "govern",
            "step_name": "govern_outcome",
            "params_key": "govern_outcome",
        }
        cfg = self.orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        instr = params.get("additional_instructions", "")
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", instr)
        self.assertNotIn("next_question", instr.lower())

    def test_reflect_not_affected_by_prior_reflect(self):
        """_build_step_cfg does not apply dynamic spec to reflect itself."""
        decision = {
            "action": "invoke",
            "primitive": "reflect",
            "step_name": "reflect_round_2",
            "params_key": "reflect_after_challenge",
            "reflect_scope": "Examine after targeted investigation",
        }
        cfg = self.orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        instr = params.get("additional_instructions", "")
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", instr)

    def test_investigate_IS_affected_by_reflect(self):
        """Confirm investigate does receive dynamic spec — positive control."""
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = self.orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        instr = params.get("additional_instructions", "")
        self.assertIn("DYNAMIC SPEC FROM REFLECT", instr)


class TestReflectScopeInjection(unittest.TestCase):
    """Verify reflect_scope from orchestrator decision becomes reflect scope param."""

    def test_reflect_scope_becomes_scope_param(self):
        """When primitive=reflect, reflect_scope from decision is used as scope."""
        state = _workflow_state_with_reflect()
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "reflect",
            "step_name": "reflect_after_challenge",
            "params_key": "reflect_after_challenge",
            "reflect_scope": "Examine why challenge found self-identification weak",
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        self.assertIn("Examine why challenge found self-identification weak", params.get("scope", ""))

    def test_reflect_scope_fallback_when_missing(self):
        """When reflect_scope absent, default scope is provided."""
        state = _workflow_state_with_reflect()
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "reflect",
            "step_name": "reflect_step",
            "params_key": "reflect_after_challenge",
            # no reflect_scope
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        self.assertIn("scope", params)
        self.assertGreater(len(params["scope"]), 0)

    def test_domain_index_injected_for_reflect(self):
        """domain_index is built and injected for reflect steps."""
        state = _workflow_state_with_reflect()
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "reflect",
            "step_name": "reflect_step",
            "params_key": "reflect_after_challenge",
            "reflect_scope": "test scope",
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg.get("params", {})
        self.assertIn("domain_index", params)
        # domain_index should contain primitive config keys
        self.assertIn("investigate_predicates", params["domain_index"])


class TestTrajectorySemantics(unittest.TestCase):
    """Verify that reflect trajectory values produce correct downstream behavior."""

    def test_continue_trajectory_passes_next_question(self):
        """continue trajectory: next_question is injected into subsequent step."""
        reflect = _reflect_output(
            trajectory="continue",
            next_question="Specific question for continue trajectory"
        )
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg["params"]
        # next_question should override the static question for investigate
        self.assertIn("Specific question for continue trajectory", params.get("question", ""))

    def test_revise_trajectory_passes_revision_context(self):
        """revise trajectory: dynamic spec still injected (same as continue)."""
        reflect = _reflect_output(
            trajectory="revise",
            revision_target="investigate_predicates",
            next_question="Targeted revise question",
        )
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg["params"]
        self.assertIn("Targeted revise question", params.get("question", ""))

    def test_escalate_trajectory_no_injection(self):
        """escalate trajectory: dynamic spec NOT injected (no next_question)."""
        reflect = _reflect_output(
            trajectory="escalate",
            next_question=None,
            escalation_reason="Cannot determine from available evidence",
        )
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg["params"]
        # escalate: no dynamic spec injected
        instr = params.get("additional_instructions", "")
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", instr)


class TestEstablishedFactsSkipping(unittest.TestCase):
    """Verify established_facts_to_skip prevents redundant re-examination."""

    def test_skip_list_in_do_not_reexamine_block(self):
        """Skip list appears under a clear DO NOT RE-EXAMINE header."""
        reflect = _reflect_output(
            established_facts=[
                "Nine violations catalogued",
                "Penalty amount $20M confirmed",
                "Self-identification dates: ¶15 ¶23 ¶105",
            ]
        )
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        instr = cfg["params"].get("additional_instructions", "")

        # All three skip facts must be present
        self.assertIn("Nine violations catalogued", instr)
        self.assertIn("Penalty amount $20M confirmed", instr)
        self.assertIn("Self-identification dates", instr)

        # Must be under a clear skip header
        upper = instr.upper()
        self.assertTrue(
            "DO NOT RE-EXAMINE" in upper or "ESTABLISHED" in upper,
            "Skip list must be under a clear DO NOT RE-EXAMINE header"
        )

    def test_empty_skip_list_no_skip_block(self):
        """Empty established_facts produces no ESTABLISHED skip block."""
        reflect = _reflect_output(established_facts=[])
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        instr = cfg["params"].get("additional_instructions", "")
        # No "ESTABLISHED — DO NOT RE-EXAMINE:" block with items
        # The block header only appears when there are items to list
        # Check that no item lines appear (lines starting with "  - ")
        lines = instr.split("\n")
        skip_item_lines = [l for l in lines
                          if l.strip().startswith("- ") and "ESTABLISHED" not in l.upper()]
        # With empty skip list, no skip items should be present after dynamic spec marker
        in_dynamic = False
        skip_items_in_dynamic = []
        for line in lines:
            if "DYNAMIC SPEC FROM REFLECT" in line:
                in_dynamic = True
            if in_dynamic and line.strip().startswith("- "):
                # Only flag if this looks like a skip list item
                # (skip items come from established_facts_to_skip)
                pass  # With empty list, no items injected
        # The real check: _apply_dynamic_spec should not call the skip block
        # when established_facts_to_skip is empty
        # Verify by checking the template_guidance didn't accidentally add skip items
        self.assertNotIn("ESTABLISHED — DO NOT RE-EXAMINE", instr)


class TestIdempotency(unittest.TestCase):
    """Verify applying dynamic spec twice does not compound."""

    def test_second_build_step_cfg_does_not_double_inject(self):
        """Calling _build_step_cfg twice produces same result — not additive."""
        reflect = _reflect_output(next_question="Is diversion compelled?")
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": "investigate",
            "step_name": "investigate_targeted",
            "params_key": "investigate_predicates",
        }
        cfg1 = orch._build_step_cfg(decision)
        cfg2 = orch._build_step_cfg(decision)

        instr1 = cfg1["params"].get("additional_instructions", "")
        instr2 = cfg2["params"].get("additional_instructions", "")

        # Both should be identical — not double-appended
        self.assertEqual(instr1, instr2)

        # The dynamic spec should appear exactly once in each
        self.assertEqual(
            instr1.count("DYNAMIC SPEC FROM REFLECT"), 1,
            "Dynamic spec block should appear exactly once"
        )


class TestOrchestratorPromptIntegration(unittest.TestCase):
    """Verify orchestrator prompt includes domain_index slot."""

    def test_orchestrator_prompt_has_domain_index(self):
        """orchestrator.txt contains {domain_index} placeholder."""
        from cognitive_core.primitives.registry import get_prompt_template
        template = get_prompt_template.__module__
        # Read file directly
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/orchestrator.txt"
        )
        content = prompt_path.read_text()
        self.assertIn("{domain_index}", content)

    def test_orchestrator_prompt_mentions_reflect(self):
        """orchestrator.txt instructs orchestrator when to use reflect."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/orchestrator.txt"
        )
        content = prompt_path.read_text()
        self.assertIn("reflect", content)
        self.assertIn("reflect_scope", content)

    def test_orchestrator_prompt_has_reflect_in_primitive_list(self):
        """orchestrator.txt lists reflect as an available primitive."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/orchestrator.txt"
        )
        content = prompt_path.read_text()
        # reflect should be in the action JSON schema
        self.assertIn("reflect", content)


class TestReflectPromptOutputContract(unittest.TestCase):
    """Verify reflect prompt produces a complete, parseable output contract."""

    def test_prompt_output_json_structure_complete(self):
        """Reflect prompt's JSON response structure contains all dynamic spec fields."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/reflect.txt"
        )
        content = prompt_path.read_text()

        # All fields that go into dynamic spec must be in the JSON contract
        dynamic_spec_fields = [
            "next_question",
            "domain_keys_relevant",
            "established_facts_to_skip",
            "hypothesis",
            "template_guidance",
            "trajectory",
            "revision_target",
            "escalation_reason",
        ]
        for field in dynamic_spec_fields:
            self.assertIn(field, content,
                f"Field '{field}' missing from reflect prompt JSON contract")

    def test_prompt_enforces_json_only_response(self):
        """Reflect prompt ends with JSON-only instruction."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/reflect.txt"
        )
        content = prompt_path.read_text()
        self.assertIn("Respond ONLY with the JSON object", content)

    def test_prompt_has_groundedness_requirements(self):
        """Reflect prompt requires grounded, traceable claims."""
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/reflect.txt"
        )
        content = prompt_path.read_text()
        self.assertIn("GROUNDEDNESS", content.upper())


# ── Run ───────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Stringent Mechanistic Tests
#
# These tests verify correctness under adversarial conditions that the first
# suite didn't reach: malformed inputs, unknown trajectory values, state
# mutation safety, injection order, one-shot semantics, and all-primitive
# coverage.
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialTrajectoryValues(unittest.TestCase):
    """Reflect behavior under malformed or unexpected trajectory values."""

    def test_unknown_trajectory_no_injection(self):
        """An unknown trajectory value does not inject dynamic spec."""
        reflect = _reflect_output(trajectory="proceed")  # not valid
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        instr = cfg["params"].get("additional_instructions", "")
        # Unknown trajectory should not inject
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", instr)

    def test_revise_with_no_next_question_no_crash(self):
        """trajectory=revise with next_question=None does not crash."""
        reflect = _reflect_output(trajectory="revise", next_question=None)
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        try:
            cfg = orch._build_step_cfg(decision)
        except Exception as e:
            self.fail(f"Should not crash with next_question=None: {e}")

    def test_revise_with_no_next_question_preserves_static_question(self):
        """trajectory=revise with next_question=None keeps the static question."""
        reflect = _reflect_output(trajectory="revise", next_question=None)
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        # Static question from domain config should be preserved
        question = cfg["params"].get("question", "")
        self.assertIn("factual predicates", question.lower())

    def test_reflect_with_zero_confidence_still_applied(self):
        """Reflect output with confidence=0.0 is still applied — confidence
        is not a gate for dynamic spec injection."""
        reflect = _reflect_output(trajectory="continue", next_question="Low confidence question")
        reflect["output"]["confidence"] = 0.0
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        # Confidence=0.0 should not prevent injection
        instr = cfg["params"].get("additional_instructions", "")
        self.assertIn("DYNAMIC SPEC FROM REFLECT", instr)


class TestStateMutationSafety(unittest.TestCase):
    """Verify that reflect operations do not mutate shared state."""

    def test_get_last_reflect_does_not_mutate_steps(self):
        """_get_last_reflect_output does not mutate workflow_state['steps']."""
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        original_steps = [dict(s) for s in state["steps"]]
        orch = _make_orchestrator_step(state)

        orch._get_last_reflect_output()

        # Steps list should be identical after call
        self.assertEqual(len(state["steps"]), len(original_steps))
        for original, current in zip(original_steps, state["steps"]):
            self.assertEqual(original["step_name"], current["step_name"])

    def test_apply_dynamic_spec_does_not_mutate_reflect_output(self):
        """_apply_dynamic_spec does not mutate the reflect output dict."""
        reflect = _reflect_output()
        original_output = json.loads(json.dumps(reflect["output"]))  # deep copy

        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        params = {"question": "original", "scope": "original scope"}
        orch._apply_dynamic_spec(params, "investigate", reflect["output"])

        # Reflect output should be unchanged
        self.assertEqual(
            reflect["output"]["next_question"],
            original_output["next_question"]
        )
        self.assertEqual(
            reflect["output"]["established_facts_to_skip"],
            original_output["established_facts_to_skip"]
        )

    def test_apply_dynamic_spec_does_not_mutate_domain_config(self):
        """_apply_dynamic_spec does not modify the static domain config params."""
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)

        # Record original domain config
        original_question = orch.primitive_configs["investigate_predicates"]["params"]["question"]
        original_scope = orch.primitive_configs["investigate_predicates"]["params"]["scope"]

        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        orch._build_step_cfg(decision)

        # Domain config should be unchanged
        self.assertEqual(
            orch.primitive_configs["investigate_predicates"]["params"]["question"],
            original_question
        )
        self.assertEqual(
            orch.primitive_configs["investigate_predicates"]["params"]["scope"],
            original_scope
        )


class TestInjectionOrderAndBoundaries(unittest.TestCase):
    """Verify injection order, boundary markers, and structural correctness."""

    def test_dynamic_spec_prepended_before_existing_instructions(self):
        """Dynamic spec block appears BEFORE existing additional_instructions (prepend).

        The directive must arrive before the LLM forms its plan, so it is
        prepended rather than appended.  See architecture finding #3 in HANDOFF.
        """
        reflect = _reflect_output(template_guidance="Dynamic guidance")
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        params = {"additional_instructions": "EXISTING STATIC INSTRUCTIONS"}
        orch._apply_dynamic_spec(params, "deliberate", reflect["output"])

        instr = params["additional_instructions"]
        existing_pos = instr.index("EXISTING STATIC INSTRUCTIONS")
        dynamic_pos = instr.index("DYNAMIC SPEC FROM REFLECT")
        self.assertLess(dynamic_pos, existing_pos,
            "Dynamic spec must be prepended before existing static instructions")

    def test_dynamic_spec_has_open_and_close_markers(self):
        """Dynamic spec block has both opening and closing boundary markers."""
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        params = {}
        orch._apply_dynamic_spec(params, "investigate", reflect["output"])

        instr = params.get("additional_instructions", "")
        self.assertIn("DYNAMIC SPEC FROM REFLECT", instr)
        self.assertIn("END DYNAMIC SPEC", instr)
        # Open marker must precede close marker
        open_pos = instr.index("DYNAMIC SPEC FROM REFLECT")
        close_pos = instr.index("END DYNAMIC SPEC")
        self.assertLess(open_pos, close_pos)

    def test_hypothesis_not_injected_for_non_investigate(self):
        """Hypothesis does NOT get injected into scope for non-investigate primitives."""
        reflect = _reflect_output(hypothesis="Test hypothesis")
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        params = {}
        orch._apply_dynamic_spec(params, "deliberate", reflect["output"])

        # Hypothesis should not appear as a scope injection for deliberate
        scope = params.get("scope", "")
        self.assertNotIn("HYPOTHESIS TO TEST", scope)

    def test_next_question_override_only_for_primitives_with_question_param(self):
        """next_question param override only applies when 'question' exists in params."""
        reflect = _reflect_output(next_question="Should not override")
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)

        # deliberate has no 'question' param — should go to additional_instructions
        params = {"instruction": "Deliberate on the findings"}
        orch._apply_dynamic_spec(params, "deliberate", reflect["output"])

        # question param should not have been created
        self.assertNotIn("question", params)
        # next_question should still appear in additional_instructions
        instr = params.get("additional_instructions", "")
        self.assertIn("Should not override", instr)


class TestMultipleReflectSteps(unittest.TestCase):
    """Verify behavior with multiple reflect steps in the trajectory."""

    def test_only_last_reflect_applies(self):
        """With two reflect steps, only the last one shapes the next primitive."""
        first = {
            "step_name": "reflect_1",
            "primitive": "reflect",
            "output": {
                "trajectory": "revise",
                "next_question": "FIRST REFLECT QUESTION",
                "template_guidance": None,
                "established_facts_to_skip": [],
                "domain_keys_relevant": [],
                "hypothesis": None,
            },
        }
        second = {
            "step_name": "reflect_2",
            "primitive": "reflect",
            "output": {
                "trajectory": "continue",
                "next_question": "SECOND REFLECT QUESTION",
                "template_guidance": None,
                "established_facts_to_skip": [],
                "domain_keys_relevant": [],
                "hypothesis": None,
            },
        }
        state = {"input": {}, "steps": [first, second], "routing_log": [],
                 "loop_counts": {}, "metadata": {}}
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        question = cfg["params"].get("question", "")

        self.assertIn("SECOND REFLECT QUESTION", question)
        self.assertNotIn("FIRST REFLECT QUESTION", question)

    def test_second_reflect_does_not_see_first_reflect_as_domain_output(self):
        """A reflect step is not treated as a domain-level step by subsequent reflect."""
        first_reflect = {
            "step_name": "reflect_1",
            "primitive": "reflect",
            "output": {
                "trajectory": "revise",
                "next_question": "First question",
                "what_established": ["First reflection claim"],
            },
        }
        second_reflect_decision = {
            "action": "invoke", "primitive": "reflect",
            "step_name": "reflect_2", "params_key": "reflect_after_challenge",
            "reflect_scope": "Second reflection scope",
        }
        state = {"input": {}, "steps": [first_reflect], "routing_log": [],
                 "loop_counts": {}, "metadata": {}}
        orch = _make_orchestrator_step(state)
        cfg = orch._build_step_cfg(second_reflect_decision)

        # Second reflect should NOT have dynamic spec from first reflect injected
        instr = cfg["params"].get("additional_instructions", "")
        self.assertNotIn("DYNAMIC SPEC FROM REFLECT", instr)

    def test_reflect_overridden_by_later_non_reflect_step(self):
        """If a non-reflect step runs after reflect, the reflect output still
        applies to the NEXT step (reflect is consumed once, not cleared)."""
        # This tests that _get_last_reflect_output correctly finds the last
        # reflect even when non-reflect steps follow it
        reflect = {
            "step_name": "reflect_1",
            "primitive": "reflect",
            "output": {
                "trajectory": "continue",
                "next_question": "After-reflect question",
                "template_guidance": None,
                "established_facts_to_skip": [],
                "domain_keys_relevant": [],
                "hypothesis": None,
            },
        }
        post_reflect_investigate = {
            "step_name": "investigate_post",
            "primitive": "investigate",
            "output": {"finding": "Something found"},
        }
        state = {
            "input": {},
            "steps": [reflect, post_reflect_investigate],
            "routing_log": [], "loop_counts": {}, "metadata": {},
        }
        orch = _make_orchestrator_step(state)
        result = orch._get_last_reflect_output()
        # Reflect is still the last reflect step — should still be found
        self.assertIsNotNone(result)
        self.assertEqual(result["next_question"], "After-reflect question")


class TestAllPrimitivesReceiveDynamicSpec(unittest.TestCase):
    """Verify every first-order primitive receives dynamic spec from reflect."""

    PRIMITIVES_AND_CONFIGS = [
        ("classify",    "retrieve_doc",            {}),
        ("verify",      "retrieve_doc",            {}),
        ("generate",    "retrieve_doc",            {}),
        ("challenge",   "retrieve_doc",            {}),
        ("deliberate",  "retrieve_doc",            {}),
        ("investigate", "investigate_predicates",  {"question": "q", "scope": "s"}),
    ]

    def _test_primitive_receives_spec(self, primitive, params_key, extra_params):
        reflect = _reflect_output(
            trajectory="continue",
            template_guidance=f"Guidance for {primitive}"
        )
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke",
            "primitive": primitive,
            "step_name": f"test_{primitive}",
            "params_key": params_key,
        }
        cfg = orch._build_step_cfg(decision)
        params = cfg["params"]

        # Must have additional_instructions with dynamic spec
        instr = params.get("additional_instructions", "")
        self.assertIn(
            "DYNAMIC SPEC FROM REFLECT", instr,
            f"Primitive '{primitive}' did not receive dynamic spec from reflect"
        )
        self.assertIn(
            f"Guidance for {primitive}", instr,
            f"Primitive '{primitive}' did not receive template_guidance"
        )

    def test_classify_receives_dynamic_spec(self):
        self._test_primitive_receives_spec("classify", "retrieve_doc", {})

    def test_verify_receives_dynamic_spec(self):
        self._test_primitive_receives_spec("verify", "retrieve_doc", {})

    def test_generate_receives_dynamic_spec(self):
        self._test_primitive_receives_spec("generate", "retrieve_doc", {})

    def test_challenge_receives_dynamic_spec(self):
        self._test_primitive_receives_spec("challenge", "retrieve_doc", {})

    def test_deliberate_receives_dynamic_spec(self):
        self._test_primitive_receives_spec("deliberate", "retrieve_doc", {})

    def test_investigate_receives_dynamic_spec(self):
        self._test_primitive_receives_spec(
            "investigate", "investigate_predicates", {"question": "q", "scope": "s"}
        )


class TestReflectPromptSemanticContracts(unittest.TestCase):
    """Verify the reflect prompt enforces the right semantic contracts."""

    def setUp(self):
        prompt_path = (
            Path(__file__).parent.parent.parent
            / "cognitive_core/primitives/prompts/reflect.txt"
        )
        self.prompt = prompt_path.read_text()

    def test_prompt_requires_sensitivity_sub_fields(self):
        """Sensitivity analysis requires all four sub-fields."""
        for field in ["load_bearing_fact", "current_value",
                      "alternative_value", "what_would_change"]:
            self.assertIn(field, self.prompt,
                f"Sensitivity sub-field '{field}' missing from reflect prompt")

    def test_prompt_distinguishes_evidence_gaps_from_reasoning_gaps(self):
        """Prompt explicitly distinguishes evidence gaps from reasoning gaps."""
        self.assertIn("reasoning gaps", self.prompt.lower(),
            "Prompt should distinguish reasoning gaps from evidence gaps")
        self.assertIn("evidence gaps", self.prompt.lower())

    def test_prompt_requires_traceable_claims(self):
        """Prompt requires established claims to be traceable to specific steps."""
        self.assertIn("traceable", self.prompt.lower(),
            "Prompt should require claims to be traceable to specific steps")

    def test_prompt_bans_unsupported_assumptions(self):
        """Prompt surfaces implicit assumptions — things taken for granted."""
        self.assertIn("took for granted", self.prompt.lower(),
            "Prompt should surface things taken for granted without verifying")

    def test_prompt_requires_one_revision_target_only(self):
        """Prompt specifies exactly one revision target — prevents scope creep."""
        self.assertIn("one revision target", self.prompt.lower(),
            "Prompt should specify exactly one revision target")

    def test_prompt_requires_next_question_to_be_answerable(self):
        """Prompt requires next_question to be answerable in principle."""
        self.assertIn("answerable", self.prompt.lower(),
            "Prompt should require next_question to be answerable in principle")


class TestOneShotSemantics(unittest.TestCase):
    """Verify that dynamic spec application is one-shot per reflect step.

    The reflect output should shape the NEXT step's spec.
    It should not accumulate or compound across multiple subsequent steps.
    The test cannot fully verify one-shot semantics without a running
    simulation, but verifies that _build_step_cfg consistently applies
    the same spec and doesn't accumulate it.
    """

    def test_dynamic_spec_same_across_multiple_calls(self):
        """Multiple calls to _build_step_cfg produce identical specs."""
        reflect = _reflect_output()
        state = _workflow_state_with_reflect(reflect_output=reflect)
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        configs = [orch._build_step_cfg(decision) for _ in range(3)]
        instrs = [c["params"].get("additional_instructions", "") for c in configs]

        # All three should be identical
        self.assertEqual(instrs[0], instrs[1])
        self.assertEqual(instrs[1], instrs[2])

        # Each should have spec exactly once
        for instr in instrs:
            self.assertEqual(instr.count("DYNAMIC SPEC FROM REFLECT"), 1)

    def test_dynamic_spec_count_does_not_grow_with_step_count(self):
        """Adding more non-reflect steps after reflect does not grow the spec."""
        reflect = _reflect_output()
        extra_steps = [
            {"step_name": f"step_{i}", "primitive": "retrieve",
             "output": {"data": {}}} for i in range(5)
        ]
        # State: reflect first, then 5 more steps
        state = {
            "input": {},
            "steps": [reflect] + extra_steps,
            "routing_log": [], "loop_counts": {}, "metadata": {},
        }
        orch = _make_orchestrator_step(state)
        decision = {
            "action": "invoke", "primitive": "investigate",
            "step_name": "investigate_targeted", "params_key": "investigate_predicates",
        }
        cfg = orch._build_step_cfg(decision)
        instr = cfg["params"].get("additional_instructions", "")
        # Dynamic spec should still appear exactly once
        self.assertEqual(instr.count("DYNAMIC SPEC FROM REFLECT"), 1)


if __name__ == "__main__":
    print("\n═══ Reflect Primitive — Mechanistic Tests ═══\n")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        # Part 1 — wiring
        TestReflectSchema,
        TestReflectRegistry,
        TestGetLastReflectOutput,
        TestApplyDynamicSpec,
        TestReflectIsolation,
        TestReflectScopeInjection,
        TestTrajectorySemantics,
        TestEstablishedFactsSkipping,
        TestIdempotency,
        TestOrchestratorPromptIntegration,
        TestReflectPromptOutputContract,
        # Part 2 — stringent mechanistic
        TestAdversarialTrajectoryValues,
        TestStateMutationSafety,
        TestInjectionOrderAndBoundaries,
        TestMultipleReflectSteps,
        TestAllPrimitivesReceiveDynamicSpec,
        TestReflectPromptSemanticContracts,
        TestOneShotSemantics,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures) + len(result.errors)
    print(f"\n{'═' * 50}")
    print(f"  {total - failures}/{total} tests passed")
    if failures:
        print(f"  {failures} FAILURES")
    else:
        print(f"  ✓ ALL TESTS PASSED")
    print(f"{'═' * 50}\n")


class TestChallengeVulnerabilityInjection(unittest.TestCase):
    """Tests for _inject_challenge_vulnerabilities.

    When challenge passes with HIGH/CRITICAL vulnerabilities, those signals
    must not be silently dropped — they must reach govern (work order) and
    generate (defensive framing on re-run).
    """

    def _workflow_state_with_challenge(self, vulnerabilities: list, survives: bool = True) -> dict:
        """Build a workflow state containing a completed challenge step."""
        return {
            "input": {"document_text": "test"},
            "steps": [
                {
                    "step_name": "challenge_determination",
                    "primitive": "challenge",
                    "output": {
                        "survives": survives,
                        "confidence": 0.8,
                        "vulnerabilities": vulnerabilities,
                        "strengths": [],
                        "overall_assessment": "test",
                        "reasoning": "test",
                        "evidence_used": [],
                        "evidence_missing": [],
                        "reasoning_quality": 0.8,
                        "outcome_certainty": 0.8,
                    },
                }
            ],
            "routing_log": [],
            "loop_counts": {},
            "metadata": {"mode": "agentic"},
        }

    def _make_orch(self, workflow_state: dict) -> object:
        from cognitive_core.engine.agentic_devs import OrchestratorStep
        cfg = {
            "name": "test_workflow",
            "mode": "agentic",
            "goal": "test",
            "available_primitives": ["generate", "challenge", "govern"],
            "constraints": {"max_steps": 10, "max_repeat": 2,
                            "must_include": [], "must_end_with": "govern",
                            "challenge_must_pass": False},
            "orchestrator": {"strategy": "test"},
            "primitive_configs": {},
        }
        orch = OrchestratorStep.__new__(OrchestratorStep)
        orch.workflow_state_ref = workflow_state
        orch.primitive_configs = {}
        orch.model = "default"
        orch.temperature = 0.1
        orch.must_include = set()
        orch.must_end_with = "govern"
        orch.challenge_must_pass = False
        orch.max_repeat = 2
        orch._terminal_invoked = False
        return orch

    def test_high_vulnerability_injected_into_govern(self):
        """HIGH vulnerability from challenge must appear in govern's additional_instructions."""
        vulns = [{"severity": "high",
                  "description": "Capital PT records use 'cervical radiculopathy' throughout",
                  "attack_vector": "Plan could argue PT addressed wrong condition",
                  "recommendation": "Resolve PT diagnosis labeling via regulatory override"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)
        params = {}
        orch._inject_challenge_vulnerabilities(params, "govern")

        instr = params.get("additional_instructions", "")
        self.assertIn("Capital PT records", instr)
        self.assertIn("HIGH", instr)
        self.assertIn("work_order", instr)

    def test_high_vulnerability_injected_into_generate(self):
        """HIGH vulnerability from challenge must appear in generate's additional_instructions on re-run."""
        vulns = [{"severity": "high",
                  "description": "Adjacent segment disease not addressed",
                  "attack_vector": "Plan could question long-term surgical outcome",
                  "recommendation": "Add disclosure of adjacent segment risk"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)
        params = {}
        orch._inject_challenge_vulnerabilities(params, "generate")

        instr = params.get("additional_instructions", "")
        self.assertIn("Adjacent segment disease", instr)
        self.assertIn("HIGH", instr)

    def test_critical_vulnerability_injected(self):
        """CRITICAL severity vulnerabilities are also injected."""
        vulns = [{"severity": "critical",
                  "description": "Fabricated statute reference",
                  "attack_vector": "Undermines entire determination",
                  "recommendation": "Remove or verify citation"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)
        params = {}
        orch._inject_challenge_vulnerabilities(params, "govern")

        instr = params.get("additional_instructions", "")
        self.assertIn("CRITICAL", instr)

    def test_medium_vulnerability_not_injected(self):
        """MEDIUM/LOW vulnerabilities are not injected — signal-to-noise."""
        vulns = [{"severity": "medium",
                  "description": "Slightly formal tone",
                  "attack_vector": "Minor",
                  "recommendation": "Soften language"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)
        params = {}
        orch._inject_challenge_vulnerabilities(params, "govern")

        instr = params.get("additional_instructions", "")
        self.assertEqual(instr, "")

    def test_no_challenge_in_state_is_noop(self):
        """When no challenge step exists, params are unchanged."""
        state = {"input": {}, "steps": [], "routing_log": [], "loop_counts": {}, "metadata": {}}
        orch = self._make_orch(state)
        params = {"additional_instructions": "existing"}
        orch._inject_challenge_vulnerabilities(params, "govern")

        self.assertEqual(params["additional_instructions"], "existing")

    def test_injection_prepends_before_existing_instructions(self):
        """Vulnerability block must appear before any existing additional_instructions."""
        vulns = [{"severity": "high", "description": "test vuln",
                  "attack_vector": "x", "recommendation": "y"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)
        params = {"additional_instructions": "EXISTING INSTRUCTIONS"}
        orch._inject_challenge_vulnerabilities(params, "govern")

        instr = params["additional_instructions"]
        vuln_pos = instr.index("test vuln")
        existing_pos = instr.index("EXISTING INSTRUCTIONS")
        self.assertLess(vuln_pos, existing_pos,
                        "Vulnerability block must precede existing instructions")

    def test_govern_message_differs_from_generate_message(self):
        """govern and generate get different framing — work_order vs defensive framing."""
        vulns = [{"severity": "high", "description": "test", "attack_vector": "x", "recommendation": "y"}]
        state = self._workflow_state_with_challenge(vulns)
        orch = self._make_orch(state)

        govern_params = {}
        orch._inject_challenge_vulnerabilities(govern_params, "govern")

        gen_params = {}
        orch._inject_challenge_vulnerabilities(gen_params, "generate")

        self.assertIn("work_order", govern_params["additional_instructions"])
        self.assertNotIn("work_order", gen_params["additional_instructions"])
        self.assertIn("artifact", gen_params["additional_instructions"])

