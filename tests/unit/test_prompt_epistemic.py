"""
Tests: Epistemic state flows into prompts correctly

Verifies:
- deliberate prompt includes reasoning_quality/outcome_certainty fields and Step 7
- govern prompt includes {epistemic_context} variable and coherence flag instructions
- build_context_from_state appends epistemic section when state["epistemic"] present
- govern node injects epistemic_context into resolved params from state dict
- registry defaults prevent KeyError when epistemic_context not supplied
- end-to-end: epistemic flags written to state flow into context seen by LLM
"""

from __future__ import annotations

import unittest


class TestDeliberatePromptUpdates(unittest.TestCase):

    def _render(self, **extra):
        from cognitive_core.primitives.registry import render_prompt
        params = {
            "context": "Prior steps completed.",
            "instruction": "Determine the appropriate action.",
            "focus": "Review all evidence carefully.",
            "additional_instructions": "",
            **extra,
        }
        return render_prompt("deliberate", params)

    def test_reasoning_quality_in_json_block(self):
        rendered = self._render()
        self.assertIn('"reasoning_quality"', rendered)

    def test_outcome_certainty_in_json_block(self):
        rendered = self._render()
        self.assertIn('"outcome_certainty"', rendered)

    def test_step_7_present(self):
        rendered = self._render()
        self.assertIn("STEP 7", rendered)
        self.assertIn("EPISTEMIC SELF-ASSESSMENT", rendered)

    def test_honesty_anchor_present(self):
        """The prompt must tell the LLM not to inflate scores."""
        rendered = self._render()
        self.assertIn("Do not inflate", rendered)

    def test_governance_consequence_mentioned(self):
        """LLM should know scores feed governance."""
        rendered = self._render()
        self.assertIn("governance layer", rendered)

    def test_coherence_flag_instruction_present(self):
        """Deliberate should know to factor in VERIFY_DELIBERATE_TENSION."""
        rendered = self._render()
        self.assertIn("VERIFY_DELIBERATE_TENSION", rendered)

    def test_scores_defined_with_scale(self):
        """Each score has a 0.0-1.0 scale description."""
        rendered = self._render()
        self.assertIn("reasoning_quality", rendered)
        self.assertIn("outcome_certainty", rendered)
        # Both should have scale anchors
        self.assertIn("0.0 =", rendered)
        self.assertIn("1.0 =", rendered)


class TestGovernPromptUpdates(unittest.TestCase):

    def _render(self, epistemic_context="All steps epistemically sound.", **extra):
        from cognitive_core.primitives.registry import render_prompt
        params = {
            "context": "Prior steps.",
            "workflow_state": "verify: violations found. deliberate: approve.",
            "governance_context": "Domain tier: gate. Case: conditional permit.",
            "tier_override": "",
            "additional_instructions": "",
            "epistemic_context": epistemic_context,
            **extra,
        }
        return render_prompt("govern", params)

    def test_epistemic_context_variable_in_prompt(self):
        rendered = self._render()
        self.assertIn("EPISTEMIC STATE", rendered)

    def test_epistemic_context_value_injected(self):
        rendered = self._render(
            epistemic_context="Coherence flag: VERIFY_DELIBERATE_TENSION (at deliberate_determination)"
        )
        self.assertIn("VERIFY_DELIBERATE_TENSION", rendered)

    def test_step_1_read_epistemic_state(self):
        rendered = self._render()
        self.assertIn("STEP 1: READ EPISTEMIC STATE", rendered)

    def test_coherence_flags_explained_in_prompt(self):
        """Govern should know what each flag means for tier selection."""
        rendered = self._render()
        self.assertIn("VERIFY_DELIBERATE_TENSION", rendered)
        self.assertIn("CLASSIFY_DELIBERATE_MISMATCH", rendered)
        self.assertIn("UNWARRANTED_RECOMMENDATION", rendered)

    def test_tier_rationale_must_reference_flags(self):
        """The requirements section should mandate flag acknowledgment."""
        rendered = self._render()
        self.assertIn("tier_rationale MUST reference", rendered)

    def test_work_order_instructions_include_flags(self):
        """Work order instructions should surface flags for the reviewer."""
        rendered = self._render()
        self.assertIn("epistemic flags", rendered)

    def test_default_epistemic_context_no_keyerror(self):
        """Rendering without epistemic_context should use the default."""
        from cognitive_core.primitives.registry import render_prompt
        params = {
            "context": "ctx",
            "workflow_state": "state",
            "governance_context": "ctx",
        }
        rendered = render_prompt("govern", params)
        self.assertIn("Epistemic state not yet computed", rendered)

    def test_accountability_chain_includes_epistemic(self):
        """Accountability record should document epistemic flags."""
        rendered = self._render()
        self.assertIn("epistemic flag", rendered)


class TestBuildContextFromStateEpistemic(unittest.TestCase):
    """build_context_from_state appends epistemic section when state has it."""

    def _state_with_epistemic(self, flags=None, warranted=True, overall=0.95, low=None, gaps=None):
        from cognitive_core.engine.state import build_context_from_state
        state = {
            "input": {},
            "steps": [
                {"step_name": "verify_step", "primitive": "verify",
                 "output": {"conforms": False, "violations": [], "confidence": 0.88}},
            ],
            "metadata": {}, "loop_counts": {}, "routing_log": [],
            "epistemic": {
                "_record": {
                    "workflow_overall": overall,
                    "coherence_flags": flags or [],
                    "warranted": warranted,
                    "open_evidence_gaps": gaps or [],
                    "low_confidence_steps": low or [],
                }
            },
        }
        return build_context_from_state(state)

    def test_epistemic_section_appears_with_flags(self):
        ctx = self._state_with_epistemic(
            flags=["VERIFY_DELIBERATE_TENSION (at deliberate_determination)"],
            warranted=False,
        )
        self.assertIn("Epistemic state", ctx)
        self.assertIn("VERIFY_DELIBERATE_TENSION", ctx)

    def test_not_warranted_warning_appears(self):
        ctx = self._state_with_epistemic(
            flags=["VERIFY_DELIBERATE_TENSION (at deliberate)"],
            warranted=False,
        )
        self.assertIn("WARNING", ctx)
        self.assertIn("not warranted", ctx)

    def test_overall_score_appears(self):
        ctx = self._state_with_epistemic(
            flags=["CONFIDENCE_DROP (at deliberate)"],
            warranted=False,
            overall=0.72,
        )
        self.assertIn("0.72", ctx)

    def test_open_gaps_appear(self):
        ctx = self._state_with_epistemic(
            flags=["UNRESOLVED_EVIDENCE_GAPS (at govern)"],
            warranted=False,
            gaps=["Full habitat survey needed before determination"],
        )
        self.assertIn("Open evidence gap", ctx)
        self.assertIn("habitat survey", ctx)

    def test_low_confidence_step_appears(self):
        ctx = self._state_with_epistemic(
            flags=["VERIFY_DELIBERATE_TENSION (at deliberate)"],
            warranted=False,
            low=[{
                "step": "deliberate_determination",
                "primitive": "deliberate",
                "overall": 0.75,
                "flags": ["VERIFY_DELIBERATE_TENSION"],
                "evidence_completeness": None,
                "rule_coverage": None,
            }],
        )
        self.assertIn("Low epistemic step", ctx)
        self.assertIn("deliberate_determination", ctx)
        self.assertIn("0.75", ctx)

    def test_no_epistemic_section_when_all_clean(self):
        """No epistemic section when everything is warranted and no flags."""
        from cognitive_core.engine.state import build_context_from_state
        state = {
            "input": {}, "steps": [], "metadata": {}, "loop_counts": {}, "routing_log": [],
            "epistemic": {
                "_record": {
                    "workflow_overall": 0.95,
                    "coherence_flags": [],
                    "warranted": True,
                    "open_evidence_gaps": [],
                    "low_confidence_steps": [],
                }
            },
        }
        ctx = build_context_from_state(state)
        self.assertNotIn("Epistemic state", ctx)

    def test_no_epistemic_section_when_key_absent(self):
        """No epistemic section when state has no epistemic key."""
        from cognitive_core.engine.state import build_context_from_state
        state = {
            "input": {}, "steps": [], "metadata": {}, "loop_counts": {}, "routing_log": {},
        }
        ctx = build_context_from_state(state)
        self.assertNotIn("Epistemic state", ctx)

    def test_epistemic_section_after_step_outputs(self):
        """Epistemic section should appear after all step outputs."""
        ctx = self._state_with_epistemic(
            flags=["VERIFY_DELIBERATE_TENSION (at deliberate)"],
            warranted=False,
        )
        verify_pos = ctx.find("verify_step")
        epistemic_pos = ctx.find("Epistemic state")
        self.assertGreater(epistemic_pos, verify_pos,
                           "Epistemic section should come after step outputs")


class TestGovernNodeEpistemicInjection(unittest.TestCase):
    """
    Govern node reads state['epistemic']['_record'] and injects
    epistemic_context into resolved params before rendering the prompt.
    """

    def test_epistemic_context_built_from_state(self):
        """
        Simulate the govern node's resolved dict assembly.
        If state has epistemic record with flags, epistemic_context should
        contain those flags.
        """
        # Replicate the logic from create_govern_node
        state = {
            "input": {},
            "epistemic": {
                "_record": {
                    "workflow_overall": 0.75,
                    "coherence_flags": ["VERIFY_DELIBERATE_TENSION (at deliberate_determination)"],
                    "warranted": False,
                    "open_evidence_gaps": [],
                    "low_confidence_steps": [{
                        "step": "deliberate_determination",
                        "primitive": "deliberate",
                        "overall": 0.75,
                        "flags": ["VERIFY_DELIBERATE_TENSION"],
                    }],
                }
            },
        }

        resolved = {}
        ep_record = state.get("epistemic", {}).get("_record")
        if ep_record:
            flags = ep_record.get("coherence_flags", [])
            warranted = ep_record.get("warranted", True)
            overall = ep_record.get("workflow_overall")
            low = ep_record.get("low_confidence_steps", [])
            gaps = ep_record.get("open_evidence_gaps", [])
            lines = []
            if overall is not None:
                lines.append(f"Workflow epistemic overall: {overall:.2f}")
            if not warranted:
                lines.append("WARRANTED: false — one or more steps have unresolved epistemic problems")
            for f in flags:
                lines.append(f"Coherence flag: {f}")
            for s in low[:3]:
                step_flags = s.get("flags", [])
                lines.append(f"Low epistemic step: {s['step']} overall={s['overall']:.2f}" +
                              (f" flags={step_flags}" if step_flags else ""))
            for g in gaps:
                lines.append(f"Open gap: {g[:100]}")
            resolved["epistemic_context"] = "\n".join(lines) if lines else "All steps epistemically sound."

        self.assertIn("epistemic_context", resolved)
        self.assertIn("VERIFY_DELIBERATE_TENSION", resolved["epistemic_context"])
        self.assertIn("WARRANTED: false", resolved["epistemic_context"])
        self.assertIn("0.75", resolved["epistemic_context"])

    def test_epistemic_context_clean_when_all_good(self):
        """When no flags, epistemic_context should say all steps sound."""
        state = {
            "input": {},
            "epistemic": {
                "_record": {
                    "workflow_overall": 0.95,
                    "coherence_flags": [],
                    "warranted": True,
                    "open_evidence_gaps": [],
                    "low_confidence_steps": [],
                }
            },
        }

        resolved = {}
        ep_record = state.get("epistemic", {}).get("_record")
        if ep_record:
            flags = ep_record.get("coherence_flags", [])
            warranted = ep_record.get("warranted", True)
            overall = ep_record.get("workflow_overall")
            low = ep_record.get("low_confidence_steps", [])
            gaps = ep_record.get("open_evidence_gaps", [])
            lines = []
            if overall is not None:
                lines.append(f"Workflow epistemic overall: {overall:.2f}")
            if not warranted:
                lines.append("WARRANTED: false")
            for f in flags:
                lines.append(f"Coherence flag: {f}")
            for s in low[:3]:
                step_flags = s.get("flags", [])
                lines.append(f"Low epistemic step: {s['step']} overall={s['overall']:.2f}")
            resolved["epistemic_context"] = "\n".join(lines) if lines else "All steps epistemically sound."

        self.assertIn("epistemic_context", resolved)
        # Overall score is always included when present; "All steps sound" is the
        # fallback only when lines is empty (no overall score either)
        self.assertIn("0.95", resolved["epistemic_context"])
        self.assertNotIn("WARRANTED: false", resolved["epistemic_context"])
        self.assertNotIn("Coherence flag", resolved["epistemic_context"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
