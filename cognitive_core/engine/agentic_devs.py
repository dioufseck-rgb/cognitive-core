"""
Cognitive Core — DEVS-Native Agentic Executor

Replaces the LangGraph hub-and-spoke agentic executor with a native
DEVS implementation. The orchestrator is an AtomicDEVS component that
sits alongside the primitive step components in a CoupledDEVS model.

Architecture:

    AgenticWorkflowModel (CoupledDEVS)
      ├── OrchestratorStep (AtomicDEVS)   — decides next primitive
      └── WorkflowStep × N  (AtomicDEVS)  — one per available primitive

    Coupling:
      OrchestratorStep.out_activate[primitive] → WorkflowStep.in_activate
      WorkflowStep.out_done                    → OrchestratorStep.in_step_done

Key properties (identical to workflow mode):

    - Every primitive step is an AtomicDEVS component.
    - ta(s) = INFINITY for waiting/running/passive states.
    - Suspension is ta(s) = INFINITY; resume is extTransition.
    - The step callback fires on every done→passive transition, giving
      agentic mode the same epistemic state computation, governance
      enforcement, and ledger recording as workflow mode.
    - Hard constraints (max_steps, must_include, must_end_with) are
      enforced by the OrchestratorStep, not communicated to the LLM
      as preferences.
    - The orchestrator's trajectory choices are recorded in the
      workflow state's routing_log, which is captured in the ledger.

The orchestrator controls the path.
The substrate controls the accountability.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from pypdevs.DEVS import AtomicDEVS, CoupledDEVS
    from pypdevs.simulator import Simulator
    from pypdevs.infinity import INFINITY
except ImportError:
    from cognitive_core.engine.devs import AtomicDEVS, CoupledDEVS, Simulator, INFINITY

from cognitive_core.engine.devs_executor import (
    StepState, StepInterrupt, StepResult, StepCallback,
    no_interrupt_callback, WorkflowStep,
)
from cognitive_core.engine.state import (
    WorkflowState, get_step_output, resolve_param, build_context_from_state,
)
from cognitive_core.primitives.registry import PRIMITIVE_CONFIGS


# ─── Orchestrator state ───────────────────────────────────────────────────────

@dataclass
class OrchestratorState:
    phase: str          # 'active' | 'running' | 'decided' | 'waiting' | 'done'
    step_count: int = 0
    last_decision: dict = field(default_factory=dict)
    finished: bool = False


# ─── Orchestrator AtomicDEVS ──────────────────────────────────────────────────

class OrchestratorStep(AtomicDEVS):
    """
    Central decision-maker in agentic execution.

    States:
        active    — ta = 0. Ready to make a decision immediately.
        running   — ta = INFINITY. LLM call in progress (background thread).
        decided   — ta = 0. Decision ready; will fire activation to chosen step.
        waiting   — ta = INFINITY. Waiting for the activated step to complete.
        done      — ta = 0. Workflow complete; signals termination.

    The step callback fires on done→passive of any WorkflowStep, which
    reinjects an activation signal into the orchestrator, moving it from
    waiting back to active for the next decision.
    """

    ORCHESTRATOR_SLA = 120  # 2 minutes for orchestrator decision

    def __init__(
        self,
        config: dict,
        workflow_state_ref: dict,
        model: str,
        temperature: float,
        decision_callback=None,
    ):
        AtomicDEVS.__init__(self, "orchestrator")

        self.config = config
        self.workflow_state_ref = workflow_state_ref
        self.model = model
        self.temperature = temperature
        self._simulator = None
        self._decision_callback = decision_callback  # called after each orchestrator decision

        # Parse constraints
        constraints = config.get("constraints", {})
        self.max_steps = constraints.get("max_steps", 10)
        self.max_repeat = constraints.get("max_repeat", 3)
        self.must_include = set(constraints.get("must_include", []))
        self.must_end_with = constraints.get("must_end_with", None)
        self.challenge_must_pass = constraints.get("challenge_must_pass", True)
        self.available_primitives = config.get("available_primitives", [])
        self.goal = config.get("goal", "")
        self.primitive_configs = config.get("primitive_configs", {})
        self.strategy = config.get("orchestrator", {}).get("strategy", "")

        # Ports: one out_activate per available primitive, one in_step_done
        self.out_activate = {}
        for prim in self.available_primitives:
            self.out_activate[prim] = self.addOutPort(f"activate_{prim}")
        self.in_step_done = self.addInPort("step_done")
        self.out_finished = self.addOutPort("finished")

        # Start active — make first decision immediately
        self.state = OrchestratorState(phase="active")
        self._terminal_invoked = False   # Thread-safe flag: terminal primitive invoked
        self._decision_lock = threading.Lock()  # Serialize decision making

    def timeAdvance(self) -> float:
        phase = self.state.phase
        if phase in ("active", "decided", "done"):
            return 0.0
        return INFINITY  # running, waiting

    def intTransition(self):
        s = self.state
        new = OrchestratorState(**vars(s))

        if s.phase == "active":
            # Kick off LLM decision in background thread
            new.phase = "running"
            self._decide_async()
            return new

        if s.phase == "decided":
            if s.last_decision.get("action") == "end":
                new.phase = "done"
                new.finished = True
            else:
                new.phase = "waiting"
            return new

        if s.phase == "done":
            # Terminal — go passive
            new.phase = "waiting"
            new.finished = True
            return new

        return s

    def extTransition(self, inputs: dict):
        s = self.state
        new = OrchestratorState(**vars(s))

        if self.in_step_done in inputs:
            # A primitive step just completed — time for next decision
            payload = inputs[self.in_step_done][0]
            new.step_count = s.step_count + 1
            # Do not reactivate if terminal primitive already invoked
            if s.phase == "waiting" and not self._terminal_invoked:
                new.phase = "active"
        return new

    def outputFnc(self) -> dict:
        s = self.state
        if s.phase == "decided":
            decision = s.last_decision
            if decision.get("action") == "end":
                return {self.out_finished: {"finished": True}}
            primitive = decision.get("primitive", "")
            if primitive in self.out_activate:
                # Build step_cfg from decision and inject into workflow state
                step_cfg = self._build_step_cfg(decision)
                return {self.out_activate[primitive]: step_cfg}
        if s.phase == "done":
            return {self.out_finished: {"finished": True}}
        return {}

    # ── Decision logic ────────────────────────────────────────────────────────

    def _decide_async(self):
        """Run the orchestrator LLM call in a background thread."""
        done_event = threading.Event()

        def _run():
            try:
                decision = self._make_decision()
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                decision = {"action": "end", "reasoning": f"Orchestrator error: {e}\n{tb}"}
                print(f"[agentic_devs] OrchestratorStep._decide_async error:\n{tb}", flush=True)
            finally:
                done_event.set()

            if self._simulator:
                # Set terminal flag immediately if govern was chosen
                if (decision.get("primitive") == self.must_end_with
                        or decision.get("action") == "end"):
                    self._terminal_invoked = True
                # Record decision in routing_log
                self._record_decision(decision)
                # Update state with decision and move to 'decided'
                self._simulator.inject(
                    model=self,
                    port=self.in_step_done,
                    value={"_decision": decision},
                    sim_time=None,
                )

        def _watchdog():
            if not done_event.wait(timeout=self.ORCHESTRATOR_SLA):
                decision = {"action": "end", "reasoning": "Orchestrator timed out"}
                self._record_decision(decision)
                if self._simulator:
                    self._simulator.inject(
                        model=self,
                        port=self.in_step_done,
                        value={"_decision": decision},
                        sim_time=None,
                    )

        threading.Thread(target=_run, daemon=True).start()
        threading.Thread(target=_watchdog, daemon=True).start()

    def extTransition(self, inputs: dict):
        """Handle both step_done (from primitives) and decision injection."""
        s = self.state
        new = OrchestratorState(**vars(s))

        if self.in_step_done in inputs:
            payload = inputs[self.in_step_done][0]

            # Decision injection from background thread
            if "_decision" in payload:
                decision = payload["_decision"]
                new.last_decision = decision
                new.phase = "decided"
                return new

            # Step completion signal from a primitive
            new.step_count = s.step_count + 1
            if s.phase == "waiting":
                new.phase = "active"

        return new

    def _make_decision(self) -> dict:
        """Call the orchestrator LLM and return a decision dict."""
        from cognitive_core.engine.llm import create_llm
        from cognitive_core.engine.nodes import extract_json

        state = self.workflow_state_ref
        steps_completed = state.get("steps", [])
        step_count = len(steps_completed)

        # ── Deterministic overrides before calling LLM ────────────────────

        # Max steps reached
        if step_count >= self.max_steps:
            return {"action": "end", "reasoning": f"Reached max_steps={self.max_steps}"}

        # Deterministic post-generate transition:
        # If generate just ran, force challenge immediately.
        # The LLM must not be asked — it will loop back to generate.
        if steps_completed:
            last = steps_completed[-1]
            if last.get("primitive") == "generate":
                challenge_key = next(
                    (k for k, v in self.primitive_configs.items()
                     if isinstance(v, dict) and v.get("primitive") == "challenge"),
                    "challenge_determination"
                )
                return {
                    "action": "invoke",
                    "reasoning": "Generate completed — forcing challenge per transition rules.",
                    "primitive": "challenge",
                    "step_name": challenge_key,
                    "params_key": challenge_key,
                    "reflect_scope": "",
                }

        # Deterministic post-challenge transition:
        # If challenge just passed (survives=True), force govern immediately.
        # The LLM must not be asked — it will loop.
        if steps_completed:
            last = steps_completed[-1]
            if last.get("primitive") == "challenge":
                survives = last.get("output", {}).get("survives", False)
                if survives:
                    executed = {s["primitive"] for s in steps_completed}
                    missing = self.must_include - executed - {"govern"}
                    if not missing:
                        govern_key = next(
                            (k for k, v in self.primitive_configs.items()
                             if isinstance(v, dict) and v.get("primitive") == "govern"),
                            "govern_appeal_outcome"
                        )
                        return {
                            "action": "invoke",
                            "reasoning": "Challenge passed — forcing govern per transition rules.",
                            "primitive": self.must_end_with or "govern",
                            "step_name": govern_key,
                            "params_key": govern_key,
                            "reflect_scope": "",
                        }

        # Deterministic end: must_end_with primitive just completed and all must_include satisfied
        if steps_completed and self.must_end_with:
            last = steps_completed[-1]
            last_prim = last.get("primitive", "")
            if last_prim == self.must_end_with:
                # For challenge: also require it passed if challenge_must_pass is set
                if last_prim == "challenge" and self.challenge_must_pass:
                    if not last.get("output", {}).get("survives"):
                        pass  # challenge failed — let orchestrator decide (regen or escalate)
                    else:
                        executed = {s["primitive"] for s in steps_completed}
                        if not (self.must_include - executed):
                            return {"action": "end", "reasoning": "Challenge passed and all required primitives executed"}
                else:
                    # govern or any other terminal primitive — end immediately
                    executed = {s["primitive"] for s in steps_completed}
                    if not (self.must_include - executed):
                        return {"action": "end", "reasoning": f"{last_prim} completed — workflow done"}

        # Hard stop: terminal primitive already invoked.
        # Uses a thread-safe flag set the moment any govern decision is made,
        # before the LLM call returns — prevents concurrent threads from
        # each deciding to invoke govern simultaneously.
        if self.must_end_with:
            executed_prims = [s["primitive"] for s in steps_completed]
            terminal_count = executed_prims.count(self.must_end_with)
            if terminal_count >= 1 or self._terminal_invoked:
                return {
                    "action": "end",
                    "reasoning": (
                        f"{self.must_end_with} already invoked "
                        f"— workflow terminated."
                    ),
                }

        # ── Deterministic repeat limit enforcement ────────────────────────
        # Count per primitive (not per step_name) to prevent LLM from
        # bypassing max_repeat by inventing new step_name variants.
        prim_counts: dict[str, int] = {}
        for s in steps_completed:
            p = s.get("primitive", "")
            prim_counts[p] = prim_counts.get(p, 0) + 1

        # If any non-terminal primitive has hit max_repeat, block it
        # by forcing the orchestrator toward govern if must_include is met.
        over_limit = [p for p, c in prim_counts.items()
                      if c >= self.max_repeat
                      and p not in (self.must_end_with, "reflect", "govern")]
        if over_limit:
            executed = {s["primitive"] for s in steps_completed}
            missing = self.must_include - executed
            if not missing:
                # All required primitives done and generate is over limit — go to govern
                if "generate" in over_limit or "challenge" in over_limit:
                    return {
                        "action": "invoke",
                        "reasoning": f"Primitives {over_limit} hit max_repeat={self.max_repeat}. Forcing govern.",
                        "primitive": self.must_end_with or "govern",
                        "step_name": "govern_appeal_outcome",
                        "params_key": "govern_appeal_outcome",
                        "reflect_scope": "",
                    }

        # ── Build prompt ───────────────────────────────────────────────────

        prompt_file = Path(__file__).parent.parent / "primitives" / "prompts" / "orchestrator.txt"
        prompt_template = prompt_file.read_text()

        steps_summary = self._build_steps_summary(steps_completed)
        routing_log = state.get("routing_log", [])
        routing_summary = "\n".join(
            f"  {r.get('from_step','')} → {r.get('to_step','')} "
            f"({r.get('decision_type','')}): {r.get('reason','')}"
            for r in routing_log
        ) or "  (no routing decisions yet)"

        prims_desc = []
        for p in self.available_primitives:
            cfgs = [k for k, v in self.primitive_configs.items()
                    if isinstance(v, dict) and v.get("primitive") == p]
            prims_desc.append(f"  {p}: configs={cfgs}")

        configs_text = json.dumps(
            {k: {"primitive": v.get("primitive"), "params": list(v.get("params", {}).keys())}
             for k, v in self.primitive_configs.items()
             if isinstance(v, dict)},
            indent=2,
        )

        constraints_text = (
            f"max_steps: {self.max_steps}\n"
            f"max_repeat per step_name: {self.max_repeat}\n"
            f"must_include: {list(self.must_include)}\n"
            f"must_end_with: {self.must_end_with}\n"
            f"challenge_must_pass: {self.challenge_must_pass}"
        )

        # Build domain index — keys from the domain config for reflect to reference
        domain_index_parts = []
        for k, v in self.primitive_configs.items():
            if isinstance(v, dict) and v.get("params"):
                domain_index_parts.append(f"  {k}: {list(v['params'].keys())}")
        domain_index_text = "\n".join(domain_index_parts) or "No domain index available."

        prompt = prompt_template.format(
            goal=self.goal,
            available_primitives="\n".join(prims_desc),
            primitive_configs=configs_text,
            constraints=constraints_text,
            steps_completed=steps_summary,
            routing_log=routing_summary,
            step_count=step_count,
            max_steps=self.max_steps,
            max_repeat=self.max_repeat,
            strategy=self.strategy,
            domain_index=domain_index_text,
        )

        orch_model = self.config.get("orchestrator", {}).get("model", self.model)
        llm = create_llm(model=orch_model, temperature=self.temperature)

        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            class HumanMessage:
                def __init__(self, content):
                    self.content = content

        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            from cognitive_core.engine.governance import _extract_text
            raw = _extract_text(response.content)
        except Exception:
            raw = str(response.content) if response.content else ""
        try:
            return extract_json(raw)
        except Exception as e:
            print(f"[agentic_devs] Failed to parse orchestrator response: {e}\nRaw: {raw[:300]}", flush=True)
            return {"action": "end", "reasoning": f"Failed to parse orchestrator response: {e}"}

    def _build_steps_summary(self, steps: list) -> str:
        if not steps:
            return "  (none — this is the first step)"
        parts = []
        for s in steps:
            out = s.get("output", {})
            summary = f"  {s['step_name']} ({s['primitive']}): "
            if out.get("error"):
                summary += f"⚠ FAILED: {str(out['error'])[:100]}"
            elif s["primitive"] == "classify":
                summary += f"→ {out.get('category','?')} (conf: {out.get('confidence','?')})"
            elif s["primitive"] == "investigate":
                summary += f"→ {str(out.get('finding','?'))[:150]}"
            elif s["primitive"] == "generate":
                summary += f"→ artifact ({len(str(out.get('artifact','')))} chars)"
            elif s["primitive"] == "challenge":
                summary += f"→ survives={out.get('survives','?')}, vulns={len(out.get('vulnerabilities',[]))}"
            elif s["primitive"] == "verify":
                summary += f"→ conforms={out.get('conforms','?')}"
            elif s["primitive"] == "retrieve":
                summary += f"→ sources: {list(out.get('data',{}).keys())}"
            elif s["primitive"] == "deliberate":
                summary += f"→ action: {out.get('recommended_action','?')}"
            elif s["primitive"] == "govern":
                tier = str(out.get("tier_applied", out.get("tier", "?"))).replace("GovernanceTier.", "")
                summary += f"→ tier={tier} disposition={out.get('disposition','?')}"
            else:
                summary += f"→ {str(out)[:100]}"
            parts.append(summary)
        return "\n".join(parts)

    def _build_step_cfg(self, decision: dict) -> dict:
        """Build a step_cfg dict from an orchestrator decision.

        If a reflect step has run recently, its dynamic spec fields are
        applied to augment the static params from the domain config:
          - next_question     → overrides question/instruction param
          - template_guidance → prepended to additional_instructions
          - established_facts_to_skip → appended to additional_instructions
          - hypothesis        → appended to question/scope for investigate
          - domain_keys_relevant → stored for reference (not injected directly)
        """
        primitive = decision.get("primitive", "")
        step_name = decision.get("step_name", primitive)
        params_key = decision.get("params_key", step_name)

        prim_config = self.primitive_configs.get(params_key, {})
        if not isinstance(prim_config, dict):
            prim_config = {}

        params = dict(prim_config.get("params", {}))

        # For reflect: inject reflect_scope from orchestrator decision
        if primitive == "reflect":
            scope = decision.get("reflect_scope", "")
            if scope:
                params["scope"] = scope
            elif "scope" not in params:
                params["scope"] = (
                    "Examine the full accumulated reasoning state. "
                    "Identify assumptions, what changed, and what the next step should ask."
                )
            # Build domain index for reflect
            domain_index_parts = []
            for k, v in self.primitive_configs.items():
                if isinstance(v, dict) and v.get("params"):
                    domain_index_parts.append(f"  {k}: {list(v['params'].keys())}")
            params["domain_index"] = "\n".join(domain_index_parts) or "No domain index."

        # Resolve ${...} references against current workflow state
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                resolved_params[k] = resolve_param(v, self.workflow_state_ref)
            else:
                resolved_params[k] = v

        # Auto-build context if not provided
        if "context" not in resolved_params:
            input_ctx = json.dumps(
                self.workflow_state_ref.get("input", {}), indent=2
            )
            prior_ctx = build_context_from_state(self.workflow_state_ref)
            resolved_params["context"] = f"Workflow Input:\n{input_ctx}\n\n{prior_ctx}"

        # ── Apply dynamic spec from most recent reflect output ──────────
        # Reflect is a second-order primitive — its output shapes subsequent
        # primitive calls. If a reflect step completed and its trajectory is
        # continue or revise, inject its dynamic spec fields.
        if primitive != "reflect" and primitive != "govern":
            reflect_output = self._get_last_reflect_output()
            if reflect_output:
                trajectory = reflect_output.get("trajectory", "continue")
                if trajectory in ("continue", "revise"):
                    self._apply_dynamic_spec(resolved_params, primitive, reflect_output)

        # Inject challenge vulnerabilities into govern and generate.
        # When challenge passes with HIGH/CRITICAL vulnerabilities, those signals
        # must not be silently dropped — they inform governance tier and defensive
        # framing. This is separate from the reflect dynamic spec path.
        if primitive in ("govern", "generate"):
            self._inject_challenge_vulnerabilities(resolved_params, primitive)

        return {
            "name": step_name,
            "primitive": primitive,
            "params": resolved_params,
            "model": prim_config.get("model", self.model),
            "temperature": prim_config.get("temperature", self.temperature),
            "sla_seconds": prim_config.get("sla_seconds", _AgenticWorkflowStep.DEFAULT_SLA_SECONDS),
        }

    def _get_last_reflect_output(self) -> dict | None:
        """Return the output of the most recent reflect step, if any."""
        steps = self.workflow_state_ref.get("steps", [])
        for step in reversed(steps):
            if step.get("primitive") == "reflect":
                output = step.get("output", {})
                if isinstance(output, dict):
                    return output
        return None


    def _inject_challenge_vulnerabilities(
        self, resolved_params: dict, primitive: str
    ) -> None:
        """Inject HIGH/CRITICAL challenge vulnerabilities into govern and generate params.

        When challenge passes with residual vulnerabilities, those signals must
        not be silently dropped. For govern: they surface into the work order so
        the human reviewer knows exactly what to examine. For generate (re-run
        after a failed challenge): they inform defensive framing so the artifact
        preemptively addresses known weak points.

        This is separate from the reflect dynamic spec path — reflect shapes
        analysis direction, this shapes output defensibility and governance routing.
        """
        steps = self.workflow_state_ref.get("steps", [])
        high_vulns = []
        for step in steps:
            if step.get("primitive") == "challenge":
                output = step.get("output", {})
                if isinstance(output, dict):
                    for v in output.get("vulnerabilities", []):
                        if v.get("severity") in ("high", "critical"):
                            high_vulns.append(v)

        if not high_vulns:
            return

        if primitive == "govern":
            header = (
                "The challenge step passed (survives=True) but flagged the following "
                "HIGH/CRITICAL vulnerabilities. These must be reflected in the work_order "
                "instructions so the human reviewer knows exactly what to examine:\n"
            )
        else:  # generate (re-run after failed challenge)
            header = (
                "Prior challenge step flagged these vulnerabilities. Address each one "
                "in the artifact — either resolve it directly with available evidence, "
                "or acknowledge the gap explicitly rather than asserting the unsupported claim:\n"
            )

        lines = [header, "CHALLENGE VULNERABILITIES REQUIRING ATTENTION:"]
        for i, v in enumerate(high_vulns, 1):
            lines.append(
                f"  {i}. [{v.get('severity', '?').upper()}] {v.get('description', '?')}\n"
                f"     Attack vector: {v.get('attack_vector', 'not specified')}\n"
                f"     Recommendation: {v.get('recommendation', 'not specified')}"
            )

        block = "\n".join(lines)
        existing = resolved_params.get("additional_instructions", "")
        resolved_params["additional_instructions"] = (
            block + ("\n\n" + existing if existing else "")
        )


    def _apply_dynamic_spec(
        self, resolved_params: dict, primitive: str, reflect_output: dict
    ) -> None:
        """Apply reflect's dynamic spec fields to the next primitive's params.

        Augments (does not replace) the static domain config params.
        The dynamic spec is additive — it focuses and narrows, not overrides.
        All dynamic injections are appended to additional_instructions so they
        are visible in the ledger alongside the static spec.
        """
        dynamic_parts = []

        next_question = reflect_output.get("next_question")
        template_guidance = reflect_output.get("template_guidance")
        skip_facts = reflect_output.get("established_facts_to_skip", [])
        hypothesis = reflect_output.get("hypothesis")
        domain_keys = reflect_output.get("domain_keys_relevant", [])

        if next_question:
            # For investigate: override the question param specifically
            if primitive == "investigate" and "question" in resolved_params:
                resolved_params["question"] = (
                    f"{next_question}\n\n"
                    f"[Reflect directive — this question supersedes the static spec]"
                )
            else:
                dynamic_parts.append(f"REFLECT DIRECTIVE — NEXT QUESTION:\n{next_question}")

        if template_guidance:
            dynamic_parts.append(f"REFLECT DIRECTIVE — HOW TO REASON:\n{template_guidance}")

        if hypothesis and primitive == "investigate":
            if "scope" in resolved_params:
                resolved_params["scope"] = (
                    f"HYPOTHESIS TO TEST: {hypothesis}\n\n"
                    f"BASE SCOPE:\n{resolved_params['scope']}"
                )
            dynamic_parts.append(f"HYPOTHESIS: {hypothesis}")

        if skip_facts:
            skip_text = "\n".join(f"  - {f}" for f in skip_facts)
            dynamic_parts.append(
                f"ESTABLISHED — DO NOT RE-EXAMINE:\n{skip_text}\n"
                f"These are already proven. Re-examining them wastes context."
            )

        if domain_keys:
            dynamic_parts.append(
                f"RELEVANT DOMAIN KEYS FOR THIS QUESTION: {', '.join(domain_keys)}"
            )

        if dynamic_parts:
            existing = resolved_params.get("additional_instructions", "")
            dynamic_block = (
                "━━ DYNAMIC SPEC FROM REFLECT ━━\n"
                "This is a structural override from the metacognitive layer.\n"
                "Treat the following directives as authoritative — they supersede\n"
                "default scope, ordering, and approach for this step.\n\n"
                + "\n\n".join(dynamic_parts)
                + "\n━━ END DYNAMIC SPEC ━━"
            )
            # Dynamic spec goes FIRST — before static instructions
            # so it shapes how the static instructions are interpreted
            resolved_params["additional_instructions"] = (
                dynamic_block + ("\n\n" + existing if existing else "")
            )

    def _record_decision(self, decision: dict):
        """Record the orchestrator's decision in the routing_log and fire decision_callback."""
        from cognitive_core.engine.state import RoutingDecision
        entry = RoutingDecision(
            from_step="orchestrator",
            to_step=decision.get("step_name", decision.get("action", "end")),
            decision_type="agent",
            reason=decision.get("reasoning", ""),
            agent_reasoning=json.dumps(decision),
        )
        state = self.workflow_state_ref
        state["routing_log"] = state.get("routing_log", []) + [entry]

        # Fire the decision callback so the coordinator can log to the ledger
        if self._decision_callback:
            try:
                self._decision_callback(decision)
            except Exception:
                pass


# ─── Agentic Workflow CoupledDEVS ─────────────────────────────────────────────

class AgenticWorkflowModel(CoupledDEVS):
    """
    Agentic workflow as a coupled DEVS model.

    Components:
        - One OrchestratorStep (hub)
        - One WorkflowStep per available primitive (spokes), all in waiting state

    Coupling:
        orchestrator.out_activate[prim] → step[prim].in_activate
        step[prim].out_done             → orchestrator.in_step_done

    The orchestrator starts active (ta=0) and makes the first decision
    immediately. Each decision activates one waiting primitive step.
    When that step completes, it fires back to the orchestrator's
    in_step_done port, triggering the next decision.
    """

    def __init__(
        self,
        config: dict,
        workflow_state: dict,
        tool_registry: Any,
        action_registry: Any,
        model: str = "default",
        temperature: float = 0.1,
        decision_callback=None,
    ):
        CoupledDEVS.__init__(self, config.get("name", "agentic_workflow"))
        self.config = config
        self.workflow_state = workflow_state

        available = config.get("available_primitives", [])

        # Orchestrator component
        self.orchestrator = OrchestratorStep(
            config=config,
            workflow_state_ref=workflow_state,
            model=model,
            temperature=temperature,
            decision_callback=decision_callback,
        )
        self.addSubModel(self.orchestrator)

        # One WorkflowStep per available primitive, all in waiting state
        self.step_components: dict[str, WorkflowStep] = {}
        for prim in available:
            # Placeholder step_cfg — will be replaced by orchestrator decision
            # via the step_cfg injection in extTransition
            step_cfg = {
                "name": f"_pending_{prim}",
                "primitive": prim,
                "params": {},
                "model": model,
                "temperature": temperature,
            }
            comp = _AgenticWorkflowStep(
                primitive=prim,
                workflow_state_ref=workflow_state,
                tool_registry=tool_registry,
                action_registry=action_registry,
                default_model=model,
                default_temperature=temperature,
            )
            self.addSubModel(comp)
            self.step_components[prim] = comp

            # Wire: orchestrator activates this primitive
            if prim in self.orchestrator.out_activate:
                self.connectPorts(
                    self.orchestrator.out_activate[prim],
                    comp.in_activate,
                )

            # Wire: primitive completion fires back to orchestrator
            self.connectPorts(
                comp.out_done,
                self.orchestrator.in_step_done,
            )


# ─── Agentic-aware WorkflowStep ───────────────────────────────────────────────

class _AgenticWorkflowStep(AtomicDEVS):
    """
    WorkflowStep variant for agentic mode.

    The key difference from WorkflowStep: the step_cfg (step name,
    params) is not known at construction time — it arrives via the
    activate port payload from the OrchestratorStep. On activation,
    the step reads the step_cfg from the payload and configures itself
    before executing.

    All other DEVS semantics are identical to WorkflowStep.
    """

    DEFAULT_SLA_SECONDS = 600  # 10 minutes — agentic steps can be long (challenge prompts esp.)

    def __init__(
        self,
        primitive: str,
        workflow_state_ref: dict,
        tool_registry: Any,
        action_registry: Any,
        default_model: str,
        default_temperature: float,
    ):
        AtomicDEVS.__init__(self, f"step_{primitive}")

        self.primitive = primitive
        self.workflow_state_ref = workflow_state_ref
        self.tool_registry = tool_registry
        self.action_registry = action_registry
        self.default_model = default_model
        self.default_temperature = default_temperature
        self._simulator = None

        self.in_activate = self.addInPort("activate")
        self.in_resume = self.addInPort("resume")
        self.out_done = self.addOutPort("done")

        # Start waiting — orchestrator will activate us when needed
        self.state = StepState(
            phase="waiting",
            step_name=f"_pending_{primitive}",
            primitive=primitive,
            params={},
            model_name=default_model,
            temperature=default_temperature,
        )

    def timeAdvance(self) -> float:
        phase = self.state.phase
        if phase in ("active", "done"):
            return 0.0
        return INFINITY

    def intTransition(self):
        s = self.state
        new = StepState(**vars(s))

        if s.phase == "active":
            new.phase = "running"
            t0 = time.time()
            self._run_step_async(new, t0)
            return new

        if s.phase == "done":
            new.phase = "waiting"  # Reset to waiting for next invocation
            return new

        return s

    def extTransition(self, inputs: dict):
        s = self.state
        new = StepState(**vars(s))

        if self.in_activate in inputs:
            payload = inputs[self.in_activate][0]
            if s.phase == "waiting":
                # Payload from orchestrator contains the step_cfg
                if isinstance(payload, dict) and "primitive" in payload:
                    new.step_name = payload.get("name", f"{self.primitive}_step")
                    new.primitive = payload.get("primitive", self.primitive)
                    new.params = payload.get("params", {})
                    new.model_name = payload.get("model", self.default_model)
                    new.temperature = payload.get("temperature", self.default_temperature)
                    # Store sla_seconds in params so watchdog can read it
                    new.params = dict(new.params)
                    new.params["sla_seconds"] = payload.get("sla_seconds", self.DEFAULT_SLA_SECONDS)
                new.phase = "active"
            return new

        if self.in_resume in inputs:
            result = inputs[self.in_resume][0]
            if s.phase == "running":
                new.phase = "done"
                new.output = result.get("output")
                new.interrupt = result.get("interrupt")
                new.elapsed_ms = result.get("elapsed_ms", 0.0)
            return new

        return s

    def outputFnc(self) -> dict:
        s = self.state
        if s.phase == "done":
            return {self.out_done: {
                "step_name": s.step_name,
                "output": s.output,
                "interrupt": s.interrupt,
                "elapsed_ms": s.elapsed_ms,
            }}
        return {}

    def _run_step_async(self, step_state: StepState, t0: float):
        done_event = threading.Event()

        def _run():
            try:
                output, interrupt = self._execute_step(step_state)
                elapsed_ms = (time.time() - t0) * 1000
                # Embed primitive type and step_name in the output dict so that
                # _step_log_callback (which receives s.output) can read them.
                # get_step_output() returns the inner primitive output which has
                # no "primitive" key — the coordinator needs it for ledger recording.
                if isinstance(output, dict) and "primitive" not in output:
                    output = {**output, "primitive": step_state.primitive,
                              "step_name": step_state.step_name}
                result = {"output": output, "interrupt": interrupt, "elapsed_ms": elapsed_ms}
            except Exception as e:
                result = {
                    "output": {"error": str(e), "step": step_state.step_name},
                    "interrupt": None,
                    "elapsed_ms": (time.time() - t0) * 1000,
                }
            finally:
                done_event.set()
            if self._simulator:
                self._simulator.inject(
                    model=self, port=self.in_resume, value=result, sim_time=None
                )

        def _watchdog():
            sla = self.state.params.get("sla_seconds", self.DEFAULT_SLA_SECONDS)
            if not done_event.wait(timeout=sla):
                if self._simulator:
                    self._simulator.inject(
                        model=self,
                        port=self.in_resume,
                        value={
                            "output": {
                                "error": f"Step '{step_state.step_name}' timed out",
                                "timeout": True,
                            },
                            "interrupt": None,
                            "elapsed_ms": (time.time() - t0) * 1000,
                        },
                        sim_time=None,
                    )

        threading.Thread(target=_run, daemon=True).start()
        threading.Thread(target=_watchdog, daemon=True).start()

    def _execute_step(self, step_state: StepState) -> tuple[dict, StepInterrupt | None]:
        from cognitive_core.engine.nodes import create_node, create_retrieve_node, create_govern_node

        state = self.workflow_state_ref

        if step_state.primitive == "retrieve":
            node_fn = create_retrieve_node(
                step_name=step_state.step_name,
                params=step_state.params,
                tool_registry=self.tool_registry,
                model=step_state.model_name,
                temperature=step_state.temperature,
            )
        elif step_state.primitive == "govern":
            node_fn = create_govern_node(
                step_name=step_state.step_name,
                params=step_state.params,
                action_registry=self.action_registry,
                model=step_state.model_name,
                temperature=step_state.temperature,
                dry_run=True,
            )
        else:
            node_fn = create_node(
                step_name=step_state.step_name,
                primitive_name=step_state.primitive,
                params=step_state.params,
                model=step_state.model_name,
                temperature=step_state.temperature,
            )

        delta = node_fn(state)
        for key, value in delta.items():
            if key in ("steps", "routing_log") and isinstance(value, list):
                state[key] = state.get(key, []) + value
            elif key == "loop_counts" and isinstance(value, dict):
                state[key] = {**state.get(key, {}), **value}
            else:
                state[key] = value

        step_output = get_step_output(state, step_state.step_name)

        interrupt = None
        requests = step_output.get("resource_requests", []) if step_output else []
        blocking = [r for r in requests if r.get("blocking", True)]
        if blocking:
            interrupt = StepInterrupt(
                reason=f"Step '{step_state.step_name}' produced {len(blocking)} "
                       "blocking resource request(s)",
                suspended_at_step=step_state.step_name,
                resource_requests=blocking,
            )

        return step_output, interrupt


# ─── Agentic Workflow Executor ────────────────────────────────────────────────

class AgenticWorkflowExecutor:
    """
    Executes an agentic workflow using DEVS simulation.

    The orchestrator selects the trajectory autonomously.
    The substrate enforces governance on that trajectory identically
    to workflow execution mode.

    The step_callback is called after each primitive step completes —
    identical to WorkflowExecutor — giving the coordinator the same
    hook for epistemic state computation, ledger recording, and
    governance enforcement.
    """

    def __init__(
        self,
        config: dict,
        tool_registry: Any = None,
        action_registry: Any = None,
        model: str = "default",
        temperature: float = 0.1,
    ):
        self.config = config
        self.tool_registry = tool_registry
        self.action_registry = action_registry
        self.model = model
        self.temperature = temperature

    def execute(
        self,
        workflow_input: dict,
        step_callback: StepCallback = no_interrupt_callback,
        decision_callback=None,
    ) -> StepResult:
        workflow_state: WorkflowState = {
            "input": workflow_input,
            "steps": [],
            "current_step": "",
            "metadata": {
                "use_case": self.config.get("name", ""),
                "description": self.config.get("description", ""),
                "mode": "agentic",
            },
            "loop_counts": {},
            "routing_log": [],
        }
        return self._run(workflow_state, step_callback, decision_callback)

    def _run(
        self,
        workflow_state: dict,
        step_callback: StepCallback,
        decision_callback=None,
    ) -> StepResult:
        interrupt_container: list[StepInterrupt] = []

        agentic_model = AgenticWorkflowModel(
            config=self.config,
            workflow_state=workflow_state,
            tool_registry=self.tool_registry,
            action_registry=self.action_registry,
            model=self.model,
            temperature=self.temperature,
            decision_callback=decision_callback,
        )

        sim = Simulator(agentic_model)
        sim.setRealTime(True)
        sim._inject_wait_timeout = 1200  # 20 min — LLM steps (esp. challenge) can be slow

        # Wire simulator back-reference into all components
        for comp in agentic_model.step_components.values():
            comp._simulator = sim
        agentic_model.orchestrator._simulator = sim

        def _on_transition(model, transition_type: str, sim_time: float):
            # Fire step callback only for primitive step completions
            if not isinstance(model, _AgenticWorkflowStep):
                return

            s = model.state
            if s.phase == "done" and s.output is not None:
                interrupt = step_callback(s.step_name, s.output, workflow_state)
                if interrupt is None and s.interrupt is not None:
                    interrupt = s.interrupt
                if interrupt is not None:
                    interrupt.state_at_interrupt = workflow_state
                    interrupt_container.append(interrupt)
                    sim.stop()

        sim.setStepCallback(_on_transition)

        def _is_done(t, original_model):
            if interrupt_container:
                return True
            steps_done = len(workflow_state.get("steps", []))
            # Orchestrator finished is authoritative — but only after at least
            # one step has completed. Avoids false-positive at t=0 when all
            # step components are in initial "waiting" state before any work.
            if agentic_model.orchestrator.state.finished and steps_done > 0:
                return True
            # Normal completion: orchestrator done + all step components passive
            return (
                agentic_model.orchestrator.state.finished
                and all(
                    comp.state.phase == "waiting"
                    for comp in agentic_model.step_components.values()
                )
            )

        sim.setTerminationCondition(_is_done)

        sim_thread = threading.Thread(target=sim.simulate, daemon=True)
        sim_thread.start()
        sim_thread.join(timeout=86400)

        if interrupt_container:
            return StepResult(
                completed=False,
                final_state=workflow_state,
                interrupt=interrupt_container[0],
            )

        return StepResult(
            completed=True,
            final_state=workflow_state,
            interrupt=None,
        )


# ─── Backward-compatible entry point ─────────────────────────────────────────

def run_agentic_workflow_devs(
    config: dict,
    workflow_input: dict,
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: Any = None,
    action_registry: Any = None,
    step_callback: StepCallback = no_interrupt_callback,
    decision_callback=None,
) -> StepResult:
    """
    Drop-in replacement for run_agentic_workflow() from agentic.py.
    Returns a StepResult instead of raw LangGraph state.
    """
    executor = AgenticWorkflowExecutor(
        config, tool_registry, action_registry, model, temperature
    )
    return executor.execute(workflow_input, step_callback, decision_callback)
