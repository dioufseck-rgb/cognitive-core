"""
Cognitive Core — DEVS Workflow Executor

Replaces LangGraph as the workflow execution engine.

Each workflow step is an AtomicDEVS component. The workflow instance
is a CoupledDEVS model. The coordinator is the simulation kernel.

Key properties:

    Suspension is ta(s) = INFINITY — a well-defined passive state.
    No subgraph recompilation. No special-case code for mid-graph resume.

    Resume is extTransition — an external event (human decision,
    specialist result) fires the passive model back into active state.

    Parallel handlers are two AtomicDEVS components both passive.
    The coupled model resumes the primary when both return results.
    This is native DEVS composition, not engineered special cases.

Backward compatibility:
    WorkflowExecutor.execute() and WorkflowExecutor.resume() replace
    stepper.step_execute() and stepper.step_resume() respectively.
    Return types are compatible with the existing StepResult contract.

Usage:
    executor = WorkflowExecutor(config, tool_registry, action_registry)
    result = executor.execute(workflow_input, step_callback)

    # On suspension:
    result = executor.resume(state_snapshot, resume_step, injected_data, step_callback)
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    from pypdevs.DEVS import AtomicDEVS, CoupledDEVS
    from pypdevs.simulator import Simulator
    from pypdevs.infinity import INFINITY
except ImportError:
    from cognitive_core.engine.devs import AtomicDEVS, CoupledDEVS, Simulator, INFINITY

from cognitive_core.engine.state import WorkflowState, get_step_output


# ─── Result types (compatible with stepper.StepResult) ───────────────────────

@dataclass
class StepInterrupt:
    reason: str
    suspended_at_step: str
    state_at_interrupt: dict[str, Any] = field(default_factory=dict)
    resource_requests: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    completed: bool
    final_state: dict[str, Any]
    interrupt: StepInterrupt | None = None


StepCallback = Callable[[str, dict[str, Any], dict[str, Any]], StepInterrupt | None]


def no_interrupt_callback(step_name, step_output, state):
    return None


# ─── Workflow Step State ──────────────────────────────────────────────────────

@dataclass
class StepState:
    """State for a single workflow step AtomicDEVS component."""
    phase: str          # 'waiting' | 'ready' | 'running' | 'done' | 'passive'
    step_name: str
    primitive: str
    params: dict
    model_name: str
    temperature: float
    output: dict | None = None
    interrupt: StepInterrupt | None = None
    elapsed_ms: float = 0.0


# ─── Workflow Step AtomicDEVS ─────────────────────────────────────────────────

class WorkflowStep(AtomicDEVS):
    """
    A single cognitive primitive step as an atomic DEVS component.

    States:
        waiting  — ta = INFINITY. Waiting for activation signal from prior step.
        active   — ta = 0. Ready to execute immediately.
        running  — ta = INFINITY. LLM call in progress (async).
        done     — ta = 0. Output ready, will fire to next step.
        passive  — ta = INFINITY. Terminal state after completion.

    The LLM call runs in a background thread. When it completes,
    it calls sim.inject() to fire an external event that moves
    the model from 'running' to 'done'.
    """

    def __init__(
        self,
        step_cfg: dict,
        workflow_state_ref: dict,
        tool_registry: Any,
        action_registry: Any,
        default_model: str,
        default_temperature: float,
    ):
        AtomicDEVS.__init__(self, step_cfg["name"])

        self.step_cfg = step_cfg
        self.workflow_state_ref = workflow_state_ref
        self.tool_registry = tool_registry
        self.action_registry = action_registry

        self.in_activate = self.addInPort("activate")   # receives activation from prior step
        self.in_resume = self.addInPort("resume")        # receives injected data on resume
        self.out_done = self.addOutPort("done")          # fires to next step on completion
        self.out_interrupt = self.addOutPort("interrupt") # fires to workflow on suspension
        self._simulator = None                           # set by Simulator after construction

        self.state = StepState(
            phase="waiting",
            step_name=step_cfg["name"],
            primitive=step_cfg["primitive"],
            params=step_cfg.get("params", {}),
            model_name=step_cfg.get("model", default_model),
            temperature=step_cfg.get("temperature", default_temperature),
        )
        self._last_transition_time = 0.0

    def timeAdvance(self) -> float:
        phase = self.state.phase
        if phase == "active":
            return 0.0          # fire immediately — kick off the LLM call
        if phase == "done":
            return 0.0          # fire immediately — pass output to next step
        return INFINITY         # waiting, running, passive — wait for external event

    def intTransition(self):
        """Internal transitions: active→running, done→passive."""
        s = self.state
        if s.phase == "active":
            # Kick off the LLM call in a background thread
            new_state = StepState(**vars(s))
            new_state.phase = "running"
            t0 = time.time()
            self._run_step_async(new_state, t0)
            return new_state

        if s.phase == "done":
            new_state = StepState(**vars(s))
            new_state.phase = "passive"
            return new_state

        return s

    def extTransition(self, inputs: dict):
        """External transitions: waiting→active (activation), running→done (LLM result)."""
        s = self.state
        new_state = StepState(**vars(s))

        if self.in_activate in inputs:
            # Activation from prior step — move to active
            if s.phase == "waiting":
                new_state.phase = "active"
            return new_state

        if self.in_resume in inputs:
            # Injected result (LLM done, or human decision for resume)
            result = inputs[self.in_resume][0]
            if s.phase == "running":
                new_state.phase = "done"
                new_state.output = result.get("output")
                new_state.interrupt = result.get("interrupt")
                new_state.elapsed_ms = result.get("elapsed_ms", 0.0)
            elif s.phase == "passive" and "injection" in result:
                # Re-activation for resume path
                new_state.phase = "active"
                # Merge injected data into workflow state
                injection = result["injection"]
                self.workflow_state_ref.setdefault("delegation_results", {}).update(injection)
            return new_state

        return s

    def outputFnc(self) -> dict:
        """Output fires before intTransition when done."""
        s = self.state
        if s.phase == "done":
            payload = {
                "step_name": s.step_name,
                "output": s.output,
                "interrupt": s.interrupt,
                "elapsed_ms": s.elapsed_ms,
            }
            return {self.out_done: payload}
        return {}

    # ── Background LLM execution ──────────────────────────────────────────────

    # Default step SLA: 300 seconds (5 minutes)
    # Override per step via step_cfg["sla_seconds"]
    DEFAULT_SLA_SECONDS = 300

    def _run_step_async(self, step_state: StepState, t0: float):
        """
        Run the LLM call in a background thread, inject result when done.

        Watchdog: if the LLM call does not complete within the step SLA,
        a timeout error is injected as the result. This prevents a hung
        thread from leaving the workflow passive indefinitely with no
        recovery path.

        The SLA is configured per step via step_cfg["sla_seconds"].
        Default is 300s (5 minutes) — generous for real LLM calls but
        finite to ensure eventual failure rather than silent hang.
        """
        sla_seconds = self.step_cfg.get("sla_seconds", self.DEFAULT_SLA_SECONDS)
        done_event = threading.Event()

        def _run():
            try:
                output, interrupt = self._execute_step(step_state)
                elapsed_ms = (time.time() - t0) * 1000
                result = {
                    "output": output,
                    "interrupt": interrupt,
                    "elapsed_ms": elapsed_ms,
                }
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
                    model=self,
                    port=self.in_resume,
                    value=result,
                    sim_time=None,  # inject at current sim time
                )

        def _watchdog():
            completed = done_event.wait(timeout=sla_seconds)
            if not completed and self._simulator:
                elapsed_ms = (time.time() - t0) * 1000
                timeout_result = {
                    "output": {
                        "error": f"Step '{step_state.step_name}' timed out "
                                 f"after {sla_seconds}s SLA",
                        "step": step_state.step_name,
                        "timeout": True,
                    },
                    "interrupt": None,
                    "elapsed_ms": elapsed_ms,
                }
                self._simulator.inject(
                    model=self,
                    port=self.in_resume,
                    value=timeout_result,
                    sim_time=None,  # inject at current sim time
                )

        threading.Thread(target=_run, daemon=True).start()
        threading.Thread(target=_watchdog, daemon=True).start()

    def _execute_step(self, step_state: StepState) -> tuple[dict, StepInterrupt | None]:
        """Execute the cognitive primitive. Returns (output, interrupt_or_None)."""
        from cognitive_core.engine.nodes import create_node, create_retrieve_node, create_govern_node
        from cognitive_core.engine.state import WorkflowState

        state = self.workflow_state_ref
        step_cfg = self.step_cfg

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
                dry_run=step_cfg.get("dry_run", True),
            )
        else:
            node_fn = create_node(
                step_name=step_state.step_name,
                primitive_name=step_state.primitive,
                params=step_state.params,
                model=step_state.model_name,
                temperature=step_state.temperature,
            )

        # Execute node with current workflow state
        delta = node_fn(state)

        # Merge delta into workflow state
        for key, value in delta.items():
            if key in ("steps", "routing_log") and isinstance(value, list):
                state[key] = state.get(key, []) + value
            elif key == "loop_counts" and isinstance(value, dict):
                state[key] = {**state.get(key, {}), **value}
            else:
                state[key] = value

        # Extract this step's output
        step_output = get_step_output(state, step_state.step_name)

        # Check for resource requests (suspension trigger)
        interrupt = None
        requests = step_output.get("resource_requests", [])
        blocking = [r for r in requests if r.get("blocking", True)]
        if blocking:
            interrupt = StepInterrupt(
                reason=f"Step '{step_state.step_name}' produced {len(blocking)} "
                       f"blocking resource request(s)",
                suspended_at_step=step_state.step_name,
                resource_requests=blocking,
            )

        return step_output, interrupt


# ─── Workflow CoupledDEVS ─────────────────────────────────────────────────────

class WorkflowModel(CoupledDEVS):
    """
    A complete workflow as a coupled DEVS model.

    Builds from the merged workflow+domain config.
    Each step is a WorkflowStep atomic component.
    Steps are connected sequentially; conditional routing is resolved
    at each step's extTransition based on the step's output.

    The first step is initialized in 'active' state.
    All subsequent steps start in 'waiting' state.
    """

    def __init__(
        self,
        config: dict,
        workflow_state: dict,
        tool_registry: Any,
        action_registry: Any,
        model: str = "default",
        temperature: float = 0.1,
        resume_step: str | None = None,
    ):
        CoupledDEVS.__init__(self, config.get("name", "workflow"))
        self.config = config
        self.workflow_state = workflow_state

        steps = config.get("steps", [])
        self.step_components: dict[str, WorkflowStep] = {}

        for i, step_cfg in enumerate(steps):
            comp = WorkflowStep(
                step_cfg=step_cfg,
                workflow_state_ref=workflow_state,
                tool_registry=tool_registry,
                action_registry=action_registry,
                default_model=model,
                default_temperature=temperature,
            )

            # If resuming, activate the resume step; otherwise activate step 0
            if resume_step:
                if step_cfg["name"] == resume_step:
                    comp.state.phase = "active"
            else:
                if i == 0:
                    comp.state.phase = "active"

            self.addSubModel(comp)
            self.step_components[step_cfg["name"]] = comp

        # Connect steps sequentially based on transitions
        self._wire_transitions(steps)

    def _wire_transitions(self, steps: list[dict]):
        """
        Wire step outputs to next step inputs based on transition config.
        For conditional transitions, the WorkflowRouter handles routing.
        """
        step_names = [s["name"] for s in steps]

        for i, step_cfg in enumerate(steps):
            src_comp = self.step_components[step_cfg["name"]]
            transitions = step_cfg.get("transitions", [])
            default_next = step_names[i + 1] if i + 1 < len(step_names) else None

            # Determine next step(s) to wire to
            if not transitions:
                if default_next:
                    dst_comp = self.step_components[default_next]
                    self.connectPorts(src_comp.out_done, dst_comp.in_activate)
            else:
                # For simple default transitions, wire directly
                # Conditional transitions are resolved at runtime in the router
                for t in transitions:
                    if "default" in t and t["default"] != "__end__":
                        target = t["default"]
                        if target in self.step_components:
                            dst_comp = self.step_components[target]
                            self.connectPorts(src_comp.out_done, dst_comp.in_activate)
                    elif "when" not in t and "default" not in t and default_next:
                        dst_comp = self.step_components[default_next]
                        self.connectPorts(src_comp.out_done, dst_comp.in_activate)


# ─── Workflow Executor ────────────────────────────────────────────────────────

class WorkflowExecutor:
    """
    Executes a workflow using DEVS simulation.

    Replaces stepper.step_execute() and stepper.step_resume().

    The executor builds a WorkflowModel (CoupledDEVS), runs the
    Simulator, and intercepts step completions via the step callback.

    Suspension:
        When a step produces a StepInterrupt, the simulator is stopped.
        The workflow state is saved and returned as a StepResult with
        completed=False.

    Resume:
        WorkflowExecutor.resume() builds a new WorkflowModel starting
        from the resume step, with prior state injected. No subgraph
        recompilation — just a new simulation from that step forward.
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
    ) -> StepResult:
        """
        Execute a workflow from the beginning.

        Args:
            workflow_input: Case input data
            step_callback: Called after each step. Return StepInterrupt to pause.

        Returns:
            StepResult(completed=True) if finished,
            StepResult(completed=False, interrupt=...) if paused.
        """
        workflow_state: WorkflowState = {
            "input": workflow_input,
            "steps": [],
            "current_step": "",
            "metadata": {
                "use_case": self.config.get("name", ""),
                "description": self.config.get("description", ""),
            },
            "loop_counts": {},
            "routing_log": [],
        }

        return self._run(workflow_state, step_callback)

    def resume(
        self,
        state_snapshot: dict,
        resume_step: str,
        injected_data: dict | None = None,
        step_callback: StepCallback = no_interrupt_callback,
    ) -> StepResult:
        """
        Resume a workflow from a saved state snapshot.

        No subgraph recompilation. Builds a WorkflowModel starting at
        resume_step, injects prior state and delegation results, runs forward.

        Args:
            state_snapshot: Saved workflow state from suspension
            resume_step: Step name to resume at
            injected_data: Delegation results or human decision to inject
            step_callback: Called after each step

        Returns:
            StepResult — same semantics as execute()
        """
        import copy
        workflow_state = copy.deepcopy(state_snapshot)
        if injected_data:
            workflow_state.setdefault("delegation_results", {}).update(injected_data)

        return self._run(workflow_state, step_callback, resume_step=resume_step)

    def _run(
        self,
        workflow_state: dict,
        step_callback: StepCallback,
        resume_step: str | None = None,
    ) -> StepResult:
        """Build the DEVS model and run simulation."""
        interrupt_container: list[StepInterrupt] = []
        completion_event = threading.Event()

        workflow_model = WorkflowModel(
            config=self.config,
            workflow_state=workflow_state,
            tool_registry=self.tool_registry,
            action_registry=self.action_registry,
            model=self.model,
            temperature=self.temperature,
            resume_step=resume_step,
        )

        sim = Simulator(workflow_model)
        sim.setRealTime(True)
        sim._inject_wait_timeout = 1200  # 20 min — LLM steps can be slow

        # Wire simulator reference into all WorkflowStep components
        # so background threads can call sim.inject() to return results.
        for comp in workflow_model.step_components.values():
            comp._simulator = sim

        def _on_transition(model: WorkflowStep, transition_type: str, sim_time: float):
            if not isinstance(model, WorkflowStep):
                return

            s = model.state

            # Step just completed (phase moved to done via extTransition)
            if s.phase == "done" and s.output is not None:
                step_name = s.step_name
                step_output = s.output

                # Call the step callback
                interrupt = step_callback(step_name, step_output, workflow_state)
                if interrupt is None and s.interrupt is not None:
                    interrupt = s.interrupt

                if interrupt is not None:
                    interrupt.state_at_interrupt = workflow_state
                    interrupt_container.append(interrupt)
                    sim.stop()
                    return

        sim.setStepCallback(_on_transition)

        def _is_done(t, original_model):
            """Terminate when interrupted or all steps passive (workflow complete)."""
            if interrupt_container:
                return True
            return all(
                comp.state.phase == "passive"
                for comp in workflow_model.step_components.values()
            )

        sim.setTerminationCondition(_is_done)

        # Run simulation in a thread (real-time sim blocks)
        sim_thread = threading.Thread(
            target=sim.simulate,
            daemon=True,
        )
        sim_thread.start()
        sim_thread.join(timeout=86400)  # 24h max — coordinator handles SLA

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


# ─── Convenience aliases (stepper compatibility) ──────────────────────────────

def step_execute(
    config: dict,
    workflow_input: dict,
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: Any = None,
    action_registry: Any = None,
    step_callback: StepCallback = no_interrupt_callback,
) -> StepResult:
    """Drop-in replacement for stepper.step_execute()."""
    executor = WorkflowExecutor(config, tool_registry, action_registry, model, temperature)
    return executor.execute(workflow_input, step_callback)


def step_resume(
    config: dict,
    state_snapshot: dict,
    resume_step: str,
    model: str = "default",
    temperature: float = 0.1,
    tool_registry: Any = None,
    action_registry: Any = None,
    step_callback: StepCallback = no_interrupt_callback,
) -> StepResult:
    """Drop-in replacement for stepper.step_resume()."""
    executor = WorkflowExecutor(config, tool_registry, action_registry, model, temperature)
    return executor.resume(state_snapshot, resume_step, step_callback=step_callback)


# ─── Stepper-compatible callbacks ────────────────────────────────────────────
# These match the stepper.py API exactly so runtime.py needs no other changes.

def resource_request_callback(
    step_name: str,
    step_output: dict,
    state: dict,
) -> StepInterrupt | None:
    """
    Callback that inspects step output for blocking ResourceRequests.
    Port of stepper.resource_request_callback.
    """
    requests = step_output.get("resource_requests", [])
    blocking = [r for r in requests if r.get("blocking", True)]

    if not blocking:
        return None

    # Filter already-fulfilled needs
    existing = set(state.get("input", {}).get("delegation", {}).keys())
    if existing:
        filtered = [r for r in blocking if r.get("need", "") not in existing]
        blocking = filtered

    if not blocking:
        return None

    return StepInterrupt(
        reason=f"Step '{step_name}' produced {len(blocking)} blocking resource request(s)",
        suspended_at_step=step_name,
        resource_requests=blocking,
    )


def combined_callback(*callbacks: StepCallback) -> StepCallback:
    """
    Combine multiple step callbacks. Returns the first interrupt produced.
    Port of stepper.combined_callback.
    """
    def _combined(step_name, step_output, state):
        for cb in callbacks:
            result = cb(step_name, step_output, state)
            if result is not None:
                return result
        return None
    return _combined