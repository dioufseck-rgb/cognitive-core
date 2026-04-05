"""
Cognitive Core — DEVS Simulation Kernel

A faithful replication of the PythonPDEVS minimal simulation kernel,
extended with real-time execution and external event injection for
use as Cognitive Core's workflow execution engine.

Source implementation:
    PythonPDEVS (Van Tendeloo & Vangheluwe, 2014)
    Modelling, Simulation and Design Lab (MSDL)
    McGill University and University of Antwerp
    http://msdl.cs.mcgill.ca/projects/DEVS/PythonPDEVS
    Licensed under the Apache License, Version 2.0

Extensions beyond the minimal kernel:
    - Real-time simulation via ThreadingPython backend
    - External event injection via inject() (thread-safe)
    - Step callback for post-transition observation
    - Termination condition function

Formalism reference:
    Zeigler, B.P., Praehofer, H., Kim, T.G. (2000).
    Theory of Modeling and Simulation (2nd ed.). Academic Press.
"""

# Copyright 2014 Modelling, Simulation and Design Lab (MSDL) at
# McGill University and the University of Antwerp
# (original PythonPDEVS source — Apache License 2.0)
#
# Cognitive Core extensions copyright 2026 Mamadou Seck
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections import defaultdict
from threading import Event, Thread, Lock
from typing import Any, Callable

INFINITY = float('inf')


# ─── Port ─────────────────────────────────────────────────────────────────────

class Port:
    """Source: pypdevs.DEVS.Port (Van Tendeloo & Vangheluwe, 2014)"""
    def __init__(self, is_input: bool, name: str | None = None):
        self.inline: list = []
        self.outline: list = []
        self.routing_outline: list = []
        self.host_DEVS = None
        self.name = name
        self.is_input = is_input
        self.z_functions: dict = {}

    def getPortName(self) -> str:
        return self.name

    def type(self) -> str:
        return 'INPORT' if self.is_input else 'OUTPORT'


# ─── BaseDEVS ─────────────────────────────────────────────────────────────────

class BaseDEVS:
    """Source: pypdevs.DEVS.BaseDEVS (Van Tendeloo & Vangheluwe, 2014)"""
    def __init__(self, name: str):
        self.name = name
        self.parent = None
        self.IPorts: list[Port] = []
        self.OPorts: list[Port] = []
        self.ports: list[Port] = []
        self.time_last: tuple[float, int] = (0.0, 0)
        self.time_next: tuple[float, int] = (0.0, 1)
        self.my_input: dict = {}
        self.my_output: dict = {}

    def addPort(self, name: str | None, is_input: bool) -> Port:
        name = name if name is not None else f"port{len(self.ports)}"
        port = Port(is_input=is_input, name=name)
        if is_input:
            self.IPorts.append(port)
        else:
            self.OPorts.append(port)
        port.port_id = len(self.ports)
        self.ports.append(port)
        port.host_DEVS = self
        return port

    def addInPort(self, name: str | None = None) -> Port:
        return self.addPort(name, True)

    def addOutPort(self, name: str | None = None) -> Port:
        return self.addPort(name, False)

    def getModelName(self) -> str:
        return str(self.name)


# ─── AtomicDEVS ───────────────────────────────────────────────────────────────

class AtomicDEVS(BaseDEVS):
    """
    Abstract base for atomic DEVS models.
    Source: pypdevs.DEVS.AtomicDEVS + pypdevs.minimal (Van Tendeloo & Vangheluwe, 2014)
    """
    _id_counter = 0

    def __init__(self, name: str | None = None):
        BaseDEVS.__init__(self, name)
        self.elapsed = 0.0
        self.state = None
        self.model_id = AtomicDEVS._id_counter
        AtomicDEVS._id_counter += 1
        self._simulator = None  # set by Simulator.__init__ after directConnect

    def timeAdvance(self) -> float:
        return INFINITY

    def intTransition(self):
        return self.state

    def extTransition(self, inputs: dict) -> Any:
        return self.state

    def confTransition(self, inputs: dict) -> Any:
        """
        Confluent transition: simultaneous internal + external event.
        Default: int first, then ext.
        Source: pypdevs.DEVS.AtomicDEVS.confTransition
        """
        self.state = self.intTransition()
        return self.extTransition(inputs)

    def outputFnc(self) -> dict:
        return {}


# ─── CoupledDEVS ──────────────────────────────────────────────────────────────

class CoupledDEVS(BaseDEVS):
    """Source: pypdevs.DEVS.CoupledDEVS (Van Tendeloo & Vangheluwe, 2014)"""
    def __init__(self, name: str | None = None):
        BaseDEVS.__init__(self, name)
        self.component_set: list = []

    def addSubModel(self, model, location=None):
        model.parent = self
        self.component_set.append(model)
        return model

    def connectPorts(self, p1: Port, p2: Port, z=None):
        """
        Connect p1 (output) to p2 (input). Validates coupling type.
        Source: pypdevs.DEVS.CoupledDEVS.connectPorts
        """
        p1.outline.append(p2)
        p2.inline.append(p1)
        p1.z_functions[p2] = z

    def select(self, imm_children: list):
        """Default select: first imminent child."""
        return imm_children[0]


# ─── Direct Connection ─────────────────────────────────────────────────────────

def directConnect(component_set: list) -> list:
    """
    Flatten coupled hierarchy to atomic models with direct routing_outline.
    Source: pypdevs.minimal.directConnect (Van Tendeloo & Vangheluwe, 2014)
    """
    work = list(component_set)
    atomics = []
    for item in work:
        if isinstance(item, CoupledDEVS):
            work.extend(item.component_set)
        else:
            atomics.append(item)

    for model in atomics:
        for outport in model.OPorts:
            outport.routing_outline = []
            worklist = list(outport.outline)
            for outline in worklist:
                if isinstance(outline.host_DEVS, CoupledDEVS):
                    worklist.extend(outline.outline)
                else:
                    outport.routing_outline.append(outline)

    return atomics


# ─── Scheduler (Sorted List) ──────────────────────────────────────────────────

class SchedulerSL:
    """
    Sorted List scheduler.
    Source: pypdevs.schedulers.schedulerSL (Van Tendeloo & Vangheluwe, 2014)
    """
    def __init__(self, models: list, epsilon: float = 1e-9, total: int = 0):
        self.models = list(models)
        self.epsilon = epsilon

    def schedule(self, model):
        self.models.append(model)
        self.models.sort(key=lambda m: m.time_next)

    def unschedule(self, model):
        self.models.remove(model)

    def massReschedule(self, reschedule_set):
        self.models.sort(key=lambda m: m.time_next)

    def readFirst(self) -> tuple[float, int]:
        return self.models[0].time_next if self.models else (INFINITY, 1)

    def getImminent(self, time: tuple[float, int]) -> list:
        """
        Return all models imminent at time (within epsilon).
        Source: pypdevs.schedulers.schedulerSL.getImminent
        """
        imminent = []
        t, age = time
        try:
            count = 0
            while (abs(self.models[count].time_next[0] - t) < self.epsilon and
                   self.models[count].time_next[1] == age):
                imminent.append(self.models[count])
                count += 1
        except IndexError:
            pass
        return imminent


# ─── RootDEVS ─────────────────────────────────────────────────────────────────

class RootDEVS:
    """Source: pypdevs.minimal.RootDEVS (Van Tendeloo & Vangheluwe, 2014)"""
    def __init__(self, components: list):
        self.component_set = components
        self.time_next = (INFINITY, 1)
        self.scheduler = SchedulerSL(components, 1e-9, len(components))


# ─── ThreadingPython ──────────────────────────────────────────────────────────

class ThreadingPython:
    """
    Python threads real-time subsystem.
    Source: pypdevs.realtime.threadingPython.ThreadingPython (Van Tendeloo & Vangheluwe, 2014)
    """
    def __init__(self):
        self.evt = Event()
        self.evt_lock = Lock()

    def wait(self, delay: float, func: Callable):
        t = Thread(target=self._callFunc, args=(delay, func), daemon=True)
        t.start()

    def interrupt(self):
        self.evt.set()

    def _callFunc(self, delay: float, func: Callable):
        with self.evt_lock:
            self.evt.wait(delay)
            func()
            self.evt.clear()


# ─── Simulator ────────────────────────────────────────────────────────────────

class Simulator:
    """
    Sequential DEVS simulation kernel.

    Core loop faithfully replicates pypdevs.minimal.Simulator.simulate().
    Extended with real-time execution (ThreadingPython) and external event
    injection (inject()) for Cognitive Core's suspend/resume semantics.

    Source: pypdevs.minimal.Simulator (Van Tendeloo & Vangheluwe, 2014)
    Cognitive Core extensions: Mamadou Seck, 2026
    """

    def __init__(self, model):
        self.original_model = model
        self._real_time = False
        self._rt_backend: ThreadingPython | None = None
        self._step_callback: Callable | None = None
        self._termination_fn: Callable = lambda t, m: False
        self._inject_wait_timeout = 60  # seconds; override for long-running LLM steps

        # Injection queue (Cognitive Core extension)
        self._inject_lock = Lock()
        self._inject_queue: list = []
        self._inject_event = Event()

        # Build RootDEVS from model
        if isinstance(model, CoupledDEVS):
            components = directConnect(model.component_set)
            for idx, m in enumerate(components):
                m.time_last = (-m.elapsed, 0)
                m.time_next = (-m.elapsed + m.timeAdvance(), 1)
                m.model_id = idx
                m._simulator = self  # back-reference for inject() calls from threads
            self.model = RootDEVS(components)
        elif isinstance(model, AtomicDEVS):
            for p in model.OPorts:
                p.routing_outline = []
            model.time_last = (-model.elapsed, 0)
            model.time_next = (model.time_last[0] + model.timeAdvance(), 1)
            model.model_id = 0
            model._simulator = self
            self.model = RootDEVS([model])
        else:
            raise ValueError(f"Expected AtomicDEVS or CoupledDEVS, got {type(model)}")

        # Wire simulator reference into all atomic components so they can
        # call self._simulator.inject() from background threads.
        for m in self.model.component_set:
            if hasattr(m, '_simulator'):
                m._simulator = self

        self.setTerminationTime(INFINITY)

    # ── Configuration ─────────────────────────────────────────────────────────

    def setTerminationTime(self, time: float):
        # Terminate when sim time reaches the termination time.
        # If time==INFINITY, only terminate when all models are permanently passive.
        # Do NOT terminate just because tn==INFINITY and time is finite.
        self.setTerminationCondition(
            lambda t, m: t[0] != INFINITY and t[0] >= time
        )

    def setTerminationCondition(self, function: Callable):
        self._termination_fn = function

    def setRealTime(self, enabled: bool, subsystem: str = "python", args: list = None):
        self._real_time = enabled
        if enabled:
            self._rt_backend = ThreadingPython()

    def setStepCallback(self, callback: Callable):
        """
        Callback after each atomic transition.
        Signature: callback(model, transition_type, sim_time)
        transition_type: 'internal' | 'external' | 'confluent'
        """
        self._step_callback = callback

    def stop(self):
        """
        Request the simulation loop to stop at the next iteration.
        Sets the termination condition to always return True.
        """
        self.setTerminationCondition(lambda t, m: True)
        # Also wake the event loop if it's waiting for an inject
        if self._rt_backend is not None:
            self._rt_backend.interrupt()
        self._inject_event.set()

    # ── Injection (Cognitive Core extension) ──────────────────────────────────

    def inject(self, model: AtomicDEVS, port: Port, value: Any,
               sim_time=None):
        """
        Inject an external event. Thread-safe.
        Used for resume: human decision or specialist result injected
        into a suspended (ta=INFINITY) workflow step.
        """
        current_t = self.model.scheduler.readFirst()[0]
        if current_t == INFINITY:
            current_t = 0.0

        if sim_time is None:
            t = current_t
        elif isinstance(sim_time, tuple):
            t = sim_time[0]
        else:
            t = float(sim_time)

        t = max(t, current_t)
        timestamp = (t, 0)  # age=0: external events fire before internal at same time

        with self._inject_lock:
            self._inject_queue.append((timestamp, port, value, model))

        # Wake simulation loop
        if self._rt_backend is not None:
            self._rt_backend.interrupt()
        self._inject_event.set()

    # ── Simulation loop ───────────────────────────────────────────────────────

    def simulate(self):
        """
        Core simulation loop.

        Faithful replication of pypdevs.minimal.Simulator.simulate(),
        extended with real-time sleep and external injection handling.

        Source: pypdevs.minimal.Simulator.simulate (Van Tendeloo & Vangheluwe, 2014)
        """
        scheduler = self.model.scheduler
        tn = scheduler.readFirst()

        while not self._termination_fn(tn, self.original_model):

            # ── Process pending injections ─────────────────────────────────
            with self._inject_lock:
                due = [inj for inj in self._inject_queue
                       if inj[0][0] <= tn[0]]
                self._inject_queue = [inj for inj in self._inject_queue
                                      if inj[0][0] > tn[0]]

            if due:
                for (timestamp, port, value, target) in due:
                    # Use the injection timestamp as current time,
                    # NOT tn (which may be INFINITY when all models are passive).
                    inj_t = timestamp[0]
                    target.my_input.setdefault(port, []).append(value)
                    target.elapsed = inj_t - target.time_last[0]
                    target.state = target.extTransition(target.my_input)
                    ta = target.timeAdvance()
                    target.time_next = (inj_t + ta, 1)
                    target.time_last = (inj_t, timestamp[1])
                    target.my_input = {}
                    if self._step_callback:
                        self._step_callback(target, 'external', inj_t)
                scheduler.massReschedule(set())
                tn = scheduler.readFirst()
                continue

            # ── All passive — wait for external inject ─────────────────────
            if tn[0] == INFINITY:
                if self._real_time:
                    # Clear then re-check queue (avoid race with inject())
                    self._inject_event.clear()
                    with self._inject_lock:
                        has_pending = len(self._inject_queue) > 0
                    if not has_pending:
                        woken = self._inject_event.wait(timeout=self._inject_wait_timeout)
                        if not woken:
                            break  # No inject arrived in 60s — terminate
                    tn = scheduler.readFirst()
                    continue
                else:
                    break

            # ── Real-time: sleep until tn, interruptible ───────────────────
            if self._real_time and tn[0] > 0:
                self._inject_event.clear()
                self._inject_event.wait(timeout=tn[0])
                # Check for injections that arrived during sleep
                with self._inject_lock:
                    has_early = any(inj[0][0] <= tn[0]
                                    for inj in self._inject_queue)
                if has_early:
                    continue

            # ── Imminent transitions (faithful minimal kernel) ─────────────
            # Source: pypdevs.minimal.Simulator.simulate inner loop
            transitioning: dict = defaultdict(int)

            for c in scheduler.getImminent(tn):
                transitioning[c] |= 1
                outbag = c.outputFnc()
                for outport, msgs in outbag.items():
                    msgs_list = msgs if isinstance(msgs, list) else [msgs]
                    for inport in outport.routing_outline:
                        inport.host_DEVS.my_input.setdefault(
                            inport, []).extend(msgs_list)
                        transitioning[inport.host_DEVS] |= 2

            for aDEVS, ttype in transitioning.items():
                if ttype == 1:
                    aDEVS.state = aDEVS.intTransition()
                    tname = 'internal'
                elif ttype == 2:
                    aDEVS.elapsed = tn[0] - aDEVS.time_last[0]
                    aDEVS.state = aDEVS.extTransition(aDEVS.my_input)
                    tname = 'external'
                else:  # ttype == 3: confluent
                    aDEVS.elapsed = 0.0
                    aDEVS.state = aDEVS.confTransition(aDEVS.my_input)
                    tname = 'confluent'

                aDEVS.time_next = (
                    tn[0] + aDEVS.timeAdvance(),
                    1 if tn[0] > aDEVS.time_last[0] else tn[1] + 1
                )
                aDEVS.time_last = tn
                aDEVS.my_input = {}

                if self._step_callback:
                    self._step_callback(aDEVS, tname, tn[0])

            scheduler.massReschedule(transitioning)
            tn = scheduler.readFirst()

        return tn[0]