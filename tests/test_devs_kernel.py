"""
Tests for the minimal DEVS simulation kernel.

Validates:
1. AtomicDEVS lifecycle: active → running → passive
2. INFINITY passive state — model does not fire without external event
3. extTransition — external event wakes passive model
4. CoupledDEVS — output routes from one atomic to another
5. Simulator.inject() — thread-safe external event injection
6. WorkflowExecutor step sequencing
"""

import sys
import time
import threading
sys.path.insert(0, '/home/claude/cognitive-core/cognitive-core')

from cognitive_core.engine.devs import (
    AtomicDEVS, CoupledDEVS, Simulator, INFINITY
)


# ─── Test 1: Simple atomic model lifecycle ────────────────────────────────────

class PingModel(AtomicDEVS):
    """Fires once after 0.01s, then goes passive."""
    def __init__(self):
        AtomicDEVS.__init__(self, "ping")
        self.state = {"phase": "active", "fired": False}

    def timeAdvance(self):
        return 0.01 if self.state["phase"] == "active" else INFINITY

    def intTransition(self):
        return {"phase": "passive", "fired": True}

    def extTransition(self, inputs):
        return self.state

    def outputFnc(self):
        return {}


def test_atomic_lifecycle():
    model = PingModel()
    sim = Simulator(model)

    transitions = []
    def on_step(m, ttype, t):
        transitions.append((m.name, ttype, t, m.state.copy()))

    sim.setStepCallback(on_step)
    sim.setTerminationTime(1.0)
    sim.simulate()

    assert len(transitions) == 1, f"Expected 1 transition, got {len(transitions)}"
    assert transitions[0][1] == "internal"
    assert model.state["phase"] == "passive"
    assert model.state["fired"] == True
    print("✓ test_atomic_lifecycle")


# ─── Test 2: INFINITY — passive model does not fire ───────────────────────────

class PassiveModel(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "passive")
        self.state = {"fired": False}

    def timeAdvance(self):
        return INFINITY

    def intTransition(self):
        return {"fired": True}

    def extTransition(self, inputs):
        return self.state

    def outputFnc(self):
        return {}


def test_passive_does_not_fire():
    model = PassiveModel()
    sim = Simulator(model)
    sim.setTerminationTime(0.1)
    sim.simulate()

    assert model.state["fired"] == False
    print("✓ test_passive_does_not_fire")


# ─── Test 3: extTransition — inject wakes passive model ──────────────────────

class WakeableModel(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "wakeable")
        self.in_wake = self.addInPort("wake")
        self.state = {"phase": "sleeping", "woken": False}

    def timeAdvance(self):
        if self.state["phase"] == "awake":
            return 0.01
        return INFINITY

    def intTransition(self):
        return {"phase": "done", "woken": self.state["woken"]}

    def extTransition(self, inputs):
        if self.in_wake in inputs:
            return {"phase": "awake", "woken": True}
        return self.state

    def outputFnc(self):
        return {}


def test_inject_wakes_passive():
    model = WakeableModel()
    sim = Simulator(model)

    # Inject wake event after 0.05s
    def inject_later():
        time.sleep(0.05)
        sim.inject(model, model.in_wake, "wake!", sim_time=0.05)

    t = threading.Thread(target=inject_later, daemon=True)
    t.start()

    sim.setRealTime(True)
    sim.setTerminationTime(1.0)
    sim.simulate()

    assert model.state["phase"] == "done"
    assert model.state["woken"] == True
    print("✓ test_inject_wakes_passive")


# ─── Test 4: CoupledDEVS — output routes between atomics ─────────────────────

class Sender(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "sender")
        self.out = self.addOutPort("out")
        self.state = {"phase": "active"}

    def timeAdvance(self):
        return 0.01 if self.state["phase"] == "active" else INFINITY

    def intTransition(self):
        return {"phase": "passive"}

    def extTransition(self, inputs):
        return self.state

    def outputFnc(self):
        return {self.out: "hello"}


class Receiver(AtomicDEVS):
    def __init__(self):
        AtomicDEVS.__init__(self, "receiver")
        self.inp = self.addInPort("in")
        self.state = {"received": None}

    def timeAdvance(self):
        return INFINITY

    def intTransition(self):
        return self.state

    def extTransition(self, inputs):
        if self.inp in inputs:
            return {"received": inputs[self.inp][0]}
        return self.state

    def outputFnc(self):
        return {}


class SenderReceiverCoupled(CoupledDEVS):
    def __init__(self):
        CoupledDEVS.__init__(self, "coupled")
        self.sender = self.addSubModel(Sender())
        self.receiver = self.addSubModel(Receiver())
        self.connectPorts(self.sender.out, self.receiver.inp)


def test_coupled_routing():
    model = SenderReceiverCoupled()
    sim = Simulator(model)
    sim.setTerminationTime(1.0)
    sim.simulate()

    assert model.receiver.state["received"] == "hello", \
        f"Expected 'hello', got {model.receiver.state['received']}"
    print("✓ test_coupled_routing")


# ─── Test 5: Sequential step chain ───────────────────────────────────────────

class CountingStep(AtomicDEVS):
    """Activates, fires after 0.01s, passes message to next."""
    def __init__(self, name, sequence):
        AtomicDEVS.__init__(self, name)
        self.inp = self.addInPort("in")
        self.out = self.addOutPort("out")
        self.state = {"phase": "waiting", "count": 0}
        self.sequence = sequence
        self._first = (len(sequence) == 0)  # first step starts active

    def timeAdvance(self):
        return 0.01 if self.state["phase"] == "active" else INFINITY

    def intTransition(self):
        return {"phase": "passive", "count": self.state["count"] + 1}

    def extTransition(self, inputs):
        if self.inp in inputs:
            return {"phase": "active", "count": self.state["count"]}
        return self.state

    def outputFnc(self):
        return {self.out: f"step_{self.name}"}


class ThreeStepWorkflow(CoupledDEVS):
    def __init__(self):
        CoupledDEVS.__init__(self, "three_step")
        seq = []
        self.step1 = self.addSubModel(CountingStep("step1", seq))
        self.step2 = self.addSubModel(CountingStep("step2", seq))
        self.step3 = self.addSubModel(CountingStep("step3", seq))

        # step1 starts active
        self.step1.state["phase"] = "active"

        # wire sequentially
        self.connectPorts(self.step1.out, self.step2.inp)
        self.connectPorts(self.step2.out, self.step3.inp)


def test_sequential_chain():
    model = ThreeStepWorkflow()
    sim = Simulator(model)

    transitions = []
    def on_step(m, ttype, t):
        if isinstance(m, CountingStep):
            transitions.append(m.name)

    sim.setStepCallback(on_step)
    sim.setTerminationTime(1.0)
    sim.simulate()

    # All three steps should have fired their internal transitions
    internal_fires = [name for name in transitions
                      if model.__dict__.get(name)]
    assert model.step1.state["count"] == 1
    assert model.step2.state["count"] == 1
    assert model.step3.state["count"] == 1
    print("✓ test_sequential_chain")


# ─── Test 6: Suspension and resume via inject ─────────────────────────────────

class SuspendableStep(AtomicDEVS):
    """Fires once, then suspends (ta=INFINITY) until injected."""
    def __init__(self):
        AtomicDEVS.__init__(self, "suspendable")
        self.in_resume = self.addInPort("resume")
        self.out = self.addOutPort("out")
        self.state = {"phase": "active", "resumed": False}

    def timeAdvance(self):
        if self.state["phase"] in ("active", "resumed"):
            return 0.01
        return INFINITY

    def intTransition(self):
        if self.state["phase"] == "active":
            return {"phase": "suspended", "resumed": False}
        if self.state["phase"] == "resumed":
            return {"phase": "passive", "resumed": True}
        return self.state

    def extTransition(self, inputs):
        if self.in_resume in inputs:
            return {"phase": "resumed", "resumed": False}
        return self.state

    def outputFnc(self):
        return {self.out: self.state["phase"]}


def test_suspend_and_resume():
    model = SuspendableStep()
    sim = Simulator(model)

    # After first fire (suspended), inject resume after 0.1s
    def resume_later():
        time.sleep(0.1)
        sim.inject(model, model.in_resume, "human_decision", sim_time=0.1)

    t = threading.Thread(target=resume_later, daemon=True)
    t.start()

    sim.setRealTime(True)
    sim.setTerminationTime(1.0)
    sim.simulate()

    assert model.state["phase"] == "passive"
    assert model.state["resumed"] == True
    print("✓ test_suspend_and_resume")


# ─── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n═══ DEVS Kernel Tests ═══\n")
    test_atomic_lifecycle()
    test_passive_does_not_fire()
    test_inject_wakes_passive()
    test_coupled_routing()
    test_sequential_chain()
    test_suspend_and_resume()
    print("\n✓ All tests passed\n")
