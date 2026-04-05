"""
DDD Agentic Mode — Live LLM Test
Military Hardship Assessment

Demonstrates autonomous epistemic trajectory selection within structural
guardrails. The orchestrator selects which primitives to invoke and in
what order — no sequence is declared. Governance, epistemic state
computation, and audit ledger recording operate identically to workflow
mode on whatever trajectory the orchestrator produces.

Two cases are run to show trajectory differentiation:
  1. Military transition (Reeves) — expected to trigger SCRA investigation
     and synthesize_approach before generating guidance
  2. Civilian overextension (Webb) — expected to use financial investigation
     and proceed directly to guidance generation

The orchestrator's decisions are printed as they happen, showing the
autonomous trajectory as it unfolds.

Usage (from repo root):
    python examples/run_ddd_test.py
    python examples/run_ddd_test.py --case reeves
    python examples/run_ddd_test.py --case webb
    python examples/run_ddd_test.py --case both

Prerequisites:
    - LLM configured (GOOGLE_API_KEY or ANTHROPIC_API_KEY in environment,
      or model configured in llm_config.yaml)
    - pip install -e . (from repo root)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT    = EXAMPLES_DIR.parent

WORKFLOW  = "loan_hardship_agentic"
DOMAIN    = "military_hardship_agentic"
CONFIG    = EXAMPLES_DIR / "coordinator_config_agentic.yaml"

CASES = {
    "reeves": EXAMPLES_DIR / "cases" / "military_hardship_reeves.json",
    "webb":   EXAMPLES_DIR / "cases" / "civilian_hardship_webb.json",
}

TIER_ICONS = {
    "auto":       "✓  AUTO",
    "spot_check": "~  SPOT CHECK",
    "gate":       "⏸  GATE",
    "hold":       "⛔ HOLD",
}


class _AgenticTrace(NullTrace):
    """
    Captures orchestrator decisions and step completions for display.
    Shows the autonomous trajectory as it unfolds.
    """
    def __init__(self):
        self.decisions: list[dict] = []
        self.steps: list[dict] = []
        self._t0 = time.time()

    def on_route_decision(self, from_step, to_step, decision_type, reason):
        if decision_type == "agent":
            elapsed = time.time() - self._t0
            entry = {
                "from": from_step,
                "to": to_step,
                "reason": reason,
                "elapsed": elapsed,
            }
            self.decisions.append(entry)
            # Print immediately so trajectory is visible as it happens
            if to_step == "__end__":
                print(f"  🧭 orchestrator → end  ({reason[:80]})")
            else:
                print(f"  🧭 orchestrator → {to_step}  ({reason[:80]})")

    def on_parse_result(self, step_name, primitive, output):
        entry = {
            "step_name": step_name,
            "primitive": primitive,
            "output": output,
            "elapsed": time.time() - self._t0,
        }
        self.steps[step_name if isinstance(step_name, str) else step_name] = entry
        # Print step completion
        conf = output.get("confidence", "?")
        if isinstance(conf, float):
            conf = f"{conf:.2f}"
        summary = _step_summary(primitive, output)
        print(f"  {_prim_icon(primitive)} {step_name}  [{primitive}]  conf={conf}")
        if summary:
            print(f"    {summary}")

    def __init__(self):
        self.decisions = []
        self.steps = {}
        self._t0 = time.time()


def _prim_icon(primitive: str) -> str:
    return {
        "retrieve":    "📥",
        "classify":    "🏷 ",
        "investigate": "🔍",
        "deliberate":  "🤔",
        "verify":      "✅",
        "generate":    "📝",
        "govern":      "⚖️ ",
        "challenge":   "⚔️ ",
    }.get(primitive, "▶ ")


def _step_summary(primitive: str, output: dict) -> str:
    if primitive == "classify":
        cat = output.get("category", "?")
        return f"→ {cat}"
    if primitive == "investigate":
        finding = output.get("finding", "")
        return f"→ {str(finding)[:120]}{'…' if len(str(finding)) > 120 else ''}"
    if primitive == "deliberate":
        action = output.get("recommended_action", "")
        return f"→ {str(action)[:100]}"
    if primitive == "generate":
        art = output.get("artifact", "")
        return f"→ {len(str(art))} chars"
    if primitive == "challenge":
        survives = output.get("survives", "?")
        vulns = len(output.get("vulnerabilities", []))
        return f"→ survives={survives}  vulnerabilities={vulns}"
    if primitive == "verify":
        conforms = output.get("conforms", "?")
        viols = len(output.get("violations", []))
        return f"→ conforms={conforms}  violations={viols}"
    return ""


def run_case(coord: Coordinator, case_path: Path) -> None:
    with open(case_path) as f:
        case = json.load(f)

    case_id      = case.get("_id", "UNKNOWN")
    description  = case.get("_description", "")
    expected_path = case.get("_expected_path", "")
    expected_tier = case.get("_expected_tier", "")

    # Strip internal fields before passing to coordinator
    case_input = {k: v for k, v in case.items() if not k.startswith("_")}

    print(f"\n{'═' * 70}")
    print(f"  CASE: {case_id}")
    print(f"  {description[:80]}")
    print(f"  Expected path: {expected_path}")
    print(f"  Expected tier: {expected_tier.upper()}")
    print(f"{'─' * 70}")
    print(f"  Autonomous trajectory (orchestrator decisions → step completions):")
    print()

    trace = _AgenticTrace()
    set_trace(trace)
    t0 = time.time()

    instance_id = coord.start(
        workflow_type=WORKFLOW,
        domain=DOMAIN,
        case_input=case_input,
    )

    elapsed = time.time() - t0
    set_trace(NullTrace())

    # Read result from store
    instance = coord.store.get_instance(instance_id)
    tier = ""
    disposition = ""
    if instance:
        tier = str(getattr(instance, "governance_tier", "")).lower().replace("governancetier.", "")
        disposition = getattr(instance, "disposition", "")

    print()
    print(f"{'─' * 70}")
    print(f"  Result:  {TIER_ICONS.get(tier, tier.upper())}  —  {disposition}")
    print(f"  Elapsed: {elapsed:.1f}s  |  Steps: {len(trace.steps)}")

    # Trajectory: read orchestrator_decision entries from ledger
    ledger = coord.store.get_ledger(instance_id=instance_id)
    orch_decisions = [
        e["details"] for e in ledger
        if e.get("action_type") == "orchestrator_decision"
    ]
    trajectory = [
        d.get("step_name", d.get("action", "end"))
        for d in orch_decisions
        if d.get("action") != "end"
    ]

    print(f"  Orchestrator decisions: {len(orch_decisions)}")
    print(f"\n  Trajectory taken:")
    if trajectory:
        print(f"    {' → '.join(trajectory)}")
        # Print each decision's reasoning
        for d in orch_decisions:
            action = d.get("action", "invoke")
            name = d.get("step_name") or "end"
            prim = d.get("primitive", "")
            reason = d.get("reasoning", "")[:80]
            icon = "⏹" if action == "end" else _prim_icon(prim)
            print(f"    {icon} → {name}  {reason}")
    else:
        print(f"    (no decisions recorded in ledger)")

    # Compare to expected
    expected_steps = [s.strip() for s in expected_path.split("→")]
    actual_matches = trajectory == expected_steps
    if expected_path:
        print(f"\n  Expected:  {' → '.join(expected_steps)}")
        print(f"  Match: {'✓ YES' if actual_matches else '△ DIFFERENT'}")

    # Epistemic state from govern step if present
    govern_output = trace.steps.get("govern_eia_determination") or trace.steps.get("govern_hardship")
    if not govern_output:
        # Find any govern step
        for name, step in trace.steps.items():
            if step.get("primitive") == "govern":
                govern_output = step
                break
    if govern_output:
        out = govern_output.get("output", {})
        rationale = out.get("tier_rationale", "")
        if rationale:
            print(f"\n  Governance rationale:")
            print(f"    {rationale[:200]}")

    print(f"{'═' * 70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DDD Agentic Mode — Live LLM Test"
    )
    parser.add_argument(
        "--case",
        choices=["reeves", "webb", "both"],
        default="both",
        help="Which case to run (default: both)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full coordinator output",
    )
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  DDD AGENTIC MODE — Live LLM Test")
    print("  Military Hardship Assessment")
    print()
    print("  Claim: The orchestrator selects the trajectory autonomously.")
    print("  The substrate enforces governance on that trajectory identically")
    print("  to workflow mode — same epistemic state, same audit ledger,")
    print("  same four-tier governance. The orchestrator controls the path;")
    print("  the substrate controls the accountability.")
    print()
    print(f"  Workflow:  {WORKFLOW}")
    print(f"  Domain:    {DOMAIN}")
    print("═" * 70)

    coord = Coordinator(
        str(CONFIG),
        db_path=str(EXAMPLES_DIR / "agentic_test.db"),
        verbose=args.verbose,
    )

    cases_to_run = []
    if args.case == "both":
        cases_to_run = [("reeves", CASES["reeves"]), ("webb", CASES["webb"])]
    else:
        cases_to_run = [(args.case, CASES[args.case])]

    for name, path in cases_to_run:
        run_case(coord, path)

    print()


if __name__ == "__main__":
    main()