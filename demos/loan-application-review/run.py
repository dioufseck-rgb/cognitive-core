"""
Loan Application Review — Full Chain Demo

Six primitives, four governance outcomes, real institutional logic.

Usage:
    python demos/loan-application-review/run.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

DEMO_DIR = Path(__file__).resolve().parent

TIER_ICONS = {
    "auto":        "✓",
    "spot_check":  "~",
    "gate":        "⏸ ",
    "hold":        "⛔",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name: str, primitive: str, output: dict) -> None:
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_application(coord: Coordinator, app: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {app['_id']}  |  {app['applicant_name']}, {app['applicant_age']}")
    print(f"  ${app['loan_amount']:,}  —  {app['loan_purpose']}")
    print(f"  Credit: {app['get_credit']['score']}  |  "
          f"DTI: {app['get_financials']['dti_ratio']:.0%}  |  "
          f"Employment: {app['get_employment']['status']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in app.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="loan_application_review",
        domain="consumer_lending",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    steps_run = list(trace.steps.keys())
    print(f"  Steps run:      {' → '.join(steps_run)}")

    classify = trace.steps.get("classify_risk", {}).get("output", {})
    if classify:
        print(f"  Risk tier:      {classify.get('category', '?')}")

    delib = trace.steps.get("deliberate_recommendation", {}).get("output", {})
    if delib:
        action = delib.get("recommended_action", "?")
        warrant = str(delib.get("warrant") or "")
        print(f"  Recommendation: {action}")
        if warrant:
            print(f"  Warrant:        {warrant[:120]}{'...' if len(warrant) > 120 else ''}")

    verify = trace.steps.get("verify_compliance", {}).get("output", {})
    if verify:
        conforms = verify.get("conforms")
        violations = len(verify.get("violations") or [])
        print(f"  Compliance:     {'✓ conforms' if conforms else f'✗ {violations} violation(s)'}")

    gov = trace.steps.get("govern_decision", {}).get("output", {})
    tier = gov.get("tier_applied") or instance.governance_tier or "?"
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    icon = TIER_ICONS.get(tier_str, "?")
    print(f"  Governance:     {icon} {tier_str.upper()}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict):
        print(f"  Work order:     → {work_order.get('target', '?')} queue")

    print(f"  Expected:       {app.get('_expected', '')}")


def main():
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("No LLM key — running in simulated mode.\n")

    coord = Coordinator(
        str(DEMO_DIR / "coordinator_config.yaml"),
        db_path=str(DEMO_DIR / "demo.db"),
    )

    with open(DEMO_DIR / "cases" / "applications.json") as f:
        applications = json.load(f)

    print("\n" + "═" * 68)
    print("  LOAN APPLICATION REVIEW — Full Chain Demo")
    print("  retrieve → classify → investigate → deliberate → verify → govern")
    print("  Six primitives. Four applications. Four governance outcomes.")
    print("═" * 68)

    for app in applications:
        run_application(coord, app)

    print(f"\n{'═' * 68}")
    print("  Every step typed. Every decision warranted. Every escalation auditable.")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    main()
