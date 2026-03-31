"""
Consumer Lending — Domain Pack Runner

Full-chain loan application review: retrieve → classify → investigate
→ deliberate → verify → govern. Six primitives. Four governance tiers.

Usage (from repo root):
    python library/domain-packs/consumer-lending/run.py
    python library/domain-packs/consumer-lending/run.py --case-id APP-003
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

PACK_DIR = Path(__file__).resolve().parent

TIER_ICONS = {
    "auto":       "✓",
    "spot_check": "~",
    "gate":       "⏸ ",
    "hold":       "⛔",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_application(coord: Coordinator, app: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {app['_id']}  |  {app['applicant_name']}, age {app['applicant_age']}")
    print(f"  ${app['loan_amount']:,}  —  {app['loan_purpose']}")
    credit = app.get("get_credit", {})
    fin = app.get("get_financials", {})
    emp = app.get("get_employment", {})
    if credit:
        print(f"  Score: {credit.get('score', '?')}  |  "
              f"DTI: {fin.get('dti_ratio', 0):.0%}  |  "
              f"Employment: {emp.get('status', '?')}")
    if app.get("_expected"):
        print(f"  Expected: {app['_expected']}")
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
    tier = gov.get("tier_applied") or getattr(instance, "governance_tier", "?")
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    icon = TIER_ICONS.get(tier_str, "?")
    print(f"  Governance:     {icon} {tier_str.upper()}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict):
        print(f"  Work order:     → {work_order.get('target', '?')} queue")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "pack.db"),
    )

    with open(PACK_DIR / "cases" / "applications.json") as f:
        applications = json.load(f)

    if args.case_id:
        applications = [a for a in applications if a.get("_id") == args.case_id]
        if not applications:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 68)
    print("  CONSUMER LENDING — Domain Pack")
    print("  retrieve → classify → [investigate] → deliberate → verify → govern")
    print(f"  {len(applications)} application(s)")
    print("═" * 68)

    for app in applications:
        run_application(coord, app)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
