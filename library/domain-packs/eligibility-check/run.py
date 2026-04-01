"""
Eligibility Check — Domain Pack Runner

Fast governed eligibility determination embeddable in any web or app workflow.
Clear cases resolve immediately with typed disposition. Borderline cases
investigate before deciding. Every determination has a warrant.

Usage (from repo root):
    python library/domain-packs/eligibility-check/run.py
    python library/domain-packs/eligibility-check/run.py --case-id ELIG-002

Embed in your application:
    coord = Coordinator('coordinator_config.yaml', db_path='eligibility.db')
    iid = coord.start('eligibility_check', 'eligibility_check', case_data)
    inst = coord.store.get_instance(iid)
    # Read disposition and warrant from governance step output
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

PACK_DIR = Path(__file__).resolve().parent

TIER_LABELS = {
    "auto":       "✓  ELIGIBLE — grant access",
    "spot_check": "~  ELIGIBLE — grant access (sampled)",
    "gate":       "⏸  REVIEW — hold for verification",
    "hold":       "⛔ HOLD — compliance required",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = output


def run_check(coord: Coordinator, check: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {check.get('_id', '?')}  |  {check.get('check_type', '?')}")
    print(f"  Benefit: {check.get('benefit', '?')}  |  "
          f"Claimed: {check.get('claimed_status', '?')}")
    if check.get("_description"):
        print(f"  {check['_description']}")
    if check.get("_expected"):
        print(f"  Expected: {check['_expected']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in check.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="eligibility_check",
        domain="eligibility_check",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    steps_run = list(trace.steps.keys())
    if steps_run:
        print(f"  Path:            {' → '.join(steps_run)}")

    classify = trace.steps.get("classify_eligibility", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence", 0)
        print(f"  Classification:  {cat}  (confidence {conf:.2f})")

    if "investigate_borderline" in trace.steps:
        finding = str(trace.steps["investigate_borderline"].get("finding", ""))
        print(f"  Investigation:   {finding[:80]}{'...' if len(finding) > 80 else ''}")

    if "deliberate_eligibility" in trace.steps:
        action = trace.steps["deliberate_eligibility"].get("recommended_action", "?")
        warrant = str(trace.steps["deliberate_eligibility"].get("warrant") or "")
        print(f"  Determination:   {action}")
        if warrant:
            print(f"  Warrant:         {warrant[:80]}{'...' if len(warrant) > 80 else ''}")

    verify = trace.steps.get("verify_eligibility", {})
    if verify:
        conforms = verify.get("conforms")
        violations = verify.get("violations", [])
        print(f"  Rules:           {'✓ all satisfied' if conforms else f'✗ {len(violations)} violation(s)'}")
        if violations:
            for v in violations[:2]:
                print(f"    Rule {str(v)[:70]}")

    gov = trace.steps.get("govern_eligibility", {})
    tier = gov.get("tier_applied") or str(getattr(instance, "governance_tier", "gate")).lower()
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    label = TIER_LABELS.get(tier_str, tier_str.upper())
    print(f"  Decision:        {label}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict) and work_order.get("target"):
        print(f"  Routed to:       {work_order.get('target')} queue")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run eligibility check pack")
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "eligibility.db"),
    )

    with open(PACK_DIR / "cases" / "example_checks.json") as f:
        checks = json.load(f)

    if args.case_id:
        checks = [c for c in checks if c.get("_id") == args.case_id]
        if not checks:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 68)
    print("  ELIGIBILITY CHECK — Domain Pack")
    print("  retrieve → classify → [investigate →] verify → govern")
    print(f"  {len(checks)} check(s)  |  subscription · student · professional · loyalty")
    print("═" * 68)

    for check in checks:
        run_check(coord, check)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
