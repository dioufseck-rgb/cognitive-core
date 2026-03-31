"""
Compliance Review — Domain Pack Runner

Reviews case, decision, or artifact conformance against explicit rules.
Verify-first design enables early exit for clean conformance.

Usage (from repo root):
    python library/domain-packs/compliance-review/run.py
    python library/domain-packs/compliance-review/run.py --case-id COMP-002
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


def run_review(coord: Coordinator, review: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {review.get('_id', '?')}  |  {review.get('decision_type', '')}")
    print(f"  Subject: {review.get('subject_description', '')}")
    if review.get("_expected"):
        print(f"  Expected: {review['_expected']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in review.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="compliance_and_conformance",
        domain="compliance_review",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    steps_run = list(trace.steps.keys())
    print(f"  Steps run:      {' → '.join(steps_run)}")

    verify = trace.steps.get("verify_conformance", {}).get("output", {})
    if verify:
        conforms = verify.get("conforms")
        violations = verify.get("violations", [])
        print(f"  Conformance:    {'✓ conforms' if conforms else f'✗ {len(violations)} violation(s)'}")
        if violations:
            for v in violations[:3]:
                print(f"    - {str(v)[:80]}")

    investigate = trace.steps.get("investigate_violations", {}).get("output", {})
    if investigate:
        finding = str(investigate.get("finding", ""))
        print(f"  Investigation:  {finding[:100]}{'...' if len(finding) > 100 else ''}")

    deliberate = trace.steps.get("deliberate_finding", {}).get("output", {})
    if deliberate:
        action = deliberate.get("recommended_action", "?")
        print(f"  Finding:        {action}")

    gov = trace.steps.get("govern_outcome", {}).get("output", {})
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

    with open(PACK_DIR / "cases" / "example_reviews.json") as f:
        reviews = json.load(f)

    if args.case_id:
        reviews = [r for r in reviews if r.get("_id") == args.case_id]
        if not reviews:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 68)
    print("  COMPLIANCE REVIEW — Domain Pack")
    print("  retrieve → verify → [investigate] → [deliberate] → generate → govern")
    print(f"  {len(reviews)} review(s)  |  verify-first design")
    print("═" * 68)

    for review in reviews:
        run_review(coord, review)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
