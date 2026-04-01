"""
E-Commerce Returns — Domain Pack Runner

Governed return processing: standard returns auto-approve in seconds,
exceptions investigate customer relationship and circumstances,
fraud patterns are challenged adversarially before escalation.

Usage (from repo root):
    python library/domain-packs/ecommerce-returns/run.py
    python library/domain-packs/ecommerce-returns/run.py --case-id RET-003

Embed in your application:
    coord = Coordinator('coordinator_config.yaml', db_path='returns.db')
    iid = coord.start('ecommerce_return', 'ecommerce_return', case_data)
    inst = coord.store.get_instance(iid)
    disposition = inst.disposition  # approve / deny / gate / hold
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

PACK_DIR = Path(__file__).resolve().parent

TIER_LABELS = {
    "auto":       "✓  APPROVE",
    "spot_check": "~  APPROVE (sampled)",
    "gate":       "⏸  AGENT REVIEW",
    "hold":       "⛔ LOSS PREVENTION",
}

PATH_LABELS = {
    "within_policy":    "standard path",
    "outside_policy":   "exception path",
    "suspected_fraud":  "fraud path",
    "carrier_claim":    "carrier claim path",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = output


def run_return(coord: Coordinator, ret: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {ret.get('_id', '?')}  |  Order {ret.get('order_id', '?')}")
    print(f"  Reason: {ret.get('return_reason', '?')}  |  "
          f"Value: ${ret.get('order_value', 0):.2f}  |  "
          f"Days since delivery: {ret.get('days_since_delivery', '?')}")
    if ret.get("_description"):
        print(f"  {ret['_description']}")
    if ret.get("_expected"):
        print(f"  Expected: {ret['_expected']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in ret.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="ecommerce_return",
        domain="ecommerce_return",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    steps_run = list(trace.steps.keys())
    if steps_run:
        print(f"  Path:            {' → '.join(steps_run)}")

    classify = trace.steps.get("classify_return", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence", 0)
        path = PATH_LABELS.get(cat, cat)
        print(f"  Classification:  {cat}  ({path}, confidence {conf:.2f})")

    verify = trace.steps.get("verify_policy", {})
    if verify:
        conforms = verify.get("conforms")
        violations = verify.get("violations", [])
        print(f"  Policy:          {'✓ conforms' if conforms else f'✗ {len(violations)} violation(s)'}")
        if violations:
            for v in violations[:2]:
                print(f"    - {str(v)[:70]}")

    if "investigate_exception" in trace.steps:
        finding = str(trace.steps["investigate_exception"].get("finding", ""))
        print(f"  Exception inv.:  {finding[:80]}{'...' if len(finding) > 80 else ''}")

    if "deliberate_exception" in trace.steps:
        action = trace.steps["deliberate_exception"].get("recommended_action", "?")
        print(f"  Exception dec.:  {action}")

    if "investigate_fraud" in trace.steps:
        finding = str(trace.steps["investigate_fraud"].get("finding", ""))
        print(f"  Fraud inv.:      {finding[:80]}{'...' if len(finding) > 80 else ''}")

    if "challenge_fraud_finding" in trace.steps:
        survives = trace.steps["challenge_fraud_finding"].get("survives")
        vulns = len(trace.steps["challenge_fraud_finding"].get("vulnerabilities") or [])
        print(f"  Fraud challenge: {'survives (uncertain)' if survives else 'fails (fraud likely)'}  "
              f"({vulns} counter-arguments)")

    gov = trace.steps.get("govern_disposition", {})
    tier = gov.get("tier_applied") or str(getattr(instance, "governance_tier", "gate")).lower()
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    label = TIER_LABELS.get(tier_str, tier_str.upper())
    print(f"  Decision:        {label}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict) and work_order.get("target"):
        print(f"  Routed to:       {work_order.get('target')} queue")
        if work_order.get("brief"):
            print(f"  Agent brief:     {str(work_order['brief'])[:80]}...")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run e-commerce returns pack")
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "returns.db"),
    )

    with open(PACK_DIR / "cases" / "example_returns.json") as f:
        returns = json.load(f)

    if args.case_id:
        returns = [r for r in returns if r.get("_id") == args.case_id]
        if not returns:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 68)
    print("  E-COMMERCE RETURNS — Domain Pack")
    print("  retrieve → classify → [verify | investigate | fraud] → govern")
    print(f"  {len(returns)} return(s)  |  standard + exception + fraud paths")
    print("═" * 68)

    for ret in returns:
        run_return(coord, ret)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
