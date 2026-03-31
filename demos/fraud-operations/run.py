"""
Fraud Operations — Demo Runner

Multi-workflow fraud investigation: triage classifies fraud type, delegates
to specialty investigation, which fires parallel handlers for regulatory
review and case resolution.

Usage:
    python demos/fraud-operations/run.py
    python demos/fraud-operations/run.py --case cases/card_fraud_cnp.json
    python demos/fraud-operations/run.py --fraud-type check_fraud

Note: This demo uses inline case JSON files. The full production setup
uses a live MCP data server (fraud_data_mcp.py) — see README for details.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

DEMO_DIR = Path(__file__).resolve().parent

TIER_ICONS = {
    "auto":       "✓",
    "spot_check": "~",
    "gate":       "⏸ ",
    "hold":       "⛔",
}

CASE_MAP = {
    "card_fraud":   "card_fraud_cnp.json",
    "check_fraud":  "check_fraud_altered_amount.json",
    "app_scam":     "app_scam_romance.json",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_case(coord: Coordinator, case: dict) -> None:
    fraud_type = case.get("fraud_type", "unknown")
    case_id = case.get("case_id", case.get("_id", "UNKNOWN"))
    description = case.get("description", "")

    print(f"\n{'─' * 68}")
    print(f"  {case_id}  |  {fraud_type}")
    if description:
        print(f"  {description[:80]}")
    print(f"{'─' * 68}")

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="fraud_triage",
        domain="fraud_triage",
        case_input=case,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    # Triage results
    classify = trace.steps.get("classify_fraud_type", {}).get("output", {})
    priority = trace.steps.get("assess_priority", {}).get("output", {})
    if classify:
        print(f"  Fraud type:    {classify.get('category', '?')}  "
              f"(confidence: {classify.get('confidence', 0):.2f})")
    if priority:
        print(f"  Priority:      {priority.get('category', '?')}")

    # Triage governance
    gov = trace.steps.get("govern_triage", {}).get("output", {})
    tier = gov.get("tier_applied") or getattr(instance, "governance_tier", "?")
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "delegated to specialist")
    icon = TIER_ICONS.get(tier_str, "?")
    print(f"  Triage:        {icon} {tier_str.upper()}  —  {disposition}")

    # Delegation
    print(f"  Delegated to:  specialty investigation (fire-and-forget)")
    print(f"                 → regulatory review + case resolution (parallel, wait-for-result)")
    print(f"                 → generate_final_report on resume")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict) and work_order.get("target"):
        print(f"  Work order:    → {work_order.get('target')} queue")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run fraud operations demo")
    parser.add_argument("--case", type=Path, default=None,
                        help="Path to a case JSON file")
    parser.add_argument("--fraud-type",
                        choices=["card_fraud", "check_fraud", "app_scam"],
                        default=None,
                        help="Fraud type (selects a sample case)")
    args = parser.parse_args()

    coord = Coordinator(
        str(DEMO_DIR / "coordinator_config.yaml"),
        db_path=str(DEMO_DIR / "demo.db"),
    )

    if args.case:
        with open(args.case) as f:
            cases = [json.load(f)]
    elif args.fraud_type:
        with open(DEMO_DIR / "cases" / CASE_MAP[args.fraud_type]) as f:
            cases = [json.load(f)]
    else:
        # Run one of each fraud type
        cases = []
        for case_file in ["card_fraud_cnp.json", "check_fraud_altered_amount.json",
                          "app_scam_romance.json"]:
            with open(DEMO_DIR / "cases" / case_file) as f:
                cases.append(json.load(f))

    print("\n" + "═" * 68)
    print("  FRAUD OPERATIONS — Multi-Workflow Demo")
    print("  triage → specialty investigation → [regulatory + case resolution] → report")
    print(f"  {len(cases)} case(s)  |  parallel-handlers coordinator")
    print("═" * 68)

    for case in cases:
        run_case(coord, case)

    print(f"\n{'═' * 68}")
    print("  The coordinator manages the full delegation chain.")
    print("  Each case suspends at deliberation, fires two parallel handlers,")
    print("  and resumes at report generation when both return.")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    main()
