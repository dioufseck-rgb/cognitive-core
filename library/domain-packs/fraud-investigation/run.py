"""
Fraud Investigation — Domain Pack Runner

Multi-workflow fraud investigation: triage → specialty investigation
→ parallel handlers (regulatory review + case resolution) → report.

Usage (from repo root):
    python library/domain-packs/fraud-investigation/run.py
    python library/domain-packs/fraud-investigation/run.py --case cases/card_fraud_cnp.json
    python library/domain-packs/fraud-investigation/run.py --fraud-type app_scam

Note: This pack uses the parallel-handlers coordinator template.
The specialty investigation suspends after deliberation, fires regulatory
review and case resolution in parallel, and resumes at generate_final_report
when both handlers return. Run time is longer than single-workflow packs.

Note on data: The fraud demo uses a SQLite MCP server to serve case data.
For library use, case data is provided directly in the case JSON.
Set DATA_MCP_CMD in coordinator_config.yaml if you have a live data source.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

PACK_DIR = Path(__file__).resolve().parent

FRAUD_TYPE_DOMAINS = {
    "card_fraud":   "card_fraud",
    "check_fraud":  "check_fraud",
    "app_scam":     "app_scam_fraud",
}

TIER_ICONS = {
    "auto":       "✓",
    "spot_check": "~",
    "gate":       "⏸ ",
    "hold":       "⛔",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}
        self.workflow_steps: dict[str, dict[str, dict]] = {}
        self._current_workflow = "primary"

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_case(coord: Coordinator, case: dict) -> None:
    fraud_type = case.get("fraud_type", "card_fraud")
    case_id = case.get("case_id", case.get("_id", "UNKNOWN"))

    print(f"\n{'─' * 68}")
    print(f"  {case_id}  |  {fraud_type}")
    print(f"  {case.get('description', '')[:80]}")
    if case.get("_expected"):
        print(f"  Expected: {case['_expected']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in case.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    # Triage workflow routes to specialty investigation based on fraud_type
    instance_id = coord.start(
        workflow_type="fraud_triage",
        domain="fraud_triage",
        case_input=case_input,
    )

    set_trace(NullTrace())

    # Triage results
    triage_classify = trace.steps.get("classify_fraud_type", {}).get("output", {})
    triage_priority = trace.steps.get("assess_priority", {}).get("output", {})
    if triage_classify:
        print(f"  Triage:          type={triage_classify.get('category', '?')}  "
              f"priority={triage_priority.get('category', '?')}")

    # The triage fire-and-forget delegation spawns the specialty investigation.
    # For packs running without a persistent event loop, we check for
    # delegated instances in the store.
    delegated = coord.store.get_delegated_instances(instance_id)
    if delegated:
        print(f"  Delegated to:    {len(delegated)} specialty workflow(s)")
        for d in delegated:
            d_instance = coord.store.get_instance(d)
            if d_instance:
                d_tier = str(getattr(d_instance, "governance_tier", "?")).lower().replace("governancetier.", "")
                d_disp = getattr(d_instance, "disposition", "?")
                icon = TIER_ICONS.get(d_tier, "?")
                print(f"    Specialty:     {icon} {d_tier.upper()}  —  {d_disp}")
    else:
        print(f"  Specialty investigation: dispatched (check coordinator store for async result)")
        print(f"  (In production: the specialty workflow runs in the coordinator event loop)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=None,
                        help="Path to a case JSON file")
    parser.add_argument("--fraud-type", choices=["card_fraud", "check_fraud", "app_scam"],
                        default="card_fraud",
                        help="Fraud type to run (selects example case)")
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "pack.db"),
    )

    if args.case:
        with open(args.case) as f:
            cases = [json.load(f)]
    else:
        case_map = {
            "card_fraud":  PACK_DIR / "cases" / "card_fraud_cnp.json",
            "check_fraud": PACK_DIR / "cases" / "check_fraud_altered_amount.json",
            "app_scam":    PACK_DIR / "cases" / "app_scam_romance.json",
        }
        with open(case_map[args.fraud_type]) as f:
            cases = [json.load(f)]

    print("\n" + "═" * 68)
    print("  FRAUD INVESTIGATION — Domain Pack")
    print("  triage → specialty investigation → [regulatory + case resolution] → report")
    print(f"  {len(cases)} case(s)  |  parallel-handlers coordinator")
    print("═" * 68)

    for case in cases:
        run_case(coord, case)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
