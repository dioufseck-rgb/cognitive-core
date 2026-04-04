"""
Permit Review — Demo Runner

Policy-grounded environmental permit review using CEQA, 14 CCR, and
municipal code. The workflow retrieves the regulatory corpus as typed
tool call responses and classifies each application against the actual
statutory text. Every finding in the audit ledger cites a specific
instrument, section, and provision.

Delegation topology (for conditional and complex cases):
  permit_intake → [fire-and-forget] → eia_assessment
    → [parallel wait-for-result] → public_notice_compliance
                                  + biological_resources_review
    → resume at deliberate_determination

Usage:
    python demos/permit-review/run.py
    python demos/permit-review/run.py --case cases/pmt_2026_00142.json
    python demos/permit-review/run.py --permit exempt
    python demos/permit-review/run.py --permit conditional
    python demos/permit-review/run.py --permit prohibited

Cases:
    pmt_2026_00142  Tenant improvement, restaurant, C-2 zone
                    → §15301 Class 1 Categorical Exemption
                    → Expected tier: SPOT CHECK

    pmt_2026_00318  Mixed-use infill, 180 units, transit corridor
                    → §15332 Class 32 exemption fails Condition (d)
                    → Initial Study required, MND pathway
                    → Expected tier: GATE (EIA specialist + parallel handlers)

    pmt_2026_00447  Concrete batch plant, regulatory floodway
                    → Municipal Code §18.52.060(B)(1) prohibition
                    → 44 CFR §60.3(d) — no variance available
                    → Expected tier: HOLD (city attorney)

Note: This demo runs on inline fixture JSON files. Tool calls in the
retrieve steps are served from the case JSON — each get_* key in the
case file corresponds to a tool call in the retrieve specification.
"""

import json
import sys
import argparse
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
    "exempt":      "pmt_2026_00142.json",
    "conditional": "pmt_2026_00318.json",
    "prohibited":  "pmt_2026_00447.json",
}

CASE_LABELS = {
    "pmt_2026_00142": ("PMT-2026-00142", "Tenant improvement — restaurant, C-2 zone"),
    "pmt_2026_00318": ("PMT-2026-00318", "Mixed-use infill — 180 units, transit corridor"),
    "pmt_2026_00447": ("PMT-2026-00447", "Concrete batch plant — regulatory floodway"),
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_case(coord: Coordinator, case: dict) -> None:
    case_id = case.get("get_application", {}).get("permit_number", "UNKNOWN")
    description = case.get("get_application", {}).get("proposed_use", "")

    label_id, label_desc = CASE_LABELS.get(case_id, (case_id, description))

    print(f"\n{'─' * 72}")
    print(f"  {label_id}")
    print(f"  {label_desc[:70]}")
    print(f"{'─' * 72}")

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="permit_intake",
        domain="permit_intake",
        case_input=case,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    # Classification result
    classify = trace.steps.get("classify_review_type", {}).get("output", {})
    if classify:
        category = classify.get("category", "?")
        confidence = classify.get("confidence", 0)
        reasoning = classify.get("reasoning", "")
        print(f"\n  CEQA classification:  {category.upper()}  "
              f"(confidence: {confidence:.2f})")
        # Extract the first statutory citation from reasoning
        if "§" in reasoning:
            citation_start = reasoning.find("§")
            citation_end = min(
                reasoning.find(".", citation_start),
                reasoning.find("\n", citation_start),
                citation_start + 60
            )
            if citation_end < citation_start:
                citation_end = citation_start + 60
            print(f"  Policy basis:         ...§{reasoning[citation_start+1:citation_end].strip()}")

    # Verify result (for exempt and prohibited cases)
    verify = trace.steps.get("verify_statutory_compliance", {}).get("output", {})
    if verify:
        conforms = verify.get("conforms", True)
        violations = verify.get("violations", [])
        status = "conforms" if conforms else f"VIOLATIONS: {len(violations)}"
        print(f"  Statutory compliance: {status}")
        for v in violations[:2]:
            rule = v.get("rule", "") if isinstance(v, dict) else str(v)
            print(f"    ✗ {str(rule)[:65]}")

    # Deliberation result
    deliberate = trace.steps.get("deliberate_determination", {}).get("output", {})
    if deliberate:
        action = deliberate.get("recommended_action", "?")
        conf = deliberate.get("confidence", 0)
        warrant = deliberate.get("warrant", "")
        print(f"  Determination:        {action}  (confidence: {conf:.2f})")
        if warrant:
            print(f"  Warrant:              {warrant[:70]}")

    # Governance
    gov = trace.steps.get("govern_determination", {}).get("output", {})
    tier = gov.get("tier_applied") or getattr(instance, "governance_tier", "?")
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "")
    tier_rationale = gov.get("tier_rationale", "")
    icon = TIER_ICONS.get(tier_str, "?")

    print(f"\n  Governance tier:      {icon}  {tier_str.upper()}")
    if disposition:
        print(f"  Disposition:          {disposition}")
    if tier_rationale:
        print(f"  Rationale:            {tier_rationale[:70]}")

    # Delegation (conditional/complex cases)
    category_val = classify.get("category", "") if classify else ""
    if category_val in ("conditional", "complex"):
        print(f"\n  Delegation:           permit_intake → [fire-and-forget] → eia_assessment")
        print(f"                        → [parallel] public_notice_compliance")
        print(f"                                   + biological_resources_review")
        print(f"                        → resume at deliberate_determination")
        print(f"                        → govern_eia_determination → GATE")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run permit review demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cases:
  exempt       PMT-2026-00142  Tenant improvement → §15301 Class 1 exemption
  conditional  PMT-2026-00318  Mixed-use infill   → Initial Study, MND pathway
  prohibited   PMT-2026-00447  Floodway industrial → §18.52.060(B)(1) bar

Examples:
  python demos/permit-review/run.py
  python demos/permit-review/run.py --permit exempt
  python demos/permit-review/run.py --case cases/pmt_2026_00447.json
        """
    )
    parser.add_argument("--case", type=Path, default=None,
                        help="Path to a case JSON file")
    parser.add_argument("--permit",
                        choices=["exempt", "conditional", "prohibited"],
                        default=None,
                        help="Run a specific case type")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Run execution MCP in dry-run mode (default: true)")
    parser.add_argument("--live", action="store_true", default=False,
                        help="Run execution MCP in live mode (sends real notifications)")
    args = parser.parse_args()

    # Resolve execution MCP to absolute path so coordinator can spawn it
    # regardless of the working directory the demo is launched from.
    mcp_script = DEMO_DIR / "execution_mcp.py"
    dry_run = not args.live
    dry_run_prefix = "DRY_RUN=true " if dry_run else ""
    execution_mcp_cmd = f"{dry_run_prefix}python {mcp_script}"

    coord = Coordinator(
        str(DEMO_DIR / "coordinator_config.yaml"),
        db_path=str(DEMO_DIR / "demo.db"),
        overrides={"execution_mcp_cmd": execution_mcp_cmd},
    )

    if args.case:
        with open(args.case) as f:
            cases = [json.load(f)]
    elif args.permit:
        with open(DEMO_DIR / "cases" / CASE_MAP[args.permit]) as f:
            cases = [json.load(f)]
    else:
        # Run all three cases
        cases = []
        for case_file in ["pmt_2026_00142.json", "pmt_2026_00318.json",
                          "pmt_2026_00447.json"]:
            with open(DEMO_DIR / "cases" / case_file) as f:
                cases.append(json.load(f))

    print("\n" + "═" * 72)
    print("  PERMIT REVIEW — Policy-Grounded Environmental Determination")
    print("  CEQA  ·  14 CCR  ·  Municipal Code  ·  44 CFR Part 60")
    print(f"  {len(cases)} case(s)  |  parallel-handlers coordinator")
    print("═" * 72)

    for case in cases:
        run_case(coord, case)

    print("═" * 72)
    print("  Every classification and verification finding cites a specific")
    print("  statutory provision retrieved from the regulatory corpus.")
    print("  The audit ledger records instrument, section, and operative")
    print("  language for every determination.")
    if any(
        json.load(open(DEMO_DIR / "cases" / "pmt_2026_00318.json")).get("_id") ==
        c.get("_id") for c in cases
    ):
        print()
        print("  The conditional case (PMT-2026-00318) exercises the full")
        print("  delegation chain: intake → EIA specialist → [parallel]")
        print("  public notice + biological resources → resume → govern.")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()