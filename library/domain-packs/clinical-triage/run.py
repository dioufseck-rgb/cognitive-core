"""
Clinical Triage — Domain Pack Runner

Assess urgency, investigate complex presentations, generate triage
disposition scripts, and govern escalation with clinician sign-off.

Usage (from repo root):
    python library/domain-packs/clinical-triage/run.py
    python library/domain-packs/clinical-triage/run.py --case-id TRIAGE-001
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


def run_contact(coord: Coordinator, contact: dict) -> None:
    print(f"\n{'─' * 68}")
    print(f"  {contact.get('_id', '?')}  |  Age {contact.get('patient_age', '?')}")
    print(f"  Chief complaint: {contact.get('chief_complaint', '')}")
    print(f"  Symptoms: {contact.get('symptoms', '')[:80]}...")
    if contact.get("_expected"):
        print(f"  Expected: {contact['_expected']}")
    print(f"{'─' * 68}")

    case_input = {k: v for k, v in contact.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="triage_and_escalation",
        domain="clinical_triage",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    steps_run = list(trace.steps.keys())
    print(f"  Steps run:     {' → '.join(steps_run)}")

    classify = trace.steps.get("classify_severity", {}).get("output", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "?"
        print(f"  Severity:      {cat}  (confidence {conf_str})")

    investigate = trace.steps.get("investigate_situation", {}).get("output", {})
    if investigate:
        finding = str(investigate.get("finding", ""))
        print(f"  Investigation: {finding[:100]}{'...' if len(finding) > 100 else ''}")

    generate = trace.steps.get("generate_response", {}).get("output", {})
    if generate:
        artifact = str(generate.get("artifact") or generate.get("content") or "")
        print(f"  Disposition:   {artifact[:120]}{'...' if len(artifact) > 120 else ''}")

    gov = trace.steps.get("govern_escalation", {}).get("output", {})
    tier = gov.get("tier_applied") or getattr(instance, "governance_tier", "?")
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    icon = TIER_ICONS.get(tier_str, "?")
    print(f"  Governance:    {icon} {tier_str.upper()}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict):
        print(f"  Work order:    → {work_order.get('target', '?')} queue")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "pack.db"),
    )

    with open(PACK_DIR / "cases" / "example_contacts.json") as f:
        contacts = json.load(f)

    if args.case_id:
        contacts = [c for c in contacts if c.get("_id") == args.case_id]
        if not contacts:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 68)
    print("  CLINICAL TRIAGE — Domain Pack")
    print("  classify → [investigate] → generate → govern")
    print(f"  {len(contacts)} contact(s)  |  human-review overlay active")
    print("═" * 68)

    for contact in contacts:
        run_contact(coord, contact)

    print(f"\n{'═' * 68}\n")


if __name__ == "__main__":
    main()
