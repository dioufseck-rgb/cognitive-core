#!/usr/bin/env python3
"""
Hardship Chain Demo — End-to-End Delegation
============================================
Runs WF1 → WF2 → WF3 → WF4 through the coordinator, proving delegation.

Usage:
    python scripts/hardship_chain_demo.py [--case mh_007_non_english] [-v]
    python scripts/hardship_chain_demo.py --case mh_001_mixed_portfolio -v
"""

import json
import os
import sys
import time
import tempfile
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_workflow(coord, workflow, domain, case_input, correlation_id="", label=""):
    """Run a single workflow through the coordinator, handling suspension."""
    from coordinator.types import InstanceStatus

    t0 = time.time()
    instance_id = coord.start(
        workflow_type=workflow,
        domain=domain,
        case_input=case_input,
        correlation_id=correlation_id,
    )
    instance = coord.get_instance(instance_id)

    # Handle governance suspension (gate/hold tiers)
    if instance.status == InstanceStatus.SUSPENDED:
        print(f"    ⏸ Suspended (governance gate) → auto-approving")
        coord.approve(instance_id, approver="chain_demo")
        instance = coord.get_instance(instance_id)

    elapsed = round(time.time() - t0, 2)
    result = instance.result or {}
    steps = result.get("steps", [])

    # Extract artifact from the last generate step
    artifact = None
    for step in reversed(steps):
        if step.get("primitive") == "generate":
            raw = step.get("artifact") or step.get("artifact_preview", "")
            if isinstance(raw, dict):
                artifact = raw
            elif isinstance(raw, str):
                try:
                    artifact = json.loads(raw)
                except (ValueError, TypeError):
                    artifact = {"_raw": raw[:500]}
            break

    status = instance.status.value if hasattr(instance.status, "value") else str(instance.status)

    return {
        "instance_id": instance_id,
        "correlation_id": instance.correlation_id,
        "workflow": workflow,
        "domain": domain,
        "status": status,
        "steps": len(steps),
        "step_names": [s.get("step_name", "?") for s in steps],
        "elapsed": elapsed,
        "artifact": artifact,
        "artifact_keys": list(artifact.keys()) if artifact and isinstance(artifact, dict) else [],
    }


def run_chain(case_name, verbose=False):
    from coordinator.runtime import Coordinator

    config_path = os.path.join(PROJECT_ROOT, "coordinator", "config.yaml")
    db_path = os.path.join(tempfile.mkdtemp(), "chain_demo.db")
    coord = Coordinator(config_path=config_path, db_path=db_path, verbose=verbose)

    # Load case metadata only — actual data comes from fixture DB
    case_path = os.path.join(PROJECT_ROOT, "cases", "synthetic", f"{case_name}.json")
    with open(case_path) as f:
        case_data = json.load(f)
    meta = case_data.pop("_meta", {})

    # Extract member_id for DB-backed tool resolution
    profile = case_data.get("get_member_profile", {})
    member_id = profile.get("member_token", f"MBR-{case_name}")

    report = {
        "case": case_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": meta,
        "member_id": member_id,
        "data_source": "fixtures/cognitive_core.db",
        "stages": [],
        "chain_status": "running",
    }

    t_total = time.time()

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: Intake Packet
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"STAGE 1: hardship_intake_packet")
    print(f"{'═'*60}")

    # Pass only identifiers — tools resolve from DB, not JSON
    wf1_input = {
        "case_id": case_name,
        "member_id": member_id,
    }
    s1 = run_workflow(coord, "hardship_intake_packet", "member_hardship", wf1_input)
    report["stages"].append(s1)
    corr_id = s1["correlation_id"]

    triage = s1["artifact"].get("preliminary_triage") if s1["artifact"] else None
    print(f"  ✓ {s1['status']} | {s1['steps']} steps | {s1['elapsed']}s")
    print(f"    Steps: {' → '.join(s1['step_names'])}")
    print(f"    Triage: {triage}")
    print(f"    Artifact keys: {s1['artifact_keys']}")

    if not s1["artifact"]:
        report["chain_status"] = "failed_stage_1"
        return report

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: Eligibility & Constraints
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"STAGE 2: eligibility_constraints_assessment")
    print(f"{'═'*60}")

    wf2_input = {
        "intake_packet": s1["artifact"],
        "case_id": case_name,
        "member_id": member_id,
    }

    s2 = run_workflow(coord, "eligibility_constraints_assessment", "hardship_eligibility",
                      wf2_input, correlation_id=corr_id)
    report["stages"].append(s2)

    review = s2["artifact"].get("required_human_review") if s2["artifact"] else None
    print(f"  ✓ {s2['status']} | {s2['steps']} steps | {s2['elapsed']}s")
    print(f"    Steps: {' → '.join(s2['step_names'])}")
    print(f"    Human review: {review}")
    print(f"    Artifact keys: {s2['artifact_keys']}")

    if not s2["artifact"]:
        report["chain_status"] = "failed_stage_2"
        return report

    # ═══════════════════════════════════════════════════════════
    # STAGE 3: Path Recommendation
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"STAGE 3: hardship_path_recommendation")
    print(f"{'═'*60}")

    wf3_input = {
        "intake_packet": s1["artifact"],
        "constraints": s2["artifact"],
    }

    s3 = run_workflow(coord, "hardship_path_recommendation", "hardship_path",
                      wf3_input, correlation_id=corr_id)
    report["stages"].append(s3)

    path = s3["artifact"].get("selected_path") if s3["artifact"] else None
    confidence = s3["artifact"].get("confidence_score") if s3["artifact"] else None
    plans = len(s3["artifact"].get("plan_options", [])) if s3["artifact"] else 0
    print(f"  ✓ {s3['status']} | {s3['steps']} steps | {s3['elapsed']}s")
    print(f"    Steps: {' → '.join(s3['step_names'])}")
    print(f"    Path: {path}")
    print(f"    Confidence: {confidence}")
    print(f"    Plan options: {plans}")

    if not s3["artifact"]:
        report["chain_status"] = "failed_stage_3"
        return report

    # ═══════════════════════════════════════════════════════════
    # STAGE 4: Contact Strategy
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"STAGE 4: member_contact_strategy")
    print(f"{'═'*60}")

    wf4_input = {
        "intake_packet": s1["artifact"],
        "constraints": s2["artifact"],
        "recommendation": s3["artifact"],
    }

    s4 = run_workflow(coord, "member_contact_strategy", "hardship_contact",
                      wf4_input, correlation_id=corr_id)
    report["stages"].append(s4)

    channel = s4["artifact"].get("channel") if s4["artifact"] else None
    language = s4["artifact"].get("language") if s4["artifact"] else None
    escalation = s4["artifact"].get("escalation_needed") if s4["artifact"] else None
    print(f"  ✓ {s4['status']} | {s4['steps']} steps | {s4['elapsed']}s")
    print(f"    Steps: {' → '.join(s4['step_names'])}")
    print(f"    Channel: {channel}")
    print(f"    Language: {language}")
    print(f"    Escalation: {escalation}")

    # ═══════════════════════════════════════════════════════════
    # CHAIN SUMMARY
    # ═══════════════════════════════════════════════════════════
    total = round(time.time() - t_total, 2)
    report["total_time"] = total
    report["total_steps"] = sum(s["steps"] for s in report["stages"])

    # Correlation chain
    chain = coord.get_correlation_chain(corr_id)
    report["correlation_chain"] = [
        {"instance_id": c.instance_id, "workflow": c.workflow_type, "domain": c.domain,
         "status": c.status.value if hasattr(c.status, "value") else str(c.status)}
        for c in chain
    ]

    # Audit ledger
    ledger = coord.get_ledger(correlation_id=corr_id)
    report["audit_entries"] = len(ledger)

    all_ok = all(s["status"] in ("completed", "auto_approved") for s in report["stages"])
    report["chain_status"] = "completed" if all_ok else "partial"

    # Final summary
    report["summary"] = {
        "triage": triage,
        "human_review": review,
        "selected_path": path,
        "confidence": confidence,
        "plan_options": plans,
        "channel": channel,
        "language": language,
        "escalation_needed": escalation,
    }

    print(f"\n{'═'*60}")
    print(f"CHAIN {'COMPLETE ✅' if all_ok else 'PARTIAL ⚠️'}")
    print(f"{'═'*60}")
    print(f"  Stages: {len(report['stages'])}/4")
    print(f"  Total steps: {report['total_steps']}")
    print(f"  Total time: {total}s")
    print(f"  Audit entries: {report['audit_entries']}")
    print(f"  Correlation chain: {len(report['correlation_chain'])} instances")
    print()
    print(f"  ┌─ WF1 Intake ────────── triage = {triage}")
    print(f"  ├─ WF2 Eligibility ───── review = {review}")
    print(f"  ├─ WF3 Recommendation ── path = {path}, confidence = {confidence}")
    print(f"  └─ WF4 Contact ───────── channel = {channel}, language = {language}")
    print()

    return report


def main():
    parser = argparse.ArgumentParser(description="Hardship Chain Demo")
    parser.add_argument("--case", default="mh_007_non_english")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output", default="chain_demo_results.json")
    args = parser.parse_args()

    report = run_chain(args.case, args.verbose)

    out = os.path.join(PROJECT_ROOT, args.output)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {out}")


if __name__ == "__main__":
    main()
