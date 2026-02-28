#!/usr/bin/env python3
"""
Batch Test Runner for Insurance Claim Adjudication
===================================================

Runs each case N times with scripted human task responses.
Captures structured results for analysis.

Usage:
    python run_batch_test.py                    # all cases, 5 runs each
    python run_batch_test.py --case hard --n 10 # hard case, 10 runs
    python run_batch_test.py --case simple medium hard --n 3
"""

import argparse
import json
import time
import os
import sys
import copy
import traceback
from datetime import datetime
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────

CASES = {
    "simple": {
        "file": "cases/insurance_claim_simple.json",
        "workflow": "claim_adjudication",
        "domain": "claims_processing",
    },
    "medium": {
        "file": "cases/insurance_claim_medium.json",
        "workflow": "claim_adjudication",
        "domain": "claims_processing",
    },
    "hard": {
        "file": "cases/insurance_claim_hard.json",
        "workflow": "claim_adjudication",
        "domain": "claims_processing",
    },
}

HUMAN_RESPONSES_FILE = "cases/fixtures/human_task_responses.json"


# ── Human Task Auto-Completer ─────────────────────────────────────

class HumanTaskAutoCompleter:
    """
    Intercepts human task publications and auto-completes them with
    scripted responses. Patches the coordinator's human task handler.
    """

    def __init__(self, responses: dict, case_name: str):
        self.responses = responses
        self.case_name = case_name
        self.completed_tasks = []

    def get_response(self, need: str, context: dict) -> dict:
        """Get the scripted response for a human task need."""
        task_data = self.responses.get(need, {})

        # New fixture structure: responses keyed by case name
        if self.case_name in task_data:
            response = task_data[self.case_name]
        elif "variants" in task_data:
            # Legacy structure fallback
            variants = task_data["variants"]
            if need == "forensic_accounting_review":
                response = variants.get(
                    "greenfield_standard" if self.case_name == "medium" else "standard", {}
                )
            elif need == "coverage_specialist_review":
                response = variants.get("exclusion_partial", {})
            elif need == "independent_appraisal":
                response = variants.get("standard", {})
            else:
                response = {}
        else:
            response = {}

        if not response:
            response = {
                "findings": f"Human task '{need}' completed. No specific data.",
                "status": "completed"
            }

        self.completed_tasks.append({
            "need": need,
            "case_name": self.case_name,
        })
        return response


# ── Result Extractor ──────────────────────────────────────────────

def extract_run_result(coordinator, instance_id: str, case_data: dict) -> dict:
    """Extract structured results from a completed/failed/suspended run."""

    instance = coordinator.store.get_instance(instance_id)
    if not instance:
        return {"status": "error", "error": "Instance not found"}

    result = {
        "instance_id": instance_id,
        "status": instance.status.value,
        "elapsed_seconds": 0,
    }

    # Governance gate suspension = workflow completed, awaiting human approval
    if instance.status.value == "suspended":
        suspension = coordinator.store.get_suspension(instance_id)
        if suspension and suspension.suspended_at_step == "__governance_gate__":
            result["status"] = "completed_pending_approval"
            result["governance_gate"] = True

    # Timing
    if hasattr(instance, 'created_at') and hasattr(instance, 'updated_at'):
        result["elapsed_seconds"] = round(instance.updated_at - instance.created_at, 1)

    # Count delegations from the correlation chain
    delegations = []
    actions = coordinator.store.get_ledger(correlation_id=instance.correlation_id)
    
    # Get all work orders for this correlation
    all_wo_ids = set()
    if actions:
        for a in actions:
            # Work order IDs may be in action details
            details = a.get('details', {}) or {}
            wo_id = details.get('work_order_id', '')
            if wo_id:
                all_wo_ids.add(wo_id)
    
    # Also check pending_work_orders on instance
    for wo_id in (instance.pending_work_orders or []):
        all_wo_ids.add(wo_id)
    
    # Try to find work orders from the store
    for wo_id in all_wo_ids:
        wo = coordinator.store.get_work_order(wo_id)
        if wo:
            delegations.append({
                "work_order_id": wo_id,
                "status": wo.status.value,
                "contract": wo.contract_name,
            })
    result["delegations"] = delegations
    result["delegation_count"] = len(delegations)

    # Count unique needs dispatched — extract from action log
    needs_dispatched = []
    if actions:
        for a in actions:
            if a['action_type'] == "interrupted_for_resources":
                details = a.get('details', {}) or {}
                needs = details.get('needs', [])
                if isinstance(needs, list):
                    needs_dispatched.extend(needs)
    result["unique_needs_dispatched"] = list(set(needs_dispatched))

    # Audit ledger
    actions = coordinator.store.get_ledger(correlation_id=instance.correlation_id)
    result["audit_entries"] = len(actions)
    result["action_types"] = [a['action_type'] for a in actions] if actions else []

    # Count interrupts and resumes
    result["interrupt_count"] = sum(
        1 for a in (actions or []) if a['action_type'] == "interrupted_for_resources"
    )
    result["resume_count"] = sum(
        1 for a in (actions or []) if a['action_type'] == "resume"
    )

    # Extract final output if completed
    if instance.result and isinstance(instance.result, dict):
        steps = instance.result.get("steps", [])
        result["steps_completed"] = len(steps)
        result["step_names"] = [s.get("step_name", "") for s in steps]

        # Find settlement recommendation
        for step in steps:
            if step.get("step_name") == "settlement_recommendation":
                # In result summary, artifact is a direct field
                artifact = step.get("artifact") or step.get("output", {}).get("artifact", "")
                if isinstance(artifact, str) and artifact.strip():
                    try:
                        settlement = json.loads(artifact)
                        result["settlement"] = settlement
                    except (json.JSONDecodeError, TypeError):
                        result["settlement_raw"] = artifact[:500]
                elif isinstance(artifact, dict):
                    result["settlement"] = artifact

            if step.get("step_name") == "compliance_check":
                # conforms might be direct or under output
                conforms = step.get("conforms")
                if conforms is None:
                    conforms = step.get("output", {}).get("conforms")
                result["compliance_conforms"] = conforms
                result["compliance_violations"] = (
                    step.get("violations") or
                    step.get("output", {}).get("violations", [])
                )

        # Extract key financial figures from settlement
        settlement = result.get("settlement", {})
        if settlement:
            result["net_payable"] = settlement.get("net_payable")
            result["total_verified"] = settlement.get("total_verified")
            result["subrogation_recommended"] = settlement.get(
                "subrogation", {}
            ).get("recommended")
    else:
        result["steps_completed"] = 0
        result["step_names"] = []

    # Ground truth comparison
    ground_truth = case_data.get("_ground_truth", {})
    if ground_truth and result.get("net_payable") is not None:
        expected = ground_truth.get("net_payable")
        actual = result["net_payable"]
        if expected and actual:
            tolerance = ground_truth.get("tolerance_pct", 15) / 100
            deviation = abs(actual - expected) / expected
            result["ground_truth_deviation_pct"] = round(deviation * 100, 1)
            result["within_tolerance"] = deviation <= tolerance

    return result


# ── Single Run ────────────────────────────────────────────────────

def run_single(
    case_name: str,
    case_config: dict,
    human_responses: dict,
    run_number: int,
    max_human_tasks: int = 5,
) -> dict:
    """Execute a single run of a case. Returns structured result."""

    case_file = case_config["file"]
    with open(case_file) as f:
        case_data = json.load(f)

    completer = HumanTaskAutoCompleter(human_responses, case_name)

    run_result = {
        "case": case_name,
        "run_number": run_number,
        "timestamp": datetime.now().isoformat(),
        "case_file": case_file,
        "human_tasks_completed": [],
    }

    try:
        from coordinator.runtime import Coordinator
        from coordinator.types import WorkOrderStatus, WorkOrderResult

        coord = Coordinator(config_path="coordinator/config.yaml")

        # Start the workflow
        instance_id = coord.start(
            workflow_type=case_config["workflow"],
            domain=case_config["domain"],
            case_input=case_data,
        )

        # Auto-complete human tasks in a loop
        human_task_count = 0
        max_iterations = 20  # safety cap

        for _ in range(max_iterations):
            instance = coord.store.get_instance(instance_id)
            if not instance:
                break
            if instance.status.value not in ("suspended",):
                break  # completed, failed, or running

            # Get the suspension record
            suspension = coord.store.get_suspension(instance_id)
            if not suspension:
                break

            # ── Governance gate suspension = workflow completed successfully ──
            # The workflow finished all steps and is now waiting for human
            # approval. For batch testing, we treat this as a completed run
            # and extract results from the suspended state.
            if suspension.suspended_at_step == "__governance_gate__":
                break  # Exit loop — results extracted below from suspension state

            # Find human task work orders (those with no handler_instance_id)
            human_wos = []
            for wo_id in suspension.work_order_ids:
                wo = coord.store.get_work_order(wo_id)
                if not wo:
                    continue
                if wo.status.value == "running" and not wo.handler_instance_id:
                    human_wos.append(wo)

            if not human_wos:
                # Not suspended for human tasks — could be waiting for
                # a provider workflow that is itself suspended (governance gate)
                break

            # Auto-complete each human task
            for wo in human_wos:
                # Resolve the need name from wo_need_map
                need = suspension.wo_need_map.get(wo.work_order_id, "unknown")
                context = wo.inputs

                response = completer.get_response(need, context)

                wo.status = WorkOrderStatus.COMPLETED
                wo.completed_at = time.time()
                wo.result = WorkOrderResult(
                    work_order_id=wo.work_order_id,
                    status="completed",
                    outputs=response,
                    completed_at=time.time(),
                )
                coord.store.save_work_order(wo)
                human_task_count += 1

                run_result["human_tasks_completed"].append({
                    "need": need,
                    "work_order_id": wo.work_order_id,
                })

            # Build external_input dict: {wo_id: outputs, need_name: outputs}
            external_input = {}
            all_providers_done = True
            for wo_id in suspension.work_order_ids:
                wo_check = coord.store.get_work_order(wo_id)
                if not wo_check:
                    continue
                if wo_check.status == WorkOrderStatus.COMPLETED and wo_check.result:
                    external_input[wo_id] = wo_check.result.outputs
                    # Also key by need name
                    need_name = suspension.wo_need_map.get(wo_id, "")
                    if need_name:
                        external_input[need_name] = wo_check.result.outputs
                elif wo_check.status.value in ("running", "created"):
                    all_providers_done = False

            if not all_providers_done:
                # Some provider workflows still running (not human tasks)
                break

            # Resume the workflow
            try:
                coord.resume(
                    instance_id=instance_id,
                    external_input=external_input,
                    resume_nonce=suspension.resume_nonce,
                )
            except Exception as e:
                run_result.setdefault("resume_errors", []).append(str(e))
                break

        # Extract results
        instance = coord.store.get_instance(instance_id)
        # The instance might have been resumed and re-suspended at a later step.
        # Follow the chain: if it completed, great. If suspended again,
        # the loop continues. If failed, we capture that.
        run_result.update(extract_run_result(coord, instance_id, case_data))

    except Exception as e:
        run_result["status"] = "error"
        run_result["error"] = str(e)
        run_result["traceback"] = traceback.format_exc()

    return run_result


# ── Batch Runner ──────────────────────────────────────────────────

def run_batch(
    cases: list[str],
    n: int,
    output_dir: str = "test_results",
) -> dict:
    """Run all specified cases N times each. Returns summary."""

    os.makedirs(output_dir, exist_ok=True)

    # Load human task responses
    with open(HUMAN_RESPONSES_FILE) as f:
        human_data = json.load(f)
    human_responses = human_data.get("human_task_responses", {})

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    print(f"\n{'='*70}")
    print(f"  BATCH TEST: {batch_id}")
    print(f"  Cases: {cases}")
    print(f"  Runs per case: {n}")
    print(f"{'='*70}\n")

    for case_name in cases:
        if case_name not in CASES:
            print(f"  ⚠ Unknown case: {case_name}, skipping")
            continue

        case_config = CASES[case_name]
        print(f"  ── {case_name.upper()} ({case_config['file']}) ──")

        for run_num in range(1, n + 1):
            print(f"    Run {run_num}/{n}...", end=" ", flush=True)
            start = time.time()

            result = run_single(
                case_name=case_name,
                case_config=case_config,
                human_responses=human_responses,
                run_number=run_num,
            )

            elapsed = time.time() - start
            result["wall_clock_seconds"] = round(elapsed, 1)
            all_results.append(result)

            status = result.get("status", "?")
            net = result.get("net_payable", "—")
            interrupts = result.get("interrupt_count", "?")
            steps = result.get("steps_completed", 0)
            deviation = result.get("ground_truth_deviation_pct", "—")

            net_str = f"${net:,.0f}" if isinstance(net, (int, float)) else str(net)
            dev_str = f"{deviation}%" if isinstance(deviation, (int, float)) else deviation

            print(
                f"{status:10s} | {steps} steps | {interrupts} interrupts | "
                f"net={net_str:>10s} | dev={dev_str:>6s} | {elapsed:.1f}s"
            )

        print()

    # ── Write Raw Results ──
    raw_path = os.path.join(output_dir, f"batch_{batch_id}_raw.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Raw results: {raw_path}")

    # ── Generate Summary ──
    summary = generate_summary(all_results, cases, n)
    summary_path = os.path.join(output_dir, f"batch_{batch_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # ── Print Summary Table ──
    print_summary_table(summary)

    return summary


# ── Summary Generation ────────────────────────────────────────────

def generate_summary(results: list[dict], cases: list[str], n: int) -> dict:
    """Generate analysis summary from batch results."""

    summary = {
        "batch_info": {
            "timestamp": datetime.now().isoformat(),
            "cases": cases,
            "runs_per_case": n,
            "total_runs": len(results),
        },
        "per_case": {},
        "overall": {},
    }

    for case_name in cases:
        case_results = [r for r in results if r.get("case") == case_name]
        if not case_results:
            continue

        cs = {
            "runs": len(case_results),
            "statuses": {},
            "completion_rate": 0,
            "avg_steps": 0,
            "avg_interrupts": 0,
            "avg_delegations": 0,
            "avg_elapsed_seconds": 0,
            "net_payable_values": [],
            "net_payable_stats": {},
            "subrogation_decisions": {},
            "ground_truth_deviations": [],
            "within_tolerance_rate": 0,
            "unique_needs_seen": {},
            "compliance_pass_rate": 0,
            "failure_modes": [],
        }

        completed = 0
        compliance_pass = 0

        for r in case_results:
            # Status counts
            status = r.get("status", "unknown")
            cs["statuses"][status] = cs["statuses"].get(status, 0) + 1

            if status in ("completed", "completed_pending_approval"):
                completed += 1

            # Steps, interrupts, delegations
            cs["avg_steps"] += r.get("steps_completed", 0)
            cs["avg_interrupts"] += r.get("interrupt_count", 0)
            cs["avg_delegations"] += r.get("delegation_count", 0)
            cs["avg_elapsed_seconds"] += r.get("wall_clock_seconds", 0)

            # Net payable
            net = r.get("net_payable")
            if net is not None and isinstance(net, (int, float)):
                cs["net_payable_values"].append(net)

            # Subrogation
            sub = r.get("subrogation_recommended")
            if sub is not None:
                sub_key = str(sub)
                cs["subrogation_decisions"][sub_key] = (
                    cs["subrogation_decisions"].get(sub_key, 0) + 1
                )

            # Ground truth deviation
            dev = r.get("ground_truth_deviation_pct")
            if dev is not None:
                cs["ground_truth_deviations"].append(dev)

            if r.get("within_tolerance"):
                compliance_pass += 1  # reusing variable loosely

            # Compliance
            if r.get("compliance_conforms") is True:
                compliance_pass += 1

            # Needs dispatched
            for need in r.get("unique_needs_dispatched", []):
                cs["unique_needs_seen"][need] = (
                    cs["unique_needs_seen"].get(need, 0) + 1
                )

            # Failure modes
            if status == "failed":
                error = r.get("error", "unknown")
                cs["failure_modes"].append(error[:200])

        n_runs = len(case_results)
        cs["completion_rate"] = round(completed / n_runs, 2) if n_runs else 0
        cs["avg_steps"] = round(cs["avg_steps"] / n_runs, 1) if n_runs else 0
        cs["avg_interrupts"] = round(cs["avg_interrupts"] / n_runs, 1) if n_runs else 0
        cs["avg_delegations"] = round(cs["avg_delegations"] / n_runs, 1) if n_runs else 0
        cs["avg_elapsed_seconds"] = round(cs["avg_elapsed_seconds"] / n_runs, 1) if n_runs else 0

        # Net payable stats
        vals = cs["net_payable_values"]
        if vals:
            cs["net_payable_stats"] = {
                "min": min(vals),
                "max": max(vals),
                "mean": round(sum(vals) / len(vals), 0),
                "spread": max(vals) - min(vals),
                "spread_pct": round(
                    (max(vals) - min(vals)) / (sum(vals) / len(vals)) * 100, 1
                ) if sum(vals) > 0 else 0,
                "count": len(vals),
            }

        # Deviation stats
        devs = cs["ground_truth_deviations"]
        if devs:
            cs["deviation_stats"] = {
                "min": round(min(devs), 1),
                "max": round(max(devs), 1),
                "mean": round(sum(devs) / len(devs), 1),
                "count": len(devs),
            }
            within = sum(1 for d in devs if d <= 15)  # 15% tolerance
            cs["within_tolerance_rate"] = round(within / len(devs), 2)

        summary["per_case"][case_name] = cs

    # Overall stats
    total = len(results)
    completed_total = sum(
        1 for r in results if r.get("status") == "completed"
    )
    summary["overall"] = {
        "total_runs": total,
        "completed": completed_total,
        "completion_rate": round(completed_total / total, 2) if total else 0,
        "total_llm_calls": "see raw results",
        "avg_wall_clock": round(
            sum(r.get("wall_clock_seconds", 0) for r in results) / total, 1
        ) if total else 0,
    }

    return summary


# ── Pretty Print ──────────────────────────────────────────────────

def print_summary_table(summary: dict):
    """Print a human-readable summary table."""

    print(f"\n{'='*70}")
    print(f"  BATCH SUMMARY")
    print(f"{'='*70}")

    overall = summary.get("overall", {})
    print(f"\n  Total runs: {overall.get('total_runs', 0)}")
    print(f"  Completed:  {overall.get('completed', 0)} ({overall.get('completion_rate', 0):.0%})")
    print(f"  Avg time:   {overall.get('avg_wall_clock', 0):.1f}s")

    for case_name, cs in summary.get("per_case", {}).items():
        print(f"\n  ── {case_name.upper()} ──")
        print(f"  Runs:         {cs['runs']}")
        print(f"  Completion:   {cs['completion_rate']:.0%}")
        print(f"  Avg steps:    {cs['avg_steps']}")
        print(f"  Avg interrupts: {cs['avg_interrupts']}")
        print(f"  Avg delegations: {cs['avg_delegations']}")
        print(f"  Avg elapsed:  {cs['avg_elapsed_seconds']:.1f}s")

        print(f"  Statuses:     {cs['statuses']}")

        nps = cs.get("net_payable_stats", {})
        if nps:
            print(f"  Net payable:")
            print(f"    Range:    ${nps.get('min', 0):,.0f} — ${nps.get('max', 0):,.0f}")
            print(f"    Mean:     ${nps.get('mean', 0):,.0f}")
            print(f"    Spread:   ${nps.get('spread', 0):,.0f} ({nps.get('spread_pct', 0):.1f}%)")

        ds = cs.get("deviation_stats")
        if ds:
            print(f"  Ground truth deviation:")
            print(f"    Range:    {ds['min']}% — {ds['max']}%")
            print(f"    Mean:     {ds['mean']}%")
            print(f"    Within 15%: {cs['within_tolerance_rate']:.0%}")

        if cs.get("subrogation_decisions"):
            print(f"  Subrogation:  {cs['subrogation_decisions']}")

        if cs.get("unique_needs_seen"):
            print(f"  Resources requested:")
            for need, count in sorted(
                cs["unique_needs_seen"].items(), key=lambda x: -x[1]
            ):
                pct = count / cs["runs"] * 100
                print(f"    {need}: {count}/{cs['runs']} ({pct:.0f}%)")

        if cs.get("failure_modes"):
            print(f"  Failure modes:")
            for fm in cs["failure_modes"][:3]:
                print(f"    • {fm[:100]}")

    print(f"\n{'='*70}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch test runner")
    parser.add_argument(
        "--case", nargs="+", default=["simple", "medium", "hard"],
        help="Cases to run (simple, medium, hard)"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of runs per case"
    )
    parser.add_argument(
        "--output", default="test_results",
        help="Output directory"
    )
    args = parser.parse_args()

    run_batch(cases=args.case, n=args.n, output_dir=args.output)


if __name__ == "__main__":
    main()
