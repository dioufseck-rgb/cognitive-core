#!/usr/bin/env python3
"""
Cognitive Core — Live SAR Batch Demo (Real LLM)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Runs 3 real SAR investigations through the full Cognitive Core stack:
  • Real LLM (Google Gemini) executing classify → investigate → generate
  • Real coordinator state machine with suspend/resume
  • Real dispatch optimizer with cost matrix and solver
  • Real batch capacity model with time-window trigger
  • Real backpressure queue with drain-on-capacity
  • Full audit trail: DDRs, reservation events, action ledger

Setup:
  pip install langchain-google-genai
  export GOOGLE_API_KEY=your-key-here
  export LLM_PROVIDER=google
  python3 demo_batch_sar_live.py

The demo processes 3 SAR cases:
  1. Marcus Chen — structuring ($87K in cash deposits below CTR threshold)
  2. Diana Reeves — unusual wire activity (Cayman Islands → brokerage)
  3. Robert & Lisa Tran — structuring + layering ($34K cash → cashier's check)

Timeline:
  • Cases 1-2 arrive → batch collects (COLLECTING)
  • Time window expires → batch fires with 2 cases
  • Case 3 arrives during execution → QUEUED
  • Batch 1 completes → queue drains → Case 3 dispatched
  • Each case runs the full sar_investigation workflow with real LLM
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Terminal colors ──
B, R = "\033[1m", "\033[0m"
GRN, RED, YEL, CYN, GRY, ORN = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[38;5;208m"
)
BLU = "\033[94m"

def banner(text, c=BLU):
    w = max(len(text) + 4, 64)
    print(f"\n{c}{'━' * w}")
    print(f"  {text}")
    print(f"{'━' * w}{R}\n")

def timeline(t, msg):
    print(f"  {GRY}t={t:>5.1f}s{R}  {msg}")

def status(label, value, color=GRN):
    print(f"           {GRY}{label}:{R} {color}{value}{R}")

def section(msg):
    print(f"\n  {CYN}── {msg} ──{R}\n")


def check_prerequisites():
    """Verify LLM provider is available."""
    provider = os.environ.get("LLM_PROVIDER", "google")
    if provider == "google":
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            print(f"\n  {RED}ERROR: GOOGLE_API_KEY not set.{R}")
            print(f"  {GRY}export GOOGLE_API_KEY=your-key-here{R}")
            print(f"  {GRY}export LLM_PROVIDER=google{R}")
            sys.exit(1)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            print(f"\n  {RED}ERROR: langchain-google-genai not installed.{R}")
            print(f"  {GRY}pip install langchain-google-genai{R}")
            sys.exit(1)
    # Quick smoke test
    try:
        from engine.llm import create_llm
        llm = create_llm(model="default", temperature=0.1)
        print(f"  {GRN}✓{R} LLM available: {type(llm).__name__}")
    except Exception as e:
        print(f"\n  {RED}ERROR: Cannot create LLM: {e}{R}")
        sys.exit(1)


def run_sar_workflow(coord, case_data, case_label, max_iterations=30):
    """
    Run a single SAR investigation through the full workflow.
    Returns (instance_id, result_dict, elapsed_seconds).
    """
    t0 = time.time()
    print(f"    {B}Starting:{R} {case_label}")
    status("alert_id", case_data["alert_id"])
    status("type", case_data["alert_type"])
    status("subject", case_data["subject"]["name"])
    status("risk_score", str(case_data["risk_score"]))

    try:
        instance_id = coord.start(
            workflow_type="sar_investigation",
            domain="structuring_sar",
            case_input=case_data,
        )
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    {RED}✗ Failed to start: {e}{R}")
        return None, {"error": str(e)}, elapsed

    # Poll for completion
    for iteration in range(max_iterations):
        instance = coord.store.get_instance(instance_id)
        if not instance:
            break

        if instance.status.value == "completed":
            elapsed = time.time() - t0
            result = instance.result or {}
            # Extract key outputs
            summary = {}
            if isinstance(result, dict):
                for key in result:
                    val = result[key]
                    if isinstance(val, dict):
                        summary[key] = {
                            k: str(v)[:100] for k, v in val.items()
                            if k in ("category", "confidence", "finding",
                                     "artifact", "conforms", "survives",
                                     "reasoning", "decision")
                        }
                    elif isinstance(val, str) and len(val) > 200:
                        summary[key] = val[:200] + "..."
                    else:
                        summary[key] = val
            print(f"    {GRN}✓ Completed{R} in {elapsed:.1f}s")
            return instance_id, summary, elapsed

        elif instance.status.value == "failed":
            elapsed = time.time() - t0
            error = getattr(instance, 'error', '') or 'unknown'
            print(f"    {RED}✗ Failed:{R} {str(error)[:200]}")
            return instance_id, {"error": str(error)[:500]}, elapsed

        elif instance.status.value == "suspended":
            # Check if governance gate (workflow done, awaiting approval)
            sus = coord.store.get_suspension(instance_id)
            if sus and sus.suspended_at_step == "__governance_gate__":
                elapsed = time.time() - t0
                result = getattr(sus, 'state_snapshot', {}) or {}
                print(f"    {GRN}✓ Completed (governance gate){R} in {elapsed:.1f}s")
                summary = {}
                for key in result:
                    val = result[key]
                    if isinstance(val, dict):
                        summary[key] = {
                            k: str(v)[:100]
                            for k, v in val.items()
                            if k in ("category", "confidence", "finding",
                                     "artifact", "conforms", "survives",
                                     "reasoning", "decision")
                        }
                return instance_id, summary, elapsed

            # Suspended for resources — check work orders
            if sus:
                for wo_id in sus.work_order_ids:
                    wo = coord.store.get_work_order(wo_id)
                    if wo and wo.status.value == "running" and not wo.handler_instance_id:
                        # Human task — auto-approve for demo
                        from coordinator.types import WorkOrderResult
                        wo.status = coord.types_module.WorkOrderStatus.COMPLETED \
                            if hasattr(coord, 'types_module') else __import__(
                                'coordinator.types', fromlist=['WorkOrderStatus']
                            ).WorkOrderStatus.COMPLETED
                        wo.result = WorkOrderResult(
                            work_order_id=wo.work_order_id,
                            status="completed",
                            outputs={"approved": True, "notes": "Auto-approved for demo"},
                        )
                        coord.store.save_work_order(wo)
                        coord._try_resume_after_all_providers(instance, sus)
                        break

        time.sleep(0.5)  # brief pause between polls

    elapsed = time.time() - t0
    instance = coord.store.get_instance(instance_id)
    final_status = instance.status.value if instance else "unknown"
    print(f"    {YEL}⚠ Timed out after {elapsed:.1f}s (status: {final_status}){R}")
    return instance_id, {"status": final_status, "timeout": True}, elapsed


def run():
    banner("COGNITIVE CORE — LIVE SAR BATCH DEMO (REAL LLM)")

    provider = os.environ.get("LLM_PROVIDER", "google")
    print(f"  {B}Provider:{R}  {provider}")
    print(f"  {B}Workflow:{R}  sar_investigation / structuring_sar")
    print(f"  {B}Cases:{R}     3 SAR alerts (structuring + unusual wire)")
    print(f"  {B}Batch:{R}     threshold=5, time_window=60s")
    print(f"  {B}Stack:{R}     Coordinator → Optimizer → Batch Capacity → LLM")
    print()

    check_prerequisites()

    # Load cases
    with open(os.path.join(_project_root, "cases", "sar_batch_cases.json")) as f:
        cases = json.load(f)

    # ═══════════════════════════════════════════════════════════════
    # SETUP: Coordinator + Optimizer + Batch Resource
    # ═══════════════════════════════════════════════════════════════

    section("SETUP")

    from coordinator.runtime import Coordinator
    from coordinator.ddd import (
        ResourceRegistration, ResourceRegistry,
        CapacityModel, CapacityState, BatchStatus,
        DDDWorkOrder,
    )
    from coordinator.optimizer import DispatchOptimizer
    from coordinator.physics import OptimizationConfig
    from coordinator.hardening import ReservationEventLog, LearningScopeEnforcer
    from coordinator.types import WorkOrderStatus

    coord = Coordinator(config_path="coordinator/config.yaml", verbose=True)

    # Register batch resource in optimizer
    batch_cap = CapacityState(
        model=CapacityModel.BATCH,
        batch_threshold=5,
        batch_timeout_seconds=60.0,
        max_execution_duration_seconds=300.0,
    )
    batch_resource = ResourceRegistration(
        resource_id="sar_ml_screener",
        resource_type="automated",
        capabilities=[("sar_screen", "structuring_sar")],
        capacity=batch_cap,
        attributes={"cost_rate": 5.0, "quality_score": 0.92},
        completed_work_orders=847,
    )
    coord._resource_registry.register(batch_resource)

    # Optimizer hooks
    res_log = ReservationEventLog()
    ddr_records = []
    def capture_ddr(decision, eligible, audit, config, solution):
        ddr_records.append({
            "wo_id": decision.work_order_id,
            "resource": decision.selected_resource_id,
            "tier": decision.tier,
        })

    optimizer = DispatchOptimizer(
        coord._resource_registry,
        ddr_callback=capture_ddr,
        reservation_log=res_log,
        learning_enforcer=LearningScopeEnforcer(),
    )
    opt_config = OptimizationConfig(
        objectives={
            "minimize_cost": 0.1,
            "minimize_wait_time": 0.3,
            "minimize_sla_risk": 0.6,
        },
        exploration_enabled=False,
    )

    print(f"  {GRN}✓{R} Coordinator initialized")
    print(f"  {GRN}✓{R} Batch resource registered (threshold=5, window=60s)")
    print(f"  {GRN}✓{R} Optimizer with DDR + reservation log + learning enforcer")

    # Track results
    all_results = {}
    total_t0 = time.time()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: CASES 1-2 ARRIVE → BATCH COLLECTS
    # ═══════════════════════════════════════════════════════════════

    banner("PHASE 1: CASES 1-2 → BATCH COLLECTS", ORN)

    sim_t0 = time.time()
    batch_cap.batch_collecting_since = sim_t0

    for i, case in enumerate(cases[:2]):
        elapsed = time.time() - sim_t0

        # Optimizer dispatch
        ddd_wo = DDDWorkOrder.create(
            "sar_investigation", f"cor_{case['alert_id']}", "sar_screen",
            priority="high", sla_seconds=86400.0,
            case_id=case["alert_id"],
        )
        decisions = optimizer.dispatch([ddd_wo], "sar_screen", "structuring_sar", opt_config)
        d = decisions[0]

        timeline(elapsed, f"{GRN}✓{R} {B}{case['alert_id']}{R} "
                 f"({case['alert_type']}) → {d.selected_resource_id or 'NONE'}")

        # Run the actual workflow with real LLM
        instance_id, result, wf_elapsed = run_sar_workflow(
            coord, case,
            f"{case['alert_id']} — {case['subject']['name']}",
        )
        all_results[case["alert_id"]] = {
            "instance_id": instance_id,
            "result": result,
            "elapsed": wf_elapsed,
            "optimizer_tier": d.tier,
        }

    print(f"\n  {CYN}Batch state:{R} {batch_cap.batch_items} items, "
          f"status={batch_cap.batch_status.value}")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: SIMULATE TIME WINDOW + BATCH EXECUTION
    # ═══════════════════════════════════════════════════════════════

    banner("PHASE 2: TIME WINDOW FIRES", YEL)

    # Force time window (in real deployment this would be a sweep tick)
    batch_cap.batch_collecting_since = time.time() - 61  # pretend 61s ago
    triggered = batch_cap.check_batch_trigger()
    timeline(time.time() - sim_t0,
             f"Time window check: {GRN if triggered else RED}{triggered}{R}")

    if triggered:
        batch_cap.trigger_batch()
        timeline(time.time() - sim_t0,
                 f"{YEL}⚡ BATCH TRIGGERED{R} with {batch_cap.batch_items} items")
        status("batch_status", batch_cap.batch_status.value, YEL)
        status("can_accept", str(batch_cap.can_accept()), RED)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: CASE 3 ARRIVES DURING EXECUTION → QUEUED
    # ═══════════════════════════════════════════════════════════════

    banner("PHASE 3: CASE 3 ARRIVES → QUEUED", RED)

    case3 = cases[2]
    ddd_wo3 = DDDWorkOrder.create(
        "sar_investigation", f"cor_{case3['alert_id']}", "sar_screen",
        priority="high", sla_seconds=86400.0,
        case_id=case3["alert_id"],
    )
    decisions3 = optimizer.dispatch([ddd_wo3], "sar_screen", "structuring_sar", opt_config)
    d3 = decisions3[0]

    timeline(time.time() - sim_t0,
             f"{RED}✗{R} {B}{case3['alert_id']}{R} ({case3['alert_type']}) → "
             f"{YEL}NO RESOURCE — optimizer says: {d3.tier}{R}")

    # Queue it
    store_wo = coord.store.create_work_order(
        requester_instance_id="bsa_system",
        correlation_id=f"cor_{case3['alert_id']}",
        contract_name="sar_screen",
    ) if hasattr(coord.store, 'create_work_order') else None

    from coordinator.types import WorkOrder as WO
    queued_wo = WO.create(
        requester_instance_id="bsa_system",
        correlation_id=f"cor_{case3['alert_id']}",
        contract_name="sar_screen",
        sla_seconds=86400.0,
        urgency="high",
    )
    queued_wo.status = WorkOrderStatus.QUEUED
    coord.store.save_work_order(queued_wo)
    coord._enqueue_for_resource(
        "sar_ml_screener", queued_wo.work_order_id,
        "bsa_system", {"need": "sar_screen", "context": case3},
        type("C", (), {"need_type": "sar_screen"})(), [],
    )

    status("queue_depth", str(coord.get_queue_depth("sar_ml_screener")), YEL)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4: BATCH COMPLETES → DRAIN → RUN CASE 3
    # ═══════════════════════════════════════════════════════════════

    banner("PHASE 4: BATCH COMPLETES → DRAIN → CASE 3", GRN)

    batch_cap.complete_batch()
    timeline(time.time() - sim_t0, f"{GRN}✓ Batch execution complete{R}")
    status("batch_status", batch_cap.batch_status.value, GRN)

    drained = coord.drain_resource_queue("sar_ml_screener")
    timeline(time.time() - sim_t0, f"{GRN}▶ Queue drained:{R} {drained} work order(s)")

    # Now run case 3 through real LLM
    instance_id3, result3, elapsed3 = run_sar_workflow(
        coord, case3,
        f"{case3['alert_id']} — {case3['subject']['name']}",
    )
    all_results[case3["alert_id"]] = {
        "instance_id": instance_id3,
        "result": result3,
        "elapsed": elapsed3,
        "optimizer_tier": d3.tier,
        "was_queued": True,
    }

    # ═══════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════

    banner("RESULTS", CYN)

    total_elapsed = time.time() - total_t0

    for alert_id, data in all_results.items():
        was_queued = data.get("was_queued", False)
        queue_tag = f" {YEL}[was queued]{R}" if was_queued else ""
        error = data.get("result", {}).get("error")
        if error:
            print(f"  {RED}✗{R} {B}{alert_id}{R}{queue_tag}")
            status("error", str(error)[:150], RED)
        else:
            print(f"  {GRN}✓{R} {B}{alert_id}{R}{queue_tag}")
            status("elapsed", f"{data['elapsed']:.1f}s")
            status("optimizer", data["optimizer_tier"])
            # Show key workflow outputs
            result = data.get("result", {})
            for step_name in ("classify_alert", "classify_filing_decision"):
                if step_name in result and isinstance(result[step_name], dict):
                    cat = result[step_name].get("category", "")
                    conf = result[step_name].get("confidence", "")
                    if cat:
                        status(step_name, f"{cat} (conf={conf})")
        print()

    # ── Audit trail ──
    section("AUDIT TRAIL")
    print(f"  DDR records: {B}{len(ddr_records)}{R}")
    for d in ddr_records:
        color = GRN if d["resource"] else RED
        print(f"    {d['wo_id'][:24]:24s} → {color}{d['resource'] or 'NONE':20s}{R} "
              f"tier={d['tier']}")

    events = res_log.get_events()
    print(f"\n  Reservation events: {B}{len(events)}{R}")
    print(f"    Acquires: {len([e for e in events if e.operation == 'acquire'])}")
    print(f"    Commits:  {len([e for e in events if e.operation == 'commit'])}")

    ledger = coord.store.get_ledger()
    queue_events = [e for e in ledger if "queue" in e.get("action_type", "")]
    print(f"\n  Queue events: {B}{len(queue_events)}{R}")
    for e in queue_events:
        print(f"    {e['action_type']}")

    print(f"\n  {B}Total elapsed: {total_elapsed:.1f}s{R}")
    print(f"  {B}LLM calls: {provider} ({os.environ.get('LLM_PROVIDER', 'google')}){R}")

    # Save results
    output_file = os.path.join(_project_root, "output", "sar_batch_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": provider,
            "total_elapsed": total_elapsed,
            "cases": {k: {
                "elapsed": v["elapsed"],
                "optimizer_tier": v["optimizer_tier"],
                "was_queued": v.get("was_queued", False),
                "result_keys": list(v.get("result", {}).keys()),
            } for k, v in all_results.items()},
            "ddr_count": len(ddr_records),
            "reservation_events": len(events),
            "queue_events": len(queue_events),
        }, f, indent=2)
    print(f"\n  {GRN}Results saved → {output_file}{R}")

    banner("DONE", GRN)


if __name__ == "__main__":
    run()
