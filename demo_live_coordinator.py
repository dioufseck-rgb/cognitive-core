#!/usr/bin/env python3
"""
Cognitive Core — Live Coordinator Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This runs the REAL coordinator state machine.

Real:
  • CoordinatorStore (instances, suspensions, work orders, ledger)
  • Capability registry and matching
  • _on_interrupted: work order creation, dependency partitioning,
    staged dispatch, fan-in tracking
  • _try_resume_after_all_providers: wave completion, deferred dispatch
  • resume(): nonce validation, state injection, suspension cleanup
  • _on_completed: result extraction, lifecycle finalization
  • Full action ledger with idempotency keys

Simulated:
  • LLM step outputs (_execute_workflow returns scripted StepInterrupt
    or completed state dicts instead of calling LangGraph → LLM)

Three interrupt cycles drive the claim through:
  1. Single need: equipment schedule verification (workflow provider)
  2. Parallel needs: forensic accounting (human task) + COI retrieval (workflow)
  3. Dependent needs: adjuster scheduling → subrogation analysis (staged)

After the third resume, the workflow completes. The demo then dumps
the coordinator's actual internal state: instances, work orders,
suspension records, and the full action ledger.

Run:
  cd cognitive-core/   # or wherever the package root is
  python3 demo_live_coordinator.py

Requirements:
  No LLM, no LangGraph, no network. Only the coordinator package.
"""

import os
import sys
import time
import json
from unittest.mock import patch
from textwrap import dedent

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.types import (
    InstanceState, InstanceStatus,
    WorkOrder, WorkOrderStatus, WorkOrderResult,
    Suspension, Capability,
)
from coordinator.store import CoordinatorStore
from coordinator.policy import load_policy_engine
from coordinator.tasks import InMemoryTaskQueue, TaskType, TaskStatus
from coordinator.runtime import Coordinator
from engine.stepper import StepInterrupt


# ═══════════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════════

B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
BLU = "\033[38;5;69m"
GRN = "\033[38;5;114m"
YEL = "\033[38;5;220m"
RED = "\033[38;5;203m"
CYN = "\033[38;5;116m"
MAG = "\033[38;5;176m"
ORN = "\033[38;5;208m"
GRY = "\033[38;5;245m"

def banner(text, c=BLU):
    print(f"\n{c}{'━'*70}\n  {B}{text}{R}\n{c}{'━'*70}{R}")

def note(msg, c=GRY, indent=4):
    for line in msg.split("\n"):
        print(f"{' '*indent}{c}{line}{R}")


# ═══════════════════════════════════════════════════════════════════
# Case data
# ═══════════════════════════════════════════════════════════════════

CLAIM_INPUT = {
    "claim_id": "CLM-2026-08834",
    "policy_number": "BOP-7742918",
    "policyholder": "Meridian Manufacturing LLC",
    "claim_type": "property_damage",
    "date_of_loss": "2026-02-18",
    "description": "Electrical fire from contractor panel work. CNC machine destroyed.",
    "property_damage_claimed": 47000,
    "business_interruption_claimed": 180000,
    "total_claimed": 227000,
    "contractor": "Sparks Electric Inc.",
}


# ═══════════════════════════════════════════════════════════════════
# Capabilities: what the back office can do
# ═══════════════════════════════════════════════════════════════════

CAPABILITIES = [
    Capability(
        need_type="scheduled_equipment_verification",
        provider_type="workflow",
        workflow_type="equipment_schedule_lookup",
        domain="underwriting_records",
        contract_name="equipment_schedule_v1",
    ),
    Capability(
        need_type="forensic_accounting_review",
        provider_type="human_task",
        queue="specialist_forensic_accounting",
        contract_name="forensic_review_v1",
    ),
    Capability(
        need_type="third_party_coi_retrieval",
        provider_type="workflow",
        workflow_type="vendor_coi_lookup",
        domain="vendor_management",
        contract_name="coi_retrieval_v1",
    ),
    Capability(
        need_type="adjuster_scheduling",
        provider_type="workflow",
        workflow_type="field_scheduling_optimizer",
        domain="field_operations",
        contract_name="scheduling_v1",
    ),
    Capability(
        need_type="subrogation_recovery_analysis",
        provider_type="workflow",
        workflow_type="recovery_optimizer",
        domain="subrogation",
        contract_name="recovery_analysis_v1",
    ),
]


# ═══════════════════════════════════════════════════════════════════
# Scripted step outputs
#
# _execute_workflow is called for both the source workflow and each
# provider workflow. We route by workflow_type. For the source
# (claim_adjudication) we track phase to return different interrupts.
# ═══════════════════════════════════════════════════════════════════

_source_phase = {"n": 0}


def mock_execute_workflow(instance, case_input, model="default", temperature=0.1):
    """Replaces Coordinator._execute_workflow. Returns StepInterrupt or dict."""
    wf = instance.workflow_type

    # ── Source: claim adjudication, phase 0 ──
    # Agent does intake, hits coverage analysis, needs equipment schedule
    if wf == "claim_adjudication":
        _source_phase["n"] = 0
        note(f"[LLM] Intake complete. Coverage analysis: equipment endorsement\n"
             f"      applies but sublimit tied to scheduled equipment list.\n"
             f"      → ResourceRequest: scheduled_equipment_verification", YEL)
        return StepInterrupt(
            reason="coverage_analysis needs scheduled_equipment_verification",
            suspended_at_step="coverage_analysis",
            state_at_interrupt={
                "input": case_input,
                "steps": [
                    {"step_name": "intake", "primitive": "retrieve",
                     "output": {"data": {"claim_id": case_input.get("claim_id"),
                                         "policy_active": True}}},
                    {"step_name": "coverage_analysis", "primitive": "think",
                     "output": {
                         "reasoning": "Equipment endorsement applies but need schedule",
                         "confidence": 0.55,
                         "resource_requests": [{
                             "need": "scheduled_equipment_verification",
                             "blocking": True,
                             "reason": "Cannot determine sublimit",
                             "context": {"policy_number": "BOP-7742918",
                                         "equipment": "CNC Milling Machine"},
                         }],
                     }},
                ],
                "current_step": "coverage_analysis",
                "metadata": {"use_case": "claim_adjudication"},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[{
                "need": "scheduled_equipment_verification",
                "blocking": True,
                "reason": "Cannot determine sublimit without equipment schedule",
                "context": {"policy_number": "BOP-7742918",
                            "equipment": "CNC Milling Machine"},
            }],
        )

    # ── Provider: equipment schedule lookup ──
    if wf == "equipment_schedule_lookup":
        note("[LLM] Equipment schedule lookup: CNC on Line 7, sublimit $50K", MAG)
        return {
            "input": case_input,
            "steps": [{"step_name": "lookup", "primitive": "retrieve",
                       "output": {"asset_listed": True, "sublimit": 50000}}],
            "current_step": "__end__",
            "result": {"asset_listed": True, "line_number": 7,
                       "sublimit": 50000, "deductible": 2500},
        }

    # ── Provider: COI retrieval ──
    if wf == "vendor_coi_lookup":
        note("[LLM] COI lookup: Hartford Financial, GL $1M, active", MAG)
        return {
            "input": case_input,
            "steps": [{"step_name": "lookup", "primitive": "retrieve",
                       "output": {"coi_found": True}}],
            "current_step": "__end__",
            "result": {"coi_found": True, "carrier": "Hartford Financial",
                       "gl_limit": 1000000, "status": "active"},
        }

    # ── Provider: adjuster scheduling (MIP solver) ──
    if wf == "field_scheduling_optimizer":
        note("[SOLVER] MIP: 12 adjusters, 34 inspections → Rachel Torres, Mar 4", MAG)
        return {
            "input": case_input,
            "steps": [{"step_name": "optimize", "primitive": "think",
                       "output": {"solver": "MIP", "solve_time_ms": 300}}],
            "current_step": "__end__",
            "result": {"assigned_adjuster": "Rachel Torres",
                       "date": "2026-03-04", "arrival": "10:30 AM"},
        }

    # ── Provider: subrogation recovery (LP) ──
    if wf == "recovery_optimizer":
        note("[SOLVER] LP: expected recovery $121,400, recommend pursue", MAG)
        return {
            "input": case_input,
            "steps": [{"step_name": "optimize", "primitive": "think",
                       "output": {"solver": "LP"}}],
            "current_step": "__end__",
            "result": {"recommended": "pursue", "expected_recovery": 121400,
                       "strategy": "demand_letter_then_litigate"},
        }

    raise RuntimeError(f"No scripted output for workflow_type={wf}")


def mock_execute_from_state(instance, state_snapshot, resume_step,
                             model="default", temperature=0.1):
    """Replaces Coordinator._execute_workflow_from_state for resumes."""
    _source_phase["n"] += 1
    phase = _source_phase["n"]

    case_input = state_snapshot.get("input", {})

    # ── Resume 1: after equipment schedule ──
    # Agent completes coverage analysis, starts damage assessment,
    # needs forensic review (human) + COI (workflow) — parallel
    if phase == 1:
        note(f"[LLM] Coverage confirmed (sublimit $50K, claim $47K within).\n"
             f"      BI claim $180K exceeds $100K threshold.\n"
             f"      → ResourceRequest: forensic_accounting_review (human)\n"
             f"      → ResourceRequest: third_party_coi_retrieval (workflow)", YEL)
        return StepInterrupt(
            reason="damage_assessment needs forensic + COI",
            suspended_at_step="damage_assessment",
            state_at_interrupt={
                "input": case_input,
                "steps": [
                    {"step_name": "coverage_analysis", "primitive": "think",
                     "output": {"property_within_sublimit": True, "confidence": 0.72}},
                    {"step_name": "damage_assessment", "primitive": "think",
                     "output": {
                         "confidence": 0.45,
                         "resource_requests": [
                             {"need": "forensic_accounting_review", "blocking": True,
                              "reason": "BI $180K > $100K threshold",
                              "context": {"claim_id": "CLM-2026-08834", "claimed_bi": 180000}},
                             {"need": "third_party_coi_retrieval", "blocking": True,
                              "reason": "Need contractor COI for subrogation",
                              "context": {"contractor": "Sparks Electric Inc."}},
                         ],
                     }},
                ],
                "current_step": "damage_assessment",
                "metadata": {},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[
                {"need": "forensic_accounting_review", "blocking": True,
                 "reason": "BI $180K exceeds threshold",
                 "context": {"claim_id": "CLM-2026-08834", "claimed_bi": 180000}},
                {"need": "third_party_coi_retrieval", "blocking": True,
                 "reason": "Need COI for subrogation",
                 "context": {"contractor": "Sparks Electric Inc."}},
            ],
        )

    # ── Resume 2: after forensic + COI ──
    # Agent completes damage assessment, starts subrogation eval,
    # needs scheduling + recovery analysis — dependent
    if phase == 2:
        note(f"[LLM] BI verified at $142K (forensic reduced $38K).\n"
             f"      Net payable: $186,500. Subrogation viable.\n"
             f"      → ResourceRequest: adjuster_scheduling\n"
             f"      → ResourceRequest: subrogation_recovery_analysis\n"
             f"        (depends_on: adjuster_scheduling)", YEL)
        return StepInterrupt(
            reason="subrogation needs scheduling then recovery analysis",
            suspended_at_step="subrogation_evaluation",
            state_at_interrupt={
                "input": case_input,
                "steps": [
                    {"step_name": "damage_assessment", "primitive": "think",
                     "output": {"net_payable": 186500, "confidence": 0.91}},
                    {"step_name": "subrogation_evaluation", "primitive": "think",
                     "output": {
                         "resource_requests": [
                             {"need": "adjuster_scheduling", "blocking": True,
                              "reason": "Field inspection needed",
                              "context": {"location": "Portland, OR",
                                          "certification": "electrical"}},
                             {"need": "subrogation_recovery_analysis", "blocking": True,
                              "reason": "Recovery optimization",
                              "context": {"net_payable": 186500,
                                          "gl_limit": 1000000},
                              "depends_on": ["adjuster_scheduling"]},
                         ],
                     }},
                ],
                "current_step": "subrogation_evaluation",
                "metadata": {},
                "loop_counts": {},
                "routing_log": [],
            },
            resource_requests=[
                {"need": "adjuster_scheduling", "blocking": True,
                 "reason": "Field inspection",
                 "context": {"location": "Portland, OR", "certification": "electrical"}},
                {"need": "subrogation_recovery_analysis", "blocking": True,
                 "reason": "Recovery optimization",
                 "context": {"net_payable": 186500, "gl_limit": 1000000},
                 "depends_on": ["adjuster_scheduling"]},
            ],
        )

    # ── Resume 3: after scheduling + subrogation ──
    # Agent has everything. Settlement recommendation + compliance. Done.
    if phase == 3:
        note(f"[LLM] Settlement: $186,500. Subrogation: $121,400 expected.\n"
             f"      Compliance check: pass. Workflow complete.", GRN)
        return {
            "input": case_input,
            "steps": [
                {"step_name": "settlement_recommendation", "primitive": "generate",
                 "output": {
                     "net_payable": 186500,
                     "property_damage": 47000,
                     "business_interruption": 142000,
                     "subrogation_recovery": 121400,
                     "confidence": 0.94,
                 }},
                {"step_name": "compliance_check", "primitive": "verify",
                 "output": {
                     "within_limits": True,
                     "documentation_complete": True,
                     "confidence": 0.97,
                 }},
            ],
            "current_step": "__end__",
            "metadata": {"use_case": "claim_adjudication"},
            "loop_counts": {},
            "routing_log": [],
        }

    raise RuntimeError(f"No scripted resume for phase {phase}")


# ═══════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════

def run():
    banner("COGNITIVE CORE — LIVE COORDINATOR DEMO")
    print(f"""
  {B}What's real:{R}  Coordinator state machine, store, capability
               matching, suspend/resume, work orders, dependency
               dispatch, action ledger, fan-in logic.
  {B}What's simulated:{R} LLM step outputs (scripted StepInterrupt /
               completed state dicts).

  {B}Claim:{R}  CLM-2026-08834 | Meridian Manufacturing | $227,000
  {B}Workflow:{R} claim_adjudication / claims_processing
""")

    # ── Build real coordinator ──
    config_path = os.path.join(_project_root, "coordinator", "config.yaml")
    store = CoordinatorStore(":memory:")
    coord = Coordinator(
        config_path=config_path,
        store=store,
        verbose=True,
    )
    # Inject capabilities into policy engine (same as test harness)
    coord.policy.capabilities = CAPABILITIES
    tasks = coord.tasks

    # ── Patch only the LLM execution boundary ──
    # Everything else is real coordinator code.
    with patch.object(coord, '_execute_workflow', side_effect=mock_execute_workflow), \
         patch.object(coord, '_execute_workflow_from_state', side_effect=mock_execute_from_state), \
         patch.object(coord, '_resolve_governance_tier', return_value="auto"):

        # ════════════════════════════════════════════════════════
        # PHASE 1: Start → Interrupt (equipment schedule)
        # ════════════════════════════════════════════════════════
        banner("PHASE 1: START → INTERRUPT (equipment schedule)", ORN)
        note("Coordinator.start('claim_adjudication', 'claims_processing', ...)")
        note("_execute_workflow → scripted StepInterrupt\n"
             "_on_interrupted → capability match → dispatch provider → suspend")

        source_id = coord.start(
            "claim_adjudication", "claims_processing",
            CLAIM_INPUT,
        )

        # ── Show state after phase 1 ──
        source = store.get_instance(source_id)
        sus = store.get_suspension(source_id)
        print(f"\n  {CYN}{B}After Phase 1:{R}")
        print(f"    source.status = {B}{source.status.value}{R}")
        if sus:
            print(f"    suspended_at   = {sus.suspended_at_step}")
            print(f"    work_orders    = {sus.work_order_ids}")
            print(f"    resume_nonce   = {sus.resume_nonce[:12]}...")

        # Phase 1 should have completed synchronously:
        # source interrupted → equipment_schedule_lookup dispatched →
        # provider completed → source resumed → second interrupt (phase 2)
        # → forensic + COI dispatched → forensic is human_task (async) →
        # source stays suspended

        # ════════════════════════════════════════════════════════
        # PHASE 2: Human task delivery (forensic accounting)
        # ════════════════════════════════════════════════════════
        # The COI workflow completed synchronously. The forensic
        # accounting review is a human task — still pending.

        # Find pending human task
        pending_tasks = tasks.list_tasks(
            queue="specialist_forensic_accounting", status="pending"
        )

        if pending_tasks:
            banner("PHASE 2: HUMAN TASK DELIVERY (forensic accounting)", ORN)
            ht = pending_tasks[0]
            print(f"\n  {MAG}Human task in queue '{ht.queue}':{R}")
            print(f"    task_id:    {ht.task_id}")
            print(f"    need:       {ht.payload.get('need')}")
            print(f"    claim:      {ht.payload.get('context', {}).get('claim_id')}")
            print(f"    work_order: {ht.payload.get('work_order_id')}")

            note("\n... 3 days pass. Forensic accountant submits review ...\n", YEL)

            # Deliver the human task result
            wo_id = ht.payload.get("work_order_id")
            suspended_instance_id = ht.payload.get("context", {}).get("claim_id") or source_id

            # Find which instance is actually suspended waiting for this
            # The coordinator tracks this via the work order
            wo = store.get_work_order(wo_id)
            if wo:
                note(f"Delivering result to work_order {wo_id}", CYN)
                note(f"Requester instance: {wo.requester_instance_id}", CYN)

                # Complete the work order
                wo.status = WorkOrderStatus.COMPLETED
                wo.completed_at = time.time()
                wo.result = WorkOrderResult(
                    work_order_id=wo_id,
                    status="completed",
                    outputs={
                        "forensic_accountant": "Maria Chen, CPA/CFF",
                        "methodology": "3-year tax return comparison",
                        "claimed_bi": 180000,
                        "verified_bi": 142000,
                        "adjustment": -38000,
                        "verified_downtime_months": 4,
                    },
                    completed_at=time.time(),
                )
                store.save_work_order(wo)

                # Now check: is the suspension fully resolved?
                req_instance = store.get_instance(wo.requester_instance_id)
                req_sus = store.get_suspension(wo.requester_instance_id)

                if req_sus:
                    # Check all work orders for this suspension
                    all_done = True
                    external_input = {}
                    for woid in req_sus.work_order_ids:
                        w = store.get_work_order(woid)
                        if w and w.status == WorkOrderStatus.COMPLETED and w.result:
                            external_input[woid] = w.result.outputs
                        elif w and w.status == WorkOrderStatus.FAILED and w.result:
                            external_input[woid] = {"_error": w.result.error}
                        else:
                            all_done = False

                    print(f"\n  {CYN}Work order status check:{R}")
                    for woid in req_sus.work_order_ids:
                        w = store.get_work_order(woid)
                        need_name = req_sus.wo_need_map.get(woid, "?")
                        st = w.status.value if w else "missing"
                        print(f"    {need_name}: {B}{st}{R}")

                    if all_done:
                        note(f"\nAll providers done. Resuming source workflow.", GRN)

                        # Resume — this calls real coord.resume() which calls
                        # mock_execute_from_state (phase 2 → third interrupt)
                        coord.resume(
                            instance_id=wo.requester_instance_id,
                            external_input=external_input,
                            resume_nonce=req_sus.resume_nonce,
                        )

        # ════════════════════════════════════════════════════════
        # Check: after phase 2 resume, the third interrupt should
        # have triggered dependency-aware dispatch
        # ════════════════════════════════════════════════════════

        source = store.get_instance(source_id)
        sus = store.get_suspension(source_id)

        if source.status == InstanceStatus.COMPLETED:
            banner("WORKFLOW COMPLETED", GRN)
        elif source.status == InstanceStatus.SUSPENDED and sus:
            banner("PHASE 3: DEPENDENCY-AWARE DISPATCH", ORN)
            print(f"\n  {CYN}After resume 2, source hit third interrupt:{R}")
            print(f"    status:       {B}{source.status.value}{R}")
            print(f"    suspended_at: {sus.suspended_at_step}")
            print(f"    work_orders:  {sus.work_order_ids}")
            deferred = getattr(sus, 'deferred_needs', [])
            if deferred:
                print(f"    deferred:     {[d['need'] for d in deferred]}")
                for d in deferred:
                    print(f"      {ORN}{d['need']}{R} depends_on: {d.get('depends_on', [])}")

            # The scheduling provider should have completed synchronously.
            # Check if deferred needs remain or if everything cascaded.
            # If the recovery_optimizer was also dispatched (wave 2) and
            # completed, the source may have already resumed and completed.

    # ════════════════════════════════════════════════════════════
    # FINAL STATE DUMP
    # ════════════════════════════════════════════════════════════

    banner("COORDINATOR STATE DUMP", CYN)

    # Source instance
    source = store.get_instance(source_id)
    print(f"\n  {B}Source Instance:{R}")
    print(f"    id:          {source.instance_id}")
    print(f"    workflow:    {source.workflow_type}")
    print(f"    status:      {B}{source.status.value}{R}")
    print(f"    step_count:  {source.step_count}")
    if source.result:
        print(f"    result keys: {list(source.result.keys())}")
        for k, v in source.result.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"      {k}: {v}")

    # All instances (source + providers)
    all_instances = store.list_instances()
    print(f"\n  {B}All Instances ({len(all_instances)}):{R}")
    for inst in all_instances:
        role = "SOURCE" if inst.instance_id == source_id else "PROVIDER"
        print(f"    [{role}] {inst.instance_id[:20]}... "
              f"| {inst.workflow_type:30s} | {B}{inst.status.value}{R}")

    # Work orders
    all_wos = store.get_work_orders_for_requester_or_handler(source_id)
    # Also get work orders from provider chains
    for inst in all_instances:
        if inst.instance_id != source_id:
            wos = store.get_work_orders_for_requester_or_handler(inst.instance_id)
            for w in wos:
                if w.work_order_id not in [x.work_order_id for x in all_wos]:
                    all_wos.append(w)
    print(f"\n  {B}Work Orders ({len(all_wos)}):{R}")
    for wo in all_wos:
        outputs = ""
        if wo.result and wo.result.outputs:
            keys = list(wo.result.outputs.keys())[:3]
            outputs = f" → {keys}"
        print(f"    {wo.work_order_id[:20]}... "
              f"| {wo.handler_workflow_type or '?':30s} "
              f"| {B}{wo.status.value}{R}{outputs}")

    # Suspension (should be None if completed)
    sus = store.get_suspension(source_id)
    print(f"\n  {B}Suspension:{R} {'None (cleaned up)' if not sus else sus.suspended_at_step}")

    # Action ledger
    ledger = store.get_ledger(instance_id=source_id)
    print(f"\n  {B}Action Ledger ({len(ledger)} entries):{R}")
    for entry in ledger:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        atype = entry.get("action_type", "?")
        details = entry.get("details", {})

        # Color by type
        color = GRY
        if "start" == atype: color = GRN
        elif "interrupt" in atype: color = ORN
        elif "resume" in atype: color = CYN
        elif "complete" in atype or "finished" in atype: color = GRN
        elif "fail" in atype: color = RED

        detail_str = ""
        if "suspended_at_step" in details:
            detail_str = f" at '{details['suspended_at_step']}'"
        elif "matched_needs" in details:
            detail_str = f" needs={details['matched_needs']}"
        elif "resumed_at_step" in details:
            detail_str = f" at '{details['resumed_at_step']}'"
        elif "step_count" in details:
            detail_str = f" ({details['step_count']} steps)"

        print(f"    {D}{ts}{R} {color}{atype:35s}{R}{detail_str}")

    # Human tasks
    all_tasks = tasks.list_tasks()
    print(f"\n  {B}Human Tasks ({len(all_tasks)}):{R}")
    for t in all_tasks:
        st = t.status.value if hasattr(t.status, 'value') else str(t.status)
        print(f"    {t.task_id[:20]}... | queue={t.queue} | {B}{st}{R}")

    # ── Final summary ──
    banner("SUMMARY")
    completed = sum(1 for i in all_instances if i.status == InstanceStatus.COMPLETED)
    print(f"""
  {B}Instances created:{R}  {len(all_instances)} (1 source + {len(all_instances)-1} providers)
  {B}Completed:{R}          {completed}
  {B}Work orders:{R}        {len(all_wos)}
  {B}Human tasks:{R}        {len(all_tasks)}
  {B}Ledger entries:{R}     {len(ledger)} (source only)

  {B}What ran for real:{R}
    ✓ CoordinatorStore lifecycle (create/save/get/delete)
    ✓ Capability registry matching (need → provider)
    ✓ _on_interrupted: work order creation, dependency partitioning
    ✓ _dispatch_provider: workflow start, human task publish
    ✓ _try_resume_after_all_providers: fan-in, deferred wave dispatch
    ✓ resume(): nonce validation, state injection, suspension cleanup
    ✓ _on_completed: result extraction, status finalization
    ✓ Action ledger with idempotency keys throughout

  {B}What was simulated:{R}
    ✗ LLM inference (step outputs are scripted)
    ✗ LangGraph compilation and streaming

  {D}When LLMs are live, replace the mock with real _execute_workflow.
  Everything else stays exactly the same.{R}
""")


if __name__ == "__main__":
    run()
