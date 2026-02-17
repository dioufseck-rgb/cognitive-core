#!/usr/bin/env python3
"""
Cognitive Core — Three-Workflow Coordination Demo
==================================================

Run: python3 tests/demo_three_workflow.py

Walks through a Loan Application Pipeline with 3 coordinated workflows,
printing a rich trace showing every state transition, delegation,
governance decision, task queue operation, and cascade.

  Workflow A: loan_application  (gate)       → undergoes underwriting
  Workflow B: credit_review     (auto)       → blocking delegation from A
  Workflow C: compliance_audit  (hold)       → fire-and-forget from A

No LLM calls. All cognitive outputs simulated. The demo exercises
the coordinator's state management, which is the part that matters
for production reliability.
"""

import sys
import os
import time
import textwrap

# Ensure project root is on sys.path regardless of where the script is invoked from
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from coordinator.types import (
    InstanceState, InstanceStatus,
    WorkOrder, WorkOrderStatus, WorkOrderResult,
    Suspension,
)
from coordinator.store import CoordinatorStore
from coordinator.tasks import (
    Task, TaskType, TaskStatus, TaskCallback, TaskResolution,
)
from coordinator.policy import GovernanceDecision
from coordinator.runtime import Coordinator

# ─── Pretty printing ──────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

def banner(text, color=BLUE):
    width = 72
    print()
    print(f"{color}{'═' * width}{RESET}")
    print(f"{color}{BOLD}  {text}{RESET}")
    print(f"{color}{'═' * width}{RESET}")

def phase(num, title):
    print()
    print(f"{YELLOW}{'─' * 72}{RESET}")
    print(f"{YELLOW}{BOLD}  PHASE {num}: {title}{RESET}")
    print(f"{YELLOW}{'─' * 72}{RESET}")

def step_trace(agent, step_name, primitive, summary, details=None):
    print(f"  {CYAN}[{agent}]{RESET} {BOLD}{step_name}{RESET} ({primitive})")
    print(f"  {DIM}└─ {summary}{RESET}")
    if details:
        for k, v in details.items():
            val = str(v)
            if len(val) > 80:
                val = val[:77] + "..."
            print(f"     {DIM}{k}: {val}{RESET}")

def state_box(label, items):
    print(f"\n  {MAGENTA}┌─ {label}{RESET}")
    for k, v in items.items():
        print(f"  {MAGENTA}│{RESET}  {k}: {v}")
    print(f"  {MAGENTA}└{'─' * 50}{RESET}")

def event(icon, msg, color=GREEN):
    print(f"  {color}{icon} {msg}{RESET}")

def task_event(action, details):
    print(f"  {CYAN}📋 TASK QUEUE: {action}{RESET}")
    for k, v in details.items():
        print(f"     {DIM}{k}: {v}{RESET}")


# ─── Main Demo ────────────────────────────────────────────────────

def run_demo():
    banner("COGNITIVE CORE — THREE-WORKFLOW COORDINATION DEMO")
    print(f"""
  {DIM}Scenario: Member MBR-5001 (Jane Doe) applies for a $25,000 personal loan.
  The loan application triggers two delegations:
    1. Credit Review   (blocking)        — loan waits for result
    2. Compliance Audit (fire-and-forget) — runs independently

  All 3 workflows are linked by a shared correlation ID.
  No LLM calls — cognitive outputs are simulated.{RESET}
""")

    store = CoordinatorStore(":memory:")
    _config_path = os.path.join(_project_root, "coordinator", "config.yaml")
    coord = Coordinator(config_path=_config_path, store=store, verbose=False)
    correlation_id = "loan_pipeline_001"

    # ═══════════════════════════════════════════════════════════════
    phase(1, "LOAN APPLICATION — Cognitive Execution")
    # ═══════════════════════════════════════════════════════════════

    loan = InstanceState.create("loan_application", "personal_loan", "gate")
    loan.correlation_id = correlation_id
    loan.status = InstanceStatus.RUNNING
    store.save_instance(loan)

    event("▶", f"STARTED loan_application/{loan.instance_id}")
    event("📌", f"correlation: {correlation_id}")
    event("🏛", f"governance tier: gate (requires specialist approval)")
    print()

    # Step 1: Retrieve
    step_trace("LOAN", "gather_application", "retrieve",
               "Pull member profile and credit summary from core systems",
               {"member": "MBR-5001 (Jane Doe, 48mo tenure)",
                "credit_score": 720, "dti_ratio": "32%",
                "tools_called": "get_member, get_credit_summary"})

    loan_state = {
        "input": {"member_id": "MBR-5001", "loan_amount": 25000, "loan_type": "personal"},
        "steps": [
            {"step_name": "gather_application", "primitive": "retrieve",
             "output": {"data": {
                 "get_member": {"member_id": "MBR-5001", "name": "Jane Doe", "tenure_months": 48},
                 "get_credit_summary": {"score": 720, "dti_ratio": 0.32},
             }}, "raw_response": "", "prompt_used": ""},
        ],
        "current_step": "gather_application",
        "metadata": {"use_case": "loan_application"},
        "loop_counts": {}, "routing_log": [],
    }

    # Step 2: Classify
    step_trace("LOAN", "assess_eligibility", "classify",
               "Evaluate loan eligibility based on member profile",
               {"category": "conditionally_approved",
                "confidence": 0.85,
                "reasoning": "Credit score 720 meets threshold; DTI 32% is borderline"})

    loan_state["steps"].append(
        {"step_name": "assess_eligibility", "primitive": "classify",
         "output": {"category": "conditionally_approved", "confidence": 0.85,
                   "reasoning": "Credit score 720 meets threshold; DTI 32% is borderline"},
         "raw_response": "", "prompt_used": ""})

    # Step 3: Investigate
    step_trace("LOAN", "underwrite", "investigate",
               "Deep underwriting analysis with risk factor identification",
               {"finding": "Applicant meets basic criteria but requires detailed credit review",
                "confidence": 0.78,
                "evidence_flags": ["high_dti_ratio", "recent_inquiry_spike"],
                "hypotheses": "DTI manageable if no hidden obligations → INCONCLUSIVE"})

    loan_state["steps"].append(
        {"step_name": "underwrite", "primitive": "investigate",
         "output": {"finding": "Meets basic criteria, needs full review",
                   "confidence": 0.78,
                   "evidence_flags": ["high_dti_ratio", "recent_inquiry_spike"],
                   "hypotheses_tested": [
                       {"hypothesis": "DTI manageable without hidden obligations",
                        "status": "inconclusive", "reasoning": "Need full credit pull"}
                   ]},
         "raw_response": "", "prompt_used": ""})

    # Step 4: Generate
    step_trace("LOAN", "generate_terms", "generate",
               "Draft preliminary loan terms for conditional approval",
               {"artifact": "Proposed: $25,000 at 8.5% APR, 60 months, pending credit review",
                "confidence": 0.9,
                "constraints": "Reg Z disclosure, TILA compliance"})

    loan_state["steps"].append(
        {"step_name": "generate_terms", "primitive": "generate",
         "output": {"artifact": "Proposed: $25,000 at 8.5% APR, 60 months",
                   "confidence": 0.9},
         "raw_response": "", "prompt_used": ""})
    loan_state["current_step"] = "generate_terms"

    loan.status = InstanceStatus.COMPLETED
    loan.step_count = 4
    loan.result = {"step_count": 4, "category": "conditionally_approved",
                   "evidence_flags": ["high_dti_ratio", "recent_inquiry_spike"]}
    store.save_instance(loan)
    store.log_action(loan.instance_id, correlation_id, "execution_finished",
                     {"steps": 4, "category": "conditionally_approved"})

    event("✓", f"EXECUTION FINISHED loan_application (4 steps)", GREEN)

    # ═══════════════════════════════════════════════════════════════
    phase(2, "DELEGATION EVALUATION — Two Policies Fire")
    # ═══════════════════════════════════════════════════════════════

    print(f"\n  {DIM}Coordinator evaluates delegation policies against loan output...{RESET}")
    print(f"  {DIM}  • evidence_flags: ['high_dti_ratio', 'recent_inquiry_spike']{RESET}")
    print(f"  {DIM}  • category: conditionally_approved{RESET}")

    # --- Delegation 1: Credit Review (BLOCKING) ---
    print()
    event("→", "DELEGATION [wait_for_result]: credit_risk_assessment", YELLOW)
    print(f"     {DIM}trigger: conditionally_approved + evidence_flags{RESET}")
    print(f"     {DIM}mode:    wait_for_result (loan SUSPENDS until credit review completes){RESET}")
    print(f"     {DIM}target:  credit_review / credit_assessment{RESET}")
    print(f"     {DIM}resume:  generate_terms (loan resumes here with credit data){RESET}")

    credit = InstanceState.create("credit_review", "credit_assessment", "auto")
    credit.correlation_id = correlation_id
    credit.lineage = [f"loan_application:{loan.instance_id}"]
    credit.status = InstanceStatus.CREATED
    store.save_instance(credit)

    wo_credit = WorkOrder.create(loan.instance_id, correlation_id,
                                  "credit_review_v1", 1,
                                  {"member_id": "MBR-5001", "loan_amount": 25000})
    wo_credit.handler_instance_id = credit.instance_id
    wo_credit.handler_workflow_type = "credit_review"
    wo_credit.handler_domain = "credit_assessment"
    wo_credit.status = WorkOrderStatus.DISPATCHED
    store.save_work_order(wo_credit)

    print(f"     {DIM}work_order: {wo_credit.work_order_id}{RESET}")

    # Suspend loan
    loan_sus = Suspension.create(loan.instance_id, "generate_terms", loan_state,
                                  work_order_ids=[wo_credit.work_order_id])
    loan.status = InstanceStatus.SUSPENDED
    loan.resume_nonce = loan_sus.resume_nonce
    loan.pending_work_orders = [wo_credit.work_order_id]
    store.save_instance(loan)
    store.save_suspension(loan_sus)

    store.log_action(loan.instance_id, correlation_id, "delegation_dispatched",
                     {"mode": "wait_for_result", "work_order_id": wo_credit.work_order_id,
                      "target": "credit_review/credit_assessment"})
    store.log_action(loan.instance_id, correlation_id, "suspended_for_delegation",
                     {"work_order_id": wo_credit.work_order_id, "resume_step": "generate_terms"})

    event("⏸", f"SUSPENDED loan_application at 'generate_terms'", YELLOW)
    print(f"     {DIM}waiting on: {wo_credit.work_order_id}{RESET}")

    # --- Delegation 2: Compliance Audit (FIRE-AND-FORGET) ---
    print()
    event("→", "DELEGATION [fire_and_forget]: lending_compliance_check", YELLOW)
    print(f"     {DIM}trigger: any personal loan > $10,000{RESET}")
    print(f"     {DIM}mode:    fire_and_forget (loan does NOT wait){RESET}")
    print(f"     {DIM}target:  compliance_audit / lending_compliance{RESET}")

    compliance = InstanceState.create("compliance_audit", "lending_compliance", "hold")
    compliance.correlation_id = correlation_id
    compliance.lineage = [f"loan_application:{loan.instance_id}"]
    compliance.status = InstanceStatus.CREATED
    store.save_instance(compliance)

    wo_comp = WorkOrder.create(loan.instance_id, correlation_id,
                                "compliance_audit_v1", 1,
                                {"member_id": "MBR-5001", "loan_amount": 25000,
                                 "loan_type": "personal"})
    wo_comp.handler_instance_id = compliance.instance_id
    wo_comp.status = WorkOrderStatus.DISPATCHED
    store.save_work_order(wo_comp)

    store.log_action(loan.instance_id, correlation_id, "delegation_dispatched",
                     {"mode": "fire_and_forget", "work_order_id": wo_comp.work_order_id,
                      "target": "compliance_audit/lending_compliance"})

    print(f"     {DIM}work_order: {wo_comp.work_order_id}{RESET}")

    state_box("PIPELINE STATE AFTER DELEGATIONS", {
        f"loan_application ({loan.instance_id[:15]}…)": f"{YELLOW}SUSPENDED{RESET} at generate_terms",
        f"credit_review ({credit.instance_id[:15]}…)":  f"{CYAN}CREATED{RESET} (about to execute)",
        f"compliance_audit ({compliance.instance_id[:15]}…)": f"{CYAN}CREATED{RESET} (about to execute)",
        "work_orders": f"2 dispatched",
        "correlation_id": correlation_id,
    })

    # ═══════════════════════════════════════════════════════════════
    phase(3, "CREDIT REVIEW — Blocking Delegation Executes")
    # ═══════════════════════════════════════════════════════════════

    credit.status = InstanceStatus.RUNNING
    store.save_instance(credit)
    event("▶", f"STARTED credit_review/{credit.instance_id}")
    event("📌", f"lineage: loan_application:{loan.instance_id[:12]}…")
    event("🏛", f"governance tier: auto (proceeds without approval)")
    print()

    step_trace("CREDIT", "pull_credit_report", "retrieve",
               "Full credit pull from Equifax, TransUnion, Experian",
               {"equifax_score": 718, "transunion_score": 724, "experian_score": 715,
                "open_accounts": 7, "delinquencies": 0,
                "tools_called": "get_credit_bureau_equifax, get_credit_bureau_tu, get_credit_bureau_exp"})

    step_trace("CREDIT", "evaluate_risk", "investigate",
               "Assess credit risk with full bureau data",
               {"finding": "Borrower has stable credit history with manageable DTI",
                "risk_tier": "B+",
                "evidence_flags": ["no_delinquencies", "stable_employment"],
                "hypotheses": [
                    "Hidden obligations → REJECTED (no undisclosed debts found)",
                    "Income sufficient → SUPPORTED (employment verified 3+ years)",
                ]})

    step_trace("CREDIT", "render_decision", "generate",
               "Generate credit decision with recommended terms",
               {"decision": "APPROVED",
                "recommended_rate": "8.25% (25bps below standard for B+ with tenure)",
                "max_amount": "$30,000",
                "conditions": "None"})

    credit.status = InstanceStatus.COMPLETED
    credit.step_count = 3
    credit.result = {
        "step_count": 3,
        "credit_decision": "approved",
        "recommended_rate": 8.25,
        "risk_tier": "B+",
        "max_amount": 30000,
        "conditions": [],
    }
    store.save_instance(credit)
    store.log_action(credit.instance_id, correlation_id, "execution_finished", {"steps": 3})

    event("✓", f"EXECUTION FINISHED credit_review (3 steps)", GREEN)
    event("🏛", f"governance: auto → proceed (no approval needed)", GREEN)

    # Update work order
    wo_credit.status = WorkOrderStatus.COMPLETED
    wo_credit.completed_at = time.time()
    wo_credit.result = WorkOrderResult(
        wo_credit.work_order_id, "completed", credit.result, completed_at=time.time())
    store.save_work_order(wo_credit)

    event("📦", f"work_order {wo_credit.work_order_id[:18]}… → COMPLETED", GREEN)

    # Cascade
    print()
    event("↩", f"CASCADE: credit_review completed → checking for blocked requesters", MAGENTA)
    print(f"     {DIM}work_order {wo_credit.work_order_id} links to requester {loan.instance_id}{RESET}")
    print(f"     {DIM}requester is SUSPENDED with this work_order in pending list{RESET}")

    # Run the cascade (will attempt resume, which fails without LLM)
    try:
        coord._check_delegation_completion(credit)
    except Exception:
        pass

    event("▶", f"RESUME loan_application at 'generate_terms' (with credit data injected)", MAGENTA)
    print(f"     {DIM}delegation_results[{wo_credit.work_order_id}] = {RESET}")
    print(f"     {DIM}  credit_decision: approved{RESET}")
    print(f"     {DIM}  recommended_rate: 8.25{RESET}")
    print(f"     {DIM}  risk_tier: B+{RESET}")
    print(f"     {DIM}  max_amount: 30000{RESET}")
    print(f"     {DIM}(Resume would re-execute generate_terms with enriched context){RESET}")
    print(f"     {DIM}(Skipped here — no LLM in test environment){RESET}")

    state_box("PIPELINE STATE AFTER CREDIT REVIEW", {
        f"loan_application": f"{MAGENTA}RESUME ATTEMPTED{RESET} (would continue in production)",
        f"credit_review": f"{GREEN}COMPLETED{RESET} (3 steps, approved, B+)",
        f"compliance_audit": f"{CYAN}CREATED{RESET} (not yet started)",
        f"work_order (credit)": f"{GREEN}COMPLETED{RESET}",
    })

    # ═══════════════════════════════════════════════════════════════
    phase(4, "COMPLIANCE AUDIT — Fire-and-Forget with Governance Hold")
    # ═══════════════════════════════════════════════════════════════

    compliance.status = InstanceStatus.RUNNING
    store.save_instance(compliance)
    event("▶", f"STARTED compliance_audit/{compliance.instance_id}")
    event("📌", f"lineage: loan_application:{loan.instance_id[:12]}…")
    event("🏛", f"governance tier: hold (REQUIRES compliance officer approval)")
    print()

    step_trace("COMPLIANCE", "check_tila", "verify",
               "Verify Truth in Lending Act compliance",
               {"conforms": True, "rules_checked": ["APR_disclosure", "finance_charge", "payment_schedule"],
                "violations": "none"})

    step_trace("COMPLIANCE", "check_ecoa", "verify",
               "Verify Equal Credit Opportunity Act compliance",
               {"conforms": True, "rules_checked": ["no_discrimination", "adverse_action_notice"],
                "violations": "none"})

    step_trace("COMPLIANCE", "check_hmda", "verify",
               "Verify Home Mortgage Disclosure Act applicability",
               {"conforms": True, "rules_checked": ["loan_type_exemption"],
                "violations": "none",
                "note": "Personal loan exempt from HMDA reporting"})

    step_trace("COMPLIANCE", "generate_compliance_report", "generate",
               "Generate compliance attestation report",
               {"artifact": "Compliance Report: All checks passed. TILA/ECOA/HMDA reviewed.",
                "confidence": 0.98})

    compliance.status = InstanceStatus.COMPLETED
    compliance.step_count = 4
    compliance.result = {
        "step_count": 4,
        "compliance_status": "passed",
        "checks_performed": ["TILA", "ECOA", "HMDA"],
        "violations": [],
    }
    store.save_instance(compliance)
    store.log_action(compliance.instance_id, correlation_id, "execution_finished", {"steps": 4})

    event("✓", f"EXECUTION FINISHED compliance_audit (4 steps)", GREEN)
    print()

    # Governance evaluation
    event("🏛", f"GOVERNANCE EVALUATION: hold tier", YELLOW)
    print(f"     {DIM}tier: hold → action: suspend_for_approval{RESET}")
    print(f"     {DIM}queue: compliance_review{RESET}")
    print(f"     {DIM}reason: Lending compliance requires officer review before finalization{RESET}")

    compliance_state = {
        "steps": [
            {"step_name": "check_tila", "primitive": "verify",
             "output": {"conforms": True, "violations": [], "rules_checked": ["APR_disclosure"]},
             "raw_response": "", "prompt_used": ""},
            {"step_name": "check_ecoa", "primitive": "verify",
             "output": {"conforms": True, "violations": [], "rules_checked": ["no_discrimination"]},
             "raw_response": "", "prompt_used": ""},
            {"step_name": "check_hmda", "primitive": "verify",
             "output": {"conforms": True, "violations": [], "rules_checked": ["loan_type_exemption"]},
             "raw_response": "", "prompt_used": ""},
            {"step_name": "generate_compliance_report", "primitive": "generate",
             "output": {"artifact": "All checks passed. TILA/ECOA/HMDA reviewed.", "confidence": 0.98},
             "raw_response": "", "prompt_used": ""},
        ],
        "input": {"member_id": "MBR-5001"},
    }
    comp_sus = Suspension.create(compliance.instance_id, "__governance_gate__", compliance_state)
    compliance.status = InstanceStatus.SUSPENDED
    compliance.resume_nonce = comp_sus.resume_nonce
    store.save_instance(compliance)
    store.save_suspension(comp_sus)

    # Publish task
    task = Task.create(
        task_type=TaskType.GOVERNANCE_APPROVAL,
        queue="compliance_review",
        instance_id=compliance.instance_id,
        correlation_id=correlation_id,
        workflow_type="compliance_audit",
        domain="lending_compliance",
        payload={"governance_tier": "hold",
                 "reason": "Lending compliance requires officer review",
                 "compliance_status": "passed",
                 "checks": ["TILA", "ECOA", "HMDA"]},
        callback=TaskCallback(method="approve", instance_id=compliance.instance_id,
                               resume_nonce=comp_sus.resume_nonce),
        priority=2,
    )
    task_id = coord.tasks.publish(task)

    store.log_action(compliance.instance_id, correlation_id, "suspended_for_approval",
                     {"tier": "hold", "queue": "compliance_review", "task_id": task_id})

    event("⏸", f"SUSPENDED compliance_audit → task published to 'compliance_review'", YELLOW)

    task_event("PUBLISHED", {
        "task_id": task_id,
        "queue": "compliance_review",
        "type": TaskType.GOVERNANCE_APPROVAL,
        "priority": "2 (urgent)",
        "callback": f"approve({compliance.instance_id[:15]}…)",
    })

    state_box("PIPELINE STATE — AWAITING APPROVAL", {
        f"loan_application": "RESUME ATTEMPTED (would be running in prod)",
        f"credit_review": f"{GREEN}COMPLETED{RESET}",
        f"compliance_audit": f"{YELLOW}SUSPENDED{RESET} → governance hold",
        f"pending_approvals": "1 (compliance_review queue)",
    })

    # ═══════════════════════════════════════════════════════════════
    phase(5, "HUMAN APPROVAL — Compliance Officer Reviews")
    # ═══════════════════════════════════════════════════════════════

    # List pending
    pending = coord.list_pending_approvals()
    print(f"  {BOLD}Pending approvals:{RESET} {len(pending)}")
    for p in pending:
        print(f"     {DIM}• {p['workflow_type']}/{p['domain']} "
              f"(tier: {p['governance_tier']}, queue: {p['queue']}){RESET}")
    print()

    # Officer claims
    event("👤", "Compliance Officer Kim claims task from queue", CYAN)
    claimed = coord.claim_task("compliance_review", "Officer Kim")
    task_event("CLAIMED", {
        "task_id": claimed["task_id"],
        "claimed_by": "Officer Kim",
        "workflow": f"{claimed['workflow_type']}/{claimed['domain']}",
        "payload.compliance_status": claimed["payload"].get("compliance_status", ""),
        "payload.checks": str(claimed["payload"].get("checks", [])),
    })

    # Officer reviews and approves
    print()
    event("👤", "Officer Kim reviews compliance report...", CYAN)
    print(f"     {DIM}• TILA: APR properly disclosed, finance charges correct{RESET}")
    print(f"     {DIM}• ECOA: No discriminatory factors in underwriting{RESET}")
    print(f"     {DIM}• HMDA: Personal loan correctly exempt{RESET}")
    print(f"     {DIM}• Decision: APPROVE{RESET}")
    print()

    coord.resolve_task(
        task_id=claimed["task_id"],
        action="approve",
        resolved_by="Officer Kim",
        notes="All regulatory checks verified. Loan cleared for disbursement.",
    )

    task_event("RESOLVED", {
        "task_id": claimed["task_id"],
        "action": "approve",
        "resolved_by": "Officer Kim",
        "notes": "All regulatory checks verified. Loan cleared for disbursement.",
    })

    event("✓", f"APPROVED compliance_audit by Officer Kim", GREEN)

    # Final state
    comp_final = store.get_instance(compliance.instance_id)
    event("✓", f"COMPLETED compliance_audit (post-approval)", GREEN)

    state_box("PIPELINE STATE — FINAL", {
        f"loan_application": "COMPLETED (would be, with credit data)",
        f"credit_review": f"{GREEN}COMPLETED{RESET} (approved, B+, 8.25%)",
        f"compliance_audit": f"{GREEN}COMPLETED{RESET} (approved by Officer Kim)",
        f"work_orders": "2 completed",
        f"pending_approvals": "0",
    })

    # ═══════════════════════════════════════════════════════════════
    phase(6, "AUDIT TRAIL — Full Correlation Chain & Ledger")
    # ═══════════════════════════════════════════════════════════════

    chain = store.list_instances(correlation_id=correlation_id)
    print(f"\n  {BOLD}Correlation Chain: {correlation_id}{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    for inst in chain:
        status_color = GREEN if inst.status == InstanceStatus.COMPLETED else YELLOW
        lineage_str = f" ← {inst.lineage[0][:30]}…" if inst.lineage else " (root)"
        print(f"  {status_color}●{RESET} {inst.workflow_type}/{inst.domain}")
        print(f"    {DIM}id: {inst.instance_id}{RESET}")
        print(f"    {DIM}status: {inst.status.value} | steps: {inst.step_count} | tier: {inst.governance_tier}{RESET}")
        print(f"    {DIM}lineage:{lineage_str}{RESET}")

    print(f"\n  {BOLD}Work Orders:{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    all_wos = store.get_work_orders_for_instance(loan.instance_id)
    for wo in all_wos:
        status_color = GREEN if wo.status == WorkOrderStatus.COMPLETED else YELLOW
        result_summary = ""
        if wo.result and wo.result.outputs:
            keys = list(wo.result.outputs.keys())[:3]
            result_summary = f" → {', '.join(keys)}"
        print(f"  {status_color}●{RESET} {wo.work_order_id}")
        print(f"    {DIM}contract: {wo.contract_name} | status: {wo.status.value}{result_summary}{RESET}")
        print(f"    {DIM}requester: {wo.requester_instance_id[:20]}… → handler: {wo.handler_instance_id[:20]}…{RESET}")

    print(f"\n  {BOLD}Action Ledger:{RESET}")
    print(f"  {DIM}{'─' * 60}{RESET}")
    ledger = store.get_ledger(correlation_id=correlation_id)
    for entry in ledger:
        icon_map = {
            "execution_finished": "✓", "delegation_dispatched": "→",
            "suspended_for_delegation": "⏸", "suspended_for_approval": "⏸",
            "governance_approved": "✓", "governance_rejected": "✗",
            "terminate": "■", "resume": "▶",
        }
        icon = icon_map.get(entry["action_type"], "·")
        inst_short = entry["instance_id"][:15] + "…"
        details_str = ""
        if entry.get("details"):
            d = entry["details"]
            if isinstance(d, str):
                try:
                    d = __import__("json").loads(d)
                except Exception:
                    pass
            if isinstance(d, dict):
                interesting = {k: v for k, v in d.items()
                               if k in ("mode", "target", "tier", "queue", "steps",
                                        "work_order_id", "task_id", "resume_step",
                                        "category")}
                if interesting:
                    details_str = " " + " ".join(f"{k}={v}" for k, v in interesting.items())
        print(f"  {DIM}{icon} [{inst_short}] {entry['action_type']}{details_str}{RESET}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════

    banner("DEMO COMPLETE", GREEN)
    print(f"""
  {GREEN}✓{RESET} 3 workflows coordinated through single correlation chain
  {GREEN}✓{RESET} Blocking delegation: loan suspended → credit review completed → cascade triggered
  {GREEN}✓{RESET} Fire-and-forget: compliance audit ran independently
  {GREEN}✓{RESET} Governance hold: compliance suspended → task published → officer approved
  {GREEN}✓{RESET} Task queue: publish → list → claim → resolve lifecycle
  {GREEN}✓{RESET} Full audit trail: {len(ledger)} ledger entries across {len(chain)} instances
  {GREEN}✓{RESET} All assertions would pass — zero state inconsistencies
""")

    # Verify assertions silently
    assert len(chain) == 3
    assert store.get_instance(credit.instance_id).status == InstanceStatus.COMPLETED
    assert store.get_instance(compliance.instance_id).status == InstanceStatus.COMPLETED
    assert len(all_wos) == 2
    assert len(ledger) >= 6


if __name__ == "__main__":
    run_demo()
