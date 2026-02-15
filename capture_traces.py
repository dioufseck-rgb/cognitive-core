#!/usr/bin/env python3
"""
Cognitive Core — Trace Capture Script

Runs all use cases (sequential + agentic), captures detailed traces
including case context, step-by-step outputs, routing decisions, and
timing. Saves everything to a single JSON file for presentation deck.

Usage:
    export GOOGLE_API_KEY=your_key
    python capture_traces.py                      # all use cases
    python capture_traces.py --only complaint      # filter by name
    python capture_traces.py --output traces.json  # custom output path
    python capture_traces.py --model gemini-2.5-pro
"""

import argparse
import json
import sys
import time
import os
import traceback
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Use case definitions with case context summaries
# ──────────────────────────────────────────────────────────────

USE_CASES = [
    # ── Sequential ────────────────────────────────────────────
    {
        "id": "card_dispute_seq",
        "title": "Card Dispute",
        "mode": "sequential",
        "workflow": "workflows/dispute_resolution.yaml",
        "domain": "domains/card_dispute.yaml",
        "case": "cases/card_clear_fraud.json",
        "case_summary": {
            "headline": "Clear Fraud — Unauthorized Amazon Purchase",
            "member": "Sarah Chen, 8-year member, excellent standing",
            "situation": "Unauthorized $847.23 Amazon purchase while card was in her possession. "
                        "Fraud score 0.92. Device fingerprint doesn't match any known device.",
            "key_facts": [
                "Transaction: $847.23 at Amazon.com on 2026-01-15",
                "Card was in member's possession (not lost/stolen)",
                "Fraud score: 0.92 — strong signal for unauthorized",
                "Device fingerprint mismatch confirms not member's device",
            ],
            "expected_outcome": "Fast-path classification (high confidence) → provisional credit → member letter",
            "regulatory": "Reg E (EFT), Reg Z (credit cards)",
        },
    },
    {
        "id": "loan_hardship_seq",
        "title": "Loan Hardship — Military",
        "mode": "sequential",
        "workflow": "workflows/loan_hardship.yaml",
        "domain": "domains/military_hardship.yaml",
        "case": "cases/military_hardship_reeves.json",
        "case_summary": {
            "headline": "Military Spouse — Medical Retirement Hardship",
            "member": "Angela Reeves, member since 2018, military spouse",
            "situation": "Husband being medically retired from Navy after 12 years. "
                        "Dual income dropping to single. Mortgage $2,847/mo + auto loan $487/mo. "
                        "Concerned about making payments during transition.",
            "key_facts": [
                "Military medical retirement — SCRA/MLA protections may apply",
                "Mortgage: $287K balance, $2,847/mo — 38% of current income",
                "Auto loan: $18.2K balance, $487/mo",
                "Transition from dual income ($142K) to single (~$65K estimate)",
                "No missed payments yet — proactively seeking help",
            ],
            "expected_outcome": "Classify → military investigation (SCRA) → guidance letter or escalation",
            "regulatory": "SCRA, MLA, ECOA, UDAAP",
        },
    },
    {
        "id": "complaint_seq",
        "title": "Member Complaint",
        "mode": "sequential",
        "workflow": "workflows/complaint_resolution.yaml",
        "domain": "domains/member_complaint.yaml",
        "case": "cases/complaint_torres.json",
        "case_summary": {
            "headline": "Inconsistent Hold Information — Service Failure",
            "member": "Michael Torres, 6-year member, $45K relationship, attrition risk 0.6",
            "situation": "Called 3 times in one week about a paycheck deposit hold. "
                        "Got conflicting information each time. Bounced 2 payments, "
                        "charged $50 in NSF fees. Now threatening to close accounts.",
            "key_facts": [
                "3 calls in 7 days — different answer each time",
                "Hold was for new account deposit verification ($3,200)",
                "2 bounced payments: Verizon $127.43, State Farm $198.00",
                "NSF fees: $50.00 total",
                "Member says: 'I'm considering closing my accounts'",
                "Relationship: $45K across checking, savings, auto loan + direct deposit",
            ],
            "expected_outcome": "Classify type+severity → investigate root cause → response with fee reversal",
            "regulatory": "UDAAP (unfair/deceptive acts)",
        },
    },
    {
        "id": "spending_advisor_seq",
        "title": "Spending Advisor",
        "mode": "sequential",
        "workflow": "workflows/spending_advisor.yaml",
        "domain": "domains/debit_spending.yaml",
        "case": "cases/spending_advisor_williams.json",
        "case_summary": {
            "headline": "Personal Finance Check-In",
            "member": "David Williams, 34, $72K income, 3-year member",
            "situation": "Asks 'How's my spending been?' — wants a general health check "
                        "on debit card spending patterns and progress toward goals.",
            "key_facts": [
                "12 months of debit card transaction data available",
                "Financial goals: emergency fund ($15K target), vacation fund ($3K)",
                "Monthly discretionary spending ~$1,800",
                "Top categories: dining, groceries, entertainment, coffee",
            ],
            "expected_outcome": "Retrieve data → classify inquiry → investigate patterns → conversational advice",
            "regulatory": "Cannot provide investment advice without licensing",
        },
    },
    {
        "id": "nurse_triage_seq",
        "title": "Nurse Triage — Cardiac",
        "mode": "sequential",
        "workflow": "workflows/nurse_triage.yaml",
        "domain": "domains/cardiac_triage.yaml",
        "case": "cases/cardiac_chest_pain.json",
        "case_summary": {
            "headline": "2 AM Chest Pain — Potential Cardiac Emergency",
            "member": "Robert Martinez, 52-year-old male",
            "situation": "Woke at 2am with chest pressure radiating to left arm. "
                        "History of hypertension and high cholesterol. Father had MI at 58. "
                        "Took antacid 30 minutes ago, no relief.",
            "key_facts": [
                "52yo male — high-risk demographic for cardiac events",
                "Chest pressure + left arm radiation (classic cardiac pattern)",
                "Onset at 2am (acute, at rest)",
                "Risk factors: hypertension, hyperlipidemia, family history",
                "Antacid ineffective — argues against GI cause",
                "Spouse on the line, asking what to do",
            ],
            "expected_outcome": "Emergent classification → verify safety protocols → 911 script with aspirin",
            "regulatory": "Schmitt-Thompson triage protocols, standard of care",
        },
    },
    {
        "id": "regulatory_impact_seq",
        "title": "Regulatory Impact — AVM Rule",
        "mode": "sequential",
        "workflow": "workflows/regulatory_impact.yaml",
        "domain": "domains/avm_regulation.yaml",
        "case": "cases/avm_regulation.json",
        "case_summary": {
            "headline": "Interagency AVM Quality Control Rule",
            "member": "Navy Federal Credit Union — $180B assets",
            "situation": "New interagency rule (CFPB, OCC, FDIC, NCUA, FHFA, FRB) requiring "
                        "quality control standards for automated valuation models in mortgage lending. "
                        "Affects HELOCs, home equity, refinancing. Comment period ends April 15.",
            "key_facts": [
                "6 agencies issuing jointly — significant regulatory weight",
                "Effective date: September 1, 2026",
                "Comment period ends: April 15, 2026 (2 months away)",
                "Affects: Mortgage Lending, Home Equity, Risk, Compliance, IT",
                "Current AVM vendors: CoreLogic, FNMA Collateral Underwriter",
                "Fair lending / nondiscrimination requirements included",
            ],
            "expected_outcome": "Classify severity → deep investigation → executive impact report",
            "regulatory": "AVM rule, ECOA/fair lending, exam preparation",
        },
    },
    {
        "id": "sar_investigation_seq",
        "title": "SAR Investigation — Structuring",
        "mode": "sequential",
        "workflow": "workflows/sar_investigation.yaml",
        "domain": "domains/structuring_sar.yaml",
        "case": "cases/sar_structuring.json",
        "case_summary": {
            "headline": "Potential Structuring — Cash Deposits Below CTR Threshold",
            "member": "James Wilson, self-employed landscaping business",
            "situation": "Alert triggered by pattern of cash deposits consistently just below "
                        "$10,000 CTR threshold. Multiple deposits in short timeframes. "
                        "Alert score 0.78.",
            "key_facts": [
                "6 cash deposits in 30 days, all between $8,500-$9,800",
                "Average: ~$9,200 (consistently below $10K CTR threshold)",
                "Deposits at multiple branches",
                "Subject claims cash-heavy landscaping business",
                "No prior SARs or suspicious activity on account",
                "Alert score: 0.78",
            ],
            "expected_outcome": "Classify alert → investigate patterns → filing decision → SAR narrative or no-file doc",
            "regulatory": "BSA/AML, CTR ($10K), SAR filing (FinCEN)",
        },
    },

    # ── Agentic ───────────────────────────────────────────────
    {
        "id": "spending_advisor_agent",
        "title": "Spending Advisor (Agentic)",
        "mode": "agentic",
        "workflow": "workflows/spending_advisor_agentic.yaml",
        "domain": "domains/debit_spending_agentic.yaml",
        "case": "cases/spending_advisor_williams.json",
        "case_summary": {
            "headline": "Same Member, Agentic Mode — Watch Self-Correction",
            "member": "David Williams, 34 (same as sequential)",
            "situation": "Same question, same data — but orchestrator decides the path. "
                        "May discover errors in generation and self-correct through "
                        "the generate → challenge → reinvestigate loop.",
            "key_facts": ["Same case data as sequential spending advisor"],
            "expected_outcome": "Orchestrator-driven path with potential self-correction loops",
            "regulatory": "Same constraints — no investment advice",
        },
    },
    {
        "id": "loan_hardship_agent",
        "title": "Military Hardship (Agentic + Think)",
        "mode": "agentic",
        "workflow": "workflows/loan_hardship_agentic.yaml",
        "domain": "domains/military_hardship_agentic.yaml",
        "case": "cases/military_hardship_reeves.json",
        "case_summary": {
            "headline": "Same Case, Agentic — Think Changes the Decision",
            "member": "Angela Reeves (same as sequential)",
            "situation": "Same case but agentic orchestrator has Think available. "
                        "Expected to run dual investigations (military + financial), "
                        "then Think synthesizes: too many unknowns → escalate to specialist "
                        "instead of generating a member letter.",
            "key_facts": ["Same case — compare outcome to sequential version"],
            "expected_outcome": "Dual investigation → Think → escalation brief (not member letter)",
            "regulatory": "SCRA, MLA (same — but Think catches what sequential misses)",
        },
    },
    {
        "id": "complaint_agent",
        "title": "Member Complaint (Agentic)",
        "mode": "agentic",
        "workflow": "workflows/complaint_resolution_agentic.yaml",
        "domain": "domains/member_complaint_agentic.yaml",
        "case": "cases/complaint_torres.json",
        "case_summary": {
            "headline": "Same Complaint, Agentic — Orchestrator Decides Escalation",
            "member": "Michael Torres (same as sequential)",
            "situation": "Same complaint — orchestrator may use Think to weigh "
                        "retention math ($45K relationship vs $50 fees) and decide "
                        "whether to escalate or resolve directly.",
            "key_facts": ["Same case — compare resolution path to sequential version"],
            "expected_outcome": "Think-informed decision on escalation vs direct resolution",
            "regulatory": "UDAAP (same)",
        },
    },
]


# ──────────────────────────────────────────────────────────────
# Trace capture
# ──────────────────────────────────────────────────────────────

class TraceCapture:
    """Captures all trace events into a structured list."""

    def __init__(self):
        self.events = []
        self.start_time = time.time()

    def _ts(self):
        return round(time.time() - self.start_time, 2)

    def on_step_start(self, step_name, primitive, loop_iteration):
        self.events.append({
            "type": "step_start", "ts": self._ts(),
            "step_name": step_name, "primitive": primitive,
            "loop_iteration": loop_iteration,
        })
        print(f"    [{self._ts():6.1f}s] {'🏷️ ' if primitive=='classify' else '🔍' if primitive=='investigate' else '💭' if primitive=='think' else '✅' if primitive=='verify' else '📝' if primitive=='generate' else '⚔️ ' if primitive=='challenge' else '📡'} {step_name}", file=sys.stderr)

    def on_llm_start(self, step_name, prompt_chars):
        self.events.append({
            "type": "llm_start", "ts": self._ts(),
            "step_name": step_name, "prompt_chars": prompt_chars,
        })

    def on_llm_end(self, step_name, response_chars, elapsed):
        self.events.append({
            "type": "llm_end", "ts": self._ts(),
            "step_name": step_name, "response_chars": response_chars,
            "llm_elapsed": round(elapsed, 2),
        })
        print(f"    [{self._ts():6.1f}s]   ↳ response ({response_chars:,} chars, {elapsed:.1f}s)", file=sys.stderr)

    def on_parse_result(self, step_name, primitive, output):
        self.events.append({
            "type": "parse_result", "ts": self._ts(),
            "step_name": step_name, "primitive": primitive,
        })
        # One-line summary
        if primitive == "classify":
            print(f"    [{self._ts():6.1f}s]   → {output.get('category','?')} ({output.get('confidence',0):.2f})", file=sys.stderr)
        elif primitive == "investigate":
            f = str(output.get('finding',''))[:70]
            print(f"    [{self._ts():6.1f}s]   → {f}... ({output.get('confidence',0):.2f})", file=sys.stderr)
        elif primitive == "challenge":
            s = output.get('survives', '?')
            n = len(output.get('vulnerabilities', []))
            print(f"    [{self._ts():6.1f}s]   → {'✓ passed' if s else f'✗ failed ({n} vulns)'}", file=sys.stderr)
        elif primitive == "generate":
            print(f"    [{self._ts():6.1f}s]   → generated {len(output.get('artifact',''))} chars", file=sys.stderr)
        elif primitive == "verify":
            print(f"    [{self._ts():6.1f}s]   → conforms: {output.get('conforms','?')}", file=sys.stderr)
        elif primitive == "retrieve":
            d = output.get('data', {})
            print(f"    [{self._ts():6.1f}s]   → {len(d)} data keys", file=sys.stderr)

    def on_parse_error(self, step_name, error):
        self.events.append({
            "type": "parse_error", "ts": self._ts(),
            "step_name": step_name, "error": str(error)[:200],
        })
        print(f"    [{self._ts():6.1f}s]   ⚠ PARSE ERROR: {str(error)[:80]}", file=sys.stderr)

    def on_route_decision(self, from_step, to_step, decision_type, reason):
        self.events.append({
            "type": "route", "ts": self._ts(),
            "from_step": from_step, "to_step": to_step,
            "decision_type": decision_type, "reason": reason,
        })
        target = "END" if to_step == "__end__" else to_step
        print(f"    [{self._ts():6.1f}s]   → route: {target} ({decision_type})", file=sys.stderr)

    def on_retrieve_start(self, step_name, source_name):
        self.events.append({
            "type": "retrieve_start", "ts": self._ts(),
            "step_name": step_name, "source_name": source_name,
        })

    def on_retrieve_end(self, step_name, source_name, status, latency_ms):
        self.events.append({
            "type": "retrieve_end", "ts": self._ts(),
            "step_name": step_name, "source_name": source_name,
            "status": status, "latency_ms": round(latency_ms, 1),
        })


def run_use_case(uc, model, temperature):
    """Run a single use case and return the full trace."""
    from engine.composer import load_three_layer, run_workflow
    from engine.agentic import run_agentic_workflow
    from engine.nodes import set_trace
    from engine.tools import create_case_registry

    config, case_input = load_three_layer(uc["workflow"], uc["domain"], uc["case"])
    is_agentic = config.get("mode") == "agentic"

    # Setup trace capture
    trace = TraceCapture()
    set_trace(trace)

    # Tool registry from case data
    has_retrieve = False
    if is_agentic:
        has_retrieve = "retrieve" in config.get("available_primitives", [])
    else:
        has_retrieve = any(s["primitive"] == "retrieve" for s in config.get("steps", []))

    tool_registry = None
    if has_retrieve:
        tool_registry = create_case_registry(case_input)

    # Run
    start = time.time()
    try:
        if is_agentic:
            final_state = run_agentic_workflow(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
            )
        else:
            final_state = run_workflow(
                config, case_input, model, temperature,
                tool_registry=tool_registry,
            )
        elapsed = time.time() - start
        error = None
    except Exception as e:
        elapsed = time.time() - start
        final_state = None
        error = str(e)
        traceback.print_exc(file=sys.stderr)

    # Build result
    result = {
        "id": uc["id"],
        "title": uc["title"],
        "mode": uc["mode"],
        "workflow_file": uc["workflow"],
        "domain_file": uc["domain"],
        "case_file": uc["case"],
        "case_summary": uc["case_summary"],
        "model": model,
        "temperature": temperature,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "trace_events": trace.events,
        "error": error,
    }

    if final_state:
        # Extract structured step results
        steps = []
        for s in final_state.get("steps", []):
            step_data = {
                "step_name": s["step_name"],
                "primitive": s["primitive"],
                "output": s["output"],
            }
            steps.append(step_data)

        result["steps"] = steps
        result["step_count"] = len(steps)
        result["step_path"] = " → ".join(s["step_name"] for s in steps)
        result["routing_log"] = final_state.get("routing_log", [])
        result["loop_counts"] = {
            k: v for k, v in final_state.get("loop_counts", {}).items() if v > 1
        }

        # Final outcome summary
        last = steps[-1] if steps else None
        if last:
            result["final_primitive"] = last["primitive"]
            if last["primitive"] == "challenge":
                result["final_survives"] = last["output"].get("survives")
                result["final_vulnerabilities"] = len(last["output"].get("vulnerabilities", []))
            elif last["primitive"] == "generate":
                result["final_artifact_length"] = len(last["output"].get("artifact", ""))

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture traces for all Cognitive Core use cases")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--output", "-o", default="traces.json")
    parser.add_argument("--only", help="Filter use cases by substring match on id/title")
    parser.add_argument("--sequential-only", action="store_true")
    parser.add_argument("--agentic-only", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Set GOOGLE_API_KEY before running.", file=sys.stderr)
        print("  export GOOGLE_API_KEY=your_key", file=sys.stderr)
        sys.exit(1)

    # Filter use cases
    cases = USE_CASES
    if args.only:
        cases = [uc for uc in cases if args.only.lower() in uc["id"].lower()
                 or args.only.lower() in uc["title"].lower()]
    if args.sequential_only:
        cases = [uc for uc in cases if uc["mode"] == "sequential"]
    if args.agentic_only:
        cases = [uc for uc in cases if uc["mode"] == "agentic"]

    print(f"\n{'═'*70}", file=sys.stderr)
    print(f"  COGNITIVE CORE — TRACE CAPTURE", file=sys.stderr)
    print(f"  {len(cases)} use cases  |  model: {args.model}", file=sys.stderr)
    print(f"  output: {args.output}", file=sys.stderr)
    print(f"{'═'*70}\n", file=sys.stderr)

    all_results = {
        "capture_metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "temperature": args.temperature,
            "total_use_cases": len(cases),
        },
        "use_cases": [],
    }

    total_start = time.time()

    for i, uc in enumerate(cases):
        print(f"\n{'─'*70}", file=sys.stderr)
        print(f"  [{i+1}/{len(cases)}] {uc['title']} ({uc['mode']})", file=sys.stderr)
        print(f"  {uc['case_summary']['headline']}", file=sys.stderr)
        print(f"{'─'*70}", file=sys.stderr)

        try:
            result = run_use_case(uc, args.model, args.temperature)
            status = "✓" if not result.get("error") else "✗"
            steps = result.get("step_count", 0)
            elapsed = result.get("elapsed_seconds", 0)
            print(f"\n  {status} {uc['title']}: {steps} steps, {elapsed:.1f}s", file=sys.stderr)
        except Exception as e:
            result = {
                "id": uc["id"],
                "title": uc["title"],
                "mode": uc["mode"],
                "case_summary": uc["case_summary"],
                "error": str(e),
                "elapsed_seconds": 0,
            }
            print(f"\n  ✗ {uc['title']}: {e}", file=sys.stderr)

        all_results["use_cases"].append(result)

    total_elapsed = time.time() - total_start

    all_results["capture_metadata"]["total_elapsed_seconds"] = round(total_elapsed, 1)
    all_results["capture_metadata"]["successful"] = sum(
        1 for uc in all_results["use_cases"] if not uc.get("error")
    )

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'═'*70}", file=sys.stderr)
    print(f"  CAPTURE COMPLETE", file=sys.stderr)
    ok = all_results["capture_metadata"]["successful"]
    print(f"  {ok}/{len(cases)} successful  |  {total_elapsed:.1f}s total", file=sys.stderr)
    print(f"  saved: {args.output} ({os.path.getsize(args.output):,} bytes)", file=sys.stderr)
    print(f"{'═'*70}\n", file=sys.stderr)
    print(f"\nNext: feed {args.output} to Claude to generate the presentation deck.", file=sys.stderr)


if __name__ == "__main__":
    main()
