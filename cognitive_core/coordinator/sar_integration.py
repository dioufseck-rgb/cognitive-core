"""
Cognitive Core — SAR Evidence Integration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bridges between the CaseEvidenceStore (SQLite) and workflow execution.

Two directions:
  1. DB → Workflow: build_case_input() constructs a rich case_input dict
     from all evidence tables, giving the LLM access to transactions,
     member profile, account history, entity lookups, etc.

  2. Workflow → DB: capture_workflow_outputs() extracts step outputs from
     the completed workflow state and writes them to sar_workflow_outputs.

Usage:
    from cognitive_core.coordinator.evidence import CaseEvidenceStore
    from cognitive_core.coordinator.sar_integration import build_case_input, capture_workflow_outputs

    evidence = CaseEvidenceStore(store)
    
    # Before starting workflow
    case_input = build_case_input(evidence, "ALT-2026-4471")
    instance_id = coord.start("sar_investigation", "structuring_sar", case_input)
    
    # After workflow completes
    instance = store.get_instance(instance_id)
    capture_workflow_outputs(evidence, "ALT-2026-4471", instance, final_state)
"""

from __future__ import annotations

import json
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cognitive_core.coordinator.evidence import CaseEvidenceStore
    from cognitive_core.coordinator.types import InstanceState


def build_case_input(evidence: CaseEvidenceStore, alert_id: str) -> dict[str, Any]:
    """
    Build a rich case_input dict from all evidence tables.
    
    This is what the LLM workflow receives as ${input}. It contains
    everything an analyst would pull up when investigating an alert:
    transactions, member profile, account baseline, wire details,
    prior history, entity lookups, and screening results.
    
    The structure is designed so the LLM can reference specific
    evidence in its reasoning, and the UI can trace which evidence
    supported each conclusion.
    """
    alert = evidence.get_alert(alert_id)
    if not alert:
        raise ValueError(f"Alert {alert_id} not found in evidence store")

    subject = evidence.get_subject(alert_id) or {}
    account = evidence.get_account(alert_id) or {}
    transactions = evidence.get_transactions(alert_id)
    prior_alerts = evidence.get_prior_alerts(alert_id)
    ctr = evidence.get_ctr_history(alert_id) or {}
    entities = evidence.get_entity_lookups(alert_id)
    ofac = evidence.get_ofac(alert_id) or {}

    # ── Format transactions for LLM consumption ──
    # Summarize + provide detail so the LLM can cite specifics
    txn_summary = _summarize_transactions(transactions)
    
    # ── Format wire details if present ──
    wire_details = []
    for txn in transactions:
        if txn.get("wire_details"):
            wd = txn["wire_details"]
            wire_details.append({
                "txn_id": txn["txn_id"],
                "date": txn["txn_date"],
                "amount": txn["amount"],
                "direction": "inbound" if txn["txn_type"] == "wire_in" else "outbound",
                **wd,
            })

    # ── Prior alert narrative ──
    if prior_alerts:
        prior_narrative = "; ".join(
            f"{pa['prior_alert_id']} ({pa['prior_date']}, {pa['prior_type']}): "
            f"{pa['disposition']}. {pa.get('notes', '')}"
            for pa in prior_alerts
        )
    else:
        prior_narrative = "No prior alerts or SARs on file for this customer."

    # ── Entity lookup narrative ──
    if entities:
        entity_narrative = "\n".join(
            f"- {ent['entity_name']} ({ent['jurisdiction']}, via {ent['source']}): "
            f"{ent['result']}. {ent.get('detail', '')}"
            for ent in entities
        )
    else:
        entity_narrative = "No entity lookups performed or no counterparty entities identified."

    # ── Build the complete input ──
    case_input = {
        # Alert metadata
        "alert_id": alert_id,
        "alert_type": alert["alert_type"],
        "risk_score": alert["risk_score"],
        "monitoring_rule": alert.get("monitoring_rule", ""),
        "generated_at": alert.get("generated_at", ""),

        # Subject (structured for LLM)
        "subject": {
            "name": subject.get("full_name", ""),
            "customer_id": subject.get("customer_id", ""),
            "relationship_since": subject.get("relationship_since", ""),
            "employment": subject.get("employer", ""),
            "title": subject.get("title", ""),
            "income": {
                "annual": subject.get("income_annual"),
                "source": subject.get("income_source", ""),
                "verified": bool(subject.get("income_verified")),
            },
            "risk_rating": subject.get("risk_rating", "low"),
            "address": subject.get("address", ""),
        },

        # Account baseline
        "account": {
            "number": account.get("account_number_masked", ""),
            "type": account.get("account_type", ""),
            "product": account.get("product_name", ""),
            "opened": account.get("opened_date", ""),
            "avg_balance_6mo": account.get("avg_balance_6mo"),
            "avg_deposits_6mo": account.get("avg_deposits_6mo"),
            "avg_cash_deposits_6mo": account.get("avg_cash_deposits_6mo"),
            "typical_range": account.get("typical_range", ""),
            "activity_notes": account.get("activity_notes", ""),
        },

        # Transaction evidence
        "triggering_activity": txn_summary["narrative"],
        "transaction_count": txn_summary["count"],
        "transaction_total": txn_summary["total"],
        "transaction_date_range": txn_summary["date_range"],
        "transactions": txn_summary["detail"],  # individual txn records

        # Wire details (if any)
        "wire_transfers": wire_details if wire_details else None,

        # Account history / baseline
        "account_history": (
            f"Average monthly deposits: ${account.get('avg_deposits_6mo') or 0:,.0f} "
            f"(6-month). Average cash deposits: ${account.get('avg_cash_deposits_6mo') or 0:,.0f}. "
            f"Average balance: ${account.get('avg_balance_6mo') or 0:,.0f}. "
            f"Typical deposit range: {account.get('typical_range') or 'N/A'}. "
            f"{account.get('activity_notes') or ''}"
        ),

        # Prior history
        "prior_alerts": prior_narrative,
        "prior_alert_count": len(prior_alerts),

        # CTR history (absence is evidence)
        "ctr_history": {
            "filings_12mo": ctr.get("filings_12mo", 0),
            "expected": ctr.get("expected_filings", ""),
            "notes": ctr.get("notes", ""),
        },

        # Entity research
        "entity_lookups": entity_narrative,
        "entity_lookup_details": [
            {
                "entity": ent["entity_name"],
                "jurisdiction": ent["jurisdiction"],
                "source": ent["source"],
                "result": ent["result"],
                "detail": ent.get("detail", ""),
            }
            for ent in entities
        ],

        # OFAC/Sanctions
        "ofac_screening": {
            "result": ofac.get("result", "not_screened"),
            "entities_screened": ofac.get("entities_screened", ""),
            "lists_checked": ofac.get("lists_checked", []),
        },
    }

    # Remove None values to keep prompts clean
    return {k: v for k, v in case_input.items() if v is not None}


def _summarize_transactions(transactions: list[dict]) -> dict[str, Any]:
    """
    Build a transaction summary + detail list for the LLM.
    """
    if not transactions:
        return {
            "narrative": "No transactions in evidence.",
            "count": 0, "total": 0, "date_range": "", "detail": [],
        }

    # Separate by type
    cash_deps = [t for t in transactions if t["txn_type"] == "cash_deposit"]
    wires_in = [t for t in transactions if t["txn_type"] == "wire_in"]
    wires_out = [t for t in transactions if t["txn_type"] == "wire_out"]
    checks = [t for t in transactions if t["txn_type"] == "cashiers_check"]
    
    # Amounts (absolute for summary)
    amounts = [abs(t["amount"]) for t in transactions]
    dates = sorted(set(t["txn_date"] for t in transactions))
    date_range = f"{dates[0]} to {dates[-1]}" if len(dates) > 1 else dates[0] if dates else ""
    
    # Unique locations
    locations = set(t.get("location") or "" for t in transactions if t.get("location"))

    # Build narrative summary
    parts = []
    if cash_deps:
        dep_total = sum(t["amount"] for t in cash_deps)
        dep_range = f"${min(t['amount'] for t in cash_deps):,.0f}–${max(t['amount'] for t in cash_deps):,.0f}"
        parts.append(
            f"{len(cash_deps)} cash deposits totaling ${dep_total:,.0f} "
            f"(range: {dep_range}) across {len(locations)} locations"
        )
    if wires_in:
        wire_total = sum(t["amount"] for t in wires_in)
        parts.append(f"{len(wires_in)} incoming wires totaling ${wire_total:,.0f}")
    if wires_out:
        wire_out_total = sum(abs(t["amount"]) for t in wires_out)
        parts.append(f"{len(wires_out)} outgoing wires totaling ${wire_out_total:,.0f}")
    if checks:
        chk_total = sum(abs(t["amount"]) for t in checks)
        payees = set(t.get("payee", "unknown") for t in checks)
        parts.append(f"{len(checks)} cashier's check(s) totaling ${chk_total:,.0f} to {', '.join(payees)}")

    narrative = f"Period: {date_range}. " + ". ".join(parts) + "."

    # Detail records for LLM citation
    detail = []
    for t in transactions:
        rec = {
            "txn_id": t["txn_id"],
            "date": t["txn_date"],
            "time": t.get("txn_time", ""),
            "type": t["txn_type"],
            "amount": t["amount"],
            "location": t.get("location", ""),
            "balance_after": t.get("running_balance"),
        }
        if t.get("payee"):
            rec["payee"] = t["payee"]
        if t.get("wire_details"):
            wd = t["wire_details"]
            rec["wire"] = {
                k: v for k, v in wd.items()
                if v and k in ("originator_name", "from", "originator_country",
                               "country", "beneficiary_name", "to", "to_bank",
                               "from_bank", "purpose", "swift_ref", "swift")
            }
        detail.append(rec)

    return {
        "narrative": narrative,
        "count": len(transactions),
        "total": sum(amounts),
        "date_range": date_range,
        "detail": detail,
    }


# ═══════════════════════════════════════════════════════════════════
# WORKFLOW → DB: Capture outputs after completion
# ═══════════════════════════════════════════════════════════════════

def capture_workflow_outputs(
    evidence: CaseEvidenceStore,
    alert_id: str,
    final_state: dict[str, Any],
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Extract step outputs from a completed workflow state and save
    them to the evidence store.
    
    Maps the workflow's internal step names to evidence store columns:
      classify_alert         → classify_output
      investigate_activity   → investigate_output
      classify_filing_decision → filing_output
      generate_narrative     → narrative_text (the artifact)
      challenge_narrative    → challenge_output
      verify_completeness    → verify_output
    
    Returns the extracted outputs dict for convenience.
    """
    steps = final_state.get("steps", [])
    
    # Build a step-name → output map
    step_outputs = {}
    for step in steps:
        name = step.get("step_name", "")
        output = step.get("output", {})
        if name and output:
            step_outputs[name] = output

    # Map to evidence store schema
    outputs = {}

    # Classification
    classify = step_outputs.get("classify_alert", {})
    if classify:
        outputs["classify"] = {
            "category": classify.get("category"),
            "confidence": classify.get("confidence"),
            "reasoning": classify.get("reasoning", ""),
            "evidence_used": classify.get("evidence_used", []),
        }

    # Investigation
    investigate = step_outputs.get("investigate_activity", {})
    if investigate:
        outputs["investigate"] = {
            "finding": investigate.get("finding", ""),
            "confidence": investigate.get("confidence"),
            "evidence_used": investigate.get("evidence_used", []),
            "evidence_missing": investigate.get("evidence_missing", []),
            "gaps": investigate.get("evidence_missing", []),
        }

    # Filing decision
    filing = step_outputs.get("classify_filing_decision", {})
    if filing:
        outputs["filing"] = {
            "decision": filing.get("category"),
            "confidence": filing.get("confidence"),
            "reasoning": filing.get("reasoning", ""),
        }

    # Narrative (the main artifact)
    narrative = step_outputs.get("generate_narrative", {})
    if narrative:
        outputs["narrative"] = narrative.get("artifact", "")

    # Challenge
    challenge = step_outputs.get("challenge_narrative", {})
    if challenge:
        outputs["challenge"] = {
            "survives": challenge.get("survives"),
            "strengths": challenge.get("strengths", []),
            "weaknesses": challenge.get("vulnerabilities", []),
            "overall_assessment": challenge.get("overall_assessment", ""),
        }

    # Verification
    verify = step_outputs.get("verify_completeness", {})
    if verify:
        outputs["verify"] = {
            "conforms": verify.get("conforms"),
            "rules_checked": verify.get("rules_checked", []),
            "violations": verify.get("violations", []),
        }

    # Closure (alternative path)
    closure = step_outputs.get("generate_closure", {})
    if closure:
        outputs["closure"] = closure.get("artifact", "")

    # Escalation (alternative path)
    escalation = step_outputs.get("escalate", {})
    if escalation:
        outputs["escalation"] = escalation.get("artifact", "")

    # Save to evidence store
    evidence.save_workflow_outputs(
        alert_id, outputs,
        elapsed_seconds=elapsed_seconds,
    )

    return outputs


# ═══════════════════════════════════════════════════════════════════
# COORDINATOR HOOKS: Wire into the coordinator lifecycle
# ═══════════════════════════════════════════════════════════════════

class SARWorkflowHooks:
    """
    Lifecycle hooks for SAR workflow execution.
    
    Attach to a coordinator to automatically:
    - Build rich case_input from evidence store before execution
    - Capture workflow outputs to evidence store after completion
    - Update alert disposition based on governance gate decision
    
    Usage:
        hooks = SARWorkflowHooks(evidence_store)
        
        # Before starting
        case_input = hooks.pre_execute("ALT-2026-4471")
        instance_id = coord.start("sar_investigation", "structuring_sar", case_input)
        
        # After completion (call from _on_completed or governance callback)
        hooks.post_execute("ALT-2026-4471", final_state, elapsed=62.0)
        
        # After examiner review
        hooks.on_examiner_decision("ALT-2026-4471", "approved", "J. Rodriguez")
    """

    def __init__(self, evidence: CaseEvidenceStore):
        self.evidence = evidence
        self._execution_starts: dict[str, float] = {}

    def pre_execute(self, alert_id: str) -> dict[str, Any]:
        """Build case_input from evidence store. Call before coord.start()."""
        self._execution_starts[alert_id] = time.time()
        return build_case_input(self.evidence, alert_id)

    def post_execute(
        self,
        alert_id: str,
        final_state: dict[str, Any],
        elapsed: float | None = None,
    ) -> dict[str, Any]:
        """Capture outputs to evidence store. Call after workflow completes."""
        if elapsed is None and alert_id in self._execution_starts:
            elapsed = time.time() - self._execution_starts.pop(alert_id, time.time())
        return capture_workflow_outputs(
            self.evidence, alert_id, final_state, elapsed_seconds=elapsed,
        )

    def on_examiner_decision(
        self,
        alert_id: str,
        decision: str,
        examiner: str | None = None,
    ) -> None:
        """Update disposition after examiner review at governance gate."""
        self.evidence.update_disposition(alert_id, decision, examiner)
