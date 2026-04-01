"""
Cognitive Core — Minimal Integration Example (Sprint 3)

A minimal Python application that embeds Cognitive Core and demonstrates
the complete case lifecycle from an external application:

    1. Submit a case           → instance_id
    2. Subscribe to events     → SSE stream
    3. Handle HITL pause       → work order → human decision → resume
    4. Fetch result            → final disposition and audit trail

Run from repo root:
    CC_COORD_CONFIG=library/domain-packs/consumer-lending/coordinator_config.yaml \\
    CC_COORD_BASE=library/domain-packs/consumer-lending \\
    python examples/integration_example.py

Or without a server (direct coordinator use):
    cd library/domain-packs/consumer-lending
    python ../../../examples/integration_example.py --direct
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────
# A: Direct coordinator integration (no HTTP server required)
# ─────────────────────────────────────────────────────────────────────

def run_direct():
    """
    Embed Cognitive Core directly in your Python process.
    No server needed — import the coordinator and call it.
    """
    from cognitive_core.coordinator.runtime import Coordinator

    PACK_DIR = REPO_ROOT / "library/domain-packs/consumer-lending"

    # 1. Initialise the coordinator
    coord = Coordinator(
        config_path=str(PACK_DIR / "coordinator_config.yaml"),
        verbose=True,
    )

    print("\n── Cognitive Core Direct Integration ──────────────────────\n")

    # 2. Prepare a case
    case_input = {
        "applicant_name": "Diane Whitfield",
        "applicant_age": 42,
        "loan_amount": 8500,
        "loan_purpose": "Medical expenses",
        "investigation_findings": "",
        "get_credit": {
            "score": 614,
            "utilisation_pct": 68,
            "derogatory_marks_24mo": 3,
            "oldest_account_years": 7,
            "payment_history": "2 lates in 18 months",
        },
        "get_financials": {
            "annual_income_verified": 42000,
            "dti_ratio": 0.48,
            "monthly_obligations": 1680,
            "requested_monthly_payment": 320,
        },
        "get_employment": {
            "status": "part_time",
            "employer": "Various",
            "tenure_years": 0.8,
            "income_source": "hourly",
            "verification_status": "unverified",
        },
        "get_banking": {
            "avg_monthly_balance": 1200,
            "nsf_events_12mo": 0,
            "account_age_years": 12,
        },
        "get_identity": {
            "verification_status": "verified",
            "fraud_flag": False,
        },
    }

    # 3. Submit case
    print(f"Submitting case for: {case_input['applicant_name']}, ${case_input['loan_amount']:,}")
    instance_id = coord.start(
        workflow_type="loan_application_review",
        domain="consumer_lending",
        case_input=case_input,
    )
    print(f"Instance ID: {instance_id}")

    # 4. Check result
    inst = coord.store.get_instance(instance_id)
    print(f"Status: {inst.status.value}")
    print(f"Governance tier: {inst.governance_tier}")

    if inst.status.value == "suspended":
        # 5. Handle HITL pause
        print("\n── GOVERNANCE GATE — Human review required ─────────────\n")
        pending = coord.list_pending_approvals()
        for task in pending:
            if task["instance_id"] == instance_id:
                print(f"  Tier:   {task['governance_tier']}")
                print(f"  Reason: {task.get('reason', '—')}")
                break

        # Work order brief
        suspension = coord.store.get_suspension(instance_id)
        if suspension:
            print(f"  Suspended at: {suspension.suspended_at_step}")

        # Submit human decision
        print("\n  Submitting decision: approve_modified")
        coord.approve(
            instance_id=instance_id,
            approver="integration_example",
            notes="approve_modified: Approve at $6,000 with income verification condition",
        )

        inst = coord.store.get_instance(instance_id)
        print(f"\nStatus after decision: {inst.status.value}")

    # 6. Fetch result and audit trail
    if inst.result:
        print(f"\n── Result ──────────────────────────────────────────────\n")
        print(json.dumps(inst.result, indent=2, default=str))

    ledger = coord.store.get_ledger(instance_id=instance_id)
    print(f"\n── Audit trail: {len(ledger)} entries ──────────────────────\n")
    for entry in ledger:
        ts = time.strftime("%H:%M:%S", time.localtime(entry["created_at"]))
        print(f"  {ts}  {entry['action_type']:<30}  {entry.get('details', {}).get('step_name', '')}")

    # 7. Verify ledger integrity
    result = coord.store.verify_ledger(instance_id)
    valid_str = "✓ verified" if result["valid"] else "✗ tampered"
    print(f"\n── Ledger integrity: {valid_str} ({result['entry_count']} entries checked) ──\n")

    return instance_id


# ─────────────────────────────────────────────────────────────────────
# B: HTTP API integration (server running separately)
# ─────────────────────────────────────────────────────────────────────

def run_via_api(base_url: str = "http://localhost:8000"):
    """
    Integrate with Cognitive Core via its REST API.

    Start the server first:
        CC_COORD_CONFIG=library/domain-packs/consumer-lending/coordinator_config.yaml \\
        CC_COORD_BASE=library/domain-packs/consumer-lending \\
        uvicorn cognitive_core.api.server:app --port 8000
    """
    try:
        import requests
    except ImportError:
        print("requests not installed. Run: pip install requests")
        return

    print(f"\n── Cognitive Core HTTP Integration ({base_url}) ────────────\n")

    case_input = {
        "applicant_name": "Priya Sharma",
        "applicant_age": 34,
        "loan_amount": 12000,
        "loan_purpose": "Home improvement",
        "investigation_findings": "",
        "get_credit": {"score": 748, "utilisation_pct": 18, "derogatory_marks_24mo": 0,
                       "oldest_account_years": 11, "payment_history": "100% on time"},
        "get_financials": {"annual_income_verified": 82000, "dti_ratio": 0.28,
                          "monthly_obligations": 1915, "requested_monthly_payment": 340},
        "get_employment": {"status": "employed_full_time", "employer": "Meridian Health",
                          "tenure_years": 6.5, "income_source": "salary",
                          "verification_status": "verified"},
        "get_banking": {"avg_monthly_balance": 4200, "nsf_events_12mo": 0, "account_age_years": 8},
        "get_identity": {"verification_status": "verified", "fraud_flag": False},
    }

    # 1. Submit case
    r = requests.post(f"{base_url}/api/start", json={
        "workflow_type": "loan_application_review",
        "domain": "consumer_lending",
        "case_input": case_input,
    })
    r.raise_for_status()
    result = r.json()
    instance_id = result["instance_id"]
    print(f"Submitted: {instance_id}")
    print(f"  Trace page: {base_url}/instances/{instance_id}/trace")
    print(f"  SSE stream: {base_url}/api/instances/{instance_id}/stream")

    # 2. Poll until terminal or suspended
    print("\nWaiting for result", end="", flush=True)
    for _ in range(60):
        time.sleep(2)
        r = requests.get(f"{base_url}/api/instances/{instance_id}")
        inst = r.json()
        status = inst.get("status")
        print(".", end="", flush=True)

        if status == "suspended":
            print(f"\n\n── HITL GATE ──────────────────────────────────────────\n")
            # 3. Get work order
            r = requests.get(f"{base_url}/api/instances/{instance_id}/workorder")
            if r.ok:
                wo = r.json()
                print(f"  Tier:   {wo['governance_tier']}")
                print(f"  Brief:  {wo['brief'][:120]}...")
                print(f"  Options: {wo['decision_options']}")

            # 4. Submit decision
            r = requests.post(f"{base_url}/api/instances/{instance_id}/decision", json={
                "decision": "approve",
                "rationale": "Strong profile, approve as submitted",
                "reviewer_id": "integration_example",
            })
            r.raise_for_status()
            print(f"\n  Decision submitted. Workflow resuming...")
            continue

        if status in ("completed", "failed", "terminated"):
            print(f"\n\nFinal status: {status}")
            if inst.get("result"):
                print(f"Result: {json.dumps(inst['result'], indent=2, default=str)[:500]}")
            break

    # 5. Verify ledger integrity
    r = requests.get(f"{base_url}/api/instances/{instance_id}/verify")
    if r.ok:
        v = r.json()
        valid_str = "✓ verified" if v["valid"] else f"✗ tampered at entry {v['first_invalid_entry']}"
        print(f"\nLedger: {valid_str} ({v['entries_checked']} entries)")

    return instance_id


# ─────────────────────────────────────────────────────────────────────
# C: SSE stream consumer
# ─────────────────────────────────────────────────────────────────────

def consume_sse(base_url: str, instance_id: str):
    """
    Subscribe to the SSE action ledger stream for a workflow instance.

    Events: step_started, step_completed, governance_decision,
            hitl_requested, hitl_resolved, workflow_completed, workflow_failed
    """
    try:
        import sseclient
        import requests
    except ImportError:
        print("sseclient-py not installed. Run: pip install sseclient-py requests")
        return

    url = f"{base_url}/api/instances/{instance_id}/stream"
    print(f"Connecting to SSE stream: {url}")

    response = requests.get(url, stream=True)
    client = sseclient.SSEClient(response)

    for event in client.events():
        if not event.data or event.data.startswith(":"):
            continue
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError:
            continue

        evt = data.get("event", event.event)
        step = data.get("step_name", "")
        primitive = data.get("primitive", "")
        elapsed = data.get("elapsed_ms", 0)
        ts = time.strftime("%H:%M:%S")

        print(f"  {ts}  {evt:<22}  {step:<28}  {primitive:<12}  {elapsed}ms")

        if evt in ("workflow_completed", "workflow_failed"):
            print(f"\n  Final: {data.get('status')} in {data.get('elapsed_seconds', 0):.1f}s")
            break


# ─────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Core Integration Example")
    parser.add_argument("--direct", action="store_true",
                        help="Run via direct coordinator import (no server needed)")
    parser.add_argument("--api", action="store_true",
                        help="Run via HTTP API (server must be running)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="API base URL (default: http://localhost:8000)")
    args = parser.parse_args()

    if args.api:
        run_via_api(args.url)
    else:
        # Default: direct coordinator usage
        run_direct()
