"""
Cognitive Core — Fixture API Registry

Registers tools into a ToolRegistry that query the fixture SQLite database
with the same parameter signatures that production REST APIs would use.

Each tool models a production API endpoint:
    get_member(member_id)           → GET /v1/members/{member_id}
    get_accounts(member_id)         → GET /v1/accounts?member_id=...
    get_transactions(member_id, **) → GET /v1/transactions?member_id=...&from=...&to=...&category=...
    get_loans(member_id)            → GET /v1/loans?member_id=...
    get_dispute(dispute_id)         → GET /v1/disputes/{dispute_id}
    get_complaint(complaint_id)     → GET /v1/complaints/{complaint_id}
    get_fraud_score(transaction_id) → GET /v1/fraud/scores?transaction_id=...
    get_devices(member_id)          → GET /v1/devices?member_id=...
    get_aml_alert(alert_id)         → GET /v1/aml/alerts/{alert_id}
    get_check_deposit(deposit_id)   → GET /v1/deposits/{deposit_id}
    get_nsf_events(member_id)       → GET /v1/accounts/{account_id}/nsf
    get_financial_goals(member_id)  → GET /v1/goals?member_id=...
    get_spending_summary(member_id) → GET /v1/spending/summary?member_id=...&month=...
    get_patient(patient_id)         → GET /v1/triage/patients/{patient_id}
    get_regulation(regulation_id)   → GET /v1/regulations/{regulation_id}

In production, swap this module for one where each function calls
the real REST endpoint. The ToolRegistry interface is identical.

Usage:
    from fixtures.api import create_service_registry
    registry = create_service_registry()
    # Pass to compose_workflow(..., tool_registry=registry)
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

import importlib.util
import sys
import os

# Import ToolRegistry directly from engine/tools.py to avoid
# engine/__init__.py which pulls in langgraph and other heavy deps.
# This keeps the fixtures package lightweight.
_tools_path = str(Path(__file__).parent.parent / "engine" / "tools.py")
_spec = importlib.util.spec_from_file_location("engine.tools", _tools_path)
_tools_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tools_mod)
ToolRegistry = _tools_mod.ToolRegistry

DB_PATH = Path(__file__).parent / "cognitive_core.db"


def _get_conn(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    # Parse JSON strings back to lists/dicts
    for k, v in d.items():
        if isinstance(v, str) and v.startswith(("[", "{")):
            try:
                d[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
    # Remove internal column
    d.pop("source_case", None)
    return d


def _rows_to_list(rows: list) -> list[dict]:
    return [_row_to_dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════
# API-shaped query functions
# ═══════════════════════════════════════════════════════════════════════
# Each function takes (context: dict) → dict, matching the DataTool protocol.
# The context dict contains the query parameters — same as what would be
# extracted from an HTTP request in production.

def _make_get_member(db_path):
    """GET /v1/members/{member_id}"""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)
        row = conn.execute("SELECT * FROM members WHERE member_id = ?", (mid,)).fetchone()
        conn.close()
        if row is None:
            return {"error": f"Member {mid} not found", "status": 404}
        return _row_to_dict(row)
    return fn


def _make_get_accounts(db_path):
    """GET /v1/accounts?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)
        rows = conn.execute("SELECT * FROM accounts WHERE member_id = ?", (mid,)).fetchall()
        conn.close()
        return {"accounts": _rows_to_list(rows), "count": len(rows)}
    return fn


def _make_get_transactions(db_path):
    """GET /v1/transactions?member_id=...&from=...&to=...&category=...&limit=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}

        query = "SELECT * FROM transactions WHERE member_id = ?"
        params: list = [mid]

        # Optional filters — same as query params in prod
        if context.get("from_date"):
            query += " AND date >= ?"
            params.append(context["from_date"])
        if context.get("to_date"):
            query += " AND date <= ?"
            params.append(context["to_date"])
        if context.get("category"):
            query += " AND category = ?"
            params.append(context["category"])
        if context.get("type"):
            query += " AND type = ?"
            params.append(context["type"])
        if context.get("merchant"):
            query += " AND merchant LIKE ?"
            params.append(f"%{context['merchant']}%")

        query += " ORDER BY date DESC"
        limit = int(context.get("limit", 100))
        query += f" LIMIT {limit}"

        conn = _get_conn(db_path)
        rows = conn.execute(query, params).fetchall()
        conn.close()
        return {"transactions": _rows_to_list(rows), "count": len(rows)}
    return fn


def _make_get_loans(db_path):
    """GET /v1/loans?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)
        rows = conn.execute("SELECT * FROM loans WHERE member_id = ?", (mid,)).fetchall()
        conn.close()
        return {"loans": _rows_to_list(rows), "count": len(rows)}
    return fn


def _make_get_dispute(db_path):
    """GET /v1/disputes/{dispute_id} or GET /v1/disputes?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        conn = _get_conn(db_path)
        did = context.get("dispute_id")
        if did:
            row = conn.execute("SELECT * FROM disputes WHERE dispute_id = ?", (did,)).fetchone()
            conn.close()
            if row is None:
                return {"error": f"Dispute {did} not found", "status": 404}
            return _row_to_dict(row)
        mid = context.get("member_id")
        if mid:
            rows = conn.execute("SELECT * FROM disputes WHERE member_id = ?", (mid,)).fetchall()
            conn.close()
            return {"disputes": _rows_to_list(rows), "count": len(rows)}
        conn.close()
        return {"error": "dispute_id or member_id required", "status": 400}
    return fn


def _make_get_complaint(db_path):
    """GET /v1/complaints/{complaint_id} or GET /v1/complaints?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        conn = _get_conn(db_path)
        cid = context.get("complaint_id")
        if cid:
            row = conn.execute("SELECT * FROM complaints WHERE complaint_id = ?", (cid,)).fetchone()
            conn.close()
            if row is None:
                return {"error": f"Complaint {cid} not found", "status": 404}
            return _row_to_dict(row)
        mid = context.get("member_id")
        if mid:
            rows = conn.execute("SELECT * FROM complaints WHERE member_id = ?", (mid,)).fetchall()
            conn.close()
            return {"complaints": _rows_to_list(rows), "count": len(rows)}
        conn.close()
        return {"error": "complaint_id or member_id required", "status": 400}
    return fn


def _make_get_fraud_score(db_path):
    """GET /v1/fraud/scores?transaction_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        txn_id = context.get("transaction_id") or context.get("transaction_detail", {}).get("transaction_id")
        if not txn_id:
            return {"error": "transaction_id required", "status": 400}
        conn = _get_conn(db_path)
        row = conn.execute(
            "SELECT * FROM fraud_scores WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
            (txn_id,)).fetchone()
        conn.close()
        if row is None:
            return {"error": f"No fraud score for {txn_id}", "status": 404}
        result = _row_to_dict(row)
        result.pop("id", None)
        return result
    return fn


def _make_get_devices(db_path):
    """GET /v1/devices?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)
        rows = conn.execute("SELECT * FROM device_fingerprints WHERE member_id = ?", (mid,)).fetchall()
        conn.close()
        return {"devices": _rows_to_list(rows), "count": len(rows)}
    return fn


def _make_get_aml_alert(db_path):
    """GET /v1/aml/alerts/{alert_id}"""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        aid = context.get("alert_id")
        if not aid:
            return {"error": "alert_id required", "status": 400}
        conn = _get_conn(db_path)
        row = conn.execute("SELECT * FROM aml_alerts WHERE alert_id = ?", (aid,)).fetchone()
        conn.close()
        if row is None:
            return {"error": f"Alert {aid} not found", "status": 404}
        return _row_to_dict(row)
    return fn


def _make_get_check_deposit(db_path):
    """GET /v1/deposits/{deposit_id} or GET /v1/deposits?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        conn = _get_conn(db_path)
        dep_id = context.get("deposit_id")
        if dep_id:
            row = conn.execute("SELECT * FROM check_deposits WHERE deposit_id = ?", (dep_id,)).fetchone()
            conn.close()
            if row is None:
                return {"error": f"Deposit {dep_id} not found", "status": 404}
            return _row_to_dict(row)
        mid = context.get("member_id")
        if mid:
            rows = conn.execute("SELECT * FROM check_deposits WHERE member_id = ?", (mid,)).fetchall()
            conn.close()
            return {"deposits": _rows_to_list(rows), "count": len(rows)}
        conn.close()
        return {"error": "deposit_id or member_id required", "status": 400}
    return fn


def _make_get_nsf_events(db_path):
    """GET /v1/accounts/{account_id}/nsf or GET /v1/nsf?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        conn = _get_conn(db_path)
        acct_id = context.get("account_id")
        if acct_id:
            rows = conn.execute("SELECT * FROM nsf_events WHERE account_id = ?", (acct_id,)).fetchall()
        else:
            mid = context.get("member_id")
            if not mid:
                conn.close()
                return {"error": "account_id or member_id required", "status": 400}
            rows = conn.execute("SELECT * FROM nsf_events WHERE member_id = ?", (mid,)).fetchall()
        conn.close()
        return {"nsf_events": _rows_to_list(rows), "count": len(rows)}
    return fn


def _make_get_financial_goals(db_path):
    """GET /v1/goals?member_id=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)
        rows = conn.execute("SELECT * FROM financial_goals WHERE member_id = ?", (mid,)).fetchall()
        conn.close()
        results = _rows_to_list(rows)
        for r in results:
            r.pop("id", None)
        return {"goals": results, "count": len(results)}
    return fn


def _make_get_spending_summary(db_path):
    """GET /v1/spending/summary?member_id=...&month=...&from=...&to=..."""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        mid = context.get("member_id") or context.get("member_profile", {}).get("member_id")
        if not mid:
            return {"error": "member_id required", "status": 400}
        conn = _get_conn(db_path)

        query = "SELECT * FROM monthly_summaries WHERE member_id = ?"
        params: list = [mid]

        if context.get("month"):
            query += " AND month = ?"
            params.append(context["month"])
        elif context.get("from_month") or context.get("to_month"):
            if context.get("from_month"):
                query += " AND month >= ?"
                params.append(context["from_month"])
            if context.get("to_month"):
                query += " AND month <= ?"
                params.append(context["to_month"])

        query += " ORDER BY month"
        rows = conn.execute(query, params).fetchall()
        conn.close()
        results = _rows_to_list(rows)
        for r in results:
            r.pop("id", None)
        return {"summaries": results, "count": len(results)}
    return fn


def _make_get_patient(db_path):
    """GET /v1/triage/patients/{patient_id}"""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        pid = context.get("patient_id", "PAT-TRIAGE-001")
        conn = _get_conn(db_path)
        row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (pid,)).fetchone()
        conn.close()
        if row is None:
            return {"error": f"Patient {pid} not found", "status": 404}
        return _row_to_dict(row)
    return fn


def _make_get_regulation(db_path):
    """GET /v1/regulations/{regulation_id}"""
    def fn(context: dict[str, Any]) -> dict[str, Any]:
        rid = context.get("regulation_id") or context.get("regulation", {}).get("federal_register")
        conn = _get_conn(db_path)
        if rid:
            row = conn.execute("SELECT * FROM regulations WHERE regulation_id = ?", (rid,)).fetchone()
        else:
            # Return first regulation if no ID specified
            row = conn.execute("SELECT * FROM regulations LIMIT 1").fetchone()
        conn.close()
        if row is None:
            return {"error": "Regulation not found", "status": 404}
        return _row_to_dict(row)
    return fn


# ═══════════════════════════════════════════════════════════════════════
# Registry factory
# ═══════════════════════════════════════════════════════════════════════

def create_service_registry(db_path: str | Path = DB_PATH) -> ToolRegistry:
    """
    Create a ToolRegistry backed by the service data layer.

    This is the DIRECT (in-process) registry — tools query SQLite
    through Python function calls. Use this when:
      - Running without an MCP server (simple dev/test)
      - Performance matters (no serialization overhead)

    For MCP-backed retrieval, start the data_services MCP server:
      python mcp_servers/data_services.py --http --port 8200
    and configure the Retrieve primitive to discover tools via MCP.

    Both paths expose identical tool names and signatures — the Retrieve
    primitive doesn't know or care which backing is in use.

    Args:
        db_path: Path to fixture SQLite database.

    Returns:
        ToolRegistry with all data service tools registered.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Fixture database not found at {db_path}. "
            f"Run: python -m fixtures.db build"
        )

    registry = ToolRegistry()

    # ── Member 360 / Core Banking ─────────────────────────────────
    registry.register(
        name="get_member",
        fn=_make_get_member(db_path),
        description="Member profile: demographics, tenure, products, credit score, military status. Params: member_id",
        latency_hint_ms=50,
        required=True,
    )

    # ── Account Management ────────────────────────────────────────
    registry.register(
        name="get_accounts",
        fn=_make_get_accounts(db_path),
        description="All accounts for a member: balances, types, status. Params: member_id",
        latency_hint_ms=40,
    )

    # ── Transaction Ledger ────────────────────────────────────────
    registry.register(
        name="get_transactions",
        fn=_make_get_transactions(db_path),
        description="Transaction history with filtering. Params: member_id (required), from_date, to_date, category, type, merchant, limit",
        latency_hint_ms=120,
    )

    # ── Loan Servicing ────────────────────────────────────────────
    registry.register(
        name="get_loans",
        fn=_make_get_loans(db_path),
        description="Active loans: mortgage, auto, personal with balances and payment status. Params: member_id",
        latency_hint_ms=60,
    )

    # ── Dispute Management ────────────────────────────────────────
    registry.register(
        name="get_dispute",
        fn=_make_get_dispute(db_path),
        description="Dispute details by ID or list disputes for a member. Params: dispute_id OR member_id",
        latency_hint_ms=70,
    )

    # ── Complaint / Case Management ───────────────────────────────
    registry.register(
        name="get_complaint",
        fn=_make_get_complaint(db_path),
        description="Complaint details by ID or list complaints for a member. Params: complaint_id OR member_id",
        latency_hint_ms=70,
    )

    # ── Real-Time Fraud Engine ────────────────────────────────────
    registry.register(
        name="get_fraud_score",
        fn=_make_get_fraud_score(db_path),
        description="Fraud risk score for a transaction: score (0-1000), risk level, contributing factors. Params: transaction_id",
        latency_hint_ms=200,
    )

    # ── Device Trust Service ──────────────────────────────────────
    registry.register(
        name="get_devices",
        fn=_make_get_devices(db_path),
        description="Known/trusted devices for a member. Params: member_id",
        latency_hint_ms=30,
    )

    # ── AML/BSA Monitoring ────────────────────────────────────────
    registry.register(
        name="get_aml_alert",
        fn=_make_get_aml_alert(db_path),
        description="AML alert details: triggering activity, subject info, prior history. Params: alert_id",
        latency_hint_ms=80,
    )

    # ── Check Processing / Reg CC ─────────────────────────────────
    registry.register(
        name="get_check_deposit",
        fn=_make_get_check_deposit(db_path),
        description="Check deposit details: hold status, Reg CC info, image quality. Params: deposit_id OR member_id",
        latency_hint_ms=60,
    )

    # ── NSF Events ────────────────────────────────────────────────
    registry.register(
        name="get_nsf_events",
        fn=_make_get_nsf_events(db_path),
        description="NSF/overdraft events and fees. Params: account_id OR member_id",
        latency_hint_ms=40,
    )

    # ── Financial Wellness ────────────────────────────────────────
    registry.register(
        name="get_financial_goals",
        fn=_make_get_financial_goals(db_path),
        description="Member financial goals: targets, progress, monthly contributions. Params: member_id",
        latency_hint_ms=50,
    )

    # ── Spend Aggregation ─────────────────────────────────────────
    registry.register(
        name="get_spending_summary",
        fn=_make_get_spending_summary(db_path),
        description="Monthly spending summaries by category. Params: member_id (required), month, from_month, to_month",
        latency_hint_ms=100,
    )

    # ── Patient Triage ────────────────────────────────────────────
    registry.register(
        name="get_patient",
        fn=_make_get_patient(db_path),
        description="Patient intake: demographics, chief complaint, history, vitals. Params: patient_id",
        latency_hint_ms=30,
    )

    # ── Regulatory Change ─────────────────────────────────────────
    registry.register(
        name="get_regulation",
        fn=_make_get_regulation(db_path),
        description="Regulation details: title, agency, effective date, institution impact. Params: regulation_id",
        latency_hint_ms=40,
    )

    return registry
