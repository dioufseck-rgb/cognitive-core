"""
Data Services MCP Server

Exposes member, account, transaction, loan, and other data services
as MCP tools. In dev, backed by the fixture SQLite database. In prod,
swap the backing implementation to call real REST APIs.

The Retrieve primitive discovers these tools at runtime via MCP —
no hardcoded tool lists in workflows or domains.

Transports:
    stdio:  python mcp_servers/data_services.py
    http:   python mcp_servers/data_services.py --http --port 8200

Requires: pip install "mcp[cli]"
"""

import json
import sqlite3
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent.parent / "fixtures" / "cognitive_core.db"

mcp = FastMCP(
    name="data_services",
    instructions=(
        "Core data services for Navy Federal member, account, transaction, "
        "loan, dispute, complaint, and fraud data. Use these tools to "
        "retrieve member information needed for case processing."
    ),
)


# ─── Helpers ──────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, str) and v.startswith(("[", "{")):
            try:
                d[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
    d.pop("source_case", None)
    return d


def _rows_to_list(rows) -> list[dict]:
    return [_row_to_dict(r) for r in rows]


# ─── Member 360 / Core Banking ───────────────────────────────────────

@mcp.tool()
def get_member(member_id: str) -> str:
    """
    Retrieve member profile: demographics, tenure, products, credit score,
    military status, relationship value.

    Args:
        member_id: Member identifier (e.g., MBR-2024-88291)
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"Member {member_id} not found", "status": 404})
    return json.dumps(_row_to_dict(row))


# ─── Account Management ──────────────────────────────────────────────

@mcp.tool()
def get_accounts(member_id: str) -> str:
    """
    Retrieve all accounts for a member: checking, savings, credit card.
    Returns balances, status, and overdraft protection.

    Args:
        member_id: Member identifier
    """
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM accounts WHERE member_id = ?", (member_id,)).fetchall()
    conn.close()
    return json.dumps({"accounts": _rows_to_list(rows), "count": len(rows)})


# ─── Transaction Ledger ──────────────────────────────────────────────

@mcp.tool()
def get_transactions(
    member_id: str,
    from_date: str = "",
    to_date: str = "",
    category: str = "",
    transaction_type: str = "",
    merchant: str = "",
    limit: int = 100,
) -> str:
    """
    Retrieve transaction history with optional filters.

    Args:
        member_id: Member identifier
        from_date: Start date filter (ISO format, e.g., 2025-01-01)
        to_date: End date filter (ISO format)
        category: Category filter (e.g., dining_out, grocery, gas)
        transaction_type: Type filter (e.g., purchase, cash_deposit, ach)
        merchant: Merchant name filter (partial match)
        limit: Max results (default 100)
    """
    query = "SELECT * FROM transactions WHERE member_id = ?"
    params: list = [member_id]

    if from_date:
        query += " AND date >= ?"
        params.append(from_date)
    if to_date:
        query += " AND date <= ?"
        params.append(to_date)
    if category:
        query += " AND category = ?"
        params.append(category)
    if transaction_type:
        query += " AND type = ?"
        params.append(transaction_type)
    if merchant:
        query += " AND merchant LIKE ?"
        params.append(f"%{merchant}%")

    query += f" ORDER BY date DESC LIMIT {limit}"

    conn = _get_conn()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return json.dumps({"transactions": _rows_to_list(rows), "count": len(rows)})


# ─── Loan Servicing ──────────────────────────────────────────────────

@mcp.tool()
def get_loans(member_id: str) -> str:
    """
    Retrieve active loans: mortgage, auto, personal.
    Returns balances, rates, payment status, and days past due.

    Args:
        member_id: Member identifier
    """
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM loans WHERE member_id = ?", (member_id,)).fetchall()
    conn.close()
    return json.dumps({"loans": _rows_to_list(rows), "count": len(rows)})


# ─── Dispute Management ──────────────────────────────────────────────

@mcp.tool()
def get_dispute(dispute_id: str = "", member_id: str = "") -> str:
    """
    Retrieve dispute details by ID, or list all disputes for a member.

    Args:
        dispute_id: Specific dispute identifier
        member_id: Member identifier (returns all disputes for member)
    """
    conn = _get_conn()
    if dispute_id:
        row = conn.execute("SELECT * FROM disputes WHERE dispute_id = ?", (dispute_id,)).fetchone()
        conn.close()
        if row is None:
            return json.dumps({"error": f"Dispute {dispute_id} not found", "status": 404})
        return json.dumps(_row_to_dict(row))
    if member_id:
        rows = conn.execute("SELECT * FROM disputes WHERE member_id = ?", (member_id,)).fetchall()
        conn.close()
        return json.dumps({"disputes": _rows_to_list(rows), "count": len(rows)})
    return json.dumps({"error": "dispute_id or member_id required", "status": 400})


# ─── Complaint / Case Management ─────────────────────────────────────

@mcp.tool()
def get_complaint(complaint_id: str = "", member_id: str = "") -> str:
    """
    Retrieve complaint details by ID, or list all complaints for a member.

    Args:
        complaint_id: Specific complaint identifier
        member_id: Member identifier (returns all complaints for member)
    """
    conn = _get_conn()
    if complaint_id:
        row = conn.execute("SELECT * FROM complaints WHERE complaint_id = ?", (complaint_id,)).fetchone()
        conn.close()
        if row is None:
            return json.dumps({"error": f"Complaint {complaint_id} not found", "status": 404})
        return json.dumps(_row_to_dict(row))
    if member_id:
        rows = conn.execute("SELECT * FROM complaints WHERE member_id = ?", (member_id,)).fetchall()
        conn.close()
        return json.dumps({"complaints": _rows_to_list(rows), "count": len(rows)})
    return json.dumps({"error": "complaint_id or member_id required", "status": 400})


# ─── Real-Time Fraud Engine ──────────────────────────────────────────

@mcp.tool()
def get_fraud_score(transaction_id: str) -> str:
    """
    Retrieve fraud risk score for a transaction.
    Returns score (0-1000), risk level, and contributing factors.

    Args:
        transaction_id: Transaction identifier
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM fraud_scores WHERE transaction_id = ? ORDER BY id DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"No fraud score for {transaction_id}", "status": 404})
    result = _row_to_dict(row)
    result.pop("id", None)
    return result if isinstance(result, str) else json.dumps(result)


# ─── Device Trust Service ────────────────────────────────────────────

@mcp.tool()
def get_devices(member_id: str) -> str:
    """
    Retrieve known/trusted devices for a member.

    Args:
        member_id: Member identifier
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM device_fingerprints WHERE member_id = ?", (member_id,)
    ).fetchall()
    conn.close()
    return json.dumps({"devices": _rows_to_list(rows), "count": len(rows)})


# ─── AML/BSA Monitoring ──────────────────────────────────────────────

@mcp.tool()
def get_aml_alert(alert_id: str) -> str:
    """
    Retrieve AML alert details: triggering activity, subject info,
    prior history, cash deposit totals.

    Args:
        alert_id: Alert identifier (e.g., AML-2026-00847)
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM aml_alerts WHERE alert_id = ?", (alert_id,)).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"Alert {alert_id} not found", "status": 404})
    return json.dumps(_row_to_dict(row))


# ─── Check Processing / Reg CC ───────────────────────────────────────

@mcp.tool()
def get_check_deposit(deposit_id: str = "", member_id: str = "") -> str:
    """
    Retrieve check deposit details: hold status, Reg CC info, image quality.

    Args:
        deposit_id: Specific deposit identifier
        member_id: Member identifier (returns all deposits for member)
    """
    conn = _get_conn()
    if deposit_id:
        row = conn.execute(
            "SELECT * FROM check_deposits WHERE deposit_id = ?", (deposit_id,)
        ).fetchone()
        conn.close()
        if row is None:
            return json.dumps({"error": f"Deposit {deposit_id} not found", "status": 404})
        return json.dumps(_row_to_dict(row))
    if member_id:
        rows = conn.execute(
            "SELECT * FROM check_deposits WHERE member_id = ?", (member_id,)
        ).fetchall()
        conn.close()
        return json.dumps({"deposits": _rows_to_list(rows), "count": len(rows)})
    return json.dumps({"error": "deposit_id or member_id required", "status": 400})


# ─── NSF Events ──────────────────────────────────────────────────────

@mcp.tool()
def get_nsf_events(member_id: str = "", account_id: str = "") -> str:
    """
    Retrieve NSF/overdraft events and fees.

    Args:
        member_id: Member identifier
        account_id: Specific account identifier
    """
    conn = _get_conn()
    if account_id:
        rows = conn.execute(
            "SELECT * FROM nsf_events WHERE account_id = ?", (account_id,)
        ).fetchall()
    elif member_id:
        rows = conn.execute(
            "SELECT * FROM nsf_events WHERE member_id = ?", (member_id,)
        ).fetchall()
    else:
        conn.close()
        return json.dumps({"error": "member_id or account_id required", "status": 400})
    conn.close()
    return json.dumps({"nsf_events": _rows_to_list(rows), "count": len(rows)})


# ─── Financial Wellness ──────────────────────────────────────────────

@mcp.tool()
def get_financial_goals(member_id: str) -> str:
    """
    Retrieve member financial goals: targets, progress, monthly contributions.

    Args:
        member_id: Member identifier
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM financial_goals WHERE member_id = ?", (member_id,)
    ).fetchall()
    conn.close()
    results = _rows_to_list(rows)
    for r in results:
        r.pop("id", None)
    return json.dumps({"goals": results, "count": len(results)})


# ─── Spend Aggregation ───────────────────────────────────────────────

@mcp.tool()
def get_spending_summary(
    member_id: str,
    month: str = "",
    from_month: str = "",
    to_month: str = "",
) -> str:
    """
    Retrieve monthly spending summaries by category.

    Args:
        member_id: Member identifier
        month: Specific month (e.g., 2025-06)
        from_month: Start of range (e.g., 2025-01)
        to_month: End of range (e.g., 2025-12)
    """
    query = "SELECT * FROM monthly_summaries WHERE member_id = ?"
    params: list = [member_id]

    if month:
        query += " AND month = ?"
        params.append(month)
    else:
        if from_month:
            query += " AND month >= ?"
            params.append(from_month)
        if to_month:
            query += " AND month <= ?"
            params.append(to_month)

    query += " ORDER BY month"
    conn = _get_conn()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    results = _rows_to_list(rows)
    for r in results:
        r.pop("id", None)
    return json.dumps({"summaries": results, "count": len(results)})


# ─── Patient Triage ──────────────────────────────────────────────────

@mcp.tool()
def get_patient(patient_id: str = "PAT-TRIAGE-001") -> str:
    """
    Retrieve patient intake: demographics, chief complaint, history, vitals.

    Args:
        patient_id: Patient identifier
    """
    conn = _get_conn()
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": f"Patient {patient_id} not found", "status": 404})
    return json.dumps(_row_to_dict(row))


# ─── Regulatory Change ───────────────────────────────────────────────

@mcp.tool()
def get_regulation(regulation_id: str = "") -> str:
    """
    Retrieve regulation details: title, agency, effective date, institution impact.

    Args:
        regulation_id: Regulation identifier (e.g., 2026-FR-00142)
    """
    conn = _get_conn()
    if regulation_id:
        row = conn.execute(
            "SELECT * FROM regulations WHERE regulation_id = ?", (regulation_id,)
        ).fetchone()
    else:
        row = conn.execute("SELECT * FROM regulations LIMIT 1").fetchone()
    conn.close()
    if row is None:
        return json.dumps({"error": "Regulation not found", "status": 404})
    return json.dumps(_row_to_dict(row))


# ─── Entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = 8200
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
