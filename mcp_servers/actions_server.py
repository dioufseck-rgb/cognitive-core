"""
Actions MCP Server

Write-side MCP server for the Act primitive. Exposes executable actions
as MCP tools — the Act primitive resolves what to do, checks authorization,
then calls these tools to close the loop.

This server includes one REAL action (send_email via Gmail SMTP) and
several simulated actions for dev/test. In production, replace the
simulations with real API integrations.

Transports:
    stdio:  python mcp_servers/actions_server.py
    http:   python mcp_servers/actions_server.py --http --port 8200

Requires: pip install "mcp[cli]"
"""

import json
import os
import smtplib
import sys
import uuid
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    name="actions",
    instructions=(
        "Action execution tools for the Cognitive Core Act primitive. "
        "These tools perform real side effects — send emails, update systems, "
        "file reports. Always check authorization before calling."
    ),
)


# ─── CONFIGURATION ────────────────────────────────────────────────────

SMTP_CONFIG = {
    "sender": os.environ.get("SMTP_SENDER", ""),
    "app_password": os.environ.get("SMTP_APP_PASSWORD", ""),
    "smtp_host": os.environ.get("SMTP_HOST", "smtp.gmail.com"),
    "smtp_port": int(os.environ.get("SMTP_PORT", "587")),
}

DEFAULT_RECIPIENT = os.environ.get("SMTP_DEFAULT_RECIPIENT", "")


# ─── REAL ACTIONS ─────────────────────────────────────────────────────

@mcp.tool()
def send_email(
    recipient: str,
    subject: str,
    body: str,
    body_format: str = "plain",
    cc: str = "",
    reply_to: str = "",
) -> str:
    """
    Send an email via Gmail SMTP. This is a REAL action with side effects.
    The email will actually be delivered.

    Authorization: system (no approval needed for notifications)

    Args:
        recipient: Email address to send to
        subject: Email subject line
        body: Email body content
        body_format: "plain" for text, "html" for HTML formatted email
        cc: Optional CC recipients (comma-separated)
        reply_to: Optional reply-to address
    """
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"Cognitive Core <{SMTP_CONFIG['sender']}>"
        msg["To"] = recipient
        msg["Subject"] = subject
        msg["X-Cognitive-Core"] = "act-primitive"
        msg["X-Execution-ID"] = uuid.uuid4().hex[:12].upper()

        if cc:
            msg["Cc"] = cc
        if reply_to:
            msg["Reply-To"] = reply_to

        # Attach body
        mime_type = "html" if body_format == "html" else "plain"
        msg.attach(MIMEText(body, mime_type))

        # Send via Gmail SMTP
        with smtplib.SMTP(SMTP_CONFIG["smtp_host"], SMTP_CONFIG["smtp_port"]) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_CONFIG["sender"], SMTP_CONFIG["app_password"])

            all_recipients = [recipient]
            if cc:
                all_recipients.extend([a.strip() for a in cc.split(",")])

            server.sendmail(SMTP_CONFIG["sender"], all_recipients, msg.as_string())

        confirmation_id = f"EMAIL-{uuid.uuid4().hex[:8].upper()}"

        return json.dumps({
            "status": "executed",
            "action": "send_email",
            "target_system": "gmail_smtp",
            "confirmation_id": confirmation_id,
            "recipient": recipient,
            "subject": subject,
            "body_length": len(body),
            "body_format": body_format,
            "sent_at": datetime.now().isoformat(),
            "reversible": False,
            "side_effects": [
                "Email delivered to recipient inbox",
                "Copy stored in sender's Sent folder",
            ],
        })

    except smtplib.SMTPAuthenticationError as e:
        return json.dumps({
            "status": "failed",
            "action": "send_email",
            "target_system": "gmail_smtp",
            "error": f"SMTP authentication failed: {e}",
            "reversible": False,
        })
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "action": "send_email",
            "target_system": "gmail_smtp",
            "error": str(e),
            "reversible": False,
        })


# ─── SIMULATED ACTIONS (replace with real APIs in production) ─────────

@mcp.tool()
def issue_provisional_credit(
    member_id: str,
    account_id: str,
    amount: float,
    reason: str,
    dispute_id: str = "",
) -> str:
    """
    Issue a provisional credit to a member's account.
    SIMULATED in dev — returns a plausible confirmation.

    Authorization: agent (supervisor if amount > $500)
    Reversible: yes

    Args:
        member_id: Member identifier
        account_id: Account to credit
        amount: Credit amount in dollars
        reason: Reason for the provisional credit
        dispute_id: Associated dispute ID if applicable
    """
    confirmation_id = f"PC-{uuid.uuid4().hex[:8].upper()}"
    rollback_handle = f"RB-{uuid.uuid4().hex[:8].upper()}"

    return json.dumps({
        "status": "simulated",
        "action": "issue_provisional_credit",
        "target_system": "core_banking",
        "confirmation_id": confirmation_id,
        "rollback_handle": rollback_handle,
        "details": {
            "member_id": member_id,
            "account_id": account_id,
            "amount": amount,
            "reason": reason,
            "dispute_id": dispute_id,
            "posted_at": datetime.now().isoformat(),
        },
        "reversible": True,
        "side_effects": [
            f"Account {account_id} balance increased by ${amount:.2f}",
            "Transaction posted to next statement",
            "Member notification sent (email + push)",
        ],
        "note": "SIMULATED — no real credit issued. Replace with core banking API in production.",
    })


@mcp.tool()
def update_case_status(
    case_id: str,
    new_status: str,
    resolution_code: str = "",
    notes: str = "",
) -> str:
    """
    Update a dispute or case status in the case management system.
    SIMULATED in dev.

    Authorization: agent
    Reversible: yes

    Args:
        case_id: Case or dispute identifier
        new_status: New status (e.g., "resolved", "pending_review", "escalated")
        resolution_code: Resolution code if closing the case
        notes: Free-text notes for the case history
    """
    confirmation_id = f"CS-{uuid.uuid4().hex[:8].upper()}"
    rollback_handle = f"RB-{uuid.uuid4().hex[:8].upper()}"

    return json.dumps({
        "status": "simulated",
        "action": "update_case_status",
        "target_system": "case_management",
        "confirmation_id": confirmation_id,
        "rollback_handle": rollback_handle,
        "details": {
            "case_id": case_id,
            "new_status": new_status,
            "resolution_code": resolution_code,
            "notes": notes,
            "updated_at": datetime.now().isoformat(),
        },
        "reversible": True,
        "side_effects": [
            f"Case {case_id} status changed to {new_status}",
            "Audit trail entry created",
            "SLA timer updated",
        ],
        "note": "SIMULATED — no real case update. Replace with case management API in production.",
    })


@mcp.tool()
def file_regulatory_report(
    report_type: str,
    subject_id: str,
    narrative: str,
    amount: float = 0.0,
    filing_deadline: str = "",
) -> str:
    """
    File a regulatory report (SAR, CTR, etc.).
    SIMULATED in dev.

    Authorization: compliance (requires compliance officer sign-off)
    Reversible: no
    Requires confirmation: yes

    Args:
        report_type: Report type — SAR, CTR, CMIR, etc.
        subject_id: Subject of the report (member ID, entity ID)
        narrative: Filing narrative
        amount: Amount involved
        filing_deadline: Regulatory filing deadline (ISO date)
    """
    confirmation_id = f"REG-{uuid.uuid4().hex[:8].upper()}"

    return json.dumps({
        "status": "simulated",
        "action": "file_regulatory_report",
        "target_system": "regulatory_filing",
        "confirmation_id": confirmation_id,
        "details": {
            "report_type": report_type,
            "subject_id": subject_id,
            "narrative_length": len(narrative),
            "amount": amount,
            "filing_deadline": filing_deadline,
            "filed_at": datetime.now().isoformat(),
        },
        "reversible": False,
        "side_effects": [
            f"{report_type} report filed with FinCEN",
            "Regulatory clock started",
            "Internal compliance log updated",
            "Subject flagged for enhanced monitoring",
        ],
        "note": "SIMULATED — no real filing. Replace with FinCEN e-filing API in production.",
    })


@mcp.tool()
def schedule_callback(
    member_id: str,
    callback_datetime: str,
    reason: str,
    assigned_to: str = "next_available",
    channel: str = "phone",
) -> str:
    """
    Schedule a follow-up callback or contact with a member.
    SIMULATED in dev.

    Authorization: agent
    Reversible: yes

    Args:
        member_id: Member to contact
        callback_datetime: When to call back (ISO datetime)
        reason: Reason for the callback
        assigned_to: Agent or queue to assign to
        channel: Contact channel — phone, video, secure_message
    """
    confirmation_id = f"CB-{uuid.uuid4().hex[:8].upper()}"

    return json.dumps({
        "status": "simulated",
        "action": "schedule_callback",
        "target_system": "scheduling",
        "confirmation_id": confirmation_id,
        "details": {
            "member_id": member_id,
            "scheduled_for": callback_datetime,
            "reason": reason,
            "assigned_to": assigned_to,
            "channel": channel,
        },
        "reversible": True,
        "side_effects": [
            "Callback added to agent queue",
            f"Member notified of scheduled {channel} callback",
        ],
        "note": "SIMULATED — no real scheduling. Replace with scheduling API in production.",
    })


# ─── RESOURCES ────────────────────────────────────────────────────────

@mcp.resource("actions://catalog")
def action_catalog() -> str:
    """Catalog of all available actions with authorization requirements."""
    return json.dumps({
        "actions": [
            {
                "name": "send_email",
                "real": True,
                "authorization": "system",
                "reversible": False,
                "description": "Send email via Gmail SMTP",
            },
            {
                "name": "issue_provisional_credit",
                "real": False,
                "authorization": "agent (supervisor if >$500)",
                "reversible": True,
                "description": "Issue provisional credit to member account",
            },
            {
                "name": "update_case_status",
                "real": False,
                "authorization": "agent",
                "reversible": True,
                "description": "Update case/dispute status",
            },
            {
                "name": "file_regulatory_report",
                "real": False,
                "authorization": "compliance",
                "reversible": False,
                "description": "File SAR/CTR with FinCEN",
            },
            {
                "name": "schedule_callback",
                "real": False,
                "authorization": "agent",
                "reversible": True,
                "description": "Schedule member follow-up",
            },
        ],
        "note": "Only send_email is a real action. Others are simulated for dev/test.",
    })


# ─── ENTRYPOINT ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = 8200
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
