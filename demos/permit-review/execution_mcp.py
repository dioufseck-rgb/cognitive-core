"""
Permit Review — Execution MCP Server

Provides governed execution tools for the permit review workflow.
Called by the coordinator's execution policy engine after tier clearance.

Tools:
  notify_applicant              Send email notification to applicant
  file_notice_of_exemption      File NOE with county recorder (stub)
  withdraw_notice_of_exemption  Reverse NOE filing (stub)
  file_notice_of_determination  File NOD with county recorder (stub)
  withdraw_notice_of_determination  Reverse NOD filing (stub)
  trigger_public_notice         Trigger public review period (stub)
  update_permit_status          Update permit management system (stub)
  revert_permit_status          Reverse permit status update (stub)
  request_additional_info       Send information request to applicant

Usage:
    python demos/permit-review/execution_mcp.py

    # With real SMTP:
    SMTP_HOST=smtp.example.com SMTP_PORT=587 \\
    SMTP_USER=permits@city.gov SMTP_PASSWORD=xxx \\
    python demos/permit-review/execution_mcp.py

    # Dry run (logs actions, no real email):
    DRY_RUN=true python demos/permit-review/execution_mcp.py

Environment:
    SMTP_HOST       SMTP server hostname (default: localhost)
    SMTP_PORT       SMTP port (default: 1025 — MailHog/debug)
    SMTP_USER       SMTP username (optional)
    SMTP_PASSWORD   SMTP password (optional)
    SMTP_FROM       From address (default: permits@city.gov)
    APPLICANT_EMAIL Applicant email for demo (default: applicant@example.com)
    DRY_RUN         If 'true', log actions without sending (default: false)
    PERMIT_DB       Path to permit status SQLite DB (default: permit_status.db)
"""

import asyncio
import json
import logging
import os
import smtplib
import sqlite3
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("mcp package not installed — run: pip install mcp")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("permit_execution_mcp")

DEMO_DIR = Path(__file__).resolve().parent

# ── Configuration ─────────────────────────────────────────────────

SMTP_HOST     = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT     = int(os.environ.get("SMTP_PORT", "1025"))
SMTP_USER     = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM     = os.environ.get("SMTP_FROM", "permits@city.gov")
APPLICANT_EMAIL = os.environ.get("APPLICANT_EMAIL", "applicant@example.com")
DRY_RUN       = os.environ.get("DRY_RUN", "false").lower() == "true"
PERMIT_DB     = os.environ.get("PERMIT_DB", str(DEMO_DIR / "permit_status.db"))

# ── Email templates ───────────────────────────────────────────────

TEMPLATES = {
    "approval_exempt": {
        "subject": "Permit {permit_number} — Approved (CEQA Categorical Exemption)",
        "body": """Dear Applicant,

Your permit application {permit_number} has been approved.

CEQA Determination: Categorically Exempt
Determination: {determination}

A Notice of Exemption has been filed with the county recorder. Your permit
will be issued upon payment of applicable fees.

If you have questions, please contact the Planning Department.

City Planning Department
"""
    },
    "approval_conditional": {
        "subject": "Permit {permit_number} — Conditionally Approved",
        "body": """Dear Applicant,

Your permit application {permit_number} has been conditionally approved.

CEQA Determination: {determination}

A 20-day public review period is required before the permit can be issued.
You will be notified when the review period concludes.

Conditions of approval have been recorded in your permit file.

City Planning Department
"""
    },
    "hold_pending_attorney_review": {
        "subject": "Permit {permit_number} — On Hold Pending Review",
        "body": """Dear Applicant,

Your permit application {permit_number} requires additional legal review
before a determination can be made.

{message}

City Planning Department
"""
    },
    "information_request": {
        "subject": "Permit {permit_number} — Additional Information Required",
        "body": """Dear Applicant,

Your permit application {permit_number} requires additional information
before we can complete our review.

Required information:
{required_items}

Please submit the requested items within 30 days to avoid application
expiration.

City Planning Department
"""
    },
}

# ── Permit status DB ──────────────────────────────────────────────

def get_permit_db():
    conn = sqlite3.connect(PERMIT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS permit_status (
            permit_number TEXT PRIMARY KEY,
            status TEXT,
            determination TEXT,
            updated_at REAL,
            notes TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS execution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            permit_number TEXT,
            tool TEXT,
            inputs TEXT,
            result TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    return conn

def log_execution(permit_number, tool, inputs, result):
    try:
        conn = get_permit_db()
        conn.execute(
            "INSERT INTO execution_log (permit_number, tool, inputs, result, timestamp) VALUES (?, ?, ?, ?, ?)",
            (permit_number, tool, json.dumps(inputs), json.dumps(result), time.time())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to log execution: {e}")

# ── Email sending ─────────────────────────────────────────────────

def send_email(to_address, subject, body):
    if DRY_RUN:
        log.info(f"[DRY RUN] Email to {to_address}")
        log.info(f"  Subject: {subject}")
        log.info(f"  Body: {body[:100]}...")
        return {"status": "dry_run", "to": to_address, "subject": subject}

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_FROM
        msg["To"] = to_address
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        log.info(f"Email sent to {to_address}: {subject}")
        return {"status": "sent", "to": to_address, "subject": subject,
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        log.error(f"Failed to send email: {e}")
        return {"status": "failed", "error": str(e)}

# ── Tool implementations ──────────────────────────────────────────

def tool_notify_applicant(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    template_name = inputs.get("template", "approval_exempt")
    determination = inputs.get("determination", "")
    message = inputs.get("message", "")

    template = TEMPLATES.get(template_name, TEMPLATES["approval_exempt"])
    subject = template["subject"].format(permit_number=permit_number)
    body = template["body"].format(
        permit_number=permit_number,
        determination=determination,
        message=message,
        required_items=inputs.get("required_items", ""),
    )

    result = send_email(APPLICANT_EMAIL, subject, body)
    log_execution(permit_number, "notify_applicant", inputs, result)
    return result


def tool_file_notice_of_exemption(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    exemption_class = inputs.get("exemption_class", "")
    project_description = inputs.get("project_description", "")

    # In production: POST to county recorder's e-filing API
    noe_number = f"NOE-{permit_number}-{int(time.time())}"
    result = {
        "status": "filed" if not DRY_RUN else "dry_run",
        "noe_number": noe_number,
        "permit_number": permit_number,
        "exemption_class": exemption_class,
        "filed_at": datetime.now().isoformat(),
    }
    log.info(f"[{'DRY RUN' if DRY_RUN else 'FILED'}] NOE {noe_number} for {permit_number}")
    log_execution(permit_number, "file_notice_of_exemption", inputs, result)

    # Update permit status
    try:
        conn = get_permit_db()
        conn.execute(
            "INSERT OR REPLACE INTO permit_status VALUES (?, ?, ?, ?, ?)",
            (permit_number, "approved_exempt", exemption_class, time.time(), noe_number)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Failed to update permit status: {e}")

    return result


def tool_withdraw_notice_of_exemption(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    result = {"status": "withdrawn" if not DRY_RUN else "dry_run",
              "permit_number": permit_number,
              "withdrawn_at": datetime.now().isoformat()}
    log.info(f"[{'DRY RUN' if DRY_RUN else 'WITHDRAWN'}] NOE for {permit_number}")
    log_execution(permit_number, "withdraw_notice_of_exemption", inputs, result)
    return result


def tool_file_notice_of_determination(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    determination_type = inputs.get("determination_type", "")
    nod_number = f"NOD-{permit_number}-{int(time.time())}"
    result = {
        "status": "filed" if not DRY_RUN else "dry_run",
        "nod_number": nod_number,
        "permit_number": permit_number,
        "determination_type": determination_type,
        "filed_at": datetime.now().isoformat(),
    }
    log.info(f"[{'DRY RUN' if DRY_RUN else 'FILED'}] NOD {nod_number} for {permit_number}")
    log_execution(permit_number, "file_notice_of_determination", inputs, result)
    return result


def tool_withdraw_notice_of_determination(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    result = {"status": "withdrawn" if not DRY_RUN else "dry_run",
              "permit_number": permit_number,
              "withdrawn_at": datetime.now().isoformat()}
    log_execution(permit_number, "withdraw_notice_of_determination", inputs, result)
    return result


def tool_trigger_public_notice(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    notice_type = inputs.get("notice_type", "mnd")
    review_period_days = inputs.get("review_period_days", 20)
    start_date = datetime.now().isoformat()
    result = {
        "status": "triggered" if not DRY_RUN else "dry_run",
        "permit_number": permit_number,
        "notice_type": notice_type,
        "review_period_days": review_period_days,
        "start_date": start_date,
        "statutory_basis": "Pub. Resources Code §21092",
    }
    log.info(f"[{'DRY RUN' if DRY_RUN else 'TRIGGERED'}] Public notice for {permit_number} "
             f"({review_period_days} days)")
    log_execution(permit_number, "trigger_public_notice", inputs, result)
    return result


def tool_update_permit_status(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    status = inputs.get("status", "")
    try:
        conn = get_permit_db()
        conn.execute(
            "INSERT OR REPLACE INTO permit_status VALUES (?, ?, ?, ?, ?)",
            (permit_number, status, "", time.time(), "")
        )
        conn.commit()
        conn.close()
        result = {"status": "updated", "permit_number": permit_number,
                  "new_status": status}
    except Exception as e:
        result = {"status": "failed", "error": str(e)}
    log_execution(permit_number, "update_permit_status", inputs, result)
    return result


def tool_revert_permit_status(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    try:
        conn = get_permit_db()
        conn.execute("DELETE FROM permit_status WHERE permit_number = ?",
                     (permit_number,))
        conn.commit()
        conn.close()
        result = {"status": "reverted", "permit_number": permit_number}
    except Exception as e:
        result = {"status": "failed", "error": str(e)}
    log_execution(permit_number, "revert_permit_status", inputs, result)
    return result


def tool_request_additional_info(inputs: dict) -> dict:
    permit_number = inputs.get("permit_number", "UNKNOWN")
    required_items = inputs.get("required_items", [])
    if isinstance(required_items, list):
        items_str = "\n".join(f"  - {item}" for item in required_items)
    else:
        items_str = str(required_items)

    body = TEMPLATES["information_request"]["body"].format(
        permit_number=permit_number,
        required_items=items_str,
    )
    subject = TEMPLATES["information_request"]["subject"].format(
        permit_number=permit_number
    )
    result = send_email(APPLICANT_EMAIL, subject, body)
    log_execution(permit_number, "request_additional_info", inputs, result)
    return result


# ── Tool dispatch ─────────────────────────────────────────────────

TOOLS = {
    "notify_applicant": tool_notify_applicant,
    "file_notice_of_exemption": tool_file_notice_of_exemption,
    "withdraw_notice_of_exemption": tool_withdraw_notice_of_exemption,
    "file_notice_of_determination": tool_file_notice_of_determination,
    "withdraw_notice_of_determination": tool_withdraw_notice_of_determination,
    "trigger_public_notice": tool_trigger_public_notice,
    "update_permit_status": tool_update_permit_status,
    "revert_permit_status": tool_revert_permit_status,
    "request_additional_info": tool_request_additional_info,
}

TOOL_SCHEMAS = [
    {"name": "notify_applicant",
     "description": "Send governed email notification to the permit applicant",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "template": {"type": "string",
                      "enum": ["approval_exempt", "approval_conditional",
                               "hold_pending_attorney_review", "information_request"]},
         "determination": {"type": "string"},
         "message": {"type": "string"},
     }, "required": ["permit_number", "template"]}},

    {"name": "file_notice_of_exemption",
     "description": "File Notice of Exemption with county recorder (14 CCR §15062)",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "exemption_class": {"type": "string"},
         "project_description": {"type": "string"},
     }, "required": ["permit_number", "exemption_class"]}},

    {"name": "withdraw_notice_of_exemption",
     "description": "Withdraw a previously filed Notice of Exemption",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
     }, "required": ["permit_number"]}},

    {"name": "file_notice_of_determination",
     "description": "File Notice of Determination with county recorder",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "determination_type": {"type": "string",
                                "enum": ["negative_declaration",
                                         "mitigated_negative_declaration",
                                         "eir_certified"]},
     }, "required": ["permit_number", "determination_type"]}},

    {"name": "withdraw_notice_of_determination",
     "description": "Withdraw a previously filed Notice of Determination",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
     }, "required": ["permit_number"]}},

    {"name": "trigger_public_notice",
     "description": "Trigger statutory public notice and review period (Pub. Resources Code §21092)",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "notice_type": {"type": "string", "enum": ["mnd", "eir", "nd"]},
         "review_period_days": {"type": "integer"},
     }, "required": ["permit_number", "notice_type"]}},

    {"name": "update_permit_status",
     "description": "Update permit status in permit management system",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "status": {"type": "string"},
     }, "required": ["permit_number", "status"]}},

    {"name": "revert_permit_status",
     "description": "Revert permit status (compensation/rollback operation)",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
     }, "required": ["permit_number"]}},

    {"name": "request_additional_info",
     "description": "Send information request to applicant and suspend workflow",
     "inputSchema": {"type": "object", "properties": {
         "permit_number": {"type": "string"},
         "required_items": {"type": "array", "items": {"type": "string"}},
     }, "required": ["permit_number", "required_items"]}},
]


# ── MCP Server ────────────────────────────────────────────────────

if MCP_AVAILABLE:
    server = Server("permit-execution-mcp")

    @server.list_tools()
    async def list_tools():
        return [Tool(**schema) for schema in TOOL_SCHEMAS]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name not in TOOLS:
            return [TextContent(type="text",
                                text=json.dumps({"error": f"Unknown tool: {name}"}))]
        try:
            result = TOOLS[name](arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            log.error(f"Tool {name} failed: {e}")
            return [TextContent(type="text",
                                text=json.dumps({"error": str(e), "tool": name}))]

    async def main():
        log.info("Permit execution MCP server starting")
        log.info(f"SMTP: {SMTP_HOST}:{SMTP_PORT} | DRY_RUN: {DRY_RUN}")
        log.info(f"Tools: {list(TOOLS.keys())}")
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream,
                             server.create_initialization_options())

    if __name__ == "__main__":
        asyncio.run(main())

else:
    # Standalone mode — can be called directly without MCP
    if __name__ == "__main__":
        import sys
        if len(sys.argv) >= 3:
            tool_name = sys.argv[1]
            inputs = json.loads(sys.argv[2])
            if tool_name in TOOLS:
                result = TOOLS[tool_name](inputs)
                print(json.dumps(result, indent=2))
            else:
                print(f"Unknown tool: {tool_name}")
                print(f"Available: {list(TOOLS.keys())}")
        else:
            print("Usage: python execution_mcp.py <tool_name> '<json_inputs>'")
            print(f"Tools: {list(TOOLS.keys())}")
