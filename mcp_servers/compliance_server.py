"""
Compliance MCP Server

Example MCP server that exposes compliance tools and resources.
In production, these would query real compliance databases, regulation
text search, examination findings, and enforcement action databases.

This server demonstrates the pattern — domain teams build MCP servers,
the Retrieve primitive discovers and calls them dynamically.

Transports:
    stdio:  python mcp_servers/compliance_server.py
    http:   python mcp_servers/compliance_server.py --http --port 8100

Requires: pip install "mcp[cli]"
"""

import json
import sys
from datetime import datetime
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="compliance",
    instructions="Compliance data and regulatory tools for financial services.",
)


# ─── TOOLS ────────────────────────────────────────────────────────────

@mcp.tool()
def search_regulations(
    query: str,
    regulation_type: str = "all",
    effective_after: str = "",
) -> str:
    """
    Search federal regulations relevant to a topic.
    Returns matching regulation sections with citations.

    Args:
        query: Search terms (e.g., "Reg E unauthorized transaction liability")
        regulation_type: Filter by type — all, cfr, cfpb, occ, ncua, fdic
        effective_after: ISO date — only return regulations effective after this date
    """
    # Production: vector search against regulation corpus
    # Dev: return representative examples
    results = [
        {
            "citation": "12 CFR 1005.6(b)",
            "title": "Regulation E — Liability of Consumer for Unauthorized Transfers",
            "summary": (
                "Consumer liability for unauthorized EFTs: $0 if reported within "
                "2 business days; up to $50 if reported within 2-60 days; "
                "unlimited if reported after 60 days."
            ),
            "effective_date": "2024-01-01",
            "relevance_score": 0.95,
        },
        {
            "citation": "12 CFR 1005.11",
            "title": "Regulation E — Procedures for Resolving Errors",
            "summary": (
                "Financial institution must investigate within 10 business days "
                "(20 for new accounts) or provisionally credit account. "
                "Must resolve within 45 calendar days (90 for POS/foreign)."
            ),
            "effective_date": "2024-01-01",
            "relevance_score": 0.92,
        },
    ]
    return json.dumps({"query": query, "results": results, "total": len(results)})


@mcp.tool()
def check_examination_findings(
    topic: str,
    severity: str = "all",
    years_back: int = 3,
) -> str:
    """
    Search recent examination findings and MRAs (Matters Requiring Attention).
    Helps understand what examiners are focused on.

    Args:
        topic: Area of focus (e.g., "dispute resolution", "BSA/AML", "UDAAP")
        severity: Filter — all, mra, mrba (matters requiring board attention)
        years_back: How many years to look back
    """
    findings = [
        {
            "finding_id": "EX-2025-0142",
            "date": "2025-06-15",
            "topic": "Dispute Resolution",
            "severity": "MRA",
            "summary": (
                "Institution failed to provide provisional credit within 10 "
                "business days for 12% of Reg E disputes. Root cause: manual "
                "processing delays in dispute classification step."
            ),
            "examiner_guidance": (
                "Implement automated classification to reduce processing time. "
                "Monitor provisional credit timelines daily."
            ),
        },
    ]
    return json.dumps({"topic": topic, "findings": findings})


@mcp.tool()
def get_enforcement_actions(
    violation_type: str,
    institution_type: str = "credit_union",
    years_back: int = 5,
) -> str:
    """
    Search public enforcement actions for comparable violations.
    Useful for understanding regulatory risk of specific practices.

    Args:
        violation_type: Type of violation (e.g., "Reg E timing", "UDAAP")
        institution_type: Filter — bank, credit_union, all
        years_back: How many years to look back
    """
    actions = [
        {
            "action_id": "NCUA-2024-0089",
            "institution_type": "credit_union",
            "violation": "Reg E dispute resolution timing",
            "penalty": "$175,000 civil money penalty",
            "date": "2024-03-22",
            "summary": (
                "Credit union failed to resolve Reg E disputes within 45 "
                "calendar days for approximately 1,200 members over 18-month "
                "period. Contributing factor: inadequate dispute tracking system."
            ),
        },
    ]
    return json.dumps({"violation_type": violation_type, "actions": actions})


@mcp.tool()
def validate_compliance_checklist(
    process_name: str,
    regulation: str,
    checklist_items: str,
) -> str:
    """
    Validate a compliance checklist against current regulatory requirements.
    Returns items that are missing, outdated, or need modification.

    Args:
        process_name: Name of the business process being checked
        regulation: Primary regulation (e.g., "Reg E", "BSA", "ECOA")
        checklist_items: JSON array of checklist items to validate
    """
    try:
        items = json.loads(checklist_items)
    except json.JSONDecodeError:
        items = [checklist_items]

    validation = {
        "process": process_name,
        "regulation": regulation,
        "items_checked": len(items),
        "issues": [
            {
                "item": "Provisional credit timeline",
                "status": "needs_update",
                "current": "10 business days",
                "required": "10 business days (20 for new accounts, 5 for Visa/MC)",
                "note": "Missing fast-track provisions for card network disputes",
            },
        ],
        "missing_items": [
            "Written explanation when claim denied",
            "Right to request documents relied upon",
        ],
        "validated_at": datetime.now().isoformat(),
    }
    return json.dumps(validation)


# ─── RESOURCES ────────────────────────────────────────────────────────

@mcp.resource("compliance://policies/dispute-resolution")
def dispute_resolution_policy() -> str:
    """Current dispute resolution policy document."""
    return json.dumps({
        "policy_id": "POL-DR-2025-001",
        "title": "Dispute Resolution Standard Operating Procedure",
        "version": "3.2",
        "effective_date": "2025-01-15",
        "sections": [
            "1. Intake and Classification",
            "2. Provisional Credit Determination",
            "3. Investigation Procedures",
            "4. Resolution and Notification",
            "5. Regulatory Compliance Checklist",
        ],
        "last_reviewed": "2025-01-10",
        "next_review": "2025-07-10",
    })


@mcp.resource("compliance://policies/bsa-aml")
def bsa_aml_policy() -> str:
    """Current BSA/AML policy document."""
    return json.dumps({
        "policy_id": "POL-BSA-2025-001",
        "title": "Bank Secrecy Act / Anti-Money Laundering Program",
        "version": "5.1",
        "effective_date": "2025-02-01",
        "sar_filing_threshold": "Suspicious activity involving $5,000+",
        "ctr_threshold": "$10,000 in cash",
    })


# ─── ENTRYPOINT ───────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = 8100
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
