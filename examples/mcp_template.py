"""
Cognitive Core — MCP Server Template (Sprint 3)

Connect Cognitive Core to your data source via the Model Context Protocol.

The retrieve primitive calls tools on MCP servers to gather case data.
This template wraps a hypothetical database in 60 lines of MCP server code.

HOW IT WORKS:
  Domain scaffold                retrieve primitive
        │                               │
        ▼                               ▼
  specification:           calls tools on MCP servers
    sources:               listed in coordinator_config.yaml
      - get_customer    ─────────────────────────────────────────────►  MCP Server
      - get_account                                                       │
      - get_transactions                                                  ▼
                                                              list_tools() + call_tool()
                                                                          │
                                                                          ▼
                                                                  your_database.query()

SETUP:
  1. Copy this file and rename it to match your data source
  2. Replace the stub implementations with your actual data access
  3. Add your server to coordinator_config.yaml:

     mcp_servers:
       - name: my-data-source
         transport: stdio
         command: python
         args: [examples/mcp_template.py]

  4. Reference your tools in domain scaffold retrieve.specification:

     retrieve:
       specification:
         sources:
           - get_customer
           - get_account

See demos/fraud-operations/fraud_data_mcp.py for a working reference.

Run standalone (stdio transport):
    python examples/mcp_template.py

Run with inspector:
    npx @modelcontextprotocol/inspector python examples/mcp_template.py
"""

from __future__ import annotations

import json
import sys
from typing import Any

# ── MCP imports ───────────────────────────────────────────────────────
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "MCP not installed. Run: pip install -e '.[runtime,mcp]'\n"
        "or: pip install mcp",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Your data access layer ─────────────────────────────────────────────
# Replace these stub functions with your actual database / API calls.

def fetch_customer(case_id: str) -> dict[str, Any]:
    """
    Fetch customer record by case_id.
    Replace with: return your_db.customers.get(case_id)
    """
    return {
        "customer_id":   case_id,
        "full_name":     "Example Customer",
        "date_of_birth": "1985-03-15",
        "account_since": "2018-06-01",
        "status":        "active",
    }


def fetch_account(case_id: str) -> dict[str, Any]:
    """
    Fetch account details by case_id.
    Replace with: return your_db.accounts.get(case_id)
    """
    return {
        "account_id":     f"ACC-{case_id}",
        "product_type":   "checking",
        "balance":        4250.00,
        "opened_date":    "2018-06-01",
        "status":         "active",
        "overdraft_limit": 500.00,
    }


def fetch_transactions(case_id: str, days: int = 90) -> list[dict[str, Any]]:
    """
    Fetch recent transactions for a case.
    Replace with: return your_db.transactions.query(case_id, days=days)
    """
    return [
        {"date": "2026-03-28", "amount": -42.50,  "merchant": "Grocery Store",    "type": "debit"},
        {"date": "2026-03-25", "amount": 2800.00,  "merchant": "Payroll Deposit",  "type": "credit"},
        {"date": "2026-03-20", "amount": -1200.00, "merchant": "Rent Payment",     "type": "debit"},
    ]


def fetch_risk_profile(case_id: str) -> dict[str, Any]:
    """
    Fetch risk signals for a case.
    Replace with: return your_risk_engine.score(case_id)
    """
    return {
        "risk_score":       420,
        "risk_tier":        "medium",
        "fraud_flags":      [],
        "velocity_alerts":  0,
        "last_updated":     "2026-03-31T00:00:00Z",
    }


# ── MCP Server definition ──────────────────────────────────────────────

server = Server("my-data-source")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Declare the tools this server exposes.
    One tool per data source your workflows need.
    Tool names must match the source names in your domain scaffold.
    """
    return [
        Tool(
            name="get_customer",
            description="Fetch customer profile by case_id",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {
                        "type": "string",
                        "description": "The case or customer identifier",
                    }
                },
                "required": ["case_id"],
            },
        ),
        Tool(
            name="get_account",
            description="Fetch account details by case_id",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "The account identifier"},
                },
                "required": ["case_id"],
            },
        ),
        Tool(
            name="get_transactions",
            description="Fetch recent transaction history by case_id",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "days":    {"type": "integer", "default": 90, "description": "Lookback window"},
                },
                "required": ["case_id"],
            },
        ),
        Tool(
            name="get_risk_profile",
            description="Fetch risk score and fraud signals by case_id",
            inputSchema={
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                },
                "required": ["case_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Route tool calls to your data access functions.
    Return data as JSON text — the retrieve primitive will parse it.
    """
    case_id = arguments.get("case_id", "")

    if name == "get_customer":
        data = fetch_customer(case_id)

    elif name == "get_account":
        data = fetch_account(case_id)

    elif name == "get_transactions":
        days = arguments.get("days", 90)
        data = fetch_transactions(case_id, days=days)

    elif name == "get_risk_profile":
        data = fetch_risk_profile(case_id)

    else:
        data = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


# ── Run ────────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
