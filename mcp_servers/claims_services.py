"""
Claims Services MCP Server

Exposes claims-specific tools as MCP endpoints:
  - calculate_settlement: Deterministic settlement arithmetic
  - get_claim, get_policy, etc.: Claims data retrieval (from fixtures DB)

In dev, backed by fixture JSON files. In prod, swap backing to
real claims management system APIs.

Transports:
    stdio:  python mcp_servers/claims_services.py
    http:   python mcp_servers/claims_services.py --http --port 8201

Requires: pip install "mcp[cli]"
"""

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

FIXTURES_DIR = Path(__file__).parent.parent / "cases" / "fixtures"

mcp = FastMCP(
    name="claims_services",
    instructions=(
        "Claims processing data services. Provides claim and policy "
        "retrieval, and deterministic settlement calculation. "
        "The calculate_settlement tool MUST be used for all settlement "
        "arithmetic — LLMs should never compute these numbers themselves."
    ),
)


# ─── Fixture Loader ──────────────────────────────────────────────────

_fixture_cache: dict[str, dict] = {}

def _load_fixture(claim_id: str) -> dict:
    """Load fixture data for a claim. Cached."""
    if claim_id in _fixture_cache:
        return _fixture_cache[claim_id]

    fixture_map = {
        "CLM-2025-90112": "simple_tools.json",
        "CLM-2025-91207": "medium_tools.json",
        "CLM-2025-88431": "hard_tools.json",
    }

    filename = fixture_map.get(claim_id)
    if filename:
        path = FIXTURES_DIR / filename
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            _fixture_cache[claim_id] = data
            return data

    # Try matching by _case_id in any fixture file
    for f in FIXTURES_DIR.glob("*_tools.json"):
        with open(f) as fp:
            data = json.load(fp)
        if data.get("_case_id") == claim_id:
            _fixture_cache[claim_id] = data
            return data

    return {}


# ─── Claims Data Tools ───────────────────────────────────────────────

@mcp.tool()
def get_claim(claim_id: str) -> str:
    """
    Retrieve claim filing details: ID, amounts, dates, claimant, adjuster notes.

    Args:
        claim_id: Claim identifier (e.g., CLM-2025-90112)
    """
    data = _load_fixture(claim_id)
    result = data.get("get_claim")
    if result is None:
        return json.dumps({"error": f"Claim {claim_id} not found", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_policy(policy_number: str = "", claim_id: str = "") -> str:
    """
    Retrieve policy details: type, limits, deductibles, endorsements, exclusions.

    Args:
        policy_number: Policy number (e.g., BOP-5518823)
        claim_id: Alternatively, look up by claim ID
    """
    # In fixture mode, we need claim_id to find the right fixture file
    # In production, policy_number would query the policy admin system
    if claim_id:
        data = _load_fixture(claim_id)
    else:
        # Search all fixtures for matching policy number
        for name in ["simple_tools.json", "medium_tools.json", "hard_tools.json"]:
            path = FIXTURES_DIR / name
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                policy = d.get("get_policy", {})
                if policy.get("policy_number") == policy_number:
                    return json.dumps(policy)
        return json.dumps({"error": f"Policy {policy_number} not found", "status": 404})

    result = data.get("get_policy")
    if result is None:
        return json.dumps({"error": "Policy not found", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_incident_report(claim_id: str) -> str:
    """
    Retrieve incident report and supporting documentation.

    Args:
        claim_id: Claim identifier
    """
    data = _load_fixture(claim_id)
    result = data.get("get_incident_report")
    if result is None:
        return json.dumps({"error": f"No incident report for {claim_id}", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_claimant_history(claim_id: str) -> str:
    """
    Retrieve prior claims history and fraud indicators for the policyholder.

    Args:
        claim_id: Claim identifier
    """
    data = _load_fixture(claim_id)
    result = data.get("get_claimant_history")
    if result is None:
        return json.dumps({"error": f"No history for {claim_id}", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_equipment_schedule(claim_id: str, equipment_query: str = "") -> str:
    """
    Retrieve the scheduled equipment list for a policy's equipment
    breakdown endorsement.

    Args:
        claim_id: Claim identifier
        equipment_query: Optional description of equipment to look up
    """
    data = _load_fixture(claim_id)
    result = data.get("equipment")
    if result is None:
        return json.dumps({"error": f"No equipment schedule for {claim_id}", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_contractor_info(claim_id: str) -> str:
    """
    Retrieve contractor/vendor profile and COI (Certificate of Insurance)
    information.

    Args:
        claim_id: Claim identifier
    """
    data = _load_fixture(claim_id)
    result = data.get("contractor")
    if result is None:
        return json.dumps({"error": f"No contractor info for {claim_id}", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_contractor_gl_limits(claim_id: str) -> str:
    """
    Retrieve contractor GL (General Liability) coverage limits and status.

    Args:
        claim_id: Claim identifier
    """
    data = _load_fixture(claim_id)
    result = data.get("contractor_gl_limits")
    if result is None:
        return json.dumps({"error": f"No GL limits for {claim_id}", "status": 404})
    return json.dumps(result)


@mcp.tool()
def get_legal_cost_estimates(claim_id: str) -> str:
    """
    Retrieve estimated legal costs for recovery actions.

    Args:
        claim_id: Claim identifier
    """
    data = _load_fixture(claim_id)
    result = data.get("estimated_legal_costs")
    if result is None:
        return json.dumps({"error": f"No legal estimates for {claim_id}", "status": 404})
    return json.dumps(result)


# ─── Settlement Calculator ───────────────────────────────────────────
# Pure logic lives in engine/settlement.py (zero MCP deps).
# This wrapper exposes it as an MCP tool.

import os, sys
# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine.settlement import calculate_settlement as _calculate_settlement_impl

@mcp.tool()
def calculate_settlement(
    line_items: str,
    deductible: float = 0,
    sublimits: str = "[]",
    coinsurance: str = "{}",
) -> str:
    """
    Deterministic settlement calculator. Computes verified totals,
    applies deductibles, sublimits, and coinsurance penalties.

    ALL settlement arithmetic MUST go through this tool. LLMs should
    never compute these numbers themselves.

    Args:
        line_items: JSON array of line items. Each item:
            {"category": "property", "claimed": 27800, "verified": 27800, "adjustment_reason": ""}
        deductible: Policy deductible amount
        sublimits: JSON array of sublimits: [{"category": "cnc_machine", "limit": 25000}]
        coinsurance: JSON object for BI coinsurance:
            {"required_pct": 0.50, "carried_limit": 200000, "annual_revenue": 4307000}
    """
    return _calculate_settlement_impl(line_items, deductible, sublimits, coinsurance)


# ─── Entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--http" in sys.argv:
        port = 8201
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run(transport="stdio")
