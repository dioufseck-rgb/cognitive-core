"""
Cognitive Core — Settlement Calculator

Pure deterministic arithmetic for insurance claim settlement.
Zero external dependencies. Imported by:
  - mcp_servers/claims_services.py (as MCP tool)
  - engine/tools.py (as direct fallback when MCP unavailable)

ALL settlement arithmetic MUST go through this module.
LLMs should never compute these numbers themselves.
"""

import json
from typing import Any


def calculate_settlement(
    line_items: str | list,
    deductible: float = 0,
    sublimits: str | list = "[]",
    coinsurance: str | dict = "{}",
) -> str:
    """
    Deterministic settlement calculator. Computes verified totals,
    applies deductibles, sublimits, and coinsurance penalties.

    Args:
        line_items: JSON array (or list) of line items. Each item:
            {"category": "property", "claimed": 27800, "verified": 27800}
        deductible: Policy deductible amount
        sublimits: JSON array (or list) of sublimits:
            [{"category": "cnc_machine", "limit": 25000}]
        coinsurance: JSON object (or dict) for BI coinsurance:
            {"required_pct": 0.50, "carried_limit": 200000, "annual_revenue": 4307000}

    Returns:
        JSON string with settlement breakdown.
    """
    # Parse JSON string inputs
    try:
        items = json.loads(line_items) if isinstance(line_items, str) else line_items
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "Invalid line_items JSON", "status": 400})

    try:
        sub = json.loads(sublimits) if isinstance(sublimits, str) else sublimits
    except (json.JSONDecodeError, TypeError):
        sub = []

    try:
        coins = json.loads(coinsurance) if isinstance(coinsurance, str) else coinsurance
    except (json.JSONDecodeError, TypeError):
        coins = {}

    sublimit_map = {s["category"]: s["limit"] for s in sub if isinstance(s, dict)}

    # Compute verified totals by category
    category_totals: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cat = item.get("category", "unknown")
        verified = item.get("verified", item.get("claimed", 0))
        try:
            verified = float(verified)
        except (TypeError, ValueError):
            verified = 0

        # Apply sublimit if applicable
        if cat in sublimit_map:
            verified = min(verified, sublimit_map[cat])

        category_totals[cat] = category_totals.get(cat, 0) + verified

    # Apply coinsurance penalty to BI if applicable
    bi_total = category_totals.get("business_interruption", 0)
    coinsurance_applied = False
    coinsurance_ratio = 1.0

    if coins and bi_total > 0:
        required_pct = coins.get("required_pct", 0)
        carried_limit = coins.get("carried_limit", 0)
        annual_revenue = coins.get("annual_revenue", 0)

        if required_pct > 0 and annual_revenue > 0:
            required_limit = required_pct * annual_revenue
            if carried_limit < required_limit:
                coinsurance_ratio = carried_limit / required_limit
                category_totals["business_interruption"] = round(
                    bi_total * coinsurance_ratio
                )
                coinsurance_applied = True

    total_verified = sum(category_totals.values())
    net_payable = max(0, total_verified - deductible)

    result = {
        "category_totals": category_totals,
        "total_verified": total_verified,
        "deductible_applied": deductible,
        "net_payable": net_payable,
        "coinsurance_applied": coinsurance_applied,
        "coinsurance_ratio": round(coinsurance_ratio, 6) if coinsurance_applied else None,
        "sublimits_applied": list(sublimit_map.keys()),
        "line_item_count": len(items),
    }

    return json.dumps(result)


def calculate_settlement_from_context(context: dict) -> dict:
    """
    Context-aware wrapper for calculate_settlement.

    Extracts settlement inputs from the full workflow context
    (case_input + prior step outputs) and calls the deterministic
    calculator. This is the version registered as a tool — the LLM
    doesn't need to construct the arguments manually.

    Extraction logic:
      1. Line items: from get_claim.claimed_amounts or prior step outputs
      2. Deductible: from get_policy.property_coverage.deductible
      3. Sublimits: from get_policy endorsements/scheduled items
      4. Coinsurance: from get_policy.business_income.coinsurance +
         daily revenue from claim data
    """
    import json as _json

    # ── Extract claim data ──
    claim = context.get("get_claim", context.get("_step_intake", {}).get("data", {}).get("get_claim", {}))
    if isinstance(claim, dict) and "data" in claim:
        claim = claim["data"]
    policy = context.get("get_policy", context.get("_step_intake", {}).get("data", {}).get("get_policy", {}))
    if isinstance(policy, dict) and "data" in policy:
        policy = policy["data"]

    # ── Build line items ──
    line_items = []
    claimed_amounts = claim.get("claimed_amounts", {})

    # Property damage
    prop = claimed_amounts.get("property_damage", {})
    if isinstance(prop, dict):
        total_prop = prop.get("total_property", sum(
            v for v in prop.values() if isinstance(v, (int, float))
        ))
        if total_prop > 0:
            line_items.append({
                "category": "property_damage",
                "claimed": total_prop,
                "verified": total_prop,  # use claimed as verified unless adjusted
            })
    elif isinstance(prop, (int, float)) and prop > 0:
        line_items.append({"category": "property_damage", "claimed": prop, "verified": prop})

    # Business interruption
    bi = claimed_amounts.get("business_interruption", {})
    if isinstance(bi, dict):
        total_bi = bi.get("total_bi", sum(
            v for v in bi.values() if isinstance(v, (int, float))
        ))
        if total_bi > 0:
            # Only include lost revenue for coinsurance, not expediting/temp costs
            lost_revenue = bi.get("lost_revenue_10_days", bi.get("lost_revenue", total_bi))
            line_items.append({
                "category": "business_interruption",
                "claimed": total_bi,
                "verified": lost_revenue,  # coinsurance applies to the revenue loss
            })
    elif isinstance(bi, (int, float)) and bi > 0:
        line_items.append({"category": "business_interruption", "claimed": bi, "verified": bi})

    # If no structured data found, try flat amounts
    if not line_items:
        for key in ("property_damage", "bodily_injury", "liability"):
            val = claimed_amounts.get(key, 0)
            if isinstance(val, (int, float)) and val > 0:
                line_items.append({"category": key, "claimed": val, "verified": val})

    # ── Extract deductible ──
    prop_coverage = policy.get("property_coverage", {})
    deductible = prop_coverage.get("deductible", 0)

    # ── Extract sublimits ──
    sublimits = []
    # Check endorsements for sublimits
    for endorsement in policy.get("endorsements", []):
        if isinstance(endorsement, dict) and "sublimit" in endorsement:
            sublimits.append({
                "category": endorsement.get("category", endorsement.get("name", "")),
                "limit": endorsement["sublimit"],
            })

    # ── Extract coinsurance ──
    coinsurance = {}
    bi_coverage = policy.get("business_income", {})
    coins_pct = bi_coverage.get("coinsurance", 0)

    if coins_pct and coins_pct > 0:
        # Convert percentage to decimal if needed
        required_pct = coins_pct / 100.0 if coins_pct > 1 else coins_pct

        # Derive annual revenue from BI claim data
        annual_revenue = 0
        if isinstance(bi, dict):
            lost_revenue = bi.get("lost_revenue_10_days", 0)
            if lost_revenue > 0:
                daily = lost_revenue / 10.0
                annual_revenue = daily * 365
            else:
                # Try to find daily revenue info in adjuster notes
                total_bi = bi.get("total_bi", 0)
                if total_bi > 0:
                    # Rough estimate from claim notes
                    annual_revenue = total_bi * 36.5  # assume ~10 day claim

        carried_limit = bi_coverage.get("limit", 0)

        if annual_revenue > 0 and carried_limit > 0:
            coinsurance = {
                "required_pct": required_pct,
                "carried_limit": carried_limit,
                "annual_revenue": annual_revenue,
            }

    # ── Call the deterministic calculator ──
    result_str = calculate_settlement(
        line_items=line_items,
        deductible=deductible,
        sublimits=sublimits,
        coinsurance=coinsurance,
    )

    try:
        result = _json.loads(result_str) if isinstance(result_str, str) else result_str
    except (_json.JSONDecodeError, TypeError):
        result = {"raw_result": result_str}

    # Add the inputs for transparency
    result["_inputs"] = {
        "line_items": line_items,
        "deductible": deductible,
        "sublimits": sublimits,
        "coinsurance": coinsurance,
    }

    return result
