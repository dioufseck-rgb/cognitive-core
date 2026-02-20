"""
Cognitive Core — Azure AI Foundry Hosted Agent Adapter

Bridges the Foundry Responses API protocol to the Cognitive Core coordinator.
Foundry sends POST /responses with OpenAI Responses format; this adapter
translates to coordinator.start() and returns governed results.

Architecture:
    Foundry Agent Service → POST /responses (port 8088)
        → parse messages into case_input
        → resolve workflow/domain from env vars or message routing
        → coordinator.start() (full governance stack)
        → format result as Responses API output

The same Docker image serves any workflow. WORKFLOW and DOMAIN env vars
determine which workflow runs. Each Foundry agent registration points
to the same image with different env vars.

Runs alongside the existing API:
    Port 8000 — Cognitive Core API (direct access, CLI, internal tools)
    Port 8088 — Foundry Responses protocol (Foundry catalog, Teams, Copilot)

Usage:
    # Standalone
    CC_PROJECT_ROOT=. WORKFLOW=claim_intake DOMAIN=synthetic_claim \
        uvicorn api.foundry_adapter:app --host 0.0.0.0 --port 8088

    # In Docker (set via Foundry agent registration env vars)
    CMD ["uvicorn", "api.foundry_adapter:app", "--host", "0.0.0.0", "--port", "8088"]

Foundry protocol reference:
    https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/hosted-agents
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("cognitive_core.foundry_adapter")


def create_foundry_app(project_root: str = ".") -> Any:
    """
    Create the Foundry Responses API adapter.

    This is a separate FastAPI app from the main API server.
    It speaks the Foundry Responses protocol on port 8088.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="Cognitive Core — Foundry Agent",
        version="0.1.0",
        description="Foundry Hosted Agent backed by Cognitive Core",
    )

    # ── Configuration ────────────────────────────────────────────

    # Default workflow/domain from env vars (set per agent registration)
    DEFAULT_WORKFLOW = os.environ.get("WORKFLOW", "")
    DEFAULT_DOMAIN = os.environ.get("DOMAIN", "")

    # Workflow routing table: maps agent names to workflow/domain pairs
    # Loaded from env var or coordinator config
    ROUTING_TABLE = _load_routing_table(project_root)

    # ── Coordinator (lazy singleton) ─────────────────────────────

    _coordinator = None

    def get_coordinator():
        nonlocal _coordinator
        if _coordinator is None:
            from coordinator.runtime import Coordinator
            config_path = os.path.join(project_root, "coordinator", "config.yaml")
            if not os.path.exists(config_path):
                config_path = None
            _coordinator = Coordinator(config_path=config_path, verbose=True)
        return _coordinator

    # ── Foundry Responses API ────────────────────────────────────

    @app.post("/responses")
    async def handle_responses(request: Request):
        """
        Foundry Responses API endpoint.

        Accepts:
            {
                "input": "string" | {"messages": [...]},
                "stream": false,
                "metadata": {...}  // optional
            }

        Returns:
            {
                "id": "resp_...",
                "object": "response",
                "output": [{"type": "message", "role": "assistant", "content": "..."}],
                "status": "completed" | "failed",
                "metadata": {...}
            }
        """
        body = await request.json()
        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        start_time = time.time()

        try:
            # ── Parse input ──────────────────────────────────────
            case_input, workflow, domain = _parse_foundry_input(
                body, DEFAULT_WORKFLOW, DEFAULT_DOMAIN, ROUTING_TABLE
            )

            if not workflow or not domain:
                return _error_response(
                    response_id, 400,
                    "Cannot determine workflow/domain. Set WORKFLOW and DOMAIN "
                    "env vars or include routing metadata in the request."
                )

            # ── Execute through coordinator ──────────────────────
            coord = get_coordinator()
            instance_id = coord.start(
                workflow_type=workflow,
                domain=domain,
                case_input=case_input,
            )

            instance = coord.get_instance(instance_id)
            if instance is None:
                return _error_response(response_id, 500, "Instance not found after start")

            # ── Handle governance suspension ─────────────────────
            # If the workflow hits a gate tier, it suspends.
            # For Foundry, we return the suspension state so the
            # consuming application can handle approval flow.
            from coordinator.types import InstanceStatus

            if instance.status == InstanceStatus.SUSPENDED:
                return _suspended_response(response_id, instance, coord)

            # ── Format successful result ─────────────────────────
            elapsed = time.time() - start_time
            result = instance.result or {}
            ledger = coord.get_ledger(instance_id=instance_id)
            chain = coord.get_correlation_chain(instance.correlation_id)

            # Build output content
            content = _format_result_content(instance, result, chain, ledger)

            return JSONResponse(content={
                "id": response_id,
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                ],
                "status": "completed",
                "usage": {
                    "total_tokens": 0,  # Not tracked at this layer
                },
                "metadata": {
                    "cognitive_core": {
                        "instance_id": instance_id,
                        "correlation_id": instance.correlation_id,
                        "workflow": instance.workflow_type,
                        "domain": instance.domain,
                        "governance_tier": instance.governance_tier,
                        "step_count": instance.step_count,
                        "elapsed_seconds": round(elapsed, 3),
                        "delegations": len(chain) - 1,
                        "audit_entries": len(ledger),
                        "chain": [
                            {
                                "instance_id": c.instance_id,
                                "workflow": c.workflow_type,
                                "domain": c.domain,
                                "tier": c.governance_tier,
                                "status": c.status.value,
                            }
                            for c in chain
                        ],
                    }
                },
            })

        except Exception as e:
            logger.exception(f"Foundry adapter error: {e}")
            return _error_response(response_id, 500, str(e))

    # ── Health (required by Foundry) ─────────────────────────────

    @app.get("/health")
    async def health():
        return JSONResponse(content={
            "status": "ok",
            "agent": DEFAULT_WORKFLOW or "multi-workflow",
            "domain": DEFAULT_DOMAIN or "routing-table",
            "timestamp": time.time(),
        })

    # ── Agent info ───────────────────────────────────────────────

    @app.get("/")
    async def agent_info():
        return JSONResponse(content={
            "name": "Cognitive Core Agent",
            "version": "0.1.0",
            "workflow": DEFAULT_WORKFLOW or "dynamic",
            "domain": DEFAULT_DOMAIN or "dynamic",
            "protocol": "responses/v1",
            "governance": "enabled",
            "capabilities": [
                "multi-agent-delegation",
                "tiered-governance",
                "audit-trail",
                "pii-redaction",
                "shadow-mode",
            ],
        })

    return app


# ═══════════════════════════════════════════════════════════════════
# Input Parsing
# ═══════════════════════════════════════════════════════════════════

def _parse_foundry_input(
    body: dict,
    default_workflow: str,
    default_domain: str,
    routing_table: dict,
) -> tuple[dict, str, str]:
    """
    Parse Foundry Responses API input into case_input + workflow/domain.

    Foundry can send input as:
      - A string: "Process claim CLM-2026-00847"
      - An object with messages: {"messages": [{"role": "user", "content": "..."}]}

    Routing logic:
      1. If body has metadata.workflow/domain → use those
      2. If body has input as structured JSON → extract case_input directly
      3. Use env var defaults (WORKFLOW, DOMAIN)
    """
    raw_input = body.get("input", "")
    metadata = body.get("metadata", {})

    # Routing: metadata overrides > env var defaults
    workflow = metadata.get("workflow", default_workflow)
    domain = metadata.get("domain", default_domain)

    # Parse input into case_input
    case_input = {}

    if isinstance(raw_input, str):
        # Simple string input — wrap as description
        case_input = {"description": raw_input, "raw_input": raw_input}

        # Try to detect workflow from routing table
        if not workflow and routing_table:
            workflow, domain = _route_from_content(raw_input, routing_table)

    elif isinstance(raw_input, dict):
        messages = raw_input.get("messages", [])
        if messages:
            # Extract the last user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                last_msg = user_msgs[-1]
                content = last_msg.get("content", "")

                # If content is valid JSON, use it as case_input directly
                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            case_input = parsed
                        else:
                            case_input = {"description": content}
                    except json.JSONDecodeError:
                        case_input = {"description": content, "raw_input": content}
                elif isinstance(content, dict):
                    case_input = content
        else:
            # Direct object input (not wrapped in messages)
            case_input = raw_input

    # If input has explicit workflow/domain fields, use them
    if "workflow" in case_input and not metadata.get("workflow"):
        workflow = case_input.pop("workflow")
    if "domain" in case_input and not metadata.get("domain"):
        domain = case_input.pop("domain")

    return case_input, workflow, domain


def _route_from_content(content: str, routing_table: dict) -> tuple[str, str]:
    """
    Simple keyword-based routing from input content.
    Production would use a classifier; this handles basic cases.
    """
    content_lower = content.lower()
    for keyword, (wf, dom) in routing_table.items():
        if keyword in content_lower:
            return wf, dom
    return "", ""


def _load_routing_table(project_root: str) -> dict:
    """
    Load workflow routing table from env var or build from config.

    Format: keyword → (workflow, domain)
    """
    # From env var (JSON)
    env_table = os.environ.get("ROUTING_TABLE", "")
    if env_table:
        try:
            return json.loads(env_table)
        except json.JSONDecodeError:
            pass

    # Build basic table from known workflows
    return {
        "claim": ("claim_intake", "synthetic_claim"),
        "damage": ("damage_assessment", "synthetic_damage"),
        "fraud": ("fraud_screening", "synthetic_fraud"),
        "dispute": ("dispute_resolution", "card_dispute"),
    }


# ═══════════════════════════════════════════════════════════════════
# Response Formatting
# ═══════════════════════════════════════════════════════════════════

def _format_result_content(
    instance: Any,
    result: dict,
    chain: list,
    ledger: list,
) -> str:
    """Format coordinator result as human-readable content for Foundry."""
    parts = []

    # Summary
    parts.append(f"Workflow: {instance.workflow_type}/{instance.domain}")
    parts.append(f"Status: {instance.status.value}")
    parts.append(f"Governance tier: {instance.governance_tier}")
    parts.append(f"Steps executed: {instance.step_count}")

    # Step results
    steps = result.get("steps", [])
    for step in steps:
        prim = step.get("primitive", "")
        name = step.get("step_name", "")
        output = step.get("output", {})

        if prim == "classify":
            parts.append(f"\n{name}: {output.get('category', '?')} (confidence: {output.get('confidence', '?')})")
        elif prim == "verify":
            parts.append(f"\n{name}: {'PASS' if output.get('conforms') else 'FAIL'}")
            violations = output.get("violations", [])
            if violations:
                parts.append(f"  Violations: {', '.join(violations)}")
        elif prim == "investigate":
            parts.append(f"\n{name}: {output.get('finding', '?')}")
            flags = output.get("evidence_flags", [])
            if flags:
                parts.append(f"  Evidence: {', '.join(flags)}")
            parts.append(f"  Recommendation: {output.get('recommendation', '?')}")
        elif prim == "think":
            parts.append(f"\n{name}: score={output.get('risk_score', '?')}, {output.get('recommendation', '?')}")
        elif prim == "generate":
            # Don't dump full generated content — summarize
            parts.append(f"\n{name}: generated")

    # Delegations
    if len(chain) > 1:
        parts.append(f"\nDelegated workflows: {len(chain) - 1}")
        for c in chain:
            if c.instance_id != instance.instance_id:
                parts.append(f"  → {c.workflow_type}/{c.domain} [{c.status.value}]")

    # Audit
    parts.append(f"\nAudit trail: {len(ledger)} entries")

    return "\n".join(parts)


def _suspended_response(response_id: str, instance: Any, coord: Any) -> Any:
    """Return a response for governance-suspended workflows."""
    from fastapi.responses import JSONResponse

    pending = coord.list_pending_approvals()
    approval_for = [p for p in pending if p.get("instance_id") == instance.instance_id]

    return JSONResponse(
        status_code=200,
        content={
            "id": response_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": (
                        f"Workflow suspended for governance review.\n"
                        f"Instance: {instance.instance_id}\n"
                        f"Tier: {instance.governance_tier}\n"
                        f"Approve via: POST /v1/approvals/{instance.instance_id}/approve"
                    ),
                }
            ],
            "status": "requires_action",
            "metadata": {
                "cognitive_core": {
                    "instance_id": instance.instance_id,
                    "correlation_id": instance.correlation_id,
                    "governance_tier": instance.governance_tier,
                    "status": "suspended",
                    "approval_required": True,
                    "approval_queue": approval_for[0].get("queue", "") if approval_for else "",
                }
            },
        },
    )


def _error_response(response_id: str, status_code: int, message: str) -> Any:
    """Return a Foundry-formatted error response."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "id": response_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": f"Error: {message}",
                }
            ],
            "status": "failed",
            "error": {"message": message},
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Module-level app for uvicorn
# ═══════════════════════════════════════════════════════════════════

try:
    app = create_foundry_app(
        project_root=os.environ.get("CC_PROJECT_ROOT", ".")
    )
except ImportError:
    app = None
