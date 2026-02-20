#!/usr/bin/env python3
"""
Cognitive Core — Foundry Agent Registration

Registers each workflow/domain pair as a Foundry hosted agent.
All agents use the same Docker image; env vars determine which
workflow runs.

Usage:
    # Register all agents
    python -m scripts.register_foundry_agents

    # Register a single agent
    python -m scripts.register_foundry_agents --workflow claim_intake --domain synthetic_claim

    # Dry run (print what would be registered)
    python -m scripts.register_foundry_agents --dry-run

Requires:
    pip install azure-ai-projects azure-identity
    AZURE_AI_PROJECT_ENDPOINT env var set
    Container image pushed to ACR
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path

# ─── Agent Definitions ───────────────────────────────────────────────
# Each entry becomes a Foundry hosted agent.
# All point to the same container image with different env vars.

AGENTS = [
    {
        "name": "claim-intake",
        "description": "Insurance claim intake orchestrator. Classifies, validates eligibility, assesses risk, and delegates to damage assessment and fraud screening.",
        "workflow": "claim_intake",
        "domain": "synthetic_claim",
        "tier": "gate",
        "cpu": "1",
        "memory": "2Gi",
        "exposure": "external",  # Foundry catalog — entry point for consumers
    },
    {
        "name": "damage-assessment",
        "description": "Physical damage assessment agent. Classifies severity, verifies documentation, generates assessment report.",
        "workflow": "damage_assessment",
        "domain": "synthetic_damage",
        "tier": "auto",
        "cpu": "0.5",
        "memory": "1Gi",
        "exposure": "internal",  # Spawned by coordinator — also callable directly
    },
    {
        "name": "fraud-screening",
        "description": "Fraud screening agent. Classifies risk level, investigates patterns, generates screening result.",
        "workflow": "fraud_screening",
        "domain": "synthetic_fraud",
        "tier": "spot_check",
        "cpu": "0.5",
        "memory": "1Gi",
        "exposure": "internal",  # Spawned by coordinator — also callable directly
    },
    {
        "name": "dispute-resolution",
        "description": "Card dispute resolution agent. Retrieves case data, classifies dispute, verifies against records, investigates, generates response with regulatory compliance.",
        "workflow": "dispute_resolution",
        "domain": "card_dispute",
        "tier": "gate",
        "cpu": "1",
        "memory": "2Gi",
        "exposure": "external",  # Foundry catalog — entry point for consumers
    },
]


def register_agents(
    agents: list[dict],
    image: str,
    endpoint: str,
    dry_run: bool = False,
):
    """Register agents with Foundry Agent Service."""

    if not dry_run:
        from azure.ai.projects import AIProjectClient
        from azure.ai.projects.models import (
            ImageBasedHostedAgentDefinition,
            ProtocolVersionRecord,
            AgentProtocol,
        )
        from azure.identity import DefaultAzureCredential

        client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential(),
        )

    for agent_def in agents:
        name = agent_def["name"]
        workflow = agent_def["workflow"]
        domain = agent_def["domain"]

        env_vars = {
            "CC_PROJECT_ROOT": "/app",
            "WORKFLOW": workflow,
            "DOMAIN": domain,
            "AZURE_AI_PROJECT_ENDPOINT": endpoint,
        }

        print(f"\n{'─'*50}")
        print(f"Agent: {name}")
        print(f"  Workflow: {workflow}")
        print(f"  Domain:   {domain}")
        print(f"  Tier:     {agent_def['tier']}")
        print(f"  Image:    {image}")
        print(f"  CPU:      {agent_def['cpu']}")
        print(f"  Memory:   {agent_def['memory']}")
        print(f"  Env:      {list(env_vars.keys())}")

        if dry_run:
            print(f"  [DRY RUN] Would register {name}")
            continue

        try:
            agent = client.agents.create_version(
                agent_name=name,
                definition=ImageBasedHostedAgentDefinition(
                    container_protocol_versions=[
                        ProtocolVersionRecord(
                            protocol=AgentProtocol.RESPONSES,
                            version="v1",
                        )
                    ],
                    cpu=agent_def["cpu"],
                    memory=agent_def["memory"],
                    image=image,
                    environment_variables=env_vars,
                ),
            )
            print(f"  ✓ Registered: {agent.name} v{agent.version}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def discover_agents(project_root: str) -> list[dict]:
    """
    Discover workflow/domain pairs from the project directory.
    Returns agent definitions for any pair found in workflows/ and domains/.
    """
    wf_dir = Path(project_root) / "workflows"
    dom_dir = Path(project_root) / "domains"

    if not wf_dir.exists() or not dom_dir.exists():
        return []

    agents = []
    for dom_file in sorted(dom_dir.glob("*.yaml")):
        try:
            with open(dom_file) as f:
                dom_config = yaml.safe_load(f)
            if not dom_config or not isinstance(dom_config, dict):
                continue

            workflow_name = dom_config.get("workflow", "")
            domain_name = dom_config.get("domain_name", dom_file.stem)
            tier = dom_config.get("governance", "gate")

            # Check workflow exists
            wf_file = wf_dir / f"{workflow_name}.yaml"
            if not wf_file.exists():
                continue

            agent_name = domain_name.replace("_", "-")
            agents.append({
                "name": agent_name,
                "description": dom_config.get("description", f"{workflow_name} / {domain_name}"),
                "workflow": workflow_name,
                "domain": domain_name,
                "tier": tier,
                "cpu": "1",
                "memory": "2Gi",
            })
        except Exception as e:
            print(f"  Warning: skipping {dom_file.name}: {e}")

    return agents


def main():
    parser = argparse.ArgumentParser(description="Register Cognitive Core workflows as Foundry agents")
    parser.add_argument("--image", default=os.environ.get("CC_CONTAINER_IMAGE", ""),
                       help="Container image URI (e.g., myacr.azurecr.io/cognitive-core:v1)")
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_AI_PROJECT_ENDPOINT", ""),
                       help="Foundry project endpoint")
    parser.add_argument("--workflow", help="Register a single workflow")
    parser.add_argument("--domain", help="Domain for single workflow registration")
    parser.add_argument("--discover", action="store_true",
                       help="Auto-discover agents from workflows/ and domains/")
    parser.add_argument("--external-only", action="store_true",
                       help="Only register external (orchestrator) agents, skip internal")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be registered without calling Azure")
    args = parser.parse_args()

    if not args.image and not args.dry_run:
        print("Error: --image or CC_CONTAINER_IMAGE required")
        sys.exit(1)

    if not args.endpoint and not args.dry_run:
        print("Error: --endpoint or AZURE_AI_PROJECT_ENDPOINT required")
        sys.exit(1)

    # Determine agent list
    if args.workflow:
        agents = [{
            "name": args.workflow.replace("_", "-"),
            "description": f"{args.workflow} / {args.domain or 'default'}",
            "workflow": args.workflow,
            "domain": args.domain or args.workflow,
            "tier": "gate",
            "cpu": "1",
            "memory": "2Gi",
        }]
    elif args.discover:
        agents = discover_agents(args.project_root)
        if not agents:
            print("No workflow/domain pairs found.")
            sys.exit(1)
        print(f"Discovered {len(agents)} agents from project directory")
    else:
        agents = AGENTS

    # Filter by exposure level
    if args.external_only:
        before = len(agents)
        agents = [a for a in agents if a.get("exposure", "external") == "external"]
        print(f"Filtered to {len(agents)} external agents (from {before} total)")

    print(f"\nRegistering {len(agents)} agents")
    print(f"Image: {args.image or '[dry-run]'}")
    print(f"Endpoint: {args.endpoint or '[dry-run]'}")

    register_agents(
        agents=agents,
        image=args.image or "dry-run-image",
        endpoint=args.endpoint or "dry-run-endpoint",
        dry_run=args.dry_run,
    )

    print(f"\n{'═'*50}")
    print(f"Done. {len(agents)} agents {'would be ' if args.dry_run else ''}registered.")


if __name__ == "__main__":
    main()
