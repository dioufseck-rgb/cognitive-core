"""
Support Ticket Triage — Quickstart Demo

The simplest complete Cognitive Core workflow.
Two YAML files + a JSON case = a working AI triage system.

Usage:
    python demos/support-ticket-triage/run.py

Requirements:
    pip install cognitive-core[runtime]
    export GOOGLE_API_KEY=...   (or OPENAI_API_KEY with LLM_PROVIDER=openai)
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

DEMO_DIR = Path(__file__).resolve().parent


class _CaptureTrace(NullTrace):
    """Capture step outputs during execution for display."""
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name: str, primitive: str, output: dict) -> None:
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_ticket(coord: Coordinator, ticket: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Ticket: {ticket['_id']} — {ticket['subject'][:50]}")
    print(f"  Customer: {ticket['customer_name']} ({ticket['customer_tier']} tier)")
    print(f"{'─' * 60}")

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="support_ticket_triage",
        domain="customer_support",
        case_input=ticket,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    classify = trace.steps.get("classify_severity", {}).get("output", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "?"
        print(f"  Severity:  {cat}  (confidence {conf_str})")

    gen = trace.steps.get("generate_response", {}).get("output", {})
    if gen:
        artifact = gen.get("artifact", {})
        if isinstance(artifact, str):
            try:
                artifact = json.loads(artifact)
            except Exception:
                pass
        if isinstance(artifact, dict):
            print(f"  Subject:   {artifact.get('subject', '—')}")
            print(f"  Escalate:  {artifact.get('escalate_to', 'null')}")
            print(f"  SLA:       {artifact.get('sla_hours', '—')} hours")
            body = artifact.get("body", "")
            if body:
                sentences = body.replace("\n", " ").split(". ")
                preview = ". ".join(sentences[:2])
                if len(sentences) > 2:
                    preview += "."
                print(f"\n  Response preview:")
                print(f"  {preview[:220]}")

    print(f"\n  Status: {instance.status.value}  |  Governance: {instance.governance_tier}")


def main():
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("No LLM key — running in simulated mode.")
        print("Set GOOGLE_API_KEY or OPENAI_API_KEY for real results.\n")

    coord = Coordinator(
        str(DEMO_DIR / "coordinator_config.yaml"),
        db_path=str(DEMO_DIR / "demo.db"),
    )

    with open(DEMO_DIR / "cases" / "tickets.json") as f:
        tickets = json.load(f)

    print("\n" + "═" * 60)
    print("  SUPPORT TICKET TRIAGE — Cognitive Core Demo")
    print("  Three tickets. Two YAML files. Zero application code.")
    print("═" * 60)

    for ticket in tickets:
        run_ticket(coord, ticket)

    print(f"\n{'═' * 60}")
    print("  Done. Try editing domain.yaml to change triage behaviour.")
    print("  No code changes needed.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
