"""
Cognitive Core - CLI Runner

Three-layer mode (standard):
    python -m engine.runner -w workflows/X.yaml -d domains/Y.yaml -c cases/Z.json

Single-file mode (backward compat):
    python -m engine.runner use_cases/X.yaml

Flags:
    --model / -m       Model alias or provider-specific name (default: "default")
    --provider / -p    LLM provider: azure_foundry, azure, openai, google, bedrock (auto-detected)
    --verbose / -v     Show detailed output after completion
    --output / -o      Save full state JSON
    --no-trace         Disable live tracing
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from engine.composer import load_use_case, load_three_layer, validate_use_case, run_workflow
from engine.agentic import validate_agentic_config, run_agentic_workflow
from engine.nodes import set_trace
from engine.tools import create_case_registry
from engine.actions import ActionRegistry, create_simulation_registry


# ---------------------------------------------------------------------------
# Live trace implementation
# ---------------------------------------------------------------------------

class ConsoleTrace:
    """Prints step-by-step progress to stderr during execution."""

    PRIM_ICONS = {
        "classify": "🏷️ ",
        "investigate": "🔍",
        "verify": "✅",
        "generate": "📝",
        "challenge": "⚔️ ",
        "retrieve": "📡",
        "think": "💭",
        "act": "⚡",
    }

    def __init__(self):
        self.workflow_start = time.time()
        self.step_start = None
        self.step_count = 0

    def _elapsed(self) -> str:
        return f"{time.time() - self.workflow_start:.1f}s"

    def _log(self, msg: str):
        print(f"  [{self._elapsed():>6s}] {msg}", file=sys.stderr, flush=True)

    def on_step_start(self, step_name: str, primitive: str, loop_iteration: int):
        self.step_count += 1
        self.step_start = time.time()
        icon = self.PRIM_ICONS.get(primitive, "  ")
        iter_label = f" (iteration {loop_iteration})" if loop_iteration > 1 else ""
        self._log(f"{icon} {step_name}{iter_label}")

    def on_llm_start(self, step_name: str, prompt_chars: int):
        self._log(f"    ↳ calling LLM ({prompt_chars:,} chars)...")

    def on_llm_end(self, step_name: str, response_chars: int, elapsed: float):
        self._log(f"    ↳ response received ({response_chars:,} chars, {elapsed:.1f}s)")

    def on_parse_result(self, step_name: str, primitive: str, output: dict):
        # Show the one-line summary based on primitive type
        if primitive == "classify":
            cat = output.get("category", "?")
            conf = output.get("confidence", 0)
            self._log(f"    → {cat} (confidence: {conf:.2f})")
        elif primitive == "investigate":
            finding = output.get("finding", "?")[:80]
            conf = output.get("confidence", 0)
            self._log(f"    → {finding}... ({conf:.2f})")
        elif primitive == "verify":
            conforms = output.get("conforms", "?")
            n_violations = len(output.get("violations", []))
            icon = "✓" if conforms else f"✗ ({n_violations} violations)"
            self._log(f"    → conforms: {icon}")
        elif primitive == "generate":
            artifact = output.get("artifact", "")
            self._log(f"    → generated {len(artifact):,} chars")
        elif primitive == "challenge":
            survives = output.get("survives", "?")
            n_vulns = len(output.get("vulnerabilities", []))
            icon = "✓ passed" if survives else f"✗ failed ({n_vulns} vulnerabilities)"
            self._log(f"    → {icon}")
        elif primitive == "retrieve":
            data = output.get("data", {})
            n_sources = len(output.get("sources_queried", []))
            n_ok = sum(1 for s in output.get("sources_queried", []) if s.get("status") == "success")
            conf = output.get("confidence", 0)
            self._log(f"    → {n_ok}/{n_sources} sources retrieved, {len(data)} data keys ({conf:.2f})")
        elif primitive == "act":
            mode = output.get("mode", "?")
            actions = output.get("actions_taken", [])
            n_exec = sum(1 for a in actions if a.get("status") == "executed")
            n_sim = sum(1 for a in actions if a.get("status") == "simulated")
            n_blocked = sum(1 for a in actions if a.get("status") == "blocked")
            needs_approval = output.get("requires_human_approval", False)
            if needs_approval:
                self._log(f"    → ⏸ APPROVAL REQUIRED ({len(actions)} actions pending)")
            elif n_exec > 0:
                conf_ids = [a.get("confirmation_id", "?") for a in actions if a.get("status") == "executed"]
                self._log(f"    → ✓ {n_exec} action(s) EXECUTED: {', '.join(conf_ids)}")
            elif n_sim > 0:
                self._log(f"    → {n_sim} action(s) simulated (dry run)")
            if n_blocked > 0:
                self._log(f"    → ✗ {n_blocked} action(s) BLOCKED (authorization)")
        elif primitive == "think":
            decision = output.get("decision", "")
            n_conclusions = len(output.get("conclusions", []))
            self._log(f"    → {n_conclusions} conclusions{': ' + decision[:60] if decision else ''}")

    def on_parse_error(self, step_name: str, error: str):
        self._log(f"    ⚠ PARSE ERROR: {error[:100]}")

    def on_route_decision(self, from_step: str, to_step: str, decision_type: str, reason: str):
        icons = {"deterministic": "⚡", "agent": "🤖", "default": "→", "loop_limit": "🔄"}
        icon = icons.get(decision_type, "?")
        target = "END" if to_step == "__end__" else to_step
        self._log(f"    {icon} route → {target} ({decision_type}: {reason[:60]})")

    def on_retrieve_start(self, step_name: str, source_name: str):
        self._log(f"    📡 fetching {source_name}...")

    def on_retrieve_end(self, step_name: str, source_name: str, status: str, latency_ms: float):
        icon = "✓" if status == "success" else "✗"
        self._log(f"    📡 {source_name}: {icon} {status} ({latency_ms:.0f}ms)")


# ---------------------------------------------------------------------------
# Post-run display
# ---------------------------------------------------------------------------

def print_step_result(step: dict, verbose: bool = False):
    output = step["output"]
    primitive = step["primitive"]

    print(f"\n{'='*70}")
    print(f"  STEP: {step['step_name']}  ({primitive.upper()})")
    print(f"{'='*70}")

    if "error" in output:
        print(f"  ⚠ ERROR: {output['error']}")
        if verbose:
            print(f"  Raw: {output.get('raw_response', 'N/A')[:500]}")
        return

    print(f"  Confidence: {output.get('confidence', 'N/A')}")
    print(f"  Reasoning: {output.get('reasoning', 'N/A')}")

    if primitive == "classify":
        print(f"\n  → Category: {output.get('category', 'N/A')}")
        for alt in output.get("alternative_categories", []):
            print(f"    - {alt['category']} ({alt['confidence']:.2f}): {alt.get('reasoning', '')}")

    elif primitive == "investigate":
        print(f"\n  → Finding: {output.get('finding', 'N/A')}")
        for h in output.get("hypotheses_tested", []):
            icon = {"supported": "✓", "rejected": "✗", "inconclusive": "?"}.get(h.get("status"), "?")
            print(f"    {icon} {h['hypothesis']} → {h['status']}")
        for a in output.get("recommended_actions", []):
            print(f"    • {a}")

    elif primitive == "verify":
        print(f"\n  → Conforms: {'✓' if output.get('conforms') else '✗'} {output.get('conforms')}")
        for v in output.get("violations", []):
            print(f"    [{v['severity'].upper()}] {v['rule']}: {v['description']}")

    elif primitive == "generate":
        artifact = output.get("artifact", "")
        print(f"\n  → Artifact ({output.get('format', 'text')}):")
        for line in artifact.split("\n"):
            print(f"    {line}")
        for c in output.get("constraints_checked", []):
            print(f"    {'✓' if c['satisfied'] else '✗'} {c['constraint']}")

    elif primitive == "challenge":
        print(f"\n  → Survives: {'✓' if output.get('survives') else '✗'} {output.get('survives')}")
        print(f"  Assessment: {output.get('overall_assessment', 'N/A')}")
        for v in output.get("vulnerabilities", []):
            print(f"    [{v['severity'].upper()}] {v['description']}")
            if verbose:
                print(f"      Attack: {v['attack_vector']}")
                print(f"      Fix: {v['recommendation']}")

    elif primitive == "retrieve":
        data = output.get("data", {})
        print(f"\n  → Data sources assembled: {list(data.keys())}")
        for sq in output.get("sources_queried", []):
            icon = "✓" if sq.get("status") == "success" else "✗"
            latency = sq.get("latency_ms", 0)
            print(f"    {icon} {sq['source']}: {sq['status']} ({latency:.0f}ms)")
            if sq.get("error"):
                print(f"      Error: {sq['error']}")
        skipped = output.get("sources_skipped", [])
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}")
        plan = output.get("retrieval_plan", "")
        if plan:
            print(f"  Plan: {plan}")
        if verbose:
            for key, val in data.items():
                print(f"\n  [{key}]:")
                preview = json.dumps(val, indent=2)[:300]
                for line in preview.split("\n"):
                    print(f"    {line}")

    elif primitive == "act":
        mode = output.get("mode", "?")
        print(f"\n  → Mode: {mode}")
        for a in output.get("actions_taken", []):
            status = a.get("status", "?")
            icon = {"executed": "✓", "simulated": "~", "blocked": "✗", "failed": "⚠"}.get(status, "?")
            print(f"    {icon} {a.get('action', '?')} → {a.get('target_system', '?')}: {status}")
            if a.get("confirmation_id"):
                print(f"      Confirmation: {a['confirmation_id']}")
            if a.get("error"):
                print(f"      Error: {a['error']}")
            if a.get("reversible") is not None:
                print(f"      Reversible: {a['reversible']}")
        for ac in output.get("authorization_checks", []):
            icon = "✓" if ac.get("result") == "passed" else "✗"
            print(f"    auth {icon} {ac.get('check', '?')}: {ac.get('result', '?')} — {ac.get('reason', '')}")
        if output.get("side_effects"):
            print(f"  Side effects:")
            for se in output["side_effects"]:
                print(f"    • {se}")
        if output.get("requires_human_approval"):
            print(f"\n  ⏸ HUMAN APPROVAL REQUIRED")
            if output.get("approval_brief"):
                print(f"    {output['approval_brief']}")

    elif primitive == "think":
        print(f"\n  → Thought: {output.get('thought', 'N/A')[:200]}...")
        if output.get("decision"):
            print(f"  Decision: {output['decision']}")
        for c in output.get("conclusions", []):
            print(f"    • {c}")

    missing = output.get("evidence_missing", [])
    if missing:
        print(f"\n  Missing evidence:")
        for e in missing:
            print(f"    ? {e['source']}: {e['description']}")


def print_summary(state: dict, elapsed: float):
    steps = state["steps"]
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Use case: {state['metadata'].get('use_case', '?')}")
    print(f"  Steps: {len(steps)}  |  Elapsed: {elapsed:.1f}s")
    print(f"  Path: {' → '.join(s['step_name'] for s in steps)}")

    loops = {k: v for k, v in state.get("loop_counts", {}).items() if v > 1}
    if loops:
        print(f"  Loops: {', '.join(f'{k}×{v}' for k, v in loops.items())}")

    print(f"\n  Confidence:")
    for s in steps:
        conf = s["output"].get("confidence", 0)
        if isinstance(conf, (int, float)):
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"    {s['step_name']:30s} {bar} {conf:.2f}")


# ---------------------------------------------------------------------------
# Action registry builder — reads SMTP config from case data
# ---------------------------------------------------------------------------

def _build_action_registry(case_data: dict) -> ActionRegistry:
    """
    Build an ActionRegistry with real or simulated actions.

    SMTP credentials come from environment variables (same pattern as
    LLM provider API keys):
        SMTP_SENDER       — sender email address
        SMTP_APP_PASSWORD  — Gmail app password
        SMTP_HOST          — SMTP host (default: smtp.gmail.com)
        SMTP_PORT          — SMTP port (default: 587)

    If SMTP_SENDER and SMTP_APP_PASSWORD are set, send_email is REAL.
    Otherwise it falls back to simulation.
    """
    import os
    import smtplib
    import uuid
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    registry = create_simulation_registry()

    # Wire real SMTP if env vars are set
    smtp_sender = os.environ.get("SMTP_SENDER", "")
    smtp_password = os.environ.get("SMTP_APP_PASSWORD", "")
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))

    if smtp_sender and smtp_password:
        def real_send_email(params: dict) -> dict:
            recipient = params.get("recipient", "")
            subject = params.get("subject", "Cognitive Core Notification")
            body = params.get("body", "")
            body_format = params.get("body_format", "plain")

            if not recipient:
                raise ValueError("No recipient email provided")

            msg = MIMEMultipart("alternative")
            msg["From"] = f"Cognitive Core <{smtp_sender}>"
            msg["To"] = recipient
            msg["Subject"] = subject
            msg["X-Cognitive-Core"] = "act-primitive"

            execution_id = uuid.uuid4().hex[:12].upper()
            msg["X-Execution-ID"] = execution_id

            mime_type = "html" if body_format == "html" else "plain"
            msg.attach(MIMEText(body, mime_type))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_sender, smtp_password)
                server.sendmail(smtp_sender, [recipient], msg.as_string())

            return {
                "confirmation_id": f"EMAIL-{execution_id}",
                "recipient": recipient,
                "subject": subject,
                "body_length": len(body),
            }

        registry.register(
            name="send_email",
            fn=real_send_email,
            description=f"Send email via {smtp_host} (REAL — credentials from env)",
            target_system="smtp",
            authorization_level="system",
            reversible=False,
            side_effects=[
                "Email delivered to recipient inbox",
                "Copy stored in sender's Sent folder",
            ],
        )
        print(f"  smtp: {smtp_sender} via {smtp_host}:{smtp_port} (LIVE)", file=sys.stderr)
    else:
        print(f"  smtp: simulated (set SMTP_SENDER + SMTP_APP_PASSWORD for live)", file=sys.stderr)

    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Core Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Three-layer (standard):
  python -m engine.runner -w workflows/dispute_resolution.yaml \\
                          -d domains/card_dispute.yaml \\
                          -c cases/card_clear_fraud.json

Single-file (legacy):
  python -m engine.runner config.yaml
        """,
    )
    parser.add_argument("config", nargs="?", help="Single-file YAML (legacy mode)")
    parser.add_argument("--workflow", "-w", help="Workflow YAML")
    parser.add_argument("--domain", "-d", help="Domain YAML")
    parser.add_argument("--case", "-c", help="Case JSON/YAML")
    parser.add_argument("--input", "-i", help="JSON string input (overrides --case)")
    parser.add_argument("--model", "-m", default="default",
                        help="Model name: alias (default/fast/standard/strong) or provider-specific (gpt-4o, gemini-2.0-flash)")
    parser.add_argument("--provider", "-p", default=None,
                        help="LLM provider: azure_foundry, azure, openai, google, bedrock (auto-detected if not set)")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", "-o", help="Save state to JSON")
    parser.add_argument("--no-trace", action="store_true", help="Disable live tracing")

    args = parser.parse_args()

    # Wire provider to environment so all create_llm calls pick it up
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider

    three_layer = args.workflow and args.domain
    if not args.config and not three_layer:
        parser.error("Provide --workflow + --domain, or a single config file")

    # Load
    try:
        if three_layer:
            config, case_input = load_three_layer(args.workflow, args.domain, args.case)
        else:
            config = load_use_case(args.config)
            case_input = None
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    is_agentic = config.get("mode") == "agentic"

    if is_agentic:
        errors = validate_agentic_config(config)
    else:
        errors = validate_use_case(config)

    if errors:
        print("Config errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print(f"✓ Valid: {config['name']} (mode: {'agentic' if is_agentic else 'sequential'})")
        if is_agentic:
            print(f"  Goal: {config.get('goal', '?')[:80]}")
            print(f"  Primitives: {config.get('available_primitives', [])}")
            print(f"  Max steps: {config.get('constraints', {}).get('max_steps', '?')}")
            for k, v in config.get("primitive_configs", {}).items():
                print(f"  config: {k} ({v.get('primitive', '?')})")
        else:
            for s in config["steps"]:
                t = s.get("transitions", [])
                modes = []
                if any("when" in tr for tr in t): modes.append("det")
                if any("agent_decide" in tr for tr in t): modes.append("agent")
                suffix = f"  [{','.join(modes)}]" if modes else ""
                print(f"  {s['primitive']:12s} {s['name']}{suffix}")
        sys.exit(0)

    # Input
    if args.input:
        workflow_input = json.loads(args.input)
    elif case_input:
        workflow_input = case_input
    else:
        workflow_input = config.get("default_input", {})

    if not workflow_input:
        parser.error("No input. Use --case, --input, or default_input in config.")

    # Setup tracing
    if not args.no_trace:
        trace = ConsoleTrace()
        set_trace(trace)

    # Header
    from engine.llm import get_provider_info, resolve_model, validate_config
    config_issues = validate_config()
    if config_issues:
        print("⚠ LLM config warnings:", file=sys.stderr)
        for issue in config_issues:
            print(f"  - {issue}", file=sys.stderr)
    pinfo = get_provider_info()
    resolved_model = resolve_model(args.model, pinfo["provider"])
    mode = "agentic" if is_agentic else ("three-layer" if three_layer else "single-file")
    print(f"\n{'─'*70}", file=sys.stderr)
    print(f"  {config['name']}  ({mode})", file=sys.stderr)
    print(f"  provider: {pinfo['provider']}  model: {resolved_model}", file=sys.stderr)
    if is_agentic:
        prims = ", ".join(config.get("available_primitives", []))
        max_s = config.get("constraints", {}).get("max_steps", "?")
        print(f"  primitives: {prims}", file=sys.stderr)
        print(f"  max_steps: {max_s}", file=sys.stderr)
    else:
        steps_list = " → ".join(s["name"] for s in config["steps"])
        print(f"  steps: {steps_list}", file=sys.stderr)
    print(f"{'─'*70}", file=sys.stderr, flush=True)

    # Build tool registry from case data (dev/test mode)
    has_retrieve = False
    if is_agentic:
        has_retrieve = "retrieve" in config.get("available_primitives", [])
    else:
        has_retrieve = any(s["primitive"] == "retrieve" for s in config.get("steps", []))

    tool_registry = None
    if has_retrieve:
        # Data sourcing priority:
        #   1. MCP data_services server (if DATA_MCP_URL or DATA_MCP_CMD is set)
        #   2. Fixture database (if cognitive_core.db exists)
        #   3. Case passthrough (legacy fallback)
        data_mcp_url = os.environ.get("DATA_MCP_URL", "")
        data_mcp_cmd = os.environ.get("DATA_MCP_CMD", "")

        if data_mcp_url or data_mcp_cmd:
            # ── MCP-backed retrieval ──
            import asyncio
            from engine.providers import MCPProvider
            from engine.tools import ToolRegistry as _TR

            tool_registry = _TR()
            if data_mcp_url:
                provider = MCPProvider(transport="http", url=data_mcp_url)
            else:
                parts = data_mcp_cmd.split()
                provider = MCPProvider(
                    transport="stdio", command=parts[0], args=parts[1:],
                )

            async def _connect_mcp():
                await provider.connect()
                provider.register_all(tool_registry)

            asyncio.get_event_loop().run_until_complete(_connect_mcp())
            if not args.no_trace:
                src = data_mcp_url or data_mcp_cmd
                print(f"  data: MCP ({src})", file=sys.stderr)
        else:
            try:
                from fixtures.api import create_service_registry
                from fixtures.db import DB_PATH
                if DB_PATH.exists():
                    # ── Fixture DB (in-process, no MCP overhead) ──
                    tool_registry = create_service_registry()
                    if not args.no_trace:
                        print(f"  data: fixture DB ({DB_PATH.name})", file=sys.stderr)
                else:
                    raise FileNotFoundError
            except (ImportError, FileNotFoundError):
                # ── Legacy case passthrough ──
                fixtures_path = None
                if args.case:
                    from pathlib import Path
                    case_p = Path(args.case)
                    fixtures_candidate = case_p.parent / "fixtures" / case_p.name
                    if fixtures_candidate.exists():
                        fixtures_path = str(fixtures_candidate)
                tool_registry = create_case_registry(workflow_input, fixtures_path=fixtures_path)
                if not args.no_trace:
                    print(f"  data: case passthrough", file=sys.stderr)
        if not args.no_trace:
            print(f"  tools: {', '.join(tool_registry.list_tools())}", file=sys.stderr)

    # Build action registry (auto-detect act steps)
    has_act = False
    if is_agentic:
        has_act = "act" in config.get("available_primitives", [])
    else:
        has_act = any(s["primitive"] == "act" for s in config.get("steps", []))

    action_registry = None
    if has_act:
        action_registry = _build_action_registry(workflow_input)
        if not args.no_trace:
            print(f"  actions: {', '.join(action_registry.list_actions())}", file=sys.stderr)

    # Run
    start = time.time()
    try:
        if is_agentic:
            final_state = run_agentic_workflow(
                config, workflow_input, args.model, args.temperature,
                tool_registry=tool_registry,
            )
        else:
            final_state = run_workflow(
                config, workflow_input, args.model, args.temperature,
                tool_registry=tool_registry,
                action_registry=action_registry,
            )
    except Exception as e:
        print(f"\n  ⚠ FAILED: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start

    print(f"\n{'─'*70}", file=sys.stderr)
    print(f"  done in {elapsed:.1f}s", file=sys.stderr)
    print(f"{'─'*70}\n", file=sys.stderr, flush=True)

    # Display results
    for step in final_state["steps"]:
        print_step_result(step, verbose=args.verbose)

    print_summary(final_state, elapsed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_state, f, indent=2, default=str)
        print(f"\n  State saved: {args.output}")


if __name__ == "__main__":
    main()
