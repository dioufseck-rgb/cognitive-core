"""
Cognitive Core - CLI Runner

Three-layer mode (standard):
    python -m engine.runner -w workflows/X.yaml -d domains/Y.yaml -c cases/Z.json

Single-file mode (backward compat):
    python -m engine.runner use_cases/X.yaml

Flags:
    --model / -m       Gemini model (default: gemini-2.0-flash)
    --verbose / -v     Show detailed output after completion
    --output / -o      Save full state JSON
    --no-trace         Disable live tracing
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

from engine.composer import load_use_case, load_three_layer, validate_use_case, run_workflow
from engine.nodes import set_trace


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

    def on_parse_error(self, step_name: str, error: str):
        self._log(f"    ⚠ PARSE ERROR: {error[:100]}")

    def on_route_decision(self, from_step: str, to_step: str, decision_type: str, reason: str):
        icons = {"deterministic": "⚡", "agent": "🤖", "default": "→", "loop_limit": "🔄"}
        icon = icons.get(decision_type, "?")
        target = "END" if to_step == "__end__" else to_step
        self._log(f"    {icon} route → {target} ({decision_type}: {reason[:60]})")


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
    parser.add_argument("--model", "-m", default="gemini-2.0-flash")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", "-o", help="Save state to JSON")
    parser.add_argument("--no-trace", action="store_true", help="Disable live tracing")

    args = parser.parse_args()

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
    errors = validate_use_case(config)
    if errors:
        print("Config errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print(f"✓ Valid: {config['name']}")
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
    mode = "three-layer" if three_layer else "single-file"
    print(f"\n{'─'*70}", file=sys.stderr)
    print(f"  {config['name']}  ({mode})", file=sys.stderr)
    print(f"  model: {args.model}", file=sys.stderr)
    steps_list = " → ".join(s["name"] for s in config["steps"])
    print(f"  steps: {steps_list}", file=sys.stderr)
    print(f"{'─'*70}", file=sys.stderr, flush=True)

    # Run
    start = time.time()
    try:
        final_state = run_workflow(config, workflow_input, args.model, args.temperature)
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
