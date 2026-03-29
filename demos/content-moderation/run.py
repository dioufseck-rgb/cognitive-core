"""
Content Moderation — Governance Demo

Shows the govern primitive in action: auto-approve safe content,
challenge borderline decisions, gate ambiguous cases for human review.

Usage:
    python demos/content-moderation/run.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

DEMO_DIR = Path(__file__).resolve().parent

TIER_LABELS = {
    "auto":        "✓  AUTO-APPROVE",
    "spot_check":  "~  SPOT CHECK",
    "gate":        "⏸  HUMAN REVIEW",
    "hold":        "⛔ COMPLIANCE HOLD",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name: str, primitive: str, output: dict) -> None:
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_post(coord: Coordinator, post: dict) -> None:
    snippet = post["content"][:80].replace("\n", " ")
    print(f"\n{'─' * 65}")
    print(f"  {post['_id']}  |  {post['content_type']}")
    print(f"  \"{snippet}{'...' if len(post['content']) > 80 else ''}\"")
    print(f"{'─' * 65}")

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="content_moderation",
        domain="content_moderation",
        case_input=post,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    classify = trace.steps.get("classify_content", {}).get("output", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "?"
        print(f"  Classification:  {cat}  (confidence {conf_str})")

    challenge = trace.steps.get("challenge_decision", {}).get("output", {})
    if challenge:
        survives = challenge.get("survives")
        vulns = len(challenge.get("vulnerabilities") or [])
        print(f"  Challenge:       survives={survives}  vulnerabilities={vulns}")

    gov = trace.steps.get("govern_disposition", {}).get("output", {})
    tier = gov.get("tier_applied") or instance.governance_tier or "?"
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    label = TIER_LABELS.get(tier_str, tier_str.upper())
    print(f"  Governance:      {label}  —  {disposition}")

    rationale = gov.get("tier_rationale", "")
    if rationale:
        print(f"  Rationale:       {str(rationale)[:120]}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict):
        print(f"  Work order:      → {work_order.get('target', '?')} queue")

    print(f"  Expected:        {post.get('_expected', '')}")


def main():
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("No LLM key — running in simulated mode.\n")

    coord = Coordinator(
        str(DEMO_DIR / "coordinator_config.yaml"),
        db_path=str(DEMO_DIR / "demo.db"),
    )

    with open(DEMO_DIR / "cases" / "posts.json") as f:
        posts = json.load(f)

    print("\n" + "═" * 65)
    print("  CONTENT MODERATION — Governance Demo")
    print("  Four posts. Four different governance outcomes.")
    print("  The govern primitive decides: auto, spot-check, gate, or hold.")
    print("═" * 65)

    for post in posts:
        run_post(coord, post)

    print(f"\n{'═' * 65}")
    print("  The govern primitive is what makes this institutional AI,")
    print("  not just AI. Disposition is typed, auditable, and consistent.")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
