"""
Content Moderation — Domain Pack Runner

Classifies content, verifies policy, challenges borderline decisions,
and governs disposition across four governance tiers.

Usage (from repo root):
    python library/domain-packs/content-moderation/run.py
    python library/domain-packs/content-moderation/run.py --case-id POST-002
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import set_trace, NullTrace

PACK_DIR = Path(__file__).resolve().parent

TIER_LABELS = {
    "auto":       "✓  AUTO",
    "spot_check": "~  SPOT CHECK",
    "gate":       "⏸  HUMAN REVIEW",
    "hold":       "⛔ HOLD",
}


class _CaptureTrace(NullTrace):
    def __init__(self):
        self.steps: dict[str, dict] = {}

    def on_parse_result(self, step_name, primitive, output):
        self.steps[step_name] = {"primitive": primitive, "output": output}


def run_post(coord: Coordinator, post: dict) -> None:
    snippet = post["content"][:80].replace("\n", " ")
    print(f"\n{'─' * 65}")
    print(f"  {post.get('_id', '?')}  |  {post.get('content_type', '')}")
    print(f"  \"{snippet}{'...' if len(post['content']) > 80 else ''}\"")
    if post.get("_expected"):
        print(f"  Expected: {post['_expected']}")
    print(f"{'─' * 65}")

    case_input = {k: v for k, v in post.items() if not k.startswith("_")}

    trace = _CaptureTrace()
    set_trace(trace)

    instance_id = coord.start(
        workflow_type="content_moderation",
        domain="content_moderation",
        case_input=case_input,
    )

    set_trace(NullTrace())
    instance = coord.store.get_instance(instance_id)

    classify = trace.steps.get("classify_content", {}).get("output", {})
    if classify:
        cat = classify.get("category", "?")
        conf = classify.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else "?"
        print(f"  Classification:  {cat}  (confidence {conf_str})")

    verify = trace.steps.get("verify_policy", {}).get("output", {})
    if verify:
        conforms = verify.get("conforms")
        violations = verify.get("violations", [])
        print(f"  Policy:          {'✓ conforms' if conforms else f'✗ {len(violations)} violation(s)'}")

    challenge = trace.steps.get("challenge_decision", {}).get("output", {})
    if challenge:
        survives = challenge.get("survives")
        vulns = len(challenge.get("vulnerabilities") or [])
        print(f"  Challenge:       {'survives' if survives else 'fails'}  ({vulns} vulnerabilities)")
    else:
        print(f"  Challenge:       skipped (high-confidence conform)")

    gov = trace.steps.get("govern_disposition", {}).get("output", {})
    tier = gov.get("tier_applied") or getattr(instance, "governance_tier", "?")
    tier_str = str(tier).lower().replace("governancetier.", "")
    disposition = gov.get("disposition", "?")
    label = TIER_LABELS.get(tier_str, tier_str.upper())
    print(f"  Governance:      {label}  —  {disposition}")

    work_order = gov.get("work_order")
    if isinstance(work_order, dict):
        print(f"  Work order:      → {work_order.get('target', '?')} queue")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str, default=None)
    args = parser.parse_args()

    coord = Coordinator(
        str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "pack.db"),
    )

    with open(PACK_DIR / "cases" / "example_posts.json") as f:
        posts = json.load(f)

    if args.case_id:
        posts = [p for p in posts if p.get("_id") == args.case_id]
        if not posts:
            print(f"Case {args.case_id!r} not found")
            sys.exit(1)

    print("\n" + "═" * 65)
    print("  CONTENT MODERATION — Domain Pack")
    print("  classify → verify → challenge → govern")
    print(f"  {len(posts)} post(s)")
    print("═" * 65)

    for post in posts:
        run_post(coord, post)

    print(f"\n{'═' * 65}\n")


if __name__ == "__main__":
    main()
