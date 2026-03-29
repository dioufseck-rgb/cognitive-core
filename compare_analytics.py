#!/usr/bin/env python3
"""
Analytics Layer Comparison
===========================
Runs the same fraud case with and without the analytics layer (causal DAG
for investigate, SDA policy for think) and prints a formatted comparison
so you can see the difference in output quality and reasoning depth.

Usage:
    python compare_analytics.py [--case card_fraud_cnp]
    python compare_analytics.py --case check_fraud_duplicate
    python compare_analytics.py --case app_scam_romance

Primitives compared:
    investigate  — v1 (free reasoning) vs v2 (causal DAG paths + gap analysis)
    think        — v1 (free reasoning) vs v2 (SDA policy + expected value + tension detection)
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

# ── Env flags: disable overhead for direct comparison ─────────────────────────
os.environ.setdefault("CC_CACHE_ENABLED", "false")
os.environ.setdefault("CC_PII_ENABLED", "false")
os.environ.setdefault("CC_EVAL_GATE_ENFORCED", "false")
os.environ.setdefault("CC_ANALYTICS_FALLBACK", "skip")

import yaml  # noqa: E402 — after env setup

ROOT = Path(__file__).parent
CASES_DIR = ROOT / "demos" / "fraud-operations" / "cases"
DOMAINS_DIR = ROOT / "demos" / "fraud-operations" / "domains"

CASE_FILES = {
    "card_fraud_cnp":          CASES_DIR / "card_fraud_cnp.json",
    "check_fraud_duplicate":   CASES_DIR / "check_fraud_duplicate.json",
    "app_scam_romance":        CASES_DIR / "app_scam_romance.json",
    "card_fraud_atm_skimmer":  CASES_DIR / "card_fraud_atm_skimmer.json",
}

DOMAIN_MAP = {
    "card_fraud_cnp":          ("card_fraud",     "card_fraud.yaml"),
    "check_fraud_duplicate":   ("check_fraud",    "check_fraud.yaml"),
    "app_scam_romance":        ("app_scam_fraud", "app_scam_fraud.yaml"),
    "card_fraud_atm_skimmer":  ("card_fraud",     "card_fraud.yaml"),
}


# ── ANSI colours ─────────────────────────────────────────────────────────────

BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
RED     = "\033[31m"
RESET   = "\033[0m"
BLUE    = "\033[34m"

def h1(s: str) -> str:
    bar = "═" * 70
    return f"\n{BOLD}{CYAN}{bar}\n  {s}\n{bar}{RESET}\n"

def h2(s: str) -> str:
    return f"\n{BOLD}{YELLOW}{'─' * 60}\n  {s}\n{'─' * 60}{RESET}\n"

def label(s: str) -> str:
    return f"{BOLD}{BLUE}{s}{RESET}"

def added(s: str) -> str:
    return f"{GREEN}+  {s}{RESET}"

def dim(s: str) -> str:
    return f"{DIM}{s}{RESET}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_case(case_name: str) -> dict:
    path = CASE_FILES[case_name]
    with open(path) as f:
        return json.load(f)


def load_domain(yaml_name: str) -> dict:
    path = DOMAINS_DIR / yaml_name
    with open(path) as f:
        return yaml.safe_load(f)


def make_state(case: dict, domain: str, prior_steps: list | None = None) -> dict:
    _INPUT_EXCLUDE = {"_meta", "description", "fraud_type", "alert_type", "risk_score"}
    return {
        "input":        {k: v for k, v in case.items() if k not in _INPUT_EXCLUDE},
        "metadata":     {"domain": domain},
        "steps":        prior_steps or [],
        "current_step": "",
        "loop_counts":  {},
    }


def run_node(step_name: str, primitive: str, params: dict,
             state: dict, eligible_artifacts: list) -> dict:
    """Run a single primitive node with controlled artifact eligibility."""
    from cognitive_core.engine.nodes import create_node

    node = create_node(step_name=step_name, primitive_name=primitive, params=params)

    def fake_list_eligible(context, artifact_type=None):
        if not eligible_artifacts:
            return []
        if artifact_type:
            return [a for a in eligible_artifacts if a.get("artifact_type") == artifact_type]
        return eligible_artifacts

    with patch("analytics.registry.AnalyticsRegistry") as MockReg:
        mock_inst = MockReg.return_value
        mock_inst.list_eligible.side_effect = fake_list_eligible
        result_state = node(state)

    return result_state["steps"][-1]["output"]


def load_real_artifacts() -> list[dict]:
    """Load analytics artifacts from the real registry config."""
    registry_path = ROOT / "config" / "analytics" / "registry.yaml"
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    return data.get("artifacts", [])


def get_eligible_for_domain(artifacts: list[dict], domain: str) -> list[dict]:
    eligible = []
    for art in artifacts:
        for pred in art.get("eligibility_predicates", []):
            field = pred.get("field")
            op    = pred.get("operator")
            val   = pred.get("value")
            if field == "domain":
                if op == "eq"  and domain == val:
                    eligible.append(art); break
                if op == "in"  and domain in val:
                    eligible.append(art); break
    return eligible


def trunc(s: str, width: int = 80) -> str:
    s = str(s).strip()
    lines = s.splitlines()
    if len(lines) > 4:
        lines = lines[:4] + ["..."]
    s = " ".join(l.strip() for l in lines if l.strip())
    return textwrap.fill(s, width=width, subsequent_indent="     ")


def fmt_list(lst: list, indent: int = 5) -> str:
    if not lst:
        return f"{'':>{indent}}(none)"
    prefix = " " * indent
    return "\n".join(f"{prefix}• {item}" for item in lst)


def fmt_ev(ev: dict | None) -> str:
    if not ev:
        return "(not computed)"
    return "  ".join(f"H{k}: {v:+.2f}" for k, v in sorted(ev.items(), key=lambda x: int(x[0])))


# ── Main comparison ───────────────────────────────────────────────────────────

def compare(case_name: str):
    print(h1(f"Analytics Layer Comparison — {case_name}"))

    case   = load_case(case_name)
    domain, domain_yaml = DOMAIN_MAP[case_name]
    dom    = load_domain(domain_yaml)

    meta = case.get("_meta", {})
    gt   = meta.get("ground_truth", {})
    print(f"  {label('Case ID:')}      {case.get('case_id', '?')}")
    print(f"  {label('Domain:')}       {domain}")
    print(f"  {label('Difficulty:')}   {meta.get('difficulty', '?')}")
    print(f"  {label('Ground truth:')} {gt.get('determination', '?')}  ({gt.get('fraud_type', '?')})")
    if case.get("description"):
        print(f"\n  {dim(textwrap.fill(case['description'], 66, subsequent_indent='  '))}")

    all_artifacts = load_real_artifacts()
    eligible      = get_eligible_for_domain(all_artifacts, domain)

    investigate_scope     = dom.get("investigate_activity", {}).get("scope", "Investigate the case.")
    think_framework       = dom.get("think_determination",  {}).get("framework", "Make a determination.")

    inv_params = {
        "question": (
            "Investigate this fraud alert to determine if fraud occurred. "
            "Assess severity, member impact, and required response.\n\n"
            + investigate_scope
        ),
        "scope":            investigate_scope,
        "available_evidence": "All case evidence is available in context above.",
        "effort_level":     "deep",
    }

    state_base = make_state(case, domain)

    # ── INVESTIGATE ───────────────────────────────────────────────────────────
    print(h2("INVESTIGATE primitive"))

    investigate_artifacts = [a for a in eligible if a["artifact_type"] == "causal_dag"]

    print(f"  Running {BOLD}v1{RESET} (no analytics)…", end="", flush=True)
    inv_v1 = run_node(
        "investigate_activity", "investigate",
        inv_params,
        state_base, eligible_artifacts=[],
    )
    print(f" done  (confidence: {inv_v1.get('confidence', '?')})")

    print(f"  Running {BOLD}v2{RESET} (causal DAG: {', '.join(a['artifact_name'] for a in investigate_artifacts) or 'none found'})…", end="", flush=True)
    inv_v2 = run_node(
        "investigate_activity", "investigate",
        inv_params,
        state_base, eligible_artifacts=investigate_artifacts,
    )
    print(f" done  (confidence: {inv_v2.get('confidence', '?')})")

    print()
    print(f"  {'V1 (baseline)':^34}  {'V2 (with causal DAG)':^34}")
    print(f"  {'─' * 34}  {'─' * 34}")

    def side(k, v1, v2, fmt=trunc):
        v1s = fmt(v1) if v1 else dim("(absent)")
        v2s = fmt(v2) if v2 else dim("(absent)")
        print(f"\n  {label(k)}")
        print(f"  V1: {v1s}")
        print(f"  V2: {v2s}")

    side("Finding", inv_v1.get("finding"), inv_v2.get("finding"))
    side("Confidence", inv_v1.get("confidence"), inv_v2.get("confidence"), fmt=str)

    # Reasoning excerpt
    print(f"\n  {label('Reasoning (first 200 chars)')}")
    r1 = str(inv_v1.get("reasoning", "")).strip()[:200]
    r2 = str(inv_v2.get("reasoning", "")).strip()[:200]
    for line in textwrap.wrap(r1, 66, subsequent_indent="     "):
        print(f"  V1: {dim(line)}")
    for line in textwrap.wrap(r2, 66, subsequent_indent="     "):
        print(f"  V2: {line}")

    # Analytics-only fields
    print(f"\n  {label('Fields added by analytics layer:')}")
    analytics_fields = [
        ("dag_version",               "DAG version used"),
        ("causal_templates_invoked",  "Templates matched"),
        ("activated_paths",           "Activated causal paths"),
        ("alternative_paths_considered", "Alternative paths considered"),
        ("unobserved_nodes",          "Unobserved nodes flagged"),
        ("evidential_gaps",           "Evidential gaps identified"),
        ("dag_divergence_flag",       "DAG divergence flag"),
        ("integration_reasoning",     "Integration reasoning"),
    ]
    any_added = False
    for key, desc in analytics_fields:
        val_v1 = inv_v1.get(key)
        val_v2 = inv_v2.get(key)
        if val_v2 is not None and val_v2 != [] and val_v2 != "":
            any_added = True
            if isinstance(val_v2, list):
                print(added(f"{desc}:"))
                print(fmt_list(val_v2, 8))
            elif isinstance(val_v2, bool):
                flag_color = RED if val_v2 else GREEN
                print(added(f"{desc}: {flag_color}{val_v2}{RESET}"))
            else:
                print(added(f"{desc}: {str(val_v2)[:120]}"))
    if not any_added:
        print(f"     {dim('(no new fields in this run — check artifact eligibility)')}")

    # ── THINK ─────────────────────────────────────────────────────────────────
    print(h2("THINK primitive"))

    sda_artifacts = [a for a in eligible if a["artifact_type"] == "sequential_decision"]

    # Build state with investigate step for think to reference
    def inv_step(output: dict, step_name: str = "investigate_activity") -> dict:
        return {
            "step_name": step_name,
            "primitive": "investigate",
            "output":    output,
        }

    state_think_v1 = make_state(case, domain, prior_steps=[inv_step(inv_v1)])
    state_think_v2 = make_state(case, domain, prior_steps=[inv_step(inv_v2)])

    def think_params(inv_out: dict) -> dict:
        return {
            "instruction": (
                "Make a fraud determination based on the investigation.\n\n"
                f"Investigation finding: {inv_out.get('finding', '')}\n"
                f"Investigation confidence: {inv_out.get('confidence', '')}\n\n"
                + think_framework
            ),
            "focus": think_framework,
            "additional_instructions": (
                "Do NOT emit resource_requests — set resource_requests to []. "
                "Use only the evidence already present in context."
            ),
        }

    print(f"  Running {BOLD}v1{RESET} (no analytics)…", end="", flush=True)
    think_v1 = run_node(
        "think_determination", "think",
        think_params(inv_v1),
        state_think_v1, eligible_artifacts=[],
    )
    print(f" done  (confidence: {think_v1.get('confidence', '?')})")

    print(f"  Running {BOLD}v2{RESET} (SDA policy: {', '.join(a['artifact_name'] for a in sda_artifacts) or 'none found'})…", end="", flush=True)
    think_v2 = run_node(
        "think_determination", "think",
        think_params(inv_v2),
        state_think_v2, eligible_artifacts=sda_artifacts,
    )
    print(f" done  (confidence: {think_v2.get('confidence', '?')})")

    print()
    side("Decision",   think_v1.get("decision"),   think_v2.get("decision"),   fmt=str)
    side("Confidence", think_v1.get("confidence"), think_v2.get("confidence"), fmt=str)

    print(f"\n  {label('Reasoning (first 200 chars)')}")
    r1 = str(think_v1.get("reasoning", "")).strip()[:200]
    r2 = str(think_v2.get("reasoning", "")).strip()[:200]
    for line in textwrap.wrap(r1, 66, subsequent_indent="     "):
        print(f"  V1: {dim(line)}")
    for line in textwrap.wrap(r2, 66, subsequent_indent="     "):
        print(f"  V2: {line}")

    print(f"\n  {label('Fields added by analytics layer:')}")
    sda_fields = [
        ("policy_class",               "Policy class"),
        ("policy_recommendation",      "Policy recommendation"),
        ("decision_horizon",           "Decision horizon"),
        ("expected_value_by_horizon",  "Expected value by horizon"),
        ("causal_consistency_check",   "Causal consistency check"),
        ("tension_flags",              "Tension flags"),
    ]
    any_sda = False
    for key, desc in sda_fields:
        val = think_v2.get(key)
        if val is not None and val != [] and val != "":
            any_sda = True
            if key == "expected_value_by_horizon":
                print(added(f"{desc}: {fmt_ev(val if isinstance(val, dict) else None)}"))
            elif isinstance(val, list):
                if val:
                    print(added(f"{desc}:"))
                    print(fmt_list(val, 8))
                else:
                    print(added(f"{desc}: {GREEN}(none — consistent){RESET}"))
            else:
                cc_color = GREEN if val == "consistent" else (RED if val == "inconsistent" else "")
                print(added(f"{desc}: {cc_color}{val}{RESET}"))
    if not any_sda:
        print(f"     {dim('(no new fields in this run — check artifact eligibility)')}")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(h2("SUMMARY"))
    print(f"  {label('Investigate')}")
    print(f"    V1 finding:    {dim(str(inv_v1.get('finding',''))[:100])}")
    print(f"    V2 finding:    {str(inv_v2.get('finding',''))[:100]}")
    activated = inv_v2.get("activated_paths", [])
    if activated:
        print(f"    Activated:     {GREEN}{', '.join(str(p) for p in activated)}{RESET}")
    gaps = inv_v2.get("evidential_gaps", [])
    if gaps:
        print(f"    Gaps:          {YELLOW}{', '.join(str(g) for g in gaps[:3])}{RESET}")

    print()
    print(f"  {label('Think')}")
    print(f"    V1 decision:   {dim(str(think_v1.get('decision',''))[:80])}")
    print(f"    V2 decision:   {str(think_v2.get('decision',''))[:80]}")
    rec = think_v2.get("policy_recommendation")
    if rec:
        print(f"    Policy rec:    {CYAN}{str(rec)[:80]}{RESET}")
    cc = think_v2.get("causal_consistency_check")
    if cc:
        cc_color = GREEN if cc == "consistent" else (RED if cc == "inconsistent" else YELLOW)
        print(f"    Consistency:   {cc_color}{cc}{RESET}")
    tf = think_v2.get("tension_flags", [])
    if tf:
        print(f"    {RED}Tension flags:{RESET}")
        print(fmt_list(tf, 8))

    print(f"\n  {label('Ground truth:'):} {BOLD}{gt.get('determination', '?')}{RESET}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare analytics layer impact on output quality")
    parser.add_argument(
        "--case",
        default="card_fraud_cnp",
        choices=list(CASE_FILES.keys()),
        help="Which case to run (default: card_fraud_cnp)",
    )
    args = parser.parse_args()

    if args.case not in CASE_FILES:
        print(f"Unknown case '{args.case}'. Available: {', '.join(CASE_FILES)}")
        sys.exit(1)

    try:
        compare(args.case)
    except Exception as e:
        import traceback
        print(f"\n{RED}Error: {e}{RESET}")
        print(dim(traceback.format_exc()))
        print(f"\n{YELLOW}Check that an LLM provider is configured:{RESET}")
        print("  export GOOGLE_API_KEY=...        (Google Gemini)")
        print("  export OPENAI_API_KEY=...         (OpenAI)")
        print("  export AZURE_OPENAI_API_KEY=...   (Azure OpenAI)")
        sys.exit(1)
