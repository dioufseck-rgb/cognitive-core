"""
Prior Authorization Appeal Review — 20-Case Benchmark Runner

Runs all 20 adversarial cases through both Cognitive Core (agentic DEVS)
and the ReAct baseline. Results are written to:

  output/benchmark/
    results.json            — machine-readable, one record per case per system
    cc/<case_id>.txt        — CC determination text
    react/<case_id>.txt     — ReAct determination text
    run_log.txt             — timestamped console output

Usage:
    # Run all 20 cases, both systems
    python demos/prior-auth-appeal/run_benchmark.py

    # CC only (faster, no API key needed for ReAct portion)
    python demos/prior-auth-appeal/run_benchmark.py --cc-only

    # ReAct only (to backfill after CC run)
    python demos/prior-auth-appeal/run_benchmark.py --react-only

    # Single case for debugging
    python demos/prior-auth-appeal/run_benchmark.py --case pa_2024_a001.json

    # Resume interrupted run (skips cases already in results.json)
    python demos/prior-auth-appeal/run_benchmark.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

DEMO_DIR  = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEMO_DIR))

from run import (
    load_documents,
    build_case_input,
    run_cognitive_core,
    run_react_agent,
    CASES_DIR,
)

OUTPUT_DIR = DEMO_DIR / "output" / "benchmark"

# ── Case discovery ──────────────────────────────────────────────────────────

def discover_cases() -> list[str]:
    """Return sorted list of benchmark case filenames (pa_2024_*.json)."""
    files = sorted(CASES_DIR.glob("pa_2024_*.json"))
    return [f.name for f in files]


def load_case(filename: str) -> dict:
    with open(CASES_DIR / filename) as f:
        return json.load(f)


# ── Result record ───────────────────────────────────────────────────────────

def make_result(
    case_id: str,
    case_type: str,
    system: str,
    determination: str,
    elapsed: float,
    tier: str | None,
    trajectory: list[str] | None,
    steps: int | None,
    error: str | None,
    ground_truth: dict,
) -> dict:
    disposition = extract_disposition(determination)
    return {
        "case_id":       case_id,
        "case_type":     case_type,
        "system":        system,                   # "cc" | "react"
        "timestamp":     datetime.utcnow().isoformat(),
        "elapsed_s":     round(elapsed, 1),
        "disposition":   disposition,
        "tier":          tier,
        "steps":         steps,
        "trajectory":    trajectory,
        "char_count":    len(determination),
        "error":         error,
        "ground_truth_answer": ground_truth.get("right_answer", ""),
        "obvious_reading":     ground_truth.get("obvious_reading", ""),
        "genuine_tension":     ground_truth.get("the_genuine_tension", ""),
        "determination": determination,
    }


def extract_disposition(text: str) -> str:
    """
    Extract the disposition keyword from a determination string.
    Returns one of: OVERTURN, UPHOLD, PARTIAL, REMAND, UNKNOWN.
    Looks for the most authoritative signal — explicit DISPOSITION: field first,
    then the strongest keyword in the opening paragraph.
    """
    if not text or text.startswith("["):
        return "ERROR"

    upper = text.upper()

    # 1. Explicit DISPOSITION: field
    for line in upper.splitlines():
        line = line.strip()
        if line.startswith("DISPOSITION:") or line.startswith("RE: FINAL APPEAL DETERMINATION"):
            for kw in ("OVERTURN", "UPHOLD", "PARTIAL", "REMAND"):
                if kw in line:
                    return kw

    # 2. First 500 chars — opening statement
    opener = upper[:500]
    for kw in ("PARTIAL", "REMAND", "OVERTURN", "UPHOLD"):  # PARTIAL before OVERTURN
        if kw in opener:
            return kw
    # UPHELD is a valid form of UPHOLD
    if "UPHELD" in opener:
        return "UPHOLD"

    # 3. Full text majority vote — count occurrences
    scores = {kw: upper.count(kw) + upper.count(kw+"ED") for kw in ("OVERTURN", "UPHOLD", "PARTIAL", "REMAND")}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "UNKNOWN"


# ── Results persistence ─────────────────────────────────────────────────────

def load_existing_results(results_path: Path) -> list[dict]:
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


def save_results(results: list[dict], results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def already_run(results: list[dict], case_id: str, system: str) -> bool:
    return any(
        r["case_id"] == case_id and r["system"] == system and r.get("error") is None
        for r in results
    )


# ── Per-case runner ─────────────────────────────────────────────────────────

def run_case_cc(case_file: str, docs: dict, verbose: bool = False) -> dict:
    case = load_case(case_file)
    case_input = build_case_input(case, docs)
    ground_truth = case.get("ground_truth_complexity", {})
    case_type = case["case_id"].split("-")[2][0]  # A, B, C, D, E, F

    try:
        result = run_cognitive_core(case_input, verbose=verbose)
        determination = str(result.get("determination", ""))
        return make_result(
            case_id=case["case_id"],
            case_type=case_type,
            system="cc",
            determination=determination,
            elapsed=result.get("elapsed", 0),
            tier=result.get("tier", "unknown"),
            trajectory=result.get("trajectory", []),
            steps=len(result.get("trajectory", [])),
            error=None,
            ground_truth=ground_truth,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return make_result(
            case_id=case["case_id"],
            case_type=case_type,
            system="cc",
            determination=f"[CC failed: {e}]",
            elapsed=0,
            tier=None,
            trajectory=None,
            steps=None,
            error=str(e) + "\n" + tb[-500:],
            ground_truth=ground_truth,
        )


def run_case_react(case_file: str, docs: dict) -> dict:
    case = load_case(case_file)
    case_input = build_case_input(case, docs)
    ground_truth = case.get("ground_truth_complexity", {})
    case_type = case["case_id"].split("-")[2][0]

    try:
        result = run_react_agent(case_input, docs)
        determination = str(result.get("determination", ""))
        return make_result(
            case_id=case["case_id"],
            case_type=case_type,
            system="react",
            determination=determination,
            elapsed=result.get("elapsed", 0),
            tier=None,
            trajectory=None,
            steps=1,
            error=None,
            ground_truth=ground_truth,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return make_result(
            case_id=case["case_id"],
            case_type=case_type,
            system="react",
            determination=f"[ReAct failed: {e}]",
            elapsed=0,
            tier=None,
            trajectory=None,
            steps=None,
            error=str(e) + "\n" + tb[-500:],
            ground_truth=ground_truth,
        )


# ── Console helpers ─────────────────────────────────────────────────────────

def print_banner():
    print(f"\n{'═' * 72}")
    print(f"  PRIOR AUTHORIZATION — 20-CASE BENCHMARK")
    print(f"  Cognitive Core vs. ReAct Agent")
    print(f"{'═' * 72}\n")


def print_case_start(case_id: str, i: int, total: int, system: str):
    print(f"[{i:2d}/{total}] {case_id}  ({system.upper()})")


def print_case_done(result: dict):
    disp = result["disposition"]
    gt   = result["ground_truth_answer"][:60] + "..." if len(result["ground_truth_answer"]) > 60 else result["ground_truth_answer"]
    # Rough correctness signal — does extracted disposition appear in ground truth?
    match = "✓" if disp in result["ground_truth_answer"].upper() else "?"
    err = f"  ERROR: {result['error'][:80]}" if result["error"] else ""
    print(
        f"       → {disp:<10} {match}  "
        f"{result['elapsed_s']:>6.1f}s  "
        f"{result['char_count']:>6,} chars"
        f"{err}"
    )


def print_progress_summary(results: list[dict]):
    cc    = [r for r in results if r["system"] == "cc"]
    react = [r for r in results if r["system"] == "react"]
    print(f"\n{'─' * 72}")
    print(f"  Progress: CC={len(cc)}/20  ReAct={len(react)}/20")
    if cc:
        disps = {}
        for r in cc:
            disps[r["disposition"]] = disps.get(r["disposition"], 0) + 1
        print(f"  CC dispositions: {disps}")
    if react:
        disps = {}
        for r in react:
            disps[r["disposition"]] = disps.get(r["disposition"], 0) + 1
        print(f"  ReAct dispositions: {disps}")
    print(f"{'─' * 72}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prior Auth 20-Case Benchmark Runner")
    parser.add_argument("--cc-only",    action="store_true", help="Only run CC")
    parser.add_argument("--react-only", action="store_true", help="Only run ReAct")
    parser.add_argument("--resume",     action="store_true", help="Skip already-completed cases")
    parser.add_argument("--verbose",    action="store_true", help="Show LLM call details")
    parser.add_argument("--case",       default=None,        help="Single case file (for debugging)")
    args = parser.parse_args()

    print_banner()

    # Discover cases
    if args.case:
        cases = [args.case]
    else:
        cases = discover_cases()

    print(f"  Cases: {len(cases)}")
    print(f"  Output: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cc").mkdir(exist_ok=True)
    (OUTPUT_DIR / "react").mkdir(exist_ok=True)

    results_path = OUTPUT_DIR / "results.json"
    results = load_existing_results(results_path) if args.resume else []

    docs = load_documents()

    total = len(cases)
    systems = []
    if not args.react_only:
        systems.append("cc")
    if not args.cc_only:
        systems.append("react")

    # Run
    for i, case_file in enumerate(cases, 1):
        case_id = case_file.replace(".json", "").replace("pa_", "PA-").upper()
        # Normalise: pa_2024_a001.json → PA-2024-A001
        case_id = case_file.replace(".json", "")
        parts = case_id.split("_")
        if len(parts) == 4:
            case_id = f"PA-{parts[1]}-{parts[2].upper()}{parts[3]}"
        else:
            case_id = case_file.replace(".json", "")

        for system in systems:
            if args.resume and already_run(results, case_id, system):
                print(f"[{i:2d}/{total}] {case_id}  ({system.upper()})  — SKIPPED (already complete)")
                continue

            print_case_start(case_id, i, total, system)
            t_start = time.time()

            if system == "cc":
                result = run_case_cc(case_file, docs, verbose=args.verbose)
            else:
                result = run_case_react(case_file, docs)

            print_case_done(result)

            # Save determination text
            det_path = OUTPUT_DIR / system / f"{case_id}.txt"
            with open(det_path, "w") as f:
                header = (
                    f"CASE: {case_id}\n"
                    f"SYSTEM: {system.upper()}\n"
                    f"DISPOSITION: {result['disposition']}\n"
                    f"ELAPSED: {result['elapsed_s']}s\n"
                    f"GROUND TRUTH: {result['ground_truth_answer']}\n"
                    f"{'=' * 72}\n\n"
                )
                f.write(header + result["determination"])

            # Strip full determination from JSON to keep results.json lean
            result_lean = {k: v for k, v in result.items() if k != "determination"}
            results.append(result_lean)
            save_results(results, results_path)

        # Progress summary every 5 cases
        if i % 5 == 0 or i == total:
            print_progress_summary(results)

    print(f"\n  Done. Results: {results_path}")
    print(f"  CC texts:    {OUTPUT_DIR / 'cc'}")
    print(f"  ReAct texts: {OUTPUT_DIR / 'react'}\n")


if __name__ == "__main__":
    main()
