"""
Loan Modification — Benchmark Runner
=====================================
Runs all cases through Cognitive Core, ReAct, and Plan-and-Solve.
Results written to output/benchmark/.

Usage:
    python demos/loan-modification/run_benchmark.py
    python demos/loan-modification/run_benchmark.py --cc-only
    python demos/loan-modification/run_benchmark.py --case lm_2024_a001.json
    python demos/loan-modification/run_benchmark.py --resume
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
    run_plan_and_solve_agent,
    CASES_DIR,
)

OUTPUT_DIR = DEMO_DIR / "output" / "benchmark"

# ── Case discovery ──────────────────────────────────────────────────────────

def discover_cases() -> list[str]:
    files = sorted(CASES_DIR.glob("lm_2024_*.json"))
    return [f.name for f in files]

def load_case(filename: str) -> dict:
    with open(CASES_DIR / filename) as f:
        return json.load(f)

# ── Result record ───────────────────────────────────────────────────────────

def extract_disposition(text: str) -> str:
    if not text or text.startswith("["):
        return "ERROR"
    upper = text.upper()
    # Look for explicit DETERMINATION: or DISPOSITION: line first
    for line in upper.splitlines():
        line = line.strip()
        for prefix in ("DETERMINATION:", "DISPOSITION:", "FINAL DETERMINATION:"):
            if line.startswith(prefix):
                for kw in ("PARTIAL", "APPROVE", "DENY", "PEND", "REFER"):
                    if kw in line:
                        return kw
    # Fall back to first strong keyword in first 500 chars
    opening = upper[:500]
    for kw in ("PARTIAL", "PEND", "REFER", "APPROVE", "DENY"):
        if kw in opening:
            return kw
    return "UNKNOWN"

def make_result(
    case_id: str,
    system: str,
    determination: str,
    elapsed: float,
    tier: str | None,
    trajectory: list[str] | None,
    error: str | None,
    ground_truth: dict,
) -> dict:
    return {
        "case_id":        case_id,
        "system":         system,
        "timestamp":      datetime.utcnow().isoformat(),
        "elapsed_s":      round(elapsed, 1),
        "disposition":    extract_disposition(determination),
        "tier":           tier,
        "trajectory":     trajectory,
        "char_count":     len(determination),
        "error":          error,
        "ground_truth_answer":  ground_truth.get("right_answer", ""),
        "obvious_reading":      ground_truth.get("obvious_reading", ""),
        "genuine_tension":      ground_truth.get("the_genuine_tension", ""),
        "determination":  determination,
    }

# ── Load/save results ───────────────────────────────────────────────────────

def load_existing_results(results_path: Path) -> list[dict]:
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []

def save_results(results: list[dict], results_path: Path):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

# ── Run one case ────────────────────────────────────────────────────────────

def run_case(filename: str, docs: dict, args, results: list, results_path: Path,
             log_file, output_dir: Path):
    case = load_case(filename)
    case_id = case["case_id"]
    case_input = build_case_input(case, docs)
    gt = case.get("ground_truth_complexity", {})

    print(f"\n{'─' * 60}")
    print(f"  {case_id}")
    print(f"  Borrower: {case['borrower']['name']}")
    print(f"  Hardship: {case['hardship']['type']}")
    print(f"  Ground truth: {gt.get('right_answer','?')[:80]}...")
    print(f"{'─' * 60}")

    if log_file:
        log_file.write(f"\n{case_id}\n")

    already_done = {(r["case_id"], r["system"]) for r in results}

    # CC
    if not args.react_only and not args.ps_only:
        if args.resume and (case_id, "cc") in already_done:
            print(f"  [skip] CC already in results")
        else:
            try:
                cc = run_cognitive_core(case_input, verbose=args.verbose)
                r = make_result(case_id, "cc",
                                cc.get("determination",""), cc["elapsed"],
                                cc.get("tier"), cc.get("trajectory"), None, gt)
                results.append(r)
                save_results(results, results_path)
                _save_text(output_dir / "cc" / f"{case_id}.txt",
                           f"CC — {case_id}\nTier: {cc.get('tier','?').upper()}\n"
                           + "=" * 60 + "\n\n" + str(cc.get("determination","")))
            except Exception as e:
                tb = traceback.format_exc()
                r = make_result(case_id, "cc", f"[ERROR: {e}]", 0, None, None, str(e), gt)
                results.append(r)
                save_results(results, results_path)
                print(f"  CC ERROR: {e}\n{tb}")

    # ReAct
    if not args.cc_only and not args.ps_only:
        if args.resume and (case_id, "react") in already_done:
            print(f"  [skip] ReAct already in results")
        else:
            try:
                react = run_react_agent(case_input, docs)
                r = make_result(case_id, "react",
                                react.get("determination",""), react.get("elapsed",0),
                                None, None, None, gt)
                results.append(r)
                save_results(results, results_path)
                _save_text(output_dir / "react" / f"{case_id}.txt",
                           f"ReAct — {case_id}\n" + "=" * 60 + "\n\n"
                           + str(react.get("determination","")))
            except Exception as e:
                r = make_result(case_id, "react", f"[ERROR: {e}]", 0, None, None, str(e), gt)
                results.append(r)
                save_results(results, results_path)
                print(f"  ReAct ERROR: {e}")

    # Plan-and-Solve
    if not args.cc_only and not args.react_only:
        if args.resume and (case_id, "plansolve") in already_done:
            print(f"  [skip] Plan-and-Solve already in results")
        else:
            try:
                ps = run_plan_and_solve_agent(case_input, docs)
                r = make_result(case_id, "plansolve",
                                ps.get("determination",""), ps.get("elapsed",0),
                                None, None, None, gt)
                results.append(r)
                save_results(results, results_path)
                _save_text(output_dir / "plansolve" / f"{case_id}.txt",
                           f"Plan-and-Solve — {case_id}\n" + "=" * 60 + "\n\n"
                           + f"PLAN:\n{ps.get('plan','')}\n\n" + "=" * 60
                           + "\n\nDETERMINATION:\n" + str(ps.get("determination","")))
            except Exception as e:
                r = make_result(case_id, "plansolve", f"[ERROR: {e}]", 0, None, None, str(e), gt)
                results.append(r)
                save_results(results, results_path)
                print(f"  Plan-and-Solve ERROR: {e}")


def _save_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ── Summary ─────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    from collections import defaultdict
    by_system = defaultdict(list)
    for r in results:
        by_system[r["system"]].append(r)

    print(f"\n{'═' * 72}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'═' * 72}")

    for system in ["cc", "react", "plansolve"]:
        rs = by_system.get(system, [])
        if not rs:
            continue
        errors = sum(1 for r in rs if r.get("error"))
        unknowns = sum(1 for r in rs if r["disposition"] == "UNKNOWN")
        avg_time = sum(r["elapsed_s"] for r in rs) / len(rs) if rs else 0
        dispositions = {}
        for r in rs:
            d = r["disposition"]
            dispositions[d] = dispositions.get(d, 0) + 1
        label = {"cc": "Cognitive Core", "react": "ReAct (Yao 2022)",
                 "plansolve": "Plan-and-Solve (Wang 2023)"}.get(system, system)
        print(f"\n  {label}")
        print(f"    Cases run: {len(rs)}  Errors: {errors}  Unknowns: {unknowns}")
        print(f"    Avg time: {avg_time:.1f}s")
        print(f"    Dispositions: {dispositions}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Loan Modification Benchmark Runner")
    parser.add_argument("--cc-only",    action="store_true")
    parser.add_argument("--react-only", action="store_true")
    parser.add_argument("--ps-only",    action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip cases already in results.json")
    parser.add_argument("--case",       default=None,
                        help="Run a single case file")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "cc").mkdir(exist_ok=True)
    (OUTPUT_DIR / "react").mkdir(exist_ok=True)
    (OUTPUT_DIR / "plansolve").mkdir(exist_ok=True)

    results_path = OUTPUT_DIR / "results.json"
    results = load_existing_results(results_path) if args.resume else []

    docs = load_documents()
    cases = [args.case] if args.case else discover_cases()

    print(f"\n{'═' * 72}")
    print(f"  LOAN MODIFICATION BENCHMARK")
    print(f"  Cases: {len(cases)}")
    systems = []
    if not args.react_only and not args.ps_only:
        systems.append("Cognitive Core")
    if not args.cc_only and not args.ps_only:
        systems.append("ReAct (Yao 2022)")
    if not args.cc_only and not args.react_only:
        systems.append("Plan-and-Solve (Wang 2023)")
    print(f"  Systems: {', '.join(systems)}")
    print(f"{'═' * 72}")

    log_path = OUTPUT_DIR / "run_log.txt"
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\nRUN: {datetime.utcnow().isoformat()}  cases={len(cases)}\n")
        for filename in cases:
            run_case(filename, docs, args, results, results_path, log_file, OUTPUT_DIR)

    print_summary(results)
    print(f"\n  Results: {results_path}")
    print(f"  Log: {log_path}")


if __name__ == "__main__":
    main()
