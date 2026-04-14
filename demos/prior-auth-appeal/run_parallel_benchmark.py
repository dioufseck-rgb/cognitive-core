"""
Prior Authorization Appeal Review — Parallel Benchmark Runner

Runs all 26 cases through Cognitive Core, ReAct, and Plan-and-Solve
concurrently. Results are written to:

  output/parallel_benchmark/
    results.json                — one record per case per system
    cc/<case_id>.txt            — CC determination text
    react/<case_id>.txt         — ReAct determination text
    plansolve/<case_id>.txt     — Plan-and-Solve determination text
    run_log.txt                 — timestamped progress log
    summary.txt                 — accuracy matrix + silent error counts

The 11-case balanced evaluation set is scored separately for accuracy
comparison (matching the paper benchmark). All 26 cases are run for
trajectory analysis and reasoning graph construction.

Usage:
    # Run all 26 cases, all three systems (default: 5 workers)
    python run_parallel_benchmark.py

    # CC only
    python run_parallel_benchmark.py --cc-only

    # Tune concurrency (watch your rate limits)
    python run_parallel_benchmark.py --workers 8

    # Resume interrupted run
    python run_parallel_benchmark.py --resume

    # Dry run — show what would run, don't execute
    python run_parallel_benchmark.py --dry-run

    # Score only — re-score existing results without re-running
    python run_parallel_benchmark.py --score-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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

OUTPUT_DIR = DEMO_DIR / "output" / "parallel_benchmark"

# ── 11-case balanced evaluation set (paper benchmark) ──────────────────────
BALANCED_SET = {
    "PA-2024-A001", "PA-2024-B004", "PA-2024-G005",   # OVERTURN
    "PA-2024-C004", "PA-2024-G004",                    # UPHOLD
    "PA-2024-D001", "PA-2024-D003",                    # REMAND
    "PA-2024-B001", "PA-2024-E001",                    # PARTIAL
    "PA-2024-C003", "PA-2024-G003",                    # Contested GT
}

# ── Thread-safe progress tracking ──────────────────────────────────────────
_lock   = threading.Lock()
_done   = 0
_total  = 0
_log_fh = None


def _log(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ── Case discovery ──────────────────────────────────────────────────────────

def discover_cases() -> list[str]:
    files = sorted(CASES_DIR.glob("pa_2024_*.json"))
    return [f.name for f in files]


def load_case(filename: str) -> dict:
    with open(CASES_DIR / filename) as f:
        return json.load(f)


def case_id_from_filename(filename: str) -> str:
    name = filename.replace(".json", "")
    parts = name.split("_")
    if len(parts) == 4:
        return f"PA-{parts[1]}-{parts[2].upper()}{parts[3]}"
    return name


# ── Disposition extraction ──────────────────────────────────────────────────

def extract_disposition(text: str) -> str:
    if not text or text.startswith("["):
        return "ERROR"
    upper = text.upper()
    for line in upper.splitlines():
        line = line.strip()
        if line.startswith("DISPOSITION:"):
            for kw in ("OVERTURN", "UPHOLD", "PARTIAL", "REMAND"):
                if kw in line:
                    return kw
    opener = upper[:500]
    for kw in ("PARTIAL", "REMAND", "OVERTURN", "UPHOLD"):
        if kw in opener:
            return kw
    if "UPHELD" in opener:
        return "UPHOLD"
    scores = {kw: upper.count(kw) for kw in ("OVERTURN", "UPHOLD", "PARTIAL", "REMAND")}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNKNOWN"


def extract_gt_disposition(answer: str) -> str:
    upper = answer.upper()
    for kw in ("PARTIAL", "REMAND", "GATE", "OVERTURN", "UPHOLD"):
        if upper.startswith(kw):
            return "UPHOLD" if kw == "GATE" else kw  # GATE GT cases resolve to UPHOLD
    return "UNKNOWN"


# ── Plan-and-Solve baseline ─────────────────────────────────────────────────

def run_plansolve_agent(case_input: dict, docs: dict) -> dict:
    """
    Plan-and-Solve: two sequential LLM calls.
    Phase 1 (Planning): produce a disposition decision tree.
    Phase 2 (Execution): follow the plan to produce the determination.
    """
    from run import _extract_domain_knowledge

    domain_knowledge = _extract_domain_knowledge(docs)

    # ── Phase 1: Planning ──────────────────────────────────────────────────
    plan_prompt = f"""You are an expert health plan appeals reviewer preparing to evaluate a prior
authorization appeal.

CASE SUMMARY:
Case ID: {case_input['case_id']}
Member: {case_input['member_info']}
Procedure: {case_input['procedure_info']}
Denial: {case_input['denial_info']}

DOMAIN KNOWLEDGE AND SOURCE DOCUMENTS:
{domain_knowledge}

INSTRUCTIONS:
Produce a numbered evaluation plan for this appeal. The plan must:
1. Be structured to reach whichever disposition the evidence supports.
   All four dispositions are equally valid: OVERTURN, UPHOLD, PARTIAL, REMAND.
   UPHOLD and REMAND are correct answers when the evidence supports them.
2. Include a procedural defect check (CHSC §1374.31(b)) as Step 1.
3. Include a disposition decision tree: under what conditions is each of
   OVERTURN / UPHOLD / PARTIAL / REMAND the correct answer for THIS case?
4. For each plan step, specify both the positive finding AND the negative finding.
5. Explicitly identify the source hierarchy that governs conflict resolution.

Output the plan only. Do not produce the final determination yet."""

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        t0 = time.time()
        plan_response = model.generate_content(plan_prompt)
        plan_text = plan_response.text if hasattr(plan_response, "text") else str(plan_response)

        # ── Phase 2: Execution ─────────────────────────────────────────────
        exec_prompt = f"""You are an expert health plan appeals reviewer. Execute the evaluation plan
below to produce a formal appeal determination.

CASE:
Case ID: {case_input['case_id']}
Member: {case_input['member_info']}
Procedure: {case_input['procedure_info']}
Denial: {case_input['denial_info']}
Appeal basis: {case_input['appeal_basis']}
Clinical record: {case_input['clinical_record']}

EVALUATION PLAN (follow this exactly):
{plan_text}

DOMAIN KNOWLEDGE AND SOURCE DOCUMENTS:
{domain_knowledge}

EXECUTION INSTRUCTIONS:
- Follow the plan's disposition decision tree step by step.
- A determination that always overturns is as wrong as one that always upholds.
- UPHOLD and REMAND are correct answers when the evidence supports them.
- State your disposition clearly: OVERTURN, UPHOLD, PARTIAL, or REMAND.
- Cite specific sources for every conclusion.

Write a formal appeal determination letter."""

        exec_response = model.generate_content(exec_prompt)
        elapsed = time.time() - t0

        memo = exec_response.text if hasattr(exec_response, "text") else str(exec_response)
        return {"determination": memo, "elapsed": elapsed, "plan": plan_text}

    except Exception as e:
        return {"determination": f"[Plan-and-Solve failed: {e}]",
                "elapsed": 0, "plan": ""}


# ── Result record ───────────────────────────────────────────────────────────

def make_result(
    case_id: str,
    case_type: str,
    system: str,
    determination: str,
    elapsed: float,
    tier: str | None,
    steps: int | None,
    trajectory: list | None,
    error: str | None,
    ground_truth: dict,
) -> dict:
    gt_answer = ground_truth.get("right_answer", "")
    disposition = extract_disposition(determination)
    gt_disp = extract_gt_disposition(gt_answer)
    correct = (
        disposition == gt_disp or
        (gt_disp in ("GATE", "REMAND") and disposition in ("GATE", "REMAND"))
    )
    in_balanced = case_id in BALANCED_SET
    return {
        "case_id":          case_id,
        "case_type":        case_type,
        "system":           system,
        "timestamp":        datetime.utcnow().isoformat(),
        "elapsed_s":        round(elapsed, 1),
        "disposition":      disposition,
        "gt_disposition":   gt_disp,
        "correct":          correct,
        "in_balanced_set":  in_balanced,
        "tier":             tier,
        "steps":            steps,
        "trajectory":       trajectory,
        "char_count":       len(determination),
        "determination":    determination,
        "error":            error,
        "ground_truth_answer": gt_answer,
        "genuine_tension":  ground_truth.get("the_genuine_tension", ""),
        "failure_mode":     ground_truth.get("failure_mode", ""),
    }


# ── Per-case runners ────────────────────────────────────────────────────────

def run_one(case_file: str, system: str, docs: dict, verbose: bool = False) -> dict:
    case = load_case(case_file)
    case_input = build_case_input(case, docs)
    ground_truth = case.get("ground_truth_complexity", {})
    case_type = case["case_id"].split("-")[2][0] if "-" in case["case_id"] else "?"
    case_id = case["case_id"]

    try:
        if system == "cc":
            result = run_cognitive_core(case_input, verbose=verbose)
            det = str(result.get("determination", ""))
            return make_result(
                case_id=case_id, case_type=case_type, system="cc",
                determination=det, elapsed=result.get("elapsed", 0),
                tier=result.get("tier"), steps=len(result.get("trajectory", [])),
                trajectory=result.get("trajectory", []),
                error=None, ground_truth=ground_truth,
            )
        elif system == "react":
            result = run_react_agent(case_input, docs)
            det = str(result.get("determination", ""))
            return make_result(
                case_id=case_id, case_type=case_type, system="react",
                determination=det, elapsed=result.get("elapsed", 0),
                tier=None, steps=1, trajectory=None,
                error=None, ground_truth=ground_truth,
            )
        elif system == "plansolve":
            result = run_plansolve_agent(case_input, docs)
            det = str(result.get("determination", ""))
            return make_result(
                case_id=case_id, case_type=case_type, system="plansolve",
                determination=det, elapsed=result.get("elapsed", 0),
                tier=None, steps=2, trajectory=None,
                error=None, ground_truth=ground_truth,
            )
        else:
            raise ValueError(f"Unknown system: {system}")

    except Exception as e:
        tb = traceback.format_exc()
        return make_result(
            case_id=case_id, case_type=case_type, system=system,
            determination=f"[{system} failed: {e}]", elapsed=0,
            tier=None, steps=None, trajectory=None,
            error=str(e) + "\n" + tb[-500:], ground_truth=ground_truth,
        )


# ── Results persistence ─────────────────────────────────────────────────────

def load_existing_results(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def already_run(results: list[dict], case_id: str, system: str) -> bool:
    return any(
        r["case_id"] == case_id and r["system"] == system and r.get("error") is None
        for r in results
    )


# ── Summary scoring ─────────────────────────────────────────────────────────

def compute_summary(results: list[dict]) -> str:
    systems = ["cc", "react", "plansolve"]
    lines = []
    lines.append("=" * 72)
    lines.append("  BENCHMARK SUMMARY")
    lines.append("=" * 72)

    for scope, label in [
        (lambda r: r["in_balanced_set"], "11-case balanced set (paper benchmark)"),
        (lambda r: True,                 "All 26 cases"),
    ]:
        lines.append(f"\n{label}")
        lines.append("-" * 60)
        scoped = [r for r in results if scope(r)]

        header = f"  {'System':<12} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Silent errors':>14}"
        lines.append(header)
        lines.append("  " + "-" * 56)

        for sys in systems:
            sys_r = [r for r in scoped if r["system"] == sys]
            if not sys_r:
                continue
            correct = sum(1 for r in sys_r if r["correct"])
            total   = len(sys_r)
            acc     = f"{100*correct//total}%" if total else "—"

            # Silent errors: incorrect AND no gate/hold routing (baselines only)
            if sys == "cc":
                # CC silent errors: wrong AND tier is auto or spot_check
                silent = sum(
                    1 for r in sys_r
                    if not r["correct"] and r.get("tier") in ("auto", "spot_check", None)
                )
            else:
                # Baselines: all errors are silent (no governance tier)
                silent = sum(1 for r in sys_r if not r["correct"])

            lines.append(f"  {sys:<12} {correct:>8} {total:>6} {acc:>10} {silent:>14}")

        # Disposition breakdown
        lines.append("\n  Dispositions by system:")
        for sys in systems:
            sys_r = [r for r in scoped if r["system"] == sys]
            if not sys_r:
                continue
            counts = {}
            for r in sys_r:
                counts[r["disposition"]] = counts.get(r["disposition"], 0) + 1
            counts_str = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            lines.append(f"    {sys:<10} {counts_str}")

    # CC tier distribution
    lines.append("\n  CC governance tier distribution (all cases):")
    cc_results = [r for r in results if r["system"] == "cc"]
    tier_counts = {}
    for r in cc_results:
        t = r.get("tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for t, c in sorted(tier_counts.items()):
        lines.append(f"    {t:<12} {c}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parallel Prior Auth Benchmark — CC + ReAct + Plan-and-Solve"
    )
    parser.add_argument("--cc-only",    action="store_true", help="Only run CC")
    parser.add_argument("--react-only", action="store_true", help="Only run ReAct")
    parser.add_argument("--ps-only",    action="store_true", help="Only run Plan-and-Solve")
    parser.add_argument("--resume",     action="store_true", help="Skip already-completed cases")
    parser.add_argument("--verbose",    action="store_true", help="Show LLM call details (CC only)")
    parser.add_argument("--workers",    type=int, default=5, help="Concurrent workers (default: 5)")
    parser.add_argument("--dry-run",    action="store_true", help="Show plan without running")
    parser.add_argument("--score-only", action="store_true", help="Re-score existing results")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sys in ("cc", "react", "plansolve"):
        (OUTPUT_DIR / sys).mkdir(exist_ok=True)

    results_path = OUTPUT_DIR / "results.json"

    # Score-only mode
    if args.score_only:
        results = load_existing_results(results_path)
        if not results:
            print("No results found. Run the benchmark first.")
            sys.exit(1)
        summary = compute_summary(results)
        print(summary)
        (OUTPUT_DIR / "summary.txt").write_text(summary)
        return

    # Determine systems to run
    systems = []
    if args.cc_only:
        systems = ["cc"]
    elif args.react_only:
        systems = ["react"]
    elif args.ps_only:
        systems = ["plansolve"]
    else:
        systems = ["cc", "react", "plansolve"]

    cases = discover_cases()
    results = load_existing_results(results_path) if args.resume else []

    # Build work queue
    work = []
    for case_file in cases:
        case_id = case_id_from_filename(case_file)
        for sys in systems:
            if args.resume and already_run(results, case_id, sys):
                continue
            work.append((case_file, sys))

    print(f"\n{'═' * 72}")
    print(f"  PARALLEL PRIOR AUTH BENCHMARK")
    print(f"  Systems: {', '.join(systems)}")
    print(f"  Cases: {len(cases)}  ·  Jobs: {len(work)}  ·  Workers: {args.workers}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'═' * 72}\n")

    if args.dry_run:
        print("Dry run — jobs that would execute:")
        for case_file, sys in work:
            case_id = case_id_from_filename(case_file)
            balanced = " [balanced]" if case_id in BALANCED_SET else ""
            print(f"  {case_id:<22} {sys}{balanced}")
        return

    global _done, _total, _log_fh
    _total = len(work)
    _done  = 0

    log_path = OUTPUT_DIR / "run_log.txt"
    _log_fh = open(log_path, "w")

    docs = load_documents()

    _log(f"Starting {_total} jobs across {args.workers} workers")

    results_lock = threading.Lock()

    def run_job(job):
        case_file, sys = job
        case_id = case_id_from_filename(case_file)
        t0 = time.time()
        result = run_one(case_file, sys, docs, verbose=args.verbose)
        elapsed = time.time() - t0

        status = "✓" if result["correct"] else "✗"
        disp   = result["disposition"]
        gt     = result["gt_disposition"]
        tier   = f" [{result['tier']}]" if result.get("tier") else ""
        err    = f" ERROR: {result['error'][:60]}" if result.get("error") else ""

        with _lock:
            global _done
            _done += 1
            _log(
                f"[{_done:3d}/{_total}] {case_id:<22} {sys:<10} "
                f"{status} {disp:<10} (GT:{gt}){tier}  {elapsed:.0f}s{err}"
            )

        # Save determination text to file
        det_path = OUTPUT_DIR / sys / f"{case_id}.txt"
        with open(det_path, "w") as f:
            header = (
                f"CASE: {case_id}\n"
                f"SYSTEM: {sys.upper()}\n"
                f"DISPOSITION: {result['disposition']}\n"
                f"GT: {result['gt_disposition']}\n"
                f"CORRECT: {result['correct']}\n"
                f"ELAPSED: {result['elapsed_s']}s\n"
                f"{'=' * 72}\n\n"
            )
            f.write(header + result.get("determination", ""))

        # Append to results — keep determination in JSON for score_benchmark.py
        # Strip only the plan text (Plan-and-Solve intermediate artifact)
        result_lean = {k: v for k, v in result.items() if k != "plan"}
        with results_lock:
            results.append(result_lean)
            save_results(results, results_path)

        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job): job for job in work}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                case_file, sys = futures[future]
                _log(f"  FATAL ERROR {case_file} {sys}: {e}")

    # Final summary
    summary = compute_summary(results)
    print(f"\n{summary}")
    (OUTPUT_DIR / "summary.txt").write_text(summary)
    _log("Done.")
    _log_fh.close()

    print(f"\n  Results: {results_path}")
    print(f"  Summary: {OUTPUT_DIR / 'summary.txt'}")
    print(f"  Log:     {log_path}\n")


if __name__ == "__main__":
    main()
