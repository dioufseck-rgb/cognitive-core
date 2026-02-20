#!/usr/bin/env python3
"""
Cognitive Core — Soundness Evaluation

Goes beyond "do tests pass?" to answer:
  1. CONSISTENCY:  Do deterministic rules produce identical outputs across N runs?
  2. SAFETY NET:   When the LLM gets it wrong, does governance catch it?
  3. MODEL SENSITIVITY: How does accuracy change across model tiers?
  4. BRITTLENESS:  Which prompt/domain patterns break down and why?

Usage:
  # Quick consistency check (3 runs, default model)
  python scripts/eval_soundness.py --mode consistency --runs 3

  # Full soundness report (5 runs)
  python scripts/eval_soundness.py --mode full --runs 5

  # Model comparison (1 run per model)
  python scripts/eval_soundness.py --mode model-compare

  # Single case deep-dive (10 runs)
  python scripts/eval_soundness.py --mode consistency --case sc_004_both_delegations --runs 10

  # Brittleness analysis
  python scripts/eval_soundness.py --mode brittleness
"""

import argparse
import json
import os
import sys
import time
import tempfile
import statistics
import copy
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

# Import the CASES dict from eval_live
from scripts.eval_live import CASES


# ═════════════════════════════════════════════════════════════════
# Core: Single-case runner (returns structured result)
# ═════════════════════════════════════════════════════════════════

def run_single_case(case_name: str, case_def: dict, project_root: str,
                    auto_approve: bool = True, verbose: bool = False) -> dict:
    """Run a single case and return detailed structured results."""
    from coordinator.runtime import Coordinator
    from coordinator.types import InstanceStatus

    case_path = os.path.join(project_root, case_def["file"])
    with open(case_path) as f:
        case_data = json.load(f)
    case_data.pop("_meta", None)

    config_path = os.path.join(project_root, "coordinator", "config.yaml")
    db_path = os.path.join(tempfile.mkdtemp(), f"eval_{case_name}_{time.time():.0f}.db")
    coord = Coordinator(config_path=config_path, db_path=db_path, verbose=verbose)

    t0 = time.time()
    try:
        instance_id = coord.start(
            workflow_type=case_def["workflow"],
            domain=case_def["domain"],
            case_input=case_data,
        )
    except Exception as e:
        return {
            "case": case_name, "status": "crashed", "error": str(e),
            "elapsed": time.time() - t0, "step_outputs": {}, "checks": {},
            "governance_events": [],
        }

    instance = coord.get_instance(instance_id)

    # Handle suspension
    governance_events = []
    if instance.status == InstanceStatus.SUSPENDED:
        governance_events.append({
            "type": "gate_suspension",
            "instance_id": instance_id,
            "action": "auto_approved" if auto_approve else "blocked",
        })
        if auto_approve:
            coord.approve(instance_id, approver="eval_runner")
            instance = coord.get_instance(instance_id)

    elapsed = time.time() - t0

    # Extract step outputs
    result = instance.result or {}
    steps = result.get("steps", [])
    step_outputs = {}
    for step in steps:
        name = step.get("step_name", "")
        step_outputs[name] = step

    # Run checks
    checks = {}
    for check_key, expected_val in case_def.get("checks", {}).items():
        step_name, field = check_key.rsplit(".", 1)
        actual = step_outputs.get(step_name, {}).get(field)
        checks[check_key] = {
            "expected": expected_val,
            "actual": actual,
            "passed": actual == expected_val,
        }

    # Risk score check
    import re as _re
    risk_range = case_def.get("risk_range")
    if risk_range:
        step_data = step_outputs.get("assess_risk", {})
        risk_score = step_data.get("risk_score")
        if risk_score is None:
            for text in [step_data.get("thought", "")] + (step_data.get("conclusions", []) or []):
                if isinstance(text, str):
                    match = _re.search(r'(?:risk.?score|score)\s*[=:]\s*(\d+)', text, _re.IGNORECASE)
                    if match:
                        risk_score = int(match.group(1))
                        break
        in_range = (risk_score is not None
                    and isinstance(risk_score, (int, float))
                    and risk_range[0] <= risk_score <= risk_range[1])
        checks["risk_score"] = {
            "expected": f"[{risk_range[0]}, {risk_range[1]}]",
            "actual": risk_score,
            "passed": in_range,
        }

    # Delegation checks
    if case_def.get("delegations") and instance.status == InstanceStatus.COMPLETED:
        chain = coord.get_correlation_chain(instance.correlation_id)
        chain_wfs = {c.workflow_type for c in chain if c.instance_id != instance_id}
        for wf_name, should_exist in case_def["delegations"].items():
            exists = wf_name in chain_wfs
            checks[f"delegation_{wf_name}"] = {
                "expected": should_exist,
                "actual": exists,
                "passed": exists == should_exist,
            }

    # Quality gate events
    ledger = coord.get_ledger(instance_id=instance_id)
    for entry in ledger:
        e_type = getattr(entry, 'event_type', '') or ''
        if 'gate' in e_type.lower() or 'suspend' in e_type.lower():
            governance_events.append({
                "type": e_type,
                "step": getattr(entry, 'step_name', None),
                "detail": getattr(entry, 'detail', None),
            })

    all_passed = all(c["passed"] for c in checks.values())
    return {
        "case": case_name,
        "status": "pass" if all_passed else "fail",
        "elapsed": round(elapsed, 2),
        "step_outputs": {k: _safe_serialize(v) for k, v in step_outputs.items()},
        "checks": checks,
        "governance_events": governance_events,
        "instance_id": instance_id,
    }


def _safe_serialize(obj, max_depth=3):
    """Serialize step output for JSON, truncating large values."""
    if max_depth <= 0:
        return str(obj)[:200]
    if isinstance(obj, dict):
        return {k: _safe_serialize(v, max_depth - 1) for k, v in obj.items()
                if k not in ("raw_response", "prompt_used")}
    if isinstance(obj, list):
        return [_safe_serialize(v, max_depth - 1) for v in obj[:10]]
    if isinstance(obj, (str,)) and len(obj) > 500:
        return obj[:500] + "..."
    return obj


# ═════════════════════════════════════════════════════════════════
# Mode 1: CONSISTENCY — same case, N runs, measure variance
# ═════════════════════════════════════════════════════════════════

def eval_consistency(case_names: list[str], n_runs: int, project_root: str, verbose: bool = False):
    """Run each case N times. Measure output determinism."""
    print("═" * 70)
    print("SOUNDNESS EVAL: CONSISTENCY")
    print(f"Cases: {len(case_names)} × {n_runs} runs = {len(case_names) * n_runs} total executions")
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'not set')}")
    print("═" * 70)

    results = {}
    for case_name in case_names:
        case_def = CASES[case_name]
        print(f"\n{'─'*60}")
        print(f"CASE: {case_name} ({n_runs} runs)")

        runs = []
        for i in range(n_runs):
            print(f"  Run {i+1}/{n_runs}...", end=" ", flush=True)
            r = run_single_case(case_name, case_def, project_root, verbose=verbose)
            runs.append(r)
            status = "✅" if r["status"] == "pass" else "❌" if r["status"] == "fail" else "💥"
            print(f"{status} ({r['elapsed']:.1f}s)")

        # Analyze consistency
        analysis = _analyze_consistency(case_name, case_def, runs)
        results[case_name] = analysis

        # Print summary
        pass_rate = analysis["pass_rate"]
        icon = "✅" if pass_rate == 1.0 else "⚠️" if pass_rate >= 0.8 else "❌"
        print(f"\n  {icon} {case_name}: {pass_rate:.0%} pass rate ({analysis['passes']}/{n_runs})")

        for check_name, check_stats in analysis["check_details"].items():
            rate = check_stats["pass_rate"]
            c_icon = "✅" if rate == 1.0 else "⚠️" if rate > 0 else "❌"
            values = check_stats.get("unique_values", [])
            print(f"    {c_icon} {check_name}: {rate:.0%} | values seen: {values}")

        if analysis.get("flaky_checks"):
            print(f"  ⚠️  FLAKY: {analysis['flaky_checks']}")

    # Summary
    _print_consistency_summary(results, n_runs)
    return results


def _analyze_consistency(case_name, case_def, runs):
    """Analyze N runs of the same case for consistency."""
    passes = sum(1 for r in runs if r["status"] == "pass")
    fails = sum(1 for r in runs if r["status"] == "fail")
    crashes = sum(1 for r in runs if r["status"] == "crashed")
    times = [r["elapsed"] for r in runs]

    check_details = {}
    flaky_checks = []

    # Collect all check keys across runs
    all_check_keys = set()
    for r in runs:
        all_check_keys.update(r.get("checks", {}).keys())

    for check_key in sorted(all_check_keys):
        values = []
        pass_count = 0
        for r in runs:
            c = r.get("checks", {}).get(check_key, {})
            values.append(c.get("actual"))
            if c.get("passed"):
                pass_count += 1

        unique_values = list(set(str(v) for v in values))
        is_deterministic = len(unique_values) == 1
        pass_rate = pass_count / len(runs) if runs else 0

        check_details[check_key] = {
            "pass_rate": pass_rate,
            "deterministic": is_deterministic,
            "unique_values": unique_values,
            "value_counts": dict(Counter(str(v) for v in values)),
        }

        if not is_deterministic:
            flaky_checks.append(check_key)

    # Governance events
    governance_fired = sum(1 for r in runs if r.get("governance_events"))

    return {
        "case": case_name,
        "n_runs": len(runs),
        "passes": passes,
        "fails": fails,
        "crashes": crashes,
        "pass_rate": passes / len(runs) if runs else 0,
        "mean_time": statistics.mean(times) if times else 0,
        "stdev_time": statistics.stdev(times) if len(times) > 1 else 0,
        "check_details": check_details,
        "flaky_checks": flaky_checks,
        "deterministic": len(flaky_checks) == 0 and crashes == 0,
        "governance_fired_count": governance_fired,
    }


def _print_consistency_summary(results, n_runs):
    print(f"\n{'═'*70}")
    print("CONSISTENCY SUMMARY")
    print(f"{'═'*70}")

    total_cases = len(results)
    fully_deterministic = sum(1 for r in results.values() if r["deterministic"])
    fully_passing = sum(1 for r in results.values() if r["pass_rate"] == 1.0)
    flaky_cases = [name for name, r in results.items() if 0 < r["pass_rate"] < 1.0]
    broken_cases = [name for name, r in results.items() if r["pass_rate"] == 0]

    print(f"  Cases:            {total_cases}")
    print(f"  Runs per case:    {n_runs}")
    print(f"  100% pass rate:   {fully_passing}/{total_cases}")
    print(f"  Deterministic:    {fully_deterministic}/{total_cases}")

    if flaky_cases:
        print(f"\n  ⚠️  FLAKY CASES (intermittent failures):")
        for name in flaky_cases:
            r = results[name]
            print(f"    {name}: {r['pass_rate']:.0%} ({r['passes']}/{r['n_runs']})")
            for ck, cd in r["check_details"].items():
                if not cd["deterministic"]:
                    print(f"      → {ck}: values={cd['value_counts']}")

    if broken_cases:
        print(f"\n  ❌ BROKEN CASES (0% pass rate):")
        for name in broken_cases:
            r = results[name]
            for ck, cd in r["check_details"].items():
                if cd["pass_rate"] == 0:
                    print(f"    {name}.{ck}: always={cd['unique_values']}")

    # Aggregate check-level stats
    all_checks = defaultdict(list)
    for r in results.values():
        for ck, cd in r["check_details"].items():
            check_type = ck.split(".")[-1] if "." in ck else ck
            all_checks[check_type].append(cd["pass_rate"])

    print(f"\n  CHECK TYPE RELIABILITY:")
    for check_type, rates in sorted(all_checks.items()):
        avg = statistics.mean(rates)
        icon = "✅" if avg == 1.0 else "⚠️" if avg >= 0.8 else "❌"
        print(f"    {icon} {check_type}: {avg:.0%} avg across {len(rates)} checks")

    print(f"{'═'*70}")


# ═════════════════════════════════════════════════════════════════
# Mode 2: BRITTLENESS — perturb inputs, check if outputs break
# ═════════════════════════════════════════════════════════════════

def eval_brittleness(project_root: str, verbose: bool = False):
    """Generate adversarial perturbations and test robustness."""
    print("═" * 70)
    print("SOUNDNESS EVAL: BRITTLENESS ANALYSIS")
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'not set')}")
    print("═" * 70)

    perturbations = _generate_perturbations()
    results = []

    for p in perturbations:
        print(f"\n{'─'*60}")
        print(f"PERTURBATION: {p['name']}")
        print(f"  Base case: {p['base_case']}")
        print(f"  Change: {p['description']}")

        case_def = copy.deepcopy(CASES[p["base_case"]])
        case_def["file"] = p["fixture_file"]

        r = run_single_case(p["name"], case_def, project_root, verbose=verbose)

        # Check specific perturbation expectations
        expected_result = p.get("expected_status", "pass")
        expected_checks = p.get("expected_checks", {})

        perturbation_passed = True
        notes = []

        if expected_result == "any":
            notes.append("no specific expectation")
        elif r["status"] != expected_result:
            perturbation_passed = False
            notes.append(f"expected status={expected_result}, got={r['status']}")

        for ck, ev in expected_checks.items():
            actual = r["checks"].get(ck, {}).get("actual")
            if actual != ev:
                perturbation_passed = False
                notes.append(f"{ck}: expected={ev}, got={actual}")

        icon = "✅" if perturbation_passed else "❌"
        print(f"  {icon} {p['name']}: {'; '.join(notes) if notes else 'as expected'}")

        results.append({
            "perturbation": p["name"],
            "base_case": p["base_case"],
            "description": p["description"],
            "expected_status": expected_result,
            "actual_status": r["status"],
            "perturbation_passed": perturbation_passed,
            "notes": notes,
            "checks": r["checks"],
        })

    # Summary
    passed = sum(1 for r in results if r["perturbation_passed"])
    print(f"\n{'═'*70}")
    print(f"BRITTLENESS SUMMARY: {passed}/{len(results)} perturbations handled correctly")
    if passed < len(results):
        print("  FAILURES:")
        for r in results:
            if not r["perturbation_passed"]:
                print(f"    {r['perturbation']}: {r['notes']}")
    print(f"{'═'*70}")

    return results


def _generate_perturbations():
    """Create perturbed test fixtures from base cases."""
    perturbations = []

    # P1: Amount at exact E3 boundary ($500,000)
    p = _load_and_modify("sc_001_simple_approve", "perturb_boundary_amount",
                         "E3 boundary: amount = exactly $500,000",
                         {"get_claim.amount": 500000})
    p["expected_checks"] = {"check_eligibility.conforms": True}
    perturbations.append(p)

    # P2: Amount at E3 violation ($500,001)
    p = _load_and_modify("sc_001_simple_approve", "perturb_over_boundary",
                         "E3 violation: amount = $500,001",
                         {"get_claim.amount": 500001})
    p["expected_checks"] = {"check_eligibility.conforms": False}
    perturbations.append(p)

    # P3: Amount at E3 violation ($0)
    p = _load_and_modify("sc_001_simple_approve", "perturb_zero_amount",
                         "E3 violation: amount = $0",
                         {"get_claim.amount": 0})
    p["expected_checks"] = {"check_eligibility.conforms": False}
    perturbations.append(p)

    # P4: Future date (clear E4 violation)
    p = _load_and_modify("sc_001_simple_approve", "perturb_future_date",
                         "E4 violation: incident date in 2027",
                         {"get_claim.incident_date": "2027-06-15"})
    p["expected_checks"] = {"check_eligibility.conforms": False}
    perturbations.append(p)

    # P5: Add maximum flags but don't change eligibility fields
    # (eligibility should still pass — flags aren't eligibility criteria)
    p = _load_and_modify("sc_001_simple_approve", "perturb_max_flags_eligible",
                         "Max flags but all E1-E4 pass — should still conform",
                         {"get_claim.flags": ["fraud_indicator", "repeat_claimant",
                                              "new_policy", "high_velocity"]})
    p["expected_checks"] = {"check_eligibility.conforms": True}
    perturbations.append(p)

    # P6: Coverage type with extra whitespace/formatting
    p = _load_and_modify("sc_001_simple_approve", "perturb_coverage_format",
                         "Coverage with spaces: 'liability, comprehensive'",
                         {"get_policy.coverage_type": "liability, comprehensive"})
    p["expected_checks"] = {"check_eligibility.conforms": True}
    p["expected_status"] = "any"  # This is an edge case — may or may not work
    perturbations.append(p)

    # P7: Claim type that doesn't match any E2 mapping explicitly
    # "other" should always pass E2
    p = _load_and_modify("sc_001_simple_approve", "perturb_other_type_minimal_coverage",
                         "claim_type=other with minimal coverage (should pass E2)",
                         {"get_claim.claim_type_hint": "other",
                          "get_policy.coverage_type": "liability"})
    p["expected_checks"] = {
        "classify_claim_type.category": "other",
        "check_eligibility.conforms": True,
    }
    perturbations.append(p)

    return perturbations


def _load_and_modify(base_case: str, perturb_name: str, description: str,
                     modifications: dict) -> dict:
    """Load a base fixture, apply modifications, save as new fixture."""
    base_def = CASES[base_case]
    base_path = os.path.join(BASE, base_def["file"])
    with open(base_path) as f:
        data = json.load(f)
    data.pop("_meta", None)

    for path, value in modifications.items():
        parts = path.split(".")
        obj = data
        for p in parts[:-1]:
            obj = obj[p]
        obj[parts[-1]] = value

    fixture_dir = os.path.join(BASE, "cases", "synthetic", "perturbations")
    os.makedirs(fixture_dir, exist_ok=True)
    fixture_path = os.path.join(fixture_dir, f"{perturb_name}.json")
    with open(fixture_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "name": perturb_name,
        "base_case": base_case,
        "description": description,
        "fixture_file": os.path.relpath(fixture_path, BASE),
        "expected_status": "any",
    }


# ═════════════════════════════════════════════════════════════════
# Report Generation
# ═════════════════════════════════════════════════════════════════

def write_report(consistency_results=None, brittleness_results=None,
                 output_path=None):
    """Write comprehensive soundness report."""
    if output_path is None:
        output_path = os.path.join(BASE, "soundness_report.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "provider": os.environ.get("LLM_PROVIDER", "unknown"),
    }

    if consistency_results:
        # Compute aggregate stats
        all_rates = [r["pass_rate"] for r in consistency_results.values()]
        all_deterministic = [r["deterministic"] for r in consistency_results.values()]
        flaky = {name: r for name, r in consistency_results.items()
                 if 0 < r["pass_rate"] < 1.0}

        report["consistency"] = {
            "total_cases": len(consistency_results),
            "mean_pass_rate": statistics.mean(all_rates) if all_rates else 0,
            "fully_deterministic": sum(all_deterministic),
            "flaky_cases": list(flaky.keys()),
            "broken_cases": [n for n, r in consistency_results.items() if r["pass_rate"] == 0],
            "per_case": {
                name: {
                    "pass_rate": r["pass_rate"],
                    "deterministic": r["deterministic"],
                    "flaky_checks": r["flaky_checks"],
                    "mean_time_s": round(r["mean_time"], 1),
                }
                for name, r in consistency_results.items()
            },
        }

    if brittleness_results:
        passed = sum(1 for r in brittleness_results if r["perturbation_passed"])
        report["brittleness"] = {
            "total_perturbations": len(brittleness_results),
            "passed": passed,
            "failed": len(brittleness_results) - passed,
            "perturbations": brittleness_results,
        }

    # Design guidance based on findings
    report["guidance"] = _generate_guidance(consistency_results, brittleness_results)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {output_path}")
    return report


def _generate_guidance(consistency_results=None, brittleness_results=None) -> list[str]:
    """Generate actionable guidance for prompt/domain/workflow authors."""
    guidance = []

    if consistency_results:
        flaky = {name: r for name, r in consistency_results.items()
                 if 0 < r["pass_rate"] < 1.0}

        # Check which check types are unreliable
        check_type_rates = defaultdict(list)
        for r in consistency_results.values():
            for ck, cd in r["check_details"].items():
                check_type = ck.split(".")[-1] if "." in ck else ck
                check_type_rates[check_type].append(cd["pass_rate"])

        for ct, rates in check_type_rates.items():
            avg = statistics.mean(rates)
            if avg < 1.0:
                if ct == "conforms":
                    guidance.append(
                        f"VERIFY RELIABILITY ({avg:.0%}): The 'verify' primitive's conforms "
                        f"output is non-deterministic. Consider: (a) using a stronger model "
                        f"tier for verify steps, (b) making rules more explicit with examples, "
                        f"(c) providing a curated 'subject' param to limit what the LLM sees."
                    )
                elif ct == "category":
                    guidance.append(
                        f"CLASSIFY RELIABILITY ({avg:.0%}): Classification outputs vary. "
                        f"Ensure the domain's 'categories' and 'criteria' fields use "
                        f"deterministic rules (numeric thresholds, exact string matches) "
                        f"rather than qualitative judgments."
                    )
                elif ct == "recommendation":
                    guidance.append(
                        f"THINK RELIABILITY ({avg:.0%}): The 'think' primitive's recommendation "
                        f"varies across runs. Ensure the domain provides an explicit, step-by-step "
                        f"formula with exact thresholds. Show the arithmetic in 'constraints'."
                    )
                elif "risk_score" in ct:
                    guidance.append(
                        f"RISK SCORE VARIANCE: Risk scores vary across runs. "
                        f"Ensure the formula is fully specified with explicit point values "
                        f"and cumulative logic. Use 'constraints' to require showing arithmetic."
                    )

        if flaky:
            guidance.append(
                f"FLAKY CASES ({len(flaky)}): These cases produce different results across "
                f"runs: {list(flaky.keys())}. This indicates the prompt or domain rules "
                f"are ambiguous enough for the LLM to interpret them differently each time. "
                f"Review the rules for these cases and make them more explicit."
            )

    if brittleness_results:
        failed = [r for r in brittleness_results if not r["perturbation_passed"]]
        boundary_failures = [r for r in failed if "boundary" in r["perturbation"]]
        flag_failures = [r for r in failed if "flag" in r["perturbation"]]

        if boundary_failures:
            guidance.append(
                "BOUNDARY SENSITIVITY: The LLM fails on exact boundary values "
                "(e.g., $500,000 exactly). For production domains, either: "
                "(a) use exclusive boundaries with clear language ('strictly less than'), "
                "(b) add explicit boundary examples in the rules, or "
                "(c) implement boundary checks in code rather than via LLM."
            )

        if flag_failures:
            guidance.append(
                "FLAG LEAKAGE: The LLM uses non-eligibility data (flags, risk factors) "
                "to influence eligibility decisions. Ensure the verify prompt's "
                "'additional_instructions' explicitly lists fields to ignore, or use "
                "the 'subject' param to limit what data reaches the LLM."
            )

    if not guidance:
        guidance.append(
            "ALL CLEAR: No systematic issues detected. The prompt/domain/workflow "
            "design appears sound for the current model and test suite."
        )

    return guidance


# ═════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cognitive Core — Soundness Evaluation")
    parser.add_argument("--mode", choices=["consistency", "brittleness", "full"],
                        default="full", help="Evaluation mode")
    parser.add_argument("--runs", type=int, default=3, help="Runs per case for consistency")
    parser.add_argument("--case", help="Single case to evaluate")
    parser.add_argument("--workflow", help="Filter to a workflow (claim_intake, etc.)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--project-root", default=BASE)
    args = parser.parse_args()

    # Validate environment
    provider = os.environ.get("LLM_PROVIDER", "")
    if not provider:
        print("Error: LLM_PROVIDER not set")
        sys.exit(1)

    # Determine cases
    if args.case:
        case_names = [args.case]
    elif args.workflow:
        case_names = [n for n, d in CASES.items() if d["workflow"] == args.workflow]
    else:
        case_names = list(CASES.keys())

    consistency_results = None
    brittleness_results = None

    if args.mode in ("consistency", "full"):
        consistency_results = eval_consistency(
            case_names, args.runs, args.project_root, args.verbose)

    if args.mode in ("brittleness", "full"):
        brittleness_results = eval_brittleness(args.project_root, args.verbose)

    # Write report
    report = write_report(consistency_results, brittleness_results)

    # Print guidance
    print(f"\n{'═'*70}")
    print("DESIGN GUIDANCE")
    print(f"{'═'*70}")
    for i, g in enumerate(report.get("guidance", []), 1):
        print(f"\n  {i}. {g}")
    print(f"\n{'═'*70}")


if __name__ == "__main__":
    main()
