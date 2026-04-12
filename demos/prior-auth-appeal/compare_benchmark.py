"""
Prior Authorization Appeal Review — Pairwise Comparison & Systematic Analysis

Reads output/benchmark/results.json produced by run_benchmark.py and generates:

  output/benchmark/
    comparison_report.md        — human-readable pairwise analysis
    systematic_differences.json — machine-readable patterns
    accuracy_matrix.txt         — disposition × ground-truth matrix

Usage:
    python demos/prior-auth-appeal/compare_benchmark.py

    # With full determination text (loads from cc/ and react/ subdirs)
    python demos/prior-auth-appeal/compare_benchmark.py --full-text
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DEMO_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_DIR / "output" / "benchmark"
CASES_DIR  = DEMO_DIR / "cases"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results() -> list[dict]:
    path = OUTPUT_DIR / "results.json"
    if not path.exists():
        print(f"No results found at {path}")
        print("Run run_benchmark.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_determination(case_id: str, system: str) -> str:
    path = OUTPUT_DIR / system / f"{case_id}.txt"
    if path.exists():
        return path.read_text()
    return ""


def load_case_meta(case_id: str) -> dict:
    # Map case_id back to filename: PA-2024-A001 → pa_2024_a001.json
    parts = case_id.split("-")
    if len(parts) == 3:
        fname = f"pa_{parts[1]}_{parts[2].lower()}.json"
    else:
        fname = case_id.lower().replace("-", "_") + ".json"
    path = CASES_DIR / fname
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Disposition analysis ─────────────────────────────────────────────────────

# Ground truth extraction: parse right_answer to canonical disposition
DISPOSITION_KEYWORDS = ["OVERTURN", "UPHOLD", "PARTIAL", "REMAND", "GATE"]

def parse_ground_truth_disposition(gt: str) -> str:
    """Extract the primary disposition keyword from a ground truth string."""
    upper = gt.upper()
    # PARTIAL before OVERTURN — "PARTIAL" cases contain OVERTURN for sub-levels
    for kw in ("PARTIAL", "REMAND", "GATE", "UPHOLD", "OVERTURN"):
        if kw in upper[:60]:  # keyword in first 60 chars = primary
            return kw
    for kw in ("PARTIAL", "REMAND", "GATE", "UPHOLD", "OVERTURN"):
        if kw in upper:
            return kw
    return "UNKNOWN"


def is_correct(predicted: str, ground_truth_str: str) -> bool | None:
    """
    Returns True if predicted matches ground truth, False if it doesn't,
    None if ground truth is ambiguous (e.g., GATE cases where GATE is
    a routing outcome not a disposition).
    """
    gt = parse_ground_truth_disposition(ground_truth_str)
    if gt == "UNKNOWN":
        return None
    # GATE ground truth — correct only if OVERTURN or GATE/REMAND (borderline)
    if gt == "GATE":
        return predicted in ("GATE", "REMAND", "OVERTURN")
    return predicted == gt


# ── Pair construction ────────────────────────────────────────────────────────

def build_pairs(results: list[dict]) -> list[dict]:
    """
    Join CC and ReAct results by case_id into paired records.
    Returns list of dicts with cc_result and react_result.
    """
    by_case: dict[str, dict] = defaultdict(dict)
    for r in results:
        by_case[r["case_id"]][r["system"]] = r

    pairs = []
    for case_id, systems in sorted(by_case.items()):
        pair = {
            "case_id":   case_id,
            "case_type": case_id.split("-")[2][0] if "-" in case_id else "?",
            "cc":        systems.get("cc"),
            "react":     systems.get("react"),
        }
        # Add ground truth from whichever record is available
        for sys_key in ("cc", "react"):
            if pair[sys_key]:
                pair["ground_truth"]          = pair[sys_key].get("ground_truth_answer", "")
                pair["obvious_reading"]        = pair[sys_key].get("obvious_reading", "")
                pair["genuine_tension"]        = pair[sys_key].get("genuine_tension", "")
                pair["ground_truth_canonical"] = parse_ground_truth_disposition(pair["ground_truth"])
                break
        pairs.append(pair)
    return pairs


# ── Systematic pattern detection ─────────────────────────────────────────────

def classify_divergence(pair: dict) -> str | None:
    """
    Classify the type of divergence between CC and ReAct.
    Returns a pattern label or None if they agree.
    """
    cc    = pair.get("cc")
    react = pair.get("react")
    if not cc or not react:
        return "INCOMPLETE"

    cc_d    = cc["disposition"]
    react_d = react["disposition"]

    if cc_d == react_d:
        return None  # Agreement

    gt = pair["ground_truth_canonical"]

    # Correctness
    cc_correct    = is_correct(cc_d, pair["ground_truth"])
    react_correct = is_correct(react_d, pair["ground_truth"])

    if cc_correct and not react_correct:
        return f"CC_CORRECT__REACT_WRONG ({react_d}→{cc_d})"
    if react_correct and not cc_correct:
        return f"REACT_CORRECT__CC_WRONG ({cc_d}→{react_d})"
    if cc_correct is None or react_correct is None:
        return f"AMBIGUOUS_GROUND_TRUTH ({cc_d} vs {react_d})"

    return f"BOTH_WRONG ({cc_d} vs {react_d}, gt={gt})"


def detect_anchor_failure(pair: dict, full_text: bool = False) -> dict:
    """
    Detect specific anchor failure patterns in ReAct output.
    These are the patterns the 20-case battery was designed to surface.
    """
    signals = {}
    react = pair.get("react")
    if not react:
        return signals

    det_text = ""
    if full_text:
        det_text = load_determination(pair["case_id"], "react").upper()
    else:
        # Work from ground truth tension description
        tension = pair.get("genuine_tension", "").upper()
        gt      = pair.get("ground_truth", "").upper()

    case_type = pair["case_type"]

    # Type A: did ReAct mention the regulatory override?
    if case_type == "A":
        if det_text:
            has_reg = "10169.5" in det_text or "APL 22-014" in det_text
            signals["regulatory_override_cited"] = has_reg
            signals["anchor_failure_likely"] = not has_reg and react["disposition"] == "UPHOLD"

    # Type B multi-level: did ReAct give a PARTIAL?
    if case_type == "B":
        signals["gave_partial"] = react["disposition"] == "PARTIAL"
        if pair["ground_truth_canonical"] == "PARTIAL":
            signals["anchor_failure_likely"] = react["disposition"] != "PARTIAL"

    # Type C: did ReAct route to GATE/REMAND on conflicting radiology?
    if case_type == "C":
        signals["routed_to_gate"] = react["disposition"] in ("GATE", "REMAND")
        if "CONFLICTING" in pair.get("genuine_tension", "").upper():
            signals["anchor_failure_likely"] = react["disposition"] not in ("GATE", "REMAND")

    # Type D: did ReAct catch the procedural defect?
    if case_type == "D":
        if det_text:
            has_procedural = "1374.31" in det_text or "IMR NOTICE" in det_text
            signals["procedural_defect_cited"] = has_procedural
        signals["anchor_failure_likely"] = (
            react["disposition"] == "UPHOLD"
            and pair["ground_truth_canonical"] == "REMAND"
        )

    # Type E: did ReAct give a PARTIAL for multi-level?
    if case_type == "E":
        signals["gave_partial"] = react["disposition"] == "PARTIAL"
        if pair["ground_truth_canonical"] == "PARTIAL":
            signals["anchor_failure_likely"] = react["disposition"] != "PARTIAL"

    # Type F: did ReAct get distracted by documentation dispute?
    if case_type == "F":
        if det_text:
            undisputed_cited = "UNDISPUTED" in det_text or "RIVERSIDE" in det_text
            signals["found_controlling_record"] = undisputed_cited
        signals["anchor_failure_likely"] = react["disposition"] != "OVERTURN"

    return signals


# ── Report generation ─────────────────────────────────────────────────────────

def generate_markdown_report(pairs: list[dict], full_text: bool = False) -> str:
    lines = []
    lines.append("# Prior Authorization Appeal Review — Benchmark Comparison Report\n")
    lines.append(f"Generated from {len(pairs)} cases (Cognitive Core vs. ReAct baseline)\n")

    # ── Summary table
    lines.append("## Summary\n")
    lines.append("| Case ID | Type | GT | CC | ReAct | Agreement | CC✓ | ReAct✓ |")
    lines.append("|---------|------|----|----|-------|-----------|-----|--------|")

    cc_correct_total    = 0
    react_correct_total = 0
    agreement_total     = 0
    divergences         = []
    evaluable           = 0

    for p in pairs:
        cc    = p.get("cc")
        react = p.get("react")
        gt    = p["ground_truth_canonical"]

        cc_d    = cc["disposition"]    if cc    else "—"
        react_d = react["disposition"] if react else "—"
        agree   = "✓" if cc_d == react_d else "✗"

        cc_ok    = is_correct(cc_d,    p["ground_truth"]) if cc    else None
        react_ok = is_correct(react_d, p["ground_truth"]) if react else None

        cc_sym    = "✓" if cc_ok    else ("?" if cc_ok    is None else "✗")
        react_sym = "✓" if react_ok else ("?" if react_ok is None else "✗")

        if cc_ok is not None:
            evaluable += 1
            if cc_ok:    cc_correct_total    += 1
            if react_ok: react_correct_total += 1
            if cc_d == react_d: agreement_total += 1

        lines.append(
            f"| {p['case_id']} | {p['case_type']} | {gt} | {cc_d} | {react_d} "
            f"| {agree} | {cc_sym} | {react_sym} |"
        )

        div = classify_divergence(p)
        if div:
            divergences.append((p["case_id"], p["case_type"], div))

    lines.append(f"\n**Evaluable cases:** {evaluable}  ")
    lines.append(f"**CC accuracy:** {cc_correct_total}/{evaluable}  ")
    lines.append(f"**ReAct accuracy:** {react_correct_total}/{evaluable}  ")
    lines.append(f"**Agreement rate:** {agreement_total}/{evaluable}\n")

    # ── Divergences
    lines.append("## Disposition Divergences\n")
    if not divergences:
        lines.append("No divergences detected.\n")
    else:
        lines.append(f"{len(divergences)} divergence(s) found:\n")
        lines.append("| Case ID | Type | Pattern |")
        lines.append("|---------|------|---------|")
        for case_id, case_type, pattern in divergences:
            lines.append(f"| {case_id} | {case_type} | {pattern} |")
        lines.append("")

    # ── By type analysis
    lines.append("## Analysis by Case Type\n")
    type_groups: dict[str, list] = defaultdict(list)
    for p in pairs:
        type_groups[p["case_type"]].append(p)

    type_descriptions = {
        "A": "Regulatory override applies, plan criteria not met — expected anchor failure: UPHOLD",
        "B": "Plan criteria met, clinical standard conflict — expected anchor failure: binary OVERTURN vs PARTIAL",
        "C": "Diagnosis boundary — expected anchor failure: binary outcome vs GATE/REMAND",
        "D": "Procedural defect — expected anchor failure: UPHOLD vs REMAND",
        "E": "Multi-level partial approval — expected anchor failure: binary vs PARTIAL",
        "F": "PT documentation dispute — expected anchor failure: distraction vs controlling record",
    }

    for t in sorted(type_groups.keys()):
        cases = type_groups[t]
        lines.append(f"### Type {t}: {type_descriptions.get(t, '')}\n")

        cc_c = sum(1 for p in cases if is_correct(p.get("cc", {}).get("disposition", ""), p["ground_truth"]) is True)
        rc_c = sum(1 for p in cases if is_correct(p.get("react", {}).get("disposition", ""), p["ground_truth"]) is True)
        n    = sum(1 for p in cases if is_correct(p.get("cc", {}).get("disposition", ""), p["ground_truth"]) is not None)

        lines.append(f"CC correct: {cc_c}/{n} | ReAct correct: {rc_c}/{n}\n")

        for p in cases:
            cc    = p.get("cc")
            react = p.get("react")
            cc_d    = cc["disposition"]    if cc    else "—"
            react_d = react["disposition"] if react else "—"
            div     = classify_divergence(p)
            anchor  = detect_anchor_failure(p, full_text=full_text)

            lines.append(f"**{p['case_id']}**  ")
            lines.append(f"GT: `{p['ground_truth_canonical']}`  CC: `{cc_d}`  ReAct: `{react_d}`  ")
            if div:
                lines.append(f"⚠ Divergence: {div}  ")
            if anchor.get("anchor_failure_likely"):
                lines.append(f"⚡ Anchor failure likely in ReAct  ")
            lines.append(f"*{p.get('genuine_tension', '')[:200]}*\n")

    # ── Governance analysis (CC only)
    lines.append("## Governance Tier Analysis (CC)\n")
    lines.append("| Case ID | Type | Disposition | Tier | Steps | Elapsed |")
    lines.append("|---------|------|-------------|------|-------|---------|")
    for p in pairs:
        cc = p.get("cc")
        if not cc:
            continue
        lines.append(
            f"| {p['case_id']} | {p['case_type']} | {cc['disposition']} "
            f"| {cc.get('tier','?')} | {cc.get('steps','?')} | {cc.get('elapsed_s','?')}s |"
        )
    lines.append("")

    # ── Systematic patterns
    lines.append("## Systematic Patterns\n")

    # Pattern 1: Cases where CC gives GATE but ReAct gives binary
    gate_cases = [
        p for p in pairs
        if p.get("cc", {}).get("tier") in ("gate", "GATE")
        and p.get("react", {}).get("disposition") not in ("GATE", "REMAND")
    ]
    if gate_cases:
        lines.append(f"### Pattern 1: CC routes GATE, ReAct gives binary answer ({len(gate_cases)} cases)\n")
        for p in gate_cases:
            lines.append(f"- {p['case_id']}: CC=GATE/{p.get('cc',{}).get('disposition','?')}, ReAct={p.get('react',{}).get('disposition','?')}")
        lines.append("")

    # Pattern 2: Cases where ReAct gives OVERTURN but correct is UPHOLD or PARTIAL
    react_overturn_wrong = [
        p for p in pairs
        if p.get("react", {}).get("disposition") == "OVERTURN"
        and p["ground_truth_canonical"] in ("UPHOLD", "PARTIAL", "REMAND")
    ]
    if react_overturn_wrong:
        lines.append(f"### Pattern 2: ReAct OVERTURNs when correct answer is not OVERTURN ({len(react_overturn_wrong)} cases)\n")
        for p in react_overturn_wrong:
            lines.append(f"- {p['case_id']} (type {p['case_type']}): GT={p['ground_truth_canonical']}, ReAct=OVERTURN")
            lines.append(f"  *{p.get('obvious_reading','')[:120]}*")
        lines.append("")

    # Pattern 3: PARTIAL misses
    partial_cases = [p for p in pairs if p["ground_truth_canonical"] == "PARTIAL"]
    if partial_cases:
        lines.append(f"### Pattern 3: PARTIAL disposition accuracy ({len(partial_cases)} cases with PARTIAL ground truth)\n")
        for p in partial_cases:
            cc_d    = p.get("cc",    {}).get("disposition", "—")
            react_d = p.get("react", {}).get("disposition", "—")
            lines.append(
                f"- {p['case_id']}: CC={cc_d} {'✓' if cc_d=='PARTIAL' else '✗'}  "
                f"ReAct={react_d} {'✓' if react_d=='PARTIAL' else '✗'}"
            )
        lines.append("")

    # Pattern 4: REMAND misses
    remand_cases = [p for p in pairs if p["ground_truth_canonical"] == "REMAND"]
    if remand_cases:
        lines.append(f"### Pattern 4: REMAND disposition accuracy ({len(remand_cases)} cases with REMAND ground truth)\n")
        for p in remand_cases:
            cc_d    = p.get("cc",    {}).get("disposition", "—")
            react_d = p.get("react", {}).get("disposition", "—")
            lines.append(
                f"- {p['case_id']}: CC={cc_d} {'✓' if cc_d=='REMAND' else '✗'}  "
                f"ReAct={react_d} {'✓' if react_d=='REMAND' else '✗'}"
            )
        lines.append("")

    # ── Timing
    lines.append("## Timing Comparison\n")
    cc_times    = [p.get("cc",    {}).get("elapsed_s", 0) for p in pairs if p.get("cc")]
    react_times = [p.get("react", {}).get("elapsed_s", 0) for p in pairs if p.get("react")]
    if cc_times:
        lines.append(f"CC avg: {sum(cc_times)/len(cc_times):.0f}s  "
                     f"min: {min(cc_times):.0f}s  max: {max(cc_times):.0f}s")
    if react_times:
        lines.append(f"ReAct avg: {sum(react_times)/len(react_times):.0f}s  "
                     f"min: {min(react_times):.0f}s  max: {max(react_times):.0f}s")
    if cc_times and react_times:
        ratio = sum(cc_times) / max(sum(react_times), 1)
        lines.append(f"CC/ReAct time ratio: {ratio:.1f}×\n")

    return "\n".join(lines)


def generate_systematic_json(pairs: list[dict]) -> dict:
    """Machine-readable systematic differences for programmatic analysis."""
    result = {
        "total_pairs":       len(pairs),
        "agreement":         0,
        "divergences":       [],
        "cc_accuracy":       {"correct": 0, "wrong": 0, "ambiguous": 0, "missing": 0},
        "react_accuracy":    {"correct": 0, "wrong": 0, "ambiguous": 0, "missing": 0},
        "by_type":           {},
        "anchor_failures":   [],
        "disposition_matrix": defaultdict(lambda: defaultdict(int)),
        "governance_tiers":  defaultdict(int),
    }

    for p in pairs:
        cc    = p.get("cc")
        react = p.get("react")
        gt    = p["ground_truth_canonical"]
        t     = p["case_type"]

        if t not in result["by_type"]:
            result["by_type"][t] = {
                "count": 0, "cc_correct": 0, "react_correct": 0,
                "agreement": 0, "divergences": []
            }
        result["by_type"][t]["count"] += 1

        cc_d    = cc["disposition"]    if cc    else None
        react_d = react["disposition"] if react else None

        # Agreement
        if cc_d and react_d:
            if cc_d == react_d:
                result["agreement"] += 1
                result["by_type"][t]["agreement"] += 1
            else:
                div = classify_divergence(p)
                result["divergences"].append({
                    "case_id":   p["case_id"],
                    "case_type": t,
                    "cc":        cc_d,
                    "react":     react_d,
                    "gt":        gt,
                    "pattern":   div,
                })
                result["by_type"][t]["divergences"].append(p["case_id"])

        # Accuracy
        for sys_key, acc_key, d in [("cc", "cc_accuracy", cc_d), ("react", "react_accuracy", react_d)]:
            if d is None:
                result[acc_key]["missing"] += 1
            else:
                ok = is_correct(d, p["ground_truth"])
                if ok is True:
                    result[acc_key]["correct"] += 1
                    if sys_key == "cc":   result["by_type"][t]["cc_correct"]    += 1
                    if sys_key == "react": result["by_type"][t]["react_correct"] += 1
                elif ok is False:
                    result[acc_key]["wrong"] += 1
                else:
                    result[acc_key]["ambiguous"] += 1

        # Disposition matrix: react_d → cc_d count
        if cc_d and react_d:
            result["disposition_matrix"][react_d][cc_d] += 1

        # Governance tiers
        if cc:
            tier = cc.get("tier", "unknown")
            result["governance_tiers"][str(tier)] += 1

        # Anchor failure detection
        anchor = detect_anchor_failure(p)
        if anchor.get("anchor_failure_likely"):
            result["anchor_failures"].append({
                "case_id":   p["case_id"],
                "case_type": t,
                "react_d":   react_d,
                "gt":        gt,
                "signals":   anchor,
            })

    # Convert defaultdicts for JSON serialisation
    result["disposition_matrix"] = {
        k: dict(v) for k, v in result["disposition_matrix"].items()
    }
    result["governance_tiers"] = dict(result["governance_tiers"])

    return result


def generate_accuracy_matrix(pairs: list[dict]) -> str:
    """Text matrix: rows = ground truth, cols = system disposition."""
    dispositions = ["OVERTURN", "UPHOLD", "PARTIAL", "REMAND", "GATE", "UNKNOWN", "ERROR"]

    for system in ("cc", "react"):
        lines = [f"\n{system.upper()} — Predicted vs Ground Truth\n"]
        lines.append(f"{'GT \\ Pred':<12}" + "".join(f"{d:<10}" for d in dispositions))
        lines.append("─" * (12 + 10 * len(dispositions)))

        gt_groups: dict[str, list] = defaultdict(list)
        for p in pairs:
            r = p.get(system)
            if r:
                gt = p["ground_truth_canonical"]
                gt_groups[gt].append(r["disposition"])

        for gt in dispositions:
            if gt not in gt_groups:
                continue
            preds = gt_groups[gt]
            row = f"{gt:<12}"
            for d in dispositions:
                count = preds.count(d)
                row += f"{count if count else '.':<10}"
            lines.append(row)

        yield "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark pairwise comparison")
    parser.add_argument("--full-text", action="store_true",
                        help="Load full determination text for deeper analysis")
    args = parser.parse_args()

    print("Loading results...")
    results = load_results()
    pairs   = build_pairs(results)

    print(f"  {len(pairs)} case pairs loaded")
    print(f"  CC results:    {sum(1 for p in pairs if p.get('cc'))}")
    print(f"  ReAct results: {sum(1 for p in pairs if p.get('react'))}\n")

    # Markdown report
    report = generate_markdown_report(pairs, full_text=args.full_text)
    report_path = OUTPUT_DIR / "comparison_report.md"
    report_path.write_text(report)
    print(f"Report: {report_path}")

    # Systematic JSON
    systematic = generate_systematic_json(pairs)
    sys_path = OUTPUT_DIR / "systematic_differences.json"
    with open(sys_path, "w") as f:
        json.dump(systematic, f, indent=2)
    print(f"Systematic: {sys_path}")

    # Accuracy matrices
    matrix_path = OUTPUT_DIR / "accuracy_matrix.txt"
    with open(matrix_path, "w") as f:
        for matrix_text in generate_accuracy_matrix(pairs):
            f.write(matrix_text + "\n\n")
    print(f"Matrix: {matrix_path}")

    # Quick console summary
    print(f"\n{'─' * 60}")
    print(f"  Cases: {len(pairs)}")
    ev = systematic["cc_accuracy"]["correct"] + systematic["cc_accuracy"]["wrong"]
    if ev:
        print(f"  CC accuracy:    {systematic['cc_accuracy']['correct']}/{ev} "
              f"({100*systematic['cc_accuracy']['correct']//ev}%)")
        print(f"  ReAct accuracy: {systematic['react_accuracy']['correct']}/{ev} "
              f"({100*systematic['react_accuracy']['correct']//ev}%)")
    print(f"  Agreement: {systematic['agreement']}/{len(pairs)}")
    print(f"  Divergences: {len(systematic['divergences'])}")
    print(f"  Anchor failures detected: {len(systematic['anchor_failures'])}")
    if systematic["anchor_failures"]:
        for af in systematic["anchor_failures"]:
            print(f"    {af['case_id']} (type {af['case_type']}): "
                  f"ReAct={af['react_d']} GT={af['gt']}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
