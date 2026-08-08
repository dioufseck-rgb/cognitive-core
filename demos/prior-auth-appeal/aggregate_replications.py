#!/usr/bin/env python3
"""
aggregate_replications.py — turn N replication outputs into the revised paper's numbers.

Usage:
    python aggregate_replications.py output/replication_1 output/replication_2 ...

Reads results.json from each replication dir. Re-derives ground truth directly from
cases/*.json (so scoring is correct even if the harness GT-extraction bug wasn't patched).
Emits:
  - Per-case modal table (markdown, drop-in for Table 6)
  - Modal accuracy + per-run range, balanced set and full battery
  - Pooled silent errors + exact one-sided 95% upper bound if zero (rule of three)
  - CC tier distribution across runs
  - C3: baseline accuracy, OVERTURN-battery vs balanced set
  - C4: per-case instability vs CC modal tier
"""
import json, sys, glob
from collections import Counter, defaultdict
from pathlib import Path

DEMO = Path(__file__).resolve().parent
BALANCED = {"PA-2024-A001","PA-2024-B004","PA-2024-G005","PA-2024-C004","PA-2024-G004",
            "PA-2024-D001","PA-2024-D003","PA-2024-B001","PA-2024-E001","PA-2024-C003","PA-2024-G003"}
EXEC_TIERS = {"auto","spot_check","spot check","spotcheck"}

def gt_from_case(path):
    d = json.load(open(path))
    ra = ((d.get("ground_truth_complexity") or {}).get("right_answer") or "").upper().lstrip()
    if ra.startswith("GATE"):                       # "GATE / REMAND — ..." → text after prefix
        ra = ra.split("/",1)[-1].lstrip() if "/" in ra else ra[4:].lstrip()
    hits = [(ra.find(kw), kw) for kw in ("PARTIAL","REMAND","OVERTURN","UPHOLD") if ra.find(kw) >= 0]
    return min(hits)[1] if hits else None           # earliest keyword anywhere; None → excluded (warned)

def main(run_dirs):
    # ground truth, independently derived
    gt = {}
    for f in glob.glob(str(DEMO/"cases"/"*.json")):
        cid = json.load(open(f)).get("case_id") or Path(f).stem.upper().replace("PA_","PA-").replace("_","-")
        g = gt_from_case(f)
        if g: gt[cid] = g
    excluded = sorted(set(json.load(open(f)).get("case_id","?") for f in glob.glob(str(DEMO/"cases"/"*.json"))) - set(gt))
    print(f"# Aggregation over {len(run_dirs)} replications")
    print(f"Labeled cases: {len(gt)}; excluded (no parseable documented label): {excluded or 'none'}")
    if excluded: print("WARNING: excluded cases above — if G003 appears here, amend its artifact to 'GATE / UPHOLD — ...' per the locked plan and re-run this script.")
    print()

    # collect: results[case][system] = list over runs of (disposition, tier)
    R = len(run_dirs)
    res = defaultdict(lambda: defaultdict(list))
    for rd in run_dirs:
        for rec in json.load(open(Path(rd)/"results.json")):
            cid, sysname = rec["case_id"], rec["system"]
            if cid in gt:
                res[cid][sysname].append(((rec.get("disposition") or "UNKNOWN").upper(),
                                          (rec.get("tier") or "").lower().replace(" ","_")))

    def modal(lst):
        c = Counter(x[0] for x in lst)
        m, n = c.most_common(1)[0]
        ties = [k for k, v in c.items() if v == n]
        return (m if len(ties) == 1 else "TIE:" + "/".join(sorted(ties))), n, len(lst)

    def analyze(cases, label):
        print(f"## {label} ({len(cases)} cases)")
        acc = {s: 0 for s in ("cc","react","plansolve")}
        per_run_acc = {s: [0]*R for s in acc}
        pooled_err = Counter(); pooled_silent = Counter()
        cases = [c for c in cases if res.get(c)]   # drop cases absent from all replications (warned below)
        rows = []
        for cid in sorted(cases):
            row = {"case": cid, "gt": gt[cid]}
            for s in acc:
                lst = res[cid].get(s, [])
                if not lst: row[s] = ("MISSING", 0, 0); print(f"WARNING: {cid} missing for {s}"); continue
                m, n, tot = modal(lst)
                ok = (m == gt[cid])
                acc[s] += ok
                for i,(d,_) in enumerate(lst):
                    per_run_acc[s][i] += (d == gt[cid])
                errs = [(d,t) for d,t in lst if d != gt[cid]]
                pooled_err[s] += len(errs)
                pooled_silent[s] += sum(1 for d,t in errs if (s!="cc") or (t in EXEC_TIERS))
                row[s] = (m, n, tot)
                if s=="cc": row["cc_tier"] = Counter(t for _,t in lst).most_common(1)[0][0]
            rows.append(row)
        # markdown table
        print("| Case | GT | CC (agree) | CC modal tier | ReAct | P&S |")
        print("|---|---|---|---|---|---|")
        for r in rows:
            f = lambda s: f"{r[s][0]}{' ✓' if r[s][0]==r['gt'] else ' ✗'} ({r[s][1]}/{r[s][2]})"
            print(f"| {r['case']} | {r['gt']} | {f('cc')} | {r.get('cc_tier','—')} | {f('react')} | {f('plansolve')} |")
        for s in acc:
            lo, hi = min(per_run_acc[s]), max(per_run_acc[s])
            print(f"**{s}**: modal accuracy {acc[s]}/{len(cases)} ({100*acc[s]//len(cases)}%), per-run range {lo}–{hi}; "
                  f"pooled errors {pooled_err[s]}, silent {pooled_silent[s]}"
                  + (f" (95% one-sided UB ≈ {min(1.0, 3/pooled_err[s]):.0%})" if s == "cc" and pooled_err[s] >= 3 and not pooled_silent[s]
                     else " (too few CC error events for a meaningful bound)" if s == "cc" and pooled_err[s] < 3 and not pooled_silent[s] else ""))
        print()
        return rows

    balanced_rows = analyze([c for c in gt if c in BALANCED], "Balanced 11-case set (headline)")
    battery_rows  = analyze(list(gt), "Full labeled battery (robustness)")

    # C3: approval prior made visible
    ov = [c for c in gt if gt[c]=="OVERTURN"]; nonov_bal = [c for c in gt if c in BALANCED and gt[c]!="OVERTURN"]
    print("## C3 — approval prior contrast (baselines)")
    for s in ("react","plansolve"):
        a = sum(modal(res[c][s])[0]==gt[c] for c in ov if res[c].get(s))
        b = sum(modal(res[c][s])[0]==gt[c] for c in nonov_bal if res[c].get(s))
        print(f"{s}: OVERTURN cases {a}/{len(ov)} vs balanced non-OVERTURN {b}/{len(nonov_bal)}")
    # C4: instability vs tier
    print("\n## C4 — instability vs CC tier (balanced set)")
    for r in balanced_rows:
        agree = r["cc"][1]; tot = r["cc"][2]
        if tot and agree < tot:
            print(f"{r['case']}: agreement {agree}/{tot}, modal tier {r.get('cc_tier')}")
    print("\n## CC tier distribution (all runs, balanced set)")
    tiers = Counter(t for c in BALANCED if c in gt for _,t in res[c].get("cc",[]))
    print(dict(tiers))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: aggregate_replications.py output/replication_1 [output/replication_2 ...]")
    main(sys.argv[1:])
