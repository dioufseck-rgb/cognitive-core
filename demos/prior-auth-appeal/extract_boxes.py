#!/usr/bin/env python3
"""
extract_boxes.py — regenerate manuscript Box 1 and Box 3 from the round5 evaluation runs.

Usage (from demos/prior-auth-appeal, against your LOCAL appeal_cc.db):
    python extract_boxes.py output/round4_replication_1 ... output/round4_replication_5

Joins each replication's B002/cc record (results.json timestamp/elapsed) to its
instance in appeal_cc.db, prints a summary of all five B002 governed runs
(tier + coherence flags), then emits Box 1 (govern step entry) and Box 3
(per-step epistemic state) for the SELECTED run.

Selection rule (pre-committed): the earliest replication whose B002 run gated AND
carries an UNRESOLVED_EVIDENCE_GAPS flag (the condition Box 3 illustrates);
if none carries it, replication 1. The choice is printed so it is documented.
"""
import sys, json, sqlite3, datetime
from pathlib import Path

DB = "appeal_cc.db"

def instances_for_case(c, case_id):
    out = []
    for iid, tier, created, updated, meta in c.execute(
            "select instance_id, governance_tier, created_at, updated_at, case_meta from instances"):
        try: m = json.loads(meta or "{}")
        except Exception: m = {}
        if m.get("case_id") == case_id:
            out.append(dict(iid=iid, tier=tier, created=created, updated=updated))
    return out

def steps(c, iid):
    rows = c.execute("select id, details, entry_hash from action_ledger "
                     "where instance_id=? and action_type='step_completed' order by id", (iid,)).fetchall()
    return [(r[0], json.loads(r[1]), r[2]) for r in rows]

def flags_of(ep):
    fl = (ep or {}).get("coherence_flags") or []
    return [f.get("flag", f) if isinstance(f, dict) else f for f in fl]

def main(run_dirs):
    c = sqlite3.connect(DB)
    recs = []
    for i, rd in enumerate(run_dirs, 1):
        for r in json.load(open(Path(rd) / "results.json")):
            if r["case_id"] == "PA-2024-B002" and r["system"] == "cc" and not r.get("error"):
                recs.append((i, r))
    inst = instances_for_case(c, "PA-2024-B002")
    print(f"B002 instances in DB: {len(inst)}; replication records: {len(recs)}\n")
    joined = []
    for repn, r in recs:
        ts = datetime.datetime.fromisoformat(r["timestamp"]).replace(tzinfo=datetime.timezone.utc).timestamp()
        start = ts - float(r.get("elapsed_s") or 0)
        best = min(inst, key=lambda x: abs(x["created"] - start))
        joined.append((repn, r, best, abs(best["created"] - start)))
    print("rep | tier(results) | instance | join-skew(s) | flags across steps")
    chosen = None
    for repn, r, ins, skew in sorted(joined):
        fl = sorted({f for _, d, _ in steps(c, ins["iid"]) for f in flags_of(d.get("epistemic"))})
        print(f"  {repn} | {r['tier']:<10} | {ins['iid']} | {skew:6.1f} | {fl}")
        if chosen is None and r["tier"] in ("gate", "hold") and any("EVIDENCE" in f for f in fl):
            chosen = (repn, ins)
    if chosen is None:
        chosen = (sorted(joined)[0][0], sorted(joined)[0][2])
    repn, ins = chosen
    print(f"\nSELECTED: replication {repn}, instance {ins['iid']} (rule: earliest gated run carrying an evidence-gaps flag)\n")

    st = steps(c, ins["iid"])
    print("=" * 70, "\nBOX 1 — govern step entry (verbatim; trim with … for layout)\n" + "=" * 70)
    for seq, d, h in st:
        if d.get("primitive") == "govern" or "govern" in (d.get("step_name") or ""):
            print(json.dumps({"seq": seq, "step": d.get("step_name"),
                              "output": d.get("output"), "epistemic": d.get("epistemic"),
                              "elapsed_ms": d.get("elapsed_ms"),
                              "entry_hash": (h or "")[:16] + "…"}, indent=1)[:4000])
    print("\n" + "=" * 70, "\nBOX 3 — per-step epistemic state\n" + "=" * 70)
    for seq, d, h in st:
        ep = d.get("epistemic") or {}
        sig = {k: v for k, v in ep.items() if isinstance(v, (int, float))}
        fl = flags_of(ep)
        keep = ["evidence_completeness","rule_coverage","citation_rate","reasoning_quality",
                "outcome_certainty","alternative_separation","overall"]
        vals = ", ".join(f"{k} {ep[k]:.2f}" for k in keep if isinstance(ep.get(k),(int,float)))
        print(f"{d.get('step_name')} [{d.get('primitive')}]: {vals} · warranted={ep.get('warranted')}")
        if fl: print(f"    coherence_flags: {json.dumps(ep.get('coherence_flags'))[:500]}")
        gaps = ep.get("inherited_gaps") or []
        if gaps: print(f"    open gaps: {json.dumps(gaps)[:500]}")

if __name__ == "__main__":
    main(sys.argv[1:])
