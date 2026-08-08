#!/usr/bin/env python3
"""
apply_patches.py — applies all locked-config patches for the revision rerun.
Run from demos/prior-auth-appeal:  python apply_patches.py
Idempotent: safe to run twice. Prints OK / ALREADY APPLIED / FAILED per patch.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ok = True

def report(name, status): print(f"[{status:^15}] {name}")

# ── Patch 1: GT extraction fix in run_parallel_benchmark.py ──────────────────
p = HERE / "run_parallel_benchmark.py"
src = p.read_text(encoding="utf-8")
NEW_FN = '''def extract_gt_disposition(answer: str) -> str:
    upper = answer.upper().lstrip()
    if upper.startswith("GATE"):  # "GATE / REMAND - ..." -> disposition after prefix
        upper = upper.split("/", 1)[-1].lstrip() if "/" in upper else upper[4:].lstrip()
    hits = [(upper.find(kw), kw) for kw in ("PARTIAL", "REMAND", "OVERTURN", "UPHOLD") if upper.find(kw) >= 0]
    return min(hits)[1] if hits else "UNKNOWN"
'''
if "GATE GT cases resolve to UPHOLD" in src:
    pat = re.compile(r"def extract_gt_disposition\(answer: str\) -> str:.*?return \"UNKNOWN\"\n", re.S)
    if pat.search(src):
        p.write_text(pat.sub(NEW_FN, src, count=1), encoding="utf-8")
        report("1. GT extraction fix (run_parallel_benchmark.py)", "OK")
    else:
        report("1. GT extraction fix — function shape unexpected, EDIT MANUALLY (plan §3.1)", "FAILED"); ok = False
elif "disposition after prefix" in src:
    report("1. GT extraction fix", "ALREADY APPLIED")
else:
    report("1. GT extraction fix — marker not found, verify manually", "FAILED"); ok = False

# self-test the new logic
import importlib.util
tests = {"GATE / REMAND — text": "REMAND", "GATE — no keyword": "UNKNOWN",
         "PARTIAL — text": "PARTIAL", "OVERTURN — text": "OVERTURN"}
ns = {}
exec(NEW_FN, ns)
for inp, want in tests.items():
    got = ns["extract_gt_disposition"](inp)
    if got != want:
        report(f"1b. GT self-test '{inp}' -> {got} (want {want})", "FAILED"); ok = False
report("1b. GT extraction self-test", "OK" if ok else "SEE ABOVE")

# ── Patch 2: G003 label clarification ────────────────────────────────────────
g = HERE / "cases" / "pa_2024_g003.json"
d = json.loads(g.read_text(encoding="utf-8"))
ra = d["ground_truth_complexity"]["right_answer"]
if ra.upper().lstrip().startswith("GATE /"):
    report("2. G003 label clarification", "ALREADY APPLIED")
elif ra.upper().lstrip().startswith("GATE"):
    stripped = ra.lstrip()
    # replace leading "GATE" (+ optional dash) with "GATE / UPHOLD —"
    rest = re.sub(r"^GATE\s*[—\-–]*\s*", "", stripped)
    d["ground_truth_complexity"]["right_answer"] = "GATE / UPHOLD — " + rest
    g.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    report("2. G003 label clarification (GATE / UPHOLD)", "OK")
else:
    report("2. G003 right_answer does not start with GATE — inspect manually", "FAILED"); ok = False

# ── Patch 3: pin all Gemini aliases in llm_config.yaml to gemini-2.5-flash ──
y = REPO / "llm_config.yaml"
ytxt = y.read_text(encoding="utf-8")
pinned = re.sub(r"gemini-3[\w\.\-]*", "gemini-2.5-flash", ytxt)
if pinned != ytxt:
    y.write_text(pinned, encoding="utf-8")
    n = len(re.findall(r"gemini-3[\w\.\-]*", ytxt))
    report(f"3. Model pin: {n} gemini-3* reference(s) -> gemini-2.5-flash", "OK")
elif "gemini-2.5-flash" in ytxt:
    report("3. Model pin (gemini-2.5-flash)", "ALREADY APPLIED")
else:
    report("3. Model pin — no gemini reference found, inspect llm_config.yaml", "FAILED"); ok = False

# ── Patch 4: remove stale 'not yet wired' docstring in epistemic.py ─────────
e = REPO / "cognitive_core" / "engine" / "epistemic.py"
etxt = e.read_text(encoding="utf-8")
if "not yet wired" in etxt or "placeholders here" in etxt:
    new = re.sub(r"[^\n]*not yet wired[^\n]*\n", "", etxt)
    new = re.sub(r"[^\n]*placeholders here[^\n]*\n", "", new)
    e.write_text(new, encoding="utf-8")
    report("4. Stale docstring removed (epistemic.py)", "OK")
else:
    report("4. Stale docstring (epistemic.py)", "ALREADY APPLIED")

# ── Summary ──────────────────────────────────────────────────────────────────
print()
if ok:
    print("ALL PATCHES APPLIED. Next: git add -A ; git commit -m \"lock eval config for revision rerun\"")
else:
    print("ONE OR MORE PATCHES FAILED — fix before running anything. Do not proceed to paid stages.")
    sys.exit(1)
