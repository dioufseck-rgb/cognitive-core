import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ok = True
def report(name, status): print(f"[{status:^15}] {name}")

# A. Token budgets
y = REPO / "llm_config.yaml"
t = y.read_text(encoding="utf-8")
t2 = re.sub(r"(investigate:\s*)8192", r"\g<1>16384", t)
t2 = re.sub(r"(classify:\s*)4096",   r"\g<1>8192",  t2)
if t2 != t:
    y.write_text(t2, encoding="utf-8"); report("A. Token budgets (investigate 16384, classify 8192)", "OK")
elif "16384" in t:
    report("A. Token budgets", "ALREADY APPLIED")
else:
    report("A. Token budgets - pattern not found", "FAILED"); ok = False

# B1. Mark truncation-salvage outputs
n = REPO / "cognitive_core" / "engine" / "nodes.py"
src = n.read_text(encoding="utf-8")
anchor1 = '''                output = {
                    "confidence": parsed.get("confidence", 0.7),'''
if '"_salvaged": True,\n                    "confidence": parsed.get("confidence", 0.7)' in src:
    report("B1. Truncation-salvage marker", "ALREADY APPLIED")
elif anchor1 in src:
    src = src.replace(anchor1, '''                output = {
                    "_salvaged": True,
                    "confidence": parsed.get("confidence", 0.7),''')
    report("B1. Truncation-salvage marker", "OK")
else:
    report("B1. anchor not found (nodes.py ~695)", "FAILED"); ok = False

# B2. Mark full-parse-failure outputs
anchor2 = '''            output = {
                "error": str(e),
                "raw_response": raw_response[:500],'''
if '"error": str(e),\n                "_salvaged": True,' in src:
    report("B2. Parse-failure marker", "ALREADY APPLIED")
elif anchor2 in src:
    src = src.replace(anchor2, '''            output = {
                "error": str(e),
                "_salvaged": True,
                "raw_response": raw_response[:500],''')
    report("B2. Parse-failure marker", "OK")
else:
    report("B2. anchor not found (nodes.py ~796)", "FAILED"); ok = False
n.write_text(src, encoding="utf-8")

# B3. Output-integrity mechanical signal
e = REPO / "cognitive_core" / "engine" / "epistemic.py"
et = e.read_text(encoding="utf-8")
anchor3 = '''    mechanical_scores = [s for s in [evidence_completeness, rule_coverage, citation_rate]
                         if s is not None]'''
patch3 = anchor3 + '''
    # Output integrity: a salvaged or unparseable output is a measured degradation,
    # not an absence of measurement.
    if output.get("error"):
        mechanical_scores.append(0.0)   # full parse failure
    elif output.get("_salvaged"):
        mechanical_scores.append(0.5)   # truncated, partial salvage'''
if "Output integrity: a salvaged or unparseable" in et:
    report("B3. Output-integrity signal", "ALREADY APPLIED")
elif anchor3 in et:
    e.write_text(et.replace(anchor3, patch3), encoding="utf-8")
    report("B3. Output-integrity signal", "OK")
else:
    report("B3. anchor not found (epistemic.py ~573)", "FAILED"); ok = False

# B4. Self-test
sys.path.insert(0, str(REPO))
try:
    import importlib
    import cognitive_core.engine.epistemic as ep
    importlib.reload(ep)
    clean  = ep.compute_step_epistemic_state({"primitive":"investigate","step_name":"t","output":{"confidence":0.9}}, [], [])
    salv   = ep.compute_step_epistemic_state({"primitive":"investigate","step_name":"t","output":{"confidence":0.9,"_salvaged":True}}, [], [])
    failed = ep.compute_step_epistemic_state({"primitive":"investigate","step_name":"t","output":{"confidence":0.0,"error":"x"}}, [], [])
    got = (round(clean.overall,2), round(salv.overall,2), round(failed.overall,2))
    good = got[0] > got[1] > got[2] and got[2] == 0.0
    report(f"B4. Self-test clean/salvaged/failed = {got}", "OK" if good else "FAILED")
    if not good: ok = False
except Exception as ex:
    report(f"B4. Self-test could not run ({type(ex).__name__}: {ex})", "WARN")

print()
if ok:
    print("ALL PATCHES APPLIED. Commit, then restart Stage 1.")
else:
    print("FAILED - send this output before running anything paid.")
    sys.exit(1)
