from pathlib import Path
import sys
ok = True
# A. run.py: capture last deliberate recommended_action from the typed ledger
r = Path("run.py"); t = r.read_text(encoding="utf-8")
if "typed_disposition_raw" in t: print("[ALREADY] run.py")
else:
    A1 = '''            if prim == "generate":'''
    N1 = '''            if prim == "deliberate":
                out = d.get("output", {})
                if isinstance(out, dict) and out.get("recommended_action"):
                    typed_disposition_raw = str(out.get("recommended_action"))
            if prim == "generate":'''
    A2 = '''    return {
        "determination": determination,
        "tier": tier,'''
    N2 = '''    return {
        "determination": determination,
        "typed_disposition_raw": typed_disposition_raw,
        "tier": tier,'''
    A0 = '''    determination = ""
    trajectory = []'''
    N0 = '''    determination = ""
    typed_disposition_raw = ""
    trajectory = []'''
    if A1 in t and A2 in t and A0 in t:
        t = t.replace(A0, N0).replace(A1, N1).replace(A2, N2)
        r.write_text(t, encoding="utf-8"); print("[OK] run.py: typed disposition exposed from ledger")
    else: print("[FAILED] run.py anchors"); ok = False
# B. harness: CC scored from the typed field, letter parsing as fallback only
h = Path("run_parallel_benchmark.py"); s = h.read_text(encoding="utf-8")
if "typed_disposition_raw" in s: print("[ALREADY] harness")
else:
    A3 = "    disposition = extract_disposition(determination)"
    N3 = '''    typed_raw = (result.get("typed_disposition_raw") or "") if isinstance(result, dict) else ""
    if system == "cc" and typed_raw:
        disposition = extract_disposition("FINAL DISPOSITION: " + typed_raw)
        if disposition in ("UNKNOWN", "ERROR"):
            disposition = extract_disposition(determination)
    else:
        disposition = extract_disposition(determination)'''
    if A3 in s: s = s.replace(A3, N3); h.write_text(s, encoding="utf-8"); print("[OK] harness: CC scored from typed record")
    else: print("[FAILED] harness anchor"); ok = False
print("DONE" if ok else "FAILED"); sys.exit(0 if ok else 1)
