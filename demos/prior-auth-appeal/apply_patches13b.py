from pathlib import Path
import sys
ok = True
# A. Revert the broken harness splice
h = Path("run_parallel_benchmark.py"); s = h.read_text(encoding="utf-8")
BROKEN = '''    typed_raw = (result.get("typed_disposition_raw") or "") if isinstance(result, dict) else ""
    if system == "cc" and typed_raw:
        disposition = extract_disposition("FINAL DISPOSITION: " + typed_raw)
        if disposition in ("UNKNOWN", "ERROR"):
            disposition = extract_disposition(determination)
    else:
        disposition = extract_disposition(determination)'''
CLEAN = "    disposition = extract_disposition(determination)"
if BROKEN in s: h.write_text(s.replace(BROKEN, CLEAN), encoding="utf-8"); print("[OK] harness splice reverted")
elif CLEAN in s: print("[ALREADY] harness clean")
else: print("[FAILED] harness state unrecognized"); ok = False
# B. run.py: typed disposition rides as the determination's first line
r = Path("run.py"); t = r.read_text(encoding="utf-8")
MARK = 'determination = "FINAL DISPOSITION: "'
if MARK in t: print("[ALREADY] run.py")
else:
    A = '''    return {
        "determination": determination,
        "typed_disposition_raw": typed_disposition_raw,
        "tier": tier,'''
    N = '''    if typed_disposition_raw:
        determination = "FINAL DISPOSITION: " + typed_disposition_raw + "\\n\\n" + (determination or "")
    return {
        "determination": determination,
        "typed_disposition_raw": typed_disposition_raw,
        "tier": tier,'''
    if A in t: r.write_text(t.replace(A, N), encoding="utf-8"); print("[OK] run.py: typed disposition prepended to determination")
    else: print("[FAILED] run.py anchor"); ok = False
print("DONE" if ok else "FAILED"); sys.exit(0 if ok else 1)
