from pathlib import Path
import re, sys
ok = True
# 1. Harness: fix --resume case-id matching (pa_2024_a001 vs PA-2024-A001)
f = Path("run_parallel_benchmark.py"); t = f.read_text(encoding="utf-8")
OLD = 'r["case_id"] == case_id and r["system"] == system and r.get("error") is None'
NEW = 'r["case_id"].upper().replace("_","-") == case_id.upper().replace("_","-") and r["system"] == system and r.get("error") is None'
if NEW in t: print("[ALREADY] resume id normalization")
elif OLD in t: t = t.replace(OLD, NEW); print("[OK] resume id normalization")
else: print("[FAILED] resume anchor"); ok = False
# 2. Harness: abnormal termination must record an error (so resume retries it)
m = re.search(r'(    in_balanced = case_id in BALANCED_SET\n)', t)
if "abnormal termination: no parseable determination" in t: print("[ALREADY] abnormal-termination marking")
elif m:
    t = t[:m.start()] + '''    if error is None and (disposition == "ERROR" or not (determination or "").strip()):
        error = "abnormal termination: no parseable determination"
''' + t[m.start():]
    print("[OK] abnormal-termination marking")
else: print("[FAILED] termination anchor"); ok = False
f.write_text(t, encoding="utf-8")
# 3. Aggregator: ERROR dispositions are infrastructure failures
p = Path("aggregate_replications.py"); s = p.read_text(encoding="utf-8")
OLDA = 'if rec.get("error"):'
NEWA = 'if rec.get("error") or (rec.get("disposition") or "").upper() == "ERROR":'
if NEWA in s: print("[ALREADY] aggregator ERROR-disposition exclusion")
elif OLDA in s: p.write_text(s.replace(OLDA, NEWA), encoding="utf-8"); print("[OK] aggregator ERROR-disposition exclusion")
else: print("[FAILED] aggregator anchor"); ok = False
print("DONE" if ok else "FAILED"); sys.exit(0 if ok else 1)
