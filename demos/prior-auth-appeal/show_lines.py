import json
for r in json.load(open("output/parallel_benchmark/results.json")):
    if r["system"]=="cc":
        d = r.get("determination","")
        print("="*15, r["case_id"], "extracted:", r["disposition"], "="*15)
        for ln in d.splitlines():
            if any(k in ln.upper() for k in ("DISPOSITION","UPHELD","OVERTURN","PARTIAL","IN PART")): print("  |", ln.strip()[:150])
