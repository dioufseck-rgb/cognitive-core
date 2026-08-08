import json
r = [x for x in json.load(open("output/parallel_benchmark/results.json")) if x["system"]=="cc"][0]
gt = r.get("guard_trace")
print("disp:", r["disposition"], "| tier:", r["tier"])
if not gt: print("guard_trace not in results record - harness drops unknown keys; tell Claude")
else:
    for s in gt:
        print("-"*60); print(s["primitive"].upper(), s["step"])
        for k in ("recommended_action","survives","trajectory","vulnerabilities","reasoning"):
            if s.get(k) not in (None,"None",""): print(f"  {k}: {s[k]}")
