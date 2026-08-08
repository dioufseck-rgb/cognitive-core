import json, sys
r = [x for x in json.load(open("output/parallel_benchmark/results.json")) if x["system"]=="cc"][0]
print("DISPOSITION:", r["disposition"], "| GT:", r["gt_disposition"], "| TIER:", r["tier"], "| steps:", r["steps"])
print("\n=== TRAJECTORY ===")
print(json.dumps(r.get("trajectory"), indent=1)[:3000])
print("\n=== DETERMINATION (tail) ===")
print(r.get("determination","")[-2500:])
