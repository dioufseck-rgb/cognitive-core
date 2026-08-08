import json
for r in json.load(open("output/parallel_benchmark/results.json")):
    if r["system"]=="cc":
        det = r.get("determination","")
        print("="*20, r["case_id"], r["tier"], "="*20)
        i = det.lower().rfind("tier")
        print(det[max(0,i-200):i+600] if i>0 else det[-600:])
