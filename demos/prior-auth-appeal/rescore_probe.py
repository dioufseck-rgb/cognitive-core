import json, importlib.util
spec = importlib.util.spec_from_file_location("h", "run_parallel_benchmark.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
for r in json.load(open("output/parallel_benchmark/results.json")):
    if r["system"]=="cc":
        new = h.extract_disposition(r.get("determination",""))
        print(f"{r['case_id']}: stored={r['disposition']} -> rescored={new} (GT {r['gt_disposition']}) tier={r['tier']}")
