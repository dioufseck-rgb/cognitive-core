from pathlib import Path
f = Path("run_parallel_benchmark.py"); t = f.read_text(encoding="utf-8")
OLD = 'if line.strip().startswith("FINAL DISPOSITION"):'
NEW = 'if line.strip().startswith("FINAL") and ("DISPOSITION" in line or "DETERMINATION" in line):'
if NEW in t: print("[ALREADY] 12b")
elif OLD in t: f.write_text(t.replace(OLD, NEW), encoding="utf-8"); print("[OK] 12b: extractor matches FINAL...DETERMINATION variants")
else: print("[FAILED] 12b anchor")
