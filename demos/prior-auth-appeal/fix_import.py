from pathlib import Path
f = Path("run_parallel_benchmark.py"); t = f.read_text(encoding="utf-8")
if "\nimport re\n" in t or t.startswith("import re"): print("[ALREADY]")
else:
    i = t.find("\nimport "); i = i if i >= 0 else t.find("\nfrom ")
    f.write_text(t[:i] + "\nimport re" + t[i:], encoding="utf-8"); print("[OK] import re added")
