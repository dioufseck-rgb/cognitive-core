import re, sys
from pathlib import Path
if len(sys.argv) != 2:
    sys.exit("usage: python set_model.py <model-string>")
target = sys.argv[1].removeprefix("models/")
HERE = Path(".").resolve()
REPO = HERE.parents[1]
PAT = re.compile(r"gemini-[\w\.\-]+")
for f in (REPO / "llm_config.yaml", HERE / "run_parallel_benchmark.py"):
    txt = f.read_text(encoding="utf-8")
    found = sorted(set(PAT.findall(txt)) - {target})
    new = PAT.sub(target, txt)
    if new != txt:
        f.write_text(new, encoding="utf-8")
        print(f"[OK] {f.name}: {', '.join(found)} -> {target}")
    else:
        print(f"[ALREADY] {f.name}: only '{target}' present")
print(f"\nAll systems now target: {target}")
