import yaml, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
targets = [HERE / "domains" / "prior_auth_appeal.yaml",
           HERE.parents[0] / "loan-modification" / "domains"]
files = [targets[0]] + (list(targets[1].glob("*.yaml")) if targets[1].exists() else [])
for f in files:
    txt = f.read_text(encoding="utf-8")
    cfg = yaml.safe_load(txt)
    if isinstance(cfg, dict) and "governance" in cfg:
        print(f"[ALREADY] {f.name}: governance = {cfg['governance']}")
        continue
    txt = txt.rstrip() + "\n\n# Declared baseline governance tier (floor). The effective tier is\n# resolved upward-only from this floor by the govern primitive, coherence\n# flags, and quality gates. Without this key the runtime defaults the\n# floor to 'gate', which makes SPOT_CHECK/AUTO unreachable.\ngovernance:\n  tier: auto\n"
    f.write_text(txt, encoding="utf-8")
    check = yaml.safe_load(f.read_text(encoding="utf-8"))
    ok = check.get("governance", {}).get("tier") == "auto"
    print(f"[{'OK' if ok else 'FAILED'}] {f.name}: declared tier = auto")
    if not ok: sys.exit(1)
print("\nDeclared-tier floor fixed. Commit, then rerun Stage 1.")
