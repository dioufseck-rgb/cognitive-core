from pathlib import Path
import re, sys
ok = True
# 1. Orchestrator-decision guard (agentic_devs.py)
f = Path("../../cognitive_core/engine/agentic_devs.py")
t = f.read_text(encoding="utf-8")
OLD = '''        primitive = decision.get("primitive", "")
        step_name = decision.get("step_name", primitive)
        params_key = decision.get("params_key", step_name)

        prim_config = self.primitive_configs.get(params_key, {})'''
NEW = '''        primitive = decision.get("primitive", "")
        step_name = decision.get("step_name", primitive)
        if not isinstance(step_name, str):
            step_name = primitive if isinstance(primitive, str) else str(step_name)
        params_key = decision.get("params_key", step_name)
        if not isinstance(params_key, str):
            # The orchestrator occasionally emits an inline dict here instead of
            # a config name; recover with the step name so the lookup stays hashable.
            params_key = step_name

        prim_config = self.primitive_configs.get(params_key, {})'''
if "lookup stays hashable" in t:
    print("[ALREADY] agentic_devs guard")
elif OLD in t:
    f.write_text(t.replace(OLD, NEW), encoding="utf-8"); print("[OK] agentic_devs guard")
else:
    print("[FAILED] agentic_devs anchor"); ok = False
# 2. Aggregator: exclude infrastructure-error rows
p = Path("aggregate_replications.py")
s = p.read_text(encoding="utf-8")
if "INFRASTRUCTURE ERRORS" in s:
    print("[ALREADY] aggregator error exclusion")
else:
    m = re.search(r"    for rd in run_dirs:\n.*?\.replace\(\" \",\"_\"\)\)\)", s, re.S)
    if not m: print("[FAILED] aggregator anchor"); ok = False
    else:
        NEWA = '''    infra_errors = Counter()
    for rd in run_dirs:
        for rec in json.load(open(Path(rd)/"results.json")):
            cid, sysname = rec["case_id"], rec["system"]
            if cid not in gt:
                continue
            if rec.get("error"):
                infra_errors[(cid, sysname)] += 1
                continue   # infrastructure failure: excluded; re-run with --resume
            res[cid][sysname].append(((rec.get("disposition") or "UNKNOWN").upper(),
                                      (rec.get("tier") or "").lower().replace(" ","_")))
    if infra_errors:
        print("INFRASTRUCTURE ERRORS (excluded — re-run those replications with --resume until zero):")
        for (cid, sysname), k in sorted(infra_errors.items()):
            print(f"  {cid} / {sysname}: {k} errored run(s)")
        print()'''
        s = s[:m.start()] + NEWA + s[m.end():]
        p.write_text(s, encoding="utf-8"); print("[OK] aggregator error exclusion")
print("DONE" if ok else "FAILED - send output"); sys.exit(0 if ok else 1)
