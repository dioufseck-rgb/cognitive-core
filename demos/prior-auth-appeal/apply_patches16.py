from pathlib import Path
r = Path("run.py"); t = r.read_text(encoding="utf-8")
if "guard_trace" in t: print("[ALREADY]")
else:
    A = '''            if prim == "deliberate":'''
    N = '''            if prim in ("deliberate", "challenge", "reflect"):
                out = d.get("output", {})
                if isinstance(out, dict):
                    guard_trace.append({
                        "step": d.get("step_name", prim), "primitive": prim,
                        "recommended_action": out.get("recommended_action"),
                        "survives": out.get("survives"),
                        "vulnerabilities": str(out.get("vulnerabilities"))[:300],
                        "trajectory": out.get("trajectory"),
                        "reasoning": str(out.get("reasoning"))[:300]})
            if prim == "deliberate":'''
    A0 = '''    typed_disposition_raw = ""'''
    N0 = '''    typed_disposition_raw = ""
    guard_trace = []'''
    A2 = '''        "typed_disposition_raw": typed_disposition_raw,'''
    N2 = '''        "typed_disposition_raw": typed_disposition_raw,
        "guard_trace": guard_trace,'''
    if A in t and A0 in t and A2 in t:
        t = t.replace(A0, N0).replace(A, N).replace(A2, N2)
        r.write_text(t, encoding="utf-8"); print("[OK] guard trace captured")
    else: print("[FAILED] anchors")
