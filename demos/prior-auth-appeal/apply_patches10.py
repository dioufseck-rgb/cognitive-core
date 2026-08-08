from pathlib import Path
f = Path("../../cognitive_core/engine/epistemic.py"); t = f.read_text(encoding="utf-8")
OLD = "    overall, warranted = compute_overall(mechanical_scores, [], coherence_flags)"
NEW = """    judgment_scores = [v for v in (output.get("reasoning_quality"), output.get("outcome_certainty"))
                       if isinstance(v, (int, float))]
    overall, warranted = compute_overall(mechanical_scores, judgment_scores, coherence_flags)"""
if "judgment_scores = [v for v in" in t: print("[ALREADY] Fix A")
elif OLD in t: f.write_text(t.replace(OLD, NEW), encoding="utf-8"); print("[OK] Fix A: judgment signals wired into step state")
else: print("[FAILED] Fix A anchor - send lines 570-585 of epistemic.py")
