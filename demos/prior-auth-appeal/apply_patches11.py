from pathlib import Path
f = Path("domains/prior_auth_appeal.yaml"); t = f.read_text(encoding="utf-8")
MARK = "PER-COMPONENT CRITERIA VERIFICATION"
ADD = """  4b. PER-COMPONENT CRITERIA VERIFICATION IS MANDATORY.
     Before deliberating, run a verify step that checks the request against
     the plan's clinical criteria RULE BY RULE. If the request contains
     multiple components (multiple spinal levels, procedures, or items),
     verify each component SEPARATELY and report conforms/violations per
     component. Deliberate must then state a disposition PER COMPONENT; a
     full OVERTURN is warranted only if every component independently
     conforms. Cross-source consistency verification does not substitute
     for criteria verification.
"""
if MARK in t: print("[ALREADY] Fix B")
else:
    anchor = "  5. CHALLENGE THE DISPOSITION EXACTLY ONCE."
    if anchor not in t: print("[FAILED] Fix B anchor"); raise SystemExit(1)
    f.write_text(t.replace(anchor, ADD + "\n" + anchor), encoding="utf-8")
    print("[OK] Fix B: per-component criteria verification mandated")
