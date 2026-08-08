from pathlib import Path
f = Path("domains/prior_auth_appeal.yaml"); t = f.read_text(encoding="utf-8")
M = "corroborated by objective findings"
if M in t: print("[ALREADY]")
else:
    A = "Never AUTO when sources conflict."
    N = """Never AUTO when sources conflict.
     A treating-physician declaration corroborated by objective clinical
     findings is not a source conflict. Verification findings that do not
     change the disposition do not by themselves require GATE."""
    if A not in t: print("[FAILED] anchor")
    else: f.write_text(t.replace(A, N, 1), encoding="utf-8"); print("[OK] corroboration clause")
