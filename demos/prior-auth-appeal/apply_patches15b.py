from pathlib import Path
import re
f = Path("domains/prior_auth_appeal.yaml"); t = f.read_text(encoding="utf-8")
if "corroborated by objective clinical" in t: print("[ALREADY]")
else:
    m = re.search(r"^(\s*)A clear UPHOLD passes AUTO\.\s*$", t, re.M)
    if not m: print("[FAILED] anchor line 502 not found")
    else:
        ind = m.group(1)
        ADD = (m.group(0) + "\n"
               + ind + "A treating-physician declaration corroborated by objective clinical\n"
               + ind + "findings is not a conflict; corroboration supports AUTO or SPOT_CHECK.\n"
               + ind + "Verification findings that do not change the disposition do not by\n"
               + ind + "themselves require GATE.")
        t = t[:m.start()] + ADD + t[m.end():]
        f.write_text(t, encoding="utf-8"); print("[OK] corroboration clause in governance_context")
