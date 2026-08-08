from pathlib import Path
import re, sys
ok = True
# A. Footer contract in domain strategy
f = Path("domains/prior_auth_appeal.yaml"); t = f.read_text(encoding="utf-8")
if "FINAL DISPOSITION:" in t: print("[ALREADY] footer contract")
else:
    anchor = "  5. CHALLENGE THE DISPOSITION EXACTLY ONCE."
    ADD = """  4c. THE DETERMINATION MUST END WITH A MACHINE-READABLE FOOTER.
     The generated determination letter must end with exactly these lines:
       FINAL DISPOSITION: <OVERTURN | UPHOLD | PARTIAL | REMAND>
       COMPONENT DISPOSITIONS: <component>: <disposition>; ... (omit if single-component)
     A determination that overturns some components and upholds others is
     PARTIAL, and must be labeled PARTIAL in the footer.

"""
    if anchor not in t: print("[FAILED] footer anchor"); ok = False
    else: f.write_text(t.replace(anchor, ADD + anchor), encoding="utf-8"); print("[OK] footer contract")
# B. Replace extractor wholesale
h = Path("run_parallel_benchmark.py"); s = h.read_text(encoding="utf-8")
if "PARTIAL(LY)?|IN PART" in s: print("[ALREADY] extractor")
else:
    m = re.search(r"def extract_disposition\(text: str\) -> str:.*?return best if scores\[best\] > 0 else \"UNKNOWN\"\n", s, re.S)
    if not m: print("[FAILED] extractor anchor"); ok = False
    else:
        NEW = '''def extract_disposition(text: str) -> str:
    if not text or text.startswith("["):
        return "ERROR"
    upper = text.upper()
    FORMS = [("PARTIAL", r"PARTIAL(LY)?|IN PART"), ("REMAND", r"REMAND(ED)?"),
             ("UPHOLD", r"UPHELD|UPHOLD(S)?"), ("OVERTURN", r"OVERTURN(ED|S)?")]
    for line in upper.splitlines():
        ls = line.strip()
        if ls.startswith("FINAL") and ("DISPOSITION" in ls or "DETERMINATION" in ls):
            for kw, pat in FORMS:
                if re.search(pat, ls):
                    return kw
    for line in upper.splitlines():
        if line.strip().startswith("DISPOSITION:"):
            for kw, pat in FORMS:
                if re.search(pat, line):
                    return kw
    opener = upper[:500]
    for kw, pat in FORMS:
        if re.search(pat, opener):
            return kw
    scores = {kw: len(re.findall(pat, upper)) for kw, pat in FORMS}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "UNKNOWN"
'''
        s = s[:m.start()] + NEW + s[m.end():]
        h.write_text(s, encoding="utf-8"); print("[OK] extractor")
print("DONE" if ok else "FAILED"); sys.exit(0 if ok else 1)
