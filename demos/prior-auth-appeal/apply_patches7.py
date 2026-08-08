from pathlib import Path
import re
f = Path("../../llm_config.yaml"); t = f.read_text(encoding="utf-8")
t2 = re.sub(r"(generate:\s*)8192", r"\g<1>16384", t)
t2 = re.sub(r"(deliberate:\s*)8192", r"\g<1>12288", t2)
t2 = re.sub(r"(challenge:\s*)8192", r"\g<1>12288", t2)
print("[OK] budgets: generate 16384, deliberate/challenge 12288" if t2 != t else "[ALREADY]")
f.write_text(t2, encoding="utf-8")
