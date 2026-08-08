from pathlib import Path
import re
f = Path("../../llm_config.yaml"); t = f.read_text(encoding="utf-8")
t2 = re.sub(r"(govern:\s*)6144", r"\g<1>12288", t)
t2 = re.sub(r"(verify:\s*)6144", r"\g<1>10240", t2)
print("[OK] govern 12288, verify 10240" if t2 != t else "[ALREADY]")
f.write_text(t2, encoding="utf-8")
