import sqlite3, json, re
db = sqlite3.connect(r"..\..\coordinator.db")
tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", tables)
found = 0
for t in tables:
    cols = [c[1] for c in db.execute(f"PRAGMA table_info({t})")]
    for row in db.execute(f"SELECT * FROM {t} ORDER BY rowid DESC LIMIT 400"):
        blob = " ".join(str(x) for x in row)
        for m in re.finditer(r'"(reasoning_quality|outcome_certainty)":\s*([0-9.]+|null)', blob):
            print(f"{t}: {m.group(1)} = {m.group(2)}")
            found += 1
        if found > 30: break
    if found > 30: break
print("none found in recent rows" if found == 0 else f"({found} shown)")
