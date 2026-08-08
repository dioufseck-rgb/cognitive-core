#!/usr/bin/env python3
"""peek.py — print case/system/disposition/tier from a run's results.json.
Usage: python peek.py [output/parallel_benchmark]"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "output/parallel_benchmark")
for r in json.load(open(d / "results.json")):
    print(f"{r['case_id']:16} {r['system']:10} disp={str(r.get('disposition')):10} "
          f"tier={str(r.get('tier')):12} correct={r.get('correct')} "
          f"override={r.get('override_source','')} err={str(r.get('error'))[:40] if r.get('error') else ''}")
