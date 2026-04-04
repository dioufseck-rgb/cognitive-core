"""
Submit a permit review case to the Cognitive Core API server.

Usage:
    python submit_case.py                          # exempt case
    python submit_case.py --case pmt_2026_00318    # conditional
    python submit_case.py --case pmt_2026_00447    # prohibited
    python submit_case.py --url http://localhost:8000

Then open the trace page at:
    http://localhost:8000/instances/{id}/trace
"""

import json
import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

DEMO_DIR = Path(__file__).resolve().parent

CASES = {
    "pmt_2026_00142": "Tenant improvement — §15301 exempt → SPOT CHECK",
    "pmt_2026_00318": "Mixed-use infill — §15332 fails → MND → GATE",
    "pmt_2026_00447": "Concrete batch plant — floodway bar → HOLD",
}

def submit(base_url: str, case_name: str):
    case_file = DEMO_DIR / "cases" / f"{case_name}.json"
    if not case_file.exists():
        print(f"Case not found: {case_file}")
        sys.exit(1)

    case = json.load(open(case_file))
    permit_number = case.get("get_application", {}).get("permit_number", case_name)

    print(f"\n  Submitting: {permit_number}")
    print(f"  {CASES.get(case_name, '')}")

    payload = {
        "workflow_type": "permit_intake",
        "domain": "permit_intake",
        "case_input": case,
    }

    try:
        r = requests.post(f"{base_url}/api/start", json=payload, timeout=300)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n  ✗ Cannot connect to {base_url}")
        print("  Start the server first:")
        print("    GOOGLE_API_KEY=your_key \\")
        print("    CC_COORD_CONFIG=demos/permit-review/coordinator_config.yaml \\")
        print("    CC_COORD_BASE=demos/permit-review \\")
        print("    uvicorn cognitive_core.api.server:app --port 8000")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n  ✗ HTTP {r.status_code}: {r.text[:200]}")
        sys.exit(1)

    data = r.json()
    instance_id = data.get("instance_id") or data.get("id") or str(data)

    print(f"\n  ✓ Started: {instance_id}")
    print(f"\n  Trace page:")
    print(f"    {base_url}/instances/{instance_id}/trace")
    print(f"\n  Status API:")
    print(f"    curl {base_url}/api/instances/{instance_id}")

    return instance_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="pmt_2026_00142",
                        choices=list(CASES.keys()),
                        help="Case to submit (default: exempt case)")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="API server URL")
    parser.add_argument("--all", action="store_true",
                        help="Submit all three cases")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  PERMIT REVIEW — API submission")
    print("═" * 60)

    if args.all:
        for case_name in CASES:
            submit(args.url, case_name)
        print()
    else:
        submit(args.url, args.case)
        print()


if __name__ == "__main__":
    main()