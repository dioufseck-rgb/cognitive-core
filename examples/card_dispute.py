"""
Card Dispute Example - Standalone Runner

Demonstrates the cognitive core framework by running a card dispute
through the full composition:

  Classify → Verify → Investigate → Classify → Generate → Challenge

Usage:
    export GOOGLE_API_KEY=your_key_here
    python -m examples.card_dispute
    python -m examples.card_dispute --verbose
    python -m examples.card_dispute --model gemini-2.5-flash
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.composer import load_use_case, run_workflow
from engine.runner import print_step_result, print_summary
from datetime import datetime


# Sample dispute scenarios for testing
SCENARIOS = {
    "clear_fraud": {
        "member_id": "MBR-2024-88291",
        "member_name": "Sarah Chen",
        "member_since": "2018-03-15",
        "account_type": "Signature Visa",
        "dispute_description": (
            "I noticed a charge of $487.32 from 'ELECTROMART ONLINE' on January 28th "
            "that I did not make. I have never shopped at this merchant. I had my card "
            "in my possession the entire time. I want this charge reversed immediately."
        ),
        "transaction_details": {
            "merchant": "ELECTROMART ONLINE",
            "amount": 487.32,
            "date": "2026-01-28",
            "category": "Electronics - Online",
            "auth_method": "card_not_present",
            "ip_geolocation": "Lagos, Nigeria",
            "device_fingerprint": "unknown_device",
        },
        "member_history": {
            "tenure_years": 8,
            "prior_disputes": 0,
            "avg_monthly_spend": 2150.00,
            "typical_merchants": "grocery, gas, restaurants, amazon",
            "fraud_alerts_prior": 0,
        },
    },
    "ambiguous_dispute": {
        "member_id": "MBR-2022-44102",
        "member_name": "James Rodriguez",
        "member_since": "2022-06-10",
        "account_type": "Platinum Rewards",
        "dispute_description": (
            "There's a charge of $299.99 from 'PREMIUM STREAMING SVCS' on February 1st. "
            "I don't recognize this merchant name. I do have some streaming subscriptions "
            "but nothing that costs this much. Maybe it's a price increase I wasn't told about?"
        ),
        "transaction_details": {
            "merchant": "PREMIUM STREAMING SVCS",
            "amount": 299.99,
            "date": "2026-02-01",
            "category": "Digital Services - Subscription",
            "auth_method": "recurring_token",
            "ip_geolocation": "Ashburn, Virginia",
            "device_fingerprint": "device_match_partial",
        },
        "member_history": {
            "tenure_years": 3,
            "prior_disputes": 1,
            "avg_monthly_spend": 3200.00,
            "typical_merchants": "streaming, restaurants, uber, electronics",
            "fraud_alerts_prior": 0,
        },
    },
    "billing_error": {
        "member_id": "MBR-2015-12003",
        "member_name": "Patricia Williams",
        "member_since": "2015-11-20",
        "account_type": "cashRewards",
        "dispute_description": (
            "I was charged $156.78 at Target on January 30th, but my receipt shows "
            "the total was $56.78. It looks like they added an extra $100. I've "
            "attached a photo of my receipt."
        ),
        "transaction_details": {
            "merchant": "TARGET STORE #1234",
            "amount": 156.78,
            "date": "2026-01-30",
            "category": "Retail - Department Store",
            "auth_method": "chip_and_pin",
            "ip_geolocation": "Virginia Beach, VA",
            "device_fingerprint": "known_device",
        },
        "member_history": {
            "tenure_years": 10,
            "prior_disputes": 0,
            "avg_monthly_spend": 1800.00,
            "typical_merchants": "target, costco, grocery, gas",
            "fraud_alerts_prior": 0,
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Card Dispute Example")
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()),
        default="clear_fraud",
        help="Which dispute scenario to run",
    )
    parser.add_argument("--model", "-m", default="gemini-2.0-flash", help="Gemini model")
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Save full state to JSON file")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")

    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for name, data in SCENARIOS.items():
            print(f"\n  {name}:")
            print(f"    Member: {data['member_name']}")
            print(f"    Dispute: {data['dispute_description'][:80]}...")
            print(f"    Amount: ${data['transaction_details']['amount']}")
        sys.exit(0)

    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        print("  export GOOGLE_API_KEY=your_key_here", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = Path(__file__).parent.parent / "use_cases" / "card_dispute.yaml"
    config = load_use_case(config_path)

    # Get scenario input
    scenario_input = SCENARIOS[args.scenario]

    print(f"\n{'#'*70}")
    print(f"  CARD DISPUTE WORKFLOW")
    print(f"  Scenario: {args.scenario}")
    print(f"  Member: {scenario_input['member_name']}")
    print(f"  Dispute: {scenario_input['dispute_description'][:80]}...")
    print(f"  Amount: ${scenario_input['transaction_details']['amount']}")
    print(f"  Model: {args.model}")
    print(f"{'#'*70}")

    print(f"\n  Composition: Classify → Verify → Investigate → Classify → Generate → Challenge")
    print(f"  Running...\n")

    start = datetime.now()

    try:
        final_state = run_workflow(
            config=config,
            workflow_input=scenario_input,
            model=args.model,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"\n⚠ Workflow failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()

    # Display results
    for step in final_state["steps"]:
        print_step_result(step, verbose=args.verbose)

    print_summary(final_state)
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_state, f, indent=2, default=str)
        print(f"  State saved to: {args.output}")


if __name__ == "__main__":
    main()
