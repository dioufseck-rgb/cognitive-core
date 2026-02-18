"""
Cognitive Core — Coordinator CLI

Run workflows through the runtime coordinator instead of the raw runner.
This adds governance tier evaluation, delegation policy checking, and
full audit logging to every workflow execution.

Usage:
    # Run a single workflow through the coordinator
    python -m coordinator.cli run \\
        --workflow dispute_resolution \\
        --domain card_dispute \\
        --case cases/card_clear_fraud.json

    # Show coordinator stats
    python -m coordinator.cli stats

    # Show correlation chain for an instance
    python -m coordinator.cli chain <instance_id>

    # Show audit ledger
    python -m coordinator.cli ledger [--instance <id>] [--correlation <id>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from coordinator.runtime import Coordinator
from coordinator.types import InstanceStatus


def cmd_run(args, coord: Coordinator):
    """Run a workflow through the coordinator."""
    # Load case input
    case_input = {}
    if args.case:
        p = Path(args.case)
        if p.exists():
            with open(p) as f:
                case_input = json.load(f) if p.suffix == ".json" else __import__("yaml").safe_load(f)
        else:
            print(f"Error: case file not found: {args.case}", file=sys.stderr)
            sys.exit(1)
    elif args.input:
        case_input = json.loads(args.input)

    if not case_input:
        print("Error: provide --case or --input", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 70}", file=sys.stderr)
    print(f"  COORDINATOR: {args.workflow}/{args.domain}", file=sys.stderr)
    print(f"{'═' * 70}", file=sys.stderr, flush=True)

    start = time.time()
    try:
        instance_id = coord.start(
            workflow_type=args.workflow,
            domain=args.domain,
            case_input=case_input,
            model=args.model,
            temperature=args.temperature,
        )
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start

    # Print results
    instance = coord.get_instance(instance_id)
    if not instance:
        print(f"\nError: instance {instance_id} not found after execution", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 70}", file=sys.stderr)
    print(f"  COORDINATOR SUMMARY", file=sys.stderr)
    print(f"{'─' * 70}", file=sys.stderr)
    print(f"  instance:    {instance.instance_id}", file=sys.stderr)
    print(f"  correlation: {instance.correlation_id}", file=sys.stderr)
    print(f"  status:      {instance.status.value}", file=sys.stderr)
    print(f"  governance:  {instance.governance_tier}", file=sys.stderr)
    print(f"  steps:       {instance.step_count}", file=sys.stderr)
    print(f"  elapsed:     {elapsed:.1f}s", file=sys.stderr)

    # Show delegations
    work_orders = coord.get_work_orders(instance_id)
    if work_orders:
        print(f"\n  DELEGATIONS:", file=sys.stderr)
        for wo in work_orders:
            print(f"    {wo.work_order_id}: {wo.contract_name} → "
                  f"{wo.handler_workflow_type}/{wo.handler_domain} "
                  f"[{wo.status.value}]", file=sys.stderr)

    # Show correlation chain
    chain = coord.get_correlation_chain(instance.correlation_id)
    if len(chain) > 1:
        print(f"\n  CORRELATION CHAIN ({instance.correlation_id}):", file=sys.stderr)
        for inst in chain:
            marker = "→" if inst.instance_id != instance_id else "●"
            print(f"    {marker} {inst.instance_id}: "
                  f"{inst.workflow_type}/{inst.domain} "
                  f"[{inst.status.value}]", file=sys.stderr)

    # Show audit ledger summary
    ledger = coord.get_ledger(correlation_id=instance.correlation_id)
    if ledger:
        print(f"\n  AUDIT LEDGER ({len(ledger)} entries):", file=sys.stderr)
        for entry in ledger:
            print(f"    {entry['action_type']:24s} "
                  f"{entry['instance_id'][:16]}", file=sys.stderr)

    print(f"{'═' * 70}\n", file=sys.stderr)

    # Show pending approvals hint
    pending = coord.list_pending_approvals()
    if pending:
        print(f"  ⏸ {len(pending)} instance(s) awaiting governance approval:", file=sys.stderr)
        for p in pending:
            print(f"    {p['instance_id']}: {p['workflow_type']}/{p['domain']} "
                  f"[{p['governance_tier']}]", file=sys.stderr)
        print(f"\n  Approve: python -m coordinator.cli approve <instance_id>", file=sys.stderr)
        print(f"  Reject:  python -m coordinator.cli reject <instance_id>", file=sys.stderr)
        print(f"  List:    python -m coordinator.cli pending\n", file=sys.stderr)

    # Output result as JSON
    if instance.result and args.verbose:
        print(json.dumps(instance.result, indent=2, default=str))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "instance": {
                    "instance_id": instance.instance_id,
                    "workflow_type": instance.workflow_type,
                    "domain": instance.domain,
                    "status": instance.status.value,
                    "governance_tier": instance.governance_tier,
                    "correlation_id": instance.correlation_id,
                    "step_count": instance.step_count,
                    "result": instance.result,
                },
                "work_orders": [
                    {
                        "work_order_id": wo.work_order_id,
                        "contract": wo.contract_name,
                        "handler": f"{wo.handler_workflow_type}/{wo.handler_domain}",
                        "status": wo.status.value,
                    }
                    for wo in work_orders
                ],
                "ledger": ledger,
            }, f, indent=2, default=str)
        print(f"  Output saved: {args.output}", file=sys.stderr)


def cmd_stats(args, coord: Coordinator):
    """Show coordinator statistics."""
    s = coord.stats()
    print(json.dumps(s, indent=2))


def cmd_pending(args, coord: Coordinator):
    """List instances awaiting governance approval."""
    approvals = coord.list_pending_approvals()
    if not approvals:
        print("No instances pending approval.")
        return

    print(f"\nPending Approvals ({len(approvals)})")
    print(f"{'─' * 70}")
    for a in approvals:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.get("created_at", 0)))
        print(f"  {a['instance_id']}")
        print(f"    type:        {a['workflow_type']}/{a['domain']}")
        print(f"    governance:  {a['governance_tier']}")
        print(f"    correlation: {a['correlation_id']}")
        print(f"    queue:       {a.get('queue', '—')}")
        print(f"    created:     {ts}")
        if a.get("sla_seconds"):
            print(f"    sla:         {a['sla_seconds']}s")
        print()
    print(f"Approve:  python -m coordinator.cli approve <instance_id> --approver 'Name'")
    print(f"Reject:   python -m coordinator.cli reject <instance_id> --reason 'Why'")


def cmd_approve(args, coord: Coordinator):
    """Approve a governance-suspended instance."""
    print(f"\n{'═' * 70}", file=sys.stderr)
    print(f"  APPROVING: {args.instance_id}", file=sys.stderr)
    print(f"{'═' * 70}", file=sys.stderr, flush=True)

    try:
        instance_id = coord.approve(
            instance_id=args.instance_id,
            approver=args.approver,
            notes=args.notes,
        )
    except ValueError as e:
        print(f"\n  ✗ FAILED: {e}", file=sys.stderr)
        print(f"    db: {args.db}", file=sys.stderr)
        # Help the user
        if "not found" in str(e).lower():
            print(f"\n  Hint: Make sure you're using the same --db path as the 'run' command.", file=sys.stderr)
            print(f"  Try:  python -m coordinator.cli --db {args.db} pending", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}", file=sys.stderr)
        sys.exit(1)

    instance = coord.get_instance(instance_id)
    if instance:
        print(f"\n{'═' * 70}", file=sys.stderr)
        print(f"  APPROVED: {instance_id}", file=sys.stderr)
        print(f"  status:   {instance.status.value}", file=sys.stderr)
        print(f"{'═' * 70}\n", file=sys.stderr)


def cmd_reject(args, coord: Coordinator):
    """Reject a governance-suspended instance."""
    try:
        instance_id = coord.reject(
            instance_id=args.instance_id,
            rejector=args.rejector,
            reason=args.reason,
        )
        print(f"Rejected: {instance_id}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_chain(args, coord: Coordinator):
    """Show correlation chain for an instance."""
    instance = coord.get_instance(args.instance_id)
    if not instance:
        print(f"Instance not found: {args.instance_id}", file=sys.stderr)
        sys.exit(1)

    chain = coord.get_correlation_chain(instance.correlation_id)
    print(f"\nCorrelation: {instance.correlation_id}")
    print(f"{'─' * 50}")
    for inst in chain:
        marker = "●" if inst.instance_id == args.instance_id else " "
        print(f"  {marker} {inst.instance_id}")
        print(f"    type:       {inst.workflow_type}/{inst.domain}")
        print(f"    status:     {inst.status.value}")
        print(f"    governance: {inst.governance_tier}")
        print(f"    steps:      {inst.step_count}")
        if inst.lineage:
            print(f"    lineage:    {' → '.join(inst.lineage)}")


def cmd_ledger(args, coord: Coordinator):
    """Show audit ledger."""
    entries = coord.get_ledger(
        instance_id=args.instance if hasattr(args, "instance") and args.instance else None,
        correlation_id=args.correlation if hasattr(args, "correlation") and args.correlation else None,
    )
    if not entries:
        print("No ledger entries found.")
        return

    print(f"\nAudit Ledger ({len(entries)} entries)")
    print(f"{'─' * 70}")
    for e in entries:
        ts = time.strftime("%H:%M:%S", time.localtime(e["created_at"]))
        print(f"  [{ts}] {e['action_type']:24s} {e['instance_id'][:20]}")
        if args.verbose:
            for k, v in e["details"].items():
                val = str(v)[:60]
                print(f"           {k}: {val}")


def main():
    # Resolve project root from CLI module location
    _cli_dir = Path(__file__).resolve().parent         # coordinator/
    _project_root = _cli_dir.parent                     # project root

    parser = argparse.ArgumentParser(
        description="Cognitive Core — Runtime Coordinator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(_cli_dir / "config.yaml"),
        help="Coordinator config YAML (default: coordinator/config.yaml)",
    )
    parser.add_argument(
        "--db", default=str(_project_root / "coordinator.db"),
        help="Coordinator database path (default: coordinator.db in project root)",
    )

    subs = parser.add_subparsers(dest="command", help="Command")

    # run
    run_p = subs.add_parser("run", help="Run a workflow through the coordinator")
    run_p.add_argument("--workflow", "-w", required=True)
    run_p.add_argument("--domain", "-d", required=True)
    run_p.add_argument("--case", "-c", help="Case JSON file")
    run_p.add_argument("--input", "-i", help="JSON string input")
    run_p.add_argument("--model", "-m", default="default")
    run_p.add_argument("--temperature", "-t", type=float, default=0.1)
    run_p.add_argument("--verbose", "-v", action="store_true")
    run_p.add_argument("--output", "-o", help="Save result JSON")

    # stats
    subs.add_parser("stats", help="Show coordinator statistics")

    # pending
    subs.add_parser("pending", help="List instances awaiting governance approval")

    # approve
    approve_p = subs.add_parser("approve", help="Approve a governance-suspended instance")
    approve_p.add_argument("instance_id")
    approve_p.add_argument("--approver", "-a", default="", help="Approver name")
    approve_p.add_argument("--notes", "-n", default="", help="Approval notes")
    approve_p.add_argument("--verbose", "-v", action="store_true")

    # reject
    reject_p = subs.add_parser("reject", help="Reject a governance-suspended instance")
    reject_p.add_argument("instance_id")
    reject_p.add_argument("--rejector", "-r", default="", help="Rejector name")
    reject_p.add_argument("--reason", default="", help="Rejection reason")

    # chain
    chain_p = subs.add_parser("chain", help="Show correlation chain")
    chain_p.add_argument("instance_id")

    # ledger
    ledger_p = subs.add_parser("ledger", help="Show audit ledger")
    ledger_p.add_argument("--instance", help="Filter by instance ID")
    ledger_p.add_argument("--correlation", help="Filter by correlation ID")
    ledger_p.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize coordinator
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: config not found at {config_path}, using defaults",
              file=sys.stderr)
        coord = Coordinator(db_path=args.db, verbose=True)
    else:
        coord = Coordinator(config_path=config_path, db_path=args.db, verbose=True)

    # Dispatch
    if args.command == "run":
        cmd_run(args, coord)
    elif args.command == "stats":
        cmd_stats(args, coord)
    elif args.command == "pending":
        cmd_pending(args, coord)
    elif args.command == "approve":
        cmd_approve(args, coord)
    elif args.command == "reject":
        cmd_reject(args, coord)
    elif args.command == "chain":
        cmd_chain(args, coord)
    elif args.command == "ledger":
        cmd_ledger(args, coord)


if __name__ == "__main__":
    main()
