#!/usr/bin/env python3
"""Smoke test: 1 run of the simple case, verbose."""

import json, time, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from coordinator.runtime import Coordinator
from coordinator.types import WorkOrderStatus, WorkOrderResult

case_file = "cases/insurance_claim_simple.json"
with open(case_file) as f:
    case_data = json.load(f)
with open("cases/fixtures/human_task_responses.json") as f:
    human_responses = json.load(f)

print(f"=== SMOKE TEST: simple case ===")
# Ground truth from case data (identity-only format)
ground_truth = case_data.get('_ground_truth', {})
if ground_truth:
    print(f"Ground truth: net_payable=${ground_truth.get('net_payable', 0):,}")
else:
    print(f"Claim ID: {case_data.get('claim_id', 'unknown')}")
t0 = time.time()

coord = Coordinator(config_path="coordinator/config.yaml")
instance_id = coord.start(
    workflow_type="claim_adjudication",
    domain="claims_processing",
    case_input=case_data,
)
print(f"Started: {instance_id}")

# Loop: auto-complete human tasks and resume
for iteration in range(15):
    instance = coord.store.get_instance(instance_id)
    status = instance.status.value if instance else "gone"
    print(f"\n--- Iteration {iteration} | status={status} ---")

    if status == "completed":
        print("✓ COMPLETED")
        break
    if status == "failed":
        print("✗ FAILED")
        break
    if status != "suspended":
        print(f"  Not suspended, done looping")
        break

    suspension = coord.store.get_suspension(instance_id)
    if not suspension:
        print("  No suspension record found")
        break

    print(f"  Suspended at step: {suspension.suspended_at_step}")
    print(f"  Work orders: {suspension.work_order_ids}")
    print(f"  wo_need_map: {suspension.wo_need_map}")
    print(f"  deferred_needs: {suspension.deferred_needs}")

    # Check each work order
    human_wos = []
    all_done = True
    for wo_id in suspension.work_order_ids:
        wo = coord.store.get_work_order(wo_id)
        if not wo:
            print(f"  WO {wo_id}: NOT FOUND")
            continue
        need = suspension.wo_need_map.get(wo_id, "?")
        handler = wo.handler_instance_id or "none"
        print(f"  WO {wo_id[:12]}...: status={wo.status.value}, need={need}, handler={handler}")

        if wo.status.value == "running" and not wo.handler_instance_id:
            human_wos.append(wo)
        elif wo.status.value in ("running", "created", "dispatched"):
            all_done = False

    if human_wos:
        for wo in human_wos:
            need = suspension.wo_need_map.get(wo.work_order_id, "unknown")
            # Find matching response
            resp = human_responses.get("default", {})
            for key, val in human_responses.items():
                if key in need.lower():
                    resp = val
                    break
            print(f"  >> Auto-completing human task: {need}")
            wo.status = WorkOrderStatus.COMPLETED
            wo.completed_at = time.time()
            wo.result = WorkOrderResult(
                work_order_id=wo.work_order_id,
                status="completed",
                outputs=resp,
                completed_at=time.time(),
            )
            coord.store.save_work_order(wo)

    # Check if all WOs done now
    external_input = {}
    ready = True
    for wo_id in suspension.work_order_ids:
        wo = coord.store.get_work_order(wo_id)
        if wo and wo.status == WorkOrderStatus.COMPLETED and wo.result:
            external_input[wo_id] = wo.result.outputs
            need_name = suspension.wo_need_map.get(wo_id, "")
            if need_name:
                external_input[need_name] = wo.result.outputs
        elif wo and wo.status.value in ("running", "created", "dispatched"):
            ready = False

    if not ready:
        print("  Some WOs still running, waiting...")
        time.sleep(2)
        continue

    if not external_input:
        print("  No external input to resume with")
        break

    print(f"  Resuming with keys: {list(external_input.keys())}")
    try:
        coord.resume(
            instance_id=instance_id,
            external_input=external_input,
            resume_nonce=suspension.resume_nonce,
        )
    except Exception as e:
        print(f"  ✗ Resume error: {e}")
        import traceback; traceback.print_exc()
        break
else:
    print("\n⚠ Hit max iterations")

elapsed = time.time() - t0
print(f"\n=== Elapsed: {elapsed:.1f}s ===")

# Extract final state
instance = coord.store.get_instance(instance_id)
print(f"Final status: {instance.status.value if instance else 'gone'}")

# Try to get the settlement
if instance:
    ledger = coord.store.get_ledger(correlation_id=instance.correlation_id)
    step_names = [a['action_type'] for a in ledger if a.get('action_type') == 'step_completed']
    print(f"Steps completed: {len(step_names)}")
    # Look for settlement in last step output
    for a in reversed(ledger):
        details = a.get('details', {})
        if 'artifact' in details:
            art = details['artifact']
            if isinstance(art, dict) and 'net_payable' in art:
                print(f"Settlement: net_payable=${art['net_payable']:,.2f}")
                break
            elif isinstance(art, dict):
                print(f"Last artifact keys: {list(art.keys())[:10]}")
