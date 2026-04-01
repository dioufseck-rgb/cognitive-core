# E-Commerce Returns

**Pattern:** P08 / P03 hybrid — Case Resolution with Exception path  
**Overlays:** adversarial (fraud path), governance  
**Coordinator:** simple  

Governed return request processing. Standard in-policy returns approve in seconds. Out-of-policy returns investigate whether an exception is warranted based on customer relationship. Suspected fraud is investigated adversarially before routing to loss prevention. Every decision — approval, denial, exception, or fraud flag — produces a warrant with specific reasons.

## The three paths

```
Standard:   retrieve → classify → verify → govern
Exception:  retrieve → classify → investigate → deliberate → verify → govern
Fraud:      retrieve → classify → investigate → challenge → govern
```

The `classify_return` step determines the path. Most returns take the standard path. High-LTV customers outside the return window take the exception path. Anomalous patterns take the fraud path.

## Governance outcomes

| Outcome | What it means | Application action |
|---------|--------------|-------------------|
| `AUTO approve` | In-policy, clear approval | Process refund or replacement immediately |
| `SPOT_CHECK approve` | Approved, flag for sampling | Process, add to QA queue |
| `GATE agent review` | Exception or high value | Show "under review", route brief to agent |
| `HOLD loss prevention` | Fraud indicators | Hold, route evidence to loss prevention |

## Files

```
ecommerce-returns/
  ecommerce_return.yaml     ← fill in your return policy and thresholds
  coordinator_config.yaml   ← governance queues and SLAs
  run.py                    ← runner / embedding reference
  workflows/
    ecommerce_return.yaml   ← the workflow (do not edit)
  cases/
    example_returns.json    ← 5 cases: standard, exception, fraud, carrier, high-value
```

## Setup

```bash
pip install -e ".[runtime]"
export ANTHROPIC_API_KEY=your_key_here
python library/domain-packs/ecommerce-returns/run.py
```

## Embedding in your application

```python
from cognitive_core.coordinator.runtime import Coordinator

coord = Coordinator('ecommerce_return/coordinator_config.yaml', db_path='returns.db')

# Build case from your order management system
case = {
    "order_id": order.id,
    "order_date": str(order.created_at.date()),
    "order_value": float(order.total),
    "days_since_delivery": (today - order.delivered_at).days,
    "return_reason": request.reason,
    "return_reason_detail": request.detail,
    "item_condition": request.reported_condition,
    "item_category": order.items[0].category,
    # Include pre-fetched customer and order data
    "get_order": order.to_dict(),
    "get_customer": customer.to_dict(),
    "get_return_history": customer.return_history(),
    "get_item_details": order.items[0].details(),
    "get_carrier_info": shipment.delivery_info(),
}

iid = coord.start('ecommerce_return', 'ecommerce_return', case)
inst = coord.store.get_instance(iid)

# Route based on disposition
gov_output = inst.step_outputs.get('govern_disposition', {})
tier = gov_output.get('tier_applied', 'gate')
disposition = gov_output.get('disposition')
work_order = gov_output.get('work_order')

if tier == 'auto':
    process_refund(order, disposition)
elif tier == 'spot_check':
    process_refund(order, disposition)
    flag_for_qa(iid)
elif tier == 'gate':
    show_under_review(customer)
    route_to_agent(work_order)
elif tier == 'hold':
    notify_loss_prevention(work_order)
```

## Customising

Open `ecommerce_return.yaml` and fill in:

| Section | What to change |
|---------|---------------|
| `classify_return.categories` | Your return reason taxonomy |
| `classify_return.exception_condition` | When to trigger exception investigation |
| `classify_return.fraud_condition` | When to trigger fraud investigation |
| `verify_policy.rules` | Your specific return policy rules (numbered) |
| `investigate_exception.scope` | What to weigh for exception decisions |
| `deliberate_exception.instruction` | Your exception approval criteria |
| `investigate_fraud.scope` | Your fraud pattern indicators |
| `challenge_fraud_finding.threat_model` | False positive risks to probe |
| `govern_disposition.governance_context` | Your governance thresholds by order value |

## The five example cases

| ID | Scenario | Expected path |
|----|----------|---------------|
| RET-001 | Defective item, 8 days, good customer | Standard → AUTO approve |
| RET-002 | Past window, 7-year $8.9K LTV customer | Exception → investigate → approve_exception |
| RET-003 | New account, prepaid card, weight discrepancy | Fraud → investigate → challenge → HOLD |
| RET-004 | Carrier damage | carrier_claim classification → route to carrier |
| RET-005 | $1,850 camera, within policy | Standard → verify conforms → GATE (high value) |

## What the warrant looks like

Every decision produces a warrant. For a denial:
```
Denial reason: Return requested 47 days after delivery. Policy rule 1 
requires return within 30 days. No exception granted: account age 14 days, 
no prior purchase history, first return request.
```

For an exception approval:
```
Exception granted: Return is 17 days outside the return window. Customer 
has been with the merchant for 2,641 days with $8,940 lifetime value. 
First exception request. Item is unopened in original packaging. 
Circumstances (gift purchase, travel delay) are documented and credible.
```

This is the difference between a decision that can be explained and one that can't.
