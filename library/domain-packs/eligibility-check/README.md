# Eligibility Check

**Pattern:** P02 — Eligibility Determination (lightweight embeddable)  
**Overlays:** exception, governance  
**Coordinator:** simple  

Fast governed eligibility determination for embedding in web apps and portals. Works for any decision of the form "does this person or account qualify for this benefit?" Clear cases resolve in seconds. Borderline cases investigate before deciding. Every outcome — approval or denial — has a warrant with specific reasons.

## What it's for

Any eligibility decision where:
- The same criteria apply to many applicants
- Denials need documented specific reasons
- Borderline cases need judgment, not just rules
- You want consistency across all decisions

**Common uses:** subscription tier upgrades, student / professional discounts, loyalty program tiers, feature gating, beta access, credit line increases, promotional offer qualification, insurance product eligibility, government benefit programs.

## The two paths

```
Standard:   retrieve → classify → verify → govern   (clearly eligible / ineligible)
Review:     retrieve → classify → investigate → deliberate → verify → govern   (borderline)
```

The classify step determines the path. Clear cases skip investigation entirely.

## Files

```
eligibility-check/
  eligibility_check.yaml    ← fill in your criteria (template with [FILL IN] markers)
  coordinator_config.yaml   ← governance queues and SLAs
  run.py                    ← runner / embedding reference
  workflows/
    eligibility_check.yaml  ← the workflow (do not edit)
  cases/
    example_checks.json     ← 4 cases across 4 different eligibility domains
```

## Setup

```bash
pip install -e ".[runtime]"
export ANTHROPIC_API_KEY=your_key_here
python library/domain-packs/eligibility-check/run.py
```

## Embedding in your application

```python
from cognitive_core.coordinator.runtime import Coordinator

coord = Coordinator('eligibility_check/coordinator_config.yaml', db_path='eligibility.db')

case = {
    "check_type": "student_discount",
    "benefit": "50pct_student_discount",
    "claimed_status": "student",
    "account_age_days": (today - account.created_at).days,
    "account_standing": account.standing,
    "verification_document": request.document_type,
    "age": user.age,
    "region": user.region,
    # Include any pre-fetched data
    "get_account": account.to_dict(),
    "get_verification": verification.to_dict(),
}

iid = coord.start('eligibility_check', 'eligibility_check', case)
inst = coord.store.get_instance(iid)

gov_output = inst.step_outputs.get('govern_eligibility', {})
tier = gov_output.get('tier_applied', 'gate')

if tier == 'auto':
    grant_benefit(user, benefit)
elif tier in ('spot_check',):
    grant_benefit(user, benefit)
    queue_verification_check(iid)
elif tier == 'gate':
    show_pending_review(user)
    route_to_agent(gov_output.get('work_order'))
elif tier == 'hold':
    request_identity_verification(user)
```

## Customising — fill in the [FILL IN] sections

Open `eligibility_check.yaml`. Every section that requires domain knowledge is marked `[FILL IN]`.

The minimum to fill in for a working eligibility check:

1. **`classify_eligibility.categories`** — define what "clearly eligible," "borderline," and "clearly ineligible" mean for your benefit
2. **`classify_eligibility.criteria`** — how to classify from the input signals
3. **`classify_eligibility.input_context`** — which input fields to present to the classifier
4. **`verify_eligibility.rules`** — your numbered eligibility rules (these become denial reasons)
5. **`govern_eligibility.governance_context`** — when to auto-approve vs gate for review

Everything else has sensible defaults that work for most eligibility checks.

## The four example cases

| ID | Domain | Scenario | Expected |
|----|--------|----------|----------|
| ELIG-001 | Subscription upgrade | Power user, clear criteria met | clearly_eligible → AUTO |
| ELIG-002 | Student discount | Gmail address, unverified student ID | borderline → needs_verification → GATE |
| ELIG-003 | Professional discount | License verified but account suspended | clearly_ineligible → deny |
| ELIG-004 | Loyalty tier upgrade | Meets spend threshold, region restriction | borderline → eligible_conditional → SPOT_CHECK |

The same workflow handles all four domains. Only the domain scaffold changes.

## Why a warrant matters

A rules engine returns `true` or `false`. This pack returns:

```
Determination: needs_verification
Warrant: Applicant claims student status but registered email is @gmail.com 
(not .edu). Student ID photo uploaded but not yet verified. Account is 14 
days old with no purchase history. Eligibility is plausible but requires 
verified documentation before discount is granted.
```

That warrant is what the agent sees in their review queue. It's what you send the customer. It's what you produce if the decision is ever disputed.
