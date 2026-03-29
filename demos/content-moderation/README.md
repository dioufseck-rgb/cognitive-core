# Content Moderation

Shows governance as a first-class primitive — not bolted on, but
structurally part of the workflow.

## What it shows

- The `govern` primitive determining tier and disposition from accumulated evidence
- The `challenge` primitive adversarially probing borderline decisions before escalation
- Four governance outcomes: auto-approve, spot-check, gate (human review), hold
- How confidence and policy violations feed into governance decisions

## Run it

```bash
export GOOGLE_API_KEY=your-key
python demos/content-moderation/run.py
```

## The workflow

```
classify_content
    ↓
verify_policy
    ├── conforms + high confidence → govern_disposition
    └── borderline or low confidence → challenge_decision → govern_disposition
```

The `govern` step reads the accumulated state — classification, policy check,
challenge result — and produces a typed disposition:

- `auto` + `proceed` — published immediately
- `spot_check` + `proceed` — published, flagged for sampling
- `gate` + `suspend` — held, work order sent to human reviewer queue
- `hold` + `hold` — compliance hold, legal review required

## The governance contract

The `govern` output is always typed and auditable:

```python
GovernOutput(
    tier_applied=GovernanceTier.GATE,
    disposition="suspend",
    work_order=WorkOrder(
        work_type="human_review",
        target="human_reviewer",
        instructions="...",
        governance_tier=GovernanceTier.GATE,
        sla="7200",
    ),
    accountability_chain=AccountabilityChain(...)
)
```

This is what makes it institutional: every decision is explainable,
every escalation is typed, and the audit trail is built-in.

## Customise it

Change `domain.yaml` to adjust:
- `classify_content.categories` — add categories relevant to your platform
- `verify_policy.rules` — your specific policy rules
- `govern_disposition.governance_context` — the tier decision logic
- `challenge_decision.perspective` — the adversarial review framing
