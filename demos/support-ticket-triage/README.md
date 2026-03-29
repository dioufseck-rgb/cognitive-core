# Support Ticket Triage

The simplest complete Cognitive Core workflow. Two YAML files define
the entire system — no code.

## What it shows

- The three-layer architecture: workflow + domain + case data
- `classify` → conditional routing → `investigate` → `generate`
- How domain expertise is expressed in plain language, not code
- How changing the domain YAML changes system behaviour without touching anything else

## Run it

```bash
# From repo root
export GOOGLE_API_KEY=your-key   # or OPENAI_API_KEY with LLM_PROVIDER=openai
python demos/support-ticket-triage/run.py
```

No LLM key? It runs in simulated mode — the coordinator state machine
executes fully, just without real LLM output.

## The workflow

```
classify_severity
  ├── critical/high → investigate_issue → generate_response
  └── medium/low   →                      generate_response
```

Three primitives. One conditional branch. That's it.

## Customise it

The entire triage logic lives in `domain.yaml`. Change the severity
categories, the investigation scope, or the response format — the workflow
YAML doesn't change.

Try:
- Adding a `challenge` step after `generate_response` for adversarial review
- Adding a `govern` step with `gate` tier for enterprise customers
- Creating a second domain (`domain_b2b.yaml`) with different escalation rules
