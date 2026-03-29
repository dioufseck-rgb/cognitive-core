# Contributing to Cognitive Core

Thank you for your interest. Cognitive Core is an early open-source release. We welcome contributions, but ask that you read this first.

## What we need most right now

**New domain packs** — the highest-value contribution. A domain pack is a workflow YAML and a domain YAML that demonstrates the framework in a new context. Good candidates:

- Healthcare triage (nurse handoff, prior auth, care coordination)
- HR workflows (candidate review, policy compliance, onboarding)
- Legal review (contract classification, clause verification, risk assessment)
- Procurement (vendor assessment, contract review, approval routing)

Look at `configs/workflows/scaffold_workflow.yaml` and `configs/domains/scaffold_domain.yaml` as starting points. Look at `demos/fraud-operations/` for a complete reference.

**New quickstart examples** in `configs/` — minimal, runnable, domain-neutral.

**LLM provider integrations** — `cognitive_core/engine/llm.py` handles provider abstraction. OpenAI and Google are supported. Azure OpenAI, Anthropic, Bedrock, and local models (Ollama) would be useful additions.

**Observability exporters** — `cognitive_core/engine/governance.py` has tracing hooks. OpenTelemetry, Datadog, and Langfuse exporters would make the framework much more useful in production.

## What we are not accepting yet

- Changes to the primitive layer contracts (`cognitive_core/primitives/schemas.py`) — these are in active theoretical development
- Changes to the coordinator state machine (`cognitive_core/coordinator/runtime.py`) without prior discussion
- New primitives — the eight-primitive taxonomy is theoretically motivated; additions require the same level of justification

If you are unsure whether something is in scope, open an issue first.

## How to contribute

### Setup

```bash
git clone https://github.com/dioufseck-rgb/cc_fraud
cd cognitive-core
pip install -e ".[dev]"
```

### Run the mechanism tests

These run without an LLM and validate the coordinator state machine:

```bash
python demos/fraud-operations/test_mechanisms.py
# All 38 assertions must pass
```

### Run unit tests

```bash
pytest tests/unit/
```

### Before opening a PR

- All 38 mechanism assertions pass
- No new imports from outside `cognitive_core.*` in package code
- New domain packs include at least one example case JSON
- New domain packs include a brief comment explaining the use case

## Issue etiquette

- **Bug reports**: include the full traceback, Python version, and a minimal reproduction
- **Feature requests**: explain the use case first, then the proposed implementation
- **Domain pack proposals**: describe the domain, the workflow structure, and why it's a good general example

## Style

We use `ruff` for linting (`ruff check .`) and have no strict formatting requirements beyond what `ruff` enforces. Type hints are welcome but not required.

## Code of conduct

Be direct and constructive. We are building infrastructure for institutional AI — precision matters more than politeness, but both are fine.
