# Cognitive Core

**An open-source framework for governed institutional AI workflows built from typed cognitive primitives.**

Most enterprise AI work embeds language models inside patterns inherited from earlier software: screens, request/response APIs, fixed pipelines. The intelligence is new; the surrounding architecture is old.

Cognitive Core is built differently. It treats AI-native workflow design as a first-class architectural problem, with three commitments:

- **Typed cognitive primitives** — eight epistemic operations that compose into any reasoning workflow
- **Configuration-first** — a new use case is a YAML file, not an application
- **Governance as a first-class concern** — escalation, audit trails, and human review are structural, not bolted on

→ **[Quickstart — run a governed workflow in five minutes](QUICKSTART.md)**  
→ **[Domain library — five ready-to-run institutional decision packs](library/README.md)**  
→ **[Position paper — the institutional AI design language](docs/institutional-intelligence.md)**

---

## The eight primitives

Every workflow is composed from eight typed epistemic operations. Each has a defined input contract, a structured output schema, and a prompt template.

| Primitive | Epistemic function | Key output fields |
|---|---|---|
| `retrieve` | Acquire evidence from external sources | `data`, `sources_queried`, `confidence` |
| `classify` | Categorical assignment under uncertainty | `category`, `alternative_categories`, `confidence` |
| `investigate` | Goal-directed inquiry until threshold | `finding`, `hypotheses_tested`, `confidence` |
| `challenge` | Adversarial examination of a conclusion | `survives`, `vulnerabilities`, `strengths` |
| `verify` | Conformance check against a rule set | `conforms`, `violations`, `rules_checked` |
| `deliberate` | Meta-cognitive synthesis → warranted action | `recommended_action`, `warrant`, `options_considered` |
| `generate` | Render reasoning into a communicable artifact | `artifact`, `format`, `constraints_checked` |
| `govern` | Determine governance tier and disposition | `tier_applied`, `disposition`, `work_order` |

All outputs inherit from `CognitiveOutput`: `confidence`, `reasoning`, `evidence_used`, `evidence_missing`.

---

## The three-layer architecture

Every workflow execution merges three independent configuration layers:

```
Workflow YAML      →  step sequence, transitions, loop bounds
Domain YAML        →  expertise, governance tier, evaluation criteria  
Case JSON          →  runtime data, served as typed tool calls
```

The merge is the framework's central insight: **a use case is a configuration, not an application.** No code is written per use case.

---

## Quickstart

### Install

```bash
pip install cognitive-core
# or with full runtime:
pip install cognitive-core[runtime]
```

### Write a workflow

```yaml
# configs/workflows/ticket_triage.yaml
name: ticket_triage

steps:
  - name: gather_ticket
    primitive: retrieve
    params:
      specification: "${domain.gather_ticket.specification}"

  - name: classify_severity
    primitive: classify
    temperature: 0.0
    params:
      categories: "${domain.classify_severity.categories}"
      criteria: "${domain.classify_severity.criteria}"
    transitions:
      - when: "output.category == 'critical'"
        goto: escalate
      - default: generate_response

  - name: generate_response
    primitive: generate
    params:
      requirements: "${domain.generate_response.requirements}"
      format: "${domain.generate_response.format}"
      constraints: "${domain.generate_response.constraints}"
    transitions:
      - default: __end__

  - name: escalate
    primitive: govern
    params:
      workflow_state: "${context}"
      governance_context: "${domain.escalate.governance_context}"
```

### Write a domain pack

```yaml
# configs/domains/customer_support.yaml
domain_name: customer_support
workflow: ticket_triage
governance: spot_check

gather_ticket:
  specification: |
    Pull the following:
      - get_ticket: The support ticket (subject, body, metadata)
      - get_customer: Customer account status and history
      - get_prior_tickets: Prior tickets from this customer (last 90 days)

classify_severity:
  categories: |
    - critical: Data loss, security breach, complete service outage.
      Requires immediate human response.
    - high: Core functionality broken, significant business impact.
      Same-day resolution required.
    - medium: Functionality degraded but workaround exists.
    - low: Questions, minor issues, feature requests.
  criteria: |
    Classify based on customer impact and urgency.
    When in doubt between critical and high, choose critical.
    Account status and history inform urgency but don't change severity.

generate_response:
  requirements: |
    Draft a response that acknowledges the issue, states next steps,
    and provides a realistic resolution timeline.
  format: json
  constraints: |
    - Never promise specific resolution times without escalation approval
    - Always include a case reference number
    - Match tone to severity: direct for critical, warm for low

escalate:
  governance_context: |
    Critical ticket requires immediate human review before any response
    is sent. Gate tier. Route to on-call queue.
```

### Run it

```python
from cognitive_core.coordinator import Coordinator

coord = Coordinator("configs/coordinator.yaml")

result = coord.start(
    workflow_type="ticket_triage",
    domain="customer_support",
    case_input={
        "ticket_id": "TKT-9821",
        "get_ticket": {"subject": "All data gone after update", ...},
        "get_customer": {"tier": "enterprise", ...},
    }
)
```

---

## Demos

Three demos cover the main architectural concepts, each runnable in simulated mode (no LLM key required) or with a live LLM.

### 1. Support ticket triage — quickstart

The simplest complete workflow. Two YAML files. Three primitives. Zero application code.

```bash
python demos/support-ticket-triage/run.py
```

Shows: three-layer architecture, `classify` → conditional routing → `investigate` → `generate`.

### 2. Content moderation — governance

Four posts, four governance outcomes. The `govern` primitive in action.

```bash
python demos/content-moderation/run.py
```

Shows: `classify` → `verify` → `challenge` → `govern` producing auto, spot-check, gate, and hold dispositions.

### 3. Loan application review — full chain

Six primitives, regulated decision logic, four applicant profiles.

```bash
python demos/loan-application-review/run.py
```

Shows: `retrieve` → `classify` → `investigate` → `deliberate` → `verify` → `govern`, with warranted recommendations and compliance checks.

### Fraud operations console — serious example

A complete multi-agent fraud investigation system with 9 cases, delegation chains, and a web UI.

```bash
# Mechanism validation (no LLM required)
python demos/fraud-operations/test_mechanisms.py

# Full demo with LLM + MCP data server
cd demos/fraud-operations && python fraud_data_mcp.py &
python -m cognitive_core.coordinator.cli --config coordinator_config.yaml run ...
```

---

The `demos/fraud-operations/` directory contains a complete multi-agent fraud operations workflow used in production. It demonstrates:

- **Multi-agent delegation** — a triage agent classifies alerts and routes to specialist agents (check fraud, card fraud, APP scam)
- **Governed escalation** — specialists complete investigations then gate for analyst review before any action
- **Typed delegation contracts** — investigation results flow into regulatory review and case resolution via typed work orders
- **Inspectable execution** — every step produces structured output; every governance decision is recorded in an audit chain

The fraud domain pack (`demos/fraud-operations/domains/check_fraud.yaml`) shows what institutional expertise looks like expressed as configuration:

```yaml
deliberate_determination:
  framework: |
    CONFIRMED FRAUD indicators:
    - Deliberate deposits through different channels to exploit hold differences
    - Rapid withdrawal of duplicate funds

    REQUIRED OUTPUT: Set the decision field to exactly one of these codes:
    - confirmed_fraud
    - likely_fraud
    - accidental_duplicate
    - refer_siu
```

Run the demo (no LLM required for mechanism validation):

```bash
python demos/fraud-operations/test_mechanisms.py
# 38 assertions, all pass, no LLM calls
```

---

## Governance model

Governance is not a compliance layer added after the fact. It is a structural primitive (`govern`) that every workflow can invoke. Four tiers:

| Tier | Meaning | Behavior |
|---|---|---|
| `auto` | Fully automated | Proceeds without human involvement |
| `spot_check` | Sampled review | Proceeds; flagged for post-completion sampling |
| `gate` | Mandatory review | Suspends until a human approves |
| `hold` | Compliance hold | Suspends pending compliance officer release |

Tier escalation is strictly upward. A `gate` workflow cannot be de-escalated to `auto` by any downstream step.

Every `govern` invocation produces an `AccountabilityChain` — an append-only record of every governance decision made during the instance's lifetime.

---

## Design principles

**The primitive layer is purely epistemic.** No primitive touches the world. `generate` produces action specifications; `govern` determines governance conditions; downstream systems execute. The boundary between reasoning and execution is explicit and enforced.

**Configuration is the product.** Application code is not written per use case. A new domain requires a workflow YAML and a domain YAML. The coordinator, engine, and governance pipeline are infrastructure.

**Governance is load-bearing from the start.** In AI-native architecture for regulated domains, the conditions under which a judgment can be trusted are inseparable from how it is produced and recorded. Governance is not an afterthought.

---

## Project status

Early open-source release. The primitive layer, three-layer architecture, coordinator, and governance pipeline are stable. The following are functional but marked experimental:

- `coordinator/optimizer.py` — dispatch optimization (assignment, VRP archetypes)
- `coordinator/federation.py` — multi-coordinator federation
- `coordinator/hardening.py` — DDR audit trail, partial failure handling
- `coordinator/resilience.py` — revalidation guards, staleness detection

---

## Repository layout

```
cognitive_core/        # installable package
├── primitives/        # schemas, registry, prompt templates
├── engine/            # execution engine, LLM providers, governance pipeline
├── coordinator/       # orchestration state machine, delegation, policy
├── analytics/         # artifact registry (causal DAGs, SDA policy models)
└── api/               # FastAPI server

library/               # domain library — patterns, overlays, coordinator templates
├── domain-packs/      # five ready-to-run packs (fraud, lending, moderation, triage, compliance)
├── patterns/          # five canonical workflow patterns (P02, P04, P05, P06, P07)
├── overlays/          # five composable modifiers
├── coordinator-templates/  # seven structural templates
└── README.md          # library entry point

demos/
├── fraud-operations/  # complete fraud ops example with 9 cases, 38 mechanism tests
├── content-moderation/
├── loan-application-review/
└── support-ticket-triage/

configs/               # example workflows and domain configs
docs/                  # position papers and design documents
examples/              # additional reference implementations
tests/                 # unit and integration tests
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most useful contributions right now:

- New domain packs — see [library/README.md](library/README.md) for the pattern taxonomy and how to build a pack
- New canonical patterns (P01, P03, P08–P13 are on the roadmap)
- Additional coordinator templates
- LLM provider integrations beyond Anthropic/Google/OpenAI
- Observability exporters (OpenTelemetry, Datadog)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Background

Cognitive Core grew out of research on institutional AI architecture — the claim that systems operating on judgment and context require a different descriptive vocabulary than traditional software. The theoretical foundation is documented in [the position paper](docs/cognitive_core_position_paper.md) (forthcoming).

The core thesis: a use case is a configuration, not an application. Every workflow is eight epistemic operations composed in sequence. Governance is a first-class primitive, not an infrastructure concern.
