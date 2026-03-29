# Cognitive Core — Framework State

> **Date:** March 2026
> **Status:** Early open-source release (v0.1.0)

---

## 1. What the Framework Is

Cognitive Core is a configuration-driven framework for governed institutional AI workflows. It decomposes any workflow into sequences of eight reusable **cognitive primitives**, each backed by a prompt template and a Pydantic output schema.

A new use case requires only:

- A **workflow YAML** — step sequence, transitions, loop controls
- A **domain YAML** — expertise injected into prompts via `${domain.*}` references
- A **case JSON** — runtime data available as tool calls

No application code is written per use case. The framework's core principle: **a use case is a configuration, not an application**.

---

## 2. Repository Layout

```
cognitive_core/              # installable package
├── primitives/              # schemas, registry, prompt templates, artifact types
├── engine/                  # execution engine, LLM providers, governance pipeline
├── coordinator/             # orchestration state machine, delegation, policy
├── analytics/               # artifact registry (causal DAGs, SDA policy models)
├── api/                     # FastAPI REST server
└── servers/                 # MCP server support

configs/                     # example coordinator config, workflows, domains, cases
demos/
└── fraud-operations/        # complete fraud operations demo (9 cases, 4 workflows)
examples/                    # additional reference domain configurations
tests/                       # unit and integration tests
```

---

## 3. Three-Layer Configuration Architecture

Every workflow execution merges three independent layers:

| Layer | Source | Purpose |
|-------|--------|---------|
| Workflow | `configs/workflows/<n>.yaml` | Step sequence, transition logic, loop bounds |
| Domain | `configs/domains/<n>.yaml` | Domain expertise, governance tier, tool specs |
| Runtime data | Case JSON | Case-specific data served as tool calls |

Domain references (`${domain.section.field}`) are resolved at merge time. Runtime references (`${step_name.field}`, `${input.field}`) are resolved at execution time by the engine's state resolver.

---

## 4. Eight Cognitive Primitives

All primitives are purely **epistemic** — none cross the epistemic/pragmatic boundary.

| Primitive | Epistemic function | Key output fields |
|-----------|-------------------|-------------------|
| `classify` | Categorical assignment under uncertainty | `category`, `confidence`, `reasoning` |
| `retrieve` | Evidence acquisition from external sources | `data`, `sources_queried`, `confidence` |
| `investigate` | Goal-directed iterative inquiry | `finding`, `hypotheses_tested`, `confidence` |
| `challenge` | Adversarial examination of a conclusion | `survives`, `vulnerabilities`, `strengths` |
| `verify` | Conformance check against a rule set | `conforms`, `violations`, `rules_checked` |
| `deliberate` | Meta-cognitive synthesis → warranted action | `recommended_action`, `warrant`, `options_considered` |
| `generate` | Render reasoning into a communicable artifact | `artifact`, `format`, `constraints_checked` |
| `govern` | Governance tier determination and disposition | `tier_applied`, `disposition`, `work_order` |

---

## 5. Governance Tiers

| Tier | Meaning | Coordinator behavior |
|------|---------|---------------------|
| `auto` | Fully automated | Proceeds without human involvement |
| `spot_check` | Sampled review | Proceeds; flagged for post-completion sampling |
| `gate` | Mandatory review | Suspends; issues WorkOrder to human queue |
| `hold` | Compliance hold | Suspends pending compliance officer release |

Tier escalation is strictly upward. Once established, a tier is locked for the life of the instance.

---

## 6. Known Gaps and Roadmap

- Production hardening modules (kill switch, PII, guardrails, semantic cache) — implemented, not all on main execution path
- Eval gate — implemented, not enforced at coordinator startup
- Service Bus task queue — SQLite is current backend; Service Bus adapter on roadmap
- Single-worker only — multi-worker requires connection pooling validation
- No OpenTelemetry/Datadog observability export
- Federation spoke instantiated, not exercised
