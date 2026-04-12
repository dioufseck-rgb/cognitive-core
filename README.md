# Cognitive Core

**A framework for governed institutional AI — typed cognitive primitives, structural governance, tamper-evident audit trails.**

Most enterprise AI work embeds language models inside patterns inherited from earlier software: screens, request/response APIs, fixed pipelines. The intelligence is new; the surrounding architecture is old.

Cognitive Core treats AI-native workflow design as a first-class architectural problem:

- **Typed cognitive primitives** — nine epistemic operations that compose into any reasoning workflow
- **Configuration-first** — a new decision domain is a YAML file, not an application
- **Structural governance** — escalation, human review, and audit trails are conditions of execution, not bolt-ons
- **Demand-driven delegation** — agentic mode lets the orchestrator reason the path; the substrate enforces governance on whatever path it takes

→ **[Quickstart — run a governed workflow in five minutes](QUICKSTART.md)**  
→ **[arXiv paper — Governed Reasoning for Institutional AI](https://arxiv.org/abs/PLACEHOLDER)**  

---

## The nine primitives

Every workflow is composed from nine typed epistemic operations. Each has a defined input contract, a structured output schema, a prompt template, and a three-layer epistemic state computation.

| Primitive | Epistemic function | Key output fields |
|---|---|---|
| `retrieve` | Acquire evidence from external sources | `data`, `sources_queried`, `confidence` |
| `classify` | Categorical assignment under uncertainty | `category`, `alternative_categories`, `confidence` |
| `investigate` | Goal-directed inquiry until threshold | `finding`, `hypotheses_tested`, `confidence` |
| `verify` | Conformance check against a rule set | `conforms`, `violations`, `rules_checked` |
| `challenge` | Adversarial examination of a conclusion | `survives`, `vulnerabilities`, `strengths` |
| `reflect` | Metacognitive synthesis over accumulated reasoning state | `trajectory`, `revision_target`, `next_question` |
| `deliberate` | Synthesis toward a warranted conclusion | `recommended_action`, `warrant`, `options_considered` |
| `generate` | Render reasoning into a communicable artifact | `artifact`, `format`, `constraints_checked` |
| `govern` | Determine governance tier and disposition | `tier_applied`, `disposition`, `work_order` |

All outputs inherit from `CognitiveOutput`: `confidence`, `reasoning`, `evidence_used`, `evidence_missing`. Six of nine primitives also elicit `reasoning_quality` and `outcome_certainty` — all except `retrieve` (quality measured mechanically) and `govern` (reads accumulated record rather than producing first-order reasoning). The `reflect` primitive reports `trajectory` and `revision_target` rather than scalar quality fields — its governance contribution is structural, not numeric.

---

## Three-layer epistemic state

Every step produces a structured epistemic state — not a single confidence scalar:

| Layer | Signals | How computed |
|---|---|---|
| Mechanical | `evidence_completeness`, `rule_coverage`, `citation_rate`, `alternative_separation` | Deterministic from observable output structure — cannot be inflated |
| Judgment | `reasoning_quality`, `outcome_certainty` | Elicited from 6 of 9 primitive prompts with governance-aware framing (40% weight) |
| Coherence | Named flags: `CLASSIFY_DELIBERATE_MISMATCH`, `VERIFY_DELIBERATE_TENSION`, etc. | Computed cross-step — detects problems no single-step analysis can see |

A flags-first governance cascade uses this state to determine tier. A `warranted` flag provides a hard governance stop independent of the aggregate score.

---

## Two execution modes

**Workflow mode** — declare the epistemic sequence in YAML. The framework executes it with full governance, epistemic accounting, and audit ledger.

**Agentic mode** — declare available primitives, constraints, and a goal. The orchestrator reasons the sequence from evidence at runtime. Hard constraints (`must_include`, `max_steps`, `must_end_with`) are enforced by the substrate — not communicated to the orchestrator as preferences it might override. The substrate controls the accountability.

Both modes use the same governance model, the same epistemic state architecture, and the same tamper-evident ledger.

---

## Three-layer configuration

Every workflow execution merges three independent configuration layers:

```
Workflow YAML   →  step sequence (or available primitives + goal for agentic mode)
Domain YAML     →  expertise, governance tier, evaluation criteria, primitive configs
Case JSON       →  runtime data, served as typed tool calls
```

**A use case is a configuration, not an application.** No code is written per domain. Deploying a new institutional decision domain requires a workflow YAML, a domain YAML, and case JSON files — no framework changes.

---

## Quickstart

### Install

```bash
git clone https://github.com/dioufseck-rgb/cognitive-core.git
cd cognitive-core
pip install -e .
```

Set an API key:

```bash
export GOOGLE_API_KEY=your_key      # Gemini (primary development provider)
export ANTHROPIC_API_KEY=your_key   # Claude
export OPENAI_API_KEY=your_key      # OpenAI
```

### Run the prior authorization appeal demo

```bash
cd demos/prior-auth-appeal
python run.py --case pa_2024_a001
```

### Run the benchmark (replicates paper results)

```bash
cd demos/prior-auth-appeal
python run_benchmark.py          # runs all 11 benchmark cases
python score_benchmark.py        # scores results against ground truth
python compare_benchmark.py      # compares CC vs ReAct vs Plan-and-Solve
```

### Run the loan modification demo

```bash
cd demos/loan-modification
python run.py
```

---

## Governance model

Four tiers. Tier escalation is strictly upward and locks for the instance lifetime.

| Tier | Meaning | Behavior |
|---|---|---|
| `auto` | Fully automated | Proceeds without human involvement |
| `spot_check` | Sampled review | Proceeds; flagged for post-completion sampling |
| `gate` | Mandatory review | Suspends until a human approves |
| `hold` | Compliance hold | Suspends pending compliance officer release |

Every `govern` invocation produces a work order recorded in the tamper-evident SHA-256 hash chain ledger. The audit trail is endogenous to the computation.

---

## Repository layout

```
cognitive_core/           — installable package
├── primitives/           — schemas, registry, nine prompt templates
├── engine/               — DEVS execution kernel, LLM providers, governance pipeline,
│                           epistemic state computation
├── coordinator/          — runtime, store, tasks, delegation, policy, resilience
└── api/
    ├── server.py         — framework API server
    └── trace.html        — single-source trace UI

demos/
├── prior-auth-appeal/    — benchmark domain: 26 cases, documents, domain config,
│   ├── cases/            — run.py, run_benchmark.py, score_benchmark.py, compare_benchmark.py
│   ├── documents/
│   ├── domains/
│   └── workflows/
└── loan-modification/    — second demonstrated domain
    ├── cases/
    ├── documents/
    ├── domains/
    └── workflows/

tests/
├── unit/                 — 153 unit tests (reflect, epistemic state, prompt behavior)
├── smoke/                — 44 governance path tests (no LLM calls required)
└── test_devs_kernel.py   — 6 DEVS execution kernel tests
```

---

## Run the tests

```bash
pytest tests/unit/ tests/smoke/ tests/test_devs_kernel.py
# 203 tests, no LLM calls required
```

---

## LLM provider support

Cognitive Core is provider-agnostic. The execution layer abstracts all LLM calls through a single `create_llm()` factory (`cognitive_core/engine/llm.py`).

| Provider | Key variable | Status |
|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | Extensively tested — primary development provider |
| Anthropic Claude | `ANTHROPIC_API_KEY` | Implemented, not yet tested end-to-end |
| OpenAI | `OPENAI_API_KEY` | Implemented, not yet tested end-to-end |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | Implemented, not yet tested end-to-end |
| Azure AI Foundry | `AZURE_AI_FOUNDRY_ENDPOINT` | Implemented, not yet tested end-to-end |
| Amazon Bedrock | `AWS_DEFAULT_REGION` | Implemented, not yet tested end-to-end |

---

## Design principles

**The primitive layer is purely epistemic.** No primitive touches the world. `generate` produces artifacts; `govern` determines governance conditions; downstream systems execute. The boundary between reasoning and execution is explicit.

**Configuration is the product.** No code is written per use case. A new domain requires a workflow YAML and a domain YAML.

**Governance is load-bearing from the start.** In regulated institutional AI, the conditions under which a judgment can be trusted are inseparable from how it is produced and recorded.

**The orchestrator controls the path; the substrate controls the accountability.** In agentic mode, autonomous trajectory selection and structural governance are not in tension — they operate at different layers.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
