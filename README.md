# Cognitive Core

**A framework for governed institutional AI — typed cognitive primitives, structural governance, tamper-evident audit trails.**

Most enterprise AI work embeds language models inside patterns inherited from earlier software: screens, request/response APIs, fixed pipelines. The intelligence is new; the surrounding architecture is old.

Cognitive Core treats AI-native workflow design as a first-class architectural problem:

- **Typed cognitive primitives** — eight epistemic operations that compose into any reasoning workflow
- **Configuration-first** — a new decision domain is a YAML file, not an application
- **Structural governance** — escalation, human review, and audit trails are conditions of execution, not bolt-ons
- **Demand-driven delegation** — agentic mode lets the orchestrator reason the path; the substrate enforces governance on whatever path it takes

→ **[Quickstart — run a governed workflow in five minutes](QUICKSTART.md)**  
→ **[Domain library — seven ready-to-run institutional decision packs](library/README.md)**  
→ **[DDD agentic demo — autonomous trajectory demonstration](demos/loan-hardship-agentic/README.md)**  
→ **[REST API reference](docs/api-reference.md)**  
→ **[Operational notes — maturity, assumptions, known limitations](OPERATIONAL_NOTES.md)**  

---

## The eight primitives

Every workflow is composed from eight typed epistemic operations. Each has a defined input contract, a structured output schema, a prompt template, and a three-layer epistemic state computation.

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

All outputs inherit from `CognitiveOutput`: `confidence`, `reasoning`, `evidence_used`, `evidence_missing`. Six of eight primitives also elicit `reasoning_quality` and `outcome_certainty` (all except `retrieve` and `govern` — see epistemic state section).

---

## Three-layer epistemic state

Every step produces a structured epistemic state — not a single confidence scalar:

| Layer | Signals | How computed |
|---|---|---|
| Mechanical | `evidence_completeness`, `rule_coverage`, `citation_rate`, `alternative_separation` | Deterministic from observable output structure — cannot be inflated |
| Judgment | `reasoning_quality`, `outcome_certainty` | Elicited from 6 of 8 primitive prompts with governance-aware framing (40% weight). `retrieve` and `govern` use mechanical signals only. |
| Coherence | Named flags: `CLASSIFY_DELIBERATE_MISMATCH`, `VERIFY_DELIBERATE_TENSION`, etc. | Computed cross-step — detects problems no single-step analysis can see |

A flags-first governance cascade uses this state to determine tier. A `warranted` flag provides a hard governance stop independent of the aggregate score.

---

## Two execution modes

**Workflow mode** — declare the epistemic sequence in YAML. The framework executes it with full governance, epistemic accounting, and audit ledger.

**Agentic mode** — declare available primitives, constraints, and a goal. The orchestrator reasons the sequence from evidence at runtime. The substrate enforces governance identically on whatever trajectory the orchestrator produces. The orchestrator controls the path; the substrate controls the accountability.

Both modes use the same governance model, the same epistemic state architecture, and the same tamper-evident ledger.

---

## Three-layer configuration

Every workflow execution merges three independent configuration layers:

```
Workflow YAML   →  step sequence (or available primitives + goal for agentic mode)
Domain YAML     →  expertise, governance tier, evaluation criteria, primitive configs
Case JSON       →  runtime data, served as typed tool calls
```

**A use case is a configuration, not an application.** No code is written per domain.

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
export ANTHROPIC_API_KEY=your_key   # Claude
export GOOGLE_API_KEY=your_key      # Gemini
export OPENAI_API_KEY=your_key      # OpenAI
```

### Run a domain pack

```bash
# Start the server pointed at the consumer-lending pack
CC_COORD_CONFIG=library/domain-packs/consumer-lending/coordinator_config.yaml \
CC_COORD_BASE=library/domain-packs/consumer-lending \
uvicorn cognitive_core.api.server:app --port 8000
```

Open `http://localhost:8000` — the landing page lists all instances. Submit a case at `/api/start`, then open the trace URL to watch execution live.

Each domain pack includes a `run.py` for direct command-line execution without the server.

### Run the agentic demonstration

```bash
python demos/loan-hardship-agentic/run.py
```

Two hardship cases run against the same configuration with no declared sequence. The orchestrator produces materially different trajectories for each; governance fires identically on both. See [demos/loan-hardship-agentic/README.md](demos/loan-hardship-agentic/README.md).

---

## Governance model

Four tiers. Tier escalation is strictly upward.

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
├── primitives/           — schemas, registry, eight prompt templates
├── engine/               — DEVS execution kernel, LLM providers, governance pipeline,
│                           epistemic state computation
├── coordinator/          — runtime, store, tasks, delegation, policy, resilience
├── analytics/            — artifact registry (causal DAGs, SDA policy models)
└── api/
    ├── server.py         — framework API server (CC_COORD_CONFIG env var)
    ├── main.py           — re-export of server.py
    └── trace.html        — single-source trace UI

demos/
└── loan-hardship-agentic/  — DDD agentic mode demonstration (two live cases)

library/                  — domain library
├── domain-packs/         — seven ready-to-run packs
│   ├── consumer-lending/
│   ├── content-moderation/
│   ├── clinical-triage/
│   ├── compliance-review/
│   ├── ecommerce-returns/
│   ├── eligibility-check/
│   └── fraud-investigation/
├── patterns/             — five canonical workflow patterns
├── overlays/             — five composable modifiers
└── coordinator-templates/— seven structural templates

configs/                  — default server configuration
docs/                     — architecture and API reference
tests/
├── smoke/                — 44 governance path tests (no LLM required)
└── test_devs_kernel.py   — 6 DEVS execution kernel tests
```

---

## Run the tests

```bash
pytest tests/smoke/ tests/test_devs_kernel.py
# 50 tests, ~2 minutes, no LLM calls required
```

---

## LLM provider support

Cognitive Core is designed to be provider-agnostic. The execution layer abstracts all LLM calls through a single `create_llm()` factory (`cognitive_core/engine/llm.py`) that returns a LangChain `BaseChatModel`. Switching providers requires no changes to framework code — only environment variables or `llm_config.yaml`.

**Supported providers**

| Provider | Key variable | Status |
|---|---|---|
| Google Gemini | `GOOGLE_API_KEY` | Extensively tested — primary development provider |
| Anthropic Claude | `ANTHROPIC_API_KEY` | Implemented, not yet tested end-to-end |
| OpenAI | `OPENAI_API_KEY` | Implemented, not yet tested end-to-end |
| Azure OpenAI | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` | Implemented, not yet tested end-to-end |
| Azure AI Foundry | `AZURE_AI_FOUNDRY_ENDPOINT` | Implemented, not yet tested end-to-end |
| Amazon Bedrock | `AWS_DEFAULT_REGION` | Implemented, not yet tested end-to-end |

**Known compatibility notes**

The framework handles the two most common cross-provider response shape differences:

- `response.content` as a list of blocks (Gemini multimodal, some Claude responses) is normalised to a plain string via `_extract_text()` at every LLM call site
- Transient errors (timeout, rate limit, 5xx) are retried with exponential backoff via `invoke_with_retry()` in `protected_llm_call`

Two failure modes are documented but not yet tested against non-Gemini providers:

- **Wrapper keys**: some models respond with `{"output": {...}}` instead of a bare JSON object. If seen, add `"Return a bare JSON object with no wrapper key"` to the affected primitive prompt and open a PR.
- **Schema compliance fidelity**: prompt compliance with the full JSON output contract varies across providers. The parser attempts five recovery strategies before falling back to `confidence=0.0`.

**Community contributions**

If you run Cognitive Core against a non-Gemini provider and find provider-specific parsing failures, please open an issue or PR. Include:
1. The provider and model name
2. The primitive that failed
3. The raw LLM response (sanitised of any sensitive data)
4. The fix — typically a prompt adjustment or an additional recovery strategy in `extract_json()`

## Design principles

**The primitive layer is purely epistemic.** No primitive touches the world. `generate` produces artifacts; `govern` determines governance conditions; downstream systems execute. The boundary between reasoning and execution is explicit.

**Configuration is the product.** No code is written per use case. A new domain requires a workflow YAML and a domain YAML.

**Governance is load-bearing from the start.** In regulated institutional AI, the conditions under which a judgment can be trusted are inseparable from how it is produced and recorded.

**The orchestrator controls the path; the substrate controls the accountability.** In agentic mode, autonomous trajectory selection and structural governance are not in tension — they operate at different layers.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
