# Cognitive Core

**A framework for governed institutional AI — typed cognitive primitives, structural governance, tamper-evident audit trails.**

Most enterprise AI work embeds language models inside patterns inherited from earlier software: screens, request/response APIs, fixed pipelines. The intelligence is new; the surrounding architecture is old.

Cognitive Core treats AI-native workflow design as a first-class architectural problem:

- **Typed cognitive primitives** — nine epistemic operations that compose into any reasoning workflow
- **Configuration-first** — a new decision domain is a YAML file, not an application
- **Structural governance** — escalation, human review, and audit trails are conditions of execution, not bolt-ons
- **Demand-driven delegation** — agentic mode lets the orchestrator reason the path; the substrate enforces governance on whatever path it takes

→ **[Prior authorization appeal — the evaluated domain](demos/prior-auth-appeal/)**  
→ **[Loan modification — second demonstrated domain](demos/loan-modification/)**  
→ **[Reproducing the IEEE Computer results](#reproducing-the-published-results)**  

---

## The nine primitives

Every workflow is composed from nine typed epistemic operations. Each has a defined input contract, a structured output schema, a prompt template, and a three-layer epistemic state computation.

| Primitive | Epistemic function | Key output fields |
|---|---|---|
| `retrieve` | Acquire evidence from external sources | `data`, `sources_queried`, `confidence` |
| `classify` | Categorical assignment under uncertainty | `category`, `alternative_categories`, `confidence` |
| `investigate` | Goal-directed inquiry until threshold | `finding`, `hypotheses_tested`, `confidence` |
| `challenge` | Adversarial examination of a conclusion | `survives`, `vulnerabilities`, `strengths` |
| `verify` | Conformance check against a rule set | `conforms`, `violations`, `rules_checked` |
| `deliberate` | Meta-cognitive synthesis → warranted action | `recommended_action`, `warrant`, `options_considered` |
| `generate` | Render reasoning into a communicable artifact | `artifact`, `format`, `constraints_checked` |
| `reflect` | Metacognitive review of the reasoning so far; directs the next step | `what_was_established`, `what_was_assumed`, `trajectory`, `next_question` |
| `govern` | Determine governance tier and disposition | `tier_applied`, `disposition`, `work_order` |

All outputs inherit from `BaseOutput`: `confidence`, `reasoning`, `evidence_used`, `evidence_missing`. Seven of the nine primitives also elicit `reasoning_quality` and `outcome_certainty`. The two exceptions are `retrieve` and `govern` (see the epistemic state section).

`reflect` reasons about the accumulated reasoning, not about the case itself. It lists what has been established and what has been assumed, identifies the single most load-bearing fact, and sets a trajectory: `continue`, `revise` (with a revision target), or `escalate` (with a reason). Its output can shape the specification of the next primitive call. Schema: `ReflectOutput` in `cognitive_core/primitives/schemas.py`; prompt: `cognitive_core/primitives/prompts/reflect.txt`.

---

## Three-layer epistemic state

Every step produces a structured epistemic state — not a single confidence scalar:

| Layer | Signals | How computed |
|---|---|---|
| Mechanical | `evidence_completeness`, `rule_coverage`, `citation_rate`, `alternative_separation` | Deterministic from observable output structure — cannot be inflated |
| Judgment | `reasoning_quality`, `outcome_certainty` | Reported by the LLM in 7 of the 9 primitives. `retrieve` and `govern` use mechanical signals only. |
| Coherence | Six named flags (below) | Computed across steps by the framework. Detects inconsistencies that no single step can see. |

**How the step score is computed** (`cognitive_core/engine/epistemic.py`, `compute_overall`):

```
combined = 0.6 × mean(mechanical) + 0.4 × mean(judgment)     # mechanical only, if no judgment signals
overall  = combined × coherence_multiplier
coherence_multiplier = max(0.3, 1.0 − Σ flag penalties)
warranted = overall ≥ 0.5  and  no critical flag present
```

Judgment signals carry less weight because they are self-reported. Mechanical signals are computed from the structure of the output.

| Coherence flag | Condition | Penalty | Critical |
|---|---|---|---|
| `CLASSIFY_DELIBERATE_MISMATCH` | Recommended action inconsistent with the classification | 0.20 | yes |
| `VERIFY_DELIBERATE_TENSION` | Verify found violations; deliberate still recommends approval | 0.25 | yes |
| `CONFIDENCE_DROP` | Step confidence fell by more than 0.25 from the prior step | 0.10 | no |
| `UNRESOLVED_EVIDENCE_GAPS` | Missing evidence from retrieve/investigate never addressed | 0.10 | no |
| `GOVERN_ESCALATION_UNEXPLAINED` | Govern tier higher than deliberate confidence suggests | 0.15 | no |
| `UNWARRANTED_RECOMMENDATION` | Deliberate recommendation has no warrant | 0.15 | no |

A flags-first governance cascade uses this state to set the tier. Domain YAML can declare gate triggers on these signals (for example, `not_warranted` or a named coherence flag). The `warranted` flag acts as a hard governance stop, independent of the aggregate score.

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

### Run a demonstrated domain

Two domains are included, both in agentic mode with all nine primitives available.

```bash
# Prior authorization appeal (the domain evaluated in the paper)
python demos/prior-auth-appeal/run.py

# Loan modification
python demos/loan-modification/run.py --case lm_2024_a001.json --compare
```

### Run the server

```bash
CC_COORD_CONFIG=demos/prior-auth-appeal/coordinator_config.yaml \
CC_COORD_BASE=demos/prior-auth-appeal \
uvicorn cognitive_core.api.server:app --port 8000
```

Open `http://localhost:8000`. Submit a case at `/api/start`, then open the trace URL to follow execution.

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

## Reproducing the published results

The results in *Governance by Design: Architectural Requirements for Institutional AI* (IEEE Computer) were produced in the prior authorization appeal domain. The evaluation artifacts (five replications, run 8–9 August 2026 with `gemini-3.5-flash`, locked configuration in `llm_config.yaml` and `demos/prior-auth-appeal/domains/prior_auth_appeal.yaml`) are committed under `demos/prior-auth-appeal/output/`.

The original evaluation commit is tagged `comsi-2026-eval`. The tag cited in the revised manuscript is `comsi-2026-r2`; the manuscript gives its full commit SHA. To regenerate every reported number from the committed outputs, without any LLM calls:

```bash
git checkout comsi-2026-r2
cd demos/prior-auth-appeal
python aggregate_replications.py output/replication_1 output/replication_2 \
    output/replication_3 output/replication_4 output/replication_5
```

The script re-derives ground truth from `cases/*.json` and reports modal accuracy, per-run ranges, pooled and silent errors with the exact (Clopper–Pearson) one-sided 95% upper bound, and the governance tier distribution. Ground-truth labels were constructed by the authors and checked by internal review.

To re-run the benchmark itself (requires `GOOGLE_API_KEY`): `python demos/prior-auth-appeal/run_benchmark.py`. Scoring of determination text is in `score_benchmark.py`.

---

## Repository layout

```
cognitive_core/           — installable package
├── primitives/           — schemas, registry, nine primitive prompt templates + orchestrator
├── engine/               — DEVS execution kernel, LLM providers, governance pipeline,
│                           epistemic state computation
├── coordinator/          — runtime, store, tasks, delegation, policy, resilience
├── analytics/            — artifact registry (causal DAGs, SDA policy models)
└── api/
    ├── server.py         — framework API server (CC_COORD_CONFIG env var)
    ├── main.py           — re-export of server.py
    └── trace.html        — single-source trace UI

demos/
├── prior-auth-appeal/    — evaluated domain: cases, domain and workflow YAML,
│                           benchmark runner, scorer, aggregation script,
│                           committed replication outputs
└── loan-modification/    — second demonstrated domain

llm_config.yaml           — provider and model configuration
tests/
├── smoke/                — governance path tests (no LLM required)
├── unit/                 — epistemic state, reflect primitive, kill switch, eval gate
└── test_devs_kernel.py   — DEVS execution kernel tests
```

---

## Run the tests

```bash
pytest tests/smoke/ tests/unit/ tests/test_devs_kernel.py
# no LLM calls required
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

**The primitive layer is purely epistemic.** No primitive touches the world. `generate` produces artifacts; `reflect` directs the next step; `govern` determines governance conditions; downstream systems execute. The boundary between reasoning and execution is explicit.

**Configuration is the product.** No code is written per use case. A new domain requires a workflow YAML and a domain YAML.

**Governance is load-bearing from the start.** In regulated institutional AI, the conditions under which a judgment can be trusted are inseparable from how it is produced and recorded.

**The orchestrator controls the path; the substrate controls the accountability.** In agentic mode, autonomous trajectory selection and structural governance are not in tension — they operate at different layers.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
