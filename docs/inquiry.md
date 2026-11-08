# Native Cognitive Core inquiry

The inquiry command starts a **Coordinator-owned, agentic Cognitive Core
workflow**. Workflow and domain YAML select registered research profiles; the
native DEVS orchestrator chooses the next operation. The Coordinator applies
its start controls, epistemic assessment, quality gates, resource work orders,
governance tiers, persistence, and audit ledger.

This is the first minimal implementation of the
[native inquiry design](inquiry-design.md). It handles framing, ordinary
reasoning, public-web search/read, critique, revision, synthesis, and blocking
user clarification. Simulation, optimization, statistical or causal estimation,
parallel research workers, and private-document connectors are not implemented.
An inquiry can describe a needed analysis as unperformed.

For a complete PowerShell walkthrough, see
[Testing in VS Code on Windows](inquiry-vscode-windows.md).

## Install

From the cloned repository root, install the research extra and your chosen
provider SDK. For the existing OpenAI adapter:

```bash
python -m pip install -e ".[inquiry]" langchain-openai
```

Other native factory adapters include Anthropic and Google; install the SDK
you use. PDF decoding is optional: `python -m pip install -e ".[pdf]"`.

Models, aliases, token settings, and provider retries use the existing
`llm_config.yaml` and native LLM factory. Set `LLM_CONFIG_PATH` when launching
outside the checkout. Supply provider credentials and `SERPER_API_KEY` through
the process environment. There are no API-key CLI arguments and no automatic
`.env` loading. Credentials are not part of the Study or its model prompts.

## Run the native offline demonstration

```bash
python -m cognitive_core.inquiry demo --output inquiries/native-demo
python -m cognitive_core.inquiry status inquiries/native-demo
python -m cognitive_core.inquiry report inquiries/native-demo
```

Expected result: **Coordinator status `completed`, research outcome
`conditional`, 12 accepted operations, 24 model attempts, one search, one read.**
Model and web responses are synthetic fixtures; no external API calls occur.
The actual Coordinator, DEVS engine, protected-call pipeline, profile nodes,
evidence validation, SQLite transactions, epistemic calculation, and governance
run normally. This verifies integration, not the research quality of a live LLM.

The fictional policy requires a receipt and, above 100 credits, prior approval.
The first finding incorrectly treats a receipt as sufficient. Verify and
Challenge identify the overclaim; Reflect directs a material correction. The
revised finding is integrated, rendered, reviewed, and governed.

### Exercise suspension and continuation

```bash
python -m cognitive_core.inquiry demo --clarify --output inquiries/native-clarify
python -m cognitive_core.inquiry status inquiries/native-clarify
python -m cognitive_core.inquiry resume inquiries/native-clarify \
  --answer "c1=Yes, restrict the answer to the fictional policy."
```

The first command suspends on a native clarification work order. The last
command creates a fresh Coordinator, commits the answer through its typed
resource contract, and continues from the stored frame. It does not rerun
accepted operations. Multiple blocking questions can be answered with repeated
`--answer ID=TEXT`; execution continues after the last required answer.

## Start a live inquiry

After configuring the model-provider key and `SERPER_API_KEY`:

```bash
python -m cognitive_core.inquiry start \
  "For an internal assistant that searches policy documents and drafts cited answers, when is a fixed retrieve-then-generate workflow sufficient, and when does model-directed multi-step retrieval improve results?" \
  --purpose "Develop practical architecture criteria for a read-only pilot using public sources" \
  --provider openai --model default \
  --output inquiries/policy-retrieval-native
```

Use a provider model ID accessible to your account if the configured alias is
unsuitable. `--context-file notes.md` adds supplied UTF-8 context (up to 32,000
characters in this CLI). A live model chooses its own path; it need not reproduce
the fixture's sequence. `--quiet` suppresses Coordinator progress messages.

The CLI is synchronous. Leave its terminal running until it completes or
suspends. Choose a new output directory for each new inquiry; existing runs are
never overwritten.

## What executes, and where to configure it

| Layer | Shipped configuration or implementation | Responsibility |
|---|---|---|
| Coordinator | [`coordinator.yaml`](../cognitive_core/inquiry/config/coordinator.yaml) | Local tools, quality gates, governance tiers, clarification capability/contract |
| Workflow | [`inquiry.yaml`](../cognitive_core/inquiry/config/workflows/inquiry.yaml) | Agentic execution profile, allowed primitives, budgets, completion requirements |
| Domain | [`reasoning_research.yaml`](../cognitive_core/inquiry/config/domains/reasoning_research.yaml) | Framing, source selection, critique, stopping guidance, operation profiles and instructions |
| Orchestration | `engine/agentic_devs.py::OrchestratorStep` | Typed choice of configured operation and Study/question/report target |
| Native node extension | `engine/profile_nodes.py` | Protected calls, exact response schema, bounded correction, registered tools |
| Research data | `inquiry/native_models.py` and `native_profiles.py` | Contracts, passage identities, pure Study reducers, bounded context projection |
| Lifecycle | `coordinator/runtime.py` and `store.py` | Checkpoints, accepted-step transactions, resource fulfillment, resume, final governance |

The CLI loads the packaged YAML by default. To customize it, copy the **whole
config directory**, retain its `workflows/` and `domains/` layout, and pass
`--config path/to/coordinator.yaml`. `--domain` selects a domain file in that
pack. Operation names, primitive identity, and `output_profile` must agree;
unknown profiles fail explicitly. The effective workflow/domain configuration
is saved with the Study. Resume uses that saved research configuration.

The shipped domain configures two operations using the same native Retrieve
profile: `discover_sources` permits `web_search`, while `read_sources` permits
`read_source`. Their `allowed_tools` and `require_new_tool_call: true` parameters
are enforced before dispatch. Empty or entirely cached plans cannot be accepted
as new research by these operations. The coordinator's specific purpose for
the operation is passed to its primitive, alongside the configured instructions.

Before each decision, the orchestrator receives the currently valid
operation/target pairs and their remaining repetition counts, calculated from
the complete accepted history. It also sees research progress and recent tool
outcomes. A rejected decision receives bounded, cumulative correction feedback;
correction calls bypass the response cache so neither exact nor semantic cache
hits can replay the rejected response. They retain the shared protected-call
controls and consume the model-attempt allocation. Final report review and
governance constraints apply to this available menu.

Pulling updated YAML does **not** replace a running study's saved configuration.
Runtime fixes apply when an older study resumes, but an existing
`research_sources` operation retains its original parameters and name. Start a
new study to use the revised default domain strategy. Recovery never clears
accepted-operation counts, consumed budgets, or recorded evidence.

The public entry points are `python -m cognitive_core.inquiry` and
`cognitive-inquiry`. The earlier `InquiryRunner`, its `cli.py`, and legacy tests
remain for compatibility and diagnosis. Public commands do not execute that
scheduler. Legacy JSON checkpoints are not automatically migrated; start a
new native run after updating.

## Persistence and evidence

| File in an output directory | Meaning |
|---|---|
| `coordinator.db` | Authoritative local SQLite state: instance, canonical Study, exact read passages, work orders/tasks, audit events |
| `run.json` | Convenience manifest identifying the instance and configuration |
| `state.json` | Inspection export; editing it cannot alter continuation |
| `report.md` | Human-readable report or provisional findings export |

The Coordinator checkpoints each accepted operation and records model/tool
reservations and actual research results. JSON/Markdown exports refresh when
the CLI returns. `status` and `report` read the database without creating an
LLM or making search calls. `report` can therefore show the latest durable
partial state after an interrupted process.

The report includes recent accepted operation names and targets, their actual
search/read/cache/failure counts, and recent contract-rejection messages. This
distinguishes source discovery, source reading, and cached or empty work when a
study stalls, without exposing raw rejected model payloads in the report.

Claims are classified as source-based, inference, assumption, or user-supplied.
Source claims must cite stored passages and quote text present in them. Reports
can cite only passages in the current synthesis. Changed source content gets
a new source identity, preserving old citations. Search snippets never become
read passages. Final review is bound to the current report and evidence; changes
invalidate it. Passing a branch Challenge cannot authorize Govern.

These are structural provenance guarantees. Whether a passage substantively
supports a claim still requires assessment and review. Model confidence and
agreement are not independent evidence.

## Limits and stopping

The default pack allows 32 accepted operations, 96 model attempts, eight
searches, and 12 source reads. It allows at most two output corrections per
operation and six repetitions of the same configured operation/target.
The final four operations or 12 remaining model attempts trigger closeout,
reserving effort for synthesis, report, final Challenge, and Govern.

Override allocations at start:

```bash
python -m cognitive_core.inquiry start "Your question" \
  --output inquiries/bounded-study --provider openai \
  --max-steps 24 --max-model-calls 72 --max-searches 4 --max-reads 6
```

Model accounting includes orchestration, schema correction, and retries at the
native protected provider boundary. A cache-served operation conservatively
consumes one reservation. These are attempt allocations, not dollar or token
caps; SDK internals and source redirects can involve additional HTTP activity.
Duplicate successful tool requests reuse their stored result.

Closeout does not guarantee success: the report still needs review and
governance. Exhausted retries, malformed output, a missing model, invalid
targets, and deadlines cannot silently become completed answers. This minimal
slice has no post-start budget enlargement command.

### Interpret lifecycle and research status separately

| Coordinator status | Meaning |
|---|---|
| `completed` | Native execution and release governance completed. Assess the research outcome and limitations separately. |
| `suspended` | A clarification, governance review, or other native work order needs a response. |
| `failed` | Execution could not continue; accepted findings and consumed usage remain checkpointed. |
| `running` | A worker is executing, or its process stopped without closing the instance. |

The Study's research outcome is separate: `open`, `answered`, `conditional`,
`evidence_limited`, or `budget_exhausted`. A completed, honestly qualified report
can have a limited research outcome. A failed run can contain useful findings,
but they are not a released final answer.

## Clarification, recovery, and governance

```bash
python -m cognitive_core.inquiry resume inquiries/policy-retrieval-native \
  --answer "c1=Public evidence for a read-only pilot; humans make consequential decisions."
```

Use the actual ID shown in `status`. The resource handler validates the answer
against a current clarification and commits the answer, task/work-order result,
and Study change together. Identical repeated fulfillment is idempotent; a
different answer cannot silently overwrite a fulfilled request.

After fixing a provider/configuration failure, use `resume DIRECTORY`.
For a `running` instance whose previous process has **definitely stopped**:

```bash
python -m cognitive_core.inquiry resume inquiries/policy-retrieval-native --recover-running
```

This local recovery preserves accepted steps, passages, and consumed allocations.
It may repeat an external request whose result was never committed. This version
does not implement distributed leases or exactly-once HTTP calls. Never recover
a run while its original process is still working.

A governance suspension is distinct from a research clarification. Inspect it
through the native Coordinator using the same database and configuration:

```bash
python -m cognitive_core.coordinator.cli \
  --config cognitive_core/inquiry/config/coordinator.yaml \
  --db inquiries/policy-retrieval-native/coordinator.db pending
```

After actually reviewing the report, use that CLI's `approve INSTANCE_ID
--approver "Your name" --notes "Review rationale"`. Clarification answers and
generic recovery cannot replace governance approval. This minimal pack uses
local SQLite tasks; hosted queues and distributed workers remain future work.

## Verification

```bash
python -m unittest discover -s tests/unit -p "test_native_inquiry*.py" -q
python -m unittest discover -s tests/unit -p "test_*.py" -q
```

Native tests exercise the real execution path with model/tool boundary fixtures:
framing, revisions, provenance, malformed-output recovery, budgets, stale/late
results, atomic acceptance, final-review binding, protected calls, clarification
across a fresh Coordinator, approval separation, and recovery. The broader suite
checks compatibility with existing framework behavior.

Progress regressions exercise a repeated unavailable choice with the real
governance cache enabled, a history longer than the prompt's recent-operation
window, and recovery of a failed SQLite study that retains the older combined
`research_sources` configuration. They require continuation through a real
source read and the final report's Challenge and Govern.

Before relying on research results, separately evaluate live runs for coverage,
source relevance, entailment, useful counterarguments, and calibrated stopping.
Passing deterministic integration tests does not establish those qualities.
