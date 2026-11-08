# Research-assisted inquiry

For a complete Windows setup and testing walkthrough, see
[Testing in VS Code on Windows](inquiry-vscode-windows.md).

The inquiry runner explores a question through framing, web research, reasoning,
challenge, revision, and synthesis. It saves the working question, subquestions,
findings, source passages, clarifications, and progress in a local directory.

This first version is for questions that can progress through ordinary reasoning
and published or supplied evidence: conceptual questions, architecture choices,
argument development, comparisons, and literature-supported explanations.
Simulation, optimization, experiment execution, and statistical or causal
estimation are outside this version's scope.

## Install

Run the commands below from the cloned repository root. Install the inquiry
dependencies and the SDK for your chosen model provider. For example, with the
existing OpenAI provider:

```bash
python -m pip install -e ".[inquiry]" langchain-openai
```

Other existing provider choices include `langchain-google-genai` and
`langchain-anthropic`. You only need the SDK you intend to use. PDF reading is
available when `pdfplumber` is installed:

```bash
python -m pip install -e ".[pdf]"
```

The runner uses Cognitive Core's existing `create_llm` factory, model aliases,
and protected model-call path. It reads the repository's `llm_config.yaml`; if
launching elsewhere, set `LLM_CONFIG_PATH` to that file. Use `--provider` to
select a provider explicitly and `--model` to choose an alias or provider model
ID. `--model default` uses the existing configuration and its environment
override, `LLM_DEFAULT_MODEL`.

Supply keys through the process environment or your secret-management setup.
For example, in a Bash session:

```bash
export SERPER_API_KEY="<your Serper key>"
export OPENAI_API_KEY="<your model-provider key>"
```

There are no API-key command-line arguments. The Serper adapter reads
`SERPER_API_KEY` directly; the key is not an inquiry input and is not included
in model prompts, source records, or checkpoints. Search requires a Serper key
in addition to the credential for your LLM provider.

## Try the offline demonstration

```bash
python -m cognitive_core.inquiry demo --output inquiries/offline-demo
python -m cognitive_core.inquiry status inquiries/offline-demo
python -m cognitive_core.inquiry report inquiries/offline-demo
```

The demo uses scripted model responses and synthetic source fixtures. It makes
no model or search API calls and requires no keys. It demonstrates the control
flow, evidence links, revision, and report generation; its findings are not
empirical research or a measurement of model quality. Choose a fresh output
directory for each demo run.

## Start a real inquiry

```bash
python -m cognitive_core.inquiry start \
  "When is an autonomous agent a useful abstraction for a business workflow?" \
  --purpose "Develop criteria for architecture decisions" \
  --output inquiries/agent-abstraction \
  --provider openai \
  --model default
```

Use a UTF-8 text or Markdown file to supply relevant material:

```bash
python -m cognitive_core.inquiry start \
  "Which parts of this workflow should use autonomous reasoning?" \
  --context-file workflow-notes.md \
  --output inquiries/workflow-design \
  --provider openai \
  --max-steps 80 \
  --max-searches 12 \
  --max-reads 18
```

Progress goes to standard error; the final status summary goes to standard
output. `--quiet` suppresses step progress. The command refuses to overwrite an
existing inquiry. The interface is a synchronous CLI and Python runner; there
is no background service or browser UI in this version.

### What the runner does

| Operation | Responsibility |
|---|---|
| **Frame** | Refine the question, establish purpose and answer criteria, propose subquestions and dependencies, and identify evidence needs or consequential clarifications. |
| **Retrieve** | Plan explicit tool calls, search with Serper, and read selected public sources. |
| **Investigate** | Develop findings and distinguish source-supported claims, inferences, assumptions, and user-supplied statements. |
| **Challenge** | Identify material weaknesses and counterexamples; accept an answer when no consequential objection is found. |
| **Reflect** | Choose whether to accept, revise, research further, reframe, ask the user, or leave an issue unresolved. |
| **Deliberate** | Integrate the branches, reconcile contradictions, and preserve qualifications. |
| **Generate** | Produce the final Markdown answer using the available findings and passage citations. |

The frame is provisional. Research or a substantive objection can change it.
Completing one branch does not complete the inquiry: the controller assembles
and reviews the overall synthesis separately. Research can recur after a
challenge; it is not limited to an initial search pass.

Independent branches receive their first reviewed round before another branch
gets a second round. Genuine dependencies still wait for their prerequisites.
Reflection receives recent findings, prior review decisions, and the branch's
remaining round/planning allowances so it can compare revisions rather than
assess each draft in isolation.

Retrieval plans retain search-result ranking within each search. Duplicate,
cached, and exhausted requests are filtered with explicit feedback. After one
corrective attempt, another plan containing no new feasible calls ends that
research pass with the evidence gap preserved. If an initial branch actually
searched but tries to proceed without any read evidence or read attempt, the
controller may read one ranked discovered candidate within the remaining
budget. That fallback is recorded; ranking does not establish source quality,
and a successful read does not establish the claim.

The inquiry controller uses Cognitive Core's typed primitive contracts and
inquiry-specific research profiles. Its stopping criteria are inquiry controls,
not an institutional approval decision. This runner does not claim the full
coordinator's governance, human work-order, or tamper-evident audit lifecycle;
it does not substitute a successful challenge for `Govern` authorization.

## Answer a clarification and resume

If the next useful step needs your judgment or missing context, the command
saves the inquiry with status `awaiting_user` and displays clarification IDs.

```bash
python -m cognitive_core.inquiry status inquiries/agent-abstraction
python -m cognitive_core.inquiry resume inquiries/agent-abstraction \
  --answer "C1=Internal document-review workflows with a human final decision" \
  --provider openai \
  --model default
```

Use the IDs shown by `status`; `C1` above is an example. Repeat `--answer` to
answer more than one clarification. The answer may contain `=` characters.
Unknown IDs, empty answers, and duplicate IDs are rejected before execution.
The saved answers become part of the inquiry context.

The runtime model and provider are selected per invocation. Supply the same
`--provider` and `--model` when resuming, or intentionally choose a different
model. `status` and `report` only read local files and never instantiate a model
or search client.

## Budgets, checkpoints, and interrupted runs

The CLI defaults to 80 steps, 12 searches, and 18 source reads. These are ceilings,
not a required number of iterations. The runner also bounds reasoning rounds,
reframing, retrieval planning, and evidence context. Python callers can adjust
these through `InquiryLimits`.

Steps count scheduled primitive invocations and uncached retrieval operations;
search/read usage includes failed attempts. These are scheduling limits, not
token or spending limits. Model-provider retries and source redirects can make
additional HTTP requests within an operation.

Invalid JSON or a response that fails its output schema can receive up to two
corrective model attempts (`InquiryLimits.max_output_repairs`). Every attempt,
including rejected outputs, consumes a scheduled step. The original schema is
still enforced: the runner never truncates an oversized tool plan, guesses
arguments, or executes a partially valid plan. Correction prompts receive the
same task and safe field/type diagnostics, without replaying the rejected raw
response. Provider/configuration failures and URL/citation/evidence violations
are not handled as output-format repairs.

The checkpoint's `output_repair` field preserves safe feedback and attempt
counts across budget pauses and interruptions. An explicit `resume` after a
failed run starts a new bounded correction batch, allowing a changed model or
configuration to recover. Reports display rejected-output and correction counts;
these are scheduled attempts, not a count of HTTP requests or provider retries.

When the step budget is exhausted, the checkpoint and available partial results
remain inspectable. Add a specific amount of effort when resuming:

```bash
python -m cognitive_core.inquiry resume inquiries/agent-abstraction \
  --additional-steps 20 \
  --additional-searches 3 \
  --additional-reads 5 \
  --provider openai
```

An added budget is cumulative; it does not reset usage. The maximum supported
per-inquiry totals are 1,000 steps, 200 searches, and 300 reads. Reaching a limit
does not turn an unresolved question into a supported conclusion.
Adding steps does not reopen a branch already closed as unresolved at its
separate revision limit. A fresh inquiry is appropriate when the frame or the
research strategy needs to be reconsidered; retain the old directory for
comparison.

The directory contains:

| File | Content |
|---|---|
| `state.json` | The versioned frame, question states, findings and reviews, sources and passages, search history, clarifications, budgets, and event history. |
| `report.md` | The latest readable report, including a final answer when completed or available partial results otherwise. |

The report displays each branch's claim classifications, available read-passage
references, latest critique, and progress counts. These are model judgments and
traceable references, not independent verification. The checkpoint also records
retrieval plans and controller interventions for diagnosing unproductive runs.

Checkpoints are written atomically. Only one writer may run an inquiry at a
time; a process lock is released when that process exits. After interruption,
inspect the saved state and use `resume`. An interrupted, read-only model or
retrieval operation may need to run again, so a crash can repeat a billable API
call. The saved checkpoint, rather than a live Python object, is the source of
progress on restart.

Exit codes are `0` for successful command handling, including a deliberate
clarification pause or exhausted budget; `1` for a failed run or command error;
`2` for invalid CLI arguments; and `130` for a keyboard interruption. Inspect
the printed `Status` or `state.json` to distinguish completion from a pause.

## Evidence and citations

Search results are discovery records. Their snippets are kept separate from
source passages read by `read_source`; a snippet alone does not become a read
source. Sources keep stable IDs, URLs, retrieval times, reading status, and
bounded text passages. Relevant passages are explicitly supplied to reasoning
and review steps.

Successful identical queries and already-read sources are reused within an
inquiry, including after restart. Source retrieval times are supplied to the
reasoning steps. This version keeps a stable research snapshot; start a new
inquiry when you need to refresh the same sources for newly changed facts.

Source-supported claims carry passage references and quoted support. The
runner checks that cited passages exist, were available to the reasoning
step, and contain the stated quotation. Final passage citations are linked
to recorded source URLs. These mechanical checks prevent nonexistent
references and invented quotations; they do not establish that a source is
correct, independent, complete, or sufficient to justify the claim. Model
agreement is likewise not independent evidence.

Source text is treated as untrusted evidence. A page's instructions do not
override the inquiry question or control the tools. Reading is bounded and
restricted to public HTTP(S) resources; failures, unsupported content, and
truncation remain visible. A missing or inaccessible source can leave a
finding conditional or unresolved.

## Validation

Run the offline tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s tests/unit -p 'test_inquiry*.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.unit.test_frame_primitive -v
```

The tests use controlled model responses and mock HTTP responses to exercise
the contracts and failure paths without spending API credits. A successful
offline run verifies the plumbing and lifecycle; a live run with configured
keys is still needed to evaluate search coverage and the quality of a real
inquiry.
