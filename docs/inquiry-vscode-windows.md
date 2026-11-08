# Testing native inquiry in VS Code on Windows

These instructions use **Windows, VS Code, and its PowerShell terminal**. Run
each command from the repository root unless a step says otherwise. The first
tests are offline; the later connectivity checks and live inquiry use your API
accounts.

The inquiry implementation is on the **`feat/reasoning-inquiry`** branch of
`dioufseck-rgb/cognitive-core`. Clone that branch to get the code, tests, and
this guide together.

If you already cloned this branch, update it from the repository root before
continuing:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
git switch feat/reasoning-inquiry
git pull --ff-only
```

This version uses the native Coordinator and DEVS executor. Start a **new output
directory** for it; the earlier standalone runner's JSON checkpoints are not
automatically migrated.

## 1. Check the prerequisites

Install Git, Python 3.11 or newer, and VS Code's **Python** extension published by
Microsoft. The implementation's Python tests were run with Python 3.12.

In VS Code, select **Terminal > New Terminal**. Choose **PowerShell** in the
terminal dropdown. Run:

```powershell
git --version
py -3 --version
```

Both should print a version. If `py` is unavailable but `python --version`
reports Python 3.11 or newer, use `python` in place of `py -3` when creating the
environment below. If neither works, install Python and open a new terminal.

## 2. Clone the inquiry branch

Create a test checkout and open it in VS Code:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\Projects" | Out-Null
Set-Location "$env:USERPROFILE\Projects"
git clone -c core.longpaths=true --branch feat/reasoning-inquiry --single-branch https://github.com/dioufseck-rgb/cognitive-core.git cognitive-core-inquiry
Set-Location .\cognitive-core-inquiry
code .
```

If `code` is unavailable, use **File > Open Folder** and select
`Projects\cognitive-core-inquiry`.

In the new VS Code window, open a PowerShell terminal. The Explorer should show
`pyproject.toml`, `llm_config.yaml`, and `cognitive_core\inquiry`.

Verify the selected branch:

```powershell
git branch --show-current
```

It should print `feat/reasoning-inquiry`.

## 3. Create a fresh Python environment

The repository includes an old checked-in environment. Create a separate
`.venv-inquiry` for your machine. Do not use the repository's `.venv`.

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
py -3 -m venv .venv-inquiry
.\.venv-inquiry\Scripts\python.exe -m pip install --upgrade pip
.\.venv-inquiry\Scripts\python.exe -m pip install -e ".[inquiry,pdf]"
```

The UTF-8 setting keeps text handling consistent; the bytecode setting avoids
rewriting old compiled files tracked by this repository. The editable install
uses your checkout directly. The `pdf` extra installs optional PDF decoding.

All commands in this guide call the environment's Python executable explicitly.
You do not need to activate it or change PowerShell's execution policy.

Tell VS Code to use the same interpreter:

1. Press **Ctrl+Shift+P**.
2. Run **Python: Select Interpreter**.
3. Select `.venv-inquiry\Scripts\python.exe`.
4. If it is absent, choose **Enter interpreter path**, then browse to that file.

Verify the executable and imports:

```powershell
.\.venv-inquiry\Scripts\python.exe -c "import sys, pydantic, httpx; print(sys.executable); print('Dependencies loaded')"
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry --help
```

The executable path should contain `.venv-inquiry`. Help should list `start`,
`resume`, `status`, `report`, and `demo`.

## 4. Run the automated tests

No API keys are needed:

```powershell
.\.venv-inquiry\Scripts\python.exe -m unittest discover -s tests/unit -p "test_*.py" -q
```

The final test summary should report `OK`; the test count grows as regressions
are added. The suite includes framing, retrieval, evidence validation, revision, dependencies,
clarification, persistence, and the existing framework tests. Exact timing will
vary. Some tests deliberately exercise errors and kill switches, so their log
messages are not failures; read the final unittest summary.

To focus on the inquiry tests:

```powershell
.\.venv-inquiry\Scripts\python.exe -m unittest discover -s tests/unit -p "test_native_inquiry*.py" -v
```

Optional VS Code test UI: run **Python: Configure Tests**, choose **unittest**,
the `tests/unit` directory, and the `test_*.py` pattern. Tests appear in the
Testing view. The terminal command above is sufficient for the initial check.

## 5. Run the offline demonstration

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry demo --output inquiries/native-offline-01
```

Expect `Coordinator status: completed`, research outcome `conditional`, **12
accepted operations, 24 model attempts, one synthetic search, and one source
read**. Only the model and web boundaries are fixtures. The Coordinator, native
DEVS executor, protected-call pipeline, epistemic assessment, SQLite checkpoint,
audit ledger, final review, and Govern actually run.

The initial finding mistakes a required receipt for a sufficient condition.
Verify and Challenge identify a missing prior-approval condition. Reflect
requests a material revision; the corrected answer preserves the limitations
of the fictional policy. This is an integration test, not live research.

Open `inquiries/native-offline-01/report.md` and press **Ctrl+Shift+V** to preview
it. Inspect `state.json` alongside it. The authoritative state is in
`coordinator.db`; `state.json` is an inspection export. These commands make no
model or search calls:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry status inquiries/native-offline-01
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry report inquiries/native-offline-01
```

Use a different output directory, such as `offline-02`, to run the demo again.
The command intentionally refuses to overwrite an existing inquiry.

Also test durable clarification in separate CLI invocations:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry demo --clarify --output inquiries/native-clarify-01
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry resume inquiries/native-clarify-01 --answer "c1=Yes, restrict the answer to the fictional policy."
```

The first invocation suspends after Frame. The second fulfills the actual native
work order and resumes from the database through a newly created Coordinator.

## 6. Configure a real model and Serper

The remaining examples use **OpenAI as the model provider**. The application
also supports the other providers in Cognitive Core's existing factory.

| Provider | SDK to install | Credential variable | CLI provider value |
|---|---|---|---|
| OpenAI | `langchain-openai` | `OPENAI_API_KEY` | `openai` |
| Anthropic | `langchain-anthropic` | `ANTHROPIC_API_KEY` | `anthropic` |
| Google | `langchain-google-genai` | `GOOGLE_API_KEY` | `google` |

Install the one SDK you intend to use. For the example:

```powershell
.\.venv-inquiry\Scripts\python.exe -m pip install langchain-openai
```

Enter keys through masked terminal prompts:

```powershell
$env:SERPER_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "Serper API key" -AsSecureString)).Password
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", (Read-Host "OpenAI API key" -AsSecureString)).Password
$env:LLM_PROVIDER = "openai"
```

Paste each key at its prompt and press Enter. The values become environment
variables for this terminal and child Python processes. They are not included
in the command text. The application needs the real plaintext values in its
process environment; masking protects terminal entry, not process memory.

Verify presence without printing the values:

```powershell
.\.venv-inquiry\Scripts\python.exe -c "import os; print('Serper configured:', bool(os.getenv('SERPER_API_KEY'))); print('Model key configured:', bool(os.getenv('OPENAI_API_KEY')))"
```

Both should say `True`. These session variables do not automatically appear in
another terminal. A `.env` file is not automatically loaded by this runner.

Use `--provider openai` on `start`; resume remembers that selection, and accepts
an explicit override. `--model default` resolves through `llm_config.yaml`.
Inspect the provider's configured alias or pass an explicit model ID your
account can use. A model that passes a connectivity check still needs to be
evaluated for the inquiry's structured outputs and research quality.

## 7. Check the three live connections separately

These checks make live requests. The first uses a model call; the second uses
one Serper search; the third reads a public page without a search request.

### A. Model access

```powershell
.\.venv-inquiry\Scripts\python.exe -c "from cognitive_core.engine.llm import create_llm; result = create_llm(provider='openai', model='default').invoke('Reply only with OK.'); print(result.content)"
```

Expect a short response such as `OK`. If the account lacks access to the
configured model, choose an available model before continuing.

### B. Serper search

```powershell
.\.venv-inquiry\Scripts\python.exe -c "from cognitive_core.inquiry.research import SerperResearchProvider; hits = SerperResearchProvider().search('Python virtual environments official documentation', num=3); print('\n'.join(hit['url'] for hit in hits))"
```

Expect a short list of URLs. This checks the key, quota, query request, and
response normalization.

### C. Source reading

```powershell
.\.venv-inquiry\Scripts\python.exe -c "from cognitive_core.inquiry.research import SerperResearchProvider; page = SerperResearchProvider().read('https://docs.python.org/3/library/venv.html'); print(page['title']); print('Characters:', len(page['text'])); print(page['text'][:400])"
```

Expect a title and nonempty text. Search can work while a source website is
inaccessible, so test both capabilities. The source reader uses direct public
HTTP(S) connections and does not inherit proxy settings. In a managed network,
an approved transport configuration may be needed; do not disable TLS checks
or organizational controls to make the test pass.

If a check fails, fix that connection before running a complete inquiry. This
implementation was verified with deterministic boundary fixtures, without your
keys. These live checks establish what works on your machine.

## 8. Start a small real inquiry

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry start "For an internal assistant that searches policy documents and drafts cited answers, when is a fixed retrieve-then-generate workflow sufficient, and when does model-directed multi-step retrieval improve results?" --purpose "Develop practical architecture criteria for a read-only pilot using public sources" --provider openai --model default --max-steps 32 --max-model-calls 96 --max-searches 8 --max-reads 12 --output inquiries/policy-native-01
```

The native orchestrator chooses the path; it is not expected to reproduce the
offline demo's sequence or length. YAML controls the strategy and operation
instructions. The runtime enforces profiles, budgets, targets, and final review.

| Status | Meaning | Next action |
|---|---|---|
| `completed` | Native execution and release governance completed. | Read the report and assess its separate research outcome. |
| `suspended` | A clarification or governance review is pending. | Answer the displayed clarification or resolve the native review task. |
| `failed` | Execution could not continue. | Inspect the saved error, fix the cause, then recover with `resume`. |
| `running` | A worker is active, or its process stopped unexpectedly. | Wait if it is active; recover only after it has stopped. |

Research outcomes are separate: `open`, `answered`, `conditional`,
`evidence_limited`, or `budget_exhausted`. A completed report can honestly be
conditional or evidence-limited. Closeout reserves effort for synthesis,
report, final Challenge, and Govern; it cannot guarantee a passing final review.

The model allocation counts orchestration, output correction, and native
transport retries. It is not a dollar or token cap; provider SDK internals and
page redirects can make additional HTTP requests. A successful cached tool
request is reused rather than charged as another search or read.

## 9. Clarify, recover, or interrupt the inquiry

First inspect its current state:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry status inquiries/policy-native-01
```

If it asks a question labeled `c1`, answer that actual question. Example:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry resume inquiries/policy-native-01 --answer "c1=Public evidence for a read-only pilot; humans make consequential decisions" --provider openai
```

Repeat `--answer` if it asks more than one question. Execution continues after
the last blocking answer. Identical repeated answers are idempotent; replacing
a fulfilled answer is rejected. Clarification fulfillment cannot approve a
governance-suspended report. See the [native guide](inquiry.md#clarification-recovery-and-governance)
for that separate review path.

After a provider failure is fixed, run `resume` without an answer. Recovery
preserves accepted operations, read passages, and consumed budgets. This minimal
version does not support the legacy `--additional-*` budget flags. Inspect the
work before choosing a larger budget for a new study.

Press **Ctrl+C** to interrupt. Once the original process has definitely stopped,
recover an instance still marked `running` with:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry resume inquiries/policy-native-01 --recover-running
```

An external request whose result was not committed may run again. Do not run
two workers on the same inquiry directory or recover while the old one is active.

If you close VS Code or open a new terminal, return to the repository root, set
`PYTHONUTF8`, `PYTHONDONTWRITEBYTECODE`, and the keys again, then resume with the
same provider/model arguments. Use terminal execution for these instructions:
environment variables set inside a terminal do not propagate into the parent
VS Code process or automatically configure an F5 debugger session.

## 10. Supply context for a more useful study

Create `study-context.md` in the repository root. For example:

```markdown
# Purpose
Develop a position for AI architecture leaders.

# Scope
Focus on enterprise document review and investigation workflows.
Assume a human makes consequential final decisions.

# Questions worth examining
- When does adaptive choice of the next step add value?
- What does autonomy cost in coordination and verification?
- What is the strongest case for a fixed workflow?

# Desired answer
State practical criteria, counterarguments, assumptions, and cited evidence.
Distinguish conceptual arguments from measured performance claims.
```

Start a new study with that context:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry start "When is agentic orchestration justified?" --context-file study-context.md --provider openai --model default --max-steps 60 --max-searches 8 --max-reads 12 --output inquiries/agent-design-02
```

## 11. Assess whether it is useful

Open the study's `report.md` in Markdown preview. In `state.json`, examine:

| Field | What to look for |
|---|---|
| `study.frames` | Did framing improve the question without silently changing your purpose? |
| `study.questions` / `study.findings` | Do branches contribute to the answer? Did objections change the findings? |
| `study.searches` / `study.sources` | Were searches discriminating and sources actually read? Are passages relevant? |
| `study.clarifications` | Were questions to you necessary, and did the answers affect the study? |
| `study.synthesis` / `study.report` | Does integration preserve uncertainty and scope? |
| `study.final_review` / `study.governance` | Is review current and tied to the actual report? |
| `steps` / `routing_log` | What did the native executor accept, and why did it choose each operation? |
| `metadata.inquiry_governance` | Were protected calls, native checkpoints, and governance hooks recorded? |

Compare the final result with one strong ordinary chat response to the same
question. Judge framing, evidence quality, corrections, synthesis, and how much
intervention you needed. A long report or many successful steps is not evidence
that the answer is better.

The native implementation checks source references and exact quoted support, but it cannot
mechanically establish that a source is true or that a quotation entails the
claim. It reuses successful searches and source snapshots within a study; use a
new inquiry for a fresh examination of the same changing facts. Numerical
experiments and simulations are outside this version.

## Troubleshooting

| Problem | What to check |
|---|---|
| `No module named cognitive_core.inquiry` | Confirm `git branch --show-current` prints `feat/reasoning-inquiry` and you are in the project root using `.venv-inquiry` Python. |
| `py` is not recognized | Install Python, or use an existing `python` that reports version 3.11+. |
| Provider dependency missing | Install the SDK for the provider you selected into `.venv-inquiry`. |
| The app tries Google when you meant OpenAI | Pass `--provider openai` on start and resume; set `LLM_PROVIDER` for direct factory helpers. |
| Key absent despite a `.env` file | Set the variables in the PowerShell session; the runner does not autoload `.env`. |
| Serper rejects access | Check the runtime key/account; for quota or rate limiting, inspect the account before retrying. |
| Source lookup/HTTP/TLS error | Test source reading independently. Follow the network's approved proxy/TLS configuration. |
| Unknown/unavailable model | Check `llm_config.yaml` and select an available model ID with `--model`. |
| Invalid JSON or structured output | Native profiles supply one exact schema and permit two corrections before failing. Inspect the error and preserved state. Fix the model/configuration cause, then recover; a repeated failure is useful bug-report evidence. |
| Output directory already exists | Use `resume` for unfinished work or a new output directory for a new study. |
| `budget_exhausted` research outcome | The report preserves limits from exhausted research allocation. Budget enlargement is not implemented in this native slice. |
| A legacy checkpoint is rejected | Start a new native output directory. The earlier JSON store is not the native Coordinator database. |

## Reference documentation

- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [VS Code Python testing](https://code.visualstudio.com/docs/python/testing)
- [Python virtual environments](https://docs.python.org/3.12/library/venv.html)
- [PowerShell Read-Host](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/read-host)
- [PowerShell environment variables](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables)
