# Testing the inquiry runner in VS Code on Windows

These instructions use **Windows, VS Code, and its PowerShell terminal**. Run
each command from the repository root unless a step says otherwise. The first
tests are offline; the later connectivity checks and live inquiry use your API
accounts.

The inquiry implementation is on the **`feat/reasoning-inquiry`** branch of
`dioufseck-rgb/cognitive-core`. Clone that branch to get the code, tests, and
this guide together.

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

The current checkout has 305 tests; the run here completed with `OK`. The suite
includes framing, retrieval, evidence validation, revision, dependencies,
clarification, persistence, and the existing framework tests. Exact timing will
vary. Some tests deliberately exercise errors and kill switches, so their log
messages are not failures; read the final unittest summary.

To focus on the inquiry tests:

```powershell
.\.venv-inquiry\Scripts\python.exe -m unittest discover -s tests/unit -p "test_inquiry*.py" -v
```

Optional VS Code test UI: run **Python: Configure Tests**, choose **unittest**,
the `tests/unit` directory, and the `test_*.py` pattern. Tests appear in the
Testing view. The terminal command above is sufficient for the initial check.

## 5. Run the offline demonstration

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry demo --output inquiries/offline-01
```

Expect `Status: completed`, 16 scheduled steps, one simulated search, one
simulated source read, and two rounds of reasoning on the demonstration branch.
The demo uses scripted responses and a fictional source; it makes no API calls.

The first finding overclaims an efficiency benefit. Challenge identifies the
unsupported claim. Reflect requests a revision, and the revised finding removes
the empirical claim. The final synthesis retains the qualification and cites
the example passage.

Open `inquiries/offline-01/report.md` and press **Ctrl+Shift+V** to preview it.
Inspect `state.json` alongside it. Run these read-only commands if useful:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry status inquiries/offline-01
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry report inquiries/offline-01
```

Use a different output directory, such as `offline-02`, to run the demo again.
The command intentionally refuses to overwrite an existing inquiry.

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

Use `--provider openai` explicitly on both `start` and `resume`: the checked-in
configuration otherwise defaults to Google. `--model default` resolves through
`llm_config.yaml`; its OpenAI mapping in this checkout is `gpt-4o-mini`. That is
an existing configuration value, not a guarantee of availability on your
account. Override `--model` with a model ID your account can use when needed.

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

If a check fails, fix that connection before running a complete inquiry. The
development workspace could not resolve a public source domain, so live
transport was not verified there. These checks establish what works on your
machine.

## 8. Start a small real inquiry

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry start "When should an enterprise AI workflow use autonomous agents instead of a fixed workflow?" --purpose "Develop practical architecture criteria and examine the strongest counterargument" --provider openai --model default --max-steps 30 --max-searches 4 --max-reads 6 --output inquiries/agent-design-01
```

The output shows the current operation and subquestion. A live model chooses its
own path; it is not expected to reproduce the offline demo's sequence or length.
The small budget is meant to let you inspect behavior early. It may stop before
the final synthesis.

| Status | Meaning | Next action |
|---|---|---|
| `completed` | The run has produced a reviewed answer, possibly with qualifications. | Read the report and assess the reasoning. |
| `awaiting_user` | A material clarification is needed. | Answer the displayed ID using `resume`. |
| `budget_exhausted` | The scheduled step allowance is used. | Inspect partial results and add effort if useful. |
| `failed` | A model/tool/configuration/contract failure prevented the next step. | Read the sanitized error, fix the cause, and resume. |
| `ready` | A saved run is ready to continue, including after a keyboard interruption. | Resume it. |

The budgets limit scheduled work; they are not dollar or token caps. Provider
retries and page redirects can make additional HTTP requests within a step.

## 9. Clarify, extend, or interrupt the inquiry

First inspect its current state:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry status inquiries/agent-design-01
```

If it asks a question labeled `C1`, answer that actual question. Example:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry resume inquiries/agent-design-01 --answer "C1=Internal document-review workflows, with a human making the final consequential decision" --provider openai --model default
```

Repeat `--answer` if it asks more than one question. IDs are specific to the
saved inquiry; do not invent an ID or reuse one that is already answered.

For a budget pause:

```powershell
.\.venv-inquiry\Scripts\python.exe -m cognitive_core.inquiry resume inquiries/agent-design-01 --additional-steps 20 --additional-searches 3 --additional-reads 5 --provider openai --model default
```

Added allowances are cumulative. A clarification answer and added budget can
be supplied together. A completed run stays complete; use a new output directory
to explore a new or revised question.

Press **Ctrl+C** to interrupt. Progress is saved; an operation interrupted before
completion may run again on resume. Do not run two writers on the same inquiry
directory at once.

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
| `frames` | Did framing make the question more useful? Was a later change justified? |
| `questions` | Do the branches contribute to the main question? Did objections lead to substantive revisions? |
| `searches` | Were queries connected to uncertainties and contrary evidence? |
| `sources` | Were important sources actually read? Are passages relevant and dated? |
| `clarifications` | Were questions to you necessary, and did your answers change the work? |
| `synthesis` | Does the integrated answer reconcile branches and preserve uncertainty? |
| `events` | Was the work productive, or did it repeat without learning? |

Compare the final result with one strong ordinary chat response to the same
question. Judge framing, evidence quality, corrections, synthesis, and how much
intervention you needed. A long report or many successful steps is not evidence
that the answer is better.

The v1 runner checks source references and exact quoted support, but it cannot
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
| Invalid JSON or structured output | Read the named field error; retry the saved step or select a suitable model. Repeated failures are a bug report, not a reason to loosen evidence checks. |
| Output directory already exists | Use `resume` for unfinished work or a new output directory for a new study. |
| `budget_exhausted` | This is a saved pause. Inspect the result, then add a bounded amount of effort. |

## Reference documentation

- [VS Code Python environments](https://code.visualstudio.com/docs/python/environments)
- [VS Code Python testing](https://code.visualstudio.com/docs/python/testing)
- [Python virtual environments](https://docs.python.org/3.12/library/venv.html)
- [PowerShell Read-Host](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/read-host)
- [PowerShell environment variables](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_environment_variables)
