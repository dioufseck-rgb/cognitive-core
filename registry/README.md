# Cognitive Core — Registry

The registry defines the cognitive building blocks of the framework.

## Prompts (`prompts/`)

Every LLM call in the framework uses a prompt template from this directory.
No prompts are hardcoded in engine code.

| File | Type | Used By | Params |
|------|------|---------|--------|
| `retrieve.txt` | Primitive | Sequential + Agentic | `specification`, `sources`, `source_results`, `context` |
| `classify.txt` | Primitive | Sequential + Agentic | `categories`, `criteria`, `confidence_threshold`, `context` |
| `investigate.txt` | Primitive | Sequential + Agentic | `question`, `scope`, `available_evidence`, `effort_level`, `context` |
| `think.txt` | Primitive | Sequential + Agentic | `instruction`, `focus`, `context` |
| `verify.txt` | Primitive | Sequential + Agentic | `rules`, `context` |
| `generate.txt` | Primitive | Sequential + Agentic | `requirements`, `format`, `constraints`, `context` |
| `challenge.txt` | Primitive | Sequential + Agentic | `perspective`, `threat_model`, `context` |
| `act.txt` | Primitive | Sequential + Agentic | `available_actions`, `action_results`, `authorization_context`, `context` |
| `orchestrator.txt` | Meta | Agentic only | `goal`, `available_primitives`, `primitive_configs`, `constraints`, `steps_completed`, `routing_log`, `step_count`, `max_steps`, `max_repeat`, `strategy` |

### Prompt Contract

- Templates use `{param_name}` placeholders filled at runtime
- Literal braces in JSON examples must be doubled: `{{`, `}}`
- All prompts request JSON-only output
- All primitive prompts share the `{context}` param (auto-populated
  from workflow state if not explicitly provided)
- The `{additional_instructions}` param is optional on all primitives
  for domain-specific prompt injection

### Editing Prompts

Prompts are the primary tuning surface. When hardening for accuracy:
- Add mandatory step sequences (extract → hypothesize → test)
- Add groundedness requirements (cite specific values)
- Add confidence calibration guidance
- Add examples of correct vs incorrect output

Changes to prompts affect all workflows using that primitive.
Domain-specific behavior goes in the domain YAML, not here.

## Schemas (`schemas.py`)

Pydantic models defining the output contract for each primitive.

| Schema | Fields | Notes |
|--------|--------|-------|
| `BaseOutput` | `confidence`, `reasoning`, `evidence_used`, `evidence_missing` | Inherited by all |
| `RetrieveOutput` | + `data`, `sources_queried`, `sources_skipped`, `retrieval_plan` | Two-phase: tools then LLM |
| `ClassifyOutput` | + `category`, `alternative_categories` | |
| `InvestigateOutput` | + `finding`, `hypotheses_tested`, `recommended_actions` | |
| `ThinkOutput` | + `thought`, `conclusions`, `decision` | Never terminal |
| `VerifyOutput` | + `conforms`, `violations`, `rules_checked` | |
| `GenerateOutput` | + `artifact`, `format`, `constraints_checked` | |
| `ChallengeOutput` | + `survives`, `vulnerabilities`, `strengths`, `overall_assessment` | |
| `ActOutput` | + `mode`, `actions_taken`, `authorization_checks`, `side_effects`, `requires_human_approval` | Write-side boundary |

### Schema Contract

- All schemas extend `BaseOutput`
- All outputs include `confidence` (0.0-1.0) and `reasoning`
- Downstream steps reference fields via `${step_name.field}` or `${_last_primitive.field}`
- Adding fields to a schema is backward-compatible
- Removing or renaming fields is a breaking change

## Primitives (`primitives.py`)

Maps primitive names to their prompt files, schemas, and param requirements.

| Primitive | Required Params | Schema | Boundary |
|-----------|----------------|--------|----------|
| `retrieve` | `specification` | `RetrieveOutput` | Read |
| `classify` | `categories`, `criteria` | `ClassifyOutput` | Read |
| `investigate` | `question`, `scope` | `InvestigateOutput` | Read |
| `think` | `instruction` | `ThinkOutput` | Read |
| `verify` | `rules` | `VerifyOutput` | Read |
| `generate` | `requirements`, `format`, `constraints` | `GenerateOutput` | Read |
| `challenge` | `perspective`, `threat_model` | `ChallengeOutput` | Read |
| `act` | `available_actions`, `authorization_context` | `ActOutput` | **Write** |

The orchestrator is **not** a primitive — it cannot be used in workflow
YAML steps. It is a meta-component that sequences primitives in agentic
mode. Its prompt lives in `prompts/orchestrator.txt` and is loaded by
`engine/agentic.py`.
