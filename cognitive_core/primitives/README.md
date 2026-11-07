# Cognitive Core — Primitive Registry

This directory defines the ten typed epistemic primitives and the orchestrator prompt.

## Files

| File | Contents |
|---|---|
| `registry.py` | `PRIMITIVE_CONFIGS`: maps each primitive to its prompt file, output schema, required and optional parameters, and defaults |
| `schemas.py` | Pydantic output schemas; all extend `BaseOutput` |
| `prompts/` | One prompt template per primitive, plus `orchestrator.txt` |
| `artifacts.py` | Artifact helpers |

## Primitives

| Primitive | Required params | Optional params | Schema |
|---|---|---|---|
| `frame` | `question` | `purpose` | `FrameOutput` |
| `retrieve` | `specification` | `sources`, `strategy` | `RetrieveOutput` |
| `classify` | `categories`, `criteria` | `confidence_threshold` | `ClassifyOutput` |
| `investigate` | `question`, `scope` | `effort_level`, `available_evidence` | `InvestigateOutput` |
| `verify` | `rules` | `subject` | `VerifyOutput` |
| `challenge` | `perspective`, `threat_model` | — | `ChallengeOutput` |
| `deliberate` | `instruction` | `focus` | `DeliberateOutput` |
| `generate` | `requirements`, `format`, `constraints` | — | `GenerateOutput` |
| `reflect` | `scope` | `domain_index` | `ReflectOutput` |
| `govern` | `workflow_state`, `governance_context` | `tier_override`, `epistemic_context` | `GovernOutput` |

Every primitive also accepts `additional_instructions` (domain-specific prompt text) and `context` (filled from workflow state if not provided).

## Output schemas

All schemas extend `BaseOutput`: `confidence` (0.0–1.0), `reasoning`, `evidence_used`, `evidence_missing`, `resource_requests`.

| Schema | Main added fields |
|---|---|
| `FrameOutput` | `working_question`, `purpose`, `scope`, `assumptions`, `subquestions`, `answer_criteria`, `clarifications` |
| `RetrieveOutput` | `data`, `sources_queried`, `sources_skipped`, `retrieval_plan` |
| `ClassifyOutput` | `category`, `alternative_categories` |
| `InvestigateOutput` | `finding`, `hypotheses_tested`, `recommended_actions`, `evidence_flags` |
| `VerifyOutput` | `conforms`, `violations`, `rules_checked` |
| `ChallengeOutput` | `survives`, `vulnerabilities`, `strengths`, `overall_assessment` |
| `DeliberateOutput` | `situation_summary`, `options_considered`, `recommended_action`, `warrant` |
| `GenerateOutput` | `artifact`, `format`, `constraints_checked` |
| `ReflectOutput` | `what_was_established`, `what_was_assumed`, `what_changed`, `sensitivity`, `trajectory`, `next_question` |
| `GovernOutput` | `tier_applied`, `tier_rationale`, `disposition`, `work_order`, `accountability_chain` |

Seven schemas also carry the LLM-reported judgment fields `reasoning_quality` and `outcome_certainty`. `FrameOutput`, `RetrieveOutput`, and `GovernOutput` do not. These fields feed the judgment layer of the epistemic state computed in `engine/epistemic.py`.

### Framing an inquiry

`FrameOutput` is a provisional question plan. Its one to eight subquestions have stable IDs, contribution statements, evidence needs, and dependencies on other subquestions. Validation rejects blank text, duplicate IDs, unknown dependencies, and dependency cycles. Clarifications specify the question for the user, why it matters, and whether it blocks progress. The caller retains the original question and versions of the frame.

Frame can run before research or revise a plan after new evidence. It identifies what needs investigation; it does not establish findings. The inquiry runner uses Frame explicitly. Registering it does not change the existing institutional agentic workflow's transition and closure rules.

### Schema contract

- Downstream steps reference fields as `${step_name.field}` or `${_last_primitive.field}`.
- Adding fields is backward-compatible. Removing or renaming fields is a breaking change.

## Prompts

- Templates use `{param_name}` placeholders filled at runtime.
- `render_prompt` protects literal JSON braces while replacing named placeholders.
- All prompts request JSON-only output.
- Domain-specific behavior belongs in the domain YAML, not in these templates. A change here affects every workflow that uses the primitive.

## The orchestrator

The orchestrator is not a primitive and cannot be used as a workflow step. In agentic mode it chooses the next primitive from the goal, the available primitives, and the steps completed so far. Its prompt is `prompts/orchestrator.txt`, loaded by `engine/agentic_devs.py`. Governance is enforced by the framework on whatever trajectory the orchestrator produces.
