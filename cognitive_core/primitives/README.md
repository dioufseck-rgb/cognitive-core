# Cognitive Core — Primitives

The primitive layer defines the nine typed epistemic operations from which all workflows are composed.

---

## Prompt templates (`prompts/`)

Every LLM call in the framework uses a prompt template from this directory. No prompts are hardcoded in engine code.

| File | Primitive | Mode | Key params |
|------|-----------|------|------------|
| `retrieve.txt` | `retrieve` | Workflow + Agentic | `specification`, `sources`, `source_results`, `context` |
| `classify.txt` | `classify` | Workflow + Agentic | `categories`, `criteria`, `confidence_threshold`, `context` |
| `investigate.txt` | `investigate` | Workflow + Agentic | `question`, `scope`, `available_evidence`, `effort_level`, `context` |
| `verify.txt` | `verify` | Workflow + Agentic | `rules`, `subject`, `context` |
| `challenge.txt` | `challenge` | Workflow + Agentic | `perspective`, `threat_model`, `context` |
| `reflect.txt` | `reflect` | Agentic (gap-filling and post-challenge) | `scope`, `domain_index`, `context` |
| `deliberate.txt` | `deliberate` | Workflow + Agentic | `instruction`, `focus`, `context` |
| `generate.txt` | `generate` | Workflow + Agentic | `requirements`, `format`, `constraints`, `context` |
| `govern.txt` | `govern` | Workflow + Agentic | `workflow_state`, `governance_context`, `epistemic_context`, `tier_override` |
| `orchestrator.txt` | — (meta) | Agentic only | `goal`, `available_primitives`, `primitive_configs`, `constraints`, `steps_completed`, `routing_log`, `step_count`, `max_steps`, `strategy` |

The orchestrator is not a primitive — it cannot be used in workflow YAML steps. It is the decision-making component that sequences primitives in agentic mode.

### Prompt contract

- Templates use `{param_name}` placeholders filled at runtime
- Literal braces in JSON examples must be doubled: `{{`, `}}`
- All primitive prompts request JSON-only output
- All primitive prompts accept `{context}` (auto-populated from workflow state) and `{additional_instructions}` (optional domain-specific injection)
- Domain-specific behavior goes in the domain YAML, not here

### Editing prompts

Prompts are the primary tuning surface for accuracy and governance behavior:
- Add mandatory reasoning steps (hypothesize → test → conclude)
- Add groundedness requirements (cite specific values from evidence)
- Add confidence calibration guidance
- Add examples of correct vs incorrect output

Changes to prompts affect all workflows using that primitive. Domain-specific constraints belong in the domain YAML `primitive_configs` section.

---

## Schemas (`schemas.py`)

Pydantic models defining the output contract for each primitive.

| Schema | Additional fields beyond `BaseOutput` |
|--------|---------------------------------------|
| `BaseOutput` | `confidence`, `reasoning`, `evidence_used`, `evidence_missing` |
| `RetrieveOutput` | `data`, `sources_queried`, `sources_skipped`, `retrieval_plan` |
| `ClassifyOutput` | `category`, `alternative_categories` |
| `InvestigateOutput` | `finding`, `hypotheses_tested`, `evidence_flags`, `missing_evidence` |
| `VerifyOutput` | `conforms`, `violations`, `rules_checked` |
| `ChallengeOutput` | `survives`, `vulnerabilities`, `strengths`, `overall_assessment` |
| `ReflectOutput` | `trajectory`, `revision_target`, `what_changed`, `open_questions`, `next_question`, `template_guidance`, `established_facts_to_skip` |
| `DeliberateOutput` | `recommended_action`, `warrant`, `situation_summary`, `options_considered` |
| `GenerateOutput` | `artifact`, `format`, `constraints_checked` |
| `GovernOutput` | `tier_applied`, `tier_rationale`, `disposition`, `work_order`, `escalation_target`, `resumption_condition`, `accountability_chain` |

### Schema contract

- All schemas extend `BaseOutput`
- All outputs include `confidence` (0.0–1.0) and `reasoning`
- Downstream steps reference prior outputs via workflow state context injection
- Adding fields to a schema is backward-compatible
- Removing or renaming fields is a breaking change

### Epistemic state

Six of nine primitives elicit `reasoning_quality` and `outcome_certainty` with governance-aware framing. The exceptions:
- `retrieve` — quality measured mechanically (evidence completeness)
- `govern` — reads accumulated record rather than producing first-order reasoning
- `reflect` — reports `trajectory` and `revision_target` rather than scalar quality fields; its governance contribution is structural

---

## Registry (`registry.py`)

Maps primitive names to prompt files, schemas, and parameter requirements. New primitives are registered here.

```python
from cognitive_core.primitives.registry import list_primitives, get_prompt_template, render_prompt

list_primitives()
# ['retrieve', 'classify', 'investigate', 'verify', 'challenge',
#  'reflect', 'deliberate', 'generate', 'govern']

template = get_prompt_template('deliberate')
rendered = render_prompt('deliberate', {'instruction': '...', 'context': '...'})
```
