# Cognitive Core

Composable AI workflows from seven cognitive primitives.
Three-layer architecture: **Workflow** × **Domain** × **Case**.
Two execution modes: **Sequential** (deterministic) and **Agentic** (LLM-orchestrated).

## Quick Start

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key

# Run the demo (guided scenarios with pause points)
./demo.sh

# Or run any three-layer combo directly:
python -m engine.runner \
  -w workflows/complaint_resolution.yaml \
  -d domains/member_complaint.yaml \
  -c cases/complaint_torres.json

# Agentic mode — same case, orchestrator decides the path:
python -m engine.runner \
  -w workflows/complaint_resolution_agentic.yaml \
  -d domains/member_complaint_agentic.yaml \
  -c cases/complaint_torres.json

# Validate a config without calling LLMs:
python -m engine.runner \
  -w workflows/dispute_resolution.yaml \
  -d domains/card_dispute.yaml \
  --validate-only
```

## The Idea

Every AI task — dispute resolution, hardship assessment, spending advice, compliance review — decomposes into the same cognitive operations. We made each operation a typed, composable building block called a **primitive**. Different compositions of the same primitives serve different use cases, configured in YAML, not code.

## Seven Primitives

| Primitive       | What It Does                        | Key Output Fields                    | Temp  |
|-----------------|-------------------------------------|--------------------------------------|-------|
| **Retrieve**    | Gather data from source systems     | `data`, `sources_queried`            | 0.1   |
| **Classify**    | Categorize with evidence            | `category`, `alternative_categories` | 0.05  |
| **Investigate** | Extract → Hypothesize → Test        | `finding`, `hypotheses_tested`       | 0.1   |
| **Think**       | Synthesize, reason, connect dots    | `thought`, `conclusions`, `decision` | 0.2   |
| **Verify**      | Rule-by-rule compliance check       | `conforms`, `violations`             | 0.0   |
| **Generate**    | Produce grounded output             | `artifact`, `constraints_checked`    | 0.3   |
| **Challenge**   | Adversarial review (stronger model) | `survives`, `vulnerabilities`        | 0.05  |

All extend `BaseOutput`: `confidence`, `reasoning`, `evidence_used`, `evidence_missing`.

## Three-Layer Architecture

| Layer        | Location           | Who Authors      | Changes      |
|--------------|--------------------|------------------|--------------|
| **Workflow** | `workflows/*.yaml` | AI Engineers     | Rarely       |
| **Domain**   | `domains/*.yaml`   | SMEs + Engineers | Per use case |
| **Case**     | `cases/*.json`     | Production APIs  | Per run      |

Multiplication: 7 workflows × 12 domains = all use cases.
New use case = new domain YAML (2-4 hours), not new code.

### Reference Resolution

```
${domain.classify.categories}  → resolved at MERGE TIME (before execution)
${input.member_statement}      → resolved at RUNTIME (from case data)
${_last_classify.category}     → resolved at RUNTIME (from prior step output)
```

## Two Modes

### Sequential (Production)

Deterministic step sequence with conditional routing, loop-on-failure, and per-step model overrides. Every path is predictable and auditable.

```yaml
steps:
  - name: classify_type
    primitive: classify
    transitions:
      - when: "output.category == 'military_transition'"
        goto: investigate_military
      - default: investigate_financial
```

### Agentic (Discovery)

LLM orchestrator at the center of a hub-and-spoke graph. Sees accumulated state, decides which primitive to invoke next. Constrained by guardrails.

```yaml
mode: agentic
constraints:
  max_steps: 10
  must_include: [classify, investigate, generate, challenge]
  must_end_with: challenge
  challenge_must_pass: true
```

**Lifecycle**: prototype in agentic → observe paths → crystallize into sequential.

## Available Workflows

### Sequential

| Workflow | Pattern | Domains |
|----------|---------|---------|
| `dispute_resolution` | Retrieve → Classify → Verify → Investigate → Generate → Challenge | `card_dispute`, `ach_dispute` |
| `loan_hardship` | Classify → Investigate (branch) → Generate → Challenge | `military_hardship` |
| `spending_advisor` | Retrieve → Classify → Investigate → Generate → Challenge | `debit_spending` |
| `complaint_resolution` | Classify ×2 → Investigate → Generate → Challenge | `member_complaint` |
| `nurse_triage` | Classify ×2 → Investigate → Verify → Generate → Challenge | `cardiac_triage` |
| `regulatory_impact` | Classify → Investigate (adaptive) → Classify → Generate → Challenge | `avm_regulation` |
| `sar_investigation` | Classify → Investigate → Classify → Generate → Challenge → Verify | `structuring_sar` |

### Agentic

| Workflow | Domains |
|----------|---------|
| `loan_hardship_agentic` | `military_hardship_agentic` |
| `spending_advisor_agentic` | `debit_spending_agentic` |
| `complaint_resolution_agentic` | `member_complaint_agentic` |
| `nurse_triage_agentic` | `cardiac_triage_agentic` |
| `regulatory_impact_agentic` | `avm_regulation_agentic` |
| `sar_investigation_agentic` | `structuring_sar_agentic` |

Every sequential workflow has an agentic counterpart sharing the same cases.

## Demo Cases

| Case File | Domain | Scenario |
|-----------|--------|----------|
| `card_clear_fraud.json` | Card dispute | Clear fraud, fast-path eligible |
| `ach_revoked_authorization.json` | ACH dispute | Subscription didn't cancel |
| `military_hardship_reeves.json` | Loan hardship | Military spouse, medical retirement |
| `spending_advisor_williams.json` | Spending advisor | "How's my spending?" |
| `spending_advisor_williams_followup.json` | Spending advisor | Follow-up drill-down |
| `complaint_torres.json` | Member complaint | Inconsistent hold info, $50 in fees |
| `cardiac_chest_pain.json` | Nurse triage | 2am chest pain, cardiac risk factors |
| `avm_regulation.json` | Regulatory impact | AVM quality control rule |
| `sar_structuring.json` | SAR investigation | Potential structuring pattern |

## LLM Parameter Cascade

Per-step model and temperature override the workflow default. Most specific wins.

```
CLI --model / --temperature
  └→ orchestrator.model (agentic)
      └→ primitive_configs.X.model (domain)
          └→ step.model (sequential workflow)
```

Challenge always uses `gemini-2.5-pro` for adversarial diversity — different model catches errors the generator's model might share as blind spots.

## Retrieve & Tool Registry

The Retrieve primitive calls data source tools through a `ToolRegistry`.

```python
# Production: register API, vector, and MCP providers
provider = APIProvider(base_url="https://core-banking.internal")
provider.add_endpoint("member_profile", "/v1/members/{member_id}")
provider.register_all(registry)

# MCP: dynamic tool discovery
mcp = MCPProvider(command="python", args=["mcp_servers/compliance_server.py"])
await mcp.connect()  # discovers tools automatically
mcp.register_all(registry)

# Dev/test: case JSON becomes the tool registry
registry = create_case_registry(case_data)
```

Two-phase architecture: tools return ground truth data, LLM assesses quality. If LLM assessment fails, data still flows through at confidence 0.8.

## Project Structure

```
cognitive-core/
├── engine/              # Framework runtime
│   ├── runner.py        # CLI entry point
│   ├── composer.py      # Three-layer merge + LangGraph compilation
│   ├── agentic.py       # Hub-and-spoke orchestrator
│   ├── nodes.py         # Node factory with tracing
│   ├── state.py         # WorkflowState, StepResult, routing
│   ├── tools.py         # ToolRegistry, DataTool protocol
│   └── providers.py     # APIProvider, VectorProvider, MCPProvider
├── registry/            # Cognitive building blocks
│   ├── primitives.py    # Primitive registry + rendering
│   ├── schemas.py       # Pydantic output contracts
│   └── prompts/         # 7 primitive + 1 orchestrator templates
├── workflows/           # Cognitive patterns (YAML)
├── domains/             # Subject matter expertise (YAML)
├── cases/               # Test case data (JSON)
├── config/              # Production tool registry
├── mcp_servers/         # Example MCP server
├── docs/                # Authoring guide, talking points
├── demo.sh              # Guided demo script
└── requirements.txt
```

## Documentation

- **[Authoring Guide](docs/AUTHORING_GUIDE.md)** — How to create a new use case
- **[Registry README](registry/README.md)** — Prompt contracts, schema details
- **[Demo Talking Points](docs/DEMO_TALKING_POINTS.md)** — Presentation guide
- **[Schema Glossary](docs/schema_glossary.yaml)** — Production field mappings

## CLI Reference

```bash
# Three-layer execution
python -m engine.runner -w WORKFLOW -d DOMAIN [-c CASE]

# Options
  --model MODEL          Override LLM model (default: gemini-2.0-flash)
  --temperature TEMP     Override temperature
  --output FILE          Save full state to JSON
  --validate-only        Check config without calling LLMs
  --quiet                Suppress trace output
  --verbose              Show full LLM prompts and responses
```
