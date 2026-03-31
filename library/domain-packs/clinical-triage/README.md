# Clinical Triage

**Pattern:** P06 — Triage and Escalation  
**Overlays:** human-review, governance  
**Coordinator:** simple  

Assesses urgency and severity of a patient contact, classifies the clinical situation into a severity tier, investigates complex or high-acuity presentations, generates a triage disposition script for clinical review, and governs escalation. All GATE and HOLD cases require clinician sign-off before the disposition is communicated.

## What it does

Five steps, one conditional branch:

1. **Retrieve** — pulls patient record and recent encounters if a patient ID is provided (skips gracefully if none)
2. **Classify** — assigns severity tier (emergent / urgent / semi-urgent / non-urgent)
3. **Investigate** — fires only for emergent and urgent presentations; performs deeper clinical assessment before generating the script
4. **Generate** — produces a triage disposition script with severity, recommended pathway, immediate instructions, and return precautions
5. **Govern** — routes to SPOT_CHECK (non-urgent, high confidence), GATE (urgent, clinician review), or HOLD (emergent, senior clinician)

**This pack supports clinical decision support — it does not make autonomous clinical decisions.** The human-review overlay is always declared. All GATE and HOLD cases require clinician sign-off.

## Files

```
clinical-triage/
  clinical_triage.yaml              ← fill in your severity tiers and response format
  workflow.yaml            ← the workflow (do not edit)
  coordinator_config.yaml  ← governance queues and SLAs
  run.py                   ← runner script
  cases/
    example_contacts.json  ← 4 contacts: emergent through non-urgent
```

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=your_key_here
python library/domain-packs/clinical-triage/run.py
python library/domain-packs/clinical-triage/run.py --case-id TRIAGE-001
```

## Customising

Open `clinical_triage.yaml` and fill in:

| Section | What to change |
|---------|---------------|
| `classify_severity.categories` | Your severity tier definitions and ESI/CTAS mapping |
| `classify_severity.criteria` | Assessment weighting and tie-breaking rules |
| `classify_severity.investigate_condition` | Which tiers trigger deeper investigation |
| `investigate_situation.scope` | What clinical questions to answer for high-acuity cases |
| `generate_response.requirements` | Required sections in your disposition script |
| `govern_escalation.governance_context` | When each tier applies in your setting |
| `human_review.sla_hours` | Review SLA by tier |

Update queue names in `coordinator_config.yaml` to match your clinical workflow infrastructure.

## The four example contacts

| ID | Presentation | Expected path |
|----|-------------|---------------|
| TRIAGE-001 | Chest pain + diaphoresis, age 58 | emergent → investigate → HOLD (senior clinician) |
| TRIAGE-002 | Fever 104.2°F, 18-month-old | urgent → investigate → GATE (clinician review) |
| TRIAGE-003 | UTI symptoms, 2 days | semi-urgent → GATE (clinician spot-check) |
| TRIAGE-004 | Mild cold symptoms | non-urgent → SPOT_CHECK |

## Adapting to other triage settings

**Mental health crisis triage:** Replace severity categories with crisis acuity levels (imminent / high / moderate / low). Update investigate scope to include safety assessment framework. Update generate requirements to include safety planning and resource referrals.

**Urgent care intake triage:** Simplify to two-tier (emergent → ED, non-emergent → treat). Remove patient record retrieval. Shorten investigation scope.

**Patient deterioration monitoring:** Replace chief complaint input with vital signs and early warning score. Update classify criteria to match your deterioration scoring protocol.

## Scaffold boundary

The scaffold covers severity tier definitions, clinical assessment criteria, investigation scope, disposition script format, and governance thresholds.

Structural changes requiring workflow edits:
- Routing emergent cases to an automatic 911 dispatch system (fire-and-forget coordinator template)
- Adding a parallel specialty consult for complex presentations (wait-for-result coordinator template)
- A two-triage-nurse review for high-acuity cases (two-stage-review coordinator template)
