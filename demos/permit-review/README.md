# Permit Review Demo

Policy-grounded environmental permit review using CEQA, 14 CCR, and municipal code. Demonstrates Cognitive Core's typed cognitive primitive architecture applied to a real institutional decision domain — the kind of workflow a city or county planning department would build.

## What this demo shows

**Policy as corpus, not configuration.** The retrieve step fetches actual regulatory text — §15301 Class 1 exemption language, §15300.2 exception conditions, 44 CFR §60.3(d) floodway prohibition, MSHCP criteria cell requirements — as typed tool call responses. The classify and verify steps reason against this retrieved text. Every determination in the audit ledger cites a specific instrument and provision.

**All four governance tiers.** Three cases cover the full tier range:
- SPOT CHECK: categorical exemption confirmed, staff authority, quality monitoring
- GATE: conditional classification, EIA specialist required, planning manager review
- HOLD: statutory bar, standard pathway closed, city attorney review

**Demand-driven delegation with parallel handlers.** The conditional case (PMT-2026-00318) exercises the full delegation chain: the intake workflow classifies the application and fires the EIA specialist via fire-and-forget. The specialist investigates all CEQA Appendix G issue areas, then suspends while two parallel handlers run concurrently — public notice compliance (§21092 review period verification) and biological resources review (MSHCP criteria cell check, CNDDB species records). The specialist resumes when both return and deliberates with combined results.

## Setup

```bash
# From the repo root
GOOGLE_API_KEY=your_key python demos/permit-review/run.py
```

## Usage

```bash
# Run all three cases
python demos/permit-review/run.py

# Run a specific case by type
python demos/permit-review/run.py --permit exempt
python demos/permit-review/run.py --permit conditional
python demos/permit-review/run.py --permit prohibited

# Run a specific case file
python demos/permit-review/run.py --case cases/pmt_2026_00447.json
```

## Cases

### PMT-2026-00142 — Tenant Improvement (SPOT CHECK)
Interior renovation of existing 3,800 sq ft commercial space. Change of use from retail to restaurant in C-2 General Commercial zone. No expansion of footprint, no ground disturbance.

Expected path: retrieve → classify (exempt, §15301 Class 1) → verify (conforms) → deliberate → generate → govern (SPOT CHECK)

Policy basis: 14 CCR §15301 — Class 1 Categorical Exemption, Existing Facilities. Confirmed: no §15300.2 exceptions apply (Zone X floodplain, no sensitive area overlays, no unusual circumstances). Restaurant is a permitted use under Municipal Code §18.24.020(A)(8).

### PMT-2026-00318 — Mixed-Use Infill (GATE)
New 6-story mixed-use building, 180 residential units + 8,400 sq ft retail, 1.2-acre urban infill site, 0.4 miles from regional transit.

Expected path: retrieve → classify (conditional, §15332 fails Condition (d)) → fire EIA specialist → specialist investigates Appendix G → parallel: public notice + biological resources → specialist resumes → deliberate (MND pathway) → govern (GATE)

Policy basis: 14 CCR §15332 Class 32 In-Fill Exemption does not apply — Condition (d) cannot be self-certified for a 180-unit project without independent assessment. Initial Study required under 14 CCR §15063. SB 743 transit proximity presumption applicable (0.4 miles from regional rail). Cultural resources Phase I survey required under 14 CCR §15064.5(b).

### PMT-2026-00447 — Concrete Batch Plant, Regulatory Floodway (HOLD)
New heavy industrial facility on 4.8-acre site. Western 2.1 acres within regulatory floodway of Ridgeline Creek per FIRM Panel 06071C0891G.

Expected path: retrieve → classify (prohibited, §18.52.060(B)(1)) → verify (2 violations) → deliberate (close — prohibited use) → generate → govern (HOLD)

Policy basis: Municipal Code §18.52.060(B)(1) categorically prohibits industrial manufacturing facilities within a regulatory floodway (implementing 44 CFR §60.3(d)). No variance available. MSHCP Criteria Cell 4872 — California red-legged frog (federally threatened) and Least Bell's vireo (federally endangered) habitat documented. Municipal Code §18.50.040 requires MSHCP consistency finding before any discretionary approval.

## Architecture

```
demos/permit-review/
├── run.py                          Entry point
├── coordinator_config.yaml         Governance tiers, delegation policies
├── domains/
│   ├── permit_intake.yaml          Classify + verify + govern (intake)
│   ├── eia_specialist.yaml         Investigate all Appendix G issue areas
│   ├── public_notice.yaml          §21092 review period verification
│   └── biological_resources.yaml  MSHCP + CNDDB + ESA assessment
├── workflows/
│   ├── permit_intake.yaml          Main entry workflow
│   ├── eia_assessment.yaml         Specialist EIA workflow
│   ├── public_notice_compliance.yaml  Parallel handler
│   └── biological_resources_review.yaml  Parallel handler
└── cases/
    ├── pmt_2026_00142.json         Tenant improvement (exempt)
    ├── pmt_2026_00318.json         Mixed-use infill (conditional)
    └── pmt_2026_00447.json         Floodway industrial (prohibited)
```

## Regulatory Corpus

The cases embed actual statutory text in the tool call responses:

- **14 CCR §15301** — Class 1 Categorical Exemption (Existing Facilities)
- **14 CCR §15332** — Class 32 Categorical Exemption (In-Fill Development)
- **14 CCR §15300.2** — Exceptions to categorical exemptions
- **14 CCR Appendix G** — Environmental Checklist significance thresholds
- **Pub. Resources Code §21092** — Public review period requirements
- **44 CFR §60.3(d)** — NFIP floodway development restrictions
- **Municipal Code §18.52.060** — Floodplain Management Overlay (implements NFIP)
- **Municipal Code §18.50.040** — MSHCP consistency requirement
- **SB 743 / Pub. Resources Code §21099** — VMT analysis requirements

## Viewing the trace

If running the API server:

```bash
GOOGLE_API_KEY=your_key \
CC_COORD_CONFIG=demos/permit-review/coordinator_config.yaml \
CC_COORD_BASE=demos/permit-review \
uvicorn cognitive_core.api.server:app --port 8000
```

Then submit a case via POST /api/start and open the trace page at:
http://localhost:8000/instances/{id}/trace

The trace page shows each step as it completes, the classification with its statutory citation, the verify findings with rule-by-rule results, and the governance determination with tier rationale. On GATE or HOLD, the HITL form shows the PLANNING REVIEW queue with the full statutory record.
