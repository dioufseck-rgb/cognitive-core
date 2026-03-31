# Domain Library Architecture

*How the Cognitive Core domain library is organised, what each layer does, and how to extend it.*

---

## What this library is

The domain library makes the framework's core claim operational: a new institutional decision domain requires configuration, not construction.

The library supplies the workflow side of that equation — canonical patterns, composable overlays, and coordinator templates so well-designed that domain knowledge is the only variable a new adopter needs to supply.

The library is evidence that the framework works and an onboarding path for teams adopting it. It is not the product. Domain packs are starter kits — the right onboarding unit, not the right long-term packaging unit. A production deployment will assemble composable workflow configurations that the packs anticipated but did not fully define.

---

## The four layers

The library is organised around four composable layers. Each has a distinct job. The layers compose — a domain scaffold declares which pattern, overlays, and coordinator template it uses, and the framework assembles the rest.

| Layer | Job | Who touches it |
|-------|-----|----------------|
| **Pattern** | The epistemic sequence — which primitives fire in what order. 13 patterns cover the institutional decision space. Stable by design. | Framework maintainers. Patterns absorb the majority of structural variation. |
| **Overlay** | A composable modifier injected into a pattern at load time. Adds a Challenge step, a delegation work order, an exception branch, or a human checkpoint. | Framework maintainers define overlays. Domain scaffolds declare which ones apply. |
| **Coordinator template** | The structural relationship between workflows — single workflow with governance, or multi-workflow with delegation chains, contracts, and suspend/resume semantics. | Framework maintainers define templates. Domain scaffolds declare which one applies. |
| **Domain scaffold** | Declares pattern + overlays + coordinator template. Fills in domain-specific parameters: categories, rules, criteria, governance thresholds, delegation conditions. | Domain owners. Covers the majority of domain variation. |

### The scaffold boundary

The scaffold covers: classification categories and criteria, verification rules, investigation scope, deliberation outcome space and criteria, report format and requirements, governance tier thresholds, delegation trigger conditions and input mappings.

The scaffold does not cover: fundamentally different step sequencing, multiple adversarial perspectives in a single step, non-standard report schemas for edge-case populations, or new coordinator topology requirements. These require pattern or overlay work — which is the correct signal that the framework needs extension, not that something is wrong.

### Why domain packs are the primary surface

The four-layer model is architecturally correct but not the right starting point for a new adopter. Asking someone to identify their pattern, select their overlays, and choose a coordinator template before writing any domain knowledge creates three degrees of freedom before they have produced anything.

The domain pack collapses this: the composition is already made, the overlays are pre-declared, the coordinator template is chosen. The domain owner fills in their expertise and runs. The layers are visible in each pack's scaffold header for users who want to understand or customise them.

---

## The 13 canonical patterns

These patterns cover the institutional decision space. Each has a distinct objective, a recognisable primitive sequence, a distinct output artifact, and a recognisable governance posture.

| # | Pattern | Default sequence | Gov. posture |
|---|---------|-----------------|--------------|
| P01 | Intake and Routing | Retrieve → Classify → Generate → Govern | Auto / Spot-check |
| P02 | Eligibility Determination | Retrieve → Classify → Verify → Deliberate → Generate → Govern | Gate |
| P03 | Exception and Variance | Retrieve → Investigate → Deliberate → Verify → Generate → Govern | Gate / Hold |
| P04 | Adversarial Review | Retrieve → Challenge → Verify → Deliberate → Generate → Govern | Gate |
| P05 | Investigation and Reporting | Retrieve → Investigate → Verify → Deliberate → Generate → Govern | Gate / Hold |
| P06 | Triage and Escalation | Retrieve → Classify → Deliberate → Generate → Govern | Auto → Gate |
| P07 | Compliance and Conformance | Retrieve → Verify → Investigate → Deliberate → Generate → Govern | Gate / Hold |
| P08 | Case Resolution | Retrieve → Deliberate → Verify → Generate → Govern | Gate |
| P09 | Monitoring and Alert Review | Retrieve → Classify → Verify → Deliberate → Govern | Auto → Gate |
| P10 | Evidence Reconciliation | Retrieve → Investigate → Verify → Deliberate → Generate → Govern | Gate |
| P11 | Approval and Authorization | Retrieve → Deliberate → Verify → Generate → Govern | Gate |
| P12 | Remediation and Action Planning | Retrieve → Investigate → Deliberate → Generate → Govern | Gate |
| P13 | Periodic Review and Recertification | Retrieve → Verify → Investigate → Deliberate → Generate → Govern | Spot-check / Gate |

**v1 status:** P02, P04, P05, P06, P07 are implemented with full workflow YAMLs. P01, P03, P08–P13 are documented in this taxonomy and on the roadmap for v2.

### Make it a pattern if:

- It has a distinct institutional objective recognisable across industries
- It has a recognisable primitive sequence not simply derivable from another pattern
- It produces a distinct output artifact
- Its governance posture is recognisable across industries

If something is a domain-specific variation of another pattern — different categories, rules, thresholds — it is a domain scaffold, not a new pattern.

---

## The 5 overlays

Overlays are composable modifiers applied at load time. Each injects steps, transitions, or parameters into a base pattern without modifying the pattern itself. Multiple overlays can apply to the same pattern.

| Overlay | What it injects | Apply when |
|---------|----------------|------------|
| **adversarial** | Challenge step after Investigate or Classify. Requires `perspective` and `threat_model` parameters. | Decision stakes are high. Fraud, abuse risk. Formal counter-position required. |
| **delegation** | Typed work-order delegation. Activates coordinator template wiring. | Specialist input needed from another workflow. Suspend/resume required. |
| **exception** | Conditional branch into exception-handling path with compensating controls. | Standard rule doesn't fit. Deviation may be justified. Non-standard approval required. |
| **human-review** | Structured brief generation and mandatory human checkpoint. | Policy requires human approval unconditionally. Case ambiguity is structurally high. |
| **governance** | Govern step with tier resolution and work order. **Always applied.** | Always. |

**v1 note:** Overlays are pre-composed into domain pack variants in v1. Full overlay injection by the composer is a v2 feature.

### Make it an overlay if:

- It modifies many patterns in the same structural way
- It does not define a full workflow objective by itself
- Removing it from a pattern leaves a complete, valid workflow

Note: P04 (Adversarial Review) exists as a standalone pattern for cases where adversarial testing IS the primary workflow objective — not a modifier. Use the overlay when adversarial testing modifies another objective.

---

## The 7 coordinator templates

Coordinator templates define the structural relationship between workflows. Users select the template that matches their structural need. Most domains use `simple.yaml`.

| Template | Structure | Use when |
|----------|-----------|----------|
| **simple** | Single workflow. Governance tiers, quality gates, HITL routing. | Decision is self-contained. |
| **fire-and-forget** | Primary completes. Handler fires independently. | Handler result not needed by primary. Background process. |
| **wait-for-result** | Primary fires one handler and suspends. Resumes when handler returns. | One specialist workflow required. |
| **parallel-handlers** | Primary fires N handlers simultaneously. Suspends until all return. | Multiple parallel specialist assessments required. |
| **sequential-lifecycle** | A fires B on completion. B fires C. Results flow forward. | Distinct sequential stages, each building on the prior. |
| **two-stage-review** | First-line produces draft. Second-line reviews with return path. | First-line / second-line structure. Appeals. Peer review. |
| **hub-and-spoke** | Orchestrator coordinates N parallel specialists. None know each other. | Complex dockets requiring independent parallel assessments. |

### Coordinator semantics: where real complexity lives

The coordinator templates appear as a selection menu. They are more than that. The coordination layer — who waits for whom, what constitutes completion, what gets injected back on resume, how handler timeouts are handled, what happens when handlers conflict — is where most of the genuine complexity in multi-workflow institutional decisions accumulates.

Each template documents its failure modes explicitly. Read them before using any multi-workflow template in production.

### Make it a coordinator template if:

- It defines a structural relationship between two or more workflows
- The same structural shape appears across many different domain combinations
- It specifies delegation mode, contracts, input mappings, or resume points

---

## The v1 domain packs

| Pack | Pattern | Overlays | Coordinator |
|------|---------|----------|-------------|
| Fraud Investigation | P05 | governance | parallel-handlers |
| Consumer Lending | P02 | exception, governance | simple |
| Content Moderation | P04 | governance | simple |
| Clinical Triage | P06 | human-review, governance | simple |
| Compliance Review | P07 | exception, governance | simple |

---

## Adding a new domain pack

**Test for fit first:**

> Can the workflow be written as a fixed sequence of epistemic steps, and can the domain expertise be written as explicit parameters?

If yes — the domain belongs in the library. If the answer is "it depends on the case in ways I can't enumerate" — the framework handles the structured portion and hands off at the point of irreducible judgment. Document that boundary in the README.

**Build steps:**

1. Identify the matching pattern from the taxonomy above
2. Copy the pattern's `workflow.yaml` to your new pack directory as `workflows/<workflow_name>.yaml`
3. Copy the appropriate coordinator template as `coordinator_config.yaml`; add `workflow_dir`, `domain_dir`, `case_dir` directives
4. Write `<domain_name>.yaml` — declare the pattern, overlays, and coordinator, then fill in each step's parameters
5. Write 3–5 example cases covering the main paths through the workflow
6. Write `run.py` following the pattern in existing packs (use `Coordinator`, not `CoordinatorRuntime`)
7. Write `README.md` — what the pack does, what to fill in, scaffold boundary

---

## Extension: v1 boundary and roadmap

In v1, framework maintainers define patterns, overlays, and coordinator templates. Domain owners fill in scaffolds. This is the right boundary for launch — it keeps the architecture coherent.

Enterprise customers will eventually ask whether they can author their own patterns or overlays. In v1: yes, you can define a new workflow YAML that references the primitive layer directly. What you cannot do yet is register it in the library taxonomy, apply overlays to it, or compose it with coordinator templates through the scaffold declaration system. That requires framework work — which is the appropriate cost signal for structural customisation.

A governed extension API — a way for customers to define new patterns or overlays within a validated namespace — is the right v2 or v3 roadmap item. The boundary is not a closed door. It is a clear cost signal: scaffold customisation is free, structural customisation has a cost.

---

## Honest limits

The library works when the decision structure is stable, the rules are statable, the outcome space is bounded, and variation lives in the data.

**Open-ended reasoning.** Litigation strategy, novel clinical synthesis, M&A judgment — the right answer depends on facts that emerge through process or resists typed output contracts. The framework handles the structured portion; the open-ended portion requires a human gate.

**Real-time decisions.** Sub-second fraud scoring, live transaction blocking. These require rule engines or ML pipelines. The framework handles investigation and disposition of flagged cases after the fact.

**Cross-instance learning.** Each instance is independent. Improvement requires manual domain scaffold updates or an external fine-tuning pipeline.
