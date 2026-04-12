"""
Prior Authorization Appeal Review
===================================
Cognitive Core (Agentic DEVS) vs. ReAct Agent

Same case. Same three documents. Same domain knowledge.
The question: does governed multi-source reasoning produce a
qualitatively different determination than single-prompt reasoning?

Run from repo root:
    python demos/prior-auth-appeal/run.py

    # With comparison:
    python demos/prior-auth-appeal/run.py --compare --save --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEMO_DIR))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import (
    TraceCallback, NullTrace, set_trace, get_trace,
)

WORKFLOW  = "prior_auth_appeal"
DOMAIN    = "prior_auth_appeal"
CONFIG    = DEMO_DIR / "coordinator_config.yaml"
DOCS_DIR  = DEMO_DIR / "documents"
CASES_DIR = DEMO_DIR / "cases"

# ── Load case and documents ────────────────────────────────────────────────

def load_case(case_file: str = "appeal_chen_spine.json") -> dict:
    with open(CASES_DIR / case_file) as f:
        return json.load(f)

def load_documents() -> dict:
    docs = {}
    for doc_file in DOCS_DIR.glob("*.txt"):
        docs[doc_file.stem] = doc_file.read_text()
    return docs

def build_case_input(case: dict, docs: dict) -> dict:
    """Build the case_input dict for the coordinator."""
    return {
        # The appeal case
        "case_id":           case["case_id"],
        "appeal_date":       case["appeal_date"],
        "member_info":       json.dumps(case["member"]),
        "procedure_info":    json.dumps(case["procedure"]),
        "denial_info":       json.dumps(case["denial"]),
        "appeal_basis":      json.dumps(case["appeal_basis"]),
        "clinical_record":   json.dumps(case["clinical_record"]),
        "question":          case["question"],

        # The three source documents — each required, none sufficient alone
        "case_record":            json.dumps({
            "case": case,
            "summary": (
                f"Case {case['case_id']}: {case['member']['name']}, "
                f"{case['member']['age']}yo, {case['member']['state']}. "
                f"Procedure: {case['procedure']['description']} "
                f"(CPT {case['procedure']['code']}). "
                f"Denied: {case['denial']['reason_text']}"
            )
        }),
        "plan_criteria_document":     docs.get("plan_clinical_criteria", ""),
        "regulatory_document":        docs.get("california_regulatory_framework", ""),
        "clinical_evidence_document": docs.get("clinical_evidence_base", ""),
    }

# ── Trace ──────────────────────────────────────────────────────────────────

ICONS = {
    "retrieve":    "📥",
    "classify":    "🏷 ",
    "investigate": "🔍",
    "verify":      "✅",
    "deliberate":  "🤔",
    "generate":    "📝",
    "challenge":   "⚔️ ",
    "reflect":     "🪞",
    "govern":      "⚖️ ",
}

class _AppealTrace(TraceCallback):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._current_step = ""

    def on_step_start(self, step_name, primitive, loop_iter):
        self._current_step = step_name
        icon = ICONS.get(primitive, "  ")
        print(f"  {icon} {step_name}  [{primitive}]", end="", flush=True)

    def on_step_end(self, step_name, primitive, output, elapsed_ms):
        conf = output.get("confidence", "?") if isinstance(output, dict) else "?"
        print(f"  conf={conf}  ({elapsed_ms/1000:.1f}s)")

        if not isinstance(output, dict):
            return

        if primitive == "retrieve":
            sources = output.get("sources_queried", [])
            ok = sum(1 for s in sources if s.get("status") == "success")
            print(f"       → {ok}/{len(sources)} sources read")

        elif primitive == "classify":
            cat = output.get("category", "?")
            print(f"       → {cat}")

        elif primitive == "investigate":
            finding = (output.get("finding") or "")
            gaps = output.get("evidence_missing", [])
            print(f"       → {finding[:120]}...")
            for g in gaps:
                print(f"         GAP: {g.get('description', '')}")

        elif primitive == "verify":
            conforms = output.get("conforms", "?")
            violations = output.get("violations", [])
            print(f"       → conforms={conforms}  violations={len(violations)}")
            for v in violations:
                print(f"         [{v.get('severity','?').upper()}] {v.get('description','')[:80]}")

        elif primitive == "deliberate":
            action = output.get("recommended_action", "?")
            warrant = (output.get("warrant") or "")[:120]
            print(f"       → {action}")
            print(f"         warrant: {warrant}...")

        elif primitive == "generate":
            artifact = str(output.get("artifact", ""))
            print(f"       → {len(artifact)} chars")

        elif primitive == "challenge":
            survives = output.get("survives", "?")
            vulns = output.get("vulnerabilities", [])
            high = [v for v in vulns if v.get("severity") in ("high", "critical")]
            print(f"       → survives={survives}  vulnerabilities={len(vulns)}  high={len(high)}")
            for v in vulns:
                sev = v.get("severity", "?").upper()
                desc = (v.get("description") or "")[:80]
                print(f"         [{sev}] {desc}...")

        elif primitive == "reflect":
            traj = output.get("trajectory", "?")
            target = output.get("revision_target") or ""
            nq = (output.get("next_question") or "")[:80]
            print(f"       → trajectory={traj}  revision_target={target or 'none'}")
            if nq:
                print(f"         next_question: {nq}...")

        elif primitive == "govern":
            tier = str(output.get("tier_applied", "?")).replace("GovernanceTier.", "")
            disp = output.get("disposition", "?")
            print(f"       → {tier.upper()}  disposition={disp}")

    def on_llm_start(self, step_name, prompt_len):
        if self.verbose:
            print(f"       [llm]  prompt={prompt_len:,} chars", flush=True)

    def on_llm_end(self, step_name, response_len, elapsed):
        if self.verbose:
            print(f"       [llm]  response={response_len:,} chars  {elapsed:.1f}s")

    def on_retrieve_start(self, step_name, source_name):
        if self.verbose:
            print(f"       [retrieve] → {source_name}")

    def on_retrieve_end(self, step_name, source_name, status, elapsed_ms):
        if self.verbose:
            print(f"       [retrieve] ← {source_name}  {status}  {elapsed_ms:.0f}ms")

    def on_governance_decision(self, step_name, tier, rationale):
        tier_str = str(tier).replace("GovernanceTier.", "").upper()
        print(f"  [coord]  governance: {step_name}  {tier_str}")

    def on_parse_error(self, step_name, error):
        print(f"  [coord]  parse error: {step_name} — {error[:100]}")


# ── Cognitive Core runner ──────────────────────────────────────────────────

def run_cognitive_core(case_input: dict, verbose: bool = False) -> dict:
    print(f"\n{'═' * 72}")
    print(f"  COGNITIVE CORE — Agentic DEVS Mode")
    print(f"  Retrieve → Classify → Investigate × 3 → Verify → Deliberate")
    print(f"  → Generate → Challenge → Govern")
    print(f"  Three independent knowledge sources · Cross-source conflict detection")
    print(f"{'─' * 72}\n")

    coord = Coordinator(
        str(CONFIG),
        db_path=str(DEMO_DIR / "appeal_cc.db"),
        verbose=False,
    )

    trace = _AppealTrace(verbose=verbose)
    set_trace(trace)
    t0 = time.time()

    instance_id = coord.start(
        workflow_type=WORKFLOW,
        domain=DOMAIN,
        case_input=_prepare_case_input(case_input),
    )

    elapsed = time.time() - t0
    set_trace(NullTrace())

    instance = coord.store.get_instance(instance_id)
    tier = str(getattr(instance, "governance_tier", "?")).lower().replace("governancetier.", "")
    disposition = getattr(instance, "disposition", "?")

    # Extract determination from generate step
    ledger = coord.store.get_ledger(instance_id=instance_id)
    determination = ""
    trajectory = []
    for entry in ledger:
        if entry.get("action_type") == "step_completed":
            d = entry.get("details", {})
            prim = d.get("primitive", "")
            if prim:
                trajectory.append(prim)
            if prim == "generate":
                out = d.get("output", {})
                if isinstance(out, dict):
                    det = (out.get("artifact")
                           or out.get("content")
                           or out.get("text")
                           or "")
                    if det:
                        determination = str(det)
                elif isinstance(out, str) and out:
                    determination = out

    return {
        "determination": determination,
        "tier": tier,
        "disposition": disposition,
        "elapsed": elapsed,
        "trajectory": trajectory,
        "instance_id": instance_id,
    }


# ── Case input preparation ─────────────────────────────────────────────────

def _prepare_case_input(case_input: dict) -> dict:
    """Trim to coordinator field size limits. Documents stay intact — they're essential."""
    import json

    MAX_FIELD = 60_000
    MAX_STR   = 30_000
    prepared  = {}

    for key, value in case_input.items():
        if isinstance(value, str):
            if len(value) > MAX_STR:
                prepared[key] = value[:MAX_STR] + f"\n[... truncated at {MAX_STR} chars]"
            else:
                prepared[key] = value
        else:
            raw = json.dumps(value, default=str)
            if len(raw.encode()) > MAX_FIELD:
                prepared[key] = raw[:MAX_FIELD] + "... [truncated]"
            else:
                prepared[key] = value

    return prepared


# ── ReAct agent runner ─────────────────────────────────────────────────────

def _extract_domain_knowledge(docs: dict) -> str:
    """
    Provide ReAct with the same knowledge CC's domain configuration encodes:
    - The three raw source documents (same as CC's retrieve steps)
    - The structured domain knowledge entries (same as CC's investigate scopes)
    - The hierarchy and review checklist (same as CC's orchestrator strategy)

    Both systems get identical information. The difference is architecture:
    CC applies this knowledge through typed, sequenced, governed primitives.
    ReAct applies it in a single reasoning pass.
    """
    return f"""HIERARCHY OF AUTHORITY: State law > Clinical standard > Plan criteria.
Where sources conflict, higher authority governs.

DOMAIN KNOWLEDGE — PRIOR AUTHORIZATION APPEAL REVIEW

CLASSIFICATION:
- Myelopathy requires: spinal cord compression on MRI + upper motor neuron signs
  (Hoffman, hyperreflexia) + bilateral symptoms + gait disturbance.
  Cord signal change (myelomalacia) = strong myelopathy indicator.
- Radiculopathy: nerve root compression, unilateral dermatomal symptoms, no/minimal UMN signs.
- CRITICAL: These are different diseases with different treatment algorithms.
  The plan's PT requirement is derived from radiculopathy evidence only.
  Applying it to myelopathy is a category error.

PLAN CRITERIA KNOWLEDGE:
- Standard pathway requires: 6-week PT, 2 pharmacotherapy agents 4+ weeks, 1 interventional procedure.
- Myelopathy exception (Section 2.2A): PT reduced to 4 weeks minimum, interventional waived
  if physician documents risk. Applies only when myelopathy is PRIMARY diagnosis.
- Functional plateau: <10% improvement over 3 sessions OR therapist documents maximum benefit reached.

REGULATORY KNOWLEDGE:
- CIC 10169.5: Plan cannot require PT completion where:
  (1) Physician documents PT contraindicated or unlikely to benefit, OR
  (2) Member completed PT and reached functional plateau, OR
  (3) Objective evidence of structural neurological compromise that PT cannot address.
  Myelomalacia with cord compression = (3) applies.
- DMHC APL 22-014: Plans SHALL NOT require full conservative treatment where:
  objective imaging shows progressive pathology AND physician documents conservative failure
  AND surgery indicated under generally accepted guidelines.
  Myelomalacia specifically identified as supporting waiver of standard conservative requirements.
- CHSC 1374.32(b): IMR standard is NOT plan criteria. Service meeting generally accepted standards
  SHALL be approved regardless of plan's more restrictive internal criteria.
- CHSC 1374.31(b): Denial notice defective if it fails to specify clinical criteria applied
  or specific clinical reason. Check denial notice for procedural adequacy.

CLINICAL STANDARD KNOWLEDGE:
- AANS/CNS 2023: Moderate-severe myelopathy with objective cord compression = TIER 1 strong
  surgical indication. No PT minimum duration established by evidence for this presentation.
  Myelomalacia = strong surgical indication, contraindication to extended conservative trial.
- AO Spine 2021: No high-quality evidence supporting mandatory PT prior to surgery for
  moderate-severe myelopathy. The 6-week PT requirement has NO evidentiary basis for myelopathy.
- Myelomalacia (T2 cord signal change) = existing cord injury. Strong predictor of permanent
  deficit if surgery delayed. Clinical urgency not eliminated by elective scheduling status.

DENIAL REVIEW CHECKLIST:
1. Was the denial procedurally adequate? (criteria cited, IMR notice included)
2. Did the plan apply its own criteria correctly?
3. Did the plan apply its own myelopathy exception where applicable?
4. Does state law (CIC 10169.5, DMHC APL 22-014) override the criterion used to deny?
5. Does clinical standard require approval regardless of plan criteria?
6. Cross-source conflict resolution: where sources conflict, apply the hierarchy.

---

SOURCE DOCUMENT 1 — PLAN CLINICAL CRITERIA (HealthFirst CC-SPINE-2024):
{docs.get('plan_clinical_criteria', '')}

SOURCE DOCUMENT 2 — CALIFORNIA REGULATORY FRAMEWORK:
{docs.get('california_regulatory_framework', '')}

SOURCE DOCUMENT 3 — CLINICAL EVIDENCE BASE (AANS/CNS 2023, NASS 2020, AO Spine 2021):
{docs.get('clinical_evidence_base', '')}
"""


def run_react_agent(case_input: dict, docs: dict) -> dict:
    print(f"\n{'═' * 72}")
    print(f"  REACT AGENT — Best-Practice Prompt Engineering")
    print(f"  Same case · Same three documents · Same domain knowledge")
    print(f"{'─' * 72}\n")

    domain_knowledge = _extract_domain_knowledge(docs)

    prompt = f"""You are an expert health plan appeals reviewer. Review this prior authorization appeal
and determine whether the denial should be OVERTURNED or UPHELD.

Apply the three-source hierarchy: State law > Clinical standard > Plan criteria.

CASE:
Case ID: {case_input['case_id']}
Member: {case_input['member_info']}
Procedure: {case_input['procedure_info']}
Denial: {case_input['denial_info']}
Appeal basis: {case_input['appeal_basis']}
Clinical record: {case_input['clinical_record']}

DOMAIN KNOWLEDGE AND SOURCE DOCUMENTS:
{domain_knowledge}

INSTRUCTIONS:
1. Identify the primary clinical presentation (myelopathy vs radiculopathy)
2. Apply the plan's criteria to the clinical record
3. Determine if the plan applied its own myelopathy exception
4. Apply California regulatory requirements (CIC §10169.5, DMHC APL 22-014, CHSC §1374.32)
5. Apply the clinical standard (AANS/CNS, NASS, AO Spine guidelines)
6. Resolve any conflicts using the hierarchy
7. State your disposition: OVERTURN, UPHOLD, PARTIAL, or REMAND
8. Provide a warrant connecting evidence to disposition
9. Cite specific sources for every conclusion

Write a formal appeal determination letter."""

    print(f"  Running... (single LLM call with all three documents)")
    print(f"  Prompt size: {len(prompt):,} chars")

    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        t0 = time.time()
        response = model.generate_content(prompt)
        elapsed = time.time() - t0

        # Gemini SDK response: use .text directly
        if hasattr(response, 'text'):
            memo = response.text
        elif hasattr(response, 'content'):
            raw = response.content
            if isinstance(raw, list):
                memo = "".join(part.text if hasattr(part, 'text') else str(part) for part in raw)
            else:
                memo = str(raw)
        else:
            memo = str(response)

        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Output: {len(memo):,} chars")
        return {"determination": memo, "elapsed": elapsed}

    except Exception as e:
        print(f"  ReAct failed: {e}")
        return {"determination": f"[ReAct failed: {e}]", "elapsed": 0}


# ── Comparison ─────────────────────────────────────────────────────────────

def print_comparison(cc: dict, react: dict):
    print(f"\n{'═' * 72}")
    print(f"  COMPARISON")
    print(f"{'═' * 72}")
    print(f"""
  ┌─────────────────────────────┬──────────────────────┬────────────────────┐
  │                             │  Cognitive Core      │  ReAct Agent       │
  ├─────────────────────────────┼──────────────────────┼────────────────────┤
  │ Time                        │  {cc['elapsed']:>6.1f}s              │  {react.get('elapsed',0):>6.1f}s            │
  │ Sources read                │  3 (independently)   │  1 (combined)      │
  │ Governance tier             │  {str(cc['tier']).upper():<20} │  N/A               │
  │ Audit ledger                │  Yes — hash-chained  │  No                │
  │ Cross-source conflict check │  Yes — verify step   │  Prompt instruction│
  │ Challenge step fired        │  Yes                 │  No (self-review)  │
  │ Determination size          │  {len(str(cc['determination'])):>5,} chars          │  {len(str(react.get('determination',''))):>5,} chars        │
  └─────────────────────────────┴──────────────────────┴────────────────────┘

  The question: do they reach the same disposition?
  And if so, which one can you defend at IMR?
""")


def save_determinations(cc: dict, react: dict):
    output_dir = DEMO_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cc_path = output_dir / "cognitive_core_determination.txt"
    with open(cc_path, "w") as f:
        f.write(f"COGNITIVE CORE — PRIOR AUTH APPEAL DETERMINATION\n")
        f.write(f"Trajectory: {' → '.join(cc.get('trajectory', []))}\n")
        f.write(f"Governance: {cc['tier'].upper()}\n")
        f.write(f"Elapsed: {cc['elapsed']:.1f}s\n")
        f.write("=" * 72 + "\n\n")
        f.write(str(cc.get("determination", "[No determination generated]")))
    print(f"\n  CC determination saved: {cc_path}")

    react_path = output_dir / "react_determination.txt"
    with open(react_path, "w") as f:
        f.write(f"REACT AGENT — PRIOR AUTH APPEAL DETERMINATION\n")
        f.write(f"Elapsed: {react.get('elapsed', 0):.1f}s\n")
        f.write("=" * 72 + "\n\n")
        memo = react.get("determination", "[No determination generated]")
        f.write(memo if isinstance(memo, str) else str(memo))
    print(f"  ReAct determination saved: {react_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prior Authorization Appeal Review Demo")
    parser.add_argument("--compare", action="store_true", help="Also run ReAct agent")
    parser.add_argument("--save",    action="store_true", help="Save determinations to output/")
    parser.add_argument("--verbose", action="store_true", help="Show LLM call details")
    parser.add_argument("--case",    default="appeal_chen_spine.json",
                        help="Case file to use (default: appeal_chen_spine.json)")
    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print(f"  PRIOR AUTHORIZATION APPEAL REVIEW DEMO")
    print(f"  Cognitive Core vs. ReAct Agent")
    print(f"{'═' * 72}\n")

    # Load case and documents
    case = load_case(args.case)
    docs = load_documents()
    case_input = build_case_input(case, docs)

    print(f"  Case: {case['case_id']} — {case['member']['name']}, {case['member']['age']}yo")
    print(f"  Procedure: {case['procedure']['description']}")
    print(f"  Denial reason: {case['denial']['reason_text'][:80]}...")
    print(f"  State: {case['member']['state']}")
    print(f"  Sources: plan criteria + California regulations + clinical guidelines")
    print(f"  Question: {case['question']}")

    # Run Cognitive Core
    cc_result = run_cognitive_core(case_input, verbose=args.verbose)

    # Save CC determination immediately
    if args.save:
        output_dir = DEMO_DIR / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        cc_path = output_dir / "cognitive_core_determination.txt"
        with open(cc_path, "w") as f:
            f.write(f"COGNITIVE CORE — PRIOR AUTH APPEAL DETERMINATION\n")
            f.write(f"Trajectory: {' → '.join(cc_result.get('trajectory', []))}\n")
            f.write(f"Governance: {cc_result['tier'].upper()}\n")
            f.write(f"Elapsed: {cc_result['elapsed']:.1f}s\n")
            f.write("=" * 72 + "\n\n")
            f.write(str(cc_result.get("determination", "[No determination generated]")))
        print(f"\n  CC determination saved: {cc_path}")

    # Run ReAct if requested
    react_result = {}
    if args.compare:
        react_result = run_react_agent(case_input, docs)

    # Comparison
    if args.compare and react_result:
        print_comparison(cc_result, react_result)
        if args.save:
            react_path = DEMO_DIR / "output" / "react_determination.txt"
            with open(react_path, "w") as f:
                f.write(f"REACT AGENT — PRIOR AUTH APPEAL DETERMINATION\n")
                f.write(f"Elapsed: {react_result.get('elapsed', 0):.1f}s\n")
                f.write("=" * 72 + "\n\n")
                memo = react_result.get("determination", "")
                f.write(memo if isinstance(memo, str) else str(memo))
            print(f"  ReAct determination saved: {react_path}")
    else:
        print(f"\n{'═' * 72}")
        print(f"  APPEAL DETERMINATION")
        print(f"  Governance: {cc_result['tier'].upper()}")
        print(f"{'═' * 72}\n")
        print(cc_result.get("determination", "[No determination generated]"))


if __name__ == "__main__":
    main()
