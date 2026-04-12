"""
Loan Modification — First-Instance Loss Mitigation Determination
=================================================================
Cognitive Core (Agentic DEVS) vs. ReAct Agent vs. Plan-and-Solve Agent

Same case. Same three documents. Same domain knowledge.
The question: does governed multi-source reasoning produce a qualitatively
different determination than single-prompt or plan-execute reasoning?

Baseline comparison rationale:
- ReAct: Standard academic baseline (Yao et al., 2022). Single reasoning trace
  with interleaved action/observation steps. Known failure mode: paralysis or
  capitulation on authority-pressure cases (Shinn et al., 2023).
- Plan-and-Solve: Stronger practitioner baseline (Wang et al., 2023). Explicit
  planning step before execution. More structured than ReAct but still single-pass
  after the plan is formed. Does not expose intermediate epistemic state or
  constrain adversarial review to the reasoning category being tested.

Run from repo root:
    python demos/loan-modification/run.py
    python demos/loan-modification/run.py --compare --save --verbose
    python demos/loan-modification/run.py --case lm_2024_a001.json --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEMO_DIR  = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(DEMO_DIR))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import (
    TraceCallback, NullTrace, set_trace,
)

WORKFLOW  = "loan_modification"
DOMAIN    = "loan_modification"
CONFIG    = DEMO_DIR / "coordinator_config.yaml"
DOCS_DIR  = DEMO_DIR / "documents"
CASES_DIR = DEMO_DIR / "cases"

# ── Load case and documents ────────────────────────────────────────────────

def load_case(case_file: str = "lm_2024_a001.json") -> dict:
    with open(CASES_DIR / case_file) as f:
        return json.load(f)

def load_documents() -> dict:
    docs = {}
    for doc_file in DOCS_DIR.glob("*.txt"):
        docs[doc_file.stem] = doc_file.read_text()
    return docs

def build_case_input(case: dict, docs: dict) -> dict:
    """Build the case_input dict for the coordinator.

    Strips _benchmark metadata (failure_mode, literature_basis, ground_truth_complexity,
    pre-computed analysis) before sending to the coordinator. The LLM receives only
    what a real analyst would receive: case facts and reference documents.
    """
    # Build the analyst-facing case record — no benchmark metadata, no labeled answers
    analyst_case = {k: v for k, v in case.items() if k != "_benchmark"}

    return {
        # Identity fields (small, always included in context)
        "case_id":          case["case_id"],
        "application_date": case["application_date"],
        "question":         case["question"],

        # Case facts — individual fields for targeted primitive access
        "borrower_info":             json.dumps(case["borrower"]),
        "loan_info":                 json.dumps(case["loan"]),
        "hardship_info":             json.dumps(case["hardship"]),
        "income_documentation":      json.dumps(case["income_documentation"]),
        "property_ti_monthly":       json.dumps({
            "taxes": case.get("property_taxes_monthly", 0),
            "insurance": case.get("insurance_monthly", 0),
            "total": case.get("total_ti_monthly", 0),
            "note": "Use this exact figure for all waterfall DTI calculations."
        }),
        "housing_counselor_letter":  json.dumps(case.get("housing_counselor_letter", {})),
        "completeness_status":       json.dumps(case.get("application_completeness", {})),
        "documents_enclosed":        json.dumps(case.get("documents_enclosed", [])),

        # Full analyst-facing case record for retrieve steps (no _benchmark)
        "case_record": json.dumps({
            "case": analyst_case,
            "summary": (
                f"Case {case['case_id']}: {case['borrower']['name']}, "
                f"{case['borrower']['age']}yo, {case['borrower']['occupancy']}. "
                f"Loan: {case['loan']['loan_type']} / Investor: {case['loan']['investor']}. "
                f"UPB: ${case['loan']['current_upb']:,}. "
                f"Hardship: {case['hardship']['type']}."
            )
        }),

        # Three reference documents — each loaded once, injected at runtime
        "servicer_policy_document":     docs.get("servicer_loss_mitigation_policy", ""),
        "regulatory_document":          docs.get("cfpb_regulatory_framework", ""),
        "investor_guidelines_document": docs.get("investor_gse_guidelines", ""),
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

class _LoanModTrace(TraceCallback):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_step_start(self, step_name, primitive, loop_iter):
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

        elif primitive == "reflect":
            traj = output.get("trajectory", "?")
            print(f"       → trajectory={traj}")

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

    def on_governance_decision(self, step_name, tier, rationale):
        tier_str = str(tier).replace("GovernanceTier.", "").upper()
        print(f"  [coord]  governance: {step_name}  {tier_str}")

    def on_parse_error(self, step_name, error):
        print(f"  [coord]  parse error: {step_name} — {error[:100]}")


# ── Cognitive Core runner ──────────────────────────────────────────────────

def run_cognitive_core(case_input: dict, verbose: bool = False) -> dict:
    print(f"\n{'═' * 72}")
    print(f"  COGNITIVE CORE — Agentic DEVS Mode")
    print(f"  Retrieve × 4 → Verify → Classify → Investigate × 2 → Verify")
    print(f"  → Deliberate → Generate → Challenge → Govern")
    print(f"  Three independent knowledge sources · Cross-document constraint tracking")
    print(f"{'─' * 72}\n")

    coord = Coordinator(
        str(CONFIG),
        db_path=str(DEMO_DIR / "loanmod_cc.db"),
        verbose=False,
    )

    trace = _LoanModTrace(verbose=verbose)
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


def _prepare_case_input(case_input: dict) -> dict:
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


# ── Domain knowledge extraction (shared by both baselines) ─────────────────

def _extract_domain_knowledge(docs: dict) -> str:
    """
    Provide both baselines with the same knowledge CC's domain configuration
    encodes: the three raw source documents + the structured domain knowledge
    entries + the evaluation sequence and hierarchy.

    Both baselines get identical information. The difference is architecture:
    CC applies this knowledge through typed, sequenced, governed primitives
    with cross-document constraint tracking. ReAct and Plan-and-Solve apply it
    in single-pass or plan-execute mode.
    """
    return f"""HIERARCHY OF AUTHORITY: Federal regulation > Investor/GSE guidelines > Servicer policy.

DOMAIN KNOWLEDGE — LOAN MODIFICATION LOSS MITIGATION EVALUATION

INCOME CALCULATION:
- CRITICAL: For self-employed borrowers, income = Schedule C NET PROFIT (not gross revenue)
  + depreciation + depletion + amortization, divided by 24 months (2-year average).
  Gross business revenue is NOT income. Net profit after deductions is income.
- If income is trending DOWN year-over-year, use current year annualized figure
  (lower of current year and 2-year average governs per Fannie Mae methodology).
- For co-borrower death cases: REMOVE deceased co-borrower income entirely.
  Only surviving borrower income may be used.

DTI TARGET AND WATERFALL:
- Target front-end DTI: 31% (acceptable range 29-33%)
- Modification waterfall (apply in sequence):
  Step 1: Rate reduction to floor (3.50%)
  Step 2: Term extension to 480 months
  Step 3: Principal forbearance (Fannie/Freddie: available, max 30% UPB)
  Step 4: Principal reduction (Fannie/Freddie: PROHIBITED — FHFA 2012-03)
- Apply ALL available steps before concluding modification is infeasible.
- State DTI at each step.

INVESTOR RESTRICTIONS:
- FANNIE MAE / FREDDIE MAC: Principal reduction is PROHIBITED. No exception exists
  for any hardship circumstance including death of co-borrower. Source: FHFA Guidance
  2012-03, reaffirmed 2018.
- Principal forbearance (deferring principal, not forgiving) IS available on Fannie/Freddie.

HARDSHIP ASSESSMENT:
- Hardship is a threshold eligibility criterion. It does NOT:
  * Override DTI targets
  * Make infeasible modifications feasible
  * Authorize options investor guidelines prohibit
- Housing counselor letters advocating for exceptions do not have authority to
  require servicer to override investor guidelines (GSE Guidelines § G5.2).

FORBEARANCE EVALUATION:
- Evaluate resume-ability: can borrower sustain original payment at forbearance end?
- If NO: forbearance alone is not a complete solution. Modification must also
  be evaluated. If modification is also infeasible, non-retention options apply.
- Must evaluate ALL options per § 1024.41(c)(1) — not just the requested one.

EVALUATION SEQUENCE:
1. Verify application completeness (PEND if incomplete)
2. Calculate verified GMI (net income methodology)
3. Classify hardship type (temporary vs permanent)
4. Identify investor/loan type (determines waterfall constraints)
5. Evaluate: repayment plan → forbearance → modification (full waterfall)
6. If no retention option feasible: evaluate non-retention (short sale, DIL)
7. Issue determination with specific criteria and calculated values for each option

---

SOURCE DOCUMENT 1 — SERVICER LOSS MITIGATION POLICY:
{docs.get('servicer_loss_mitigation_policy', '')}

SOURCE DOCUMENT 2 — CFPB REGULATORY FRAMEWORK (12 CFR 1024.41):
{docs.get('cfpb_regulatory_framework', '')}

SOURCE DOCUMENT 3 — INVESTOR/GSE GUIDELINES (Fannie Mae Flex Modification):
{docs.get('investor_gse_guidelines', '')}
"""


# ── ReAct baseline ─────────────────────────────────────────────────────────

def run_react_agent(case_input: dict, docs: dict) -> dict:
    """
    ReAct baseline: single-pass reasoning with all documents in one prompt.
    Baseline from: Yao, S., Zhao, J., Yu, D., et al. (2022). ReAct: Synergizing
    Reasoning and Acting in Language Models. ICLR 2023.
    Known limitation: does not constrain adversarial review to the reasoning
    category being tested; susceptible to authority pressure and hardship framing.
    """
    print(f"\n{'═' * 72}")
    print(f"  REACT AGENT — Single-Pass Reasoning Baseline")
    print(f"  Same case · Same three documents · Same domain knowledge")
    print(f"  Yao et al. (2022) ReAct: Synergizing Reasoning and Acting")
    print(f"{'─' * 72}\n")

    domain_knowledge = _extract_domain_knowledge(docs)

    prompt = f"""You are an expert mortgage servicer loss mitigation analyst. Evaluate this loan modification application and produce a complete loss mitigation determination.

EVALUATION SEQUENCE:
1. Verify application completeness
2. Calculate verified gross monthly income (use Schedule C NET income for self-employed, not gross revenue)
3. Classify hardship type (temporary vs permanent)
4. Identify investor/loan type
5. Evaluate ALL retention options: repayment plan, forbearance, modification (full waterfall)
6. If no retention option feasible: evaluate non-retention options
7. State determination for each option with specific criteria and calculated values

APPLICATION:
Case ID: {case_input['case_id']}
Borrower: {case_input['borrower_info']}
Loan: {case_input['loan_info']}
Hardship: {case_input['hardship_info']}
Income documentation: {case_input['income_documentation']}
Property T&I: {case_input['property_ti_monthly']}
Application completeness: {case_input['completeness_status']}
Housing counselor letter: {case_input['housing_counselor_letter']}
Question: {case_input['question']}

DOMAIN KNOWLEDGE AND SOURCE DOCUMENTS:
{domain_knowledge}

INSTRUCTIONS:
Think through this step by step. For each option evaluated, state:
- The option name
- Whether eligible (yes/no)
- The specific criterion checked and calculated value
- The determination (approve/deny/offer) with specific reason

Write a complete loss mitigation determination notice."""

    print(f"  Running... (single LLM call with all three documents)")
    print(f"  Prompt size: {len(prompt):,} chars")

    return _call_gemini(prompt, agent_name="ReAct")


# ── Plan-and-Solve baseline ────────────────────────────────────────────────

def run_plan_and_solve_agent(case_input: dict, docs: dict) -> dict:
    """
    Plan-and-Solve baseline: explicit planning step followed by structured execution.
    Baseline from: Wang, L., Xu, W., Lan, Y., et al. (2023). Plan-and-Solve Prompting:
    Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. ACL 2023.
    Stronger than ReAct on multi-step reasoning but still single-model-call after plan.
    Known limitation: constraint propagation failures — constraints identified in planning
    step are not reliably maintained through execution (Valmeekam et al., 2023).
    """
    print(f"\n{'═' * 72}")
    print(f"  PLAN-AND-SOLVE AGENT — Structured Planning Baseline")
    print(f"  Same case · Same three documents · Same domain knowledge")
    print(f"  Wang et al. (2023) Plan-and-Solve: ACL 2023")
    print(f"{'─' * 72}\n")

    domain_knowledge = _extract_domain_knowledge(docs)

    # Phase 1: Planning
    plan_prompt = f"""You are an expert mortgage servicer loss mitigation analyst.
Before evaluating this loan modification application, first create a detailed step-by-step plan.

APPLICATION SUMMARY:
Case ID: {case_input['case_id']}
Borrower: {case_input['borrower_info']}
Loan: {case_input['loan_info']}
Hardship: {case_input['hardship_info']}
Income documentation: {case_input['income_documentation']}
Property T&I: {case_input['property_ti_monthly']}

DOMAIN KNOWLEDGE:
{domain_knowledge}

TASK: Create a precise evaluation plan. For each step, specify:
1. What you will calculate or verify
2. What data you will use
3. What criteria you will apply
4. What constraint or investor restriction you need to check

Be explicit about:
- Which income figure you will use and why (gross vs net, trending-down rule)
- Which investor restrictions apply to this loan type
- The full waterfall sequence you will apply for modification
- What options must be evaluated

Output a numbered plan only. Do not evaluate yet."""

    print(f"  Phase 1: Planning...")
    plan_result = _call_gemini(plan_prompt, agent_name="Plan-and-Solve (plan phase)")
    plan_text = plan_result.get("determination", "")
    print(f"  Plan generated: {len(plan_text):,} chars")

    # Phase 2: Execute against plan
    solve_prompt = f"""You are an expert mortgage servicer loss mitigation analyst.
Execute the following evaluation plan to produce a complete loss mitigation determination.

EVALUATION PLAN:
{plan_text}

APPLICATION:
Case ID: {case_input['case_id']}
Borrower: {case_input['borrower_info']}
Loan: {case_input['loan_info']}
Hardship: {case_input['hardship_info']}
Income documentation: {case_input['income_documentation']}
Property T&I: {case_input['property_ti_monthly']}
Application completeness: {case_input['completeness_status']}
Housing counselor letter: {case_input['housing_counselor_letter']}
Question: {case_input['question']}

DOMAIN KNOWLEDGE AND SOURCE DOCUMENTS:
{domain_knowledge}

INSTRUCTIONS:
Execute the plan step by step. Show your calculations at each step.
For the modification waterfall: calculate and state DTI at each step.
State each investor restriction as you apply it.
Do not skip steps.

Write a complete loss mitigation determination notice with specific reasons
and calculated values for each option approved or denied."""

    print(f"  Phase 2: Executing plan...")
    solve_result = _call_gemini(solve_prompt, agent_name="Plan-and-Solve (solve phase)")
    elapsed = plan_result.get("elapsed", 0) + solve_result.get("elapsed", 0)

    return {
        "determination": solve_result.get("determination", ""),
        "plan": plan_text,
        "elapsed": elapsed,
    }


# ── Shared Gemini caller ───────────────────────────────────────────────────

def _call_gemini(prompt: str, agent_name: str = "agent") -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        t0 = time.time()
        response = model.generate_content(prompt)
        elapsed = time.time() - t0

        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'content'):
            raw = response.content
            if isinstance(raw, list):
                text = "".join(
                    part.text if hasattr(part, 'text') else str(part)
                    for part in raw
                )
            else:
                text = str(raw)
        else:
            text = str(response)

        print(f"  Elapsed: {elapsed:.1f}s  Output: {len(text):,} chars")
        return {"determination": text, "elapsed": elapsed}

    except Exception as e:
        print(f"  {agent_name} failed: {e}")
        return {"determination": f"[{agent_name} failed: {e}]", "elapsed": 0}


# ── Comparison ─────────────────────────────────────────────────────────────

def print_comparison(cc: dict, react: dict, ps: dict):
    print(f"\n{'═' * 72}")
    print(f"  COMPARISON")
    print(f"{'═' * 72}")
    print(f"""
  ┌──────────────────────────┬──────────────┬──────────────┬──────────────┐
  │                          │ Cognitive    │ ReAct        │ Plan-and-    │
  │                          │ Core         │ (Yao 2022)   │ Solve        │
  │                          │              │              │ (Wang 2023)  │
  ├──────────────────────────┼──────────────┼──────────────┼──────────────┤
  │ Time                     │ {cc['elapsed']:>6.1f}s      │ {react.get('elapsed',0):>6.1f}s      │ {ps.get('elapsed',0):>6.1f}s      │
  │ Sources read             │ 3 (sep.)     │ 1 (combined) │ 1 (combined) │
  │ Governance tier          │ {str(cc['tier']).upper():<12} │ N/A          │ N/A          │
  │ Audit ledger             │ Yes          │ No           │ No           │
  │ Constraint propagation   │ Governed     │ Single-pass  │ Plan→Execute │
  │ Income calc verified     │ Verify step  │ Prompt only  │ Plan only    │
  │ Investor restriction     │ Domain yaml  │ Prompt only  │ Plan only    │
  │ Output size              │ {len(str(cc['determination'])):>5,} chars  │ {len(str(react.get('determination',''))):>5,} chars  │ {len(str(ps.get('determination',''))):>5,} chars  │
  └──────────────────────────┴──────────────┴──────────────┴──────────────┘

  The question: do they all reach the same determination?
  And if not, which failure modes does each system exhibit?
  (See case ground_truth_complexity.scoring for evaluation criteria)
""")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Loan Modification Loss Mitigation Demo")
    parser.add_argument("--compare",   action="store_true", help="Also run ReAct and Plan-and-Solve baselines")
    parser.add_argument("--react-only", action="store_true", help="Run ReAct only")
    parser.add_argument("--ps-only",   action="store_true", help="Run Plan-and-Solve only")
    parser.add_argument("--save",      action="store_true", help="Save determinations to output/")
    parser.add_argument("--verbose",   action="store_true", help="Show LLM call details")
    parser.add_argument("--case",      default="lm_2024_a001.json",
                        help="Case file to use (default: lm_2024_a001.json)")
    args = parser.parse_args()

    print(f"\n{'═' * 72}")
    print(f"  LOAN MODIFICATION — LOSS MITIGATION DETERMINATION DEMO")
    print(f"  Cognitive Core vs. ReAct vs. Plan-and-Solve")
    print(f"{'═' * 72}\n")

    case = load_case(args.case)
    docs = load_documents()
    case_input = build_case_input(case, docs)

    print(f"  Case: {case['case_id']}")
    print(f"  Borrower: {case['borrower']['name']}, {case['borrower']['age']}yo")
    print(f"  Loan type: {case['loan']['loan_type']} / Investor: {case['loan']['investor']}")
    print(f"  Hardship: {case['hardship']['type']}")
    print(f"  Sources: servicer policy + CFPB Reg X + investor/GSE guidelines")
    print(f"  Question: {case['question'][:80]}...")

    output_dir = DEMO_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cc_result = {}
    react_result = {}
    ps_result = {}

    if not args.react_only and not args.ps_only:
        cc_result = run_cognitive_core(case_input, verbose=args.verbose)
        if args.save:
            _save(output_dir / "cognitive_core_determination.txt",
                  f"COGNITIVE CORE — LOAN MODIFICATION DETERMINATION\n"
                  f"Trajectory: {' → '.join(cc_result.get('trajectory', []))}\n"
                  f"Governance: {cc_result['tier'].upper()}\n"
                  f"Elapsed: {cc_result['elapsed']:.1f}s\n"
                  + "=" * 72 + "\n\n"
                  + str(cc_result.get("determination", "")))

    if args.compare or args.react_only:
        react_result = run_react_agent(case_input, docs)
        if args.save:
            _save(output_dir / "react_determination.txt",
                  f"REACT AGENT — LOAN MODIFICATION DETERMINATION\n"
                  f"Elapsed: {react_result.get('elapsed',0):.1f}s\n"
                  + "=" * 72 + "\n\n"
                  + str(react_result.get("determination", "")))

    if args.compare or args.ps_only:
        ps_result = run_plan_and_solve_agent(case_input, docs)
        if args.save:
            _save(output_dir / "plan_and_solve_determination.txt",
                  f"PLAN-AND-SOLVE AGENT — LOAN MODIFICATION DETERMINATION\n"
                  f"Elapsed: {ps_result.get('elapsed',0):.1f}s\n"
                  + "=" * 72 + "\n\n"
                  + f"PLAN:\n{ps_result.get('plan','')}\n\n"
                  + "=" * 72 + "\n\n"
                  + f"DETERMINATION:\n{str(ps_result.get('determination',''))}")

    if args.compare and cc_result and react_result and ps_result:
        print_comparison(cc_result, react_result, ps_result)
    elif not args.react_only and not args.ps_only:
        print(f"\n{'═' * 72}")
        print(f"  LOSS MITIGATION DETERMINATION")
        print(f"  Governance: {cc_result.get('tier','?').upper()}")
        print(f"{'═' * 72}\n")
        print(cc_result.get("determination", "[No determination generated]"))


def _save(path: Path, content: str):
    with open(path, "w") as f:
        f.write(content)
    print(f"\n  Saved: {path}")


if __name__ == "__main__":
    main()
