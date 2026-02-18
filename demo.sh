#!/usr/bin/env bash
# Cognitive Core — Demo Script
# Usage: bash demo.sh | bash demo.sh 3 | bash demo.sh --list
set -euo pipefail

B='\033[1m'; BB='\033[1;97m'; DIM='\033[2m'; UL='\033[4m'
WHT='\033[97m'; CYN='\033[96m'; GRN='\033[92m'; YLW='\033[93m'
MAG='\033[95m'; RED='\033[91m'
BG_BLU='\033[44m'; BG_CYN='\033[46m'; BG_MAG='\033[45m'; BG_GRN='\033[42m'
BG_YLW='\033[43m'; BG_RED='\033[41m'; BG_DK='\033[48;5;236m'; R='\033[0m'

banner() {
    local num="$1" title="$2"
    local w=62 content="   DEMO ${num}  ▸  ${title}"
    local pad=$(( w - ${#content} ))
    echo ""; echo ""
    echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' $w '')${R}"
    echo -e "${B}${BG_BLU}${WHT}${content}$(printf '%*s' $pad '')${R}"
    echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' $w '')${R}"
    echo ""
}
section()  { echo ""; echo -e "  ${B}${BG_DK}${WHT} ▎ ${1} ${R}"; echo ""; }
line()     { echo -e "     ${BB}${1}${R}"; }
dim()      { echo -e "     ${DIM}${1}${R}"; }
pipeline() { echo -e "     ${B}${CYN}$1${R}"; }
bullet()   { echo -e "     ${GRN}•${R}  ${WHT}${1}${R}"; }
sep()      { echo -e "  ${DIM}────────────────────────────────────────────────────────────${R}"; }
tag() { echo -e "     ${B}${2}${1}${R}  ${WHT}${3}${R}"; }
pause_run() { echo ""; echo -e "     ${B}${YLW}▶  Press Enter to run${R}${DIM}  (Ctrl+C to skip)${R}"; read -r; }

run_case() {
    echo ""; echo -e "  ${B}${BG_DK}${WHT} ▎ EXECUTING ${R}"; echo ""
    echo -e "     ${DIM}LLM_PROVIDER=${LLM_PROVIDER:-google} python -m engine.runner -w workflows/${1}.yaml -d domains/${2}.yaml -c cases/${3}.json${R}"
    echo ""; sep; echo ""
    python -m engine.runner -w "workflows/${1}.yaml" -d "domains/${2}.yaml" -c "cases/${3}.json" 2>&1 || true
    echo ""; sep
}

list_demos() {
    echo ""; echo -e "${B}${BG_BLU}${WHT}  Cognitive Core — Demo Script                                 ${R}"; echo ""
    echo -e "  ${B}${WHT} #   Use Case                    Pipeline${R}"
    echo -e "  ${DIM}───  ──────────────────────────  ────────────────────────────────────────${R}"
    echo -e "  ${CYN} 1${R}   Spending Advisor             ${DIM}Retrieve → Classify → Investigate → Generate → Challenge${R}"
    echo -e "  ${CYN} 2${R}   Nurse Triage                 ${DIM}Classify² → Investigate → Verify → Generate → Challenge${R}"
    echo -e "  ${CYN} 3${R}   Regulatory Impact             ${DIM}Classify → Investigate² → Classify → Generate → Challenge${R}"
    echo -e "  ${CYN} 4${R}   Card Fraud Dispute            ${DIM}Retrieve → Classify → Verify → fast-path → Generate → Challenge${R}"
    echo -e "  ${CYN} 5${R}   ACH Dispute                   ${DIM}Same workflow as #4, different domain${R}"
    echo -e "  ${CYN} 6${R}   Loan Hardship                 ${DIM}Retrieve → Classify → Branch → Investigate → Agent → Generate${R}"
    echo -e "  ${CYN} 7${R}   SAR Investigation             ${DIM}Retrieve → Classify → Investigate loop → Filing → Verify${R}"
    echo -e "  ${CYN} 8${R}   Complaint + Act               ${DIM}Full pipeline + write-side Act (email delivery)${R}"
    echo -e "  ${CYN} 9${R}   Spending Advisor ${MAG}(agentic)${R}    ${DIM}Orchestrator chooses primitives dynamically${R}"
    echo -e "  ${CYN}10${R}   Loan Hardship ${MAG}(agentic)${R}       ${DIM}Orchestrator + military-specific reasoning${R}"
    echo ""
}

DEMO_NUM="${1:-all}"
[[ "$DEMO_NUM" == "--list" || "$DEMO_NUM" == "-l" ]] && { list_demos; exit 0; }
should_run() { [[ "$DEMO_NUM" == "all" || "$DEMO_NUM" == "$1" ]]; }

echo ""
echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' 62 '')${R}"
echo -e "${B}${BG_BLU}${WHT}          COGNITIVE CORE  —  Demo Walkthrough                  ${R}"
echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' 62 '')${R}"
echo ""
echo -e "     ${BB}LLM_PROVIDER${R}${WHT}=${R}${CYN}${LLM_PROVIDER:-not set (defaults to azure)}${R}"
echo -e "     ${BB}10 use cases${R}${WHT}, increasing complexity${R}"
echo -e "     ${BB}8 primitives${R}${WHT}: Retrieve  Classify  Investigate  Think  Verify  Generate  Challenge  Act${R}"
echo -e "     ${BB}3 data modes${R}${WHT}: MCP → Fixture DB → Case passthrough${R}"
echo ""
[ ! -f "fixtures/cognitive_core.db" ] && { echo -e "     ${YLW}Building fixture database...${R}"; python -m fixtures.db build; echo ""; }

# ══════════════════════════════════════════════════════════════════════
#  1 — SPENDING ADVISOR
# ══════════════════════════════════════════════════════════════════════
if should_run 1; then
banner "1/10" "SPENDING ADVISOR"
section "THE PROBLEM"
line "Marcus Williams asks via mobile app:"
echo -e "     ${B}${YLW}\"I feel like I spend way too much on takeout and coffee.${R}"
echo -e "     ${B}${YLW} Can you show me how bad it actually is?\"${R}"
echo ""
dim "Simplest retrieval workflow — the 'hello world' of Cognitive Core."
section "PIPELINE"
pipeline "Retrieve → Classify → Investigate → Generate → Challenge"
dim "5 steps. No branching. No agent decisions."
section "PRIMITIVES"
bullet "Retrieve — member profile, accounts, 762 transactions, goals, spending summaries"
bullet "Classify — question type (spending analysis, goal tracking, comparison)"
bullet "Investigate — crunch the numbers: \$/mo on coffee, dining, trends"
bullet "Generate — conversational, personalized response"
bullet "Challenge — are the numbers actually right?"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member, get_accounts, get_transactions, get_financial_goals, get_spending_summary"
tag "👁 WATCH" "$GRN" "Retrieve selecting 5 of 16 available tools"
tag "👁 WATCH" "$GRN" "Investigate extracting concrete dollar amounts from raw transactions"
tag "✓ EXPECT" "$CYN" "Personalized spending breakdown with numbers tied to goals"
pause_run; run_case spending_advisor debit_spending spending_advisor_williams
fi

# ══════════════════════════════════════════════════════════════════════
#  2 — NURSE TRIAGE
# ══════════════════════════════════════════════════════════════════════
if should_run 2; then
banner "2/10" "NURSE TRIAGE"
section "THE PROBLEM"
line "2:15 AM call. David Park, 42yo male:"
echo -e "     ${B}${RED}\"Woke up with bad chest tightness... pressure in the center...${R}"
echo -e "     ${B}${RED} left arm tingly... My wife told me to call.\"${R}"
echo ""
line "History: hypertension, high cholesterol, father had MI at 58"
line "Vitals: BP 158/95, HR 92"
echo ""
echo -e "     ${B}${BG_RED}${WHT} ⚠  LIFE-OR-DEATH. Must not under-triage. ${R}"
section "PIPELINE"
pipeline "Classify → Classify → Investigate → Verify → Generate → Challenge"
dim "No Retrieve — all data is caller-provided intake."
dim "Verify is the SAFETY GATE: checks triage against clinical rules BEFORE generating."
section "PRIMITIVES"
bullet "Classify (×2) — symptom classification + urgency level"
bullet "Investigate — clinical differential: ACS vs GERD vs anxiety"
bullet "Verify — safety gate: catches red flags requiring 911"
bullet "Generate — nurse telephone script"
bullet "Challenge — could this script harm the patient?"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "Case JSON only (no service lookups — caller-provided)"
tag "👁 WATCH" "$GRN" "Verify catching cardiac red flags (pressure + arm tingling + family hx)"
tag "👁 WATCH" "$GRN" "Urgency classification: should be EMERGENT or IMMEDIATE"
tag "✓ EXPECT" "$CYN" "Script directing patient to call 911 or ER immediately"
pause_run; run_case nurse_triage cardiac_triage cardiac_chest_pain
fi

# ══════════════════════════════════════════════════════════════════════
#  3 — REGULATORY IMPACT
# ══════════════════════════════════════════════════════════════════════
if should_run 3; then
banner "3/10" "REGULATORY IMPACT ANALYSIS"
section "THE PROBLEM"
line "New interagency rule on Automated Valuation Models (AVMs)."
line "Six federal agencies. Effective September 2026."
echo -e "     ${B}${YLW}What changes? What's the cost? What's the timeline?${R}"
echo ""
dim "Introduces AGENT ROUTING — the LLM decides what to do next."
section "PIPELINE"
pipeline "Classify → Investigate → Investigate → Classify → Generate → Challenge"
dim "Two investigation passes: regulatory requirements + institutional impact."
dim "Agent decides if first investigation is sufficient. Generate→Challenge loops up to 3×."
section "PRIMITIVES"
bullet "Classify — regulation type (new rule, amendment, guidance)"
bullet "Investigate (×2) — regulatory requirements + institutional impact"
bullet "Generate — executive impact report"
bullet "Challenge — is the report defensible to examiners?"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "Case JSON only (regulation text + institution context)"
tag "👁 WATCH" "$GRN" "Agent routing decision between investigation steps"
tag "👁 WATCH" "$GRN" "Generate→Challenge loop: survives on first pass?"
tag "✓ EXPECT" "$CYN" "Executive-ready regulatory impact report with timeline"
pause_run; run_case regulatory_impact avm_regulation avm_regulation
fi

# ══════════════════════════════════════════════════════════════════════
#  4 — CARD FRAUD DISPUTE
# ══════════════════════════════════════════════════════════════════════
if should_run 4; then
banner "4/10" "CARD FRAUD DISPUTE"
section "THE PROBLEM"
line "Sarah Chen, 8-year member:"
echo -e "     ${B}${YLW}\"\$487.32 from ELECTROMART ONLINE on Jan 28th — I did not${R}"
echo -e "     ${B}${YLW} make this charge. Never shopped there. Had my card.\"${R}"
echo ""
line "Fraud score: 892/1000 (high). IP: Lagos, Nigeria. Unknown device."
dim "Clear-cut fraud. The system should fast-path it."
section "PIPELINE"
pipeline "Retrieve → Classify → Verify → Investigate → Fast-path → ... → Generate → Challenge"
dim "9 steps defined, but FAST-PATH ROUTING skips the full"
dim "investigation when classification confidence ≥ 0.95."
section "PRIMITIVES"
bullet "Retrieve — member, fraud score, device fingerprints, transaction history"
bullet "Classify — dispute type (unauthorized, billing error, etc.)"
bullet "Verify — do records match the member's claim?"
bullet "Classify (fast-path) — high confidence → skip investigation"
bullet "Generate — member notification letter"
bullet "Challenge — Reg E compliance, UDAAP, no internal details leaked"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member, get_transactions, get_fraud_score, get_devices"
tag "👁 WATCH" "$GRN" "Fast-path: classify at ≥0.95 confidence skips investigation"
tag "👁 WATCH" "$GRN" "Challenge checking Reg E provisional credit requirements"
tag "✓ EXPECT" "$CYN" "Compliant notification with provisional credit, no fraud score leaked"
pause_run; run_case dispute_resolution card_dispute card_clear_fraud
fi

# ══════════════════════════════════════════════════════════════════════
#  5 — ACH DISPUTE (DOMAIN SWAP)
# ══════════════════════════════════════════════════════════════════════
if should_run 5; then
banner "5/10" "ACH DISPUTE — DOMAIN SWAPPABILITY"
section "THE PROBLEM"
line "Thomas Wright cancelled his gym membership in December."
line "Premier Fitness LLC still debiting \$189.99/month via ACH."
echo -e "     ${B}${YLW}Second unauthorized charge since cancellation.${R}"
section "THREE-LAYER ARCHITECTURE"
echo ""
echo -e "     ${B}${BG_CYN}${WHT} SAME workflow ${R}  dispute_resolution  (9 steps)"
echo -e "     ${B}${BG_MAG}${WHT} DIFFERENT domain ${R}  ach_dispute  (ACH-specific knowledge)"
echo -e "     ${B}${BG_GRN}${WHT} DIFFERENT case ${R}  ach_revoked_authorization"
echo ""
line "Workflow (HOW) × Domain (WHAT) × Case (WHO) = unique execution"
section "CARD vs ACH"
echo -e "     ${DIM}Card dispute:${R}  Reg Z, card networks, device fingerprints"
echo -e "     ${BB}ACH dispute:${R}   Reg E, originator/ODFI, SEC codes, authorization chain"
dim "Same primitives, same workflow, completely different domain knowledge."
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member, get_transactions (ACH history), get_dispute"
tag "👁 WATCH" "$GRN" "Domain-specific classification: 'revoked_authorization'"
tag "👁 WATCH" "$GRN" "Reg E timelines (different from Reg Z)"
tag "✓ EXPECT" "$CYN" "ACH-specific resolution with correct regulatory references"
pause_run; run_case dispute_resolution ach_dispute ach_revoked_authorization
fi

# ══════════════════════════════════════════════════════════════════════
#  6 — LOAN HARDSHIP
# ══════════════════════════════════════════════════════════════════════
if should_run 6; then
banner "6/10" "MILITARY LOAN HARDSHIP"
section "THE PROBLEM"
line "Angela Reeves:"
echo -e "     ${B}${YLW}\"My husband was medically retired from the Navy last month.${R}"
echo -e "     ${B}${YLW} VA disability is less than active duty pay. We have a${R}"
echo -e "     ${B}${YLW} mortgage (\$1,842/mo) and auto loan (\$548/mo).${R}"
echo -e "     ${B}${YLW} Never missed a payment. Is there anything you can do?\"${R}"
echo ""
dim "Introduces CONDITIONAL BRANCHING + AGENT-DECIDED ESCALATION."
section "PIPELINE"
pipeline "Retrieve → Classify → [Branch] → Investigate → Classify → [Agent Decide] → Generate → Challenge"
echo ""
echo -e "     ${B}Classify routes to different paths:${R}"
echo -e "       ${CYN}military_transition${R}  →  investigate_military (SCRA/VA benefits)"
echo -e "       ${RED}disaster_hardship${R}    →  escalate (human specialist)"
echo -e "       ${DIM}default${R}              →  investigate_financial (general hardship)"
dim "After investigation, agent decides: generate guidance OR escalate."
section "PRIMITIVES"
bullet "Retrieve — member profile (military_status), loans, accounts"
bullet "Classify — hardship type with military-specific categories"
bullet "Investigate — SCRA protections, VA benefits, DTI analysis"
bullet "Classify — assistance type (forbearance, modification, etc.)"
bullet "Agent Decide — clear enough to generate, or needs a human?"
bullet "Generate — member-facing hardship guidance letter"
bullet "Challenge — regulatory review (SCRA compliance)"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member (military_status=spouse_of_retired), get_loans, get_accounts"
tag "👁 WATCH" "$GRN" "Classification routing to military-specific investigation"
tag "👁 WATCH" "$GRN" "Agent deciding 'generate guidance' vs 'escalate to specialist'"
tag "✓ EXPECT" "$CYN" "Hardship guidance mentioning SCRA protections and VA benefits"
pause_run; run_case loan_hardship military_hardship military_hardship_reeves
fi

# ══════════════════════════════════════════════════════════════════════
#  7 — SAR INVESTIGATION
# ══════════════════════════════════════════════════════════════════════
if should_run 7; then
banner "7/10" "SAR INVESTIGATION"
section "THE PROBLEM"
line "AML alert score 87. Robert Mancini, 'Mancini Consulting LLC.'"
echo ""
echo -e "     ${B}${RED}January 2026: seven cash deposits totaling \$64,900${R}"
echo -e "     ${B}${RED}ALL under \$10,000. Alternating between two branches.${R}"
echo -e "     ${B}${RED}Then a \$45,000 wire to Cayman National Bank.${R}"
echo -e "     ${B}${RED}Prior monthly average: \$12,000.${R}"
echo ""
dim "Classic structuring pattern. Most complex sequential workflow."
section "PIPELINE"
pipeline "Retrieve → Classify → Investigate (loop ×2) → Filing Decision → Narrative → Challenge → Verify"
dim "Investigation LOOPS for more evidence. loop_fallback → filing decision."
dim "SAR narrative → adversarial challenge → FinCEN completeness check."
section "PRIMITIVES"
bullet "Retrieve — AML alert, member profile, transaction history"
bullet "Classify — alert typology (structuring, wire, rapid movement)"
bullet "Investigate (×1-2) — hypothesis testing against extracted data"
echo -e "       ${DIM}Tests: sub-\$10K pattern, branch alternation (smurfing),${R}"
echo -e "       ${DIM}Cayman wire (high-risk jurisdiction), 5× volume spike${R}"
bullet "Classify — filing decision (file SAR, close, request more info)"
bullet "Generate — FinCEN-compliant SAR narrative"
bullet "Challenge — fact-check every dollar amount and date"
bullet "Verify — FinCEN completeness checklist"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_aml_alert, get_member, get_transactions"
tag "👁 WATCH" "$GRN" "Investigation loop: confidence rising ~0.75 → ~0.85"
tag "👁 WATCH" "$GRN" "Hypothesis testing: structuring, smurfing, high-risk jurisdiction"
tag "👁 WATCH" "$GRN" "SAR narrative: factual, objective, no tipping-off language"
tag "✓ EXPECT" "$CYN" "FinCEN-compliant SAR surviving challenge + verification"
pause_run; run_case sar_investigation structuring_sar sar_structuring
fi

# ══════════════════════════════════════════════════════════════════════
#  8 — COMPLAINT + ACT
# ══════════════════════════════════════════════════════════════════════
if should_run 8; then
banner "8/10" "COMPLAINT RESOLUTION + ACT"
section "THE PROBLEM"
line "Mamadou Diouf-Seck:"
echo -e "     ${B}${YLW}\"I deposited a \$4,200 check from my employer on Feb 10th${R}"
echo -e "     ${B}${YLW} via mobile deposit. It's Friday and funds STILL not${R}"
echo -e "     ${B}${YLW} available. A payment bounced. I got charged an NSF fee.${R}"
echo -e "     ${B}${YLW} This is costing me money.\"${R}"
echo ""
echo -e "     ${B}${BG_MAG}${WHT} ✦  FIRST WORKFLOW WITH THE ACT PRIMITIVE  ✦ ${R}"
dim "Doesn't just analyze — it DOES something. Sends an actual email."
section "PIPELINE"
pipeline "Retrieve → Classify → Classify → Investigate → Generate → Challenge → ACT → Generate"
echo ""
echo -e "     ${DIM}Steps 1-6: read-side cognitive processing${R}"
echo -e "     ${B}${MAG}Step 7: ACT — write-side boundary (sends email)${R}"
echo -e "     ${DIM}Step 8: internal resolution summary${R}"
section "PRIMITIVES"
bullet "Retrieve — member, account balances, check deposit (Reg CC hold), NSF events"
bullet "Classify (×2) — complaint type + severity/urgency"
bullet "Investigate — is the hold justified? Reg CC §229.13(b) analysis"
bullet "Generate — empathetic email response"
bullet "Challenge — compliance + tone review"
bullet "Act — send_email with the approved response"
bullet "Generate — internal case summary"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member, get_accounts, get_check_deposit, get_nsf_events, get_complaint"
tag "👁 WATCH" "$GRN" "Act primitive: executes send_email (or simulates if no SMTP)"
tag "👁 WATCH" "$GRN" "Reg CC analysis: hold justified under §229.13(b)?"
tag "👁 WATCH" "$GRN" "Does response address the \$29 NSF fee and offer waiver?"
tag "✓ EXPECT" "$CYN" "Email sent (or simulated) + internal case summary"
echo ""
if [ -n "${SMTP_SENDER:-}" ]; then
    echo -e "     ${B}${BG_GRN}${WHT} ✉  SMTP configured — real email will be sent ${R}"
else
    echo -e "     ${B}${BG_YLW}${WHT} ✉  SMTP not set — email will be simulated ${R}"
    dim "Set SMTP_SENDER + SMTP_APP_PASSWORD for live delivery"
fi
pause_run; run_case complaint_resolution_act check_clearing_complaint check_clearing_complaint_diouf
fi

# ══════════════════════════════════════════════════════════════════════
#  9 — SPENDING ADVISOR AGENTIC
# ══════════════════════════════════════════════════════════════════════
if should_run 9; then
banner "9/10" "SPENDING ADVISOR (AGENTIC)"
section "THE PROBLEM"
line "Same question as Demo 1: Marcus Williams asks about spending."
echo ""
echo -e "     ${B}${BG_MAG}${WHT} ✦  PARADIGM SHIFT: choreography → orchestration  ✦ ${R}"
dim "The ORCHESTRATOR decides which primitives to use and in what order."
section "ARCHITECTURE"
echo -e "     ${B}Mode:${R}       ${MAG}AGENTIC${R}"
echo -e "     ${B}Available:${R}  retrieve  classify  investigate  ${CYN}think${R}  generate  challenge"
echo ""
dim "The orchestrator plans and executes dynamically:"
echo -e "     ${BB}1.${R} Retrieve data          ${BB}4.${R} ${CYN}Think${R} — connect findings to goals"
echo -e "     ${BB}2.${R} Classify the question   ${BB}5.${R} Generate advice"
echo -e "     ${BB}3.${R} Investigate patterns    ${BB}6.${R} Challenge its own output"
section "SEQUENTIAL vs AGENTIC"
echo -e "     ${DIM}Demo 1:${R}  Fixed pipeline, always 5 steps in order"
echo -e "     ${BB}Demo 9:${R}  Orchestrator adapts. Simple → skip Think. Complex → add steps."
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "Same service registry — orchestrator discovers and calls tools"
tag "👁 WATCH" "$GRN" "Orchestrator's step selection — does it use Think?"
tag "👁 WATCH" "$GRN" "Fewer or more steps than the sequential version?"
tag "✓ EXPECT" "$CYN" "Same quality advice, adaptively planned"
pause_run; run_case spending_advisor_agentic debit_spending_agentic spending_advisor_williams
fi

# ══════════════════════════════════════════════════════════════════════
#  10 — LOAN HARDSHIP AGENTIC
# ══════════════════════════════════════════════════════════════════════
if should_run 10; then
banner "10/10" "MILITARY HARDSHIP (AGENTIC)"
section "THE PROBLEM"
line "Same case as Demo 6: Angela Reeves, military hardship."
echo ""
echo -e "     ${B}${BG_MAG}${WHT} ✦  Most complex demo — full autonomous orchestration  ✦ ${R}"
dim "Agentic mode with military-specific domain knowledge,"
dim "multiple investigation paths, high-stakes compliance."
section "ARCHITECTURE"
echo -e "     ${B}Mode:${R}       ${MAG}AGENTIC${R}"
echo -e "     ${B}Available:${R}  retrieve  classify  investigate  ${CYN}think${R}  verify  generate  challenge"
echo ""
dim "Orchestrator should recognize the need to:"
echo -e "     ${BB}1.${R} Retrieve member + loan data    ${BB}5.${R} Verify SCRA compliance"
echo -e "     ${BB}2.${R} Classify hardship type          ${BB}6.${R} Generate member guidance"
echo -e "     ${BB}3.${R} Investigate SCRA/VA protections  ${BB}7.${R} Challenge for compliance"
echo -e "     ${BB}4.${R} ${CYN}Think${R} — weigh DTI, equity, capacity"
section "SEQUENTIAL vs AGENTIC"
echo -e "     ${DIM}Demo 6:${R}   Fixed routing with branch paths"
echo -e "     ${BB}Demo 10:${R}  Orchestrator recognizes military status, adapts strategy"
section "WHAT TO WATCH"
tag "📡 DATA" "$MAG" "get_member (military_status=spouse_of_retired), get_loans, get_accounts"
tag "👁 WATCH" "$GRN" "Does the orchestrator detect military status early?"
tag "👁 WATCH" "$GRN" "Does it invoke SCRA/VA-specific investigation?"
tag "👁 WATCH" "$GRN" "Does it use Think to synthesize before generating?"
tag "✓ EXPECT" "$CYN" "Adaptive military hardship assessment with SCRA analysis"
pause_run; run_case loan_hardship_agentic military_hardship_agentic military_hardship_reeves
fi

# ══════════════════════════════════════════════════════════════════════
#  WRAP-UP
# ══════════════════════════════════════════════════════════════════════
echo ""; echo ""
echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' 62 '')${R}"
echo -e "${B}${BG_BLU}${WHT}                     Demo Complete                             ${R}"
echo -e "${B}${BG_BLU}${WHT}$(printf '%*s' 62 '')${R}"
echo ""
echo -e "     ${B}${UL}What you saw:${R}"
echo ""
echo -e "     ${B}${GRN}Primitives${R}     Retrieve  Classify  Investigate  Think  Verify  Generate  Challenge  Act"
echo ""
echo -e "     ${B}${GRN}Data modes${R}     Fixture DB → MCP server → Production APIs"
echo -e "                    ${DIM}Same tool names, same signatures, swap the backing${R}"
echo ""
echo -e "     ${B}${GRN}Three-layer${R}    Workflow ${DIM}(how)${R} × Domain ${DIM}(what)${R} × Case ${DIM}(who)${R}"
echo -e "                    ${DIM}Card vs ACH dispute: same workflow, different domain${R}"
echo ""
echo -e "     ${B}${GRN}Orchestration${R}  Sequential ${DIM}(fixed steps)${R} → Agentic ${DIM}(LLM plans)${R}"
echo -e "                    ${DIM}Same use case, two execution modes${R}"
echo ""
echo -e "     ${B}${GRN}Features${R}       Fast-path routing, investigation loops, agent decisions,"
echo -e "                    conditional branching, loop fallback, generate→challenge"
echo -e "                    refinement, write-side Act"
echo ""
