"""
Cognitive Core — Fixture Database Builder

Normalizes case JSON files into a SQLite database whose tables model
production data services. Each table corresponds to an API endpoint:

    members              → GET /v1/members/{member_id}
    accounts             → GET /v1/accounts/{account_id}
                           GET /v1/accounts?member_id=...
    transactions         → GET /v1/transactions?member_id=...&from=...&to=...
    loans                → GET /v1/loans?member_id=...
    disputes             → GET /v1/disputes/{dispute_id}
    complaints           → GET /v1/complaints/{complaint_id}
    fraud_scores         → GET /v1/fraud/scores?transaction_id=...
    device_fingerprints  → GET /v1/devices?member_id=...
    aml_alerts           → GET /v1/aml/alerts/{alert_id}
    check_deposits       → GET /v1/deposits/{deposit_id}
    financial_goals      → GET /v1/goals?member_id=...
    monthly_summaries    → GET /v1/spending/summary?member_id=...&month=...
    nsf_events           → GET /v1/accounts/{account_id}/nsf
    patients             → GET /v1/triage/patients/{patient_id}
    regulations          → GET /v1/regulations/{regulation_id}

Usage:
    from fixtures.db import build_fixture_db
    build_fixture_db()   # reads cases/*.json, writes fixtures/cognitive_core.db
"""

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "cognitive_core.db"
CASES_DIR = Path(__file__).parent.parent / "cases"
FULL_CASES_DIR = CASES_DIR / "full"  # Rich source data for DB population
FIXTURES_DIR = CASES_DIR / "fixtures"

SCHEMA = """
CREATE TABLE IF NOT EXISTS members (
    member_id          TEXT PRIMARY KEY,
    name               TEXT,
    email              TEXT,
    phone              TEXT,
    preferred_contact  TEXT,
    member_since       TEXT,
    tenure_years       REAL,
    branch             TEXT,
    account_type       TEXT,
    credit_score       INTEGER,
    income_band        TEXT,
    age                INTEGER,
    relationship_value REAL,
    segment            TEXT,
    military_status    TEXT,
    products           TEXT,
    prior_disputes     INTEGER DEFAULT 0,
    prior_fraud_alerts INTEGER DEFAULT 0,
    prior_hardship_requests INTEGER DEFAULT 0,
    payment_history    TEXT,
    avg_monthly_spend  REAL,
    source_case        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id          TEXT PRIMARY KEY,
    member_id           TEXT NOT NULL,
    type                TEXT NOT NULL,
    balance_current     REAL,
    balance_available   REAL,
    monthly_avg_balance REAL,
    overdraft_protection INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'active',
    source_case         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id     TEXT PRIMARY KEY,
    member_id          TEXT NOT NULL,
    account_id         TEXT,
    date               TEXT NOT NULL,
    merchant           TEXT,
    category           TEXT,
    amount             REAL NOT NULL,
    type               TEXT,
    auth_method        TEXT,
    location           TEXT,
    ip_geolocation     TEXT,
    device_fingerprint TEXT,
    avs_result         TEXT,
    cvv_result         TEXT,
    destination        TEXT,
    trace_number       TEXT,
    sec_code           TEXT,
    source_case        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS loans (
    loan_number      TEXT PRIMARY KEY,
    member_id        TEXT NOT NULL,
    type             TEXT NOT NULL,
    original_amount  REAL,
    current_balance  REAL,
    monthly_payment  REAL,
    rate             REAL,
    term             TEXT,
    ltv              REAL,
    months_remaining INTEGER,
    payment_status   TEXT,
    days_past_due    INTEGER DEFAULT 0,
    source_case      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disputes (
    dispute_id     TEXT PRIMARY KEY,
    member_id      TEXT NOT NULL,
    transaction_id TEXT,
    type           TEXT,
    description    TEXT,
    filed_date     TEXT,
    status         TEXT DEFAULT 'open',
    channel        TEXT,
    originator     TEXT,
    odfi           TEXT,
    merchant       TEXT,
    amount         REAL,
    source_case    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    member_id    TEXT NOT NULL,
    filed_date   TEXT,
    channel      TEXT,
    type         TEXT,
    subject      TEXT,
    description  TEXT,
    status       TEXT DEFAULT 'open',
    source_case  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fraud_scores (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL,
    member_id      TEXT NOT NULL,
    score          INTEGER,
    risk_level     TEXT,
    factors        TEXT,
    model_version  TEXT,
    scored_at      TEXT,
    source_case    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS device_fingerprints (
    device_id   TEXT PRIMARY KEY,
    member_id   TEXT NOT NULL,
    type        TEXT,
    first_seen  TEXT,
    trusted     INTEGER DEFAULT 1,
    source_case TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS aml_alerts (
    alert_id                   TEXT PRIMARY KEY,
    member_id                  TEXT NOT NULL,
    alert_type                 TEXT,
    alert_score                INTEGER,
    subject_name               TEXT,
    business_name              TEXT,
    business_type              TEXT,
    period_start               TEXT,
    period_end                 TEXT,
    total_cash_deposits        REAL,
    ctr_filed                  INTEGER,
    prior_sars                 INTEGER DEFAULT 0,
    prior_alerts               INTEGER DEFAULT 0,
    prior_alert_disposition    TEXT,
    account_balance            REAL,
    avg_monthly_deposits_prior REAL,
    triggering_transactions    TEXT,
    source_case                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS check_deposits (
    deposit_id              TEXT PRIMARY KEY,
    member_id               TEXT NOT NULL,
    account_id              TEXT,
    check_amount            REAL,
    check_issuer            TEXT,
    check_issuer_bank       TEXT,
    check_number            TEXT,
    deposited_via           TEXT,
    deposited_date          TEXT,
    image_quality           TEXT,
    endorsement_valid       INTEGER,
    duplicate_check         INTEGER,
    issuer_account_verified INTEGER,
    hold_applied            INTEGER,
    hold_reason             TEXT,
    hold_amount             REAL,
    first_200_available     TEXT,
    remaining_available     TEXT,
    reg_cc_max_hold_date    TEXT,
    hold_notice_sent        INTEGER,
    hold_notice_date        TEXT,
    hold_notice_method      TEXT,
    source_case             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nsf_events (
    fee_id      TEXT PRIMARY KEY,
    member_id   TEXT NOT NULL,
    account_id  TEXT,
    date        TEXT,
    payee       TEXT,
    amount      REAL,
    fee_charged REAL,
    source_case TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS financial_goals (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id            TEXT NOT NULL,
    goal_name            TEXT NOT NULL,
    target_amount        REAL,
    current_amount       REAL,
    monthly_contribution REAL,
    target_date          TEXT,
    target_monthly       REAL,
    description          TEXT,
    source_case          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monthly_summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id   TEXT NOT NULL,
    month       TEXT NOT NULL,
    total_spending REAL,
    categories  TEXT,
    source_case TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,
    name            TEXT,
    age             INTEGER,
    sex             TEXT,
    call_time       TEXT,
    chief_complaint TEXT,
    known_conditions TEXT,
    medications      TEXT,
    allergies        TEXT,
    family_history   TEXT,
    smoking          TEXT,
    last_physical    TEXT,
    bmi              REAL,
    has_bp_monitor   INTEGER,
    reported_bp      TEXT,
    reported_hr      TEXT,
    temperature      TEXT,
    source_case      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS regulations (
    regulation_id       TEXT PRIMARY KEY,
    title               TEXT,
    agency              TEXT,
    federal_register    TEXT,
    effective_date      TEXT,
    comment_period_ends TEXT,
    summary             TEXT,
    institution_name    TEXT,
    institution_assets  TEXT,
    affected_lines      TEXT,
    current_usage       TEXT,
    current_vendor      TEXT,
    source_case         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_accounts_member ON accounts(member_id);
CREATE INDEX IF NOT EXISTS idx_transactions_member ON transactions(member_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(member_id, date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(member_id, category);
CREATE INDEX IF NOT EXISTS idx_loans_member ON loans(member_id);
CREATE INDEX IF NOT EXISTS idx_disputes_member ON disputes(member_id);
CREATE INDEX IF NOT EXISTS idx_complaints_member ON complaints(member_id);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_txn ON fraud_scores(transaction_id);
CREATE INDEX IF NOT EXISTS idx_devices_member ON device_fingerprints(member_id);
CREATE INDEX IF NOT EXISTS idx_goals_member ON financial_goals(member_id);
CREATE INDEX IF NOT EXISTS idx_summaries_member ON monthly_summaries(member_id, month);
CREATE INDEX IF NOT EXISTS idx_nsf_member ON nsf_events(member_id);
CREATE INDEX IF NOT EXISTS idx_check_deposits_member ON check_deposits(member_id);
"""


def _json_dumps(obj: Any) -> str:
    """Safe JSON serialize for storing arrays/dicts in TEXT columns."""
    if obj is None:
        return None
    if isinstance(obj, (list, dict)):
        return json.dumps(obj)
    return str(obj)


def _extract_member_id(case: dict, case_name: str) -> str | None:
    """Find the member_id from various case data shapes."""
    if "member_id" in case:
        return case["member_id"]
    for key in ("member_profile", "subject"):
        if isinstance(case.get(key), dict) and "member_id" in case[key]:
            return case[key]["member_id"]
    # Generate one for cases that don't have explicit member_id
    if case_name == "cardiac_chest_pain":
        return "PAT-TRIAGE-001"
    if case_name == "avm_regulation":
        return None  # no member
    return None


def _load_card_fraud(db: sqlite3.Connection, case: dict, case_name: str):
    """card_clear_fraud → members, accounts, transactions, fraud_scores, devices, disputes."""
    mp = case["member_profile"]
    mid = mp["member_id"]

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, mp.get("member_name"), None, None, None,
         mp.get("member_since"), mp.get("tenure_years"), None,
         mp.get("account_type"), mp.get("credit_score"), None, None,
         mp.get("total_relationship_value"), None, None,
         _json_dumps(mp.get("products")),
         mp.get("prior_disputes", 0), mp.get("prior_fraud_alerts", 0), 0, None, None,
         case_name))

    # Disputed transaction
    td = case["transaction_detail"]
    txn_id = td["transaction_id"]
    db.execute("""INSERT OR REPLACE INTO transactions VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (txn_id, mid, None, td["date"], td["merchant"], td.get("category"),
         td["amount"], "purchase", td.get("auth_method"), None,
         td.get("ip_geolocation"), td.get("device_fingerprint"),
         td.get("avs_result"), td.get("cvv_result"), None, None, None,
         case_name))

    # Fraud score
    fs = case["fraud_score"]
    db.execute("""INSERT INTO fraud_scores
        (transaction_id, member_id, score, risk_level, factors, model_version, scored_at, source_case)
        VALUES (?,?,?,?,?,?,?,?)""",
        (txn_id, mid, fs["score"], fs["risk_level"],
         _json_dumps(fs["factors"]), fs.get("model_version"), fs.get("scored_at"),
         case_name))

    # Device fingerprints
    for dev in case.get("device_fingerprints", {}).get("known_devices", []):
        db.execute("""INSERT OR REPLACE INTO device_fingerprints VALUES (?,?,?,?,?,?)""",
            (dev["device_id"], mid, dev.get("type"), dev.get("first_seen"), 1, case_name))

    # Dispute record
    di = case.get("dispute_intake", {})
    dispute_id = f"DSP-{txn_id}"
    db.execute("""INSERT OR REPLACE INTO disputes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (dispute_id, mid, txn_id, "card_fraud", di.get("dispute_description"),
         di.get("dispute_filed_date"), "open", di.get("dispute_channel"),
         None, None, td["merchant"], td["amount"], case_name))


def _load_ach_dispute(db: sqlite3.Connection, case: dict, case_name: str):
    """ach_revoked_authorization → members, transactions, disputes."""
    mid = case["member_id"]

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, case.get("member_name"), None, None, None,
         case.get("member_since"), case.get("member_history", {}).get("tenure_years"),
         None, case.get("account_type"), None, None, None, None, None, None, None,
         case.get("member_history", {}).get("prior_disputes", 0),
         case.get("member_history", {}).get("fraud_alerts_prior", 0), 0, None,
         case.get("member_history", {}).get("avg_monthly_spend"),
         case_name))

    # Current + prior debits as transactions
    td = case["transaction_details"]
    for i, txn in enumerate(td.get("prior_debits_from_originator", [])):
        txn_id = f"ACH-{mid}-{txn['date']}"
        db.execute("""INSERT OR REPLACE INTO transactions VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (txn_id, mid, None, txn["date"], td["originator"], "ach_debit",
             txn["amount"], "recurring_ach_debit", "ach", None, None, None,
             None, None, None, td.get("trace_number"), td.get("sec_code"),
             case_name))

    # Dispute
    dispute_id = f"DSP-ACH-{mid}"
    db.execute("""INSERT OR REPLACE INTO disputes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (dispute_id, mid, None, "ach_unauthorized", case.get("dispute_description"),
         None, "open", None, td.get("originator"), td.get("odfi"),
         None, td.get("amount"), case_name))


def _load_check_clearing(db: sqlite3.Connection, case: dict, case_name: str):
    """check_clearing_complaint_diouf → members, accounts, complaints, check_deposits, nsf."""
    mp = case["member_profile"]
    mid = mp["member_id"]

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, mp.get("name"), mp.get("email"), mp.get("phone"),
         mp.get("preferred_contact"), None, None, None, None, None, None, None,
         None, None, None, None, 0, 0, 0, None, None, case_name))

    # Accounts
    for acct in case.get("accounts", []):
        db.execute("""INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?,?,?,?,?)""",
            (acct["account_id"], mid, acct["type"], None, None, None, 0, "active",
             case_name))

    # Complaint
    comp = case.get("complaint", {})
    db.execute("""INSERT OR REPLACE INTO complaints VALUES (?,?,?,?,?,?,?,?,?)""",
        (comp.get("complaint_id"), mid, comp.get("filed_date"), comp.get("channel"),
         comp.get("type"), comp.get("subject"), comp.get("description"), "open",
         case_name))

    # Load fixtures if available (check_deposit_detail, account_detail, etc.)
    fixtures_path = FIXTURES_DIR / f"{case_name}.json"
    if fixtures_path.exists():
        fixtures = json.load(open(fixtures_path))

        # Account detail enrichment
        ad = fixtures.get("account_detail", {})
        if ad:
            db.execute("""UPDATE accounts SET
                balance_current=?, balance_available=?, monthly_avg_balance=?,
                overdraft_protection=?
                WHERE account_id=?""",
                (ad.get("balance_current"), ad.get("balance_available"),
                 ad.get("monthly_average_balance"),
                 1 if ad.get("overdraft_protection") else 0,
                 ad.get("account_id")))

        # Check deposit
        cd = fixtures.get("check_deposit_detail", {})
        if cd:
            db.execute("""INSERT OR REPLACE INTO check_deposits VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (cd.get("deposit_id"), mid, ad.get("account_id"),
                 cd.get("check_amount"), cd.get("check_issuer"),
                 cd.get("check_issuer_bank"), cd.get("check_number"),
                 cd.get("deposited_via"), cd.get("deposited_date"),
                 cd.get("image_quality"),
                 1 if cd.get("endorsement_valid") else 0,
                 1 if cd.get("duplicate_check") else 0,
                 1 if cd.get("issuer_account_verified") else 0,
                 1 if cd.get("hold_applied") else 0,
                 cd.get("hold_reason"), cd.get("hold_amount"),
                 cd.get("first_200_available_date"),
                 cd.get("remaining_available_date"),
                 cd.get("reg_cc_maximum_hold_date"),
                 1 if cd.get("hold_notice_sent") else 0,
                 cd.get("hold_notice_date"), cd.get("hold_notice_method"),
                 case_name))

        # NSF events
        for nsf in ad.get("recent_nsf", []):
            db.execute("""INSERT OR REPLACE INTO nsf_events VALUES (?,?,?,?,?,?,?,?)""",
                (nsf.get("fee_id"), mid, ad.get("account_id"), nsf.get("date"),
                 nsf.get("payee"), nsf.get("amount"), nsf.get("fee_charged"),
                 case_name))

        # Member enrichment from fixtures
        fmp = fixtures.get("member_profile", {})
        if fmp:
            db.execute("""UPDATE members SET
                member_since=COALESCE(?, member_since),
                tenure_years=COALESCE(?, tenure_years),
                branch=COALESCE(?, branch),
                products=COALESCE(?, products),
                relationship_value=COALESCE(?, relationship_value),
                segment=COALESCE(?, segment)
                WHERE member_id=?""",
                (fmp.get("member_since"), fmp.get("membership_tenure_years"),
                 fmp.get("branch"), _json_dumps(fmp.get("products")),
                 fmp.get("relationship_value"), fmp.get("segment"), mid))


def _load_military_hardship(db: sqlite3.Connection, case: dict, case_name: str):
    """military_hardship_reeves → members, loans."""
    mid = case["member_id"]
    mp = case.get("member_profile", {})

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, case.get("member_name"), None, None, None,
         case.get("member_since"), mp.get("tenure_years"), None, None,
         mp.get("credit_score"), None, None,
         mp.get("total_relationship_value"), None,
         mp.get("military_status"),
         _json_dumps(mp.get("other_products")),
         0, 0, mp.get("prior_hardship_requests", 0),
         mp.get("payment_history"), None, case_name))

    # Loans
    for loan_type, loan in case.get("loan_details", {}).items():
        db.execute("""INSERT OR REPLACE INTO loans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (loan["loan_number"], mid, loan_type,
             loan.get("original_amount"), loan.get("current_balance"),
             loan.get("monthly_payment"), loan.get("rate"), loan.get("type"),
             loan.get("ltv"), loan.get("months_remaining"),
             loan.get("payment_status"), loan.get("days_past_due", 0),
             case_name))


def _load_sar(db: sqlite3.Connection, case: dict, case_name: str):
    """sar_structuring → members, aml_alerts, transactions."""
    subj = case["subject"]
    mid = f"AML-{subj.get('name', 'unknown').replace(' ', '-').upper()}"

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, subj.get("name"), None, None, None,
         subj.get("member_since"), None, None,
         subj.get("account_type"), None, None, None, None, None, None, None,
         0, 0, 0, None, None, case_name))

    # AML alert
    ta = case.get("triggering_activity", {})
    ah = case.get("account_history", {})
    period = ta.get("period", "")
    period_parts = period.split(" to ") if " to " in period else [period, period]

    db.execute("""INSERT OR REPLACE INTO aml_alerts VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (case["alert_id"], mid, case.get("alert_type"), case.get("alert_score"),
         subj.get("name"), subj.get("business_name"), subj.get("business_type"),
         period_parts[0], period_parts[1] if len(period_parts) > 1 else None,
         ta.get("total_cash_deposits"), 1 if ta.get("ctr_filed") else 0,
         ah.get("prior_sars", 0), ah.get("prior_alerts", 0),
         ah.get("prior_alert_disposition"), ah.get("account_balance"),
         ah.get("avg_monthly_deposits_prior_6mo"),
         _json_dumps(ta.get("transactions")),
         case_name))

    # Individual transactions from the alert
    for txn in ta.get("transactions", []):
        txn_id = f"AML-TXN-{mid}-{txn['date']}-{txn['amount']}"
        db.execute("""INSERT OR REPLACE INTO transactions VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (txn_id, mid, None, txn["date"], None, None,
             txn["amount"], txn.get("type"), None, txn.get("location"),
             None, None, None, None, txn.get("destination"), None, None,
             case_name))


def _load_spending_advisor(db: sqlite3.Connection, case: dict, case_name: str):
    """spending_advisor_williams → members, transactions, financial_goals, monthly_summaries."""
    mp = case["member_profile"]
    mid = mp["member_id"]

    db.execute("""INSERT OR REPLACE INTO members VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, mp.get("name"), None, None, None,
         mp.get("member_since"), None, None, None, None,
         mp.get("income_band"), mp.get("age"), None, None, None,
         _json_dumps(mp.get("products")),
         0, 0, 0, None, None, case_name))

    # Accounts (inferred)
    db.execute("""INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?,?,?,?,?)""",
        (f"CHK-{mid}", mid, "checking", mp.get("checking_balance"),
         mp.get("checking_balance"), None, 0, "active", case_name))
    db.execute("""INSERT OR REPLACE INTO accounts VALUES (?,?,?,?,?,?,?,?,?)""",
        (f"SAV-{mid}", mid, "savings", mp.get("savings_balance"),
         mp.get("savings_balance"), None, 0, "active", case_name))

    # Transactions
    for txn in case.get("transactions", []):
        db.execute("""INSERT OR REPLACE INTO transactions VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (txn["id"], mid, f"CHK-{mid}", txn["date"], txn.get("merchant"),
             txn.get("category"), txn["amount"], "debit_purchase", None, None,
             None, None, None, None, None, None, None, case_name))

    # Financial goals
    for goal_name, goal in case.get("financial_goals", {}).items():
        db.execute("""INSERT INTO financial_goals
            (member_id, goal_name, target_amount, current_amount,
             monthly_contribution, target_date, target_monthly, description, source_case)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (mid, goal_name, goal.get("target"), goal.get("current"),
             goal.get("monthly_contribution"), goal.get("target_date"),
             goal.get("target_monthly"), goal.get("description"), case_name))

    # Monthly summaries
    for month, summary in case.get("monthly_summaries", {}).items():
        db.execute("""INSERT INTO monthly_summaries
            (member_id, month, total_spending, categories, source_case)
            VALUES (?,?,?,?,?)""",
            (mid, month, summary.get("total_spending"),
             _json_dumps(summary.get("categories")), case_name))


def _load_cardiac_triage(db: sqlite3.Connection, case: dict, case_name: str):
    """cardiac_chest_pain → patients."""
    ph = case.get("patient_history", {})
    vit = case.get("vitals_if_available", {})
    patient_id = "PAT-TRIAGE-001"

    db.execute("""INSERT OR REPLACE INTO patients VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, case.get("caller_name"), case.get("caller_age"),
         case.get("caller_sex"), case.get("call_time"),
         case.get("chief_complaint"),
         _json_dumps(ph.get("known_conditions")),
         _json_dumps(ph.get("medications")),
         _json_dumps(ph.get("allergies")),
         ph.get("family_history"), ph.get("smoking"),
         ph.get("last_physical"), ph.get("bmi"),
         1 if vit.get("has_home_bp_monitor") else 0,
         vit.get("reported_bp"), vit.get("reported_hr"),
         vit.get("temperature"), case_name))


def _load_avm_regulation(db: sqlite3.Connection, case: dict, case_name: str):
    """avm_regulation → regulations."""
    reg = case.get("regulation", {})
    inst = case.get("institution_context", {})

    db.execute("""INSERT OR REPLACE INTO regulations VALUES (
        ?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (reg.get("federal_register", f"REG-{case_name}"),
         reg.get("title"), reg.get("agency"), reg.get("federal_register"),
         reg.get("effective_date"), reg.get("comment_period_ends"),
         reg.get("summary"), inst.get("name"), inst.get("assets"),
         _json_dumps(inst.get("affected_business_lines")),
         inst.get("current_avm_usage"), inst.get("current_vendor"),
         case_name))


# Dispatch table: case filename stem → loader function
_LOADERS = {
    "card_clear_fraud": _load_card_fraud,
    "ach_revoked_authorization": _load_ach_dispute,
    "check_clearing_complaint_diouf": _load_check_clearing,
    "military_hardship_reeves": _load_military_hardship,
    "sar_structuring": _load_sar,
    "spending_advisor_williams": _load_spending_advisor,
    "spending_advisor_williams_followup": _load_spending_advisor,
    "cardiac_chest_pain": _load_cardiac_triage,
    "avm_regulation": _load_avm_regulation,
}


def build_fixture_db(db_path: str | Path | None = None) -> Path:
    """
    Build fixture database from all case JSON files.

    Returns path to the created database.
    """
    db_path = Path(db_path) if db_path else DB_PATH

    # Remove existing
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)

    loaded = []
    skipped = []

    for case_file in sorted(FULL_CASES_DIR.glob("*.json")):
        case_name = case_file.stem
        loader = _LOADERS.get(case_name)
        if loader is None:
            skipped.append(case_name)
            continue

        case_data = json.load(open(case_file))
        try:
            loader(conn, case_data, case_name)
            loaded.append(case_name)
        except Exception as e:
            print(f"  ⚠ {case_name}: {e}")
            skipped.append(case_name)

    conn.commit()

    # Summary
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor]
    table_counts = {}
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        if count > 0:
            table_counts[table] = count

    conn.close()

    return db_path, loaded, skipped, table_counts


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"

    if cmd == "build":
        path, loaded, skipped, counts = build_fixture_db()
        print(f"✓ Built {path}")
        print(f"  Cases loaded: {', '.join(loaded)}")
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}")
        print(f"  Tables with data:")
        for table, count in sorted(counts.items()):
            print(f"    {table:30s} {count:>6} rows")

    elif cmd == "list":
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        for (table,) in cursor:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:30s} {count:>6} rows")
        conn.close()

    elif cmd == "query" and len(sys.argv) >= 3:
        table = sys.argv[2]
        key = sys.argv[3] if len(sys.argv) > 3 else None
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        if key:
            # Auto-detect primary key column
            pk_info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            pk_col = next((r["name"] for r in pk_info if r["pk"] == 1), pk_info[0]["name"])
            rows = conn.execute(f"SELECT * FROM {table} WHERE {pk_col} = ?", (key,)).fetchall()
        else:
            rows = conn.execute(f"SELECT * FROM {table} LIMIT 5").fetchall()
        for row in rows:
            print(json.dumps(dict(row), indent=2))
        conn.close()

    else:
        print("Usage:")
        print("  python -m fixtures.db build     # Build from cases/")
        print("  python -m fixtures.db list       # Table row counts")
        print("  python -m fixtures.db query TABLE [KEY]")
