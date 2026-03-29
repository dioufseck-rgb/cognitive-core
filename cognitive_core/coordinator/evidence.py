"""
Cognitive Core — Case Evidence Store
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extends the coordinator store with domain-specific tables for
case evidence. This is the data layer that both the AI workflow
and the examiner UI read from.

Tables:
  sar_alerts         — alert metadata (ID, type, risk score, rule)
  sar_subjects       — customer profile (demographics, employment, income)
  sar_accounts       — account baseline (type, balances, typical activity)
  sar_transactions   — individual transaction records
  sar_wire_details   — wire transfer specifics (originator, beneficiary, SWIFT)
  sar_prior_alerts   — prior alert/SAR history for the subject
  sar_ctr_history    — CTR filing history (including absence-as-evidence)
  sar_entity_lookups — counterparty/entity research results
  sar_ofac           — OFAC/sanctions screening results
  sar_workflow_outputs — AI workflow step outputs (classify, investigate, etc.)

Design:
  - All tables keyed by alert_id (the case identifier)
  - JSON columns for nested/variable-structure data
  - Loader ingests fixture files or API responses
  - Query methods return typed dicts ready for UI consumption
  - Same SQLite database as coordinator state — one system
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cognitive_core.coordinator.store import CoordinatorStore


class CaseEvidenceStore:
    """
    Case evidence layer on top of CoordinatorStore.
    
    Usage:
        store = CoordinatorStore("coordinator.db")
        evidence = CaseEvidenceStore(store)
        evidence.load_fixture("cases/sar_fixtures.json")
        
        alert = evidence.get_alert("ALT-2026-4471")
        txns = evidence.get_transactions("ALT-2026-4471")
        outputs = evidence.get_workflow_outputs("ALT-2026-4471")
    """

    def __init__(self, store: CoordinatorStore):
        self.store = store
        self.db = store.db
        self._create_evidence_tables()

    def _create_evidence_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sar_alerts (
                alert_id TEXT PRIMARY KEY,
                generated_at TEXT,
                alert_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                monitoring_rule TEXT,
                disposition TEXT DEFAULT 'pending_review',
                assigned_to TEXT,
                batch_id TEXT,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS sar_subjects (
                alert_id TEXT PRIMARY KEY REFERENCES sar_alerts(alert_id),
                customer_id TEXT,
                full_name TEXT NOT NULL,
                date_of_birth TEXT,
                ssn_masked TEXT,
                address TEXT,
                phone TEXT,
                email TEXT,
                employer TEXT,
                title TEXT,
                employed_since TEXT,
                naics_code TEXT,
                income_annual REAL,
                income_source TEXT,
                income_verified INTEGER DEFAULT 0,
                risk_rating TEXT DEFAULT 'low',
                relationship_since TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_accounts (
                alert_id TEXT PRIMARY KEY REFERENCES sar_alerts(alert_id),
                account_number_masked TEXT,
                account_type TEXT,
                product_name TEXT,
                opened_date TEXT,
                status TEXT DEFAULT 'active',
                avg_balance_6mo REAL,
                avg_deposits_6mo REAL,
                avg_cash_deposits_6mo REAL,
                typical_range TEXT,
                typical_counterparties TEXT,
                activity_notes TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL REFERENCES sar_alerts(alert_id),
                txn_id TEXT NOT NULL,
                txn_date TEXT NOT NULL,
                txn_time TEXT,
                txn_type TEXT NOT NULL,
                amount REAL NOT NULL,
                location TEXT,
                teller_id TEXT,
                atm_id TEXT,
                running_balance REAL,
                reference TEXT,
                payee TEXT,
                check_number TEXT,
                wire_details TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_prior_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL REFERENCES sar_alerts(alert_id),
                prior_alert_id TEXT NOT NULL,
                prior_date TEXT,
                prior_type TEXT,
                disposition TEXT,
                analyst TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_ctr_history (
                alert_id TEXT PRIMARY KEY REFERENCES sar_alerts(alert_id),
                filings_12mo INTEGER DEFAULT 0,
                expected_filings TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_entity_lookups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT NOT NULL REFERENCES sar_alerts(alert_id),
                entity_name TEXT NOT NULL,
                jurisdiction TEXT,
                source TEXT,
                result TEXT,
                detail TEXT,
                searched_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_ofac (
                alert_id TEXT PRIMARY KEY REFERENCES sar_alerts(alert_id),
                screening_id TEXT,
                entities_screened TEXT,
                lists_checked TEXT,
                result TEXT NOT NULL,
                screened_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sar_workflow_outputs (
                alert_id TEXT PRIMARY KEY REFERENCES sar_alerts(alert_id),
                classify_output TEXT,
                investigate_output TEXT,
                filing_output TEXT,
                narrative_text TEXT,
                challenge_output TEXT,
                verify_output TEXT,
                completed_at REAL,
                elapsed_seconds REAL
            );

            CREATE INDEX IF NOT EXISTS idx_sar_txn_alert ON sar_transactions(alert_id);
            CREATE INDEX IF NOT EXISTS idx_sar_txn_date ON sar_transactions(txn_date);
            CREATE INDEX IF NOT EXISTS idx_sar_entity_alert ON sar_entity_lookups(alert_id);
            CREATE INDEX IF NOT EXISTS idx_sar_prior_alert ON sar_prior_alerts(alert_id);
        """)
        self.db.commit()

    # ─── Fixture Loader ──────────────────────────────────────────

    def load_fixture(self, path: str | Path) -> int:
        """
        Load a fixture JSON file into the evidence tables.
        Returns the number of cases loaded.
        """
        with open(path) as f:
            data = json.load(f)

        cases = data.get("cases", [])
        for case in cases:
            self._load_case(case)
        self.db.commit()
        return len(cases)

    def load_case_dict(self, case: dict[str, Any]) -> str:
        """Load a single case dict into the store. Returns alert_id."""
        alert_id = self._load_case(case)
        self.db.commit()
        return alert_id

    def _load_case(self, case: dict[str, Any]) -> str:
        alert = case["alert"]
        alert_id = alert["alert_id"]
        now = time.time()

        # Alert
        self.db.execute("""
            INSERT OR REPLACE INTO sar_alerts
            (alert_id, generated_at, alert_type, risk_score, monitoring_rule, disposition, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (alert_id, alert.get("generated_at"), alert["alert_type"],
              alert["risk_score"], alert.get("monitoring_rule"),
              alert.get("disposition", "pending_review"), now))

        # Subject
        subj = case["subject"]
        inc = subj.get("income", {})
        self.db.execute("""
            INSERT OR REPLACE INTO sar_subjects
            (alert_id, customer_id, full_name, date_of_birth, ssn_masked,
             address, phone, email, employer, title, employed_since,
             naics_code, income_annual, income_source, income_verified,
             risk_rating, relationship_since)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (alert_id, subj.get("customer_id"), subj["full_name"],
              subj.get("date_of_birth"), subj.get("ssn_masked"),
              subj.get("address"), subj.get("phone"), subj.get("email"),
              subj.get("employment", {}).get("employer") or subj.get("employer"),
              subj.get("employment", {}).get("title") or subj.get("title"),
              subj.get("employment", {}).get("since") or subj.get("since"),
              subj.get("employment", {}).get("naics_code") or subj.get("naics"),
              inc.get("stated_annual") or inc.get("annual"),
              inc.get("source"),
              1 if inc.get("verified") else 0,
              subj.get("risk_rating", {}).get("current") if isinstance(subj.get("risk_rating"), dict) else subj.get("risk_rating", "low"),
              subj.get("relationship_since")))

        # Account
        acct = case["account"]
        summary = acct.get("monthly_activity_summary", {})
        self.db.execute("""
            INSERT OR REPLACE INTO sar_accounts
            (alert_id, account_number_masked, account_type, product_name,
             opened_date, avg_balance_6mo, avg_deposits_6mo, avg_cash_deposits_6mo,
             typical_range, typical_counterparties, activity_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (alert_id,
              acct.get("account_number_masked") or acct.get("number"),
              acct.get("account_type") or acct.get("type"),
              acct.get("product") or acct.get("product_name"),
              acct.get("opened") or acct.get("opened_date"),
              acct.get("average_balance_6mo") or acct.get("avg_balance_6mo"),
              summary.get("avg_deposits_6mo") or acct.get("avg_deposits_6mo"),
              summary.get("avg_cash_deposits_6mo") or acct.get("avg_cash_6mo"),
              acct.get("typical_range") or str(summary.get("typical_deposit_range", "")),
              json.dumps(summary.get("typical_counterparties", [])),
              summary.get("notes") or acct.get("activity_notes", "")))

        # Transactions
        for txn in case.get("transactions", []):
            wire = txn.get("wire_detail") or txn.get("wire")
            self.db.execute("""
                INSERT INTO sar_transactions
                (alert_id, txn_id, txn_date, txn_time, txn_type, amount,
                 location, teller_id, atm_id, running_balance, reference,
                 payee, check_number, wire_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, txn.get("txn_id") or txn.get("id"),
                  txn.get("date") or txn.get("txn_date"),
                  txn.get("time") or txn.get("txn_time"),
                  txn.get("type") or txn.get("txn_type"),
                  txn["amount"],
                  txn.get("branch") or txn.get("location"),
                  txn.get("teller_id") or txn.get("teller"),
                  txn.get("atm_id"),
                  txn.get("running_balance") or txn.get("balance"),
                  txn.get("ref") or txn.get("reference"),
                  txn.get("payee"),
                  txn.get("check_number") or txn.get("check_no"),
                  json.dumps(wire) if wire else None))

        # Prior alerts
        for pa in case.get("prior_alerts", []):
            self.db.execute("""
                INSERT INTO sar_prior_alerts
                (alert_id, prior_alert_id, prior_date, prior_type,
                 disposition, analyst, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, pa.get("alert_id") or pa.get("prior_alert_id"),
                  pa.get("date") or pa.get("prior_date"),
                  pa.get("type") or pa.get("prior_type"),
                  pa["disposition"], pa.get("analyst"), pa.get("notes")))

        # CTR history
        ctr = case.get("ctr_history") or case.get("ctr", {})
        if ctr:
            self.db.execute("""
                INSERT OR REPLACE INTO sar_ctr_history
                (alert_id, filings_12mo, expected_filings, notes)
                VALUES (?, ?, ?, ?)
            """, (alert_id, ctr.get("filings_12mo") or ctr.get("filings", 0),
                  ctr.get("expected_filings") or ctr.get("expected"),
                  ctr.get("notes") or ctr.get("note")))

        # Entity lookups
        for ent in case.get("entity_lookups", case.get("entities", [])):
            self.db.execute("""
                INSERT INTO sar_entity_lookups
                (alert_id, entity_name, jurisdiction, source, result, detail, searched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (alert_id, ent.get("entity") or ent.get("entity_name"),
                  ent.get("jurisdiction"), ent.get("source"),
                  ent["result"],
                  ent.get("details") or ent.get("detail"),
                  ent.get("searched_at")))

        # OFAC
        ofac = case.get("ofac_screening") or case.get("ofac", {})
        if ofac:
            self.db.execute("""
                INSERT OR REPLACE INTO sar_ofac
                (alert_id, screening_id, entities_screened, lists_checked, result, screened_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (alert_id, ofac.get("screening_id"),
                  ofac.get("entity_screened") or ofac.get("screened"),
                  json.dumps(ofac.get("lists_checked", [])),
                  ofac["result"],
                  ofac.get("timestamp") or ofac.get("screened_at")))

        # AI workflow outputs (if present)
        ai = case.get("ai")
        if ai:
            self.save_workflow_outputs(alert_id, ai)

        return alert_id

    # ─── AI Workflow Output ──────────────────────────────────────

    def save_workflow_outputs(
        self, alert_id: str, outputs: dict[str, Any],
        elapsed_seconds: float | None = None,
    ) -> None:
        """Save AI workflow step outputs for a case."""
        self.db.execute("""
            INSERT OR REPLACE INTO sar_workflow_outputs
            (alert_id, classify_output, investigate_output, filing_output,
             narrative_text, challenge_output, verify_output,
             completed_at, elapsed_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (alert_id,
              json.dumps(outputs.get("classify", outputs.get("classify_alert"))),
              json.dumps(outputs.get("investigate", outputs.get("investigate_activity"))),
              json.dumps(outputs.get("filing", outputs.get("classify_filing"))),
              outputs.get("narrative", outputs.get("narrative_text", "")),
              json.dumps(outputs.get("challenge")),
              json.dumps(outputs.get("verify")),
              time.time(), elapsed_seconds))
        self.db.commit()

    # ─── Query Methods ───────────────────────────────────────────

    def list_alerts(self, disposition: str | None = None) -> list[dict]:
        """List all alerts, optionally filtered by disposition."""
        q = "SELECT a.*, s.full_name, s.customer_id FROM sar_alerts a JOIN sar_subjects s ON a.alert_id = s.alert_id"
        params = []
        if disposition:
            q += " WHERE a.disposition = ?"
            params.append(disposition)
        q += " ORDER BY a.risk_score DESC"
        rows = self.db.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def get_alert(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def get_subject(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_subjects WHERE alert_id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def get_account(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_accounts WHERE alert_id = ?", (alert_id,)).fetchone()
        if row:
            d = dict(row)
            d["typical_counterparties"] = json.loads(d.get("typical_counterparties") or "[]")
            return d
        return None

    def get_transactions(self, alert_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sar_transactions WHERE alert_id = ? ORDER BY txn_date, txn_time",
            (alert_id,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("wire_details"):
                d["wire_details"] = json.loads(d["wire_details"])
            results.append(d)
        return results

    def get_prior_alerts(self, alert_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sar_prior_alerts WHERE alert_id = ? ORDER BY prior_date",
            (alert_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_ctr_history(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_ctr_history WHERE alert_id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None

    def get_entity_lookups(self, alert_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM sar_entity_lookups WHERE alert_id = ? ORDER BY searched_at",
            (alert_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_ofac(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_ofac WHERE alert_id = ?", (alert_id,)).fetchone()
        if row:
            d = dict(row)
            d["lists_checked"] = json.loads(d.get("lists_checked") or "[]")
            return d
        return None

    def get_workflow_outputs(self, alert_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM sar_workflow_outputs WHERE alert_id = ?", (alert_id,)).fetchone()
        if row:
            d = dict(row)
            for key in ("classify_output", "investigate_output", "filing_output", "challenge_output", "verify_output"):
                if d.get(key):
                    d[key] = json.loads(d[key])
            return d
        return None

    def get_full_case(self, alert_id: str) -> dict | None:
        """Get complete case evidence package for UI rendering."""
        alert = self.get_alert(alert_id)
        if not alert:
            return None
        return {
            "alert": alert,
            "subject": self.get_subject(alert_id),
            "account": self.get_account(alert_id),
            "transactions": self.get_transactions(alert_id),
            "prior_alerts": self.get_prior_alerts(alert_id),
            "ctr_history": self.get_ctr_history(alert_id),
            "entity_lookups": self.get_entity_lookups(alert_id),
            "ofac": self.get_ofac(alert_id),
            "workflow_outputs": self.get_workflow_outputs(alert_id),
        }

    # ─── Disposition Updates ─────────────────────────────────────

    def update_disposition(self, alert_id: str, disposition: str, assigned_to: str | None = None) -> None:
        """Update alert disposition (pending_review, approved, rejected, escalated)."""
        self.db.execute(
            "UPDATE sar_alerts SET disposition = ?, assigned_to = COALESCE(?, assigned_to) WHERE alert_id = ?",
            (disposition, assigned_to, alert_id)
        )
        self.db.commit()

    # ─── Statistics ──────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary statistics across all loaded cases."""
        alerts = self.db.execute("SELECT COUNT(*) as c FROM sar_alerts").fetchone()
        pending = self.db.execute("SELECT COUNT(*) as c FROM sar_alerts WHERE disposition = 'pending_review'").fetchone()
        txns = self.db.execute("SELECT COUNT(*) as c FROM sar_transactions").fetchone()
        entities = self.db.execute("SELECT COUNT(*) as c FROM sar_entity_lookups").fetchone()
        outputs = self.db.execute("SELECT COUNT(*) as c FROM sar_workflow_outputs").fetchone()
        return {
            "total_alerts": alerts["c"],
            "pending_review": pending["c"],
            "total_transactions": txns["c"],
            "total_entity_lookups": entities["c"],
            "workflow_outputs_complete": outputs["c"],
        }
