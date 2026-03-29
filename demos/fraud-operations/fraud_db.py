"""
Cognitive Core — Fraud Data SQLite Layer

Simple key-value store for tool data keyed by (case_id, tool_name).
Seeded from *_tools.json fixture files; queried by the MCP server.

Schema:
    case_tools(case_id TEXT, tool_name TEXT, data TEXT,
               UNIQUE(case_id, tool_name))
"""

import json
import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS case_tools (
            case_id   TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            data      TEXT NOT NULL,
            UNIQUE(case_id, tool_name)
        )
    """)
    conn.commit()
    return conn


def import_fixture_file(db_path: str | Path, fixture_path: str | Path) -> int:
    """
    Import a single *_tools.json fixture file into the DB.

    The file must have a top-level "_case_id" key. All other keys
    that start with "get_" are imported as (case_id, tool_name, data).

    Returns number of rows inserted (0 if already present).
    """
    with open(fixture_path) as f:
        raw = json.load(f)

    case_id = raw.get("_case_id") or raw.get("_claim_id", "")
    if not case_id:
        return 0

    rows = [
        (case_id, key, json.dumps(value))
        for key, value in raw.items()
        if key.startswith("get_") and not key.startswith("_")
    ]
    if not rows:
        return 0

    conn = _connect(db_path)
    try:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO case_tools (case_id, tool_name, data) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def import_fixtures_dir(db_path: str | Path, fixtures_dir: str | Path) -> int:
    """
    Scan a directory and import all *_tools.json files not yet in DB.

    Returns total rows inserted across all files.
    """
    fixtures_dir = Path(fixtures_dir)
    if not fixtures_dir.is_dir():
        return 0

    total = 0
    for fixture_file in sorted(fixtures_dir.glob("*_tools.json")):
        try:
            inserted = import_fixture_file(db_path, fixture_file)
            total += inserted
        except Exception as exc:
            print(f"  ⚠ Skipping {fixture_file.name}: {exc}")
    return total


def get_tool_data(db_path: str | Path, case_id: str, tool_name: str) -> dict[str, Any] | None:
    """Fetch a single tool's data for a case. Returns None if not found."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT data FROM case_tools WHERE case_id = ? AND tool_name = ?",
            (case_id, tool_name),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])
    finally:
        conn.close()


def list_tool_names(db_path: str | Path) -> list[str]:
    """Return distinct tool names available across all cases."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT tool_name FROM case_tools ORDER BY tool_name"
        ).fetchall()
        return [r["tool_name"] for r in rows]
    finally:
        conn.close()
