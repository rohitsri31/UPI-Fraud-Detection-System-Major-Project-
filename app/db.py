"""Small SQLite store for scored transactions.

SQLite needs no setup, which suits a base project. To move to MySQL or MongoDB later,
keep these four function names and signatures and change only what is inside them.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    amount REAL NOT NULL,
    txn_type TEXT NOT NULL,
    hour INTEGER NOT NULL,
    is_new_payee INTEGER NOT NULL,
    is_new_device INTEGER NOT NULL,
    is_new_city INTEGER NOT NULL,
    txns_last_hour INTEGER NOT NULL,
    amount_ratio REAL NOT NULL,
    probability REAL NOT NULL,
    flagged INTEGER NOT NULL,
    risk_level TEXT NOT NULL,
    indicators TEXT NOT NULL
)
"""


def _connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path):
    with closing(_connect(path)) as conn, conn:
        conn.execute(SCHEMA)


def save_transaction(path, amount, txn_type, result):
    f = result["features"]
    values = (datetime.now(timezone.utc).isoformat(timespec="seconds"), amount, txn_type,
              int(f["hour"]), int(f["is_new_payee"]), int(f["is_new_device"]), int(f["is_new_city"]),
              int(f["txns_last_hour"]), f["amount_ratio"], result["fraud_probability"],
              int(result["flagged"]), result["risk_level"], json.dumps(result["indicators"]))
    with closing(_connect(path)) as conn, conn:
        cursor = conn.execute(
            "INSERT INTO transactions (created_at, amount, txn_type, hour, is_new_payee, is_new_device, "
            "is_new_city, txns_last_hour, amount_ratio, probability, flagged, risk_level, indicators) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        return cursor.lastrowid


def recent_transactions(path, flagged_only=False, limit=25):
    query = "SELECT * FROM transactions"
    if flagged_only:
        query += " WHERE flagged = 1"
    query += " ORDER BY id DESC LIMIT ?"
    with closing(_connect(path)) as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["indicators"] = json.loads(item["indicators"])
        item["flagged"] = bool(item["flagged"])
        out.append(item)
    return out


def stats(path):
    with closing(_connect(path)) as conn:
        total, flagged = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(flagged), 0) FROM transactions").fetchone()
    return {"total": total, "flagged": int(flagged), "flag_rate": (flagged / total) if total else 0.0}
