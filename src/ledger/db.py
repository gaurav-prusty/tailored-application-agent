"""SQLite layer for the Claims Ledger.

1. Only module that writes SQL; all other code (sub-agents) call these functions.

CLI:
    python -m src.ledger.db init    # Create db/ledger.sqlite + schema
    python -m src.ledger.db count   # Print row count
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# parents[2] climbs db.py -> ledger/ -> src/ -> project root, so paths resolve correctly regardless of cwd when invoked via `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "db" / "ledger.sqlite"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass
class Claim:
    """One row of the `claim` table — the canonical Python type for a ledger entry."""
    id: str
    category: str
    text: str
    source_role: str | None
    source_company: str | None
    has_metric: bool
    raw_original: str


def _connect() -> sqlite3.Connection:
    """Open a connection with Row factory enabled (dict-like row access)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the schema if it doesn't exist. Idempotent."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with _connect() as conn:
        conn.executescript(schema_sql)
    print(f"Initialized {DB_PATH}")


def insert_claim(claim: Claim) -> None:
    """Insert one claim. Raises sqlite3.IntegrityError on duplicate id or CHECK violation."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO claim (
                id, category, text, source_role, source_company,
                has_metric, raw_original
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.id,
                claim.category,
                claim.text,
                claim.source_role,
                claim.source_company,
                int(claim.has_metric),
                claim.raw_original,
            ),
        )


def insert_claims(claims: Iterable[Claim]) -> int:
    """Bulk-insert claims in a single transaction. Returns the number inserted."""
    rows = [
        (
            c.id, c.category, c.text, c.source_role, c.source_company,
            int(c.has_metric), c.raw_original,
        )
        for c in claims
    ]
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO claim (
                id, category, text, source_role, source_company,
                has_metric, raw_original
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_claims(
    *,
    category: str | None = None,
    source_role: str | None = None,
) -> list[Claim]:
    """Fetch claims, optionally filtered by category and/or source_role."""
    where_parts: list[str] = []
    params: list[object] = []
    if category is not None:
        where_parts.append("category = ?")
        params.append(category)
    if source_role is not None:
        where_parts.append("source_role = ?")
        params.append(source_role)

    sql = "SELECT * FROM claim"
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    sql += " ORDER BY id"

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [
        Claim(
            id=row["id"],
            category=row["category"],
            text=row["text"],
            source_role=row["source_role"],
            source_company=row["source_company"],
            has_metric=bool(row["has_metric"]),
            raw_original=row["raw_original"],
        )
        for row in rows
    ]


def count_claims() -> int:
    """Return the total number of rows in the claim table."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]


def _cli() -> None:
    """Tiny dispatcher for `python -m src.ledger.db {init|count}`."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"init", "count"}:
        print("usage: python -m src.ledger.db {init|count}", file=sys.stderr)
        sys.exit(2)

    cmd = sys.argv[1]
    if cmd == "init":
        init_db()
    elif cmd == "count":
        print(f"{count_claims()} claims in {DB_PATH}")


if __name__ == "__main__":
    _cli()
