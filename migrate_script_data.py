"""
One-shot migration: unwrap JSON-encoded script_data strings in SQLite.

The script_data column was previously typed as JSON in SQLAlchemy, causing
plain string scripts to be stored as JSON-encoded string literals, e.g.:
    "\"The full script text\""

This script:
1. Reads every row's raw script_data value via raw SQL (bypassing SA JSON decoding)
2. If the value is a JSON-encoded string (starts/ends with "), unwraps it
3. Writes the clean plain text back

Safe to re-run: rows already containing plain text are left unchanged.
"""

import json
import sqlite3
import os

DB_PATH = os.getenv("DATABASE_PATH", "data/content_history.db")


def unwrap_if_json_string(raw):
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, str):
                return parsed  # successfully unwrapped
        except json.JSONDecodeError:
            pass
    return raw  # already plain text — leave as-is


def run_migration():
    print(f"Connecting to: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    rows = cursor.execute("SELECT id, script_data FROM content_history").fetchall()
    print(f"Found {len(rows)} rows to inspect.")

    updated = 0
    for row in rows:
        raw = row["script_data"]
        clean = unwrap_if_json_string(raw)
        if clean != raw:
            cursor.execute(
                "UPDATE content_history SET script_data = ? WHERE id = ?",
                (clean, row["id"])
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"Migration complete. {updated} row(s) updated.")


if __name__ == "__main__":
    run_migration()
