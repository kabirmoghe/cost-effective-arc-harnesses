"""
migrate.py

Usage:
    python migrate.py            # apply all pending migrations
    python migrate.py --status   # show which migrations have been applied

Requires:
    pip install psycopg2-binary python-dotenv
"""

import os
import sys
import argparse
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation"
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_connection():
    return psycopg2.connect(DB_URL)


def ensure_migrations_table(conn):
    with conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    TEXT        PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)


def get_applied(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        return {row[0] for row in cur.fetchall()}


def get_pending(applied):
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [f for f in files if f.stem not in applied]


def apply_migration(conn, path: Path):
    sql = path.read_text()
    with conn, conn.cursor() as cur:
        cur.execute(sql)
        cur.execute(
            "INSERT INTO schema_migrations (version) VALUES (%s)",
            (path.stem,)
        )


def cmd_migrate():
    conn = get_connection()

    ensure_migrations_table(conn)
    applied = get_applied(conn)
    pending = get_pending(applied)

    if not pending:
        print("No pending migrations.")
        return

    for path in pending:
        print(f"  applying  {path.name} ...", end=" ", flush=True)
        apply_migration(conn, path)
        print("done")

    print(f"\n{len(pending)} migration(s) applied.")
    conn.close()


def cmd_status():
    conn = get_connection()

    ensure_migrations_table(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version")
        applied = {row[0]: row[1] for row in cur.fetchall()}

    all_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    if not all_files:
        print("No migration files found.")
        return

    print(f"{'version':<40} {'status':<12} applied_at")
    print("-" * 70)
    for f in all_files:
        if f.stem in applied:
            ts = applied[f.stem].strftime("%Y-%m-%d %H:%M")
            print(f"{f.stem:<40} {'applied':<12} {ts}")
        else:
            print(f"{f.stem:<40} {'pending':<12}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run database migrations.")
    parser.add_argument("--status", action="store_true", help="Show migration status.")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    else:
        cmd_migrate()