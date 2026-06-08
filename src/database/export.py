"""
export.py

Export the evaluation database for the published repo.

The Postgres database is the canonical source of truth for run data, but it is
848 MB (mostly per-agent traces in `definers`/`explorers`) — too large to commit.
This script produces two small, committed artifacts that let a reader verify the
headline numbers without the live DB, plus an on-demand full dump for transfer.

Committed artifacts (default):
    python -m database.export
      -> exports/runs_summary.csv   one row per eval run: split, accuracy, pass@k,
                                    tokens, and the config knobs (N/M/temperature).
      -> exports/schema.sql         schema-only pg_dump (structure reference,
                                    complements database/migrations/).

Full local dump (NOT committed; gitignored):
    python -m database.export --full
      -> exports/arc_evaluation_full.sql.gz   complete pg_dump.

Restore the full dump into a fresh database:
    createdb arc_evaluation
    gunzip -c exports/arc_evaluation_full.sql.gz | psql "$DATABASE_URL"

`schema.sql` and `--full` need `pg_dump`. The dump is produced inside the compose
container (`arc_evaluation`) when it is running — this guarantees the client
version matches the server — otherwise a host `pg_dump` on PATH is used.
DATABASE_URL is read from the environment (.env), defaulting to the local compose DB.
"""

import os
import csv
import sys
import gzip
import shutil
import argparse
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation",
)

EXPORTS_DIR = Path(__file__).parent / "exports"

# Compose container name (see database/compose.yaml). pg_dump is run inside it
# when available so the client version matches the server.
DB_CONTAINER = os.getenv("ARC_DB_CONTAINER", "arc_evaluation")

# Columns written to runs_summary.csv. Scalar table columns first, then keys
# pulled out of the `data` JSONB (tolerant of absent keys — schema varies by
# architecture, e.g. baseline rows have no pass@k).
SUMMARY_COLUMNS = [
    "run_id",
    "created_at",
    "system",           # approach: baseline / CoT / arc_base / pipeline / orchestrator
    "dataset",          # split: training / evaluation
    "num_tasks",
    "accuracy",
    "accuracy_test_pair_level",
    "accuracy_file_level",
    "accuracy_pass_at_1",
    "accuracy_pass_at_2",
    "accuracy_pass_at_m",
    "total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
    "num_explorers",
    "explorer_temperature",
    "num_definers",
    "mode",
    "ablation",
    "source_explorer_run_id",
]


def _container_running() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{DB_CONTAINER}$", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        return DB_CONTAINER in out.stdout.split()
    except (subprocess.SubprocessError, OSError):
        return False


def _pg_dump_command(url: str, extra: list[str]) -> tuple[list[str], dict] | None:
    """
    Build a pg_dump command (+ env) for the given DSN, preferring the compose
    container so the client version matches the server. Returns None if no
    pg_dump is available either way.
    """
    p = urlparse(url)
    user = p.username or "postgres"
    db = (p.path or "/").lstrip("/")
    env = os.environ.copy()

    if _container_running():
        # Connect locally inside the container as the DB user.
        pw = ["-e", f"PGPASSWORD={p.password}"] if p.password else []
        cmd = ["docker", "exec", *pw, DB_CONTAINER,
               "pg_dump", "-U", user, "-d", db, *extra]
        return cmd, env

    if shutil.which("pg_dump"):
        if p.password:
            env["PGPASSWORD"] = p.password
        cmd = ["pg_dump", "-h", p.hostname or "localhost", "-p", str(p.port or 5432),
               "-U", user, "-d", db, *extra]
        return cmd, env

    return None


def export_runs_summary(conn) -> Path:
    """Write run-level metrics, one row per eval run, to runs_summary.csv."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, created_at, system, dataset, data "
            "FROM evals ORDER BY created_at"
        )
        rows = cur.fetchall()

    out = EXPORTS_DIR / "runs_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for run_id, created_at, system, dataset, data in rows:
            data = data or {}
            tokens = data.get("tokens") or {}
            w.writerow([
                run_id,
                created_at.isoformat() if created_at else "",
                system,
                dataset,
                data.get("num_tasks"),
                data.get("accuracy"),
                data.get("accuracy_test_pair_level"),
                data.get("accuracy_file_level"),
                data.get("accuracy_pass_at_1"),
                data.get("accuracy_pass_at_2"),
                data.get("accuracy_pass_at_m"),
                tokens.get("total"),
                tokens.get("total_prompt"),
                tokens.get("total_completion"),
                data.get("num_explorers"),
                data.get("explorer_temperature"),
                data.get("num_definers"),
                data.get("mode"),
                data.get("ablation"),
                data.get("source_explorer_run_id"),
            ])
    print(f"[export] wrote {out} ({len(rows)} runs)")
    return out


def export_schema(url: str) -> Path | None:
    """Schema-only pg_dump → exports/schema.sql. Best-effort; skipped if no pg_dump."""
    built = _pg_dump_command(url, ["--schema-only", "--no-owner", "--no-privileges"])
    if built is None:
        print("[export] no pg_dump (host or container); skipping schema.sql", file=sys.stderr)
        return None
    cmd, env = built
    proc = subprocess.run(cmd, capture_output=True, env=env)
    if proc.returncode != 0:
        print(f"[export] schema dump failed; skipping schema.sql:\n{proc.stderr.decode().strip()}",
              file=sys.stderr)
        return None
    out = EXPORTS_DIR / "schema.sql"
    out.write_bytes(proc.stdout)
    print(f"[export] wrote {out}")
    return out


def export_full(url: str) -> Path:
    """Complete gzipped pg_dump → exports/arc_evaluation_full.sql.gz (gitignored)."""
    built = _pg_dump_command(url, ["--no-owner", "--no-privileges"])
    if built is None:
        sys.exit("[export] no pg_dump (host or container); cannot produce --full dump")
    cmd, env = built
    out = EXPORTS_DIR / "arc_evaluation_full.sql.gz"
    with gzip.open(out, "wb") as gz:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)
        assert proc.stdout is not None
        shutil.copyfileobj(proc.stdout, gz)
        proc.stdout.close()
        if proc.wait() != 0:
            out.unlink(missing_ok=True)
            sys.exit("[export] pg_dump failed")
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main():
    ap = argparse.ArgumentParser(description="Export the ARC evaluation database.")
    ap.add_argument("--full", action="store_true",
                    help="Also produce the complete gzipped dump (gitignored).")
    args = ap.parse_args()

    EXPORTS_DIR.mkdir(exist_ok=True)
    conn = psycopg2.connect(DB_URL)
    try:
        export_runs_summary(conn)
    finally:
        conn.close()
    export_schema(DB_URL)
    if args.full:
        export_full(DB_URL)


if __name__ == "__main__":
    main()
