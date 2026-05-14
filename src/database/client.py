"""Best-effort DB client for recording eval runs.

Disk writes remain authoritative; DB inserts are belt-and-suspenders. Any DB
error is logged to stderr and swallowed, and the offending transaction is
rolled back so subsequent calls on the same connection still work.

Usage (pipeline):
    client = EvalClient()
    run_id = str(uuid7())               # mint id locally so disk writes work even if DB is down
    client.register_run(run_id, "pipeline", "evaluation")
    ...
    exp_id = client.record_explorer(run_id, task_id, agent_idx, output, metadata)
    def_id = client.record_definer(run_id, task_id, agent_idx, [exp_id, ...], output, metadata)
    ...
    client.finalize_pipeline_run(run_id)
    client.close()

Usage (baseline / CoT, one-shot):
    client = EvalClient()
    run_id = client.record_baseline_run("baseline", "training", results, token_usage)
    client.close()
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Callable, TypeVar

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

from database.ids import uuid7
from database.metrics import compute_baseline_metrics, compute_pipeline_metrics

load_dotenv()

DEFAULT_DB_URL = "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation"

T = TypeVar("T")


class EvalClient:
    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
        self._conn: psycopg2.extensions.connection | None = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def _try(self, label: str, fn: Callable[[], T]) -> T | None:
        try:
            return fn()
        except psycopg2.OperationalError as e:
            print(f"[db] {label} hit OperationalError ({e}); reconnecting and retrying once", file=sys.stderr)
            self.close()
            try:
                return fn()
            except Exception as e2:
                print(f"[db] {label} failed after reconnect (best-effort, continuing): {e2}", file=sys.stderr)
                if self._conn is not None and not self._conn.closed:
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass
                return None
        except Exception as e:
            print(f"[db] {label} failed (best-effort, continuing): {e}", file=sys.stderr)
            if self._conn is not None and not self._conn.closed:
                try:
                    self._conn.rollback()
                except Exception:
                    pass
            return None

    # ----- pipeline -----

    def register_run(self, run_id: str, system: str, dataset: str) -> str | None:
        """Insert an evals row for a caller-minted run_id. Pipeline owns the id
        so disk writes share the same identifier even if the DB is unreachable."""

        def _do() -> str:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO evals (run_id, system, dataset, data) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (run_id) DO NOTHING",
                    (run_id, system, dataset, Json({})),
                )
            return run_id

        return self._try(f"register_run({system}/{dataset}/{run_id})", _do)

    def record_explorer(
        self,
        run_id: str,
        task_id: str,
        agent_idx: int,
        output: dict,
        metadata: dict,
    ) -> str | None:
        agent_id = str(uuid.uuid4())
        agent = f"pattern_explorer_{agent_idx}"

        def _do() -> str:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO explorers (agent_id, run_id, task_id, agent, output, metadata) "
                    "VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_id, task_id, agent) DO NOTHING",
                    (agent_id, run_id, task_id, agent, Json(output), Json(metadata)),
                )
            return agent_id

        return self._try(f"record_explorer({task_id}/{agent_idx})", _do)

    def record_definer(
        self,
        run_id: str,
        task_id: str,
        agent_idx: int,
        parent_explorer_ids: list[str],
        output: dict,
        metadata: dict,
    ) -> str | None:
        agent_id = str(uuid.uuid4())
        agent = f"transformation_definer_{agent_idx}"

        def _do() -> str:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO definers "
                    "(agent_id, run_id, task_id, agent, parent_explorer_ids, output, metadata) "
                    "VALUES (%s, %s, %s, %s, %s::uuid[], %s, %s) "
                    "ON CONFLICT (run_id, task_id, agent) DO NOTHING",
                    (
                        agent_id,
                        run_id,
                        task_id,
                        agent,
                        parent_explorer_ids,
                        Json(output),
                        Json(metadata),
                    ),
                )
            return agent_id

        return self._try(f"record_definer({task_id}/{agent_idx})", _do)

    def finalize_pipeline_run(self, run_id: str) -> dict[str, Any] | None:
        def _do() -> dict[str, Any]:
            with self._get_conn() as conn, conn.cursor() as cur:
                metrics = compute_pipeline_metrics(cur, run_id)
                cur.execute(
                    "UPDATE evals SET data = %s WHERE run_id = %s",
                    (Json(metrics), run_id),
                )
            return metrics

        return self._try(f"finalize_pipeline_run({run_id})", _do)

    # ----- baseline / CoT one-shot -----

    def record_baseline_run(
        self,
        system: str,
        dataset: str,
        results: list[dict],
        token_usage: dict | None = None,
    ) -> str | None:
        run_id = str(uuid7())
        data = compute_baseline_metrics(results, token_usage)

        def _do() -> str:
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO evals (run_id, system, dataset, data) VALUES (%s, %s, %s, %s)",
                    (run_id, system, dataset, Json(data)),
                )
            return run_id

        return self._try(f"record_baseline_run({system}/{dataset})", _do)
