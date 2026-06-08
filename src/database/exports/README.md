# Database exports

The Postgres database (`evals`, `explorers`, `definers`) is the canonical source
of truth for every run. It is ~850 MB — too large to commit — so this directory
holds small, version-controlled artifacts that let a reader inspect results
without the live DB, plus instructions for restoring a full dump.

Regenerate everything here with:

```bash
cd src
python -m database.export          # runs_summary.csv + schema.sql
python -m database.export --full   # + arc_evaluation_full.sql.gz (gitignored)
```

`pg_dump` runs inside the compose container (`arc_evaluation`) when it is up, so
the client version matches the server automatically.

## Committed files

| File | What |
|---|---|
| `runs_summary.csv` | One row per eval run: approach (`system`), split (`dataset`), accuracy + pass@k, token totals, and the config knobs (N explorers, M definers, explorer temperature, mode, ablation, source explorer run). The fastest way to map a `run_id` to its headline numbers. |
| `schema.sql` | Schema-only `pg_dump` — full table/constraint/index definitions. Complements `../migrations/` (which is the incremental history). |

## Full dump (not committed)

`arc_evaluation_full.sql.gz` is the complete database (all per-agent traces). It
is gitignored. To restore it into a fresh database:

```bash
createdb arc_evaluation        # or use the compose service in ../compose.yaml
gunzip -c arc_evaluation_full.sql.gz | psql "$DATABASE_URL"
```
