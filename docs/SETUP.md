# Setup & running the harnesses

Everything runs from the `src/` directory. All commands below assume `cd src`
first.

## 1. Prerequisites

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) (dependency management; `uv.lock` is committed)
- Docker (for the local Postgres tracking database)
- An OpenRouter API key

## 2. Install dependencies

```bash
cd src
uv sync            # creates .venv from pyproject.toml + uv.lock
```

(Equivalently, `pip install -e .` into a Python 3.10+ environment.)

## 3. Configure environment

```bash
cd src
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY (and DATABASE_URL if not using the default)
```

`OPENROUTER_API_KEY` is the only key needed for the headline runs. See
`.env.example` for the full list and what each var is for.

## 4. Start the database

The DB is the canonical source of truth for run data; disk JSON is a backup.

```bash
cd src
docker compose up -d           # starts Postgres 15 on host port 5433 (see compose.yaml)
python -m database.migrate     # apply schema migrations
python -m database.migrate --status   # verify
```

To inspect or export results, see [`database/exports/README.md`](../src/database/exports/README.md):

```bash
python -m database.export          # runs_summary.csv + schema.sql
python -m database.export --full   # complete gzipped dump (gitignored)
```

## 5. Run a harness

All harnesses share `--split {training,evaluation}`, `--limit N` (cap tasks),
`--task <id>` (single task), `--output <path>`, and
`--provider {openrouter,openai,...}`. Each run is recorded as one row in `evals`.

> **Provider:** the headline model is DeepSeek V3.2 via OpenRouter, pinned to the
> AtlasCloud FP8 backend in `shared/llm.py`. Use `--provider openrouter`.
> Note `tool_choice="required"` is **not** supported by AtlasCloud FP8 — the
> pipeline/orchestrator agents use `"auto"` accordingly.

> **Two concurrency models — don't confuse them:**
> - Baseline / CoT / arc_base use **`--workers`** (thread pool): 1 worker ≈ 1
>   in-flight API call. Bump to 20–30 on Atlas.
> - Pipeline / Orchestrator use **`--max-concurrent-tasks`** (asyncio semaphore):
>   each task fans out to `num_explorers + 1` agents, so in-flight calls ≈
>   `max_concurrent_tasks × (num_explorers + 1)`. Keep total ≲ 50–100.

### Baseline (no-CoT, single call per test pair)

```bash
python -m baseline.run --split evaluation --provider openrouter --workers 24
```

### CoT (single call with reasoning prelude)

```bash
python -m CoT.run --split evaluation --provider openrouter --workers 24
```

### arc_base (faithful port of the ARC Prize "Base LLM" harness, pass@2)

```bash
python -m arc_base.run --split evaluation --provider openrouter --workers 24
```

### Pipeline (primary architecture: N explorers → M definers, pass@k)

Full end-to-end (locked headline config N=5 / t=0.5 / M=5 + refinement):

```bash
python -m pipeline.run --split evaluation --provider openrouter \
  --num-explorers 5 --explorer-temperature 0.5 --num-definers 5 \
  --definer-refinement --max-concurrent-tasks 20
```

Explorer-only (produce a reusable explorer set), then a definer-only re-run that
reuses it (cheap pass@k sweeps and the input to the Orchestrator):

```bash
python -m pipeline.run --explorers-only --num-explorers 5 \
  --explorer-temperature 0.5 --run-id <RUN_A> ...
python -m pipeline.run --from-explorers <RUN_A> --num-definers 5 \
  --definer-refinement --max-concurrent-tasks 20
```

Other useful flags: `--resume` (skip already-done tasks; needs `--run-id`),
`--quiet`, `--ablation act-only` (remove the definer `think` tool).

### Reflective Orchestrator (single agentic loop with mid-loop explorer spawn)

Always a definer-only run off a prior pipeline explorer set (`--from-explorers`
is required):

```bash
python -m orchestrator.run --split evaluation --provider openrouter \
  --from-explorers <EXPLORER_RUN_ID> --num-definers 5 --max-concurrent-tasks 20
```

## 6. Reproduce figures & analysis

Analysis scripts read from the DB and write to `docs/`. Figure generators live in
`src/figures/`. See the **Writeup asset map** in
[`RESEARCH_TRAJECTORY.md`](../RESEARCH_TRAJECTORY.md) for the full
script → artifact → figure mapping. Examples:

```bash
python -m scripts.bootstrap_cis        # docs/bootstrap_cis.json
python -m scripts.unbiased_passk       # docs/unbiased_passk.json
python -m figures.headline_pareto      # docs/figures/headline_pareto.{pdf,png}
```

Observed run times (400 tasks, V3.2/Atlas): baseline/CoT ≈ 5 min each at high
worker counts; full pipeline ≈ 1–1.5 h at `--max-concurrent-tasks 20`;
definer-only re-runs ≈ 40–50 min.
