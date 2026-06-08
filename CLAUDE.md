# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Senior thesis research on **Agentic Architectures for ARC AGI**. The goal is to perform meta-review of high-performance open architectures on ARC AGI and use findings to inform a custom architecture.

**Scope:** Focus primarily on ARC AGI 1; if successful, extend to ARC AGI 2.

**Model:** DeepSeek V3.2 (open-weight). DeepSeek's first-party `deepseek-chat` alias moved to V4 (`deepseek-v4-flash`) in May 2026 — silently, since it's an alias. V3.2 is now accessed via **OpenRouter** (`deepseek/deepseek-v3.2`), with the inference backend pinned to **AtlasCloud FP8** (chosen 2026-05-19 after a cross-provider probe: 16/16 clean parses, no NOT_ENOUGH_BALANCE-style failures, and ~30% cheaper end-to-end than Friendli native on our input-heavy workload). Model + provider pinned explicitly in `src/shared/llm.py` (the `openrouter` provider entry); `get_extra_body()` emits the `provider.only=["atlas-cloud/fp8"]` request body. Friendli native is kept as a drift-check fallback if FP8 quant ever shows headline accuracy drift vs native. Lesson: never depend on moving aliases.

**`tool_choice="required"` is NOT supported by AtlasCloud FP8** (returns 404 from OpenRouter). The pipeline's explorer + definer use `tool_choice="auto"` instead — the system prompt + tool design steer the model to a tool call reliably, and the loop handles a no-tool-call step gracefully. Other providers (Friendli, OpenAI direct) accept `"required"`; if switching back, adjust accordingly.

**Why not V4-flash:** probed 2026-05-18. The non-thinking V4-flash slug truncates ~27% of CoT outputs at 16K tokens and the `extra_body={"thinking":{"type":"disabled"}}` flag is stochastically honored. Not a stable substrate. Pivoted back to V3.2.

## Directory Structure

```
arc_official_repo/          # Official ARC AGI 1 repo - contains training/evaluation task sets
arc_sample_tooling_2023/    # Reference tooling from 2023 (outdated)
architecture/               # Architecture diagrams
archive/                    # Deprecated code + tarballed pre-DB pipeline outputs (mirrors original tree)
```

### `src/` — Implementation

**`shared/`** — Common utilities used by all approaches:
- `types.py` — `Task`, `Pair`, `Grid` types for ARC data
- `loader.py` — `load_task()`, `get_task_ids()` for loading from arc_official_repo
- `llm.py` — OpenAI-compatible client factory (`create_client`, `create_async_client`)
- `code_exec.py` — Subprocess-isolated execution of generated `transform()` code
- `formatting.py` — `grid_to_ascii()` for rendering grids

**`baseline/`** — No-CoT few-shot baseline. Single LLM call per test pair: system prompt + training examples → JSON grid output. `run.py` has `evaluate()` with `--workers` for parallel execution.

**`CoT/`** — Chain-of-thought baseline. Same structure as baseline but with CoT reasoning before output.

**`arc_base/`** — Faithful port of the ARC Prize "Base LLM" harness: their verbatim system prompt template + `json.dumps` grid format (`--Example {i}--` / `INPUT:` / `OUTPUT:`) + **2 attempts per test pair, same context, temp 0.0, max_tokens 4024** (per the upstream `models.yml`). Scored pass@2 = "any attempt correct" per pair. Reference baseline for comparison against the leaderboard's published harness; not the thesis's primary approach.

**`pipeline/`** — Two-stage agentic pipeline (primary approach):
- `agents/pattern_explorer/` — Stage 1: N parallel explorer agents analyze training examples via iterative tool-use loop (`think`, `note_pattern`), producing pattern documents with synthesis.
- `agents/transformation_definer/` — Stage 2: M parallel definer agents consume compiled explorer findings, synthesize a `transform(grid)` Python function via `think` + `define_transformation` tools. Includes repair loop on execution errors. M>1 enables pass@k experiments (`--num-definers`).
- `selection.py` — pass@k selection logic: dedup candidates by predicted test grids, rank by training-set exact-match score (tie-broken by `agent_idx`), take top-k. Reports unbiased pass@k (Codex estimator) + naive pass@k + pass@M ceiling.
- Each agent has `core.py` (loop), `tools.py` (definitions), `types.py` (result dataclasses), `context/` (prompts + message rendering).
- `run.py` — CLI orchestrator with `--resume`, `--quiet`, `--num-definers`, `--from-explorers <run_id>` (definer-only re-run reusing a prior run's explorer findings), semaphore-based concurrency.
- `io.py` — Serialization helpers. `render.py` — Markdown rendering CLI.
- **On-disk layout:** `output/pipeline/{run_id}/{task_id}/{pattern_explorer,transformation_definer}_{idx}.json`, plus `logs/` and `summary.json` per run. `run_id` is a UUID v7 (same value as `evals.run_id` in the DB).

**`database/`** — Postgres tracking layer (canonical source of truth; disk writes are belt-and-suspenders):
- `migrations/` — SQL migrations applied by `migrate.py`. Schema: `evals` (one row per run), `explorers` and `definers` (one row per agent invocation, FK to evals).
- `client.py` — `EvalClient` with `register_run`, `record_explorer`, `record_definer`, `finalize_pipeline_run`, `record_baseline_run`. Best-effort: DB failures log to stderr and don't crash the run. One reconnect-and-retry on `OperationalError`.
- `metrics.py` — `compute_pipeline_metrics` (pass@1, pass@2, pass@M, plus unbiased Codex estimator via `pipeline.selection`; top-level `accuracy` = pass@2) and `compute_baseline_metrics` (single-attempt file-level, used by baseline/CoT/arc_base) aggregations.
- `ids.py` — UUID v7 generator (RFC 9562). Stdlib gains this in Python 3.14.
- `consolidate.py` — one-time merge of partial pre-DB runs into canonical UUID-keyed rows. Kept as documentation; not for regular use.
- `backfill.py`, `backfill_baselines.py` — recovery tools for re-importing tarballed pre-DB JSON output. Kept for that purpose only.
- Local Postgres runs via `compose.yaml` (port 5433, db `arc_evaluation`).

**`metrics/`** — Frozen snapshot of pre-DB benchmark results:
- `{baseline,CoT,pipeline}_{train,eval}_metrics[_full].json` — accuracy + token usage per approach/split, generated before the DB existed.
- `compute_costs.py` — applies DeepSeek list pricing (PRICING dict at top of file) to token counts and writes `cost_per_task.json`.
- New runs no longer write to this folder; `finalize_pipeline_run` / `record_baseline_run` write aggregate metrics to `evals.data` (JSONB) instead.

**Deprecated (moved to `archive/`):** `arc_agent/`, `sample_agent_episodes/`, `test_definer.py`, `testing_models.py`, top-level `*_context.txt` files, `metrics/generate_pipeline_metrics.py`. Old pipeline output tarballs in `archive/src/output/`.

### Concurrency model

- **Baseline / CoT** use `--workers` (ThreadPoolExecutor): 1 worker = 1 task = 1 in-flight API call.
- **Pipeline** uses `--max-concurrent-tasks` (asyncio.Semaphore). Each task fans out to `num_explorers + 1` concurrent agents, so in-flight API calls ≈ `max_concurrent_tasks × (num_explorers + 1)`.
- Practical ceiling on DeepSeek API: ~50-100 concurrent requests before throughput plateaus and the httpx default 100-connection pool saturates.

### Observed Run Times (400 tasks, DeepSeek V3.2)

- **Baseline / CoT** at `--workers=8`: ~5 min each.
- **Pipeline (full)** at `--max-concurrent-tasks=8`, `--num-explorers=2`: ~6-7 hours (~1 task/min throughput) — dominated by explorer phase.
- **Pipeline (definer-only re-run)** with `--from-explorers <run_id>`, M=5 at `--max-concurrent-tasks=20` on OpenRouter/Friendli: ~40-50 min. Explorer cost is zero (reused); definers fan out concurrently.
- **Pipeline (full e2e on V3.2/AtlasCloud FP8)**, N=2/M=5 at `--max-concurrent-tasks=20`: ~70-90 min projected, ~$45-50 in tokens (probe rate: ~400K tokens/task, prompt-heavy 9:1).

## ARC Task Data Format

Tasks in `arc_official_repo/data/{training,evaluation}/` (400 tasks each):
- JSON format with `train` (demonstration pairs) and `test` (evaluation pairs)
- Each pair has `input` and `output` grids
- Grid: 2D array of integers 0-9 (colors), size 1x1 to 30x30
- 3 attempts allowed per test input; must produce exact match

## Current Benchmark Results

All numbers are preliminary single-run point estimates (`temperature=0`, which is not fully deterministic on this stack). Proper runs with confidence intervals are pending.

**Original baselines** (April 2026, first-party DeepSeek V3.2, single-attempt — **no longer reproducible** since the `deepseek-chat` alias moved to V4):

| Approach | Train acc | Eval acc | Eval $/task | Eval total $ |
|---|---|---|---|---|
| Baseline (no-CoT) | 28.8% | 15.5% | $0.0017 | $0.70 |
| CoT | 31.0% | 16.5% | $0.0025 | $1.04 |
| Pipeline (2 explorers + 1 definer, pass@1) | 64.5% | 44.0% | $0.0596 | $23.86 |

**Pass@k / M-definer selection** (May 2026, OpenRouter/Friendli V3.2, eval split, definer-only re-runs reusing April N=2 explorer traces):

| Setup | pass@1 | pass@2 | pass@M | unbiased pass@1 | unbiased pass@2 |
|---|---|---|---|---|---|
| M=1 (control) | 40.6% | — | — | — | — |
| M=5 | 51.0% | 53.5% | 53.75% | 39.65% | 47.1% |

The M=1 control (40.6%) ≈ the April first-party 44%, confirming OpenRouter/Friendli is consistent with the old stack. M=5 vs M=1 isolates the M>1 + train-score-selection effect (explorers held fixed). The naive-vs-unbiased gap (+6.4pp at pass@2) measures selection-vs-random; the naive-vs-ceiling gap (only 0.25pp) shows selection captures ~99.5% of the achievable maximum — the system is **generation-bound, not selection-bound**. Per-task correct-definer counts are strongly bimodal (46.2% have 0/5, 22.8% have 5/5).

**Reference point — ARC Prize leaderboard, DeepSeek V3.2 "Base LLM"**: 57% on ARC-AGI-1 semi-private eval, $0.120/task. Their published harness ([prompt_manager.py](https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/prompts/prompt_manager.py)) does **2 attempts per test pair at temp 0.0, max_tokens 4024**, same context both times (verified against the committed `models.yml` and `main.py`). Pass@2 scoring marks the pair correct if either attempt matches — relies entirely on API non-determinism for diversity between the two. Whether the leaderboard's 57% was actually produced with this harness is uncertain (the leaderboard entry doesn't link to the benchmarking repo for V3.2 the way it does for R1 et al.) — `src/arc_base/` replicates the published methodology as a sanity-check reference.

## Research Topics Under Investigation

**Canonical living plan: see [`RESEARCH_TRAJECTORY.md`](./RESEARCH_TRAJECTORY.md) at the repo root.** That file owns the current research path (architectures, cross-model panel, deferred items, methodology, run log). Update it as work progresses rather than this section.

**Naming for paper artifacts:** the headline directed pipeline cell is called **Pipeline** (internally B7); the agentic-loop architecture is called **Reflective Orchestrator** or **Orchestrator** (internally B8). Internal B-codes (B0, B4, B5, B5b, B6, B7, B7.5, B7.6, B8) persist in code identifiers (prompt files, DB tags) for stability — see the Reference Key in `RESEARCH_TRAJECTORY.md`. Use the human names in all paper-facing surfaces: docs, plots, figure captions, table headers.

Additional informal notes in `Random Resources & Notes.md`.

## Reference Material

- ARC AGI 1: https://github.com/fchollet/ARC-AGI
- ARC AGI 2: https://github.com/arcprize/ARC-AGI-2
- ARC Prize benchmarking harness: https://github.com/arcprize/arc-agi-benchmarking
- ARC Prize leaderboard: https://arcprize.org/leaderboard
- "On the Measure of Intelligence": https://arxiv.org/abs/1911.01547
- High-scoring ARC approach (Berman): https://jeremyberman.substack.com/p/how-i-got-the-highest-score-on-arc-agi-again
- DreamCoder: https://arxiv.org/pdf/2006.08381
- DeepSeek pricing: https://api-docs.deepseek.com/quick_start/pricing
  