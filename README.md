# Agentic Architectures for ARC-AGI

Senior thesis research. **Question:** how far can a cheap, general-purpose
open-weight model be pushed on ARC-AGI by harness and agentic-pipeline design —
*without ARC-specific fine-tuning*? The approach is a meta-review of
high-performance open architectures, used to inform a custom one.

**Scope:** ARC-AGI 1 (400-task evaluation split); extends to ARC-AGI 2 if successful.

**Model:** DeepSeek V3.2 (open-weight), accessed via **OpenRouter** with the
inference backend pinned to **AtlasCloud FP8** (`deepseek/deepseek-v3.2`,
`provider.only=["atlas-cloud/fp8"]`). DeepSeek's first-party `deepseek-chat`
alias silently moved to V4 in May 2026, so the model + provider are pinned
explicitly in `src/shared/llm.py`. Friendli native is kept as a drift-check
fallback. The generality claim is cross-checked on a second open model
(Qwen3-235B).

## Approaches

Each is a single deliberate step up in scaffolding; all share the same ASCII grid
format so the architectural lift isn't confounded by prompt format.

- **Baseline** — no-CoT few-shot. One LLM call per test pair.
- **CoT** — same, with a chain-of-thought reasoning prelude.
- **arc_base** — faithful port of the ARC Prize "Base LLM" harness (2 attempts,
  pass@2) as a leaderboard-comparable reference point.
- **Pipeline** (primary) — two stages: N parallel pattern-explorer agents analyze
  the training examples, then M parallel transformation-definer agents each
  synthesize a `transform(grid)` Python function (with a train-feedback repair
  loop). For M>1, candidates are ranked by training-set accuracy and the top-k
  submitted (pass@k).
- **Reflective Orchestrator** — replaces the Pipeline's fixed explore→define
  split with a single agentic loop that can **spawn fresh focused exploration
  mid-loop** when its current transformation is failing. This relief valve for
  wrong-abstraction failures is the largest single-axis lift in the project.

> Internal code labels (B7 = Pipeline, B8 = Reflective Orchestrator, etc.) persist
> in identifiers for stability; paper-facing surfaces use the human names. See the
> Reference Key in [`RESEARCH_TRAJECTORY.md`](./RESEARCH_TRAJECTORY.md).

## Headline results

DeepSeek V3.2 / AtlasCloud FP8, eval split, 400 tasks, task-level accuracy. Full
panel with bootstrap confidence intervals, paired deltas, ablations, and the
cross-model comparison is in [`RESEARCH_TRAJECTORY.md`](./RESEARCH_TRAJECTORY.md).

| Architecture | Metric | Accuracy | Cost/task |
|---|---|---|---|
| Baseline (no-CoT) | pass@1 | 15.50% | $0.002 |
| CoT | pass@1 | 30.00% | $0.004 |
| Pipeline (N=5 / t=0.5 / M=5 + refinement) | pass@2 | 57.50% | $0.25 |
| **Reflective Orchestrator** (M=5) | **pass@2** | **67.25%** | $0.62 |

The agentic harness takes the same general-purpose model from 15.50% to **67.25%**
(+51.75pp end-to-end), with the Orchestrator +9.75pp over the directed Pipeline.
The system is **generation-bound, not selection-bound** (selection captures ~95%
of the achievable ceiling), and ~75% of the Orchestrator's unique wins come from
mid-loop explorer spawning. The lift replicates in direction and magnitude on
Qwen3-235B.

**Reference point** — ARC Prize leaderboard, DeepSeek V3.2 "Base LLM": 57% on the
semi-private eval at $0.120/task.

> Accuracies are single-run point estimates (`temperature` 0–0.5, not fully
> deterministic on this stack). Bootstrap CIs capture sampling noise; run-to-run
> noise across reps is not yet characterized.

## Getting started

See **[`docs/SETUP.md`](./docs/SETUP.md)** for the full setup and the exact run
command for each harness. In brief, from `src/`:

```bash
uv sync                                   # install
cp .env.example .env                      # set OPENROUTER_API_KEY
docker compose up -d && python -m database.migrate   # start + migrate the tracking DB
python -m pipeline.run --split evaluation --provider openrouter \
  --num-explorers 5 --explorer-temperature 0.5 --num-definers 5 \
  --definer-refinement --max-concurrent-tasks 20
```

All runs are recorded in a local Postgres database (canonical source of truth);
per-agent JSON is also written to disk. Inspect or export results via
`python -m database.export` — see [`src/database/exports/`](./src/database/exports/).

## Repository layout

```text
arc_official_repo/   # Official ARC-AGI 1 task sets (training + evaluation)
architecture/        # Architecture diagrams
archive/             # Deprecated code + pre-DB output (provenance only)
docs/                # Paper figures, results JSON, analysis, SETUP.md
src/                 # Implementation (see below)
RESEARCH_TRAJECTORY.md   # Canonical living plan: claims, CIs, run log, asset map
```

Inside `src/`:

```text
shared/        # Task types, loader, LLM client/provider config, grid formatting
baseline/ CoT/ arc_base/    # Single-call harnesses
pipeline/      # N explorers → M definers, refinement, pass@k selection
orchestrator/  # Reflective Orchestrator: single agentic loop + mid-loop spawn
database/      # Postgres tracking layer: client, migrations, metrics, exports
figures/ scripts/   # Paper-figure generators and analysis scripts
```

## Reference material

- ARC-AGI-1: <https://github.com/fchollet/ARC-AGI>
- ARC Prize benchmarking harness: <https://github.com/arcprize/arc-agi-benchmarking>
- ARC Prize leaderboard: <https://arcprize.org/leaderboard>
- "On the Measure of Intelligence" (Chollet 2019): <https://arxiv.org/abs/1911.01547>
