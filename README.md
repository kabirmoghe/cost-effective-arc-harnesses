# Senior Thesis Research: Agentic Architectures for ARC AGI

**Goal:** Meta-review high-performance open architectures on ARC AGI, and use the findings to inform a custom architecture.

**Scope:** Focus on ARC AGI 1; extend to ARC AGI 2 if successful.

**Model:** DeepSeek V3.2 (open-weight). DeepSeek's first-party `deepseek-chat` alias moved to V4 in May 2026, so V3.2 is now accessed via OpenRouter with the inference backend pinned to the Friendli provider (`deepseek/deepseek-v3.2`).

## Approaches

- **Baseline** — no-CoT few-shot. One LLM call per test pair (single attempt).
- **CoT** — few-shot with chain-of-thought reasoning (single attempt).
- **Pipeline** — two-stage agentic approach: N parallel pattern-explorer agents analyze training examples, then M parallel transformation-definer agents each synthesize a `transform(grid)` Python function (with a repair loop on execution errors). For M>1, candidates are ranked by training-set accuracy and the top-k submitted (pass@k).

All runs are tracked in a local Postgres database; per-agent JSON output is also written to disk.

## Current Results

> Numbers are preliminary single-run point estimates (`temperature=0`, which is not fully deterministic on this stack). Proper runs with confidence intervals are pending.

**Original baselines** (April 2026, first-party DeepSeek V3.2 — no longer reproducible since the alias moved to V4):

| Approach | Train acc | Eval acc |
| --- | --- | --- |
| Baseline (no-CoT) | 28.8% | 15.5% |
| CoT | 31.0% | 16.5% |
| Pipeline (2 explorers + 1 definer, pass@1) | 64.5% | 44.0% |

**Pass@k / M-definer selection** (May 2026, OpenRouter/Friendli V3.2, eval split, explorer phase held fixed by reusing the April traces):

| Setup | pass@1 | pass@2 | pass@M |
| --- | --- | --- | --- |
| M=1 (control) | 40.6% | — | — |
| M=5 | 51.0% | 53.5% | 53.75% |

The M=1 control (40.6%) ≈ the April first-party 44%, confirming the OpenRouter stack is consistent. The jump to M=5 is the value of generating multiple candidates and selecting by training-set accuracy. The system is **generation-bound, not selection-bound** — selection captures ~99.5% of the achievable ceiling, so the lever for further gains is candidate generation, not selection logic.

## Directory Structure

```text
arc_official_repo/          # Official ARC AGI 1 repo: training/evaluation task sets
arc_sample_tooling_2023/    # Reference tooling from 2023 (outdated)
architecture/               # Architecture diagrams
archive/                    # Deprecated code and pre-DB pipeline outputs
src/                        # Implementation: shared/, baseline/, CoT/, pipeline/, database/, metrics/
```
