# Senior Thesis Research: Agentic Architectures for ARC AGI

**Goal:** Meta-review high-performance open architectures on ARC AGI, and use the findings to inform a custom architecture.

**Scope:** Focus on ARC AGI 1; extend to ARC AGI 2 if successful.

**Model:** All approaches use DeepSeek's `deepseek-chat` alias (currently DeepSeek V3.2, open-weight).

## Approaches

- **Baseline** — no-CoT few-shot. One LLM call per test pair.
- **CoT** — few-shot with chain-of-thought reasoning.
- **Pipeline** — two-stage agentic approach: N parallel pattern-explorer agents analyze training examples, then a transformation-definer agent synthesizes a `transform(grid)` Python function (with a repair loop on execution errors).

All runs are tracked in a local Postgres database; per-agent JSON output is also written to disk.

## Current Results

Single-attempt (pass@1), `temperature=0`, DeepSeek V3.2.

| Approach | Train acc | Eval acc | Eval $/task |
| --- | --- | --- | --- |
| Baseline (no-CoT) | 28.8% | 15.5% | $0.0017 |
| CoT | 31.0% | 16.5% | $0.0025 |
| Pipeline (3 explorers + definer) | 64.5% | 44.0% | $0.0596 |

## Directory Structure

```text
arc_official_repo/          # Official ARC AGI 1 repo: training/evaluation task sets
arc_sample_tooling_2023/    # Reference tooling from 2023 (outdated)
architecture/               # Architecture diagrams
archive/                    # Deprecated code and pre-DB pipeline outputs
src/                        # Implementation: shared/, baseline/, CoT/, pipeline/, database/, metrics/
```
