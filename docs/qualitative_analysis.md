# Qualitative failure-mode analysis (B4 eval, V3.2 / AtlasCloud FP8)

Run: pipeline eval `019e3e84-2d28-7445-afb9-113ff0508ad8` (N=2 explorers, M=5 definers, pass@2 = 50.5%).
Method: inspect representative definer outputs in the Streamlit visualizer (`src/viz/app.py`), then for tasks of interest patch the model's actual code against the official task JSON to test fix-locality.

Three tasks bracket the spectrum of failure modes. They are deliberately preserved as anchors for downstream architecture decisions (B5b refinement, B7.5 code-gen-vs-direct-grid, Reflective Orchestrator).

---

## 1. `0607ce86` — wrong abstraction

| Definer | Train | Test |
|---|---|---|
| 0–4 | all 0/3 | all ❌ |

**Task pattern.** Input has 5×6 rectangular blocks of a fixed template (`6 6 6 6 6 6` row + four rows of `8 8 3 3 8 8`) corrupted with stray `3`s both inside and outside the blocks. Output cleans the noise: blocks snap to the canonical template; all non-block cells become `0`.

**Definers' model.** All 5 definers framed the task as "find vertical stripes of fixed width and canonicalize them per row." That is the wrong abstraction at the structural level — the right unit is a 2D block, not a 1D stripe.

**Bug hierarchy.**
- Definer_0 also has a variable-shadowing bug: in the apply step, `pattern = tuple(grid[r][start:start+width])` uses bare `width` (the loop variable from the search phase, which by exit has been reassigned to the largest tried value), instead of the captured `best_width`. Fixing this in isolation does not recover the result.
- The dominant failure mode for all 5 is that the algorithm anchors on the first non-zero column and produces stripe-shaped outputs, not block-shaped outputs. Each definer has a slightly different bug in the canonicalization step; these differ across the 5 but are downstream of the same abstraction error.

**Refinement viability.** Low. Train-feedback refinement can show "your predicted output differs from the expected output by ~178 cells" — but recovering would require the model to read the failure and conclude *the unit of analysis is wrong*, not *I have a bug in stripe canonicalization*. That is pattern-rediscovery, not bug-fixing; it sits in the explorer's specialty, not the definer's. This is the canonical motivating example for the Reflective-Orchestrator design (PLAN.md §"Reflective Orchestrator").

---

## 2. `1e97544e` — wrong fallback on an unseen edge case

| Definer | Train | Test |
|---|---|---|
| 0 | **3/3** | ❌ |
| 1 | 1/3 | ❌ |
| 2 | 1/3 | ❌ |
| 3 | 1/3 | ❌ |
| 4 | 2/3 | ❌ |

**Task pattern.** Fill the 23×23 grid with a cyclic 1–9 sequence. Cells follow a "doubled diagonal" structure: position `(i, i)` and `(i, i-1)` share the same value; values decrement cyclically going left from there and increment going right.

**Definer_0's model.** Correct. The transform:
- Computes `diag_val = ((i + offset) % N) + 1` for each row's diagonal.
- Fills cells left-to-right: read left neighbor, increment cyclically; force the diagonal cell to `diag_val`.

**Bug.** The fallback for the case "col 0 has no left neighbor and the cell is empty":
```python
else:
    # First column has no left neighbor, start from 1
    output[i][j] = 1
```
This case never occurs in any train pair (every train row has some leftward anchor cell). It occurs on two test rows (rows 4 and 5) whose entire prefix is empty. The constant `1` is wrong; the cascade rightward then produces `1 2 3 4 ...` instead of the correct `3 4 5 6 ...`. Result: 9 wrong cells out of 529, all in rows 4–5, cols 0–4.

**Verified fix (5 lines).** Derive col 0 from the diagonal value of the same row, accounting for the doubled-diagonal step:
```python
else:
    if i == 0:
        output[i][j] = diag_val
    else:
        val = diag_val - (i - 1)
        output[i][j] = ((val - 1) % N) + 1
```
With this patch alone: train still 3/3, test now **1/1**. The model's mental model is otherwise complete.

**Refinement viability.** Mixed. *Locally* the bug is exactly the shape refinement is best at — a small contradiction between intended rule and observed output. But: train-feedback refinement would not surface it, because the transform already scores 100% on train. The bug is invisible from train data alone. This is direct evidence that **train-feedback refinement has a fundamental coverage limit**: it can only fix bugs that train pairs exercise. Edge cases not seen in train are out of scope unless we also expose intermediate test predictions to the agent — which would defeat the eval methodology.

---

## 3. `05a7bcf2` — right reasoning, wonky implementation

| Definer | Train | Test |
|---|---|---|
| 0 | 0/3 | ❌ |
| 1 | 0/3 | ❌ |
| 2 | 0/3 | ❌ |
| 3 | 0/3 | ❌ |
| 4 | 1/3 | ❌ |

**Task pattern.** A row or column of `8`s acts as a baseline. Above it: 4-shapes. Below it: 2-noise. The transformation involves several stacked sub-operations: extend the bottom row of each 4-shape downward toward the baseline (as 4s), convert the original 4s above the new extension to 3s, project an 8-shadow of each 4-shape's columns through the baseline downward to the bottom of the grid, preserve the 2-noise except where overridden by the 8-shadow.

**Definers' reasoning.** Broadly correct. All 5 verbal summaries identify the structural elements (8-baseline, 4-shapes, 2-noise) and propose plausible rules (projection, mirroring, extension, overlap precedence). Differences across summaries are mostly word-level synonyms of the same idea.

**Code outcome.** None can execute the geometry on more than one train pair. The bugs are scattered across loop bounds, overlap precedence, and projection direction — each definer gets one or two sub-operations approximately right and breaks the others.

**Refinement viability.** Plausibly helpful on individual implementation bugs (loop-bound off-by-one, wrong overlap precedence) but the surface area is large. This is the strongest motivating example for **B7.5 (code-gen vs direct-grid generation)**: the model's mental model is fine; the translation step to executable Python is what fails. A direct-output-grid architecture would skip that translation entirely.

---

## Cross-cutting pattern

| Task | Reasoning quality | Code quality | Where the failure lives | Primary intervention |
|---|---|---|---|---|
| `0607ce86` | Wrong abstraction | Bugs in wrong abstraction | Exploration / framing | Reflective Orchestrator |
| `1e97544e` | Correct rule | One missing edge case | Spec completeness on unseen data | (out of reach for train-feedback refinement) |
| `05a7bcf2` | Correct rule family | Wonky implementation | Translation: rule → code | B7.5 direct-output-grid |

The "reasoning is decent, code is wonky" intuition is most cleanly true for `05a7bcf2`. For `0607ce86` the reasoning is itself wrong. For `1e97544e` the reasoning is correct but specification is incomplete. Train-feedback refinement (the B5 hypothesis) is well-targeted only when the bug is (a) localized and (b) exercised by the training data — a narrow intersection.

## Implications for downstream architecture choices

- **B5b train-feedback refinement** should not be expected to recover the dominant failure class (wrong-abstraction, `0607ce86`-type). Its value is in the spec-completeness band — and only when train pairs actually surface the bug, which `1e97544e` shows is not guaranteed. The ablation is still worth running for the bugs it does cover, but expectations should be moderated.
- **B7.5 direct-output-grid** is well-motivated by `05a7bcf2`-type failures: correct mental model, broken code-gen translation. The retroactive-trace-reuse variant (replay existing think blocks, swap the final tool call to direct-grid submission) is *not* a clean ablation because the existing think blocks were authored for the code-gen goal — they end at "implement step X in Python" rather than at "the test output should look like…". Run B7.5 as a fresh definer pass with a `submit_transformation(grid)` tool, reusing the B4 explorer traces, so the only varying axis is the output modality.
- **Reflective Orchestrator** is the right intervention for `0607ce86`-type failures where the definer needs explorer-level work mid-trajectory.

---

*Source run for all examples: `019e3e84-2d28-7445-afb9-113ff0508ad8`. Patches verified against `arc_official_repo/data/evaluation/{task_id}.json`. Visualizer used for inspection: `src/viz/app.py`.*
