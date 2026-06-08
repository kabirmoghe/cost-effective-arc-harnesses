# Research Trajectory

Living document — the current shape of the thesis work. For deep historical notes (architecture-code log B0→B8), see git history of this file pre-2026-06-01.

---

## Thesis question

How far can a cheap, general-purpose open-source model be pushed on ARC-AGI by harness and agentic-pipeline design — i.e. *without ARC-specific fine-tuning*? The model itself can (and almost certainly will) have undergone general RL post-training — RLHF, agentic-capability RL, reasoning RL. That's modern-OS default and orthogonal to the thesis. What we exclude is **ARC-specialized RL fine-tuning**: training the model directly on ARC training tasks so it learns the benchmark's distribution. Approaches like *the Architects* explicitly do this; their lift mixes architecture and ARC-specific fine-tuning, so they're not a clean comparison against a general-purpose model with a clever harness.

The model-selection axis is *generality of the harness lift across general-purpose open models*, not the identity of any one model.

---

## Headline (locked, V3.2/AtlasCloud FP8, eval split, 400 tasks)

All accuracies reported task-level (a task is correct if any of its test pairs is correct); this matches the natural unit of pass@k for the pipeline-family architectures and makes Baseline → CoT → Pipeline → Orchestrator deltas directly comparable. Per-test-pair accuracy (the `evals.data.accuracy` convention) is ~1.5–2.5pp lower for Baseline/CoT due to the 419-vs-400 test-pair-to-task ratio. **All architectures use the same ASCII-with-coordinate-headers grid format** (`shared/formatting.py::grid_to_ascii`) so the architectural lift is not confounded by prompt-format choice — see `docs/format_invariance_check.json` for the empirical check showing baseline/CoT are within sampling noise of the JSON-format ARC-Prize-harness numbers.

| Architecture | Metric | Accuracy | 95% bootstrap CI | Cost/task | Run id |
|---|---|---|---|---|---|
| Direct (no-CoT) baseline | pass@1 | 15.50% | [12.00%, 19.25%] (±3.63pp) | $0.002 | `019e8e9d-...` |
| CoT baseline | pass@1 | 30.00% | [25.50%, 34.50%] (±4.50pp) | $0.004 | `019e8ecf-...` |
| Pipeline (locked: N=5/t=0.5/M=5 + refinement) | pass@1 | 54.75% | [49.75%, 59.50%] (±4.87pp) | $0.25 | `019e7ec8-...` |
| **Pipeline** (locked) | **pass@2** | **57.50%** | [52.75%, 62.50%] (±4.88pp) | $0.25 | `019e7ec8-...` |
| **Reflective Orchestrator** (locked: M=5) | **pass@2** | **67.25%** | [62.75%, 71.75%] (±4.50pp) | $0.62 | `019e818d-...` |

**Paired Δ table (joint task resampling, B=10K, seed pinned; all on the same 400 eval tasks):**

| Transition | Δ (pp) | 95% paired CI | CI excludes 0 |
|---|---|---|---|
| Baseline → CoT | **+14.50** | [+9.75, +19.25] | ✓ |
| CoT → Pipeline (pass@1) | **+24.75** | [+20.00, +29.50] | ✓ |
| CoT → Pipeline (pass@2) | **+27.50** | [+23.00, +32.00] | ✓ |
| Pipeline pass@1 → pass@2 (selection lift) | +2.75 | [+1.25, +4.50] | ✓ |
| **Pipeline → Reflective Orchestrator (pass@2)** | **+9.75** | [+6.50, +13.25] | ✓ |
| **Baseline → Reflective Orchestrator (pass@2)** | **+51.75** | [+46.50, +57.00] | ✓ |
| Pipeline → Act-only ablation (`think` removed) | −5.75 | [−9.00, −2.50] | ✓ |
| Pipeline → Refinement-off ablation | −0.75 | [−2.00, +0.25] | × (5.0% of reps > 0) |

Paired CIs are materially tighter than unpaired-subtraction would imply (~half-width −1pp for each transition) because the paired bootstrap controls for task-level difficulty as a common confounder. CIs capture **sampling noise** only (would the result change if we'd drawn a different 400 tasks); they do *not* capture run-to-run model non-determinism, which would require multiple reps (deferred). Full panel in `docs/bootstrap_cis.json` (regenerated end-to-end via `src/scripts/bootstrap_cis.py`).

**Headline lift:** the agentic harness takes the cheap, general-purpose V3.2 model from a 15.50% no-scaffold baseline to **67.25%** pass@2 — **+51.75pp end-to-end [+46.50, +57.00]**, +9.75pp over the directed-pipeline baseline it's proposed against. V3.2 has undergone standard general RL post-training (RLHF, agentic-capability RL) but no ARC-specific fine-tuning, which is the relevant scope condition.

Selection captures ~95% of the achievable ceiling for both architectures → the system is **generation-bound**, not selection-bound. Lift is overwhelmingly generation-side (unbiased Codex pass@1: Pipeline 46.40% → Orchestrator 56.21%, +9.81pp).

---

## Architectures

### Direct baseline (no-CoT)
Single LLM call. System prompt + few-shot training examples + test input → JSON-grid output. No reasoning step.

### CoT baseline
Single call with reasoning prelude. Two variants tested:
- **Structured** (default): `{reasoning, output}` JSON schema. Used for the headline.
- **Free-form**: prose reasoning + terminal `{"output": ...}` block. Implemented in `src/CoT/run_freeform.py` for cross-model consistency on models that destabilize structured JSON CoT (notably Kimi K2). Not used for headline.

### Pipeline (locked: N=5/t=0.5/M=5 + refinement)
Two-stage architecture:
- **Phase 1 — explorers:** N=5 parallel `PatternExplorer` agents at temperature 0.5, each iterating `think` + `note_pattern` for up to 10 steps, ending with a synthesis. Produces N PatternDocuments.
- **Phase 2 — definers:** M=5 parallel `TransformationDefiner` agents consume the compiled explorer findings, synthesize a `transform(grid)` Python function via `think` + `define_transformation` tools. Refinement loop: after each define, the definer evaluates on train pairs and gets failure feedback (predicted-vs-expected grids + diff + cell list); up to 3 refinement passes. Exits when train hits 100%, refinement cap reached, or steps exhaust.
- **Selection:** dedup-aware pass@k over the M candidates, ranked by training-set exact-match score (tie-broken by `agent_idx`).

Headline run id: `019e7ec8-b83c-7ee2-96e8-12e3afd28d37`.

### Reflective Orchestrator (locked: M=5)

**Motivation.** The Pipeline definer faces a structural problem: it receives a baked-in batch of explorer findings before it starts and has *no mechanism to revisit exploration*. When the explorers miss the right abstraction (a common failure mode on ~35% of Pipeline-failed tasks, per trace inspection), the definer can correctly diagnose from train-feedback that its current hypothesis is wrong, but can only refine within the same wrong frame — there's no path back to pattern discovery. Pattern discovery is the explorer's specialty; the definer doing it inline produces "analysis paralysis" traces (correctly identifying what's wrong, never committing to a new define).

The Orchestrator's design insight is to make this loop closed-by-construction: give the definer a tool to **spawn fresh, focused exploration mid-loop**, so when its current transformation is failing it can request new abstractions rather than getting stuck refining the wrong one. The guidance the definer passes to the spawned explorers is primarily *negative* — confirmed dead-ends and ruled-out hypotheses — because that's the highest-information signal the definer has after seeing its own attempts fail, and it sharpens the focused explorers' search more than open-ended re-exploration would.

The headline framing: spawn is the *relief valve* for wrong-abstraction failures. The architecture preserves the Pipeline's parallel-decomposition + selection structure but makes the explorer/definer boundary dynamic — exploration is no longer a one-shot upstream phase.

**Architecture.** Single agentic loop replacing the Pipeline's directed Phase 1 + Phase 2. The same Phase 1 explorer cell seeds initial findings, then a single orchestrator loop has five tools available throughout:

- `think` — scratchpad
- `define_transformation(code)` — first commit on the rule
- `submit_refined_transformation(code)` — refined version after seeing train feedback
- `explore_new_patterns(guidance)` — spawn K=2 focused PatternExplorers with negative-leaning guidance; spawned PatternDocuments append to the live exploration result
- `done(reason)` — agent-mediated exit

Three exit paths: structural `train == 1.0`, agent-mediated `done`, fuses (`MAX_ITERATIONS=15`, `MAX_CONSECUTIVE_EXEC_ERRORS=5`, `MAX_SPAWN_CALLS=3`).

Headline run id: `019e818d-2907-7694-8fdd-dae6138a2b1e`. Exit distribution (n=1983 definers): perfect_train 65.8% · max_iterations 25.8% (cost driver) · done 7.3% · no_tool_call 1.1% · consecutive_exec_errors 0.05%.

**Spawn is the dominant mechanism for the orchestrator's lift:** of the 45 tasks where Orchestrator beats Pipeline, 31 (69%) are spawn-induced — the winning definer called `explore_new_patterns`. Of the 40 tasks Orchestrator solves that Baseline + CoT + Pipeline all fail, 30 are spawn-induced (~75%). Canonical example: task `3a301edc` (geometric-vs-arithmetic rule trap, `docs/canonical_3a301edc.md`) — all 5 Orchestrator definers reached 5/5 train; only the spawning one got test correct.

---

## Ablations on the locked Pipeline

Two design choices in the Pipeline are tested by removing them from the locked (N=5/t=0.5/M=5) cell and re-measuring pass@2. Both are *capability removals*, not hyperparameter sweeps — they answer "does this feature materially contribute?" rather than "what's the optimum?".

| Removed capability | Method | pass@2 | Δ vs locked Pipeline (paired CI) | Cost/task | Notes |
|---|---|---|---|---|---|
| **None (locked Pipeline baseline)** | full architecture | **57.50%** [52.75, 62.50] | — | $0.25 | run `019e7ec8-...` |
| **`think` tool** (act-only definer) | removed `think` from Phase 1 + Phase 2 definer tool set; model must reason inside the structured tool fields | 51.75% [46.75, 56.75] | **−5.75pp** [−9.00, −2.50] (CI excludes 0; 0% of reps > 0) | $0.17 (−32%) | dedicated run, same explorer cell; run `019e7ff3-...` |
| **Refinement loop** (Phase 2) | subsampled attempts[0] (Phase 1 only) from the locked run — methodologically equivalent because Phase 1 uses `SYSTEM_PROMPT` (no refinement awareness) and the prompt swap to `SYSTEM_PROMPT_B5B` only happens *after* Phase 1 commits | 56.75% [51.75, 61.50] | **−0.75pp** [−2.00, +0.25] (CI touches 0; 95% of reps ≤ 0) | not separately measured (same call budget) | subsampled, no new run; data in `docs/refinement_ablation.json` |

**Interpretation:**
- The `think` tool contributes ~6pp — comparable to halving the explorer fleet (N=5→N=2). It is a *major* architectural component, not overhead, and the paired CI [−9.00, −2.50] cleanly excludes 0 (0% of bootstrap reps put the ablation above the locked Pipeline). The model's reasoning is materially better when given a scratchpad separate from the structured tool fields.
- Refinement (Phase 2 train-feedback loop) contributes a smaller +0.75pp at pass@2 (paired CI [−0.25pp, +2.00pp] on the Pipeline → Refinement-off transition — touches 0; sign consistent in 95% of bootstrap reps). Pass@1 only +0.25pp (CI [−0.50pp, +1.25pp]). The effect is **directional but not statistically significant** at n=400 + M=5; we can't confidently distinguish it from selection-layer noise. Most of refinement's per-definer benefit is absorbed by the selection layer — refinement only matters when it converts a wrong→right definer, and that's a small fraction of the M=5 ensemble already. The `think` ablation by contrast is clearly load-bearing (paired CI a full ±2.5pp from zero).
- Combined picture: the agentic decomposition + the `think` scratchpad are the load-bearing structural choices; refinement is a small additive boost.

Deferred ablations (no data yet):
- **Direct-grid definer** (B7.5): `define_output(grids)` instead of `define_transformation(code)` — isolates the code-gen vs grid-emission lever within the Pipeline. Not on critical path.
- **Orchestrator spawn-off**: orchestrator with `explore_new_patterns` removed. Would isolate spawn's contribution from the loop-only contribution. Deferred — would need a fresh ~$250 run, and cross-system analysis already shows 75% of unique wins are spawn-induced (strong correlational evidence).

---

## Reference key (internal code labels → human names)

Internal B-codes survive in code identifiers (prompt files, DB tags, schema fields) for stability; the paper uses human names exclusively.

| Code | Human name | What it is |
|---|---|---|
| B0 | Baseline / CoT runs | Single-call no-scaffold cells |
| B4 | Pipeline (initial e2e) | First end-to-end pipeline, N=2/M=5 |
| B5 / B5b | Pipeline with refinement | Train-feedback definer loop; B5b moves refinement instructions from system prompt to inline user message |
| B6 | M × definer-temp sweep | 2D plane sweep on high-leverage axes |
| B7 | **Pipeline (locked)** | Headline directed-pipeline cell: N=5/t=0.5/M=5 + refinement |
| B7.5 | Direct-grid ablation (deferred) | Within Pipeline: `define_output(grids)` instead of `define_transformation(code)` |
| B7.6 | Act-only ablation | Remove `think` tool from definer; 51.75% pass@2, −5.75pp vs Pipeline → think is a major architectural component |
| B8 | **Reflective Orchestrator (locked)** | Single agentic loop with spawn relief valve |

Pre-Pipeline cells (B4/B5/B5b/B6) and the act-only ablation (B7.6) are reported as the lineage in the writeup but are not active research targets.

---

## Cross-model validation

The generality claim: the harness lift transfers across open coders, not just V3.2. Single-provider per model (per `feedback_one_run_per_provider`), aiming for FP8-equivalent quantization where available. Full-400-task panel for V3.2; **matched n=99 subset** for Qwen3 (Pipeline + Orchestrator) because alibaba's per-model rate limits forced a hard stop at 100 distinct tasks (429 storm at conc=20 → throttled at conc=15 → confined). To keep the comparison apples-to-apples, the cross-model panel is computed on the 99 common tasks at **M=3** (Qwen3's available ensemble depth; V3.2's M=5 subsampled to M=3 — same temperature, i.i.d. definers).

### Full-400 panel (V3.2 only)

All values task-level (any test pair correct) for consistency with the headline panel above. **All architectures now use the same ASCII-with-coordinate-headers grid format** (see `docs/format_invariance_check.json` for the JSON→ASCII verification on V3.2 baseline + CoT).

| Model | Provider | Baseline pass@1 | CoT pass@1 | Pipeline pass@2 | Orchestrator pass@2 |
|---|---|---|---|---|---|
| DeepSeek V3.2 | atlas-cloud/fp8 | 15.50% | 30.00% | 57.50% | 67.25% |
| Kimi K2 0905 (JSON-format, test-pair, dropped) | atlas-cloud/fp8 | 11.5% | 8.4% | **dropped** | — |

### Matched n=99 subset (V3.2 vs Qwen3, M=3)

Accuracies and cost per task on the 99 common tasks; full data + paired bootstrap CIs in `docs/cross_model_matched_full.json` and `docs/cross_model_pareto_n99.json`. Paired comparison uses n=99 — the intersection of the 100 tasks covered by each Qwen3 pipeline-family run (each architecture independently dropped one different task: B7 lost one to the rate-limit-storm partial-dir cleanup; B8 lost one to a single mid-loop failure).

| Model | Baseline pass@1 | CoT pass@1 | Pipeline pass@2 | Orchestrator pass@2 |
|---|---|---|---|---|
| DeepSeek V3.2 | 15.15% / 0.15¢ | 32.32% / 0.36¢ | 59.60% / 20.24¢ | **69.70% / 40.01¢** |
| Qwen3-235B | 17.17% / 0.10¢ | 28.28% / 0.50¢ | 48.48% / 12.51¢ | **54.55% / 21.15¢** |

**Same-model lifts (Orchestrator − Pipeline, paired bootstrap on Δ, B=10K, seed pinned):**
- V3.2: **+10.10pp** [+3.03, +17.17] — CI excludes 0
- Qwen3: **+6.06pp** [+0.00, +12.12] — directional, CI touches 0 at n=99

**Cross-model gaps (Qwen3 − V3.2, paired bootstrap):**
- Pipeline: −11.11pp [−18.18, −5.05]
- Orchestrator: −15.15pp [−23.23, −7.07] — gap slightly wider for Orchestrator

**Takeaway:** the Pipeline → Orchestrator lift replicates on a second open-weight model in direction and magnitude. Qwen3 trails V3.2 in absolute level on both architectures and benefits slightly less from the loop (Orchestrator lift is +10pp on V3.2 vs +6pp on Qwen3) — but the architectural progression holds. Headline cross-model claim survives a second model; the absolute-level finding is V3.2-specific.

**Notes:**
- **V3.2 baseline + CoT use the ASCII-format runs (canonical, 2026-06-03):** `019e8e9d-...` (baseline) and `019e8ecf-...` (CoT) — same ASCII-with-coordinate-headers format as the pipeline agents, so the architectural-lift comparison is not format-confounded. JSON-vs-ASCII verification: baseline within −0.5pp, CoT +2.25pp at task level — both within sampling noise. See `docs/format_invariance_check.json` for the per-task swing breakdown.
- **Earlier V3.2 CoT runs (historical):** `019e3e30-...` (May 19, JSON format, fully measured) and `019e85fc-...` (June 2, JSON format, stitched, ~16 telemetry-lost test pairs) are no longer used in the headline panel but remain in the DB and the run-log section for provenance.
- **Kimi K2 dropped:** structured CoT 7.25% / freeform CoT (probed) 9.0% — both *under* its own baseline (10.25%). Targeted probe on 14 baseline-failing tasks: freeform parsed 13/14 cleanly (vs 0/14 for structured), but Kimi got 0/14 right. Kimi can't solve these tasks regardless of emission format. Documented as a panel exclusion: "CoT does not lift Kimi K2 on ARC-AGI; structured JSON destabilizes its emission, but the underlying solve capacity is also absent."
- **Qwen3 rate-limit incident:** alibaba per-model rate limits trigger a 429 storm at conc=20 (429 count climbed 1 → 64 → 118 → 259 in 6 min with throughput collapse). Stopped at 91 cleanly-completed + 4 partial → cleaned partials, resumed at conc=8 to reach n=100 (one task didn't write definers — 99 in DB). Same conc=6 ceiling held for the Orchestrator run.

---

## Key findings (paper claims)

1. **Largest single-axis lift in the project:** Orchestrator vs Pipeline = **+9.75pp pass@2** at +148% cost. Pareto-dominates Pipeline at every M.
2. **System is generation-bound, not selection-bound.** Selection captures ~95% of the achievable ceiling for both architectures. Unbiased pass@1 lift mirrors pass@2 lift (+9.81pp), confirming Orchestrator's gain is generation-side.
3. **Spawn is the breakthrough mechanism.** 75% of Orchestrator's unique-win tasks (40 tasks failed by all of Baseline/CoT/Pipeline) involve mid-loop explorer spawning. The relief-valve hypothesis the architecture was designed around is empirically validated.
4. **The `think` tool is a major architectural component, not overhead.** Removing it drops Pipeline by 5.75pp — comparable to halving the explorer fleet (N=5→N=2).
5. **M-saturation at k=3 for Pipeline** under train-score selection: definers cluster bimodally (all-correct or all-wrong) on this model/temperature, so additional definers add nothing the selector can use. Diversity-restoring interventions (temp sweep, alternative selectors) are the upstream lever.
6. **Cross-model: Pipeline → Orchestrator lift replicates on Qwen3-235B** (matched n=99, M=3): V3.2 lift +10.10pp pass@2 [CI excludes 0]; Qwen3 lift +6.06pp [CI touches 0]. Direction + magnitude consistent across two open-weight models. Qwen3 trails V3.2 in absolute level on both architectures (cross-model gap ~11–15pp), but the architectural progression holds — supports the thesis claim that the harness, not the model, drives the lift.

---

## Future work / open questions

**Active / near-term:**
- **Cross-model panel extension** — current panel uses Qwen3 at n=99 (rate-limit-capped). If alibaba rate limits ease (or a parallel provider lands a comparable Qwen3 backend), extend to full 400 to tighten the Qwen3 Orchestrator−Pipeline CI (currently touches 0 at n=99). Optional third model (Llama-3.3-70B via nebius/fp8 is provider-vetted in `shared/llm.py`) could ground the generality claim further.

**Deferred (with reasons):**
- **Orchestrator early-exit prompt** — 25.8% of definers hit `max_iterations` averaging 4.55 attempts + 2.19 spawns all at train_score=0; they should call `done` earlier. Skipped pre-deadline due to re-run risk vs reward (~$250, +0 to −2pp possible if it kills legitimate late recoveries).
- **Direct-grid definer ablation (B7.5)** — within-pipeline ablation: `define_output(grids)` instead of `define_transformation(code)`. Cleanly isolates the code-gen lever. Not on the critical path.
- **Self-certainty / Borda explorer ranking** — Brown et al. 2025 method. Probe revealed all 12 OpenRouter backends for V3.2 return `logprobs = None`; first-party API is sunset. Revisit if V3.2 gains logprobs or the stack moves to a logprob-supported model. Future-work paragraph in writeup.
- **Noise-floor characterization (C8)** — 3× repeats of the headline cell for run-to-run σ on the headline table. Single-rep noise across ablations suggested ~±2pp; locked-cell variance not directly measured.
- **Selection tie-breaker (cell-accuracy secondary tier)** — ≤10 pass@1 tasks max, 0 at pass@2. Low value vs generation-side work.

---

## Methodology

**Single-axis ablation.** Each architecture in the lineage differs from the predecessor by exactly one deliberate design choice. Grid representation (ASCII vs JSON integer arrays) and shared task framing (color legend, "apply the same rule" language) are centralized in `shared/` so cells differ only in their intended axis.

**Subsampling from largest-N/M run.** Within a fixed (temperature, refinement, model) cell, explorers and definers within a run are independent samples from the same distribution. One (N=5, t=0.5) cell covers all N ∈ {1..5} at that temp via subsampling; one (M=5) cell covers all M ∈ {1..5}. Temperature is **not** subsample-able. This is how the M-sweep + N-sweep Pareto curves are generated without redundant compute.

**Unbiased pass@k (Codex estimator).** For each task with `c` correct samples out of `n`, pass@k = 1 − C(n−c, k)/C(n, k). Isolates generation quality from selection quality. Used in the Orchestrator vs Pipeline comparison to confirm the lift is generation-side.

**Bootstrap confidence intervals.** Per-architecture pass@2 CIs from unpaired bootstrap on the 400 per-task binary outcomes (B=10,000 resamples, seed pinned). Architecture-vs-architecture deltas use the **paired bootstrap**: resample 400 task indices jointly so both arms see the same tasks each bootstrap iteration, then compute Δ = mean(b) − mean(a). Pairing exploits the fact that both architectures were evaluated on the exact same task set — task-level difficulty becomes a controlled-for common factor rather than independent noise. Result: paired CIs are materially tighter than what naive subtraction of the per-arch CIs would imply. Bootstrap captures **sampling noise only** (i.e., "would the result change if we'd drawn a different 400 tasks from the same task population"); it does NOT capture **run-to-run noise** (i.e., "would the same 400 tasks score differently on a re-run because the LLM is non-deterministic at temp=0.5"). Run-to-run noise needs repeated runs to estimate (single-rep noise across ablations suggested ~±2pp informally; locked-cell variance not directly measured — deferred). See `src/scripts/bootstrap_cis.py` and `docs/bootstrap_cis.json`.

**Cost accounting.** Workload is ~87/13 prompt/completion split (loop snowballs prompts across steps; outputs per step stay small). At AtlasCloud FP8 pricing ($0.30/Mp + $0.38/Mc): Pipeline = $0.25/task, Orchestrator = $0.62/task end-to-end.

---

## Cost notes (Atlas vs Friendli, 2026-05-29)

| Provider | Input $/M | Output $/M | Precision |
|---|---|---|---|
| AtlasCloud FP8 | $0.30 | $0.38 | Explicitly FP8 (slug `atlas-cloud/fp8`) |
| Friendli | $0.50 | $1.50 | Undocumented quant — slug is `friendli` (no suffix); higher price suggests BF16, unverified |

End-to-end Friendli ≈ 2.03× AtlasCloud FP8 at our 87/13 ratio. Friendli single-rep drift-check on locked Pipeline came in *lower* than Atlas (55.75% vs 57.75%), within run-to-run noise band but opposite of a quantization-loss story. **Decision:** AtlasCloud FP8 is the headline substrate. Friendli kept available for drift-checks only.

**Prompt caching** (DeepSeek/OpenRouter cache-hit pricing ~10% of regular input) deferred — trace prefixes are identical across consecutive steps within one agent; naive estimate ~$30-40 savings per full Pipeline run. Pin if cost optimization becomes binding.

---

## Sequencing (the path)

1. ✅ Baseline + CoT on V3.2/AtlasCloud (single-call cells)
2. ✅ Pipeline end-to-end (B4) → refinement (B5/B5b) → M×t sweep (B6) → locked Pipeline (B7)
3. ✅ Act-only ablation (B7.6) — establishes `think` as a major architectural component
4. ✅ Reflective Orchestrator (B8) — locked headline, +9.75pp over Pipeline
5. ✅ Cross-model panel — Qwen3 Pipeline + Orchestrator at matched n=99 (M=3); Kimi K2 excluded; V3.2 anchored at full 400. Lift replicates in direction + magnitude.
6. ✅ Writeup figures: all headline figures generated and committed to `docs/figures/` (headline Pareto, Pipeline-vs-Orchestrator M-sweep, N×t×M faceted + surface, cross-model Pareto); per-architecture tables integrated in this doc; presentation slide deck landed in `docs/figures/slides/`. Prose writeup ongoing.
7. ✅ Repo made self-referential (2026-06-07): `.gitignore` rewritten to track docs/figures/results as part of the body of work (previously extension-ignored); `src/output/` raw traces remain DB-canonical and untracked.

---

## Run log (condensed — headline cells only)

| Date | Run ID | Architecture | Config | Metric |
|---|---|---|---|---|
| 2026-05-19 | `019e3df9-...` | Baseline | V3.2/Atlas, eval | 13.75% file / 14.6% test-pair |
| 2026-06-01 | `019e8524-...` | Baseline (re-run, strengthened prompt) | V3.2/Atlas, eval | 14.75% / 15.5% |
| 2026-05-19 | `019e3e30-...` | CoT | V3.2/Atlas, eval | 26.5% / 28.4% |
| 2026-06-02 | `019e85fc-...` | CoT (stitched re-run) | V3.2/Atlas, eval | 26.25% / 27.7% (16 pairs lost telemetry — true ∈ [27.7%, 31.5%]) |
| 2026-05-19 | `019e3e84-...` | Pipeline (B4 initial) | N=2/M=5, eval | pass@1 47.9% / pass@2 50.5% |
| 2026-05-31 | `019e7ec8-...` | **Pipeline (locked)** | N=5/t=0.5/M=5 + refinement, eval | **pass@2 57.50%**, $100.27 |
| 2026-05-31 | `019e7ff3-...` | Act-only ablation (B7.6) | Pipeline minus `think` tool, eval | pass@2 51.75%, $68.26 |
| 2026-06-01 | `019e818d-...` | **Reflective Orchestrator (locked)** | M=5, eval | **pass@2 67.25%**, $247.53 |
| 2026-06-01 | `019e8521-...` | Baseline | Qwen3-235B / alibaba, eval | 12.75% / 14.1% |
| 2026-06-01 | `019e856c-...` | CoT | Qwen3-235B / alibaba, eval | 24.75% / 26.7% |
| 2026-06-01 | `019e853e-...` | Baseline | Kimi K2 / atlas-fp8, eval | 10.25% / 11.5% |
| 2026-06-01 | `019e8559-...` | CoT (Kimi — excluded from panel) | atlas-fp8, eval | 7.25% / 8.4% |
| 2026-06-02 | `019e85ef-...` | Pipeline (Qwen3) | N=5/t=0.5/M=3 + refinement, alibaba, eval | n=99: pass@1 47.47% / pass@2 48.48% |
| 2026-06-02 | `019e8773-...` | Reflective Orchestrator (Qwen3) | M=3, definer-only from `019e85ef-...`, alibaba, eval | n=99: pass@1 52.53% / pass@2 54.55% |
| 2026-06-03 | `019e8e9d-...` | **Baseline (ASCII-format, canonical)** | V3.2/Atlas, eval | 15.50% (task-level) — supersedes `019e8524-...` |
| 2026-06-03 | `019e8ecf-...` | **CoT (ASCII-format, canonical)** | V3.2/Atlas, eval | 30.00% (task-level) — supersedes `019e3e30-...` |
| 2026-06-03 | `019e8edc-...` | Baseline (ASCII-format) | Qwen3-235B / alibaba, eval (full 400) | 14.80% test-pair / 17.17% (n=99 task-level) |
| 2026-06-03 | `019e8f0e-...` | CoT (ASCII-format, n=100 subset) | Qwen3-235B / alibaba, eval | 28.85% test-pair / 28.28% (n=99 task-level) |

**Historical cells** (pre-headline lineage; archived for provenance):
- 2026-04-17 `019e1ac0-...` — original first-party V3.2 Pipeline N=2/M=1
- 2026-05-14 `019e289c-...` — OR/Friendli V3.2 M=1 control
- 2026-05-15 `019e2c5b-...` — arc_base pass@2 = 4.3% (published-harness sanity check)
- 2026-05-29 — Friendli Pipeline drift-check (55.75% pass@2)

---

## Writeup asset map

Everything the paper might reference, by section it likely supports.

### Figures (paper-quality, vector PDFs in `docs/figures/`)
| File | Use |
|---|---|
| `docs/figures/headline_pareto.{pdf,png}` | Main result figure: Baseline → CoT → Pipeline → Orchestrator cost-vs-accuracy progression |
| `docs/figures/m_sweep_pipeline_vs_orchestrator.{pdf,png}` | Pipeline vs Orchestrator M-sweep curves at locked explorer cell |
| `docs/figures/pipeline_n_t_m_faceted.{pdf,png}` | Pipeline N×M curves faceted by explorer temperature t; locked cell starred |
| `docs/figures/pipeline_n_t_m_surface.{pdf,png}` | Pipeline N×t×M sweep as a 3-axis surface; companion view to the faceted figure |
| `docs/figures/cross_model_pareto.{pdf,png}` | Cross-model Pareto (V3.2 vs Qwen3) on matched n=99 subset; all 4 architectures per model |

Generator scripts in `src/figures/` (share `paper.mplstyle` + `_style.COLORS`).

**Presentation deck** (`docs/figures/slides/`, generated by `src/figures/slides/`): talk-oriented figures derived from the same data as the paper figures — ARC task walk-through (`slide02`, canonical `3a301edc`), architecture build-up (`slide03` external-only → `slide08` pipeline-added), candidate distribution (`slide10`), full Pareto (`slide13`), lift waterfall (`slide15`), annotated Pareto (`slide16`). Defer-styled for slides, not the paper body.

### Data files (`docs/`)
| File | Contents |
|---|---|
| `docs/pareto_orchestrator.json` | M-sweep raw data for Pipeline + Orchestrator; pricing assumptions |
| `docs/pareto_pipeline_sweep_v32.json` | Full 45-cell N×t×M surface + per-cell Pareto front (Pipeline) |
| `docs/pareto_all.json` | Combined Pareto export across all architectures |
| `docs/refinement_ablation.json` | Refinement-on vs refinement-off pass@k numbers with methodology note |
| `docs/bootstrap_cis.json` | Per-architecture pass@2 CIs + paired Orchestrator−Pipeline Δ CI (B=10K, seed pinned) |
| `docs/unbiased_passk.json` | Codex unbiased pass@k for Pipeline + Orchestrator at multiple k |
| `docs/cross_system_hard_tasks.json` | 4-way (Baseline/CoT/Pipeline/Orchestrator) per-task win matrix; spawn-induced vs loop-only breakdown |
| `docs/canonical_3a301edc.md` | Side-by-side trace of the canonical "spawn unlocks geometric rule" task |
| `docs/qualitative_analysis.md` | Earlier-stage qualitative notes (still useful for §Related Work / §Discussion) |
| `docs/swing_analysis/pooled_summary.json` + `docs/swing_analysis/per_cell/` | Per-cell B5b refinement swing decomposition (rewrite vs surgical edit) — supports the trace-fidelity methodology footnote |
| `docs/cross_model_matched_full.json` | 4-way matched-subset comparison (V3.2/Qwen3 × Pipeline/Orchestrator) with paired bootstrap CIs |
| `docs/cross_model_pareto_n99.json` | Cross-model Pareto data: per-cell pass@k + cost on matched n=99 (M=3 cap); methodology + pricing |
| `docs/cross_model_matched_b7.json` | Earlier-stage matched-subset comparison restricted to Pipeline (Qwen3 vs V3.2) — superseded by `cross_model_matched_full.json` |
| `docs/format_invariance_check.json` | JSON-vs-ASCII grid-format verification on V3.2 baseline + CoT (single-call architectures); supports the methods footnote that all architectures share the ASCII format and the lift is not format-confounded |

### Analysis scripts (`src/scripts/`)
| Script | Produces |
|---|---|
| `src/scripts/pareto_orchestrator.py` | Computes M-sweep + writes `docs/pareto_orchestrator.json` |
| `src/scripts/pareto_analysis.py` | Computes full N×t×M surface + writes `docs/pareto_pipeline_sweep_v32.json` |
| `src/scripts/export_pareto_all.py` | Combined Pareto export across architectures |
| `src/scripts/unbiased_passk.py` | Codex estimator |
| `src/scripts/refinement_ablation.py` | Refinement subsampling ablation (this session) |
| `src/scripts/bootstrap_cis.py` | Per-arch + paired-Δ bootstrap CIs |
| `src/scripts/cross_system_hard_tasks.py` | 4-way task win matrix |
| `src/scripts/render_canonical_trace.py` | Generates `docs/canonical_3a301edc.md` from DB traces |
| `src/scripts/swing_analysis.py` | B5b refinement swing per-cell analysis |
| `src/scripts/stitch_v32_cot.py` | Stitcher for the paused/resumed V3.2 CoT re-run |
| `src/scripts/cross_model_matched_full.py` | 4-way (V3.2/Qwen3 × Pipeline/Orchestrator) matched-subset paired comparison + bootstrap CIs |
| `src/scripts/cross_model_pareto.py` | Computes cross-model Pareto data + pricing on n=99; writes `docs/cross_model_pareto_n99.json` |
| `src/figures/cross_model_pareto.py` | Renders the cross-model Pareto figure |

### Code (paper-architecturally relevant; canonical implementations)
| Module | What |
|---|---|
| `src/pipeline/` | Pipeline architecture: explorers, definers, refinement loop, selection |
| `src/pipeline/selection.py` | Dedup-aware pass@k selection logic — referenced in methodology |
| `src/orchestrator/` | Reflective Orchestrator: single agentic loop, spawn primitive, 5 tools |
| `src/orchestrator/spawn.py` | Mid-loop focused-explorer spawn implementation |
| `src/baseline/`, `src/CoT/` | Single-call baselines (both structured and `_freeform` variants) |
| `src/shared/llm.py` | Provider config (model + provider pinning, OpenRouter routing) |
| `src/database/` | Eval tracking schema + metrics computation; canonical source of truth for run data |

### Specs / design docs
| File | What |
|---|---|
| `RESEARCH_TRAJECTORY.md` (this file) | Living research plan; canonical claims + run log + asset index |
| `CLAUDE.md` | Project-wide instructions + naming convention pointer |

### Reference material (external)
- ARC-AGI-1 task data: `data/` (training + evaluation splits), vendored from fchollet/ARC-AGI
- ARC Prize leaderboard: <https://arcprize.org/leaderboard> (for the V3.2-published-harness reference point)
- "On the Measure of Intelligence" (Chollet 2019): arxiv:1911.01547 — frames why ARC matters
- Codex unbiased pass@k formula (Chen et al. 2021) — methodology source
- Brown et al. 2025 self-certainty paper (arxiv:2502.18581) — future-work motivation

### What's *missing* from this map (gaps to be aware of)
- **Cross-model panel at full 400** — Qwen3 cells exist only at n=99 (rate-limit-capped); V3.2 cells exist at full 400. The Qwen3 Orchestrator−Pipeline paired Δ CI just touches 0 at n=99 — directional signal is clear but doesn't hit standard significance at the current sample size. Would tighten with more Qwen3 tasks.
- **Direct-grid definer ablation (B7.5)** — not run, no data. Listed in deferred-work section.
- **Noise-floor / C8 reps** — not run. Single-rep noise informally bounded at ~±2pp; not measured directly.
- **Orchestrator spawn-off ablation** — not run. Only correlational evidence from cross-system analysis.
- **External flagship reference point (C10)** — not run. Could add one Claude/GPT data point to the headline Pareto if time allows.
