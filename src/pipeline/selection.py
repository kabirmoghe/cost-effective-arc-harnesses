"""Select top-k transformations from M definer results for pass@k evaluation.

A "candidate" is one definer's transformation. Selection:
  1. drop candidates that never produced executable code (final_error / no code)
  2. dedup by predicted test-output grids — a submission is a grid, so two
     definers with identical test predictions collapse to one candidate
  3. rank by training-set score (fraction of train pairs the transform solves)
  4. take the top k; pass@k holds if any of them is correct on all test pairs

Two adapters build the common CandidateView: candidate_from_result (live
TransformationResult objects, used by the pipeline runner) and
candidate_from_output (definer `output` JSON, used by DB metrics).

**B5 note:** for definers with multiple refinement attempts, the top-level
TransformationResult fields (`code`, `test_results`, `train_num_*`, etc.) are
populated by the definer driver from `best_attempt()` — the attempt with the
highest train-score (ties broken by later iter). So both adapters get the
best-by-train candidate "for free" via the top-level fields; no explicit
attempts handling needed here. The full attempts history lives in
`result.attempts` / `output["attempts"]` for post-hoc analysis (e.g.
final-vs-best diagnostic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CandidateView:
    agent_idx: int
    train_score: float
    correct: bool  # correct on ALL test pairs (file-level)
    num_test_correct: int
    num_test_total: int
    prediction_key: tuple
    excluded: bool  # never produced executable code


def _grid_key(grid):
    if grid is None:
        return None
    return tuple(tuple(row) for row in grid)


def candidate_from_result(result) -> CandidateView:
    """Build a CandidateView from a live TransformationResult."""
    excluded = result.final_error is not None or not result.code
    pred_key = tuple(
        _grid_key(tr.predicted_output)
        for tr in sorted(result.test_results, key=lambda r: r.test_index)
    )
    return CandidateView(
        agent_idx=result.agent_idx,
        train_score=result.train_score,
        correct=bool(result.correct),
        num_test_correct=result.num_correct,
        num_test_total=len(result.test_results),
        prediction_key=pred_key,
        excluded=excluded,
    )


def candidate_from_output(agent_idx: int, output: dict) -> CandidateView:
    """Build a CandidateView from a definer `output` JSON dict (DB row)."""
    excluded = output.get("final_error") is not None or not output.get("code")
    test_results = output.get("test_results") or []
    pred_key = tuple(
        _grid_key(tr.get("predicted_output"))
        for tr in sorted(test_results, key=lambda r: r.get("test_index", 0))
    )
    train_total = output.get("train_num_total", 0) or 0
    train_correct = output.get("train_num_correct", 0) or 0
    train_score = train_correct / train_total if train_total else 0.0
    return CandidateView(
        agent_idx=agent_idx,
        train_score=train_score,
        correct=bool(output.get("correct")),
        num_test_correct=sum(1 for tr in test_results if tr.get("correct")),
        num_test_total=len(test_results),
        prediction_key=pred_key,
        excluded=excluded,
    )


def _view_dict(c: CandidateView) -> dict[str, Any]:
    return {
        "agent_idx": c.agent_idx,
        "train_score": c.train_score,
        "correct": c.correct,
        "num_test_correct": c.num_test_correct,
        "num_test_total": c.num_test_total,
    }


def select_pass_at_k(candidates: list[CandidateView], k: int) -> dict[str, Any]:
    """Rank, dedup, and take top-k. Returns pass@k plus the ranking detail."""
    pool = [c for c in candidates if not c.excluded]

    # dedup by predicted test outputs; keep the highest-train-score representative
    by_key: dict[tuple, CandidateView] = {}
    for c in pool:
        cur = by_key.get(c.prediction_key)
        if cur is None or (c.train_score, -c.agent_idx) > (cur.train_score, -cur.agent_idx):
            by_key[c.prediction_key] = c
    unique = sorted(by_key.values(), key=lambda c: (-c.train_score, c.agent_idx))

    topk = unique[:k]
    return {
        "k": k,
        "num_candidates": len(candidates),
        "num_unique": len(unique),
        "pass_at_k": any(c.correct for c in topk),
        "selected": [_view_dict(c) for c in topk],
        "ranking": [_view_dict(c) for c in unique],
    }
