"""Streamlit visualizer for cross-approach ARC analysis.

Run from `src/`:
    uv run streamlit run viz/app.py

Compares Baseline / CoT / Pipeline runs from the Postgres DB.
- Cross-tab tab: 2×2×2 (×2 for pipeline metric) success/failure matrix.
- Task drill-down tab: per-task grids — train pairs, test target, baseline +
  CoT predictions, all M pipeline definer predictions + code + reasoning.

Baseline predicted grids require the `predicted` field in `evals.data->results`
(post-patch). Pre-patch baseline runs are still selectable but their grid cells
render as "(not stored)".
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from matplotlib.colors import ListedColormap

# Make src/ importable so we can reuse shared utilities.
SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shared.loader import load_task  # noqa: E402
from CoT.extract_response import extract_response as cot_extract  # noqa: E402
from pipeline.selection import candidate_from_output, select_pass_at_k  # noqa: E402

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kabirmoghe:postgres@localhost:5433/arc_evaluation",
)

# ARC palette (10 colors, 0-9). Matches the official viewer.
ARC_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 sky
    "#870C25",  # 9 maroon
]
ARC_CMAP = ListedColormap(ARC_COLORS)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


@st.cache_resource
def get_conn():
    return psycopg2.connect(DB_URL)


@st.cache_data(ttl=300)
def list_runs() -> pd.DataFrame:
    """All eval rows with display-friendly columns, newest first."""
    sql = """
    SELECT run_id::text, system, dataset,
           (data->>'accuracy')::float AS accuracy,
           COALESCE((data->>'num_tasks')::int, 0) AS num_tasks,
           created_at
    FROM evals
    ORDER BY created_at DESC
    """
    with get_conn().cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["run_id", "system", "dataset", "accuracy", "num_tasks", "created_at"])
    df["label"] = df.apply(
        lambda r: f"{r['system']} | {r['dataset']} | acc={r['accuracy']:.3f} | n={r['num_tasks']} | {r['created_at']:%Y-%m-%d}",
        axis=1,
    )
    return df


@st.cache_data(ttl=300)
def load_baseline_results(run_id: str) -> pd.DataFrame:
    """results list from evals.data; one row per (task_id, test_index)."""
    sql = "SELECT data->'results' AS r FROM evals WHERE run_id = %s"
    with get_conn().cursor() as cur:
        cur.execute(sql, (run_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return pd.DataFrame()
    return pd.DataFrame(row[0])


@st.cache_data(ttl=300)
def load_pipeline_definers(run_id: str) -> pd.DataFrame:
    """One row per (task_id, agent_idx) with definer output."""
    sql = """
    SELECT task_id, agent, output
    FROM definers
    WHERE run_id = %s
    ORDER BY task_id, agent
    """
    with get_conn().cursor() as cur:
        cur.execute(sql, (run_id,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["task_id", "agent", "output"])
    if df.empty:
        return df
    df["agent_idx"] = df["agent"].str.extract(r"(\d+)$").astype(int)
    return df


def _compute_task_pass_at_2(definers_df: pd.DataFrame) -> pd.DataFrame:
    """Per-task pass@2 and pass@M using the canonical pipeline selection logic
    (dedup by prediction grid + rank by train score + tie-break by agent_idx).
    Returns one row per task with the boolean outcomes."""
    if definers_df.empty:
        return pd.DataFrame(columns=["task_id", "pass_at_2", "pass_at_m"])
    rows = []
    for task_id, grp in definers_df.groupby("task_id"):
        cands = [candidate_from_output(int(r["agent_idx"]), r["output"] or {})
                 for _, r in grp.iterrows()]
        p2 = select_pass_at_k(cands, k=2)["pass_at_k"]
        pM = any(c.correct for c in cands if not c.excluded)
        rows.append({"task_id": task_id, "pass_at_2": p2, "pass_at_m": pM})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def render_grid(ax, grid, title=None, target=None):
    """Render a grid onto a matplotlib axis. If `target` provided, overlay an X
    on cells that mismatch."""
    if grid is None:
        ax.text(0.5, 0.5, "(no grid)", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        return
    arr = np.array(grid)
    if arr.ndim != 2:
        ax.text(0.5, 0.5, "(malformed)", ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="gray")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        return
    arr_clipped = np.clip(arr, 0, 9)
    ax.imshow(arr_clipped, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#555555", linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])

    if target is not None:
        tarr = np.array(target)
        if tarr.shape == arr.shape:
            mismatches = arr != tarr
            ys, xs = np.where(mismatches)
            ax.scatter(xs, ys, marker="x", s=40, color="white", linewidths=1.2)

    if title:
        ax.set_title(title, fontsize=9)


def fig_for_grids(grids: list[tuple[str, Optional[list[list[int]]]]],
                  target: Optional[list[list[int]]] = None,
                  cell_in: float = 1.6) -> plt.Figure:
    """Make a row of grids with optional target-overlay X marks on predictions."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(cell_in * n, cell_in + 0.4))
    if n == 1:
        axes = [axes]
    for ax, (title, g) in zip(axes, grids):
        # Overlay target X only for grids whose title implies a prediction
        is_prediction = any(k in title.lower() for k in ("predict", "definer", "baseline", "cot"))
        render_grid(ax, g, title=title, target=target if is_prediction else None)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Per-task aggregation
# --------------------------------------------------------------------------- #


@dataclass
class TaskOutcomes:
    task_id: str
    baseline_correct: Optional[bool]
    cot_correct: Optional[bool]
    pipeline_p2: Optional[bool]
    pipeline_pm: Optional[bool]


def build_outcomes(baseline_df: pd.DataFrame,
                   cot_df: pd.DataFrame,
                   pipeline_pk_df: pd.DataFrame) -> pd.DataFrame:
    """Merge per-task pass/fail across approaches. Per-task = correct on test_index=0
    for baseline/CoT (the common case); pipeline uses pass@2 and pass@M from selection."""

    def by_task(df, col):
        if df.empty:
            return pd.DataFrame(columns=["task_id", col])
        sub = df[df["test_index"] == 0].copy() if "test_index" in df.columns else df.copy()
        return sub[["task_id", "correct"]].rename(columns={"correct": col})

    b = by_task(baseline_df, "baseline_correct")
    c = by_task(cot_df, "cot_correct")
    out = b.merge(c, on="task_id", how="outer")

    if not pipeline_pk_df.empty:
        out = out.merge(pipeline_pk_df, on="task_id", how="outer")
        out = out.rename(columns={"pass_at_2": "pipeline_p2", "pass_at_m": "pipeline_pm"})
    else:
        out["pipeline_p2"] = None
        out["pipeline_pm"] = None

    for col in ["baseline_correct", "cot_correct", "pipeline_p2", "pipeline_pm"]:
        if col not in out.columns:
            out[col] = None
    return out.sort_values("task_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# UI: sidebar
# --------------------------------------------------------------------------- #


st.set_page_config(page_title="ARC analysis", layout="wide", page_icon="🔬")
st.title("ARC cross-approach analyzer")

runs = list_runs()


def run_picker(label: str, system: str, default_split: str = "evaluation"):
    sub = runs[runs["system"] == system]
    if sub.empty:
        st.sidebar.warning(f"No {system} runs in DB.")
        return None
    # Sort preference: default split first, then most recent
    sub = sub.copy()
    sub["is_default_split"] = (sub["dataset"] == default_split).astype(int)
    sub = sub.sort_values(["is_default_split", "created_at"], ascending=[False, False])
    options = sub["run_id"].tolist()
    labels = dict(zip(sub["run_id"], sub["label"]))
    return st.sidebar.selectbox(label, options=options, format_func=lambda r: labels[r], key=f"sel_{system}")


st.sidebar.header("Runs")
baseline_run = run_picker("Baseline", "baseline")
cot_run = run_picker("CoT", "CoT")
pipeline_run = run_picker("Pipeline", "pipeline")

if not pipeline_run:
    st.error("Need at least one pipeline run in the DB to render anything useful.")
    st.stop()

pipeline_meta = runs[runs["run_id"] == pipeline_run].iloc[0]
baseline_meta = runs[runs["run_id"] == baseline_run].iloc[0] if baseline_run else None
cot_meta = runs[runs["run_id"] == cot_run].iloc[0] if cot_run else None

# Split-consistency warning
selected_splits = {pipeline_meta["dataset"]}
if baseline_meta is not None: selected_splits.add(baseline_meta["dataset"])
if cot_meta is not None: selected_splits.add(cot_meta["dataset"])
if len(selected_splits) > 1:
    st.sidebar.warning(f"⚠️ Mixed splits selected: {sorted(selected_splits)}. Comparisons across splits are not meaningful.")
else:
    st.sidebar.caption(f"Split: **{next(iter(selected_splits))}**")

split = pipeline_meta["dataset"]

# --------------------------------------------------------------------------- #
# Data load
# --------------------------------------------------------------------------- #

baseline_df = load_baseline_results(baseline_run) if baseline_run else pd.DataFrame()
cot_df = load_baseline_results(cot_run) if cot_run else pd.DataFrame()
pipeline_df = load_pipeline_definers(pipeline_run)
pipeline_pk = _compute_task_pass_at_2(pipeline_df)
outcomes = build_outcomes(baseline_df, cot_df, pipeline_pk)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_crosstab, tab_task = st.tabs(["🔀 Cross-tab", "🔍 Task drill-down"])

# --------- Cross-tab tab --------------------------------------------------- #

with tab_crosstab:
    st.subheader("Per-task success/failure across approaches")
    st.caption(
        "B = Baseline (test_index=0 correct) · C = CoT (test_index=0 correct) · "
        "P2 = Pipeline pass@2 (top-2 by train score) · PM = Pipeline pass@M (any definer correct)"
    )

    # Build label rows for the cross-tab
    df = outcomes.copy()
    for col in ["baseline_correct", "cot_correct", "pipeline_p2", "pipeline_pm"]:
        df[col] = df[col].astype("boolean")

    def fmt(v):
        if pd.isna(v): return "—"
        return "✅" if v else "❌"

    # Headline counts
    cols = st.columns(4)
    for i, (col_name, label) in enumerate([
        ("baseline_correct", "Baseline ✅"),
        ("cot_correct", "CoT ✅"),
        ("pipeline_p2", "Pipeline pass@2 ✅"),
        ("pipeline_pm", "Pipeline pass@M ✅"),
    ]):
        n_ok = int(df[col_name].sum()) if df[col_name].notna().any() else 0
        n_tot = int(df[col_name].notna().sum())
        cols[i].metric(label, f"{n_ok}/{n_tot}" if n_tot else "—",
                       f"{n_ok/n_tot*100:.1f}%" if n_tot else "")

    st.markdown("---")
    st.markdown("#### Joint counts (B × C × Pipeline)")
    metric_choice = st.radio("Pipeline metric for cross-tab:",
                             ["pass@2 (selection)", "pass@M (ceiling)"],
                             horizontal=True, index=0)
    pcol = "pipeline_p2" if "pass@2" in metric_choice else "pipeline_pm"

    df["b"] = df["baseline_correct"].map(fmt)
    df["c"] = df["cot_correct"].map(fmt)
    df["p"] = df[pcol].map(fmt)
    df["combo"] = "B=" + df["b"] + " · C=" + df["c"] + " · P=" + df["p"]
    counts = df.groupby("combo").size().reset_index(name="n_tasks").sort_values("n_tasks", ascending=False)
    st.dataframe(counts, use_container_width=True, hide_index=True)

    # Key cells
    st.markdown("#### Key regression / win categories")
    cat_defs = [
        ("Pipeline-only wins (B❌ C❌ P✅)",
         (df["baseline_correct"] == False) & (df["cot_correct"] == False) & (df[pcol] == True)),
        ("Pipeline regression (B✅ or C✅, P❌)",
         ((df["baseline_correct"] == True) | (df["cot_correct"] == True)) & (df[pcol] == False)),
        ("All-pass (B✅ C✅ P✅)",
         (df["baseline_correct"] == True) & (df["cot_correct"] == True) & (df[pcol] == True)),
        ("All-fail (B❌ C❌ P❌)",
         (df["baseline_correct"] == False) & (df["cot_correct"] == False) & (df[pcol] == False)),
    ]
    for label, mask in cat_defs:
        cat = df.loc[mask, ["task_id", "b", "c", "p"]].rename(
            columns={"b": "B", "c": "C", "p": "P"})
        with st.expander(f"{label} — {len(cat)} tasks"):
            if cat.empty:
                st.caption("(none)")
            else:
                st.dataframe(cat, use_container_width=True, hide_index=True, height=min(35 * (len(cat) + 1), 400))


# --------- Task drill-down tab -------------------------------------------- #

with tab_task:
    st.subheader("Per-task view")

    # Filter chips: by outcome category
    filter_choice = st.selectbox(
        "Filter task list by category:",
        [
            "All tasks",
            "Pipeline-only wins (B❌ C❌ P_p2 ✅)",
            "Pipeline regression vs baseline (B✅, P_p2 ❌)",
            "Pipeline regression vs CoT (C✅, P_p2 ❌)",
            "All-fail (B❌ C❌ P_p2 ❌)",
            "Selection miss (PM ✅ but P_p2 ❌)",
        ],
    )
    mask = pd.Series(True, index=outcomes.index)
    o = outcomes
    if filter_choice.startswith("Pipeline-only wins"):
        mask = (o["baseline_correct"] == False) & (o["cot_correct"] == False) & (o["pipeline_p2"] == True)
    elif "regression vs baseline" in filter_choice:
        mask = (o["baseline_correct"] == True) & (o["pipeline_p2"] == False)
    elif "regression vs CoT" in filter_choice:
        mask = (o["cot_correct"] == True) & (o["pipeline_p2"] == False)
    elif filter_choice.startswith("All-fail"):
        mask = (o["baseline_correct"] == False) & (o["cot_correct"] == False) & (o["pipeline_p2"] == False)
    elif "Selection miss" in filter_choice:
        mask = (o["pipeline_pm"] == True) & (o["pipeline_p2"] == False)

    filtered_ids = outcomes.loc[mask, "task_id"].tolist()
    st.caption(f"{len(filtered_ids)} task(s) in this category.")
    if not filtered_ids:
        st.info("No tasks match this filter.")
        st.stop()

    task_id = st.selectbox("Task:", filtered_ids)
    row = outcomes[outcomes["task_id"] == task_id].iloc[0]

    # Header chips
    st.markdown(
        f"**{task_id}** &nbsp;|&nbsp; "
        f"Baseline: {'✅' if row['baseline_correct'] is True else ('❌' if row['baseline_correct'] is False else '—')} &nbsp;|&nbsp; "
        f"CoT: {'✅' if row['cot_correct'] is True else ('❌' if row['cot_correct'] is False else '—')} &nbsp;|&nbsp; "
        f"Pipeline pass@2: {'✅' if row['pipeline_p2'] is True else ('❌' if row['pipeline_p2'] is False else '—')} &nbsp;|&nbsp; "
        f"pass@M: {'✅' if row['pipeline_pm'] is True else ('❌' if row['pipeline_pm'] is False else '—')}"
    )

    # Load the task
    try:
        task = load_task(task_id, split)
    except Exception as e:
        st.error(f"Couldn't load task {task_id} from split {split}: {e}")
        st.stop()

    # --- Train pairs --- #
    st.markdown("### Training examples")
    for i, ex in enumerate(task.train):
        st.markdown(f"**Train {i}**")
        fig = fig_for_grids([(f"input", ex.input), (f"output", ex.output)])
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # --- Test pair --- #
    st.markdown("### Test pair (target)")
    test_ex = task.test[0]  # focus on test_index=0 for the cross-approach view
    fig = fig_for_grids([("test input", test_ex.input), ("expected output", test_ex.output)])
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    expected_test = test_ex.output

    # --- Baseline + CoT predictions --- #
    st.markdown("### Baseline vs CoT predictions (test_index=0)")

    def get_single_result(df, task_id):
        if df.empty: return None
        sub = df[df["task_id"] == task_id]
        if sub.empty: return None
        if "test_index" in sub.columns:
            sub = sub[sub["test_index"] == 0]
        if sub.empty: return None
        return sub.iloc[0].to_dict()

    b_res = get_single_result(baseline_df, task_id)
    c_res = get_single_result(cot_df, task_id)

    # Extract baseline predicted grid (post-patch field, may be missing for old runs)
    b_pred = None
    if b_res:
        b_pred = b_res.get("predicted")  # post-patch
        if b_pred is None and not b_res.get("error"):
            # Pre-patch: no predicted grid stored
            b_pred = None

    def _safe_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return str(x) if not isinstance(x, str) else x

    # Extract CoT predicted grid by parsing raw_response
    c_pred = None
    c_raw = _safe_str(c_res.get("raw_response")) if c_res else None
    if c_raw:
        try:
            c_pred = cot_extract(c_raw)
        except Exception:
            c_pred = None

    def _safe_err(res):
        e = res.get("error")
        if e is None or (isinstance(e, float) and pd.isna(e)):
            return None
        return str(e)

    cols = st.columns(2)
    with cols[0]:
        if b_res is None:
            st.caption("Baseline: no record for this task.")
        elif _safe_err(b_res):
            st.error(f"Baseline errored: {_safe_err(b_res)[:200]}")
        elif b_pred is None:
            st.caption("Baseline predicted grid not stored on this run (pre-patch). Re-run baseline to enable.")
        else:
            fig = fig_for_grids([("baseline predicted", b_pred)], target=expected_test)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.caption(f"Result: {'✅ correct' if b_res.get('correct') else '❌ wrong'} · tokens: {int(b_res.get('prompt_tokens',0) or 0):,} prompt / {int(b_res.get('completion_tokens',0) or 0):,} completion")
    with cols[1]:
        if c_res is None:
            st.caption("CoT: no record for this task.")
        elif _safe_err(c_res):
            st.error(f"CoT errored: {_safe_err(c_res)[:200]}")
        elif c_pred is None:
            st.caption("CoT response unparseable.")
        else:
            fig = fig_for_grids([("CoT predicted", c_pred)], target=expected_test)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            st.caption(f"Result: {'✅ correct' if c_res.get('correct') else '❌ wrong'} · tokens: {int(c_res.get('prompt_tokens',0) or 0):,} prompt / {int(c_res.get('completion_tokens',0) or 0):,} completion")

    if c_raw:
        with st.expander("CoT reasoning"):
            try:
                payload = json.loads(c_raw, strict=False)
                st.markdown(payload.get("reasoning", "(no reasoning field)"))
            except Exception:
                st.text(c_raw[:3000])

    # --- Pipeline definers --- #
    st.markdown("### Pipeline definers (M parallel agents)")
    task_defs = pipeline_df[pipeline_df["task_id"] == task_id].sort_values("agent_idx")
    if task_defs.empty:
        st.caption("No definer records for this task in the selected pipeline run.")
    else:
        for _, drow in task_defs.iterrows():
            out = drow["output"] or {}
            tc = out.get("train_num_correct", 0)
            tt = out.get("train_num_total", 0)
            test_results = out.get("test_results", [])
            tr0 = test_results[0] if test_results else {}
            pred = tr0.get("predicted_output")
            correct = bool(tr0.get("correct", False))
            err = out.get("final_error")
            train_score = (tc / tt) if tt else 0.0
            header = (
                f"**Definer {drow['agent_idx']}** &nbsp;|&nbsp; "
                f"train: {tc}/{tt} ({train_score*100:.0f}%) &nbsp;|&nbsp; "
                f"test_0: {'✅' if correct else '❌'}"
            )
            if err:
                header += " &nbsp;|&nbsp; ⚠️ error"
            st.markdown(header)
            grids_row = [("predicted output", pred)]
            fig = fig_for_grids(grids_row, target=expected_test)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            with st.expander("Code + reasoning"):
                summary = out.get("transformation_summary") or "(no summary)"
                reasoning = out.get("reasoning") or "(no reasoning)"
                code = out.get("code") or ""
                st.markdown(f"**Summary:** {summary}")
                st.markdown(f"**Reasoning:**\n\n{reasoning}")
                st.code(code, language="python")
                if err:
                    st.error(f"final_error: {err[:500]}")
            st.markdown("---")
