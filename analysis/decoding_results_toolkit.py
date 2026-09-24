#!/usr/bin/env python3

"""
Reusable building blocks for ad-hoc analyses of mvpa_workflow.py's
decoding_results.csv (raw, one row per decoded TR -- see
utils.mvpa_common.timecourse_decoding / workflows.generate_report.py's
compile_group_decoding). Not part of the general report pipeline -- this is
for one-off, per-project scripts (see analysis/plot_valence_evidence_by_operation.py
for a worked example) that need their own filters/conditions/windows/stats
without editing shared code.

Everything here is a small, composable function rather than one big
configurable pipeline, so a new analysis picks just the pieces it needs:

  loading:      load_decoding_results
  value column: select_evidence_value        (e.g. "self"-evidence, or any
                                               explicit evidence_<category>)
  filtering:    apply_filters                 (exact/membership/regex/callable)
  conditions:   derive_label                  (e.g. task -> "pos"/"neg",
                                               trial_type -> "face"/"place")
  aggregation:  aggregate_by_subject_window,   subject x window_index, one
                average_across_groups          column per condition level;
                                                average_across_groups combines
                                                several such tables with equal
                                                weight each (e.g. face + place)
  baseline:     subtract_baseline             (matched on shared index, e.g.
                                               subject + window_index)
  windows:      bin_by_size, bin_by_edges     (adjustable stat windows --
                                               uniform bin size, or an
                                               explicit list of (start, end)
                                               TR ranges)
  stats:        compare_conditions_by_bin,     one row per bin; STAT_METHODS
                STAT_METHODS                   is swappable/extensible, or
                                                pass your own callable
  plotting:     plot_conditions                one line + SE-across-subjects
                                                band per condition, onto a
                                                given axes
"""

import os

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from workflows.generate_report import subject_paths


# =====================================================
# Loading
# =====================================================

def load_decoding_results(analysis_output_dir: str, desc: str, subjects: list) -> pd.DataFrame:
    """Every listed subject's own decoding_results.csv concatenated into one
    raw (one row per decoded TR) dataframe. Subjects with no file are
    skipped with a printed note, not an error -- a partial group is still
    useful to look at. Raises if none of them have one."""
    frames = []
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if not os.path.exists(p["decoding_raw"]):
            print(f"  (!) {s}: no decoding_results.csv found -- skipping")
            continue
        frames.append(pd.read_csv(p["decoding_raw"], dtype={"subject": str}))
    if not frames:
        raise SystemExit("No decoding_results.csv found for any subject in scope.")
    return pd.concat(frames, ignore_index=True)


# =====================================================
# Value column
# =====================================================

def select_evidence_value(df: pd.DataFrame, value: str = "self", new_col: str = "value") -> pd.DataFrame:
    """Adds `new_col`:
      - value="self" (default): each row's own evidence_<its regressor_label>
        -- "how much did the classifier believe this trial was its own true
        category" -- rows whose regressor_label has no matching evidence_
        column are dropped (count printed).
      - any other string: that literal evidence_<value> column (or `value`
        itself, if it's already spelled "evidence_...").
    """
    df = df.copy()
    if value == "self":
        evidence_cols = [c for c in df.columns if c.startswith("evidence_")]
        categories = [c.replace("evidence_", "") for c in evidence_cols]
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        unknown = ~df["regressor_label"].isin(cat_to_idx)
        if unknown.any():
            print(f"  (!) {int(unknown.sum())} row(s) had a regressor_label with no matching evidence_ column -- dropped")
        df = df[~unknown].copy()
        evidence_matrix = df[evidence_cols].to_numpy()
        row_idx = df["regressor_label"].map(cat_to_idx).to_numpy()
        df[new_col] = evidence_matrix[np.arange(len(df)), row_idx]
    else:
        col = value if value.startswith("evidence_") else f"evidence_{value}"
        if col not in df.columns:
            raise SystemExit(f"select_evidence_value: no column {col!r} in this decoding_results.csv")
        df[new_col] = df[col]
    return df


# =====================================================
# Filtering
# =====================================================

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Generic row filter -- `filters` maps column name -> criterion:
      - a scalar (str/int/...): exact match
      - a list/tuple/set: membership (isin)
      - a callable: applied to the column, kept where truthy
      - a compiled-looking usage: pass {"column": {"regex": pattern}} for a
        substring/regex match (case-insensitive by default)
    Unknown columns raise immediately (a typo'd filter should never silently
    match everything)."""
    mask = pd.Series(True, index=df.index)
    for column, criterion in filters.items():
        if column not in df.columns:
            raise SystemExit(f"apply_filters: no column {column!r} in this dataframe")
        if callable(criterion):
            mask &= df[column].map(criterion).astype(bool)
        elif isinstance(criterion, dict) and "regex" in criterion:
            mask &= df[column].astype(str).str.contains(
                criterion["regex"], case=criterion.get("case", False), na=False, regex=True
            )
        elif isinstance(criterion, (list, tuple, set)):
            mask &= df[column].isin(criterion)
        else:
            mask &= df[column] == criterion
    return df[mask]


# =====================================================
# Condition/label derivation
# =====================================================

def derive_label(df: pd.DataFrame, column: str, mapping: dict, new_col: str, regex: bool = False, case: bool = False, drop_unmatched: bool = True) -> pd.DataFrame:
    """Adds `new_col`, mapping `column`'s values through `mapping`
    ({label: criterion}, checked in dict order -- first match wins):
      - regex=False (default): criterion is an exact value, or a list of
        values (isin).
      - regex=True: criterion is a regex/substring pattern tested against
        `column` (case-insensitive unless case=True) -- e.g. deriving
        "stimulus" from trial_type via {"face": "face", "place": "place"}.
    Rows matching no key are dropped by default (count printed) -- pass
    drop_unmatched=False to keep them with `new_col` set to None instead."""
    labels = pd.Series(pd.array([None] * len(df), dtype="object"), index=df.index)
    for label, criterion in mapping.items():
        still_unmatched = labels.isna()
        if regex:
            mask = df[column].astype(str).str.contains(criterion, case=case, na=False, regex=True)
        else:
            values = criterion if isinstance(criterion, (list, tuple, set)) else [criterion]
            mask = df[column].isin(values)
        labels = labels.mask(mask & still_unmatched, label)

    df = df.copy()
    df[new_col] = labels
    unmatched = df[new_col].isna().sum()
    if unmatched and drop_unmatched:
        print(f"  (!) {unmatched} row(s) matched no {new_col!r} label from {column!r} -- dropped")
        df = df[df[new_col].notna()]
    return df


# =====================================================
# Aggregation
# =====================================================

def aggregate_by_subject_window(df: pd.DataFrame, condition_col: str, value_col: str = "value",
                                 subject_col: str = "subject", window_col: str = "window_index") -> pd.DataFrame:
    """(subject, window_index) x condition_col's levels -- mean of value_col
    across every row sharing that (subject, window, condition). Call this
    once per subgroup you want kept separate (e.g. once per stimulus), or
    once on the whole dataframe for no subgrouping at all."""
    grouped = df.groupby([subject_col, window_col, condition_col])[value_col].mean()
    return grouped.unstack(condition_col)


def average_across_groups(group_tables: dict) -> pd.DataFrame:
    """Equal-weighted average of several aggregate_by_subject_window() tables
    (e.g. one per stimulus) sharing the same index shape -- concat +
    groupby-mean over the index handles a (subject, window) combination
    missing from one table by just averaging whichever tables actually have
    it, rather than producing NaN."""
    tables = list(group_tables.values())
    combined = pd.concat(tables)
    return combined.groupby(level=combined.index.names).mean()


# =====================================================
# Baseline subtraction
# =====================================================

def subtract_baseline(wide_df: pd.DataFrame, baseline_df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """wide_df - baseline_df, matched on their shared index (e.g. subject +
    window_index) via an inner join -- an index value missing from either
    side is simply excluded, not guessed at. `columns` restricts which
    condition columns to subtract (default: whichever are present on both
    sides)."""
    cols = columns or [c for c in wide_df.columns if c in baseline_df.columns]
    aligned_a, aligned_b = wide_df[cols].align(baseline_df[cols], join="inner")
    return aligned_a - aligned_b


# =====================================================
# Stat windows (binning)
# =====================================================

def bin_by_size(wide_df: pd.DataFrame, bin_size: int = 3, window_level: str = "window_index", tr_offset: int = 1) -> tuple:
    """Non-overlapping, equal-size bins (bin 0 = the first `bin_size` TRs,
    bin 1 = the next `bin_size`, ...). Returns (per_subject_bin, bins_meta):
      - per_subject_bin: (subject, bin) x condition columns, mean of
        wide_df's values across each bin's TRs.
      - bins_meta: one row per bin (bin, tr_start, tr_end), 1-indexed TR
        numbers by default (tr_offset=1, matching "TRs 1-3" convention) --
        pass tr_offset=0 to keep window_index's own numbering instead.
    """
    df = wide_df.reset_index()
    df["bin"] = df[window_level] // bin_size
    value_cols = [c for c in df.columns if c not in ("subject", window_level, "bin")]
    per_subject_bin = df.groupby(["subject", "bin"])[value_cols].mean()

    bins_meta = df[["bin"]].drop_duplicates().sort_values("bin").reset_index(drop=True)
    bins_meta["tr_start"] = bins_meta["bin"] * bin_size + tr_offset
    bins_meta["tr_end"] = bins_meta["bin"] * bin_size + bin_size - 1 + tr_offset
    return per_subject_bin, bins_meta


def bin_by_edges(wide_df: pd.DataFrame, edges: list, window_level: str = "window_index", tr_offset: int = 1) -> tuple:
    """Explicit, arbitrary (possibly unequal-size, possibly non-contiguous)
    windows -- `edges` is a list of (tr_start, tr_end) pairs, both inclusive
    and already in the same 1-indexed TR numbering bins_meta reports (i.e.
    window_index = tr - tr_offset). A TR not covered by any window is
    excluded entirely, rather than assigned to some default bin. Same
    return shape as bin_by_size."""
    df = wide_df.reset_index()
    df["bin"] = pd.NA
    for bin_id, (tr_start, tr_end) in enumerate(edges):
        mask = (df[window_level] >= tr_start - tr_offset) & (df[window_level] <= tr_end - tr_offset)
        df.loc[mask, "bin"] = bin_id
    df = df[df["bin"].notna()].copy()
    df["bin"] = df["bin"].astype(int)

    value_cols = [c for c in df.columns if c not in ("subject", window_level, "bin")]
    per_subject_bin = df.groupby(["subject", "bin"])[value_cols].mean()

    bins_meta = pd.DataFrame(
        [{"bin": i, "tr_start": s, "tr_end": e} for i, (s, e) in enumerate(edges)]
    )
    return per_subject_bin, bins_meta


# =====================================================
# Stat methods (pluggable)
# =====================================================

def _paired_diff_summary(a_vals: np.ndarray, b_vals: np.ndarray) -> dict:
    diff = np.asarray(a_vals) - np.asarray(b_vals)
    return {"n": len(diff), "mean_diff": float(np.mean(diff)), "sd_diff": float(np.std(diff, ddof=1))}


def ttest_1samp_diff(a_vals, b_vals, alternative="greater") -> dict:
    """One-sample t-test of the per-subject (a - b) difference against 0 --
    the standard "paired difference score" test: each subject contributes
    exactly one number."""
    summary = _paired_diff_summary(a_vals, b_vals)
    t_stat, p_value = scipy_stats.ttest_1samp(np.asarray(a_vals) - np.asarray(b_vals), popmean=0.0, alternative=alternative)
    summary.update(stat=float(t_stat), p_value=float(p_value), method="ttest_1samp_diff")
    return summary


def ttest_rel(a_vals, b_vals, alternative="greater") -> dict:
    """scipy's own paired t-test -- mathematically identical to
    ttest_1samp_diff, kept as a separate named option for clarity/API
    completeness (and because it's the more familiar name)."""
    summary = _paired_diff_summary(a_vals, b_vals)
    t_stat, p_value = scipy_stats.ttest_rel(a_vals, b_vals, alternative=alternative)
    summary.update(stat=float(t_stat), p_value=float(p_value), method="ttest_rel")
    return summary


def wilcoxon_signed_rank(a_vals, b_vals, alternative="greater") -> dict:
    """Non-parametric alternative to the paired t-test -- use when the
    per-subject differences look non-normal/have outliers."""
    summary = _paired_diff_summary(a_vals, b_vals)
    stat, p_value = scipy_stats.wilcoxon(a_vals, b_vals, alternative=alternative)
    summary.update(stat=float(stat), p_value=float(p_value), method="wilcoxon")
    return summary


STAT_METHODS = {
    "ttest_1samp_diff": ttest_1samp_diff,
    "ttest_rel": ttest_rel,
    "wilcoxon": wilcoxon_signed_rank,
}


def compare_conditions_by_bin(per_subject_bin: pd.DataFrame, cond_a: str, cond_b: str,
                               method="ttest_1samp_diff", alternative: str = "greater",
                               bins_meta: pd.DataFrame = None) -> pd.DataFrame:
    """One row per bin -- `method` is a name in STAT_METHODS (swap the test
    with one line), or your own callable(a_vals, b_vals, alternative) -> dict
    with at least {"stat", "p_value"}. A bin with fewer than 2 subjects
    having both conditions is skipped (can't test a single point).
    `bins_meta` (from bin_by_size/bin_by_edges), if given, is merged in for
    tr_start/tr_end columns."""
    test_fn = STAT_METHODS[method] if isinstance(method, str) else method

    rows = []
    for bin_id, bin_df in per_subject_bin.reset_index().groupby("bin"):
        sub = bin_df[[cond_a, cond_b]].dropna()
        if len(sub) < 2:
            continue
        result = test_fn(sub[cond_a].to_numpy(), sub[cond_b].to_numpy(), alternative)
        result["bin"] = int(bin_id)
        rows.append(result)

    out = pd.DataFrame(rows)
    if bins_meta is not None and not out.empty:
        out = out.merge(bins_meta, on="bin", how="left")
    return out


# =====================================================
# Plotting
# =====================================================

def plot_conditions(ax, wide_df: pd.DataFrame, colors: dict, window_level: str = "window_index",
                     tr_offset: int = 1, title: str = None, ylabel: str = None):
    """One mean +/- SE-across-subjects line per condition in `colors`
    ({condition_label: matplotlib_color}), onto a given axes -- wide_df is
    (subject, window_index)-indexed with one column per condition."""
    for condition, color in colors.items():
        if condition not in wide_df.columns:
            continue
        series = wide_df[condition].dropna()
        if series.empty:
            continue
        by_window = series.groupby(level=window_level)
        mean = by_window.mean()
        n = by_window.count()
        se = by_window.std(ddof=1) / np.sqrt(n)
        tr = mean.index.to_numpy() + tr_offset
        ax.fill_between(tr, mean - se, mean + se, alpha=0.2, color=color)
        ax.plot(tr, mean, color=color, linewidth=1.5, label=condition)
    if title:
        ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("TR")
    ax.legend(fontsize=8)
