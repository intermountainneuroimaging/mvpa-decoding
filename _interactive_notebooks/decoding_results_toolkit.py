#!/usr/bin/env python3

"""
Reusable building blocks for ad-hoc analyses of mvpa_workflow.py's
decoding_results.csv (raw, one row per decoded TR -- see
utils.mvpa_common.timecourse_decoding / workflows.generate_report.py's
compile_group_decoding). Not part of the general report pipeline -- this is
for one-off, per-project scripts (see _interactive_notebooks/plot_valence_evidence_by_operation.py
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
  trial balance: balance_preceding_category    equal say for trials that follow a same-
                                                category vs different/no-category trial
                                                per subject x condition cell, by random
                                                subsampling or by weights (all trials);
                weighted_group_mean            the weighted means that go with the weights
  baseline:     subtract_baseline             (matched on shared index, e.g.
                                               subject + window_index)
  windows:      bin_by_size, bin_by_edges     (adjustable stat windows --
                                               uniform bin size, or an
                                               explicit list of (start, end)
                                               TR ranges)
  stats:        compare_conditions_by_bin,     one row per bin; STAT_METHODS
                STAT_METHODS, fdr_correct      is swappable/extensible, or
                                                pass your own callable; each
                                                comparison's p-values also get
                                                an FDR-adjusted p_fdr column
  reliability:  split_half_group_difference    how reproducible a group-level mean
                                                difference between two conditions
                                                is when every participant's own
                                                trials are randomly halved
  plotting:     plot_conditions                one line + SE-across-subjects
                                                band per condition, onto a
                                                given axes
"""

import os
import warnings

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

def weighted_group_mean(df: pd.DataFrame, by, value_col: str = "value", weight_col: str = None) -> pd.Series:
    """Mean of value_col within each group of `by` (a column name or list), named value_col.
    With `weight_col` it is the weighted mean sum(w * v) / sum(w); rows with a missing value
    count for neither sum, as in a plain groupby mean. weight_col=None is exactly
    df.groupby(by)[value_col].mean()."""
    if weight_col is None:
        return df.groupby(by)[value_col].mean()
    valid = df[value_col].notna()
    weights = df[weight_col].where(valid)
    keys = [df[b] for b in ([by] if isinstance(by, str) else list(by))]
    numerator = (df[value_col] * df[weight_col]).where(valid).groupby(keys).sum()
    denominator = weights.groupby(keys).sum()
    return (numerator / denominator.where(denominator > 0)).rename(value_col)


def aggregate_by_subject_window(df: pd.DataFrame, condition_col: str, value_col: str = "value",
                                 subject_col: str = "subject", window_col: str = "window_index",
                                 weight_col: str = None) -> pd.DataFrame:
    """(subject, window_index) x condition_col's levels -- mean of value_col
    across every row sharing that (subject, window, condition), weighted by
    `weight_col` if given (see weighted_group_mean). Call this
    once per subgroup you want kept separate (e.g. once per stimulus), or
    once on the whole dataframe for no subgrouping at all."""
    grouped = weighted_group_mean(df, [subject_col, window_col, condition_col], value_col, weight_col)
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


def max_reliable_window_index(df: pd.DataFrame, group_cols, min_fraction: float = 0.9,
                               subject_col: str = "subject", window_col: str = "window_index") -> int:
    """Latest window_index at which every combination of `group_cols` (e.g.
    stimulus x operation x valence) still has at least `min_fraction` of the
    dataset's own total subject count contributing. A trial's real end time
    varies (jittered ITI to the next trial's own anchor), so only a
    shrinking handful of unusually long trials still have data at the far
    tail of the window -- averaging just 1-2 subjects there produces a
    noisy, unrepresentative single-point spike (and an undefined SE, since
    a sample std needs at least 2 points). Filtering a dataframe to
    window_index <= this value keeps every plotted/tested point supported
    by close to the full group, instead of quietly plotting a handful of
    outlier trials as if they were a real group average. Raises if no
    window_index meets the threshold at all."""
    total_subjects = df[subject_col].nunique()
    min_n = min_fraction * total_subjects
    group_cols = list(group_cols)
    counts = df.groupby(group_cols + [window_col])[subject_col].nunique()
    # reindex to the full cross-product of every observed group_cols
    # combination x every observed window_index, so a combination that's
    # completely absent at a given window (not just sparse -- e.g. every
    # one of its trials ended before that point) is filled in as n=0 and
    # correctly disqualifies that window, rather than silently vanishing
    # from the groupby result and never being checked at all
    observed_groups = df[group_cols].drop_duplicates()
    observed_windows = pd.DataFrame({window_col: sorted(df[window_col].unique())})
    full_index = pd.MultiIndex.from_frame(observed_groups.merge(observed_windows, how="cross"))
    counts = counts.reindex(full_index, fill_value=0)
    worst_per_window = counts.groupby(level=window_col).min()
    reliable = worst_per_window[worst_per_window >= min_n]
    if reliable.empty:
        raise ValueError(
            f"max_reliable_window_index: no window_index has >= {min_fraction:.0%} of "
            f"{total_subjects} subjects across every {list(group_cols)} combination"
        )
    return int(reliable.index.max())


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


FDR_METHODS = ("bh", "by")


def fdr_correct(p_values, method: str = "bh") -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values), same order and
    length as `p_values`. "bh" is Benjamini-Hochberg (valid for independent
    or positively-dependent tests); "by" is the more conservative
    Benjamini-Yekutieli (valid under any dependence). NaN p-values pass
    through as NaN and don't count towards the number of tests. Thresholding
    the result at alpha controls the expected fraction of false discoveries
    among the bins called significant at alpha."""
    if method not in FDR_METHODS:
        raise ValueError(f"unknown FDR method {method!r} -- expected one of {FDR_METHODS}")
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = scipy_stats.false_discovery_control(p[ok], method=method)
    return out


def compare_conditions_by_bin(per_subject_bin: pd.DataFrame, cond_a: str, cond_b: str,
                               method="ttest_1samp_diff", alternative: str = "greater",
                               bins_meta: pd.DataFrame = None, fdr: str = "bh") -> pd.DataFrame:
    """One row per bin -- `method` is a name in STAT_METHODS (swap the test
    with one line), or your own callable(a_vals, b_vals, alternative) -> dict
    with at least {"stat", "p_value"}. A bin with fewer than 2 subjects
    having both conditions is skipped (can't test a single point).
    `bins_meta` (from bin_by_size/bin_by_edges), if given, is merged in for
    tr_start/tr_end columns.

    Every bin of one call is one *family* of tests (one condition contrast
    across the time windows): `p_value` is the raw per-bin p, and `p_fdr`
    (inserted right after it) is that family's FDR-adjusted p -- see
    fdr_correct. Call this once per contrast (e.g. once per operation, or
    per valence x operation) so each contrast is corrected on its own.
    Pass fdr=None for raw p-values only (no p_fdr column)."""
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
    if fdr and not out.empty:
        out.insert(out.columns.get_loc("p_value") + 1, "p_fdr", fdr_correct(out["p_value"], fdr))
    if bins_meta is not None and not out.empty:
        out = out.merge(bins_meta, on="bin", how="left")
    return out


# =====================================================
# Split-half reliability of a group-level finding
# =====================================================

def _split_masks(n_units: int, n_splits: int, rng) -> tuple:
    """(mask_a, mask_b), boolean (n_splits, n_units): every row is an
    independent random split of the units into two equal halves (with an odd
    count, one random unit is left out of both)."""
    half = n_units // 2
    ranks = rng.random((n_splits, n_units)).argsort(axis=1).argsort(axis=1)
    return ranks < half, (ranks >= half) & (ranks < 2 * half)


def _half_means(matrix: np.ndarray, mask: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """NaN-aware mean of `matrix` (units x timepoints) over each row of
    `mask` (n_splits x units) -> (n_splits, timepoints); weighted by `weights`
    (one per unit) if given."""
    valid = np.isfinite(matrix)
    selected = mask.astype(float) if weights is None else mask.astype(float) * weights[None, :]
    sums = selected @ np.where(valid, matrix, 0.0)
    counts = selected @ valid.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def split_half_group_difference(df: pd.DataFrame, condition_col: str, cond_a: str, cond_b: str,
                                 bin_size: int = 3, value_col: str = "value", subject_col: str = "subject",
                                 time_col: str = "window_index", trial_cols=("boldfile", "trial_index"),
                                 tr_offset: int = 0, n_splits: int = 1000, seed: int = 0,
                                 min_trials: int = 2, weight_col: str = None) -> pd.DataFrame:
    """If the same participants were run again, how reliable is the group-level
    finding "condition a differs from condition b" in each time bin?

    `df` is long-format, one row per (subject, trial_cols..., timepoint), and
    should already be the data subset for this contrast -- e.g. only maintain
    trials for a maintain pos-vs-neg comparison; only rows whose
    `condition_col` is cond_a or cond_b are used. It mirrors the statistics'
    own pipeline exactly: a subject's value for a condition and bin is the mean
    over the bin's timepoints of the mean over trials at each timepoint (as
    aggregate_by_subject_window + bin_by_size build it), and the group
    difference is the mean over subjects of (a - b). `full_mean_diff` is
    therefore the same number compare_conditions_by_bin reports as mean_diff.

    Then, n_splits times: every subject's trials in each of the two conditions
    are randomly cut into two equal halves (separately per condition, an odd
    trial left out), the group mean difference is computed in each half, and
    the halves are compared. With `weight_col`, every trial carries a weight
    (one per trial, constant over its timepoints) and all means -- the full data
    and each half -- are weighted means, as in aggregate_by_subject_window(...,
    weight_col=...); the weights are the full-data ones, so a half's mean is a
    weighted mean of the trials that landed in it. Subjects need at least `min_trials` (>= 2) trials
    in BOTH conditions to take part; the rest are left out of every split and
    of full_mean_diff (n_subjects counts those who took part).

    Returns one row per bin: bin, tr_start/tr_end (as bin_by_size), n_subjects,
    full_mean_diff, half_ci_low/half_ci_high (2.5/97.5th percentile of the
    half-split group differences, A and B halves pooled), p_half_a_gt_b (share
    of half-splits showing a > b), p_same_direction (share showing the
    full-data direction), p_halves_agree (share of splits where half A and
    half B agree on direction), and n_splits (splits with both halves
    estimable). Because the random halves are exchangeable, p_halves_agree =
    2 * p_same_direction - 1 exactly -- one piece of information, shown two
    ways.

    How to read them -- carefully. The two halves partition the same data, so
    both inherit its realized noise and their estimates sit symmetrically
    around full_mean_diff. The probabilities are therefore a transformation of
    how many trial-to-trial standard errors full_mean_diff sits from zero,
    not independent evidence that an effect is real: a bin with NO true effect
    still averages p_same_direction ~0.75 (never below 0.5), and its
    p_halves_agree is spread roughly uniformly over 0-1, so only values near 1
    mean much (p_halves_agree >= 0.95 is about as unusual as p = 0.05). Each
    half also holds only half of every subject's trials, so a half-sized study
    replicates less often than the full one would; and the same participants
    sit in both halves, so this says nothing about new participants."""
    trial_cols = list(trial_cols)
    min_trials = max(int(min_trials), 2)
    sub = df[df[condition_col].isin([cond_a, cond_b])]
    if sub.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    index_cols = [subject_col, condition_col] + trial_cols
    wide = sub.pivot_table(index=index_cols, columns=time_col, values=value_col, aggfunc="mean")
    unit_weights = (sub.groupby(index_cols)[weight_col].first().reindex(wide.index).to_numpy(dtype=float)
                    if weight_col is not None else None)
    times = wide.columns.to_numpy()
    bins = np.unique(times // bin_size)
    bin_matrix = ((times[:, None] // bin_size) == bins[None, :]).astype(float)  # (timepoints, bins)

    def bin_means(x):  # NaN-aware mean over each bin's timepoints: (..., timepoints) -> (..., bins)
        valid = np.isfinite(x).astype(float)
        counts = valid @ bin_matrix
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, (np.where(valid > 0, x, 0.0) @ bin_matrix) / counts, np.nan)

    full_diffs, half_a, half_b = [], [], []
    position = np.arange(len(wide))
    for _, g in wide.groupby(level=subject_col, sort=False):
        conds = g.index.get_level_values(condition_col)
        values = g.to_numpy(dtype=float)
        rows = position[wide.index.get_indexer(g.index)]
        wa = wb = None
        if unit_weights is not None:
            wa, wb = unit_weights[rows][conds == cond_a], unit_weights[rows][conds == cond_b]
        ma, mb = values[conds == cond_a], values[conds == cond_b]
        if len(ma) < min_trials or len(mb) < min_trials:
            continue
        full_diffs.append((bin_means(_half_means(ma, np.ones((1, len(ma)), dtype=bool), wa))
                           - bin_means(_half_means(mb, np.ones((1, len(mb)), dtype=bool), wb)))[0])
        (a1, a2), (b1, b2) = _split_masks(len(ma), n_splits, rng), _split_masks(len(mb), n_splits, rng)
        half_a.append(bin_means(_half_means(ma, a1, wa)) - bin_means(_half_means(mb, b1, wb)))
        half_b.append(bin_means(_half_means(ma, a2, wa)) - bin_means(_half_means(mb, b2, wb)))
    if not full_diffs:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        full = np.nanmean(np.vstack(full_diffs), axis=0)                  # (bins,)
        group_a, group_b = np.nanmean(np.stack(half_a), axis=0), np.nanmean(np.stack(half_b), axis=0)  # (n_splits, bins)
    pooled = np.concatenate([group_a, group_b], axis=0)

    rows = []
    for j, bin_id in enumerate(bins):
        h = pooled[:, j][np.isfinite(pooled[:, j])]
        both = np.isfinite(group_a[:, j]) & np.isfinite(group_b[:, j])
        have_full = np.isfinite(full[j]) and full[j] != 0 and h.size > 0
        rows.append({
            "bin": int(bin_id),
            "tr_start": int(bin_id * bin_size + tr_offset),
            "tr_end": int(bin_id * bin_size + bin_size - 1 + tr_offset),
            "n_subjects": len(full_diffs),
            "full_mean_diff": float(full[j]),
            "half_ci_low": float(np.percentile(h, 2.5)) if h.size else np.nan,
            "half_ci_high": float(np.percentile(h, 97.5)) if h.size else np.nan,
            "p_half_a_gt_b": float((h > 0).mean()) if h.size else np.nan,
            "p_same_direction": float((np.sign(h) == np.sign(full[j])).mean()) if have_full else np.nan,
            "p_halves_agree": float((np.sign(group_a[both, j]) == np.sign(group_b[both, j])).mean()) if both.any() else np.nan,
            "n_splits": int(both.sum()),
        })
    return pd.DataFrame(rows)


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


def annotate_significance(ax, bin_stats: pd.DataFrame, alpha: float = 0.05, color: str = "black",
                           p_col: str = "p_value", tr_start_col: str = "tr_start", tr_end_col: str = "tr_end"):
    """Draws a short horizontal bar spanning [tr_start, tr_end] plus a "*"
    centered above it, for every row of `bin_stats` (e.g. from
    compare_conditions_by_bin) with p_col < alpha -- a compact way to show
    exactly which time windows a stat test called significant directly on
    the timecourse plot itself, rather than only in a separate table.
    Extends the axes' ylim upward to make room, so it never overlaps the
    existing lines/bands."""
    sig = bin_stats[bin_stats[p_col] < alpha]
    if sig.empty:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    bar_y = ymax + 0.06 * span
    star_y = bar_y + 0.03 * span
    for _, row in sig.iterrows():
        ax.plot([row[tr_start_col], row[tr_end_col]], [bar_y, bar_y], color=color, linewidth=2, solid_capstyle="butt")
        ax.text((row[tr_start_col] + row[tr_end_col]) / 2, star_y, "*", ha="center", va="bottom", fontsize=14, color=color)
    ax.set_ylim(ymin, star_y + 0.08 * span)


def draw_significance_table(ax, rows, alpha: float = 0.05, p_col: str = "p_fdr", tr_seconds: float = 1.0,
                            tr_start_col: str = "tr_start", tr_end_col: str = "tr_end",
                            sig_color: str = "0.55", ns_color: str = "0.93", legend_label: str = None,
                            label_fontsize: float = 8, window_numbers: bool = True, stars: bool = False,
                            frame: bool = False, xaxis: bool = False, show_legend: bool = True,
                            value_col: str = "mean_diff"):
    """A graphic table of which time windows differ significantly, in the style
    of a trajectory figure's footer: one row per comparison, one column per
    stat window, a dark cell where the window is significant and a light one
    where it is not.

    `ax` is a dedicated axes meant to sit directly under a time-series plot
    and share its x-axis (sharex=True): columns are drawn at their real time
    positions, from tr_start * tr_seconds to (tr_end + 1) * tr_seconds, so each
    cell lines up with the stretch of the plot it describes (tr_start/tr_end
    are inclusive 0-indexed TR numbers, i.e. bin_by_size(..., tr_offset=0)).

    `rows` is a list, top to bottom, of {"parts": [(text, color), ...],
    "stats": DataFrame} -- `parts` is the row label as colored pieces (e.g.
    [("Switch", "blue"), (" vs. ", "black"), ("Suppress", "red")]) and `stats`
    has tr_start_col, tr_end_col and p_col, one row per window. A window with
    a missing p is left blank. Windows are numbered 1..N under the first row's
    columns. By default the axes is stripped of ticks and spines.

    stars=True writes a white "*" in every significant cell; frame=True keeps a
    thin black outline around the table; xaxis=True keeps the bottom tick
    labels (shared with the plot above), so the time axis can sit underneath
    the table instead of the plot; show_legend=False omits the shading legend.
    draw_stat_strip below is the one-row version of these settings.

    A row may also carry "direction": "negative" or "positive" -- then a
    window only counts as significant when p < alpha AND that row's value_col
    (the window's mean difference, "mean_diff" by default) is below / above
    zero; a window significant in the other direction is drawn as not
    significant. Rows without "direction" count either sign."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    from matplotlib.patches import Patch, Rectangle

    n = len(rows)
    ax.set_ylim(0, n)
    ax.set_yticks([])
    ax.tick_params(bottom=xaxis, labelbottom=xaxis)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(frame)
        if frame:
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(0.8)

    for i, row in enumerate(rows):
        y0 = n - 1 - i
        for _, w in row["stats"].iterrows():
            p = w[p_col]
            if pd.isna(p):
                continue
            x0 = w[tr_start_col] * tr_seconds
            width = (w[tr_end_col] + 1 - w[tr_start_col]) * tr_seconds
            significant = p < alpha
            if row.get("direction") == "negative":
                significant = significant and w[value_col] < 0
            elif row.get("direction") == "positive":
                significant = significant and w[value_col] > 0
            ax.add_patch(Rectangle((x0, y0), width, 1, facecolor=sig_color if significant else ns_color,
                                   edgecolor="white", linewidth=1.5))
            if stars and significant:
                ax.text(x0 + width / 2, y0 + 0.42, "*", color="white", ha="center", va="center",
                        fontsize=12, fontweight="bold")
        pieces = [TextArea(text, textprops=dict(color=color, fontsize=label_fontsize,
                                                fontweight="normal" if color == "black" else "bold"))
                  for text, color in row["parts"]]
        box = AnchoredOffsetbox(loc="center right", child=HPacker(children=pieces, align="baseline", pad=0, sep=0),
                                bbox_to_anchor=(-0.01, y0 + 0.5), bbox_transform=ax.get_yaxis_transform(),
                                frameon=False, borderpad=0, pad=0)
        ax.add_artist(box)

    if window_numbers and rows and not rows[0]["stats"].empty:
        # numbered from the first row's windows (every row of one table shares the same windows)
        for k, (_, w) in enumerate(rows[0]["stats"].sort_values(tr_start_col).iterrows()):
            xc = (w[tr_start_col] + w[tr_end_col] + 1) / 2 * tr_seconds
            ax.text(xc, -0.12, str(k + 1), ha="center", va="top", fontsize=7, color="gray", clip_on=False)

    if show_legend:
        label = legend_label or f"{p_col} < {alpha}"
        ax.legend(handles=[Patch(facecolor=sig_color, label=label), Patch(facecolor=ns_color, label="n.s.")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.22 if window_numbers else -0.05), ncol=2, frameon=False,
                  fontsize=8, handlelength=1.2)


def draw_stat_strip(ax, stats, label: str = "Positive vs. Negative", alpha: float = 0.05, p_col: str = "p_fdr",
                    tr_seconds: float = 1.0, **kwargs):
    """A single-row stat strip to sit directly under a time-series plot: one
    cell per stat window at its real time position, dark with a white "*"
    where `stats` (tr_start, tr_end, p_col) is significant and light where it
    is not, outlined in black, with the time axis tick labels underneath (it
    shares the plot's x-axis, so create both with sharex). `label` is written
    to the left of the strip. Extra keyword arguments pass through to
    draw_significance_table."""
    settings = dict(sig_color="0.35", ns_color="0.93", stars=True, frame=True, xaxis=True, show_legend=False,
                    window_numbers=False)
    settings.update(kwargs)
    draw_significance_table(ax, [{"parts": [(label, "black")], "stats": stats}], alpha=alpha, p_col=p_col,
                            tr_seconds=tr_seconds, **settings)


def baseline_correct_timecourses(df: pd.DataFrame, group_cols, baseline_windows, value_col: str = "value",
                                  window_col: str = "window_index", weight_col: str = None):
    """Per-participant baseline correction of trial-level evidence.

    For every group in `group_cols` (e.g. subject, operation, category,
    valence) the trials are first averaged into that group's time course (mean
    over trials at each `window_col` value). The baseline is the mean of that
    time course over `baseline_windows` (e.g. (0, 1, 2): the first three TRs of
    the trial, i.e. its first three view-period timepoints), and it is
    subtracted from `value_col` at every timepoint of the group. Doing it per
    trial row is the same as subtracting it from the group's time course, and it
    means any later pooling or averaging (across groups, then across
    participants) works on already-corrected values.

    Groups with no data at any baseline window have no baseline and are
    dropped (they cannot be corrected). With `weight_col` the group's time
    course is the weighted mean over trials (see weighted_group_mean). Returns
    (corrected copy, number of rows dropped)."""
    group_cols = list(group_cols)
    data = df.dropna(subset=group_cols + [window_col])
    timecourse = weighted_group_mean(data, group_cols + [window_col], value_col, weight_col).reset_index()
    baseline = (timecourse[timecourse[window_col].isin(list(baseline_windows))]
                .groupby(group_cols)[value_col].mean().rename("_baseline").to_frame())
    out = df.join(baseline, on=group_cols)
    missing = out["_baseline"].isna()
    n_dropped = int(missing.sum())
    out = out[~missing].copy()
    out[value_col] = out[value_col] - out["_baseline"]
    return out.drop(columns="_baseline"), n_dropped


def baseline_group_names(df: pd.DataFrame, group_cols, valence_col: str = "valence", operation_col: str = "operation") -> dict:
    """{group key: plain-language name} for every distinct combination of `group_cols` in df, in
    reading order (pos before neg, then operation, then category). The key is the tuple of the
    combination's values in `group_cols` order -- what a groupby on those columns yields -- and the
    name is e.g. "positive faces": valence reads positive / negative, an operation keeps its name,
    any other column (the category) is pluralised (face -> faces). Words run valence, operation,
    category."""
    cols = list(group_cols)
    combos = df[cols].dropna().drop_duplicates()
    valence_order = {"pos": 0, "neg": 1}
    nouns = {"face": "faces", "place": "places", "item": "items"}
    category_cols = [c for c in cols if c not in (valence_col, operation_col)]
    ordered = ([valence_col] if valence_col in cols else []) + ([operation_col] if operation_col in cols else []) + category_cols
    rows = []
    for _, r in combos.iterrows():
        sort_key = tuple((valence_order.get(r[c], 2) if c == valence_col else str(r[c])) for c in ordered)
        words = []
        for c in ordered:
            v = str(r[c])
            words.append({"pos": "positive", "neg": "negative"}.get(v, v) if c == valence_col else (v if c == operation_col else nouns.get(v, v)))
        rows.append((sort_key, tuple(r[c] for c in cols), " ".join(words)))
    return {key: name for _, key, name in sorted(rows, key=lambda t: t[0])}


def describe_baseline_groups(df: pd.DataFrame, group_cols, valence_col: str = "valence", operation_col: str = "operation") -> list:
    """The names of baseline_group_names(), in reading order, e.g. ["positive faces",
    "positive places", "negative faces", "negative places"] for group_cols = ("object", "valence"):
    the trial groups that each get their own baseline in
    baseline_correct_timecourses(df, ["subject", *group_cols], ...)."""
    return list(baseline_group_names(df, group_cols, valence_col, operation_col).values())


def baseline_only_bins(baseline_windows, bin_size: int) -> set:
    """Stat bins (bin_by_size numbering) that lie entirely inside the baseline
    windows. After baseline correction such a bin averages to zero in every
    participant by construction, so it carries no information to test (and a
    t-test on it would only amplify floating-point noise): leave it out of the
    statistics."""
    windows = set(baseline_windows)
    candidate_bins = {w // bin_size for w in windows}
    return {b for b in candidate_bins if all(t in windows for t in range(b * bin_size, (b + 1) * bin_size))}


BALANCE_MODES = ("subsample", "weighted")


def balance_trial_table(trials: pd.DataFrame, within=("operation", "category", "valence"), seed: int = 0,
                        mode: str = "subsample"):
    """Within every subject x `within` cell, give trials that follow a trial of the SAME
    category as the one before them and trials that follow a trial of a DIFFERENT category
    or no trial at all (the first trial of an acquisition) the same total say. Let k be the
    size of the smaller group in a cell; `trials` is trial_order_analysis.build_trial_table()'s output.

    mode="subsample": the smaller group is kept whole and the larger one is randomly thinned
    to k trials (a fixed `seed` makes the draw reproducible; cells are visited in sorted
    order). Every kept trial has weight 1.

    mode="weighted": every trial of a cell with k > 0 is kept, with weight k / (size of its
    group): the probability that "subsample" would have kept it. A weighted mean over these
    trials is the average of the subsample mean over every possible draw, so it uses all
    trials with no randomness (`seed` is not used). The smaller group has weight 1.

    A cell with no trial in one of the two groups keeps nothing. A trial whose predecessor
    has no type (a truncated trial) is left out: whether its predecessor had the same
    category is unknown.

    Returns (kept, report): `kept` is the retained trials (subject, boldfile, trial_index plus
    `predecessor`, "same" or "different", and `weight`); `report` has one row per subject x
    cell with the trial counts before balancing, the number kept per group (subsample) or
    the size of the smaller group (weighted) in n_kept_per_group, and each group's inclusion
    probability."""
    if mode not in BALANCE_MODES:
        raise ValueError(f"mode must be one of {BALANCE_MODES}, got {mode!r}")
    from _interactive_notebooks.trial_order_analysis import with_predecessor_flags
    within = list(within)
    flagged = with_predecessor_flags(trials)
    flagged = flagged[flagged["trial_type"].notna()].copy()
    same = flagged["same_category"].astype("boolean")
    is_same = same.fillna(False).astype(bool)
    is_different = (~same.fillna(True)).astype(bool) | flagged["is_first_trial"]      # known to differ, or no predecessor
    flagged["predecessor"] = np.select([is_same, is_different], ["same", "different"], default="unknown")
    n_unknown_predecessor = int((flagged["predecessor"] == "unknown").sum())
    flagged = flagged[flagged["predecessor"] != "unknown"]
    keys = ["subject", "boldfile", "trial_index"]
    rng = np.random.default_rng(seed)
    kept_parts, rows = [], []
    for cell, g in flagged.groupby(["subject"] + within, sort=True):
        same_rows = g[g["predecessor"] == "same"].sort_values(keys)
        diff_rows = g[g["predecessor"] == "different"].sort_values(keys)
        k = min(len(same_rows), len(diff_rows))
        for part in (same_rows, diff_rows):
            if not k:
                continue
            if mode == "subsample":
                kept_parts.append(part.iloc[np.sort(rng.choice(len(part), size=k, replace=False))].assign(weight=1.0))
            else:
                kept_parts.append(part.assign(weight=k / len(part)))
        rows.append(dict(zip(["subject"] + within, cell if isinstance(cell, tuple) else (cell,)),
                         n_same_category=len(same_rows), n_different_or_no_category=len(diff_rows), n_kept_per_group=k,
                         inclusion_probability_same_category=k / len(same_rows) if len(same_rows) else np.nan,
                         inclusion_probability_different_or_no_category=k / len(diff_rows) if len(diff_rows) else np.nan))
    columns = keys + ["predecessor", "weight"]
    kept = pd.concat(kept_parts)[columns] if kept_parts else flagged.iloc[0:0].assign(weight=1.0)[columns]
    report = pd.DataFrame(rows)
    report.attrs["n_trials_with_unknown_predecessor"] = n_unknown_predecessor
    report.attrs["mode"] = mode
    return kept.reset_index(drop=True), report


def balance_preceding_category(decoding: pd.DataFrame, within=("operation", "category", "valence"), seed: int = 0,
                               mode: str = "subsample"):
    """balance_trial_table() on the trials of a raw decoding table (one row per decoded TR, with
    subject, session, task, run, boldfile, trial_index, window_index, volume_of_interest and
    trial_type). Returns (kept, report) -- see balance_trial_table."""
    from _interactive_notebooks.trial_order_analysis import build_trial_table
    return balance_trial_table(build_trial_table(decoding), within=within, seed=seed, mode=mode)


def first_significant_bin(bin_stats: pd.DataFrame, alpha: float = 0.05, p_col: str = "p_value",
                          value_col: str = "mean_diff", direction: str = None, bin_col: str = "bin"):
    """The earliest row of `bin_stats` (by bin_col) with p_col < alpha -- and,
    when direction is "negative" / "positive", value_col below / above zero --
    or None if no bin qualifies."""
    if direction not in (None, "negative", "positive"):
        raise ValueError(f"direction must be None, 'negative' or 'positive', got {direction!r}")
    sig = bin_stats[bin_stats[p_col] < alpha]
    if direction == "negative":
        sig = sig[sig[value_col] < 0]
    elif direction == "positive":
        sig = sig[sig[value_col] > 0]
    if sig.empty:
        return None
    return sig.sort_values(bin_col).iloc[0]


def mark_onsets_in_rows(ax, onsets, marker: str = "v", size: float = 70, row_step: float = 0.055,
                        label: str = None):
    """One marker per row in a strip at the bottom of the axes, so markers for
    different conditions can never overlap, whatever their x positions.

    `onsets` is a list of (x, color) in the order the rows should appear, top
    to bottom; x is None for a condition with nothing to mark (its row is
    left empty, so every condition keeps the same row on every plot). Rows
    are placed in axes-fraction coordinates (x stays in data units) and the
    y-axis is extended downward so the strip is clear of the plotted data.
    `label`, if given, is written small at the left of the strip."""
    n = len(onsets)
    if n == 0:
        return
    strip_top = 0.03 + n * row_step          # strip spans axes fractions [0, strip_top]
    ymin, ymax = ax.get_ylim()
    # new_ymin chosen so the old ymin sits at strip_top (a fraction of the NEW axis height)
    ax.set_ylim(ymin - strip_top / (1 - strip_top) * (ymax - ymin), ymax)
    trans = ax.get_xaxis_transform()         # x: data, y: axes fraction
    for k, (x, color) in enumerate(onsets):
        if x is None:
            continue
        y = 0.03 + (n - 1 - k + 0.5) * row_step
        ax.scatter([x], [y], marker=marker, s=size, color=color, edgecolor="black", linewidth=0.5,
                   zorder=5, transform=trans, clip_on=False)
    ax.axhline(ymin, color="lightgray", linewidth=0.6, zorder=0)   # where the data area ends
    if label:
        ax.text(0.005, strip_top - 0.004, label, transform=ax.transAxes, fontsize=7, color="gray",
                va="top", ha="left")


def mark_significance_onset(ax, bin_stats: pd.DataFrame, value_col: str = "mean_diff", alpha: float = 0.05,
                             color: str = "black", marker: str = "v", size: float = 100,
                             p_col: str = "p_value", bin_col: str = "bin", tr_start_col: str = "tr_start",
                             direction: str = None, first_only: bool = False):
    """Marks the *onset* of each contiguous run of significant bins (p_col <
    alpha, consecutive `bin_col` values) with a downward-pointing triangle,
    directly on the line itself -- at that bin's own tr_start_col position
    and value_col's value there (e.g. "mean_diff" from
    compare_conditions_by_bin, the same per-bin group mean the plotted
    line's curve already shows) -- rather than annotate_significance's bar
    spanning every significant bin. Useful for a plot (e.g. a
    baseline-subtraction plot) where "when does this line begin to
    significantly differ" matters more than a full summary of every
    significant window; a significant run that stops and later resumes
    gets one marker per resumption, not one continuous span.

    direction=None (default) counts any significant bin; "negative" /
    "positive" counts only bins whose value_col is below / above zero (e.g.
    "negative" on an operation-minus-baseline plot: only where the
    condition is significantly *less* than the baseline). first_only=True
    marks just the single earliest qualifying bin instead of every run's
    onset. The p-values are used as given: for a one-sided question, build
    bin_stats with the matching `alternative`."""
    if direction not in (None, "negative", "positive"):
        raise ValueError(f"direction must be None, 'negative' or 'positive', got {direction!r}")
    sig = bin_stats[bin_stats[p_col] < alpha]
    if direction == "negative":
        sig = sig[sig[value_col] < 0]
    elif direction == "positive":
        sig = sig[sig[value_col] > 0]
    sig = sig.sort_values(bin_col)
    if sig.empty:
        return
    if first_only:
        onsets = sig.iloc[[0]]
    else:
        is_onset = sig[bin_col].diff() != 1
        onsets = sig[is_onset]
    ax.scatter(onsets[tr_start_col], onsets[value_col], marker=marker, s=size, color=color,
               edgecolor="black", linewidth=0.5, zorder=5)
