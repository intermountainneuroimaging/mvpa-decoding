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


def _half_means(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """NaN-aware mean of `matrix` (units x timepoints) over each row of
    `mask` (n_splits x units) -> (n_splits, timepoints)."""
    valid = np.isfinite(matrix)
    sums = mask.astype(float) @ np.where(valid, matrix, 0.0)
    counts = mask.astype(float) @ valid.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def split_half_group_difference(df: pd.DataFrame, condition_col: str, cond_a: str, cond_b: str,
                                 bin_size: int = 3, value_col: str = "value", subject_col: str = "subject",
                                 time_col: str = "window_index", trial_cols=("boldfile", "trial_index"),
                                 tr_offset: int = 0, n_splits: int = 1000, seed: int = 0,
                                 min_trials: int = 2) -> pd.DataFrame:
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
    the halves are compared. Subjects need at least `min_trials` (>= 2) trials
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
    wide = sub.pivot_table(index=[subject_col, condition_col] + trial_cols, columns=time_col,
                           values=value_col, aggfunc="mean")
    times = wide.columns.to_numpy()
    bins = np.unique(times // bin_size)
    bin_matrix = ((times[:, None] // bin_size) == bins[None, :]).astype(float)  # (timepoints, bins)

    def bin_means(x):  # NaN-aware mean over each bin's timepoints: (..., timepoints) -> (..., bins)
        valid = np.isfinite(x).astype(float)
        counts = valid @ bin_matrix
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, (np.where(valid > 0, x, 0.0) @ bin_matrix) / counts, np.nan)

    full_diffs, half_a, half_b = [], [], []
    for _, g in wide.groupby(level=subject_col, sort=False):
        conds = g.index.get_level_values(condition_col)
        values = g.to_numpy(dtype=float)
        ma, mb = values[conds == cond_a], values[conds == cond_b]
        if len(ma) < min_trials or len(mb) < min_trials:
            continue
        full_diffs.append((bin_means(_half_means(ma, np.ones((1, len(ma)), dtype=bool)))
                           - bin_means(_half_means(mb, np.ones((1, len(mb)), dtype=bool))))[0])
        (a1, a2), (b1, b2) = _split_masks(len(ma), n_splits, rng), _split_masks(len(mb), n_splits, rng)
        half_a.append(bin_means(_half_means(ma, a1)) - bin_means(_half_means(mb, b1)))
        half_b.append(bin_means(_half_means(ma, a2)) - bin_means(_half_means(mb, b2)))
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


def mark_significance_onset(ax, bin_stats: pd.DataFrame, value_col: str = "mean_diff", alpha: float = 0.05,
                             color: str = "black", marker: str = "v", size: float = 100,
                             p_col: str = "p_value", bin_col: str = "bin", tr_start_col: str = "tr_start"):
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
    gets one marker per resumption, not one continuous span."""
    sig = bin_stats[bin_stats[p_col] < alpha].sort_values(bin_col)
    if sig.empty:
        return
    is_onset = sig[bin_col].diff() != 1
    onsets = sig[is_onset]
    ax.scatter(onsets[tr_start_col], onsets[value_col], marker=marker, s=size, color=color,
               edgecolor="black", linewidth=0.5, zorder=5)
