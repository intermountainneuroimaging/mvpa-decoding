#!/usr/bin/env python3

"""
Clearvale-specific analysis (not part of the general report pipeline):
positive- vs. negative-valence classifier evidence, per operation, with a
maintain-baseline-subtracted view and 3-TR-bin significance testing.

Reads each subject's own decoding_results.csv (raw, one row per decoded TR --
see mvpa_workflow.py/generate_report.py) and, for each operation category
(regressor_label -- e.g. maintain/suppress/switch/clear) plots the
classifier's *self*-evidence (evidence_<that row's own true category>) over
time, split into WMpos ("pos", red) vs. WMneg ("neg", blue) trials:

  1. Raw evidence, one page per operation, 3 panels: face-only, place-only,
     and collapsed across stimulus (pooling every trial regardless of
     face/place) -- the face/place split is purely a sanity check (are there
     odd stimulus-specific differences?); the collapsed panel is the one
     that matters for the pos-vs-neg question itself.
  2. The same, for suppress/switch/clear only, with maintain's own
     (subject- and window-matched) evidence subtracted out first -- does a
     removal operation's pos/neg pattern look different from maintain's own,
     once maintain's baseline pattern is removed?
  3. A significance table: for every operation (raw) and every
     suppress/switch/clear (baseline-subtracted), the timecourse is binned
     into non-overlapping 3-TR chunks; for each chunk, this subject's own
     mean pos-evidence minus mean neg-evidence is a paired difference score,
     tested one-tailed (H1: pos > neg) via a one-sample t-test on those
     per-subject difference scores across the group.

Stimulus category (face/place) is read from trial_type (".*face.*" /
".*place.*", the same convention model_conditions.timecourse_decoding.overlay
entries already use elsewhere in this repo); valence from task (exact
"WMpos"/"WMneg"). Both are parsed directly from decoding_results.csv, not
from the config -- this script only needs --analysis-output-dir and
--desc/--config to find that file, not the full model_conditions.

Deliberately a separate, standalone script rather than a change to
generate_report.py -- this is a one-off analysis for a specific write-up,
not a general reporting feature.

Usage:
    python analysis/plot_valence_evidence_by_operation.py \\
        --analysis-output-dir /path/to/out --config configs/config-kfold.clearvale-operation.json \\
        --subjects configs/subject-list.txt --output-dir ./valence_evidence_analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from workflows.generate_report import resolve_desc, list_subject_dirs, parse_subjects_arg, subject_paths

BIN_SIZE_DEFAULT = 3
VALENCE_COLORS = {"pos": "red", "neg": "blue"}
VALENCE_TASK_MAP = {"WMpos": "pos", "WMneg": "neg"}


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-output-dir", required=True, help="Same value passed to mvpa_workflow.py.")
    parser.add_argument("--desc", default=None, help="Classifier folder name (model.desc, sanitized). One of --desc/--config is required.")
    parser.add_argument("--config", default=None, help="mvpa config JSON -- supplies model.desc when --desc is omitted.")
    parser.add_argument(
        "--subjects", default=None,
        help="Restrict to these subjects -- a comma-separated list (e.g. \"001,004,010\") or a path to a "
             "text file listing them (one per line and/or comma-separated). Omit to use every subject "
             "found under analysis-output-dir/desc."
    )
    parser.add_argument("--bin-size", type=int, default=BIN_SIZE_DEFAULT, help=f"TRs per non-overlapping significance-test bin (default {BIN_SIZE_DEFAULT}).")
    parser.add_argument(
        "--alternative", default="greater", choices=["greater", "less", "two-sided"],
        help="One-tailed direction for the per-bin t-test on (pos - neg) difference scores against 0 "
             "-- \"greater\" (default) tests pos > neg."
    )
    parser.add_argument("--output-dir", required=True, help="Where to write the PDF + stats CSVs.")
    return parser.parse_args()


# =====================================================
# Data loading + derived columns
# =====================================================

def load_group_raw(analysis_output_dir: str, desc: str, subjects: list) -> pd.DataFrame:
    """Every subject's own decoding_results.csv concatenated -- same file
    generate_report.py's compile_group_decoding reads, just loaded directly
    here so this script has no other dependency on generate_report.py's own
    CLI/report-building code."""
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


def add_derived_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """valence (pos/neg, from task), stimulus (face/place, from trial_type),
    and self_evidence (evidence_<this row's own regressor_label> -- how much
    the classifier believed this trial was its own true category) -- rows
    where valence or stimulus can't be resolved are dropped (counts
    printed), since they can't participate in either comparison."""
    df = raw.copy()

    df["valence"] = df["task"].map(VALENCE_TASK_MAP)
    unresolved_valence = df["valence"].isna().sum()
    if unresolved_valence:
        print(f"  (!) {unresolved_valence} row(s) had a task value other than WMpos/WMneg -- dropped")
    df = df[df["valence"].notna()]

    is_face = df["trial_type"].str.contains("face", case=False, na=False)
    is_place = df["trial_type"].str.contains("place", case=False, na=False)
    df["stimulus"] = np.select([is_face, is_place], ["face", "place"], default=None)
    unresolved_stimulus = (df["stimulus"].isna()).sum()
    if unresolved_stimulus:
        print(f"  (!) {unresolved_stimulus} row(s) had no face/place in trial_type -- dropped")
    df = df[df["stimulus"].notna()]

    evidence_cols = [c for c in df.columns if c.startswith("evidence_")]
    categories = [c.replace("evidence_", "") for c in evidence_cols]
    cat_to_col_idx = {c: i for i, c in enumerate(categories)}
    unknown_op = ~df["regressor_label"].isin(cat_to_col_idx)
    if unknown_op.any():
        print(f"  (!) {int(unknown_op.sum())} row(s) had a regressor_label with no matching evidence_ column -- dropped")
    df = df[~unknown_op]

    evidence_matrix = df[evidence_cols].to_numpy()
    row_idx = df["regressor_label"].map(cat_to_col_idx).to_numpy()
    df = df.copy()
    df["self_evidence"] = evidence_matrix[np.arange(len(df)), row_idx]

    return df


# =====================================================
# Aggregation
# =====================================================

def per_subject_window_means(df: pd.DataFrame, operation: str) -> dict:
    """For one operation, returns {"face": wide_df, "place": wide_df,
    "collapsed": wide_df} -- each wide_df is subject x window_index, one
    column per valence (pos/neg), averaged across every trial sharing that
    (subject, window_index, valence[, stimulus]) -- "collapsed" pools face
    and place trials together directly (not an average of the two stimulus
    means), matching "classifier evidence regardless of stimulus type"."""
    op_df = df[df["regressor_label"] == operation]

    out = {}
    for stim in ("face", "place"):
        subset = op_df[op_df["stimulus"] == stim]
        grouped = subset.groupby(["subject", "window_index", "valence"])["self_evidence"].mean().unstack("valence")
        out[stim] = grouped
    out["collapsed"] = op_df.groupby(["subject", "window_index", "valence"])["self_evidence"].mean().unstack("valence")
    return out


def subtract_baseline(op_means: dict, baseline_means: dict) -> dict:
    """op_means - baseline_means (e.g. maintain), matched on (subject,
    window_index) via an inner join -- a subject/window/valence missing from
    either side (e.g. near a trial-count edge case) is simply excluded
    rather than guessed at."""
    out = {}
    for key, op_df in op_means.items():
        base_df = baseline_means[key]
        common_cols = [c for c in ("pos", "neg") if c in op_df.columns and c in base_df.columns]
        aligned_op, aligned_base = op_df[common_cols].align(base_df[common_cols], join="inner")
        out[key] = aligned_op - aligned_base
    return out


# =====================================================
# Plotting
# =====================================================

def _plot_pos_neg_panel(ax, wide_df: pd.DataFrame, title: str):
    """wide_df: subject x window_index, columns "pos"/"neg" (whichever are
    present) -- group mean +/- SE across subjects, one line per valence."""
    for valence, color in VALENCE_COLORS.items():
        if valence not in wide_df.columns:
            continue
        series = wide_df[valence].dropna()
        if series.empty:
            continue
        by_window = series.groupby("window_index")
        mean = by_window.mean()
        n = by_window.count()
        se = by_window.std(ddof=1) / np.sqrt(n)
        tr = mean.index.to_numpy() + 1  # 1-indexed TR, matching "TRs 1-3" convention
        ax.fill_between(tr, mean - se, mean + se, alpha=0.2, color=color)
        ax.plot(tr, mean, color=color, linewidth=1.5, label=valence)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("TR")
    ax.legend(fontsize=8)


def render_operation_page(pdf, operation: str, op_means: dict, suptitle: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, key in zip(axes, ("face", "place", "collapsed")):
        _plot_pos_neg_panel(ax, op_means[key], key)
    axes[0].set_ylabel("classifier evidence (own true category)")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# =====================================================
# Significance testing
# =====================================================

def bin_and_test(wide_df: pd.DataFrame, bin_size: int, alternative: str) -> pd.DataFrame:
    """wide_df: subject x window_index, columns "pos"/"neg". Bins window_index
    into non-overlapping chunks of `bin_size` (bin 0 = TRs 1..bin_size, bin 1
    = the next bin_size TRs, ...), takes each subject's own mean pos and mean
    neg evidence within a bin, forms a per-subject (pos - neg) difference
    score, and one-sample t-tests that difference against 0 -- the paired
    equivalent of a pos-vs-neg comparison, since each subject contributes
    exactly one difference score per bin."""
    if "pos" not in wide_df.columns or "neg" not in wide_df.columns:
        return pd.DataFrame()

    df = wide_df[["pos", "neg"]].dropna().reset_index()
    df["bin"] = df["window_index"] // bin_size

    rows = []
    for bin_id, bin_df in df.groupby("bin"):
        per_subject = bin_df.groupby("subject")[["pos", "neg"]].mean()
        diff = (per_subject["pos"] - per_subject["neg"]).to_numpy()
        n = len(diff)
        if n < 2:
            continue
        t_stat, p_value = stats.ttest_1samp(diff, popmean=0.0, alternative=alternative)
        rows.append({
            "bin": int(bin_id),
            "tr_start": int(bin_id * bin_size) + 1,
            "tr_end": int(bin_id * bin_size) + bin_size,
            "n_subjects": n,
            "mean_diff_pos_minus_neg": float(np.mean(diff)),
            "sd_diff": float(np.std(diff, ddof=1)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
        })
    return pd.DataFrame(rows)


# =====================================================
# Main
# =====================================================

def main():
    args = parse_args()
    desc = resolve_desc(args.desc, args.config)
    subjects_arg = parse_subjects_arg(args.subjects) if args.subjects else None
    subjects = list_subject_dirs(args.analysis_output_dir, desc, subjects=subjects_arg)
    print(f"Scope: {len(subjects)} subject(s): {subjects}")

    raw = load_group_raw(args.analysis_output_dir, desc, subjects)
    df = add_derived_columns(raw)

    operations = [c for c in df["regressor_label"].unique()]
    # config declaration order isn't available here (no config-only
    # dependency by design) -- "maintain" first if present, since it's the
    # baseline every other operation gets compared/subtracted against,
    # otherwise alphabetical
    operations = sorted(operations, key=lambda o: (o != "maintain", o))
    print(f"Operations found: {operations}")

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{desc}_valence_evidence.pdf")
    stats_rows = []

    means_by_operation = {op: per_subject_window_means(df, op) for op in operations}

    with PdfPages(pdf_path) as pdf:
        # 1. raw pos vs. neg evidence, one page per operation
        for op in operations:
            render_operation_page(pdf, op, means_by_operation[op], f"{op}: pos vs. neg classifier evidence")
            for key in ("face", "place", "collapsed"):
                bin_stats = bin_and_test(means_by_operation[op][key], args.bin_size, args.alternative)
                if bin_stats.empty:
                    continue
                bin_stats.insert(0, "operation", op)
                bin_stats.insert(1, "stimulus_scope", key)
                bin_stats.insert(2, "baseline_subtracted", False)
                stats_rows.append(bin_stats)

        # 2. maintain-baseline-subtracted, suppress/switch/clear only
        if "maintain" in means_by_operation:
            baseline = means_by_operation["maintain"]
            for op in operations:
                if op == "maintain":
                    continue
                adjusted = subtract_baseline(means_by_operation[op], baseline)
                render_operation_page(
                    pdf, op, adjusted,
                    f"{op} minus maintain baseline: pos vs. neg classifier evidence"
                )
                for key in ("face", "place", "collapsed"):
                    bin_stats = bin_and_test(adjusted[key], args.bin_size, args.alternative)
                    if bin_stats.empty:
                        continue
                    bin_stats.insert(0, "operation", op)
                    bin_stats.insert(1, "stimulus_scope", key)
                    bin_stats.insert(2, "baseline_subtracted", True)
                    stats_rows.append(bin_stats)
        else:
            print("  (!) no \"maintain\" operation found -- skipping baseline-subtracted pages/stats")

    print(f"PDF written to: {pdf_path}")

    if stats_rows:
        stats_df = pd.concat(stats_rows, ignore_index=True)
        stats_path = os.path.join(args.output_dir, f"{desc}_valence_evidence_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Stats table written to: {stats_path}")


if __name__ == "__main__":
    main()
