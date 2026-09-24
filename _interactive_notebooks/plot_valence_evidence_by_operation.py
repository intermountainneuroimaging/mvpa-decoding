#!/usr/bin/env python3

"""
Clearvale-specific analysis (not part of the general report pipeline):
positive- vs. negative-valence classifier evidence, per operation, with a
maintain-baseline-subtracted view and binned significance testing.

This is a thin, project-specific configuration of the reusable building
blocks in _interactive_notebooks/decoding_results_toolkit.py -- to point this same
approach at a different filter/condition/window/stat combination (a
different operation set, a different valence coding, two-tailed instead of
one-tailed, a non-parametric test, wider or narrower bins, ...), either
edit the constants below or write a new short script against the toolkit
directly; the toolkit itself never needs to change.

For each operation category (regressor_label -- e.g. maintain/suppress/
switch/clear) plots the classifier's *self*-evidence (evidence_<that row's
own true category>) over time, split into WMpos ("pos", red) vs. WMneg
("neg", blue) trials:

  1. Raw evidence, one page per operation, 3 panels: face-only, place-only,
     and collapsed across stimulus (the average of the face-only and
     place-only means, giving each stimulus equal weight regardless of
     trial-count imbalance -- not a direct pool of every trial) -- the
     face/place split is purely a sanity check (are there odd
     stimulus-specific differences?); the collapsed panel is the one that
     matters for the pos-vs-neg question itself.
  2. The same, for suppress/switch/clear only, with maintain's own
     (subject- and window-matched) evidence subtracted out first -- does a
     removal operation's pos/neg pattern look different from maintain's own,
     once maintain's baseline pattern is removed?
  3. A significance table: for every operation (raw) and every
     suppress/switch/clear (baseline-subtracted), the timecourse is binned
     into non-overlapping TR chunks (--bin-size, default 3); for each chunk,
     this subject's own mean pos-evidence minus mean neg-evidence is a
     paired difference score, tested one-tailed (H1: pos > neg, by default)
     via a one-sample t-test on those per-subject difference scores across
     the group -- see --method to swap in a different test.

Stimulus category (face/place) is read from trial_type (".*face.*" /
".*place.*", the same convention model_conditions.timecourse_decoding.overlay
entries already use elsewhere in this repo); valence from task (exact
"WMpos"/"WMneg"). Both are parsed directly from decoding_results.csv, not
from the config -- this script only needs --analysis-output-dir and
--desc/--config to find that file, not the full model_conditions.

Usage:
    python _interactive_notebooks/plot_valence_evidence_by_operation.py \\
        --analysis-output-dir /path/to/out --config configs/config-kfold.clearvale-operation.json \\
        --subjects configs/subject-list.txt --output-dir ./valence_evidence_analysis
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from workflows.generate_report import resolve_desc, list_subject_dirs, parse_subjects_arg
from _interactive_notebooks.decoding_results_toolkit import (
    load_decoding_results, select_evidence_value, derive_label,
    aggregate_by_subject_window, average_across_groups, subtract_baseline,
    bin_by_size, compare_conditions_by_bin, plot_conditions, STAT_METHODS,
)

# -- the only project-specific knobs; everything downstream is generic
# (_interactive_notebooks/decoding_results_toolkit.py) --
VALENCE_MAPPING = {"pos": "WMpos", "neg": "WMneg"}  # task -> condition label (exact match)
STIMULUS_MAPPING = {"face": "face", "place": "place"}  # trial_type -> group label (regex)
VALENCE_COLORS = {"pos": "red", "neg": "blue"}
BASELINE_OPERATION = "maintain"
BIN_SIZE_DEFAULT = 3


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
        "--method", default="ttest_1samp_diff", choices=list(STAT_METHODS),
        help="Per-bin stat test on the (pos - neg) difference score (default ttest_1samp_diff)."
    )
    parser.add_argument(
        "--alternative", default="greater", choices=["greater", "less", "two-sided"],
        help="One-tailed direction for the per-bin test -- \"greater\" (default) tests pos > neg."
    )
    parser.add_argument("--output-dir", required=True, help="Where to write the PDF + stats CSVs.")
    return parser.parse_args()


def per_operation_means(df: pd.DataFrame, operation: str) -> dict:
    """{"face": wide_df, "place": wide_df, "collapsed": wide_df} for one
    operation -- see _interactive_notebooks/decoding_results_toolkit.py's
    aggregate_by_subject_window/average_across_groups."""
    op_df = df[df["regressor_label"] == operation]
    per_stimulus = {
        stim: aggregate_by_subject_window(op_df[op_df["stimulus"] == stim], condition_col="valence")
        for stim in STIMULUS_MAPPING
    }
    return {**per_stimulus, "collapsed": average_across_groups(per_stimulus)}


def render_operation_page(pdf, op_means: dict, suptitle: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, key in zip(axes, ("face", "place", "collapsed")):
        plot_conditions(ax, op_means[key], VALENCE_COLORS, title=key)
    axes[0].set_ylabel("classifier evidence (own true category)")
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def compute_bin_stats(wide_df: pd.DataFrame, bin_size: int, method: str, alternative: str) -> pd.DataFrame:
    per_subject_bin, bins_meta = bin_by_size(wide_df, bin_size=bin_size)
    return compare_conditions_by_bin(per_subject_bin, "pos", "neg", method=method, alternative=alternative, bins_meta=bins_meta)


def main():
    args = parse_args()
    desc = resolve_desc(args.desc, args.config)
    subjects_arg = parse_subjects_arg(args.subjects) if args.subjects else None
    subjects = list_subject_dirs(args.analysis_output_dir, desc, subjects=subjects_arg)
    print(f"Scope: {len(subjects)} subject(s): {subjects}")

    raw = load_decoding_results(args.analysis_output_dir, desc, subjects)
    df = select_evidence_value(raw, value="self", new_col="value")
    df = derive_label(df, "task", VALENCE_MAPPING, new_col="valence")
    df = derive_label(df, "trial_type", STIMULUS_MAPPING, new_col="stimulus", regex=True)

    operations = sorted(df["regressor_label"].unique(), key=lambda o: (o != BASELINE_OPERATION, o))
    print(f"Operations found: {operations}")

    means_by_operation = {op: per_operation_means(df, op) for op in operations}

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, f"{desc}_valence_evidence.pdf")
    stats_rows = []

    with PdfPages(pdf_path) as pdf:
        for op in operations:
            render_operation_page(pdf, means_by_operation[op], f"{op}: pos vs. neg classifier evidence")
            for scope in ("face", "place", "collapsed"):
                bin_stats = compute_bin_stats(means_by_operation[op][scope], args.bin_size, args.method, args.alternative)
                if bin_stats.empty:
                    continue
                bin_stats.insert(0, "operation", op)
                bin_stats.insert(1, "stimulus_scope", scope)
                bin_stats.insert(2, "baseline_subtracted", False)
                stats_rows.append(bin_stats)

        if BASELINE_OPERATION in means_by_operation:
            baseline = means_by_operation[BASELINE_OPERATION]
            for op in operations:
                if op == BASELINE_OPERATION:
                    continue
                adjusted = {scope: subtract_baseline(means_by_operation[op][scope], baseline[scope]) for scope in ("face", "place", "collapsed")}
                render_operation_page(pdf, adjusted, f"{op} minus {BASELINE_OPERATION} baseline: pos vs. neg classifier evidence")
                for scope in ("face", "place", "collapsed"):
                    bin_stats = compute_bin_stats(adjusted[scope], args.bin_size, args.method, args.alternative)
                    if bin_stats.empty:
                        continue
                    bin_stats.insert(0, "operation", op)
                    bin_stats.insert(1, "stimulus_scope", scope)
                    bin_stats.insert(2, "baseline_subtracted", True)
                    stats_rows.append(bin_stats)
        else:
            print(f"  (!) no {BASELINE_OPERATION!r} operation found -- skipping baseline-subtracted pages/stats")

    print(f"PDF written to: {pdf_path}")

    if stats_rows:
        stats_df = pd.concat(stats_rows, ignore_index=True)
        stats_path = os.path.join(args.output_dir, f"{desc}_valence_evidence_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Stats table written to: {stats_path}")


if __name__ == "__main__":
    main()
