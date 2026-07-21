#!/usr/bin/env python3

"""
Generate a PDF report (accuracy/AUC, confusion-style matrices, timecourse
decoding, importance maps) from mvpa_workflow.py's output -- either for one
subject, or aggregated across every subject found for a given classifier
("desc").

Usage:
    # group report -- aggregates every subject found under <dir>/<desc>/*/
    python generate_report.py --analysis-output-dir ./out --desc gm_valence_classifier \\
        --config examples/config-2.example.json --master-spreadsheet master_spreadsheet.csv

    # single-subject report -- scoped to just <dir>/<desc>/4057/
    python generate_report.py --analysis-output-dir ./out --desc gm_valence_classifier \\
        --subject 4057 --config examples/config-2.example.json --master-spreadsheet master_spreadsheet.csv

--config/--master-spreadsheet are both optional and only needed for
timecourse-decoding annotation (median event duration + TR, both derived
from real data, not hardcoded) -- without them the report still renders,
just without those annotations.
"""

import argparse
import glob
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
import numpy as np
import pandas as pd

from mvpa_common import label_rows, get_bold_header_info, resolve_window_times


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--analysis-output-dir", required=True,
        help="Same --analysis-output-dir passed to mvpa_workflow.py -- results are read from "
             "<this>/<desc>/*/{cv,model,decoding}/"
    )
    parser.add_argument("--desc", required=True, help="Classifier folder name (model.desc, sanitized) under analysis-output-dir")
    parser.add_argument("--subject", default=None, help="Restrict the report to one subject (single-subject report). Omit for a group report across all subjects found.")
    parser.add_argument("--config", default=None, help="mvpa config JSON -- supplies timecourse_decoding conditions/window for annotation. Optional.")
    parser.add_argument("--master-spreadsheet", default=None, help="master_spreadsheet.csv -- needed for TR + median trial duration (timecourse annotation). Optional.")
    parser.add_argument("--output", default=None, help="Output PDF path. Defaults to <dir>/<desc>/report_<desc>.pdf (group) or <dir>/<desc>/<subject>/report_<subject>.pdf (single-subject).")
    return parser.parse_args()


# =====================================================
# Subject-scope discovery + file layout
# =====================================================

def list_subject_dirs(analysis_output_dir: str, desc: str, subject: str = None) -> list:
    base = os.path.join(analysis_output_dir, desc)
    if subject:
        if not os.path.isdir(os.path.join(base, subject, "model")):
            raise SystemExit(f"No results found for subject {subject!r} at {os.path.join(base, subject)}")
        return [subject]

    subjects = sorted(
        name for name in os.listdir(base)
        if os.path.isdir(os.path.join(base, name, "model"))
    ) if os.path.isdir(base) else []
    if not subjects:
        raise SystemExit(f"No subject result folders found under {base}")
    return subjects


def subject_paths(analysis_output_dir: str, desc: str, subject: str) -> dict:
    base = os.path.join(analysis_output_dir, desc, subject)
    return {
        "cv_total": os.path.join(base, "cv", f"{subject}_cv_results_total_scores.csv"),
        "model_total": os.path.join(base, "model", f"{subject}_model_results_total_scores.csv"),
        "model_auc": os.path.join(base, "model", f"{subject}_model_results_auc.csv"),
        "model_accuracy": os.path.join(base, "model", f"{subject}_model_results_accuracy.csv"),
        "model_evidence": os.path.join(base, "model", f"{subject}_model_results_evidence.csv"),
        "model_impa": os.path.join(base, "model", f"{subject}_impa_native.nii.gz"),
        # the report always uses the pre-aggregated summary (one row per
        # window_index/regressor_label), never the raw per-TR decoding_results.csv --
        # group-level stats (mean +/- SE across subjects or folds) need one value per
        # group per subject/fold, not individual trials.
        "decoding": os.path.join(base, "decoding", f"{subject}_summary_decoding_results.csv"),
    }


def has_fold_files(analysis_output_dir: str, desc: str, subject: str) -> bool:
    base = os.path.join(analysis_output_dir, desc, subject)
    return len(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_total_scores.csv"))) > 0


def fold_paths(analysis_output_dir: str, desc: str, subject: str) -> dict:
    """{fold_id: {..same keys as subject_paths' model/decoding entries..}} for every
    fold found for this subject (empty dict if test_decode_cv wasn't used)."""
    base = os.path.join(analysis_output_dir, desc, subject)
    totals = sorted(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_total_scores.csv")))
    fold_ids = [int(os.path.basename(p).split("_fold")[1].split("_")[0]) for p in totals]
    return {
        fid: {
            "model_total": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_total_scores.csv"),
            "model_auc": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_auc.csv"),
            "model_impa": os.path.join(base, "model", f"{subject}_fold{fid}_impa_native.nii.gz"),
            "decoding": os.path.join(base, "decoding", f"{subject}_fold{fid}_summary_decoding_results.csv"),
        }
        for fid in fold_ids
    }


def load_scalar_csv(path: str) -> float:
    return float(np.loadtxt(path))


def load_labeled_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def infer_categories(analysis_output_dir: str, desc: str, subjects: list) -> list:
    """Category order, read from the first available model_results_auc.csv (already
    saved with category labels by save_model_results) -- no config needed."""
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p["model_auc"]):
            return load_labeled_csv(p["model_auc"]).index.tolist()
    return []


# =====================================================
# Timecourse annotation info (best-effort -- never raises)
# =====================================================

def load_annotation_info(config_path, master_spreadsheet_path):
    """Returns (window, tr, median_duration_by_condition), any of which may be
    None/empty if the optional inputs are missing or insufficient -- annotation is
    strictly best-effort and never blocks the rest of the report."""
    if not config_path or not master_spreadsheet_path:
        return None, None, {}
    if not os.path.isfile(config_path):
        print(f"(!) --config {config_path} not found -- skipping timecourse annotation")
        return None, None, {}
    if not os.path.isfile(master_spreadsheet_path):
        print(f"(!) --master-spreadsheet {master_spreadsheet_path} not found -- skipping timecourse annotation")
        return None, None, {}

    with open(config_path) as f:
        cfg = json.load(f)
    tc_cfg = cfg.get("model_conditions", {}).get("timecourse_decoding")
    if not tc_cfg:
        print("(!) config has no model_conditions.timecourse_decoding -- skipping timecourse annotation")
        return None, None, {}

    window = tc_cfg.get("window")
    conditions = tc_cfg.get("conditions", {})

    master = pd.read_csv(
        master_spreadsheet_path, dtype={"subject": str, "session": str, "task": str, "trial_type": str}
    )
    labeled = label_rows(master, conditions)

    # median duration per condition, from one row per source event (dedupe the
    # per-volume explosion via boldfile+trial_index)
    dedup = labeled.drop_duplicates(subset=["boldfile", "trial_index"])
    median_duration = dedup.groupby("regressor_label")["duration"].median().to_dict()

    # TR derived from the data (majority across boldfiles), not hardcoded
    trs = []
    for boldfile in dedup["boldfile"].unique():
        if os.path.exists(boldfile):
            try:
                tr, _ = get_bold_header_info(boldfile)
                trs.append(tr)
            except Exception:
                pass

    if not trs:
        print("(!) could not read TR from any boldfile -- timecourse x-axis will stay in window_index units")
        return window, None, median_duration

    tr_counts = Counter(trs)
    tr = tr_counts.most_common(1)[0][0]
    if len(tr_counts) > 1:
        print(f"(!) multiple distinct TRs found across boldfiles ({dict(tr_counts)}) -- using the majority TR={tr}")

    return window, tr, median_duration


# =====================================================
# Report pages
# =====================================================

def render_title_page(pdf, desc, subjects, config_path, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    scope = f"single subject ({subjects[0]})" if len(subjects) == 1 else f"{len(subjects)} subjects"
    lines = [
        f"MVPA Report: {desc}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scope: {scope}",
    ]
    if len(subjects) > 1:
        lines.append("Subjects: " + ", ".join(subjects))
    lines.append(f"Config: {config_path or '(not provided -- timecourse annotation skipped)'}")
    lines.append(f"Output: {output_path}")
    ax.text(0.05, 0.92, "\n".join(lines), fontsize=13, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def render_accuracy_auc_page(pdf, analysis_output_dir, desc, subjects, fold_flags):
    cv_totals, model_totals, auc_by_subject = {}, {}, {}
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p["cv_total"]):
            cv_totals[s] = load_scalar_csv(p["cv_total"])
        if os.path.exists(p["model_total"]):
            model_totals[s] = load_scalar_csv(p["model_total"])
        if os.path.exists(p["model_auc"]):
            auc_by_subject[s] = load_labeled_csv(p["model_auc"]).iloc[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    # --- left: internal-CV vs held-out test accuracy ---
    ax = axes[0]
    x = np.arange(len(subjects))
    width = 0.35
    ax.bar(x - width / 2, [cv_totals.get(s, np.nan) for s in subjects], width, label="internal CV (training)")
    ax.bar(x + width / 2, [model_totals.get(s, np.nan) for s in subjects], width, label="held-out test")

    if len(subjects) == 1 and fold_flags.get(subjects[0]):
        folds = fold_paths(analysis_output_dir, desc, subjects[0])
        fold_vals = [load_scalar_csv(f["model_total"]) for f in folds.values() if os.path.exists(f["model_total"])]
        if fold_vals:
            ax.scatter([x[0] + width / 2] * len(fold_vals), fold_vals, color="black", zorder=3, s=20, label="per-fold test")

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.set_title("Accuracy: internal CV vs. held-out test")

    # --- right: per-class AUC ---
    ax = axes[1]
    if auc_by_subject:
        auc_df = pd.DataFrame(auc_by_subject)  # rows=category, cols=subject
        categories = auc_df.index.tolist()
        if len(subjects) > 1:
            ax.boxplot([auc_df.loc[c].dropna().values for c in categories], tick_labels=categories)
        else:
            ax.bar(categories, auc_df.iloc[:, 0].values)
            if fold_flags.get(subjects[0]):
                folds = fold_paths(analysis_output_dir, desc, subjects[0])
                for f in folds.values():
                    if os.path.exists(f["model_auc"]):
                        fold_auc = load_labeled_csv(f["model_auc"]).iloc[:, 0].reindex(categories)
                        ax.scatter(categories, fold_auc.values, color="black", s=15, zorder=3)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("AUC")
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    ax.set_title("Per-class AUC" + (" across subjects" if len(subjects) > 1 else ""))

    fig.suptitle(f"{desc}: accuracy & AUC", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def render_confusion_matrices_page(pdf, analysis_output_dir, desc, subjects):
    if len(subjects) == 1:
        p = subject_paths(analysis_output_dir, desc, subjects[0])
        acc = load_labeled_csv(p["model_accuracy"]) if os.path.exists(p["model_accuracy"]) else None
        evi = load_labeled_csv(p["model_evidence"]) if os.path.exists(p["model_evidence"]) else None
        title_suffix = f"subject {subjects[0]}"
    else:
        accs, evis = [], []
        for s in subjects:
            p = subject_paths(analysis_output_dir, desc, s)
            if os.path.exists(p["model_accuracy"]):
                accs.append(load_labeled_csv(p["model_accuracy"]))
            if os.path.exists(p["model_evidence"]):
                evis.append(load_labeled_csv(p["model_evidence"]))
        acc = sum(accs) / len(accs) if accs else None
        evi = sum(evis) / len(evis) if evis else None
        title_suffix = f"mean across {len(subjects)} subjects"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, name in zip(axes, [acc, evi], ["Accuracy", "Evidence"]):
        if mat is None:
            ax.axis("off")
            continue
        im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=max(1.0, float(np.nanmax(mat.values))))
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("Predicted / evidence for")
        ax.set_ylabel("True condition")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{desc}: confusion-style matrices ({title_suffix})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def render_timecourse_pages(pdf, analysis_output_dir, desc, subjects, fold_flags, window, tr, median_duration):
    frames = []
    use_fold_variability = len(subjects) == 1 and fold_flags.get(subjects[0])

    for s in subjects:
        if use_fold_variability:
            for f in fold_paths(analysis_output_dir, desc, s).values():
                if os.path.exists(f["decoding"]):
                    frames.append(pd.read_csv(f["decoding"]))
        else:
            p = subject_paths(analysis_output_dir, desc, s)
            if os.path.exists(p["decoding"]):
                frames.append(pd.read_csv(p["decoding"]))

    if not frames:
        print("(!) no summary_decoding_results.csv found for the subjects in scope -- skipping timecourse page")
        return

    combined = pd.concat(frames, ignore_index=True)
    evidence_cols = [c for c in combined.columns if c.startswith("evidence_")]
    categories = [c.replace("evidence_", "") for c in evidence_cols]
    true_conditions = sorted(combined["regressor_label"].unique())

    n_rows, n_cols = len(true_conditions), len(categories)
    if n_rows == 0 or n_cols == 0:
        print("(!) decoding_results.csv has no evidence_* columns or regressor_label values -- skipping timecourse page")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex=True, sharey=True, squeeze=False)
    variability_label = "+/- SE across folds" if use_fold_variability else "+/- SE across subjects"
    x_is_seconds = tr is not None
    x_label = "Time from window start (s)" if x_is_seconds else "window_index"

    for i, true_cond in enumerate(true_conditions):
        subset = combined[combined["regressor_label"] == true_cond]
        for j, cat in enumerate(categories):
            ax = axes[i][j]
            agg = (
                subset.groupby("window_index")[f"evidence_{cat}"]
                .agg(mean="mean", se=lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
                .reset_index()
                .sort_values("window_index")
            )
            x = agg["window_index"] * tr if x_is_seconds else agg["window_index"]
            ax.plot(x, agg["mean"], color="black", linewidth=1.5)
            ax.fill_between(x, agg["mean"] - agg["se"], agg["mean"] + agg["se"], alpha=0.25, color="black")

            if window is not None and x_is_seconds:
                dur = median_duration.get(true_cond)
                if dur is not None:
                    window_start_offset, _ = resolve_window_times(window, onset=0, duration=dur)
                    onset_mark = -window_start_offset
                    end_mark = dur - window_start_offset
                    ax.axvline(onset_mark, color="gray", linestyle="--", linewidth=0.75)
                    ax.axvline(end_mark, color="gray", linestyle=":", linewidth=0.75)
                    if i == 0:
                        ylim = ax.get_ylim()
                        ax.text(onset_mark, ylim[1], "onset", fontsize=7, ha="center", va="bottom")
                        ax.text(end_mark, ylim[1], "event end", fontsize=7, ha="center", va="bottom")

            if i == 0:
                ax.set_title(f"evidence: {cat}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"true: {true_cond}", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=9)

    fig.suptitle(f"{desc}: timecourse decoding ({variability_label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def _plot_categories_page(pdf, impa_path, title, regressor_categories):
    from nilearn import plotting

    img = nib.load(impa_path)
    data = img.get_fdata()
    n_cat = data.shape[3] if data.ndim == 4 else 1

    fig, axes = plt.subplots(n_cat, 1, figsize=(8.5, 3 * n_cat), squeeze=False)
    for c in range(n_cat):
        vol = data[..., c] if data.ndim == 4 else data
        cat_img = nib.Nifti1Image(vol, img.affine)
        label = regressor_categories[c] if c < len(regressor_categories) else f"class {c}"
        plotting.plot_stat_map(cat_img, bg_img=None, display_mode="ortho", axes=axes[c][0], title=f"{title}: {label}")
    pdf.savefig(fig)
    plt.close(fig)


def _render_fold_mosaic(pdf, fold_files: dict, mean_file: str, regressor_categories):
    mean_data = nib.load(mean_file).get_fdata()
    n_cat = mean_data.shape[3] if mean_data.ndim == 4 else 1
    fold_ids = sorted(fold_files.keys())
    n_cols = len(fold_ids) + 1

    fig, axes = plt.subplots(n_cat, n_cols, figsize=(2.2 * n_cols, 2.2 * n_cat), squeeze=False)
    for c in range(n_cat):
        mean_vol = mean_data[..., c] if mean_data.ndim == 4 else mean_data
        vmax = float(np.nanmax(np.abs(mean_vol))) or 1.0

        for col, fid in enumerate(fold_ids):
            fold_vol_full = nib.load(fold_files[fid]).get_fdata()
            vol = fold_vol_full[..., c] if fold_vol_full.ndim == 4 else fold_vol_full
            mid_z = vol.shape[2] // 2
            ax = axes[c][col]
            ax.imshow(np.rot90(vol[:, :, mid_z]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"fold {fid}", fontsize=9)

        mid_z = mean_vol.shape[2] // 2
        ax = axes[c][-1]
        ax.imshow(np.rot90(mean_vol[:, :, mid_z]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if c == 0:
            ax.set_title("mean", fontsize=9)

        label = regressor_categories[c] if c < len(regressor_categories) else f"class {c}"
        axes[c][0].set_ylabel(label, fontsize=9)

    fig.suptitle("Fold-to-fold importance map consistency (mid-axial slice)", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def render_importance_pages(pdf, analysis_output_dir, desc, subjects, fold_flags, regressor_categories):
    # Masks are native-space, per subject -- there is no common voxel grid across
    # subjects, so importance maps are never averaged across subjects (only across
    # folds, within one subject, where the grid is guaranteed shared).
    if len(subjects) > 1:
        for s in subjects:
            p = subject_paths(analysis_output_dir, desc, s)
            if os.path.exists(p["model_impa"]):
                _plot_categories_page(pdf, p["model_impa"], s, regressor_categories)
        return

    s = subjects[0]
    p = subject_paths(analysis_output_dir, desc, s)
    if os.path.exists(p["model_impa"]):
        _plot_categories_page(pdf, p["model_impa"], f"{s} (aggregated)", regressor_categories)

    if fold_flags.get(s):
        folds = fold_paths(analysis_output_dir, desc, s)
        fold_files = {fid: f["model_impa"] for fid, f in folds.items() if os.path.exists(f["model_impa"])}
        if fold_files and os.path.exists(p["model_impa"]):
            _render_fold_mosaic(pdf, fold_files, p["model_impa"], regressor_categories)


# =====================================================
# Main
# =====================================================

def main():
    args = parse_args()

    subjects = list_subject_dirs(args.analysis_output_dir, args.desc, args.subject)
    fold_flags = {s: has_fold_files(args.analysis_output_dir, args.desc, s) for s in subjects}
    regressor_categories = infer_categories(args.analysis_output_dir, args.desc, subjects)
    window, tr, median_duration = load_annotation_info(args.config, args.master_spreadsheet)

    if args.output:
        output_path = args.output
    elif len(subjects) == 1:
        output_path = os.path.join(args.analysis_output_dir, args.desc, subjects[0], f"report_{subjects[0]}.pdf")
    else:
        output_path = os.path.join(args.analysis_output_dir, args.desc, f"report_{args.desc}.pdf")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    print(f"Report scope: {len(subjects)} subject(s): {subjects}")
    with PdfPages(output_path) as pdf:
        render_title_page(pdf, args.desc, subjects, args.config, output_path)
        render_accuracy_auc_page(pdf, args.analysis_output_dir, args.desc, subjects, fold_flags)
        render_confusion_matrices_page(pdf, args.analysis_output_dir, args.desc, subjects)
        render_timecourse_pages(pdf, args.analysis_output_dir, args.desc, subjects, fold_flags, window, tr, median_duration)
        render_importance_pages(pdf, args.analysis_output_dir, args.desc, subjects, fold_flags, regressor_categories)

    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
