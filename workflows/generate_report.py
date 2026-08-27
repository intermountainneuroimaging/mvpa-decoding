#!/usr/bin/env python3

"""
Generate a PDF report (accuracy/AUC, confusion-style matrices, timecourse
decoding, importance maps) from mvpa_workflow.py's output -- either for one
subject, or aggregated across every subject found for a given classifier
("desc"). mvpa_workflow.py's k-fold cross-validation (model.kfold_cv, under
model/) and independent test-set evaluation (model_conditions.testing, under
test/) are entirely independent and shown as separate "CV"/"held-out-test" sections
throughout the report -- a subject may have either, both, or neither.
Fold-variability panels render automatically when k-fold output is present
(detected by the presence of `_fold{N}_*` files under model/).

Usage:
    # group report -- aggregates every subject found under <dir>/<desc>/*/ --
    # desc is read from --config's model.desc, same sanitization the workflow
    # scripts use, so it always matches where they actually wrote output
    python generate_report.py --analysis-output-dir ./out \\
        --config examples/config-generalization.example.json --master-spreadsheet master_spreadsheet.csv

    # single-subject report -- scoped to just <dir>/<desc>/4057/
    python generate_report.py --analysis-output-dir ./out --subject 4057 \\
        --config examples/config-generalization.example.json --master-spreadsheet master_spreadsheet.csv

    # --desc still works directly, if you'd rather not point at a config
    python generate_report.py --analysis-output-dir ./out --desc gm_valence_classifier

Exactly one of --desc/--config is required, to know which classifier's
output to read. --master-spreadsheet is always optional, and --config's
timecourse_decoding conditions/window/overlay are used for annotation
best-effort even when --desc is also given -- without --master-spreadsheet
(or without --config at all) the report still renders, just without those
annotations.
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import label_rows, get_bold_header_info, resolve_window_times, quick_safe, impa_tag


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--analysis-output-dir", required=True,
        help="Same --analysis-output-dir passed to mvpa_workflow.py -- results are read from "
             "<this>/<desc>/*/{model,test,decoding}/"
    )
    parser.add_argument(
        "--desc", default=None,
        help="Classifier folder name (model.desc, sanitized) under analysis-output-dir. Optional if "
             "--config is given -- read from the config's model.desc instead (same sanitization the "
             "workflow scripts use, so it always matches where they wrote output). One of --desc/--config "
             "is required."
    )
    parser.add_argument("--subject", default=None, help="Restrict the report to one subject (single-subject report). Omit for a group report across all subjects found.")
    parser.add_argument(
        "--config", default=None,
        help="mvpa config JSON. Supplies model.desc (see --desc) when --desc is omitted, and "
             "timecourse_decoding conditions/window/overlay for annotation either way. Optional only if "
             "--desc is given explicitly."
    )
    parser.add_argument("--master-spreadsheet", default=None, help="master_spreadsheet.csv -- needed for TR + median trial duration (timecourse annotation). Optional.")
    parser.add_argument("--output", default=None, help="Output PDF path. Defaults to <dir>/<desc>/report_<desc>.pdf (group) or <dir>/<desc>/<subject>/report_<subject>.pdf (single-subject).")
    return parser.parse_args()


def resolve_desc(desc_arg: str, config_path: str) -> str:
    """--desc if given directly; otherwise model.desc read from --config and
    sanitized via quick_safe -- the exact same value the workflow scripts
    themselves use to name their output folder, so the two can never
    silently disagree. Raises SystemExit if neither is available, or if
    --config is given but unreadable/missing model.desc."""
    if desc_arg:
        return desc_arg
    if not config_path:
        raise SystemExit("Either --desc or --config (with model.desc) is required.")
    if not os.path.isfile(config_path):
        raise SystemExit(f"--config {config_path} not found -- can't read model.desc from it.")

    with open(config_path) as f:
        cfg = json.load(f)
    model_desc = cfg.get("model", {}).get("desc")
    if not model_desc:
        raise SystemExit(f"--config {config_path} has no model.desc -- pass --desc explicitly instead.")
    return quick_safe(model_desc)


def resolve_mnispace(config_path: str) -> bool:
    """model.mnispace from --config -- False (space unknown/unasserted) when
    no config is given, the file can't be read, or the key is absent, same
    default the workflow scripts themselves use (see mvpa_common.impa_tag).
    Best-effort, like load_annotation_info -- never raises, since a bad
    config here should still let the rest of the report render."""
    if not config_path or not os.path.isfile(config_path):
        return False
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return bool(cfg.get("model", {}).get("mnispace", False))
    except (json.JSONDecodeError, OSError):
        return False


# =====================================================
# Subject-scope discovery + file layout
# =====================================================

def _subject_has_results(base: str, subject: str) -> bool:
    """model/ (k-fold) or test/ (independent test set) -- a subject can have
    either, both, or (in a training+timecourse_decoding-only run) neither,
    in which case they're not a "result" for report purposes."""
    subj_base = os.path.join(base, subject)
    return os.path.isdir(os.path.join(subj_base, "model")) or os.path.isdir(os.path.join(subj_base, "test"))


def list_subject_dirs(analysis_output_dir: str, desc: str, subject: str = None) -> list:
    base = os.path.join(analysis_output_dir, desc)
    if subject:
        if not _subject_has_results(base, subject):
            raise SystemExit(f"No results found for subject {subject!r} at {os.path.join(base, subject)}")
        return [subject]

    subjects = sorted(
        name for name in os.listdir(base)
        if _subject_has_results(base, name)
    ) if os.path.isdir(base) else []
    if not subjects:
        raise SystemExit(f"No subject result folders found under {base}")
    return subjects


def subject_paths(analysis_output_dir: str, desc: str, subject: str, mnispace: bool = False) -> dict:
    base = os.path.join(analysis_output_dir, desc, subject)
    tag = impa_tag(mnispace)
    return {
        # K-fold cross-validation, entirely within model_conditions.training
        # (mvpa_workflow.py's run_kfold) -- aggregated across folds. model/ is
        # exclusively this family's directory.
        "kfold_total": os.path.join(base, "model", f"{subject}_model_results_total_scores.csv"),
        "kfold_auc": os.path.join(base, "model", f"{subject}_model_results_auc.csv"),
        "kfold_accuracy": os.path.join(base, "model", f"{subject}_model_results_accuracy.csv"),
        "kfold_evidence": os.path.join(base, "model", f"{subject}_model_results_evidence.csv"),
        # filename tag depends on model.mnispace (see impa_tag): "impa_mni" when
        # the input BOLD/mask were confirmed-by-config to already be in MNI
        # space, plain "impa" otherwise (space left unasserted, since it isn't
        # reliably knowable from the file itself -- see render_importance_pages).
        "kfold_impa": os.path.join(base, "model", f"{subject}_{tag}.nii.gz"),
        # always "impa_mni" regardless of the mnispace argument above -- either
        # because mnispace=true made the workflow write that filename directly
        # (in which case this coincides with "kfold_impa"), or because the user
        # separately resampled the plain "impa" file via `hcp_resample.py
        # --direction native2mni --output .../model/{subject}_impa_mni.nii.gz`.
        "kfold_impa_mni": os.path.join(base, "model", f"{subject}_impa_mni.nii.gz"),
        # Independent test set: one classifier fit on the complete training
        # set, evaluated against model_conditions.testing (mvpa_workflow.py).
        # test/ is exclusively this family's directory -- same filenames as
        # the kfold family above, just under a different directory.
        "test_total": os.path.join(base, "test", f"{subject}_model_results_total_scores.csv"),
        "test_auc": os.path.join(base, "test", f"{subject}_model_results_auc.csv"),
        "test_accuracy": os.path.join(base, "test", f"{subject}_model_results_accuracy.csv"),
        "test_evidence": os.path.join(base, "test", f"{subject}_model_results_evidence.csv"),
        "test_impa": os.path.join(base, "test", f"{subject}_{tag}.nii.gz"),
        # see "kfold_impa_mni" above -- same idea, in test/. Presence of either
        # this or "kfold_impa_mni" is how render_importance_pages decides a
        # cross-subject group average is spatially valid -- see
        # resolve_group_impa_mni (held-out-test preferred, CV/kfold as fallback).
        "test_impa_mni": os.path.join(base, "test", f"{subject}_impa_mni.nii.gz"),
        # "decoding" (the pre-aggregated summary) isn't read by the timecourse page
        # itself -- it reads "decoding_raw" instead, so trial-to-trial variability
        # within each subject (never retained in the summary) is available for
        # the plot -- see summarize_raw_for_timecourse. Kept here for any other
        # consumer that wants the plain one-row-per-group summary. Always the
        # complete-training-set classifier's decoding -- never per-fold.
        "decoding": os.path.join(base, "decoding", f"{subject}_summary_decoding_results.csv"),
        "decoding_raw": os.path.join(base, "decoding", f"{subject}_decoding_results.csv"),
    }


def has_fold_files(analysis_output_dir: str, desc: str, subject: str) -> bool:
    base = os.path.join(analysis_output_dir, desc, subject)
    return len(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_total_scores.csv"))) > 0


def fold_paths(analysis_output_dir: str, desc: str, subject: str, mnispace: bool = False) -> dict:
    """{fold_id: {kfold_total/kfold_auc/kfold_impa}} for every k-fold fold found
    for this subject (empty dict if model.kfold_cv wasn't configured/run). Always
    under model/ -- k-fold is the only fold-based family; there's no such thing
    as a "test fold" since the independent test-set evaluation is a single fit."""
    base = os.path.join(analysis_output_dir, desc, subject)
    totals = sorted(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_total_scores.csv")))
    fold_ids = [int(os.path.basename(p).split("_fold")[1].split("_")[0]) for p in totals]
    tag = impa_tag(mnispace)
    return {
        fid: {
            "kfold_total": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_total_scores.csv"),
            "kfold_auc": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_auc.csv"),
            "kfold_impa": os.path.join(base, "model", f"{subject}_fold{fid}_{tag}.nii.gz"),
        }
        for fid in fold_ids
    }


def load_scalar_csv(path: str) -> float:
    return float(np.loadtxt(path))


def load_labeled_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def infer_categories(analysis_output_dir: str, desc: str, subjects: list) -> list:
    """Category order, read from the first available model_results_auc.csv (already
    saved with category labels by save_model_results) -- no config needed. Checks
    the CV (kfold) family first, then held-out-test -- either is equally valid,
    since regressor_categories is shared across both."""
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        for key in ("kfold_auc", "test_auc"):
            if os.path.exists(p[key]):
                return load_labeled_csv(p[key]).index.tolist()
    return []


def compile_group_summary(analysis_output_dir: str, desc: str, subjects: list) -> pd.DataFrame:
    """One row per (subject, family) with every scalar/vector/matrix metric
    mvpa_workflow.py produced for that family flattened into its own column --
    total_accuracy, auc_<category>, accuracy_<true>_<pred>, evidence_<true>_<pred>
    -- so the whole group's numbers live in one spreadsheet instead of scattered
    across each subject's own model/test CSVs. A subject missing a family (no
    model.kfold_cv, or no model_conditions.testing) simply contributes no row
    for that family; a subject/category set that differs from the rest just
    leaves the mismatched columns blank for that row (pandas fills NaN)."""
    rows = []
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        for family, prefix in (("CV", "kfold"), ("held-out-test", "test")):
            total_key, auc_key, acc_key, evi_key = f"{prefix}_total", f"{prefix}_auc", f"{prefix}_accuracy", f"{prefix}_evidence"
            if not os.path.exists(p[total_key]):
                continue
            row = {"subject": s, "family": family, "total_accuracy": load_scalar_csv(p[total_key])}
            if os.path.exists(p[auc_key]):
                for cat, val in load_labeled_csv(p[auc_key]).iloc[:, 0].items():
                    row[f"auc_{cat}"] = val
            if os.path.exists(p[acc_key]):
                acc = load_labeled_csv(p[acc_key])
                for true_cat in acc.index:
                    for pred_cat in acc.columns:
                        row[f"accuracy_{true_cat}_{pred_cat}"] = acc.loc[true_cat, pred_cat]
            if os.path.exists(p[evi_key]):
                evi = load_labeled_csv(p[evi_key])
                for true_cat in evi.index:
                    for pred_cat in evi.columns:
                        row[f"evidence_{true_cat}_{pred_cat}"] = evi.loc[true_cat, pred_cat]
            rows.append(row)
    return pd.DataFrame(rows)


# =====================================================
# Timecourse annotation info (best-effort -- never raises)
# =====================================================

def load_annotation_info(config_path, master_spreadsheet_path):
    """Returns (window, tr, median_duration_by_condition, overlay_conditions), any
    of which may be None/empty if the optional inputs are missing or insufficient --
    annotation is strictly best-effort and never blocks the rest of the report.
    overlay_conditions is model_conditions.timecourse_decoding.overlay verbatim
    (empty dict if absent) -- see summarize_raw_with_overlay for how it's used."""
    if not config_path or not master_spreadsheet_path:
        return None, None, {}, {}
    if not os.path.isfile(config_path):
        print(f"(!) --config {config_path} not found -- skipping timecourse annotation")
        return None, None, {}, {}
    if not os.path.isfile(master_spreadsheet_path):
        print(f"(!) --master-spreadsheet {master_spreadsheet_path} not found -- skipping timecourse annotation")
        return None, None, {}, {}

    with open(config_path) as f:
        cfg = json.load(f)
    tc_cfg = cfg.get("model_conditions", {}).get("timecourse_decoding")
    if not tc_cfg:
        print("(!) config has no model_conditions.timecourse_decoding -- skipping timecourse annotation")
        return None, None, {}, {}

    window = tc_cfg.get("window")
    conditions = tc_cfg.get("conditions", {})
    overlay_conditions = tc_cfg.get("overlay", {})

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
        return window, None, median_duration, overlay_conditions

    tr_counts = Counter(trs)
    tr = tr_counts.most_common(1)[0][0]
    if len(tr_counts) > 1:
        print(f"(!) multiple distinct TRs found across boldfiles ({dict(tr_counts)}) -- using the majority TR={tr}")

    return window, tr, median_duration, overlay_conditions


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


def _draw_chance_line(ax, level):
    """Dashed reference line + label, shared by the accuracy and AUC panels below --
    the two have different chance levels (1/n_categories vs. always 0.5, see
    render_accuracy_auc_page) but the same visual treatment."""
    ax.axhline(level, linestyle="--", color="gray", linewidth=1)
    ax.text(0.98, level, "chance level", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="gray")


def render_accuracy_auc_page(pdf, analysis_output_dir, desc, subjects, fold_flags, regressor_categories):
    """CV (k-fold, entirely within model_conditions.training) and held-out-test (the
    complete-training-set classifier evaluated against model_conditions.testing)
    are independent -- a subject/group may have either, both, or neither. Both
    panels render whichever families have data, side by side when both exist."""
    kfold_totals, test_totals = {}, {}
    kfold_auc_by_subject, test_auc_by_subject = {}, {}
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p["kfold_total"]):
            kfold_totals[s] = load_scalar_csv(p["kfold_total"])
        if os.path.exists(p["test_total"]):
            test_totals[s] = load_scalar_csv(p["test_total"])
        if os.path.exists(p["kfold_auc"]):
            kfold_auc_by_subject[s] = load_labeled_csv(p["kfold_auc"]).iloc[:, 0]
        if os.path.exists(p["test_auc"]):
            test_auc_by_subject[s] = load_labeled_csv(p["test_auc"]).iloc[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    # --- left: accuracy ---
    ax = axes[0]
    families = []
    if kfold_totals:
        families.append(("CV", kfold_totals, "C0"))
    if test_totals:
        families.append(("held-out-test", test_totals, "C1"))

    if not families:
        ax.axis("off")
    elif len(subjects) > 1:
        # group report -- one bar per family (mean across subjects), with each
        # subject's own value scattered on top (beeswarm-style, deterministic
        # spread rather than random jitter so the figure is reproducible)
        labels_acc = [name for name, _, _ in families]
        per_family_vals = [[vals[s] for s in subjects if s in vals] for _, vals, _ in families]
        bar_colors = [c for _, _, c in families]
        means = [np.mean(v) if v else np.nan for v in per_family_vals]
        ax.bar(labels_acc, means, color=bar_colors, alpha=0.7, zorder=1)

        for xi, vals in enumerate(per_family_vals):
            if not vals:
                continue
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(xi + jitter, vals, color="black", zorder=3, s=20)

        ax.set_title("Accuracy: " + " vs. ".join(labels_acc) + "\n(bars = mean, dots = individual subjects)")
    else:
        s = subjects[0]
        x = np.arange(len(families))
        width = 0.5
        ax.bar(x, [vals.get(s, np.nan) for _, vals, _ in families], width, color=[c for _, _, c in families])

        if kfold_totals:
            folds = fold_paths(analysis_output_dir, desc, s)
            fold_vals = [load_scalar_csv(f["kfold_total"]) for f in folds.values() if os.path.exists(f["kfold_total"])]
            if fold_vals:
                cv_x = [name for name, _, _ in families].index("CV")
                ax.scatter([cv_x] * len(fold_vals), fold_vals, color="black", zorder=3, s=20, label="per-fold")
                ax.legend(fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([name for name, _, _ in families])
        ax.set_title("Accuracy: " + " vs. ".join(name for name, _, _ in families))

    ax.set_ylabel("Accuracy")
    if regressor_categories and families:
        _draw_chance_line(ax, 1.0 / len(regressor_categories))

    # --- right: per-class AUC ---
    ax = axes[1]
    auc_families = []
    if kfold_auc_by_subject:
        auc_families.append(("CV", pd.DataFrame(kfold_auc_by_subject), "C0"))
    if test_auc_by_subject:
        auc_families.append(("held-out-test", pd.DataFrame(test_auc_by_subject), "C1"))

    if not auc_families:
        ax.axis("off")
    else:
        categories = auc_families[0][1].index.tolist()
        n_fam = len(auc_families)
        width = 0.8 / n_fam
        for fi, (name, auc_df, color) in enumerate(auc_families):
            offset = (fi - (n_fam - 1) / 2) * width
            positions = np.arange(len(categories)) + offset
            if len(subjects) > 1:
                bp = ax.boxplot([auc_df.loc[c].dropna().values for c in categories],
                                 positions=positions, widths=width * 0.9, patch_artist=True)
                for box in bp["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(0.6)
            else:
                s = subjects[0]
                ax.bar(positions, auc_df.reindex(categories).iloc[:, 0].values, width * 0.9, color=color, label=name)
                if name == "CV" and fold_flags.get(s):
                    folds = fold_paths(analysis_output_dir, desc, s)
                    for f in folds.values():
                        if os.path.exists(f["kfold_auc"]):
                            fold_auc = load_labeled_csv(f["kfold_auc"]).iloc[:, 0].reindex(categories)
                            ax.scatter(positions, fold_auc.values, color="black", s=15, zorder=3)
        if len(subjects) == 1:
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylabel("AUC")
        _draw_chance_line(ax, 0.5)

    auc_title = "Per-class AUC" + (" across subjects" if len(subjects) > 1 else "")
    if len(auc_families) > 1:
        auc_title += " (CV vs. held-out-test)"
    ax.set_title(auc_title)

    fig.suptitle(f"{desc}: accuracy & AUC", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def _load_confusion_family(analysis_output_dir, desc, subjects, acc_key, evi_key):
    """(accuracy_matrix, evidence_matrix) for one family (CV or held-out-test) -- the
    single subject's own matrices, or the mean across subjects for a group
    report. Either/both may be None if that family has no files at all."""
    if len(subjects) == 1:
        p = subject_paths(analysis_output_dir, desc, subjects[0])
        acc = load_labeled_csv(p[acc_key]) if os.path.exists(p[acc_key]) else None
        evi = load_labeled_csv(p[evi_key]) if os.path.exists(p[evi_key]) else None
        return acc, evi

    accs, evis = [], []
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p[acc_key]):
            accs.append(load_labeled_csv(p[acc_key]))
        if os.path.exists(p[evi_key]):
            evis.append(load_labeled_csv(p[evi_key]))
    acc = sum(accs) / len(accs) if accs else None
    evi = sum(evis) / len(evis) if evis else None
    return acc, evi


def render_confusion_matrices_page(pdf, analysis_output_dir, desc, subjects):
    """CV (k-fold) and held-out-test (independent test set) each get their own row --
    a row is omitted entirely (not left blank) when that family has no files
    for any subject in scope. Every populated cell is labeled with its value
    to 2 decimal places."""
    title_suffix = f"subject {subjects[0]}" if len(subjects) == 1 else f"mean across {len(subjects)} subjects"

    families = []
    kfold_acc, kfold_evi = _load_confusion_family(analysis_output_dir, desc, subjects, "kfold_accuracy", "kfold_evidence")
    if kfold_acc is not None or kfold_evi is not None:
        families.append(("CV", kfold_acc, kfold_evi))
    test_acc, test_evi = _load_confusion_family(analysis_output_dir, desc, subjects, "test_accuracy", "test_evidence")
    if test_acc is not None or test_evi is not None:
        families.append(("held-out-test", test_acc, test_evi))

    if not families:
        return

    fig, axes = plt.subplots(len(families), 2, figsize=(11, 5 * len(families)), squeeze=False)
    for row, (family_label, acc, evi) in enumerate(families):
        for col, (mat, name) in enumerate([(acc, "Accuracy"), (evi, "Evidence")]):
            ax = axes[row][col]
            if mat is None:
                ax.axis("off")
                continue
            values = mat.values
            im = ax.imshow(values, cmap="viridis", vmin=0, vmax=max(1.0, float(np.nanmax(values))))
            ax.set_xticks(range(len(mat.columns)))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index)
            ax.set_xlabel("Predicted / evidence for")
            ax.set_ylabel("True condition")
            ax.set_title(f"{family_label}: {name}")
            fig.colorbar(im, ax=ax, fraction=0.046)

            for yi in range(values.shape[0]):
                for xi in range(values.shape[1]):
                    val = values[yi, xi]
                    if np.isnan(val):
                        continue
                    # light text on dark cells, dark text on light cells --
                    # im.norm(val) is this cell's position in [vmin, vmax]
                    text_color = "white" if im.norm(val) < 0.6 else "black"
                    ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.suptitle(f"{desc}: confusion-style matrices ({title_suffix})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def summarize_raw_for_timecourse(raw_df: pd.DataFrame, overlay_conditions: dict = None) -> pd.DataFrame:
    """One row per (window_index, regressor_label[, overlay_label]) -- for each
    evidence_* column, both the mean *and* the trial-to-trial standard error
    (std across the trials sharing that group, within this one subject's/fold's
    own raw rows, /sqrt(n)), suffixed evidence_*_se. Computed directly from the
    raw per-TR file rather than the pre-aggregated summary CSV specifically so
    this trial-level spread is available at all -- the summary only ever kept
    the mean (mvpa_common.summarize_decoding() has no equivalent). When
    overlay_conditions is given, rows are additionally tagged via label_rows
    (dropping unmatched rows, count printed) and grouped by overlay_label as
    an extra key."""
    df = raw_df
    group_cols = ["window_index", "regressor_label"]
    if overlay_conditions:
        df = label_rows(df, overlay_conditions, label_column="overlay_label")
        dropped = len(raw_df) - len(df)
        if dropped:
            print(f"  (!) {dropped} row(s) matched no overlay condition -- dropped from the timecourse plot")
        group_cols = group_cols + ["overlay_label"]

    evidence_cols = [c for c in df.columns if c.startswith("evidence_")]
    grouped = df.groupby(group_cols)[evidence_cols]
    means = grouped.mean()
    ses = grouped.agg(lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0).add_suffix("_se")
    return pd.concat([means, ses], axis=1).reset_index()


def render_timecourse_pages(pdf, analysis_output_dir, desc, subjects, window, tr, median_duration,
                             overlay_conditions=None):
    """Timecourse decoding always comes from the complete-training-set
    classifier (mvpa_workflow.py never runs it per-fold), so there's exactly
    one decoding_raw file per subject regardless of whether model.kfold_cv
    is also configured -- no fold-level variability source anymore."""
    overlay_conditions = overlay_conditions or {}
    frames = []

    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p["decoding_raw"]):
            frames.append(summarize_raw_for_timecourse(pd.read_csv(p["decoding_raw"]), overlay_conditions))

    if not frames:
        print("(!) no decoding_results.csv found for the subjects in scope -- skipping timecourse page")
        return

    combined = pd.concat(frames, ignore_index=True)
    evidence_cols = [c for c in combined.columns if c.startswith("evidence_") and not c.endswith("_se")]
    categories = [c.replace("evidence_", "") for c in evidence_cols]
    true_conditions = sorted(combined["regressor_label"].unique())
    overlay_categories = sorted(combined["overlay_label"].unique()) if overlay_conditions else [None]

    n_rows, n_cols = len(true_conditions), len(categories)
    if n_rows == 0 or n_cols == 0:
        print("(!) decoding_results.csv has no evidence_* columns or regressor_label values -- skipping timecourse page")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharex=True, sharey=True, squeeze=False)
    # zero-width for a single-subject report -- nothing to average across when
    # there's only one subject's own decoding
    variability_label = "darker band: +/- SE across subjects; lighter band: +/- trial-to-trial SE"
    x_is_seconds = tr is not None
    x_label = "Time from window start (s)" if x_is_seconds else "window_index"
    overlay_colors = plt.get_cmap("tab10").colors

    for i, true_cond in enumerate(true_conditions):
        subset = combined[combined["regressor_label"] == true_cond]
        for j, cat in enumerate(categories):
            ax = axes[i][j]
            for k, overlay_cat in enumerate(overlay_categories):
                line = subset if overlay_cat is None else subset[subset["overlay_label"] == overlay_cat]
                color = "black" if overlay_cat is None else overlay_colors[k % len(overlay_colors)]
                agg = (
                    line.groupby("window_index").agg(
                        mean=(f"evidence_{cat}", "mean"),
                        se=(f"evidence_{cat}", lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0),
                        # mean of each subject's/fold's own within-group trial-to-trial
                        # SE -- distinct from `se` above (SE *across* subjects/folds)
                        trial_se=(f"evidence_{cat}_se", "mean"),
                    )
                    .reset_index()
                    .sort_values("window_index")
                )
                x = agg["window_index"] * tr if x_is_seconds else agg["window_index"]
                plot_kwargs = {"label": overlay_cat} if overlay_cat is not None else {}
                # lighter/wider trial-to-trial band drawn first (behind), then the
                # more opaque subject/fold-level band on top, then the mean line --
                # keeps both regions individually legible even where they overlap
                ax.fill_between(x, agg["mean"] - agg["trial_se"], agg["mean"] + agg["trial_se"],
                                 alpha=0.12, color=color, zorder=1)
                ax.fill_between(x, agg["mean"] - agg["se"], agg["mean"] + agg["se"],
                                 alpha=0.25, color=color, zorder=2)
                ax.plot(x, agg["mean"], color=color, linewidth=1.5, zorder=3, **plot_kwargs)

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

    handles, labels = axes[0][0].get_legend_handles_labels()
    trial_se_proxy = Patch(facecolor="black", alpha=0.12)
    handles = handles + [trial_se_proxy]
    labels = labels + ["trial-to-trial SE"]
    fig.legend(handles, labels, loc="outside upper right", fontsize=8, title="overlay" if overlay_conditions else None)

    fig.suptitle(f"{desc}: timecourse decoding\n({variability_label})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def _plot_categories_page(pdf, impa, title, regressor_categories, mosaic=False, mnispace=False):
    """impa is either a path to a NIfTI file, or an already-loaded/in-memory
    nibabel image (e.g. a group-average built without ever touching disk).

    mnispace controls whether nilearn draws its own bundled MNI152 template
    underneath as anatomical context (bg_img left at its default) or omits
    it entirely (bg_img=None): true only when the map is actually known to
    be in MNI space, so the overlay is never shown misleadingly registered
    against a template it doesn't really share a grid with. mosaic=True
    (only ever called with confirmed-MNI data -- see render_importance_pages'
    group-mean branch) always shows the template regardless of this flag,
    since that caller has already established the map is in MNI space by
    construction (it's built from model_impa_mni files).

    mosaic=True switches from a single ortho (3-slice) cut to nilearn's
    "mosaic" display -- many tiled slices, much more of the map's spatial
    extent visible at once. A mosaic is much taller than one ortho row, so
    (unlike the ortho case, which stacks every category into one figure)
    each category gets its own page -- otherwise categories would overlap
    into an illegible mess trying to share one page's worth of vertical
    space."""
    from nilearn import plotting

    img = nib.load(impa) if isinstance(impa, (str, os.PathLike)) else impa
    data = img.get_fdata()
    n_cat = data.shape[3] if data.ndim == 4 else 1

    if mosaic:
        for c in range(n_cat):
            vol = data[..., c] if data.ndim == 4 else data
            cat_img = nib.Nifti1Image(vol, img.affine)
            label = regressor_categories[c] if c < len(regressor_categories) else f"class {c}"
            fig = plt.figure(figsize=(11, 6))
            plotting.plot_stat_map(cat_img, display_mode="mosaic", figure=fig, title=f"{title}: {label}", draw_cross=False)
            pdf.savefig(fig)
            plt.close(fig)
        return

    # omit bg_img entirely (nilearn's own MNI152 default) when mnispace, pass
    # None (no background at all) otherwise -- see docstring above
    bg_kwargs = {} if mnispace else {"bg_img": None}
    fig, axes = plt.subplots(n_cat, 1, figsize=(8.5, 3 * n_cat), squeeze=False)
    for c in range(n_cat):
        vol = data[..., c] if data.ndim == 4 else data
        cat_img = nib.Nifti1Image(vol, img.affine)
        label = regressor_categories[c] if c < len(regressor_categories) else f"class {c}"
        plotting.plot_stat_map(cat_img, display_mode="ortho", axes=axes[c][0], title=f"{title}: {label}",
                                draw_cross=False, **bg_kwargs)
    pdf.savefig(fig)
    plt.close(fig)


def _render_fold_mosaic(pdf, fold_files: dict, mean_file: str, regressor_categories, mnispace=False):
    """One row per category, one column per fold plus a trailing "mean" column --
    same mid-axial slice (by world-space z coordinate, not just a matching array
    index, so it's the same anatomical location even if a fold's array happens to
    differ in shape) across every panel for direct visual comparison. Uses
    nilearn's plot_stat_map (single z-cut) rather than a raw imshow so it gets the
    same MNI152 background as the ortho/mosaic pages above when mnispace confirms
    the map is actually in MNI space -- see _plot_categories_page's docstring for
    why that's conditional rather than always-on."""
    from nilearn import plotting

    bg_kwargs = {} if mnispace else {"bg_img": None}

    mean_img = nib.load(mean_file)
    mean_data = mean_img.get_fdata()
    n_cat = mean_data.shape[3] if mean_data.ndim == 4 else 1
    fold_ids = sorted(fold_files.keys())
    n_cols = len(fold_ids) + 1

    fig, axes = plt.subplots(n_cat, n_cols, figsize=(2.2 * n_cols, 2.2 * n_cat), squeeze=False)
    for c in range(n_cat):
        # captured before plot_stat_map touches axes[c][0] below -- nilearn draws
        # its own child axes over/instead of the one we hand it, which silently
        # drops any set_ylabel() applied afterward (set_title survives, ylabel
        # doesn't -- see the fig.text() placement at the end of this loop, which
        # sidesteps the problem entirely by not depending on that axes object)
        row_position = axes[c][0].get_position()

        mean_vol = mean_data[..., c] if mean_data.ndim == 4 else mean_data
        vmax = float(np.nanmax(np.abs(mean_vol))) or 1.0
        mid_z_index = mean_vol.shape[2] // 2
        z_mm = (mean_img.affine @ np.array([0, 0, mid_z_index, 1]))[2]

        for col, fid in enumerate(fold_ids):
            fold_img = nib.load(fold_files[fid])
            fold_vol_full = fold_img.get_fdata()
            vol = fold_vol_full[..., c] if fold_vol_full.ndim == 4 else fold_vol_full
            cat_img = nib.Nifti1Image(vol, fold_img.affine)
            ax = axes[c][col]
            plotting.plot_stat_map(cat_img, display_mode="z", cut_coords=[z_mm], axes=ax, vmax=vmax,
                                    colorbar=False, draw_cross=False, annotate=False, **bg_kwargs)
            if c == 0:
                ax.set_title(f"fold {fid}", fontsize=9)

        mean_cat_img = nib.Nifti1Image(mean_vol, mean_img.affine)
        ax = axes[c][-1]
        plotting.plot_stat_map(mean_cat_img, display_mode="z", cut_coords=[z_mm], axes=ax, vmax=vmax,
                                colorbar=False, draw_cross=False, annotate=False, **bg_kwargs)
        if c == 0:
            ax.set_title("mean", fontsize=9)

        label = regressor_categories[c] if c < len(regressor_categories) else f"class {c}"
        fig.text(0.02, (row_position.y0 + row_position.y1) / 2, label, fontsize=9,
                  rotation=90, va="center", ha="center")

    fig.suptitle("Fold-to-fold importance map consistency (mid-axial slice)", fontsize=12)
    pdf.savefig(fig)
    plt.close(fig)


def resolve_group_impa_mni(analysis_output_dir: str, desc: str, subjects: list) -> tuple:
    """Which subjects have an MNI-registered importance map, preferring the
    held-out-test family's test_impa_mni and falling back to the CV (k-fold)
    family's kfold_impa_mni only when held-out-test isn't available for that
    subject -- neither is ever written by the workflow script itself unless
    model.mnispace is set, otherwise only by the user separately running
    `hcp_resample.py --direction native2mni` on the corresponding plain impa
    file. Returns ({subject: (path, family_label)}, [subjects missing both])
    -- pure path-existence check, no image I/O; shape compatibility is
    checked separately at load time."""
    available, missing = {}, []
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        if os.path.exists(p["test_impa_mni"]):
            available[s] = (p["test_impa_mni"], "held-out-test")
        elif os.path.exists(p["kfold_impa_mni"]):
            available[s] = (p["kfold_impa_mni"], "CV")
        else:
            missing.append(s)
    return available, missing


def render_importance_pages(pdf, analysis_output_dir, desc, subjects, fold_flags, regressor_categories, output_dir,
                             mnispace=False):
    """A subject can have two independent importance-map families now: "CV"
    (kfold_impa -- the mean importance map across every k-fold fold's own fit,
    each fold trained on a different subset of runs, broken out fold-by-fold in
    the mosaic below) and "held-out-test" (test_impa -- the one classifier fit
    on the complete training set, whose weights are also what gets evaluated
    against model_conditions.testing when that's configured). Either, both, or
    neither may exist for a subject depending on what model.kfold_cv/
    model_conditions.testing were configured.

    Neither family's space is asserted/known (whatever the input BOLD/mask
    happened to be in), so subjects aren't guaranteed to share a common voxel
    grid -- per-subject maps are therefore never averaged across subjects
    (only across folds, within one subject's CV family, where the grid is
    guaranteed shared) -- see resolve_group_impa_mni for the one exception: if
    held-out-test (or, failing that, CV) has been separately resampled into
    MNI space (shared grid) for enough subjects, a real group average becomes
    possible."""
    if len(subjects) > 1:
        mni_paths, missing = resolve_group_impa_mni(analysis_output_dir, desc, subjects)
        if mni_paths:
            if missing:
                print(f"  (!) {len(missing)} subject(s) have no MNI-registered importance map "
                      f"({', '.join(missing)}) -- group map averaged across the remaining {len(mni_paths)}")
            n_cv_fallback = sum(1 for _, fam in mni_paths.values() if fam == "CV")
            if n_cv_fallback:
                print(f"  (!) {n_cv_fallback} subject(s) had no held-out-test MNI map -- used their CV "
                      f"(k-fold) MNI map for the group average instead")

            imgs = {s: nib.load(path) for s, (path, _) in mni_paths.items()}
            ref_subject, ref_img = next(iter(imgs.items()))
            mismatched = [s for s, img in imgs.items() if img.shape != ref_img.shape]
            if mismatched:
                print(f"  (!) {len(mismatched)} subject(s) MNI importance map shape doesn't match "
                      f"the reference grid ({ref_subject}'s {ref_img.shape}): {', '.join(mismatched)} -- "
                      f"excluded from the group average")

            included = [s for s in imgs if s not in mismatched]
            mean_data = np.mean([imgs[s].get_fdata() for s in included], axis=0)
            mean_img = nib.Nifti1Image(mean_data, ref_img.affine)

            # save alongside the PDF -- the plotted page is a quick look, this
            # is the actual data for anyone who wants to load it elsewhere
            # (e.g. a group-level stats tool, or a different viewer/threshold)
            saved_path = os.path.join(output_dir, f"{desc}_group_mean_impa_mni.nii.gz")
            nib.save(mean_img, saved_path)
            print(f"  Saved group mean importance map: {saved_path}")

            _plot_categories_page(pdf, mean_img, f"group mean, MNI space (n={len(included)} subjects)",
                                   regressor_categories, mosaic=True)
            return

        print("  (!) no subject has an MNI-registered importance map -- falling back to per-subject maps "
              "(space not asserted), which aren't guaranteed spatially comparable across subjects. Resample "
              "each subject's impa via `hcp_resample.py --direction native2mni` to enable a group map.")
        for s in subjects:
            p = subject_paths(analysis_output_dir, desc, s, mnispace=mnispace)
            if os.path.exists(p["kfold_impa"]):
                _plot_categories_page(pdf, p["kfold_impa"], f"{s} (CV, aggregated across folds)",
                                       regressor_categories, mnispace=mnispace)
            if os.path.exists(p["test_impa"]):
                _plot_categories_page(pdf, p["test_impa"], f"{s} (held-out-test, full training set)",
                                       regressor_categories, mnispace=mnispace)
        return

    s = subjects[0]
    p = subject_paths(analysis_output_dir, desc, s, mnispace=mnispace)

    if os.path.exists(p["kfold_impa"]):
        _plot_categories_page(pdf, p["kfold_impa"], f"{s} (CV, aggregated across folds)",
                               regressor_categories, mnispace=mnispace)
        if fold_flags.get(s):
            folds = fold_paths(analysis_output_dir, desc, s, mnispace=mnispace)
            fold_files = {fid: f["kfold_impa"] for fid, f in folds.items() if os.path.exists(f["kfold_impa"])}
            if fold_files:
                _render_fold_mosaic(pdf, fold_files, p["kfold_impa"], regressor_categories, mnispace=mnispace)

    if os.path.exists(p["test_impa"]):
        _plot_categories_page(pdf, p["test_impa"], f"{s} (held-out-test, full training set)",
                               regressor_categories, mnispace=mnispace)


# =====================================================
# Main
# =====================================================

def main():
    args = parse_args()
    desc = resolve_desc(args.desc, args.config)

    subjects = list_subject_dirs(args.analysis_output_dir, desc, args.subject)
    fold_flags = {s: has_fold_files(args.analysis_output_dir, desc, s) for s in subjects}
    regressor_categories = infer_categories(args.analysis_output_dir, desc, subjects)
    window, tr, median_duration, overlay_conditions = load_annotation_info(args.config, args.master_spreadsheet)
    mnispace = resolve_mnispace(args.config)

    if args.output:
        output_path = args.output
    elif len(subjects) == 1:
        output_path = os.path.join(args.analysis_output_dir, desc, subjects[0], f"report_{subjects[0]}.pdf")
    else:
        output_path = os.path.join(args.analysis_output_dir, desc, f"report_{desc}.pdf")
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    print(f"Report scope: {len(subjects)} subject(s): {subjects}")
    with PdfPages(output_path) as pdf:
        render_title_page(pdf, desc, subjects, args.config, output_path)
        render_accuracy_auc_page(pdf, args.analysis_output_dir, desc, subjects, fold_flags, regressor_categories)
        render_confusion_matrices_page(pdf, args.analysis_output_dir, desc, subjects)
        render_timecourse_pages(pdf, args.analysis_output_dir, desc, subjects, window, tr, median_duration,
                                 overlay_conditions)
        render_importance_pages(pdf, args.analysis_output_dir, desc, subjects, fold_flags, regressor_categories,
                                 os.path.dirname(output_path), mnispace=mnispace)

    print(f"Report written to: {output_path}")

    if len(subjects) > 1:
        summary = compile_group_summary(args.analysis_output_dir, desc, subjects)
        summary_path = os.path.join(os.path.dirname(output_path), f"{desc}_group_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Group summary spreadsheet saved to: {summary_path}")


if __name__ == "__main__":
    main()
