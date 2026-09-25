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
timecourse_decoding trial_start_event/conditions/overlay are used for
annotation best-effort even when --desc is also given -- without
--master-spreadsheet (or without --config at all) the report still renders,
just without those per-event timing annotations.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import label_rows, label_rows_optional, get_bold_header_info, quick_safe, impa_tag, partition_into_trials, qualifying_boldfiles


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
        "--subjects", default=None,
        help="Restrict a group report to just these subjects, instead of every subject folder found under "
             "analysis-output-dir/desc. Either a comma-separated list (e.g. \"1,2,3\") or a path to a text "
             "file listing subject IDs (one per line and/or comma-separated -- whichever's convenient to "
             "generate). Still a group report (unlike --subject, singular) -- ignored if --subject is also given."
    )
    parser.add_argument(
        "--config", default=None,
        help="mvpa config JSON. Supplies model.desc (see --desc) when --desc is omitted, and "
             "timecourse_decoding trial_start_event/conditions/overlay for annotation either way. Optional "
             "only if --desc is given explicitly."
    )
    parser.add_argument(
        "--master-spreadsheet", default=None,
        help="master_spreadsheet.csv produced by generate_master_spreadsheet.py -- needed to annotate the "
             "timecourse page with real per-trial event timing (onset/duration of every real event type "
             "observed inside a trial). Optional; without it the timecourse page still renders, just "
             "without those annotations."
    )
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


def parse_subjects_arg(value: str) -> list:
    """--subjects accepts either a literal comma-separated list ("1,2,3") or
    a path to a text file listing subject IDs -- one per line, comma-separated
    on one line, or a mix of both (whichever's more convenient to generate),
    with blank lines and surrounding whitespace ignored either way."""
    text = open(value).read() if os.path.isfile(value) else value
    return [s.strip() for s in re.split(r"[,\n]", text) if s.strip()]


def list_subject_dirs(analysis_output_dir: str, desc: str, subject: str = None, subjects: list = None) -> list:
    """`subject` (singular) takes priority -- a single-subject report, exactly
    as before `subjects` existed. Otherwise `subjects` (plural), if given,
    restricts a *group* report to just that explicit list (each one checked
    for results, same as `subject` is) instead of auto-discovering every
    subject folder under analysis_output_dir/desc."""
    base = os.path.join(analysis_output_dir, desc)
    if subject:
        if not _subject_has_results(base, subject):
            raise SystemExit(f"No results found for subject {subject!r} at {os.path.join(base, subject)}")
        return [subject]

    if subjects:
        missing = [s for s in subjects if not _subject_has_results(base, s)]
        if missing:
            raise SystemExit(f"No results found for subject(s) {missing!r} under {base}")
        return list(subjects)

    found = sorted(
        name for name in os.listdir(base)
        if _subject_has_results(base, name)
    ) if os.path.isdir(base) else []
    if not found:
        raise SystemExit(f"No subject result folders found under {base}")
    return found


def subject_paths(analysis_output_dir: str, desc: str, subject: str, mnispace: bool = False) -> dict:
    base = os.path.join(analysis_output_dir, desc, subject)
    tag = impa_tag(mnispace)
    return {
        # K-fold cross-validation, entirely within model_conditions.training
        # (mvpa_workflow.py's run_kfold) -- aggregated across folds. model/ is
        # exclusively this family's directory.
        # "metadata" is total_scores/whole_voxels/selected_voxels/feature_percent
        # (every plain-scalar metric save_model_results wrote) collected into one
        # small CSV instead of 4 near-empty single-line files -- see load_metadata_csv.
        "kfold_metadata": os.path.join(base, "model", f"{subject}_model_results_metadata.csv"),
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
        # raw, one row per held-out sample across all k-fold folds (task,
        # trial_type, run, predicted_label, correct, evidence_<category>) --
        # see build_cv_raw_results/run_kfold. Absent if model.kfold_cv wasn't
        # configured.
        "kfold_cv_raw": os.path.join(base, "model", f"{subject}_cv_results.csv"),
        # Independent test set: one classifier fit on the complete training
        # set, evaluated against model_conditions.testing (mvpa_workflow.py).
        # test/ is exclusively this family's directory -- same filenames as
        # the kfold family above, just under a different directory.
        "test_metadata": os.path.join(base, "test", f"{subject}_model_results_metadata.csv"),
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
        # written by mvpa_workflow.py's double-dipping guard only when it
        # actually found training/testing or training/timecourse boldfile
        # overlap for this subject -- see render_double_dipping_page.
        "double_dipping_report": os.path.join(base, f"{subject}_double_dipping_report.json"),
    }


def has_fold_files(analysis_output_dir: str, desc: str, subject: str) -> bool:
    base = os.path.join(analysis_output_dir, desc, subject)
    return len(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_metadata.csv"))) > 0


def fold_paths(analysis_output_dir: str, desc: str, subject: str, mnispace: bool = False) -> dict:
    """{fold_id: {kfold_metadata/kfold_auc/kfold_impa}} for every k-fold fold found
    for this subject (empty dict if model.kfold_cv wasn't configured/run). Always
    under model/ -- k-fold is the only fold-based family; there's no such thing
    as a "test fold" since the independent test-set evaluation is a single fit."""
    base = os.path.join(analysis_output_dir, desc, subject)
    metadata_files = sorted(glob.glob(os.path.join(base, "model", f"{subject}_fold*_model_results_metadata.csv")))
    fold_ids = [int(os.path.basename(p).split("_fold")[1].split("_")[0]) for p in metadata_files]
    tag = impa_tag(mnispace)
    return {
        fid: {
            "kfold_metadata": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_metadata.csv"),
            "kfold_auc": os.path.join(base, "model", f"{subject}_fold{fid}_model_results_auc.csv"),
            "kfold_impa": os.path.join(base, "model", f"{subject}_fold{fid}_{tag}.nii.gz"),
        }
        for fid in fold_ids
    }


def load_metadata_csv(path: str) -> dict:
    """{metric_name: value} from a *_model_results_metadata.csv (total_scores/
    whole_voxels/selected_voxels/feature_percent, or whichever plain-scalar
    metrics save_model_results was given) -- see save_model_results."""
    return pd.read_csv(path, index_col=0)["value"].to_dict()


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
    total_accuracy, whole_voxels, selected_voxels, feature_percent,
    auc_<category>, accuracy_<true>_<pred>, evidence_<true>_<pred> -- so the
    whole group's numbers live in one spreadsheet instead of scattered
    across each subject's own model/test CSVs. A subject missing a family (no
    model.kfold_cv, or no model_conditions.testing) simply contributes no row
    for that family; a subject/category set that differs from the rest just
    leaves the mismatched columns blank for that row (pandas fills NaN)."""
    rows = []
    for s in subjects:
        p = subject_paths(analysis_output_dir, desc, s)
        for family, prefix in (("CV", "kfold"), ("held-out-test", "test")):
            metadata_key, auc_key, acc_key, evi_key = f"{prefix}_metadata", f"{prefix}_auc", f"{prefix}_accuracy", f"{prefix}_evidence"
            if not os.path.exists(p[metadata_key]):
                continue
            metadata = load_metadata_csv(p[metadata_key])
            row = {"subject": s, "family": family, "total_accuracy": metadata.get("total_scores")}
            for metric in ("whole_voxels", "selected_voxels", "feature_percent"):
                if metric in metadata:
                    row[metric] = metadata[metric]
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


def compile_group_decoding(analysis_output_dir: str, desc: str, subjects: list) -> pd.DataFrame:
    """Every subject's full per-TR timecourse decoding output
    (decoding_raw -- one row per decoded volume, not the pre-aggregated
    per-(window_index, regressor_label) summary) concatenated into one
    table -- each row already carries its own "subject" column (from
    build_timecourse_instructions), so no extra tagging is needed here.
    Subjects with no timecourse_decoding output simply contribute nothing."""
    frames = [
        pd.read_csv(p["decoding_raw"], dtype={"subject": str})
        for s in subjects
        for p in [subject_paths(analysis_output_dir, desc, s)]
        if os.path.exists(p["decoding_raw"])
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compile_group_cv_results(analysis_output_dir: str, desc: str, subjects: list) -> pd.DataFrame:
    """Every subject's full cross-validation hold-out sample detail
    (kfold_cv_raw -- one row per held-out sample across all k-fold folds,
    carrying its own task/trial_type/run/boldfile plus predicted_label/
    correct/evidence_<category>) concatenated into one table -- each row
    already carries its own "subject" column (from training_df/master_
    spreadsheet), so no extra tagging is needed here. Subjects with no
    model.kfold_cv output simply contribute nothing."""
    frames = [
        pd.read_csv(p["kfold_cv_raw"], dtype={"subject": str})
        for s in subjects
        for p in [subject_paths(analysis_output_dir, desc, s)]
        if os.path.exists(p["kfold_cv_raw"])
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =====================================================
# Timecourse annotation info (best-effort -- never raises)
# =====================================================

def resolve_marker_label(distinct_types: list, annotation_labels: dict = None, max_label_values: int = 4) -> str:
    """The single value when every instance at a position agrees. Otherwise
    (a mutually-exclusive "content" position, e.g. one of several trained
    categories or operations) -- the first entry of annotation_labels whose
    own value list fully covers distinct_types wins (config order); config
    values not present in distinct_types are fine, only the reverse
    (something in distinct_types missing from the group) disqualifies a
    group. Falls back to the distinct values themselves, joined with "/"
    (truncated with "..." past max_label_values), when nothing configured
    covers this exact combination."""
    if len(distinct_types) == 1:
        return distinct_types[0]

    remaining = set(distinct_types)
    for name, group_values in (annotation_labels or {}).items():
        if remaining <= set(group_values):
            return name

    shown = distinct_types[:max_label_values]
    return "/".join(shown) + ("/..." if len(distinct_types) > max_label_values else "")


def compute_event_markers(full_frame_df: pd.DataFrame, trial_start_event: dict, timecourse_conditions: dict = None, annotation_labels: dict = None, min_frequency: float = 0.5, max_label_values: int = 4):
    """Returns a list of {"trial_type", "mean_start", "std_start", "mean_duration"}
    (all in window_index/TR units), one per real *ordinal position* (1st real
    sub-event in the trial, 2nd, 3rd, ...) observed often enough (>=
    min_frequency of trials have something there) to annotate --
    rare/misaligned positions are dropped rather than adding noisy one-off
    markers. Before ranking, consecutive real events within the same trial
    that share the identical trial_type are merged into one sub-event first
    (the same "a repeated run of the same real value is one occurrence" rule
    partition_into_trials already applies to trial boundaries) -- e.g. a
    block design's 10 repeated same-category stimulus flashes are one
    annotated sub-event, not 10.

    Grouping by ordinal position rather than literal trial_type matters
    whenever a trial's Nth event is one of several **mutually exclusive**
    real values -- e.g. one of several trained categories in a block design,
    or one of several operations in a working-memory design -- since any
    single value alone is often well under min_frequency (each only ever
    occupying its own fraction of trials) even though *something* reliably
    happens at that position in nearly every trial. "trial_type" is resolved
    by resolve_marker_label -- the single value when every instance at that
    position agrees, otherwise annotation_labels' friendly name for that
    combination if configured (e.g. {"item": ["bottle", "cat", ...]}), else
    the distinct values themselves joined with "/" -- so the marker still
    shows *where* that position falls even when *what* happens there varies
    trial to trial.

    Frames before a boldfile's first anchor (trial_index == 0) aren't part
    of any real trial and are excluded. timecourse_conditions, when given,
    scopes full_frame_df to the same qualifying boldfiles
    build_timecourse_instructions would actually decode (see
    qualifying_boldfiles) -- so annotation reflects only the runs really in
    scope, not every boldfile in full_frame_df."""
    if timecourse_conditions:
        full_frame_df = full_frame_df[full_frame_df["boldfile"].isin(
            qualifying_boldfiles(full_frame_df, timecourse_conditions)
        )]
    trials = partition_into_trials(full_frame_df, trial_start_event)
    trials = trials[trials["trial_index"] > 0]
    if trials.empty:
        return []

    n_trials = trials[["boldfile", "trial_index"]].drop_duplicates().shape[0]

    events = trials.dropna(subset=["event_index"]).groupby(["boldfile", "event_index"], sort=False).agg(
        trial_type=("trial_type", "first"),
        trial_index=("trial_index", "first"),
        start=("window_index", "min"),
        n_frames=("window_index", "size"),
    ).reset_index()
    events = events.sort_values(["boldfile", "trial_index", "start"]).reset_index(drop=True)

    # merge consecutive real events (same trial, back-to-back, identical
    # trial_type) into one combined sub-event before ranking -- the same "a
    # repeated run of the same real value is one occurrence" rule
    # partition_into_trials already applies to trial boundaries, applied
    # here to sub-event positions within a trial (e.g. a block design's 10
    # repeated same-category stimulus flashes are one annotated sub-event,
    # not 10)
    same_trial = (events["boldfile"] == events["boldfile"].shift()) & (events["trial_index"] == events["trial_index"].shift())
    same_type = events["trial_type"] == events["trial_type"].shift()
    run_id = (~(same_trial & same_type)).cumsum()
    events = events.groupby(run_id, sort=False).agg(
        boldfile=("boldfile", "first"),
        trial_index=("trial_index", "first"),
        trial_type=("trial_type", "first"),
        start=("start", "min"),
        n_frames=("n_frames", "sum"),
    ).reset_index(drop=True)

    events["position"] = events.groupby(["boldfile", "trial_index"]).cumcount() + 1

    markers = []
    for position, group in events.groupby("position"):
        frequency = group[["boldfile", "trial_index"]].drop_duplicates().shape[0] / n_trials
        if frequency < min_frequency:
            continue
        distinct_types = sorted(group["trial_type"].unique())
        label = resolve_marker_label(distinct_types, annotation_labels, max_label_values)
        markers.append({
            "trial_type": label,
            "mean_start": float(group["start"].mean()),
            "std_start": float(group["start"].std(ddof=0)) if len(group) > 1 else 0.0,
            "mean_duration": float(group["n_frames"].mean()),
        })

    return sorted(markers, key=lambda m: m["mean_start"])


def load_annotation_info(config_path, master_spreadsheet_path):
    """Returns (event_markers, tr, overlay_conditions), any of which may be
    None/empty if the optional inputs are missing or insufficient --
    annotation is strictly best-effort and never blocks the rest of the
    report. event_markers is compute_event_markers' output: one entry per
    real event type reliably observed inside a trial, used to draw per-event
    timing annotations on the timecourse page -- labeled per
    model_conditions.timecourse_decoding.annotation_labels when given (see
    resolve_marker_label), a friendly name for a group of mutually exclusive
    trial_type values (empty dict if absent, meaning no group gets a
    friendly name -- resolve_marker_label falls back to auto-joining the
    values). overlay_conditions is
    model_conditions.timecourse_decoding.overlay verbatim (empty dict if
    absent) -- see summarize_raw_for_timecourse/resolve_overlay_styles for
    how it's used (filtering/labeling and, via each entry's own optional
    "color"/"line_type", the timecourse page's per-trace styling)."""
    if not config_path or not master_spreadsheet_path:
        return [], None, {}
    if not os.path.isfile(config_path):
        print(f"(!) --config {config_path} not found -- skipping timecourse annotation")
        return [], None, {}
    if not os.path.isfile(master_spreadsheet_path):
        print(f"(!) --master-spreadsheet {master_spreadsheet_path} not found -- skipping timecourse annotation")
        return [], None, {}

    with open(config_path) as f:
        cfg = json.load(f)
    tc_cfg = cfg.get("model_conditions", {}).get("timecourse_decoding")
    if not tc_cfg:
        print("(!) config has no model_conditions.timecourse_decoding -- skipping timecourse annotation")
        return [], None, {}

    trial_start_event = tc_cfg.get("trial_start_event")
    timecourse_conditions = tc_cfg.get("conditions", {})
    overlay_conditions = tc_cfg.get("overlay", {})
    annotation_labels = tc_cfg.get("annotation_labels", {})

    full_frame = pd.read_csv(
        master_spreadsheet_path, dtype={"subject": str, "session": str, "task": str, "trial_type": str}
    )
    event_markers = (
        compute_event_markers(full_frame, trial_start_event, timecourse_conditions, annotation_labels)
        if trial_start_event else []
    )

    # TR derived from the data (majority across boldfiles), not hardcoded
    trs = []
    for boldfile in full_frame["boldfile"].unique():
        if os.path.exists(boldfile):
            try:
                tr, _ = get_bold_header_info(boldfile)
                trs.append(tr)
            except Exception:
                pass

    if not trs:
        print("(!) could not read TR from any boldfile -- timecourse x-axis will stay in window_index units")
        return event_markers, None, overlay_conditions

    tr_counts = Counter(trs)
    tr = tr_counts.most_common(1)[0][0]
    if len(tr_counts) > 1:
        print(f"(!) multiple distinct TRs found across boldfiles ({dict(tr_counts)}) -- using the majority TR={tr}")

    return event_markers, tr, overlay_conditions


# =====================================================
# Report pages
# =====================================================

def _wrap_lines_to_page(fig, ax, lines: list, fontsize: int, x0: float = 0.05) -> list:
    """Soft-wrap every line to whatever actually fits on the page -- measured
    from the real renderer rather than a guessed character count (the axes
    box is narrower than the full figure, and font substitution/dpi can
    shift actual glyph width). Needed for any page that might include a full
    filesystem path or a long comma-joined list with no natural break point
    (100+ chars on a real cluster run), which would otherwise run off the
    page edge instead of onto a continuation line."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = ax.text(0, 0, "M" * 40, fontsize=fontsize, family="monospace")
    char_width_px = probe.get_window_extent(renderer=renderer).width / 40
    probe.remove()
    axes_width_px = ax.get_window_extent(renderer=renderer).width
    avail_px = axes_width_px * (1 - x0) - char_width_px * 2  # small right-margin buffer
    wrap_width = max(20, int(avail_px / char_width_px))

    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(
            line, width=wrap_width, subsequent_indent="    ",
            break_long_words=True, break_on_hyphens=False,
        ) or [line])
    return wrapped_lines


def _wrap_suptitle(fig, text: str, fontsize: int) -> str:
    """Soft-wrap a fig.suptitle()-bound string to the actual figure width --
    measured from the real renderer, the same technique _wrap_lines_to_page
    uses for a page of body text, just measured against the whole figure
    (fig.bbox) rather than one axes, since a suptitle spans the full figure
    width. Without this, a long model.desc (a real classifier name easily
    runs 40-60+ chars) silently overflows past the figure's right edge
    instead of wrapping onto a second line, since matplotlib never wraps
    suptitle text on its own. Preserves embedded "\\n" (each existing line
    is wrapped independently, not joined into one paragraph)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "M" * 40, fontsize=fontsize, fontweight="bold")
    char_width_px = probe.get_window_extent(renderer=renderer).width / 40
    probe.remove()
    avail_px = fig.bbox.width * 0.94  # small margin buffer each side
    wrap_width = max(20, int(avail_px / char_width_px))

    wrapped_lines = []
    for line in text.split("\n"):
        # model.desc is typically one long underscore-joined token with no
        # spaces at all (e.g. "vvps_category_loc2WM_timecourse_classifier"),
        # so textwrap sees it as a single unbreakable word and (via
        # break_long_words) would chop it at an arbitrary character offset
        # ("timecour"/"se_classifier") -- inserting a space after each "_"
        # gives textwrap real break points to wrap on, restored afterward
        wrapped = textwrap.wrap(
            line.replace("_", "_ "), width=wrap_width, break_long_words=True, break_on_hyphens=False,
        ) or [line]
        wrapped_lines.extend(w.replace("_ ", "_") for w in wrapped)
    return "\n".join(wrapped_lines)


def _row_major_legend_order(items: list, ncol: int) -> list:
    """Matplotlib's multi-column Legend always reads its handles/labels list
    column-major -- top-to-bottom within a column, then the next column to
    the right -- so asking for ncol=4 does NOT mean "4 entries left to
    right, then wrap"; it silently means "split the list into up to 4
    columns, read down each one first". This permutes `items` (already in
    the order you actually want to see, left-to-right then top-to-bottom)
    into whatever order produces that reading once matplotlib re-splits it
    column-major.

    Mirrors matplotlib's own column-size balancing exactly: the leftmost
    columns get ceil(n_remaining/columns_remaining) items and later columns
    get one fewer once the count no longer divides evenly, rather than
    always-equal-size columns -- get this wrong and the permutation lands
    one row off for any n that isn't a clean multiple of ncol."""
    n = len(items)
    if n == 0 or ncol <= 1:
        return list(items)

    col_sizes = []
    remaining = n
    for c in range(ncol):
        size = math.ceil(remaining / (ncol - c))
        col_sizes.append(size)
        remaining -= size
    col_offsets = [sum(col_sizes[:c]) for c in range(ncol)]

    ordered = [None] * n
    for row in range(col_sizes[0]):  # the first column is always the tallest (or tied)
        for col in range(ncol):
            desired_idx = row * ncol + col
            if desired_idx >= n or row >= col_sizes[col]:
                continue
            ordered[col_offsets[col] + row] = items[desired_idx]
    return ordered


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

    wrapped_lines = _wrap_lines_to_page(fig, ax, lines, fontsize=13)
    ax.text(0.05, 0.92, "\n".join(wrapped_lines), fontsize=13, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def load_double_dipping_report(analysis_output_dir: str, desc: str, subject: str) -> dict:
    """Contents of <subject>_double_dipping_report.json -- written by
    mvpa_workflow.py's double-dipping guard (README.md section 6) only when
    it actually found model_conditions.training/testing or
    training/timecourse_decoding boldfile overlap for this subject. {} (the
    common case) means the guard found nothing to report -- training and
    testing/timecourse were genuinely independent."""
    path = subject_paths(analysis_output_dir, desc, subject)["double_dipping_report"]
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _double_dipping_message(section_label: str, info: dict) -> str:
    """One human-readable line for a "test" or "timecourse" entry from a
    double-dipping report. `handling` is one of "kfold_substitution"
    (decoded on N unique held-out fold(s) instead of the full-training
    model), "overwritten" (model.allow_train_test_overlap let it proceed
    anyway), or "skipped" (no safe substitute was available, so nothing was
    computed/written for that section)."""
    handling = info["handling"]
    n = len(info["overlap_boldfiles"])
    if handling == "kfold_substitution":
        n_folds = info.get("n_folds_used", "?")
        return (f"{section_label}: {n} bold file(s) overlapped with model_conditions.training -- "
                f"decoding computed on {n_folds} unique fold(s) instead of the full-training model.")
    if handling == "overwritten":
        return (f"{section_label}: {n} bold file(s) overlapped with model_conditions.training -- "
                f"user overwrite (model.allow_train_test_overlap) -- data leakage may be present.")
    if handling == "skipped":
        return (f"{section_label}: {n} bold file(s) overlapped with model_conditions.training -- "
                f"skipped entirely, no output produced for this subject.")
    return f"{section_label}: overlap detected, handling {handling!r} not recognized."


def render_double_dipping_page(pdf, analysis_output_dir, desc, subjects):
    """A dedicated warning page -- rendered only when at least one subject in
    scope actually has a double-dipping report (mvpa_workflow.py's guard,
    see README.md section 6's "Double-dipping guard") -- so a well-formed
    config with genuinely independent training/testing/timecourse produces
    no page at all, exactly as before this guard existed."""
    reports = {s: load_double_dipping_report(analysis_output_dir, desc, s) for s in subjects}
    reports = {s: r for s, r in reports.items() if r}
    if not reports:
        return

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    lines = [
        "⚠ Data Independence Warning",
        "",
        "model_conditions.training shared bold file(s) with testing and/or",
        "timecourse_decoding for the subject(s) below -- evaluating or decoding",
        "on data a classifier was (even partly) trained on inflates apparent",
        "performance. See README.md section 6, \"Double-dipping guard\".",
        "",
    ]
    for s in subjects:
        report = reports.get(s)
        if not report:
            continue
        lines.append(f"Subject {s}:")
        if "test" in report:
            lines.append("  " + _double_dipping_message("Held-out test evaluation", report["test"]))
        if "timecourse" in report:
            lines.append("  " + _double_dipping_message("Timecourse decoding", report["timecourse"]))
        lines.append("")

    wrapped_lines = _wrap_lines_to_page(fig, ax, lines, fontsize=11)
    ax.text(0.05, 0.94, "\n".join(wrapped_lines), fontsize=11, va="top", family="monospace", color="darkred")
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
        if os.path.exists(p["kfold_metadata"]):
            kfold_totals[s] = load_metadata_csv(p["kfold_metadata"])["total_scores"]
        if os.path.exists(p["test_metadata"]):
            test_totals[s] = load_metadata_csv(p["test_metadata"])["total_scores"]
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
            fold_vals = [load_metadata_csv(f["kfold_metadata"])["total_scores"]
                         for f in folds.values() if os.path.exists(f["kfold_metadata"])]
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

    fig.suptitle(_wrap_suptitle(fig, f"{desc}: accuracy & AUC", 14), fontsize=14, fontweight="bold")
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

    fig.suptitle(_wrap_suptitle(fig, f"{desc}: confusion-style matrices ({title_suffix})", 14), fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


def broadcast_trial_label(df: pd.DataFrame, label_col: str, trial_cols=("boldfile", "trial_index")) -> pd.Series:
    """label_col's values, broadcast across every row of the same
    (boldfile, trial_index) trial -- since continuous timecourse decoding
    labels only the real content event within a trial (regressor_label is
    blank for its fixation/rest/ITI tail, see build_timecourse_instructions),
    grouping directly by label_col would silently drop that trailing,
    unlabeled portion of the trial from the timecourse plot instead of
    showing the full trial through to its end. Returns label_col's dtype;
    a trial with no non-null value anywhere keeps all-null (still excluded,
    same as before -- there's nothing to plot it as). A trial with more than
    one *distinct* non-null value is unexpected (conditions should partition
    trials, not split one down the middle) and prints a warning; the
    alphabetically-first value wins for that trial, deterministically."""
    multi = (
        df.groupby(list(trial_cols))[label_col]
        .agg(lambda s: s.dropna().nunique())
    )
    ambiguous = multi[multi > 1]
    if len(ambiguous):
        print(f"  (!) {len(ambiguous)} trial(s) have more than one distinct {label_col} value -- "
              f"using the alphabetically-first one for the timecourse plot")

    ordered = df.sort_values(list(trial_cols) + [label_col], na_position="last")
    broadcast = ordered.groupby(list(trial_cols))[label_col].transform("first")
    return broadcast.reindex(df.index)


def summarize_raw_for_timecourse(raw_df: pd.DataFrame, overlay_conditions: dict = None) -> pd.DataFrame:
    """One row per (window_index, regressor_label[, overlay_label]) -- for each
    evidence_* column, both the mean *and* the trial-to-trial standard error
    (std across the trials sharing that group, within this one subject's/fold's
    own raw rows, /sqrt(n)), suffixed evidence_*_se. Computed directly from the
    raw per-TR file rather than the pre-aggregated summary CSV specifically so
    this trial-level spread is available at all -- the summary only ever kept
    the mean (mvpa_common.summarize_decoding() has no equivalent).

    regressor_label (and overlay_label, when overlay_conditions is given) are
    first broadcast across each row's whole trial (broadcast_trial_label) --
    otherwise grouping directly by these per-frame labels would silently
    plot only a trial's labeled content event and cut off its unlabeled
    fixation/rest/ITI tail, instead of the full trial through to its end.
    When overlay_conditions is given, rows are additionally tagged via
    label_rows (dropping unmatched rows, count printed) and grouped by
    overlay_label as an extra key. Each overlay entry may carry "color"/
    "line_type" alongside its query fields (label_rows/evaluate_query_node
    only ever read the query keys they need, so these are ignored here and
    picked up separately by resolve_overlay_styles for the actual plot
    styling)."""
    df = raw_df.copy()
    df["regressor_label"] = broadcast_trial_label(df, "regressor_label")
    group_cols = ["window_index", "regressor_label"]
    if overlay_conditions:
        # label_rows_optional (not label_rows) so a trial's unlabeled tail
        # survives long enough to inherit its trial's own overlay_label
        # below -- only whole trials that never matched any overlay
        # condition get dropped, not just their trailing unmatched frames
        df = label_rows_optional(df, overlay_conditions, label_column="overlay_label")
        df["overlay_label"] = broadcast_trial_label(df, "overlay_label")
        before = len(df)
        df = df[df["overlay_label"].notna()]
        dropped = before - len(df)
        if dropped:
            print(f"  (!) {dropped} row(s) belonged to a trial matching no overlay condition -- dropped from the timecourse plot")
        group_cols = group_cols + ["overlay_label"]

    evidence_cols = [c for c in df.columns if c.startswith("evidence_")]
    grouped = df.groupby(group_cols)[evidence_cols]
    means = grouped.mean()
    ses = grouped.agg(lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0).add_suffix("_se")
    return pd.concat([means, ses], axis=1).reset_index()


# cycled for an overlay category's line_type when it's given as an integer
# index, or when no line_type is specified at all and this category needed
# one anyway; matplotlib also accepts any of these (or "solid"/"dashed"/
# "dotted"/"dashdot") as a literal explicit "line_type" string
OVERLAY_LINESTYLES = ["-", "--", ":", "-."]


def resolve_overlay_styles(overlay_conditions: dict) -> dict:
    """{category_name: (color, line_type)} for every entry in
    model_conditions.timecourse_decoding.overlay, resolving each entry's own
    optional "color"/"line_type" (set explicitly to fully control the plot)
    against sensible defaults for whichever one is omitted:

      - line_type: an int indexes OVERLAY_LINESTYLES; a string (e.g. "--" or
        "dashed") is passed straight through to matplotlib; omitted defaults
        to solid ("-").
      - color: an int indexes the standard tab10 palette; a string (e.g.
        "#1f77b4" or "red") is passed straight through to matplotlib;
        omitted auto-assigns from that same palette, cycling separately
        *within* each resolved line_type -- restarting at palette index 0
        for the first color-less entry of each line_type -- so e.g. two
        conditions sharing line_type="dashed" but no explicit color still
        land on different colors, while still lining up with same-numbered
        solid/dashed pairs that *do* share a color (see README.md section 4).

    Entries that never appear in the data still get resolved (harmless --
    render_timecourse_pages only ever looks up categories actually present)."""
    trace_colors = plt.get_cmap("tab10").colors

    line_types = {}
    for name, entry in overlay_conditions.items():
        lt = entry.get("line_type")
        if lt is None:
            line_types[name] = "-"
        elif isinstance(lt, int) and not isinstance(lt, bool):
            line_types[name] = OVERLAY_LINESTYLES[lt % len(OVERLAY_LINESTYLES)]
        else:
            line_types[name] = lt

    auto_color_counters = {}  # resolved line_type -> next auto-assigned palette index
    colors = {}
    for name, entry in overlay_conditions.items():
        c = entry.get("color")
        if c is None:
            lt = line_types[name]
            idx = auto_color_counters.get(lt, 0)
            colors[name] = trace_colors[idx % len(trace_colors)]
            auto_color_counters[lt] = idx + 1
        elif isinstance(c, int) and not isinstance(c, bool):
            colors[name] = trace_colors[c % len(trace_colors)]
        else:
            colors[name] = c

    return {name: (colors[name], line_types[name]) for name in overlay_conditions}


def resolve_overlay_groups(overlay_conditions: dict) -> dict:
    """{category_name: group value or None} for every entry in
    model_conditions.timecourse_decoding.overlay, reading each entry's own
    optional "group" key -- a plain label (any JSON scalar) used to gather
    that category together with every other category sharing the same
    "group" value into its own contiguous block of rows (one row per true
    condition, all showing only that group's categories), each block capped
    with its own legend -- see resolve_timecourse_groups. Unlike
    color/line_type there's no auto-assigned default: a category with
    "group" unset stays ungrouped (None), and render_timecourse_pages only
    switches into grouped mode at all when at least one category in scope
    has it set."""
    return {name: entry.get("group") for name, entry in overlay_conditions.items()}


def resolve_timecourse_groups(true_conditions: list, overlay_categories: list, overlay_groups: dict) -> list:
    """One (group_value, row_specs) block per distinct "group" value declared
    on overlay_categories, in first-seen (config declaration) order --
    row_specs is [(true_condition, overlay_categories_for_this_group), ...],
    one row per true condition (existing order), all sharing this block's
    group value so they render as one contiguous set of rows followed by one
    legend, before the next group's rows start -- e.g. splitting an
    operation x valence overlay into a "pos" block and a "neg" block, each
    showing all true conditions for that valence together. Every
    overlay_categories entry must already have a resolved (non-None) group
    value -- the caller (render_timecourse_pages) is responsible for
    deciding whether grouped mode applies at all and for giving any unset
    entries a fallback value."""
    group_values = list(dict.fromkeys(overlay_groups[c] for c in overlay_categories))
    return [
        (group_val, [
            (true_cond, [c for c in overlay_categories if overlay_groups[c] == group_val])
            for true_cond in true_conditions
        ])
        for group_val in group_values
    ]


def _build_overlay_legend(cats: list, overlay_styles: dict, ncol: int, overlay_conditions: dict) -> tuple:
    """(handles, labels) for one overlay legend covering exactly `cats` --
    always ending with a "trial-to-trial SE" proxy patch, ragged-row-centered
    and row-major ordered for an `ncol`-column layout (see
    _row_major_legend_order). Shared by both today's single whole-page
    legend (one block, cats=every overlay category) and each "group" block's
    own legend (cats=just that block's categories) -- either way this is the
    complete self-contained legend for whatever it's covering."""
    if overlay_conditions and cats and cats != [None]:
        handles = [
            Line2D([0], [0], color=overlay_styles.get(c, ("black", "-"))[0],
                   linestyle=overlay_styles.get(c, ("black", "-"))[1], linewidth=1.5)
            for c in cats
        ]
        labels = list(cats)
    else:
        handles, labels = [], []
    trial_se_proxy = Patch(facecolor="black", alpha=0.12)
    handles = handles + [trial_se_proxy]
    labels = labels + ["trial-to-trial SE"]

    # A ragged final row (fewer than ncol entries) would otherwise render
    # flush-left with empty space to its right -- pad it to a full row with
    # invisible entries split as evenly as possible on either side, so it
    # reads centered instead.
    remainder = len(labels) % ncol
    if remainder:
        pad_total = ncol - remainder
        left_pad, right_pad = pad_total // 2, pad_total - pad_total // 2
        blank = Patch(facecolor="none", edgecolor="none")
        handles = handles[:-remainder] + [blank] * left_pad + handles[-remainder:] + [blank] * right_pad
        labels = labels[:-remainder] + [""] * left_pad + labels[-remainder:] + [""] * right_pad

    # handles/labels are now in the order we want read left-to-right,
    # top-to-bottom (config declaration order, then the SE proxy last, with
    # the padding above centering whatever ended up in the final row) --
    # permute for matplotlib's column-major fill so it actually reads that
    # way instead of down each column first (see _row_major_legend_order)
    handles, labels = zip(*_row_major_legend_order(list(zip(handles, labels)), ncol))
    return list(handles), list(labels)


def draw_event_annotations(ax, event_markers, tr, show_labels):
    """One annotation per compute_event_markers() entry: a light shaded
    axvspan over the event's typical [start, start+duration) plus a dotted
    axvline at its start and a text label there, when that event reliably
    starts at the same relative time across trials (std_start < half a TR).
    Otherwise ("blurry" -- real trial-to-trial jitter in when it starts) skip
    the crisp line/edges and instead fade several progressively wider,
    fainter bands outward from the mean start -- sized off std_start -- so
    the boundary visibly softens rather than showing a falsely precise edge."""
    ylim = ax.get_ylim()
    for marker in event_markers:
        start = marker["mean_start"] * tr
        duration = marker["mean_duration"] * tr
        std = marker["std_start"] * tr
        label = marker["trial_type"]

        if std < 0.5 * tr:
            ax.axvspan(start, start + duration, color="gray", alpha=0.12, zorder=0)
            ax.axvline(start, color="gray", linestyle=":", linewidth=0.75, zorder=0)
        else:
            for band, alpha in zip((0.5, 1.0, 1.5, 2.0), (0.10, 0.07, 0.05, 0.03)):
                half_width = band * std
                ax.axvspan(start - half_width, start + duration + half_width, color="gray", alpha=alpha, zorder=0)
            label = f"{label} (variable timing)"

        if show_labels:
            ax.text(start, ylim[1], label, fontsize=6.5, ha="left", va="bottom", rotation=45)


def render_timecourse_pages(pdf, analysis_output_dir, desc, subjects, event_markers, tr,
                             overlay_conditions=None):
    """Timecourse decoding always comes from the complete-training-set
    classifier (mvpa_workflow.py never runs it per-fold), so there's exactly
    one decoding_raw file per subject regardless of whether model.kfold_cv
    is also configured -- no fold-level variability source anymore.

    Each subplot's x-axis spans the full trial (anchor to next anchor) by
    default, since decoding_results.csv now covers every frame of every
    trial continuously and summarize_raw_for_timecourse broadcasts each
    row's regressor_label/overlay_label across its whole trial (see
    broadcast_trial_label) -- so a trial's unlabeled fixation/rest/ITI tail
    still plots as part of its content event's own line, instead of being
    silently cut off where the real per-frame label stops. event_markers
    (compute_event_markers' output, via load_annotation_info) draws the
    real per-trial event timing on top -- see draw_event_annotations.

    overlay_conditions both filters/labels rows (as always) and, via each
    entry's own optional "color"/"line_type", controls exactly how its trace
    is drawn -- see resolve_overlay_styles. Omit both and a category falls
    back to auto-assigned styling; omit overlay_conditions entirely and
    every true-condition subplot is a single solid black line, as before
    overlay existed at all.

    An entry's optional "group" key splits the grid further: once any
    overlay category in scope has "group" set, categories sharing the same
    "group" value are gathered into their own contiguous block of rows (one
    row per true condition, in first-seen config order across blocks), each
    block capped with its own legend before the next block starts -- e.g. an
    operation x valence overlay can put maintain/suppress/switch/clear
    (still colored per resolve_overlay_styles) into a "pos" block and a
    "neg" block, each block showing every true condition for that valence
    together with its own legend, instead of one legend for the whole page.
    See resolve_timecourse_groups."""
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
    # evidence_cols (and therefore categories) are already in config order --
    # decoding_raw writes them via regressor_categories, itself
    # list(training_conditions.keys()) from mvpa_workflow.py -- so reuse
    # that same order for the true-condition rows instead of alphabetizing
    # them, so the matrix reads like a standard one (e.g. "maintain" lands
    # at row 0, col 0, matching the confusion-matrix page's convention)
    categories = [c.replace("evidence_", "") for c in evidence_cols]
    present_true_conditions = set(combined["regressor_label"].unique())
    true_conditions = [c for c in categories if c in present_true_conditions]
    # defensive: a true-condition label with no matching evidence_ column
    # shouldn't happen (regressor_label is always drawn from the same
    # regressor_categories that produced categories above), but append it
    # rather than silently dropping it if it ever does
    true_conditions += sorted(present_true_conditions - set(categories))
    # config declaration order (dict order, same as json.load preserves),
    # not alphabetical -- so the legend/plot order matches how the config
    # itself groups things (e.g. all 4 operations' "_pos" and "_neg" entries
    # declared adjacent to each other), filtered to categories that actually
    # have data (an overlay entry can legitimately match zero rows)
    present_overlay_categories = set(combined["overlay_label"].unique()) if overlay_conditions else set()
    overlay_categories = [c for c in overlay_conditions if c in present_overlay_categories] if overlay_conditions else [None]
    overlay_styles = resolve_overlay_styles(overlay_conditions) if overlay_conditions else {}
    overlay_groups = resolve_overlay_groups(overlay_conditions) if overlay_conditions else {}

    # grouped mode only kicks in once at least one in-scope overlay category
    # actually has "group" set -- otherwise every true_condition stays a
    # single block with every overlay category layered together (one legend
    # for the whole page), exactly as before "group" existed
    if overlay_conditions and any(overlay_groups.get(c) is not None for c in overlay_categories):
        missing_group = [c for c in overlay_categories if overlay_groups.get(c) is None]
        if missing_group:
            print(f"  (!) overlay {missing_group} has no \"group\" set while other overlay entries do -- "
                  f"giving each its own group block")
            for c in missing_group:
                overlay_groups[c] = c
        blocks = resolve_timecourse_groups(true_conditions, overlay_categories, overlay_groups)
    else:
        blocks = [(None, [(true_cond, overlay_categories) for true_cond in true_conditions])]

    n_cols = len(categories)
    if not any(row_specs for _, row_specs in blocks) or n_cols == 0:
        print("(!) decoding_results.csv has no evidence_* columns or regressor_label values -- skipping timecourse page")
        return

    legend_ncol = 4
    # one flat list of grid rows, alternating a block's data rows (one per
    # true condition) with a single legend row right after it -- this is
    # what turns "group" into a genuinely separate legend per block instead
    # of one legend for the whole page
    grid_rows = []
    for group_val, row_specs in blocks:
        for k, (true_cond, cats) in enumerate(row_specs):
            grid_rows.append({"kind": "data", "true_cond": true_cond, "group": group_val,
                               "cats": cats, "is_first_in_block": k == 0})
        grid_rows.append({"kind": "legend", "group": group_val, "cats": cats})

    data_row_indices = [i for i, r in enumerate(grid_rows) if r["kind"] == "data"]
    last_data_row = data_row_indices[-1]

    # data rows all get equal height; a legend row's height scales with how
    # many lines of legend_ncol-wide entries its own block actually needs
    # (trial-to-trial SE proxy always adds one more entry to every block's
    # legend, even a block with just one overlay category, so it reads as a
    # complete, self-contained legend on its own)
    def _legend_height(n_cats):
        n_lines = -(-(n_cats + 1) // legend_ncol)  # ceil
        return 0.22 + 0.2 * n_lines

    height_ratios = [1.0 if r["kind"] == "data" else _legend_height(len(r["cats"])) for r in grid_rows]
    unit_height = 2.3
    fig = plt.figure(figsize=(3 * n_cols, unit_height * sum(height_ratios) + 0.8), layout="constrained")
    gs = fig.add_gridspec(nrows=len(grid_rows), ncols=n_cols, height_ratios=height_ratios)

    # zero-width for a single-subject report -- nothing to average across when
    # there's only one subject's own decoding
    variability_label = "darker band: +/- SE across subjects; lighter band: +/- trial-to-trial SE"
    x_is_seconds = tr is not None
    x_label = "Time from trial start (s)" if x_is_seconds else "window_index"

    first_ax = None
    for i, row in enumerate(grid_rows):
        if row["kind"] == "legend":
            legend_ax = fig.add_subplot(gs[i, :])
            legend_ax.axis("off")
            handles, labels = _build_overlay_legend(row["cats"], overlay_styles, legend_ncol, overlay_conditions)
            title = None if not overlay_conditions else ("overlay" if row["group"] is None else f"overlay ({row['group']})")
            legend_ax.legend(handles, labels, loc="center", ncol=legend_ncol, fontsize=8, title=title)
            continue

        true_cond, group_val, cats = row["true_cond"], row["group"], row["cats"]
        subset = combined[combined["regressor_label"] == true_cond]
        for j, cat in enumerate(categories):
            ax = fig.add_subplot(gs[i, j], sharex=first_ax, sharey=first_ax)
            if first_ax is None:
                first_ax = ax
            for overlay_cat in cats:
                line = subset if overlay_cat is None else subset[subset["overlay_label"] == overlay_cat]
                color, linestyle = overlay_styles.get(overlay_cat, ("black", "-"))

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
                # lighter/wider trial-to-trial band drawn first (behind), then the
                # more opaque subject/fold-level band on top, then the mean line --
                # keeps both regions individually legible even where they overlap
                ax.fill_between(x, agg["mean"] - agg["trial_se"], agg["mean"] + agg["trial_se"],
                                 alpha=0.12, color=color, zorder=1)
                ax.fill_between(x, agg["mean"] - agg["se"], agg["mean"] + agg["se"],
                                 alpha=0.25, color=color, zorder=2)
                ax.plot(x, agg["mean"], color=color, linestyle=linestyle, linewidth=1.5, zorder=3)

            if event_markers and x_is_seconds:
                draw_event_annotations(ax, event_markers, tr, show_labels=row["is_first_in_block"])

            if row["is_first_in_block"]:
                ax.set_title(f"evidence: {cat}", fontsize=10)
            if j == 0:
                row_label = f"true: {true_cond}" if group_val is None else f"true: {true_cond}\n({group_val})"
                ax.set_ylabel(row_label, fontsize=10)
            if i == last_data_row:
                ax.set_xlabel(x_label, fontsize=9)

    fig.suptitle(_wrap_suptitle(fig, f"{desc}: timecourse decoding\n({variability_label})", 14), fontsize=14, fontweight="bold")
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

    subjects_arg = parse_subjects_arg(args.subjects) if args.subjects else None
    subjects = list_subject_dirs(args.analysis_output_dir, desc, args.subject, subjects_arg)
    fold_flags = {s: has_fold_files(args.analysis_output_dir, desc, s) for s in subjects}
    regressor_categories = infer_categories(args.analysis_output_dir, desc, subjects)
    event_markers, tr, overlay_conditions = load_annotation_info(args.config, args.master_spreadsheet)
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
        render_double_dipping_page(pdf, args.analysis_output_dir, desc, subjects)
        render_accuracy_auc_page(pdf, args.analysis_output_dir, desc, subjects, fold_flags, regressor_categories)
        render_confusion_matrices_page(pdf, args.analysis_output_dir, desc, subjects)
        render_timecourse_pages(pdf, args.analysis_output_dir, desc, subjects, event_markers, tr,
                                 overlay_conditions)
        render_importance_pages(pdf, args.analysis_output_dir, desc, subjects, fold_flags, regressor_categories,
                                 os.path.dirname(output_path), mnispace=mnispace)

    print(f"Report written to: {output_path}")

    if len(subjects) > 1:
        summary = compile_group_summary(args.analysis_output_dir, desc, subjects)
        summary_path = os.path.join(os.path.dirname(output_path), f"{desc}_group_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Group summary spreadsheet saved to: {summary_path}")

        decoding = compile_group_decoding(args.analysis_output_dir, desc, subjects)
        if not decoding.empty:
            decoding_path = os.path.join(os.path.dirname(output_path), f"{desc}_group_decoding_results.csv")
            decoding.to_csv(decoding_path, index=False)
            print(f"Group decoding spreadsheet saved to: {decoding_path}")

        cv_results = compile_group_cv_results(args.analysis_output_dir, desc, subjects)
        if not cv_results.empty:
            cv_results_path = os.path.join(os.path.dirname(output_path), f"{desc}_group_cv_results.csv")
            cv_results.to_csv(cv_results_path, index=False)
            print(f"Group cross-validation spreadsheet saved to: {cv_results_path}")


if __name__ == "__main__":
    main()
