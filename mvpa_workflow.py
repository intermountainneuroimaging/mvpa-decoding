#!/usr/bin/env python3

"""
Within-subject MVPA decoding: trains a classifier on model_conditions.training,
cross-validates internally on that same training data (GroupKFold on `run`),
evaluates on model_conditions.testing, and predicts at every TR across a
timecourse_decoding window built independently of however volume_of_interest
was computed in master_spreadsheet.csv. Reads the merged config (event_extraction
+ model_conditions + model sections) plus master_spreadsheet.csv produced by
generate_master_spreadsheet.py -- see README.md sections 3-5 for the config
format and "Running mvpa_workflow.py" for what each step does.

Works for any number of training/testing conditions (2 or more): class lists
are always derived from what the classifier actually learned (clf.classes_),
not from what happens to appear in a given fold's held-out data.

This script is dedicated to the independent-train/independent-test case --
training and testing come from separate model_conditions sections and may
be entirely different tasks/data. A separate k-fold workflow (same-data,
split-by-run training/testing) is planned as its own script with its own
config format, not a mode switch grafted onto this one.

Outputs, under <analysis-output-dir>/<model.desc>/<subject>/:
    <subject>_trial_pivot.csv           -- sanity check, pre-model_conditions
    cv/<subject>_cv_results_{metric}.csv        -- internal training-CV metrics
    cv/<subject>_cv_impa_native.nii.gz          -- CV-fold-averaged importance map
    model/<subject>_model_results_{metric}.csv  -- held-out test metrics
    model/<subject>_impa_native.nii.gz          -- final importance map
    decoding/<subject>_decoding_results.csv         -- raw, one row per decoded TR
    decoding/<subject>_summary_decoding_results.csv -- averaged per (window_index, regressor_label)

Usage:
    python mvpa_workflow.py --subject 4057 --config examples/config-2.example.json \\
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import importlib

# processing
try:
    from nilearn.maskers import NiftiMasker
except Exception:
    from nilearn.input_data import NiftiMasker

from sklearn.preprocessing import StandardScaler

# classification
from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, permutation_test_score
from sklearn.pipeline import Pipeline

from mvpa_common import (
    compute_volume_range, resolve_window_times, build_trial_pivot_table,
    resolve_config_root, quick_safe, label_rows, get_single_match, get_bold_header_info,
)

# grouping used for the timecourse decoding output -- the relative timepoint
# within each event's decode window, crossed with the classification label.
TIMECOURSE_GROUPING = ["window_index", "regressor_label"]


# =====================================================
# Argument Parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID to run, matching master_spreadsheet.csv's 'subject' column exactly (e.g. 4057)"
    )

    parser.add_argument(
        "--analysis-output-dir",
        required=True,
        help="Root output directory -- results are written under <this>/<model.desc>/<subject>/{cv,model,decoding}/"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the mvpa config JSON (event_extraction + model_conditions + model sections). "
             "See README.md sections 3-5 and examples/config-*.example.json."
    )

    parser.add_argument(
        "--master-spreadsheet",
        required=True,
        help="Path to master_spreadsheet.csv produced by generate_master_spreadsheet.py"
    )

    return parser.parse_args()


# =====================================================
# Performance Monitor
# =====================================================

import time
import resource
from contextlib import contextmanager


@contextmanager
def track_runtime(label: str = "run"):
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        c1 = time.process_time()

        # ru_maxrss: on Linux it's KB; on macOS it's bytes.
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = r / 1024.0  # Linux -> MB (KB/1024). If on macOS, change to r/(1024**2).

        print(f"[{label}] wall={t1 - t0:.3f}s | cpu={c1 - c0:.3f}s | peak_rss≈{rss_mb:.1f} MB")


# =====================================================
# Helper Functions
# =====================================================

def apply_regressor_codes(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    df = df.copy()
    df["regressor"] = pd.Categorical(
        df["regressor_label"], categories=categories, ordered=True
    ).codes + 1
    return df


def balance(xdf: pd.DataFrame) -> pd.DataFrame:
    df = xdf.copy()
    df["ID"] = df.index  # keep original row id

    # target number of rows to keep per regressor (lowest common denominator)
    k = df.groupby("regressor").size().min()

    # count rows per (regressor, run, trial_index) to prioritize fuller trials
    pair_counts = (
        df.groupby(["regressor", "run", "trial_index"])
          .size()
          .rename("pair_n")
          .reset_index()
    )

    # merge counts back so each row knows how "full" its (run, trial_index) group is
    df2 = df.merge(pair_counts, on=["regressor", "run", "trial_index"], how="left")

    # sort so we:
    #  1) for each regressor, consider the most-populated (run, trial_index) pairs first
    #  2) within a pair, keep highest volume_of_interest first (then we'll cap total to k)
    df2 = df2.sort_values(
        ["regressor", "pair_n", "run", "trial_index", "volume_of_interest"],
        ascending=[True, False, True, True, False]
    )

    # take first k rows per regressor (after prioritization + within-pair trimming)
    out = df2.groupby("regressor", group_keys=False).head(k)

    # optional: final ordering for downstream use
    out = out.sort_values(["run", "trial_index", "volume_of_interest"])

    # drop helper column if you want
    out = out.drop(columns=["pair_n"])

    return out


def decision_evidence(clf, rawdata):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(rawdata)

    raw_scores = clf.decision_function(rawdata)

    if len(clf.classes_) == 2:
        raw_scores = raw_scores.reshape(-1, 1)
        prob1 = 1 / (1 + np.exp(-raw_scores))
        return np.hstack([1 - prob1, prob1])

    e = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class ShapeError(Exception):
    pass


_masker_cache = {}


def load_images_and_mask(labeled_df: pd.DataFrame, mask_pattern_template: str, mask_root: str):
    """Load BOLD patterns for every (subject, session, boldfile) group in
    labeled_df, z-score, and slice to each row's volume_of_interest."""

    matrices = []
    labels = []
    indices = []
    masker = None

    for boldfile, group in labeled_df.groupby("boldfile", sort=False):
        if not os.path.exists(boldfile):
            raise FileNotFoundError(f"boldfile referenced by master_spreadsheet does not exist: {boldfile}")

        subject = group["subject"].iloc[0]
        session = group["session"].iloc[0]
        mask_key = (subject, session)

        if mask_key not in _masker_cache:
            mask_pattern = mask_pattern_template.format(subject=subject, session=session)
            mask_file = get_single_match(os.path.join(mask_root, mask_pattern))
            print(f"Using Mask File: {mask_file}")
            bold_tr, _ = get_bold_header_info(boldfile)
            _masker_cache[mask_key] = NiftiMasker(mask_img=mask_file, standardize=False, detrend=False, t_r=bold_tr)
        masker = _masker_cache[mask_key]

        # apply mask
        masked_data = masker.fit_transform(boldfile)

        # apply z-transform
        z_patterns = StandardScaler().fit_transform(masked_data)
        z_patterns = np.nan_to_num(z_patterns)

        # crop data to selected volumes
        vols = (
            pd.to_numeric(group["volume_of_interest"], errors="raise")
            .astype(int)
            .to_numpy()
        )

        z_patterns = z_patterns[vols, :]
        # stack image data to match instructions
        matrices.append(z_patterns)

        # stack the regression labels as well to be 100% sure they data and labels align
        labels.extend(group["regressor"].to_numpy())

        # stack selected indices to later extract volume metadata
        indices.extend(group.index.to_numpy())

        print(f"Sucessfully loaded: {boldfile}")

    if not matrices:
        raise ValueError("No rows to load -- check that model_conditions' queries actually match this subject's data.")

    # all image data stacked
    X = np.vstack(matrices)
    Y = np.array(labels).reshape(-1, 1)
    idx = np.array(indices)

    # Check X and Y have same number of rows (observations)
    if X.shape[0] != Y.shape[0]:
        raise ShapeError("Image Data and Labels Do No Align... Can't Proceed!!")

    return X, Y, idx, masker


def build_timecourse_instructions(labeled_df: pd.DataFrame, window: dict) -> pd.DataFrame:
    """One row per source event in labeled_df (grouped by boldfile+trial_index),
    re-expanded into fresh volume_of_interest rows per `window`, tagged with a
    window_index (position within that event's recomputed window)."""

    rows = []
    for (boldfile, trial_index), group in labeled_df.groupby(["boldfile", "trial_index"], sort=False):
        first = group.iloc[0]
        tr, n_frames = get_bold_header_info(boldfile)

        start_time, stop_time = resolve_window_times(window, first["onset"], first["duration"])
        start_vol, stop_vol = compute_volume_range(start_time, stop_time, tr, n_frames)

        for vol in range(start_vol, stop_vol):
            rows.append({
                "subject": first["subject"],
                "session": first["session"],
                "task": first["task"],
                "run": first["run"],
                "trial_type": first["trial_type"],
                "trial_index": trial_index,
                "regressor_label": first["regressor_label"],
                "regressor": first["regressor"],
                "boldfile": boldfile,
                "volume_of_interest": vol,
                "window_index": vol - start_vol,
            })

    return pd.DataFrame(rows)


def import_from_path(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def default_model_config() -> dict:
    return {
        "desc": "default_classifier",
        "featureSelection": {
            "model": "ANOVA",
            "feat_p": 0.05,
        },
        "classifier": {
            "name": "sklearn.linear_model.LogisticRegression",
            "params": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 5000,
                "class_weight": "balanced"
            }
        },
        "cv": {
            "strategy": "GroupKFold",
            "n_splits": "infer"
        }
    }


def merge_with_defaults(user_cfg, base):
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                recursive_update(d[k], v)
            else:
                d[k] = v
        return d

    return recursive_update(base, user_cfg)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        full_cfg = json.load(f)

    if "event_extraction" not in full_cfg:
        raise SystemExit("config missing required 'event_extraction' section")
    if "model_conditions" not in full_cfg:
        raise SystemExit("config missing required 'model_conditions' section")

    full_cfg["model"] = merge_with_defaults(full_cfg.get("model", {}), default_model_config())
    return full_cfg


def save_model_results(output_pattern, results, categories):
    """
    Save a dict of model outputs to disk, one file per metric.

    Parameters
    ----------
    output_pattern : str
        Format string used to build output filenames. Must include a '{metric}'
        placeholder, e.g.:
            "/path/to/out/{metric}.csv"
            "/path/to/out/sub-01_run-2_{metric}.csv"

    results : dict[str, array-like]
        Mapping from metric name -> value to save. Supported value shapes:

        1) Square matrix (C, C)
           - Interpreted as a class-by-class matrix (e.g., confusion matrix,
             importance matrix).
           - Saved as a CSV with row/column labels from `categories`.

        2) Column vector (C,) or (C, 1)
           - Interpreted as one value per category/class.
           - Saved as a single-column CSV indexed by `categories`.

        3) Anything else (e.g., (S, E), (n_features,), scalar)
           - Saved via np.savetxt as numeric CSV (no labels).
           - Scalars are promoted to 1D.

    categories : sequence of str
        Category/class labels in the same order used by the model outputs.
        Length defines C.

    Notes
    -----
    - All outputs are written as CSV files.
    - Parent directories are created automatically.
    """
    categories = list(categories)
    C = len(categories)

    for metric, x in results.items():

        # Build output path for this metric and ensure parent directory exists
        output_file = output_pattern.format(metric=metric)
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        # Coerce to numpy array (without forcing extra dims yet)
        x = np.asarray(x)

        # Promote scalars to shape (1,) so savetxt works
        if x.ndim == 0:
            x = np.atleast_1d(x)

        # Case 1: category-by-category matrix
        if x.shape == (C, C):
            df = pd.DataFrame(x, index=categories, columns=categories)
            df.to_csv(output_file, index=True)

        # Case 2: one value per category (accept (C,) or (C,1))
        elif x.shape == (C,) or x.shape == (C, 1):
            x_vec = x.reshape(C)  # ensures 1D length-C
            df = pd.DataFrame({metric: x_vec}, index=categories)
            df.to_csv(output_file, index=True)

        # Case 3: everything else (no labels)
        else:
            np.savetxt(output_file, x, delimiter=",", fmt="%.6f")

        print(f"[{metric}] saved -> {output_file} (shape={x.shape})")


def average_fold_results(fold_results: list) -> dict:
    """Average a list of model_performance-style result dicts (scalar or array-valued
    metrics, all sharing the same keys/shapes) elementwise across folds."""
    mean_results = {}
    for k in fold_results[0].keys():
        values = [res[k] for res in fold_results]
        if np.isscalar(values[0]):
            mean_results[k] = float(np.mean(values))
        else:
            mean_results[k] = np.mean(np.stack(values, axis=0), axis=0)
    return mean_results


def resolve_feature_threshold(training_data, training_labels, feat_p: float) -> float:
    """The ANOVA p-value cutoff for feature selection, widened from feat_p
    until it selects at least 5 voxels. Shared by model_classification() and
    permutation_significance() so the permutation test's null-distribution
    pipelines use the exact same selection rule the real model did, not a
    different unwidened threshold."""
    _, xP = f_classif(training_data, training_labels)
    xP = np.nan_to_num(xP, nan=1.0)
    thr = feat_p
    while np.sum(xP < thr) < 5 and thr <= 1.0:
        thr *= 1.1
    return thr


def build_classifier_pipeline(thr: float, classifier_name: str, classifier_params: dict) -> Pipeline:
    """An unfit Pipeline(ANOVA feature selection, classifier) using the given
    p-value threshold. GenericUnivariateSelect(mode="fpr", param=thr) selects
    features with p < thr -- the built-in sklearn equivalent of the manual
    xP < thr mask this replaced (verified to select identical voxels,
    including NaN p-value handling for zero-variance voxels)."""
    Cls = import_from_path(classifier_name)
    return Pipeline([
        ("feature_selection", GenericUnivariateSelect(score_func=f_classif, mode="fpr", param=thr)),
        ("classifier", Cls(**classifier_params)),
    ])


# cross_validation
def model_classification(training_data, training_labels, feat_p: float, classifier_name: str, classifier_params: dict):
    """Fit an ANOVA-feature-selection + classifier Pipeline. Bundling both
    steps into one estimator -- rather than externally tracking a voxel
    boolean mask, as before -- means the whole thing can be refit as a
    single unit, which permutation_significance() below needs (it refits
    fresh on every permutation's shuffled training labels)."""

    print("Training classifier...")

    thr = resolve_feature_threshold(training_data, training_labels, feat_p)
    pipe = build_classifier_pipeline(thr, classifier_name, classifier_params)
    pipe.fit(training_data, training_labels)

    return pipe


def model_performance(pipe, testing_data, testing_labels):

    print("Testing model performance...")

    clf = pipe.named_steps["classifier"]
    xfeat = pipe.named_steps["feature_selection"].get_support()

    # classes the classifier was actually trained on -- not np.unique(testing_labels),
    # which would drift shape-to-shape if a given fold's held-out data happens to be
    # missing one of 3+ classes entirely, breaking cross-fold averaging in main().
    xclass = clf.classes_
    n_class = len(xclass)

    # keep original (whole-brain) feature count to size impa_full below
    n_samples, n_features = testing_data.shape

    # apply model -- the pipeline applies feature selection internally, so the
    # full (unsliced) testing_data goes in
    xpred = pipe.predict(testing_data)

    # total model accuracy
    ttl_score = accuracy_score(testing_labels, xpred)

    # special case where classifier is binary (yes/no) -- only codes one label
    if n_class == 2:

        # voxel weights
        impa = np.vstack((clf.coef_, -clf.coef_))
        ## important volume 0 and volume 1 are mat*-1 of eachother.. Compute 1 tail ttests always
        print(impa.shape)

        # evidence: sigmoid on decision function → class-1 prob; other is 1-p
        d = pipe.decision_function(testing_data)
        p1 = 1.0 / (1.0 + np.exp(-d))
        p0 = 1.0 - p1
        xevi = np.vstack([p0, p1]).T

    else:

        # voxel weights
        impa = clf.coef_

        # evidence: multinomial OV(A)R decision_function → pass through sigmoid per class
        d = pipe.decision_function(testing_data)  # shape: (n, n_class)
        xevi = 1.0 / (1.0 + np.exp(-d))

    # store importance values in original dataformat
    impa_full = np.zeros((n_class, n_features), dtype=impa.dtype)
    impa_full[:, xfeat] = impa

    # normalized confusion matrix, and evidence matrix
    acc_mx = np.zeros((n_class, n_class))
    evi_mx = np.zeros((n_class, n_class))
    for xx in range(n_class):
        cls = xclass[xx]
        idxs = np.where(testing_labels == cls)[0]
        if idxs.size == 0:
            continue
        pred_slice = xpred[idxs]
        evi_slice  = xevi[idxs, :]
        for yy in range(n_class):
            ycond = xclass[yy]
            acc_mx[xx, yy] = (pred_slice == ycond).sum() / len(pred_slice)
            evi_mx[xx, yy] = float(np.mean(evi_slice[:, yy])) if len(evi_slice) else 0.0

    # ROC/AUC per class for this fold

    # One-vs-rest indicator matrix
    Y = (testing_labels[:, None] == xclass[None, :]).astype(np.uint8)
    # AUC per class → returns 1D array length n_class
    auc = np.array([
        roc_auc_score(Y[:, j], xevi[:, j])
        if Y[:, j].min() != Y[:, j].max()  # avoid single-class error
        else np.nan
        for j in range(n_class)
    ], dtype=float)

    # record model results
    xout = {
        'total_scores': ttl_score,
        'accuracy': acc_mx,  #acc_mx
        'evidence': evi_mx,  #evi_mx

        'auc': auc
    }

    return xout, impa_full


def permutation_significance(training_data, training_labels, testing_data, testing_labels, n_permutations, random_state,
                              feat_p: float, classifier_name: str, classifier_params: dict):
    """Real-vs-null significance for the held-out test evaluation, via
    sklearn.model_selection.permutation_test_score (the tool nilearn's own
    decoding docs recommend for exactly this fMRI-classification case).

    A PredefinedSplit with test_fold=-1 for every training row and =0 for
    every testing row encodes today's fixed train/test partition as a single
    CV fold. permutation_test_score then reshuffles the combined label
    vector and reruns that same fit/score structure -- including feature
    selection, since the Pipeline from model_classification() is refit fresh
    each round -- which is the textbook-correct way to build a null
    distribution for a fixed train/test split (not a naive shuffle done
    outside the fit/CV structure, which would be optimistic).

    One permutation_test_score call per metric (accuracy, roc_auc_ovr --
    the same one-vs-rest convention model_performance()'s own AUC already
    uses, just via sklearn's built-in scorer instead of a second hand-rolled
    computation)."""
    X = np.vstack([training_data, testing_data])
    y = np.concatenate([training_labels, testing_labels])
    test_fold = np.concatenate([
        np.full(len(training_labels), -1),
        np.zeros(len(testing_labels)),
    ])
    cv = PredefinedSplit(test_fold)

    # same widened threshold the real model fit used (see resolve_feature_threshold),
    # resolved once from the real (unpermuted) training data and held fixed as a
    # pipeline hyperparameter across every permutation round
    thr = resolve_feature_threshold(training_data, training_labels, feat_p)

    rows = []
    for metric in ("accuracy", "roc_auc_ovr"):
        pipe = build_classifier_pipeline(thr, classifier_name, classifier_params)
        score, _, p_value = permutation_test_score(
            pipe, X, y, cv=cv, scoring=metric,
            n_permutations=n_permutations, random_state=random_state, n_jobs=-1,
        )
        print(f"  permutation test [{metric}]: real={score:.4f}, p={p_value:.4g} ({n_permutations} permutations)")
        rows.append({"metric": metric, "real_score": score, "p_value": p_value, "n_permutations": n_permutations})

    return pd.DataFrame(rows)


def timecourse_decoding(pipe, timecourse_data, timecourse_labels, timecourse_df, regressor_categories,
                         feat_p: float, subject_id: str, model_descr: str):
    """Predict the trained classifier on every already-recomputed timecourse-decoding
    volume. Returns (raw, summary):
      - raw: one row per volume actually decoded, with its own prediction and
        evidence_<category> columns -- the actual per-TR data, not an average.
      - summary: raw grouped by (window_index, regressor_label) and averaged across
        every trial sharing that group -- the confusion-style timecourse view."""

    # pipeline applies feature selection internally -- full (unsliced) data goes in
    predictions = pipe.predict(timecourse_data)
    global_accuracy = accuracy_score(timecourse_labels, predictions)
    print(f"Global accuracy: {global_accuracy:.4f}")

    evidence = decision_evidence(pipe, timecourse_data)

    code_to_label = {i + 1: cat for i, cat in enumerate(regressor_categories)}

    raw = timecourse_df.reset_index(drop=True).copy()
    raw["predicted_label"] = [code_to_label.get(p, p) for p in predictions]
    raw["correct"] = predictions == timecourse_labels
    for i, cat in enumerate(regressor_categories):
        raw[f"evidence_{cat}"] = evidence[:, i]

    n_selected = int(pipe.named_steps["feature_selection"].get_support().sum())
    raw["threshold_p"] = feat_p
    raw["selected_voxels"] = n_selected
    raw["whole_voxels"] = timecourse_data.shape[1]
    raw["feature_percent"] = 100 * n_selected / timecourse_data.shape[1]

    evidence_cols = [c for c in raw.columns if c.startswith("evidence")]
    other_cols = [c for c in raw.columns if not c.startswith("evidence")]
    raw = raw[other_cols + evidence_cols]
    raw.insert(1, "model_descr", model_descr)  # "subject" is already a column, from timecourse_df's own BIDS entity

    summary = summarize_decoding(raw, regressor_categories, subject_id, model_descr)

    return raw, summary


def summarize_decoding(raw: pd.DataFrame, regressor_categories: list, subject_id: str, model_descr: str) -> pd.DataFrame:
    """Collapse a raw (one-row-per-decoded-TR) decoding table down to one row per
    (window_index, regressor_label), averaging Accuracy/evidence across every trial
    sharing that group."""
    rows = []
    for (window_index, regressor_label), group in raw.groupby(TIMECOURSE_GROUPING, sort=False):
        row = {
            "subject": subject_id,
            "model_descr": model_descr,
            "window_index": window_index,
            "regressor_label": regressor_label,
            "trial_count": len(group),
            "Accuracy": group["correct"].mean(),
        }
        for cat in regressor_categories:
            row[f"evidence_{cat}"] = group[f"evidence_{cat}"].mean()
        row["threshold_p"] = group["threshold_p"].mean()
        row["selected_voxels"] = group["selected_voxels"].mean()
        row["whole_voxels"] = group["whole_voxels"].mean()
        row["feature_percent"] = group["feature_percent"].mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    evidence_cols = [c for c in summary.columns if c.startswith("evidence")]
    other_cols = [c for c in summary.columns if not c.startswith("evidence")]
    return summary[other_cols + evidence_cols]


# =====================================================
# Main Workflow
# =====================================================

def main(args):
    subject_id = args.subject
    analysis_output_dir = args.analysis_output_dir
    master_spreadsheet_file = args.master_spreadsheet

    full_cfg = load_config(args.config)

    event_cfg = full_cfg["event_extraction"]
    derivatives_root = resolve_config_root(
        event_cfg, "derivatives_root", event_cfg["bids_root"], "event_extraction.derivatives_root"
    )
    # masks are typically co-located with preprocessed/derivative BOLD data, but can
    # be overridden independently (e.g. a separate hand-drawn ROI directory).
    mask_root = resolve_config_root(
        full_cfg["model"].get("mask", {}), "mask_root", derivatives_root, "model.mask.mask_root"
    )
    model_conditions = full_cfg["model_conditions"]

    training_conditions = model_conditions["training"]["conditions"]
    testing_conditions = model_conditions["testing"]["conditions"]
    timecourse_conditions = model_conditions["timecourse_decoding"]["conditions"]
    timecourse_window = model_conditions["timecourse_decoding"]["window"]

    # class label order shared across training/testing/timecourse regressor codes
    regressor_categories = list(training_conditions.keys())

    # model settings (mask/featureSelection/classifier/cv/desc)
    model_cfg = full_cfg["model"]
    model_descr = quick_safe(model_cfg["desc"])
    mask_pattern_template = model_cfg["mask"]["mask_pattern"]
    feat_p = model_cfg["featureSelection"]["feat_p"]
    classifier_name = model_cfg["classifier"]["name"]
    classifier_params = model_cfg["classifier"]["params"]

    print(f"Subject: {subject_id}")

    # ------------------------------------------------
    # Prepare Instructions
    # ------------------------------------------------

    master = pd.read_csv(
        master_spreadsheet_file,
        dtype={"subject": str, "session": str, "task": str, "trial_type": str}
    )

    # remove any bad rows
    count1 = master[master["subject"] == subject_id].shape[0]
    master = master[~(master["volume_of_interest"].isna() | np.isinf(master["volume_of_interest"]))]
    count2 = master[master["subject"] == subject_id].shape[0]
    print(f"Removing Bad Rows from Instructions Sheet... {count1 - count2} rows out of {count1}\n")

    subject_df = master[master["subject"] == subject_id]
    if subject_df.empty:
        raise SystemExit(f"No rows found for subject {subject_id!r} in {master_spreadsheet_file}")

    # -------------------------------------------------
    # Trial Pivot Table (sanity check, not used for modeling)
    # -------------------------------------------------

    trial_pivot = build_trial_pivot_table(subject_df)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, f"{subject_id}_trial_pivot.csv")
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    trial_pivot.to_csv(output_file, index=False)
    print(f"Trial pivot table (sanity check) saved to: {output_file}")

    # -------------------------------------------------
    # Load Data
    # -------------------------------------------------

    training_df = apply_regressor_codes(label_rows(subject_df, training_conditions), regressor_categories)
    testing_df = apply_regressor_codes(label_rows(subject_df, testing_conditions), regressor_categories)
    timecourse_labeled = apply_regressor_codes(label_rows(subject_df, timecourse_conditions), regressor_categories)
    timecourse_instr = build_timecourse_instructions(timecourse_labeled, timecourse_window)

    training_data, training_labels, training_ids, masker = load_images_and_mask(training_df, mask_pattern_template, mask_root)
    testing_data, testing_labels, testing_ids, masker = load_images_and_mask(testing_df, mask_pattern_template, mask_root)
    timecourse_data, timecourse_labels, timecourse_ids, masker = load_images_and_mask(timecourse_instr, mask_pattern_template, mask_root)

    training_df = training_df.loc[training_ids, :]
    testing_df = testing_df.loc[testing_ids, :]
    timecourse_instr = timecourse_instr.loc[timecourse_ids, :]
    print("...Done")

    # make sure labels are flat
    training_labels = training_labels.ravel()
    testing_labels = testing_labels.ravel()
    timecourse_labels = timecourse_labels.ravel()

    # -------------------------------------------------
    # K-Fold Cross Validation
    # -------------------------------------------------

    # ensure training data is balanced fold cv (drop extra volumes as needed)
    training_df_balanced = balance(training_df)
    training_df_balanced.reset_index(inplace=True)

    # lets process model k-fold times -- use "run" to generate folds
    ps = PredefinedSplit(training_df_balanced["run"])
    folds = list(ps.split())

    ii_results = []
    ii_impa = []

    for i, (train_idx, test_idx) in enumerate(folds, start=1):

        # in cross validation we take the training set and split it to train ~80% of the data
        print(f"Processing Fold {i}")
        xregs = training_labels[training_df_balanced.loc[train_idx, "index"].values]
        xpat  = training_data[training_df_balanced.loc[train_idx, "index"].values, :]

        xclf = model_classification(xpat, xregs, feat_p, classifier_name, classifier_params)

        # test model performance on hold out data
        holdout_xregs = training_labels[training_df_balanced.loc[test_idx, "index"].values]
        holdout_xpat  = training_data[training_df_balanced.loc[test_idx, "index"].values, :]

        xout, impa = model_performance(xclf, holdout_xpat, holdout_xregs)

        # store fold model performance and importance map (impa)
        ii_results.append(xout)
        ii_impa.append(impa)

    print("Storing cross-validation performance metrics.")

    # summarize model performance across all folds
    mean_results = average_fold_results(ii_results)
    mean_kfold_importance_map = np.mean(np.stack(ii_impa, axis=0), axis=0)

    output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "cv", f"{subject_id}" + "_cv_results_{metric}.csv")
    save_model_results(output_pattern, mean_results, regressor_categories)

    # -------------------------------------------------
    # Model Classification / Testing / Time Course Decoding
    # -------------------------------------------------

    print("Training classifier...")

    # train on full "training" set now
    xclf = model_classification(training_data, training_labels, feat_p, classifier_name, classifier_params)

    # record final model performance
    xout, importance_map = model_performance(xclf, testing_data, testing_labels)

    print("Time Course Decoding...")
    raw_decoding, summary_decoding = timecourse_decoding(
        xclf, timecourse_data, timecourse_labels, timecourse_instr, regressor_categories,
        feat_p, subject_id, model_descr,
    )

    output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_model_results_{metric}.csv")
    save_model_results(output_pattern, xout, regressor_categories)

    # -------------------------------------------------
    # Permutation-based significance test (optional -- model.permutation_test)
    # -------------------------------------------------

    permutation_test_cfg = full_cfg["model"].get("permutation_test")
    if permutation_test_cfg is not None:
        n_permutations = permutation_test_cfg.get("n_permutations", 1000)
        random_state = permutation_test_cfg.get("random_state", 0)
        print(f"Permutation testing ({n_permutations} permutations)...")
        permutation_results = permutation_significance(
            training_data, training_labels, testing_data, testing_labels, n_permutations, random_state,
            feat_p, classifier_name, classifier_params,
        )
        permutation_file = os.path.join(
            analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_permutation_test.csv"
        )
        permutation_results.to_csv(permutation_file, index=False)
        print(f"Permutation test results saved to: {permutation_file}")

    # -------------------------------------------------
    # Importance Map
    # -------------------------------------------------

    # importance map averaged across the internal training-CV folds above --
    # a within-training generalization diagnostic
    img1 = masker.inverse_transform(mean_kfold_importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "cv", f"{subject_id}" + "_cv_impa_native.nii.gz")
    img1.to_filename(output_file)

    # final model importance map
    img = masker.inverse_transform(importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_impa_native.nii.gz")
    img.to_filename(output_file)

    # -------------------------------------------------
    # Time Course Decoding output -- raw (one row per decoded TR) + summary
    # (raw grouped by window_index/regressor_label and averaged across trials)
    # -------------------------------------------------

    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_decoding_results.csv")
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    raw_decoding.to_csv(output_file, index=False)

    summary_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_summary_decoding_results.csv")
    summary_decoding.to_csv(summary_file, index=False)

    print(f"Results saved to: {output_file} (raw) and {summary_file} (summary)")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    with track_runtime():
        main(args)
