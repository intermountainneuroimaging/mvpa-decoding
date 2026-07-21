#!/usr/bin/env python3

"""
Within-subject MVPA decoding script.

Usage:
    python mvpa_workflow.py --subject 4057 --config gm_object_classifier.json \
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir $PWD
"""

import os
import argparse
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from pathlib import Path
from types import SimpleNamespace
import json
import importlib
import re

# processing
try:
    from nilearn.maskers import NiftiMasker
except Exception:
    from nilearn.input_data import NiftiMasker

from sklearn.preprocessing import StandardScaler

# classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import PredefinedSplit
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score

from mvpa_common import evaluate_query_node, compute_volume_range, resolve_window_times, build_trial_pivot_table, resolve_config_root

# grouping used for the timecourse decoding output -- the relative timepoint
# within each event's decode window, crossed with the classification label.
TIMECOURSE_GROUPING = ["window_index", "regressor_label"]


# =====================================================
# Argument Parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run within-subject decoding.")

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID (e.g., 4057)"
    )

    parser.add_argument(
        "--analysis-output-dir",
        required=True,
        help="Filepath to outputs."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the mvpa config JSON (event_extraction + model_conditions + model sections)"
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

def label_rows(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Tag rows matching any condition's query with a 'regressor_label' column
    (first matching condition wins, in dict-insertion order), dropping rows
    that match none."""
    labeled = []
    for name, query in conditions.items():
        mask = evaluate_query_node(query, df)
        subset = df[mask].copy()
        subset["regressor_label"] = name
        labeled.append(subset)
    combined = pd.concat(labeled)
    return combined[~combined.index.duplicated(keep="first")]


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


def stack_patterns(df):
    return np.vstack(df["patterns"].values).reshape(
        df.shape[0],
        len(df.patterns.iloc[0])
    )


def linear_weight_map(clf, n_voxels, feat_mask=None, class_index=0):
    """
    Returns full-length voxel weight vector (n_voxels,).
    If feat_mask is provided (boolean mask of selected voxels),
    fills unselected voxels with 0.
    """
    # For binary LR/SVM: coef_ shape is (1, n_features)
    w = np.ravel(clf.coef_)  # (n_features_selected,) or (n_features_full,)

    if feat_mask is None:
        if w.size != n_voxels:
            raise ValueError("n_voxels doesn't match coef_ length; provide feat_mask.")
        return w

    full = np.zeros(n_voxels, dtype=float)
    full[feat_mask] = w
    return full


class ShapeError(Exception):
    pass


def get_single_match(pattern: str) -> str:
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected 1 file, found {len(matches)}:\n" +
            "\n".join(str(m) for m in matches)
        )

    return str(matches[0])


_bold_header_cache = {}


def get_bold_header_info(boldfile: str):
    """Return (tr, n_frames) for a boldfile, read once and cached."""
    if boldfile not in _bold_header_cache:
        header = nib.load(boldfile).header
        _bold_header_cache[boldfile] = (float(header.get_zooms()[3]), int(header.get_data_shape()[-1]))
    return _bold_header_cache[boldfile]


_masker_cache = {}


def load_images_and_mask(labeled_df: pd.DataFrame):
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
            mask_pattern = cfg.mask.mask_pattern.format(subject=subject, session=session)
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


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def quick_safe(name):
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))


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


# cross_validation
def model_classification(training_data, training_labels):

    "return feature selection idx and model object"

    print("Training classifier...")

    xF, xP = f_classif(training_data, training_labels)
    xF = np.nan_to_num(xF)
    xP[np.isnan(xP)] = 1
    thr = cfg.featureSelection.feat_p
    # ensure at least 5 voxels are selected for feature selection
    while np.sum(xP < thr) < 5 and thr <= 1.0:
        thr *= 1.1

    xfeat = xP < thr
    n_selvoxs = np.sum(xfeat)

    training_data_xfeat = training_data[:, xfeat]

    # maximually flexible classifier
    Cls = import_from_path(cfg.classifier.name)
    classifier = Cls(**cfg.classifier.params.__dict__)

    clf = classifier.fit(training_data_xfeat, training_labels)

    return clf, xfeat


def model_performance(clf, xfeat, testing_data, testing_labels):

    print("Testing model performance...")

    # classes the classifier was actually trained on -- not np.unique(testing_labels),
    # which would drift shape-to-shape if a given fold's held-out data happens to be
    # missing one of 3+ classes entirely, breaking cross-fold averaging in main().
    xclass = clf.classes_
    n_class = len(xclass)

    # prep testing data with feature selection / mask - keep info to recover original size
    n_samples, n_features = testing_data.shape
    testing_data_xfeat = testing_data[:, xfeat]

    # apply model
    xpred = clf.predict(testing_data_xfeat)

    # total model accuracy
    ttl_score = accuracy_score(testing_labels, xpred)

    # special case where classifier is binary (yes/no) -- only codes one label
    if n_class == 2:

        # voxel weights
        impa = np.vstack((clf.coef_, -clf.coef_))
        ## important volume 0 and volume 1 are mat*-1 of eachother.. Compute 1 tail ttests always
        print(impa.shape)

        # evidence: sigmoid on decision function → class-1 prob; other is 1-p
        d = clf.decision_function(testing_data_xfeat)
        p1 = 1.0 / (1.0 + np.exp(-d))
        p0 = 1.0 - p1
        xevi = np.vstack([p0, p1]).T

    else:

        # voxel weights
        impa = clf.coef_

        # evidence: multinomial OV(A)R decision_function → pass through sigmoid per class
        d = clf.decision_function(testing_data_xfeat)  # shape: (n, n_class)
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



def timecourse_decoding(clf, xfeat, timecourse_data, timecourse_labels, timecourse_df, regressor_categories):

    # apply feature selection
    timecourse_data_xfeat = timecourse_data[:, xfeat]

    predictions = clf.predict(timecourse_data_xfeat)
    global_accuracy = accuracy_score(timecourse_labels, predictions)

    print(f"Global accuracy: {global_accuracy:.4f}")

    evidence = decision_evidence(clf, timecourse_data_xfeat)
    timecourse_df = timecourse_df.reset_index(drop=True)
    timecourse_df["evidence"] = list(evidence)

    # -------------------------------------------------
    # Group-level metrics
    # -------------------------------------------------

    out = (
        timecourse_df
        .groupby(TIMECOURSE_GROUPING)
        .apply(lambda x: x.index.tolist())
        .reset_index(name="data_index")
    )

    for idx, row in out.iterrows():
        inds = row["data_index"]

        out.loc[idx, "trial_count"] = len(inds)

        group_acc = accuracy_score(
            timecourse_labels[inds],
            predictions[inds]
        )

        evidence_by_group = np.mean(evidence[inds], axis=0)

        out.loc[idx, "Accuracy"] = group_acc

        for i, cat in enumerate(regressor_categories):
            out.loc[idx, f"evidence_{cat}"] = evidence_by_group[i]

    out["threshold_p"] = cfg.featureSelection.feat_p
    out["selected_voxels"] = timecourse_data_xfeat.shape[1]
    out["whole_voxels"] = timecourse_data.shape[1]
    out["feature_percent"] = 100 * timecourse_data_xfeat.shape[1] / timecourse_data.shape[1]

    evidence_cols = [c for c in out.columns if c.startswith("evidence")]
    other_cols = [c for c in out.columns if not c.startswith("evidence")]
    out = out[other_cols + evidence_cols]

    out.insert(0, "subject", subject_id)
    out.insert(1, "model_descr", model_descr)

    out.drop(columns="data_index", inplace=True)

    return out



# =====================================================
# Main Workflow
# =====================================================

def main():

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

    training_data, training_labels, training_ids, masker = load_images_and_mask(training_df)
    testing_data, testing_labels, testing_ids, masker = load_images_and_mask(testing_df)

    training_df = training_df.loc[training_ids, :]
    testing_df = testing_df.loc[testing_ids, :]
    print("...Done")

    # make sure labels are flat
    training_labels = training_labels.ravel()
    testing_labels = testing_labels.ravel()

    # -------------------------------------------------
    # K-Fold Cross Validation
    # -------------------------------------------------

    # ensure training data is balanced fold cv (drop extra volumes as needed)
    training_df_balanced = balance(training_df)
    training_df_balanced.reset_index(inplace=True)

    # lets process model k-fold times -- use "run" to generate folds
    ps = PredefinedSplit(training_df_balanced["run"])
    folds = list(ps.split())
    n_folds = len(folds)

    ii_results = []
    ii_impa = []

    for i, (train_idx, test_idx) in enumerate(folds, start=1):

        # in cross validation we take the training set and split it to train ~80% of the data
        print(f"Processing Fold {i}")
        xregs = training_labels[training_df_balanced.loc[train_idx, "index"].values]
        xpat  = training_data[training_df_balanced.loc[train_idx, "index"].values, :]

        xclf, xfeat = model_classification(xpat, xregs)

        # test model performance on hold out data
        holdout_xregs = training_labels[training_df_balanced.loc[test_idx, "index"].values]
        holdout_xpat  = training_data[training_df_balanced.loc[test_idx, "index"].values, :]

        xout, impa = model_performance(xclf, xfeat, holdout_xpat, holdout_xregs)

        # store fold model performance and importance map (impa)
        ii_results.append(xout)
        ii_impa.append(impa)

    print("Storing cross-validation performance metrics.")

    # summarize model perfromance across all folds
    mean_results = {}

    for k in ii_results[0].keys():
        values = [res[k] for res in ii_results]

        if np.isscalar(values[0]):
            mean_results[k] = float(np.mean(values))
        else:
            mean_results[k] = np.mean(np.stack(values, axis=0), axis=0)

    mean_kfold_importance_map = np.mean(np.stack(ii_impa, axis=0), axis=0)

    output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "cv", f"{subject_id}" + "_cv_results_{metric}.csv")
    save_model_results(output_pattern, mean_results, regressor_categories)

    # -------------------------------------------------
    # Model Classification
    # -------------------------------------------------

    print("Training classifier...")

    # train on full "training" set now
    xclf, xfeat = model_classification(training_data, training_labels)

    # record final model performance
    xout, importance_map = model_performance(xclf, xfeat, testing_data, testing_labels)

    output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_model_results_{metric}.csv")
    save_model_results(output_pattern, xout, regressor_categories)

    # -------------------------------------------------
    # Importance Map
    # -------------------------------------------------

    # kfold importance maps (averaged across folds)
    img1 = masker.inverse_transform(importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "cv", f"{subject_id}" + "_cv_impa_native.nii.gz")
    img1.to_filename(output_file)

    # final trained model importance map
    img = masker.inverse_transform(importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_impa_native.nii.gz")
    img.to_filename(output_file)

    # -------------------------------------------------
    # Time Course Decoding
    # -------------------------------------------------

    print("Time Course Decoding...")

    timecourse_labeled = apply_regressor_codes(label_rows(subject_df, timecourse_conditions), regressor_categories)
    timecourse_instr = build_timecourse_instructions(timecourse_labeled, timecourse_window)

    timecourse_data, timecourse_labels, timecourse_ids, _ = load_images_and_mask(timecourse_instr)
    timecourse_instr = timecourse_instr.loc[timecourse_ids, :]
    timecourse_labels = timecourse_labels.ravel()

    out = timecourse_decoding(xclf, xfeat, timecourse_data, timecourse_labels, timecourse_instr, regressor_categories)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_decoding_results.csv")
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":

    # access arguments from all functions
    args = parse_args()
    print(args)

    subject_id = args.subject
    analysis_output_dir = args.analysis_output_dir
    master_spreadsheet_file = args.master_spreadsheet

    # load mvpa configuration
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

    # model settings (mask/featureSelection/classifier/cv/desc), dot-access
    cfg = dict_to_namespace(full_cfg["model"])
    model_descr = quick_safe(cfg.desc)

    # run main -- track performance
    with track_runtime():
        main()
