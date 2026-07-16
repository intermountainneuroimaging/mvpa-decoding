#!/usr/bin/env python3

"""
Within-subject MVPA decoding script.

Usage:
    python mvpa_workflow.py --subject 4003 --input-scaffold inputs.json --model-config vvs_object_classifier.json --analysis-output-dir $PWD
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
import fnmatch
from pathlib import Path

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

#  == extra stuff to eventually write into arguements ==

session_id = "*1"
tr = 0.46



# =====================================================
# Argument Parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run within-subject decoding.")

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID (e.g., 4003)"
    )
    
    parser.add_argument(
        "--analysis-output-dir",
        required=True,
        help="Filepath to outputs."
    )

    parser.add_argument(
        "--input-scaffold",
        required=True,
        help="Path to JSON file containing all input scaffolding (see... [link] for more details)"
    )
    
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to JSON file containing all configuration settings for MVPA analysis (see... [link] for more details)"
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

def balance(xdf: pd.DataFrame) -> pd.DataFrame:
    df = xdf.copy()
    df["ID"] = df.index  # keep original row id

    # target number of rows to keep per regressor (lowest common denominator)
    k = df.groupby("regressor").size().min()

    # count rows per (regressor, selector, identifier) to prioritize fuller trials
    pair_counts = (
        df.groupby(["regressor", "selector", "identifier"])
          .size()
          .rename("pair_n")
          .reset_index()
    )

    # merge counts back so each row knows how "full" its (selector, identifier) group is
    df2 = df.merge(pair_counts, on=["regressor", "selector", "identifier"], how="left")

    # sort so we:
    #  1) for each regressor, consider the most-populated (selector,identifier) pairs first
    #  2) within a pair, keep highest pul_vols first (then we’ll cap total to k)
    df2 = df2.sort_values(
        ["regressor", "pair_n", "selector", "identifier", "pul_vols"],
        ascending=[True, False, True, True, False]
    )

    # take first k rows per regressor (after prioritization + within-pair trimming)
    out = df2.groupby("regressor", group_keys=False).head(k)

    # optional: final ordering for downstream use
    out = out.sort_values(["selector", "identifier", "pul_vols"])

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


def load_images_and_mask(instr, target_files, root):
    
    # Start by limiting the spreadsheet to the provided subject (always)
    intr_slice = instr.loc[(instr["subject"] == subject_id)]

    # nifti data is stored by run - we need to access by looping
    

    # load mask -- come back to!
    mask_pattern = cfg.mask.mask_pattern
    mask_file = get_single_match(mask_pattern.format(**{**globals(), **locals()}))
    print(f"Using Mask File: {mask_file}")

    masker = NiftiMasker(mask_img=mask_file, standardize=False, detrend=False, t_r=tr)

    matrices = []
    labels = []
    indices = []

    # nifti data is stored by run - we need to access by looping
    for f in target_files:
        
        mask = intr_slice["bold_file"].astype(str).apply(lambda p: fnmatch.fnmatchcase(Path(p).name, f))
        itr_instr = intr_slice[mask]
                
        bold_pattern = itr_instr["bold_file"].unique()
        if len(bold_pattern) > 1:
            raise RuntimeError("Expected to locate a single matching file in instruction sheet... not sure how to proceed")
        
        bold_pattern = bold_pattern[0]
        bold_file = get_single_match(bold_pattern.format(root=root))

        # apply mask
        masked_data = masker.fit_transform(bold_file)

        # apply z-transform
        z_patterns = StandardScaler().fit_transform(masked_data)
        z_patterns = np.nan_to_num(z_patterns)

        # crop data to selected volumes
        vols = (
            pd.to_numeric(itr_instr["pul_vols"], errors="raise")
            .astype(int)
            .to_numpy()
        )

        z_patterns = z_patterns[vols, :]
        # stack image data to match instructions
        matrices.append(z_patterns)

        # stack the regression labels as well to be 100% sure they data and labels align
        labels.extend(itr_instr["regressor"].to_numpy())
        
        # stack selected indies to later extract volume metadata
        indices.extend(itr_instr.index.to_numpy())
        
        print(f"Sucessfully loaded: {bold_file}")

    # all image data stacked
    X = np.vstack(matrices)
    Y = np.array(labels).reshape(-1, 1)
    idx = np.array(indices)

    # Check X and Y have same number of rows (observations)
    if X.shape[0] != Y.shape[0]:
        raise ShapeError("Image Data and Labels Do No Align... Can't Proceed!!")
        
    return X, Y, idx, masker
    
    
def assign_windows_by_seq(group, seq_cols, window_col):
    group = group.copy()
    group[window_col] = group.groupby(seq_cols).cumcount() + 1
    return group


def load_input_scaffold(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)
    

def import_from_path(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def default_config():
    return {
        "config_version": "1.0",
        "created_by": "auto",
        "notes": "Default MVPA config",
        "targets": [
            {
                "name": "default_classifier",
                "regressor_column": "clusterID",
                "categories": ["face", "place"],
                "drop_na": True
            }
        ],
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
                "multi_class": "ovr",
                "class_weight": "balanced"
            }
        },
        "cv": {
            "strategy": "GroupKFold",
            "n_splits": "infer"
        },
        "decoding": {
            "grouping_categories": ["window"]
        }
    }


def merge_with_defaults(user_cfg):
    base = default_config()

    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                recursive_update(d[k], v)
            else:
                d[k] = v
        return d

    return recursive_update(base, user_cfg)


def load_config(cfg_path=None):
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            user_cfg = json.load(f)
        cfg = merge_with_defaults(user_cfg)
    else:
        cfg = default_config()

    return cfg


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

    
def quick_safe(name):
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))


def make_safe_foldername(name, max_length=100):
    """
    Convert an arbitrary string into a filesystem-safe folder name.

    Parameters
    ----------
    name : str
        Input string to sanitize.
    max_length : int, optional
        Maximum allowed length of the folder name.

    Returns
    -------
    str
        Safe folder name.
    """

    # Normalize unicode (removes accents, etc.)
    name = unicodedata.normalize("NFKD", str(name))
    name = name.encode("ascii", "ignore").decode("ascii")

    # Replace spaces with underscore
    name = name.replace(" ", "_")

    # Remove illegal characters (Windows + Unix safe)
    name = re.sub(r'[<>:"/\\|?*\']', '', name)

    # Keep only alphanumeric, dash, underscore, dot
    name = re.sub(r'[^A-Za-z0-9._-]', '', name)

    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)

    # Strip leading/trailing punctuation
    name = name.strip("._-")

    # Truncate if too long
    name = name[:max_length]

    # Fallback if empty
    if not name:
        name = "untitled"

    return name


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
    
    xclass = np.unique(testing_labels)
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
    Y = (testing_labels[:, None] == np.arange(1, n_class + 1)).astype(np.uint8)
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
    

    
def timecourse_decoding(clf, xfeat, testing_data, testing_labels, testing_df):
    
    # apply feature selection
    testing_data_xfeat = testing_data[:, xfeat]
    
    predictions = clf.predict(testing_data_xfeat)
    global_accuracy = accuracy_score(testing_labels, predictions)

    print(f"Global accuracy: {global_accuracy:.4f}")

    evidence = decision_evidence(clf, testing_data_xfeat)
    testing_df["evidence"] = list(evidence)

    # -------------------------------------------------
    # Group-level metrics
    # -------------------------------------------------
    
    # we need local index values going forward
    testing_df.reset_index(drop=True, inplace=True)
    
    # create a "window" descriptor to apply decoding for each relative trial timepoint 
    testing_df = testing_df.groupby('selector', group_keys=False).apply(
        assign_windows_by_seq, seq_cols=['identifier'], window_col='window'
    )
    
    out = (
        testing_df
        .groupby(grouping_categories)
        .apply(lambda x: x.index.tolist())
        .reset_index(name="data_index")
    )

    for idx, row in out.iterrows():
        inds = row["data_index"]

        out.loc[idx, "trial_count"] = len(inds)

        group_acc = accuracy_score(
            testing_labels[inds],
            predictions[inds]
        )

        evidence_by_group = np.mean(evidence[inds], axis=0)

        out.loc[idx, "Accuracy"] = group_acc

        for i, cat in enumerate(regressor.categories):
            out.loc[idx, f"evidence_{cat}"] = evidence_by_group[i]

    out["threshold_p"] = cfg.featureSelection.feat_p
    out["selected_voxels"] = testing_data_xfeat.shape[1]
    out["whole_voxels"] = testing_data.shape[1]
    out["feature_percent"] = 100 * testing_data_xfeat.shape[1] / testing_data.shape[1]

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
#     print(f"Mask: {maskname}")
#     print(f"Reading data from: {root}")
    
    # ------------------------------------------------
    # Prepare Instructions
    # ------------------------------------------------
    
    # load set of instructions for all subjects, all runs, all trials
    instr = pd.read_csv(
        'master_spreadsheet_with_bold_file.csv',
        dtype={'subject': str, 'trial_type': str, 'operation': str}
    )
    
    # Lets force an order here...
    instr["regressor"] = pd.Categorical(
        instr[regressor.column],
        categories=regressor.categories,
        ordered=True
    )

    # make sure regressor labels are numeric (starting from here)
    instr["regressor"] = instr["regressor"].cat.codes + 1
    
    # remove any bad rows
    print("Reviewing Bad Rows from Instructions Sheet...")
    count1 = instr[instr["subject"] == subject_id].shape[0]
    
    # drop them
    instr = instr[~(instr["pul_vols"].isna() | np.isinf(instr["pul_vols"]))]
    count2 = instr[instr["subject"] == subject_id].shape[0]
    
    print(f"Removing Bad Rows from Instructions Sheet... {count1-count2} rows out of {count1}\n")
    
    # -------------------------------------------------
    # Logging Configurations
    # -------------------------------------------------
    # TODO
    
    # -------------------------------------------------
    # Load Data
    # -------------------------------------------------
    
    # training data
    training_files = inputs.trainingdata.funcfiles
    root = inputs.trainingdata.root
    training_data, training_labels, training_ids, masker = load_images_and_mask(instr, training_files, root)
    
    # testing data
    testing_files = inputs.testingdata.funcfiles
    root = inputs.testingdata.root
    testing_data, testing_labels, testing_ids, masker = load_images_and_mask(instr, testing_files, root)
    
    training_df = instr.loc[training_ids,:]
    testing_df = instr.loc[testing_ids,:]
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

    # lets process model k-fold times -- use "RUN" (aka selector) to generate folds
    ps = PredefinedSplit(training_df_balanced["selector"])
    folds = list(ps.split())
    n_folds = len(folds)

    ii_results = []
    ii_impa = []

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        
        # in cross validation we take the training set and split it to train ~80% of the data
        print(f"Processing Fold {i}")
        xregs = training_labels[training_df_balanced.loc[train_idx,"index"].values]
        xpat  = training_data[training_df_balanced.loc[train_idx,"index"].values,:]

        xclf, xfeat = model_classification(xpat, xregs)

        # test model performance on hold out data
        holdout_xregs = training_labels[training_df_balanced.loc[test_idx,"index"].values]
        holdout_xpat  = training_data[training_df_balanced.loc[test_idx,"index"].values,:]
        
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
    
    output_pattern = os.path.join(analysis_output_dir, model_descr,subject_id, "cv",f"{subject_id}"+"_cv_results_{metric}.csv")
    save_model_results(output_pattern, mean_results, regressor.categories)

    # -------------------------------------------------
    # Model Classification
    # -------------------------------------------------

    print("Training classifier...")
    
    # train on full "training" set now
    xclf, xfeat = model_classification(training_data, training_labels)
    
    # record final model performance
    xout, importance_map = model_performance(xclf, xfeat, testing_data, testing_labels)
    
    output_pattern = os.path.join(analysis_output_dir, model_descr,subject_id, "model",f"{subject_id}"+"_model_results_{metric}.csv")
    save_model_results(output_pattern, xout, regressor.categories)

    # -------------------------------------------------
    # Importance Map
    # -------------------------------------------------
    
    # kfold importance maps (averaged across folds)
    img1 = masker.inverse_transform(importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr,subject_id, "cv",f"{subject_id}"+"_cv_impa_native.nii.gz")
    img1.to_filename(output_file)
    
    # final trained model importance map
    img = masker.inverse_transform(importance_map)
    output_file = os.path.join(analysis_output_dir, model_descr,subject_id, "model",f"{subject_id}"+"_impa_native.nii.gz")
    img.to_filename(output_file)

    # -------------------------------------------------
    # Time Course Decoding
    # -------------------------------------------------

    print("Time Course Decoding...")
    
    out = timecourse_decoding(xclf, xfeat, testing_data, testing_labels, testing_df)
    output_file = os.path.join(analysis_output_dir, model_descr,subject_id, "decoding",f"{subject_id}"+"_decoding_results.csv")
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    
    # access arguments from all functions
    args = parse_args()
    print(args)
    
    subject_id = args.subject
    config_file = args.model_config
    inputs_config_file = args.input_scaffold
    analysis_output_dir = args.analysis_output_dir

    
    # load mvpa configurations 
    cfg = dict_to_namespace(load_config(config_file))
    
    # load mvpa inputs
    inputs = dict_to_namespace(load_input_scaffold(inputs_config_file))
    
    mask_pattern = cfg.mask.mask_pattern
    model_descr = quick_safe(cfg.desc)
    
    # set up remaining needed configuration settings
    
    # model regressor
    regressor = SimpleNamespace(
        column = cfg.targets[0].regressor_column,
        categories = cfg.targets[0].categories
    )

    # decoding output organization 
    grouping_categories = cfg.decoding.grouping_categories
    
    # run main -- track performance
    with track_runtime():
        main()

# python mvpa_workflow.py --subject 4001 --mask native_vvs_transformed_mask.nii.gz --model-descr vvs_object_classifier