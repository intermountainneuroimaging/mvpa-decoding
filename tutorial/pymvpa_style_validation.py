#!/usr/bin/env python3

"""
External validation of mvpa_generalization_workflow.py's Haxby-dataset results, using
PyMVPA's own documented methodology instead of PyMVPA itself -- pymvpa2
(last released ~2020) isn't installable here: its legacy setup.py needs
numpy.distutils, which depends on Python's stdlib distutils, removed in
Python 3.12 (PEP 632). Pinning numpy<2/scipy<1.14 doesn't help (the
blocker is the Python version, not numpy's), and no Python 3.10/3.11 is
available on this machine as a fallback.

So this reimplements PyMVPA's own canonical approach for exactly this
dataset, grounded in their tutorial docs (fetched from pymvpa.org):
  - Their explicit "replicates the original Haxby et al. (2001) study"
    classifier is kNN with correlation distance -- for k=1 against
    per-category training averages, that's a nearest-centroid correlation
    (template-matching) classifier, the same logic the 2001 paper itself
    used.
  - Their alternative canonical classifier is linear SVM.
  - Their default cross-validation is leave-one-chunk-out (chunk = run
    here) via NFoldPartitioner.

Deliberately does NOT import anything from mvpa_common.py/
generate_master_spreadsheet.py/mvpa_generalization_workflow.py -- only nibabel/numpy/
pandas/sklearn -- so this is a genuinely separate feature-extraction and
classification code path, not a re-skin of our own pipeline. It does reuse
the same preprocessed BOLD data and mask mvpa_generalization_workflow.py used (so any
difference in the numbers reflects classifier/CV methodology, not
different inputs), loaded directly via nibabel + numpy boolean indexing
rather than nilearn's NiftiMasker.

Feature extraction is intentionally simpler than mvpa_common.py's
compute_volume_range: one volume per trial (nearest TR to onset +
hemodynamic_lag), not a multi-volume window.

Usage:
    python tutorial/pymvpa_style_validation.py
    python tutorial/pymvpa_style_validation.py --our-results-dir out/haxby_object_classifier/1/model
"""

import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC

FUNC_DIR = "tutorial/haxby-data/derivatives/sub-1/func"
EVENTS_DIR = "tutorial/haxby-data/sub-1/func"
MASK_FILE = "tutorial/haxby-data/derivatives/sub-1/masks/native_epi_mask.nii.gz"
HEMODYNAMIC_LAG = 4.0  # matches tutorial/config-haxby.example.json
TRAIN_RUNS = set(range(1, 10))   # runs 1-9
TEST_RUNS = set(range(10, 13))   # runs 10-12


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--our-results-dir", default=None,
        help="Path to mvpa_generalization_workflow.py's model/ output dir (<analysis-output-dir>/<desc>/<subject>/model) "
             "to print alongside for comparison. Omit to skip that comparison row."
    )
    return parser.parse_args()


def load_run(bold_path: str, events_path: str, mask: np.ndarray):
    """Independent feature extraction: z-score each voxel's timeseries, then
    take the single volume nearest (onset + hemodynamic_lag) / TR per trial.
    Returns (X, y) for this run -- X shape (n_trials, n_voxels)."""
    img = nib.load(bold_path)
    tr = float(img.header.get_zooms()[3])
    data = img.get_fdata()  # (x, y, z, t)
    n_frames = data.shape[-1]

    voxel_ts = data[mask].T  # (n_frames, n_voxels)
    mean = voxel_ts.mean(axis=0, keepdims=True)
    std = voxel_ts.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (voxel_ts - mean) / std

    events = pd.read_csv(events_path, sep="\t")
    X, y = [], []
    for _, row in events.iterrows():
        target_vol = int(round((row["onset"] + HEMODYNAMIC_LAG) / tr))
        target_vol = min(max(target_vol, 0), n_frames - 1)
        X.append(z[target_vol])
        y.append(row["trial_type"])

    return np.array(X), np.array(y)


def load_all_runs():
    """Returns X (n_trials, n_voxels), y (category), chunks (run number)."""
    mask = nib.load(MASK_FILE).get_fdata().astype(bool)

    bold_files = sorted(glob.glob(os.path.join(FUNC_DIR, "*_desc-preproc_bold.nii.gz")))
    if not bold_files:
        raise SystemExit(f"No preprocessed BOLD files found under {FUNC_DIR} -- run tutorial/preprocess_haxby.sh first.")

    X_all, y_all, chunks_all = [], [], []
    for bold_path in bold_files:
        base = os.path.basename(bold_path)
        run = int(base.split("run-")[1].split("_")[0])
        events_path = os.path.join(EVENTS_DIR, base.split("_desc-preproc")[0] + "_events.tsv")

        X, y = load_run(bold_path, events_path, mask)
        X_all.append(X)
        y_all.append(y)
        chunks_all.append(np.full(len(y), run))
        print(f"  loaded run-{run:02d}: {len(y)} trials, {X.shape[1]} voxels")

    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(chunks_all)


def nearest_centroid_correlation(X_train, y_train, X_test, categories):
    """PyMVPA's own 'replicates the original Haxby et al. (2001) study'
    classifier: kNN with correlation distance, k=1 against per-category
    training averages -- a nearest-centroid template-matching classifier,
    the same logic the 2001 paper itself used. The correlation values
    themselves (not calibrated probabilities) are used as per-class
    evidence for AUC -- roc_auc_score's OvR mode only needs relative
    ordering, no calibration required."""
    centroids = np.array([X_train[y_train == cat].mean(axis=0) for cat in categories])

    # Pearson correlation of every test pattern against every centroid
    Xc = X_test - X_test.mean(axis=1, keepdims=True)
    Cc = centroids - centroids.mean(axis=1, keepdims=True)
    num = Xc @ Cc.T
    denom = np.linalg.norm(Xc, axis=1, keepdims=True) * np.linalg.norm(Cc, axis=1, keepdims=True).T
    denom[denom == 0] = 1.0
    corr = num / denom  # (n_test, n_categories)

    predictions = categories[np.argmax(corr, axis=1)]
    return predictions, corr


def linear_svm(X_train, y_train, X_test, categories):
    """PyMVPA's alternative canonical classifier (LinearCSVMC analog).
    decision_function -> sigmoid evidence, the same AUC methodology
    model_performance() in mvpa_generalization_workflow.py already uses, so only the
    classifier differs between the two pipelines, not the scoring formula."""
    clf = SVC(kernel="linear", class_weight="balanced")
    clf.fit(X_train, y_train)
    d = clf.decision_function(X_test)  # (n_test, n_categories) for multiclass OVR
    evidence = 1.0 / (1.0 + np.exp(-d))
    predictions = clf.classes_[np.argmax(d, axis=1)]
    return predictions, evidence, clf.classes_


def score(y_true, predictions, evidence, categories):
    acc = accuracy_score(y_true, predictions)
    Y = (y_true[:, None] == categories[None, :]).astype(np.uint8)
    auc = roc_auc_score(Y, evidence, multi_class="ovr", average="macro")
    return acc, auc


def evaluate_matched_split(X, y, chunks, categories):
    train_mask = np.isin(chunks, list(TRAIN_RUNS))
    test_mask = np.isin(chunks, list(TEST_RUNS))
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    results = {}

    predictions, corr = nearest_centroid_correlation(X_train, y_train, X_test, categories)
    results["nearest_centroid_correlation"] = score(y_test, predictions, corr, categories)

    predictions, evidence, clf_classes = linear_svm(X_train, y_train, X_test, categories)
    results["linear_svm"] = score(y_test, predictions, evidence, clf_classes)

    return results


def evaluate_leave_one_run_out(X, y, chunks, categories):
    """PyMVPA's own default CV scheme (NFoldPartitioner: leave-one-chunk-out),
    mean accuracy/AUC across all 12 folds."""
    runs = sorted(np.unique(chunks))
    fold_results = {"nearest_centroid_correlation": [], "linear_svm": []}

    for held_out_run in runs:
        train_mask = chunks != held_out_run
        test_mask = chunks == held_out_run
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        predictions, corr = nearest_centroid_correlation(X_train, y_train, X_test, categories)
        fold_results["nearest_centroid_correlation"].append(score(y_test, predictions, corr, categories))

        predictions, evidence, clf_classes = linear_svm(X_train, y_train, X_test, categories)
        fold_results["linear_svm"].append(score(y_test, predictions, evidence, clf_classes))

    return {
        method: (
            float(np.mean([r[0] for r in results])),
            float(np.mean([r[1] for r in results])),
        )
        for method, results in fold_results.items()
    }


def load_our_results(results_dir: str):
    total_path = glob.glob(os.path.join(results_dir, "*_model_results_total_scores.csv"))
    auc_path = glob.glob(os.path.join(results_dir, "*_model_results_auc.csv"))
    if not total_path or not auc_path:
        print(f"(!) --our-results-dir {results_dir!r} doesn't contain the expected "
              f"*_model_results_total_scores.csv/_auc.csv files -- skipping that comparison row")
        return None
    accuracy = float(np.loadtxt(total_path[0]))
    auc_df = pd.read_csv(auc_path[0], index_col=0)
    auc = float(auc_df.iloc[:, 0].mean())
    return accuracy, auc


def main():
    args = parse_args()

    print("Loading data (independent feature extraction, single volume per trial)...")
    X, y, chunks = load_all_runs()
    categories = np.array(sorted(np.unique(y)))
    print(f"Total: {X.shape[0]} trials, {X.shape[1]} voxels, {len(categories)} categories: {list(categories)}\n")

    print("=== Matched split (train runs 1-9, test runs 10-12 -- same split as tutorial/config-haxby.example.json) ===")
    matched = evaluate_matched_split(X, y, chunks, categories)
    for method, (acc, auc) in matched.items():
        print(f"  {method}: accuracy={acc:.4f}, AUC={auc:.4f}")

    print("\n=== Leave-one-run-out (PyMVPA's own default NFoldPartitioner scheme, 12 folds) ===")
    loro = evaluate_leave_one_run_out(X, y, chunks, categories)
    for method, (acc, auc) in loro.items():
        print(f"  {method}: mean accuracy={acc:.4f}, mean AUC={auc:.4f}")

    print("\n=== Comparison ===")
    print(f"{'source':45s} {'protocol':20s} accuracy   AUC")
    if args.our_results_dir:
        our = load_our_results(args.our_results_dir)
        if our:
            print(f"{'mvpa_generalization_workflow.py (this repo)':45s} {'matched split':20s} {our[0]:.4f}     {our[1]:.4f}")
    for method, (acc, auc) in matched.items():
        print(f"{method:45s} {'matched split':20s} {acc:.4f}     {auc:.4f}")
    for method, (acc, auc) in loro.items():
        print(f"{method:45s} {'leave-one-run-out':20s} {acc:.4f}     {auc:.4f}")


if __name__ == "__main__":
    main()
