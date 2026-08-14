#!/usr/bin/env python3

"""
Within-subject MVPA decoding: trains a classifier on model_conditions.training,
cross-validates internally on that same training data (GroupKFold on `run`),
evaluates on model_conditions.testing, and predicts at every TR across a
timecourse_decoding window built independently of however volume_of_interest
was computed in master_spreadsheet.csv. Reads the merged config (event_extraction
+ model_conditions + model sections) plus master_spreadsheet.csv produced by
generate_master_spreadsheet.py -- see README.md sections 3-5 for the config
format and "Running mvpa_generalization_workflow.py" for what each step does.

Works for any number of training/testing conditions (2 or more): class lists
are always derived from what the classifier actually learned (clf.classes_),
not from what happens to appear in a given fold's held-out data.

This script is dedicated to the independent-train/independent-test case --
training and testing come from separate model_conditions sections and may
be entirely different tasks/data. For same-task data split into folds by
`run`, see mvpa_kfold_workflow.py instead -- a separate script (not a mode
switch on this one) that shares its classification/decoding primitives with
this one via mvpa_common.py.

Outputs, under <analysis-output-dir>/<model.desc>/<subject>/:
    <subject>_trial_pivot.csv           -- sanity check, pre-model_conditions
    cv/<subject>_cv_results_{metric}.csv        -- internal training-CV metrics
    cv/<subject>_cv_impa_native.nii.gz          -- CV-fold-averaged importance map
    model/<subject>_model_results_{metric}.csv  -- held-out test metrics
    model/<subject>_impa_native.nii.gz          -- final importance map
    decoding/<subject>_decoding_results.csv         -- raw, one row per decoded TR
    decoding/<subject>_summary_decoding_results.csv -- averaged per (window_index, regressor_label)

Usage:
    python mvpa_generalization_workflow.py --subject 4057 \\
        --config examples/config-generalization.example.json \\
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import PredefinedSplit

from mvpa_common import (
    build_trial_pivot_table, resolve_config_root, quick_safe, label_rows,
    track_runtime, load_config, apply_regressor_codes, balance,
    load_images_and_mask, build_timecourse_instructions,
    model_classification, model_performance, permutation_significance,
    timecourse_decoding, save_model_results, average_fold_results,
)


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
