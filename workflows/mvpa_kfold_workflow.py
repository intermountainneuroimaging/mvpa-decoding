#!/usr/bin/env python3

"""
Same-task k-fold MVPA decoding: unlike mvpa_generalization_workflow.py's independent-train/
independent-test case, this script assumes model_conditions.training and
model_conditions.testing (possibly different conditions) are drawn from the
same task/run pool. Runs are split into folds (model.kfold_cv), and for each
fold: train on every run NOT in the held-out group, test + timecourse-decode
only on the held-out group's rows, then aggregate every fold's results into
one final answer -- same classifier/accuracy/decoding steps as
mvpa_generalization_workflow.py, reused from mvpa_common.py so the two scripts can never
drift apart on how a model is actually fit or scored.

Fold membership can be automatic (leave-one-run-out, or an n-way split over
runs) or fully user-specified -- see model.kfold_cv in README.md. Whichever
strategy is used, the resolved fold membership is always logged to
<subject>_kfold_folds.json for later inspection.

Outputs, under <analysis-output-dir>/<model.desc>/<subject>/:
    <subject>_trial_pivot.csv                        -- sanity check, pre-model_conditions
    model/<subject>_kfold_folds.json                  -- {fold_id: [held-out run ids]}
    model/<subject>_fold{N}_model_results_{metric}.csv    -- per-fold held-out test metrics
    model/<subject>_fold{N}_impa[_mni].nii.gz      -- per-fold importance map
    model/<subject>_fold{N}_permutation_test.csv          -- per-fold significance (optional)
    decoding/<subject>_fold{N}_decoding_results.csv       -- per-fold raw decoding (*)
    decoding/<subject>_fold{N}_summary_decoding_results.csv (*)
    model/<subject>_model_results_{metric}.csv        -- aggregated (averaged across folds)
    model/<subject>_impa[_mni].nii.gz          -- aggregated importance map. Filename tag
                                                    depends on model.mnispace (default
                                                    false): "_impa_mni" when the input
                                                    BOLD/mask are configured as already
                                                    being in MNI space, plain "_impa"
                                                    otherwise (space left unasserted)
    decoding/<subject>_decoding_results.csv               -- aggregated raw (pooled across folds) (*)
    decoding/<subject>_summary_decoding_results.csv       -- aggregated summary (*)

(*) only written when model_conditions.timecourse_decoding is configured --
omit that whole section to skip timecourse decoding entirely (no decoding/
output at all, no extra runtime for that step).

The aggregated files use the exact same names mvpa_generalization_workflow.py writes, so
generate_report.py (and any other downstream consumer) doesn't need to know
which workflow produced a given subject's results -- and its existing
`_fold{N}_*`-detection already renders fold-variability panels against the
per-fold files above with no changes needed on that end.

Usage:
    python mvpa_kfold_workflow.py --subject 4057 --config examples/config-kfold.example.json \\
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import (
    build_trial_pivot_table, quick_safe, label_rows,
    track_runtime, load_config, apply_regressor_codes,
    load_images_and_mask, build_timecourse_instructions,
    model_classification, model_performance, permutation_significance,
    timecourse_decoding, summarize_decoding, save_model_results, average_fold_results, impa_tag,
)

KFOLD_STRATEGIES = ("per_run", "group_kfold", "explicit_groups")


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
        help="Root output directory -- results are written under <this>/<model.desc>/<subject>/{model,decoding}/"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the mvpa config JSON (event_extraction + model_conditions + model sections, "
             "with model.kfold_cv set). See README.md's K-fold workflow section and "
             "examples/config-kfold.example.json."
    )

    parser.add_argument(
        "--master-spreadsheet",
        required=True,
        help="Path to master_spreadsheet.csv produced by generate_master_spreadsheet.py"
    )

    return parser.parse_args()


# =====================================================
# model.kfold_cv: validation + fold resolution
# =====================================================

def validate_kfold_cv_config(kfold_cv_cfg) -> None:
    """Cheap, config-only validation of model.kfold_cv -- run at config-load time
    so a typo fails fast, before any (possibly slow) BOLD loading happens. The
    data-dependent checks (do the referenced runs actually exist for this subject)
    still happen later, in resolve_kfold_folds, once the subject's data is available."""
    if kfold_cv_cfg is None:
        raise SystemExit(
            "model.kfold_cv is required for mvpa_kfold_workflow.py -- see README.md's K-fold "
            "workflow section, or use mvpa_generalization_workflow.py for the independent-train/independent-test case."
        )
    strategy = kfold_cv_cfg.get("strategy")
    if strategy not in KFOLD_STRATEGIES:
        raise SystemExit(f"model.kfold_cv.strategy must be one of {KFOLD_STRATEGIES}, got {strategy!r}")
    if strategy == "group_kfold" and not isinstance(kfold_cv_cfg.get("n_splits"), int):
        raise SystemExit("model.kfold_cv.strategy='group_kfold' requires an integer 'n_splits'")
    if strategy == "explicit_groups" and not isinstance(kfold_cv_cfg.get("held_out_runs"), list):
        raise SystemExit("model.kfold_cv.strategy='explicit_groups' requires a 'held_out_runs' list")


def resolve_kfold_folds(kfold_cv_cfg: dict, testing_df: pd.DataFrame, timecourse_instr) -> list:
    """Return a list of held-out run-id groups implementing kfold_cv_cfg's strategy.
    Folds are built only over runs that actually appear in this subject's
    testing/timecourse_decoding-eligible data -- not the whole master_spreadsheet --
    so every fold corresponds to data that will actually be evaluated.
    timecourse_instr is None when model_conditions.timecourse_decoding isn't
    configured -- folds are then built from testing_df's runs alone."""
    strategy = kfold_cv_cfg.get("strategy")
    tc_runs = set(timecourse_instr["run"]) if timecourse_instr is not None else set()
    universe_runs = sorted(set(testing_df["run"]) | tc_runs)

    if not universe_runs:
        raise SystemExit(
            "model.kfold_cv: no runs found in testing/timecourse_decoding-eligible data for this "
            "subject -- nothing to fold over."
        )

    if strategy == "per_run":
        return [[r] for r in universe_runs]

    if strategy == "group_kfold":
        n_splits = kfold_cv_cfg.get("n_splits")
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
            raise SystemExit(
                f"model.kfold_cv.strategy='group_kfold' requires an integer 'n_splits' >= 2, got {n_splits!r}"
            )
        if n_splits > len(universe_runs):
            raise SystemExit(
                f"model.kfold_cv.n_splits={n_splits} exceeds the number of distinct runs available "
                f"({len(universe_runs)}: {universe_runs})"
            )
        return [list(g) for g in np.array_split(np.array(universe_runs), n_splits)]

    if strategy == "explicit_groups":
        held_out_run_groups = kfold_cv_cfg.get("held_out_runs")
        if not isinstance(held_out_run_groups, list) or not held_out_run_groups or not all(
            isinstance(g, list) and g for g in held_out_run_groups
        ):
            raise SystemExit(
                "model.kfold_cv.strategy='explicit_groups' requires a non-empty 'held_out_runs' list of "
                "non-empty run-id lists -- one inner list per fold, listing the run(s) held out for that fold"
            )
        covered = {r for g in held_out_run_groups for r in g}
        uncovered = [r for r in universe_runs if r not in covered]
        if uncovered:
            print(f"(!) model.kfold_cv.held_out_runs doesn't cover run(s) {uncovered} that appear in this "
                  f"subject's testing/timecourse_decoding data -- those rows will never be evaluated in any fold")
        unknown = sorted({r for g in held_out_run_groups for r in g if r not in universe_runs})
        if unknown:
            print(f"(!) model.kfold_cv.held_out_runs references run(s) {unknown} that don't appear in this "
                  f"subject's testing/timecourse_decoding-eligible data -- they'll produce empty folds")
        return held_out_run_groups

    raise SystemExit(f"model.kfold_cv.strategy must be one of {KFOLD_STRATEGIES}, got {strategy!r}")


# =====================================================
# Per-fold execution + aggregation
# =====================================================

def run_kfold(kfold_cv_cfg, permutation_test_cfg, masker, impa_filename_tag,
              analysis_output_dir, model_descr, subject_id, regressor_categories,
              feature_selection_cfg, classifier_name, classifier_params,
              training_df, training_data, training_labels,
              testing_df, testing_data, testing_labels,
              timecourse_instr, timecourse_data, timecourse_labels):
    """Repeatedly hold out a group of runs: train on the rest, test+decode only on the
    held-out group, then aggregate. Per-fold outputs are also saved -- for transparency,
    and so generate_report.py can detect and render fold-variability panels. Returns
    (aggregated_impa, aggregated_model_xout, aggregated_raw, aggregated_summary) --
    aggregated_raw/aggregated_summary are empty DataFrames when
    model_conditions.timecourse_decoding isn't configured (timecourse_instr is
    None), since there's then nothing to decode in any fold."""

    fold_groups = resolve_kfold_folds(kfold_cv_cfg, testing_df, timecourse_instr)
    print(f"model.kfold_cv: {len(fold_groups)} fold(s), strategy={kfold_cv_cfg.get('strategy')!r}")

    folds_manifest = {}
    model_results, model_impas, decoding_raws = [], [], []

    for fold_id, held_out_runs in enumerate(fold_groups, start=1):
        folds_manifest[fold_id] = [int(r) for r in held_out_runs]

        train_mask = (~training_df["run"].isin(held_out_runs)).to_numpy()
        test_mask = testing_df["run"].isin(held_out_runs).to_numpy()
        tc_mask = timecourse_instr["run"].isin(held_out_runs).to_numpy() if timecourse_instr is not None else None
        has_tc_rows = tc_mask is not None and tc_mask.any()

        if not test_mask.any() and not has_tc_rows:
            print(f"  (!) fold {fold_id} (held-out runs {held_out_runs}): no testing or "
                  f"timecourse_decoding rows -- skipping")
            continue
        if not train_mask.any():
            print(f"  (!) fold {fold_id} (held-out runs {held_out_runs}): no training rows remain "
                  f"once these runs are excluded -- skipping")
            continue

        tc_summary = f" / {int(tc_mask.sum())} timecourse" if tc_mask is not None else ""
        print(f"  Fold {fold_id}: held-out runs {held_out_runs} "
              f"({int(train_mask.sum())} train / {int(test_mask.sum())} test{tc_summary} rows)")

        fold_train_data = training_data[train_mask]
        fold_train_labels = training_labels[train_mask]

        xclf = model_classification(fold_train_data, fold_train_labels, feature_selection_cfg, classifier_name, classifier_params)

        if test_mask.any():
            fold_test_data = testing_data[test_mask]
            fold_test_labels = testing_labels[test_mask]

            xout, impa = model_performance(xclf, fold_test_data, fold_test_labels)
            output_pattern = os.path.join(
                analysis_output_dir, model_descr, subject_id, "model",
                f"{subject_id}_fold{fold_id}" + "_model_results_{metric}.csv"
            )
            save_model_results(output_pattern, xout, regressor_categories)
            model_results.append(xout)
            model_impas.append(impa)

            fold_impa_file = os.path.join(
                analysis_output_dir, model_descr, subject_id, "model",
                f"{subject_id}_fold{fold_id}_{impa_filename_tag}.nii.gz"
            )
            masker.inverse_transform(impa).to_filename(fold_impa_file)

            if permutation_test_cfg is not None:
                n_permutations = permutation_test_cfg.get("n_permutations", 1000)
                random_state = permutation_test_cfg.get("random_state", 0)
                print(f"    permutation testing ({n_permutations} permutations)...")
                fold_permutation_results = permutation_significance(
                    fold_train_data, fold_train_labels, fold_test_data, fold_test_labels,
                    n_permutations, random_state, feature_selection_cfg, classifier_name, classifier_params,
                )
                fold_permutation_file = os.path.join(
                    analysis_output_dir, model_descr, subject_id, "model",
                    f"{subject_id}_fold{fold_id}_permutation_test.csv"
                )
                fold_permutation_results.to_csv(fold_permutation_file, index=False)
        else:
            print(f"  (!) fold {fold_id}: no held-out testing rows -- skipping model_performance for this fold")

        if has_tc_rows:
            fold_raw, fold_summary = timecourse_decoding(
                xclf, timecourse_data[tc_mask], timecourse_labels[tc_mask],
                timecourse_instr.loc[tc_mask], regressor_categories,
                feature_selection_cfg, subject_id, model_descr,
            )
            fold_raw.insert(2, "fold", fold_id)
            fold_summary.insert(2, "fold", fold_id)

            fold_decoding_file = os.path.join(
                analysis_output_dir, model_descr, subject_id, "decoding",
                f"{subject_id}_fold{fold_id}_decoding_results.csv"
            )
            fold_summary_file = os.path.join(
                analysis_output_dir, model_descr, subject_id, "decoding",
                f"{subject_id}_fold{fold_id}_summary_decoding_results.csv"
            )
            Path(os.path.dirname(fold_decoding_file)).mkdir(parents=True, exist_ok=True)
            fold_raw.to_csv(fold_decoding_file, index=False)
            fold_summary.to_csv(fold_summary_file, index=False)
            decoding_raws.append(fold_raw)
        elif tc_mask is not None:
            print(f"  (!) fold {fold_id}: no held-out timecourse_decoding rows -- skipping decoding for this fold")

    if not model_results:
        raise SystemExit(
            "model.kfold_cv: every fold was skipped -- no held-out testing rows were ever available. "
            "Check your fold strategy against the runs actually present in testing_conditions."
        )

    manifest_file = os.path.join(
        analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_kfold_folds.json"
    )
    Path(os.path.dirname(manifest_file)).mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(folds_manifest, f, indent=2)
    print(f"Fold manifest saved to: {manifest_file}")

    aggregated_model_xout = average_fold_results(model_results)
    aggregated_impa = np.mean(np.stack(model_impas, axis=0), axis=0)

    # Raw rows from different folds are genuinely disjoint trials (folds partition
    # runs), so the aggregate raw table is just a concatenation -- no averaging needed.
    # The aggregate summary is then computed fresh from that pooled raw table, which
    # weights every trial equally regardless of which fold it came from (rather than
    # averaging each fold's own summary, which would implicitly weight folds equally
    # even if they held out different numbers of trials).
    aggregated_raw = pd.concat(decoding_raws, ignore_index=True) if decoding_raws else pd.DataFrame()
    aggregated_summary = (
        summarize_decoding(aggregated_raw, regressor_categories, subject_id, model_descr)
        if not aggregated_raw.empty else pd.DataFrame()
    )

    return aggregated_impa, aggregated_model_xout, aggregated_raw, aggregated_summary


# =====================================================
# Main Workflow
# =====================================================

def main(args):
    subject_id = args.subject
    analysis_output_dir = args.analysis_output_dir
    master_spreadsheet_file = args.master_spreadsheet

    full_cfg = load_config(args.config)
    kfold_cv_cfg = full_cfg["model"].get("kfold_cv")
    validate_kfold_cv_config(kfold_cv_cfg)

    model_conditions = full_cfg["model_conditions"]

    training_conditions = model_conditions["training"]["conditions"]
    testing_conditions = model_conditions["testing"]["conditions"]
    # optional -- model_conditions.timecourse_decoding entirely absent means
    # skip timecourse decoding: no decoding/ output files (per fold or
    # aggregated), no report timecourse page (generate_report.py already
    # skips that page when it finds no decoding_results.csv, so no
    # report-side change is needed for this).
    timecourse_cfg = model_conditions.get("timecourse_decoding")
    timecourse_conditions = timecourse_cfg["conditions"] if timecourse_cfg else None
    timecourse_window = timecourse_cfg["window"] if timecourse_cfg else None

    regressor_categories = list(training_conditions.keys())

    model_cfg = full_cfg["model"]
    model_descr = quick_safe(model_cfg["desc"])
    mask_pattern_template = model_cfg.get("mask", {}).get("mask_pattern")
    impa_filename_tag = impa_tag(model_cfg.get("mnispace", False))
    feature_selection_cfg = model_cfg["featureSelection"]
    classifier_name = model_cfg["classifier"]["name"]
    classifier_params = model_cfg["classifier"]["params"]
    permutation_test_cfg = model_cfg.get("permutation_test")

    print(f"Subject: {subject_id}")

    # ------------------------------------------------
    # Prepare Instructions
    # ------------------------------------------------

    master = pd.read_csv(
        master_spreadsheet_file,
        dtype={"subject": str, "session": str, "task": str, "trial_type": str}
    )

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
    if timecourse_cfg is not None:
        timecourse_labeled = apply_regressor_codes(label_rows(subject_df, timecourse_conditions), regressor_categories)
        timecourse_instr = build_timecourse_instructions(timecourse_labeled, timecourse_window)
    else:
        timecourse_instr = None

    training_data, training_labels, training_ids, masker = load_images_and_mask(training_df, mask_pattern_template)
    testing_data, testing_labels, testing_ids, masker = load_images_and_mask(testing_df, mask_pattern_template)

    training_df = training_df.loc[training_ids, :]
    testing_df = testing_df.loc[testing_ids, :]
    print("...Done")

    training_labels = training_labels.ravel()
    testing_labels = testing_labels.ravel()

    if timecourse_instr is not None:
        timecourse_data, timecourse_labels, timecourse_ids, masker = load_images_and_mask(timecourse_instr, mask_pattern_template)
        timecourse_instr = timecourse_instr.loc[timecourse_ids, :]
        timecourse_labels = timecourse_labels.ravel()
    else:
        timecourse_data = timecourse_labels = None

    # -------------------------------------------------
    # K-Fold Train / Test / Decode
    # -------------------------------------------------

    print("Training + testing + decoding via model.kfold_cv...")
    aggregated_impa, xout, raw_decoding, summary_decoding = run_kfold(
        kfold_cv_cfg, permutation_test_cfg, masker, impa_filename_tag,
        analysis_output_dir, model_descr, subject_id, regressor_categories,
        feature_selection_cfg, classifier_name, classifier_params,
        training_df, training_data, training_labels,
        testing_df, testing_data, testing_labels,
        timecourse_instr, timecourse_data, timecourse_labels,
    )

    # -------------------------------------------------
    # Aggregated outputs -- same filenames mvpa_generalization_workflow.py uses, so
    # generate_report.py and other downstream consumers don't need to know
    # which workflow produced a given subject's results.
    # -------------------------------------------------

    output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_model_results_{metric}.csv")
    save_model_results(output_pattern, xout, regressor_categories)

    img = masker.inverse_transform(aggregated_impa)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_{impa_filename_tag}.nii.gz")
    img.to_filename(output_file)

    if not raw_decoding.empty:
        output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_decoding_results.csv")
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        raw_decoding.to_csv(output_file, index=False)

        summary_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_summary_decoding_results.csv")
        summary_decoding.to_csv(summary_file, index=False)

        print(f"Aggregated results saved to: {output_file} (raw) and {summary_file} (summary)")
    else:
        print("No model_conditions.timecourse_decoding in config -- skipped, no decoding/ output written.")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    with track_runtime():
        main(args)
