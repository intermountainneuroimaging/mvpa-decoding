#!/usr/bin/env python3

"""
Unified MVPA decoding workflow: fits a classifier on model_conditions.training,
then runs up to three independent, config-driven steps -- each only runs (and
only writes its own output) when its config section is present:

  1. model.kfold_cv        -- k-fold cross-validates entirely within
                               model_conditions.training (train on non-held-out
                               runs, evaluate on held-out runs of that same
                               training-labeled data).
  2. model_conditions.testing        -- one fit on the complete training set,
                               evaluated against a genuinely separate test set.
  3. model_conditions.timecourse_decoding -- predicts at every TR of every run in
                               the full-frame spreadsheet (event_extraction.
                               full_frame_output_file), continuously and without
                               exclusions, using that same complete-training-set
                               fit (never per-fold) -- unless step 2's guard below
                               substitutes a held-out k-fold classifier instead.
                               Rows are grouped into trials anchored on
                               trial_start_event, with window_index counting up
                               from 0 at each anchor.

The complete-training-set classifier (step 2/3's "full model") is always fit,
regardless of which of the above are configured, since timecourse decoding
needs it whether or not a test set exists. See README.md sections 3-6 for the
config format and "Running mvpa_workflow.py" for what each step does.

Double-dipping guard: a subject whose model_conditions.training shares a
bold file with testing/timecourse_decoding has its held-out test evaluation
skipped entirely (no honest substitute classifier exists) and its
overlapping timecourse rows decoded with their own held-out model.kfold_cv
classifier instead of the full-training fit (or skipped too, if kfold_cv
isn't configured) -- see boldfile_overlap() and README.md section 6's
"Double-dipping guard" for the full explanation, and
model.allow_train_test_overlap to opt out.

Works for any number of training conditions (2 or more): class lists are
always derived from what the classifier actually learned (clf.classes_), not
from what happens to appear in a given fold's/test set's data.

Outputs, under <analysis-output-dir>/<model.desc>/<subject>/:
    <subject>_trial_pivot.csv                     -- sanity check, pre-model_conditions

  model.kfold_cv configured -- k-fold CV entirely within model_conditions.training:
    model/<subject>_kfold_folds.json                    -- {fold_id: {"training": [acquisition
                                                            names], "testing": [acquisition names]}}
    model/<subject>_fold{N}_model_results_{metric}.csv  -- per-fold held-out metrics
    model/<subject>_fold{N}_impa[_mni].nii.gz           -- per-fold importance map
    model/<subject>_fold{N}_permutation_test.csv        -- per-fold significance (optional)
    model/<subject>_model_results_{metric}.csv          -- aggregated across folds
    model/<subject>_impa[_mni].nii.gz                   -- aggregated importance map
    model/<subject>_cv_results.csv                      -- raw, one row per held-out
                                                            sample across all folds (task,
                                                            trial_type, run, predicted_label,
                                                            correct, evidence_<category>, ...)

  model_conditions.testing configured -- one fit on all of training, evaluated
  against all of testing:
    test/<subject>_model_results_{metric}.csv           -- held-out test metrics
    test/<subject>_impa[_mni].nii.gz                    -- that fit's importance map
    test/<subject>_permutation_test.csv                 -- significance (optional)

  model_conditions.timecourse_decoding configured -- always the complete-training
  classifier, never per-fold:
    decoding/<subject>_decoding_results.csv             -- raw, one row per BOLD volume in the
                                                            full-frame spreadsheet (every frame,
                                                            no exclusions), real trial_type per row
    decoding/<subject>_summary_decoding_results.csv     -- averaged per (window_index, regressor_label)

`model/` is therefore exclusively k-fold's directory, `test/` is exclusively
the independent-test-set evaluation's directory -- a subject can have either,
both, or neither depending on what's configured. Importance map filenames use
"_impa_mni" instead of plain "_impa" when model.mnispace is set -- see
utils.mvpa_common.impa_tag.

Usage:
    python mvpa_workflow.py --subject 4057 \\
        --config examples/config-generalization.example.json \\
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out \\
        [--full-frame-spreadsheet master_spreadsheet_full.csv]  # only needed for timecourse_decoding
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import (
    build_trial_pivot_table, quick_safe, label_rows,
    track_runtime, load_config, apply_regressor_codes,
    load_images_and_mask, build_timecourse_instructions,
    model_classification, model_performance, permutation_significance,
    timecourse_decoding, save_model_results, average_fold_results, impa_tag,
    build_cv_raw_results, parse_bids_entities,
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
        help="Root output directory -- results are written under <this>/<model.desc>/<subject>/{model,test,decoding}/"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the mvpa config JSON (event_extraction + model_conditions + model sections). "
             "See README.md sections 3-6 and examples/config-*.example.json."
    )

    parser.add_argument(
        "--master-spreadsheet",
        required=True,
        help="Path to master_spreadsheet.csv produced by generate_master_spreadsheet.py"
    )

    parser.add_argument(
        "--full-frame-spreadsheet",
        default=None,
        help="Path to the full-frame spreadsheet produced by generate_master_spreadsheet.py's "
             "event_extraction.full_frame_output_file -- required only when "
             "model_conditions.timecourse_decoding is configured (one row per BOLD volume, "
             "real per-frame trial_type, no exclusions)."
    )

    return parser.parse_args()


# =====================================================
# model.kfold_cv: validation + fold resolution (training-only)
# =====================================================

def validate_kfold_cv_config(kfold_cv_cfg) -> None:
    """Cheap, config-only validation of model.kfold_cv -- run at config-load time
    so a typo fails fast, before any (possibly slow) BOLD loading happens. The
    data-dependent checks (do the referenced runs actually exist for this subject)
    still happen later, in resolve_kfold_folds, once the subject's data is
    available. Only called when model.kfold_cv is actually present -- k-fold
    itself is optional."""
    if kfold_cv_cfg is None:
        raise SystemExit("model.kfold_cv, once present, must be a config object -- got None")
    strategy = kfold_cv_cfg.get("strategy")
    if strategy not in KFOLD_STRATEGIES:
        raise SystemExit(f"model.kfold_cv.strategy must be one of {KFOLD_STRATEGIES}, got {strategy!r}")
    if strategy == "group_kfold" and not isinstance(kfold_cv_cfg.get("n_splits"), int):
        raise SystemExit("model.kfold_cv.strategy='group_kfold' requires an integer 'n_splits'")
    if strategy == "explicit_groups" and not isinstance(kfold_cv_cfg.get("held_out_runs"), list):
        raise SystemExit("model.kfold_cv.strategy='explicit_groups' requires a 'held_out_runs' list")


def resolve_kfold_folds(kfold_cv_cfg: dict, training_df: pd.DataFrame) -> list:
    """Return a list of held-out run-id groups implementing kfold_cv_cfg's strategy.
    Folds are built only over runs that actually appear in this subject's
    model_conditions.training data -- not the whole master_spreadsheet -- so every
    fold corresponds to data that will actually be evaluated."""
    strategy = kfold_cv_cfg.get("strategy")
    universe_runs = sorted(set(training_df["run"]))

    if not universe_runs:
        raise SystemExit(
            "model.kfold_cv: no runs found in model_conditions.training data for this "
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
                  f"subject's training data -- those rows will never be held out in any fold")
        unknown = sorted({r for g in held_out_run_groups for r in g if r not in universe_runs})
        if unknown:
            print(f"(!) model.kfold_cv.held_out_runs references run(s) {unknown} that don't appear in this "
                  f"subject's training data -- they'll produce empty folds")
        return held_out_run_groups

    raise SystemExit(f"model.kfold_cv.strategy must be one of {KFOLD_STRATEGIES}, got {strategy!r}")


# =====================================================
# Double-dipping guard: training vs. testing/timecourse independence
# =====================================================

def boldfile_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> set:
    """Boldfiles shared between two already-labeled dataframes -- a non-empty
    result means at least one scan contributes rows to both, e.g. training a
    classifier on part of a run and then evaluating it (test or timecourse
    decoding) on another part of that same run. Keyed by boldfile rather than
    run number, since two different tasks can reuse the same run number for
    genuinely different scans -- boldfile is the only unambiguous "same scan"
    identifier."""
    return set(df_a["boldfile"]) & set(df_b["boldfile"])


# =====================================================
# K-fold: per-fold execution + aggregation (training-only)
# =====================================================

def _acquisition_name(boldfile: str) -> str:
    """A short, preprocessing/space-independent acquisition label parsed from
    boldfile's own BIDS entities -- e.g. "sub-1_ses-A1_task-loc_run-01_bold"
    -- deliberately excluding dir/space/desc (constant across every
    acquisition in a given analysis, so they'd only add noise here). Used to
    label kfold_folds.json's per-fold training/testing acquisition lists,
    since a run number alone is ambiguous across tasks that reuse it (see
    boldfile_overlap)."""
    entities = parse_bids_entities(boldfile)
    parts = [f"{key}-{entities[key]}" for key in ("sub", "ses", "task", "run") if key in entities]
    return "_".join(parts) + "_bold"


def run_kfold(kfold_cv_cfg, fold_groups, permutation_test_cfg, masker, impa_filename_tag,
              analysis_output_dir, model_descr, subject_id, regressor_categories,
              feature_selection_cfg, classifier_name, classifier_params,
              training_df, training_data, training_labels):
    """Repeatedly hold out a group of runs from model_conditions.training: train on
    the rest, evaluate on the held-out group, then aggregate. Per-fold outputs are
    also saved -- for transparency, and so generate_report.py can detect and
    render fold-variability panels. Returns (aggregated_impa, aggregated_model_xout,
    boldfile_to_pipe) -- the third value maps every held-out boldfile (of every
    fold that actually ran, i.e. wasn't skipped) to that fold's own fitted pipe,
    which never trained on that boldfile's data. main() uses this to decode a
    model_conditions.timecourse_decoding row with a genuinely held-out classifier
    when its boldfile also appears in model_conditions.training, instead of the
    full-training classifier (which would have trained on it) -- see
    README.md's double-dipping warning. fold_groups is resolved by the caller
    (via resolve_kfold_folds) rather than here, so the trial-pivot-table sanity
    check can annotate rows with the exact same fold membership this function
    actually runs on."""

    print(f"model.kfold_cv: {len(fold_groups)} fold(s), strategy={kfold_cv_cfg.get('strategy')!r}")

    folds_manifest = {}
    model_results, model_impas, cv_raw_frames = [], [], []
    boldfile_to_pipe = {}

    for fold_id, held_out_runs in enumerate(fold_groups, start=1):
        train_mask = (~training_df["run"].isin(held_out_runs)).to_numpy()
        test_mask = training_df["run"].isin(held_out_runs).to_numpy()

        # full acquisition names (not just run numbers) for both sides of
        # this fold's split -- a run number alone doesn't say which task/
        # session it belongs to, so this is what actually lets someone
        # confirm exactly what went into training vs. was held out
        folds_manifest[fold_id] = {
            "training": sorted({_acquisition_name(bf) for bf in training_df.loc[train_mask, "boldfile"]}),
            "testing": sorted({_acquisition_name(bf) for bf in training_df.loc[test_mask, "boldfile"]}),
        }

        if not test_mask.any():
            print(f"  (!) fold {fold_id} (held-out runs {held_out_runs}): no held-out rows -- skipping")
            continue
        if not train_mask.any():
            print(f"  (!) fold {fold_id} (held-out runs {held_out_runs}): no training rows remain "
                  f"once these runs are excluded -- skipping")
            continue

        print(f"  Fold {fold_id}: held-out runs {held_out_runs} "
              f"({int(train_mask.sum())} train / {int(test_mask.sum())} held-out rows)")

        fold_train_data = training_data[train_mask]
        fold_train_labels = training_labels[train_mask]
        fold_test_data = training_data[test_mask]
        fold_test_labels = training_labels[test_mask]

        xclf = model_classification(fold_train_data, fold_train_labels, feature_selection_cfg, classifier_name, classifier_params)
        xout, impa = model_performance(xclf, fold_test_data, fold_test_labels)

        # per-held-out-sample detail (task/trial_type/run/boldfile plus
        # predicted_label/correct/evidence) -- documents exactly which run
        # was held out for this fold and how each of its samples was
        # classified; concatenated across folds and saved once below
        held_out_df = training_df.loc[test_mask]
        cv_raw_frames.append(build_cv_raw_results(
            xclf, fold_test_data, fold_test_labels, held_out_df, regressor_categories,
            feature_selection_cfg, model_descr, fold_id,
        ))

        # keyed by boldfile (not run number) -- two different tasks can reuse
        # the same run number for genuinely different scans, so boldfile is
        # the only unambiguous "same scan" identifier for the timecourse
        # double-dipping substitution this dict exists for
        for bf in training_df.loc[test_mask, "boldfile"].unique():
            boldfile_to_pipe[bf] = xclf

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

    if not model_results:
        raise SystemExit(
            "model.kfold_cv: every fold was skipped -- no held-out training rows were ever available. "
            "Check your fold strategy against the runs actually present in model_conditions.training."
        )

    manifest_file = os.path.join(
        analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_kfold_folds.json"
    )
    Path(os.path.dirname(manifest_file)).mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(folds_manifest, f, indent=2)
    print(f"Fold manifest saved to: {manifest_file}")

    cv_raw_file = os.path.join(
        analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_cv_results.csv"
    )
    pd.concat(cv_raw_frames, ignore_index=True).to_csv(cv_raw_file, index=False)
    print(f"Cross-validation hold-out sample results saved to: {cv_raw_file}")

    aggregated_model_xout = average_fold_results(model_results)
    aggregated_impa = np.mean(np.stack(model_impas, axis=0), axis=0)

    return aggregated_impa, aggregated_model_xout, boldfile_to_pipe


# =====================================================
# Main Workflow
# =====================================================

def main(args):
    subject_id = args.subject
    analysis_output_dir = args.analysis_output_dir
    master_spreadsheet_file = args.master_spreadsheet
    full_frame_spreadsheet_file = args.full_frame_spreadsheet

    full_cfg = load_config(args.config)
    model_conditions = full_cfg["model_conditions"]

    training_conditions = model_conditions["training"]["conditions"]
    # optional -- omit either section entirely to skip that step: no test/ or
    # decoding/ output, no report section, no extra runtime for that step.
    testing_cfg = model_conditions.get("testing")
    testing_conditions = testing_cfg["conditions"] if testing_cfg else None
    timecourse_cfg = model_conditions.get("timecourse_decoding")
    timecourse_conditions = timecourse_cfg["conditions"] if timecourse_cfg else None
    trial_start_event = timecourse_cfg["trial_start_event"] if timecourse_cfg else None
    if timecourse_cfg is not None and not full_frame_spreadsheet_file:
        raise SystemExit(
            "model_conditions.timecourse_decoding is configured but --full-frame-spreadsheet "
            "wasn't given -- continuous timecourse decoding reads its own full-frame spreadsheet "
            "(event_extraction.full_frame_output_file), not master_spreadsheet.csv."
        )

    # class label order shared across training/testing/timecourse regressor codes
    regressor_categories = list(training_conditions.keys())

    # model settings (mask/featureSelection/classifier/kfold_cv/desc)
    model_cfg = full_cfg["model"]
    model_descr = quick_safe(model_cfg["desc"])
    mask_pattern_template = model_cfg.get("mask", {}).get("mask_pattern")
    impa_filename_tag = impa_tag(model_cfg.get("mnispace", False))
    feature_selection_cfg = model_cfg["featureSelection"]
    classifier_name = model_cfg["classifier"]["name"]
    classifier_params = model_cfg["classifier"]["params"]
    permutation_test_cfg = model_cfg.get("permutation_test")

    # optional escape hatch for the double-dipping guard below -- see
    # README.md's warning. Default False: independence between training and
    # testing/timecourse is enforced automatically, not opt-in.
    allow_train_test_overlap = model_cfg.get("allow_train_test_overlap", False)

    # optional -- omitting model.kfold_cv entirely skips k-fold cross-validation
    # (no model/ output at all, no extra runtime for that step)
    kfold_cv_cfg = model_cfg.get("kfold_cv")
    if kfold_cv_cfg is not None:
        validate_kfold_cv_config(kfold_cv_cfg)

    if kfold_cv_cfg is None and testing_cfg is None and timecourse_cfg is None:
        print("(!) none of model.kfold_cv, model_conditions.testing, or "
              "model_conditions.timecourse_decoding are configured -- this run "
              "will only produce the trial pivot table sanity check, no model "
              "output at all.")

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
    # Load Data
    # -------------------------------------------------

    training_df = apply_regressor_codes(label_rows(subject_df, training_conditions), regressor_categories)
    testing_df = (
        apply_regressor_codes(label_rows(subject_df, testing_conditions), regressor_categories)
        if testing_conditions is not None else None
    )
    if timecourse_cfg is not None:
        full_frame = pd.read_csv(
            full_frame_spreadsheet_file,
            dtype={"subject": str, "session": str, "task": str, "trial_type": str}
        )
        full_frame_subject_df = full_frame[full_frame["subject"] == subject_id]
        if full_frame_subject_df.empty:
            raise SystemExit(f"No rows found for subject {subject_id!r} in {full_frame_spreadsheet_file}")
        timecourse_instr = build_timecourse_instructions(full_frame_subject_df, timecourse_conditions, trial_start_event)
        timecourse_instr = apply_regressor_codes(timecourse_instr, regressor_categories)
    else:
        timecourse_instr = None

    # -------------------------------------------------
    # Double-dipping guard -- a classifier evaluated on data from the same
    # scan(s) it was (even partly) trained on has inflated apparent
    # performance, since fMRI noise is autocorrelated within a run. Detected
    # per subject (boldfile overlap can vary subject to subject even under
    # the same config, e.g. missing runs) rather than assumed from the
    # config alone.
    #   - training/testing overlap: the held-out test evaluation is skipped
    #     entirely (no test/ output) -- there is no substitute classifier
    #     that's both "the one full-training fit" and "never trained on the
    #     overlapping run", so skipping is the only honest option.
    #   - training/timecourse overlap: if model.kfold_cv is configured, the
    #     overlapping run(s) are decoded with their own held-out k-fold
    #     classifier instead of the full-training one (see run_kfold's
    #     boldfile_to_pipe) -- non-overlapping rows still use the
    #     full-training classifier as before. If model.kfold_cv isn't
    #     configured, there's no held-out classifier to substitute, so
    #     timecourse decoding is skipped entirely.
    #   - model.allow_train_test_overlap: true disables both of the above,
    #     restoring the original (leakage-prone) behavior -- see README.md's
    #     warning before using this.
    # -------------------------------------------------
    run_test_evaluation = testing_df is not None
    use_kfold_for_timecourse = False  # set True below when kfold-substitution should be used
    skip_timecourse = False

    # structured record of what (if anything) the guard did -- written to
    # <subject>_double_dipping_report.json below and read back by
    # generate_report.py to render an in-report warning (see README.md
    # section 6/7's "Double-dipping guard"). "handling" is one of
    # "skipped" / "kfold_substitution" / "overwritten"; None means no
    # overlap was found (no report entry needed for that section).
    double_dipping_report = {}

    if testing_df is not None:
        overlap = boldfile_overlap(training_df, testing_df)
        if overlap:
            print(f"(!) DOUBLE-DIPPING WARNING: model_conditions.training and model_conditions.testing "
                  f"share {len(overlap)} bold file(s) for subject {subject_id}: {sorted(overlap)}. "
                  f"Evaluating a classifier on data from a run it was (even partly) trained on inflates "
                  f"apparent generalization performance.")
            if allow_train_test_overlap:
                print("  -> model.allow_train_test_overlap is set -- evaluating against model_conditions.testing anyway.")
                handling = "overwritten"
            else:
                print("  -> Skipping the held-out test evaluation for this subject (no test/ output). "
                      "Set model.allow_train_test_overlap: true to evaluate anyway (NOT recommended -- see README.md).")
                run_test_evaluation = False
                handling = "skipped"
            double_dipping_report["test"] = {
                "overlap_boldfiles": sorted(overlap), "handling": handling,
            }

    if timecourse_instr is not None:
        overlap = boldfile_overlap(training_df, timecourse_instr)
        if overlap:
            print(f"(!) DOUBLE-DIPPING WARNING: model_conditions.training and model_conditions.timecourse_decoding "
                  f"share {len(overlap)} bold file(s) for subject {subject_id}: {sorted(overlap)}. "
                  f"Decoding data from a run the classifier was (even partly) trained on inflates apparent decoding accuracy.")
            if allow_train_test_overlap:
                print("  -> model.allow_train_test_overlap is set -- decoding with the full-training classifier anyway.")
                handling = "overwritten"
            elif kfold_cv_cfg is not None:
                print("  -> Decoding the overlapping run(s) with their own held-out k-fold classifier instead of "
                      "the full-training model; non-overlapping rows still use the full-training model.")
                use_kfold_for_timecourse = True
                handling = "kfold_substitution"
            else:
                print("  -> model.kfold_cv is not configured, so there is no held-out classifier available for "
                      "these rows. Skipping timecourse decoding for this subject. Configure model.kfold_cv, or "
                      "set model.allow_train_test_overlap: true to decode anyway (NOT recommended -- see README.md).")
                skip_timecourse = True
                handling = "skipped"
            double_dipping_report["timecourse"] = {
                "overlap_boldfiles": sorted(overlap), "handling": handling,
            }

    # resolved once here (rather than inside run_kfold) so the exact same fold
    # membership is available for both the pivot table below and the real
    # per-fold execution -- no risk of the two drifting apart, no duplicate
    # warning prints from resolve_kfold_folds
    fold_groups = resolve_kfold_folds(kfold_cv_cfg, training_df) if kfold_cv_cfg is not None else None

    # -------------------------------------------------
    # Trial Pivot Table (sanity check, not used for modeling) -- every row
    # this subject has in master_spreadsheet.csv (not just the ones
    # model_conditions selects), tagged with which training/testing condition
    # (if any) each row matched, plus -- when model.kfold_cv is configured --
    # one column per fold showing "train"/"test"/blank for that fold
    # specifically (fold membership is entirely within training_condition
    # rows now, since k-fold no longer touches testing rows at all).
    # -------------------------------------------------

    pivot_source = subject_df.copy()
    pivot_source["training_condition"] = training_df["regressor_label"].reindex(pivot_source.index)
    if testing_df is not None:
        pivot_source["testing_condition"] = testing_df["regressor_label"].reindex(pivot_source.index)
    if fold_groups is not None:
        for fold_id, held_out_runs in enumerate(fold_groups, start=1):
            held_out_runs = set(held_out_runs)
            in_training = pivot_source.index.isin(training_df.index)
            is_train = in_training & (~pivot_source["run"].isin(held_out_runs))
            is_test = in_training & (pivot_source["run"].isin(held_out_runs))
            pivot_source[f"fold{fold_id}_split"] = np.select([is_train, is_test], ["train", "test"], default="")
    trial_pivot = build_trial_pivot_table(pivot_source)
    output_file = os.path.join(analysis_output_dir, model_descr, subject_id, f"{subject_id}_trial_pivot.csv")
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    trial_pivot.to_csv(output_file, index=False)
    print(f"Trial pivot table (sanity check) saved to: {output_file}")

    training_data, training_labels, training_ids, masker = load_images_and_mask(training_df, mask_pattern_template)
    training_df = training_df.loc[training_ids, :]
    training_labels = training_labels.ravel()

    if testing_df is not None and run_test_evaluation:
        testing_data, testing_labels, testing_ids, masker = load_images_and_mask(testing_df, mask_pattern_template)
        testing_df = testing_df.loc[testing_ids, :]
        testing_labels = testing_labels.ravel()
    else:
        # not loaded at all when the double-dipping guard already decided to
        # skip the test evaluation -- testing_df itself (if it exists) has
        # already done its job annotating the trial pivot table above
        testing_data = testing_labels = None

    if timecourse_instr is not None and not skip_timecourse:
        timecourse_data, timecourse_labels, timecourse_ids, masker = load_images_and_mask(timecourse_instr, mask_pattern_template)
        timecourse_instr = timecourse_instr.loc[timecourse_ids, :]
        timecourse_labels = timecourse_labels.ravel()
    else:
        timecourse_data = timecourse_labels = None
    print("...Done")

    # -------------------------------------------------
    # Full-training classifier -- always fit; used for timecourse decoding
    # below, and (if model_conditions.testing is configured) evaluated
    # against the held-out test set. Its importance map is only ever
    # extracted/saved as part of that test evaluation, so a
    # training+timecourse-only config (no test set, no kfold) legitimately
    # produces no importance map at all -- timecourse decoding still runs
    # fine off the in-memory classifier.
    # -------------------------------------------------

    print("Training classifier on full training set...")
    xclf_full = model_classification(training_data, training_labels, feature_selection_cfg, classifier_name, classifier_params)

    # -------------------------------------------------
    # K-Fold Cross-Validation (optional -- model.kfold_cv)
    # -------------------------------------------------

    kfold_boldfile_to_pipe = {}

    if fold_groups is not None:
        print("K-fold cross-validating within model_conditions.training...")
        aggregated_impa, kfold_xout, kfold_boldfile_to_pipe = run_kfold(
            kfold_cv_cfg, fold_groups, permutation_test_cfg, masker, impa_filename_tag,
            analysis_output_dir, model_descr, subject_id, regressor_categories,
            feature_selection_cfg, classifier_name, classifier_params,
            training_df, training_data, training_labels,
        )

        output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}" + "_model_results_{metric}.csv")
        save_model_results(output_pattern, kfold_xout, regressor_categories)

        img = masker.inverse_transform(aggregated_impa)
        output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "model", f"{subject_id}_{impa_filename_tag}.nii.gz")
        img.to_filename(output_file)

        if use_kfold_for_timecourse:
            tc_overlap = boldfile_overlap(training_df, timecourse_instr)
            uncovered = tc_overlap - set(kfold_boldfile_to_pipe.keys())
            if uncovered:
                print(f"  (!) {len(uncovered)} overlapping bold file(s) belong to a fold that was itself "
                      f"skipped (no held-out or no training rows) -- those timecourse rows will fall back "
                      f"to the full-training classifier: {sorted(uncovered)}")
            # how many *distinct* held-out fold classifiers actually decoded
            # an overlapping row -- e.g. a group_kfold fold spanning several
            # overlapping runs counts once, not once per run
            n_folds_used = len({id(kfold_boldfile_to_pipe[bf]) for bf in tc_overlap if bf in kfold_boldfile_to_pipe})
            double_dipping_report["timecourse"]["n_folds_used"] = n_folds_used

    # -------------------------------------------------
    # Test / Generalization (optional -- model_conditions.testing)
    # -------------------------------------------------

    if testing_df is not None and run_test_evaluation:
        print("Evaluating full-training classifier against model_conditions.testing...")
        xout, importance_map = model_performance(xclf_full, testing_data, testing_labels)

        output_pattern = os.path.join(analysis_output_dir, model_descr, subject_id, "test", f"{subject_id}" + "_model_results_{metric}.csv")
        save_model_results(output_pattern, xout, regressor_categories)

        img = masker.inverse_transform(importance_map)
        output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "test", f"{subject_id}_{impa_filename_tag}.nii.gz")
        img.to_filename(output_file)

        if permutation_test_cfg is not None:
            n_permutations = permutation_test_cfg.get("n_permutations", 1000)
            random_state = permutation_test_cfg.get("random_state", 0)
            print(f"Permutation testing ({n_permutations} permutations)...")
            permutation_results = permutation_significance(
                training_data, training_labels, testing_data, testing_labels, n_permutations, random_state,
                feature_selection_cfg, classifier_name, classifier_params,
            )
            permutation_file = os.path.join(
                analysis_output_dir, model_descr, subject_id, "test", f"{subject_id}_permutation_test.csv"
            )
            permutation_results.to_csv(permutation_file, index=False)
            print(f"Permutation test results saved to: {permutation_file}")

    # -------------------------------------------------
    # Time Course Decoding (optional -- model_conditions.timecourse_decoding)
    # -------------------------------------------------

    if timecourse_instr is not None and not skip_timecourse:
        print("Time Course Decoding...")
        raw_decoding, summary_decoding = timecourse_decoding(
            xclf_full, timecourse_data, timecourse_labels, timecourse_instr, regressor_categories,
            feature_selection_cfg, subject_id, model_descr,
            boldfile_to_pipe=kfold_boldfile_to_pipe if use_kfold_for_timecourse else None,
        )

        output_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_decoding_results.csv")
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        raw_decoding.to_csv(output_file, index=False)

        summary_file = os.path.join(analysis_output_dir, model_descr, subject_id, "decoding", f"{subject_id}" + "_summary_decoding_results.csv")
        summary_decoding.to_csv(summary_file, index=False)

        print(f"Results saved to: {output_file} (raw) and {summary_file} (summary)")

    # -------------------------------------------------
    # Double-dipping report (only written if the guard above actually found
    # something) -- read back by generate_report.py to render an in-report
    # warning instead of this only ever being visible in the job log.
    # -------------------------------------------------

    if double_dipping_report:
        report_file = os.path.join(
            analysis_output_dir, model_descr, subject_id, f"{subject_id}_double_dipping_report.json"
        )
        with open(report_file, "w") as f:
            json.dump(double_dipping_report, f, indent=2)
        print(f"Double-dipping report saved to: {report_file}")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    with track_runtime():
        main(args)
