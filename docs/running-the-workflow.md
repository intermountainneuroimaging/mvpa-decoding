[← README](../README.md) · [all docs](README.md)

# Running `workflows/mvpa_workflow.py`

`mvpa_workflow.py` replaces two previous, separate scripts
(`mvpa_generalization_workflow.py` and `mvpa_kfold_workflow.py`) with one:
it always fits a classifier on the complete `model_conditions.training` set,
then runs up to three further steps -- each **independently optional**,
each only writing its own output when its config section is present:

1. **`model.kfold_cv`** ([model config](model-config.md)) -- k-fold cross-validates entirely
   within `model_conditions.training` (train on non-held-out runs, evaluate
   on held-out runs of that same training data). Output under `model/`.
2. **`model_conditions.testing`** ([model_conditions](model-conditions.md)) -- evaluates the
   complete-training-set classifier against a genuinely separate test set.
   Output under `test/`.
3. **`model_conditions.timecourse_decoding`** ([model_conditions](model-conditions.md)) -- continuously
   decodes every BOLD volume of every run in `master_spreadsheet.csv`, with
   no `hemodynamic_lag` shift and no exclusions, using that same
   complete-training-set classifier (never per-fold, even when
   `model.kfold_cv` is also configured). Output under `decoding/`.

Any combination of the three is valid -- all three, any two, one, or none
(in which case the run only produces the trial pivot table sanity check
below, with a printed warning that nothing else was configured). This
example runs all three against `tutorial/config-haxby.example.json`'s
training(1-9)/testing(10-12) split:

```
python workflows/mvpa_workflow.py --subject 1 --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv --analysis-output-dir ./out
```

There's no separate `inputs.json`/`--input-scaffold` -- everything comes
from the one config plus `master_spreadsheet.csv`. For a given `--subject`,
the script:

1. Filters `master_spreadsheet.csv` to that subject and writes a **trial
   pivot table** (see below) -- a sanity check, computed before any
   `model_conditions` filtering.
2. Resolves `model_conditions.training`/`testing`'s rows via
   `label_conditions_with_lag` -- administrative events excluded as
   candidates, `conditions` matched against each real event's own values
   (first match wins, in the order conditions are listed --
   `validate_model_config.py` already warns about ambiguous overlaps), then
   `event_extraction.hemodynamic_lag`-shifted volumes selected for each match
   (see [model_conditions](model-conditions.md)). Loads BOLD patterns directly from each row's `boldfile`
   (already a concrete, resolved path -- no glob/pattern matching needed at
   this stage), z-scores, and slices to `volume_of_interest`.
3. **Always** trains a classifier on the complete `training` set -- this is
   what `timecourse_decoding` predicts with below, and (when `testing` is
   configured) what gets evaluated against it.
4. If `model.kfold_cv` is configured: resolves fold membership from
   `training` data alone (`resolve_kfold_folds`), then repeatedly trains a
   *fresh* classifier on the non-held-out folds and evaluates on the
   held-out fold, aggregating across folds. Writes per-fold and aggregated
   accuracy/evidence/AUC and importance-map NIfTIs, plus the fold manifest,
   under `<analysis-output-dir>/<desc>/<subject>/model/`.
5. If `model_conditions.testing` is configured **and the double-dipping
   guard below doesn't skip it for this subject**: evaluates the
   complete-training-set classifier (from step 3) against `testing`,
   writing accuracy/evidence/AUC and an importance-map NIfTI under
   `<analysis-output-dir>/<desc>/<subject>/test/`.
6. If `model_conditions.timecourse_decoding` is configured **and the
   double-dipping guard below doesn't skip it for this subject**: reads
   `master_spreadsheet.csv` for this subject (no lag, no exclusions),
   restricts to boldfiles that *qualify* (any row matching any `conditions`
   query -- see [model_conditions](model-conditions.md)), partitions every volume of those boldfiles into
   trials anchored on `trial_start_event` (`window_index` counts up from 0 at
   each match, until the next one), labels each volume's `regressor_label`
   via `conditions` where its real `trial_type` matches (blank otherwise --
   that volume is still decoded, just unscored), predicts with the
   complete-training-set classifier from step 3 (or, per the guard below,
   each overlapping run's
   own held-out k-fold classifier) on **every** volume of every qualifying
   boldfile, and writes two files to
   `<analysis-output-dir>/<desc>/<subject>/decoding/`:
   - `{subject}_decoding_results.csv` -- **raw**, one row per volume of every
     qualifying boldfile (every frame, nothing skipped within it), with its
     own `predicted_label` and `evidence_<category>` columns. This is the real
     per-TR data -- use it for anything that needs trial-level detail
     (custom stats, sanity-checking individual trials, filtering to whichever
     real events you actually care about).
   - `{subject}_summary_decoding_results.csv` -- the raw table grouped by
     `(window_index, regressor_label)` and averaged across every trial in
     that group (trial-count-weighted, not an average of averages) -- the
     confusion-style timecourse view `generate_report.py` ([reports](reports.md)) reads.

This works for **any number of conditions (2 or more)** in every step --
`model_performance` derives its class list from `clf.classes_` (what the
classifier actually learned), not from whatever happens to appear in a
given fold's/test set's held-out data, so accuracy/evidence/AUC stay
consistently shaped regardless of how many conditions you configure.

`tutorial/haxby-data/derivatives/sub-1/masks/native_epi_mask.nii.gz` is a
real mask already, so this runs as shown above with no extra setup -- if
your own dataset doesn't have one yet, point `model.mask.mask_pattern` at a
real (or throwaway, for testing) mask file first.

## ⚠️ Double-dipping guard: training vs. testing/timecourse independence

**A classifier evaluated on data from a run it was even partly trained on
has inflated apparent performance.** fMRI noise is autocorrelated within a
run (thermal drift, motion, physiological signal), so a train/test split
that isn't clean at the run level lets the classifier partially "recognize"
the run itself rather than genuinely generalizing -- the textbook
non-independence/circular-analysis problem. It's an easy trap to fall into
here specifically because `model_conditions.training`/`testing`/
`timecourse_decoding` are three independent queries over the same
`master_spreadsheet.csv` -- `training`/`testing` resolved through
`label_conditions_with_lag`, `timecourse_decoding` read continuously,
unshifted -- and nothing stops two of them from matching rows out of the
very same bold file (e.g. reusing the same task/run range, or a
`kfold_cv.example.json`-style config where `testing` is left
byte-identical to `training` for convenience).

`mvpa_workflow.py` checks this **per subject** (not assumed from the config
alone -- boldfile overlap can differ subject to subject, e.g. missing
runs), by boldfile rather than run number (two different tasks can reuse
the same run number for genuinely different scans, so boldfile is the only
unambiguous "same scan" identifier):

- **`training`/`testing` overlap**: the held-out test evaluation is
  **skipped entirely** for that subject -- no `test/` output. There's no
  substitute classifier that's both "the one full-training fit" and "never
  trained on the overlapping run," so skipping is the only honest option.
- **`training`/`timecourse_decoding` overlap**: if `model.kfold_cv` (section
  5) is configured, the overlapping run(s) are decoded with **their own
  held-out k-fold classifier** instead of the full-training one -- the same
  classifier that never saw that run during its own k-fold evaluation.
  Non-overlapping rows still use the full-training classifier as usual. If
  `model.kfold_cv` isn't configured, there's no held-out classifier to
  substitute, so timecourse decoding is **skipped entirely** for that
  subject instead.

Either case prints a `(!) DOUBLE-DIPPING WARNING` naming the exact
overlapping boldfile(s), so it's visible in the job log even when you don't
notice it in the config.

**`model.allow_train_test_overlap: true`** disables both of the above,
restoring the original behavior (evaluate/decode with the full-training
classifier regardless of overlap) -- the warning still prints, but nothing
is skipped or substituted.

> **⚠️ Only set `model.allow_train_test_overlap: true` if you have a
> specific, considered reason the overlap in your config is not a
> methodological problem** (e.g. you are intentionally re-scoring the
> training fit for a sanity check, not reporting it as a generalization or
> decoding result). Left at its default (`false`, or simply omitted), the
> guard above is exactly what protects you from silently publishing
> inflated accuracy/AUC or decoding numbers due to train/test leakage
> within a run. When in doubt, leave it unset and fix the overlapping
> `model_conditions` query instead.

## Trial pivot table (sanity check)

Written to `<analysis-output-dir>/<desc>/<subject>/<subject>_trial_pivot.csv`
before any `model_conditions` filtering -- one row per event retained in
`master_spreadsheet.csv` (i.e. every row of that subject's events.tsv files
across all runs, minus the hardcoded fixation/block/postRT exclusions above),
tagged with `training_condition` (which `model_conditions.training` query,
if any, matched that row) and, when `model_conditions.testing` is
configured, `testing_condition` likewise. When `model.kfold_cv` is
configured, one additional `fold{N}_split` column per fold shows `"train"`/
`"test"`/`""` for that fold specifically -- fold membership is entirely
within `training_condition` rows, since k-fold no longer touches `testing`
rows at all. Useful for eyeballing whether the volume counts and fold
membership per trial look right -- not used by the modeling steps
themselves.

## Outputs

Copied from `mvpa_workflow.py`'s own module docstring -- the authoritative
listing -- under `<analysis-output-dir>/<model.desc>/<subject>/`:

```
<subject>_trial_pivot.csv                     -- sanity check, pre-model_conditions

model.kfold_cv configured -- k-fold CV entirely within model_conditions.training:
  model/<subject>_kfold_folds.json                    -- {fold_id: {"training": [acquisition
                                                          names], "testing": [acquisition names]}}
  model/<subject>_fold{N}_model_results_{metric}.csv  -- per-fold held-out metrics ({metric}:
                                                          accuracy/evidence/auc, one file each, plus
                                                          "metadata" -- total_scores/whole_voxels/
                                                          selected_voxels/feature_percent together
                                                          in one small file, see save_model_results)
  model/<subject>_fold{N}_impa[_mni].nii.gz           -- per-fold importance map
  model/<subject>_fold{N}_permutation_test.csv        -- per-fold significance (optional)
  model/<subject>_model_results_{metric}.csv          -- aggregated across folds
  model/<subject>_impa[_mni].nii.gz                   -- aggregated importance map
  model/<subject>_cv_results.csv                      -- raw, one row per held-out sample
                                                          across all folds (task, trial_type,
                                                          run, fold, predicted_label, correct,
                                                          evidence_<category>)

model_conditions.testing configured -- one fit on all of training, evaluated
against all of testing:
  test/<subject>_model_results_{metric}.csv           -- held-out test metrics
  test/<subject>_impa[_mni].nii.gz                     -- that fit's importance map
  test/<subject>_permutation_test.csv                 -- significance (optional)

model_conditions.timecourse_decoding configured -- always the complete-training
classifier, never per-fold:
  decoding/<subject>_decoding_results.csv             -- raw, one row per decoded TR
  decoding/<subject>_summary_decoding_results.csv     -- averaged per (window_index, regressor_label)
```

`model/` is therefore exclusively k-fold's directory, `test/` is exclusively
the independent-test-set evaluation's directory -- a subject can have
either, both, or neither depending on what's configured. Importance-map
filenames use `_impa_mni` instead of plain `_impa` when `model.mnispace` is
set ([model config](model-config.md)) -- see `utils.mvpa_common.impa_tag`.

`generate_report.py` already detects `_fold{N}_*` files under `model/`
purely by their presence on disk (see [reports](reports.md)) -- fold-variability panels
(accuracy/AUC overlays, timecourse bands, an importance-map consistency
mosaic) render automatically whenever `model.kfold_cv` output exists, with
no report-side configuration needed.
