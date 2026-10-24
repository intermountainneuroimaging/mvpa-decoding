# `config.clearvale-operation-iti-liblinear-kfold.json` -- every change, annotated

JSON has no comments, so each difference from the two files it was built from
is listed here. Built from:

- **operation config** = `config.clearvale-operation-valence-groups (1).json` (conditions, GM mask, folds, timecourse overlay)
- **category config** = `config.clearvale-category-valence-groups (1).json` (the updated classifier settings)

The equal trial length is done inside the training itself (`equalize_trial_length`), so the default route needs no
separate restriction step. The restriction step (`restriction.clearvale-operation-iti.json`, stage 2b) is an optional second
route that gives the **identical training set**: see "Two routes, one training set". The repo changes that make the new keys
work are listed under "Code changes" at the bottom.

## Substitutions (a value that existed in the operation config now has a new value)

| Key | Operation config | New | Why |
|---|---|---|---|
| `notes` | "Report-only alternate view of ..." | describes this analysis | the old note described a report-only re-render, which this is not |
| `model.desc` | `gm_operation_kfold_classifier` | `gm_operation_iti_liblin_onevrest_kfold_classifier` | `desc` names the results folder. Reusing the old name would write into (and mix with) the previous operation results |
| `model.classifier.params.C` | `0.5` | `50` | updated value from the category config |
| `model.classifier.params.solver` | `lbfgs` | `liblinear` | updated value from the category config |
| `pipeline` (section name) | `pipeline` | `master` | the section was renamed; the old name is refused by the slurm scripts with a message |

## Additions (keys that did not exist in the operation config)

| Key | Value | Why / what it does |
|---|---|---|
| `model.classifier.params.random_state` | `6100` | from the category config; makes liblinear's solver reproducible |
| `model.classifier.one_vs_rest` | `true` | from the category config. Required: scikit-learn >= 1.8 refuses `liblinear` on 3+ classes unless wrapped one-vs-rest (4 operations here) |
| `model.classifier.evidence` | `"sigmoid"` | from the category config. The evidence in every output is an independent sigmoid of each class's one-vs-rest score: rows do **not** sum to 1, and values are not calibrated probabilities. Accuracy is unaffected |
| `model_conditions.training.extend_events_until` | `trial_type` regex `view_.*` | **new pipeline option.** Each operation event's window runs from the operation onset to the onset of the next trial's view cue, i.e. **operation period + the ITI after it** (Nature Communications method). Every TR in it is labelled with the operation. The last trial of a run has no following cue: its window runs to the end of the scan and it is cut to the same TR count as every other trial (see "What the analysis will do" 4) |
| `model_conditions.training.equalize_trial_length` | `true` | **new pipeline option.** Within each subject, **every training trial is cut to the subject's shortest operation + ITI window**: the first TRs are kept, the extras are removed from the **end of the ITI**. Only that subject's own trials are compared. It replaces the separate restriction step (stage 2b) for this analysis -- no list file, no extra cluster job. Each subject also gets `model/<subject>_training_trial_length_audit.csv` and `_summary.csv` listing every trial and every TR cut |
| `model_conditions.timecourse_decoding.annotation_labels.view` | `["view_face", "view_place"]` | cosmetic (from the category config): labels the view-cue marker on report/notebook figures |

## Copied unchanged from the operation config (checked, not edited)

`event_extraction` (including `hemodynamic_lag: 4.6`) · the four training conditions (WM tasks only,
`.*maintain.*` etc.) · `timecourse_decoding.conditions` / `trial_start_event` (`view_.*`) / `overlay`
(face/place colour, pos/neg group) · `model.mnispace` · `model.mask` (grey-matter mask) ·
`featureSelection` (ANOVA, p < 0.05) · `penalty: l2`, `max_iter: 10000`, `class_weight: balanced` ·
`kfold_cv` explicit groups `[[1,3],[2,4],[5,7],[6,8]]` (each fold holds out one pos and one neg run;
valid for all four Clearvale protocols).

## Deliberately NOT taken from the category config

| Category setting | Why left out |
|---|---|
| `model.mask` = vvps temporal ROI | category-specific; the operation analysis keeps the grey-matter mask |
| `model.permutation_test` | held back for now; to be added later (the operation config never had one). When wanted: `"permutation_test": {"n_permutations": 1000, "random_state": 0}` inside `model` |
| `training`/`testing` conditions (loc -> WM) | the operation classifier is trained directly on the WM runs, evaluated by k-fold |

## What the analysis will do (read this against your intent)

1. **Training samples.** For every working-memory trial: the operation period plus the ITI that follows,
   shifted by the 4.6 s hemodynamic lag, one training sample per TR, all labelled with that trial's operation.
2. **Equal length.** Every trial of a subject is cut to that subject's shortest operation + ITI window; extra
   TRs come off the end of the ITI. From the existing decoding tables the windows are 11-17 TRs (median 14), so
   expect **11 TRs per trial (6 operation + 5 ITI)** -- in the existing data all 68 subjects have a shortest window of 11 TRs. Check
   `trs_per_trial` in each subject's `model/<subject>_training_trial_length_summary.csv`; `n_trs_cut` per trial is in the audit file.
3. **Trial counts are not balanced.** You asked for equal *length*, so no trial is dropped to equalize the number per
   operation; `class_weight: balanced` handles the (small) imbalance. (Equalizing counts too needs the separate restriction
   step with a `drop_trial_strategy` of `"random"` or `"spread_runs"`; the two ways of setting the TR count cannot be combined.)
4. **Last trial of each run.** It has no following cue, so its window runs to the end of the scan and it is cut to the shared TR count like
   every other trial. It never sets that count, and it is kept only if the scan leaves it that many TRs *after the 4.6 s (10-TR) lag*. In the
   existing data that is 416 of the 543 last trials; the other 127 end too close to the scan end (64 of them have their whole lagged window outside the
   scan) and are dropped (audit status `open_window_too_short`, listed in the subject's audit file). No other trial's lag-shifted window runs past the end of a scan.
5. **Scope.** Equalization and the extended window apply to training and the k-fold folds only.
   `timecourse_decoding` is unchanged: it decodes every volume of the held-out runs, unshifted, with the
   fold's own classifier.

## Run order

```bash
# 1. master spreadsheet (stage 2), unchanged -- if it already exists you can skip this
# 2. (no restriction step: the equal trial length is applied inside stage 3)
# 3. the decoding; each subject's log prints the TRs per trial and how many were cut
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/config.clearvale-operation-iti-liblinear-kfold.json
#    then read <analysis-output-dir>/gm_operation_iti_liblin_onevrest_kfold_classifier/<subject>/model/
#    <subject>_training_trial_length_summary.csv (per condition) and _audit.csv (every trial)
# 4. group report as usual
```

Validate first: `python utils/validate_model_config.py --config <config> --master-spreadsheet <master>`
(it now also checks that `extend_events_until` matches rows).

## Two routes, one training set

| Route | Steps | Where the equal length is applied |
|---|---|---|
| **A (default)** | stage 2 -> stage 3 | `model_conditions.training.equalize_trial_length: true` in this config |
| **B (optional)** | stage 2 -> 2b -> stage 3 | the restriction list (`restriction.clearvale-operation-iti.json`, `equalize_trial_length: true`, `drop_trial_strategy: "none"`) |
| A + B | both | both -- harmless: a list built with the rule has already capped every trial, so equalizing again changes nothing |

Both routes select exactly the same volumes of the same trials (tested on synthetic data, including the last-trial rule and
subjects with different shortest windows; row order is identical too). Requirements for that guarantee: the restriction's
primary group covers exactly the trials the training conditions select (the 2b log warns with `(!)` otherwise), and
`drop_trial_strategy` is `"none"`.

Route B commands:

```bash
sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh \
    configs/config.clearvale-operation-iti-liblinear-kfold.json \
    configs/restriction.clearvale-operation-iti.json \
    /pl/active/banich/studies/Clearvale/analysis/mvpa-decoding/master_spreadsheet_trial_restriction_operation_iti.csv
# stage 3 with the list as its second argument (no config edit needed):
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/config.clearvale-operation-iti-liblinear-kfold.json \
    /pl/active/banich/studies/Clearvale/analysis/mvpa-decoding/master_spreadsheet_trial_restriction_operation_iti.csv
```

## Code changes this config depends on (all in the repo, tested)

- `utils/mvpa_common.py`: `extended_window_durations()` and the `extend_until` argument of `label_conditions_with_lag()`.
- `utils/mvpa_common.py` (also): `common_window_trs()`, `equalize_trial_lengths()` and the `equalize_length` / `report` arguments of
  `label_conditions_with_lag()` -- the same TR rule the restriction step uses.
- `workflows/mvpa_workflow.py`: passes `extend_events_until` and `equalize_trial_length` through and writes the per-subject audit and summary.
- `workflows/restrict_master_spreadsheet.py`: shares `common_window_trs()`; counts TRs over the extended window; `drop_trial_strategy`
  (`random` / `spread_runs` / `none`; replaces `drop_strategy`, the old name still works); audit/summary list the TRs cut. Not needed for this config.
- `utils/validate_model_config.py`: validates `extend_events_until` and `equalize_trial_length`.
- `slurm/2b_sbatch_restrict_master_spreadsheet.sh`: optional third argument = output path (restriction step only).
- Docs: `docs/model-conditions.md`, `docs/trial-restriction.md`.
