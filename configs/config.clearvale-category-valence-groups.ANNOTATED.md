# Clearvale face / place / item category classifier -- configs, annotated

JSON has no comments, so every key is explained here. Same layout as
`config.clearvale-operation-iti-liblinear-kfold.ANNOTATED.md`.

| File | What it is |
|---|---|
| `config.clearvale-category-valence-groups.json` | **Model B.** Category classifier trained on the localizer, *all* training trials (108 face / 108 place / 54 item per subject). Results folder `vvps_category_loc2WM_timecourse_liblin_onevrest_classifier` |
| `config.clearvale-category-fullyrestricted-valence-groups.json` | **Model C.** Identical to B plus **balanced training** (`trial_restriction`): 54 / 54 / 54 trials, face and place 27 pos + 27 neg, 4 TRs per trial. Results folder `..._fullyrestricted_classifier` |
| `restriction.clearvale-category.json` | the stage-2b file that builds Model C's trial list (see "The restriction file") |

The older normalized-probability model (multinomial, lbfgs, C = 0.5, "Model A") is no longer in `configs/`; it is in git
history and in the results folders.

## What the analysis does

1. **Training** (`model_conditions.training`): the localizer (`task` = `loc`) trials of each condition -- face, place, item --
   by `trial_type` regex, each shifted forward by `hemodynamic_lag` (4.6 s = 10 TRs) before volumes are taken.
2. **k-fold** (`model.kfold_cv`, `per_run`): leave one localizer acquisition out, within the training set.
3. **Held-out test** (`model_conditions.testing`): one classifier on all of training, scored on the working-memory
   *view* cues (`view_face`, `view_place` in the WMpos/WMneg tasks). No item class: there are no item trials in the WM tasks.
4. **Timecourse decoding** (`model_conditions.timecourse_decoding`): that classifier applied to every volume of the WM runs,
   unshifted, anchored on each trial's view cue (`view_.*`), with the operation/valence overlay used by the report.
5. **Permutation test**: 1000 permutations, seed 0.

## Key by key (shared by B and C)

| Key | Value | What it does / why |
|---|---|---|
| `event_extraction.hemodynamic_lag` | `4.6` | seconds the BOLD response lags the event; training/testing windows are `[onset + 4.6, onset + 4.6 + duration)`. Must match the restriction step (stage 2b reads it from here) |
| `event_extraction.bold_glob` / `events_glob` | MNI152NLin6Asym `desc-preproc` BOLD, `*_events.tsv` | which scans and event files go into the master spreadsheet |
| `training.conditions` | `face` / `place` / `item` = `task` exact `loc` AND `trial_type` regex `.*face.*` / `.*place.*` / `.*item.*` | the three classes. Trial types in this study are `positive_face`, `negative_face`, `positive_place`, `negative_place`, `neutral_item`. Valence is **not** a class: pos and neg trials share a class |
| `testing.conditions` | `face` / `place` = `task` regex `^WM.*` AND `trial_type` `view_face` / `view_place` | the WM view cues, scored against the localizer-trained classes |
| `timecourse_decoding.trial_start_event` | `trial_type` regex `view_.*` | each view cue starts a trial; `window_index` counts from 0 there |
| `timecourse_decoding.annotation_labels` | `view`, `operation` groups | only names the event markers on report/notebook figures |
| `timecourse_decoding.overlay` | `maintain/suppress/switch/clear` x `pos/neg` | report only: operation colour, valence block (`group`). Does not change any decoding |
| `model.desc` | see table above | names the results folder (`<analysis-output-dir>/<desc>/`). A new `desc` = a new folder, nothing is overwritten |
| `model.mnispace` / `model.mask` | `true`, vvps temporal ROI (3 mm) | decode within the vvps temporal mask in MNI space |
| `model.featureSelection` | ANOVA, `feat_p` 0.05 | voxels with p < 0.05 for the class effect, chosen inside each fold |
| `model.classifier.name` + `params` | `LogisticRegression`, `l2`, `C` 50, `liblinear`, `max_iter` 10000, `class_weight` balanced, `random_state` 6100 | the model. `balanced` weights compensate class-size differences (B has 54 item vs 108 face/place; C has none) |
| `model.classifier.one_vs_rest` | `true` | one independent binary classifier per class. Required for `liblinear` on 3+ classes in scikit-learn >= 1.8 |
| `model.classifier.evidence` | `"sigmoid"` | evidence = independent sigmoid of each class's one-vs-rest score: rows do **not** sum to 1 and are not calibrated probabilities. Accuracy is unaffected; the evidence plots are |
| `model.kfold_cv.strategy` | `per_run` | leave one acquisition out |
| `model.permutation_test` | 1000, `random_state` 0 | significance of the held-out test (and per fold) |
| `master.*` | script, BIDS/HCP roots, GM mask, output dir, master spreadsheet | cluster paths for the slurm stages. Section **renamed from `pipeline`**; the old name is refused by the stage scripts |

## How the two models differ

| | Model B (`...category-valence-groups.json`) | Model C (`...category-fullyrestricted-valence-groups.json`) |
|---|---|---|
| `model.desc` | `..._liblin_onevrest_classifier` | `..._liblin_onevrest_fullyrestricted_classifier` |
| `training.trial_restriction` | none (every localizer trial) | `.../master_spreadsheet_trial_restriction.csv` (stage 2b list) |
| training trials per subject | 108 face / 108 place / 54 item | 54 / 54 / 54 (face and place 27 pos + 27 neg), 4 TRs each |
| classifier, mask, folds, permutation test, evidence | same | same |

History of the classifier settings: the earlier model had `C` 0.5, `lbfgs` (multinomial), no `random_state`, no
`one_vs_rest`, default (normalized probability) evidence. B and C add `C` 50, `liblinear`, `random_state` 6100,
`one_vs_rest: true`, `evidence: "sigmoid"`.

## The restriction file (`restriction.clearvale-category.json`, Model C only)

Written for `workflows/restrict_master_spreadsheet.py`; `docs/trial-restriction.md` has the full description.

| Key | Value | Meaning here |
|---|---|---|
| `primary_groups` | `face`, `place`, `item` = `task` `loc` AND `trial_type` regex `.*face.*` / `.*place.*` / `.*item.*` | the classes to equalize. Exactly the same trials the training conditions select (the `task` clause keeps the WM trials, which also contain "face"/"place", out of it) |
| `secondary_groups` | `positive` = `positive_.*`, `negative` = `negative_.*` | the valence split inside a class |
| `secondary_applies_to` | `["face", "place"]` | only face and place are split; item trials are all `neutral_item`, so item is equalized in total only |
| `drop_strategy` | `"spread_runs"` | drop from whichever run holds the most trials of a class, so every run keeps the same number (18 per run per class) and per-run cross-validation stays balanced. (`"random"` could leave a run short) |
| `seed` | `0` | makes the choice of dropped trials reproducible (with the subject ID) |
| `balance_trial_counts` | not set = `true` | trial counts are equalized (this is the point of Model C) |

Result per subject (3 localizer runs x 36 face, 36 place, 18 item): **54 face (27 + 27), 54 place (27 + 27), 54 item**, every
trial cut to the shortest trial of that subject (4 TRs here), extras removed from the end.

**Status of this file.** It was rebuilt from the trial-restriction audit table of the 88-subject run
(`master_spreadsheet_trial_restriction_audit.csv`): the group queries label all 23,760 audited trials exactly as the audit
does, the counts and the 18-per-run pattern match. `seed` (0) and `drop_strategy` (`spread_runs`) are inferred; if you still have
the file you originally ran, diff it against this one before assuming the same trial selection.

Things to know from that audit: 5 of 23,760 trials had no volume left after the lag (`outside_scan`, subjects 017 and 026) and
were dropped; only subjects with enough usable trials in every class get a list (the others are *excluded* and stage 3
refuses them); the 20 subjects missing from the group results were excluded on purpose upstream (QA).

## Run order (Model C; Model B skips 2b)

```bash
# 2. master spreadsheet (shared by every analysis)
# 2b. Model C only: the balanced trial list. Default output name = master_spreadsheet_trial_restriction.csv,
#     which is what the config points to
sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh \
    configs/config.clearvale-category-fullyrestricted-valence-groups.json \
    configs/restriction.clearvale-category.json
#    read the log for "(!)" coverage warnings and the summary for excluded subjects
# 3. decoding
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/config.clearvale-category-fullyrestricted-valence-groups.json
#    Model B:  ... configs/config.clearvale-category-valence-groups.json
# 4. group report as usual
```

The operation analysis shares this master spreadsheet and writes its own list
(`master_spreadsheet_trial_restriction_operation_iti.csv`, third argument of stage 2b), so the two never overwrite each other.

## Check before running

- `python utils/validate_model_config.py --config <config> --master-spreadsheet <master>` -- the one warning it prints
  (condition names differ between `training` and `testing`) is expected: `testing` has no `item`.
- Model C: rerun stage 2b whenever `hemodynamic_lag`, the restriction groups, the events files or the master spreadsheet change.
