[← README](../README.md) · [all docs](README.md)

# `model`

Read by `mvpa_workflow.py`. Everything the analysis itself needs that isn't
about *which rows* to use (that's `model_conditions`'s job):

```json
"model": {
  "desc": "haxby_object_classifier",
  "mask": {
    "mask_pattern": "tutorial/haxby-data/derivatives/sub-{subject}/masks/native_epi_mask.nii.gz"
  },
  "mnispace": false,
  "featureSelection": {
    "model": "ANOVA",
    "feat_p": 0.05
  },
  "classifier": {
    "name": "sklearn.linear_model.LogisticRegression",
    "params": {
      "penalty": "l2",
      "C": 0.5,
      "solver": "lbfgs",
      "max_iter": 10000,
      "class_weight": "balanced"
    }
  }
}
```

| Field | Meaning |
|---|---|
| `desc` | Short name for this classifier variant; sanitized into the output folder name. |
| `mask.mask_pattern` | *(optional)* The full path to the mask NIfTI -- absolute, or relative to wherever the workflow script is run from (same convention `bids_root`/`derivatives_root` use); never resolved against either of those or any other root. Include `{subject}`/`{session}` placeholders (filled in from whichever row is being loaded) for one native-space mask per subject, as in the example above -- or omit them entirely for a single shared mask used for every subject, e.g. one MNI-space group mask (a template with no placeholders just formats to itself, so every subject resolves to the same literal path). Can still contain glob wildcards either way -- resolved the same way as bold-file lookups. Omit `mask` (or `mask_pattern`) entirely and every voxel is used instead -- a warning is printed, since a real analysis almost always wants a real mask (huge feature count otherwise, including background/non-brain voxels). A *configured* `mask_pattern` that matches no file is still a hard error, not a fallback -- only leaving it unset falls back. |
| `mask.detrend` | *(optional, default `true`)* Whether to linearly detrend each run's BOLD patterns (nilearn's own `NiftiMasker` detrending) before the per-run `StandardScaler` z-score that always runs regardless. Set `false` to skip detrending -- a warning is printed per (subject, session) when disabled, the same way an unset `mask_pattern` prints one, since skipping detrending is the unusual case worth flagging rather than a silent default. |
| `mnispace` | *(optional, default `false`)* Set `true` when the BOLD/mask this subject's model is fit on are already registered to MNI space -- there's no way to detect this automatically from the file itself, so it's an explicit claim you make. Controls two things at once (see [reports](reports.md)): the importance-map filename (`{subject}_impa_mni.nii.gz` instead of plain `{subject}_impa.nii.gz`), and whether `generate_report.py` plots it against nilearn's bundled MNI152 template as anatomical background. Setting this `true` also means every subject's importance map already carries the exact filename `generate_report.py`'s cross-subject group averaging looks for -- no separate `hcp_resample.py --direction native2mni` step needed. Leave `false` (or omit) for native-space or otherwise-unregistered data. |
| `featureSelection.feat_p` | ANOVA p-value threshold -- voxels with `p < feat_p` are kept, widened automatically until at least 5 voxels are selected. Ignored when `n_voxels` is set. |
| `featureSelection.n_voxels` | *(optional)* Select exactly this many voxels by ANOVA F-score instead, regardless of significance (sklearn's `SelectKBest` equivalent) -- takes priority over `feat_p` when both are present. Useful for keeping feature count fixed across subjects/folds whose signal strength (and thus a p-value threshold's actual voxel count) varies. `model_results_metadata.csv` still records whichever mode was actually used: `selected_voxels` (identical to `n_voxels` here) is populated either way. |
| `classifier` | Any importable scikit-learn-style estimator: `name` is a dotted import path, `params` are passed straight through as kwargs. |
| `classifier.one_vs_rest` | *(optional, default `false`)* Wrap the estimator in sklearn's `OneVsRestClassifier` (one independent binary classifier per class). Required for `LogisticRegression` with `solver: "liblinear"` on 3+ classes -- scikit-learn >= 1.8 raises a `ValueError` for that combination unless wrapped (earlier versions applied one-vs-rest silently, 1.7.x with a deprecation warning). Sits next to `name`/`params`, not inside `params`. Importance maps and evidence work unchanged: weights are stacked from the per-class estimators, and evidence comes from the wrapper's `predict_proba` (per-class scores normalized to sum to 1 across classes). |
| `classifier.evidence` | *(optional, default `"probability"`)* How the per-class "evidence" in every output is computed -- one definition, applied identically to the held-out/k-fold evidence matrices, `cv_results.csv`, the timecourse `decoding_results.csv`, the permutation test's AUC, and therefore every report plot. `"probability"` (default): the classifier's normalized class probabilities (`predict_proba`, rows sum to 1). `"sigmoid"`: an independent sigmoid of each class's one-vs-rest decision score, `1 / (1 + exp(-score))` -- not normalized across classes (rows generally don't sum to 1) and not a calibrated probability. The sigmoid is only literally a *one-vs-rest* score when the model is fit one-vs-rest (`classifier.one_vs_rest: true`); with a multinomial model the scores being transformed are softmax logits, and a warning is printed. Predictions/accuracy never depend on this setting -- only the evidence values (and the AUCs built from them) change. Sits next to `name`/`params`. |
| `kfold_cv` | *(optional)* Cross-validates entirely within `model_conditions.training` -- see below. Omitting it entirely skips k-fold cross-validation: no `model/` output at all, no extra runtime for that step. |

Omit either of `featureSelection`/`classifier` and it falls back to a
default (ANOVA @ p<0.05, `LogisticRegression`); omit `mask` and every voxel
is used (with a warning) -- only `desc` is truly required.

## `kfold_cv` (optional): cross-validating within `model_conditions.training`

There's no automatic internal-CV diagnostic anymore -- earlier versions of
this pipeline derived a leave-one-run-out (or stratified-trial) split
automatically from the training data itself. That heuristic is retired: if
you want a k-fold-style diagnostic today, configure it explicitly via
`model.kfold_cv`, e.g. the old leave-one-run-out default's equivalent:

```json
"model": {
  ...,
  "kfold_cv": {
    "strategy": "per_run"
  }
}
```

`mvpa_workflow.py` repeatedly holds out a group of runs from
`model_conditions.training`: trains on the rest of `training`, evaluates on
the held-out group, then aggregates across every fold -- entirely within
`training` data, never touching `testing`. `resolve_kfold_folds` (in
`mvpa_workflow.py`) resolves fold membership from just `(kfold_cv_cfg,
training_df)`:

| `strategy` | Meaning |
|---|---|
| `"per_run"` | Automatic, leave-one-run-out -- one fold per distinct run found in this subject's `model_conditions.training` data. |
| `"group_kfold"` | Automatic -- requires an integer `n_splits` (>= 2, <= the number of distinct training runs); runs are split into `n_splits` contiguous groups. |
| `"explicit_groups"` | User-defined -- requires `"held_out_runs"`. See below. |

Whichever strategy is used, the resolved fold membership is always written
to `model/{subject}_kfold_folds.json` -- `{fold_id: {"training": [acquisition
names], "testing": [acquisition names]}}`, e.g. `{"1": {"training":
["sub-1_ses-A1_task-loc_run-02_bold", ...], "testing":
["sub-1_ses-A1_task-loc_run-01_bold"]}}` -- full acquisition names (subject/
session/task/run, parsed from each row's own `boldfile`), not just bare run
numbers, since a run number alone is ambiguous across tasks that reuse it
(see `boldfile_overlap`). So an automatic split is just as inspectable after
the fact as an explicit one.

### `strategy: "explicit_groups"` and `held_out_runs`

`held_out_runs` is a list of lists -- **one inner list per fold, and each
inner list is that fold's held-out run(s)**, not what to train on:

```json
"kfold_cv": {
  "strategy": "explicit_groups",
  "held_out_runs": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
}
```

For a training set spanning runs 1-12, this example produces 4 folds,
grouping them into 3-run blocks. For fold 1 (`[1, 2, 3]`): that fold trains
on every `training`-condition row whose `run` is *not* 1, 2, or 3, and
evaluates on the rows whose `run` *is* 1, 2, or 3. Folds 2-4 work the same
way against `[4, 5, 6]`, `[7, 8, 9]`, and `[10, 11, 12]`. Each fold trains
and evaluates independently -- run IDs that never appear in `held_out_runs`
are simply never held out, so they're always available for training but
never scored on their own.

A few things worth knowing before writing your own:

- **The lists don't have to partition the runs.** A run can appear in more
  than one fold's held-out set (evaluated more than once, in different
  folds), and runs can be left out of `held_out_runs` entirely (always
  trained on, never held out and scored). Use this deliberately -- e.g. to
  build unequal-sized folds, or to only ever evaluate a specific subset of
  runs -- not by accident.
- **Coverage is checked, but only warned about, not enforced.** If a run
  present in this subject's `model_conditions.training` data isn't covered
  by any group in `held_out_runs`, you'll see `(!) model.kfold_cv.held_out_runs
  doesn't cover run(s) [...]` -- those rows are simply never evaluated in any
  fold, the run itself is not an error. Conversely, if `held_out_runs`
  references a run ID that doesn't exist in this subject's training data at
  all, you'll see `(!) model.kfold_cv.held_out_runs references run(s) [...]
  that don't appear ...` and that fold ends up with 0 held-out rows
  (skipped at runtime with its own warning, not a crash).
- **Run IDs must match exactly, no type coercion.** `training_df["run"]`
  (from `master_spreadsheet.csv`) is typically integers, and `held_out_runs`
  is matched against it with `.isin()` -- a JSON string `"1"` will never
  match integer `1`, it'll just silently produce an empty fold (with the
  "references run(s) that don't appear" warning above) rather than raising.
  If your runs come out as strings, write `held_out_runs` as strings too
  (`"held_out_runs": [["1", "2"]]`), matching whatever
  `master_spreadsheet.csv`'s `run` column actually contains.
- **Which config `run` values are valid to reference** -- the "universe" of
  runs `held_out_runs` is checked against is the set of `run` values present
  in this subject's *matched* `model_conditions.training` rows specifically
  -- not every run in `master_spreadsheet.csv`, and not `testing`/
  `timecourse_decoding` (unlike before this pipeline's k-fold and
  independent-test-set steps were split into two separate config sections,
  fold membership now only ever depends on `training`). A run with no
  training-condition trials for this subject won't show up in that
  universe, so referencing it in `held_out_runs` triggers the "doesn't
  appear" warning even though the run genuinely exists in the data -- it
  just has nothing to fold over.

`model.permutation_test` (below) works the same way here as for the
independent-test-set evaluation, except it runs **once per fold**, on that
fold's own train/held-out split within `training` --
`model/{subject}_fold{N}_permutation_test.csv`. Folds aren't combined into
one pooled p-value; interpret them fold-by-fold.

### Cross-validation hold-out sample detail: `model/{subject}_cv_results.csv`

In addition to the per-fold/aggregated accuracy summaries above, every fold
also writes its held-out samples out at the trial level -- **one row per
held-out sample, across all folds** -- to `model/{subject}_cv_results.csv`,
in the same raw-table style as `decoding_results.csv` (see [reports](reports.md)):
`subject`, `model_descr`, `fold`, plus every column already carried by that
row's `model_conditions.training` match (`task`, `trial_type`, `run`,
`boldfile`, `trial_index`, `regressor_label`, ...), then `predicted_label`,
`correct`, `evidence_<category>`, and that fold's own feature-selection
footprint (`threshold_p`/`selected_voxels`/`whole_voxels`/`feature_percent`).

Two volume columns are carried through, deliberately distinct: `volume_of_interest`
is the matched event's own real, unshifted position (same meaning as in
`master_spreadsheet.csv`/`decoding_results.csv`), while `volume_of_interest_withlag`
is the `hemodynamic_lag`-shifted frame actually fetched from the BOLD file to
train/evaluate the classifier (see `label_conditions_with_lag`) -- don't
confuse the two when lining this table up against event timing elsewhere.
Since every row is scored by whichever fold held its own run out, `fold`
plus `run` together document exactly which run was held out (and evaluated)
in each fold -- e.g. with `"per_run"`, fold 1's rows are all the run-1 rows,
fold 2's are all the run-2 rows, and so on. `generate_report.py`'s group
report concatenates every subject's file into
`{desc}_group_cv_results.csv` (see [reports](reports.md)), so you can load the whole
group's hold-out predictions in your own software without touching this
pipeline's internals.

<details>
<summary>Historical note: the retired automatic internal-CV heuristic</summary>

Prior to `model.kfold_cv`, `mvpa_generalization_workflow.py` derived a CV
split from the training data itself, with no config knob:

- **Leave-one-run-out** (`PredefinedSplit` on the `run` column, one fold per
  distinct run) when every training run contains the same set of conditions
  -- i.e. every run is a full replicate of the training task. This was the
  default case, and matches the leave-one-run-out scheme in the paper this
  pipeline replicates (see [THEORY.md](../THEORY.md)) -- reproduce it today via
  `model.kfold_cv: {"strategy": "per_run"}`.
- **4-fold stratified CV over trials pooled across all runs** (not scoped to
  any one run) when runs *didn't* all share the same conditions -- holding
  out a whole run in that case would risk silently dropping a condition from
  one side of a fold entirely, so fold membership was built directly from
  trials instead. Each condition's trials were gathered from every run
  together and partitioned into 4 folds independently (`StratifiedKFold`),
  grouped by `(run, trial_index)` so every volume belonging to one event
  stayed on the same side of its fold -- row-level splitting would let
  correlated volumes from the same trial leak across train/test. If the
  rarest condition had fewer than 4 trials, the fold count was reduced
  automatically (down to a minimum of 2) so every fold still got at least
  one trial of every condition; fewer than 2 trials for the rarest condition
  was a hard error. A warning was printed when this fallback triggered, and
  each fold's train/test row counts and held-out trials were logged as
  they're built.

Either way this only ever touched `model_conditions.training` data -- it
never looked at `testing`/`timecourse_decoding` rows, same as
`model.kfold_cv` today.

</details>

## `permutation_test` (optional): significance testing for held-out results

Accuracy/AUC on their own don't say whether a classifier is doing better
than chance -- add `permutation_test` to `model` to find out, via
[`sklearn.model_selection.permutation_test_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
(the tool nilearn's own decoding docs recommend for this exact fMRI
classification case):

```json
"model": {
  ...,
  "permutation_test": {
    "n_permutations": 1000,
    "random_state": 0
  }
}
```

Omitting `permutation_test` entirely skips it -- no extra runtime, no output
file. Present (even as `{}`) runs it, with `n_permutations` defaulting to
1000 and `random_state` to 0 if unset -- independently for each of
`model.kfold_cv` (if configured, once per fold: `training` vs. that fold's
own held-out runs) and `model_conditions.testing` (if configured: the
complete `training` set vs. `testing`). Whichever of those two steps are
configured, each gets its own permutation test; neither depends on the
other being present.

For each of `accuracy` and `roc_auc_ovr`, the train/held-out split in play
(a fold's split, or the full training/testing split) is encoded as a single
fixed `PredefinedSplit` fold, then `permutation_test_score` repeatedly
reshuffles the combined label vector and refits the *entire* pipeline
(feature selection + classifier, both) on each shuffle -- the
textbook-correct way to build a null distribution for a fixed train/test
split, not a naive shuffle done outside the fit structure. Writes
`model/{subject}_fold{N}_permutation_test.csv` per fold for `model.kfold_cv`
(not pooled across folds -- interpret them fold-by-fold), and/or
`test/{subject}_permutation_test.csv` for `model_conditions.testing`
(`metric,real_score,p_value,n_permutations` either way).

`accuracy`'s `real_score` matches the corresponding `model_results_metadata.csv`'s
`total_scores` value exactly (same split, same fixed feature-selection threshold,
same classifier config -- `model/{subject}_fold{N}_model_results_metadata.csv`
for a kfold fold, `test/{subject}_model_results_metadata.csv` for the
independent test set). `roc_auc_ovr`'s `real_score` will be *close to* the
mean of the corresponding `model_results_auc.csv`'s per-category values --
both are one-vs-rest AUC built from the same evidence
(`decision_evidence`: by default `predict_proba()`/softmax, rows sum to 1;
an independent per-class sigmoid instead if `classifier.evidence` is
`"sigmoid"` -- either way the same definition for both), so they should agree closely; sklearn's
scorer averages the OvR AUCs itself while `model_results_auc.csv` reports
them per category, so don't expect bit-identical numbers, just closely
matching ones. `p_value` is the fraction of permuted-label refits that
scored as well or better than the real fit.

This costs `n_permutations` extra fits (parallelized across cores via
`n_jobs=-1`) on top of the one real fit -- cheap relative to a single
subject's BOLD loading time in practice, but scales with `n_permutations`
(and, for `model.kfold_cv`, with the number of folds too, since each fold
pays for its own `n_permutations` fits), so drop it for quick iteration and
turn it on for a result you're about to report.

## `allow_train_test_overlap` (optional): ⚠️ double-dipping guard override

```json
"model": {
  ...,
  "allow_train_test_overlap": true
}
```

Defaults to `false` (or simply omit it). `mvpa_workflow.py` automatically
detects when `model_conditions.training` shares a bold file with `testing`
or `timecourse_decoding` for a given subject and, by default, skips (or
substitutes a held-out k-fold classifier for) whichever step would
otherwise double-dip -- see [running the workflow](running-the-workflow.md)'s "Double-dipping guard" for the full
explanation. Setting this to `true` disables that protection and restores
the original behavior. **Only do this if you have a specific, considered
reason the overlap in your config isn't a methodological problem** -- see
the warning in [running the workflow](running-the-workflow.md) before using it.
