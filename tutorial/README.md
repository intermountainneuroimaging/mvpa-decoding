# Tutorial: object-category decoding on the Haxby et al. (2001) dataset

This walks through the full pipeline -- download, a minimal preprocessing
pass, table-building, config validation, classifier training/CV, and
timecourse decoding -- against a real public dataset, end to end, with the
actual commands and actual output from running them. It also documents,
deliberately and in detail, everywhere this tutorial cuts corners relative to
a real analysis. **Read the "Where this tutorial oversimplifies" section
before citing these numbers as anything more than a pipeline demonstration.**

`config-haxby.example.json`, `expected_events_haxby.example.json`, and
`preprocess_haxby.sh` in this folder are the exact files used below.

## The dataset

[OpenNeuro ds000105](https://openneuro.org/datasets/ds000105)
(DOI `10.18112/openneuro.ds000105.v3.0.0`) -- the original Haxby et al. (2001)
*Science* "Distributed and overlapping representations of faces and objects
in ventral temporal cortex" dataset. 6 subjects, 12 runs each, block design:
each run presents 8 categories (bottle, cat, chair, face, house, scissors,
scrambledpix, shoe) in blocks of 12 stimuli (SOA 2s, each stimulus shown for
0.5s), separated by rest. TR = 2.5s, volumes are small (40x64x64). This
tutorial uses subject 1 only.

This is **raw** BIDS data as archived on OpenNeuro -- there is no linked
fMRIPrep derivatives dataset for ds000105, so this tutorial does its own
(minimal) preprocessing, below.

## Prerequisites

- Network access to `s3.amazonaws.com` (OpenNeuro's public S3 mirror -- no
  account, API key, or `aws`/`datalad` CLI required, plain `curl` works).
- The same Python environment the rest of this repo uses (`pandas`, `numpy`,
  `nibabel`, `nilearn`, `scikit-learn`).
- **FSL** (tested against 6.0.7) on `PATH`/`FSLDIR` set, for the preprocessing
  step -- see [fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index)
  if you don't already have it.
- ~300MB free disk space for one subject's raw functional data (preprocessing
  writes a similarly-sized copy alongside it).

## Step 1: Download one subject's data

```bash
mkdir -p tutorial/haxby-data/sub-1/func
BASE="https://s3.amazonaws.com/openneuro.org/ds000105/sub-1/func"
for run in 01 02 03 04 05 06 07 08 09 10 11 12; do
  curl -s -o "tutorial/haxby-data/sub-1/func/sub-1_task-objectviewing_run-${run}_bold.nii.gz" \
      "$BASE/sub-1_task-objectviewing_run-${run}_bold.nii.gz"
  curl -s -o "tutorial/haxby-data/sub-1/func/sub-1_task-objectviewing_run-${run}_events.tsv" \
      "$BASE/sub-1_task-objectviewing_run-${run}_events.tsv"
done
```

This pulls 12 runs x (1 `bold.nii.gz` + 1 `events.tsv`) = 24 files, ~25MB
each BOLD file, ~2KB each events file. `tutorial/haxby-data/` is gitignored --
this tutorial doesn't check the data itself into the repo.

**Note on filenames**: ds000105's events files have no `ses-` entity at all
(e.g. `sub-1_task-objectviewing_run-01_events.tsv`) -- valid BIDS, since
session labels are optional for single-session studies. 

## Step 2: Basic preprocessing

**This step is deliberately minimal -- see the caveats below before treating
it as a real preprocessing pipeline.** It does exactly three things, per run,
using FSL command-line tools:

1. **Motion correction** (`mcflirt`): align every volume to that run's own
   mean volume.
2. **Coregistration** (`flirt` + `applyxfm4D`): rigid-body (6 dof) register
   each run's motion-corrected mean volume to run 1's mean volume -- the
   common template -- then apply that single transform to every volume of
   the run at once.
3. **Linear detrending** (`fsl_glm`): fit a per-voxel GLM with a design of
   `[intercept, centered linear ramp]` and keep the residual (`--out_res`)
   -- exactly a linear detrend, voxel by voxel, along time. This leaves
   intensities centered on ~0 (positive and negative), so a constant `+10000`
   offset is added back afterward -- **only inside the brain mask** (via
   `fslmaths -mul mask`, broadcast across the 4D series) -- to keep in-brain
   voxels strictly positive, matching the nonnegative-intensity convention
   raw BOLD data normally has. Voxels outside the mask are left at their
   ~0 detrended value.

```bash
export FSLDIR=/path/to/fsl   # if not already set
bash tutorial/preprocess_haxby.sh
```

The whole script is ~90 lines (`tutorial/preprocess_haxby.sh`) built entirely
on FSL tools, and also computes one whole-brain mask (`bet`) from the average
of every run's coregistered (pre-detrend) mean volume -- pre-detrend since
detrending removes the intensity contrast `bet`'s segmentation relies on,
and post-coregistration since that's the point at which every run shares one
grid. It writes `..._desc-preproc_bold.nii.gz` files plus a mask into a
`tutorial/haxby-data/derivatives/` folder, structured so the pipeline can
find them via `derivatives_root` + `bold_glob` instead of `bids_root` --
exactly the fMRIPrep-derivatives use case those fields exist for.

Runtime: ~9 minutes for all 12 runs (mostly `mcflirt` and `applyxfm4D`, each
~20s/run).

## Step 3: The config

`config-haxby.example.json` provides an exemplar for running MVPA on a
preprocessed dataset. Since the Haxby dataset has no separate localizer and
main-task design, we hold out a fixed block of runs (10-12) as an
independent test set (`model_conditions.testing`) while training on the
rest (1-9) -- this is `mvpa_workflow.py`'s independent-test-set step,
written under `test/` and labeled "Full" by `generate_report.py`. This
config doesn't set `model.kfold_cv`, so it doesn't also cross-validate
within the 9 training runs -- see
[`config-kfold-haxby.example.json`](config-kfold-haxby.example.json) and
[README.md section 5](../README.md#5-model) for the fold-based alternative
(`model.kfold_cv`, written under `model/` and labeled "CV"), which portions
training data itself into successive held-out folds instead of relying on a
single fixed split. A few configuration settings to pay attention to:

- **`derivatives_root`/`bold_glob` point at the preprocessed data**, not
  `bids_root` -- `bids_root` still finds the events.tsv files (co-located
  with the *raw* BOLD, which is otherwise unused once preprocessing has run),
  while `bold_glob` matches the `_desc-preproc_bold.nii.gz` naming
  `preprocess_haxby.sh` writes.
- **Train/test split is by `run`, not `task`.** ds000105 has one task
  (`objectviewing`) repeated across all 12 runs -- there's no separate
  localizer task to train on like the other examples in this repo use. So
  `model_conditions.training` selects runs 1-9 and `model_conditions.testing`
  /`timecourse_decoding` select runs 10-12, both via `{"column": "run",
  "match": "in", "values": [...]}` rather than a `task` filter.
- **8 conditions, detailed in the task and train blocks** One per object category. This is also a real test of the classifier code path for >2 classes.

`event_extraction.hemodynamic_lag` is set to 4.0s (a generic HRF-peak
estimate, not tuned for this subject/dataset). `mask.mask_pattern` is
`sub-{subject}/masks/native_epi_mask.nii.gz`, resolved relative to
`derivatives_root` (masks default there).

## Step 4: Build the volume table

```
python workflows/generate_master_spreadsheet.py --config tutorial/config-haxby.example.json
```

Output:

```
Found 12 events file(s) under tutorial/haxby-data
Wrote 1452 rows to master_spreadsheet_haxby.csv
```

1452 = 12 runs x 121 volumes -- one row per BOLD volume of every run, no
exclusions, no `hemodynamic_lag` shift (see README.md section 3). Each
volume's `trial_type`/`onset`/`duration` come from whichever real event
covers it, including the `rest` gaps between blocks (blank `trial_type`
there, not dropped) and the individual 0.5s stimulus presentations within
each block (this pipeline links *events* to frames, not *blocks*, so several
consecutive volumes typically share the same category `trial_type` within a
block, rather than one row per block).

## Step 5: Validate the model config

```
python utils/validate_model_config.py --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv
```

Output (all 8 categories, training/testing/timecourse_decoding):

```
  [training] 'bottle': 90 rows       [testing] 'bottle': 29 rows       [timecourse_decoding] 'bottle': 29 rows
  [training] 'cat': 89 rows          [testing] 'cat': 30 rows          [timecourse_decoding] 'cat': 30 rows
  [training] 'chair': 88 rows        [testing] 'chair': 30 rows        [timecourse_decoding] 'chair': 30 rows
  [training] 'face': 89 rows         [testing] 'face': 30 rows         [timecourse_decoding] 'face': 30 rows
  [training] 'house': 89 rows        [testing] 'house': 29 rows        [timecourse_decoding] 'house': 29 rows
  [training] 'scissors': 90 rows     [testing] 'scissors': 30 rows     [timecourse_decoding] 'scissors': 30 rows
  [training] 'scrambledpix': 90 rows [testing] 'scrambledpix': 29 rows [timecourse_decoding] 'scrambledpix': 29 rows
  [training] 'shoe': 86 rows         [testing] 'shoe': 30 rows         [timecourse_decoding] 'shoe': 30 rows
  [timecourse_decoding] trial_start_event: 948 rows

0 error(s), 0 warning(s)
```

All three sections read the same `master_spreadsheet_haxby.csv` now, and
these counts are each category's real per-block volume count -- close to,
but not exactly, 108/36 for training/testing (9/3 runs x 12 stimuli) since
this is a direct count of real per-frame `trial_type` values, not the
`hemodynamic_lag`-shifted volume selection `mvpa_workflow.py` actually uses
for training/testing (see README.md section 4's `label_conditions_with_lag`)
-- a close, not exact, sanity-check proxy. `timecourse_decoding`'s own counts
happen to match `testing`'s here only because this config's `timecourse_decoding.conditions`
are byte-identical to `testing`'s (both real per-frame `trial_type` counts,
no lag either way).

## Step 6: Train and evaluate

```
python workflows/mvpa_workflow.py --subject 1 --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv --analysis-output-dir ./haxby_out
```

Since this config sets `model_conditions.testing`/`timecourse_decoding` but
not `model.kfold_cv`, `mvpa_workflow.py` fits one classifier on the full
9-run training set (`label_conditions_with_lag` selects its actual,
`hemodynamic_lag`-shifted volumes from `master_spreadsheet_haxby.csv`) and
writes only `test/` (the "Full" independent-test-set family) and `decoding/`
output for this subject -- no `model/` directory at all, since there's no
k-fold step configured here. Ran in ~20 seconds without `model.permutation_test`
configured (roughly +6 minutes with it, per the 1000-permutation run below).
This dataset's tiny volumes (40x64x64, ~23K-voxel mask) make it fast compared
to this repo's other, larger sample data.

## Results

Numbers below are from the current FSL-based `preprocess_haxby.sh`
(`mcflirt`/`flirt`/`applyxfm4D`/`fsl_glm`, including the in-mask `+10000`
offset -- see [Step 2](#step-2-basic-preprocessing)). Held-out test accuracy
(runs 10-12, never touched during training, written under `test/`): **0.524**
(chance = 0.125 for 8 balanced classes).

*(An earlier version of this walkthrough, run against a predecessor script
that computed a leave-one-run-out cross-validation diagnostic automatically
from `model_conditions.training` alone, also reported 9-fold internal-CV
accuracy of 0.405 across training runs 1-9. `mvpa_workflow.py` no longer
does this automatically -- add `model.kfold_cv: {"strategy": "per_run"}` to
reproduce the equivalent CV diagnostic explicitly, written under `model/`
and labeled "CV" in `generate_report.py`; see
[`config-kfold-haxby.example.json`](config-kfold-haxby.example.json) for a
complete example of that.)*

**Held-out confusion matrix** (the "Full" family, `test/`) -- rows = actual category, columns =
predicted, cells = proportion of that category's trials predicted as each
column:

| actual \ predicted | bottle | cat | chair | face | house | scissors | scrambledpix | shoe |
|---|---|---|---|---|---|---|---|---|
| bottle | **0.611** | 0.028 | 0.111 | 0.056 | 0.000 | 0.000 | 0.000 | 0.194 |
| cat | 0.028 | **0.472** | 0.028 | 0.111 | 0.028 | 0.278 | 0.000 | 0.056 |
| chair | 0.000 | 0.056 | 0.333 | 0.000 | 0.083 | 0.167 | 0.000 | 0.361 |
| face | 0.111 | 0.083 | 0.000 | **0.667** | 0.000 | 0.139 | 0.000 | 0.000 |
| house | 0.000 | 0.028 | 0.083 | 0.000 | **0.889** | 0.000 | 0.000 | 0.000 |
| scissors | 0.500 | 0.028 | 0.056 | 0.028 | 0.056 | 0.139 | 0.083 | 0.111 |
| scrambledpix | 0.028 | 0.000 | 0.000 | 0.083 | 0.056 | 0.056 | **0.639** | 0.139 |
| shoe | 0.167 | 0.056 | 0.111 | 0.083 | 0.056 | 0.028 | 0.056 | **0.444** |

**AUC per category (held-out):**

| bottle | cat | chair | face | house | scissors | scrambledpix | shoe |
|---|---|---|---|---|---|---|---|
| 0.812 | 0.752 | 0.793 | 0.932 | 0.965 | 0.627 | 0.955 | 0.799 |

(mean 0.829, matching `model_results_total_scores.csv`'s permutation-test
`roc_auc_ovr` real_score -- both now derive from the same normalized-
probability evidence, `decision_evidence()`'s `predict_proba()`/softmax,
where earlier versions of this table used an independent per-class sigmoid
that didn't sum to 1 and diverged slightly from that reference value.)

Held-out accuracy is well above the 12.5% chance level --
`face`, `house`, and `scrambledpix` are decoded almost perfectly (AUC
0.93-0.97), directionally consistent with the classic Haxby finding that
ventral temporal cortex carries distinguishable, distributed patterns for
these categories. `scissors` is the weakest category (AUC 0.63), plausibly
confusable with `chair`/other elongated-object categories (see the
confusion matrix's `bottle`<->`scissors`/`chair`<->`shoe` cross-talk) in a
whole-brain mask this crude. `permutation_test` (README.md section 5,
1000 permutations) confirms both accuracy and AUC are significant at
p < 0.001 -- see [`generate_report.py`](../README.md#7-generating-a-report-workflowsgenerate_reportpy)
or `test/1_permutation_test.csv` for the full numbers if you run it
yourself.

## Where this tutorial oversimplifies

This demonstrates that the pipeline runs correctly end-to-end on real,
external data (with a real preprocessing step) and produces a real (not
spurious) signal -- it is **not** a rigorous reanalysis of this dataset, and
the specific numbers above shouldn't be treated as a proper replication.
Concretely, relative to how this data would normally be analyzed:

- **Preprocessing is minimal, not a real pipeline.** `preprocess_haxby.sh`
  does exactly three things (via `mcflirt`/`flirt`/`applyxfm4D`/`fsl_glm`) --
  motion correction, rigid coregistration to run 1, linear detrending (plus a
  constant in-mask offset to keep intensities positive) -- and nothing else:
  - **No slice-timing correction.**
  - **Rigid-body only.** Coregistration to run 1 assumes a rigid (6 dof)
    transform is sufficient (no affine/nonlinear warp), and run 1 itself is
    an arbitrary native-space reference -- not a template (e.g. MNI) -- so
    results are not in any standardized space and can't be directly compared
    across subjects.
  - **Default FSL settings throughout** (`mcflirt`'s and `flirt`'s defaults,
    no custom search/cost-function tuning) purely for tutorial simplicity. A
    real pipeline would still want to inspect registration quality (e.g. via
    `slicesdir`) rather than assume it converged.
  - **Coregistration is estimated once per run** (each run's mean volume,
    post-`mcflirt`, to the template), then applied to every volume of that
    run via a single matrix -- if within-run motion is large relative to the
    across-run misalignment, this single per-run estimate may not represent
    the whole run's true alignment to the template equally well throughout.
  - **No confound regression** (motion parameters, physiological noise), no
    smoothing, no high-pass filtering beyond the linear detrend (which
    removes only a straight-line trend, not slower nonlinear drift).
- **A crude, non-anatomical mask.** The mask is computed via FSL's `bet`
  (default settings) on the across-run average mean image -- standard brain
  extraction, but not a grey-matter segmentation or an anatomically-defined
  ROI. The original Haxby et al. paper's classic analyses used a hand-defined
  ventral temporal cortex mask; this tutorial's mask has no anatomical
  specificity.
- **Per-event, not per-block, windowing.** Each of the 12 individual 0.5s
  stimulus presentations in a block is treated as its own event (own
  `trial_index`, own ~1-volume window), rather than modeling/averaging each
  ~24s block as a single trial the way many classic Haxby-dataset tutorials
  do. This is a finer-grained (and noisier) sampling of the same signal.
- **`hemodynamic_lag` (4.0s) is a generic estimate**, not fit or validated
  for this subject or dataset.
- **Single subject, one arbitrary train/test split.** Runs 1-9 vs. 10-12 was
  picked for simplicity, not counterbalanced or cross-validated at the
  block/run-order level, and no claim is made that this generalizes to the
  other 5 subjects in the dataset.

### If you wanted this to be a real analysis

Preprocess with fMRIPrep or similar comprehsnive preprocessing workflow (motion correction, slice-timing correction,
registration to a common per-subject reference *and* a standard template,
confound outputs), use an anatomically informed mask (e.g. a grey-matter or
ventral-temporal ROI transformed into each run's native space), and validate
`hemodynamic_lag` and the event/window scheme against the literature or your
own HRF estimates before trusting the resulting numbers scientifically.
