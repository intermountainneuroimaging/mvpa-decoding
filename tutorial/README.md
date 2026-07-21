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
session labels are optional for single-session studies. Getting this dataset
working is what surfaced a real bug: `generate_master_spreadsheet.py` used to
*require* `ses` and would have silently skipped every file here. That's now
fixed -- a missing `ses` just means `session` is absent from that row instead
of the file being rejected.

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

`config-haxby.example.json` differs from this repo's other examples in a few
ways, driven by the dataset itself rather than by choice:

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
- **8 conditions, not 2.** One per object category. This is also a real test
  of the classifier code path for >2 classes.

`event_extraction.hemodynamic_lag` is set to 4.0s (a generic HRF-peak
estimate, not tuned for this subject/dataset). `mask.mask_pattern` is
`sub-{subject}/masks/native_epi_mask.nii.gz`, resolved relative to
`derivatives_root` (masks default there) -- no `{session}` in the pattern,
matching the session-less data.

## Step 4: Build the volume table

```
python generate_master_spreadsheet.py --config tutorial/config-haxby.example.json
```

Output:

```
Found 12 events file(s) under tutorial/haxby-data
Wrote 1152 rows to master_spreadsheet_haxby.csv
```

1152 = 12 runs x 8 categories x 12 stimuli/block. Each individual 0.5s
stimulus presentation maps to essentially one BOLD volume (TR=2.5s, so a
0.5s window plus the lag rounds to a single frame) -- this pipeline links
*events* to frames, not *blocks*, so each of the 12 stimulus presentations in
a block is its own row with the same `trial_type`, not one row per block.

## Step 5: Validate the model config

```
python validate_model_config.py --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv
```

Output (all 8 categories, training/testing/timecourse_decoding):

```
  [training] 'bottle': 108 rows      [testing] 'bottle': 36 rows      [timecourse_decoding] 'bottle': 36 rows
  [training] 'cat': 108 rows         [testing] 'cat': 36 rows         [timecourse_decoding] 'cat': 36 rows
  [training] 'chair': 108 rows       [testing] 'chair': 36 rows       [timecourse_decoding] 'chair': 36 rows
  [training] 'face': 108 rows        [testing] 'face': 36 rows        [timecourse_decoding] 'face': 36 rows
  [training] 'house': 108 rows       [testing] 'house': 36 rows       [timecourse_decoding] 'house': 36 rows
  [training] 'scissors': 108 rows    [testing] 'scissors': 36 rows    [timecourse_decoding] 'scissors': 36 rows
  [training] 'scrambledpix': 108 rows [testing] 'scrambledpix': 36 rows [timecourse_decoding] 'scrambledpix': 36 rows
  [training] 'shoe': 108 rows        [testing] 'shoe': 36 rows        [timecourse_decoding] 'shoe': 36 rows

0 error(s), 0 warning(s)
```

Perfectly balanced (9 runs x 12 stimuli = 108 for training; 3 runs x 12 = 36
for testing/decoding), across all 8 categories -- confirms the run-based
train/test split lines up cleanly with the data, and that `boldfile` rows
correctly resolved to the preprocessed derivatives.

## Step 6: Train and evaluate

```
python mvpa_workflow.py --subject 1 --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv --analysis-output-dir ./haxby_out
```

Ran in ~23 seconds (9-fold CV + final model + timecourse decoding). This
dataset's tiny volumes (40x64x64, ~23K-voxel mask) make it fast compared to
this repo's other, larger sample data.

## Results

**Cross-validated (9-fold, leave-one-run-out across training runs 1-9) confusion matrix** --
rows = actual category, columns = predicted, cells = proportion of that
category's trials predicted as each column:

| actual \ predicted | bottle | cat | chair | face | house | scissors | scrambledpix | shoe |
|---|---|---|---|---|---|---|---|---|
| bottle | 0.269 | 0.009 | 0.204 | 0.130 | 0.000 | 0.167 | 0.065 | 0.157 |
| cat | 0.074 | 0.398 | 0.148 | 0.148 | 0.056 | 0.111 | 0.009 | 0.056 |
| chair | 0.167 | 0.065 | 0.343 | 0.056 | 0.028 | 0.167 | 0.037 | 0.139 |
| face | 0.148 | 0.120 | 0.028 | **0.639** | 0.000 | 0.000 | 0.019 | 0.046 |
| house | 0.102 | 0.046 | 0.102 | 0.000 | **0.667** | 0.037 | 0.037 | 0.009 |
| scissors | 0.241 | 0.074 | 0.148 | 0.037 | 0.009 | 0.269 | 0.093 | 0.130 |
| scrambledpix | 0.130 | 0.019 | 0.065 | 0.028 | 0.111 | 0.019 | **0.620** | 0.009 |
| shoe | 0.148 | 0.074 | 0.056 | 0.019 | 0.028 | 0.093 | 0.000 | **0.583** |

Overall CV accuracy: **0.473** (chance = 0.125 for 8 balanced classes) --
up from 0.363 on raw, unpreprocessed data.

**Held-out test (runs 10-12, never touched during training or CV):**

| actual \ predicted | bottle | cat | chair | face | house | scissors | scrambledpix | shoe |
|---|---|---|---|---|---|---|---|---|
| bottle | **0.528** | 0.028 | 0.111 | 0.111 | 0.000 | 0.000 | 0.000 | 0.222 |
| cat | 0.056 | 0.389 | 0.000 | 0.167 | 0.028 | 0.278 | 0.028 | 0.056 |
| chair | 0.028 | 0.083 | 0.333 | 0.000 | 0.056 | 0.111 | 0.000 | 0.389 |
| face | 0.028 | 0.000 | 0.000 | **0.889** | 0.000 | 0.083 | 0.000 | 0.000 |
| house | 0.000 | 0.028 | 0.083 | 0.000 | **0.889** | 0.000 | 0.000 | 0.000 |
| scissors | 0.389 | 0.000 | 0.000 | 0.000 | 0.056 | 0.278 | 0.139 | 0.139 |
| scrambledpix | 0.028 | 0.028 | 0.028 | 0.056 | 0.028 | 0.083 | **0.667** | 0.083 |
| shoe | 0.194 | 0.000 | 0.028 | 0.111 | 0.000 | 0.083 | 0.056 | **0.528** |

Overall held-out accuracy: **0.563** -- up from 0.306 on raw, unpreprocessed
data.

**AUC per category (held-out):**

| bottle | cat | chair | face | house | scissors | scrambledpix | shoe |
|---|---|---|---|---|---|---|---|
| 0.82 | 0.78 | 0.85 | 0.97 | 0.98 | 0.68 | 0.96 | 0.80 |

Both CV and held-out accuracy are well above the 12.5% chance level, and
substantially higher than the raw-data run of this same tutorial (CV 0.363,
held-out 0.306, AUC 0.43-0.84) -- `face`, `house`, and `scrambledpix` are now
decoded almost perfectly (AUC 0.96-0.98), directionally consistent with the
classic Haxby finding that ventral temporal cortex carries distinguishable,
distributed patterns for these categories. `scissors` remains the weakest
category (AUC 0.68) even after preprocessing, and is plausibly confusable
with `chair`/other elongated-object categories in a mask this crude.

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

Preprocess with fMRIPrep (motion correction, slice-timing correction,
registration to a common per-subject reference *and* a standard template,
confound outputs), use an anatomically informed mask (e.g. a grey-matter or
ventral-temporal ROI transformed into each run's native space), and validate
`hemodynamic_lag` and the event/window scheme against the literature or your
own HRF estimates before trusting the resulting numbers scientifically.
