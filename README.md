# MVPA workflow

This covers the whole pipeline: turning raw BIDS-style events files into a
single searchable `master_spreadsheet.csv`, defining/validating which rows of
that table count as which MVPA classification condition, and actually
training/cross-validating a classifier and running timecourse decoding.

See [THEORY.md](THEORY.md) for the scientific background and use case this
pipeline replicates (Kim et al., 2020, *Nature Communications*) and how each
config section/output maps back to that paper's analyses.

One JSON config, three top-level sections, three scripts:

| Stage | Script | Reads section |
|---|---|---|
| 1. Build the volume table | `workflows/generate_master_spreadsheet.py` | `event_extraction` (+ optional `expected_events.json`) |
| 2. Define & validate MVPA conditions | `utils/validate_model_config.py` | `model_conditions` |
| 3. Train/decode | `workflows/mvpa_generalization_workflow.py` | `model` (+ `model_conditions` to select/label rows) |

Step 3 has two interchangeable scripts, same config format:
`workflows/mvpa_generalization_workflow.py` for independent train/test data
(possibly different tasks entirely -- the default, described first) or
`workflows/mvpa_kfold_workflow.py` for same-task data split into folds by
`run` (section 6) -- pick whichever matches your design, not both. All
scripts take the **same** config file via `--config`. Shared logic -- BIDS
filename parsing, the query DSL, window math, and (for the two train/decode
scripts) the actual classification/decoding primitives -- lives in
`utils/mvpa_common.py`, imported by all of them. Everything below is grounded in
`tutorial/config-haxby.example.json`, a complete config that runs end-to-end
against `tutorial/haxby-data/` (Haxby et al. 2001 / OpenNeuro ds000105 --
downloaded fresh by `tutorial/preprocess_haxby.sh`, not checked into this
repo -- see [tutorial/README.md](tutorial/README.md) for the full
walkthrough). `examples/config-generalization-template.example.json` is the
same shape but written as a fill-in-your-own-paths template, including the
derivative-data field (`derivatives_root`) described below.

## Running tests

```
pip install -r requirements-dev.txt
pytest
```

The suite (`tests/`) uses synthetic fixtures only -- no dependency on the
gitignored real data under `examples/sample-data/` or `tutorial/haxby-data/`
-- so it runs the same locally and in CI (`.github/workflows/tests.yml`,
which runs on every push/PR to `main`).

## 1. What input data is assumed

You need a directory tree containing, for every scan run you want in the table:

- **An events file**: tab-separated `.tsv` with at least `onset`, `duration`,
  `trial_type` columns (standard BIDS events file). Its **filename** must
  contain BIDS key-value entities `sub-`, `task-`, `run-` somewhere in it (in
  any order, with any other entities mixed in between) -- e.g.
  `sub-1_task-objectviewing_run-01_events.tsv`. `ses-` is optional -- include
  it for a multi-session dataset, omit it entirely for a single-session one
  (like the tutorial data below); either way it's inferred, never configured.
- **A matching BOLD file**: a `.nii.gz` whose filename contains the word
  `bold` plus the *same* `sub-`/`task-`/`run-` (and `ses-`, if present) values
  as the events file, e.g. `sub-1_task-objectviewing_run-01_bold.nii.gz`.
  There must be **exactly one** such match per events file -- zero or
  multiple matches cause that events file to be skipped with a warning, not
  a crash.

Example tree (trimmed from `tutorial/haxby-data/`, real files this repo's
tutorial runs against -- see [tutorial/README.md](tutorial/README.md) for
how to download it):

```
tutorial/haxby-data/
└── sub-1/
    └── func/
        ├── sub-1_task-objectviewing_run-01_events.tsv
        ├── sub-1_task-objectviewing_run-01_bold.nii.gz
        ├── sub-1_task-objectviewing_run-02_events.tsv
        └── sub-1_task-objectviewing_run-02_bold.nii.gz
```

This assumes events and BOLD files are co-located and share a naming
convention. If your preprocessed data lives elsewhere (a separate 
`derivatives/` tree, a different naming scheme, etc.), see
[Using preprocessed/derivative data](#using-preprocessedderivative-data-eg-fmriprep) below --
`derivatives_root`/`bold_glob` to decouple BOLD-file discovery from this assumption
entirely.

**Inferred from the data, never configured:**
- `subject`, `session`, `task`, `run` -- parsed out of the events filename
  (`session` is `""` when there's no `ses-` entity, as in the tree above).
- Any *other* BIDS entity in the events filename (e.g. `dir-pa`) -- captured
  automatically as its own extra column, named after the entity key. Different
  designs can carry different entities; whatever shows up, shows up as a column.
- **TR** and **frame count** -- read directly from the matched BOLD file's
  NIfTI header (`get_zooms()[3]` and `get_data_shape()[-1]`), never from a
  config value. If a run's BOLD file is missing, that run cannot be processed
  at all (no way to know its TR), so it's skipped.

## 2. The config file

One JSON file with three top-level sections. `tutorial/config-haxby.example.json`
is a complete, runnable example (`examples/config-generalization-template.example.json` is the same
shape as a fill-in-your-own-paths template):

```json
{
  "config_version": "1.0",
  "created_by": "AKH",
  "notes": "Haxby et al. 2001 (OpenNeuro ds000105) 8-way object category classifier",

  "event_extraction": { "...": "see section 3" },
  "model_conditions": { "...": "see section 4" },
  "model": { "...": "see section 5" }
}
```

## 3. `event_extraction`

Read by `generate_master_spreadsheet.py`.

```json
"event_extraction": {
  "bids_root": "tutorial/haxby-data",
  "events_glob": "**/*_events.tsv",
  "hemodynamic_lag": 4.0,
  "output_file": "master_spreadsheet_haxby.csv",
  "expected_events_file": "tutorial/expected_events_haxby.example.json"
}
```

| Field | Meaning |
|---|---|
| `bids_root` | Directory to search under for events.tsv files. |
| `events_glob` | Glob (supports `**`) used to find events.tsv files under `bids_root`. |
| `hemodynamic_lag` | Seconds added to every event's `onset` before converting to volume indices. Override per-run with `--hemodynamic-lag`. |
| `output_file` | Where the resulting table is written. Override with `--output`. |
| `expected_events_file` | *(optional)* Path to a template of expected `trial_type` values -- see below. Override with `--expected-events`. |
| `derivatives_root` | *(optional)* Directory to search under for BOLD files, if different from `bids_root` -- e.g. a separate fMRIPrep `derivatives/` tree. Omit the key (or set it to `null`) to inherit `bids_root`. Setting it to `""` is **not** the same as omitting it -- an explicit empty string is honored literally (resolves to the current working directory) and prints a warning, since that's almost never what's intended. See [Using preprocessed/derivative data](#using-preprocessedderivative-data-eg-fmriprep). |
| `bold_glob` | *(optional)* Needed whenever BOLD filenames don't follow the default lookup (match on `sub`/`ses`/`task`/`run` tokens + `"bold"` in the filename) -- e.g. fMRIPrep's `desc-`/`space-` suffixes, or when `derivatives_root` returns more than one match per run. A format string resolved relative to `derivatives_root`, with `{subject}`/`{session}`/`{task}`/`{run}` placeholders -- plus **any other BIDS entity found in the events filename is available under its own raw key**, e.g. `{dir}` for a `dir-pa`/`dir-ap` entity, `{acq}` for `acq-*`, etc. No code change needed for a new entity; if it's in the filename, it's usable in `bold_glob`. |

### `expected_events.json` (optional, separate file)

A flat JSON list of every `trial_type` value you expect to see *somewhere*
across the whole dataset (no single run needs to contain all of them). After
building the table, the script diffs this list against what was actually
observed and prints warnings for both directions -- values you expected but
never saw, and values you saw but didn't expect (typos, unlisted new
conditions) -- exactly the kind of stray-space or misspelled `trial_type`
that's easy to miss by eye across a dozen events.tsv files but shows up
immediately as an unexpected value here.

```json
[
  "bottle",
  "cat",
  "chair",
  "face",
  "house",
  "scissors",
  "scrambledpix",
  "shoe"
]
```

### Running it

```
python workflows/generate_master_spreadsheet.py --config tutorial/config-haxby.example.json
```

Output (`master_spreadsheet_haxby.csv`) -- one row per BOLD volume that overlapped
an event's active window:

| Column | Meaning |
|---|---|
| `subject`, `session`, `task`, `run` | *Inferred* from the events filename. |
| `volume_of_interest` | *Computed*: the BOLD frame index, from `onset + hemodynamic_lag` through `onset + hemodynamic_lag + duration`, using the BOLD file's own TR, clipped to its frame count. |
| `trial_type` | Verbatim from the events file -- never reinterpreted, split, or renamed. |
| `trial_index` | *Computed*: 1-based sequential index (in onset order) among this run's *retained* events -- i.e. after the hardcoded exclusions below, so it's always contiguous. Identifies "which event produced this volume," used by `mvpa_generalization_workflow.py` for trial-balancing and for recomputing `timecourse_decoding`'s window. |
| `onset`, `duration` | Verbatim from the events file, repeated across every volume belonging to that event. |
| `boldfile`, `eventfile` | Resolved source file paths, for traceability/sorting. |
| *(varies)* | Any other BIDS entity found in the filename, e.g. `dir` -- *inferred*, present only if that entity appears in your filenames (the tutorial data has none). |

Example real output row (from `tutorial/haxby-data`; `session` is empty since
this dataset has no `ses-` entity):

```
subject  session  volume_of_interest  trial_type  trial_index  onset  duration  task           run  boldfile                                                                                eventfile
1                 65                  house       51           160.0  0.5       objectviewing  1    tutorial/haxby-data/derivatives/sub-1/func/..._run-01_desc-preproc_bold.nii.gz  tutorial/haxby-data/sub-1/func/..._run-01_events.tsv
```

### Hardcoded exclusions

`generate_master_spreadsheet.py` drops a fixed set of administrative/non-trial
`trial_type` values before windowing -- typically not used in the MVPA analyses therefore it  isn't exposed as a config option. Edit the `EXCLUDED_TRIAL_TYPE_EXACT`
/ `EXCLUDED_TRIAL_TYPE_SUBSTRINGS` constants near the top of the script to
change the list:

| Match | Excludes |
|---|---|
| exact: `start_block`, `end_block` | structural block markers |
| substring (case-insensitive): `fixation` | `trial_fixation`, `BaselineFixation`, `EndFixation`, etc. |
| substring (case-insensitive): `postrt` | post-response-time administrative events |

`rest_block` is **not** excluded -- it's a real experimental condition in some
designs, not a structural marker. Exclusions (and invalid-duration rows) are
dropped *before* `trial_index` is assigned, so `trial_index` is always a
contiguous `1..N` over exactly the events that end up in the output table --
not the row's raw position in the source events.tsv, which would otherwise
leave gaps wherever an excluded row used to sit.

### Using preprocessed/derivative data (e.g. fMRIPrep)

By default, BOLD files are searched for under `bids_root` -- fine when raw
events.tsv and preprocessed BOLD data live side by side. That's often not the
case: fMRIPrep (and most BIDS derivative pipelines) write outputs to a
separate `derivatives/` tree with its own naming convention (`space-`,
`desc-preproc`, etc.), sometimes on a different disk or mount entirely.

Two config fields decouple BOLD-file discovery from the events-file layout.
`tutorial/config-haxby.example.json` actually needs this -- raw events.tsv
files live under `tutorial/haxby-data/`, but the (minimally) preprocessed
BOLD data `tutorial/preprocess_haxby.sh` writes lives in its own
`derivatives/` subfolder with a `desc-preproc` suffix:

```json
"event_extraction": {
  "bids_root": "tutorial/haxby-data",
  "derivatives_root": "tutorial/haxby-data/derivatives",
  "bold_glob": "sub-{subject}/func/sub-{subject}_task-{task}_run-{run}_desc-preproc_bold.nii.gz",
  "events_glob": "**/*_events.tsv"
}
```

- `derivatives_root` -- where to search for BOLD files. Defaults to `bids_root` if
  omitted, so this is fully backward compatible.
- `bold_glob` -- resolved relative to `derivatives_root` (not `bids_root`) once
  `derivatives_root` is set. Use it whenever the default lookup (match on
  `sub`/`ses`/`task`/`run` tokens + `"bold"` in the filename) would either miss
  the file or return more than one match (e.g. multiple `space-*` variants of
  the same run) -- both cause that run to be skipped with a warning, not a
  crash.

`model.mask.mask_pattern` (see [section 5](#5-model)) is unrelated to either
of these -- it's always a full path in its own right, not resolved against
`bids_root`/`derivatives_root`. `examples/config-generalization-template.example.json`
is a template showing the `event_extraction` fields above filled in.

#### Diagnosing a "no matching BOLD file found" error

Getting `derivatives_root`/`bold_glob` right on a real dataset is fiddly, so
this failure prints real diagnostics, not just "not found": the parsed
`sub`/`task`/`run` entities, the exact `bold_glob` template *and* what it
formatted to *and* the full path actually searched (or, with no `bold_glob`,
how many `.nii.gz` files were scanned and what tokens they were checked
against), plus -- if nothing matched -- a listing of whatever `.nii.gz` files
under `derivatives_root` *do* contain that subject ID, so you can compare
their real naming against your `bold_glob`. If *nothing* contains the
subject ID at all, `derivatives_root` itself is almost certainly wrong.

Pass `--verbose` to `generate_master_spreadsheet.py` to print this same
search detail for every events file, not just the ones that fail -- useful
to confirm resolution is doing what you expect even when it "works".

## 4. `model_conditions`

Read by `validate_model_config.py`. This defines, for each of three
sections (`training`, `testing` -- both required -- and `timecourse_decoding`,
optional), a set of named **conditions** -- the classifier's class labels --
each backed by a **query** that selects which `master_spreadsheet.csv` rows
belong to it.

### The query language

A query is a small recursive boolean tree over *any* column of
`master_spreadsheet.csv` (`trial_type`, `task`, `run`, `subject`, ...):

```json
{"column": "trial_type", "match": "exact", "value": "face"}
{"column": "run", "match": "in", "values": ["10", "11", "12"]}
{"column": "trial_type", "match": "regex", "value": ".*face.*"}
{"and": [<query>, <query>, ...]}
{"or":  [<query>, <query>, ...]}
{"not": <query>}
```

- `exact`/`in` compare the column's string value directly.
- `regex` uses `re.fullmatch` against the whole value (not a partial search).
- `and`/`or`/`not` nest arbitrarily, so you can combine column filters however
  you need (e.g. "this task AND this trial_type, but NOT that specific value").

### Section by section

**`training`** -- rows used to fit the classifier. In the example, the first
9 of Haxby's 12 runs, split into its 8 object categories by an exact match on
`trial_type` (2 of 8 shown):

```json
"model_conditions": {
  "training": {
    "conditions": {
      "face":  {"and": [{"column": "trial_type", "match": "exact", "value": "face"},
                         {"column": "run", "match": "in", "values": ["1","2","3","4","5","6","7","8","9"]}]},
      "house": {"and": [{"column": "trial_type", "match": "exact", "value": "house"},
                         {"column": "run", "match": "in", "values": ["1","2","3","4","5","6","7","8","9"]}]}
    }
  }
}
```

**`testing`** -- held-out rows used to score the trained classifier. In the
example, the remaining 3 runs (10-12) -- same task, same categories, held out
by run rather than by a different task. That's what this particular example
happens to do; `training`/`testing` can just as easily reference genuinely
different tasks (e.g. train on a localizer, test on a separate main-task
run) -- the query language doesn't care which, `task` is just another
column:

```json
"testing": {
  "conditions": {
    "face":  {"and": [{"column": "trial_type", "match": "exact", "value": "face"},
                       {"column": "run", "match": "in", "values": ["10","11","12"]}]},
    "house": {"and": [{"column": "trial_type", "match": "exact", "value": "house"},
                       {"column": "run", "match": "in", "values": ["10","11","12"]}]}
  }
}
```

**`timecourse_decoding`** -- *(optional)* same idea, but for the trial-by-trial
decoding sweep. Here the example also adds a required **`window`**. Omit the
whole section (not just leave it empty) to skip timecourse decoding
entirely -- no `decoding/` output files (per-subject or per-fold), no extra
runtime for that step, and `generate_report.py`'s timecourse page is
automatically skipped too (it already skips whenever it finds no
`decoding_results.csv` for any subject in scope, so there's nothing extra to
configure on the report side):

```json
"timecourse_decoding": {
  "conditions": {
    "face":  {"and": [{"column": "trial_type", "match": "exact", "value": "face"},
                       {"column": "run", "match": "in", "values": ["10","11","12"]}]},
    "house": {"and": [{"column": "trial_type", "match": "exact", "value": "house"},
                       {"column": "run", "match": "in", "values": ["10","11","12"]}]}
  },
  "window": {
    "start": {"reference": "onset", "offset_seconds": 0},
    "end": {"reference": "offset_end", "offset_seconds": 10}
  }
}
```

`window` describes the decode window around each matched event, **independent
of `hemodynamic_lag`** used when the table was built -- `reference` is
`"onset"` (the event's own onset) or `"offset_end"` (`onset + duration`), and
`offset_seconds` shifts that reference point (can be negative). The example
above reads as "decode from stimulus onset, with no lag, through 10 seconds
past the event's end."

**`overlay`** -- *(optional)* only read by `generate_report.py`, not by
either workflow script. Same name-to-query shape as `conditions`, but for a
category that's independent of the classifier's own conditions. Haxby's raw
events don't carry a secondary factor the way some designs do (e.g. a
manipulation embedded in `trial_type` alongside the category itself) -- so
as a syntax illustration, this example overlays each category's evidence
curve by *which* testing run a trial came from, using `run` (any column
works, not just ones that look condition-like):

```json
"overlay": {
  "runs_10_11": {"column": "run", "match": "in", "values": ["10", "11"]},
  "run_12":     {"column": "run", "match": "exact", "value": "12"}
}
```

When present, the timecourse page (section 7) overlays one colored line per
overlay category within each existing subplot, instead of a single line --
see that section for how it changes the plot. Rows matching none of the
overlay queries are dropped from that plot only (a count is printed);
everything else about the pipeline -- the classifier, its evidence values,
`decoding_results.csv`/`summary_decoding_results.csv` -- is unaffected,
since `overlay` is evaluated entirely inside `generate_report.py` against
data the workflow scripts already wrote.

### Running it

```
python utils/validate_model_config.py --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv
```

Without `--master-spreadsheet`, only the JSON structure is checked (valid
`match` types, regexes that actually compile, `window` well-formed, etc.).
With it, every condition's query is run against the real table and you
additionally get:

- **Error** if a condition matches 0 rows (dead query -- likely a typo or a
  task/trial_type that doesn't exist in this dataset).
- **Warning** if two conditions in the same section overlap on any row
  (ambiguous label -- the same volume would count as two classes).
- **Warning** if the condition *names* differ between sections (training
  should generally define the same classes as testing/decoding).

Example output against `tutorial/haxby-data` (all 8 categories shown):

```
Validating tutorial/config-haxby.example.json against master_spreadsheet_haxby.csv
  [training] 'bottle': 108 rows
  [training] 'cat': 108 rows
  [training] 'chair': 108 rows
  [training] 'face': 108 rows
  [training] 'house': 108 rows
  [training] 'scissors': 108 rows
  [training] 'scrambledpix': 108 rows
  [training] 'shoe': 108 rows
  [testing] 'bottle': 36 rows
  [testing] 'cat': 36 rows
  [testing] 'chair': 36 rows
  [testing] 'face': 36 rows
  [testing] 'house': 36 rows
  [testing] 'scissors': 36 rows
  [testing] 'scrambledpix': 36 rows
  [testing] 'shoe': 36 rows
  [timecourse_decoding] 'bottle': 36 rows
  [timecourse_decoding] 'cat': 36 rows
  [timecourse_decoding] 'chair': 36 rows
  [timecourse_decoding] 'face': 36 rows
  [timecourse_decoding] 'house': 36 rows
  [timecourse_decoding] 'scissors': 36 rows
  [timecourse_decoding] 'scrambledpix': 36 rows
  [timecourse_decoding] 'shoe': 36 rows

0 error(s), 0 warning(s)
```

## 5. `model`

Read by `mvpa_generalization_workflow.py`. Everything the analysis itself needs that isn't
about *which rows* to use (that's `model_conditions`'s job):

```json
"model": {
  "desc": "haxby_object_classifier",
  "mask": {
    "mask_pattern": "tutorial/haxby-data/derivatives/sub-{subject}/masks/native_epi_mask.nii.gz"
  },
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
| `mask.mask_pattern` | The full path to the mask NIfTI -- absolute, or relative to wherever the workflow script is run from (same convention `bids_root`/`derivatives_root` use); never resolved against either of those or any other root. Include `{subject}`/`{session}` placeholders (filled in from whichever row is being loaded) for one native-space mask per subject, as in the example above -- or omit them entirely for a single shared mask used for every subject, e.g. one MNI-space group mask (a template with no placeholders just formats to itself, so every subject resolves to the same literal path). Can still contain glob wildcards either way -- resolved the same way as bold-file lookups. |
| `featureSelection.feat_p` | ANOVA p-value threshold -- voxels with `p < feat_p` are kept, widened automatically until at least 5 voxels are selected. Ignored when `n_voxels` is set. |
| `featureSelection.n_voxels` | *(optional)* Select exactly this many voxels by ANOVA F-score instead, regardless of significance (sklearn's `SelectKBest` equivalent) -- takes priority over `feat_p` when both are present. Useful for keeping feature count fixed across subjects/folds whose signal strength (and thus a p-value threshold's actual voxel count) varies. `model_results_auc.csv`-adjacent output files still record whichever mode was actually used: `threshold_p` is `NaN` in this mode, since there's no threshold, but `selected_voxels` (identical to `n_voxels` here) is populated either way. |
| `classifier` | Any importable scikit-learn-style estimator: `name` is a dotted import path, `params` are passed straight through as kwargs. |

Omit either of `featureSelection`/`classifier` and it falls back to a
default (ANOVA @ p<0.05, `LogisticRegression`) -- only `desc` and `mask` are
meaningfully required.

There's no `model.cv` field -- it's not a configurable knob, it's determined
automatically from the training data itself (`resolve_internal_cv_folds` in
`mvpa_generalization_workflow.py`):

- **Leave-one-run-out** (`PredefinedSplit` on the `run` column, one fold per
  distinct run) when every training run contains the same set of conditions
  -- i.e. every run is a full replicate of the training task. This is the
  default case, and matches the leave-one-run-out scheme in the paper this
  pipeline replicates (see [THEORY.md](THEORY.md)).
- **4-fold stratified CV over trials pooled across all runs** (not scoped to
  any one run) when runs *don't* all share the same conditions -- holding
  out a whole run in that case would risk silently dropping a condition from
  one side of a fold entirely, so fold membership is built directly from
  trials instead. Each condition's trials are gathered from every run
  together and partitioned into 4 folds independently (`StratifiedKFold`),
  grouped by `(run, trial_index)` so every volume belonging to one event
  stays on the same side of its fold -- row-level splitting would let
  correlated volumes from the same trial leak across train/test. If the
  rarest condition has fewer than 4 trials, the fold count is reduced
  automatically (down to a minimum of 2) so every fold still gets at least
  one trial of every condition; fewer than 2 trials for the rarest condition
  is a hard error. A warning is printed when this fallback triggers, and
  each fold's train/test row counts and held-out trials are logged as
  they're built.

Either way this only ever touches `model_conditions.training` data -- it
never looks at `testing`/`timecourse_decoding` rows. It's a sanity-check
diagnostic (does the classifier find real signal within its own training
task at all?), separate from and complementary to the actual held-out test
against `testing_conditions` (does that pattern generalize to new/different
data?) -- `generate_report.py`'s accuracy/AUC page (section 7) plots both
side by side as "internal CV (training)" vs. "held-out test". (For same-task
data where you want to control fold membership yourself for the *actual*
evaluation, not just this internal diagnostic, that's `model.kfold_cv` on
`mvpa_kfold_workflow.py` instead -- see section 6.)

### `permutation_test` (optional): significance testing for the held-out result

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
file, today's exact behavior. Present (even as `{}`) runs it, with
`n_permutations` defaulting to 1000 and `random_state` to 0 if unset.

For each of `accuracy` and `roc_auc_ovr`, the training/testing split is
encoded as a single fixed `PredefinedSplit` fold, then
`permutation_test_score` repeatedly reshuffles the combined label vector and
refits the *entire* pipeline (feature selection + classifier, both) on each
shuffle -- the textbook-correct way to build a null distribution for a fixed
train/test split, not a naive shuffle done outside the fit structure. Writes
`model/{subject}_permutation_test.csv` (`metric,real_score,p_value,n_permutations`).

`accuracy`'s `real_score` matches `model_results_total_scores.csv` exactly
(same split, same fixed feature-selection threshold, same classifier
config). `roc_auc_ovr`'s `real_score` will be *close to but not exactly*
the mean of `model_results_auc.csv`'s per-category values -- sklearn's
`roc_auc_ovr` scorer uses the classifier's `predict_proba()` (softmax
across classes) as evidence, while `model_performance`'s own per-category
AUC uses an independent sigmoid of `decision_function` per class (not
softmax-normalized) -- both are legitimate one-vs-rest AUC computations,
they just start from different per-class evidence, so don't expect
bit-identical numbers between the two files for this metric. `p_value` is
the fraction of permuted-label refits that scored as well or better than
the real fit.

This costs `n_permutations` extra fits (parallelized across cores via
`n_jobs=-1`) on top of the one real fit -- cheap relative to a single
subject's BOLD loading time in practice, but scales with `n_permutations`,
so drop it for quick iteration and turn it on for a result you're about to
report.

`mvpa_generalization_workflow.py` is dedicated to this independent-train/independent-test
case (training and testing come from separate `model_conditions` sections,
possibly different tasks entirely). For same-task data split into
training/testing by `run` per fold, see `mvpa_kfold_workflow.py` and
`model.kfold_cv` below instead -- a separate script, not a mode switch on
this one.

## Running `workflows/mvpa_generalization_workflow.py`

```
python workflows/mvpa_generalization_workflow.py --subject 1 --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv --analysis-output-dir ./out
```

There's no separate `inputs.json`/`--input-scaffold` anymore -- everything
comes from the one config plus `master_spreadsheet.csv`. For a given
`--subject`, the script:

1. Filters `master_spreadsheet.csv` to that subject and writes a **trial
   pivot table** (see below) -- a sanity check, computed before any
   condition filtering.
2. Evaluates `model_conditions.training`/`testing`'s queries to label and
   select rows (a row matching more than one condition takes the first
   match, in the order conditions are listed -- `validate_model_config.py`
   already warns about that case).
3. Loads BOLD patterns directly from each row's `boldfile` (already a
   concrete, resolved path -- no glob/pattern matching needed at this stage),
   z-scores, and slices to `volume_of_interest`.
4. Cross-validates (`GroupKFold` on `run`) and trains a final classifier,
   writing accuracy/evidence/AUC and importance-map NIfTIs under
   `<analysis-output-dir>/<desc>/<subject>/{cv,model}/`. This works for
   **any number of conditions (2 or more)** -- `model_performance` derives
   its class list from `clf.classes_` (what the classifier actually learned),
   not from whatever happens to appear in a given CV fold's held-out data,
   so accuracy/evidence/AUC stay consistently shaped across folds regardless
   of how many conditions you configure.
5. For `timecourse_decoding`: relabels rows via that section's own
   conditions, then **recomputes a fresh volume range per source event**
   from `model_conditions.timecourse_decoding.window` and each event's
   `onset`/`duration`/`trial_index` (independent of whatever
   `hemodynamic_lag` was used to build `volume_of_interest` originally),
   predicts with the trained classifier, and writes two files to
   `<analysis-output-dir>/<desc>/<subject>/decoding/`:
   - `{subject}_decoding_results.csv` -- **raw**, one row per volume actually
     decoded, with its own `predicted_label` and `evidence_<category>`
     columns. This is the real per-TR data -- use it for anything that needs
     trial-level detail (custom stats, sanity-checking individual trials).
   - `{subject}_summary_decoding_results.csv` -- the raw table grouped by
     `(window_index, regressor_label)` and averaged across every trial in
     that group (trial-count-weighted, not an average of averages) -- the
     confusion-style timecourse view `generate_report.py` (section 7) reads.

`tutorial/haxby-data/derivatives/sub-1/masks/native_epi_mask.nii.gz` is a
real mask already, so this runs as shown above with no extra setup -- if
your own dataset doesn't have one yet, point `model.mask.mask_pattern` at a
real (or throwaway, for testing) mask file first.

### Trial pivot table (sanity check)

Written to `<analysis-output-dir>/<desc>/<subject>/<subject>_trial_pivot.csv`
before any `model_conditions` filtering -- one row per event retained in
`master_spreadsheet.csv` (i.e. every row of that subject's events.tsv files
across all runs, minus the hardcoded fixation/block/postRT exclusions above),
with its `volume_of_interest` values spread across `vol_of_interest_1..N`
columns (`N` = the widest trial; shorter trials are NaN-padded). Useful for
eyeballing whether the volume counts per trial look right -- not used by the
modeling steps themselves.

## 6. K-fold workflow (`workflows/mvpa_kfold_workflow.py`)

For same-task data: `model_conditions.training`/`testing` may still be
different conditions, but both are expected to be drawn from the same task's
runs. Instead of one train-once/test-once split, `mvpa_kfold_workflow.py`
repeatedly holds out a group of runs, trains on the rest, tests + decodes
only on the held-out group, then aggregates every fold into one final
answer -- the same `model_classification`/`model_performance`/
`timecourse_decoding` steps `mvpa_generalization_workflow.py` uses (both scripts import
them from `mvpa_common.py`, so they can never drift apart on how a model is
actually fit or scored). See `tutorial/config-kfold-haxby.example.json` for
a complete example (12-run leave-one-run-out over `tutorial/haxby-data`'s 8
object categories -- the same categories `tutorial/config-haxby.example.json`
uses for section 5's independent-train/test example, just with `training`/
`testing`/`timecourse_decoding` referencing every run instead of a fixed
9-run/3-run split, since fold membership is `model.kfold_cv`'s job here).

```
python workflows/mvpa_kfold_workflow.py --subject 1 --config tutorial/config-kfold-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv --analysis-output-dir ./out
```

### `model.kfold_cv`

Required for this script (it's the whole point of running it) -- how runs
are split into folds:

```json
"model": {
  ...,
  "kfold_cv": {
    "strategy": "per_run"
  }
}
```

| `strategy` | Meaning |
|---|---|
| `"per_run"` | Automatic, leave-one-run-out -- one fold per run found in this subject's testing/timecourse_decoding-eligible data. |
| `"group_kfold"` | Automatic -- requires an integer `n_splits` (>= 2, <= the number of distinct runs); runs are split into `n_splits` contiguous groups. |
| `"explicit_groups"` | User-defined -- requires `"held_out_runs"`. See below. |

Whichever strategy is used, the resolved fold membership is always written
to `model/{subject}_kfold_folds.json` (`{fold_id: [held-out run ids]}`) --
so an automatic split is just as inspectable after the fact as an explicit
one.

#### `strategy: "explicit_groups"` and `held_out_runs`

`held_out_runs` is a list of lists -- **one inner list per fold, and each
inner list is that fold's held-out run(s)**, not what to train on:

```json
"kfold_cv": {
  "strategy": "explicit_groups",
  "held_out_runs": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
}
```

This example produces 4 folds, grouping Haxby's 12 runs into 3-run blocks.
For fold 1 (`[1, 2, 3]`): training uses every training-condition row whose
`run` is *not* 1, 2, or 3; testing/timecourse_decoding use only rows whose
`run` *is* 1, 2, or 3. Folds 2-4 work the same way against `[4, 5, 6]`,
`[7, 8, 9]`, and `[10, 11, 12]`. Each fold trains and evaluates
independently -- run IDs that never appear in `held_out_runs` are simply
never held out, so they're always available for training but never scored
on their own.

A few things worth knowing before writing your own:

- **The lists don't have to partition the runs.** A run can appear in more
  than one fold's held-out set (evaluated more than once, in different
  folds), and runs can be left out of `held_out_runs` entirely (always
  trained on, never held out and scored). Use this deliberately -- e.g. to
  build unequal-sized folds, or to only ever evaluate a specific subset of
  runs -- not by accident.
- **Coverage is checked, but only warned about, not enforced.** If a run
  present in this subject's testing/timecourse_decoding data isn't covered
  by any group in `held_out_runs`, you'll see `(!) model.kfold_cv.held_out_runs
  doesn't cover run(s) [...]` -- those rows are simply never evaluated in any
  fold, the run itself is not an error. Conversely, if `held_out_runs`
  references a run ID that doesn't exist in this subject's data at all,
  you'll see `(!) model.kfold_cv.held_out_runs references run(s) [...] that
  don't appear ...` and that fold ends up with 0 test/timecourse rows
  (skipped at runtime with its own warning, not a crash).
- **Run IDs must match exactly, no type coercion.** `training_df["run"]` /
  `testing_df["run"]` (from `master_spreadsheet.csv`) are typically integers,
  and `held_out_runs` is matched against them with `.isin()` -- a JSON string
  `"1"` will never match integer `1`, it'll just silently produce an empty
  fold (with the "references run(s) that don't appear" warning above) rather
  than raising. If your runs come out as strings, write `held_out_runs` as
  strings too (`"held_out_runs": [["1", "2"]]`), matching whatever
  `master_spreadsheet.csv`'s `run` column actually contains.
- **Which config `run` values are valid to reference** -- the "universe" of
  runs `held_out_runs` is checked against is the set of `run` values present
  across `model_conditions.testing` and `model_conditions.timecourse_decoding`'s
  *matched* rows for this subject, not every run in `master_spreadsheet.csv`.
  A run with no testing/timecourse rows for this subject (e.g. it only has
  training-condition trials) won't show up in that universe, so referencing
  it in `held_out_runs` triggers the "doesn't appear" warning even though the
  run genuinely exists in the data -- it just has nothing to evaluate on.

`model.permutation_test` (see above) works the same way here as in
`mvpa_generalization_workflow.py`, except it runs **once per fold**, on that fold's own
train/test split -- `model/{subject}_fold{N}_permutation_test.csv`. Folds
aren't combined into one pooled p-value; interpret them fold-by-fold.

### Outputs

Per-fold files (fold `N`'s own held-out evaluation) alongside aggregated
files (averaged/pooled across every fold, same filenames `mvpa_generalization_workflow.py`
writes -- so `generate_report.py` and other downstream consumers don't need
to know which workflow produced a given subject's results):

| Per-fold | Aggregated |
|---|---|
| `model/{subject}_fold{N}_model_results_{metric}.csv` | `model/{subject}_model_results_{metric}.csv` |
| `model/{subject}_fold{N}_impa_native.nii.gz` | `model/{subject}_impa_native.nii.gz` |
| `model/{subject}_fold{N}_permutation_test.csv` *(optional)* | -- (not combined across folds) |
| `decoding/{subject}_fold{N}_decoding_results.csv` *(only if `timecourse_decoding` configured)* | `decoding/{subject}_decoding_results.csv` *(same)* |
| `decoding/{subject}_fold{N}_summary_decoding_results.csv` *(same)* | `decoding/{subject}_summary_decoding_results.csv` *(same)* |

Aggregation: scalar/matrix metrics and importance maps are averaged across
folds; decoding rows from different folds are genuinely disjoint trials
(folds partition runs), so the aggregated raw table is a concatenation, and
the aggregated summary is recomputed fresh from that pooled table
(trial-count-weighted, not an average of per-fold averages).

`generate_report.py` already detects `_fold{N}_*` files purely by their
presence on disk (see section 7) -- fold-variability panels (accuracy/AUC
overlays, timecourse bands, an importance-map consistency mosaic) render
automatically against this script's output with no changes needed.

## 7. Generating a report (`workflows/generate_report.py`)

Produces a multi-page PDF from `mvpa_generalization_workflow.py`'s output -- accuracy/AUC,
confusion-style accuracy/evidence matrices, annotated timecourse decoding,
and importance maps. One script, two scales, switched with `--subject`:

```
# group report -- aggregates every subject found under <dir>/<desc>/*/ --
# desc is read from --config's model.desc (same sanitization the workflow
# scripts use), so it always matches where they actually wrote output
python workflows/generate_report.py --analysis-output-dir ./out \
    --config tutorial/config-haxby.example.json --master-spreadsheet master_spreadsheet_haxby.csv

# single-subject report -- scoped to just <dir>/<desc>/1/
python workflows/generate_report.py --analysis-output-dir ./out --subject 1 \
    --config tutorial/config-haxby.example.json --master-spreadsheet master_spreadsheet_haxby.csv

# --desc still works directly, if you'd rather not point at a config
python workflows/generate_report.py --analysis-output-dir ./out --desc haxby_object_classifier
```

| Flag | Meaning |
|---|---|
| `--analysis-output-dir` | Same value used for `mvpa_generalization_workflow.py --analysis-output-dir`. |
| `--desc`/`--config` | **Exactly one required.** `--desc` names the classifier folder directly; `--config` reads it from the config's own `model.desc` instead (`quick_safe`-sanitized, identical to what the workflow scripts used to name their output folder) -- since it's the same value, the two can't drift apart the way a hand-typed `--desc` can. `--config` also supplies annotation (see below) even when `--desc` is given directly. |
| `--subject` | *(optional)* Restrict the report to one subject. Omit to aggregate over every subject folder found under `<dir>/<desc>/`. |
| `--config` | Supplies `model.desc` (see above, when `--desc` is omitted) and `model_conditions.timecourse_decoding` (conditions + window, and optionally `overlay` -- see section 4) for timecourse annotation either way. Without it (i.e. using `--desc` alone), the timecourse page still renders, just unannotated (and never split by overlay). |
| `--master-spreadsheet` | *(optional)* Needed alongside `--config` to compute each condition's median trial duration and each subject's TR (both derived from real data, not hardcoded) -- used to convert `window_index` to seconds and mark trial onset/end on the timecourse plot. Without it, the x-axis stays in raw `window_index` units and annotation is skipped. |
| `--output` | *(optional)* Defaults to `<dir>/<desc>/report_<desc>.pdf` (group) or `<dir>/<desc>/<subject>/report_<subject>.pdf` (single-subject). |

**Fold-variability panels are automatic, not configured.**
`generate_report.py` detects `_fold{N}_*` files (accuracy/AUC overlays,
timecourse bands, an importance-map consistency mosaic) purely by their
presence on disk -- `mvpa_generalization_workflow.py`'s single independent-train/
independent-test case doesn't produce them, but `mvpa_kfold_workflow.py`
(section 6) does, so these panels render automatically for its output with
no report-side changes needed.

**Group accuracy/AUC panels show mean + per-subject scatter.** For a
multi-subject report, the accuracy panel plots one bar each for mean
internal-CV and mean held-out-test accuracy, with every subject's own value
scattered on top (deterministic beeswarm spread, not random jitter, so the
figure is reproducible run to run); the AUC panel does the analogous thing
per category via a boxplot. Single-subject reports keep the original
per-subject/per-fold bar chart instead, since there's only one point per
metric.

**Importance maps are averaged across subjects only when they share a
common grid.** `model_impa` (`model/{subject}_impa_native.nii.gz`) is
always native-space and per-subject -- there's no common voxel grid to
average onto directly, unlike a normalized-space group analysis, so by
default the group report shows one native-space page per subject instead
(only within-subject fold-to-fold averaging, same subject same grid,
happens automatically). If you resample each subject's `model_impa` into a
shared MNI grid -- via `hcp_resample.py --direction native2mni` (section 8)
-- and save the result as `model/{subject}_impa_mni.nii.gz` right alongside
it, `generate_report.py` detects that suffix and plots a single group-mean
page in MNI space instead of per-subject native pages. Subjects missing
`_impa_mni.nii.gz`, or whose map doesn't match the other subjects' grid
shape, are excluded from the average with a printed warning rather than
failing the whole report; the group page's title records how many subjects
went into the average. The averaged map is also saved as its own NIfTI file
(`{desc}_group_mean_impa_mni.nii.gz`) right alongside the PDF -- the plotted
page is a quick look, the file is the actual data for loading elsewhere
(a group-level stats tool, a different viewer, a different threshold).
It's plotted with nilearn's "mosaic" display (many tiled slices across all
three planes, with nilearn's own bundled MNI152 template as an anatomical
background) rather than the compact 3-slice "ortho" view used for
native-space maps -- ortho has no shared template to plot against, so it
stays background-free there. One page per category, since a mosaic needs
much more room than ortho's single row. `slurm/4_sbatch_generate_report.sh`
(section 9) runs this resampling automatically for every subject before
generating the report -- manual `hcp_resample.py` calls are only needed if
you're generating a
report outside that pipeline.

The timecourse page always reads the raw per-TR `decoding_results.csv`, not
`summary_decoding_results.csv` -- the summary only ever kept each group's
mean, never the spread across the trials that went into it, so it can't
supply what's plotted now (see below). `summarize_raw_for_timecourse`
first collapses the raw rows to one row per subject/fold (mean *and*
trial-to-trial SE per `(window_index, regressor_label[, overlay_label])`),
exactly mirroring the within-subject/fold averaging
`mvpa_common.summarize_decoding()` already does for the summary CSV, just
computed independently so the trial-level spread survives too.

Each panel plots **two** overlapping shaded bands around the mean line
(same color, different opacity, drawn so both stay legible where they
overlap), and they answer different questions:
- **Darker band -- SE across subjects (or folds, for `mvpa_kfold_workflow.py`
  output)**: how consistent is the *group-level* estimate? Degenerates to a
  zero-width band for a single-subject, no-fold report -- nothing to
  average across when there's only one value.
- **Lighter band -- trial-to-trial SE**: how consistent is decoding across
  an individual subject's *own* trials, averaged across whatever's in
  scope (one subject, or several)? This is what fills in the single-subject
  case above with real information instead of a zero-width band, and is
  shown in every scope, band or no band.

Both compose with `model_conditions.timecourse_decoding.overlay` (section 4)
when it's configured -- each overlay category gets its own color, its own
two bands, plus a legend explaining both the categories and what the
lighter band means.

## 8. Resampling MNI <-> native space (`utils/hcp_resample.py`)

This pipeline's masks/classification are entirely native-space (see the
`native_*_mask.nii.gz` examples throughout), but data processed through HCP
Pipelines is commonly in MNI space and needs to move between the two --
e.g. bringing an MNI-space preprocessed BOLD run into native space to match
this pipeline's native masks, or bringing this pipeline's own native-space
importance map into MNI space for group-level comparison (save the result
as `model/{subject}_impa_mni.nii.gz` to have `generate_report.py`'s group
report pick it up automatically -- see section 7).
`hcp_resample.py` is a standalone script for exactly that, independent of
the classification/reporting scripts (`model_conditions`/`model` don't
apply here at all -- it takes plain CLI flags, one file at a time, same as
`mvpa_generalization_workflow.py`/`mvpa_kfold_workflow.py` leave batching
across subjects to your own SLURM array wrapper).

It wraps FSL's `applywarp` -- **requires FSL on `PATH`** (`module load fsl`
on a cluster; locally, `export FSLDIR=...` and
`export PATH="$FSLDIR/bin:$PATH"`, same setup `tutorial/preprocess_haxby.sh`
already documents). This is the same external dependency the tutorial's
preprocessing script already requires; no new Python package is added.

```
python utils/hcp_resample.py --input bold_mni.nii.gz --output bold_native.nii.gz \
    --direction mni2native --subject 001 --session D1 \
    --derivatives-root /path/to/derivatives/bids-hcp \
    --reference sub-001_ses-D1_task-func_run-03_bold.nii.gz --interp trilinear
```

| Flag | Meaning |
|---|---|
| `--input`/`--output` | The file to resample and where to write the result. |
| `--direction` | `mni2native` or `native2mni` -- selects which HCP warp file to use. |
| `--subject`/`--session` | Used to resolve the warp file path. Omit `--session` for sessionless datasets (same convention as `mask_pattern` elsewhere in this README -- just leave `{session}` out of the pattern). |
| `--derivatives-root` | Root of the `bids-hcp`-style derivatives tree containing `sub-{subject}/[ses-{session}/]MNINonLinear/xfms/`. |
| `--reference` | Target-space reference image (`applywarp --ref`). **Not auto-templated** -- pass it explicitly. This pipeline's native grid (EPI-resolution masks) and HCP's own native T1w grid differ, and guessing wrong here would silently misalign the output rather than error. |
| `--interp` | *(optional, default `trilinear`)* `nn`/`trilinear`/`sinc`/`spline` -- use `nn` for masks or other discrete-label images. |
| `--datatype` | *(optional)* Force the output data type, e.g. `int` after `--interp nn` on a mask. |
| `--xfm-pattern` | *(optional)* Override the warp-file path template (`{subject}`/`{session}` placeholders, resolved relative to `--derivatives-root`) for a non-standard layout. |
| `--warp-convention` | *(optional)* Force `applywarp --abs`/`--rel`. Leave unset by default -- real HCP-generated warp fields carry their own absolute/relative convention in the file header, which `applywarp` auto-detects; only set this if `applywarp` complains or the result looks wrong. |

### HCP warp file convention

Standard HCP Pipelines output layout, under each subject's
`MNINonLinear/xfms/`:

| Direction | File |
|---|---|
| `mni2native` | `standard2acpc_dc.nii.gz` |
| `native2mni` | `acpc_dc2standard.nii.gz` |

`--xfm-pattern` overrides this if your derivatives tree organizes these
differently.

## 9. Running the full pipeline end-to-end (`slurm/0_submit_mvpa_pipeline.sh`)

`slurm/0_submit_mvpa_pipeline.sh` chains every stage above into one SLURM
submission, so there's a single command that goes from HCP Pipelines output
to a group PDF report. The four stage scripts are numbered `1_`-`4_` to match
the order they run in (`0_` for the orchestrator itself, so it sorts first
in a directory listing too).

**One file to edit for a new study/config: `slurm/pipeline_vars.sh`.** Every
dataset path, the config filename, and the output/spreadsheet locations are
centralized there and sourced by each of `1_`-`4_`, instead of being
hardcoded separately in every stage script. Repoint a deployment at a
different study by editing this one file.

That file is built around `SCRIPTS_DIR`, an explicit variable holding the
repo root, used to locate `workflows/`, `utils/`, `configs/`, etc. instead
of assuming a job's own working directory happens to already be the repo
root. `0_` resolves its own real location (reliable here because it's
invoked directly via `bash`, never through `sbatch`, which would otherwise
obscure the original file path) and `export`s `SCRIPTS_DIR` before
submitting each job, so it's already set correctly by inheritance when a
stage script sources `pipeline_vars.sh`. Run `0_` from anywhere:

```
bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh
```

Standalone submission of an individual stage script doesn't inherit
`SCRIPTS_DIR` this way -- `pipeline_vars.sh` falls back to the job's own
working directory when `SCRIPTS_DIR` isn't already set, so submit from the
repo root (`sbatch slurm/1_batch_resample_native_mask.sh`) or export it
yourself first (`export SCRIPTS_DIR=/path/to/mvpa_banich`). Either way, a
job's `--output`/`--error` log paths are plain `#SBATCH` directives (no
variable substitution happens in them, since they're parsed before the
script body ever runs), so they always resolve relative to wherever
`sbatch` was actually invoked from, regardless of `SCRIPTS_DIR` -- `0_`
`cd`s to the repo root before submitting so its own jobs' logs land in the
right place.

`0_` submits the four numbered jobs via `sbatch --parsable`, each depending
on the previous one via `--dependency=afterok` (which, for an array job,
only fires once *every* array task has succeeded):

| Stage | Script | Type |
|---|---|---|
| 1. Mask resample | `slurm/1_batch_resample_native_mask.sh` | per-subject/session array job (section 8) |
| 2. Master spreadsheet | `slurm/2_sbatch_generate_master_spreadsheet.sh` | single job (section 1) |
| 3. K-fold classifier | `slurm/3_batch_run_mvpa_workflow.sh` | per-subject array job (section 6); also writes each subject's own single-subject report |
| 4. Group report | `slurm/4_sbatch_generate_report.sh` | single job (section 7); resamples every subject's importance map to MNI (section 8) before building the group report |

Every stage script can still be run standalone (e.g. to rerun just one stage
after fixing a subject-specific failure) with plain `sbatch slurm/<script>.sh`
-- the orchestrator only adds the dependency chaining on top, it doesn't own
any logic itself.

`logs/` is created automatically; each stage's own `logs/<job>_%A_%a.{out,err}`
(or `_%j.{out,err}` for the two single jobs) is where to look first if a
stage fails, since `--dependency=afterok` will simply never submit the next
one -- no separate failure notification.

See **[tutorial/README.md](tutorial/README.md)** for a full walkthrough
against this same real public data, downloaded fresh (not checked into this
repo) -- every command, the actual output and results, including a minimal
preprocessing pass (`tutorial/preprocess_haxby.sh`, built on FSL's `mcflirt`/
`flirt`/`applyxfm4D`/`fsl_glm`: motion correction, rigid coregistration to a
common run, linear detrending) written into a `derivatives/` folder and
picked up via `derivatives_root`/`bold_glob` -- and a detailed accounting of
where the tutorial still oversimplifies (rigid-only alignment with default
settings, no slice-timing correction, no normalization to a standard
template, a crude `bet` brain mask rather than an anatomical one). Its
sessionless filenames (no `ses-` entity at all) are also what surfaced a
real bug: `generate_master_spreadsheet.py` used to *require* `ses` and would
have silently rejected every file in a session-less dataset like this one --
fixed, since BIDS session labels are optional for single-session studies.
