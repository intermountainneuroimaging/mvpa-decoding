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
| 1. Build the volume table | `generate_master_spreadsheet.py` | `event_extraction` (+ optional `expected_events.json`) |
| 2. Define & validate MVPA conditions | `validate_model_config.py` | `model_conditions` |
| 3. Train/decode | `mvpa_workflow.py` | `model` (+ `model_conditions` to select/label rows) |

All three scripts take the **same** config file via `--config`. Shared logic
(BIDS filename parsing, the query DSL, window math) lives in `mvpa_common.py`,
imported by all three. Everything below is grounded in
`examples/config-2.example.json`, a complete config that runs end-to-end
against `examples/sample-data/`. `examples/config-1.example.json` is the same
shape but written as a fill-in-your-own-paths template, including the
derivative-data fields (`derivatives_root`, `mask_root`) described below.

## 1. What input data is assumed

You need a directory tree containing, for every scan run you want in the table:

- **An events file**: tab-separated `.tsv` with at least `onset`, `duration`,
  `trial_type` columns (standard BIDS events file). Its **filename** must
  contain BIDS key-value entities `sub-`, `ses-`, `task-`, `run-` somewhere in
  it (in any order, with any other entities mixed in between) --
  e.g. `sub-4057_ses-A1_task-loc_dir-pa_run-01_events.tsv`.
- **A matching BOLD file**: a `.nii.gz` whose filename contains the word
  `bold` plus the *same* `sub-`/`ses-`/`task-`/`run-` values as the events
  file, e.g. `sub-4057_ses-A1_task-loc_dir-pa_run-01_bold.nii.gz`. There must
  be **exactly one** such match per events file -- zero or multiple matches
  cause that events file to be skipped with a warning, not a crash.

Example tree (trimmed from `examples/sample-data/`, real files this repo's
examples run against):

```
examples/sample-data/
└── sub-4057/
    └── ses-A1/
        └── func/
            ├── sub-4057_ses-A1_task-loc_dir-pa_run-01_events.tsv
            ├── sub-4057_ses-A1_task-loc_dir-pa_run-01_bold.nii.gz
            ├── sub-4057_ses-A1_task-WMpos_dir-pa_run-01_events.tsv
            └── sub-4057_ses-A1_task-WMpos_dir-pa_run-01_bold.nii.gz
```

This assumes events and BOLD files are co-located and share a naming
convention. If your preprocessed data lives elsewhere (a separate fMRIPrep
`derivatives/` tree, a different naming scheme, etc.), see
[Using preprocessed/derivative data](#using-preprocessedderivative-data-eg-fmriprep) below --
`derivatives_root`/`bold_glob` decouple BOLD-file discovery from this assumption
entirely.

**Inferred from the data, never configured:**
- `subject`, `session`, `task`, `run` -- parsed out of the events filename.
- Any *other* BIDS entity in the events filename (e.g. `dir-pa`) -- captured
  automatically as its own extra column, named after the entity key. Different
  designs can carry different entities; whatever shows up, shows up as a column.
- **TR** and **frame count** -- read directly from the matched BOLD file's
  NIfTI header (`get_zooms()[3]` and `get_data_shape()[-1]`), never from a
  config value. If a run's BOLD file is missing, that run cannot be processed
  at all (no way to know its TR), so it's skipped.

## 2. The config file

One JSON file with three top-level sections. `examples/config-2.example.json`
is a complete, runnable example (`examples/config-1.example.json` is the same
shape as a fill-in-your-own-paths template):

```json
{
  "config_version": "1.0",
  "created_by": "AKH",
  "notes": "MVPA face/place; masking on grey matter",

  "event_extraction": { "...": "see section 3" },
  "model_conditions": { "...": "see section 4" },
  "model": { "...": "see section 5" }
}
```

## 3. `event_extraction`

Read by `generate_master_spreadsheet.py`.

```json
"event_extraction": {
  "bids_root": "examples/sample-data",
  "events_glob": "**/*_events.tsv",
  "hemodynamic_lag": 4.6,
  "output_file": "master_spreadsheet.csv",
  "expected_events_file": "examples/expected_events.example.json"
}
```

| Field | Meaning |
|---|---|
| `bids_root` | Directory to search under for events.tsv files. |
| `events_glob` | Glob (supports `**`) used to find events.tsv files under `bids_root`. |
| `hemodynamic_lag` | Seconds added to every event's `onset` before converting to volume indices. Override per-run with `--hemodynamic-lag`. |
| `output_file` | Where the resulting table is written. Override with `--output`. |
| `expected_events_file` | *(optional)* Path to a template of expected `trial_type` values -- see below. Override with `--expected-events`. |
| `derivatives_root` | *(optional)* Directory to search under for BOLD files, if different from `bids_root` -- e.g. a separate fMRIPrep `derivatives/` tree. Defaults to `bids_root`. See [Using preprocessed/derivative data](#using-preprocessedderivative-data-eg-fmriprep). |
| `bold_glob` | *(optional)* Needed whenever BOLD filenames don't follow the default lookup (match on `sub`/`ses`/`task`/`run` tokens + `"bold"` in the filename) -- e.g. fMRIPrep's `desc-`/`space-` suffixes, or when `derivatives_root` returns more than one match per run. A format string with `{subject}`/`{session}`/`{task}`/`{run}` placeholders, resolved relative to `derivatives_root`. |

### `expected_events.json` (optional, separate file)

A flat JSON list of every `trial_type` value you expect to see *somewhere*
across the whole dataset (no single run needs to contain all of them). After
building the table, the script diffs this list against what was actually
observed and prints warnings for both directions -- values you expected but
never saw, and values you saw but didn't expect (typos, unlisted new
conditions). This is how a stray-space typo like `"positive _face_image"` in
`examples/sample-data` got caught during development.

```json
[
  "start_block",
  "end_block",
  "rest_block",
  "trial_fixation",
  "view_face",
  "view_place",
  "suppress_face",
  "suppress_place",
  "maintain_face",
  "maintain_place",
  "breath_face",
  "breath_place",
  "track_face",
  "track_place",
  "positive_face_image",
  "negative_face_image",
  "positive_place_image",
  "negative_place_image"
]
```

### Running it

```
python generate_master_spreadsheet.py --config examples/config-2.example.json
```

Output (`master_spreadsheet.csv`) -- one row per BOLD volume that overlapped
an event's active window:

| Column | Meaning |
|---|---|
| `subject`, `session`, `task`, `run` | *Inferred* from the events filename. |
| `volume_of_interest` | *Computed*: the BOLD frame index, from `onset + hemodynamic_lag` through `onset + hemodynamic_lag + duration`, using the BOLD file's own TR, clipped to its frame count. |
| `trial_type` | Verbatim from the events file -- never reinterpreted, split, or renamed. |
| `trial_index` | *Computed*: 1-based sequential index (in onset order) among this run's *retained* events -- i.e. after the hardcoded exclusions below, so it's always contiguous. Identifies "which event produced this volume," used by `mvpa_workflow.py` for trial-balancing and for recomputing `timecourse_decoding`'s window. |
| `onset`, `duration` | Verbatim from the events file, repeated across every volume belonging to that event. |
| `boldfile`, `eventfile` | Resolved source file paths, for traceability/sorting. |
| *(varies)* | Any other BIDS entity found in the filename, e.g. `dir` -- *inferred*, present only if that entity appears in your filenames. |

Example real output rows (from `examples/sample-data`):

```
subject  session  volume_of_interest  trial_type   trial_index  onset   duration  task   run  boldfile  eventfile  dir
4057     A1       50                  view_place   2            18.507  2.764     WMneg  3    examples/sample-data/sub-4057/.../..._run-03_bold.nii.gz  examples/sample-data/sub-4057/.../..._run-03_events.tsv  pa
```

### Hardcoded exclusions

`generate_master_spreadsheet.py` drops a fixed set of administrative/non-trial
`trial_type` values before windowing -- they're never useful to any analysis,
so this isn't exposed as a config option. Edit the `EXCLUDED_TRIAL_TYPE_EXACT`
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

Two config fields decouple BOLD-file discovery from the events-file layout:

```json
"event_extraction": {
  "bids_root": "/data/raw_bids",
  "derivatives_root": "/data/derivatives/fmriprep",
  "bold_glob": "sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
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

`model.mask.mask_root` (see [section 5](#5-model)) works the same way for
mask files, and defaults to `derivatives_root` -- masks are usually derivative
products co-located with preprocessed BOLD data, but can be pointed elsewhere
independently (e.g. a separate hand-drawn ROI directory) if needed.
`examples/config-1.example.json` is a template showing all of these fields
filled in.

## 4. `model_conditions`

Read by `validate_model_config.py`. This defines, for each of three required
sections (`training`, `testing`, `timecourse_decoding`), a set of named
**conditions** -- the classifier's class labels -- each backed by a **query**
that selects which `master_spreadsheet.csv` rows belong to it.

### The query language

A query is a small recursive boolean tree over *any* column of
`master_spreadsheet.csv` (`trial_type`, `task`, `run`, `subject`, `dir`, ...):

```json
{"column": "trial_type", "match": "exact", "value": "view_face"}
{"column": "trial_type", "match": "in", "values": ["view_face", "view_place"]}
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

**`training`** -- rows used to fit the classifier. In the example, localizer
(`loc`) task trials, split into `face`/`place` by a substring match on
`trial_type`:

```json
"model_conditions": {
  "training": {
    "conditions": {
      "face":  {"and": [{"column": "task", "match": "exact", "value": "loc"},
                         {"column": "trial_type", "match": "regex", "value": ".*face.*"}]},
      "place": {"and": [{"column": "task", "match": "exact", "value": "loc"},
                         {"column": "trial_type", "match": "regex", "value": ".*place.*"}]}
    }
  }
}
```

**`testing`** -- held-out rows used to score the trained classifier. In the
example, any working-memory task run (`task` starting with `WM`):

```json
"testing": {
  "conditions": {
    "face":  {"and": [{"column": "task", "match": "regex", "value": "^WM.*"},
                       {"column": "trial_type", "match": "regex", "value": ".*face.*"}]},
    "place": {"and": [{"column": "task", "match": "regex", "value": "^WM.*"},
                       {"column": "trial_type", "match": "regex", "value": ".*place.*"}]}
  }
}
```

**`timecourse_decoding`** -- same idea, but for the trial-by-trial decoding
sweep. Here the example also excludes the `view_*` cue rows with `not`/`in`,
and adds a required **`window`**:

```json
"timecourse_decoding": {
  "conditions": {
    "face":  {"and": [{"column": "task", "match": "regex", "value": "^WM.*"},
                       {"column": "trial_type", "match": "regex", "value": ".*face.*"},
                       {"not": {"column": "trial_type", "match": "in", "values": ["view_face"]}}]},
    "place": {"and": [{"column": "task", "match": "regex", "value": "^WM.*"},
                       {"column": "trial_type", "match": "regex", "value": ".*place.*"},
                       {"not": {"column": "trial_type", "match": "in", "values": ["view_place"]}}]}
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

### Running it

```
python validate_model_config.py --config examples/config-2.example.json \
    --master-spreadsheet master_spreadsheet.csv
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

Example output against `examples/sample-data`:

```
Validating examples/config-2.example.json against master_spreadsheet.csv
  [training] 'face': 524 rows
  [training] 'place': 536 rows
  [testing] 'face': 2020 rows
  [testing] 'place': 2022 rows
  [timecourse_decoding] 'face': 1009 rows
  [timecourse_decoding] 'place': 1012 rows

0 error(s), 0 warning(s)
```

## 5. `model`

Read by `mvpa_workflow.py`. Everything the analysis itself needs that isn't
about *which rows* to use (that's `model_conditions`'s job):

```json
"model": {
  "desc": "gm_object_classifier",
  "mask": {
    "mask_pattern": "sub-{subject}/ses-{session}/masks/native_gm_transformed_mask.nii.gz"
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
  },
  "cv": {
    "strategy": "GroupKFold",
    "n_splits": "infer"
  }
}
```

| Field | Meaning |
|---|---|
| `desc` | Short name for this classifier variant; sanitized into the output folder name. |
| `mask.mask_root` | *(optional)* Directory to search under for mask files. Defaults to `event_extraction.derivatives_root` (which itself defaults to `bids_root`) -- override independently if masks live somewhere else, e.g. a separate ROI directory. |
| `mask.mask_pattern` | Path to the per-subject mask NIfTI, resolved relative to `mask_root` with `{subject}`/`{session}` filled in from whichever row is being loaded (can still contain glob wildcards -- resolved the same way as bold-file lookups). |
| `featureSelection` | ANOVA voxel-selection threshold used before fitting the classifier. |
| `classifier` | Any importable scikit-learn-style estimator: `name` is a dotted import path, `params` are passed straight through as kwargs. |
| `cv` | Cross-validation bookkeeping (folds are actually built from the `run` column via `PredefinedSplit`). |

Omit any of `featureSelection`/`classifier`/`cv` and it falls back to a
default (ANOVA @ p<0.05, `LogisticRegression`, `GroupKFold`) -- only `desc`
and `mask` are meaningfully required.

## Running `mvpa_workflow.py`

```
python mvpa_workflow.py --subject 4057 --config examples/config-2.example.json \
    --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
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
   predicts with the trained classifier, and writes per-relative-timepoint
   accuracy/evidence to `<analysis-output-dir>/<desc>/<subject>/decoding/`.

`examples/sample-data` has no `masks/` directory, so running this against it
requires pointing `model.mask.mask_pattern` (and `mask_root`, if the mask
doesn't live under `derivatives_root`) at a real (or throwaway, for testing) mask
file first.

### Trial pivot table (sanity check)

Written to `<analysis-output-dir>/<desc>/<subject>/<subject>_trial_pivot.csv`
before any `model_conditions` filtering -- one row per event retained in
`master_spreadsheet.csv` (i.e. every row of that subject's events.tsv files
across all runs, minus the hardcoded fixation/block/postRT exclusions above),
with its `volume_of_interest` values spread across `vol_of_interest_1..N`
columns (`N` = the widest trial; shorter trials are NaN-padded). Useful for
eyeballing whether the volume counts per trial look right -- not used by the
modeling steps themselves.

## Tutorial: a real external dataset (Haxby et al. 2001 / OpenNeuro ds000105)

See **[tutorial/README.md](tutorial/README.md)** for a full walkthrough
against real public data (downloaded fresh, not checked into this repo) --
every command, the actual output and results, including a minimal
preprocessing pass (`tutorial/preprocess_haxby.py`: rigid motion correction,
rigid coregistration to a common run, linear detrending) written into a
`derivatives/` folder and picked up via `derivatives_root`/`bold_glob` -- and
a detailed accounting of where the tutorial still oversimplifies (rigid-only
alignment with deliberately cheap settings, no slice-timing correction, no
normalization to a standard template, a crude intensity-based mask rather
than an anatomical one). It's also the only example here with **8**
classification conditions (not 2) and no `ses-` entity in its filenames at
all, which is what surfaced a real bug: `generate_master_spreadsheet.py`
used to *require* `ses` and would have silently rejected every file in a
session-less dataset like this one -- fixed, since BIDS session labels are
optional for single-session studies. `tutorial/config-haxby.example.json` is
the exact config used.
