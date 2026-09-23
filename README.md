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
| 3. Train/decode | `workflows/mvpa_workflow.py` | `model` (+ `model_conditions` to select/label rows) |

Step 3 is one script running up to three independent, config-driven steps --
each only runs (and only writes its own output) when its config section is
present: `model.kfold_cv` cross-validates entirely within
`model_conditions.training` (section 5, output under `model/`);
`model_conditions.testing` evaluates one classifier fit on the complete
training set against a genuinely separate test set (optional -- section 4,
output under `test/`); `model_conditions.timecourse_decoding` predicts at
every TR across a decode window, using that same complete-training-set fit
(section 4, output under `decoding/`). Any combination -- one, two, all
three, or none of them beyond the sanity-check trial pivot table -- is valid
in a single run; see [section 6](#6-running-workflowsmvpa_workflowpy) for
the full breakdown. All scripts take the **same** config file via
`--config`. Shared logic -- BIDS filename parsing, the query DSL, window
math, and the actual classification/decoding primitives -- lives in
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
| `trial_index` | *Computed*: 1-based sequential index (in onset order) among this run's *retained* events -- i.e. after the hardcoded exclusions below, so it's always contiguous. Identifies "which event produced this volume," used by `mvpa_workflow.py` for trial-balancing and for recomputing `timecourse_decoding`'s window. |
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
sections (`training` -- required -- and `testing`/`timecourse_decoding`,
both optional), a set of named **conditions** -- the classifier's class labels --
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

**`testing`** -- *(optional)* held-out rows used to score the classifier
`mvpa_workflow.py` fits on the complete `training` set. In the example, the
remaining 3 runs (10-12) -- same task, same categories, held out by run
rather than by a different task. That's what this particular example
happens to do; `training`/`testing` can just as easily reference genuinely
different tasks (e.g. train on a localizer, test on a separate main-task
run) -- the query language doesn't care which, `task` is just another
column. Omit the whole section (not just leave it empty) to skip this
independent-test-set evaluation entirely -- no `test/` output at all for
that subject (no `test/{subject}_model_results_{metric}.csv`, no
`test/{subject}_impa[_mni].nii.gz`, no `test/{subject}_permutation_test.csv`
even if `model.permutation_test` is configured), no extra runtime for that
step. This is unlike `training`, which is always required -- a classifier
is always fit on the complete `training` set regardless of what else is
configured (that fit is what `timecourse_decoding`, below, predicts with,
and -- when `testing` *is* present -- what gets scored against it), and
`model.kfold_cv` ([section 5](#5-model)) cross-validates entirely within
`training` data too, so there has to be something there either way:

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
decoding sweep. Always uses the one classifier fit on the complete
`training` set (never per-fold, even when `model.kfold_cv` is also
configured). Here the example also adds a required **`window`**. Omit the
whole section (not just leave it empty) to skip timecourse decoding
entirely -- no `decoding/` output files at all, no extra runtime for that
step, and `generate_report.py`'s timecourse page is automatically skipped
too (it already skips whenever it finds no `decoding_results.csv` for any
subject in scope, so there's nothing extra to configure on the report side):

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
either workflow script. Same name-to-query shape as `conditions` -- each
entry both **filters/labels events** (its query, exactly like a `conditions`
entry) **and controls how that category is drawn** on the timecourse page
(section 7), via two further optional keys on the same entry:

- **`color`** -- an integer indexes the standard palette (`0`-`9`, wrapping);
  a string is passed straight through to matplotlib as a literal color
  (`"#1f77b4"`, `"red"`, ...). Omit it to auto-assign (see below).
- **`line_type`** -- an integer indexes `solid, dashed, dotted, dashdot`; a
  string is passed straight through to matplotlib (`"--"`, `"dashed"`, ...).
  Omit it and that category draws solid.

For example, 8 categories -- 4 conditions crossed with 2 sub-conditions --
organized so `1A`/`1B` share a color and differ only by line style, same for
`2A`/`2B`, `3A`/`3B`, `4A`/`4B`:

```json
"overlay": {
  "1A": {"column": "trial_type", "match": "exact", "value": "cond1A", "color": 0, "line_type": "-"},
  "1B": {"column": "trial_type", "match": "exact", "value": "cond1B", "color": 0, "line_type": "--"},
  "2A": {"column": "trial_type", "match": "exact", "value": "cond2A", "color": 1, "line_type": "-"},
  "2B": {"column": "trial_type", "match": "exact", "value": "cond2B", "color": 1, "line_type": "--"}
}
```

**If `color` is omitted, it auto-assigns from the standard palette,
restarting at index 0 separately for each resolved `line_type`.** So the
same 4-color/2-style grouping above can be built with no `color` keys at
all, as long as same-colored entries are declared in matching order within
each `line_type`:

```json
"overlay": {
  "1A": {"column": "trial_type", "match": "exact", "value": "cond1A", "line_type": "-"},
  "2A": {"column": "trial_type", "match": "exact", "value": "cond2A", "line_type": "-"},
  "1B": {"column": "trial_type", "match": "exact", "value": "cond1B", "line_type": "--"},
  "2B": {"column": "trial_type", "match": "exact", "value": "cond2B", "line_type": "--"}
}
```
(`1A`/`2A` are the first/second color-less `line_type="-"` entries -> palette
indices 0/1; `1B`/`2B` restart that counter for `line_type="--"` -> indices
0/1 again -- landing on the same two colors as the explicit version.)

An `overlay` with only one entry per condition, or with no `color`/
`line_type` on any entry, behaves exactly like a single independent split
always has: one color per category, all solid. This also covers the earlier,
simpler use case of splitting by a secondary factor without needing per-trace
control at all -- e.g. Haxby's raw events don't carry a secondary factor the
way some designs do, so as a syntax illustration, this overlays each
category's evidence curve by *which* testing run a trial came from, letting
color auto-assign:

```json
"overlay": {
  "runs_10_11": {"column": "run", "match": "in", "values": ["10", "11"]},
  "run_12":     {"column": "run", "match": "exact", "value": "12"}
}
```

When present, the timecourse page (section 7) overlays one line per overlay
category within each existing subplot, instead of a single line -- see that
section for how it changes the plot. Rows matching none of the overlay
queries are dropped from that plot only (a count is printed); everything
else about the pipeline -- the classifier, its evidence values,
`decoding_results.csv`/`summary_decoding_results.csv` -- is unaffected,
since `overlay` is evaluated entirely inside `generate_report.py` against
data the workflow scripts already wrote (the `color`/`line_type` keys are
likewise inert everywhere else -- `evaluate_query_node`/`validate_query_node`
only ever read the query keys they need, so they simply ignore both).

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
| `mnispace` | *(optional, default `false`)* Set `true` when the BOLD/mask this subject's model is fit on are already registered to MNI space -- there's no way to detect this automatically from the file itself, so it's an explicit claim you make. Controls two things at once (see [section 7](#7-generate_reportpy)): the importance-map filename (`{subject}_impa_mni.nii.gz` instead of plain `{subject}_impa.nii.gz`), and whether `generate_report.py` plots it against nilearn's bundled MNI152 template as anatomical background. Setting this `true` also means every subject's importance map already carries the exact filename `generate_report.py`'s cross-subject group averaging looks for -- no separate `hcp_resample.py --direction native2mni` step needed. Leave `false` (or omit) for native-space or otherwise-unregistered data. |
| `featureSelection.feat_p` | ANOVA p-value threshold -- voxels with `p < feat_p` are kept, widened automatically until at least 5 voxels are selected. Ignored when `n_voxels` is set. |
| `featureSelection.n_voxels` | *(optional)* Select exactly this many voxels by ANOVA F-score instead, regardless of significance (sklearn's `SelectKBest` equivalent) -- takes priority over `feat_p` when both are present. Useful for keeping feature count fixed across subjects/folds whose signal strength (and thus a p-value threshold's actual voxel count) varies. `model_results_auc.csv`-adjacent output files still record whichever mode was actually used: `threshold_p` is `NaN` in this mode, since there's no threshold, but `selected_voxels` (identical to `n_voxels` here) is populated either way. |
| `classifier` | Any importable scikit-learn-style estimator: `name` is a dotted import path, `params` are passed straight through as kwargs. |
| `kfold_cv` | *(optional)* Cross-validates entirely within `model_conditions.training` -- see below. Omitting it entirely skips k-fold cross-validation: no `model/` output at all, no extra runtime for that step. |

Omit either of `featureSelection`/`classifier` and it falls back to a
default (ANOVA @ p<0.05, `LogisticRegression`); omit `mask` and every voxel
is used (with a warning) -- only `desc` is truly required.

### `kfold_cv` (optional): cross-validating within `model_conditions.training`

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

#### Cross-validation hold-out sample detail: `model/{subject}_cv_results.csv`

In addition to the per-fold/aggregated accuracy summaries above, every fold
also writes its held-out samples out at the trial level -- **one row per
held-out sample, across all folds** -- to `model/{subject}_cv_results.csv`,
in the same raw-table style as `decoding_results.csv` (see section 7):
`subject`, `model_descr`, `fold`, plus every column already carried by that
row's `model_conditions.training` match (`task`, `trial_type`, `run`,
`boldfile`, `trial_index`, `regressor_label`, ...), then `predicted_label`,
`correct`, `evidence_<category>`, and that fold's own feature-selection
footprint (`threshold_p`/`selected_voxels`/`whole_voxels`/`feature_percent`).
Since every row is scored by whichever fold held its own run out, `fold`
plus `run` together document exactly which run was held out (and evaluated)
in each fold -- e.g. with `"per_run"`, fold 1's rows are all the run-1 rows,
fold 2's are all the run-2 rows, and so on. `generate_report.py`'s group
report concatenates every subject's file into
`{desc}_group_cv_results.csv` (see section 7), so you can load the whole
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
  pipeline replicates (see [THEORY.md](THEORY.md)) -- reproduce it today via
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

### `permutation_test` (optional): significance testing for held-out results

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

`accuracy`'s `real_score` matches the corresponding `model_results_total_scores.csv`
exactly (same split, same fixed feature-selection threshold, same classifier
config -- `model/{subject}_fold{N}_model_results_total_scores.csv` for a
kfold fold, `test/{subject}_model_results_total_scores.csv` for the
independent test set). `roc_auc_ovr`'s `real_score` will be *close to* the
mean of the corresponding `model_results_auc.csv`'s per-category values --
both are one-vs-rest AUC built from the same normalized-probability evidence
(`decision_evidence`: `predict_proba()`/softmax, rows sum to 1 -- not an
independent per-class sigmoid), so they should agree closely; sklearn's
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

### `allow_train_test_overlap` (optional): ⚠️ double-dipping guard override

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
otherwise double-dip -- see section 6's "Double-dipping guard" for the full
explanation. Setting this to `true` disables that protection and restores
the original behavior. **Only do this if you have a specific, considered
reason the overlap in your config isn't a methodological problem** -- see
the warning in section 6 before using it.

## 6. Running `workflows/mvpa_workflow.py`

`mvpa_workflow.py` replaces two previous, separate scripts
(`mvpa_generalization_workflow.py` and `mvpa_kfold_workflow.py`) with one:
it always fits a classifier on the complete `model_conditions.training` set,
then runs up to three further steps -- each **independently optional**,
each only writing its own output when its config section is present:

1. **`model.kfold_cv`** (section 5) -- k-fold cross-validates entirely
   within `model_conditions.training` (train on non-held-out runs, evaluate
   on held-out runs of that same training data). Output under `model/`.
2. **`model_conditions.testing`** (section 4) -- evaluates the
   complete-training-set classifier against a genuinely separate test set.
   Output under `test/`.
3. **`model_conditions.timecourse_decoding`** (section 4) -- predicts at
   every TR across a decode window, using that same complete-training-set
   classifier (never per-fold, even when `model.kfold_cv` is also
   configured). Output under `decoding/`.

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
2. Evaluates `model_conditions.training`/`testing`/`timecourse_decoding`'s
   queries to label and select rows (a row matching more than one condition
   takes the first match, in the order conditions are listed --
   `validate_model_config.py` already warns about that case). Loads BOLD
   patterns directly from each row's `boldfile` (already a concrete,
   resolved path -- no glob/pattern matching needed at this stage),
   z-scores, and slices to `volume_of_interest`.
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
   double-dipping guard below doesn't skip it for this subject**: relabels
   rows via that section's own conditions, then **recomputes a fresh volume
   range per source event** from `model_conditions.timecourse_decoding.window`
   and each event's `onset`/`duration`/`trial_index` (independent of
   whatever `hemodynamic_lag` was used to build `volume_of_interest`
   originally), predicts with the complete-training-set classifier from
   step 3 (or, per the guard below, each overlapping run's own held-out
   k-fold classifier), and writes two files to
   `<analysis-output-dir>/<desc>/<subject>/decoding/`:
   - `{subject}_decoding_results.csv` -- **raw**, one row per volume actually
     decoded, with its own `predicted_label` and `evidence_<category>`
     columns. This is the real per-TR data -- use it for anything that needs
     trial-level detail (custom stats, sanity-checking individual trials).
   - `{subject}_summary_decoding_results.csv` -- the raw table grouped by
     `(window_index, regressor_label)` and averaged across every trial in
     that group (trial-count-weighted, not an average of averages) -- the
     confusion-style timecourse view `generate_report.py` (section 7) reads.

This works for **any number of conditions (2 or more)** in every step --
`model_performance` derives its class list from `clf.classes_` (what the
classifier actually learned), not from whatever happens to appear in a
given fold's/test set's held-out data, so accuracy/evidence/AUC stay
consistently shaped regardless of how many conditions you configure.

`tutorial/haxby-data/derivatives/sub-1/masks/native_epi_mask.nii.gz` is a
real mask already, so this runs as shown above with no extra setup -- if
your own dataset doesn't have one yet, point `model.mask.mask_pattern` at a
real (or throwaway, for testing) mask file first.

### ⚠️ Double-dipping guard: training vs. testing/timecourse independence

**A classifier evaluated on data from a run it was even partly trained on
has inflated apparent performance.** fMRI noise is autocorrelated within a
run (thermal drift, motion, physiological signal), so a train/test split
that isn't clean at the run level lets the classifier partially "recognize"
the run itself rather than genuinely generalizing -- the textbook
non-independence/circular-analysis problem. It's an easy trap to fall into
here specifically because `model_conditions.training`/`testing`/
`timecourse_decoding` are three independent queries over the same
`master_spreadsheet.csv`: nothing stops two of them from matching rows out
of the very same bold file (e.g. reusing the same task/run range, or a
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

### Trial pivot table (sanity check)

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

### Outputs

Copied from `mvpa_workflow.py`'s own module docstring -- the authoritative
listing -- under `<analysis-output-dir>/<model.desc>/<subject>/`:

```
<subject>_trial_pivot.csv                     -- sanity check, pre-model_conditions

model.kfold_cv configured -- k-fold CV entirely within model_conditions.training:
  model/<subject>_kfold_folds.json                    -- {fold_id: [held-out run ids]}
  model/<subject>_fold{N}_model_results_{metric}.csv  -- per-fold held-out metrics
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
set (section 5) -- see `utils.mvpa_common.impa_tag`.

`generate_report.py` already detects `_fold{N}_*` files under `model/`
purely by their presence on disk (see section 7) -- fold-variability panels
(accuracy/AUC overlays, timecourse bands, an importance-map consistency
mosaic) render automatically whenever `model.kfold_cv` output exists, with
no report-side configuration needed.

## 7. Generating a report (`workflows/generate_report.py`)

Produces a multi-page PDF from `mvpa_workflow.py`'s output -- accuracy/AUC,
confusion-style accuracy/evidence matrices, annotated timecourse decoding,
and importance maps. One script, two scales, switched with `--subject`.
`mvpa_workflow.py`'s two output families -- `model.kfold_cv` (under
`model/`) and `model_conditions.testing` (under `test/`) -- are entirely
independent and shown throughout as separate **"CV"**/**"held-out-test"** sections; a
subject/report may have either, both, or neither, and every page below
renders whichever families actually have data:

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
| `--analysis-output-dir` | Same value used for `mvpa_workflow.py --analysis-output-dir`. |
| `--desc`/`--config` | **Exactly one required.** `--desc` names the classifier folder directly; `--config` reads it from the config's own `model.desc` instead (`quick_safe`-sanitized, identical to what `mvpa_workflow.py` used to name its output folder) -- since it's the same value, the two can't drift apart the way a hand-typed `--desc` can. `--config` also supplies annotation (see below) even when `--desc` is given directly. |
| `--subject` | *(optional)* Restrict the report to one subject. Omit to aggregate over every subject folder found under `<dir>/<desc>/`. |
| `--config` | Supplies `model.desc` (see above, when `--desc` is omitted) and `model_conditions.timecourse_decoding` (conditions + window, and optionally `overlay` -- see section 4) for timecourse annotation either way. Without it (i.e. using `--desc` alone), the timecourse page still renders, just unannotated (and never split by overlay). |
| `--master-spreadsheet` | *(optional)* Needed alongside `--config` to compute each condition's median trial duration and each subject's TR (both derived from real data, not hardcoded) -- used to convert `window_index` to seconds and mark trial onset/end on the timecourse plot. Without it, the x-axis stays in raw `window_index` units and annotation is skipped. |
| `--output` | *(optional)* Defaults to `<dir>/<desc>/report_<desc>.pdf` (group) or `<dir>/<desc>/<subject>/report_<subject>.pdf` (single-subject). |

**A "Data Independence Warning" page appears automatically, right after the
title page, whenever it has something to say.** If `mvpa_workflow.py`'s
double-dipping guard (section 6) skipped the held-out test evaluation,
substituted held-out k-fold classifiers for overlapping timecourse rows, or
ran anyway because `model.allow_train_test_overlap` was set, that's shown
here per subject -- reading straight from each subject's
`{subject}_double_dipping_report.json` (only written when the guard
actually found something). A report where training/testing/timecourse were
genuinely independent for every subject gets no such page at all, exactly
as before this guard existed.

**Fold-variability panels are automatic, not configured.**
`generate_report.py` detects `_fold{N}_*` files under `model/` (accuracy/AUC
overlays, timecourse bands, an importance-map consistency mosaic) purely by
their presence on disk -- they render automatically whenever `model.kfold_cv`
(section 5) produced them, with no report-side configuration needed. A
subject with only `model_conditions.testing` configured (no `model.kfold_cv`)
simply has no fold files, so no fold panels, same as before.

**Accuracy/AUC panels show CV and held-out-test side by side, whichever
exist.** The accuracy panel plots one bar per family present (CV from
`model/`, held-out-test from `test/`) against a chance-level reference line;
for a multi-subject report
that's the mean across subjects with every subject's own value scattered on
top (deterministic beeswarm spread, not random jitter, so the figure is
reproducible run to run), while a single-subject report also overlays that
subject's own per-fold CV values as black dots. The AUC panel does the
analogous thing per category -- a boxplot per family for a multi-subject
report, grouped bars (plus per-fold overlay dots for CV) for a
single-subject one -- also against a chance-level line. Either panel is
simply blank when neither family has data for the subjects in scope.

**The confusion-matrices page is a CV/held-out-test grid.** Up to two rows --
"CV" (from `model/kfold_accuracy`/`kfold_evidence`) and "held-out-test" (from
`test/test_accuracy`/`test_evidence`) -- each with two columns, Accuracy and
Evidence; a row is omitted entirely (not left blank) when that family has no
files for any subject in scope, and a multi-subject report averages each
family's matrices across subjects first. Every populated cell is labeled
with its value to 2 decimal places, colored light-on-dark or dark-on-light
depending on the cell's own intensity so the numbers stay legible against
the `viridis` colormap underneath. Both matrices' rows sum to 1 (a genuine
distribution over categories per true condition): Accuracy is the fraction
of that condition's trials predicted as each category; Evidence is the mean
`decision_evidence()` (normalized `predict_proba()`/softmax, not an
independent per-class sigmoid) across those same trials.

**Importance maps are averaged across subjects only when they share a
common grid.** A subject can have two independent importance-map families:
"CV" (`kfold_impa` -- the mean importance map across every k-fold fold's
own fit) and "held-out-test" (`test_impa` -- the one classifier fit on the complete
training set, whose weights are also what gets evaluated against
`model_conditions.testing` when that's configured). Either, both, or
neither may exist depending on what `model.kfold_cv`/`model_conditions.testing`
were configured. Neither family's space is asserted/known by default (native,
MNI, or otherwise -- whatever the input BOLD/mask happened to be in;
`model/{subject}_impa.nii.gz` / `test/{subject}_impa.nii.gz`), since it isn't
reliably knowable from the file itself. There's no guarantee subjects share
a common voxel grid in that case, unlike a normalized-space group analysis,
so the group report shows per-subject pages instead (one page per family per
subject; only within-subject fold-to-fold averaging, same subject same
grid, happens automatically for CV). A subject's per-subject page is
plotted against nilearn's bundled MNI152 template as anatomical background
exactly when the config asserts MNI space (see below) -- otherwise it's
shown background-free, since overlaying a template the map isn't actually
registered to would be misleading, not just decorative.

Two ways to get a group-mean page in MNI space instead. If your data is
already in MNI space, set `model.mnispace: true` (section 5) -- the
workflow script then writes each importance map directly as
`{subject}_impa_mni.nii.gz` (under `model/` and/or `test/`, whichever
families are configured), and `generate_report.py` reads that same
`model.mnispace` setting from `--config` to know to look for it, no
resampling step required. Otherwise, resample each subject's native-space
`impa` into a shared MNI grid after the fact -- via `hcp_resample.py
--direction native2mni` (section 8) -- and save the result as
`{subject}_impa_mni.nii.gz` right alongside it. Either way,
`generate_report.py`'s `resolve_group_impa_mni` looks for `test_impa_mni`
(held-out-test) first for each subject, falling back to `kfold_impa_mni`
(CV) only when held-out-test isn't available for that subject -- so a group
of subjects with a mix of CV-only, held-out-test-only, and both-configured
runs can still all contribute to one group average, each via whichever
family it has.
Subjects missing both, or whose map doesn't match the other subjects' grid
shape, are excluded from the average with a printed warning rather than
failing the whole report; the group page's title records how many subjects
went into the average, and the console log separately flags how many were
included via the CV fallback. The averaged map is also saved as its own
NIfTI file (`{desc}_group_mean_impa_mni.nii.gz`) right alongside the PDF --
the plotted page is a quick look, the file is the actual data for loading
elsewhere (a group-level stats tool, a different viewer, a different
threshold). It's plotted with nilearn's "mosaic" display (many tiled slices
across all three planes) rather than the compact 3-slice "ortho" view used
for per-subject pages -- one page per category, since a mosaic needs much
more room than ortho's single row. `slurm/4_sbatch_generate_report.sh`
(section 9) runs the `hcp_resample.py` resampling automatically for every
subject before generating the report (skipped entirely for subjects whose
importance map is already named `_impa_mni.nii.gz`, i.e. ran with
`mnispace: true`) -- manual `hcp_resample.py` calls are only needed if
you're generating a report outside that pipeline.

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
- **Darker band -- SE across subjects (or folds, when `model.kfold_cv`
  output is present)**: how consistent is the *group-level* estimate? Degenerates to a
  zero-width band for a single-subject, no-fold report -- nothing to
  average across when there's only one value.
- **Lighter band -- trial-to-trial SE**: how consistent is decoding across
  an individual subject's *own* trials, averaged across whatever's in
  scope (one subject, or several)? This is what fills in the single-subject
  case above with real information instead of a zero-width band, and is
  shown in every scope, band or no band.

Both compose with `model_conditions.timecourse_decoding.overlay` (section 4)
when it's configured -- each overlay category gets its own two SE bands,
plus a legend naming the categories and what the lighter band means. Each
category's actual color/line style come from `resolve_overlay_styles`
(section 4's `color`/`line_type`, per entry, with sensible auto-assigned
defaults for whichever is omitted).

**A group report (no `--subject`) also writes its own spreadsheets**
alongside the PDF, in the same folder as `--output`, so the underlying
per-trial data is available to load in your own software rather than only
ever viewed through the PDF's plots:
- `{desc}_group_summary.csv` -- one row per (subject, family), every
  scalar/vector/matrix metric flattened into its own column.
- `{desc}_group_decoding_results.csv` -- every subject's raw
  `decoding_results.csv` (timecourse decoding, one row per decoded TR)
  concatenated together.
- `{desc}_group_cv_results.csv` -- every subject's raw `model/{subject}_cv_results.csv`
  (cross-validation hold-out sample detail, section 5) concatenated
  together -- one row per held-out sample across every subject's folds,
  carrying `task`/`trial_type`/`run`/`fold`/`predicted_label`/`correct`/
  `evidence_<category>`.

Each is only written if at least one subject in scope actually has that
output (e.g. `{desc}_group_cv_results.csv` is skipped entirely if no
subject ran with `model.kfold_cv` configured).

## 8. Resampling MNI <-> native space (`utils/hcp_resample.py`)

This pipeline's masks/classification are entirely native-space (see the
`native_*_mask.nii.gz` examples throughout), but data processed through HCP
Pipelines is commonly in MNI space and needs to move between the two --
e.g. bringing an MNI-space preprocessed BOLD run into native space to match
this pipeline's native masks, or bringing this pipeline's own native-space
importance map into MNI space for group-level comparison (save the result
as `{subject}_impa_mni.nii.gz`, right alongside the plain `impa` file under
`model/` or `test/`, to have `generate_report.py`'s group report pick it up
automatically -- see section 7).
`hcp_resample.py` is a standalone script for exactly that, independent of
the classification/reporting scripts (`model_conditions`/`model` don't
apply here at all -- it takes plain CLI flags, one file at a time, same as
`mvpa_workflow.py` leaves batching across subjects to your own SLURM array
wrapper).

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

**The only required input is your config JSON.** Every stage script takes
it as a required first argument and reads all of its study-specific paths
(dataset roots, output/spreadsheet locations, the MNI template) from a
`"pipeline"` section in that same file -- there's no separate bash file to
hand-edit for a new study/config anymore, and no risk of the four stage
scripts silently disagreeing about which study they're pointed at, since
they all read the exact same config path:

```json
"pipeline": {
  "scripts_dir": "/path/to/mvpa-decoding",
  "bids_hcp_root": "/path/to/study/bids-hcp",
  "hcppipe_root": "/path/to/study/HCPPipe",
  "group_gm_mask": "/path/to/study/masks/group_gm_mask.nii.gz",
  "output_dir": "/path/to/study/mvpa-decoding",
  "master_spreadsheet": "/path/to/study/mvpa-decoding/master_spreadsheet.csv",
  "mni_template": "/path/to/MNI152_T1_2mm_brain.nii.gz"
}
```

| Field | Required by | Meaning |
|---|---|---|
| `scripts_dir` | every stage | This repo's root on the cluster -- how every stage script locates `workflows/`, `utils/`, etc. See below for why this comes from the config rather than an exported variable or the job's own location. |
| `bids_hcp_root` | stages 1, 3 | Root of the BIDS-HCP derivatives tree -- where stage 1 writes each subject/session's resampled native-space mask (section 8's `--derivatives-root`-adjacent layout: `sub-{subject}/ses-{session}/func/`), and where stage 3 lists subjects from. |
| `hcppipe_root` | stage 1; optional for stage 4 | Root of the raw HCP Pipelines output -- where stage 1 finds each session's first functional run (for its native reference grid) and resolves HCP warp fields (section 8). Stage 4 also uses it, but only to resample a plain `{subject}_impa.nii.gz` into `_impa_mni.nii.gz` -- skip it entirely if every classifier under `output_dir` uses `model.mnispace: true` (which writes `_impa_mni.nii.gz` directly, with no plain `_impa.nii.gz` ever produced), and stage 4 will just skip that resample step. |
| `group_gm_mask` | stage 1 only | The group-level MNI-space GM mask stage 1 resamples into each subject/session's native space. Not needed if your data is already MNI-space (`model.mnispace: true`, mask pointed straight at a shared MNI-space mask) and stage 1 never runs. |
| `output_dir` | stages 3, 4 | Where `mvpa_workflow.py`/`generate_report.py` write everything -- passed straight through as `--analysis-output-dir`. |
| `master_spreadsheet` | stages 2, 3, 4 | Where stage 2 writes `master_spreadsheet.csv` and every later stage reads it from. |
| `mni_template` | optional, stage 4 only | Reference grid for stage 4's importance-map resampling. Falls back to `$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz` (resolved after `module load fsl`) when omitted; moot entirely when `hcppipe_root` is also omitted, since that skips the resample step altogether. |

Every field is a full, already-resolved absolute path -- no `{subject}`/
`{session}` placeholders (those are per-file, not per-root) and no implicit
shared-prefix convenience the way a single hand-edited `ANALYSIS_ROOT` bash
variable used to provide; if you want that, build these paths with a
shared prefix in whatever generates your config, not in the pipeline
scripts themselves.

**Only the fields a given stage actually reads are required for it.**
`slurm/resolve_pipeline_config.sh` validates only the subset of fields the
*calling* stage script needs (each of `1_`-`4_` sets its own
`REQUIRED_PIPELINE_FIELDS` before sourcing it) -- an omitted field the
caller doesn't require just resolves to an empty string rather than
failing, so e.g. an all-MNI-space study (section 5's `model.mnispace`) that
never runs stage 1 and has nothing for stage 4 to resample can drop
`hcppipe_root`/`group_gm_mask` from its config entirely. A field a stage
*does* require still fails fast with a clear message if missing. There's
nothing left in `resolve_pipeline_config.sh` itself to hand-edit.

**Each of `1_`-`4_` is fully standalone given just its config-path
argument -- no dependency on `0_` having run first, an already-exported
`SCRIPTS_DIR`, or the job's working directory happening to be the repo
root.** That's *because* `scripts_dir` lives in the config: `sbatch` copies
a batch script into its own spool file before running it, so a stage
script's own location (`${BASH_SOURCE[0]}`) never points back at the real
repo the way it reliably does for `0_` (which is invoked directly via
`bash`, never through `sbatch`) -- and a stage script's working directory
is just wherever `sbatch` happened to be invoked from unless `--chdir` is
passed. Each of `1_`-`4_` therefore reads `pipeline.scripts_dir` out of its
config argument *before* it can even locate `resolve_pipeline_config.sh`,
via a small bootstrap block at the top of the script (see any of `1_`-`4_`
for the exact sequence), and passes it along as a plain shell variable from
there. Run any stage on its own, from anywhere:

```
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json
```

`0_` still resolves its own location (reliable for the reason above) to
`cd` to the repo root and build the `sbatch` paths it submits, but that's
purely for its own use -- it doesn't export anything the child jobs
depend on. Run it the same way, from anywhere:

```
bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh configs/my-study.json
```

`--output`/`--error` log paths are plain `#SBATCH` directives (no variable
substitution happens in them, since they're parsed before the script body
ever runs), so they always resolve relative to wherever `sbatch` was
actually invoked from -- `0_` `cd`s to the repo root before submitting so
its own jobs' logs land in the right place; submitting a stage script
directly from somewhere else just means its logs land there instead.

`0_` submits the four numbered jobs via `sbatch --parsable`, each depending
on the previous one via `--dependency=afterok` (which, for an array job,
only fires once *every* array task has succeeded):

| Stage | Script | Type |
|---|---|---|
| 1. Mask resample | `slurm/1_batch_resample_native_mask.sh` | per-subject/session array job (section 8) |
| 2. Master spreadsheet | `slurm/2_sbatch_generate_master_spreadsheet.sh` | single job (section 1) |
| 3. Classifier/decoding | `slurm/3_batch_run_mvpa_workflow.sh` | per-subject array job (section 6) -- runs whichever of `model.kfold_cv`/`model_conditions.testing`/`model_conditions.timecourse_decoding` are configured, each independently; also writes each subject's own single-subject report |
| 4. Group report | `slurm/4_sbatch_generate_report.sh` | single job (section 7); resamples every subject's importance map to MNI (section 8) before building the group report |

Every stage script can still be run standalone (e.g. to rerun just one stage
after fixing a subject-specific failure) with plain
`sbatch slurm/<script>.sh configs/my-study.json` -- the orchestrator only
adds the dependency chaining on top, it doesn't own any logic itself.

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
