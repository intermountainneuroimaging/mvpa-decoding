[← README](../README.md) · [all docs](README.md)

# `event_extraction`

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
| `hemodynamic_lag` | Seconds added to a matched event's `onset` before selecting volumes for **training/testing** (and, transitively, `model.kfold_cv`) -- read here, but applied later, by `mvpa_workflow.py` ([model_conditions](model-conditions.md)/[running the workflow](running-the-workflow.md)), not by this script. `generate_master_spreadsheet.py` itself ignores it entirely; `model_conditions.timecourse_decoding` never applies it either. |
| `output_file` | Where the resulting table is written. Override with `--output`. |
| `expected_events_file` | *(optional)* Path to a template of expected `trial_type` values -- see below. Override with `--expected-events`. |
| `derivatives_root` | *(optional)* Directory to search under for BOLD files, if different from `bids_root` -- e.g. a separate fMRIPrep `derivatives/` tree. Omit the key (or set it to `null`) to inherit `bids_root`. Setting it to `""` is **not** the same as omitting it -- an explicit empty string is honored literally (resolves to the current working directory) and prints a warning, since that's almost never what's intended. See [Using preprocessed/derivative data](#using-preprocessedderivative-data-eg-fmriprep). |
| `bold_glob` | *(optional)* Needed whenever BOLD filenames don't follow the default lookup (match on `sub`/`ses`/`task`/`run` tokens + `"bold"` in the filename) -- e.g. fMRIPrep's `desc-`/`space-` suffixes, or when `derivatives_root` returns more than one match per run. A format string resolved relative to `derivatives_root`, with `{subject}`/`{session}`/`{task}`/`{run}` placeholders -- plus **any other BIDS entity found in the events filename is available under its own raw key**, e.g. `{dir}` for a `dir-pa`/`dir-ap` entity, `{acq}` for `acq-*`, etc. No code change needed for a new entity; if it's in the filename, it's usable in `bold_glob`. |

## `expected_events.json` (optional, separate file)

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

## Running it

```
python workflows/generate_master_spreadsheet.py --config tutorial/config-haxby.example.json
```

Output (`master_spreadsheet_haxby.csv`) -- one row per BOLD volume,
`0..n_frames-1`, for every processed run. Nothing is excluded and no
`hemodynamic_lag` shift is applied here -- every real event (administrative
rows like `fixation` included) gets its own frames, verbatim onset/duration,
and a volume covered by no event at all gets a blank `trial_type`/`onset`/
`duration`/`event_index` rather than being skipped:

| Column | Meaning |
|---|---|
| `subject`, `session`, `task`, `run` | *Inferred* from the events filename. |
| `volume_of_interest` | *Computed*: `0..n_frames-1`, the BOLD file's own frame count -- every volume gets a row, always. |
| `trial_type`, `onset`, `duration` | Verbatim from whichever real events.tsv row's `[onset, onset+duration)` span covers this volume (later-onset event wins on overlap) -- blank if none does. |
| `event_index` | *Computed*: a contiguous `1..N` id (in onset order) over every real event in this run, administrative rows included, so rows produced by the same event can be grouped back together -- blank for volumes covered by no event. This is the raw event id `model_conditions.timecourse_decoding` groups trials from ([model_conditions](model-conditions.md)/[running the workflow](running-the-workflow.md)); `training`/`testing` re-derive their *own*, `hemodynamic_lag`-shifted, exclusion-filtered volume selection from it at run time instead of reading it directly (see `trial_index` in [running the workflow](running-the-workflow.md)). |
| `boldfile`, `eventfile` | Resolved source file paths, for traceability/sorting. |
| *(varies)* | Any other BIDS entity found in the filename, e.g. `dir` -- *inferred*, present only if that entity appears in your filenames (the tutorial data has none). |

Example real output row (from `tutorial/haxby-data`; `session` is empty since
this dataset has no `ses-` entity):

```
subject  session  volume_of_interest  trial_type  event_index  onset  duration  task           run  boldfile                                                                                eventfile
1                 65                  house       51           160.0  0.5       objectviewing  1    tutorial/haxby-data/derivatives/sub-1/func/..._run-01_desc-preproc_bold.nii.gz  tutorial/haxby-data/sub-1/func/..._run-01_events.tsv
```

## Administrative/non-trial events

`master_spreadsheet.csv` itself excludes nothing -- a fixed set of
administrative/non-trial `trial_type` values is instead ignored later, only
as *candidates* for `training`/`testing` ([model_conditions](model-conditions.md)/[running the workflow](running-the-workflow.md)): `mvpa_common.py`'s
`EXCLUDED_TRIAL_TYPE_EXACT`/`EXCLUDED_TRIAL_TYPE_SUBSTRINGS` constants (not
exposed as a config option on purpose -- edit them directly to change the
list):

| Match | Excludes |
|---|---|
| exact: `start_block`, `end_block` | structural block markers |
| substring (case-insensitive): `fixation` | `trial_fixation`, `BaselineFixation`, `EndFixation`, etc. |
| substring (case-insensitive): `postrt` | post-response-time administrative events |

`rest_block` is **not** excluded -- it's a real experimental condition in
some designs, not a structural marker. `model_conditions.timecourse_decoding`
doesn't apply this exclusion at all -- administrative rows are decoded and
appear in `decoding_results.csv` like any other real event, left for you to
filter on afterward ([model_conditions](model-conditions.md)).

## Using preprocessed/derivative data (e.g. fMRIPrep)

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

`model.mask.mask_pattern` (see [model config](model-config.md)) is unrelated to either
of these -- it's always a full path in its own right, not resolved against
`bids_root`/`derivatives_root`. `examples/config-generalization-template.example.json`
is a template showing the `event_extraction` fields above filled in.

### Diagnosing a "no matching BOLD file found" error

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
