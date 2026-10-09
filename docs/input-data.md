[← README](../README.md) · [all docs](README.md)

# Input data: what the pipeline assumes

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
tutorial runs against -- see [tutorial/README.md](../tutorial/README.md) for
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
[Using preprocessed/derivative data](event-extraction.md#using-preprocessedderivative-data-eg-fmriprep) below --
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
