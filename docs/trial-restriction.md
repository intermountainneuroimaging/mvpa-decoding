[← README](../README.md) · [all docs](README.md)

# Optional: force balanced training trials (`workflows/restrict_master_spreadsheet.py`)

An optional step between the master spreadsheet and `mvpa_workflow.py`. It
forces every subject to have exactly the same number of **training** trials --
and the same number of TRs per trial -- in each condition of interest, and
writes the selection as a small trial list. Nothing is rewritten: your
events.tsv files and the master spreadsheet stay exactly as they are. The
workflow uses the list only when the config names it (below), and only for
training and so `model.kfold_cv`; `model_conditions.testing` and
`timecourse_decoding` are left unbalanced.

```
python workflows/restrict_master_spreadsheet.py \
    --master-spreadsheet master_spreadsheet.csv \
    --restriction-config examples/config-restriction.example.json \
    --config mvpa_config.json            # event_extraction.hemodynamic_lag, and a coverage check
```

then, in the mvpa config:

```json
"model_conditions": {
  "training": {
    "conditions": { ... },
    "trial_restriction": "master_spreadsheet_trial_restriction.csv"
  }
}
```

On the cluster, `sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh
configs/my-study.json configs/restriction.json` runs the step with the same
module/conda setup as the other stages (the config supplies
`pipeline.scripts_dir` and `pipeline.master_spreadsheet`); it is not part of
`0_submit_mvpa_pipeline.sh`, so submit it yourself between stages 2 and 3.
Then give stage 3 the list either way:

```bash
# (a) per run, no config edit: optional second argument of stage 3
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json \
    /path/to/master_spreadsheet_trial_restriction.csv
# (b) or set model_conditions.training.trial_restriction in the config and submit as usual
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json
```

(a) is passed to `mvpa_workflow.py` as `--trial-restriction` and overrides (b)
if both are set. The same flag works when you run `mvpa_workflow.py` directly.

Leave `trial_restriction` out and nothing changes. The workflow never looks
for the file on its own, and it stops with a message if the file is missing,
was built for a different `hemodynamic_lag`, no longer matches the events
files (trial onsets differ), lacks the subject (excluded, see below), or was
built from a different master spreadsheet (its boldfiles aren't in it).

**The supplementary config** (`examples/config-restriction.example.json`)
defines the groups; queries are the same `column`/`match`/`and`/`or`/`not`
trees as `model_conditions` ([model_conditions](model-conditions.md)), evaluated against each *trial* -- a
retained events.tsv row, administrative events like fixation never count --
using its events.tsv columns plus `subject`/`session`/`task`/`run`/`boldfile`/
`eventfile`. A trial matching several groups goes to the first in config
order.

| Key | Meaning |
|---|---|
| `primary_groups` | `{name: query}`. Every subject gets exactly the same number of trials in each. Trials in no primary group are listed unchanged. |
| `secondary_groups` | `{name: query}`, optional. Balanced *within* each primary group. |
| `secondary_applies_to` | Optional list of primary group names that get the secondary split (default: all of them). Groups not listed are only equalized in total. |
| `seed` | Integer (default 0). Dropped trials are chosen by a generator seeded from this and the subject ID, so reruns give identical output. |
| `drop_strategy` | `"random"` (default), or `"spread_runs"`: drop from whichever run currently holds the most trials of that group, so the survivors stay evenly spread over runs (use this with a `per_run` `model.kfold_cv`). |

**What it does, per subject (all of that subject's runs pooled):**

1. *Trial counts.* Trials are dropped at random until every primary group has
   the same count and, where secondary groups apply, that count is split
   equally across them -- the largest balanced set that exists. Subject 001
   with 36 face (18 positive + 18 negative), 36 place (18 + 18) and 12 item
   trials ends with 12 face (6 + 6), 12 place (6 + 6), 12 item. A trial in a
   secondary-split group that matches no secondary group can't be balanced
   and is dropped.
2. *TR counts.* Every kept trial is limited to the TR count of the shortest
   kept trial of that subject: its **first** *n* TRs are used and the extra
   TRs are cut off the **end**. TRs are counted the way training selects them
   (the window `[onset + hemodynamic_lag, + duration)`, clipped to the scan),
   so `hemodynamic_lag` must match your analysis config. One trial cut short
   by the end of the scan therefore shortens them all.

A subject for whom no balanced set exists (some group, or some secondary
group within it, has no usable trials) is **excluded**: it gets no rows in the
list, is reported, and `mvpa_workflow.py` refuses to run it with the list
set -- it is never silently run unbalanced.

**Outputs** (next to the master spreadsheet unless `--output` is given):

| File | Contents |
|---|---|
| `<stem>_trial_restriction.csv` | The list `trial_restriction` points at: one row per trial training may use -- `subject, boldfile, trial_index, onset, trial_type, primary_group, secondary_group, n_trs, hemodynamic_lag`. Kept balanced trials carry their `n_trs`; trials in no primary group are listed with `n_trs` blank (whole window). A trial not listed is not used. |
| `<stem>_trial_restriction_summary.csv` | Per subject x primary x secondary group (plus an `(all)` row per subject): `n_trials_original`/`n_trials_restricted`, `trs_per_trial_restricted`, and `n_trs_original`/`n_trs_restricted` (TRs training would select). Excluded subjects appear with their `exclusion_reason`. |
| `<stem>_trial_restriction_audit.csv` | Every governed trial and its `status` (`kept`, `balance_drop`, `no_secondary_label`, `outside_scan`, `subject_excluded`). |

`trial_index` is the 1..N numbering of a run's retained events in onset order
that `label_conditions_with_lag` already uses. Because the list is keyed by
it, rerun the step whenever `hemodynamic_lag`, the group definitions, the
events files or the master spreadsheet change (the lag and trial onsets are
checked for you).

**Things to know**

- *Cover what the conditions select.* Trials in no primary group are listed
  untouched. If a `model_conditions.training` condition selects trials your
  primary-group query doesn't (say it adds a run restriction the primary
  group lacks), those stay unbalanced -- pass `--config` and the script
  prints a warning naming the condition and how many trials are uncovered.
- *Balance is per subject, across runs.* With `drop_strategy: "random"` a run
  can end up with few trials of a group; prefer `"spread_runs"` for
  per-run cross-validation.
- *Testing and timecourse decoding are not restricted* (for now), so with a
  held-out `testing` section the test trials are not balanced.
