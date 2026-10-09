[← README](../README.md) · [all docs](README.md)

# `model_conditions`

Read by `validate_model_config.py`. This defines, for each of three
sections (`training` -- required -- and `testing`/`timecourse_decoding`,
both optional), a set of named **conditions** -- the classifier's class labels --
each backed by a **query** that selects which `master_spreadsheet.csv` rows
belong to it.

## The query language

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

## Section by section

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
`model.kfold_cv` ([model config](model-config.md)) cross-validates entirely within
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

**`hemodynamic_lag` and which volumes `training`/`testing` actually use.**
Neither section's query directly selects `master_spreadsheet.csv` rows the
way `timecourse_decoding`'s does below -- `mvpa_workflow.py` ([running the workflow](running-the-workflow.md))
resolves them through `mvpa_common.label_conditions_with_lag` instead: for
each real event (administrative rows excluded, see [event_extraction](event-extraction.md)) whose own
`trial_type`/`task`/`run`/... match a condition's query, it shifts forward by
`event_extraction.hemodynamic_lag` ([event_extraction](event-extraction.md)) before selecting volumes --
`[onset + hemodynamic_lag, onset + duration + hemodynamic_lag)` -- since the
BOLD response to a real-world event peaks several seconds after it, not
during it. Those *shifted* volumes get that condition's `regressor_label`,
independent of whichever real (unshifted) event `master_spreadsheet.csv`
itself says is happening there at that moment. `model.kfold_cv` ([model config](model-config.md))
inherits this automatically, since it cross-validates entirely within
whatever `training` already resolved this way.

**`timecourse_decoding`** -- *(optional)* continuous, real-event decoding,
read from `master_spreadsheet.csv` like `training`/`testing`, but --
critically -- **never** through `label_conditions_with_lag`: it decodes every
volume continuously, real-time, with no `hemodynamic_lag` shift and no
exclusions. Always uses the one classifier fit on the complete `training` set
(never per-fold, even when
`model.kfold_cv` is also configured). Omit the whole section (not just leave
it empty) to skip timecourse decoding entirely -- no `decoding/` output
files at all, no extra runtime for that step, and `generate_report.py`'s
timecourse page is automatically skipped too (it already skips whenever it
finds no `decoding_results.csv` for any subject in scope, so there's nothing
extra to configure on the report side):

```json
"timecourse_decoding": {
  "conditions": {
    "face":  {"column": "trial_type", "match": "exact", "value": "face"},
    "house": {"column": "trial_type", "match": "exact", "value": "house"}
  },
  "trial_start_event": {"column": "trial_type", "match": "exact", "value": "view_stimulus"}
}
```

`conditions` now works at two levels, not one:

- **Which boldfiles are in scope.** A boldfile qualifies for continuous
  decoding if **any** of its real rows match **any** condition's query --
  e.g. a `{"column": "run", "match": "in", "values": [...]}` clause inside a
  condition still restricts decoding to just those runs, exactly as before,
  since `run`/`task` are constant across every row of one boldfile. A
  boldfile with no matching row at all (e.g. a run only ever used for
  `training`) is excluded entirely.
- **Which real `trial_type` values count as a scored category.** Once a
  boldfile qualifies, **every** BOLD volume in it is decoded -- nothing is
  subset or skipped *within* a qualifying boldfile. Rows are grouped into
  trials by `trial_start_event` (a single query, same shape as one entry of
  `conditions`): each real event that matches it starts a new trial, running
  until the next match (or the end of the run, for the last one) --
  `window_index` counts up from 0 at each `trial_start_event` match. A
  volume whose real `trial_type` matches none of `conditions` (a fixation
  frame, the `trial_start_event` cue itself if it isn't also a trained
  category, ITI, ...) is still decoded and still appears in
  `decoding_results.csv`, just with `regressor_label` blank, left for you to
  filter on afterward.

`conditions` itself is **optional** here -- omit it (or leave it an empty
object) and it falls back to `testing`'s own `conditions` (error if neither
is present; see `resolve_timecourse_conditions` in `utils/mvpa_common.py`).
This is the recommended default: `testing.conditions` already defines
"which real event is true to this trained category" for the held-out test
evaluation, and the timecourse page should normally score against that same
definition. Keeping two separately-written copies of the same query risks
them drifting apart with nothing to catch it -- exactly what happened in
practice on this project (a config's `timecourse_decoding.conditions`
matched the operate period, while `testing.conditions` correctly matched
the view period the classifier was actually trained/tested to detect, so
the timecourse page silently scored the wrong event). Set
`timecourse_decoding.conditions` explicitly only when you deliberately want
the timecourse page's "true condition" to mean something different from
`testing`'s -- e.g. scoring evidence for the originally-viewed category
during a later operate/manipulation period, per [THEORY.md](../THEORY.md).

**Marking trial starts explicitly instead of `trial_start_event`.** If your
events.tsv generation can emit it, add a row tagged exactly `trial_start`
(any onset/duration -- only its presence and onset matter) at the start of
every trial. `partition_into_trials` (`utils/mvpa_common.py`) prefers these
over `trial_start_event` automatically, **per boldfile** -- a boldfile with
any `trial_start`-tagged row uses those directly and never consults
`trial_start_event` at all; a boldfile with none falls back to
`trial_start_event` exactly as described above. `trial_start_event` stays
**required** either way (there's no way to know in advance whether a given
boldfile will have `trial_start` rows), but a dataset that tags every run
explicitly can point it at anything reasonable -- it will simply never be
consulted. A dataset can freely mix events.tsv files that do and don't tag
trial starts explicitly; each boldfile is resolved independently.

**Block-boundary markers end a trial early, automatically.** A trial
normally runs anchor-to-anchor, however long that gap turns out to be --
fine for an ordinary short inter-trial fixation/ITI (its evidence keeps
plotting as part of the preceding trial, which is what you want), but not
for an occasional long block-boundary rest period: since `generate_report.py`
broadcasts a trial's one overlay/regressor label across its *entire* window
so the line doesn't cut off early, a long rest period swept into whichever
condition happened to precede it would stretch that condition's timecourse
line noticeably farther than the others, purely because of which
condition's trials happen to sit at the end of a block. `partition_into_trials`
fixes this at the source: any real event matching `is_trial_end_marker`
(`utils/mvpa_common.py` -- the same hardcoded `start_block`/`end_block`/
`EndFixation` policy `is_excluded_trial_type` already uses, not a
per-project config option) closes the trial there instead of letting it run
through to the next anchor; volumes from that point onward are left outside
any active trial (`trial_index=0`, same convention as a boldfile's leading
pre-anchor frames). An ordinary short administrative event within a trial
(`fixation`/`trial_fixation`/`postrt`) does *not* trigger this -- it's
deliberately narrower than `is_excluded_trial_type`, so those keep
broadcasting as part of their trial's tail as before.

**`annotation_labels`** -- *(optional)* a friendly display name for a group
of mutually exclusive `trial_type` values, read only by `generate_report.py`
for the timecourse page's per-event annotations (see
`compute_event_markers`/`resolve_marker_label`). Without it, a trial
position whose real event is one of several mutually exclusive values (e.g.
one of several trained categories, or one of several operations) is labeled
by auto-joining the distinct values (`"bottle/cat/chair/face/..."`, one
`/`-joined group per position, truncated after 4 by default) -- often
correct but not always what you'd want to show:

```json
"annotation_labels": {
  "item": ["bottle", "cat", "chair", "face", "house", "scissors", "scrambledpix", "shoe"]
}
```

A group applies only when it **fully covers** the exact set of values
observed at that position (extra values in the group that never occur there
are fine; a value observed there but missing from the group disqualifies
it); the first matching group wins, in config order. A position where every
trial agrees on one single real value is never relabeled this way, even if
that value happens to appear in a configured group -- grouping only ever
replaces the auto-joined fallback for genuinely mixed positions.

Two cases are handled automatically, with no config needed: any observed
value containing `"nan"` (e.g. `view_nan`, from a malformed trial with a
missing/undefined item logged verbatim in events.tsv) is dropped before a
position's values are matched against `annotation_labels`, so a rare
malformed trial can't force the auto-joined fallback at an otherwise
consistent position; and a position whose real value(s) are *all*
administrative/excluded content (`is_excluded_trial_type` --
fixation/`trial_fixation`/`EndFixation`/`postrt`/...) is left unlabeled
(blank text, boundary still drawn) rather than printing a raw value like
`"trial_fixation"` as if it were a real condition.

**`overlay`** -- *(optional)* only read by `generate_report.py`, not by
either workflow script. Same name-to-query shape as `conditions` -- each
entry both **filters/labels events** (its query, exactly like a `conditions`
entry) **and controls how that category is drawn** on the timecourse page
([reports](reports.md)), via two further optional keys on the same entry:

- **`color`** -- an integer indexes the standard palette (`0`-`9`, wrapping);
  a string is passed straight through to matplotlib as a literal color
  (`"#1f77b4"`, `"red"`, ...). Omit it to auto-assign (see below).
- **`line_type`** -- an integer indexes `solid, dashed, dotted, dashdot`; a
  string is passed straight through to matplotlib (`"--"`, `"dashed"`, ...).
  Omit it and that category draws solid.
- **`group`** -- *(optional)* any label (string, number, ...); gathers that
  category together with every other category sharing the same `group`
  value into its own contiguous block of rows (one row per true condition),
  capped with its own legend right after it, before the next group's block
  starts. Once *any* overlay entry sets `group`, the single shared legend at
  the bottom of the page is replaced by one legend per distinct `group`
  value (first-seen, i.e. config declaration order) -- `color`/`line_type`
  still apply within a block exactly as above. Omit `group` entirely (the
  default) and every category stays in one block with one legend for the
  whole page, as before `group` existed.

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

**The same 4-condition/2-sub-condition split can instead go on separate
blocks, each with its own legend** rather than separate line styles sharing
one legend -- swap `line_type` for `group`:

```json
"overlay": {
  "1A": {"column": "trial_type", "match": "exact", "value": "cond1A", "color": 0, "group": "A"},
  "1B": {"column": "trial_type", "match": "exact", "value": "cond1B", "color": 0, "group": "B"},
  "2A": {"column": "trial_type", "match": "exact", "value": "cond2A", "color": 1, "group": "A"},
  "2B": {"column": "trial_type", "match": "exact", "value": "cond2B", "color": 1, "group": "B"}
}
```
For 2 true conditions this produces two blocks, each a 2-row x (number of
true conditions' evidence categories) grid: group "A"'s block (both true
conditions, showing only `1A`/`2A`) with its own legend right below it,
then group "B"'s block (both true conditions, showing only `1B`/`2B`) with
its own legend -- one report, not two separate ones, and `1A`/`2A` still
share a color with `1B`/`2B` across the two blocks. `color`/`line_type`/
`group` are independent and can combine on the same entry (e.g. still dash
*within* a group's block for a third factor).

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

When present, the timecourse page ([reports](reports.md)) overlays one line per overlay
category within each existing subplot, instead of a single line -- see that
section for how it changes the plot. Rows matching none of the overlay
queries are dropped from that plot only (a count is printed); everything
else about the pipeline -- the classifier, its evidence values,
`decoding_results.csv`/`summary_decoding_results.csv` -- is unaffected,
since `overlay` is evaluated entirely inside `generate_report.py` against
data the workflow scripts already wrote (the `color`/`line_type`/`group` keys
are likewise inert everywhere else -- `evaluate_query_node`/`validate_query_node`
only ever read the query keys they need, so they simply ignore all three).

## Running it

```
python utils/validate_model_config.py --config tutorial/config-haxby.example.json \
    --master-spreadsheet master_spreadsheet_haxby.csv
```

Without `--master-spreadsheet`, only the JSON structure is checked (valid
`match` types, regexes that actually compile, `trial_start_event` present,
etc.). With it, every condition's query (and `timecourse_decoding`'s
`trial_start_event`) is run against the real table and you additionally get:

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
