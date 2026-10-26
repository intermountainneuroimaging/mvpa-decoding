[← README](../README.md) · [all docs](README.md)

# Generating a report (`workflows/generate_report.py`)

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
| `--config` | Supplies `model.desc` (see above, when `--desc` is omitted) and `model_conditions.timecourse_decoding` (`trial_start_event`/conditions, and optionally `overlay` -- see [model_conditions](model-conditions.md)) for timecourse annotation either way. Without it (i.e. using `--desc` alone), the timecourse page still renders, just unannotated (and never split by overlay). |
| `--master-spreadsheet` | *(optional)* Needed alongside `--config` to compute each subject's TR and the real per-trial event timing (both derived from real data, not hardcoded) -- used to convert `window_index` to seconds and annotate the timecourse plot with every real event observed inside a trial. Without it, the x-axis stays in raw `window_index` units and annotation is skipped. |
| `--output` | *(optional)* Defaults to `<dir>/<desc>/report_<desc>.pdf` (group) or `<dir>/<desc>/<subject>/report_<subject>.pdf` (single-subject). |

**A "Data Independence Warning" page appears automatically, right after the
title page, whenever it has something to say.** If `mvpa_workflow.py`'s
double-dipping guard ([running the workflow](running-the-workflow.md)) skipped the held-out test evaluation,
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
([model config](model-config.md)) produced them, with no report-side configuration needed. A
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
the `viridis` colormap underneath. Accuracy's rows sum to 1 (the fraction of
that condition's trials predicted as each category); Evidence is the mean
`decision_evidence()` across those same trials -- by default normalized
`predict_proba()`/softmax, so its rows sum to 1 too, but with
`classifier.evidence: "sigmoid"` they are independent per-class sigmoids and
generally don't.

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
already in MNI space, set `model.mnispace: true` ([model config](model-config.md)) -- the
workflow script then writes each importance map directly as
`{subject}_impa_mni.nii.gz` (under `model/` and/or `test/`, whichever
families are configured), and `generate_report.py` reads that same
`model.mnispace` setting from `--config` to know to look for it, no
resampling step required. Otherwise, resample each subject's native-space
`impa` into a shared MNI grid after the fact -- via `hcp_resample.py
--direction native2mni` ([resampling](resampling.md)) -- and save the result as
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
([SLURM pipeline](slurm-pipeline.md)) runs the `hcp_resample.py` resampling automatically for every
subject before generating the report (skipped entirely for subjects whose
importance map is already named `_impa_mni.nii.gz`, i.e. ran with
`mnispace: true`) -- manual `hcp_resample.py` calls are only needed if
you're generating a report outside that pipeline.

**Which voxels differ from zero across subjects.** Whenever the group page
exists (MNI maps for at least 3 subjects) `generate_report.py` also tests every
voxel's weight against zero across subjects, per class
(`--importance-permutations`, default 10000; 0 skips it). Each subject's
feature selection zeroes the voxels it left out, so those zeros are treated as
*missing*, not as weights of 0: a subject contributes to a voxel only if its map is
non-zero there, a voxel is tested only if at least `--importance-min-subjects`
(default 10) subjects kept it, and the statistic is the plain mean weight of the
subjects that kept it. Its null distribution comes from random sign flipping of
whole subjects' maps (one flip is applied to all voxels of a subject, so spatial
correlation is preserved); p is two-sided with a +1 correction, so the smallest
possible p is 1/(n+1). A voxel is "significant" when p < `--importance-p-voxel`
(0.001) *and* it survives Benjamini-Hochberg FDR at `--importance-fdr-q`
(0.05; each class is its own family), optionally also in a cluster of at
least `--importance-min-cluster` voxels. Files, next to the PDF:
`{desc}_group_impa_mni_stats.nii.gz` (volumes: mean weight with zeros, mean
weight among subjects that kept the voxel, p, significant 0/1 -- each one volume per class --
then the number of subjects that kept the voxel; the `.json` beside it names them and records
the settings) and `{desc}_group_impa_signflip_summary.csv`. Details and limits
of the test (permutation p-values are coarse, which makes FDR conservative with few
voxels at the floor; conditional on selection) are in `utils/importance_stats.py`.
`_interactive_notebooks/operation_results.py --stats-nii ...` draws the significant voxels on the
cortical surface from that one small file.

**Importance maps that say how a voxel contributes (the paper's approach).**
A raw classifier weight does not say whether a voxel helps by being *more* or *less* active than
average, so `mvpa_workflow.py` also writes each subject's per-class mean z-scored activity next to the
weights (`{subject}_impa_act[_mni].nii.gz`), and the group report forms, per subject and class,
weight x activity where the signs agree (Kim et al. 2020; McDuff et al. 2009):
**BOLD activation evidence** (weight > 0, activity > 0: the voxel supports the class when its
activity is higher than average) and **BOLD deactivation evidence** (weight < 0, activity < 0: it
supports the class when its activity is lower). Weight and activity of opposite sign (activity
arguing against the class) and voxels the subject's feature selection left out are not shown.
Each subject's two maps are z-scored together, averaged over subjects, and the top
`--polarity-top-fraction` (0.05) of each group map is kept, clusters under
`--polarity-min-cluster` (10) voxels dropped and the result smoothed (`--polarity-fwhm`, 12 mm).
Written next to the PDF: `{desc}_group_impa_polarity_maps.nii.gz` (+ `.json` naming the volumes) and
`{desc}_group_impa_polarity_summary.csv`; `operation_results.py --polarity-nii ...` draws it on the
cortical surface. Subjects whose stage 3 predates the activity maps are left out (rerun stage 3 for them).

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

Both compose with `model_conditions.timecourse_decoding.overlay` ([model_conditions](model-conditions.md))
when it's configured -- each overlay category gets its own two SE bands,
plus a legend naming the categories and what the lighter band means. Each
category's actual color/line style come from `resolve_overlay_styles`
([model_conditions](model-conditions.md)'s `color`/`line_type`, per entry, with sensible auto-assigned
defaults for whichever is omitted). If any overlay entry also sets `group`,
categories sharing a `group` value are gathered into their own contiguous
block of rows, each block capped with its own legend right after it
(`resolve_timecourse_groups`) -- e.g. an operation x valence overlay can put
each valence in its own block with its own legend, all in one report,
rather than one shared legend for the whole page or a separate config/report
run per valence.

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
  (cross-validation hold-out sample detail, [model config](model-config.md)) concatenated
  together -- one row per held-out sample across every subject's folds,
  carrying `task`/`trial_type`/`run`/`fold`/`predicted_label`/`correct`/
  `evidence_<category>`.

Each is only written if at least one subject in scope actually has that
output (e.g. `{desc}_group_cv_results.csv` is skipped entirely if no
subject ran with `model.kfold_cv` configured).
