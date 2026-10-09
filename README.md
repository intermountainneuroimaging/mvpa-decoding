# MVPA decoding workflow

Decode what a person was doing, seeing or thinking from their fMRI activity.
Given BIDS-style event timing files and preprocessed BOLD data, this pipeline
trains a classifier to tell your conditions apart (for example face vs. place,
or positive vs. negative trials), measures how well it works, follows the
classifier's output moment by moment through each trial, and writes per-subject
and group PDF reports. It replicates the approach of Kim et al. (2020,
*Nature Communications*); see [THEORY.md](THEORY.md) for the science and how
each output maps back to that paper.

You describe the whole study once, in a JSON **config file**. Every script in
the pipeline reads that same file. Settings are referred to below by their
path in the config file, for example `event_extraction.hemodynamic_lag` means
the `hemodynamic_lag` setting inside the `event_extraction` section.

> [!WARNING]
> ## READ YOUR CONFIG FILE, LINE BY LINE, BEFORE YOU EXECUTE ANYTHING
>
> **Every line of the config file matters, and every line controls the
> workflow.** The config file decides which events count as which condition, how
> the brain data are timed and masked, how the classifier is built, what is
> held out for testing, which analyses are carried out, and where the results
> are saved. A wrong value usually does **not** stop the pipeline with an
> error: it carries on and produces results that look perfectly plausible and
> are wrong.
>
> **Do not copy and paste a config file from another study or another
> classifier without understanding EVERY LINE of it.** Settings that are right
> for someone else's data are silently wrong for yours: the task names and
> trial-type patterns that define a condition, the acquisitions chosen for
> training and testing, the hemodynamic lag, the mask and whether it is in MNI
> space, the classifier settings, the balancing list, and the output name
> (`model.desc`), which, if reused, overwrites earlier results.
>
> Before you trust any result, for every setting in your file you should be able
> to say what it does and why it has the value it has. If you cannot, look it up
> in the [documentation](docs/README.md) first. [Creating your config
> file](#creating-your-config-file) at the bottom of this page has a checklist.

## What it does

### The idea

fMRI records a pattern of activity across many voxels at every time point (one
**volume** per **TR**). If a brain region carries information about a
condition, the pattern it produces during that condition differs from the
pattern during another condition, and a classifier can learn the difference.
The pipeline does three things with that idea:

- **Cross-validation** (`model.kfold_cv`) asks how well the classifier works on
  data it was not trained on, by holding out part of the training data in turn
  (for example one acquisition at a time) and predicting it from the rest. It
  needs only one dataset.
- **A held-out test** (`model_conditions.testing`) trains one classifier on all
  the training data and scores it on a separate test set, which can be a
  different task (for example train on a localizer task, test on a working
  memory task). This measures how well what was learned generalizes.
- **Timecourse decoding** (`model_conditions.timecourse_decoding`) applies that
  same classifier to every volume of every acquisition in turn, and reports its
  output across each trial, so you can see when information about a condition
  rises and falls.

Each of the three is carried out only if its section is present in your config
file, and any combination is allowed. The classifier is fit separately for each subject.
All results are saved as plain CSV and NIfTI files, and the report summarizes
them in a PDF.

### How the pieces fit together

```
events.tsv + BOLD files            (one pair for every acquisition)
        │  generate_master_spreadsheet.py
        ▼
master_spreadsheet.csv             one row per volume, labelled from the events
        │  restrict_master_spreadsheet.py       (optional: balance the trial counts)
        ▼
        │  mvpa_workflow.py                     once for each subject
        ▼
cross-validation (model/)    held-out test (test/)    timecourse decoding (decoding/)
        │  generate_report.py
        ▼
PDF report (one subject, or the whole group)
```

1. **The master spreadsheet** is one table for the whole dataset, with a row for
   every volume of every acquisition. Each row says what was happening at that
   moment (`trial_type`, `onset`, `duration`), who and which scan it came from
   (`subject`, `session`, `task`, `run`, `boldfile`, `eventfile`), and where the
   events came from. The repetition time and number of volumes are read from
   each BOLD file's header, never typed in.
2. **Conditions are defined by queries on that table.** In the config file's
   `model_conditions` section you name each condition and give a small rule that
   picks its rows, such as "`task` is `loc` and `trial_type` contains `face`".
   The same rules language is used everywhere, so a condition can be as simple
   or specific as you need.
3. **The hemodynamic lag** (`event_extraction.hemodynamic_lag`, in seconds) is
   the delay between an event and the peak of its BOLD response. When the
   classifier is trained and tested, each trial's volumes are taken from that
   many seconds after the event began, not during it. Timecourse decoding does
   not apply the lag, since it follows the data in real time.
4. **Balancing** (optional) removes trials at random so every condition
   contributes the same number of examples to training, and trims trials to a
   common length. It never edits your event files or the master spreadsheet; it
   writes a separate list that the classifier training reads.
5. **The classifier** is any scikit-learn estimator you name in the config file
   (`model.classifier`), applied to the voxels inside a brain mask
   (`model.mask`) after a voxel selection step (`model.featureSelection`).
   Optional extras include a permutation test for significance
   (`model.permutation_test`).
6. **The report** collects accuracy, AUC, evidence, importance maps and the
   timecourse plots into a PDF for one subject or the whole group.

### Terms used in these documents

| Term | Meaning |
|---|---|
| **Acquisition** | One continuous scan: one BOLD file and its events file. BIDS names it with a `run-` entity, and the master spreadsheet stores it in a column named `run`. These documents call the unit an acquisition and avoid using "run" for anything else. |
| **Trial** | One event row of an events file (such as one face presentation), not a fixation or other administrative event. |
| **Volume / TR** | One 3D brain image, taken every repetition time (TR). |
| **Condition** | A class the classifier learns to tell apart, defined by a query in your config file. |
| **Evidence** | The classifier's output for each class on a sample. By default it is a probability that sums to 1 across classes; see [docs/model-config.md](docs/model-config.md) for the unnormalized option. |
| **Importance map** | A brain map of the voxel weights the classifier relied on. |

## Quick start

You need Python 3 with the packages in `requirements.txt`. For every
acquisition you also need a BIDS-style `*_events.tsv` (with columns `onset`,
`duration` and `trial_type`) and its matching `*_bold.nii.gz`; see
[docs/input-data.md](docs/input-data.md) for exactly what is expected. Execute
every command from the repository's top folder.

```bash
pip install -r requirements.txt
```

### 1. Create a config file

Start from the example closest to your study, but treat it only as a template to
read and rewrite, never as something to copy and go: you must understand every
line before you use it. See [Creating your config file](#creating-your-config-file)
at the bottom of this page. Everything after this step reads the file you make
here, so a mistake here carries through every later step.

### 2. Build the master spreadsheet

```bash
python workflows/generate_master_spreadsheet.py --config my-config.json
```

This finds every events file under the folder set by `event_extraction.bids_root`
in your config file, locates the matching BOLD file, and writes one row per
volume. The table is saved to the path set by `event_extraction.output_file` in
your config file. Events files with no matching BOLD file (or with more than
one) are skipped with a printed warning, so read the messages it prints before
moving on. Details: [docs/event-extraction.md](docs/event-extraction.md).

### 3. Check your conditions

```bash
python utils/validate_model_config.py --config my-config.json \
    --master-spreadsheet master_spreadsheet.csv
```

This reads the conditions defined under `model_conditions` in your config file.
It reports malformed queries, and, because you gave it the spreadsheet, any
condition that matches no rows and any pair of conditions that overlap. This
step is optional but catches the most common mistakes (a typo in a trial type,
a condition that silently matches nothing) in seconds.

### 4. Balance trials across conditions

*Optional; for studies with heavily imbalanced training data between conditions,
restricting the input dataset to a balanced number of exemplars can help improve
model training.*

You describe the groups to balance in a small separate file (see
[`examples/config-restriction.example.json`](examples/config-restriction.example.json)):
a set of *primary* groups that must each end up with the same number of trials
for every subject, and optionally *secondary* groups (for example positive and
negative) that are kept equal inside each primary group. The step then drops
trials at random to reach those counts and shortens every kept trial to the
same number of TRs.

```bash
python workflows/restrict_master_spreadsheet.py \
    --master-spreadsheet master_spreadsheet.csv \
    --restriction-config restriction.json --config my-config.json
```

The step writes a trial list and a summary table next to the master
spreadsheet. Read the summary: a subject who cannot be balanced (for example one
with no trials at all in a group) is excluded and listed there with the reason.
To use the list, set `model_conditions.training.trial_restriction` in your
config file to the list's path, or pass it to step 5 with `--trial-restriction`.
Without that setting nothing is restricted. Details:
[docs/trial-restriction.md](docs/trial-restriction.md).

### 5. Train and evaluate the classifier

The classifier is trained and evaluated on a per-subject basis: each execution
of the command below handles one subject, and `--subject` is required. Repeat
the command for every subject, using the ID exactly as it appears in the
spreadsheet's `subject` column.

```bash
python workflows/mvpa_workflow.py --subject 001 --config my-config.json \
    --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
```

To go through several subjects in one shell session:

```bash
for s in 001 002 003; do
    python workflows/mvpa_workflow.py --subject $s --config my-config.json \
        --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
done
```

For each subject the script builds the training set from `model_conditions.training`,
fits the classifier, and then carries out whichever analyses your config file
asks for: cross-validation if `model.kfold_cv` is present, a held-out test if
`model_conditions.testing` is present, and timecourse decoding if
`model_conditions.timecourse_decoding` is present. It also guards against
evaluating on data the classifier was trained on, and tells you when it skips
an analysis for that reason. Details:
[docs/running-the-workflow.md](docs/running-the-workflow.md).

### 6. Make the report

```bash
# group report: every subject found under <analysis-output-dir>/<model.desc>/
python workflows/generate_report.py --analysis-output-dir ./out \
    --config my-config.json --master-spreadsheet master_spreadsheet.csv

# report for a single subject
python workflows/generate_report.py --analysis-output-dir ./out --subject 001 \
    --config my-config.json --master-spreadsheet master_spreadsheet.csv
```

The report reads the folder named by `model.desc` in your config file, so pass
the same config file you trained with. The PDF holds the accuracy and AUC
figures, confusion-style matrices of accuracy and evidence, the annotated
timecourse plots, and the importance maps, and a set of CSV tables is saved
next to it for your own analysis. Details: [docs/reports.md](docs/reports.md).

### On a SLURM cluster

Add a `master` section to your config file with the study's paths (dataset
roots, output folder, master spreadsheet location; see
[docs/slurm-pipeline.md](docs/slurm-pipeline.md)), then submit the stages in
order. Each stage script takes the config file path as its first argument and
loads its own modules and environment, so you can submit any of them from any
folder:

```bash
# build the master spreadsheet (once)
sbatch slurm/2_sbatch_generate_master_spreadsheet.sh configs/my-study.json

# optional: balance the trials (once; see step 4)
sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh configs/my-study.json configs/restriction.json

# train and evaluate the classifier: an array job, one task per subject
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json \
    /path/to/master_spreadsheet_trial_restriction.csv   # omit this argument if you skipped balancing

# build the group report (once)
sbatch slurm/4_sbatch_generate_report.sh configs/my-study.json
```

Before submitting stage 3, set `--array` inside the script to your number of
subjects. Wait for each stage to finish before submitting the next, or let
`bash slurm/0_submit_mvpa_pipeline.sh configs/my-study.json` chain stages 1 to 4
for you with the right dependencies. Each stage writes a log to `logs/`; that is
the first place to look if something fails.

## Outputs

Every step writes plain CSV, JSON or NIfTI files you can open in your own
software. This section lists what each step produces, where it goes, and what
is inside. In file names, `<subject>` is the subject ID, `<desc>` is the
`model.desc` setting from your config file (made safe for use as a folder
name), `{N}` is a fold number, and `[_mni]` is added to importance-map names when
`model.mnispace` is `true` in your config file.

### Step 2: the master spreadsheet

One CSV for the whole dataset, saved at the path set by `event_extraction.output_file`
in your config file. It has one row for every volume of every acquisition:

| Column | Meaning |
|---|---|
| `volume_of_interest` | The volume's number within its acquisition, counting from 0 |
| `trial_type`, `onset`, `duration` | The event taking place during that volume, copied from the events file (blank if none) |
| `event_index` | A number identifying that event within the acquisition, so rows from the same event can be grouped |
| `subject`, `session`, `task`, `run` | Where the volume came from, read from the events file's name (`run` holds the acquisition number) |
| `boldfile`, `eventfile` | The full paths of the BOLD file and events file |

Any other BIDS entity in the events file's name (such as `dir`) becomes an extra
column. The spreadsheet is the input to every later step.

### Step 3: checking conditions

Nothing is written. The result is the text the command prints.

### Step 4: balancing trials (optional)

Three files, written next to the master spreadsheet with the spreadsheet's name
as a prefix:

| File | What is in it |
|---|---|
| `<name>_trial_restriction.csv` | The list that training reads (set `model_conditions.training.trial_restriction` in your config file to its path). One row for every trial the classifier may use: `subject`, `boldfile`, `trial_index`, `onset`, `trial_type`, `primary_group`, `secondary_group`, `n_trs` (the number of TRs kept for the trial, blank meaning all of them) and the `hemodynamic_lag` it was built for |
| `<name>_trial_restriction_summary.csv` | For every subject and group: how many trials and TRs there were before balancing and how many remain, and whether the subject is included or excluded, with the reason if excluded |
| `<name>_trial_restriction_audit.csv` | Every balanced trial and what happened to it (`kept`, dropped to balance the counts, dropped for having no secondary label, dropped for falling outside the scan, or its subject was excluded) |

### Step 5: the classifier, one folder per subject

Everything is written under `<analysis-output-dir>/<desc>/<subject>/`, where the
output folder is the `--analysis-output-dir` you pass on the command line:

```
<subject>_trial_pivot.csv                        always

model/                                           only if model.kfold_cv is set
  <subject>_kfold_folds.json
  <subject>_fold{N}_model_results_{metric}.csv
  <subject>_fold{N}_impa[_mni].nii.gz
  <subject>_fold{N}_permutation_test.csv         only if model.permutation_test is set
  <subject>_model_results_{metric}.csv
  <subject>_impa[_mni].nii.gz
  <subject>_cv_results.csv

test/                                            only if model_conditions.testing is set
  <subject>_model_results_{metric}.csv
  <subject>_impa[_mni].nii.gz
  <subject>_permutation_test.csv                 only if model.permutation_test is set

decoding/                                        only if model_conditions.timecourse_decoding is set
  <subject>_decoding_results.csv
  <subject>_summary_decoding_results.csv

<subject>_double_dipping_report.json             only if the safeguard skipped or changed an analysis
```

What the files contain:

| File | What is in it |
|---|---|
| `<subject>_trial_pivot.csv` | A sanity check written before anything is trained. One row for every trial in the master spreadsheet, tagged with the training condition (and testing condition) it matched, and, with cross-validation, one `fold{N}_split` column per fold saying whether the trial was `train`, `test` or neither in that fold. Use it to confirm your conditions and folds are what you intended |
| `…_model_results_accuracy.csv` | A conditions-by-conditions table. Each row is a true condition and each column a predicted condition; the value is the fraction of that condition's samples given that prediction, so every row sums to 1 and the diagonal is the per-condition accuracy |
| `…_model_results_evidence.csv` | The same layout, but each value is the average evidence the classifier gave a column's condition for samples of the row's condition. By default evidence is a normalized probability, so rows sum to 1; see [docs/model-config.md](docs/model-config.md) for the unnormalized option |
| `…_model_results_auc.csv` | One AUC value for each condition (how well the evidence for that condition separates its samples from the rest) |
| `…_model_results_metadata.csv` | Overall accuracy (`total_scores`), the number of voxels in the mask (`whole_voxels`), how many were selected (`selected_voxels`), and the percentage selected (`feature_percent`) |
| `…_impa[_mni].nii.gz` | An importance map: a brain image with one volume for each condition, giving the weight the classifier put on every voxel (zero for voxels that were not selected). With cross-validation there is one for each fold plus the average across folds |
| `…_permutation_test.csv` | The result of the permutation test (set by `model.permutation_test`), which estimates how often chance alone would score as well. With cross-validation there is one for each fold; the folds are not pooled |
| `<subject>_kfold_folds.json` | Which acquisitions were used to train and which were held out in each fold, so the split can always be checked |
| `<subject>_cv_results.csv` | Cross-validation at the trial level: one row for each held-out sample across all folds, with its `fold`, the trial's own columns (`task`, `trial_type`, `run`, `trial_index`, `regressor_label`, and so on), the `predicted_label`, whether it was `correct`, the `evidence_<condition>` values, and that fold's voxel-selection counts. `volume_of_interest` is the event's real position and `volume_of_interest_withlag` is the lag-shifted volume actually used |
| `<subject>_decoding_results.csv` | Timecourse decoding: one row for every decoded volume, with `trial_index`, `window_index` (the volume's position within its trial, counting from 0), the trial's condition (`regressor_label`, blank where the volume is not scored), `predicted_label`, `correct`, and the `evidence_<condition>` values |
| `<subject>_summary_decoding_results.csv` | The same decoding results averaged over trials for each `window_index` and condition, ready to plot |
| `<subject>_double_dipping_report.json` | A record of any analysis the safeguard skipped or changed because its data overlapped the training data. The report PDF shows it as a warning |

`model/` belongs only to cross-validation and `test/` only to the held-out test,
so a subject can have either, both or neither folder depending on your config
file. Full detail on every file is in
[docs/running-the-workflow.md](docs/running-the-workflow.md#outputs).

### Step 6: reports

| File | What is in it |
|---|---|
| `<analysis-output-dir>/<desc>/report_<desc>.pdf` | The group report (the path can be changed with `--output`): accuracy and AUC across subjects, the accuracy and evidence matrices, the annotated timecourse plots, and group importance maps. Cross-validation and held-out test results appear side by side wherever both exist |
| `<analysis-output-dir>/<desc>/<subject>/report_<subject>.pdf` | The report for one subject, made with `--subject` |
| `<desc>_group_summary.csv` | Written beside a group report: one row for each subject and analysis, with every metric laid out in its own column |
| `<desc>_group_decoding_results.csv` | Written beside a group report: every subject's `decoding_results.csv` joined into one table, which is what the interactive notebooks in `_interactive_notebooks/` read |
| `<desc>_group_cv_results.csv` | Written beside a group report: every subject's `cv_results.csv` joined into one table |
| `<desc>_group_mean_impa_mni.nii.gz` | Written beside a group report when subjects' importance maps are in MNI space: their average |

A group file is written only if at least one subject has that output. The
details of each report page are in [docs/reports.md](docs/reports.md).

## Try it on public data

[tutorial/README.md](tutorial/README.md) takes you through the whole pipeline on
the Haxby et al. (2001) object-viewing dataset: downloading it, a minimal
preprocessing pass, the config file, every command and the actual results. It
is the quickest way to see each stage working before you set up your own study.

## Documentation

The detailed reference is in [docs/](docs/README.md), one page for each part of
the pipeline:

| Page | Covers |
|---|---|
| [input-data](docs/input-data.md) | The files the pipeline expects and what it works out from them |
| [event-extraction](docs/event-extraction.md) | Building the master spreadsheet, finding BOLD files, the hemodynamic lag |
| [trial-restriction](docs/trial-restriction.md) | Balancing the number of trials and TRs across conditions |
| [model-conditions](docs/model-conditions.md) | The query language, and defining training, testing and timecourse conditions |
| [model-config](docs/model-config.md) | Brain mask, voxel selection, classifier, cross-validation, permutation tests, evidence |
| [running-the-workflow](docs/running-the-workflow.md) | What `mvpa_workflow.py` does for each subject, the double-dipping guard, every output file |
| [reports](docs/reports.md) | The PDF reports |
| [resampling](docs/resampling.md) | Moving brain maps between MNI and native space |
| [slurm-pipeline](docs/slurm-pipeline.md) | The cluster pipeline and the `master` config section |

## Testing the code

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use synthetic data only, so they behave the same on any machine and in
continuous integration (`.github/workflows/tests.yml`, triggered by every push
and pull request to `main`).

## Creating your config file

> [!WARNING]
> **Every line of this file matters. Do not paste in a config file from another
> study or classifier and move on.** Read each line, find what it does in the
> reference pages below, and decide its value for *your* data. Settings carried
> over unchanged from a similar file are the most common way to get results that
> look fine and are wrong.

One JSON file describes the whole study. The examples below are good places to
start reading from, but they are written for other datasets: use them to see the
shape of a file and which settings exist, then set every value yourself.

| Start from | When it fits |
|---|---|
| [`tutorial/config-haxby.example.json`](tutorial/config-haxby.example.json) | You want a complete working example: train on some acquisitions, test on others (used by the tutorial) |
| [`examples/config-kfold.example.json`](examples/config-kfold.example.json) | You want cross-validation within your data, holding out one acquisition at a time, with a `master` section for the cluster |
| [`examples/config-generalization.example.json`](examples/config-generalization.example.json) | You want to train on one task and test on another |
| [`examples/config-generalization-template.example.json`](examples/config-generalization-template.example.json) | The same, written as a fill-in-your-own-paths template |
| [`examples/config-restriction.example.json`](examples/config-restriction.example.json) | The separate small file that defines groups for [trial balancing](docs/trial-restriction.md) (step 4) |

The config file has up to four sections. This skeleton shows the shape, not a
complete working file, so start from one of the examples above:

```json
{
  "event_extraction": {
    "bids_root": "/path/to/bids",
    "events_glob": "**/*_events.tsv",
    "hemodynamic_lag": 4.6,
    "output_file": "master_spreadsheet.csv"
  },

  "model_conditions": {
    "training": {
      "conditions": {
        "face":  {"column": "trial_type", "match": "regex", "value": ".*face.*"},
        "place": {"column": "trial_type", "match": "regex", "value": ".*place.*"}
      }
    }
  },

  "model": {
    "desc": "my_classifier",
    "mask": {"mask_pattern": "/path/to/sub-{subject}/masks/mask.nii.gz"},
    "classifier": {"name": "sklearn.linear_model.LogisticRegression",
                   "params": {"C": 0.5, "max_iter": 10000}},
    "kfold_cv": {"strategy": "per_run"}
  },

  "master": { "...": "cluster paths, only used by the slurm/ scripts" }
}
```

What each section is for:

| Section | What you set in it | Reference |
|---|---|---|
| `event_extraction` | Where the events files are (`bids_root`, `events_glob`), where the BOLD files are if they are somewhere else (`derivatives_root`, `bold_glob`), the hemodynamic lag, and where to save the master spreadsheet (`output_file`) | [docs/event-extraction.md](docs/event-extraction.md) |
| `model_conditions` | The conditions, as queries on the master spreadsheet's columns. `training` is required. Add `testing` for a held-out test and `timecourse_decoding` for timecourse decoding | [docs/model-conditions.md](docs/model-conditions.md) |
| `model` | `desc` (the name of the output folder), the brain mask, voxel selection, the classifier, and optionally cross-validation (`kfold_cv`) and a permutation test (`permutation_test`) | [docs/model-config.md](docs/model-config.md) |
| `master` | The study's absolute paths for the SLURM scripts. Not needed if you execute the Python scripts directly | [docs/slurm-pipeline.md](docs/slurm-pipeline.md) |

Two settings are worth a moment of thought before you start. `model.kfold_cv.strategy`
chooses how data is held out: `per_run` leaves one acquisition out at a time (the
value is spelled `per_run` because it follows the BIDS name), while other
strategies group acquisitions or let you choose the held-out ones yourself.
`model.desc` is the name of the folder your results go in, so give each variant
of a classifier its own name to avoid overwriting earlier results.

**Checklist before you trust a config file.** Go through it from top to bottom
and confirm each of these for your own data:

- **Paths**: every folder, file pattern and glob points at *your* data and
  masks, and the `{subject}`, `{session}` and similar placeholders are right.
- **Conditions**: each condition's query selects exactly the trials you mean, and
  nothing else (the task names and `trial_type` patterns are specific to a study).
- **Training and testing**: the acquisitions and tasks in `training` and
  `testing` are the ones you intend, and testing does not share acquisitions with
  training.
- **Timing**: `event_extraction.hemodynamic_lag` suits your design and TR.
- **Mask and space**: `model.mask` is the mask you intend, and `model.mnispace`
  correctly says whether your data are in MNI space.
- **Classifier**: `model.classifier` and its settings, `model.featureSelection`,
  and any cross-validation (`model.kfold_cv`) or permutation test
  (`model.permutation_test`) are what you intend.
- **Balancing**: if you use `model_conditions.training.trial_restriction`, the
  list was built with this config file's lag and matches these conditions.
- **Output name**: `model.desc` is new, so you do not overwrite earlier results.

When your config file is ready:

1. Build the master spreadsheet (step 2), then check the conditions with
   `utils/validate_model_config.py` and the `--master-spreadsheet` option (step 3).
   It reports structural mistakes and conditions that match nothing.
2. Try a single subject (step 5) before starting the whole dataset.
3. If you balance trials, set `model_conditions.training.trial_restriction`
   ([docs/trial-restriction.md](docs/trial-restriction.md)).

Config files you want to keep for a particular study can be stored in `configs/`.
