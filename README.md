# MVPA decoding workflow

Decode what a person was doing or perceiving from their fMRI activity. Given
BIDS-style event timing files and preprocessed BOLD data, this pipeline trains
a classifier to tell your conditions apart (for example face vs. place, or
positive vs. negative trials), checks how well it generalizes, decodes the
condition continuously across each trial's timecourse, and writes per-subject
and group PDF reports. It replicates the approach of Kim et al. (2020,
*Nature Communications*); see [THEORY.md](THEORY.md) for the science and how
each output maps back to that paper.

You describe the study once, in a JSON config. Every script reads that same
file.

## What it does

```
events.tsv + BOLD files
        │  generate_master_spreadsheet.py
        ▼
master_spreadsheet.csv        one row per BOLD volume, labelled from the events
        │  restrict_master_spreadsheet.py      (optional: balance trial counts)
        ▼
        │  mvpa_workflow.py                    one run per subject
        ▼
cross-validation (model/)   held-out test (test/)   timecourse decoding (decoding/)
        │  generate_report.py
        ▼
PDF report (one subject, or the whole group)
```

## Quick start

You need Python 3 with the packages in `requirements.txt`, and for every run:
a BIDS-style `*_events.tsv` (columns `onset`, `duration`, `trial_type`) and its
matching `*_bold.nii.gz`. Details: [docs/input-data.md](docs/input-data.md).
Commands are run from the repo root.

```bash
pip install -r requirements.txt
```

**1. Write a config.** Copy an example and edit its paths and conditions; see
[Creating your config file](#creating-your-config-file) at the bottom.

**2. Build the master spreadsheet** (one for the whole dataset):

```bash
python workflows/generate_master_spreadsheet.py --config my-config.json
```

It is written to `event_extraction.output_file`.

**3. Check your conditions** (optional, but it catches typos early). With the
spreadsheet it also reports conditions that match no rows or overlap:

```bash
python utils/validate_model_config.py --config my-config.json \
    --master-spreadsheet master_spreadsheet.csv
```

**4. Balance trial counts across conditions** (optional; skip it if your
conditions already have equal numbers of trials):

```bash
python workflows/restrict_master_spreadsheet.py \
    --master-spreadsheet master_spreadsheet.csv \
    --restriction-config restriction.json --config my-config.json
```

Then add the file it prints to `model_conditions.training.trial_restriction`
in your config, or pass it to step 5 as `--trial-restriction`. Check the
`..._trial_restriction_summary.csv` it writes: a subject with no balanced set
is excluded. Details: [docs/trial-restriction.md](docs/trial-restriction.md).

**5. Run the classifier**, one subject per run (`--subject` is required; loop
over your subject IDs, or use the cluster route below):

```bash
python workflows/mvpa_workflow.py --subject 001 --config my-config.json \
    --master-spreadsheet master_spreadsheet.csv --analysis-output-dir ./out
```

Which analyses run depends on which sections your config has: cross-validation
(`model.kfold_cv`), a held-out test set (`model_conditions.testing`) and
timecourse decoding (`model_conditions.timecourse_decoding`), in any
combination. Details: [docs/running-the-workflow.md](docs/running-the-workflow.md).

**6. Make the report:**

```bash
python workflows/generate_report.py --analysis-output-dir ./out \
    --config my-config.json --master-spreadsheet master_spreadsheet.csv          # group
python workflows/generate_report.py --analysis-output-dir ./out --subject 001 \
    --config my-config.json --master-spreadsheet master_spreadsheet.csv          # one subject
```

Details: [docs/reports.md](docs/reports.md).

### On a SLURM cluster

Add a `pipeline` section to the config (paths for the dataset, output folder
and master spreadsheet; see [docs/slurm-pipeline.md](docs/slurm-pipeline.md)),
then submit the stages. Each takes the config path as its first argument and
sets up its own modules and environment:

```bash
sbatch slurm/2_sbatch_generate_master_spreadsheet.sh configs/my-study.json
sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh configs/my-study.json configs/restriction.json  # optional
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json \
    /path/to/master_spreadsheet_trial_restriction.csv     # drop the 2nd argument if you skipped 2b
sbatch slurm/4_sbatch_generate_report.sh configs/my-study.json
```

Stage 3 is an array job over subjects: set `--array` in the script to your
subject count. To chain stages 1-4 with dependencies in one command, use
`bash slurm/0_submit_mvpa_pipeline.sh configs/my-study.json`.

## What you get

Under `<analysis-output-dir>/<model.desc>/<subject>/`:

| Folder | Contents | Produced when |
|---|---|---|
| `model/` | Cross-validated accuracy, AUC and evidence, importance maps, per-trial predictions | `model.kfold_cv` is set |
| `test/` | Metrics and importance map from one fit on all training data, evaluated on the test set | `model_conditions.testing` is set |
| `decoding/` | The classifier's output for every decoded volume, by trial and condition | `model_conditions.timecourse_decoding` is set |

plus `<subject>_trial_pivot.csv`, a sanity check of how rows were labelled.
The report PDF summarizes these. Full file list:
[docs/running-the-workflow.md](docs/running-the-workflow.md#outputs).

## Try it on public data

[tutorial/README.md](tutorial/README.md) walks through the whole pipeline on
the Haxby et al. (2001) object-viewing dataset: download, minimal
preprocessing, config, every command and the real results.

## Documentation

Everything else is in [docs/](docs/README.md), one page per part:

| Page | Covers |
|---|---|
| [input-data](docs/input-data.md) | The files the pipeline expects and what it infers from them |
| [event-extraction](docs/event-extraction.md) | The master spreadsheet; finding BOLD files; `hemodynamic_lag` |
| [trial-restriction](docs/trial-restriction.md) | Balancing trial and TR counts across conditions |
| [model-conditions](docs/model-conditions.md) | The query language; defining training, testing and timecourse conditions |
| [model-config](docs/model-config.md) | Mask, feature selection, classifier, cross-validation, permutation tests |
| [running-the-workflow](docs/running-the-workflow.md) | What `mvpa_workflow.py` runs, the double-dipping guard, outputs |
| [reports](docs/reports.md) | The PDF reports |
| [resampling](docs/resampling.md) | MNI to native space and back |
| [slurm-pipeline](docs/slurm-pipeline.md) | The cluster pipeline and its `pipeline` config section |

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use synthetic data only, so they run the same locally and in CI
(`.github/workflows/tests.yml`, on every push and pull request to `main`).

## Creating your config file

One JSON file drives everything. The quickest route is to copy the closest
example and change the paths and conditions:

| Start from | When |
|---|---|
| [`tutorial/config-haxby.example.json`](tutorial/config-haxby.example.json) | Runnable on the tutorial data; train on some runs, test on others |
| [`examples/config-kfold.example.json`](examples/config-kfold.example.json) | Cross-validation within your data (split by run), with a `pipeline` section for the cluster |
| [`examples/config-generalization.example.json`](examples/config-generalization.example.json) | Train on one task, test on another |
| [`examples/config-generalization-template.example.json`](examples/config-generalization-template.example.json) | The same, as a fill-in-your-own-paths template |
| [`examples/config-restriction.example.json`](examples/config-restriction.example.json) | The separate small file that defines groups for [trial balancing](docs/trial-restriction.md) |

The main config has up to four sections:

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

  "pipeline": { "...": "cluster paths, only for the slurm/ scripts" }
}
```

| Section | You say | Reference |
|---|---|---|
| `event_extraction` | Where the events and BOLD files are, and the hemodynamic lag (seconds between an event and its BOLD peak) | [docs/event-extraction.md](docs/event-extraction.md) |
| `model_conditions` | Which trials count as which condition, as queries over the master spreadsheet's columns. `training` is required; add `testing` and `timecourse_decoding` for those analyses | [docs/model-conditions.md](docs/model-conditions.md) |
| `model` | Brain mask, feature selection, classifier, cross-validation, permutation test | [docs/model-config.md](docs/model-config.md) |
| `pipeline` | Study paths for the SLURM scripts. Not needed when you run the Python scripts directly | [docs/slurm-pipeline.md](docs/slurm-pipeline.md) |

The skeleton above is for orientation, not a complete working config: start
from an example. When you have edited it:

1. Build the master spreadsheet (step 2), then run
   `utils/validate_model_config.py` with `--master-spreadsheet` (step 3). It
   reports structural mistakes and conditions that match nothing.
2. Run one subject (step 5) before launching the whole dataset.
3. If you balance trials, add `model_conditions.training.trial_restriction`
   ([docs/trial-restriction.md](docs/trial-restriction.md)).

Config files you want to keep per study can live in `configs/`.
