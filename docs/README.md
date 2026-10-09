[← README](../README.md)

# Documentation

The main [README](../README.md) is the short version: what the workflow is for and the
steps to run it. These pages are the reference for each part, one per config
section or script. Read only the ones you need.

| Page | Covers | Config section / script |
|---|---|---|
| [input-data.md](input-data.md) | What files the pipeline expects (events + BOLD), what it infers from them | -- |
| [event-extraction.md](event-extraction.md) | Building the master spreadsheet; BOLD discovery, derivatives, `hemodynamic_lag` | `event_extraction`, `generate_master_spreadsheet.py` |
| [trial-restriction.md](trial-restriction.md) | Optional: force balanced trial and TR counts for training | restriction config, `restrict_master_spreadsheet.py` |
| [model-conditions.md](model-conditions.md) | The query language; defining training / testing / timecourse conditions | `model_conditions`, `validate_model_config.py` |
| [model-config.md](model-config.md) | Mask, feature selection, classifier, k-fold CV, permutation tests, double-dipping override | `model` |
| [running-the-workflow.md](running-the-workflow.md) | What `mvpa_workflow.py` runs, the double-dipping guard, every output file | `mvpa_workflow.py` |
| [reports.md](reports.md) | PDF reports (single subject and group), importance maps | `generate_report.py` |
| [resampling.md](resampling.md) | Moving maps between MNI and native space | `utils/hcp_resample.py` |
| [slurm-pipeline.md](slurm-pipeline.md) | Running everything on a SLURM cluster; the `master` config section | `slurm/*` |

Other documents: [THEORY.md](../THEORY.md) (scientific background and how each
output maps to Kim et al., 2020), [tutorial/README.md](../tutorial/README.md)
(a complete walkthrough on public data, with real output).

## How the pieces fit

One JSON config is read by every script via `--config`. Its sections map to
the stages: `event_extraction` builds the master spreadsheet,
`model_conditions` says which rows (trials) count as which condition, and
`model` says how to fit and evaluate the classifier. Shared logic (BIDS
filename parsing, the query language, the classification and decoding
primitives) lives in `utils/mvpa_common.py`.

`mvpa_workflow.py` runs up to three independent, config-driven analyses,
each only when its section is present: `model.kfold_cv` cross-validates
within `model_conditions.training` (output under `model/`);
`model_conditions.testing` fits on all of training and evaluates on a
separate test set (output under `test/`); `model_conditions.timecourse_decoding`
decodes every volume of every run with that same full-training fit (output
under `decoding/`). Any combination is valid; see
[running-the-workflow.md](running-the-workflow.md).

Everything in these pages is grounded in `tutorial/config-haxby.example.json`,
a complete config that runs end to end on `tutorial/haxby-data/` (Haxby et al.
2001, OpenNeuro ds000105; downloaded by `tutorial/preprocess_haxby.sh`, not
checked in). `examples/config-generalization-template.example.json` has the
same shape as a fill-in-your-own-paths template.
