# Keeping the results organized

Every model's results live in one folder of the organized results tree
(`_interactive_notebooks/results_organized/`, or `$MVPA_RESULTS_ROOT`):

```
<project: clearvale | mindmem>/<classifier: category | operation>/<evidence: norm_proba | sigmoid>/<sorting: all_trials | restricted>/<roi>_<classifier settings>/
    group_summary.csv  group_cv_results.csv  report.pdf  model_config.json  README_model.md
    raw_data/group_decoding_results.csv
    importance_maps/        importance_group_mean / _signflip_stats / _polarity_maps (.nii.gz, .json), surface_*.png
    decoding/
      uncorrected/                                  confusion_matrices, timecourse_decoding, stats_and_figures.pdf (operation)
        stats_3TR-bins_fdr/                         stats_and_figures_<N>bins.pdf|.ipynb, <comparison>_<N>bins.csv, run_settings_<N>bins.json
      corrected/baseline-correction-viewoffset_TR4-6/grouping-per-category/stats_3TR-bins_fdr/  (same files)
```

`<roi>` is `vvps` or `gm`; `<N>bins` (or `<N>windows` for sliding windows) is the size of the FDR family
(7bins = the 3-TR bins from TR 7 on, 9bins = every bin). `_interactive_notebooks/results_layout.py` holds all of these
rules in one place; the notebooks, `operation_results.py` and the ingest command all use it.

## 1. New results downloaded from the cluster

Download the stage-4 folder of a model (the `<desc>_group_*` files and `report_<desc>.pdf`) together with the model's
config `.json`, then:

```bash
python3 _interactive_notebooks/results_layout.py ingest <download_folder>            # shows what would happen
python3 _interactive_notebooks/results_layout.py ingest <download_folder> --run      # copies it into the layout
```

The project, classifier type, evidence method, trial sorting, ROI and classifier settings are read from the config
(`--config`, or a config `.json` inside the download); any of them can be given explicitly (`--project`, `--classifier`,
`--evidence`, `--sorting`, `--roi`, `--model-type`). Files already in place are left alone, a different file at the same
destination is not overwritten unless `--overwrite`, unrecognised files are listed and left where they are, and
`MANIFEST.csv` and a `README_model.md` are added.

## 2. Analyses (notebooks)

`valence_evidence_by_operation.ipynb` (category classifiers) and `operation_evidence_by_category.ipynb` (operation
classifier): set `MODEL_DIR` to the model folder (relative to the results root) and the settings (baseline correction,
grouping, statistics window); the notebook reads `raw_data/group_decoding_results.csv` and writes everything into the
matching `decoding/...` folder under the right names. The last cell copies the saved notebook beside its results.

`python3 _interactive_notebooks/operation_results.py --model-dir <model folder>` writes the operation classifier's
confusion matrices, time courses, importance figures and `stats_and_figures.pdf` into the model folder.
