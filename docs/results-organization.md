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
rules in one place; the notebooks and the ingest command all use it; the notebooks' own helpers are in `_interactive_notebooks/evidence_common.py`.

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
classifier): set `GROUP_CSV_PATH` to the model's `raw_data/group_decoding_results.csv` (the model folder is the folder above `raw_data/`; absolute, or relative to the results root) and the settings (baseline correction,
grouping, statistics window); the notebook reads `raw_data/group_decoding_results.csv` and writes everything into the
matching `decoding/...` folder under the right names. The last cell copies the saved notebook beside its results.

`operation_evidence_by_category.ipynb` is the one place for the operation classifier's `stats_and_figures`: it draws the cross-validated confusion
matrices and the raw-evidence time courses (saved on their own in `decoding/uncorrected/`), the evidence-by-category statistics, and the
importance maps on the cortical surface (saved in `importance_maps/`; only missing ones are drawn unless `REDRAW_IMPORTANCE_MAPS = True`,
and they need a nilearn version whose `plot_surf_stat_map` accepts `darkness`), and puts them all in `stats_and_figures_<N>bins.pdf`.
