"""_interactive_notebooks/results_layout.py: where results go, how they are named, and ingesting a download."""

import json
import os

import pandas as pd
import pytest

from _interactive_notebooks import results_layout as layout


def category_config(evidence="sigmoid", restricted=False, mask="/data/Clearvale/masks/mask_vvps_temporal_3mm.nii.gz"):
    clf = {"name": "sklearn.linear_model.LogisticRegression",
           "params": {"penalty": "l2", "C": 50, "solver": "liblinear"}, "one_vs_rest": True, "evidence": evidence}
    if evidence != "sigmoid":
        clf = {"name": "sklearn.linear_model.LogisticRegression", "params": {"C": 0.5, "solver": "lbfgs"}}
    training = {"trial_restriction": "/data/Clearvale/x.csv"} if restricted else {}
    return {"event_extraction": {"bids_root": "/data/Clearvale/bids/"},
            "model": {"desc": "vvps_category_loc2WM_timecourse_liblin_onevrest_classifier", "mask": {"mask_pattern": mask}, "classifier": clf},
            "model_conditions": {"training": training, "testing": {}, "timecourse_decoding": {}}}


def operation_config():
    return {"event_extraction": {"bids_root": "/data/Clearvale/bids/"},
            "model": {"desc": "gm_operation_iti_liblin_onevrest_kfold_classifier",
                      "mask": {"mask_pattern": "/data/Clearvale/masks/bin_wager_gm_mask_3mm.nii.gz"},
                      "classifier": {"name": "sklearn.linear_model.LogisticRegression", "params": {"C": 50, "solver": "liblinear"},
                                     "one_vs_rest": True, "evidence": "sigmoid"}},
            "model_conditions": {"training": {}, "timecourse_decoding": {}}}


class TestModelInfo:
    def test_sigmoid_all_trials_category_model(self):
        info = layout.model_info_from_config(category_config())
        assert info == layout.ModelInfo("clearvale", "category", "sigmoid", "all_trials", "vvps_liblinear_onevrest_C50")

    def test_restricted_and_normalised_probability_multinomial(self):
        info = layout.model_info_from_config(category_config(evidence="norm_proba", restricted=True))
        assert (info.evidence, info.sorting, info.model) == ("norm_proba", "restricted", "vvps_multinomial_lbfgs_C0.5")

    def test_operation_model_is_gm_and_kfold_only(self):
        info = layout.model_info_from_config(operation_config())
        assert info.classifier == "operation" and info.model == "gm_liblinear_onevrest_C50_kfold"

    def test_project_must_be_clear_or_overridden(self):
        cfg = json.loads(json.dumps(category_config()).replace("Clearvale", "somewhere"))
        with pytest.raises(ValueError):
            layout.model_info_from_config(cfg)
        assert layout.model_info_from_config(cfg, project="mindmem").project == "mindmem"

    def test_roi_label(self):
        assert layout.roi_label("/m/mask_vvps_temporal_3mm.nii.gz") == "vvps"
        assert layout.roi_label("/m/bin_wager_gm_mask_3mm.nii.gz") == "gm"

    def test_path_under_root(self, tmp_path):
        info = layout.model_info_from_config(category_config())
        assert info.path(str(tmp_path)) == os.path.join(str(tmp_path), "clearvale", "category", "sigmoid", "all_trials", "vvps_liblinear_onevrest_C50")


class TestFolderNames:
    def test_baseline_folders(self):
        assert layout.baseline_folder((1, 2, 3)) == "baseline-correction-viewonset_TR1-3"
        assert layout.baseline_folder((4, 5, 6)) == "baseline-correction-viewoffset_TR4-6"
        assert layout.baseline_folder((2, 3, 4)) == "baseline-correction_TR2-4"

    def test_grouping_folders(self):
        assert layout.grouping_folder(("object", "operation", "valence")) == "grouping-per-operation-category-valence"
        assert layout.grouping_folder(("object", "valence")) == "grouping-per-category-valence"
        assert layout.grouping_folder(("object",), balance="weighted") == "grouping-per-category_balanced-preceding-weighted"

    def test_window_folders(self):
        assert layout.window_folder(False, 3) == "stats_3TR-bins_fdr"
        assert layout.window_folder(False, 1) == "stats_1TR-bins_fdr"
        assert layout.window_folder(True, 3) == "stats_sliding-3TR_fdr"

    def test_decoding_dir(self):
        m = "/r/model"
        assert layout.decoding_dir(m) == "/r/model/decoding/uncorrected/stats_3TR-bins_fdr"
        assert layout.decoding_dir(m, True, (4, 5, 6), ("object",), sliding=True) == \
            "/r/model/decoding/corrected/baseline-correction-viewoffset_TR4-6/grouping-per-category/stats_sliding-3TR_fdr"

    def test_stat_file_names(self):
        assert layout.stat_file("pos_vs_neg", 7) == "pos_vs_neg_7bins.csv"
        assert layout.stat_file("stats_and_figures", 18, sliding=True, ext=".pdf") == "stats_and_figures_18windows.pdf"
        assert layout.stat_file("pos_vs_neg", None) == "pos_vs_neg.csv"

    def test_family_size_counts_windows_with_a_corrected_p(self):
        table = pd.DataFrame({"operation": ["a"] * 4 + ["b"] * 4, "p_fdr": [None, None, 0.1, 0.2, None, 0.3, 0.4, 0.5]})
        assert layout.family_size(table) == 3
        assert layout.family_size(table.drop(columns="p_fdr")) is None


class TestIngest:
    def _download(self, tmp_path, cfg):
        d = tmp_path / "download"; d.mkdir()
        desc = cfg["model"]["desc"]
        (d / "config.json").write_text(json.dumps(cfg))
        for suffix in ("_group_summary.csv", "_group_cv_results.csv", "_group_decoding_results.csv", "_group_mean_impa_mni.nii.gz"):
            (d / f"{desc}{suffix}").write_text("x,y\n1,2\n")
        (d / f"report_{desc}.pdf").write_bytes(b"%PDF")
        (d / "mystery.txt").write_text("?")
        return d

    def test_plan_puts_every_file_where_the_layout_says(self, tmp_path):
        cfg = category_config()
        info, desc, plan, unmatched = layout.plan_ingest(str(self._download(tmp_path, cfg)), root=str(tmp_path / "root"))
        dests = {os.path.relpath(dst, info.path(str(tmp_path / "root"))) for _, dst, _ in plan}
        assert dests == {"model_config.json", "group_summary.csv", "group_cv_results.csv", "raw_data/group_decoding_results.csv",
                         "importance_maps/importance_group_mean.nii.gz", "report.pdf"}
        assert unmatched == ["mystery.txt"] and all(action == "copy" for _, _, action in plan)

    def test_run_copies_writes_readme_and_manifest_and_is_repeatable(self, tmp_path):
        root = tmp_path / "root"
        download = self._download(tmp_path, category_config())
        info, _, plan, _ = layout.ingest(str(download), root=str(root), run=True)
        model = info.path(str(root))
        assert os.path.isfile(os.path.join(model, "raw_data", "group_decoding_results.csv"))
        assert os.path.isfile(os.path.join(model, "README_model.md")) and os.path.isfile(root / "MANIFEST.csv")
        assert os.path.isfile(download / "config.json")                       # copied, not moved
        _, _, again, _ = layout.ingest(str(download), root=str(root), run=True)
        assert all(action == "same" for _, _, action in again)

    def test_a_different_file_at_a_destination_is_not_overwritten_by_default(self, tmp_path):
        root = tmp_path / "root"; download = self._download(tmp_path, category_config())
        info, _, _, _ = layout.ingest(str(download), root=str(root), run=True)
        target = os.path.join(info.path(str(root)), "group_summary.csv")
        (download / (category_config()["model"]["desc"] + "_group_summary.csv")).write_text("new,content\n")
        _, _, plan, _ = layout.ingest(str(download), root=str(root), run=True)
        assert open(target).read() == "x,y\n1,2\n" and any(action == "differs" for _, _, action in plan)
        layout.ingest(str(download), root=str(root), run=True, overwrite=True)
        assert open(target).read() == "new,content\n"

    def test_dry_run_changes_nothing(self, tmp_path):
        root = tmp_path / "root"
        layout.ingest(str(self._download(tmp_path, category_config())), root=str(root))
        assert not root.exists()
