"""generate_report.py: subject-scope discovery + file layout + small CSV
loaders. Fake directory trees built under tmp_path -- no dependency on real
mvpa_workflow.py output."""

import json

import numpy as np
import pandas as pd
import pytest

from workflows.generate_report import (
    list_subject_dirs,
    subject_paths,
    has_fold_files,
    fold_paths,
    load_scalar_csv,
    load_labeled_csv,
    infer_categories,
    compile_group_summary,
    compile_group_decoding,
    summarize_raw_for_timecourse,
    load_annotation_info,
    resolve_desc,
    resolve_group_impa_mni,
    resolve_mnispace,
)


def _make_subject(tmp_path, desc, subject, with_model_dir=True, with_test_dir=False):
    base = tmp_path / desc / subject
    if with_model_dir:
        (base / "model").mkdir(parents=True)
    if with_test_dir:
        (base / "test").mkdir(parents=True, exist_ok=True)
    return base


# =====================================================
# list_subject_dirs
# =====================================================

class TestListSubjectDirs:
    def test_explicit_subject_found(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        result = list_subject_dirs(str(tmp_path), "desc1", subject="01")
        assert result == ["01"]

    def test_explicit_subject_not_found_raises(self, tmp_path):
        (tmp_path / "desc1").mkdir()
        with pytest.raises(SystemExit):
            list_subject_dirs(str(tmp_path), "desc1", subject="99")

    def test_no_subject_lists_all_with_model_dir_sorted(self, tmp_path):
        _make_subject(tmp_path, "desc1", "02")
        _make_subject(tmp_path, "desc1", "01")
        _make_subject(tmp_path, "desc1", "03", with_model_dir=False)  # neither model/ nor test/ -- excluded
        result = list_subject_dirs(str(tmp_path), "desc1")
        assert result == ["01", "02"]

    def test_test_dir_only_subject_is_included(self, tmp_path):
        # training + testing configured, no model.kfold_cv -- no model/ dir at all
        _make_subject(tmp_path, "desc1", "01", with_model_dir=False, with_test_dir=True)
        result = list_subject_dirs(str(tmp_path), "desc1")
        assert result == ["01"]

    def test_no_subject_dirs_at_all_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            list_subject_dirs(str(tmp_path), "missing_desc")


# =====================================================
# subject_paths
# =====================================================

class TestSubjectPaths:
    def test_kfold_paths_under_model_dir(self):
        paths = subject_paths("/out", "desc1", "01")
        assert paths["kfold_total"] == "/out/desc1/01/model/01_model_results_total_scores.csv"
        assert paths["kfold_auc"] == "/out/desc1/01/model/01_model_results_auc.csv"
        assert paths["kfold_accuracy"] == "/out/desc1/01/model/01_model_results_accuracy.csv"
        assert paths["kfold_evidence"] == "/out/desc1/01/model/01_model_results_evidence.csv"
        assert paths["kfold_impa"] == "/out/desc1/01/model/01_impa.nii.gz"
        assert paths["kfold_impa_mni"] == "/out/desc1/01/model/01_impa_mni.nii.gz"

    def test_test_paths_under_test_dir(self):
        paths = subject_paths("/out", "desc1", "01")
        assert paths["test_total"] == "/out/desc1/01/test/01_model_results_total_scores.csv"
        assert paths["test_auc"] == "/out/desc1/01/test/01_model_results_auc.csv"
        assert paths["test_accuracy"] == "/out/desc1/01/test/01_model_results_accuracy.csv"
        assert paths["test_evidence"] == "/out/desc1/01/test/01_model_results_evidence.csv"
        assert paths["test_impa"] == "/out/desc1/01/test/01_impa.nii.gz"
        assert paths["test_impa_mni"] == "/out/desc1/01/test/01_impa_mni.nii.gz"

    def test_decoding_paths_unaffected_by_kfold_or_test(self):
        paths = subject_paths("/out", "desc1", "01")
        assert paths["decoding"] == "/out/desc1/01/decoding/01_summary_decoding_results.csv"
        assert paths["decoding_raw"] == "/out/desc1/01/decoding/01_decoding_results.csv"

    def test_mnispace_true_points_both_impa_families_at_the_mni_filename(self):
        paths = subject_paths("/out", "desc1", "01", mnispace=True)
        assert paths["kfold_impa"] == "/out/desc1/01/model/01_impa_mni.nii.gz"
        assert paths["test_impa"] == "/out/desc1/01/test/01_impa_mni.nii.gz"
        # coincides with kfold_impa_mni/test_impa_mni -- the workflow wrote the
        # MNI-confirmed file directly, so no separate hcp_resample.py step is
        # needed for cross-subject group averaging to kick in
        assert paths["kfold_impa"] == paths["kfold_impa_mni"]
        assert paths["test_impa"] == paths["test_impa_mni"]


# =====================================================
# has_fold_files / fold_paths
# =====================================================

class TestFoldFiles:
    def test_has_fold_files_false_when_none_exist(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        assert has_fold_files(str(tmp_path), "desc1", "01") is False

    def test_has_fold_files_true_and_fold_paths_covers_each_fold(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        for fid in (1, 2):
            (base / "model" / f"01_fold{fid}_model_results_total_scores.csv").write_text("0.5")

        assert has_fold_files(str(tmp_path), "desc1", "01") is True

        folds = fold_paths(str(tmp_path), "desc1", "01")
        assert sorted(folds.keys()) == [1, 2]
        assert folds[1]["kfold_total"] == str(base / "model" / "01_fold1_model_results_total_scores.csv")
        assert folds[1]["kfold_impa"] == str(base / "model" / "01_fold1_impa.nii.gz")
        assert folds[2]["kfold_auc"] == str(base / "model" / "01_fold2_model_results_auc.csv")
        # no per-fold decoding anymore -- timecourse decoding is never fold-based
        assert "decoding" not in folds[1]
        assert "decoding_raw" not in folds[1]

    def test_fold_paths_mnispace_true_uses_mni_filename(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        (base / "model" / "01_fold1_model_results_total_scores.csv").write_text("0.5")
        folds = fold_paths(str(tmp_path), "desc1", "01", mnispace=True)
        assert folds[1]["kfold_impa"] == str(base / "model" / "01_fold1_impa_mni.nii.gz")


# =====================================================
# resolve_group_impa_mni
# =====================================================

class TestResolveGroupImpaMni:
    def test_no_subjects_have_mni_map(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        _make_subject(tmp_path, "desc1", "02")
        available, missing = resolve_group_impa_mni(str(tmp_path), "desc1", ["01", "02"])
        assert available == {}
        assert missing == ["01", "02"]

    def test_some_subjects_have_full_test_mni_map(self, tmp_path):
        base01 = _make_subject(tmp_path, "desc1", "01", with_test_dir=True)
        _make_subject(tmp_path, "desc1", "02")
        (base01 / "test" / "01_impa_mni.nii.gz").write_text("fake")

        available, missing = resolve_group_impa_mni(str(tmp_path), "desc1", ["01", "02"])
        assert available == {"01": (str(base01 / "test" / "01_impa_mni.nii.gz"), "held-out-test")}
        assert missing == ["02"]

    def test_all_subjects_have_full_test_mni_map(self, tmp_path):
        for sub in ("01", "02"):
            base = _make_subject(tmp_path, "desc1", sub, with_test_dir=True)
            (base / "test" / f"{sub}_impa_mni.nii.gz").write_text("fake")

        available, missing = resolve_group_impa_mni(str(tmp_path), "desc1", ["01", "02"])
        assert sorted(available.keys()) == ["01", "02"]
        assert all(fam == "held-out-test" for _, fam in available.values())
        assert missing == []

    def test_falls_back_to_kfold_cv_mni_map_when_full_test_absent(self, tmp_path):
        # subject has no test/ MNI map at all, but does have a kfold model/ one
        base = _make_subject(tmp_path, "desc1", "01")
        (base / "model" / "01_impa_mni.nii.gz").write_text("fake")

        available, missing = resolve_group_impa_mni(str(tmp_path), "desc1", ["01"])
        assert available == {"01": (str(base / "model" / "01_impa_mni.nii.gz"), "CV")}
        assert missing == []

    def test_prefers_full_test_over_cv_kfold_when_both_present(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01", with_test_dir=True)
        (base / "model" / "01_impa_mni.nii.gz").write_text("fake-cv")
        (base / "test" / "01_impa_mni.nii.gz").write_text("fake-full")

        available, missing = resolve_group_impa_mni(str(tmp_path), "desc1", ["01"])
        assert available == {"01": (str(base / "test" / "01_impa_mni.nii.gz"), "held-out-test")}


# =====================================================
# load_scalar_csv / load_labeled_csv
# =====================================================

class TestLoaders:
    def test_load_scalar_csv(self, tmp_path):
        path = tmp_path / "total_scores.csv"
        np.savetxt(path, [0.75], delimiter=",", fmt="%.6f")
        assert load_scalar_csv(str(path)) == pytest.approx(0.75)

    def test_load_labeled_csv_indexed_by_category(self, tmp_path):
        path = tmp_path / "auc.csv"
        pd.DataFrame({"auc": [0.7, 0.8]}, index=["face", "place"]).to_csv(path)
        df = load_labeled_csv(str(path))
        assert df.index.tolist() == ["face", "place"]
        assert df["auc"].tolist() == [0.7, 0.8]


# =====================================================
# infer_categories
# =====================================================

class TestInferCategories:
    def test_reads_categories_from_kfold_auc_csv(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        pd.DataFrame({"auc": [0.7, 0.8]}, index=["face", "place"]).to_csv(base / "model" / "01_model_results_auc.csv")

        categories = infer_categories(str(tmp_path), "desc1", ["01"])
        assert categories == ["face", "place"]

    def test_falls_back_to_test_auc_csv_when_no_kfold(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01", with_test_dir=True)
        pd.DataFrame({"auc": [0.6, 0.9]}, index=["face", "place"]).to_csv(base / "test" / "01_model_results_auc.csv")

        categories = infer_categories(str(tmp_path), "desc1", ["01"])
        assert categories == ["face", "place"]

    def test_returns_empty_list_when_no_auc_csv_found(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        categories = infer_categories(str(tmp_path), "desc1", ["01"])
        assert categories == []


# =====================================================
# compile_group_summary
# =====================================================

def _write_family_csvs(base, subdir, subject, total, auc, acc, evi, categories):
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / f"{subject}_model_results_total_scores.csv", [total], delimiter=",", fmt="%.6f")
    pd.DataFrame({"auc": auc}, index=categories).to_csv(d / f"{subject}_model_results_auc.csv")
    pd.DataFrame(acc, index=categories, columns=categories).to_csv(d / f"{subject}_model_results_accuracy.csv")
    pd.DataFrame(evi, index=categories, columns=categories).to_csv(d / f"{subject}_model_results_evidence.csv")


class TestCompileGroupSummary:
    def test_one_row_per_subject_per_family(self, tmp_path):
        categories = ["face", "place"]
        base01 = _make_subject(tmp_path, "desc1", "01", with_test_dir=True)
        _write_family_csvs(base01, "model", "01", 0.7, [0.8, 0.75], [[0.9, 0.1], [0.2, 0.8]], [[0.85, 0.15], [0.25, 0.75]], categories)
        _write_family_csvs(base01, "test", "01", 0.6, [0.65, 0.7], [[0.7, 0.3], [0.4, 0.6]], [[0.65, 0.35], [0.45, 0.55]], categories)

        base02 = _make_subject(tmp_path, "desc1", "02")
        _write_family_csvs(base02, "model", "02", 0.65, [0.72, 0.68], [[0.8, 0.2], [0.3, 0.7]], [[0.78, 0.22], [0.32, 0.68]], categories)

        summary = compile_group_summary(str(tmp_path), "desc1", ["01", "02"])

        assert sorted(zip(summary["subject"], summary["family"])) == [
            ("01", "CV"), ("01", "held-out-test"), ("02", "CV"),
        ]

        row01_cv = summary[(summary["subject"] == "01") & (summary["family"] == "CV")].iloc[0]
        assert row01_cv["total_accuracy"] == pytest.approx(0.7)
        assert row01_cv["auc_face"] == pytest.approx(0.8)
        assert row01_cv["auc_place"] == pytest.approx(0.75)
        assert row01_cv["accuracy_face_face"] == pytest.approx(0.9)
        assert row01_cv["accuracy_place_face"] == pytest.approx(0.2)
        assert row01_cv["evidence_face_place"] == pytest.approx(0.15)

        row02_cv = summary[(summary["subject"] == "02") & (summary["family"] == "CV")].iloc[0]
        assert row02_cv["total_accuracy"] == pytest.approx(0.65)

        # subject 02 has no test/ family at all -- no row for it
        assert not ((summary["subject"] == "02") & (summary["family"] == "held-out-test")).any()

    def test_no_subjects_have_any_results(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        summary = compile_group_summary(str(tmp_path), "desc1", ["01"])
        assert summary.empty

    def test_includes_voxel_footprint_columns(self, tmp_path):
        categories = ["face", "place"]
        base = _make_subject(tmp_path, "desc1", "01")
        _write_family_csvs(base, "model", "01", 0.7, [0.8, 0.75], [[0.9, 0.1], [0.2, 0.8]], [[0.85, 0.15], [0.25, 0.75]], categories)
        np.savetxt(base / "model" / "01_model_results_whole_voxels.csv", [17806], delimiter=",", fmt="%.6f")
        np.savetxt(base / "model" / "01_model_results_selected_voxels.csv", [42], delimiter=",", fmt="%.6f")
        np.savetxt(base / "model" / "01_model_results_feature_percent.csv", [0.2359], delimiter=",", fmt="%.6f")

        summary = compile_group_summary(str(tmp_path), "desc1", ["01"])
        row = summary.iloc[0]
        assert row["whole_voxels"] == pytest.approx(17806)
        assert row["selected_voxels"] == pytest.approx(42)
        assert row["feature_percent"] == pytest.approx(0.2359)

    def test_voxel_footprint_columns_absent_when_files_missing(self, tmp_path):
        categories = ["face", "place"]
        base = _make_subject(tmp_path, "desc1", "01")
        _write_family_csvs(base, "model", "01", 0.7, [0.8, 0.75], [[0.9, 0.1], [0.2, 0.8]], [[0.85, 0.15], [0.25, 0.75]], categories)

        summary = compile_group_summary(str(tmp_path), "desc1", ["01"])
        assert "whole_voxels" not in summary.columns


# =====================================================
# compile_group_decoding
# =====================================================

class TestCompileGroupDecoding:
    def test_concatenates_every_subject_decoding_raw(self, tmp_path):
        for subject, n_rows in (("01", 2), ("02", 3)):
            d = tmp_path / "desc1" / subject / "decoding"
            d.mkdir(parents=True)
            pd.DataFrame({
                "subject": [subject] * n_rows,
                "window_index": list(range(n_rows)),
                "regressor_label": ["face"] * n_rows,
                "evidence_face": [0.5] * n_rows,
            }).to_csv(d / f"{subject}_decoding_results.csv", index=False)

        combined = compile_group_decoding(str(tmp_path), "desc1", ["01", "02"])
        assert len(combined) == 5
        assert sorted(combined["subject"].unique()) == ["01", "02"]
        assert (combined[combined["subject"] == "02"]["window_index"] == [0, 1, 2]).all()

    def test_subject_missing_decoding_output_is_skipped(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")  # no decoding/ dir at all
        d = tmp_path / "desc1" / "02" / "decoding"
        d.mkdir(parents=True)
        pd.DataFrame({"subject": ["02"], "window_index": [0], "evidence_face": [0.5]}).to_csv(
            d / "02_decoding_results.csv", index=False
        )

        combined = compile_group_decoding(str(tmp_path), "desc1", ["01", "02"])
        assert list(combined["subject"]) == ["02"]

    def test_no_subjects_have_decoding_output(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        combined = compile_group_decoding(str(tmp_path), "desc1", ["01"])
        assert combined.empty


# =====================================================
# summarize_raw_for_timecourse
# =====================================================

class TestSummarizeRawForTimecourse:
    def test_groups_by_window_index_and_regressor_label_without_overlay(self):
        raw = pd.DataFrame({
            "window_index": [0, 0, 0, 0],
            "regressor_label": ["face", "face", "face", "face"],
            "evidence_face": [0.8, 0.6, 0.4, 0.2],
            "evidence_place": [0.2, 0.4, 0.6, 0.8],
        })
        result = summarize_raw_for_timecourse(raw)

        assert "overlay_label" not in result.columns
        assert set(result.columns) >= {"window_index", "regressor_label", "evidence_face", "evidence_face_se", "evidence_place"}
        row = result.iloc[0]
        assert row["evidence_face"] == pytest.approx(0.5)  # mean(0.8, 0.6, 0.4, 0.2)

    def test_trial_to_trial_se_is_std_error_across_trials(self):
        # 4 trials, sample std = 0.2582, se = std/sqrt(4)
        raw = pd.DataFrame({
            "window_index": [0, 0, 0, 0],
            "regressor_label": ["face", "face", "face", "face"],
            "evidence_face": [0.8, 0.6, 0.4, 0.2],
        })
        result = summarize_raw_for_timecourse(raw)
        expected_se = pd.Series([0.8, 0.6, 0.4, 0.2]).std(ddof=1) / (4 ** 0.5)
        assert result.iloc[0]["evidence_face_se"] == pytest.approx(expected_se)

    def test_se_is_zero_for_a_single_trial(self):
        raw = pd.DataFrame({"window_index": [0], "regressor_label": ["face"], "evidence_face": [0.8]})
        result = summarize_raw_for_timecourse(raw)
        assert result.iloc[0]["evidence_face_se"] == 0.0

    def test_groups_by_window_index_regressor_label_overlay_label(self):
        raw = pd.DataFrame({
            "window_index": [0, 0, 0, 0],
            "regressor_label": ["face", "face", "face", "face"],
            "trial_type": ["maintain_face", "maintain_face", "suppress_face", "suppress_face"],
            "evidence_face": [0.8, 0.6, 0.4, 0.2],
            "evidence_place": [0.2, 0.4, 0.6, 0.8],
        })
        overlay_conditions = {
            "maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"},
            "suppress": {"column": "trial_type", "match": "regex", "value": ".*suppress.*"},
        }
        result = summarize_raw_for_timecourse(raw, overlay_conditions)

        assert set(result.columns) >= {"window_index", "regressor_label", "overlay_label", "evidence_face", "evidence_place", "evidence_face_se"}
        maintain_row = result[result["overlay_label"] == "maintain"].iloc[0]
        assert maintain_row["evidence_face"] == pytest.approx(0.7)
        suppress_row = result[result["overlay_label"] == "suppress"].iloc[0]
        assert suppress_row["evidence_face"] == pytest.approx(0.3)

    def test_unmatched_rows_dropped_with_warning(self, capsys):
        raw = pd.DataFrame({
            "window_index": [0, 0],
            "regressor_label": ["face", "face"],
            "trial_type": ["maintain_face", "unrelated_trial"],
            "evidence_face": [0.8, 0.5],
        })
        overlay_conditions = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"}}
        result = summarize_raw_for_timecourse(raw, overlay_conditions)

        assert len(result) == 1
        assert "1 row(s) matched no overlay condition" in capsys.readouterr().out


# =====================================================
# load_annotation_info: overlay_conditions
# =====================================================

def _write_config(tmp_path, timecourse_decoding_extra):
    config = {
        "model_conditions": {
            "timecourse_decoding": {
                "conditions": {"face": {"column": "trial_type", "match": "regex", "value": ".*face.*"}},
                "window": {
                    "start": {"reference": "onset", "offset_seconds": 0},
                    "end": {"reference": "offset_end", "offset_seconds": 10},
                },
                **timecourse_decoding_extra,
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path), config


def _write_master_spreadsheet(tmp_path, boldfile):
    master = pd.DataFrame([{
        "subject": "01", "session": "", "task": "test", "run": 1,
        "trial_type": "maintain_face", "trial_index": 1, "onset": 0.0, "duration": 2.0,
        "volume_of_interest": 0, "boldfile": boldfile, "eventfile": "x",
    }])
    path = tmp_path / "master_spreadsheet.csv"
    master.to_csv(path, index=False)
    return str(path)


class TestLoadAnnotationInfoOverlay:
    def test_returns_overlay_conditions_when_present(self, tmp_path, synthetic_bold_file):
        overlay = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"}}
        config_path, config = _write_config(tmp_path, {"overlay": overlay})
        master_path = _write_master_spreadsheet(tmp_path, synthetic_bold_file)

        window, tr, median_duration, overlay_conditions = load_annotation_info(config_path, master_path)

        assert overlay_conditions == overlay
        assert tr == pytest.approx(2.0)  # synthetic_bold_file's TR

    def test_empty_overlay_conditions_when_absent(self, tmp_path, synthetic_bold_file):
        config_path, _ = _write_config(tmp_path, {})
        master_path = _write_master_spreadsheet(tmp_path, synthetic_bold_file)

        _, _, _, overlay_conditions = load_annotation_info(config_path, master_path)

        assert overlay_conditions == {}

    def test_empty_overlay_conditions_when_config_missing(self):
        _, _, _, overlay_conditions = load_annotation_info(None, None)
        assert overlay_conditions == {}


# =====================================================
# resolve_desc
# =====================================================

class TestResolveDesc:
    def test_desc_arg_used_directly_when_given(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {"desc": "from_config"}}))
        assert resolve_desc("from_cli", str(config_path)) == "from_cli"

    def test_falls_back_to_config_model_desc(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {"desc": "gm_object_classifier"}}))
        assert resolve_desc(None, str(config_path)) == "gm_object_classifier"

    def test_config_desc_is_sanitized_same_as_workflow_scripts(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {"desc": "gm valence/classifier!"}}))
        assert resolve_desc(None, str(config_path)) == "gm_valence_classifier_"

    def test_raises_when_neither_given(self):
        with pytest.raises(SystemExit):
            resolve_desc(None, None)

    def test_raises_when_config_missing_on_disk(self, tmp_path):
        with pytest.raises(SystemExit):
            resolve_desc(None, str(tmp_path / "nope.json"))

    def test_raises_when_config_has_no_model_desc(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {}}))
        with pytest.raises(SystemExit):
            resolve_desc(None, str(config_path))


# =====================================================
# resolve_mnispace
# =====================================================

class TestResolveMnispace:
    def test_true_when_configured(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {"mnispace": True}}))
        assert resolve_mnispace(str(config_path)) is True

    def test_false_when_configured_false(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {"mnispace": False}}))
        assert resolve_mnispace(str(config_path)) is False

    def test_false_when_key_absent(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model": {}}))
        assert resolve_mnispace(str(config_path)) is False

    def test_false_when_no_config_given(self):
        assert resolve_mnispace(None) is False

    def test_false_when_config_missing_on_disk(self, tmp_path):
        assert resolve_mnispace(str(tmp_path / "nope.json")) is False

    def test_false_when_config_is_bad_json(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{not valid json")
        assert resolve_mnispace(str(config_path)) is False
