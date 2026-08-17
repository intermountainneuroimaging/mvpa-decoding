"""generate_report.py: subject-scope discovery + file layout + small CSV
loaders. Fake directory trees built under tmp_path -- no dependency on real
mvpa_generalization_workflow.py output."""

import json

import numpy as np
import pandas as pd
import pytest

from generate_report import (
    list_subject_dirs,
    subject_paths,
    has_fold_files,
    fold_paths,
    load_scalar_csv,
    load_labeled_csv,
    infer_categories,
    summarize_raw_with_overlay,
    load_annotation_info,
    resolve_desc,
)


def _make_subject(tmp_path, desc, subject, with_model_dir=True):
    base = tmp_path / desc / subject
    if with_model_dir:
        (base / "model").mkdir(parents=True)
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
        _make_subject(tmp_path, "desc1", "03", with_model_dir=False)  # no model/ -- excluded
        result = list_subject_dirs(str(tmp_path), "desc1")
        assert result == ["01", "02"]

    def test_no_subject_dirs_at_all_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            list_subject_dirs(str(tmp_path), "missing_desc")


# =====================================================
# subject_paths
# =====================================================

class TestSubjectPaths:
    def test_paths_are_named_by_subject_under_expected_subdirs(self):
        paths = subject_paths("/out", "desc1", "01")
        assert paths["model_total"] == "/out/desc1/01/model/01_model_results_total_scores.csv"
        assert paths["model_auc"] == "/out/desc1/01/model/01_model_results_auc.csv"
        assert paths["cv_total"] == "/out/desc1/01/cv/01_cv_results_total_scores.csv"
        assert paths["decoding"] == "/out/desc1/01/decoding/01_summary_decoding_results.csv"
        assert paths["decoding_raw"] == "/out/desc1/01/decoding/01_decoding_results.csv"


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
        assert folds[1]["model_total"] == str(base / "model" / "01_fold1_model_results_total_scores.csv")
        assert folds[2]["decoding"] == str(base / "decoding" / "01_fold2_summary_decoding_results.csv")
        assert folds[2]["decoding_raw"] == str(base / "decoding" / "01_fold2_decoding_results.csv")


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
    def test_reads_categories_from_first_available_auc_csv(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        pd.DataFrame({"auc": [0.7, 0.8]}, index=["face", "place"]).to_csv(base / "model" / "01_model_results_auc.csv")

        categories = infer_categories(str(tmp_path), "desc1", ["01"])
        assert categories == ["face", "place"]

    def test_returns_empty_list_when_no_auc_csv_found(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        categories = infer_categories(str(tmp_path), "desc1", ["01"])
        assert categories == []


# =====================================================
# summarize_raw_with_overlay
# =====================================================

class TestSummarizeRawWithOverlay:
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
        result = summarize_raw_with_overlay(raw, overlay_conditions)

        assert set(result.columns) >= {"window_index", "regressor_label", "overlay_label", "evidence_face", "evidence_place"}
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
        result = summarize_raw_with_overlay(raw, overlay_conditions)

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
