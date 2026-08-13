"""generate_report.py: subject-scope discovery + file layout + small CSV
loaders. Fake directory trees built under tmp_path -- no dependency on real
mvpa_workflow.py output."""

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
