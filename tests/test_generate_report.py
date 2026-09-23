"""generate_report.py: subject-scope discovery + file layout + small CSV
loaders. Fake directory trees built under tmp_path -- no dependency on real
mvpa_workflow.py output."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import pytest

from workflows.generate_report import (
    list_subject_dirs,
    subject_paths,
    has_fold_files,
    fold_paths,
    load_metadata_csv,
    load_labeled_csv,
    infer_categories,
    compile_group_summary,
    compile_group_decoding,
    compile_group_cv_results,
    summarize_raw_for_timecourse,
    resolve_overlay_styles,
    resolve_overlay_groups,
    resolve_timecourse_groups,
    render_timecourse_pages,
    _build_overlay_legend,
    load_annotation_info,
    resolve_desc,
    resolve_group_impa_mni,
    resolve_mnispace,
    load_double_dipping_report,
    render_double_dipping_page,
    _wrap_suptitle,
    _row_major_legend_order,
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
        assert paths["kfold_metadata"] == "/out/desc1/01/model/01_model_results_metadata.csv"
        assert paths["kfold_auc"] == "/out/desc1/01/model/01_model_results_auc.csv"
        assert paths["kfold_accuracy"] == "/out/desc1/01/model/01_model_results_accuracy.csv"
        assert paths["kfold_evidence"] == "/out/desc1/01/model/01_model_results_evidence.csv"
        assert paths["kfold_impa"] == "/out/desc1/01/model/01_impa.nii.gz"
        assert paths["kfold_impa_mni"] == "/out/desc1/01/model/01_impa_mni.nii.gz"
        assert paths["kfold_cv_raw"] == "/out/desc1/01/model/01_cv_results.csv"

    def test_test_paths_under_test_dir(self):
        paths = subject_paths("/out", "desc1", "01")
        assert paths["test_metadata"] == "/out/desc1/01/test/01_model_results_metadata.csv"
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
            (base / "model" / f"01_fold{fid}_model_results_metadata.csv").write_text("0.5")

        assert has_fold_files(str(tmp_path), "desc1", "01") is True

        folds = fold_paths(str(tmp_path), "desc1", "01")
        assert sorted(folds.keys()) == [1, 2]
        assert folds[1]["kfold_metadata"] == str(base / "model" / "01_fold1_model_results_metadata.csv")
        assert folds[1]["kfold_impa"] == str(base / "model" / "01_fold1_impa.nii.gz")
        assert folds[2]["kfold_auc"] == str(base / "model" / "01_fold2_model_results_auc.csv")
        # no per-fold decoding anymore -- timecourse decoding is never fold-based
        assert "decoding" not in folds[1]
        assert "decoding_raw" not in folds[1]

    def test_fold_paths_mnispace_true_uses_mni_filename(self, tmp_path):
        base = _make_subject(tmp_path, "desc1", "01")
        (base / "model" / "01_fold1_model_results_metadata.csv").write_text("0.5")
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
# load_metadata_csv / load_labeled_csv
# =====================================================

class TestLoaders:
    def test_load_metadata_csv(self, tmp_path):
        path = tmp_path / "metadata.csv"
        pd.DataFrame({"value": {"total_scores": 0.75, "whole_voxels": 5000}}).to_csv(path)
        metadata = load_metadata_csv(str(path))
        assert metadata["total_scores"] == pytest.approx(0.75)
        assert metadata["whole_voxels"] == pytest.approx(5000)

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

def _write_family_csvs(base, subdir, subject, total, auc, acc, evi, categories, **extra_metadata):
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": {"total_scores": total, **extra_metadata}}).to_csv(d / f"{subject}_model_results_metadata.csv")
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
        _write_family_csvs(base, "model", "01", 0.7, [0.8, 0.75], [[0.9, 0.1], [0.2, 0.8]], [[0.85, 0.15], [0.25, 0.75]], categories,
                            whole_voxels=17806, selected_voxels=42, feature_percent=0.2359)

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
# compile_group_cv_results
# =====================================================

class TestCompileGroupCvResults:
    def test_concatenates_every_subject_cv_raw(self, tmp_path):
        for subject, n_rows in (("01", 2), ("02", 3)):
            d = tmp_path / "desc1" / subject / "model"
            d.mkdir(parents=True)
            pd.DataFrame({
                "subject": [subject] * n_rows,
                "fold": list(range(1, n_rows + 1)),
                "task": ["WM"] * n_rows,
                "trial_type": ["maintain"] * n_rows,
                "run": list(range(1, n_rows + 1)),
                "predicted_label": ["face"] * n_rows,
                "correct": [True] * n_rows,
            }).to_csv(d / f"{subject}_cv_results.csv", index=False)

        combined = compile_group_cv_results(str(tmp_path), "desc1", ["01", "02"])
        assert len(combined) == 5
        assert sorted(combined["subject"].unique()) == ["01", "02"]
        assert set(combined.columns) >= {"task", "trial_type", "run", "fold"}

    def test_subject_missing_cv_output_is_skipped(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")  # model/ exists but no cv_results.csv
        d = tmp_path / "desc1" / "02" / "model"
        d.mkdir(parents=True)
        pd.DataFrame({"subject": ["02"], "fold": [1], "run": [1]}).to_csv(
            d / "02_cv_results.csv", index=False
        )

        combined = compile_group_cv_results(str(tmp_path), "desc1", ["01", "02"])
        assert list(combined["subject"]) == ["02"]

    def test_no_subjects_have_cv_output(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        combined = compile_group_cv_results(str(tmp_path), "desc1", ["01"])
        assert combined.empty


# =====================================================
# double-dipping report (load_double_dipping_report / render_double_dipping_page)
# =====================================================

def _write_double_dipping_report(tmp_path, desc, subject, report):
    base = tmp_path / desc / subject
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{subject}_double_dipping_report.json").write_text(json.dumps(report))


class TestLoadDoubleDippingReport:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        assert load_double_dipping_report(str(tmp_path), "desc1", "01") == {}

    def test_returns_parsed_contents_when_present(self, tmp_path):
        report = {"test": {"overlap_boldfiles": ["run-1.nii.gz"], "handling": "skipped"}}
        _write_double_dipping_report(tmp_path, "desc1", "01", report)
        assert load_double_dipping_report(str(tmp_path), "desc1", "01") == report


class TestRenderDoubleDippingPage:
    def test_no_page_when_no_subject_has_a_report(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_double_dipping_page(pdf, str(tmp_path), "desc1", ["01"])
            assert pdf.get_pagecount() == 0

    def test_page_rendered_when_a_subject_has_a_report(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        _write_double_dipping_report(tmp_path, "desc1", "01", {
            "test": {"overlap_boldfiles": ["run-1.nii.gz"], "handling": "skipped"},
            "timecourse": {"overlap_boldfiles": ["run-1.nii.gz"], "handling": "kfold_substitution", "n_folds_used": 3},
        })
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_double_dipping_page(pdf, str(tmp_path), "desc1", ["01"])
            assert pdf.get_pagecount() == 1

    def test_only_affected_subjects_produce_a_report_but_page_still_renders(self, tmp_path):
        _make_subject(tmp_path, "desc1", "01")
        _make_subject(tmp_path, "desc1", "02")
        _write_double_dipping_report(tmp_path, "desc1", "02", {
            "test": {"overlap_boldfiles": ["run-1.nii.gz", "run-2.nii.gz"], "handling": "overwritten"},
        })
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_double_dipping_page(pdf, str(tmp_path), "desc1", ["01", "02"])
            assert pdf.get_pagecount() == 1


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
# resolve_overlay_styles
# =====================================================

class TestResolveOverlayStyles:
    def test_explicit_color_and_line_type_indices(self):
        overlay = {
            "1A": {"column": "trial_type", "match": "exact", "value": "x", "color": 0, "line_type": 0},
            "1B": {"column": "trial_type", "match": "exact", "value": "y", "color": 0, "line_type": 1},
        }
        styles = resolve_overlay_styles(overlay)
        assert styles["1A"][0] == styles["1B"][0]  # same color index -> same color
        assert styles["1A"][1] != styles["1B"][1]  # different line_type index -> different style
        assert styles["1A"][1] == "-"
        assert styles["1B"][1] == "--"

    def test_explicit_literal_color_and_line_type_strings(self):
        overlay = {
            "a": {"column": "trial_type", "match": "exact", "value": "x", "color": "#1f77b4", "line_type": "dashed"},
        }
        styles = resolve_overlay_styles(overlay)
        assert styles["a"] == ("#1f77b4", "dashed")

    def test_missing_line_type_defaults_to_solid(self):
        overlay = {"a": {"column": "trial_type", "match": "exact", "value": "x"}}
        styles = resolve_overlay_styles(overlay)
        assert styles["a"][1] == "-"

    def test_missing_color_restarts_at_zero_per_line_type(self):
        # 1A/2A share line_type (default solid), 1B/2B share line_type "--" --
        # auto-color-assignment should restart at palette index 0 within
        # each line_type group independently
        overlay = {
            "1A": {"column": "trial_type", "match": "exact", "value": "1a", "line_type": 0},
            "2A": {"column": "trial_type", "match": "exact", "value": "2a", "line_type": 0},
            "1B": {"column": "trial_type", "match": "exact", "value": "1b", "line_type": 1},
            "2B": {"column": "trial_type", "match": "exact", "value": "2b", "line_type": 1},
        }
        styles = resolve_overlay_styles(overlay)
        trace_colors = plt.get_cmap("tab10").colors
        assert styles["1A"][0] == trace_colors[0]
        assert styles["2A"][0] == trace_colors[1]
        assert styles["1B"][0] == trace_colors[0]  # restarted, matches 1A
        assert styles["2B"][0] == trace_colors[1]  # restarted, matches 2A
        assert styles["1A"][1] == "-"
        assert styles["1B"][1] == "--"

    def test_explicit_color_mixed_with_auto_color_in_same_line_type(self):
        # an explicit color doesn't consume/shift the auto-assignment counter
        # for other color-less entries sharing its line_type
        overlay = {
            "explicit": {"column": "trial_type", "match": "exact", "value": "x", "color": 5},
            "auto1": {"column": "trial_type", "match": "exact", "value": "y"},
            "auto2": {"column": "trial_type", "match": "exact", "value": "z"},
        }
        styles = resolve_overlay_styles(overlay)
        trace_colors = plt.get_cmap("tab10").colors
        assert styles["explicit"][0] == trace_colors[5]
        assert styles["auto1"][0] == trace_colors[0]
        assert styles["auto2"][0] == trace_colors[1]

    def test_bool_is_not_treated_as_int_index(self):
        # JSON true/false parse to Python bool, a subclass of int -- must not
        # be misread as color/line_type index 1/0
        overlay = {"a": {"column": "trial_type", "match": "exact", "value": "x", "color": True, "line_type": False}}
        styles = resolve_overlay_styles(overlay)
        assert styles["a"] == (True, False)


# =====================================================
# resolve_overlay_groups / resolve_timecourse_groups
# =====================================================

class TestResolveOverlayGroups:
    def test_reads_group_key_per_entry(self):
        overlay = {
            "maintain_pos": {"column": "trial_type", "match": "exact", "value": "x", "group": "pos"},
            "maintain_neg": {"column": "trial_type", "match": "exact", "value": "y", "group": "neg"},
        }
        assert resolve_overlay_groups(overlay) == {"maintain_pos": "pos", "maintain_neg": "neg"}

    def test_missing_group_key_is_none(self):
        overlay = {"a": {"column": "trial_type", "match": "exact", "value": "x"}}
        assert resolve_overlay_groups(overlay) == {"a": None}


class TestResolveTimecourseGroups:
    def test_splits_into_one_block_per_group_value(self):
        true_conditions = ["face", "place"]
        overlay_categories = ["maintain_pos", "maintain_neg", "suppress_pos", "suppress_neg"]
        overlay_groups = {"maintain_pos": "pos", "suppress_pos": "pos", "maintain_neg": "neg", "suppress_neg": "neg"}

        blocks = resolve_timecourse_groups(true_conditions, overlay_categories, overlay_groups)

        assert len(blocks) == 2  # one block per group value
        pos_val, pos_rows = blocks[0]
        neg_val, neg_rows = blocks[1]
        assert pos_val == "pos"
        assert pos_rows == [("face", ["maintain_pos", "suppress_pos"]), ("place", ["maintain_pos", "suppress_pos"])]
        assert neg_val == "neg"
        assert neg_rows == [("face", ["maintain_neg", "suppress_neg"]), ("place", ["maintain_neg", "suppress_neg"])]

    def test_group_value_order_is_first_seen_in_overlay_categories(self):
        # "neg" declared before "pos" in overlay_categories -- blocks should
        # follow that order, not alphabetical
        true_conditions = ["face"]
        overlay_categories = ["maintain_neg", "maintain_pos"]
        overlay_groups = {"maintain_neg": "neg", "maintain_pos": "pos"}

        blocks = resolve_timecourse_groups(true_conditions, overlay_categories, overlay_groups)

        assert [group_val for group_val, _ in blocks] == ["neg", "pos"]

    def test_single_group_value_produces_one_block_with_every_true_condition(self):
        # every overlay category shares the same group value -- degenerates
        # to today's single-block/single-legend behavior, just still routed
        # through this function
        true_conditions = ["face", "place"]
        overlay_categories = ["maintain", "suppress"]
        overlay_groups = {"maintain": "all", "suppress": "all"}

        blocks = resolve_timecourse_groups(true_conditions, overlay_categories, overlay_groups)

        assert len(blocks) == 1
        group_val, rows = blocks[0]
        assert group_val == "all"
        assert rows == [("face", ["maintain", "suppress"]), ("place", ["maintain", "suppress"])]


# =====================================================
# _build_overlay_legend
# =====================================================

class TestBuildOverlayLegend:
    def test_no_overlay_conditions_gives_only_se_proxy(self):
        handles, labels = _build_overlay_legend([None], overlay_styles={}, ncol=4, overlay_conditions={})
        # padded to a full ncol=4 row with blank entries around the one real label
        assert [l for l in labels if l] == ["trial-to-trial SE"]
        assert len(labels) == 4
        assert len(handles) == 4

    def test_covers_only_the_given_cats(self):
        overlay = {"a": {}, "b": {}, "c": {}}
        styles = {"a": ("C0", "-"), "b": ("C1", "-"), "c": ("C2", "-")}
        handles, labels = _build_overlay_legend(["a", "b"], styles, ncol=4, overlay_conditions=overlay)
        # "c" excluded -- this legend is scoped to just its own block/cats
        assert "c" not in labels
        assert "a" in labels and "b" in labels
        assert "trial-to-trial SE" in labels

    def test_pads_ragged_row_to_full_ncol_width(self):
        overlay = {"a": {}}
        styles = {"a": ("C0", "-")}
        # 1 category + SE proxy = 2 labels, ncol=4 -> padded to 4 with blanks
        handles, labels = _build_overlay_legend(["a"], styles, ncol=4, overlay_conditions=overlay)
        assert len(labels) == 4
        assert len(handles) == 4


# =====================================================
# render_timecourse_pages -- "group" grid + per-block legends
# =====================================================

def _write_decoding_raw(tmp_path, desc, subject, rows):
    d = tmp_path / desc / subject / "decoding"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / f"{subject}_decoding_results.csv", index=False)


def _synthetic_timecourse_rows(overlay_trial_types, n_windows=3):
    """rows for 2 true conditions (face/place) x given overlay trial_types x
    n_windows, with plausible evidence_face/evidence_place values -- enough
    for render_timecourse_pages to actually build a grid."""
    rows = []
    for true_cond in ("face", "place"):
        for trial_type in overlay_trial_types:
            for w in range(n_windows):
                rows.append({
                    "subject": "01", "model_descr": "m", "window_index": w,
                    "regressor_label": true_cond, "trial_type": trial_type,
                    "evidence_face": 0.6 if true_cond == "face" else 0.4,
                    "evidence_place": 0.4 if true_cond == "face" else 0.6,
                })
    return rows


class TestRenderTimecoursePagesGroups:
    def test_no_group_key_keeps_single_block_single_legend(self, tmp_path):
        overlay = {
            "maintain": {"column": "trial_type", "match": "exact", "value": "maintain"},
            "suppress": {"column": "trial_type", "match": "exact", "value": "suppress"},
        }
        _write_decoding_raw(tmp_path, "desc1", "01", _synthetic_timecourse_rows(["maintain", "suppress"]))
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_timecourse_pages(pdf, str(tmp_path), "desc1", ["01"], window=None, tr=None,
                                     median_duration={}, overlay_conditions=overlay)
            assert pdf.get_pagecount() == 1

    def test_group_key_splits_into_separate_blocks_with_own_legends(self, tmp_path):
        overlay = {
            "maintain_pos": {"column": "trial_type", "match": "exact", "value": "maintain_pos", "group": "pos", "color": 0},
            "maintain_neg": {"column": "trial_type", "match": "exact", "value": "maintain_neg", "group": "neg", "color": 0},
        }
        _write_decoding_raw(tmp_path, "desc1", "01", _synthetic_timecourse_rows(["maintain_pos", "maintain_neg"]))
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_timecourse_pages(pdf, str(tmp_path), "desc1", ["01"], window=None, tr=None,
                                     median_duration={}, overlay_conditions=overlay)
            assert pdf.get_pagecount() == 1
        # the actual 2-block (pos/neg) x 2-true-condition grid shape, and
        # that each block gets its own legend, is verified directly via
        # TestResolveTimecourseGroups + TestBuildOverlayLegend above (the
        # render closes its own figure via plt.close(), so it can't be
        # inspected from here) -- this just confirms grouped mode renders
        # without crashing end to end

    def test_missing_group_on_some_entries_falls_back_without_crashing(self, tmp_path, capsys):
        # "suppress" has no "group" set while "maintain_pos"/"maintain_neg" do --
        # should warn and still render rather than raising
        overlay = {
            "maintain_pos": {"column": "trial_type", "match": "exact", "value": "maintain_pos", "group": "pos"},
            "maintain_neg": {"column": "trial_type", "match": "exact", "value": "maintain_neg", "group": "neg"},
            "suppress": {"column": "trial_type", "match": "exact", "value": "suppress"},
        }
        _write_decoding_raw(tmp_path, "desc1", "01",
                             _synthetic_timecourse_rows(["maintain_pos", "maintain_neg", "suppress"]))
        pdf_path = tmp_path / "out.pdf"
        with PdfPages(str(pdf_path)) as pdf:
            render_timecourse_pages(pdf, str(tmp_path), "desc1", ["01"], window=None, tr=None,
                                     median_duration={}, overlay_conditions=overlay)
            assert pdf.get_pagecount() == 1
        assert "group" in capsys.readouterr().out.lower()


# =====================================================
# _row_major_legend_order
# =====================================================

def _matplotlib_column_major_read(items, ncol):
    """Re-derive what matplotlib's own Legend would display, reading `items`
    column-major with its column-size-balancing (leftmost columns get
    ceil(n_remaining/columns_remaining), later ones one fewer) -- used here
    to verify _row_major_legend_order's output round-trips back to the
    original row-major order, without hardcoding the permutation itself."""
    n = len(items)
    col_sizes = []
    remaining = n
    for c in range(ncol):
        size = -(-remaining // (ncol - c))  # ceil
        col_sizes.append(size)
        remaining -= size
    cols = []
    i = 0
    for size in col_sizes:
        cols.append(items[i:i + size])
        i += size
    rows = []
    for row in range(col_sizes[0]):
        for col in cols:
            if row < len(col):
                rows.append(col[row])
    return rows


class TestRowMajorLegendOrder:
    def test_exact_multiple_of_ncol(self):
        items = list(range(8))
        ordered = _row_major_legend_order(items, 4)
        assert _matplotlib_column_major_read(ordered, 4) == items

    def test_not_a_multiple_of_ncol(self):
        # 9 items, 4 columns -- the case that actually motivated this fix
        # (an 8-category overlay + the trial-to-trial SE proxy)
        items = list(range(9))
        ordered = _row_major_legend_order(items, 4)
        assert _matplotlib_column_major_read(ordered, 4) == items

    def test_various_sizes_round_trip(self):
        for n in range(1, 15):
            for ncol in range(1, 6):
                items = [f"item{i}" for i in range(n)]
                ordered = _row_major_legend_order(items, ncol)
                assert sorted(ordered) == sorted(items)  # no items lost or duplicated
                assert _matplotlib_column_major_read(ordered, ncol) == items

    def test_empty_list(self):
        assert _row_major_legend_order([], 4) == []

    def test_ncol_of_one_is_unchanged(self):
        items = ["a", "b", "c"]
        assert _row_major_legend_order(items, 1) == items


# =====================================================
# _wrap_suptitle
# =====================================================

class TestWrapSuptitle:
    def test_short_text_unchanged(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            assert _wrap_suptitle(fig, "short title", 14) == "short title"
        finally:
            plt.close(fig)

    def test_long_underscore_joined_desc_wraps_without_overflowing(self):
        # 6 inches -- the narrowest a real timecourse page ever gets
        # (figsize=(3 * n_cols, ...) with the minimum realistic n_cols=2)
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            long_desc = "vvps_category_loc2WM_timecourse_classifier: timecourse decoding"
            wrapped = _wrap_suptitle(fig, long_desc, 14)
            lines = wrapped.split("\n")
            assert len(lines) > 1  # actually wrapped, not left as one line

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            probe = fig.text(0, 0, "M" * 40, fontsize=14, fontweight="bold")
            char_width_px = probe.get_window_extent(renderer=renderer).width / 40
            probe.remove()
            for line in lines:
                assert len(line) * char_width_px <= fig.bbox.width  # fits within the figure
        finally:
            plt.close(fig)

    def test_breaks_after_underscores_not_mid_word(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            wrapped = _wrap_suptitle(fig, "aaaaaaaaaa_bbbbbbbbbb_cccccccccc_dddddddddd", 14)
            lines = wrapped.split("\n")
            assert len(lines) > 1  # actually wrapped
            # every line but the last should end right after an underscore
            # (a real token boundary), never mid-token
            for line in lines[:-1]:
                assert line.endswith("_")
        finally:
            plt.close(fig)

    def test_preserves_embedded_newlines(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            wrapped = _wrap_suptitle(fig, "line one\nline two", 14)
            assert wrapped == "line one\nline two"
        finally:
            plt.close(fig)


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

    def test_overlay_conditions_may_carry_color_and_line_type(self, tmp_path, synthetic_bold_file):
        overlay = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*", "color": 0, "line_type": "--"}}
        config_path, config = _write_config(tmp_path, {"overlay": overlay})
        master_path = _write_master_spreadsheet(tmp_path, synthetic_bold_file)

        _, _, _, overlay_conditions = load_annotation_info(config_path, master_path)

        assert overlay_conditions == overlay

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
