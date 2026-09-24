"""analysis/decoding_results_toolkit.py: the generic, reusable building
blocks (filters, condition derivation, aggregation, baseline subtraction,
binning, pluggable stats, plotting) that any one-off decoding_results.csv
analysis script is meant to compose -- see
tests/test_plot_valence_evidence_by_operation.py for how a specific script
wires these together."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from analysis.decoding_results_toolkit import (
    load_decoding_results,
    select_evidence_value,
    apply_filters,
    derive_label,
    aggregate_by_subject_window,
    average_across_groups,
    subtract_baseline,
    bin_by_size,
    bin_by_edges,
    ttest_1samp_diff,
    ttest_rel,
    wilcoxon_signed_rank,
    compare_conditions_by_bin,
    plot_conditions,
    STAT_METHODS,
)


# =====================================================
# load_decoding_results
# =====================================================

class TestLoadDecodingResults:
    def _write(self, tmp_path, desc, subject, n_rows):
        d = tmp_path / desc / subject / "decoding"
        d.mkdir(parents=True)
        pd.DataFrame({"subject": [subject] * n_rows, "window_index": list(range(n_rows))}).to_csv(
            d / f"{subject}_decoding_results.csv", index=False
        )

    def test_concatenates_every_subject(self, tmp_path):
        self._write(tmp_path, "desc1", "01", 2)
        self._write(tmp_path, "desc1", "02", 3)
        combined = load_decoding_results(str(tmp_path), "desc1", ["01", "02"])
        assert len(combined) == 5
        assert sorted(combined["subject"].unique()) == ["01", "02"]

    def test_missing_subject_skipped_not_raised(self, tmp_path, capsys):
        self._write(tmp_path, "desc1", "02", 2)
        (tmp_path / "desc1" / "01").mkdir(parents=True)
        combined = load_decoding_results(str(tmp_path), "desc1", ["01", "02"])
        assert list(combined["subject"].unique()) == ["02"]
        assert "skipping" in capsys.readouterr().out

    def test_no_subjects_at_all_raises(self, tmp_path):
        (tmp_path / "desc1" / "01").mkdir(parents=True)
        with pytest.raises(SystemExit):
            load_decoding_results(str(tmp_path), "desc1", ["01"])


# =====================================================
# select_evidence_value
# =====================================================

class TestSelectEvidenceValue:
    def test_self_picks_own_true_category(self):
        df = pd.DataFrame({
            "regressor_label": ["maintain", "suppress"],
            "evidence_maintain": [0.7, 0.2],
            "evidence_suppress": [0.3, 0.8],
        })
        result = select_evidence_value(df, value="self")
        assert result["value"].tolist() == pytest.approx([0.7, 0.8])

    def test_self_drops_unknown_regressor_label(self, capsys):
        df = pd.DataFrame({
            "regressor_label": ["maintain", "bogus"],
            "evidence_maintain": [0.7, 0.2],
        })
        result = select_evidence_value(df, value="self")
        assert len(result) == 1
        assert "dropped" in capsys.readouterr().out

    def test_explicit_category_name(self):
        df = pd.DataFrame({"regressor_label": ["maintain"], "evidence_maintain": [0.7], "evidence_suppress": [0.3]})
        result = select_evidence_value(df, value="suppress")
        assert result["value"].tolist() == pytest.approx([0.3])

    def test_explicit_already_prefixed_column_name(self):
        df = pd.DataFrame({"evidence_suppress": [0.3]})
        result = select_evidence_value(df, value="evidence_suppress")
        assert result["value"].tolist() == pytest.approx([0.3])

    def test_missing_explicit_column_raises(self):
        df = pd.DataFrame({"evidence_maintain": [0.7]})
        with pytest.raises(SystemExit):
            select_evidence_value(df, value="nonexistent")


# =====================================================
# apply_filters
# =====================================================

class TestApplyFilters:
    def test_scalar_exact_match(self):
        df = pd.DataFrame({"task": ["WMpos", "WMneg", "WMpos"]})
        assert len(apply_filters(df, {"task": "WMpos"})) == 2

    def test_list_membership(self):
        df = pd.DataFrame({"run": [1, 2, 3, 4]})
        assert apply_filters(df, {"run": [2, 4]})["run"].tolist() == [2, 4]

    def test_callable_predicate(self):
        df = pd.DataFrame({"window_index": [0, 1, 2, 3, 4]})
        result = apply_filters(df, {"window_index": lambda v: v >= 3})
        assert result["window_index"].tolist() == [3, 4]

    def test_regex_criterion(self):
        df = pd.DataFrame({"trial_type": ["WM_maintain_face", "WM_suppress_place"]})
        result = apply_filters(df, {"trial_type": {"regex": "face"}})
        assert result["trial_type"].tolist() == ["WM_maintain_face"]

    def test_multiple_filters_combine_with_and(self):
        df = pd.DataFrame({"task": ["WMpos", "WMpos", "WMneg"], "run": [1, 2, 1]})
        result = apply_filters(df, {"task": "WMpos", "run": 1})
        assert len(result) == 1

    def test_unknown_column_raises(self):
        df = pd.DataFrame({"task": ["WMpos"]})
        with pytest.raises(SystemExit):
            apply_filters(df, {"nonexistent": "x"})


# =====================================================
# derive_label
# =====================================================

class TestDeriveLabel:
    def test_exact_mapping(self):
        df = pd.DataFrame({"task": ["WMpos", "WMneg"]})
        result = derive_label(df, "task", {"pos": "WMpos", "neg": "WMneg"}, new_col="valence")
        assert result["valence"].tolist() == ["pos", "neg"]

    def test_exact_mapping_list_criterion(self):
        df = pd.DataFrame({"run": [1, 2, 3, 4]})
        result = derive_label(df, "run", {"early": [1, 2], "late": [3, 4]}, new_col="epoch")
        assert result["epoch"].tolist() == ["early", "early", "late", "late"]

    def test_regex_mapping_first_match_wins(self):
        df = pd.DataFrame({"trial_type": ["WM_maintain_face", "WM_suppress_place"]})
        result = derive_label(df, "trial_type", {"face": "face", "place": "place"}, new_col="stimulus", regex=True)
        assert result["stimulus"].tolist() == ["face", "place"]

    def test_unmatched_dropped_by_default(self, capsys):
        df = pd.DataFrame({"task": ["WMpos", "loc"]})
        result = derive_label(df, "task", {"pos": "WMpos", "neg": "WMneg"}, new_col="valence")
        assert len(result) == 1
        assert "dropped" in capsys.readouterr().out

    def test_unmatched_kept_when_requested(self):
        df = pd.DataFrame({"task": ["WMpos", "loc"]})
        result = derive_label(df, "task", {"pos": "WMpos", "neg": "WMneg"}, new_col="valence", drop_unmatched=False)
        assert len(result) == 2
        assert result["valence"].isna().sum() == 1


# =====================================================
# aggregate_by_subject_window / average_across_groups
# =====================================================

class TestAggregateBySubjectWindow:
    def test_means_within_group(self):
        df = pd.DataFrame({
            "subject": ["01", "01", "01"], "window_index": [0, 0, 1],
            "valence": ["pos", "pos", "neg"], "value": [0.6, 0.8, 0.5],
        })
        wide = aggregate_by_subject_window(df, condition_col="valence")
        assert wide.loc[("01", 0), "pos"] == pytest.approx(0.7)
        assert wide.loc[("01", 1), "neg"] == pytest.approx(0.5)


class TestAverageAcrossGroups:
    def test_equal_weight_average(self):
        idx = pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"])
        face = pd.DataFrame({"pos": [0.8]}, index=idx)
        place = pd.DataFrame({"pos": [0.6]}, index=idx)
        result = average_across_groups({"face": face, "place": place})
        assert result.loc[("01", 0), "pos"] == pytest.approx(0.7)

    def test_missing_from_one_group_still_averages_available(self):
        idx_face = pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"])
        idx_place = pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"])
        face = pd.DataFrame({"pos": [0.8, 0.9]}, index=idx_face)
        place = pd.DataFrame({"pos": [0.6]}, index=idx_place)
        result = average_across_groups({"face": face, "place": place})
        assert result.loc[("01", 0), "pos"] == pytest.approx(0.7)  # mean(0.8, 0.6)
        assert result.loc[("01", 1), "pos"] == pytest.approx(0.9)  # only face available


# =====================================================
# subtract_baseline
# =====================================================

class TestSubtractBaseline:
    def test_matched_index_subtracted(self):
        idx = pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"])
        op = pd.DataFrame({"pos": [0.8, 0.7], "neg": [0.6, 0.5]}, index=idx)
        base = pd.DataFrame({"pos": [0.5, 0.5], "neg": [0.4, 0.4]}, index=idx)
        result = subtract_baseline(op, base)
        assert result["pos"].tolist() == pytest.approx([0.3, 0.2])

    def test_unmatched_rows_excluded(self):
        op = pd.DataFrame({"pos": [0.8, 0.7]},
                           index=pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"]))
        base = pd.DataFrame({"pos": [0.5]},
                             index=pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"]))
        result = subtract_baseline(op, base)
        assert len(result) == 1

    def test_columns_restricted_when_given(self):
        idx = pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"])
        op = pd.DataFrame({"pos": [0.8], "neg": [0.6]}, index=idx)
        base = pd.DataFrame({"pos": [0.5], "neg": [0.4]}, index=idx)
        result = subtract_baseline(op, base, columns=["pos"])
        assert list(result.columns) == ["pos"]


# =====================================================
# bin_by_size / bin_by_edges
# =====================================================

class TestBinBySize:
    def test_non_overlapping_bins_1_indexed_tr(self):
        idx = pd.MultiIndex.from_product([["01"], range(6)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=idx)
        per_subject_bin, bins_meta = bin_by_size(wide, bin_size=3)
        assert list(zip(bins_meta["tr_start"], bins_meta["tr_end"])) == [(1, 3), (4, 6)]
        assert per_subject_bin.loc[("01", 0), "pos"] == pytest.approx(0.2)  # mean(0.1,0.2,0.3)
        assert per_subject_bin.loc[("01", 1), "pos"] == pytest.approx(0.5)

    def test_tr_offset_zero_keeps_window_index_numbering(self):
        idx = pd.MultiIndex.from_product([["01"], range(3)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": [0.1, 0.2, 0.3]}, index=idx)
        _, bins_meta = bin_by_size(wide, bin_size=3, tr_offset=0)
        assert bins_meta.iloc[0]["tr_start"] == 0
        assert bins_meta.iloc[0]["tr_end"] == 2


class TestBinByEdges:
    def test_explicit_windows(self):
        idx = pd.MultiIndex.from_product([["01"], range(8)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}, index=idx)
        per_subject_bin, bins_meta = bin_by_edges(wide, edges=[(1, 2), (7, 8)])
        assert len(bins_meta) == 2
        assert per_subject_bin.loc[("01", 0), "pos"] == pytest.approx(0.15)  # TR1-2 -> window 0,1
        assert per_subject_bin.loc[("01", 1), "pos"] == pytest.approx(0.75)  # TR7-8 -> window 6,7

    def test_trs_not_covered_by_any_window_excluded(self):
        idx = pd.MultiIndex.from_product([["01"], range(4)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": [0.1, 0.2, 0.3, 0.4]}, index=idx)
        per_subject_bin, _ = bin_by_edges(wide, edges=[(1, 1)])
        assert len(per_subject_bin) == 1
        assert per_subject_bin.loc[("01", 0), "pos"] == pytest.approx(0.1)


# =====================================================
# stat methods
# =====================================================

class TestStatMethods:
    def test_ttest_1samp_diff_and_ttest_rel_agree(self):
        rng = np.random.default_rng(0)
        a = 0.6 + rng.normal(0, 0.05, 20)
        b = 0.5 + rng.normal(0, 0.05, 20)
        r1 = ttest_1samp_diff(a, b, alternative="greater")
        r2 = ttest_rel(a, b, alternative="greater")
        assert r1["stat"] == pytest.approx(r2["stat"])
        assert r1["p_value"] == pytest.approx(r2["p_value"])

    def test_detects_a_clear_positive_difference(self):
        rng = np.random.default_rng(1)
        a = 0.6 + rng.normal(0, 0.01, 10)
        b = 0.5 + rng.normal(0, 0.01, 10)
        result = ttest_1samp_diff(a, b, alternative="greater")
        assert result["mean_diff"] == pytest.approx(0.1, abs=0.02)
        assert result["p_value"] < 0.001

    def test_wilcoxon_runs_and_reports_method(self):
        rng = np.random.default_rng(2)
        a = 0.6 + rng.normal(0, 0.05, 12)
        b = 0.5 + rng.normal(0, 0.05, 12)
        result = wilcoxon_signed_rank(a, b, alternative="greater")
        assert result["method"] == "wilcoxon"
        assert 0 <= result["p_value"] <= 1

    def test_stat_methods_registry_contains_all_three(self):
        assert set(STAT_METHODS) == {"ttest_1samp_diff", "ttest_rel", "wilcoxon"}


# =====================================================
# compare_conditions_by_bin
# =====================================================

class TestCompareConditionsByBin:
    def test_one_row_per_bin_with_named_method(self):
        rng = np.random.default_rng(0)
        rows = []
        for subj in range(6):
            for bin_id in (0, 1):
                rows.append({"subject": str(subj), "bin": bin_id,
                             "pos": 0.6 + rng.normal(0, 0.01), "neg": 0.5 + rng.normal(0, 0.01)})
        per_subject_bin = pd.DataFrame(rows).set_index(["subject", "bin"])
        result = compare_conditions_by_bin(per_subject_bin, "pos", "neg", method="ttest_1samp_diff", alternative="greater")
        assert len(result) == 2
        assert (result["p_value"] < 0.01).all()

    def test_custom_callable_method(self):
        def always_significant(a_vals, b_vals, alternative):
            return {"stat": 99.0, "p_value": 0.0001}

        per_subject_bin = pd.DataFrame({
            "subject": ["01", "02"], "bin": [0, 0], "pos": [0.6, 0.7], "neg": [0.4, 0.5],
        }).set_index(["subject", "bin"])
        result = compare_conditions_by_bin(per_subject_bin, "pos", "neg", method=always_significant, alternative="greater")
        assert result.iloc[0]["stat"] == 99.0

    def test_bin_with_fewer_than_2_subjects_skipped(self):
        per_subject_bin = pd.DataFrame({
            "subject": ["01"], "bin": [0], "pos": [0.6], "neg": [0.4],
        }).set_index(["subject", "bin"])
        result = compare_conditions_by_bin(per_subject_bin, "pos", "neg")
        assert result.empty

    def test_bins_meta_merged_in(self):
        rng = np.random.default_rng(0)
        rows = [{"subject": str(s), "bin": 0, "pos": 0.6 + rng.normal(0, 0.01), "neg": 0.5 + rng.normal(0, 0.01)} for s in range(4)]
        per_subject_bin = pd.DataFrame(rows).set_index(["subject", "bin"])
        bins_meta = pd.DataFrame({"bin": [0], "tr_start": [1], "tr_end": [3]})
        result = compare_conditions_by_bin(per_subject_bin, "pos", "neg", bins_meta=bins_meta)
        assert result.iloc[0]["tr_start"] == 1
        assert result.iloc[0]["tr_end"] == 3


# =====================================================
# plot_conditions (smoke test)
# =====================================================

class TestPlotConditions:
    def test_draws_one_line_per_present_condition(self):
        idx = pd.MultiIndex.from_product([["01", "02"], range(4)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": np.linspace(0.4, 0.8, 8), "neg": np.linspace(0.3, 0.6, 8)}, index=idx)
        fig, ax = plt.subplots()
        plot_conditions(ax, wide, {"pos": "red", "neg": "blue"}, title="test")
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_missing_condition_column_skipped_not_raised(self):
        idx = pd.MultiIndex.from_product([["01"], range(4)], names=["subject", "window_index"])
        wide = pd.DataFrame({"pos": np.linspace(0.4, 0.8, 4)}, index=idx)
        fig, ax = plt.subplots()
        plot_conditions(ax, wide, {"pos": "red", "neg": "blue"})
        assert len(ax.lines) == 1
        plt.close(fig)
