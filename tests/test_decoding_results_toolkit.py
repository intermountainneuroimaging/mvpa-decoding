"""_interactive_notebooks/decoding_results_toolkit.py: the generic, reusable building
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

from _interactive_notebooks.decoding_results_toolkit import (
    load_decoding_results,
    balance_trial_table,
    weighted_group_mean,
    describe_baseline_groups,
    baseline_group_names,
    balance_preceding_category,
    select_evidence_value,
    apply_filters,
    derive_label,
    aggregate_by_subject_window,
    average_across_groups,
    max_reliable_window_index,
    subtract_baseline,
    bin_by_size,
    bin_by_edges,
    sliding_windows,
    ttest_1samp_diff,
    ttest_rel,
    wilcoxon_signed_rank,
    compare_conditions_by_bin,
    plot_conditions,
    annotate_significance,
    mark_significance_onset,
    first_significant_bin,
    mark_onsets_in_rows,
    draw_significance_table,
    draw_stat_strip,
    baseline_correct_timecourses,
    baseline_only_bins,
    STAT_METHODS,
    fdr_correct,
    split_half_group_difference,
    _split_masks,
    _half_means,
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

    def test_self_value_is_numeric_dtype_with_nan_regressor_label(self):
        # regression: a row-wise `.apply(lambda r: ... if pd.notna(...) else
        # pd.NA)` (the pattern this function replaces in
        # valence_evidence_by_operation.ipynb) can silently produce an
        # object-dtype column when floats are mixed with pd.NA, which later
        # crashes scipy.stats.ttest_1samp ("data type <class
        # 'numpy.object_'> not inexact") -- the vectorized numpy-indexing
        # approach here must never do that, even with real NaN present
        # before the unknown rows are dropped
        df = pd.DataFrame({
            "regressor_label": ["maintain", None, "suppress"],
            "evidence_maintain": [0.7, 0.5, 0.2],
            "evidence_suppress": [0.3, 0.5, 0.8],
        })
        result = select_evidence_value(df, value="self")
        assert pd.api.types.is_numeric_dtype(result["value"])

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
# max_reliable_window_index
# =====================================================

class TestMaxReliableWindowIndex:
    def _df(self, rows):
        return pd.DataFrame(rows, columns=["subject", "operation", "window_index"])

    def test_finds_the_last_window_with_full_coverage(self):
        # 4 subjects at window 0-1, only 1 subject still has data at window 2
        # (a variable-trial-length tail) -- window 2 must be excluded
        rows = [(s, "maintain", w) for s in "1234" for w in (0, 1)]
        rows += [("1", "maintain", 2)]
        df = self._df(rows)
        assert max_reliable_window_index(df, group_cols=["operation"], min_fraction=1.0) == 1

    def test_the_weakest_combination_sets_the_cutoff(self):
        # "maintain" has full coverage through window 2, but "suppress"
        # already drops off at window 1 -- the cutoff must reflect the
        # weakest combination, not the strongest
        rows = [(s, "maintain", w) for s in "1234" for w in (0, 1, 2)]
        rows += [(s, "suppress", w) for s in "1234" for w in (0,)]
        rows += [("1", "suppress", 1)]
        df = self._df(rows)
        assert max_reliable_window_index(df, group_cols=["operation"], min_fraction=1.0) == 0

    def test_min_fraction_allows_partial_dropout(self):
        # 4 subjects at window 0, 3 of 4 (75%) at window 1 -- a 0.5 threshold
        # still accepts window 1, a 0.9 threshold does not
        rows = [(s, "maintain", 0) for s in "1234"]
        rows += [(s, "maintain", 1) for s in "123"]
        df = self._df(rows)
        assert max_reliable_window_index(df, group_cols=["operation"], min_fraction=0.5) == 1
        assert max_reliable_window_index(df, group_cols=["operation"], min_fraction=0.9) == 0

    def test_raises_when_nothing_meets_the_threshold(self):
        # 2 total subjects, but the only window_index that exists has just 1
        # -- no window anywhere reaches the required 100% coverage
        rows = [("1", "maintain", 0), ("2", "maintain", 1)]
        df = self._df(rows)
        with pytest.raises(ValueError, match="no window_index"):
            max_reliable_window_index(df, group_cols=["operation"], min_fraction=1.0)


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


class TestSlidingWindows:
    def _wide(self, n=8):
        idx = pd.MultiIndex.from_product([["01", "02"], range(n)], names=["subject", "window_index"])
        return pd.DataFrame({"pos": np.arange(2 * n, dtype=float), "neg": np.zeros(2 * n)}, index=idx)

    def test_one_window_per_start_tr_each_the_mean_of_its_trs(self):
        per_subject_bin, meta = sliding_windows(self._wide(), window=3, tr_offset=0)
        assert list(meta["bin"]) == [0, 1, 2, 3, 4, 5]          # 8 TRs hold 6 windows of 3
        assert per_subject_bin.loc[("01", 0), "pos"] == pytest.approx(1.0)       # mean(0, 1, 2)
        assert per_subject_bin.loc[("01", 3), "pos"] == pytest.approx(4.0)       # mean(3, 4, 5): shifted by 1 TR only
        assert per_subject_bin.loc[("02", 5), "pos"] == pytest.approx(14.0)      # mean(13, 14, 15)

    def test_each_window_is_reported_at_the_tr_it_is_centred_on(self):
        _, meta = sliding_windows(self._wide(), window=3)         # 1-indexed TRs
        assert list(zip(meta["window_tr_start"], meta["window_tr_end"]))[:2] == [(1, 3), (2, 4)]
        assert list(meta["tr_start"]) == [2, 3, 4, 5, 6, 7] and list(meta["tr_end"]) == list(meta["tr_start"])

    def test_report_at_start_and_end_and_even_windows(self):
        assert list(sliding_windows(self._wide(), 3, report_at="start")[1]["tr_start"]) == [1, 2, 3, 4, 5, 6]
        assert list(sliding_windows(self._wide(), 3, report_at="end")[1]["tr_start"]) == [3, 4, 5, 6, 7, 8]
        with pytest.raises(ValueError):
            sliding_windows(self._wide(), 2)                      # no centre TR

    def test_window_one_is_bin_size_one(self):
        sliding, _ = sliding_windows(self._wide(), window=1, tr_offset=0)
        binned, _ = bin_by_size(self._wide(), bin_size=1, tr_offset=0)
        pd.testing.assert_frame_equal(sliding, binned, check_names=False)

    def test_window_longer_than_the_data_gives_no_windows(self):
        per_subject_bin, meta = sliding_windows(self._wide(2), window=3)
        assert per_subject_bin.empty and meta.empty


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
# fdr_correct / FDR in compare_conditions_by_bin
# =====================================================

class TestFdrCorrect:
    def test_bh_matches_hand_calculation_and_keeps_order(self):
        # sorted: .001, .04, .2 -> .001*3/1, .04*3/2, .2*3/3 = .003, .06, .2
        np.testing.assert_allclose(fdr_correct([0.001, 0.2, 0.04]), [0.003, 0.2, 0.06])

    def test_bh_equal_adjusted_values_when_all_just_pass(self):
        np.testing.assert_allclose(fdr_correct([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05] * 5)

    def test_adjusted_never_below_raw_and_never_above_one(self):
        p = np.random.default_rng(0).uniform(0, 1, 50)
        q = fdr_correct(p)
        assert (q >= p - 1e-12).all() and (q <= 1.0).all()

    def test_nan_passes_through_and_is_not_counted_as_a_test(self):
        # without the NaN counting, m would be 3 and these would be .03/.045
        np.testing.assert_allclose(fdr_correct([0.01, np.nan, 0.03]), [0.02, np.nan, 0.03])

    def test_by_is_more_conservative_than_bh(self):
        p = [0.001, 0.01, 0.04, 0.3]
        assert (fdr_correct(p, "by") >= fdr_correct(p, "bh")).all()

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown FDR method"):
            fdr_correct([0.1, 0.2], method="holm")

    def test_all_nan_input_is_fine(self):
        assert np.isnan(fdr_correct([np.nan, np.nan])).all()


class TestCompareConditionsByBinFdr:
    def _per_subject_bin(self, n_bins=4, n_subj=8, seed=0):
        rng = np.random.default_rng(seed)
        rows = [{"subject": str(s), "bin": b, "pos": 0.55 + rng.normal(0, 0.05), "neg": 0.5 + rng.normal(0, 0.05)}
                for s in range(n_subj) for b in range(n_bins)]
        return pd.DataFrame(rows).set_index(["subject", "bin"])

    def test_p_fdr_column_follows_p_value_and_is_bh_across_the_bins(self):
        result = compare_conditions_by_bin(self._per_subject_bin(), "pos", "neg", alternative="two-sided")
        cols = list(result.columns)
        assert cols[cols.index("p_value") + 1] == "p_fdr"
        np.testing.assert_allclose(result["p_fdr"], fdr_correct(result["p_value"], "bh"))
        assert (result["p_fdr"] >= result["p_value"] - 1e-12).all()

    def test_a_raw_significant_bin_can_lose_significance_after_correction(self):
        ps = iter([0.04, 0.045, 0.5, 0.9])  # one per bin, in bin order
        def fixed_p(a, b, alternative):
            return {"stat": 0.0, "p_value": next(ps)}
        result = compare_conditions_by_bin(self._per_subject_bin(n_bins=4), "pos", "neg", method=fixed_p)
        assert (result["p_value"].iloc[:2] < 0.05).all()
        np.testing.assert_allclose(result["p_fdr"], [0.09, 0.09, 2 / 3, 0.9])
        assert (result["p_fdr"] >= 0.05).all()

    def test_fdr_none_gives_raw_p_only(self):
        result = compare_conditions_by_bin(self._per_subject_bin(), "pos", "neg", fdr=None)
        assert "p_fdr" not in result.columns

    def test_also_fdr_adds_the_extra_method_over_the_same_family(self):
        result = compare_conditions_by_bin(self._per_subject_bin(n_bins=6), "pos", "neg", fdr="bh", fdr_from_bin=2,
                                           also_fdr=("by",))
        cols = list(result.columns)
        assert cols[cols.index("p_fdr") + 1] == "p_fdr_by"
        later = result["bin"] >= 2
        np.testing.assert_allclose(result.loc[later, "p_fdr_by"], fdr_correct(result.loc[later, "p_value"], "by"))
        assert result.loc[~later, "p_fdr_by"].isna().all()
        assert (result.loc[later, "p_fdr_by"] >= result.loc[later, "p_fdr"] - 1e-12).all()

    def test_bins_before_test_from_bin_are_not_tested_at_all(self):
        result = compare_conditions_by_bin(self._per_subject_bin(n_bins=6), "pos", "neg", alternative="two-sided",
                                           fdr_from_bin=2, test_from_bin=2)
        early, late = result[result["bin"] < 2], result[result["bin"] >= 2]
        assert len(early) == 2 and early[["stat", "p_value", "p_fdr"]].isna().all().all()
        assert early["mean_diff"].notna().all() and early["n"].eq(8).all()      # described, not tested
        assert late[["stat", "p_value", "p_fdr"]].notna().all().all()
        np.testing.assert_allclose(late["p_fdr"], fdr_correct(late["p_value"], "bh"))   # the family is the tested bins

    def test_by_method_selectable(self):
        bh = compare_conditions_by_bin(self._per_subject_bin(), "pos", "neg", fdr="bh")
        by = compare_conditions_by_bin(self._per_subject_bin(), "pos", "neg", fdr="by")
        assert (by["p_fdr"] >= bh["p_fdr"] - 1e-12).all()

    def test_each_call_is_its_own_family(self):
        # correcting two contrasts separately must differ from one big family
        a = compare_conditions_by_bin(self._per_subject_bin(seed=1), "pos", "neg")
        b = compare_conditions_by_bin(self._per_subject_bin(seed=2), "pos", "neg")
        pooled = fdr_correct(np.concatenate([a["p_value"], b["p_value"]]))
        assert not np.allclose(np.concatenate([a["p_fdr"], b["p_fdr"]]), pooled)

    def test_significance_markers_can_use_p_fdr(self):
        fig, ax = plt.subplots()
        stats = pd.DataFrame({"bin": [0, 1], "tr_start": [0, 3], "tr_end": [2, 5],
                              "p_value": [0.04, 0.04], "p_fdr": [0.08, 0.01], "mean_diff": [0.1, 0.2]})
        annotate_significance(ax, stats, alpha=0.05, p_col="p_fdr")
        assert len(ax.texts) == 1  # only the bin whose FDR-adjusted p passes, not both raw-significant ones
        plt.close(fig)


# =====================================================
# split-half reliability of a group-level finding
# =====================================================

def _trial_table(n_subj=30, n_trials=12, n_times=26, effect_by_bin=None, noise_sd=0.3, seed=0, missing=0.0):
    """Long-format trial x timepoint table with a known pos - neg effect per 3-TR bin."""
    rng = np.random.default_rng(seed)
    effect = np.zeros(n_times // 3 + 1) if effect_by_bin is None else np.asarray(effect_by_bin, dtype=float)
    rows = []
    for s in range(n_subj):
        base = rng.normal(0.4, 0.1)
        for valence, sign in (("pos", 0.5), ("neg", -0.5)):
            for trial in range(n_trials):
                for w in range(n_times):
                    if missing and rng.random() < missing:
                        continue
                    rows.append((str(s), valence, "run1.nii.gz", trial, w, base + sign * effect[w // 3] + rng.normal(0, noise_sd)))
    return pd.DataFrame(rows, columns=["subject", "valence", "boldfile", "trial_index", "window_index", "value"])


class TestSplitHalfHelpers:
    def test_split_masks_are_equal_disjoint_halves(self):
        mask_a, mask_b = _split_masks(10, 200, np.random.default_rng(0))
        assert (mask_a.sum(axis=1) == 5).all() and (mask_b.sum(axis=1) == 5).all()
        assert not (mask_a & mask_b).any()
        assert len({tuple(r) for r in mask_a}) > 100  # genuinely different splits

    def test_split_masks_odd_count_drops_one_unit(self):
        mask_a, mask_b = _split_masks(7, 50, np.random.default_rng(0))
        assert (mask_a.sum(axis=1) == 3).all() and (mask_b.sum(axis=1) == 3).all()
        assert ((mask_a | mask_b).sum(axis=1) == 6).all()

    def test_half_means_are_nan_aware_means_over_the_masked_units(self):
        matrix = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        mask = np.array([[True, True, False], [False, True, True]])
        out = _half_means(matrix, mask)
        np.testing.assert_allclose(out, [[2.0, 2.0], [4.0, 6.0]])  # NaN skipped, not propagated

    def test_half_means_all_nan_column_is_nan(self):
        out = _half_means(np.array([[np.nan], [np.nan]]), np.array([[True, True]]))
        assert np.isnan(out).all()


class TestSplitHalfGroupDifference:
    def test_full_mean_diff_is_the_stats_pipelines_own_mean_diff(self):
        # incl. missing timepoints -- the number this reports must be the one the
        # statistics tables report, bin for bin
        df = _trial_table(n_subj=20, effect_by_bin=[0, 0.02, 0.05, 0, 0, 0.1, 0, 0, 0], missing=0.03, seed=1)
        res = split_half_group_difference(df, "valence", "pos", "neg", bin_size=3, n_splits=50, seed=0)
        wide = aggregate_by_subject_window(df, condition_col="valence")
        per_bin, meta = bin_by_size(wide, bin_size=3, tr_offset=0)
        stats = compare_conditions_by_bin(per_bin, "pos", "neg", alternative="two-sided", bins_meta=meta)
        merged = res.merge(stats, on="bin", suffixes=("", "_stats"))
        assert len(merged) == len(stats) == len(res)
        np.testing.assert_allclose(merged["full_mean_diff"], merged["mean_diff"], atol=1e-12)
        assert (merged["tr_start"] == merged["tr_start_stats"]).all() and (merged["tr_end"] == merged["tr_end_stats"]).all()

    def test_sliding_full_mean_diff_is_the_sliding_stats_own_mean_diff(self):
        df = _trial_table(n_subj=20, effect_by_bin=[0, 0.02, 0.05, 0, 0, 0.1, 0, 0, 0], missing=0.03, seed=1)
        res = split_half_group_difference(df, "valence", "pos", "neg", bin_size=3, n_splits=30, seed=0, sliding=True)
        wide = aggregate_by_subject_window(df, condition_col="valence")
        per_bin, meta = sliding_windows(wide, window=3, tr_offset=0)
        stats = compare_conditions_by_bin(per_bin, "pos", "neg", alternative="two-sided", bins_meta=meta)
        merged = res.merge(stats, on="bin", suffixes=("", "_stats"))
        assert len(merged) == len(stats) == len(res) > len(df["window_index"].unique()) // 3     # one window per start TR
        np.testing.assert_allclose(merged["full_mean_diff"], merged["mean_diff"], atol=1e-12)
        assert (merged["tr_start"] == merged["tr_start_stats"]).all() and (merged["window_tr_end"] == merged["window_tr_end_stats"]).all()

    def test_only_the_two_named_conditions_are_used(self):
        base = _trial_table(n_subj=12, seed=2)
        extra = base[base["valence"] == "pos"].assign(valence="other", value=lambda d: d["value"] + 5.0)
        with_extra = pd.concat([base, extra], ignore_index=True)
        a = split_half_group_difference(with_extra, "valence", "pos", "neg", n_splits=40, seed=3)
        b = split_half_group_difference(base, "valence", "pos", "neg", n_splits=40, seed=3)
        pd.testing.assert_frame_equal(a, b)

    def test_strong_positive_effect_replicates_in_both_halves(self):
        df = _trial_table(effect_by_bin=[0.2] * 9, seed=4)
        res = split_half_group_difference(df, "valence", "pos", "neg", n_splits=200, seed=0)
        assert (res["p_half_a_gt_b"] > 0.99).all() and (res["p_same_direction"] > 0.99).all()
        assert (res["p_halves_agree"] > 0.99).all()
        assert (res["half_ci_low"] > 0).all()

    def test_strong_negative_effect_is_direction_consistent_but_not_a_gt_b(self):
        df = _trial_table(effect_by_bin=[-0.2] * 9, seed=5)
        res = split_half_group_difference(df, "valence", "pos", "neg", n_splits=200, seed=0)
        assert (res["p_half_a_gt_b"] < 0.01).all()      # almost no half shows pos > neg
        assert (res["p_same_direction"] > 0.99).all()   # ...because the finding is neg > pos, reliably
        assert (res["half_ci_high"] < 0).all()

    def test_halves_agree_is_exactly_two_p_same_direction_minus_one(self):
        res = split_half_group_difference(_trial_table(effect_by_bin=[0, 0, 0.02, 0.04, 0, 0, 0, 0, 0], seed=6),
                                          "valence", "pos", "neg", n_splits=400, seed=0)
        np.testing.assert_allclose(res["p_halves_agree"], 2 * res["p_same_direction"] - 1, atol=1e-9)

    def test_band_is_centered_on_the_full_data_estimate(self):
        res = split_half_group_difference(_trial_table(effect_by_bin=[0.05] * 9, seed=7),
                                          "valence", "pos", "neg", n_splits=500, seed=0)
        assert (np.abs((res["half_ci_low"] + res["half_ci_high"]) / 2 - res["full_mean_diff"]) < 0.01).all()

    def test_no_true_effect_does_not_hover_at_chance(self):
        # the point of the docstring's warning: halves share the data's realized
        # noise, so a no-effect bin averages p_same_direction ~0.75 (never below
        # 0.5) and p_halves_agree ~0.5 spread over 0-1 -- not "50% = nothing"
        rng = np.random.default_rng(0)
        same, agree = [], []
        for _ in range(60):
            res = split_half_group_difference(_trial_table(n_subj=15, n_trials=8, n_times=3, seed=int(rng.integers(1e9))),
                                              "valence", "pos", "neg", n_splits=200, seed=int(rng.integers(1e9)))
            same.append(res.loc[0, "p_same_direction"]); agree.append(res.loc[0, "p_halves_agree"])
        assert 0.66 < np.mean(same) < 0.84
        assert min(same) > 0.48
        assert 0.36 < np.mean(agree) < 0.64
        assert np.std(agree) > 0.2  # broad, not concentrated at any one value

    def test_subjects_with_too_few_trials_in_either_condition_are_left_out(self):
        df = _trial_table(n_subj=10, n_trials=6, seed=8)
        thin = ~((df["subject"].isin(["0", "1"])) & (df["valence"] == "neg") & (df["trial_index"] >= 1))  # 1 neg trial
        thin &= ~((df["subject"] == "2") & (df["valence"] == "pos") & (df["trial_index"] >= 3))         # 3 pos trials: fine
        res = split_half_group_difference(df[thin], "valence", "pos", "neg", n_splits=50, seed=0, min_trials=2)
        assert (res["n_subjects"] == 8).all()

    def test_min_trials_is_never_below_two(self):
        df = _trial_table(n_subj=5, n_trials=1, seed=9)
        assert split_half_group_difference(df, "valence", "pos", "neg", n_splits=20, min_trials=1).empty

    def test_same_seed_is_deterministic(self):
        df = _trial_table(n_subj=10, seed=10)
        a = split_half_group_difference(df, "valence", "pos", "neg", n_splits=60, seed=5)
        b = split_half_group_difference(df, "valence", "pos", "neg", n_splits=60, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_trial_identity_comes_from_trial_cols(self):
        # two boldfiles sharing trial indices must count as distinct trials
        df = _trial_table(n_subj=6, n_trials=2, seed=11)
        df = pd.concat([df, df.assign(boldfile="run2.nii.gz")], ignore_index=True)
        res = split_half_group_difference(df, "valence", "pos", "neg", n_splits=30, seed=0)
        assert (res["n_subjects"] == 6).all()  # 4 trials per condition per subject -> eligible (2 would also be)

    def test_no_rows_for_the_conditions_returns_empty(self):
        assert split_half_group_difference(_trial_table(n_subj=4), "valence", "x", "y").empty

    def test_columns(self):
        res = split_half_group_difference(_trial_table(n_subj=8), "valence", "pos", "neg", n_splits=30)
        assert list(res.columns) == ["bin", "tr_start", "tr_end", "n_subjects", "full_mean_diff", "half_ci_low",
                                     "half_ci_high", "p_half_a_gt_b", "p_same_direction", "p_halves_agree", "n_splits"]


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


# =====================================================
# annotate_significance
# =====================================================

class TestAnnotateSignificance:
    def test_draws_a_bar_and_star_only_for_significant_bins(self):
        bin_stats = pd.DataFrame({
            "tr_start": [1, 4, 7], "tr_end": [3, 6, 9], "p_value": [0.2, 0.001, 0.04],
        })
        fig, ax = plt.subplots()
        ax.plot([1, 9], [0, 1])  # something already on the axes, for a real ylim
        n_lines_before = len(ax.lines)
        n_texts_before = len(ax.texts)
        annotate_significance(ax, bin_stats, alpha=0.05)
        # one new line (the bar) + one new text ("*") per significant bin (2 of 3)
        assert len(ax.lines) == n_lines_before + 2
        assert len(ax.texts) == n_texts_before + 2
        assert [t.get_text() for t in ax.texts] == ["*", "*"]
        plt.close(fig)

    def test_no_significant_bins_draws_nothing(self):
        bin_stats = pd.DataFrame({"tr_start": [1], "tr_end": [3], "p_value": [0.5]})
        fig, ax = plt.subplots()
        ax.plot([1, 3], [0, 1])
        n_lines_before = len(ax.lines)
        annotate_significance(ax, bin_stats, alpha=0.05)
        assert len(ax.lines) == n_lines_before
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_extends_ylim_to_make_room(self):
        bin_stats = pd.DataFrame({"tr_start": [1], "tr_end": [3], "p_value": [0.001]})
        fig, ax = plt.subplots()
        ax.plot([1, 3], [0, 1])
        ymin, ymax_before = ax.get_ylim()
        annotate_significance(ax, bin_stats, alpha=0.05)
        assert ax.get_ylim()[1] > ymax_before


class TestMarkSignificanceOnset:
    def test_marks_only_the_first_bin_of_a_contiguous_significant_run(self):
        # bins 1-2-3 all significant, back to back -- only bin 1 (the run's
        # own onset) gets a marker, not every bin in the run
        bin_stats = pd.DataFrame({
            "bin": [0, 1, 2, 3, 4],
            "tr_start": [0, 3, 6, 9, 12],
            "mean_diff": [0.01, -0.05, -0.08, -0.10, -0.02],
            "p_value": [0.5, 0.01, 0.02, 0.001, 0.6],
        })
        fig, ax = plt.subplots()
        ax.plot([0, 12], [0, -0.1])
        mark_significance_onset(ax, bin_stats, alpha=0.05)
        collections = ax.collections
        assert len(collections) == 1
        offsets = collections[0].get_offsets()
        assert len(offsets) == 1
        assert offsets[0][0] == 3  # bin 1's own tr_start
        assert offsets[0][1] == pytest.approx(-0.05)  # bin 1's own mean_diff
        plt.close(fig)

    def test_marks_each_resumption_of_a_stopped_and_restarted_run(self):
        # significant at bins 0-1, not at bin 2, significant again at bin 3
        # -- two onsets: bin 0 and bin 3
        bin_stats = pd.DataFrame({
            "bin": [0, 1, 2, 3],
            "tr_start": [0, 3, 6, 9],
            "mean_diff": [-0.05, -0.06, 0.0, -0.07],
            "p_value": [0.01, 0.02, 0.5, 0.03],
        })
        fig, ax = plt.subplots()
        ax.plot([0, 9], [0, -0.07])
        mark_significance_onset(ax, bin_stats, alpha=0.05)
        offsets = ax.collections[0].get_offsets()
        assert sorted(x for x, y in offsets) == [0, 9]
        plt.close(fig)

    def _mixed_stats(self):
        # significant ABOVE zero at bins 0-1, not at 2, significant BELOW zero at bins 3-4 and again at 6
        return pd.DataFrame({
            "bin": [0, 1, 2, 3, 4, 5, 6],
            "tr_start": [0, 3, 6, 9, 12, 15, 18],
            "mean_diff": [0.05, 0.06, 0.0, -0.07, -0.08, -0.01, -0.09],
            "p_value": [0.01, 0.02, 0.5, 0.03, 0.01, 0.6, 0.04],
        })

    def test_direction_negative_ignores_significant_positive_bins(self):
        fig, ax = plt.subplots()
        ax.plot([0, 18], [0, 0])
        mark_significance_onset(ax, self._mixed_stats(), alpha=0.05, direction="negative")
        assert sorted(x for x, y in ax.collections[0].get_offsets()) == [9, 18]
        plt.close(fig)

    def test_direction_positive_ignores_significant_negative_bins(self):
        fig, ax = plt.subplots()
        ax.plot([0, 18], [0, 0])
        mark_significance_onset(ax, self._mixed_stats(), alpha=0.05, direction="positive")
        assert [x for x, y in ax.collections[0].get_offsets()] == [0]
        plt.close(fig)

    def test_first_only_marks_just_the_earliest_qualifying_bin(self):
        fig, ax = plt.subplots()
        ax.plot([0, 18], [0, 0])
        mark_significance_onset(ax, self._mixed_stats(), alpha=0.05, direction="negative", first_only=True)
        offsets = ax.collections[0].get_offsets()
        assert len(offsets) == 1 and offsets[0][0] == 9 and offsets[0][1] == pytest.approx(-0.07)
        plt.close(fig)

    def test_first_only_without_direction_takes_the_first_significant_bin_of_either_sign(self):
        fig, ax = plt.subplots()
        ax.plot([0, 18], [0, 0])
        mark_significance_onset(ax, self._mixed_stats(), alpha=0.05, first_only=True)
        assert [x for x, y in ax.collections[0].get_offsets()] == [0]
        plt.close(fig)

    def test_direction_with_no_qualifying_bin_draws_nothing(self):
        bin_stats = pd.DataFrame({"bin": [0, 1], "tr_start": [0, 3], "mean_diff": [0.05, 0.06], "p_value": [0.01, 0.02]})
        fig, ax = plt.subplots()
        ax.plot([0, 3], [0, 0])
        mark_significance_onset(ax, bin_stats, alpha=0.05, direction="negative", first_only=True)
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_bad_direction_raises(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="direction"):
            mark_significance_onset(ax, self._mixed_stats(), direction="down")
        plt.close(fig)

    def test_no_significant_bins_draws_nothing(self):
        bin_stats = pd.DataFrame({"bin": [0], "tr_start": [0], "mean_diff": [0.01], "p_value": [0.5]})
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        mark_significance_onset(ax, bin_stats, alpha=0.05)
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_marker_style_is_a_downward_triangle_by_default(self):
        bin_stats = pd.DataFrame({"bin": [0], "tr_start": [0], "mean_diff": [-0.05], "p_value": [0.01]})
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, -0.05])
        mark_significance_onset(ax, bin_stats, alpha=0.05)
        paths = ax.collections[0].get_paths()
        assert len(paths) == 1
        # matplotlib's "v" marker path -- just confirm it's not the default circle
        from matplotlib.markers import MarkerStyle
        assert paths[0].vertices.shape == MarkerStyle("v").get_path().vertices.shape
        plt.close(fig)


class TestFirstSignificantBin:
    STATS = pd.DataFrame({
        "bin": [0, 1, 2, 3, 4],
        "tr_start": [0, 3, 6, 9, 12],
        "mean_diff": [0.05, 0.0, -0.04, -0.07, -0.08],
        "p_value": [0.01, 0.5, 0.2, 0.03, 0.01],
    })

    def test_any_direction_returns_earliest_significant(self):
        assert first_significant_bin(self.STATS, alpha=0.05)["bin"] == 0

    def test_negative_skips_significant_positive_bins_and_nonsignificant_negatives(self):
        row = first_significant_bin(self.STATS, alpha=0.05, direction="negative")
        assert row["bin"] == 3 and row["tr_start"] == 9

    def test_positive(self):
        assert first_significant_bin(self.STATS, alpha=0.05, direction="positive")["bin"] == 0

    def test_none_when_nothing_qualifies(self):
        assert first_significant_bin(self.STATS, alpha=0.001, direction="negative") is None
        assert first_significant_bin(self.STATS.iloc[0:0], alpha=0.05) is None

    def test_uses_given_p_column_and_rejects_bad_direction(self):
        stats = self.STATS.assign(p_fdr=[0.9, 0.9, 0.9, 0.9, 0.02])
        assert first_significant_bin(stats, p_col="p_fdr", direction="negative")["bin"] == 4
        with pytest.raises(ValueError, match="direction"):
            first_significant_bin(stats, direction="sideways")


class TestMarkOnsetsInRows:
    def _axes(self):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([0, 12], [-0.1, 0.1])
        return fig, ax

    def test_same_x_markers_get_distinct_rows_so_they_cannot_overlap(self):
        fig, ax = self._axes()
        mark_onsets_in_rows(ax, [(6.9, "blue"), (6.9, "red"), (6.9, "orange")])
        fig.canvas.draw()
        pts = [ax.transAxes.inverted().transform(ax.transData.transform((0, 0)))]  # force transforms to exist
        offs = [c.get_offsets()[0] for c in ax.collections]
        disp = np.array([ax.get_xaxis_transform().transform(o) for o in offs])
        assert len(disp) == 3 and all(abs(disp[:, 0] - disp[0, 0]) < 1e-6)       # same time position kept
        gaps = np.abs(np.diff(np.sort(disp[:, 1])))
        marker_height_px = np.sqrt(70) * fig.dpi / 72                              # scatter size is area in pt^2
        assert gaps.min() > marker_height_px
        plt.close(fig)

    def test_rows_follow_input_order_top_to_bottom_and_keep_x(self):
        fig, ax = self._axes()
        mark_onsets_in_rows(ax, [(3.0, "blue"), (6.0, "red")])
        ys = [ax.get_xaxis_transform().transform(c.get_offsets()[0])[1] for c in ax.collections]
        assert ys[0] > ys[1]      # first item is the top row
        assert [c.get_offsets()[0][0] for c in ax.collections] == [3.0, 6.0]
        plt.close(fig)

    def test_none_leaves_its_row_empty_but_reserves_it(self):
        fig, ax = self._axes()
        mark_onsets_in_rows(ax, [(3.0, "blue"), (None, "red"), (6.0, "orange")])
        assert len(ax.collections) == 2
        ys = sorted(ax.get_xaxis_transform().transform(c.get_offsets()[0])[1] for c in ax.collections)
        middle_row = ax.get_xaxis_transform().transform((0, 0.03 + 1.5 * 0.055))[1]
        assert ys[0] < middle_row < ys[1]
        plt.close(fig)

    def test_strip_is_below_the_data_and_only_extends_ylim_downward(self):
        fig, ax = self._axes()
        ymin, ymax = ax.get_ylim()
        mark_onsets_in_rows(ax, [(3.0, "blue"), (6.0, "red"), (9.0, "orange")])
        new_ymin, new_ymax = ax.get_ylim()
        assert new_ymax == ymax and new_ymin < ymin
        top_of_strip = ax.transData.inverted().transform(ax.transAxes.transform((0, 0.03 + 3 * 0.055)))[1]
        assert top_of_strip == pytest.approx(ymin, abs=1e-6)       # old data floor sits just above the strip
        plt.close(fig)

    def test_empty_list_does_nothing(self):
        fig, ax = self._axes()
        before = ax.get_ylim()
        mark_onsets_in_rows(ax, [])
        assert ax.get_ylim() == before and len(ax.collections) == 0
        plt.close(fig)


class TestDrawSignificanceTable:
    @staticmethod
    def _stats(p_values):
        return pd.DataFrame({
            "tr_start": [0, 3, 6][:len(p_values)],
            "tr_end": [2, 5, 8][:len(p_values)],
            "p_fdr": p_values,
        })

    def _rows(self):
        return [
            {"parts": [("A", "blue"), (" vs. ", "black"), ("B", "red")], "stats": self._stats([0.01, 0.5, 0.04])},
            {"parts": [("A", "blue"), (" vs. ", "black"), ("C", "orange")], "stats": self._stats([0.9, 0.9, 0.001])},
        ]

    def _table(self, rows=None, **kw):
        fig, (top, tab) = plt.subplots(2, 1, sharex=True)
        top.plot([0, 9], [0, 1])
        draw_significance_table(tab, rows or self._rows(), alpha=0.05, p_col="p_fdr", tr_seconds=2.0, **kw)
        return fig, tab

    def test_one_cell_per_window_shaded_by_significance(self):
        fig, tab = self._table()
        cells = [p for p in tab.patches]
        assert len(cells) == 6
        dark = {(p.get_x(), p.get_y()) for p in cells if p.get_facecolor()[0] < 0.7}
        # row 0 is the TOP row (y=1): windows 0 and 2 significant; row 1 (y=0): window 2 only
        assert dark == {(0.0, 1), (12.0, 1), (12.0, 0)}
        plt.close(fig)

    def test_cells_sit_at_their_real_time_positions(self):
        fig, tab = self._table()
        first_row = sorted((p.get_x(), p.get_width()) for p in tab.patches if p.get_y() == 1)
        assert first_row == [(0.0, 6.0), (6.0, 6.0), (12.0, 6.0)]   # TRs 0-2, 3-5, 6-8 at 2 s/TR, end inclusive
        plt.close(fig)

    def test_missing_p_leaves_the_cell_blank(self):
        rows = [{"parts": [("A", "blue")], "stats": self._stats([0.01, np.nan, 0.04])}]
        fig, tab = self._table(rows)
        assert len(tab.patches) == 2
        plt.close(fig)

    def test_row_labels_are_colored_pieces_in_row_order(self):
        from matplotlib.offsetbox import AnchoredOffsetbox
        fig, tab = self._table()
        boxes = [a for a in tab.artists if isinstance(a, AnchoredOffsetbox)]
        assert len(boxes) == 2
        texts = [[t.get_text() for t in b.get_child().get_children()] for b in boxes]
        colors = [[t._text.get_color() for t in b.get_child().get_children()] for b in boxes]
        assert texts == [["A", " vs. ", "B"], ["A", " vs. ", "C"]]
        assert colors == [["blue", "black", "red"], ["blue", "black", "orange"]]
        plt.close(fig)

    def test_windows_are_numbered_and_legend_names_the_threshold(self):
        fig, tab = self._table(legend_label="FDR p < 0.05")
        assert [t.get_text() for t in tab.texts] == ["1", "2", "3"]
        assert [t.get_text() for t in tab.get_legend().get_texts()] == ["FDR p < 0.05", "n.s."]
        plt.close(fig)

    def test_numbers_can_be_turned_off(self):
        fig, tab = self._table(window_numbers=False)
        assert len(tab.texts) == 0
        plt.close(fig)

    def test_axes_is_stripped_of_ticks_and_spines(self):
        fig, tab = self._table()
        assert list(tab.get_yticks()) == []
        assert not any(sp.get_visible() for sp in tab.spines.values())
        plt.close(fig)


class TestDrawStatStrip:
    STATS = pd.DataFrame({"tr_start": [0, 3, 6], "tr_end": [2, 5, 8], "p_fdr": [0.01, 0.5, 0.04]})

    def _strip(self, stats=None, **kw):
        fig, (top, strip) = plt.subplots(2, 1, sharex=True)
        top.plot([0, 9], [0, 1])
        draw_stat_strip(strip, self.STATS if stats is None else stats, alpha=0.05, p_col="p_fdr", tr_seconds=2.0, **kw)
        return fig, strip

    def test_white_star_in_each_significant_cell_only(self):
        fig, strip = self._strip()
        stars = [t for t in strip.texts if t.get_text() == "*"]
        assert len(stars) == 2 and all(t.get_color() == "white" for t in stars)
        assert sorted(t.get_position()[0] for t in stars) == [3.0, 15.0]     # centers of bins 0 and 2 at 2 s/TR
        plt.close(fig)

    def test_significant_cells_are_dark_and_others_light(self):
        fig, strip = self._strip()
        by_x = {p.get_x(): p.get_facecolor()[0] for p in strip.patches}
        assert by_x[0.0] < 0.5 and by_x[6.0] > 0.85 and by_x[12.0] < 0.5
        plt.close(fig)

    def test_framed_with_time_axis_labels_kept_and_no_legend(self):
        fig, strip = self._strip()
        assert all(sp.get_visible() for sp in strip.spines.values())
        assert strip.xaxis.get_tick_params()["labelbottom"] is True or strip.xaxis.get_ticklabels()
        assert strip.get_legend() is None and len(strip.texts) == 2
        plt.close(fig)

    def test_label_is_written_to_the_left(self):
        from matplotlib.offsetbox import AnchoredOffsetbox
        fig, strip = self._strip(label="Face: pos vs. neg")
        boxes = [a for a in strip.artists if isinstance(a, AnchoredOffsetbox)]
        assert [t.get_text() for t in boxes[0].get_child().get_children()] == ["Face: pos vs. neg"]
        plt.close(fig)

    def test_empty_stats_draw_an_empty_strip_without_error(self):
        fig, strip = self._strip(stats=self.STATS.iloc[0:0])
        assert len(strip.patches) == 0 and len(strip.texts) == 0
        plt.close(fig)

    def test_default_table_is_unchanged_by_the_new_options(self):
        fig, (top, tab) = plt.subplots(2, 1, sharex=True)
        draw_significance_table(tab, [{"parts": [("A", "blue")], "stats": self.STATS}], p_col="p_fdr", tr_seconds=2.0)
        assert not any(sp.get_visible() for sp in tab.spines.values())
        assert not [t for t in tab.texts if t.get_text() == "*"]
        assert tab.get_legend() is not None
        plt.close(fig)


class TestSignificanceTableDirection:
    # bins: 0 significant ABOVE zero, 1 significant BELOW zero, 2 not significant, 3 significant BELOW zero
    STATS = pd.DataFrame({"tr_start": [0, 3, 6, 9], "tr_end": [2, 5, 8, 11], "p_fdr": [0.01, 0.02, 0.5, 0.03],
                          "mean_diff": [0.05, -0.04, -0.02, -0.06]})

    def _dark_cells(self, direction):
        fig, (top, tab) = plt.subplots(2, 1, sharex=True)
        row = {"parts": [("X", "black")], "stats": self.STATS}
        if direction:
            row["direction"] = direction
        draw_significance_table(tab, [row], p_col="p_fdr", tr_seconds=1.0, stars=True)
        dark = sorted(p.get_x() for p in tab.patches if p.get_facecolor()[0] < 0.7)
        stars = sorted(t.get_position()[0] for t in tab.texts if t.get_text() == "*")
        plt.close(fig)
        return dark, stars

    def test_no_direction_counts_either_sign(self):
        assert self._dark_cells(None)[0] == [0.0, 3.0, 9.0]

    def test_negative_ignores_a_window_significant_above_zero(self):
        dark, stars = self._dark_cells("negative")
        assert dark == [3.0, 9.0] and stars == [4.5, 10.5]

    def test_positive_counts_only_windows_above_zero(self):
        assert self._dark_cells("positive")[0] == [0.0]

    def test_rows_with_and_without_direction_can_share_a_table(self):
        fig, (top, tab) = plt.subplots(2, 1, sharex=True)
        rows = [{"parts": [("below", "black")], "stats": self.STATS, "direction": "negative"},
                {"parts": [("either", "black")], "stats": self.STATS}]
        draw_significance_table(tab, rows, p_col="p_fdr", tr_seconds=1.0)
        top_row = sorted(p.get_x() for p in tab.patches if p.get_y() == 1 and p.get_facecolor()[0] < 0.7)
        bottom_row = sorted(p.get_x() for p in tab.patches if p.get_y() == 0 and p.get_facecolor()[0] < 0.7)
        assert top_row == [3.0, 9.0] and bottom_row == [0.0, 3.0, 9.0]
        plt.close(fig)


class TestBaselineCorrection:
    @staticmethod
    def _data():
        rows = []
        for subject, offset in (("s1", 0.0), ("s2", 1.0)):
            for valence, shift in (("pos", 0.0), ("neg", 0.5)):
                for trial in range(4):
                    for w in range(6):
                        # a shared shape plus subject / valence offsets and trial noise that averages out
                        noise = (-1) ** trial * 0.1
                        rows.append({"subject": subject, "valence": valence, "trial_index": trial, "window_index": w,
                                     "value": offset + shift + 0.1 * w + noise})
        return pd.DataFrame(rows)

    def test_each_groups_mean_baseline_becomes_zero(self):
        out, dropped = baseline_correct_timecourses(self._data(), ["subject", "valence"], (0, 1, 2))
        assert dropped == 0
        tc = out.groupby(["subject", "valence", "window_index"])["value"].mean().reset_index()
        first3 = tc[tc.window_index < 3].groupby(["subject", "valence"])["value"].mean()
        assert np.allclose(first3, 0.0)

    def test_subtracts_one_constant_per_group_so_shape_is_preserved(self):
        data = self._data()
        out, _ = baseline_correct_timecourses(data, ["subject", "valence"], (0, 1, 2))
        diff = (data["value"] - out["value"]).groupby([data["subject"], data["valence"]]).agg(["min", "max"])
        assert np.allclose(diff["min"], diff["max"])           # the same constant for every row of a group
        # s1/pos baseline = mean(0, 0.1, 0.2) = 0.1 (trial noise cancels in the trial average)
        assert np.isclose(diff.loc[("s1", "pos"), "min"], 0.1)

    def test_baseline_is_the_mean_of_the_trial_averaged_time_course(self):
        # unequal trial counts per window: baseline must be the mean of the per-window MEANS, not of all rows
        df = pd.DataFrame({"subject": "s", "window_index": [0, 0, 0, 1, 2], "value": [1.0, 1.0, 1.0, 4.0, 7.0]})
        out, _ = baseline_correct_timecourses(df, ["subject"], (0, 1, 2))
        assert np.allclose(out["value"], df["value"] - 4.0)    # mean(1, 4, 7) = 4, not mean of all rows (2.8)

    def test_groups_without_any_baseline_data_are_dropped_and_counted(self):
        df = pd.DataFrame({"subject": ["a", "a", "b", "b"], "window_index": [0, 5, 4, 5], "value": [1.0, 2.0, 3.0, 4.0]})
        out, dropped = baseline_correct_timecourses(df, ["subject"], (0, 1, 2))
        assert dropped == 2 and set(out["subject"]) == {"a"}

    def test_input_is_not_modified_and_index_is_kept(self):
        data = self._data(); before = data.copy()
        out, _ = baseline_correct_timecourses(data, ["subject", "valence"], (0, 1, 2))
        pd.testing.assert_frame_equal(data, before)
        assert list(out.index) == list(data.index)

    def test_condition_differences_in_the_baseline_are_removed_per_group(self):
        data = self._data()
        raw = data.groupby(["valence", "window_index"])["value"].mean().unstack(0)
        out, _ = baseline_correct_timecourses(data, ["subject", "valence"], (0, 1, 2))
        cor = out.groupby(["valence", "window_index"])["value"].mean().unstack(0)
        assert abs((raw["pos"] - raw["neg"]).iloc[:3].mean()) > 0.4          # raw pos/neg differ by the 0.5 offset
        assert abs((cor["pos"] - cor["neg"]).iloc[:3].mean()) < 1e-9         # ...and not after correction

    def test_baseline_only_bins(self):
        assert baseline_only_bins((0, 1, 2), 3) == {0}
        assert baseline_only_bins((0, 1, 2), 1) == {0, 1, 2}
        assert baseline_only_bins((0, 1, 2), 2) == {0}      # bin 1 covers TRs 2-3, only half inside the baseline
        assert baseline_only_bins((0, 1, 2), 4) == set()    # no bin is wholly inside
        assert baseline_only_bins((3, 4, 5), 3) == {1}
        assert baseline_only_bins((6, 7, 8), 3) == {2}       # the first three operation-phase TRs are exactly one stat bin
        assert baseline_only_bins((7, 8, 9), 3) == set()     # a shifted baseline straddles bins, so none is wholly inside

    def test_baseline_only_sliding_windows_are_numbered_by_their_first_tr(self):
        assert baseline_only_bins((0, 1, 2), 3, sliding=True) == {0}
        assert baseline_only_bins((0, 1, 2, 3), 3, sliding=True) == {0, 1}
        assert baseline_only_bins((0, 1, 2), 1, sliding=True) == {0, 1, 2}
        assert baseline_only_bins((0, 1, 2), 4, sliding=True) == set()


# ---------------------------------------------------------------- balancing on the preceding trial

def _ordered_trial_table(sequences):
    """sequences: {(subject, boldfile, valence): [trial_type, ...]} -> a build_trial_table-shaped frame."""
    rows = []
    for (subject, boldfile, valence), seq in sequences.items():
        for i, tt in enumerate(seq, start=1):
            prev = "<start>" if i == 1 else seq[i - 2]
            rows.append(dict(subject=subject, boldfile=boldfile, trial_index=i, valence=valence, trial_type=tt,
                             operation=tt.split("_")[0] if tt else np.nan, category=tt.split("_")[1] if tt else np.nan,
                             previous_trial=prev))
    return pd.DataFrame(rows)


class TestBalancePrecedingCategory:
    # acquisition: clear_face, clear_face, clear_face, clear_place, clear_face
    #   predecessors: start (diff), face (same), face (same), face (diff), place (diff)
    #   -> clear_face: same = 2, different = 2 (start, place) ; clear_place: same 0, different 1
    SEQ = ["clear_face", "clear_face", "clear_face", "clear_place", "clear_face"]

    def test_each_cell_keeps_equal_numbers_of_same_and_different(self):
        kept, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ * 2}))
        cell = report.set_index(["subject", "operation", "category", "valence"])
        r = cell.loc[("1", "clear", "face", "pos")]
        assert r["n_same_category"] == 5 and r["n_different_or_no_category"] == 3 and r["n_kept_per_group"] == 3
        assert kept["predecessor"].value_counts().to_dict() == {"same": 3, "different": 3}      # clear_place has no same-category trial: nothing kept

    def test_first_trial_of_an_acquisition_counts_as_different(self):
        _, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): ["clear_face", "clear_face"]}))
        r = report.iloc[0]
        assert r["n_same_category"] == 1 and r["n_different_or_no_category"] == 1 and r["n_kept_per_group"] == 1

    def test_a_cell_missing_one_group_keeps_nothing(self):
        kept, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ}))
        r = report.set_index(["operation", "category"]).loc[("clear", "place")]
        assert r["n_same_category"] == 0 and r["n_kept_per_group"] == 0
        assert not ((kept["trial_index"] == 4)).any()

    def test_balancing_is_per_subject(self):
        kept, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ, ("2", "b", "pos"): ["clear_face"] * 5}))
        per = report[(report["category"] == "face")].set_index("subject")
        assert per.loc["1", "n_kept_per_group"] == 2           # same 2, different 2
        assert per.loc["2", "n_kept_per_group"] == 1           # same 4, different 1 (the first trial)
        assert set(kept["subject"]) == {"1", "2"}

    def test_kept_trials_are_a_subset_and_seed_is_reproducible(self):
        trials = _ordered_trial_table({("1", "a", "pos"): self.SEQ * 4})
        k1, _ = balance_trial_table(trials, seed=3)
        k2, _ = balance_trial_table(trials, seed=3)
        assert k1.equals(k2)
        keys = set(map(tuple, trials[["subject", "boldfile", "trial_index"]].to_numpy()))
        assert set(map(tuple, k1[["subject", "boldfile", "trial_index"]].to_numpy())) <= keys
        k3, _ = balance_trial_table(trials, seed=4)
        assert len(k3) == len(k1)

    def test_within_can_pool_categories(self):
        _, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ}), within=("operation", "valence"))
        r = report.iloc[0]
        assert r["n_same_category"] == 2 and r["n_different_or_no_category"] == 3 and r["n_kept_per_group"] == 2

    def test_trial_after_a_typeless_trial_is_left_out_and_counted(self):
        trials = _ordered_trial_table({("1", "a", "pos"): ["clear_face", "clear_face", "clear_face"]})
        trials.loc[1, ["trial_type", "operation", "category"]] = np.nan        # the middle trial has no type
        trials.loc[2, "previous_trial"] = np.nan                                  # so its successor's predecessor is unknown
        kept, report = balance_trial_table(trials)
        assert report.attrs["n_trials_with_unknown_predecessor"] == 1
        assert not (kept["trial_index"] == 3).any()

    def test_balance_preceding_category_works_from_a_raw_decoding_table(self):
        rows = []
        vol = 0
        for i, tt in enumerate(["clear_face", "clear_face", "clear_place"], start=1):
            for w, label in enumerate([tt.replace("clear", "view")] * 2 + [tt] * 2 + ["trial_fixation"] * 2):
                rows.append(dict(subject="1", session="A1", task="WMpos", run=1, boldfile="a", trial_index=i, window_index=w,
                                 volume_of_interest=vol, trial_type=label))
                vol += 1
        kept, report = balance_preceding_category(pd.DataFrame(rows))
        assert set(report.columns) >= {"subject", "operation", "category", "valence", "n_kept_per_group"}
        assert len(kept) == 2 and set(kept["predecessor"]) == {"same", "different"}


class TestWeightedBalancing:
    SEQ = ["clear_face", "clear_face", "clear_face", "clear_place", "clear_face"]   # clear_face: same 2, different 2

    def test_weighted_mode_keeps_every_trial_of_a_balanceable_cell_with_inclusion_probability_weights(self):
        seq = self.SEQ * 2                                       # clear_face: same 5, different 3 -> k = 3
        kept, report = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): seq}), mode="weighted")
        faces = kept[kept["trial_index"].isin([i for i, tt in enumerate(seq, start=1) if tt == "clear_face"])]
        assert len(kept) == len(faces) == 8                      # all 8 face trials; clear_place has no same-category trial: dropped
        w = faces.groupby("predecessor")["weight"].agg(["count", "sum", "min", "max"])
        assert w.loc["same", "count"] == 5 and w.loc["same", "sum"] == pytest.approx(3.0) and w.loc["same", "min"] == pytest.approx(3 / 5)
        assert w.loc["different", "sum"] == pytest.approx(3.0) and w.loc["different", "max"] == pytest.approx(1.0)
        r = report.set_index("category").loc["face"]
        assert r["inclusion_probability_same_category"] == pytest.approx(0.6) and r["inclusion_probability_different_or_no_category"] == 1.0

    def test_weighted_mode_does_not_depend_on_the_seed(self):
        trials = _ordered_trial_table({("1", "a", "pos"): self.SEQ * 3})
        k1, _ = balance_trial_table(trials, seed=1, mode="weighted")
        k2, _ = balance_trial_table(trials, seed=2, mode="weighted")
        assert k1.equals(k2)

    def test_subsample_mode_weights_are_all_one(self):
        kept, _ = balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ * 2}))
        assert (kept["weight"] == 1.0).all()

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode"):
            balance_trial_table(_ordered_trial_table({("1", "a", "pos"): self.SEQ}), mode="bootstrap")

    def test_weighted_mean_is_the_average_over_every_possible_subsample(self):
        """The point of the weighted mode: weights k/n make the weighted mean equal the mean of the
        subsample mean over all equally likely draws."""
        from itertools import combinations
        rng = np.random.default_rng(0)
        n_same, n_diff, k = 5, 3, 3
        same_vals, diff_vals = rng.normal(size=n_same), rng.normal(size=n_diff)
        draws = [np.mean(np.concatenate([same_vals[list(c)], diff_vals])) for c in combinations(range(n_same), k)]
        df = pd.DataFrame({"g": "x", "value": np.concatenate([same_vals, diff_vals]),
                           "weight": [k / n_same] * n_same + [k / n_diff] * n_diff})
        assert weighted_group_mean(df, "g", "value", "weight").loc["x"] == pytest.approx(np.mean(draws))


class TestWeightedMeans:
    def test_no_weight_column_is_the_plain_groupby_mean(self):
        df = pd.DataFrame({"g": list("aabb"), "value": [1.0, 3.0, 5.0, np.nan]})
        assert weighted_group_mean(df, "g").to_dict() == df.groupby("g")["value"].mean().to_dict()

    def test_weighted_mean_and_missing_values(self):
        df = pd.DataFrame({"g": list("aaab"), "value": [1.0, 3.0, np.nan, 2.0], "w": [1.0, 3.0, 10.0, 0.5]})
        out = weighted_group_mean(df, ["g"], "value", "w")
        assert out["a"] == pytest.approx((1 * 1 + 3 * 3) / 4)       # the NaN row counts for neither sum
        assert out["b"] == pytest.approx(2.0) and out.name == "value"

    def test_aggregate_by_subject_window_weights(self):
        df = pd.DataFrame({"subject": "1", "window_index": 0, "cond": ["a", "a", "b"], "value": [0.0, 1.0, 5.0], "w": [1.0, 3.0, 1.0]})
        wide = aggregate_by_subject_window(df, condition_col="cond", weight_col="w")
        assert wide.loc[("1", 0), "a"] == pytest.approx(0.75) and wide.loc[("1", 0), "b"] == 5.0
        assert aggregate_by_subject_window(df, condition_col="cond").loc[("1", 0), "a"] == pytest.approx(0.5)

    def test_baseline_correction_uses_the_weighted_time_course(self):
        rows = []
        for trial, (w, v) in enumerate([(1.0, [0.0, 0.0, 10.0]), (3.0, [4.0, 4.0, 4.0])]):
            for t, val in enumerate(v):
                rows.append(dict(subject="1", cond="x", trial=trial, window_index=t, value=val, w=w))
        df = pd.DataFrame(rows)
        out, dropped = baseline_correct_timecourses(df, ["subject", "cond"], (0, 1), weight_col="w")
        assert dropped == 0
        base = (0 * 1 + 4 * 3) / 4                                    # weighted mean at windows 0 and 1
        assert out.loc[(out["trial"] == 0) & (out["window_index"] == 2), "value"].iloc[0] == pytest.approx(10 - base)
        plain, _ = baseline_correct_timecourses(df, ["subject", "cond"], (0, 1))
        assert plain.loc[(plain["trial"] == 0) & (plain["window_index"] == 2), "value"].iloc[0] == pytest.approx(10 - 2.0)


class TestWeightedSplitHalf:
    def _long(self, n_subjects=6, n_trials=8, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_subjects):
            for cond, shift in (("a", 0.5), ("b", 0.0)):
                for tr in range(n_trials):
                    weight = rng.uniform(0.2, 1.0)
                    for w in range(6):
                        rows.append(dict(subject=str(s), cond=cond, boldfile="f", trial_index=(cond == "b") * 100 + tr,
                                         window_index=w, value=shift + rng.normal(), weight=weight))
        return pd.DataFrame(rows)

    def test_unit_weights_reproduce_the_unweighted_result(self):
        df = self._long().assign(weight=1.0)
        a = split_half_group_difference(df, "cond", "a", "b", bin_size=3, n_splits=50, seed=1)
        b = split_half_group_difference(df, "cond", "a", "b", bin_size=3, n_splits=50, seed=1, weight_col="weight")
        pd.testing.assert_frame_equal(a, b)

    def test_full_mean_diff_is_the_weighted_statistics_pipeline_number(self):
        df = self._long()
        sh = split_half_group_difference(df, "cond", "a", "b", bin_size=3, n_splits=20, seed=0, weight_col="weight")
        wide = aggregate_by_subject_window(df, condition_col="cond", weight_col="weight")
        per_bin, _ = bin_by_size(wide, bin_size=3, tr_offset=0)
        stats_table = compare_conditions_by_bin(per_bin, "a", "b", method="ttest_1samp_diff", alternative="two-sided", fdr=None)
        merged = sh.merge(stats_table, on="bin")
        assert np.allclose(merged["full_mean_diff"], merged["mean_diff"])

    def test_weights_change_the_answer(self):
        df = self._long()
        plain = split_half_group_difference(df, "cond", "a", "b", bin_size=3, n_splits=20, seed=0)
        weighted = split_half_group_difference(df, "cond", "a", "b", bin_size=3, n_splits=20, seed=0, weight_col="weight")
        assert not np.allclose(plain["full_mean_diff"], weighted["full_mean_diff"])


class TestBaselineGroupsAndDescription:
    def _frame(self):
        rows = []
        for v in ("pos", "neg"):
            for obj in ("face", "place"):
                for op in ("maintain", "clear"):
                    for w in range(3):
                        rows.append(dict(subject="1", valence=v, object=obj, operation=op, window_index=w, value=float(w) + (v == "pos")))
        return pd.DataFrame(rows)

    def test_names_of_valence_by_category_groups(self):
        assert describe_baseline_groups(self._frame(), ("object", "valence")) == [
            "positive faces", "positive places", "negative faces", "negative places"]

    def test_names_with_the_operation_included(self):
        names = describe_baseline_groups(self._frame(), ("object", "operation", "valence"))
        assert len(names) == 8 and names[0] == "positive clear faces" and names[-1] == "negative maintain places"

    def test_coarser_groups_pool_operations_into_one_baseline(self):
        df = self._frame().assign(value=lambda d: d["value"] + np.where(d["operation"] == "clear", 10.0, 0.0))
        out, dropped = baseline_correct_timecourses(df, ["subject", "object", "valence"], (0,))
        assert dropped == 0
        pos_face = out[(out["valence"] == "pos") & (out["object"] == "face")]
        # the baseline is the mean over BOTH operations at window 0 ((1 + 11) / 2 = 6), subtracted from each
        assert pos_face.loc[(pos_face["operation"] == "maintain") & (pos_face["window_index"] == 0), "value"].iloc[0] == pytest.approx(1 - 6)
        assert pos_face.loc[(pos_face["operation"] == "clear") & (pos_face["window_index"] == 0), "value"].iloc[0] == pytest.approx(11 - 6)
        fine, _ = baseline_correct_timecourses(df, ["subject", "object", "operation", "valence"], (0,))
        assert (fine.loc[fine["window_index"] == 0, "value"] == 0).all()           # per operation, window 0 is zero by construction


class TestBaselineGroupNames:
    def test_group_names_dict_keys_are_the_groupby_keys(self):
        rows = [dict(valence=v, object=o, operation=op) for v in ("pos", "neg") for o in ("face", "place") for op in ("maintain", "clear")]
        names = baseline_group_names(pd.DataFrame(rows), ("object", "valence"))
        assert names[("face", "pos")] == "positive faces" and names[("place", "neg")] == "negative places"
        assert list(names.values()) == ["positive faces", "positive places", "negative faces", "negative places"]


class TestFdrFromBin:
    def _per_subject_bin(self):
        rng = np.random.default_rng(0)
        rows = []
        for subject in range(12):
            for bin_id, shift in enumerate([1.0, 1.0, 0.4, 0.0, 0.0]):
                rows.append({"subject": str(subject), "bin": bin_id, "a": shift + rng.normal(0, 0.5), "b": rng.normal(0, 0.5)})
        return pd.DataFrame(rows).set_index(["subject", "bin"])

    def test_default_corrects_every_bin(self):
        out = compare_conditions_by_bin(self._per_subject_bin(), "a", "b", alternative="two-sided")
        assert out["p_fdr"].notna().all()

    def test_bins_before_the_start_are_tested_but_not_in_the_family(self):
        data = self._per_subject_bin()
        out = compare_conditions_by_bin(data, "a", "b", alternative="two-sided", fdr_from_bin=2)
        assert out.loc[out["bin"] < 2, "p_fdr"].isna().all() and out.loc[out["bin"] < 2, "p_value"].notna().all()
        assert out.loc[out["bin"] >= 2, "p_fdr"].notna().all()

    def test_the_family_is_only_the_later_bins(self):
        data = self._per_subject_bin()
        out = compare_conditions_by_bin(data, "a", "b", alternative="two-sided", fdr_from_bin=2)
        later = out[out["bin"] >= 2]
        expected = fdr_correct(later["p_value"].to_numpy(), "bh")
        assert np.allclose(later["p_fdr"].to_numpy(), expected)
        # not the same as correcting everything and keeping the later rows: the early (strong) bins no longer help the later ones
        full = compare_conditions_by_bin(data, "a", "b", alternative="two-sided")
        assert not np.allclose(later["p_fdr"].to_numpy(), full.loc[full["bin"] >= 2, "p_fdr"].to_numpy())
