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
    select_evidence_value,
    apply_filters,
    derive_label,
    aggregate_by_subject_window,
    average_across_groups,
    max_reliable_window_index,
    subtract_baseline,
    bin_by_size,
    bin_by_edges,
    ttest_1samp_diff,
    ttest_rel,
    wilcoxon_signed_rank,
    compare_conditions_by_bin,
    plot_conditions,
    annotate_significance,
    mark_significance_onset,
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
