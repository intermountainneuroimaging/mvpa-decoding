"""analysis/plot_valence_evidence_by_operation.py: the numerical core
(valence/stimulus derivation, self-evidence selection, baseline subtraction,
binned one-sample t-test) -- not the CLI or plotting, which are exercised via
real synthetic-data runs instead (see the session's own verification)."""

import numpy as np
import pandas as pd
import pytest

from analysis.plot_valence_evidence_by_operation import (
    add_derived_columns,
    per_subject_window_means,
    subtract_baseline,
    bin_and_test,
)


def _raw_row(subject, window_index, regressor_label, trial_type, task, **evidence):
    row = {
        "subject": subject, "window_index": window_index,
        "regressor_label": regressor_label, "trial_type": trial_type, "task": task,
    }
    row.update({f"evidence_{k}": v for k, v in evidence.items()})
    return row


class TestAddDerivedColumns:
    def test_valence_from_task(self):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.6, suppress=0.4),
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMneg", maintain=0.5, suppress=0.5),
        ])
        df = add_derived_columns(raw)
        assert df["valence"].tolist() == ["pos", "neg"]

    def test_unresolvable_valence_dropped(self, capsys):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "loc", maintain=0.6, suppress=0.4),
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.6, suppress=0.4),
        ])
        df = add_derived_columns(raw)
        assert len(df) == 1
        assert "dropped" in capsys.readouterr().out

    def test_stimulus_from_trial_type(self):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.6, suppress=0.4),
            _raw_row("01", 0, "maintain", "WM_maintain_place", "WMpos", maintain=0.6, suppress=0.4),
        ])
        df = add_derived_columns(raw)
        assert df["stimulus"].tolist() == ["face", "place"]

    def test_unresolvable_stimulus_dropped(self, capsys):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_object", "WMpos", maintain=0.6, suppress=0.4),
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.6, suppress=0.4),
        ])
        df = add_derived_columns(raw)
        assert len(df) == 1
        assert "dropped" in capsys.readouterr().out

    def test_self_evidence_picks_own_true_category(self):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.7, suppress=0.3),
            _raw_row("01", 0, "suppress", "WM_suppress_face", "WMpos", maintain=0.2, suppress=0.8),
        ])
        df = add_derived_columns(raw)
        assert df["self_evidence"].tolist() == pytest.approx([0.7, 0.8])

    def test_unknown_regressor_label_dropped(self, capsys):
        raw = pd.DataFrame([
            _raw_row("01", 0, "bogus_category", "WM_maintain_face", "WMpos", maintain=0.7, suppress=0.3),
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.7, suppress=0.3),
        ])
        df = add_derived_columns(raw)
        assert len(df) == 1
        assert "dropped" in capsys.readouterr().out


class TestPerSubjectWindowMeans:
    def test_collapses_pools_face_and_place_directly(self):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.8, suppress=0.2),
            _raw_row("01", 0, "maintain", "WM_maintain_place", "WMpos", maintain=0.6, suppress=0.4),
        ])
        df = add_derived_columns(raw)
        means = per_subject_window_means(df, "maintain")
        # collapsed pools both trials together: (0.8 + 0.6) / 2 = 0.7
        assert means["collapsed"].loc[("01", 0), "pos"] == pytest.approx(0.7)
        assert means["face"].loc[("01", 0), "pos"] == pytest.approx(0.8)
        assert means["place"].loc[("01", 0), "pos"] == pytest.approx(0.6)

    def test_only_selects_rows_for_the_given_operation(self):
        raw = pd.DataFrame([
            _raw_row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.9, suppress=0.1),
            _raw_row("01", 0, "suppress", "WM_suppress_face", "WMpos", maintain=0.1, suppress=0.9),
        ])
        df = add_derived_columns(raw)
        means = per_subject_window_means(df, "suppress")
        assert means["collapsed"].loc[("01", 0), "pos"] == pytest.approx(0.9)  # suppress's own self-evidence


class TestSubtractBaseline:
    def test_matched_subject_and_window_subtracted(self):
        op = pd.DataFrame({"pos": [0.8, 0.7], "neg": [0.6, 0.5]},
                           index=pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"]))
        base = pd.DataFrame({"pos": [0.5, 0.5], "neg": [0.4, 0.4]},
                             index=pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"]))
        result = subtract_baseline({"collapsed": op}, {"collapsed": base})
        assert result["collapsed"]["pos"].tolist() == pytest.approx([0.3, 0.2])
        assert result["collapsed"]["neg"].tolist() == pytest.approx([0.2, 0.1])

    def test_unmatched_rows_excluded_not_guessed(self):
        op = pd.DataFrame({"pos": [0.8, 0.7]},
                           index=pd.MultiIndex.from_tuples([("01", 0), ("01", 1)], names=["subject", "window_index"]))
        base = pd.DataFrame({"pos": [0.5]},
                             index=pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"]))
        result = subtract_baseline({"collapsed": op}, {"collapsed": base})
        assert len(result["collapsed"]) == 1  # window 1 dropped, no baseline for it


class TestBinAndTest:
    def test_detects_a_clear_positive_difference(self):
        # 6 subjects, pos consistently ~0.1 higher than neg across a 3-TR bin
        rng = np.random.default_rng(0)
        rows = []
        for subj in range(6):
            for w in range(3):
                rows.append({"subject": str(subj), "window_index": w,
                             "pos": 0.6 + rng.normal(0, 0.01), "neg": 0.5 + rng.normal(0, 0.01)})
        wide = pd.DataFrame(rows).set_index(["subject", "window_index"])
        result = bin_and_test(wide, bin_size=3, alternative="greater")
        assert len(result) == 1
        assert result.iloc[0]["n_subjects"] == 6
        assert result.iloc[0]["mean_diff_pos_minus_neg"] == pytest.approx(0.1, abs=0.02)
        assert result.iloc[0]["p_value"] < 0.001

    def test_null_difference_not_significant(self):
        # independent noise on pos/neg (not literally identical, which would
        # give every subject a difference of exactly 0 -- zero variance,
        # an undefined/NaN t-statistic rather than a real null result)
        rng = np.random.default_rng(1)
        rows = []
        for subj in range(8):
            for w in range(3):
                rows.append({"subject": str(subj), "window_index": w,
                             "pos": 0.5 + rng.normal(0, 0.02), "neg": 0.5 + rng.normal(0, 0.02)})
        wide = pd.DataFrame(rows).set_index(["subject", "window_index"])
        result = bin_and_test(wide, bin_size=3, alternative="greater")
        assert result.iloc[0]["p_value"] > 0.1

    def test_bin_boundaries_are_1_indexed_trs(self):
        rows = []
        for subj in range(4):
            for w in range(6):
                rows.append({"subject": str(subj), "window_index": w, "pos": 0.5, "neg": 0.5})
        wide = pd.DataFrame(rows).set_index(["subject", "window_index"])
        result = bin_and_test(wide, bin_size=3, alternative="greater")
        assert list(zip(result["tr_start"], result["tr_end"])) == [(1, 3), (4, 6)]

    def test_missing_valence_column_returns_empty(self):
        wide = pd.DataFrame({"pos": [0.5]}, index=pd.MultiIndex.from_tuples([("01", 0)], names=["subject", "window_index"]))
        result = bin_and_test(wide, bin_size=3, alternative="greater")
        assert result.empty
