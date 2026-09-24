"""analysis/plot_valence_evidence_by_operation.py: the thin, project-specific
wiring (valence/stimulus/baseline-operation choices) on top of
analysis/decoding_results_toolkit.py's generic building blocks -- the
generic mechanics themselves are covered by tests/test_decoding_results_toolkit.py."""

import pandas as pd
import pytest

from analysis.decoding_results_toolkit import select_evidence_value, derive_label
from analysis.plot_valence_evidence_by_operation import (
    per_operation_means,
    compute_bin_stats,
    VALENCE_MAPPING,
    STIMULUS_MAPPING,
)


def _prepared_df(rows):
    """rows -> the same derived-column shape main() builds before calling
    per_operation_means (select_evidence_value + derive_label x2)."""
    raw = pd.DataFrame(rows)
    df = select_evidence_value(raw, value="self")
    df = derive_label(df, "task", VALENCE_MAPPING, new_col="valence")
    df = derive_label(df, "trial_type", STIMULUS_MAPPING, new_col="stimulus", regex=True)
    return df


def _row(subject, window_index, regressor_label, trial_type, task, **evidence):
    row = {"subject": subject, "window_index": window_index,
           "regressor_label": regressor_label, "trial_type": trial_type, "task": task}
    row.update({f"evidence_{k}": v for k, v in evidence.items()})
    return row


class TestPerOperationMeans:
    def test_selects_only_the_given_operation(self):
        df = _prepared_df([
            _row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.9, suppress=0.1),
            _row("01", 0, "suppress", "WM_suppress_face", "WMpos", maintain=0.1, suppress=0.9),
        ])
        means = per_operation_means(df, "suppress")
        assert means["face"].loc[("01", 0), "pos"] == pytest.approx(0.9)

    def test_collapsed_averages_face_and_place(self):
        df = _prepared_df([
            _row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.9, suppress=0.1),
            _row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.7, suppress=0.3),
            _row("01", 0, "maintain", "WM_maintain_place", "WMpos", maintain=0.6, suppress=0.4),
        ])
        means = per_operation_means(df, "maintain")
        assert means["face"].loc[("01", 0), "pos"] == pytest.approx(0.8)
        assert means["place"].loc[("01", 0), "pos"] == pytest.approx(0.6)
        assert means["collapsed"].loc[("01", 0), "pos"] == pytest.approx(0.7)  # mean of the two stimulus means

    def test_both_valences_present_as_separate_columns(self):
        df = _prepared_df([
            _row("01", 0, "maintain", "WM_maintain_face", "WMpos", maintain=0.8, suppress=0.2),
            _row("01", 0, "maintain", "WM_maintain_face", "WMneg", maintain=0.6, suppress=0.4),
        ])
        means = per_operation_means(df, "maintain")
        assert set(means["face"].columns) == {"pos", "neg"}


class TestComputeBinStats:
    def test_returns_pos_vs_neg_bin_rows(self):
        import numpy as np
        rng = np.random.default_rng(0)
        idx = pd.MultiIndex.from_product([[f"{s:02d}" for s in range(6)], range(3)], names=["subject", "window_index"])
        wide = pd.DataFrame({
            "pos": 0.6 + rng.normal(0, 0.01, 18),
            "neg": 0.5 + rng.normal(0, 0.01, 18),
        }, index=idx)
        result = compute_bin_stats(wide, bin_size=3, method="ttest_1samp_diff", alternative="greater")
        assert len(result) == 1
        assert result.iloc[0]["p_value"] < 0.01
        assert result.iloc[0]["tr_start"] == 1
        assert result.iloc[0]["tr_end"] == 3
