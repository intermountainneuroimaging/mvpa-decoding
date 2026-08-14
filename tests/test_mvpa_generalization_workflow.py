"""mvpa_generalization_workflow.py: internal training-CV fold selection --
leave-one-run-out when every training run replicates the same conditions,
otherwise a stratified trial-level holdout fallback. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from mvpa_generalization_workflow import (
    training_runs_have_matching_conditions,
    stratified_trial_holdout_split,
    resolve_internal_cv_folds,
)


def _trial_volumes(run, trial_index, regressor_label, regressor, n_volumes=3):
    return [
        {"run": run, "trial_index": trial_index, "regressor_label": regressor_label, "regressor": regressor}
        for _ in range(n_volumes)
    ]


def _balanced_multi_run_df(n_runs=3, n_trials_per_condition_per_run=10, n_volumes=3):
    rows = []
    for run in range(1, n_runs + 1):
        for trial_index in range(1, n_trials_per_condition_per_run * 2 + 1):
            label = "face" if trial_index % 2 == 0 else "place"
            code = 1 if label == "face" else 2
            rows.extend(_trial_volumes(run, trial_index, label, code, n_volumes))
    return pd.DataFrame(rows)


# =====================================================
# training_runs_have_matching_conditions
# =====================================================

class TestTrainingRunsHaveMatchingConditions:
    def test_true_when_every_run_has_same_conditions(self):
        df = pd.DataFrame({
            "run": [1, 1, 2, 2, 3, 3],
            "regressor_label": ["face", "place", "face", "place", "face", "place"],
        })
        assert training_runs_have_matching_conditions(df) is True

    def test_false_when_a_run_is_missing_a_condition(self):
        df = pd.DataFrame({
            "run": [1, 1, 2, 2, 3],
            "regressor_label": ["face", "place", "face", "place", "face"],  # run 3 has no "place"
        })
        assert training_runs_have_matching_conditions(df) is False

    def test_true_for_a_single_run(self):
        df = pd.DataFrame({"run": [1, 1], "regressor_label": ["face", "place"]})
        assert training_runs_have_matching_conditions(df) is True


# =====================================================
# stratified_trial_holdout_split
# =====================================================

class TestStratifiedTrialHoldoutSplit:
    def test_no_row_overlap_between_train_and_test(self):
        df = _balanced_multi_run_df()
        train_idx, test_idx = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == len(df)

    def test_no_trial_split_across_train_and_test(self):
        df = _balanced_multi_run_df()
        train_idx, test_idx = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        train_trials = set(map(tuple, df.iloc[train_idx][["run", "trial_index"]].to_numpy()))
        test_trials = set(map(tuple, df.iloc[test_idx][["run", "trial_index"]].to_numpy()))
        assert train_trials & test_trials == set()

    def test_test_fraction_is_approximately_25_percent(self):
        df = _balanced_multi_run_df()
        train_idx, test_idx = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        fraction = len(test_idx) / len(df)
        assert fraction == pytest.approx(0.25, abs=0.05)

    def test_class_balance_preserved_in_test_set(self):
        df = _balanced_multi_run_df()
        train_idx, test_idx = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        counts = df.iloc[test_idx]["regressor_label"].value_counts()
        assert counts["face"] == counts["place"]

    def test_deterministic_given_same_random_state(self):
        df = _balanced_multi_run_df()
        train_idx1, test_idx1 = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        train_idx2, test_idx2 = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        np.testing.assert_array_equal(sorted(test_idx1), sorted(test_idx2))

    def test_condition_with_fewer_than_2_trials_kept_entirely_in_train(self, capsys):
        rows = _trial_volumes(1, 1, "rare", 3, n_volumes=2)  # only 1 trial for "rare"
        rows += _balanced_multi_run_df(n_runs=1, n_trials_per_condition_per_run=5).to_dict("records")
        df = pd.DataFrame(rows)
        train_idx, test_idx = stratified_trial_holdout_split(df, test_size=0.25, random_state=0)
        rare_rows = df[df["regressor_label"] == "rare"]
        assert set(rare_rows.index) <= set(train_idx)
        assert "only 1 trial(s)" in capsys.readouterr().out


# =====================================================
# resolve_internal_cv_folds
# =====================================================

class TestResolveInternalCvFolds:
    def test_homogeneous_runs_use_leave_one_run_out(self):
        df = pd.DataFrame({
            "run": [1, 1, 2, 2, 3, 3],
            "trial_index": [1, 2, 1, 2, 1, 2],
            "regressor_label": ["face", "place", "face", "place", "face", "place"],
        })
        folds = resolve_internal_cv_folds(df)
        assert len(folds) == 3  # one fold per distinct run

    def test_heterogeneous_runs_fall_back_to_single_stratified_split(self, capsys):
        df = _balanced_multi_run_df()
        # make run 3 missing the "place" condition entirely
        df = df[~((df["run"] == 3) & (df["regressor_label"] == "place"))].reset_index(drop=True)
        folds = resolve_internal_cv_folds(df)
        assert len(folds) == 1
        assert "do not all contain the same condition" in capsys.readouterr().out
