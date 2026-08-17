"""mvpa_generalization_workflow.py: internal training-CV fold selection --
leave-one-run-out when every training run replicates the same conditions,
otherwise n_splits-fold stratified CV over trials pooled across all runs
(fold membership not scoped to any one run). Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from workflows.mvpa_generalization_workflow import (
    training_runs_have_matching_conditions,
    stratified_trial_kfold_split,
    resolve_internal_cv_folds,
)


def _trial_volumes(run, trial_index, regressor_label, n_volumes=3):
    return [
        {"run": run, "trial_index": trial_index, "regressor_label": regressor_label}
        for _ in range(n_volumes)
    ]


def _balanced_multi_run_df(n_runs=3, n_trials_per_condition_per_run=10, n_volumes=3):
    rows = []
    for run in range(1, n_runs + 1):
        for trial_index in range(1, n_trials_per_condition_per_run * 2 + 1):
            label = "face" if trial_index % 2 == 0 else "place"
            rows.extend(_trial_volumes(run, trial_index, label, n_volumes))
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
# stratified_trial_kfold_split
# =====================================================

class TestStratifiedTrialKfoldSplit:
    def test_returns_n_splits_folds(self):
        df = _balanced_multi_run_df()
        folds = stratified_trial_kfold_split(df, n_splits=4, random_state=0)
        assert len(folds) == 4

    def test_no_row_overlap_within_any_fold(self):
        df = _balanced_multi_run_df()
        for train_idx, test_idx in stratified_trial_kfold_split(df, n_splits=4, random_state=0):
            assert len(set(train_idx) & set(test_idx)) == 0
            assert len(train_idx) + len(test_idx) == len(df)

    def test_no_trial_split_across_train_and_test_within_any_fold(self):
        df = _balanced_multi_run_df()
        for train_idx, test_idx in stratified_trial_kfold_split(df, n_splits=4, random_state=0):
            train_trials = set(map(tuple, df.iloc[train_idx][["run", "trial_index"]].to_numpy()))
            test_trials = set(map(tuple, df.iloc[test_idx][["run", "trial_index"]].to_numpy()))
            assert train_trials & test_trials == set()

    def test_every_row_is_tested_exactly_once_across_all_folds(self):
        df = _balanced_multi_run_df()
        folds = stratified_trial_kfold_split(df, n_splits=4, random_state=0)
        test_counts = np.zeros(len(df), dtype=int)
        for _, test_idx in folds:
            test_counts[test_idx] += 1
        assert (test_counts == 1).all()

    def test_class_balance_approximately_preserved_in_each_fold(self):
        df = _balanced_multi_run_df()
        for _, test_idx in stratified_trial_kfold_split(df, n_splits=4, random_state=0):
            counts = df.iloc[test_idx]["regressor_label"].value_counts()
            assert counts["face"] == pytest.approx(counts["place"], abs=3)

    def test_deterministic_given_same_random_state(self):
        df = _balanced_multi_run_df()
        folds1 = stratified_trial_kfold_split(df, n_splits=4, random_state=0)
        folds2 = stratified_trial_kfold_split(df, n_splits=4, random_state=0)
        for (_, test1), (_, test2) in zip(folds1, folds2):
            np.testing.assert_array_equal(sorted(test1), sorted(test2))

    def test_reduces_n_splits_when_rarest_condition_has_fewer_trials(self, capsys):
        rows = []
        for run in (1, 2):
            for trial_index in range(1, 7):
                rows.extend(_trial_volumes(run, trial_index, "common"))
        rows.extend(_trial_volumes(1, 100, "rare"))
        rows.extend(_trial_volumes(2, 100, "rare"))
        df = pd.DataFrame(rows)  # "rare" has 2 trials total -- fewer than the default n_splits=4

        folds = stratified_trial_kfold_split(df, n_splits=4, random_state=0)
        assert len(folds) == 2
        assert "reducing internal-CV from 4 to 2 fold(s)" in capsys.readouterr().out

    def test_raises_when_rarest_condition_has_fewer_than_2_trials(self):
        rows = []
        for trial_index in range(1, 7):
            rows.extend(_trial_volumes(1, trial_index, "common"))
        rows.extend(_trial_volumes(1, 100, "rare"))  # only 1 trial
        df = pd.DataFrame(rows)

        with pytest.raises(SystemExit):
            stratified_trial_kfold_split(df, n_splits=4, random_state=0)


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

    def test_heterogeneous_runs_fall_back_to_4_fold_stratified_split(self, capsys):
        df = _balanced_multi_run_df()
        # make run 3 missing the "place" condition entirely
        df = df[~((df["run"] == 3) & (df["regressor_label"] == "place"))].reset_index(drop=True)
        folds = resolve_internal_cv_folds(df)
        assert len(folds) == 4
        assert "do not all contain the same condition" in capsys.readouterr().out

    def test_n_splits_is_configurable(self):
        df = _balanced_multi_run_df()
        df = df[~((df["run"] == 3) & (df["regressor_label"] == "place"))].reset_index(drop=True)
        folds = resolve_internal_cv_folds(df, n_splits=3)
        assert len(folds) == 3
