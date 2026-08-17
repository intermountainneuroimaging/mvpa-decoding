"""mvpa_kfold_workflow.py: fold-strategy resolution/validation, and a small
synthetic end-to-end smoke test of run_kfold. Synthetic data only -- no
dependency on real BOLD data (a fake masker stands in for NiftiMasker)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflows.mvpa_kfold_workflow import (
    validate_kfold_cv_config,
    resolve_kfold_folds,
    run_kfold,
)

CLASSIFIER_NAME = "sklearn.linear_model.LogisticRegression"
CLASSIFIER_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}


# =====================================================
# validate_kfold_cv_config
# =====================================================

class TestValidateKfoldCvConfig:
    def test_none_raises(self):
        with pytest.raises(SystemExit):
            validate_kfold_cv_config(None)

    def test_bad_strategy_raises(self):
        with pytest.raises(SystemExit):
            validate_kfold_cv_config({"strategy": "bogus"})

    def test_group_kfold_missing_n_splits_raises(self):
        with pytest.raises(SystemExit):
            validate_kfold_cv_config({"strategy": "group_kfold"})

    def test_explicit_groups_missing_held_out_runs_raises(self):
        with pytest.raises(SystemExit):
            validate_kfold_cv_config({"strategy": "explicit_groups"})

    def test_valid_per_run_passes(self):
        validate_kfold_cv_config({"strategy": "per_run"})  # no exception

    def test_valid_group_kfold_passes(self):
        validate_kfold_cv_config({"strategy": "group_kfold", "n_splits": 3})  # no exception

    def test_valid_explicit_groups_passes(self):
        validate_kfold_cv_config({"strategy": "explicit_groups", "held_out_runs": [[1], [2]]})  # no exception


# =====================================================
# resolve_kfold_folds
# =====================================================

def _df_with_runs(runs):
    return pd.DataFrame({"run": runs})


class TestResolveKfoldFolds:
    def test_per_run_is_leave_one_run_out(self):
        testing_df = _df_with_runs([1, 1, 2, 3])
        timecourse_instr = _df_with_runs([1, 2, 3])
        folds = resolve_kfold_folds({"strategy": "per_run"}, testing_df, timecourse_instr)
        assert folds == [[1], [2], [3]]

    def test_group_kfold_splits_runs_into_n_contiguous_groups(self):
        testing_df = _df_with_runs([1, 2, 3, 4])
        timecourse_instr = _df_with_runs([])
        folds = resolve_kfold_folds({"strategy": "group_kfold", "n_splits": 2}, testing_df, timecourse_instr)
        assert len(folds) == 2
        assert sorted(r for g in folds for r in g) == [1, 2, 3, 4]

    def test_group_kfold_n_splits_exceeding_runs_raises(self):
        testing_df = _df_with_runs([1, 2])
        timecourse_instr = _df_with_runs([])
        with pytest.raises(SystemExit):
            resolve_kfold_folds({"strategy": "group_kfold", "n_splits": 5}, testing_df, timecourse_instr)

    def test_explicit_groups_returned_as_is(self):
        testing_df = _df_with_runs([1, 2, 3, 4])
        timecourse_instr = _df_with_runs([])
        held_out_runs = [[1, 2], [3, 4]]
        folds = resolve_kfold_folds(
            {"strategy": "explicit_groups", "held_out_runs": held_out_runs}, testing_df, timecourse_instr
        )
        assert folds == held_out_runs

    def test_explicit_groups_uncovered_run_warns(self, capsys):
        testing_df = _df_with_runs([1, 2, 3])
        timecourse_instr = _df_with_runs([])
        resolve_kfold_folds({"strategy": "explicit_groups", "held_out_runs": [[1, 2]]}, testing_df, timecourse_instr)
        assert "doesn't cover" in capsys.readouterr().out

    def test_explicit_groups_unknown_run_warns(self, capsys):
        testing_df = _df_with_runs([1, 2])
        timecourse_instr = _df_with_runs([])
        resolve_kfold_folds({"strategy": "explicit_groups", "held_out_runs": [[1, 2, 99]]}, testing_df, timecourse_instr)
        assert "references run(s)" in capsys.readouterr().out

    def test_no_runs_at_all_raises(self):
        testing_df = _df_with_runs([])
        timecourse_instr = _df_with_runs([])
        with pytest.raises(SystemExit):
            resolve_kfold_folds({"strategy": "per_run"}, testing_df, timecourse_instr)


# =====================================================
# run_kfold (synthetic end-to-end smoke test)
# =====================================================

class FakeImg:
    def to_filename(self, path):
        Path(path).touch()


class FakeMasker:
    def inverse_transform(self, data):
        return FakeImg()


def _build_fold_data(runs=(1, 2, 3), n_per_run_per_class=5, n_features=10, seed=0):
    """Synthetic training/testing/timecourse data spread across runs, with the
    first 3 features carrying a real class-mean shift (separable enough for
    ANOVA + LogisticRegression) -- 2 classes, coded 1/2 like apply_regressor_codes
    would produce."""
    rng = np.random.default_rng(seed)

    def _block(run):
        X_parts, y_parts, run_parts = [], [], []
        for cls in (1, 2):
            block = rng.normal(loc=0.0, scale=1.0, size=(n_per_run_per_class, n_features))
            block[:, :3] += cls * 4.0
            X_parts.append(block)
            y_parts.append(np.full(n_per_run_per_class, cls))
            run_parts.append(np.full(n_per_run_per_class, run))
        return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(run_parts)

    train_X, train_y, train_run = [], [], []
    test_X, test_y, test_run = [], [], []
    tc_X, tc_y, tc_run = [], [], []
    for run in runs:
        X, y, r = _block(run)
        train_X.append(X); train_y.append(y); train_run.append(r)
        X, y, r = _block(run + 100)  # different seed offset per call via rng state, still same run id
        test_X.append(X); test_y.append(y); test_run.append(np.full(len(y), run))
        X, y, r = _block(run + 200)
        tc_X.append(X); tc_y.append(y); tc_run.append(np.full(len(y), run))

    training_data = np.vstack(train_X)
    training_labels = np.concatenate(train_y)
    training_df = pd.DataFrame({"run": np.concatenate(train_run)})

    testing_data = np.vstack(test_X)
    testing_labels = np.concatenate(test_y)
    testing_df = pd.DataFrame({"run": np.concatenate(test_run)})

    timecourse_data = np.vstack(tc_X)
    timecourse_labels = np.concatenate(tc_y)
    n_tc = len(timecourse_labels)
    timecourse_instr = pd.DataFrame({
        "run": np.concatenate(tc_run),
        "subject": ["01"] * n_tc,
        "window_index": list(range(n_per_run_per_class * 2)) * len(runs),
        "regressor_label": ["face" if c == 1 else "place" for c in timecourse_labels],
    })

    return (training_df, training_data, training_labels,
            testing_df, testing_data, testing_labels,
            timecourse_instr, timecourse_data, timecourse_labels)


class TestRunKfold:
    def test_per_run_produces_fold_and_aggregated_outputs(self, tmp_path):
        (training_df, training_data, training_labels,
         testing_df, testing_data, testing_labels,
         timecourse_instr, timecourse_data, timecourse_labels) = _build_fold_data()

        aggregated_impa, xout, raw_decoding, summary_decoding = run_kfold(
            kfold_cv_cfg={"strategy": "per_run"},
            permutation_test_cfg=None,
            masker=FakeMasker(),
            analysis_output_dir=str(tmp_path), model_descr="test_model", subject_id="01",
            regressor_categories=["face", "place"],
            feat_p=0.05, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
            training_df=training_df, training_data=training_data, training_labels=training_labels,
            testing_df=testing_df, testing_data=testing_data, testing_labels=testing_labels,
            timecourse_instr=timecourse_instr, timecourse_data=timecourse_data, timecourse_labels=timecourse_labels,
        )

        base = tmp_path / "test_model" / "01"

        # manifest logs all 3 folds
        manifest = json.loads((base / "model" / "01_kfold_folds.json").read_text())
        assert sorted(int(k) for k in manifest.keys()) == [1, 2, 3]
        assert manifest["1"] == [1]

        # per-fold outputs exist for every fold
        for fold_id in (1, 2, 3):
            assert (base / "model" / f"01_fold{fold_id}_model_results_total_scores.csv").exists()
            assert (base / "model" / f"01_fold{fold_id}_impa_native.nii.gz").exists()
            assert (base / "decoding" / f"01_fold{fold_id}_decoding_results.csv").exists()
            assert (base / "decoding" / f"01_fold{fold_id}_summary_decoding_results.csv").exists()

        # aggregated results
        assert xout["total_scores"] > 0.7  # cleanly separable synthetic data
        assert aggregated_impa.shape == (2, training_data.shape[1])
        assert len(raw_decoding) == len(timecourse_labels)  # every timecourse row covered exactly once
        assert "fold" in raw_decoding.columns

    def test_with_permutation_test_writes_per_fold_file(self, tmp_path):
        (training_df, training_data, training_labels,
         testing_df, testing_data, testing_labels,
         timecourse_instr, timecourse_data, timecourse_labels) = _build_fold_data(
            n_per_run_per_class=8, n_features=30, seed=7,
        )

        run_kfold(
            kfold_cv_cfg={"strategy": "per_run"},
            permutation_test_cfg={"n_permutations": 3, "random_state": 0},
            masker=FakeMasker(),
            analysis_output_dir=str(tmp_path), model_descr="test_model", subject_id="01",
            regressor_categories=["face", "place"],
            feat_p=0.5, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
            training_df=training_df, training_data=training_data, training_labels=training_labels,
            testing_df=testing_df, testing_data=testing_data, testing_labels=testing_labels,
            timecourse_instr=timecourse_instr, timecourse_data=timecourse_data, timecourse_labels=timecourse_labels,
        )

        base = tmp_path / "test_model" / "01"
        fold1_permutation = pd.read_csv(base / "model" / "01_fold1_permutation_test.csv")
        assert sorted(fold1_permutation["metric"].tolist()) == ["accuracy", "roc_auc_ovr"]
        assert (fold1_permutation["n_permutations"] == 3).all()
