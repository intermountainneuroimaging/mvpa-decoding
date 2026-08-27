"""mvpa_workflow.py: model.kfold_cv validation/fold resolution (now purely
within model_conditions.training -- no more testing_df/timecourse_instr
involvement), and a small synthetic end-to-end smoke test of run_kfold.
Synthetic data only -- no dependency on real BOLD data (a fake masker stands
in for NiftiMasker)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from workflows.mvpa_workflow import (
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
# resolve_kfold_folds -- purely a function of model_conditions.training now
# =====================================================

def _df_with_runs(runs):
    return pd.DataFrame({"run": runs})


class TestResolveKfoldFolds:
    def test_per_run_is_leave_one_run_out(self):
        training_df = _df_with_runs([1, 1, 2, 3])
        folds = resolve_kfold_folds({"strategy": "per_run"}, training_df)
        assert folds == [[1], [2], [3]]

    def test_group_kfold_splits_runs_into_n_contiguous_groups(self):
        training_df = _df_with_runs([1, 2, 3, 4])
        folds = resolve_kfold_folds({"strategy": "group_kfold", "n_splits": 2}, training_df)
        assert len(folds) == 2
        assert sorted(r for g in folds for r in g) == [1, 2, 3, 4]

    def test_group_kfold_n_splits_exceeding_runs_raises(self):
        training_df = _df_with_runs([1, 2])
        with pytest.raises(SystemExit):
            resolve_kfold_folds({"strategy": "group_kfold", "n_splits": 5}, training_df)

    def test_explicit_groups_returned_as_is(self):
        training_df = _df_with_runs([1, 2, 3, 4])
        held_out_runs = [[1, 2], [3, 4]]
        folds = resolve_kfold_folds({"strategy": "explicit_groups", "held_out_runs": held_out_runs}, training_df)
        assert folds == held_out_runs

    def test_explicit_groups_uncovered_run_warns(self, capsys):
        training_df = _df_with_runs([1, 2, 3])
        resolve_kfold_folds({"strategy": "explicit_groups", "held_out_runs": [[1, 2]]}, training_df)
        assert "doesn't cover" in capsys.readouterr().out

    def test_explicit_groups_unknown_run_warns(self, capsys):
        training_df = _df_with_runs([1, 2])
        resolve_kfold_folds({"strategy": "explicit_groups", "held_out_runs": [[1, 2, 99]]}, training_df)
        assert "references run(s)" in capsys.readouterr().out

    def test_no_runs_at_all_raises(self):
        training_df = _df_with_runs([])
        with pytest.raises(SystemExit):
            resolve_kfold_folds({"strategy": "per_run"}, training_df)


# =====================================================
# run_kfold (synthetic end-to-end smoke test, training-only)
# =====================================================

class FakeImg:
    def to_filename(self, path):
        Path(path).touch()


class FakeMasker:
    def inverse_transform(self, data):
        return FakeImg()


def _build_training_fold_data(runs=(1, 2, 3), n_per_run_per_class=5, n_features=10, seed=0):
    """Synthetic training data spread across runs, with the first 3 features
    carrying a real class-mean shift (separable enough for ANOVA +
    LogisticRegression) -- 2 classes, coded 1/2 like apply_regressor_codes
    would produce. Unlike the old mvpa_kfold_workflow.py fixture, there's no
    separate testing_df/timecourse data -- k-fold now holds out runs entirely
    within this one training set."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts, run_parts = [], [], []
    for run in runs:
        for cls in (1, 2):
            block = rng.normal(loc=0.0, scale=1.0, size=(n_per_run_per_class, n_features))
            block[:, :3] += cls * 4.0
            X_parts.append(block)
            y_parts.append(np.full(n_per_run_per_class, cls))
            run_parts.append(np.full(n_per_run_per_class, run))
    training_data = np.vstack(X_parts)
    training_labels = np.concatenate(y_parts)
    training_df = pd.DataFrame({"run": np.concatenate(run_parts)})
    return training_df, training_data, training_labels


class TestRunKfold:
    def test_per_run_produces_fold_and_aggregated_outputs(self, tmp_path):
        training_df, training_data, training_labels = _build_training_fold_data()
        fold_groups = resolve_kfold_folds({"strategy": "per_run"}, training_df)

        aggregated_impa, xout = run_kfold(
            kfold_cv_cfg={"strategy": "per_run"},
            fold_groups=fold_groups,
            permutation_test_cfg=None,
            masker=FakeMasker(),
            impa_filename_tag="impa",
            analysis_output_dir=str(tmp_path), model_descr="test_model", subject_id="01",
            regressor_categories=["face", "place"],
            feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
            training_df=training_df, training_data=training_data, training_labels=training_labels,
        )

        base = tmp_path / "test_model" / "01"

        # manifest logs all 3 folds
        manifest = json.loads((base / "model" / "01_kfold_folds.json").read_text())
        assert sorted(int(k) for k in manifest.keys()) == [1, 2, 3]
        assert manifest["1"] == [1]

        # per-fold outputs exist for every fold
        for fold_id in (1, 2, 3):
            assert (base / "model" / f"01_fold{fold_id}_model_results_total_scores.csv").exists()
            assert (base / "model" / f"01_fold{fold_id}_impa.nii.gz").exists()

        # aggregated results
        assert xout["total_scores"] > 0.7  # cleanly separable synthetic data
        assert aggregated_impa.shape == (2, training_data.shape[1])

    def test_with_permutation_test_writes_per_fold_file(self, tmp_path):
        training_df, training_data, training_labels = _build_training_fold_data(
            n_per_run_per_class=8, n_features=30, seed=7,
        )
        fold_groups = resolve_kfold_folds({"strategy": "per_run"}, training_df)

        run_kfold(
            kfold_cv_cfg={"strategy": "per_run"},
            fold_groups=fold_groups,
            permutation_test_cfg={"n_permutations": 3, "random_state": 0},
            masker=FakeMasker(),
            impa_filename_tag="impa",
            analysis_output_dir=str(tmp_path), model_descr="test_model", subject_id="01",
            regressor_categories=["face", "place"],
            feature_selection_cfg={"feat_p": 0.5}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
            training_df=training_df, training_data=training_data, training_labels=training_labels,
        )

        base = tmp_path / "test_model" / "01"
        fold1_permutation = pd.read_csv(base / "model" / "01_fold1_permutation_test.csv")
        assert sorted(fold1_permutation["metric"].tolist()) == ["accuracy", "roc_auc_ovr"]
        assert (fold1_permutation["n_permutations"] == 3).all()

    def test_all_folds_skipped_raises(self, tmp_path):
        # only 1 run -- per_run holds it out entirely, leaving no training rows
        training_df, training_data, training_labels = _build_training_fold_data(runs=(1,))
        fold_groups = resolve_kfold_folds({"strategy": "per_run"}, training_df)

        with pytest.raises(SystemExit):
            run_kfold(
                kfold_cv_cfg={"strategy": "per_run"},
                fold_groups=fold_groups,
                permutation_test_cfg=None,
                masker=FakeMasker(),
                impa_filename_tag="impa",
                analysis_output_dir=str(tmp_path), model_descr="test_model", subject_id="01",
                regressor_categories=["face", "place"],
                feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
                training_df=training_df, training_data=training_data, training_labels=training_labels,
            )
