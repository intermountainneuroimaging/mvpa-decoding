"""mvpa_workflow.py: the now-parameterized classification/decoding functions.
Synthetic data only -- small, deterministic, separable-enough for the
classifier steps to behave predictably without depending on real BOLD data."""

import numpy as np
import pandas as pd
import pytest

from mvpa_workflow import (
    apply_regressor_codes,
    balance,
    decision_evidence,
    save_model_results,
    average_fold_results,
    resolve_feature_threshold,
    build_classifier_pipeline,
    model_classification,
    model_performance,
    timecourse_decoding,
    summarize_decoding,
    permutation_significance,
)

CLASSIFIER_NAME = "sklearn.linear_model.LogisticRegression"
CLASSIFIER_PARAMS = {"max_iter": 1000, "class_weight": "balanced"}


def _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=0):
    """Deterministic synthetic X/y where the first 3 features carry a real
    class-mean shift and the rest are pure noise -- enough signal for ANOVA
    feature selection + LogisticRegression to behave predictably."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for cls in range(1, n_classes + 1):
        block = rng.normal(loc=0.0, scale=1.0, size=(n_per_class, n_features))
        block[:, :3] += cls * 4.0  # informative features
        X_parts.append(block)
        y_parts.append(np.full(n_per_class, cls))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


# =====================================================
# apply_regressor_codes
# =====================================================

class TestApplyRegressorCodes:
    def test_codes_are_1_indexed_in_category_order(self):
        df = pd.DataFrame({"regressor_label": ["place", "face", "place"]})
        out = apply_regressor_codes(df, ["face", "place"])
        assert out["regressor"].tolist() == [2, 1, 2]


# =====================================================
# balance
# =====================================================

class TestBalance:
    def test_downsamples_to_smallest_regressor_group(self):
        df = pd.DataFrame({
            "regressor": [1, 1, 1, 2, 2],
            "run": [1, 1, 2, 1, 2],
            "trial_index": [1, 2, 3, 1, 2],
            "volume_of_interest": [0, 1, 2, 0, 1],
        })
        out = balance(df)
        counts = out["regressor"].value_counts()
        assert counts[1] == 2
        assert counts[2] == 2


# =====================================================
# decision_evidence
# =====================================================

class FakeBinaryClf:
    classes_ = np.array([1, 2])

    def decision_function(self, X):
        return np.array([2.0, -2.0, 0.0])


class FakeMulticlassClf:
    classes_ = np.array([1, 2, 3])

    def decision_function(self, X):
        return np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])


class TestDecisionEvidence:
    def test_binary_uses_sigmoid_and_sums_to_one(self):
        evi = decision_evidence(FakeBinaryClf(), rawdata=None)
        assert evi.shape == (3, 2)
        np.testing.assert_allclose(evi.sum(axis=1), 1.0)
        assert evi[0, 1] > 0.5  # strongly positive decision -> class-1 (index 1) favored

    def test_multiclass_uses_softmax_and_sums_to_one(self):
        evi = decision_evidence(FakeMulticlassClf(), rawdata=None)
        assert evi.shape == (2, 3)
        np.testing.assert_allclose(evi.sum(axis=1), 1.0)
        assert evi[0].argmax() == 0
        assert evi[1].argmax() == 2

    def test_uses_predict_proba_when_available(self):
        class FakeProbaClf:
            def predict_proba(self, X):
                return np.array([[0.1, 0.9]])
        evi = decision_evidence(FakeProbaClf(), rawdata=None)
        np.testing.assert_array_equal(evi, [[0.1, 0.9]])


# =====================================================
# save_model_results
# =====================================================

class TestSaveModelResults:
    def test_square_matrix_saved_with_row_col_labels(self, tmp_path):
        categories = ["face", "place"]
        results = {"accuracy": np.array([[0.9, 0.1], [0.2, 0.8]])}
        pattern = str(tmp_path / "{metric}.csv")
        save_model_results(pattern, results, categories)
        df = pd.read_csv(tmp_path / "accuracy.csv", index_col=0)
        assert list(df.index) == categories
        assert list(df.columns) == categories

    def test_column_vector_saved_one_value_per_category(self, tmp_path):
        categories = ["face", "place"]
        results = {"auc": np.array([0.7, 0.8])}
        pattern = str(tmp_path / "{metric}.csv")
        save_model_results(pattern, results, categories)
        df = pd.read_csv(tmp_path / "auc.csv", index_col=0)
        assert df["auc"].tolist() == [0.7, 0.8]
        assert list(df.index) == categories

    def test_other_shape_saved_unlabeled(self, tmp_path):
        categories = ["face", "place"]
        results = {"total_scores": 0.75}  # scalar -> not (C,C) or (C,) -- unlabeled path
        pattern = str(tmp_path / "{metric}.csv")
        save_model_results(pattern, results, categories)
        loaded = np.loadtxt(tmp_path / "total_scores.csv", delimiter=",")
        assert float(loaded) == pytest.approx(0.75)


# =====================================================
# average_fold_results
# =====================================================

class TestAverageFoldResults:
    def test_averages_scalars_and_arrays_elementwise(self):
        fold_results = [
            {"total_scores": 0.6, "accuracy": np.array([[1.0, 0.0], [0.0, 1.0]])},
            {"total_scores": 0.8, "accuracy": np.array([[0.0, 1.0], [1.0, 0.0]])},
        ]
        avg = average_fold_results(fold_results)
        assert avg["total_scores"] == pytest.approx(0.7)
        np.testing.assert_allclose(avg["accuracy"], [[0.5, 0.5], [0.5, 0.5]])


# =====================================================
# resolve_feature_threshold / build_classifier_pipeline
# =====================================================

class TestResolveFeatureThreshold:
    def test_widens_threshold_until_min_voxels_selected(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        thr = resolve_feature_threshold(X, y, feat_p=1e-6)
        # starting threshold is far too strict for any real data -- must have widened
        assert thr > 1e-6

    def test_keeps_feat_p_when_already_enough_voxels(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        thr = resolve_feature_threshold(X, y, feat_p=0.9)
        assert thr == pytest.approx(0.9)


class TestBuildClassifierPipeline:
    def test_pipeline_has_feature_selection_then_classifier_steps(self):
        pipe = build_classifier_pipeline(0.1, CLASSIFIER_NAME, CLASSIFIER_PARAMS)
        assert list(pipe.named_steps.keys()) == ["feature_selection", "classifier"]
        assert type(pipe.named_steps["classifier"]).__name__ == "LogisticRegression"


# =====================================================
# model_classification / model_performance (end-to-end, tiny synthetic data)
# =====================================================

class TestModelClassificationAndPerformance:
    def test_binary_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=1)
        pipe = model_classification(X, y, feat_p=0.05, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, impa_full = model_performance(pipe, X, y)

        assert xout["total_scores"] > 0.8  # cleanly separable data
        assert xout["accuracy"].shape == (2, 2)
        assert xout["evidence"].shape == (2, 2)
        assert xout["auc"].shape == (2,)
        assert impa_full.shape == (2, X.shape[1])

    def test_multiclass_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=3, seed=2)
        pipe = model_classification(X, y, feat_p=0.05, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, impa_full = model_performance(pipe, X, y)

        assert xout["total_scores"] > 0.7
        assert xout["accuracy"].shape == (3, 3)
        assert xout["evidence"].shape == (3, 3)
        assert xout["auc"].shape == (3,)
        assert impa_full.shape == (3, X.shape[1])


# =====================================================
# timecourse_decoding / summarize_decoding
# =====================================================

class TestTimecourseDecoding:
    def _fitted_pipe_and_categories(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=3)
        pipe = model_classification(X, y, feat_p=0.05, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        return pipe

    def test_raw_and_summary_shapes(self):
        pipe = self._fitted_pipe_and_categories()
        X, y = _separable_data(n_per_class=4, n_features=10, n_classes=2, seed=4)
        categories = ["face", "place"]

        timecourse_df = pd.DataFrame({
            "subject": ["01"] * len(y),
            "window_index": [0, 1, 2, 3] * 2,
            "regressor_label": [categories[c - 1] for c in y],
        })

        raw, summary = timecourse_decoding(
            pipe, X, y, timecourse_df, categories, feat_p=0.05, subject_id="01", model_descr="test_model",
        )

        assert len(raw) == len(y)
        assert "predicted_label" in raw.columns
        assert "correct" in raw.columns
        assert "evidence_face" in raw.columns and "evidence_place" in raw.columns
        assert raw["model_descr"].unique().tolist() == ["test_model"]

        assert set(summary.columns) >= {"subject", "model_descr", "window_index", "regressor_label", "trial_count", "Accuracy"}
        # one summary row per (window_index, regressor_label) actually present
        assert len(summary) == raw.groupby(["window_index", "regressor_label"]).ngroups


class TestSummarizeDecoding:
    def test_averages_correct_and_evidence_within_group(self):
        raw = pd.DataFrame({
            "window_index": [0, 0, 1, 1],
            "regressor_label": ["face", "face", "face", "face"],
            "correct": [True, False, True, True],
            "evidence_face": [0.9, 0.5, 0.8, 1.0],
            "evidence_place": [0.1, 0.5, 0.2, 0.0],
            "threshold_p": [0.05, 0.05, 0.05, 0.05],
            "selected_voxels": [5, 5, 5, 5],
            "whole_voxels": [10, 10, 10, 10],
            "feature_percent": [50.0, 50.0, 50.0, 50.0],
        })
        summary = summarize_decoding(raw, ["face", "place"], subject_id="01", model_descr="test_model")

        assert len(summary) == 2  # (0, face) and (1, face)
        row0 = summary[summary["window_index"] == 0].iloc[0]
        assert row0["Accuracy"] == pytest.approx(0.5)
        assert row0["evidence_face"] == pytest.approx(0.7)


# =====================================================
# permutation_significance (fast smoke test, n_permutations=5)
# =====================================================

class TestPermutationSignificance:
    def test_returns_one_row_per_metric_with_plausible_values(self):
        # A generous feature count and a loose starting feat_p keep the fixed
        # selection threshold (widened once from the real training data) from
        # occasionally selecting 0 voxels on a shuffled-label permutation round
        # -- a real possibility with a tiny synthetic feature count.
        X_train, y_train = _separable_data(n_per_class=8, n_features=30, n_classes=2, seed=5)
        X_test, y_test = _separable_data(n_per_class=6, n_features=30, n_classes=2, seed=6)

        result = permutation_significance(
            X_train, y_train, X_test, y_test,
            n_permutations=5, random_state=0,
            feat_p=0.5, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
        )

        assert sorted(result["metric"].tolist()) == ["accuracy", "roc_auc_ovr"]
        assert result["p_value"].between(0.0, 1.0).all()
        assert result["real_score"].between(0.0, 1.0).all()
        assert (result["n_permutations"] == 5).all()
