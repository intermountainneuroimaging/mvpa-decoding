"""mvpa_common.py: shared, pure utilities -- the highest-value test target
since these were already parameterized/global-free before this session's
consolidation pass, and are used by every other script in the repo."""

import math

import numpy as np
import pandas as pd
import pytest

from utils.mvpa_common import (
    parse_bids_entities,
    resolve_config_root,
    compute_volume_range,
    build_trial_pivot_table,
    validate_query_node,
    evaluate_query_node,
    validate_window_bound,
    validate_window,
    resolve_window_times,
    quick_safe,
    label_rows,
    apply_regressor_codes,
    balance,
    decision_evidence,
    save_model_results,
    average_fold_results,
    resolve_feature_selection_params,
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
# compute_volume_range
# =====================================================

class TestComputeVolumeRange:
    def test_exact_tr_boundary(self):
        # start=0, stop=2*tr -> exactly 2 volumes, no rounding needed
        start, stop = compute_volume_range(0.0, 2.0, tr=1.0, n_frames=10)
        assert (start, stop) == (0, 2)

    def test_sub_tr_duration_keeps_at_least_one_volume(self):
        # 0.2s duration at TR=1.0 rounds to 0 volumes without the max(1, ...) floor
        start, stop = compute_volume_range(0.5, 0.7, tr=1.0, n_frames=10)
        assert stop - start == 1

    def test_real_off_by_one_regression_case(self):
        # The exact numbers from the bug this was fixed for this session: a
        # 2.774s trial at TR=0.46 is 6 volumes, not 7 (floor(start)+ceil(stop)
        # used to inflate by a full volume whenever onset isn't TR-aligned).
        onset, hemodynamic_lag, duration, tr = 18.507, 4.6, 2.774, 0.46
        start_time = onset + hemodynamic_lag
        stop_time = start_time + duration
        start, stop = compute_volume_range(start_time, stop_time, tr, n_frames=1000)
        assert stop - start == 6

    def test_clips_to_n_frames(self):
        start, stop = compute_volume_range(9.0, 12.0, tr=1.0, n_frames=10)
        assert stop == 10
        assert start == 9

    def test_never_returns_negative_range(self):
        # start_vol beyond n_frames should still yield a valid (possibly empty) range
        start, stop = compute_volume_range(20.0, 22.0, tr=1.0, n_frames=10)
        assert stop >= start


# =====================================================
# resolve_config_root
# =====================================================

class TestResolveConfigRoot:
    def test_missing_key_inherits_default(self):
        assert resolve_config_root({}, "derivatives_root", "/default", "label") == "/default"

    def test_explicit_null_inherits_default(self):
        assert resolve_config_root({"derivatives_root": None}, "derivatives_root", "/default", "label") == "/default"

    def test_explicit_empty_string_is_literal_not_default(self, capsys):
        result = resolve_config_root({"derivatives_root": ""}, "derivatives_root", "/default", "label")
        assert result == ""
        assert "label" in capsys.readouterr().out

    def test_present_value_is_used(self):
        assert resolve_config_root({"derivatives_root": "/custom"}, "derivatives_root", "/default", "label") == "/custom"


# =====================================================
# parse_bids_entities
# =====================================================

class TestParseBidsEntities:
    def test_standard_entities(self):
        entities = parse_bids_entities("sub-01_ses-A1_task-loc_run-01_events.tsv")
        assert entities == {"sub": "01", "ses": "A1", "task": "loc", "run": "01"}

    def test_missing_session_is_absent_not_error(self):
        entities = parse_bids_entities("sub-1_task-objectviewing_run-01_events.tsv")
        assert "ses" not in entities
        assert entities["sub"] == "1"

    def test_arbitrary_extra_entity(self):
        entities = parse_bids_entities("sub-01_ses-A1_task-loc_dir-pa_run-01_bold.nii.gz")
        assert entities["dir"] == "pa"


# =====================================================
# Query DSL: validate_query_node / evaluate_query_node
# =====================================================

class TestQueryDSL:
    def test_valid_leaf_exact(self):
        assert validate_query_node({"column": "trial_type", "match": "exact", "value": "face"}, None) == []

    def test_invalid_match_type(self):
        errors = validate_query_node({"column": "trial_type", "match": "bogus", "value": "face"}, None)
        assert len(errors) == 1

    def test_unknown_column_flagged_when_valid_columns_given(self):
        errors = validate_query_node({"column": "nope", "match": "exact", "value": "x"}, {"trial_type"})
        assert len(errors) == 1

    def test_mixing_bool_and_leaf_is_invalid(self):
        errors = validate_query_node({"and": [], "column": "trial_type"}, None)
        assert len(errors) == 1

    def test_regex_requires_valid_pattern(self):
        errors = validate_query_node({"column": "trial_type", "match": "regex", "value": "["}, None)
        assert len(errors) == 1

    def test_evaluate_exact(self):
        df = pd.DataFrame({"trial_type": ["face", "place", "face"]})
        mask = evaluate_query_node({"column": "trial_type", "match": "exact", "value": "face"}, df)
        assert mask.tolist() == [True, False, True]

    def test_evaluate_in(self):
        df = pd.DataFrame({"trial_type": ["face", "place", "house"]})
        mask = evaluate_query_node({"column": "trial_type", "match": "in", "values": ["face", "house"]}, df)
        assert mask.tolist() == [True, False, True]

    def test_evaluate_regex_fullmatch_not_partial(self):
        df = pd.DataFrame({"trial_type": ["view_face", "face"]})
        mask = evaluate_query_node({"column": "trial_type", "match": "regex", "value": "face"}, df)
        # fullmatch: only the exact "face" row matches, not "view_face"
        assert mask.tolist() == [False, True]

    def test_evaluate_and(self):
        df = pd.DataFrame({"task": ["loc", "loc", "WM"], "trial_type": ["face", "place", "face"]})
        query = {"and": [
            {"column": "task", "match": "exact", "value": "loc"},
            {"column": "trial_type", "match": "exact", "value": "face"},
        ]}
        assert evaluate_query_node(query, df).tolist() == [True, False, False]

    def test_evaluate_or(self):
        df = pd.DataFrame({"trial_type": ["face", "place", "house"]})
        query = {"or": [
            {"column": "trial_type", "match": "exact", "value": "face"},
            {"column": "trial_type", "match": "exact", "value": "house"},
        ]}
        assert evaluate_query_node(query, df).tolist() == [True, False, True]

    def test_evaluate_not(self):
        df = pd.DataFrame({"trial_type": ["face", "place"]})
        query = {"not": {"column": "trial_type", "match": "exact", "value": "face"}}
        assert evaluate_query_node(query, df).tolist() == [False, True]


# =====================================================
# timecourse_decoding window: validate_window / resolve_window_times
# =====================================================

class TestWindow:
    def test_valid_window(self):
        window = {
            "start": {"reference": "onset", "offset_seconds": 0},
            "end": {"reference": "offset_end", "offset_seconds": 10},
        }
        assert validate_window(window) == []

    def test_invalid_reference(self):
        errors = validate_window_bound({"reference": "bogus", "offset_seconds": 0}, "path")
        assert len(errors) == 1

    def test_end_before_start_same_reference_is_invalid(self):
        window = {
            "start": {"reference": "onset", "offset_seconds": 5},
            "end": {"reference": "onset", "offset_seconds": 2},
        }
        errors = validate_window(window)
        assert len(errors) == 1

    def test_resolve_window_times_onset_reference(self):
        window = {
            "start": {"reference": "onset", "offset_seconds": 0},
            "end": {"reference": "offset_end", "offset_seconds": 10},
        }
        start, stop = resolve_window_times(window, onset=5.0, duration=2.0)
        assert start == 5.0
        assert stop == 17.0  # onset + duration + 10

    def test_resolve_window_times_negative_offset(self):
        window = {
            "start": {"reference": "onset", "offset_seconds": -2},
            "end": {"reference": "onset", "offset_seconds": 0},
        }
        start, stop = resolve_window_times(window, onset=5.0, duration=2.0)
        assert start == 3.0
        assert stop == 5.0


# =====================================================
# quick_safe
# =====================================================

class TestQuickSafe:
    def test_replaces_unsafe_characters(self):
        assert quick_safe("gm valence/classifier!") == "gm_valence_classifier_"

    def test_leaves_safe_characters_alone(self):
        assert quick_safe("gm_valence-classifier.v2") == "gm_valence-classifier.v2"


# =====================================================
# label_rows
# =====================================================

class TestLabelRows:
    def test_labels_matching_rows(self):
        df = pd.DataFrame({"trial_type": ["face", "place", "house"]})
        conditions = {
            "face": {"column": "trial_type", "match": "exact", "value": "face"},
            "place": {"column": "trial_type", "match": "exact", "value": "place"},
        }
        labeled = label_rows(df, conditions)
        assert sorted(labeled["regressor_label"].tolist()) == ["face", "place"]

    def test_unmatched_rows_dropped(self):
        df = pd.DataFrame({"trial_type": ["face", "house"]})
        conditions = {"face": {"column": "trial_type", "match": "exact", "value": "face"}}
        labeled = label_rows(df, conditions)
        assert len(labeled) == 1

    def test_first_matching_condition_wins(self):
        df = pd.DataFrame({"trial_type": ["face"]})
        conditions = {
            "first": {"column": "trial_type", "match": "exact", "value": "face"},
            "second": {"column": "trial_type", "match": "regex", "value": ".*"},
        }
        labeled = label_rows(df, conditions)
        assert labeled["regressor_label"].iloc[0] == "first"

    def test_custom_label_column(self):
        df = pd.DataFrame({"trial_type": ["maintain_face", "suppress_place"]})
        conditions = {
            "maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"},
            "suppress": {"column": "trial_type", "match": "regex", "value": ".*suppress.*"},
        }
        labeled = label_rows(df, conditions, label_column="overlay_label")
        assert "overlay_label" in labeled.columns
        assert "regressor_label" not in labeled.columns
        assert sorted(labeled["overlay_label"].tolist()) == ["maintain", "suppress"]


# =====================================================
# build_trial_pivot_table
# =====================================================

class TestBuildTrialPivotTable:
    def test_one_row_per_trial_nan_padded_to_widest(self):
        df = pd.DataFrame({
            "boldfile": ["a", "a", "a", "a"],
            "trial_index": [1, 1, 1, 2],
            "volume_of_interest": [10, 11, 12, 20],
            "trial_type": ["face", "face", "face", "place"],
        })
        pivot = build_trial_pivot_table(df)

        assert len(pivot) == 2  # one row per (boldfile, trial_index)
        assert "vol_of_interest_1" in pivot.columns
        assert "vol_of_interest_3" in pivot.columns  # widest trial has 3 volumes

        trial1 = pivot[pivot["trial_index"] == 1].iloc[0]
        assert trial1["vol_of_interest_1"] == 10
        assert trial1["vol_of_interest_3"] == 12

        trial2 = pivot[pivot["trial_index"] == 2].iloc[0]
        assert math.isnan(trial2["vol_of_interest_2"])  # NaN-padded, shorter trial


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
# resolve_feature_selection_params / build_classifier_pipeline
# =====================================================

class TestResolveFeatureSelectionParams:
    def test_widens_threshold_until_min_voxels_selected(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        mode, param = resolve_feature_selection_params(X, y, {"feat_p": 1e-6})
        # starting threshold is far too strict for any real data -- must have widened
        assert mode == "fpr"
        assert param > 1e-6

    def test_keeps_feat_p_when_already_enough_voxels(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        mode, param = resolve_feature_selection_params(X, y, {"feat_p": 0.9})
        assert mode == "fpr"
        assert param == pytest.approx(0.9)

    def test_n_voxels_returns_k_best_unchanged(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        # feat_p also present (as merge_with_defaults always injects it) --
        # n_voxels must win, no widening/threshold logic involved
        mode, param = resolve_feature_selection_params(X, y, {"feat_p": 0.05, "n_voxels": 3})
        assert (mode, param) == ("k_best", 3)


class TestBuildClassifierPipeline:
    def test_fpr_pipeline_has_feature_selection_then_classifier_steps(self):
        pipe = build_classifier_pipeline("fpr", 0.1, CLASSIFIER_NAME, CLASSIFIER_PARAMS)
        assert list(pipe.named_steps.keys()) == ["feature_selection", "classifier"]
        assert type(pipe.named_steps["classifier"]).__name__ == "LogisticRegression"

    def test_k_best_pipeline_selects_exact_voxel_count(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2)
        pipe = build_classifier_pipeline("k_best", 3, CLASSIFIER_NAME, CLASSIFIER_PARAMS)
        pipe.fit(X, y)
        assert int(pipe.named_steps["feature_selection"].get_support().sum()) == 3


# =====================================================
# model_classification / model_performance (end-to-end, tiny synthetic data)
# =====================================================

class TestModelClassificationAndPerformance:
    def test_binary_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=1)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, impa_full = model_performance(pipe, X, y)

        assert xout["total_scores"] > 0.8  # cleanly separable data
        assert xout["accuracy"].shape == (2, 2)
        assert xout["evidence"].shape == (2, 2)
        assert xout["auc"].shape == (2,)
        assert impa_full.shape == (2, X.shape[1])

    def test_multiclass_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=3, seed=2)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, impa_full = model_performance(pipe, X, y)

        assert xout["total_scores"] > 0.7
        assert xout["accuracy"].shape == (3, 3)
        assert xout["evidence"].shape == (3, 3)
        assert xout["auc"].shape == (3,)
        assert impa_full.shape == (3, X.shape[1])

    def test_n_voxels_selects_exact_count_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=1)
        pipe = model_classification(X, y, feature_selection_cfg={"n_voxels": 4}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        assert int(pipe.named_steps["feature_selection"].get_support().sum()) == 4
        xout, impa_full = model_performance(pipe, X, y)
        assert xout["total_scores"] > 0.8  # cleanly separable data


# =====================================================
# timecourse_decoding / summarize_decoding
# =====================================================

class TestTimecourseDecoding:
    def _fitted_pipe_and_categories(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=3)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
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
            pipe, X, y, timecourse_df, categories, feature_selection_cfg={"feat_p": 0.05}, subject_id="01", model_descr="test_model",
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
            feature_selection_cfg={"feat_p": 0.5}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS,
        )

        assert sorted(result["metric"].tolist()) == ["accuracy", "roc_auc_ovr"]
        assert result["p_value"].between(0.0, 1.0).all()
        assert result["real_score"].between(0.0, 1.0).all()
        assert (result["n_permutations"] == 5).all()
