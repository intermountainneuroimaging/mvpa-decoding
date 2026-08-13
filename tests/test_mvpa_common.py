"""mvpa_common.py: shared, pure utilities -- the highest-value test target
since these were already parameterized/global-free before this session's
consolidation pass, and are used by every other script in the repo."""

import math

import numpy as np
import pandas as pd
import pytest

from mvpa_common import (
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
)


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
