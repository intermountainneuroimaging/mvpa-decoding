"""mvpa_common.py: shared, pure utilities -- the highest-value test target
since these were already parameterized/global-free before this session's
consolidation pass, and are used by every other script in the repo."""

import math

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from utils.mvpa_common import (
    parse_bids_entities,
    resolve_config_root,
    compute_volume_range,
    build_full_frame_table,
    partition_into_trials,
    build_timecourse_instructions,
    qualifying_boldfiles,
    label_rows_optional,
    label_conditions_with_lag,
    is_excluded_trial_type,
    build_trial_pivot_table,
    validate_query_node,
    evaluate_query_node,
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
    load_images_and_mask,
    impa_tag,
    extract_importance_map,
    build_cv_raw_results,
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
# impa_tag
# =====================================================

class TestImpaTag:
    def test_false_gives_plain_impa(self):
        assert impa_tag(False) == "impa"

    def test_true_gives_mni_suffixed_impa(self):
        assert impa_tag(True) == "impa_mni"


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

    def test_evaluate_regex_missing_value_never_matches(self):
        # a continuous-decoding gap/rest frame has no real trial_type at all
        # (None/NaN) -- must never match, not crash. Depending on the
        # column's pandas dtype, astype(str) can leave a real missing value
        # as an actual NaN rather than stringifying it to "nan", so this
        # specifically guards against passing that non-str value to
        # re.Pattern.fullmatch
        df = pd.DataFrame({"trial_type": ["face", None]})
        mask = evaluate_query_node({"column": "trial_type", "match": "regex", "value": ".*face.*"}, df)
        assert mask.tolist() == [True, False]

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
# Continuous timecourse decoding: build_full_frame_table / partition_into_trials
# / label_rows_optional / build_timecourse_instructions
# =====================================================

def _events(*rows):
    """rows: (onset, duration, trial_type) tuples -> a raw events dataframe."""
    return pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])


class TestBuildFullFrameTable:
    def test_continuous_coverage_no_exclusions(self):
        # a 5-TR view_face event (the exact shape from the "decode every frame"
        # example), then a 3-TR maintain, then a 2-TR fixation -- fixation
        # would be dropped by generate_master_spreadsheet.py's own exclusion
        # policy, but build_full_frame_table never excludes anything
        events = _events((0.0, 5.0, "view_face"), (5.0, 3.0, "maintain"), (8.0, 2.0, "fixation"))
        table = build_full_frame_table(events, tr=1.0, n_frames=10)

        assert table["volume_of_interest"].tolist() == list(range(10))
        assert table["trial_type"].tolist() == ["view_face"] * 5 + ["maintain"] * 3 + ["fixation"] * 2
        assert table["event_index"].tolist() == [1] * 5 + [2] * 3 + [3] * 2
        assert not table["trial_type"].isna().any()

    def test_gap_with_no_covering_event_is_nan(self):
        events = _events((0.0, 2.0, "view_face"))  # only covers vols 0-1 of 5
        table = build_full_frame_table(events, tr=1.0, n_frames=5)
        assert table.loc[:1, "trial_type"].tolist() == ["view_face", "view_face"]
        assert table.loc[2:, "trial_type"].isna().all()
        assert table.loc[2:, "event_index"].isna().all()

    def test_invalid_duration_row_produces_no_coverage(self):
        events = _events((0.0, float("nan"), "bad_row"), (0.0, 3.0, "view_face"))
        table = build_full_frame_table(events, tr=1.0, n_frames=3)
        # the NaN-duration row is simply never a candidate -- view_face covers everything
        assert table["trial_type"].tolist() == ["view_face"] * 3

    def test_overlap_later_onset_wins(self):
        events = _events((0.0, 3.0, "a"), (1.0, 3.0, "b"))  # a: vols 0-2, b: vols 1-3
        table = build_full_frame_table(events, tr=1.0, n_frames=4)
        assert table["trial_type"].tolist() == ["a", "b", "b", "b"]


def _full_frame_df(boldfile, *rows):
    """rows: (volume_of_interest, trial_type, onset, event_index) tuples."""
    df = pd.DataFrame(rows, columns=["volume_of_interest", "trial_type", "onset", "event_index"])
    df["boldfile"] = boldfile
    return df


VIEW_FACE_ANCHOR = {"column": "trial_type", "match": "exact", "value": "view_face"}


class TestPartitionIntoTrials:
    def test_window_index_resets_at_each_anchor(self):
        # two back-to-back trials in one run: view_face(5 TRs) -> maintain(3) ->
        # fixation(2), twice -- window_index must count 0..9 within each, not
        # keep climbing across the boundary
        rows = []
        for trial, base in enumerate((0, 10)):
            rows += [(base + i, "view_face", float(base), 3 * trial + 1) for i in range(5)]
            rows += [(base + 5 + i, "maintain", float(base + 5), 3 * trial + 2) for i in range(3)]
            rows += [(base + 8 + i, "fixation", float(base + 8), 3 * trial + 3) for i in range(2)]
        df = _full_frame_df("run1", *rows)

        result = partition_into_trials(df, VIEW_FACE_ANCHOR)

        first_trial = result[result["volume_of_interest"] < 10].sort_values("volume_of_interest")
        second_trial = result[result["volume_of_interest"] >= 10].sort_values("volume_of_interest")
        assert first_trial["trial_index"].unique().tolist() == [1]
        assert first_trial["window_index"].tolist() == list(range(10))
        assert second_trial["trial_index"].unique().tolist() == [2]
        assert second_trial["window_index"].tolist() == list(range(10))

    def test_leading_frames_before_first_anchor_get_trial_index_zero(self):
        rows = [(0, "instructions", 0.0, 1), (1, "instructions", 0.0, 1)]
        rows += [(2 + i, "view_face", 2.0, 2) for i in range(3)]
        df = _full_frame_df("run1", *rows)

        result = partition_into_trials(df, VIEW_FACE_ANCHOR).sort_values("volume_of_interest")
        leading = result[result["trial_type"] == "instructions"]
        assert leading["trial_index"].tolist() == [0, 0]
        assert leading["window_index"].tolist() == [0, 1]  # own volume_of_interest, no anchor yet
        assert result[result["trial_type"] == "view_face"]["trial_index"].unique().tolist() == [1]

    def test_no_anchor_found_leaves_everything_trial_index_zero(self):
        df = _full_frame_df("run1", (0, "maintain", 0.0, 1), (1, "maintain", 0.0, 1))
        result = partition_into_trials(df, VIEW_FACE_ANCHOR)
        assert (result["trial_index"] == 0).all()

    def test_block_of_repeated_same_type_events_collapses_to_one_trial(self):
        # a Haxby-style block design: the anchor matches EVERY stimulus
        # (any of several categories), and the same category repeats back to
        # back many times in a row with nothing else between them -- that
        # whole run must be ONE trial (block), not one trial per repeat
        anchor = {"column": "trial_type", "match": "in", "values": ["scissors", "face"]}
        rows = [(i, "scissors", float(i), i + 1) for i in range(6)]  # 6 separate "scissors" events
        rows += [(6 + i, "face", float(6 + i), 6 + i + 1) for i in range(4)]  # then 4 "face" events
        df = _full_frame_df("run1", *rows)

        result = partition_into_trials(df, anchor).sort_values("volume_of_interest")

        assert result["trial_index"].tolist() == [1] * 6 + [2] * 4
        assert result["window_index"].tolist() == list(range(6)) + list(range(4))

    def test_same_type_recurring_after_a_gap_starts_a_new_trial(self):
        # unlike the collapsed-run case above, the same trial_type occurring
        # again *after* a different, non-matching real event happened in
        # between (e.g. the next trial's own cue) must still start a fresh
        # trial -- this is the real Clearvale shape (view_face -> operation
        # -> probe -> view_face again for the next trial)
        rows = [(0, "view_face", 0.0, 1), (1, "maintain", 1.0, 2), (2, "view_face", 2.0, 3)]
        df = _full_frame_df("run1", *rows)

        result = partition_into_trials(df, VIEW_FACE_ANCHOR).sort_values("volume_of_interest")
        assert result["trial_index"].tolist() == [1, 1, 2]
        assert result["window_index"].tolist() == [0, 1, 0]

    def test_explicit_trial_start_tag_takes_priority_over_trial_start_event(self):
        # a "trial_start" tagged row exists -- it must be used as the anchor
        # directly, ignoring trial_start_event entirely (which here would
        # give a completely different, wrong answer if it were consulted)
        rows = [
            (0, "trial_start", 0.0, 1),
            (1, "view_face", 1.0, 2),  # would itself match VIEW_FACE_ANCHOR, but must NOT be used
            (2, "maintain", 2.0, 3),
        ]
        df = _full_frame_df("run1", *rows)
        result = partition_into_trials(df, VIEW_FACE_ANCHOR).sort_values("volume_of_interest")
        assert result["trial_index"].tolist() == [1, 1, 1]
        assert result["window_index"].tolist() == [0, 1, 2]

    def test_trial_start_event_fallback_used_when_no_explicit_tag_present(self):
        # no "trial_start" row anywhere in this boldfile -- falls back to
        # trial_start_event exactly as before
        rows = [(0, "view_face", 0.0, 1), (1, "maintain", 1.0, 2)]
        df = _full_frame_df("run1", *rows)
        result = partition_into_trials(df, VIEW_FACE_ANCHOR).sort_values("volume_of_interest")
        assert result["trial_index"].tolist() == [1, 1]

    def test_mixed_dataset_each_boldfile_resolved_independently(self):
        # one boldfile has explicit "trial_start" tags, the other doesn't --
        # each must be partitioned by its own appropriate mechanism
        tagged = _full_frame_df("run_tagged", (0, "trial_start", 0.0, 1), (1, "maintain", 1.0, 2))
        untagged = _full_frame_df("run_untagged", (0, "view_face", 0.0, 1), (1, "maintain", 1.0, 2))
        df = pd.concat([tagged, untagged], ignore_index=True)

        result = partition_into_trials(df, VIEW_FACE_ANCHOR)
        assert (result[result["boldfile"] == "run_tagged"]["trial_index"] == 1).all()
        assert (result[result["boldfile"] == "run_untagged"]["trial_index"] == 1).all()


class TestLabelRowsOptional:
    def test_unmatched_rows_kept_with_none_label(self):
        df = pd.DataFrame({"trial_type": ["maintain", "fixation", "suppress"]})
        conditions = {
            "maintain": {"column": "trial_type", "match": "exact", "value": "maintain"},
            "suppress": {"column": "trial_type", "match": "exact", "value": "suppress"},
        }
        result = label_rows_optional(df, conditions)
        assert len(result) == 3  # nothing dropped, unlike label_rows
        assert result["regressor_label"].tolist()[0] == "maintain"
        assert pd.isna(result["regressor_label"].tolist()[1])
        assert result["regressor_label"].tolist()[2] == "suppress"


# =====================================================
# is_excluded_trial_type
# =====================================================

class TestIsExcludedTrialType:
    @pytest.mark.parametrize("trial_type", ["fixation", "Fixation", "start_block", "end_block", "postrt", "trial_postrt_x"])
    def test_excluded_types(self, trial_type):
        assert is_excluded_trial_type(trial_type) is True

    @pytest.mark.parametrize("trial_type", ["face", "place", "view_face", "suppress_place"])
    def test_non_excluded_types(self, trial_type):
        assert is_excluded_trial_type(trial_type) is False


# =====================================================
# label_conditions_with_lag
# =====================================================

def _write_events_tsv(tmp_path, *rows, name="events.tsv"):
    """rows: (onset, duration, trial_type) tuples -> a real events.tsv on disk."""
    path = tmp_path / name
    pd.DataFrame(rows, columns=["onset", "duration", "trial_type"]).to_csv(path, sep="\t", index=False)
    return str(path)


def _full_frame_metadata_df(boldfile, eventfile, subject="01", session="", task="WM", run=1):
    """label_conditions_with_lag only reads full_frame_df for each boldfile's
    constant metadata (subject/session/task/run/eventfile) -- it re-derives
    the actual event list from eventfile itself (see the fix below), not from
    full_frame_df's own (lossy) trial_type/onset/duration/event_index
    columns. One row is enough regardless of how many real events/volumes
    exist."""
    return pd.DataFrame([{
        "boldfile": boldfile, "eventfile": eventfile, "volume_of_interest": 0,
        "trial_type": None, "onset": None, "duration": None, "event_index": None,
        "subject": subject, "session": session, "task": task, "run": run,
    }])


class TestLabelConditionsWithLag:
    def test_selects_lag_shifted_volumes_for_matched_event(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.mvpa_common.get_bold_header_info", lambda boldfile: (1.0, 20))
        eventfile = _write_events_tsv(tmp_path, (2.0, 3.0, "maintain_face"))
        df = _full_frame_metadata_df("run1", eventfile)
        conditions = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"}}

        result = label_conditions_with_lag(df, conditions, hemodynamic_lag=2.0)

        # start = onset(2.0) + lag(2.0) = 4.0 -> vol 4; 3.0s duration / 1.0s TR = 3 volumes
        assert result["volume_of_interest"].tolist() == [4, 5, 6]
        assert (result["regressor_label"] == "maintain").all()
        assert (result["trial_type"] == "maintain_face").all()  # real, unshifted label
        assert (result["trial_index"] == 1).all()

    def test_administrative_events_are_never_candidates(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.mvpa_common.get_bold_header_info", lambda boldfile: (1.0, 20))
        eventfile = _write_events_tsv(tmp_path, (2.0, 3.0, "fixation"))
        df = _full_frame_metadata_df("run1", eventfile)
        # a deliberately permissive query that would otherwise match anything
        conditions = {"anything": {"column": "trial_type", "match": "regex", "value": ".*"}}

        result = label_conditions_with_lag(df, conditions, hemodynamic_lag=0.0)
        assert result.empty

    def test_unmatched_events_produce_no_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.mvpa_common.get_bold_header_info", lambda boldfile: (1.0, 20))
        eventfile = _write_events_tsv(tmp_path, (2.0, 1.0, "probe"))
        df = _full_frame_metadata_df("run1", eventfile)
        conditions = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"}}

        result = label_conditions_with_lag(df, conditions, hemodynamic_lag=0.0)
        assert result.empty

    def test_zero_lag_matches_the_events_own_span(self, tmp_path, monkeypatch):
        monkeypatch.setattr("utils.mvpa_common.get_bold_header_info", lambda boldfile: (1.0, 20))
        eventfile = _write_events_tsv(tmp_path, (5.0, 2.0, "maintain_face"))
        df = _full_frame_metadata_df("run1", eventfile)
        conditions = {"maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"}}

        result = label_conditions_with_lag(df, conditions, hemodynamic_lag=0.0)
        assert result["volume_of_interest"].tolist() == [5, 6]

    def test_events_overwritten_in_the_dense_table_are_still_selected(self, tmp_path, monkeypatch):
        # regression: two real events close enough together that a dense,
        # single-winner-per-volume table (build_full_frame_table) would have
        # the second one fully overwrite the first (both resolve to volume 0
        # at this TR) -- label_conditions_with_lag must NOT enumerate
        # candidates from that lossy table; it re-reads eventfile directly,
        # so both events must still show up as independent training rows
        monkeypatch.setattr("utils.mvpa_common.get_bold_header_info", lambda boldfile: (2.5, 20))
        eventfile = _write_events_tsv(
            tmp_path, (0.0, 0.5, "maintain_face"), (0.4, 0.5, "suppress_face"),
        )
        # sanity: confirm the premise -- these two really do collide in the
        # dense table (both need volume 0 at TR=2.5s), with "suppress_face"
        # (later onset) winning and "maintain_face" completely disappearing
        raw = pd.read_csv(eventfile, sep="\t")
        dense = build_full_frame_table(raw, tr=2.5, n_frames=20)
        assert dense.loc[0, "trial_type"] == "suppress_face"
        assert "maintain_face" not in dense["trial_type"].values

        df = _full_frame_metadata_df("run1", eventfile)
        conditions = {
            "maintain": {"column": "trial_type", "match": "regex", "value": ".*maintain.*"},
            "suppress": {"column": "trial_type", "match": "regex", "value": ".*suppress.*"},
        }
        result = label_conditions_with_lag(df, conditions, hemodynamic_lag=0.0)

        assert set(result["regressor_label"]) == {"maintain", "suppress"}
        assert (result[result["regressor_label"] == "maintain"]["volume_of_interest"] == 0).all()


class TestQualifyingBoldfiles:
    def test_run_restriction_inside_a_condition_scopes_by_boldfile(self):
        # mirrors a real config where a condition ANDs trial_type with a run
        # restriction -- run is constant per boldfile, so this should
        # reproduce exactly the same run-level scoping it used to enforce
        # per-row
        df = pd.DataFrame({
            "boldfile": ["run10", "run10", "run1", "run1"],
            "trial_type": ["bottle", "cat", "bottle", "cat"],
            "run": ["10", "10", "1", "1"],
        })
        conditions = {
            "bottle": {"and": [{"column": "trial_type", "match": "exact", "value": "bottle"},
                                {"column": "run", "match": "in", "values": ["10", "11", "12"]}]},
        }
        assert qualifying_boldfiles(df, conditions) == {"run10"}


class TestBuildTimecourseInstructions:
    def test_decodes_every_frame_with_real_trial_type(self):
        rows = [(i, "view_face", 0.0, 1) for i in range(5)]  # the 5-TR view_face example
        rows += [(5 + i, "maintain", 5.0, 2) for i in range(3)]
        rows += [(8 + i, "fixation", 8.0, 3) for i in range(2)]
        df = _full_frame_df("run1", *rows)
        for extra_col, value in (("subject", "01"), ("session", ""), ("task", "WM"), ("run", 1)):
            df[extra_col] = value

        conditions = {"maintain": {"column": "trial_type", "match": "exact", "value": "maintain"}}
        instr = build_timecourse_instructions(df, conditions, VIEW_FACE_ANCHOR)

        # every single input frame comes out -- nothing skipped
        assert len(instr) == len(df)
        view_face_rows = instr[instr["trial_type"] == "view_face"].sort_values("window_index")
        assert view_face_rows["window_index"].tolist() == [0, 1, 2, 3, 4]
        assert view_face_rows["regressor_label"].isna().all()  # not one of `conditions`

        maintain_rows = instr[instr["trial_type"] == "maintain"]
        assert (maintain_rows["regressor_label"] == "maintain").all()

        fixation_rows = instr[instr["trial_type"] == "fixation"]
        assert fixation_rows["regressor_label"].isna().all()

    def test_boldfile_with_no_matching_condition_is_excluded_entirely(self):
        # "run1" has a maintain event (qualifies); "run2" only ever has
        # view_face/fixation, never matching `conditions` -- e.g. a
        # training-only run in a same-task, run-split design -- so it must
        # be excluded from continuous decoding altogether, not just left
        # with every row unscored
        qualifying_rows = [(i, "view_face", 0.0, 1) for i in range(2)]
        qualifying_rows += [(2 + i, "maintain", 2.0, 2) for i in range(2)]
        df_qualifying = _full_frame_df("run1", *qualifying_rows)

        non_qualifying_rows = [(i, "view_face", 0.0, 1) for i in range(2)]
        non_qualifying_rows += [(2 + i, "fixation", 2.0, 2) for i in range(2)]
        df_non_qualifying = _full_frame_df("run2", *non_qualifying_rows)

        df = pd.concat([df_qualifying, df_non_qualifying], ignore_index=True)
        for extra_col, value in (("subject", "01"), ("session", ""), ("task", "WM"), ("run", 1)):
            df[extra_col] = value

        conditions = {"maintain": {"column": "trial_type", "match": "exact", "value": "maintain"}}
        instr = build_timecourse_instructions(df, conditions, VIEW_FACE_ANCHOR)

        assert set(instr["boldfile"]) == {"run1"}
        assert len(instr) == len(df_qualifying)


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

    def test_scalar_metrics_collected_into_one_metadata_file(self, tmp_path):
        # total_scores/whole_voxels/selected_voxels/feature_percent are all
        # plain scalars -- collected together into one metadata.csv instead
        # of one near-empty file apiece
        categories = ["face", "place"]
        results = {
            "total_scores": 0.75,
            "whole_voxels": 5000,
            "selected_voxels": 250,
            "feature_percent": 5.0,
            "accuracy": np.array([[0.9, 0.1], [0.2, 0.8]]),  # non-scalar, stays its own file
        }
        pattern = str(tmp_path / "{metric}.csv")
        save_model_results(pattern, results, categories)

        assert not (tmp_path / "total_scores.csv").exists()
        assert not (tmp_path / "whole_voxels.csv").exists()
        assert (tmp_path / "accuracy.csv").exists()

        metadata = pd.read_csv(tmp_path / "metadata.csv", index_col=0)["value"]
        assert metadata["total_scores"] == pytest.approx(0.75)
        assert metadata["whole_voxels"] == pytest.approx(5000)
        assert metadata["selected_voxels"] == pytest.approx(250)
        assert metadata["feature_percent"] == pytest.approx(5.0)

    def test_no_scalar_metrics_writes_no_metadata_file(self, tmp_path):
        categories = ["face", "place"]
        results = {"accuracy": np.array([[0.9, 0.1], [0.2, 0.8]])}
        pattern = str(tmp_path / "{metric}.csv")
        save_model_results(pattern, results, categories)
        assert not (tmp_path / "metadata.csv").exists()


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
# load_images_and_mask
# =====================================================

def _labeled_df(boldfile, subject, session="", n_vols=3):
    """Distinct subject id per test -- load_images_and_mask's masker cache is
    module-level, keyed by (subject, session), so reusing one across tests
    would silently reuse a previous test's masker instead of building a
    fresh one for the config under test."""
    return pd.DataFrame({
        "boldfile": [boldfile] * n_vols,
        "subject": [subject] * n_vols,
        "session": [session] * n_vols,
        "volume_of_interest": list(range(n_vols)),
        "regressor": ([1, 2, 1] * n_vols)[:n_vols],
    })


class TestLoadImagesAndMask:
    def test_no_mask_pattern_uses_every_voxel_and_warns(self, synthetic_bold_file, capsys):
        df = _labeled_df(synthetic_bold_file, subject="load_test_no_mask")
        X, Y, idx, masker = load_images_and_mask(df, mask_pattern_template=None)
        assert X.shape == (3, 5 * 5 * 5)  # every voxel of the 5x5x5 synthetic volume
        assert "No model.mask.mask_pattern configured" in capsys.readouterr().out

    def test_empty_string_mask_pattern_also_means_no_mask(self, synthetic_bold_file):
        # falsy, same as omitting the key entirely (merge_with_defaults never
        # injects a default for it, so "" only happens if a config sets it
        # explicitly -- treated the same as unset, not as a literal empty path)
        df = _labeled_df(synthetic_bold_file, subject="load_test_empty_mask")
        X, Y, idx, masker = load_images_and_mask(df, mask_pattern_template="")
        assert X.shape == (3, 5 * 5 * 5)

    def test_configured_mask_pattern_missing_file_raises(self, synthetic_bold_file, tmp_path):
        df = _labeled_df(synthetic_bold_file, subject="load_test_missing_mask")
        missing_pattern = str(tmp_path / "does_not_exist.nii.gz")
        with pytest.raises(FileNotFoundError):
            load_images_and_mask(df, mask_pattern_template=missing_pattern)

    def test_configured_mask_pattern_restricts_voxels(self, synthetic_bold_file, tmp_path):
        mask_data = np.zeros((5, 5, 5), dtype=np.uint8)
        mask_data[:3, :, :] = 1  # 3*5*5 = 75 of the 125 voxels
        mask_path = tmp_path / "mask.nii.gz"
        nib.Nifti1Image(mask_data, np.eye(4)).to_filename(str(mask_path))

        df = _labeled_df(synthetic_bold_file, subject="load_test_real_mask")
        X, Y, idx, masker = load_images_and_mask(df, mask_pattern_template=str(mask_path))
        assert X.shape == (3, 75)


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

class TestExtractImportanceMap:
    def test_matches_model_performance_without_needing_labeled_data(self):
        # extract_importance_map should produce exactly what model_performance
        # returns as its impa_full, without ever touching testing_data/testing_labels
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=1)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        _, impa_from_model_performance = model_performance(pipe, X, y)
        impa_standalone = extract_importance_map(pipe, n_features=X.shape[1])
        np.testing.assert_array_equal(impa_standalone, impa_from_model_performance)

    def test_multiclass_shape(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=3, seed=2)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        impa = extract_importance_map(pipe, n_features=X.shape[1])
        assert impa.shape == (3, X.shape[1])


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
        assert xout["whole_voxels"] == X.shape[1]
        assert xout["selected_voxels"] == int(pipe.named_steps["feature_selection"].get_support().sum())
        assert xout["feature_percent"] == pytest.approx(100 * xout["selected_voxels"] / xout["whole_voxels"])

    def test_multiclass_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=3, seed=2)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, impa_full = model_performance(pipe, X, y)

        assert xout["total_scores"] > 0.7
        assert xout["accuracy"].shape == (3, 3)
        assert xout["evidence"].shape == (3, 3)
        assert xout["auc"].shape == (3,)
        assert impa_full.shape == (3, X.shape[1])

    def test_multiclass_evidence_rows_are_normalized_probabilities(self):
        # each row of the evidence matrix is a mean of per-trial evidence
        # vectors that individually sum to 1 (decision_evidence's softmax/
        # predict_proba normalization) -- so the row mean must too, unlike a
        # naive per-class sigmoid, which has no such constraint
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=4, seed=4)
        pipe = model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        xout, _ = model_performance(pipe, X, y)
        np.testing.assert_allclose(xout["evidence"].sum(axis=1), 1.0, atol=1e-8)

    def test_n_voxels_selects_exact_count_end_to_end(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=1)
        pipe = model_classification(X, y, feature_selection_cfg={"n_voxels": 4}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)
        assert int(pipe.named_steps["feature_selection"].get_support().sum()) == 4
        xout, impa_full = model_performance(pipe, X, y)
        assert xout["total_scores"] > 0.8  # cleanly separable data
        assert xout["whole_voxels"] == 10
        assert xout["selected_voxels"] == 4
        assert xout["feature_percent"] == pytest.approx(40.0)


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

    def test_boldfile_to_pipe_dispatches_overlapping_rows_to_substitute_pipe(self):
        # deliberately not a real fitted classifier -- a deterministic stand-in
        # so the dispatch assertion doesn't depend on any real model's
        # probabilistic behavior. predicted_label=99 is impossible for a real
        # 2-class (1/2) pipe, so its presence unambiguously proves this pipe
        # (not the fallback) decoded a given row.
        class FakeSubPipe:
            def __init__(self, predicted_label, n_selected, n_features):
                self._predicted_label = predicted_label
                self.named_steps = {"feature_selection": self}
                self._n_selected = n_selected
                self._n_features = n_features

            def get_support(self):
                return np.array([True] * self._n_selected + [False] * (self._n_features - self._n_selected))

            def predict(self, X):
                return np.full(X.shape[0], self._predicted_label)

            def predict_proba(self, X):
                proba = np.zeros((X.shape[0], 2))
                proba[:, 0] = 1.0
                return proba

        pipe = self._fitted_pipe_and_categories()
        categories = ["face", "place"]
        X, y = _separable_data(n_per_class=2, n_features=10, n_classes=2, seed=4)

        timecourse_df = pd.DataFrame({
            "subject": ["01"] * 4,
            "window_index": [0, 1, 0, 1],
            "regressor_label": [categories[c - 1] for c in y],
            "boldfile": ["shared_run.nii.gz", "shared_run.nii.gz", "other_run.nii.gz", "other_run.nii.gz"],
        })

        substitute = FakeSubPipe(predicted_label=99, n_selected=3, n_features=10)
        boldfile_to_pipe = {"shared_run.nii.gz": substitute}

        raw, _ = timecourse_decoding(
            pipe, X, y, timecourse_df, categories, feature_selection_cfg={"feat_p": 0.05},
            subject_id="01", model_descr="test_model", boldfile_to_pipe=boldfile_to_pipe,
        )

        shared_rows = raw[raw["boldfile"] == "shared_run.nii.gz"]
        other_rows = raw[raw["boldfile"] == "other_run.nii.gz"]

        assert (shared_rows["predicted_label"] == 99).all()
        assert shared_rows["selected_voxels"].unique().tolist() == [3]
        assert shared_rows["whole_voxels"].unique().tolist() == [10]

        # rows on the non-overlapping boldfile still fall back to `pipe`
        assert (other_rows["predicted_label"] != 99).all()

    def test_no_boldfile_to_pipe_behaves_exactly_as_before(self):
        # boldfile_to_pipe omitted entirely -- every row decoded by `pipe`,
        # matching pre-dispatch behavior
        pipe = self._fitted_pipe_and_categories()
        categories = ["face", "place"]
        X, y = _separable_data(n_per_class=4, n_features=10, n_classes=2, seed=4)
        timecourse_df = pd.DataFrame({
            "subject": ["01"] * len(y),
            "window_index": [0, 1, 2, 3] * 2,
            "regressor_label": [categories[c - 1] for c in y],
        })
        raw, _ = timecourse_decoding(
            pipe, X, y, timecourse_df, categories, feature_selection_cfg={"feat_p": 0.05},
            subject_id="01", model_descr="test_model",
        )
        expected_predictions = pipe.predict(X)
        np.testing.assert_array_equal(raw["correct"].to_numpy(), expected_predictions == y)


class TestBuildCvRawResults:
    def _fitted_pipe(self):
        X, y = _separable_data(n_per_class=15, n_features=10, n_classes=2, seed=3)
        return model_classification(X, y, feature_selection_cfg={"feat_p": 0.05}, classifier_name=CLASSIFIER_NAME, classifier_params=CLASSIFIER_PARAMS)

    def test_raw_columns_and_row_count(self):
        pipe = self._fitted_pipe()
        X, y = _separable_data(n_per_class=4, n_features=10, n_classes=2, seed=4)
        categories = ["face", "place"]
        held_out_df = pd.DataFrame({
            "subject": ["01"] * len(y),
            "task": ["WM"] * len(y),
            "trial_type": ["maintain"] * len(y),
            "run": [1, 1, 2, 2, 1, 1, 2, 2],
            "boldfile": ["run-1.nii.gz", "run-1.nii.gz", "run-2.nii.gz", "run-2.nii.gz"] * 2,
        })

        raw = build_cv_raw_results(
            pipe, X, y, held_out_df, categories, feature_selection_cfg={"feat_p": 0.05},
            model_descr="test_model", fold_id=1,
        )

        assert len(raw) == len(y)
        assert list(raw.columns[:3]) == ["subject", "model_descr", "fold"]
        for col in ("task", "trial_type", "run", "boldfile", "predicted_label", "correct",
                    "evidence_face", "evidence_place", "threshold_p", "selected_voxels",
                    "whole_voxels", "feature_percent"):
            assert col in raw.columns
        assert raw["model_descr"].unique().tolist() == ["test_model"]
        assert raw["fold"].unique().tolist() == [1]

    def test_correct_and_predicted_label_match_pipe_predictions(self):
        pipe = self._fitted_pipe()
        X, y = _separable_data(n_per_class=5, n_features=10, n_classes=2, seed=5)
        categories = ["face", "place"]
        held_out_df = pd.DataFrame({"run": [1] * len(y), "boldfile": ["run-1.nii.gz"] * len(y)})

        raw = build_cv_raw_results(
            pipe, X, y, held_out_df, categories, feature_selection_cfg={"feat_p": 0.05},
            model_descr="test_model", fold_id=2,
        )

        expected_predictions = pipe.predict(X)
        np.testing.assert_array_equal(raw["correct"].to_numpy(), expected_predictions == y)
        expected_labels = [categories[p - 1] for p in expected_predictions]
        assert raw["predicted_label"].tolist() == expected_labels


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
