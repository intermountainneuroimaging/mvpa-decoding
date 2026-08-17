"""generate_master_spreadsheet.py: BIDS events.tsv -> master_spreadsheet.csv
table builder. Synthetic fixtures only (tiny NIfTI + tiny events.tsv written
to tmp_path) -- no dependency on gitignored real data."""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from workflows.generate_master_spreadsheet import (
    is_excluded_trial_type,
    find_bold_file,
    process_events_file,
)


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
# find_bold_file
# =====================================================

class TestFindBoldFile:
    def test_bold_glob_with_generic_entity(self, tmp_path):
        # dir- is not one of the hardcoded aliases -- must be available generically
        (tmp_path / "sub-01_task-loc_dir-pa_run-01_bold.nii.gz").write_bytes(b"")
        entities = {"sub": "01", "task": "loc", "dir": "pa", "run": "01"}
        matches = find_bold_file(
            str(tmp_path), entities,
            bold_glob="sub-{subject}_task-{task}_dir-{dir}_run-{run}_bold.nii.gz",
        )
        assert len(matches) == 1

    def test_bold_glob_missing_entity_returns_empty_not_crash(self, tmp_path):
        entities = {"sub": "01", "task": "loc", "run": "01"}  # no "dir"
        matches = find_bold_file(str(tmp_path), entities, bold_glob="sub-{subject}_dir-{dir}_bold.nii.gz")
        assert matches == []

    def test_no_bold_glob_fallback_matches_on_tokens(self, tmp_path):
        (tmp_path / "sub-01_task-loc_run-01_desc-preproc_bold.nii.gz").write_bytes(b"")
        entities = {"sub": "01", "task": "loc", "run": "01"}
        matches = find_bold_file(str(tmp_path), entities, bold_glob=None)
        assert len(matches) == 1

    def test_no_match_returns_empty_list(self, tmp_path):
        entities = {"sub": "99", "task": "loc", "run": "01"}
        matches = find_bold_file(str(tmp_path), entities, bold_glob=None)
        assert matches == []


# =====================================================
# process_events_file
# =====================================================

@pytest.fixture
def bids_run(tmp_path):
    """A tiny real NIfTI (TR=1.0s, 20 frames) + a matching events.tsv with a mix
    of valid, excluded, and invalid rows -- named/located so find_bold_file's
    default (no bold_glob) lookup finds the NIfTI from the events.tsv path."""
    data = np.random.default_rng(0).random((4, 4, 4, 20)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((2.0, 2.0, 2.0, 1.0))  # TR = 1.0s
    img.to_filename(str(tmp_path / "sub-01_task-test_run-01_bold.nii.gz"))

    events = pd.DataFrame([
        {"onset": 0.0, "duration": 2.0, "trial_type": "face"},
        {"onset": 3.0, "duration": 2.0, "trial_type": "fixation"},   # excluded
        {"onset": 6.0, "duration": float("nan"), "trial_type": "place"},  # invalid
        {"onset": 9.0, "duration": 2.0, "trial_type": "place"},
    ])
    events_path = tmp_path / "sub-01_task-test_run-01_events.tsv"
    events.to_csv(events_path, sep="\t", index=False)
    return str(events_path), str(tmp_path)


class TestProcessEventsFile:
    def test_excludes_administrative_and_invalid_rows(self, bids_run):
        events_path, derivatives_root = bids_run
        table = process_events_file(events_path, derivatives_root, hemodynamic_lag=0.0)
        # 2 valid trials (face, place) remain; fixation + NaN-duration dropped
        assert set(table["trial_type"].unique()) == {"face", "place"}

    def test_trial_index_is_contiguous_over_kept_rows(self, bids_run):
        events_path, derivatives_root = bids_run
        table = process_events_file(events_path, derivatives_root, hemodynamic_lag=0.0)
        assert sorted(table["trial_index"].unique().tolist()) == [1, 2]

    def test_one_row_per_volume(self, bids_run):
        events_path, derivatives_root = bids_run
        table = process_events_file(events_path, derivatives_root, hemodynamic_lag=0.0)
        # each 2.0s trial at TR=1.0s -> 2 volumes -> 2 rows per trial, 2 trials = 4 rows
        assert len(table) == 4

    def test_missing_bold_file_returns_none(self, tmp_path):
        events = pd.DataFrame([{"onset": 0.0, "duration": 2.0, "trial_type": "face"}])
        events_path = tmp_path / "sub-99_task-test_run-01_events.tsv"
        events.to_csv(events_path, sep="\t", index=False)
        result = process_events_file(str(events_path), str(tmp_path), hemodynamic_lag=0.0)
        assert result is None
