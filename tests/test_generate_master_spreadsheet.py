"""generate_master_spreadsheet.py: BIDS events.tsv -> master_spreadsheet.csv
table builder. Synthetic fixtures only (tiny NIfTI + tiny events.tsv written
to tmp_path) -- no dependency on gitignored real data."""

import json
import sys

import numpy as np
import nibabel as nib
import pandas as pd
import pytest

from workflows.generate_master_spreadsheet import (
    find_bold_file,
    process_events_file,
    main,
)


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
    of administrative, invalid, and real rows -- named/located so
    find_bold_file's default (no bold_glob) lookup finds the NIfTI from the
    events.tsv path."""
    data = np.random.default_rng(0).random((4, 4, 4, 20)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((2.0, 2.0, 2.0, 1.0))  # TR = 1.0s
    img.to_filename(str(tmp_path / "sub-01_task-test_run-01_bold.nii.gz"))

    events = pd.DataFrame([
        {"onset": 0.0, "duration": 2.0, "trial_type": "face"},
        {"onset": 3.0, "duration": 2.0, "trial_type": "fixation"},   # administrative, but not excluded here
        {"onset": 6.0, "duration": float("nan"), "trial_type": "place"},  # invalid -- never covers anything
        {"onset": 9.0, "duration": 2.0, "trial_type": "place"},
    ])
    events_path = tmp_path / "sub-01_task-test_run-01_events.tsv"
    events.to_csv(events_path, sep="\t", index=False)
    return str(events_path), str(tmp_path)


class TestProcessEventsFile:
    def test_one_row_per_volume_including_administrative_types(self, bids_run):
        events_path, derivatives_root = bids_run
        table = process_events_file(events_path, derivatives_root)
        # 20-frame run -> exactly one row per volume, nothing excluded/dropped
        assert table["volume_of_interest"].tolist() == list(range(20))

        by_vol = table.set_index("volume_of_interest")["trial_type"]
        assert by_vol.loc[0] == "face" and by_vol.loc[1] == "face"
        assert by_vol.loc[3] == "fixation" and by_vol.loc[4] == "fixation"
        assert by_vol.loc[9] == "place" and by_vol.loc[10] == "place"

        # the NaN-duration row never covers anything; gaps stay NaN, not dropped
        assert len(table) == 20
        assert pd.isna(by_vol.loc[2])
        assert pd.isna(by_vol.loc[19])

    def test_event_index_is_contiguous_over_valid_events(self, bids_run):
        events_path, derivatives_root = bids_run
        table = process_events_file(events_path, derivatives_root)
        # face, fixation, place (the NaN-duration "place" row is invalid, never assigned an event_index)
        assert sorted(table["event_index"].dropna().unique().tolist()) == [1, 2, 3]

    def test_missing_bold_file_returns_none(self, tmp_path):
        events = pd.DataFrame([{"onset": 0.0, "duration": 2.0, "trial_type": "face"}])
        events_path = tmp_path / "sub-99_task-test_run-01_events.tsv"
        events.to_csv(events_path, sep="\t", index=False)
        result = process_events_file(str(events_path), str(tmp_path))
        assert result is None


# =====================================================
# main()
# =====================================================

class TestMain:
    def _config(self, tmp_path, output_file):
        event_extraction = {
            "bids_root": str(tmp_path),
            "output_file": str(output_file),
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"event_extraction": event_extraction}))
        return str(config_path)

    def test_writes_one_row_per_volume(self, tmp_path, bids_run, monkeypatch):
        _events_path, _derivatives_root = bids_run  # writes the NIfTI/events.tsv under tmp_path
        output_file = tmp_path / "master_spreadsheet.csv"
        config_path = self._config(tmp_path, output_file)

        monkeypatch.setattr(sys, "argv", ["generate_master_spreadsheet.py", "--config", config_path])
        main()

        assert output_file.exists()
        table = pd.read_csv(output_file)
        assert len(table) == 20  # one row per volume of the 20-frame bids_run fixture
