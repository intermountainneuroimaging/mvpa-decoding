"""Shared pytest fixtures -- synthetic data only, no dependency on gitignored
real data (examples/sample-data, tutorial/haxby-data), so the suite runs
identically in CI as it does locally."""

import numpy as np
import nibabel as nib
import pandas as pd
import pytest


@pytest.fixture
def synthetic_bold_file(tmp_path):
    """A tiny, real 4D NIfTI file (5x5x5 voxels, 20 volumes, TR=2.0s) written to disk."""
    rng = np.random.default_rng(0)
    data = rng.random((5, 5, 5, 20)).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((3.0, 3.0, 3.0, 2.0))  # TR = 2.0s
    path = tmp_path / "sub-01_task-test_run-01_bold.nii.gz"
    img.to_filename(str(path))
    return str(path)


@pytest.fixture
def synthetic_master_spreadsheet():
    """A small master_spreadsheet-shaped DataFrame (matching
    generate_master_spreadsheet.py's output columns): 2 conditions across 2 runs,
    2 volumes per trial."""
    rows = []
    for run in (1, 2):
        for trial_index, (trial_type, onset) in enumerate([("face", 0.0), ("place", 5.0)], start=1):
            for vol_offset in range(2):
                rows.append({
                    "subject": "01",
                    "session": "",
                    "task": "test",
                    "run": run,
                    "trial_type": trial_type,
                    "trial_index": trial_index,
                    "onset": onset,
                    "duration": 2.0,
                    "volume_of_interest": vol_offset,
                    "boldfile": f"/fake/sub-01_run-{run:02d}_bold.nii.gz",
                    "eventfile": f"/fake/sub-01_run-{run:02d}_events.tsv",
                })
    return pd.DataFrame(rows)
