"""operation_results: the overlap table between the top-weights and the significant-voxel maps."""

import numpy as np
import pytest

from _interactive_notebooks import operation_results as opr


class TestOverlapTable:
    def test_counts_and_dice(self):
        mean = np.zeros((4, 4, 4, 4))
        mean[0, 0, 0, 0], mean[1, 1, 1, 0], mean[2, 2, 2, 0] = 5.0, 4.0, -3.0
        sig = np.zeros(mean.shape, dtype=bool)
        sig[0, 0, 0, 0] = sig[2, 2, 2, 0] = True
        table = opr.importance_overlap_table(mean, sig, top_fraction=1.0)   # every non-zero voxel is "top"
        row = table.set_index("operation").loc["maintain"]
        assert (row.voxels_top, row.voxels_significant, row.voxels_both) == (3, 2, 2)
        assert row.dice == pytest.approx(2 * 2 / 5)
        assert (row.significant_positive, row.significant_negative) == (1, 1)
