"""operation_results: the overlap table between the top-weights and the significant-voxel maps."""

import os

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


class TestCombinedPdf:
    def test_vector_figures_and_png_pages_end_up_in_one_pdf(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png = tmp_path / "surface.png"
        fig_png = plt.figure(figsize=(4, 2))
        fig_png.savefig(png, dpi=130)
        plt.close(fig_png)
        fig = plt.figure()
        fig.text(0.5, 0.5, "vector page")
        out = tmp_path / "all.pdf"
        assert opr.combine_pages([("figure", fig), ("image", str(png))], str(out)) == 2
        plt.close(fig)
        assert out.read_bytes().count(b"/Type /Page\n") + out.read_bytes().count(b"/Type /Page ") >= 2

    def test_pages_are_ordered_confusion_timecourse_then_importance_maps(self):
        pages = [("importance_maps_surface_polarity", "image", "p"), ("timecourse_decoding", "figure", "t"),
                 ("importance_maps_surface", "image", "w"), ("confusion_matrices", "figure", "c"),
                 ("importance_maps_surface_significant", "image", "s"), ("extra", "figure", "x")]
        assert [item for _, item in opr.order_pages(pages)] == ["c", "t", "s", "p", "w", "x"]

    def test_polarity_maps_use_another_colour_map_than_the_signed_maps(self):
        assert opr.POLARITY_CMAP != "RdBu_r"


class TestOrganizedLayout:
    def test_inputs_are_found_in_the_model_folder(self, tmp_path):
        (tmp_path / "raw_data").mkdir(); (tmp_path / "importance_maps").mkdir()
        for rel in ("group_summary.csv", "raw_data/group_decoding_results.csv", "importance_maps/importance_signflip_stats.nii.gz"):
            (tmp_path / rel).write_text("x")
        found = opr.model_inputs(str(tmp_path))
        assert set(found) == {"summary_csv", "group_csv", "stats_nii"}          # no polarity maps in this model
        assert found["group_csv"].endswith(os.path.join("raw_data", "group_decoding_results.csv"))

    def test_figures_go_to_decoding_uncorrected_and_importance_maps(self):
        assert opr.layout_target("/m", "confusion_matrices", ".pdf") == "/m/decoding/uncorrected/confusion_matrices.pdf"
        assert opr.layout_target("/m", "timecourse_decoding", ".png") == "/m/decoding/uncorrected/timecourse_decoding.png"
        assert opr.layout_target("/m", "importance_maps_surface_significant", ".png") == "/m/importance_maps/surface_significant.png"
        assert opr.layout_target("/m", "importance_maps_surface", ".png") == "/m/importance_maps/surface_top20.png"
