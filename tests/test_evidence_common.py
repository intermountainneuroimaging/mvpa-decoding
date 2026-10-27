"""_interactive_notebooks/evidence_common.py: what the two evidence notebooks share."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from _interactive_notebooks import evidence_common as common


@pytest.fixture(autouse=True)
def fresh_settings():
    common.settings.clear()
    yield
    common.settings.clear()


def group_table(n_subjects=3, n_trials=4, seed=0):
    """A tiny group decoding table of a category classifier: per subject, task and trial, 6 view frames (view_<cat>), 6 operation
    frames (<op>_<cat>) and 2 fixation frames; regressor_label is set on the view frames only."""
    rng = np.random.default_rng(seed)
    rows = []
    ops = ["maintain", "suppress", "switch", "clear"]
    for s in range(n_subjects):
        for task in ("WMpos", "WMneg"):
            boldfile = f"s{s}_{task}"
            for trial in range(n_trials):
                cat = "face" if trial % 2 == 0 else "place"
                op = ops[trial % 4]
                for w in range(14):
                    tt = f"view_{cat}" if w < 6 else (f"{op}_{cat}" if w < 12 else "trial_fixation")
                    rows.append(dict(subject=f"{s:02d}", task=task, boldfile=boldfile, trial_index=trial + 1, window_index=w,
                                     volume_of_interest=trial * 14 + w, trial_type=tt, regressor_label=cat if w < 6 else np.nan,
                                     evidence_face=rng.random(), evidence_place=rng.random(), evidence_item=rng.random()))
    return pd.DataFrame(rows)


def cfg(path, **overrides):
    base = dict(GROUP_CSV_PATH=str(path), DESC="model", MODEL_TITLE=None, TR_SECONDS=0.46, MIN_SUBJECT_FRACTION=0.9, EXTRA_TRAILING_TRS=2,
                BALANCE_PRECEDING_CATEGORY=False, BALANCE_WITHIN=("operation", "object", "valence"), BALANCE_MODE="subsample", BALANCE_SEED=0,
                BASELINE_CORRECTION=False, BASELINE_TIMEPOINTS=(4, 5, 6), BASELINE_GROUP_BY=("object",),
                DEFAULT_BASELINE_GROUP_BY=("object", "operation", "valence"), TEST_BASELINE_BIN_WHEN_NOT_ZERO=True,
                BIN_SIZE=3, SLIDING=False, OUTPUT_DIR="unused")
    base.update(overrides)
    return base


class TestPrepareEvidence:
    def _run(self, tmp_path, **overrides):
        path = tmp_path / "g.csv"; group_table().to_csv(path, index=False)
        return common.prepare_evidence(cfg(path, **overrides), primary="object", secondary="operation",
                                       secondary_regex=r"(maintain|suppress|switch|clear)")

    def test_every_trial_gets_one_window_of_the_same_length_keyed_on_its_own_labels(self, tmp_path):
        ev = self._run(tmp_path)
        df = ev.df
        assert ev.WINDOW_LEN == ev.max_window + 1 + 2
        sizes = df.groupby(["boldfile", "trial_index"]).size()
        last_trial = sizes.index.get_level_values("trial_index") == sizes.index.get_level_values("trial_index").max()
        assert sizes[~last_trial].eq(ev.WINDOW_LEN).all()               # the last trial of a run has no later frames to borrow
        assert sizes[last_trial].le(ev.WINDOW_LEN).all()
        assert set(df["object"]) == {"face", "place"} and set(df["valence"]) == {"pos", "neg"}
        assert set(df["operation"]) <= {"maintain", "suppress", "switch", "clear"}
        assert (df["regressor_label"] == df["object"]).all() and df["value"].notna().all()
        np.testing.assert_allclose(df["time_sec"], df["window_index"] * 0.46)

    def test_value_is_the_evidence_for_the_owning_trials_own_category(self, tmp_path):
        ev = self._run(tmp_path)
        d = ev.df[ev.df["window_index"] == 1]
        expected = np.where(d["object"] == "face", d["evidence_face"], d["evidence_place"])
        np.testing.assert_allclose(d["value"], expected)

    def test_baseline_correction_makes_each_participants_category_curve_zero_over_the_baseline_trs(self, tmp_path):
        ev = self._run(tmp_path, BASELINE_CORRECTION=True)
        curve = ev.df.groupby(["subject", "object", "window_index"])["value"].mean().reset_index()
        baseline = curve[curve["window_index"].isin(ev.BASELINE_WINDOWS)].groupby(["subject", "object"])["value"].mean()
        assert ev.BASELINE_WINDOWS == (3, 4, 5) and baseline.abs().max() < 1e-12
        assert "TRs 4-6" in ev.BASELINE_NOTE and ev.BASELINE_ONLY_BINS == {1}

    def test_no_baseline_note_for_the_default_grouping(self, tmp_path):
        ev = self._run(tmp_path, BASELINE_CORRECTION=True, BASELINE_GROUP_BY=("object", "operation", "valence"))
        assert ev.BASELINE_NOTE is None and ev.OPERATION_HAS_OWN_BASELINE

    def test_the_shared_settings_are_configured_for_the_figures(self, tmp_path):
        self._run(tmp_path, BASELINE_CORRECTION=True)
        assert common.setting("BASELINE_ONLY_BINS") == {1} and common.setting("MODEL_HEAD") == "model"

    def test_a_missing_table_names_the_ingest_command(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ingest"):
            common.prepare_evidence(cfg(tmp_path / "nope.csv"), "object", "operation", r"(maintain)")


class TestWindowsAndFootnotes:
    def _wide(self):
        idx = pd.MultiIndex.from_product([["a", "b"], range(9)], names=["subject", "window_index"])
        return pd.DataFrame({"pos": np.arange(18.0), "neg": np.zeros(18)}, index=idx)

    def test_make_windows_follows_the_sliding_setting(self):
        common.configure(SLIDING=False, BIN_SIZE=3)
        assert len(common.make_windows(self._wide())[1]) == 3
        common.configure(SLIDING=True, BIN_SIZE=3)
        assert len(common.make_windows(self._wide())[1]) == 7

    def test_baseline_bins_are_dropped_unless_the_comparison_is_not_zero_there(self):
        wide = self._wide()
        per_bin, _ = common.bin_by_size(wide, bin_size=3, tr_offset=0)
        common.configure(BASELINE_ONLY_BINS={1}, TEST_BASELINE_BIN_WHEN_NOT_ZERO=True)
        assert 1 not in common.drop_baseline_bins(per_bin, True).index.get_level_values("bin")
        assert 1 in common.drop_baseline_bins(per_bin, False).index.get_level_values("bin")

    def test_pos_vs_neg_is_zero_only_when_valence_has_its_own_baseline(self):
        common.configure(BASELINE_GROUP_BY=("object",))
        assert not common.pos_vs_neg_is_zero_in_baseline(None)
        common.configure(BASELINE_GROUP_BY=("object", "valence"))
        assert common.pos_vs_neg_is_zero_in_baseline(None) and not common.pos_vs_neg_is_zero_in_baseline("clear")

    def test_significance_note_says_when_the_statistics_start_and_how_windows_are_built(self):
        common.configure(FDR_METHOD="bh", FDR_FIRST_TR=7, SLIDING=False, BIN_SIZE=3, ALPHA=0.05, STAT_METHOD="ttest_1samp_diff",
                         POS_VS_NEG_ALTERNATIVE="two-sided")
        note = common.significance_note("valence")
        assert "Benjamini-Hochberg" in note and "TR 7" in note and "non-overlapping 3-TR windows" in note and "Yekutieli" not in note
        common.configure(SLIDING=True)
        assert "slid one TR at a time" in common.significance_note("valence") and "p_fdr_by" in common.significance_note("valence")

    def test_a_figure_without_a_kind_has_no_significance_footnote_and_no_pad(self):
        fig = plt.figure()
        assert common.figure_notes(fig) == "" and common.baseline_note_pad(fig) == 0.0
        fig._note_kind = "split"
        common.configure(FDR_FIRST_TR=7, FDR_METHOD="bh", ALPHA=0.05)
        assert "split-half" in common.figure_notes(fig) and common.baseline_note_pad(fig) > 0
        plt.close(fig)


class TestEventMarkers:
    def test_marker_positions_of_a_view_operation_fixation_trial(self):
        df = group_table()
        df = df[df["window_index"].isin(range(14))]
        markers = common.compute_marker_positions(df)
        starts = [m["mean_start"] for m in markers]
        assert starts[0] == 0.0 and 6.0 in starts and 12.0 in starts


class TestPages:
    def test_overlap_table_counts_and_dice(self):
        mean = np.zeros((4, 4, 4, 4))
        mean[0, 0, 0, 0], mean[1, 1, 1, 0], mean[2, 2, 2, 0] = 5.0, 4.0, -3.0
        sig = np.zeros(mean.shape, dtype=bool)
        sig[0, 0, 0, 0] = sig[2, 2, 2, 0] = True
        table = common.importance_overlap_table(mean, sig, ("maintain", "suppress", "switch", "clear"), top_fraction=1.0)
        row = table.set_index("operation").loc["maintain"]
        assert (row.voxels_top, row.voxels_significant, row.voxels_both) == (3, 2, 2)
        assert row.dice == pytest.approx(2 * 2 / 5)
        assert (row.significant_positive, row.significant_negative) == (1, 1)

    def test_order_pages_puts_unknown_names_last(self):
        pages = [("b", "image", "p"), ("x", "figure", "u"), ("a", "figure", "t")]
        assert common.order_pages(pages, ("a", "b")) == [("figure", "t"), ("image", "p"), ("figure", "u")]

    def test_vector_figures_and_png_pages_end_up_in_one_pdf(self, tmp_path):
        png = tmp_path / "surface.png"
        fig_png = plt.figure(figsize=(4, 2)); fig_png.savefig(png, dpi=130); plt.close(fig_png)
        fig = plt.figure(); fig.add_subplot(111).plot([0, 1])
        out = tmp_path / "all.pdf"
        assert common.combine_pages([("figure", fig), ("image", str(png))], str(out)) == 2
        plt.close(fig)
        data = out.read_bytes()
        assert data.count(b"/Type /Page\n") + data.count(b"/Type /Page ") >= 2

    def test_toy_data_has_a_clear_effect_in_the_last_bin(self):
        toy = common.make_toy_data()
        last = toy[toy["window_index"] >= 9].groupby("valence")["value"].mean()
        assert last["pos"] - last["neg"] > 0.05
