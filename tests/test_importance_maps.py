"""utils.importance_maps: polarity-matched importance (weight x class-mean activity), per-subject z-scoring and
the group top-fraction maps, on small synthetic arrays."""

import numpy as np
import pytest

from utils import importance_maps as im


class TestClassMeanActivity:
    def test_z_scored_over_training_samples_then_averaged_per_class(self):
        # voxel 0 is high for class 1 and low for class 2; voxel 1 is constant
        data = np.array([[2.0, 5.0], [4.0, 5.0], [0.0, 5.0], [-2.0, 5.0]])
        labels = np.array([1, 1, 2, 2])
        act = im.class_mean_activity(data, labels, [1, 2])
        z = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std()
        np.testing.assert_allclose(act[0], [z[:2].mean(), 0.0])
        np.testing.assert_allclose(act[1], [z[2:].mean(), 0.0])
        assert act[0, 0] > 0 > act[1, 0]

    def test_classes_axis_follows_the_given_order_and_missing_class_is_zero(self):
        data = np.random.default_rng(0).normal(size=(10, 3))
        act = im.class_mean_activity(data, np.array([1] * 5 + [2] * 5), [2, 1, 3])
        assert act.shape == (3, 3) and not act[2].any()
        np.testing.assert_allclose(act[0], im.class_mean_activity(data, np.array([1] * 5 + [2] * 5), [2])[0])

    def test_activity_tag(self):
        assert im.activity_tag("impa") == "impa_act" and im.activity_tag("impa_mni") == "impa_act_mni"


class TestPolarityImportance:
    def test_matching_signs_split_into_activation_and_deactivation_evidence(self):
        w = np.array([2.0, -3.0, 4.0, -1.0, 0.0])
        a = np.array([1.5, -2.0, -1.0, 0.5, 3.0])        # +/+ , -/- , +/- , -/+ , unselected
        pos, neg = im.polarity_importance(w, a)
        np.testing.assert_allclose(pos, [3.0, 0, 0, 0, 0])
        np.testing.assert_allclose(neg, [0, 6.0, 0, 0, 0])
        assert (pos >= 0).all() and (neg >= 0).all()

    def test_opposing_polarity_and_unselected_voxels_are_ignored(self):
        pos, neg = im.polarity_importance(np.array([1.0, -1.0, 0.0]), np.array([-1.0, 1.0, 2.0]))
        assert not pos.any() and not neg.any()


class TestZscoreImportance:
    def test_one_distribution_for_positive_and_negative_values(self):
        pos = np.array([1.0, 2.0, 0.0, 0.0])
        neg = np.array([0.0, 0.0, 3.0, 4.0])
        zp, zn = im.zscore_importance(pos, neg)
        both = np.concatenate([zp[pos != 0], zn[neg != 0]])
        assert both.mean() == pytest.approx(0.0, abs=1e-12) and both.std() == pytest.approx(1.0)
        assert (zp[pos == 0] == 0).all() and (zn[neg == 0] == 0).all()      # ignored voxels stay 0
        assert zn[3] > zn[2] > zp[1] > zp[0]                                 # order of the magnitudes is kept

    def test_class_axis_is_done_per_class_and_degenerate_maps_are_zero(self):
        pos = np.zeros((2, 2, 2, 2)); neg = np.zeros((2, 2, 2, 2))
        pos[0, 0, 0, 0], pos[1, 1, 1, 0], neg[0, 1, 0, 0] = 1.0, 2.0, 5.0
        pos[0, 0, 0, 1] = 7.0                         # class 1 has one value only: no distribution
        zp, zn = im.zscore_importance(pos, neg)
        assert zp[..., 0].any() and not zp[..., 1].any() and not zn[..., 1].any()


class TestTopFractionAndSmoothing:
    def test_keeps_the_top_fraction_and_drops_small_clusters(self):
        m = np.zeros((12, 12, 12))
        m[2:5, 2:5, 2:5] = 10.0 + np.arange(27).reshape(3, 3, 3)      # a 27-voxel cluster of the highest values
        m[9, 9, 9] = 100.0                                             # one huge isolated voxel
        m[7:9, 0:2, 0:2] = 1.0                                         # low values: outside the top fraction
        mask, thr = im.top_fraction_clusters(m, top_fraction=0.5, min_cluster=10)
        assert mask[2:5, 2:5, 2:5].any() and not mask[9, 9, 9] and not mask[7:9, 0:2, 0:2].any()
        keep_isolated, _ = im.top_fraction_clusters(m, top_fraction=0.5, min_cluster=1)
        assert keep_isolated[9, 9, 9]

    def test_empty_map(self):
        mask, thr = im.top_fraction_clusters(np.zeros((4, 4, 4)))
        assert not mask.any() and np.isnan(thr)

    def test_smoothing_spreads_but_keeps_the_peak_location_and_zero_means_none(self):
        v = np.zeros((15, 15, 15)); v[7, 7, 7] = 1.0
        s = im.smooth_fwhm(v, fwhm_mm=12.0, voxel_mm=3.0)
        assert np.unravel_index(s.argmax(), s.shape) == (7, 7, 7)
        assert s[7, 7, 7] < 1.0 and s[7, 7, 9] > 0
        np.testing.assert_array_equal(im.smooth_fwhm(v, 0, 3.0), v)


class TestGroupMaps:
    def make_pairs(self, n=12, shape=(10, 10, 10), k=2, seed=0):
        """Every subject: voxel (2,2,2) cluster has weight + and activity + for class 0 (activation evidence),
        voxel cluster at (7,7,7) has weight - and activity - for class 0 (deactivation evidence); noise elsewhere."""
        rng = np.random.default_rng(seed)
        pairs = []
        for _ in range(n):
            w = rng.normal(0, 0.1, size=shape + (k,)); a = rng.normal(0, 0.5, size=shape + (k,))
            w[1:4, 1:4, 1:4, 0] = 1.0 + rng.random(); a[1:4, 1:4, 1:4, 0] = 1.0
            w[6:9, 6:9, 6:9, 0] = -(1.0 + rng.random()); a[6:9, 6:9, 6:9, 0] = -1.0
            pairs.append((w, a))
        return pairs

    def test_finds_the_positive_and_negative_clusters(self):
        out = im.group_importance_maps(self.make_pairs(), top_fraction=0.02, min_cluster=10, fwhm_mm=0)
        assert out["n_subjects"] == 12 and out["top_activation"].shape == (10, 10, 10, 2)
        assert out["top_activation"][2, 2, 2, 0] > 0 and out["top_deactivation"][7, 7, 7, 0] > 0
        assert out["top_activation"][7, 7, 7, 0] == 0 and out["top_deactivation"][2, 2, 2, 0] == 0   # each cluster is in the map of its own polarity: activation vs deactivation
        # the smoothing is applied to the kept values only
        smooth = im.group_importance_maps(self.make_pairs(), top_fraction=0.02, min_cluster=10, fwhm_mm=12)
        assert smooth["smooth_activation"][2, 2, 2, 0] > 0 and smooth["smooth_activation"][0, 0, 0, 0] >= 0
        assert out["thresholds"].shape == (2, 2)

    def test_needs_subjects(self):
        with pytest.raises(ValueError):
            im.group_importance_maps([])
