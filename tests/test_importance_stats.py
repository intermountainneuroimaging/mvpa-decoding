"""utils.importance_stats: sign-flip test of importance maps with subject-specific feature selection."""

import numpy as np
import pytest

from utils import importance_stats as ist


def make_maps(n_subjects=40, shape=(6, 6, 6), k=4, keep=0.7, seed=0):
    """Noise maps in which each subject keeps a random `keep` fraction of voxels (the rest are zero in all
    classes, like a voxel feature selection left out). Voxel (1,1,1): weight +1 for class 0 and -1 for class 1
    in every subject that kept it. Voxel (2,2,2) is kept by only 5 subjects, all +1 for class 0."""
    rng = np.random.default_rng(seed)
    maps = rng.normal(0, 0.3, size=(n_subjects,) + shape + (k,)).astype(np.float32)
    kept = rng.random((n_subjects,) + shape) < keep
    kept[:, 1, 1, 1] = True
    kept[:, 2, 2, 2] = False
    kept[:5, 2, 2, 2] = True
    maps[:, 1, 1, 1, 0] += 1.0
    maps[:, 1, 1, 1, 1] -= 1.0
    maps[:5, 2, 2, 2, 0] = 1.0
    maps *= kept[..., None]
    return maps, kept


class TestSignFlipTest:
    def test_consistent_voxels_are_found_and_noise_is_not(self):
        maps, _ = make_maps()
        r = ist.sign_flip_test(maps, n_perm=6000, seed=1, min_subjects=10)   # p floor 1.7e-4: FDR over ~200 voxels needs p < .00025
        assert r["p"][1, 1, 1, 0] < 0.001 and r["p"][1, 1, 1, 1] < 0.001
        assert r["mean_selected"][1, 1, 1, 0] > 0.8 and r["mean_selected"][1, 1, 1, 1] < -0.8
        sig = ist.significant_voxels(r["p"])
        assert sig[1, 1, 1, 0] and sig[1, 1, 1, 1]
        assert sig[..., 2].sum() == 0 and sig[..., 3].sum() == 0            # noise-only classes

    def test_unselected_voxels_are_missing_not_zeros(self):
        maps, kept = make_maps()
        r = ist.sign_flip_test(maps, n_perm=100, seed=0, min_subjects=10)
        np.testing.assert_array_equal(r["n_selected"], kept.sum(axis=0))
        v = (1, 1, 1)
        # the mean over the subjects that kept it is not diluted by the others' zeros
        assert r["mean_selected"][v + (0,)] == pytest.approx(maps[:, 1, 1, 1, 0].mean(), rel=1e-4)
        partial = (0, 0, 0)
        n = int(kept[:, 0, 0, 0].sum())
        assert n < len(maps)
        assert r["mean_selected"][partial + (2,)] == pytest.approx(maps[kept[:, 0, 0, 0], 0, 0, 0, 2].mean(), rel=1e-4)
        assert r["mean_all"][partial + (2,)] == pytest.approx(maps[:, 0, 0, 0, 2].mean(), rel=1e-4)

    def test_min_subjects_leaves_rarely_selected_voxels_untested(self):
        maps, _ = make_maps()
        r = ist.sign_flip_test(maps, n_perm=500, seed=0, min_subjects=10)
        assert not r["tested"][2, 2, 2] and np.isnan(r["p"][2, 2, 2]).all()      # 5 subjects, all +1
        loose = ist.sign_flip_test(maps, n_perm=500, seed=0, min_subjects=5)
        assert loose["tested"][2, 2, 2] and loose["p"][2, 2, 2, 0] < 0.2           # tested, but 5 subjects cannot reach p < .001

    def test_p_is_never_below_one_over_n_plus_one_and_is_reproducible(self):
        maps, _ = make_maps(n_subjects=30)
        a = ist.sign_flip_test(maps, n_perm=999, seed=3, min_subjects=10)["p"]
        b = ist.sign_flip_test(maps, n_perm=999, seed=3, min_subjects=10)["p"]
        np.testing.assert_array_equal(a, b)
        assert np.nanmin(a) >= 1 / 1000 - 1e-12

    def test_null_p_values_are_roughly_uniform(self):
        # pure noise, sign-symmetric: about 5% of tested voxel-classes fall below 0.05
        rng = np.random.default_rng(5)
        maps = rng.normal(size=(30, 8, 8, 8, 2)).astype(np.float32)
        p = ist.sign_flip_test(maps, n_perm=1000, seed=0, min_subjects=10)["p"]
        assert 0.03 < (p < 0.05).mean() < 0.07

    def test_matches_the_exact_distribution_for_a_tiny_group(self):
        # 4 subjects: all 16 sign patterns, p of |mean| for the all-positive voxel = 2 / 16 (the pattern and its mirror)
        maps = np.zeros((4, 1, 1, 1, 1), dtype=np.float32)
        maps[:, 0, 0, 0, 0] = [1.0, 2.0, 3.0, 4.0]
        p = ist.sign_flip_test(maps, n_perm=20000, seed=0, min_subjects=4)["p"][0, 0, 0, 0]
        assert p == pytest.approx(2 / 16, abs=0.02)

    def test_zero_permutations_skips_the_test_and_too_few_subjects_raise(self):
        maps, _ = make_maps(n_subjects=12)
        assert "p" not in ist.sign_flip_test(maps, n_perm=0)
        with pytest.raises(ValueError, match="at least 3"):
            ist.sign_flip_test(maps[:2])


class TestSignificantVoxels:
    def test_both_the_voxel_threshold_and_fdr_must_pass(self):
        p = np.full((10, 10, 10), 0.5)
        p[0, 0, 0] = 1e-6
        p[1, 1, 1] = 0.0005        # p < 0.001 but, among 1000 voxels, adjusted p = 0.5: fails FDR
        sig = ist.significant_voxels(p, p_voxel=0.001, fdr_q=0.05)
        assert sig[0, 0, 0] and not sig[1, 1, 1]
        assert ist.significant_voxels(p, p_voxel=0.01, fdr_q=0.6)[1, 1, 1]

    def test_nan_voxels_are_not_tested_or_counted(self):
        p = np.full((4, 4, 4), np.nan)
        p[0, 0, 0], p[1, 1, 1] = 1e-6, 0.4
        sig = ist.significant_voxels(p)
        assert sig[0, 0, 0] and sig.sum() == 1       # with only two tested voxels, the tiny p survives FDR

    def test_extent_threshold_drops_small_clusters(self):
        p = np.full((8, 8, 8), 0.9)
        p[1:3, 1:3, 1:3] = 1e-9
        p[6, 6, 6] = 1e-9
        assert ist.significant_voxels(p)[6, 6, 6]
        sig = ist.significant_voxels(p, min_cluster=5)
        assert sig[1:3, 1:3, 1:3].all() and not sig[6, 6, 6]

    def test_each_class_is_its_own_fdr_family(self):
        p = np.full((6, 6, 6, 2), 0.5)
        p[0, 0, 0, 0] = 1e-9
        sig = ist.significant_voxels(p)
        assert sig[0, 0, 0, 0] and not sig[..., 1].any()


class TestPackingAndFiles:
    def test_pack_and_split_round_trip(self):
        rng = np.random.default_rng(0)
        shape = (3, 3, 3, 4)
        mean_all, mean_sel, p = rng.normal(size=shape), rng.normal(size=shape), rng.random(shape)
        p[0, 0, 0, :] = np.nan
        sig = rng.random(shape) > 0.5
        n_sel = rng.integers(0, 60, size=shape[:3])
        out = ist.split_stats(ist.pack_stats(mean_all, mean_sel, p, sig, n_sel))
        np.testing.assert_allclose(out[0], mean_all, rtol=1e-5)
        np.testing.assert_allclose(out[1], mean_sel, rtol=1e-5)
        assert (out[2][0, 0, 0] == 1.0).all()                 # untested voxels are stored as p = 1
        np.testing.assert_array_equal(out[3], sig)
        np.testing.assert_array_equal(out[4], n_sel)

    def test_save_stats_writes_nifti_and_json(self, tmp_path):
        import json
        import nibabel as nib
        maps, _ = make_maps(n_subjects=20)
        r = ist.sign_flip_test(maps, n_perm=500, seed=0, min_subjects=10)
        path, rows = ist.save_stats(str(tmp_path / "demo"), np.eye(4), r, ["a", "b", "c", "d"],
                                    dict(n_subjects=20, n_perm=500, seed=0, min_subjects=10))
        stack = nib.load(path).get_fdata()
        assert stack.shape == (6, 6, 6, 17)
        meta = json.load(open(tmp_path / "demo_stats.json"))
        assert meta["classes"] == ["a", "b", "c", "d"] and meta["n_perm"] == 500 and meta["n_selected_volume"] == 16
        assert [r_["cls"] for r_ in rows] == ["a", "b", "c", "d"]

    def test_load_subject_maps_stacks_and_checks_shape(self, tmp_path):
        import nibabel as nib
        paths = []
        for i in range(3):
            path = tmp_path / f"s{i}.nii.gz"
            nib.save(nib.Nifti1Image(np.full((4, 4, 4, 2), i, dtype=np.float32), np.eye(4)), path)
            paths.append(str(path))
        stack, affine = ist.load_subject_maps(paths)
        assert stack.shape == (3, 4, 4, 4, 2) and stack[2].mean() == 2 and affine.shape == (4, 4)
        bad = tmp_path / "bad.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.float32), np.eye(4)), bad)
        with pytest.raises(ValueError, match="shape"):
            ist.load_subject_maps(paths + [str(bad)])
