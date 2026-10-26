"""Group-level test of importance maps: which voxels carry a weight that differs from zero across subjects.

Why not a plain one-sample t-test on the subjects' maps: each subject's importance map is zero at every voxel
that subject's feature selection (ANOVA, p < feat_p, refit inside every fold) left out, and what is left differs
from subject to subject. A zero there means "not selected", not "weight of zero". Entering those zeros as data
shrinks the mean and inflates the variance of every voxel that only some subjects kept, and mixes "how often
the voxel was selected" into "what its weight was". So here:

  * a subject contributes to a voxel only if its map is non-zero there (a voxel with all four class weights zero
    is "not selected"); the others are missing, not zero;
  * a voxel is tested only if at least `min_subjects` subjects selected it (n_selected is saved with the result);
  * the statistic is the plain group mean of the weights of the subjects that selected the voxel, and its null
    distribution comes from random sign flipping: under the null that each subject's weight at the voxel is
    symmetric around zero, flipping the sign of a whole subject's map is as likely as not. Because the same
    flip is applied to all voxels of a subject, the spatial correlation between voxels is kept;
  * p is two-sided, p = (1 + #{|permuted mean| >= |observed mean|}) / (1 + n_perm), per voxel and class (the
    smallest possible p is 1 / (n_perm + 1)), then Benjamini-Hochberg FDR across the tested voxels of each class.
    Permutation p-values are coarse: with m tested voxels BH only passes voxel k if p_k <= k * q / m, so voxels
    sitting at the p floor need k >= m / (q * (n_perm + 1)) of their kind (about 120 for 60 000 voxels and
    10 000 flips at q = .05). Strong, widespread effects clear that easily; a few isolated voxels do not.

What this does and does not tell you: it is a test of sign consistency across the subjects who kept the
voxel (conditional on selection). Selection is by an omnibus ANOVA that does not know the sign of a class's
weight, so it does not favour one sign, but a voxel that few subjects kept is weak evidence however consistent
it is (hence min_subjects). The subject's own map is the mean of its folds' maps, so a voxel kept in only some of
a subject's folds has a weight scaled down by the fraction of folds that kept it; this does not change its sign.
"""
import json

import numpy as np

VOLUME_ORDER = ("mean_all", "mean_selected", "p_signflip", "significant")     # K volumes each, then n_selected (1)


def load_subject_maps(paths):
    """(n_subjects, X, Y, Z, K) float32 stack of the subjects' 4-D maps, plus the first map's affine."""
    import nibabel as nib
    stack, affine = None, None
    for i, path in enumerate(paths):
        img = nib.load(path)
        if stack is None:
            shape = img.shape if len(img.shape) == 4 else img.shape + (1,)
            stack = np.zeros((len(paths),) + tuple(shape), dtype=np.float32)
            affine = img.affine
        elif tuple(img.shape) != stack.shape[1:] and not (len(img.shape) == 3 and tuple(img.shape) + (1,) == stack.shape[1:]):
            raise ValueError(f"{path}: shape {img.shape} differs from the first map's {stack.shape[1:]}")
        data = np.asarray(img.dataobj, dtype=np.float32)
        stack[i] = data if data.ndim == 4 else data[..., None]
    return stack, affine


def sign_flip_test(maps, n_perm: int = 10000, seed: int = 0, min_subjects: int = 10, chunk: int = 100):
    """Voxelwise sign-flip permutation test of the group mean weight against zero, per class.

    `maps`: (n_subjects, X, Y, Z, K); a voxel a subject did not select is 0 in all K classes. Returns a dict:
    mean_all (X,Y,Z,K: sum / n_subjects, zeros included -- the map the report draws), mean_selected
    (X,Y,Z,K: mean over the subjects that selected the voxel), n_selected (X,Y,Z), tested (X,Y,Z bool:
    n_selected >= min_subjects) and p (X,Y,Z,K; NaN where not tested; absent when n_perm == 0)."""
    maps = np.asarray(maps, dtype=np.float32)
    n_sub = maps.shape[0]
    k = maps.shape[-1]
    if n_sub < 3:
        raise ValueError(f"need at least 3 subjects, got {n_sub}")
    selected = (maps != 0).any(axis=-1)                                  # (n_sub, X, Y, Z)
    n_selected = selected.sum(axis=0)
    tested = n_selected >= min_subjects
    total = maps.sum(axis=0, dtype=np.float64)                           # (X, Y, Z, K)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_selected = np.where(n_selected[..., None] > 0, total / np.maximum(n_selected, 1)[..., None], 0.0)
    result = dict(mean_all=total / n_sub, mean_selected=mean_selected, n_selected=n_selected, tested=tested)
    if n_perm <= 0 or not tested.any():
        return result

    x = maps[:, tested, :].reshape(n_sub, -1)                            # (n_sub, V*K), only tested voxels
    observed = np.abs(x.sum(axis=0, dtype=np.float64))
    floor = observed * (1 - 1e-5) - 1e-12                                # float32 sums: ties must count as >=
    exceed = np.zeros(x.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(m, n_sub))
        exceed += (np.abs(signs @ x) >= floor).sum(axis=0)
        done += m
    p_flat = (1.0 + exceed) / (1.0 + n_perm)
    p = np.full(maps.shape[1:], np.nan)
    p[tested, :] = p_flat.reshape(-1, k)
    result["p"] = p
    return result


def significant_voxels(p, p_voxel: float = 0.001, fdr_q: float = 0.05, min_cluster: int = 0):
    """Boolean array (shape of p): p < `p_voxel` (uncorrected, the cluster-forming threshold) AND significant
    after Benjamini-Hochberg FDR at `fdr_q`. The FDR family is the tested voxels (non-NaN p) of one class,
    each class its own family. `min_cluster` > 0 additionally drops (6-connected) clusters smaller than that
    many voxels, the extent threshold."""
    from scipy import ndimage, stats
    p = np.asarray(p, dtype=float)
    volumes = p[..., None] if p.ndim == 3 else p
    keep = np.zeros(volumes.shape, dtype=bool)
    for v in range(volumes.shape[-1]):
        pv = volumes[..., v]
        tested = ~np.isnan(pv)
        sig = np.zeros(pv.shape, dtype=bool)
        if tested.any():
            adjusted = stats.false_discovery_control(pv[tested], method="bh")
            sig[tested] = (adjusted < fdr_q) & (pv[tested] < p_voxel)
        if min_cluster and sig.any():
            labels, n_labels = ndimage.label(sig)
            sizes = ndimage.sum_labels(sig, labels, index=np.arange(1, n_labels + 1))
            sig &= ~np.isin(labels, 1 + np.flatnonzero(sizes < min_cluster))
        keep[..., v] = sig
    return keep[..., 0] if p.ndim == 3 else keep


def pack_stats(mean_all, mean_selected, p, significant, n_selected):
    """One float32 4-D array (X, Y, Z, 4K + 1): mean_all, mean_selected, p (1 where not tested), significant
    (0/1), then n_selected. split_stats() is the inverse."""
    p = np.where(np.isnan(p), 1.0, p)
    return np.concatenate([mean_all, mean_selected, p, significant.astype(float), n_selected[..., None]],
                          axis=-1).astype(np.float32)


def split_stats(stack):
    """(mean_all, mean_selected, p, significant (bool), n_selected) from a pack_stats() array."""
    k = (stack.shape[-1] - 1) // 4
    return (stack[..., :k], stack[..., k:2 * k], stack[..., 2 * k:3 * k], stack[..., 3 * k:4 * k] > 0.5, stack[..., 4 * k])


def save_stats(path_prefix, affine, result, classes, params, p_voxel=0.001, fdr_q=0.05, min_cluster=0):
    """Writes <prefix>_stats.nii.gz (pack_stats layout), <prefix>_stats.json (what is in it and the settings)
    and returns (path, per-class summary rows as a list of dicts)."""
    import nibabel as nib
    p = result["p"]
    sig = significant_voxels(p, p_voxel, fdr_q, min_cluster)
    stack = pack_stats(result["mean_all"], result["mean_selected"], p, sig, result["n_selected"])
    path = f"{path_prefix}_stats.nii.gz"
    nib.save(nib.Nifti1Image(stack, affine), path)
    k = len(classes)
    meta = dict(volumes={name: list(range(i * k, (i + 1) * k)) for i, name in enumerate(VOLUME_ORDER)},
                n_selected_volume=4 * k, classes=list(classes), p_voxel=p_voxel, fdr_q=fdr_q, min_cluster=min_cluster,
                **params)
    with open(f"{path_prefix}_stats.json", "w") as f:
        json.dump(meta, f, indent=2)
    rows = []
    for i, cls in enumerate(classes):
        s = sig[..., i]
        mean = result["mean_selected"][..., i]
        rows.append(dict(cls=cls, voxels_tested=int(result["tested"].sum()), voxels_significant=int(s.sum()),
                         significant_positive=int((s & (mean > 0)).sum()), significant_negative=int((s & (mean < 0)).sum())))
    return path, rows
