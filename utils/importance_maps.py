"""Importance maps that say HOW a voxel contributes (BOLD activation evidence vs BOLD deactivation evidence) and
the group-level top-fraction maps (the Kim et al. 2020 / McDuff et al. 2009 approach).

A classifier weight alone does not say whether a voxel helps identify a class by being MORE active or LESS
active than average (a weight can be positive where the class's activity is below average). The importance
value of a voxel is its weight x the class's mean z-scored BOLD activity there. It is positive (the voxel
supports the class) in two cases, and the two are kept apart:

  * BOLD activation evidence    -- weight > 0 and mean BOLD activity > 0: the voxel supports the class when its
                                   activity is HIGHER than average;
  * BOLD deactivation evidence  -- weight < 0 and mean BOLD activity < 0: the voxel supports the class when its
                                   activity is LOWER than average (value = weight x activity, a positive magnitude).

The remaining voxels are not shown: weight and activity of opposite sign give a NEGATIVE importance value (the
voxel's activity argues against the class), and voxels the subject's feature selection left out (weight 0) have
none. Both are set to 0, as in the paper.

Per subject and class the activation and deactivation values are z-scored together as one distribution of
magnitudes (non-zero voxels), then separated again. The group maps are the mean of the subjects' z-scored
maps; the top `top_fraction` (5% in the paper) of the non-zero group values are kept for the activation map
and for the deactivation map, clusters smaller than `min_cluster` voxels (10) are removed, and the result is
smoothed with a Gaussian of `fwhm_mm` (12 mm) for display.

The workflow saves each subject's per-class mean activity next to the weights (class_mean_activity,
`<subject>_impa_act[_mni].nii.gz`); the importance values are formed here, from those two files.
"""
import numpy as np


def activity_tag(impa_tag: str) -> str:
    """Filename token of the class-mean-activity map that goes with an importance map tagged `impa_tag`
    ("impa" -> "impa_act", "impa_mni" -> "impa_act_mni")."""
    return impa_tag.replace("impa", "impa_act", 1)


def class_mean_activity(data, labels, classes) -> np.ndarray:
    """(n_classes, n_features): every voxel is z-scored across the training samples (mean 0, SD 1; a voxel
    with no variance stays 0), then averaged over the samples of each class, in `classes` order. Positive =
    that class is more active than the average training sample, negative = less active."""
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels)
    sd = data.std(axis=0)
    z = np.where(sd > 0, (data - data.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    return np.vstack([z[labels == cls].mean(axis=0) if (labels == cls).any() else np.zeros(data.shape[1])
                      for cls in classes])


def polarity_importance(weights, activity):
    """(activation, deactivation), both >= 0 and shaped like `weights`: weight x activity where both are positive
    (BOLD activation evidence) or both negative (BOLD deactivation evidence); 0 elsewhere (opposite signs and
    unselected voxels)."""
    weights, activity = np.asarray(weights, dtype=np.float64), np.asarray(activity, dtype=np.float64)
    product = weights * activity
    activation = np.where((weights > 0) & (activity > 0), product, 0.0)
    deactivation = np.where((weights < 0) & (activity < 0), product, 0.0)
    return activation, deactivation


def zscore_importance(activation, deactivation):
    """Z-score the non-zero values of an activation-evidence and a deactivation-evidence map together (one
    distribution of magnitudes, as in the paper) and return the two maps again; zeros (ignored voxels) stay 0.
    Arrays with a class axis last are handled one class at a time. A map with fewer than 2 non-zero values, or no
    variance, comes back all zero."""
    activation, deactivation = np.asarray(activation, dtype=np.float64), np.asarray(deactivation, dtype=np.float64)
    if activation.ndim == 4:
        pairs = [zscore_importance(activation[..., i], deactivation[..., i]) for i in range(activation.shape[-1])]
        return np.stack([a for a, _ in pairs], axis=-1), np.stack([b for _, b in pairs], axis=-1)
    values = np.concatenate([activation[activation != 0], deactivation[deactivation != 0]])
    if values.size < 2 or values.std() == 0:
        return np.zeros_like(activation), np.zeros_like(deactivation)
    mu, sd = values.mean(), values.std()
    return (np.where(activation != 0, (activation - mu) / sd, 0.0), np.where(deactivation != 0, (deactivation - mu) / sd, 0.0))


def top_fraction_clusters(group_map, top_fraction: float = 0.05, min_cluster: int = 10):
    """Boolean mask of the voxels of `group_map` in the top `top_fraction` of its positive values, with
    26-connected clusters smaller than `min_cluster` voxels removed. Returns (mask, threshold)."""
    from scipy import ndimage
    group_map = np.asarray(group_map, dtype=np.float64)
    positive = group_map[group_map > 0]
    if positive.size == 0:
        return np.zeros(group_map.shape, dtype=bool), np.nan
    threshold = np.percentile(positive, 100 * (1 - top_fraction))
    mask = group_map >= threshold
    if min_cluster > 1 and mask.any():
        labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
        mask &= ~np.isin(labels, 1 + np.flatnonzero(sizes < min_cluster))
    return mask, threshold


def smooth_fwhm(volume, fwhm_mm: float, voxel_mm: float):
    """Gaussian smoothing of a 3-D volume with the given FWHM (mm) on a grid of `voxel_mm` isotropic voxels."""
    from scipy import ndimage
    if not fwhm_mm:
        return np.asarray(volume, dtype=np.float64)
    sigma = fwhm_mm / (2 * np.sqrt(2 * np.log(2))) / voxel_mm
    return ndimage.gaussian_filter(np.asarray(volume, dtype=np.float64), sigma=sigma)


def group_importance_maps(weight_activity_pairs, top_fraction: float = 0.05, min_cluster: int = 10,
                          fwhm_mm: float = 12.0, voxel_mm: float = 3.0):
    """Group-level BOLD activation / deactivation evidence maps from an iterable of (weights, activity) arrays
    (one pair per subject, each shaped (X, Y, Z, K)). Returns a dict with group_activation / group_deactivation
    (the mean z-scored map, K volumes), top_activation / top_deactivation (the top-fraction, cluster-filtered
    values), smooth_activation / smooth_deactivation (the latter, smoothed), thresholds (K, 2: activation,
    deactivation) and n_subjects. Streams over subjects."""
    total_act = total_deact = None
    n = 0
    for weights, activity in weight_activity_pairs:
        act, deact = polarity_importance(weights, activity)
        act, deact = zscore_importance(act, deact)
        total_act = act if total_act is None else total_act + act
        total_deact = deact if total_deact is None else total_deact + deact
        n += 1
    if n == 0:
        raise ValueError("no subjects")
    group_act, group_deact = total_act / n, total_deact / n
    k = group_act.shape[-1]
    out = {name: np.zeros_like(group_act) for name in ("top_activation", "top_deactivation", "smooth_activation", "smooth_deactivation")}
    thresholds = np.full((k, 2), np.nan)
    for i in range(k):
        for j, (group, key) in enumerate(((group_act, "activation"), (group_deact, "deactivation"))):
            mask, thr = top_fraction_clusters(group[..., i], top_fraction, min_cluster)
            kept = np.where(mask, group[..., i], 0.0)
            out[f"top_{key}"][..., i] = kept
            out[f"smooth_{key}"][..., i] = smooth_fwhm(kept, fwhm_mm, voxel_mm)
            thresholds[i, j] = thr
    out.update(group_activation=group_act, group_deactivation=group_deact, thresholds=thresholds, n_subjects=n)
    return out
