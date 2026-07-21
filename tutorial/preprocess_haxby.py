#!/usr/bin/env python3
"""
Minimal preprocessing for the Haxby tutorial: rigid-body motion correction
(each volume -> its own run's first volume), rigid coregistration to a
common template (run 1's first volume), and linear detrending. See
tutorial/README.md for what this deliberately skips (slice-timing, affine/
nonlinear normalization, confound regression, ...).

Usage:
    python tutorial/preprocess_haxby.py
"""

import glob
import os

import nibabel as nib
import numpy as np
from scipy.signal import detrend
from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
from dipy.align.transforms import RigidTransform3D
from nilearn.masking import compute_multi_epi_mask

SRC_GLOB = "tutorial/haxby-data/sub-1/func/*_bold.nii.gz"
OUT_DIR = "tutorial/haxby-data/derivatives/sub-1/func"
MASK_DIR = "tutorial/haxby-data/derivatives/sub-1/masks"

# Deliberately cheap settings (few iterations, coarse sampling) -- enough to
# correct the small motion typical of this dataset, not a rigorous estimate.
AFFREG = AffineRegistration(
    metric=MutualInformationMetric(nbins=16, sampling_proportion=0.15),
    level_iters=[8], sigmas=[0.0], factors=[2],
)


def rigid_transform(static, moving, affine):
    return AFFREG.optimize(static, moving, RigidTransform3D(), None,
                            static_grid2world=affine, moving_grid2world=affine)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    files = sorted(glob.glob(SRC_GLOB))
    runs = [nib.load(f) for f in files]
    affine = runs[0].affine
    template = runs[0].get_fdata()[..., 0]  # run 1, volume 0 = the common reference

    mean_imgs = []  # pre-detrend temporal means, for mask computation below
    for path, img in zip(files, runs):
        data = img.get_fdata()
        n = data.shape[-1]

        # motion correction: align every volume to this run's own first volume
        corrected = np.stack(
            [rigid_transform(data[..., 0], data[..., t], affine).transform(data[..., t]) for t in range(n)],
            axis=-1,
        )
        # coregistration: one rigid transform from this run's reference to the template, applied to all volumes
        to_template = rigid_transform(template, corrected[..., 0], affine)
        corrected = np.stack([to_template.transform(corrected[..., t]) for t in range(n)], axis=-1)
        mean_imgs.append(nib.Nifti1Image(corrected.mean(axis=-1).astype(np.float32), affine))

        # linear detrending (per voxel, along time) -- for analysis, not for mask computation
        # (detrending removes the intensity contrast compute_epi_mask relies on)
        detrended = detrend(corrected, axis=-1, type="linear")

        # pass the original header through -- a fresh default header would lose the real TR
        out_path = os.path.join(OUT_DIR, os.path.basename(path).replace("_bold.nii.gz", "_desc-preproc_bold.nii.gz"))
        nib.Nifti1Image(detrended.astype(np.float32), affine, img.header).to_filename(out_path)
        print(f"wrote {out_path}")

    mask_path = os.path.join(MASK_DIR, "native_epi_mask.nii.gz")
    compute_multi_epi_mask(mean_imgs).to_filename(mask_path)
    print(f"wrote {mask_path}")


if __name__ == "__main__":
    main()
