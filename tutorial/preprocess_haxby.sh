#!/usr/bin/env bash
#
# Minimal FSL-based preprocessing for the Haxby tutorial:
#   Pass 1 (per run): motion correction (mcflirt, every volume -> this run's own
#     mean volume), coregistration (flirt 6 dof run-mean -> run 1's mean -- the
#     common template -- applied to the whole run via applyxfm4D).
#   Mask: one whole-brain mask (bet) from the average of every run's
#     coregistered (pre-detrend) mean volume -- needs every run's coreg output
#     from pass 1 first, since it's computed post-coregistration/pre-detrend.
#   Pass 2 (per run): linear detrending (fsl_glm, design = [intercept, centered
#     linear ramp], --out_res gives the detrended residual series -- mean ~0,
#     including negative values), then a constant offset (+10000) added back
#     *only inside the mask*, so in-brain voxels end up strictly positive again
#     (detrended residuals are tiny relative to 10000) while out-of-mask voxels
#     are left at their ~0 detrended value.
# See tutorial/README.md for what this deliberately skips (slice-timing,
# affine/nonlinear normalization, confound regression, ...).
#
# Usage:
#   export FSLDIR=/path/to/fsl   # if not already set
#   bash tutorial/preprocess_haxby.sh
#
# Tested against FSL 6.0.7.

set -euo pipefail

: "${FSLDIR:?set FSLDIR to your FSL install, e.g. export FSLDIR=/Users/you/fsl}"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"

SRC_DIR="tutorial/haxby-data/sub-1/func"
OUT_DIR="tutorial/haxby-data/derivatives/sub-1/func"
MASK_DIR="tutorial/haxby-data/derivatives/sub-1/masks"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$OUT_DIR" "$MASK_DIR"

OFFSET=10000

runs=("$SRC_DIR"/*_bold.nii.gz)
template_mean=""
mean_files=()
bases=()

# --- Pass 1: motion correction + coregistration (needed by every run before
# the mask can be computed) ---
for src in "${runs[@]}"; do
    base=$(basename "$src" _bold.nii.gz)
    bases+=("$base")
    echo "=== $base (motion correction + coregistration) ==="

    # 1) Motion correction: align every volume to this run's own mean volume
    mcflirt -in "$src" -out "$WORK_DIR/${base}_mcf" -meanvol -mats

    if [ -z "$template_mean" ]; then
        # run 1's motion-corrected mean volume is the common template -- nothing
        # to coregister it to (identity), so its "coreg" output is just its mcflirt output
        template_mean="$WORK_DIR/${base}_mcf_mean_reg.nii.gz"
        cp "$WORK_DIR/${base}_mcf.nii.gz" "$WORK_DIR/${base}_coreg.nii.gz"
    else
        # 2) Coregistration: rigid-body align this run's mean volume to the
        # template, then apply that single transform to every volume at once
        flirt -in "$WORK_DIR/${base}_mcf_mean_reg.nii.gz" -ref "$template_mean" \
            -dof 6 -omat "$WORK_DIR/${base}_to_template.mat"
        applyxfm4D "$WORK_DIR/${base}_mcf.nii.gz" "$template_mean" \
            "$WORK_DIR/${base}_coreg.nii.gz" "$WORK_DIR/${base}_to_template.mat" -singlematrix
    fi

    # for the mask, below -- averaged pre-detrend (detrending removes the mean
    # intensity contrast bet's segmentation relies on)
    fslmaths "$WORK_DIR/${base}_coreg.nii.gz" -Tmean "$WORK_DIR/${base}_mean"
    mean_files+=("$WORK_DIR/${base}_mean")
done

# --- Mask: average every run's coregistered (pre-detrend) mean volume -- now
# all on the same grid, post-coregistration -- then skull-strip with bet.
# Computed here (between the two passes) since pass 2's offset step needs it.
fslmerge -t "$WORK_DIR/all_means" "${mean_files[@]}"
fslmaths "$WORK_DIR/all_means" -Tmean "$WORK_DIR/grand_mean"
bet "$WORK_DIR/grand_mean" "$MASK_DIR/native_epi" -m -n
MASK="$MASK_DIR/native_epi_mask.nii.gz"
echo "wrote $MASK"

# in-mask-only offset image (10000 inside the mask, 0 outside) -- broadcasts
# across every run's 4D series when added, via fslmaths' 3D-into-4D broadcast
fslmaths "$MASK" -mul "$OFFSET" "$WORK_DIR/mask_offset"

# --- Pass 2: linear detrending + in-mask offset ---
for base in "${bases[@]}"; do
    echo "=== $base (detrending + offset) ==="

    # 3) Linear detrending: design = [intercept, centered linear ramp]; the GLM
    # residual (--out_res) is exactly the per-voxel linear-trend-removed series
    n_vols=$(fslnvols "$WORK_DIR/${base}_coreg.nii.gz")
    design="$WORK_DIR/${base}_design.mat"
    python3 - "$n_vols" "$design" <<'PYEOF'
import sys
n = int(sys.argv[1])
path = sys.argv[2]
ramp = [t - (n - 1) / 2 for t in range(n)]
with open(path, "w") as f:
    f.write(f"/NumWaves 2\n/NumPoints {n}\n/Matrix\n")
    for t in range(n):
        f.write(f"1 {ramp[t]:.6f}\n")
PYEOF

    fsl_glm -i "$WORK_DIR/${base}_coreg.nii.gz" -d "$design" \
        -o "$WORK_DIR/${base}_betas" --out_res="$WORK_DIR/${base}_detrended.nii.gz"

    # 4) Add the in-mask-only offset back -- keeps in-brain voxels strictly
    # positive (detrended residuals are tiny relative to 10000) without
    # touching out-of-mask voxels (offset image is 0 there)
    out_path="$OUT_DIR/${base}_desc-preproc_bold.nii.gz"
    fslmaths "$WORK_DIR/${base}_detrended.nii.gz" -add "$WORK_DIR/mask_offset" "$out_path"
    echo "wrote $out_path"
done
