#!/bin/bash
#
#SBATCH --job-name=resample_native_mask
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=00:30:00
#SBATCH --array=1-69
#SBATCH --output=logs/resample_native_mask_%A_%a.out
#SBATCH --error=logs/resample_native_mask_%A_%a.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#
# Per-subject array job: resample the group-level MNI GM mask into each of
# that subject's sessions' native space via hcp_resample.py, writing
# native_gm_transformed_mask.nii.gz -- one mask per (subject, session), not
# per run, since every functional run within a session shares the same
# native grid (this is why model.mask.mask_pattern elsewhere in this repo
# is keyed on {subject}/{session} only). The first functional run found for
# each session is used as the --reference grid; which run it happens to be
# doesn't matter, since all of a session's runs share that same grid.
#
# `logs/` must already exist (sbatch does not create --output's parent dir).
# --array count must match the number of subjects under HCPPIPE_ROOT --
# check with: ls -d $HCPPIPE_ROOT/sub-* | wc -l
#
# Stage 1 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Can also be run standalone -- but,
# like the orchestrator, must be submitted from the repo root (`sbatch
# slurm/1_batch_resample_native_mask.sh`), not from within slurm/, since its
# own paths (workflows/, utils/, configs/, logs/) are all repo-root-relative.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

HCPPIPE_ROOT=/pl/active/banich/studies/Clearvale/analysis/HCPPipe
BIDS_HCP_ROOT=/pl/active/banich/studies/Clearvale/analysis/bids-hcp
GROUP_MASK=/pl/active/banich/studies/Clearvale/analysis/feat/group-analyses/bin_wager_gm_mask.nii.gz

# get subject for this array task (same pattern as batch_run_mvpa_workflow.sh)
subject=`ls -d $HCPPIPE_ROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`

if [ -z "$subject" ]; then
    echo "No subject found for array task $SLURM_ARRAY_TASK_ID under $HCPPIPE_ROOT -- exiting."
    exit 1
fi

echo "=== subject $subject ==="

for session_dir in "$HCPPIPE_ROOT"/sub-"$subject"/ses-*; do
    [ -d "$session_dir" ] || continue
    session=$(basename "$session_dir" | cut -d"-" -f2)

    # any one functional run's SBRef under this session -- every run shares
    # the same native grid, so which one is picked doesn't matter
    reference=$(find "$session_dir" -iname "*_bold_SBRef_nonlin.nii.gz" | sort | head -n 1)

    if [ -z "$reference" ]; then
        echo "  (!) sub-$subject ses-$session: no functional run SBRef found under $session_dir -- skipping"
        continue
    fi

    output_dir="$BIDS_HCP_ROOT/sub-$subject/ses-$session/func"
    mkdir -p "$output_dir"
    output="$output_dir/native_gm_transformed_mask.nii.gz"

    echo "  ses-$session: reference=$reference"
    echo "  ses-$session: output=$output"
    python utils/hcp_resample.py \
        --input "$GROUP_MASK" \
        --output "$output" \
        --direction mni2native \
        --subject "$subject" --session "$session" \
        --derivatives-root "$HCPPIPE_ROOT" \
        --reference "$reference" \
        --interp nn
done
