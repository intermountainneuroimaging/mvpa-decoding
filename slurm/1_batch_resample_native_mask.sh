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
# --array count must match the number of subjects under HCPPIPE_ROOT --
# check with: ls -d $HCPPIPE_ROOT/sub-* | wc -l
#
# Stage 1 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Fully standalone -- no dependency on
# 0_ having run first or exported anything: submit directly with
# `sbatch slurm/1_batch_resample_native_mask.sh configs/my-study.json` from
# anywhere. The config path (`$1`) is the only required input; its
# pipeline.scripts_dir is how this script locates the rest of the repo (see
# resolve_pipeline_config.sh) -- --output/--error log paths above are still
# plain SBATCH directives (no variable substitution happens in them), so
# they resolve relative to wherever `sbatch` was invoked from regardless.

set -e
umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch $(basename "$0") <config.json>" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
export CONFIG_FILE

# pipeline.scripts_dir, read directly rather than relying on an exported
# SCRIPTS_DIR or the job's working directory -- sbatch copies this script
# into its own spool file before running it, so neither is reliable here
# (see resolve_pipeline_config.sh's own header comment for the full reasoning)
SCRIPTS_DIR=$(python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    print(json.load(f).get('pipeline', {}).get('scripts_dir', ''))
")
if [ -z "$SCRIPTS_DIR" ]; then
    echo "$(basename "$0"): $CONFIG_FILE's \"pipeline\" section is missing required field: scripts_dir" >&2
    exit 1
fi
export SCRIPTS_DIR

# this stage builds the per-subject/session native mask, so it needs the
# study's HCP derivatives root, the group MNI mask to resample, and where
# to write the result -- everything else in "pipeline" is irrelevant here
export REQUIRED_PIPELINE_FIELDS="bids_hcp_root hcppipe_root group_gm_mask"
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"

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
    python "$SCRIPTS_DIR/utils/hcp_resample.py" \
        --input "$GROUP_GM_MASK" \
        --output "$output" \
        --direction mni2native \
        --subject "$subject" --session "$session" \
        --derivatives-root "$HCPPIPE_ROOT" \
        --reference "$reference" \
        --interp nn
done
