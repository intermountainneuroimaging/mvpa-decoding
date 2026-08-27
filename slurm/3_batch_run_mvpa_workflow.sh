#!/bin/bash
#
#SBATCH --job-name=mvpa_workflow
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=04:00:00
#SBATCH --array=1-90
#SBATCH --output=logs/mvpa_workflow_%A_%a.out
#SBATCH --error=logs/mvpa_workflow_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Per-subject classifier array job (mvpa_workflow.py) -- runs whichever of
# model.kfold_cv / model_conditions.testing / model_conditions.timecourse_decoding
# are configured, each independently. Expects master_spreadsheet.csv to
# already exist -- generate it first (2_sbatch_generate_master_spreadsheet.sh
# / 0_submit_mvpa_pipeline.sh). Also writes each subject's own single-subject
# report right after their classifier run finishes
# (workflows/generate_report.py --subject).
#
# --time is a rough starting estimate, not a measured one: a k-fold run
# with model.kfold_cv.strategy="per_run" and model.permutation_test both
# set pays for every fold *and* every fold's own 1000-permutation test --
# substantially more than a training+testing-only run. Check actual wall
# time from the first array task's log and adjust --time/--array count
# before submitting the rest at scale.
#
# --array count must match your subject count -- check with:
#   ls -d $BIDS_HCP_ROOT/sub-* | wc -l
#
# Stage 3 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Fully standalone -- no dependency on
# 0_ having run first or exported anything: submit directly with
# `sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json` from
# anywhere. The config path (`$1`) is the only required input -- passed to
# every array task the same way (SLURM_ARRAY_TASK_ID is a separate env var,
# not a positional arg); its pipeline.scripts_dir is how this script
# locates the rest of the repo (see resolve_pipeline_config.sh) --
# --output/--error log paths above are still plain SBATCH directives (no
# variable substitution happens in them), so they resolve relative to
# wherever `sbatch` was invoked from regardless.

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

# this stage runs the classifier + per-subject report -- it needs the
# subject list (bids_hcp_root) and where to write/read results, not the
# HCP derivatives root or group mask (those are stage 1/4-only)
export REQUIRED_PIPELINE_FIELDS="bids_hcp_root output_dir master_spreadsheet"
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"

# get subject for this array task
subject=`ls -d $BIDS_HCP_ROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`

# --------------------------------------------
# gm operation (maintain/suppress/switch/clear) classifier
# --------------------------------------------
python "$SCRIPTS_DIR/workflows/mvpa_workflow.py" --subject $subject --config $CONFIG_FILE \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR

python "$SCRIPTS_DIR/workflows/generate_report.py" --analysis-output-dir $OUTPUT_DIR --config $CONFIG_FILE \
    --master-spreadsheet $MASTER_SPREADSHEET --subject $subject
