#!/bin/bash
#
#SBATCH --job-name=mvpa_master_spreadsheet_kfold
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=1:00:00
#SBATCH --output=logs/mvpa_master_spreadsheet_kfold_%j.out
#SBATCH --error=logs/mvpa_master_spreadsheet_kfold_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Single (non-array) job: builds master_spreadsheet.csv once for the whole
# Clearvale dataset, from the k-fold operation classifier's own config (its
# event_extraction section is what matters here -- model_conditions/model are
# irrelevant to spreadsheet generation).
#
# Stage 2 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Fully standalone -- no dependency on
# 0_ having run first or exported anything: submit directly with
# `sbatch slurm/2_sbatch_generate_master_spreadsheet.sh configs/my-study.json`
# from anywhere. The config path (`$1`) is the only required input; its
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

source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"

python "$SCRIPTS_DIR/workflows/generate_master_spreadsheet.py" --config $CONFIG_FILE \
    --output $MASTER_SPREADSHEET
