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
# k-fold classifier -> group report). Can also be run standalone -- submit
# from the repo root (`sbatch slurm/2_sbatch_generate_master_spreadsheet.sh
# configs/my-study.json`) so slurm/pipeline_vars.sh's SCRIPTS_DIR fallback
# and the --output/--error log paths above (plain SBATCH directives, not
# variable-substituted) both resolve correctly -- or export SCRIPTS_DIR and
# pass --chdir yourself. The config path is a required argument (`$1`) --
# see pipeline_vars.sh for the "pipeline" section it reads every other path
# from.

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

source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"

python "$SCRIPTS_DIR/workflows/generate_master_spreadsheet.py" --config $CONFIG_FILE \
    --output $MASTER_SPREADSHEET
