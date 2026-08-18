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
# from the repo root (`sbatch slurm/2_sbatch_generate_master_spreadsheet.sh`)
# so slurm/pipeline_vars.sh's SCRIPTS_DIR fallback and the --output/--error
# log paths above (plain SBATCH directives, not variable-substituted) both
# resolve correctly -- or export SCRIPTS_DIR and pass --chdir yourself.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"

python "$SCRIPTS_DIR/workflows/generate_master_spreadsheet.py" --config $CONFIG_FILE \
    --output $MASTER_SPREADSHEET
