#!/bin/bash
#
#SBATCH --job-name=mvpa_kfold_report
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=00:30:00
#SBATCH --output=logs/mvpa_kfold_report_%j.out
#SBATCH --error=logs/mvpa_kfold_report_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Single (non-array) job: stage 4, the final stage of 0_submit_mvpa_pipeline.sh
# -- builds one group PDF report (report_<desc>.pdf) aggregating every subject
# the preceding 3_batch_run_mvpa_kfold_workflow.sh array job wrote output for.
# Expects that array job to have already completed for every subject
# (0_submit_mvpa_pipeline.sh's --dependency=afterok on the whole array job is
# what guarantees this -- afterok on an array job id waits for every task in
# it, not just the first).
#
# --time is a rough starting estimate (PDF/plot rendering across every
# subject's output, no measured runtime yet) -- check the first run's actual
# wall time and adjust before relying on it.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

MASTER_SPREADSHEET=master_spreadsheet.csv
OUTPUT_DIR=out
CONFIG_DIR=configs

python workflows/generate_report.py --analysis-output-dir $OUTPUT_DIR \
    --config $CONFIG_DIR/config-kfold.clearvale-operation.json --master-spreadsheet $MASTER_SPREADSHEET
