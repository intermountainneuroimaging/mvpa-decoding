#!/bin/bash
#
# Submits the full MVPA pipeline as two chained SLURM jobs:
#   1. sbatch_generate_master_spreadsheet.sh -- single job, builds master_spreadsheet.csv
#   2. batch_run_mvpa_workflow.sh -- per-subject array job, runs the 3 classifiers
# Job 2 is submitted with --dependency=afterok on job 1, so it only starts (and
# only if job 1) succeeds.
#
# Run this directly (not via sbatch): bash submit_mvpa_pipeline.sh

set -eo pipefail

mkdir -p logs

spreadsheet_jobid=$(sbatch --parsable sbatch_generate_master_spreadsheet.sh)
echo "Submitted master_spreadsheet job: $spreadsheet_jobid"

array_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid batch_run_mvpa_workflow.sh)
echo "Submitted classifier array job: $array_jobid (depends on $spreadsheet_jobid)"
