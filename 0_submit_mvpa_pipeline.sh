#!/bin/bash
#
# Submits the full Clearvale k-fold MVPA pipeline as four chained SLURM jobs,
# numbered to match the order they run in:
#   1_batch_resample_native_mask.sh              -- per-subject/session array job,
#     resamples the group GM mask into each session's native space
#   2_sbatch_generate_master_spreadsheet_kfold.sh -- single job, builds master_spreadsheet.csv
#   3_batch_run_mvpa_kfold_workflow.sh            -- per-subject array job, runs the
#     k-fold operation classifier
#   4_sbatch_generate_kfold_report.sh             -- single job, builds the group PDF report
# Each job is submitted with --dependency=afterok on the previous one, so a
# stage only starts if every task in the stage before it succeeded --
# `afterok` on an array job id waits for every task in that array, not just
# the first.
#
# Each stage script can also be run standalone (e.g. to rerun just one stage
# after fixing a subject-specific failure) -- this script only adds the
# chaining on top.
#
# Run this directly (not via sbatch): bash 0_submit_mvpa_pipeline.sh

set -eo pipefail

mkdir -p logs

resample_jobid=$(sbatch --parsable 1_batch_resample_native_mask.sh)
echo "Submitted mask-resample array job: $resample_jobid"

spreadsheet_jobid=$(sbatch --parsable --dependency=afterok:$resample_jobid 2_sbatch_generate_master_spreadsheet_kfold.sh)
echo "Submitted master_spreadsheet job: $spreadsheet_jobid (depends on $resample_jobid)"

kfold_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid 3_batch_run_mvpa_kfold_workflow.sh)
echo "Submitted k-fold classifier array job: $kfold_jobid (depends on $spreadsheet_jobid)"

report_jobid=$(sbatch --parsable --dependency=afterok:$kfold_jobid 4_sbatch_generate_kfold_report.sh)
echo "Submitted group report job: $report_jobid (depends on $kfold_jobid)"
