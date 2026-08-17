#!/bin/bash
#
# Submits the full Clearvale k-fold MVPA pipeline as four chained SLURM jobs,
# numbered to match the order they run in:
#   1_batch_resample_native_mask.sh         -- per-subject/session array job,
#     resamples the group GM mask into each session's native space
#   2_sbatch_generate_master_spreadsheet.sh -- single job, builds master_spreadsheet.csv
#   3_batch_run_mvpa_workflow.sh            -- per-subject array job, runs the
#     k-fold operation classifier (+ each subject's own single-subject report)
#   4_sbatch_generate_report.sh             -- single job, resamples every
#     subject's importance map to MNI space, then builds the group PDF report
# Each job is submitted with --dependency=afterok on the previous one, so a
# stage only starts if every task in the stage before it succeeded --
# `afterok` on an array job id waits for every task in that array, not just
# the first.
#
# Each stage script can also be run standalone (e.g. to rerun just one stage
# after fixing a subject-specific failure) -- this script only adds the
# chaining on top.
#
# Every path here (the sibling stage scripts below, and inside each of them:
# workflows/, utils/, configs/, logs/, master_spreadsheet.csv) is relative to
# the repo root, not to slurm/ where this script lives -- sbatch runs a job
# in whatever directory it was submitted from, not the submitted script's own
# directory, so this must be run from the repo root, not from within slurm/:
#
#   bash slurm/0_submit_mvpa_pipeline.sh

module use /curc/sw/modules/slurm
module load slurm/alpine

set -eo pipefail

mkdir -p logs

resample_jobid=$(sbatch --parsable slurm/1_batch_resample_native_mask.sh)
echo "Submitted mask-resample array job: $resample_jobid"

spreadsheet_jobid=$(sbatch --parsable --dependency=afterok:$resample_jobid slurm/2_sbatch_generate_master_spreadsheet.sh)
echo "Submitted master_spreadsheet job: $spreadsheet_jobid (depends on $resample_jobid)"

kfold_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid slurm/3_batch_run_mvpa_workflow.sh)
echo "Submitted k-fold classifier array job: $kfold_jobid (depends on $spreadsheet_jobid)"

report_jobid=$(sbatch --parsable --dependency=afterok:$kfold_jobid slurm/4_sbatch_generate_report.sh)
echo "Submitted group report job: $report_jobid (depends on $kfold_jobid)"
