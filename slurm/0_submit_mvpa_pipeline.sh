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
# Study-specific variables (dataset paths, config filename, output dirs) are
# centralized in slurm/pipeline_vars.sh, sourced by each stage script --
# that's the one file to edit, not each of 1_-4_ individually. It's built
# around SCRIPTS_DIR, the repo root -- this script resolves its own real
# location (reliable here because it's invoked directly via `bash`, never
# through sbatch, which would otherwise obscure the original file path) and
# exports SCRIPTS_DIR before submitting each job, so 1_-4_ locate
# workflows/, utils/, configs/, etc. explicitly through it rather than
# depending on the job's working directory.
#
# Each stage script can also be run standalone (e.g. to rerun just one stage
# after fixing a subject-specific failure) -- this script only adds the
# chaining on top.
#
# Run from any directory:
#
#   bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh
#
# Standalone submission of an individual stage script doesn't inherit
# SCRIPTS_DIR this way -- slurm/pipeline_vars.sh falls back to the job's own
# working directory, so submit from the repo root (`sbatch
# slurm/1_batch_resample_native_mask.sh`) or export it yourself first
# (`export SCRIPTS_DIR=/path/to/mvpa_banich`). Either way, --output/--error
# log paths below are plain SBATCH directives (no variable substitution
# happens in them), so they always resolve relative to wherever `sbatch` was
# actually invoked from, regardless of SCRIPTS_DIR -- this script `cd`s to
# the repo root first so its own submissions land in the right place.

set -eo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPTS_DIR="$(dirname "$SLURM_DIR")"
cd "$SCRIPTS_DIR"

module use /curc/sw/modules/slurm
module load slurm/alpine

mkdir -p logs

resample_jobid=$(sbatch --parsable "$SCRIPTS_DIR/slurm/1_batch_resample_native_mask.sh")
echo "Submitted mask-resample array job: $resample_jobid"

spreadsheet_jobid=$(sbatch --parsable --dependency=afterok:$resample_jobid "$SCRIPTS_DIR/slurm/2_sbatch_generate_master_spreadsheet.sh")
echo "Submitted master_spreadsheet job: $spreadsheet_jobid (depends on $resample_jobid)"

kfold_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid "$SCRIPTS_DIR/slurm/3_batch_run_mvpa_workflow.sh")
echo "Submitted k-fold classifier array job: $kfold_jobid (depends on $spreadsheet_jobid)"

report_jobid=$(sbatch --parsable --dependency=afterok:$kfold_jobid "$SCRIPTS_DIR/slurm/4_sbatch_generate_report.sh")
echo "Submitted group report job: $report_jobid (depends on $kfold_jobid)"
