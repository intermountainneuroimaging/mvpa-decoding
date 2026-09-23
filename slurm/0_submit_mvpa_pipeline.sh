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
# The only required input is a single config JSON -- event_extraction/
# model_conditions/model as usual, plus a "pipeline" section (dataset paths,
# output dirs, MNI template) that every stage script reads via
# slurm/resolve_pipeline_config.sh instead of a separately-hand-edited bash file. This
# script requires that config path as its own argument and forwards it,
# unchanged, to every stage it submits -- so all four jobs agree on it by
# construction, not by four people remembering to edit the same file.
#
# An optional second argument restricts stage 4's group report to just the
# given subjects -- either a comma-separated list (e.g. "001,004,010") or a
# path to a text file listing them -- forwarded as-is to generate_report.py's
# --subjects, which detects which form it got. A relative file path is
# resolved against the directory this script was invoked from (before the
# `cd` below), so it works the same regardless of where sbatch's own working
# directory ends up. Stages 1-3 are unaffected and still process every
# subject they auto-discover under bids_hcp_root/hcppipe_root -- this only
# narrows which of those subjects' already-computed results stage 4
# aggregates into the group PDF.
#
# This script resolves its own real location (reliable here because it's
# invoked directly via `bash`, never through sbatch, which would otherwise
# obscure the original file path) purely for its own use -- `cd`ing to the
# repo root so its own submissions/logs land in the right place, and
# building the sbatch paths below. Each of 1_-4_ does *not* depend on this
# in any way: sbatch copies a batch script into its own spool file before
# running it, so a stage script's own location and this export would both
# be unreliable for it anyway -- instead, every stage script reads
# pipeline.scripts_dir out of the config itself (see
# resolve_pipeline_config.sh), making each one independently runnable with
# nothing but its own config-path argument, whether launched by this
# orchestrator or `sbatch`'d directly on its own.
#
# Each stage script can also be run standalone (e.g. to rerun just one stage
# after fixing a subject-specific failure) -- this script only adds the
# dependency chaining on top.
#
# Run from any directory:
#
#   bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh configs/my-study.json
#   bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh configs/my-study.json 1,2,3

set -eo pipefail

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: bash $0 <config.json> [subject1,subject2,...]" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# optional -- forwarded to stage 4 only (see header comment above). If it's
# a file, resolve it to an absolute path now, while still in the caller's
# own working directory -- the `cd` below changes directory before this
# reaches sbatch, which would otherwise break a relative file path.
SUBJECT_LIST="$2"
if [ -n "$SUBJECT_LIST" ] && [ -f "$SUBJECT_LIST" ]; then
    SUBJECT_LIST="$(cd "$(dirname "$SUBJECT_LIST")" && pwd)/$(basename "$SUBJECT_LIST")"
fi

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(dirname "$SLURM_DIR")"
cd "$SCRIPTS_DIR"

module use /curc/sw/modules/slurm
module load slurm/alpine

mkdir -p logs

resample_jobid=$(sbatch --parsable "$SCRIPTS_DIR/slurm/1_batch_resample_native_mask.sh" "$CONFIG_FILE")
echo "Submitted mask-resample array job: $resample_jobid"

spreadsheet_jobid=$(sbatch --parsable --dependency=afterok:$resample_jobid "$SCRIPTS_DIR/slurm/2_sbatch_generate_master_spreadsheet.sh" "$CONFIG_FILE")
echo "Submitted master_spreadsheet job: $spreadsheet_jobid (depends on $resample_jobid)"

kfold_jobid=$(sbatch --parsable --dependency=afterok:$spreadsheet_jobid "$SCRIPTS_DIR/slurm/3_batch_run_mvpa_workflow.sh" "$CONFIG_FILE")
echo "Submitted k-fold classifier array job: $kfold_jobid (depends on $spreadsheet_jobid)"

report_jobid=$(sbatch --parsable --dependency=afterok:$kfold_jobid "$SCRIPTS_DIR/slurm/4_sbatch_generate_report.sh" "$CONFIG_FILE" "$SUBJECT_LIST")
echo "Submitted group report job: $report_jobid (depends on $kfold_jobid)"
