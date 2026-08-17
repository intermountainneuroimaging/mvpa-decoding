#!/bin/bash
#
#SBATCH --job-name=mvpa_kfold_workflow
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=04:00:00
#SBATCH --array=1-90
#SBATCH --output=logs/mvpa_kfold_workflow_%A_%a.out
#SBATCH --error=logs/mvpa_kfold_workflow_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Per-subject k-fold classifier array job (mvpa_kfold_workflow.py, not
# mvpa_generalization_workflow.py -- same-task data, split into folds by
# run). Expects master_spreadsheet.csv to already exist -- generate it first
# (2_sbatch_generate_master_spreadsheet.sh / 0_submit_mvpa_pipeline.sh).
# `logs/` must also already exist (sbatch does not create --output's parent
# dir). Also writes each subject's own single-subject report right after
# their classifier run finishes (workflows/generate_report.py --subject).
#
# --time is a rough starting estimate, not a measured one: a k-fold run
# with model.kfold_cv.strategy="per_run" and model.permutation_test both
# set pays for every fold *and* every fold's own 1000-permutation test --
# substantially more than a single mvpa_generalization_workflow.py run.
# Check actual wall time from the first array task's log and adjust
# --time/--array count before submitting the rest at scale.
#
# --array count must match your subject count -- check with:
#   ls -d $DATAROOT/sub-* | wc -l
#
# Stage 3 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Can also be run standalone -- but,
# like the orchestrator, must be submitted from the repo root (`sbatch
# slurm/3_batch_run_mvpa_workflow.sh`), not from within slurm/, since its own
# paths (workflows/, configs/, logs/) are all repo-root-relative.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

DATAROOT=/pl/active/banich/studies/Clearvale/analysis/bids-hcp/
MASTER_SPREADSHEET=master_spreadsheet.csv
OUTPUT_DIR=out
CONFIG_DIR=configs

# get subject for this array task
subject=`ls -d $DATAROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`

# --------------------------------------------
# gm operation (maintain/suppress/switch/clear) k-fold classifier
# --------------------------------------------
python workflows/mvpa_kfold_workflow.py --subject $subject --config $CONFIG_DIR/config-kfold.clearvale-operation.json \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR
    
python workflows/generate_report.py --analysis-output-dir $OUTPUT_DIR --config $CONFIG_DIR/config-kfold.clearvale-operation.json --subject $subject
