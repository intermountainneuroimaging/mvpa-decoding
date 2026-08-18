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
# (2_sbatch_generate_master_spreadsheet.sh / 0_submit_mvpa_pipeline.sh). Also
# writes each subject's own single-subject report right after their
# classifier run finishes (workflows/generate_report.py --subject).
#
# --time is a rough starting estimate, not a measured one: a k-fold run
# with model.kfold_cv.strategy="per_run" and model.permutation_test both
# set pays for every fold *and* every fold's own 1000-permutation test --
# substantially more than a single mvpa_generalization_workflow.py run.
# Check actual wall time from the first array task's log and adjust
# --time/--array count before submitting the rest at scale.
#
# --array count must match your subject count -- check with:
#   ls -d $BIDS_HCP_ROOT/sub-* | wc -l
#
# Stage 3 of 0_submit_mvpa_pipeline.sh (mask resample -> master spreadsheet ->
# k-fold classifier -> group report). Can also be run standalone -- submit
# from the repo root (`sbatch slurm/3_batch_run_mvpa_workflow.sh`) so
# slurm/pipeline_vars.sh's SCRIPTS_DIR fallback and the --output/--error log
# paths above (plain SBATCH directives, not variable-substituted) both
# resolve correctly -- or export SCRIPTS_DIR and pass --chdir yourself.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"

# get subject for this array task
subject=`ls -d $BIDS_HCP_ROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`

# --------------------------------------------
# gm operation (maintain/suppress/switch/clear) k-fold classifier
# --------------------------------------------
python "$SCRIPTS_DIR/workflows/mvpa_kfold_workflow.py" --subject $subject --config $CONFIG_FILE \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR

python "$SCRIPTS_DIR/workflows/generate_report.py" --analysis-output-dir $OUTPUT_DIR --config $CONFIG_FILE --subject $subject
