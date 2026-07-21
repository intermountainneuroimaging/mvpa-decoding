#!/bin/bash
#
#SBATCH --job-name=mvpa_workflow
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb278_asc4
#SBATCH --time=6:00:00
#SBATCH --array=1-61
#SBATCH --output=logs/mvpa_workflow_%A_%a.out
#SBATCH --error=logs/mvpa_workflow_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Per-subject classifier array job. Expects master_spreadsheet.csv to already
# exist -- submit via submit_mvpa_pipeline.sh, which runs
# sbatch_generate_master_spreadsheet.sh first and chains this job after it with
# --dependency=afterok. `logs/` must also already exist (sbatch does not create
# --output's parent dir).

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

DATAROOT=/pl/active/banich/studies/mindmem/analysis/bids-hcp/
MASTER_SPREADSHEET=master_spreadsheet.csv
OUTPUT_DIR=out
CONFIG_DIR=configs

# get subject for this array task
subject=`ls -d $DATAROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`

# --------------------------------------------
# vvs category classifier
# --------------------------------------------
python mvpa_workflow.py --subject $subject --config $CONFIG_DIR/vvs_object_classifier.json \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR

# --------------------------------------------
# gm category classifier
# --------------------------------------------
python mvpa_workflow.py --subject $subject --config $CONFIG_DIR/gm_object_classifier.json \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR

# --------------------------------------------
# gm valence classifier
# --------------------------------------------
python mvpa_workflow.py --subject $subject --config $CONFIG_DIR/gm_valence_classifier.json \
    --master-spreadsheet $MASTER_SPREADSHEET --analysis-output-dir $OUTPUT_DIR