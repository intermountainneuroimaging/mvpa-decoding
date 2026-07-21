#!/bin/bash
#
#SBATCH --job-name=mvpa_master_spreadsheet
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --account=ucb278_asc4
#SBATCH --time=1:00:00
#SBATCH --output=logs/mvpa_master_spreadsheet_%j.out
#SBATCH --error=logs/mvpa_master_spreadsheet_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# Single (non-array) job: builds master_spreadsheet.csv once for the whole
# dataset. Submitted on its own via submit_mvpa_pipeline.sh, which then chains
# batch_run_mvpa_workflow.sh (the per-subject array job) after this completes.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

CONFIG_DIR=configs
MASTER_SPREADSHEET=master_spreadsheet.csv

# Any one of the classifier configs works here -- their event_extraction sections
# are identical (same bids_root/derivatives_root/subjects/tasks); only
# model_conditions/model differ between vvs/gm object/gm valence.
python generate_master_spreadsheet.py --config $CONFIG_DIR/gm_object_classifier.json \
    --output $MASTER_SPREADSHEET
