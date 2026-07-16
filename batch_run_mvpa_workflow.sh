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

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

DATAROOT=/pl/active/banich/studies/mindmem/analysis/bids-hcp/

# get feat folder for analysis
subject=`ls -d $DATAROOT/sub-* | rev | cut -d"/" -f1 | rev | cut -d"-" -f2 | sed -n "$SLURM_ARRAY_TASK_ID p"`


# --------------------------------------------
# vvs category classifier
# --------------------------------------------
MASKFILE=native_vvs_transformed_mask.nii.gz
RESULTS_FOLDER=vvs_object_classifier
python mvpa_workflow.py --subject $subject --data-path $DATAROOT --mask $MASKFILE --model-descr $RESULTS_FOLDER


# --------------------------------------------
# gm category classifier
# --------------------------------------------
MASKFILE=native_gm_transformed_mask.nii.gz
RESULTS_FOLDER=gm_object_classifier
python mvpa_workflow.py --subject $subject --data-path $DATAROOT --mask $MASKFILE --model-descr $RESULTS_FOLDER

# --------------------------------------------
# gm valence classifier
# --------------------------------------------
MASKFILE=native_gm_transformed_mask.nii.gz
RESULTS_FOLDER=gm_valence_classifier
python mvpa_workflow.py --subject $subject --data-path $DATAROOT --mask $MASKFILE --model-descr $RESULTS_FOLDER