#!/bin/bash
#
# Shared variables for the Clearvale k-fold MVPA SLURM pipeline (slurm/0_-4_).
# Edit values here once instead of hunting through every stage script.
# Sourced by each of 1_-4_ (after their own module loads, since MNI_TEMPLATE
# depends on $FSLDIR) via `source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"`
# -- not run directly, has no #SBATCH directives or steps of its own.
#
# SCRIPTS_DIR is the repo root -- every workflows/utils/config/output path
# below is built from it explicitly, instead of assuming a job's working
# directory happens to be the repo root. 0_submit_mvpa_pipeline.sh resolves
# its own real location and exports SCRIPTS_DIR before submitting each job,
# so it's already correctly set here when launched through the orchestrator.
# Falls back to the job's own working directory ($(pwd)) otherwise -- i.e.
# for a stage script submitted standalone, which still needs to be
# submitted from the repo root (or with --chdir=/path/to/repo) for this
# fallback to resolve correctly, same as before.
SCRIPTS_DIR="${SCRIPTS_DIR:-$(pwd)}"

# --- locations under the repo ---
CONFIG_DIR="$SCRIPTS_DIR/configs"
CONFIG_FILE="$CONFIG_DIR/config-kfold.clearvale-operation.json"
OUTPUT_DIR="$SCRIPTS_DIR/out"
MASTER_SPREADSHEET="$SCRIPTS_DIR/master_spreadsheet.csv"

# --- Clearvale study paths (cluster-absolute, not repo-relative) ---
BIDS_HCP_ROOT=/pl/active/banich/studies/Clearvale/analysis/bids-hcp
HCPPIPE_ROOT=/pl/active/banich/studies/Clearvale/analysis/HCPPipe
GROUP_GM_MASK=/pl/active/banich/studies/Clearvale/analysis/feat/group-analyses/bin_wager_gm_mask.nii.gz
MNI_TEMPLATE=$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz
