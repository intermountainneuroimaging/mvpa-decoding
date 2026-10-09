#!/bin/bash
#
#SBATCH --job-name=mvpa_restrict_trials
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=1:00:00
#SBATCH --output=logs/mvpa_restrict_trials_%j.out
#SBATCH --error=logs/mvpa_restrict_trials_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# OPTIONAL single (non-array) job between stages 2 and 3: builds the balanced
# training-trial list (workflows/restrict_master_spreadsheet.py, docs/trial-restriction.md) for the whole dataset from master_spreadsheet.csv. Skip it and
# the pipeline is unchanged. It is not part of 0_submit_mvpa_pipeline.sh --
# run it yourself after stage 2 and before stage 3:
#
#   sbatch slurm/2b_sbatch_restrict_master_spreadsheet.sh configs/my-study.json configs/restriction.json
#
# Arguments:
#   $1  the mvpa config -- pipeline.scripts_dir / pipeline.master_spreadsheet
#       are read from it, and so is event_extraction.hemodynamic_lag (the TR
#       counts depend on it), plus model_conditions.training, which is checked
#       against the restriction groups
#   $2  the supplementary restriction config (primary_groups / secondary_groups)
#
# Writes next to pipeline.master_spreadsheet (stem = its name without .csv):
#   <stem>_trial_restriction.csv          -- the list stage 3 reads
#   <stem>_trial_restriction_summary.csv  -- trials / TRs per subject and group, excluded subjects
#   <stem>_trial_restriction_audit.csv    -- every governed trial and whether it was kept
# Then add the first file's path to the config stage 3 reads:
#   "model_conditions": {"training": {..., "trial_restriction": "<that path>"}}
# Read the summary first: a subject with no balanced set is excluded, and its
# stage 3 task will refuse to run. Read the log for "(!)" coverage warnings.
#
# Rerun whenever hemodynamic_lag, the restriction groups, the events files or
# the master spreadsheet change (stage 3 also checks the lag and trial onsets).
# Fully standalone, like the other stages: `$1` is the only config it needs to
# find the repo, via pipeline.scripts_dir (see resolve_pipeline_config.sh).

set -e
umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

CONFIG_FILE="$1"
RESTRICTION_CONFIG="$2"
if [ -z "$CONFIG_FILE" ] || [ -z "$RESTRICTION_CONFIG" ]; then
    echo "Usage: sbatch $(basename "$0") <config.json> <restriction.json>" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi
if [ ! -f "$RESTRICTION_CONFIG" ]; then
    echo "Restriction config file not found: $RESTRICTION_CONFIG" >&2
    exit 1
fi
export CONFIG_FILE

# pipeline.scripts_dir, read directly rather than relying on an exported
# SCRIPTS_DIR or the job's working directory -- sbatch copies this script
# into its own spool file before running it, so neither is reliable here
# (see resolve_pipeline_config.sh's own header comment for the full reasoning)
SCRIPTS_DIR=$(python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    print(json.load(f).get('pipeline', {}).get('scripts_dir', ''))
")
if [ -z "$SCRIPTS_DIR" ]; then
    echo "$(basename "$0"): $CONFIG_FILE's \"pipeline\" section is missing required field: scripts_dir" >&2
    exit 1
fi
export SCRIPTS_DIR

# reads the master spreadsheet stage 2 wrote and writes beside it
export REQUIRED_PIPELINE_FIELDS="master_spreadsheet"
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"

python "$SCRIPTS_DIR/workflows/restrict_master_spreadsheet.py" --config $CONFIG_FILE \
    --master-spreadsheet $MASTER_SPREADSHEET --restriction-config $RESTRICTION_CONFIG
