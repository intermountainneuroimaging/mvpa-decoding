#!/bin/bash
#
# Shared variables for the Clearvale k-fold MVPA SLURM pipeline (slurm/0_-4_).
# The only required input to the whole pipeline is CONFIG_FILE -- every
# study-specific path below (analysis output locations, HCP derivatives
# roots, the group GM mask, the MNI template) is read directly from that
# config's own "pipeline" section, so there's nothing else to edit here or
# anywhere else. Each stage script (1_-4_) requires a config path as its own
# first argument (`sbatch slurm/N_....sh <config.json>`), validates it, and
# exports CONFIG_FILE before sourcing this file -- see any of 1_-4_ for the
# exact validation block.
#
# Sourced via `source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"` -- not run
# directly, has no #SBATCH directives or steps of its own.
#
# SCRIPTS_DIR is the repo root -- every workflows/utils path below is built
# from it explicitly, instead of assuming a job's working directory happens
# to be the repo root. 0_submit_mvpa_pipeline.sh resolves its own real
# location and exports SCRIPTS_DIR before submitting each job, so it's
# already correctly set here when launched through the orchestrator. Falls
# back to the job's own working directory ($(pwd)) otherwise -- i.e. for a
# stage script submitted standalone, which still needs to be submitted from
# the repo root (or with --chdir=/path/to/repo) for this fallback to
# resolve correctly, same as before.
SCRIPTS_DIR="${SCRIPTS_DIR:-$(pwd)}"

if [ -z "$CONFIG_FILE" ]; then
    echo "pipeline_vars.sh: CONFIG_FILE is not set -- each stage script requires a config path as its first argument (e.g. \`sbatch slurm/2_sbatch_generate_master_spreadsheet.sh configs/my-study.json\`)." >&2
    exit 1
fi

# --- everything else comes from config.pipeline ---
#
# "pipeline": {
#   "bids_hcp_root": "/path/to/bids-hcp",
#   "hcppipe_root": "/path/to/HCPPipe",
#   "group_gm_mask": "/path/to/group_gm_mask.nii.gz",
#   "output_dir": "/path/to/mvpa-decoding-output",
#   "master_spreadsheet": "/path/to/mvpa-decoding-output/master_spreadsheet.csv",
#   "mni_template": "/path/to/MNI152_T1_2mm_brain.nii.gz"   (optional -- see below)
# }
#
# Each field is a full, already-resolved absolute path -- no {subject}/
# {session} placeholders (those are per-file, not per-root) and no implicit
# shared-prefix magic the way the old hardcoded ANALYSIS_ROOT bash variable
# provided; if you want that convenience, build these paths with a shared
# prefix in whatever generates your config, not here.
_pipeline_json=$(python3 - "$CONFIG_FILE" <<'PYEOF'
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path) as f:
    cfg = json.load(f)

pipeline = cfg.get("pipeline", {})
required = ["bids_hcp_root", "hcppipe_root", "group_gm_mask", "output_dir", "master_spreadsheet"]
missing = [k for k in required if k not in pipeline]
if missing:
    print(f"pipeline_vars.sh: {config_path}'s \"pipeline\" section is missing required "
          f"field(s): {missing}", file=sys.stderr)
    sys.exit(1)

for key in required:
    print(f"{key.upper()}={shlex.quote(str(pipeline[key]))}")

# optional -- falls back to $FSLDIR's own bundled template (set below,
# after this script's module loads) when absent from the config
if "mni_template" in pipeline:
    print(f"MNI_TEMPLATE={shlex.quote(str(pipeline['mni_template']))}")
PYEOF
) || exit 1
eval "$_pipeline_json"
unset _pipeline_json

# only set if the config didn't already provide one above
MNI_TEMPLATE="${MNI_TEMPLATE:-$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz}"
