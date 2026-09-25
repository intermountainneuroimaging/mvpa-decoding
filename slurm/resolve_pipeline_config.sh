#!/bin/bash
#
# Resolves every study-specific path the Clearvale k-fold MVPA SLURM
# pipeline (slurm/0_-4_) needs as plain shell variables -- the repo root
# itself, analysis output locations, HCP derivatives roots, the group GM
# mask, the MNI template -- by reading them directly from the config's own
# "pipeline" section. There is nothing to hand-edit in this file itself;
# it's a resolver, not a variables file -- the only required input to the
# whole pipeline is the config path each stage script takes as its own
# first argument (`sbatch slurm/N_....sh <config.json>`).
#
# Each of 1_-4_ is independently bootstrapped from just that config path --
# no dependency on an already-exported SCRIPTS_DIR or on the job's working
# directory happening to be the repo root, both of which sbatch makes
# unreliable: sbatch copies the batch script into its own spool file before
# running it, so a stage script's own ${BASH_SOURCE[0]} never points back
# at the real repo (unlike 0_submit_mvpa_pipeline.sh, invoked directly via
# `bash`, where that trick is reliable), and a stage script's working
# directory is just wherever `sbatch` happened to be invoked from unless
# --chdir is passed. Each of 1_-4_ therefore reads CONFIG_FILE's own
# pipeline.scripts_dir *first*, before it can even locate this file, via a
# small bootstrap block -- see any of 1_-4_ for the exact sequence:
#
#   CONFIG_FILE="$1"                          # validated first
#   SCRIPTS_DIR=$(python3 -c '...' "$CONFIG_FILE")   # read from config
#   source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"
#
# -- not run directly itself, has no #SBATCH directives or steps of its own.
#
# "pipeline": {
#   "scripts_dir": "/path/to/mvpa-decoding",
#   "bids_hcp_root": "/path/to/bids-hcp",
#   "hcppipe_root": "/path/to/HCPPipe",
#   "group_gm_mask": "/path/to/group_gm_mask.nii.gz",
#   "output_dir": "/path/to/mvpa-decoding-output",
#   "master_spreadsheet": "/path/to/mvpa-decoding-output/master_spreadsheet.csv",
#   "full_frame_master_spreadsheet": "/path/to/mvpa-decoding-output/master_spreadsheet_full.csv"  (optional --
#                                     only needed when a config's model_conditions.timecourse_decoding is
#                                     configured; must match that config's own event_extraction.full_frame_output_file)
#   "mni_template": "/path/to/MNI152_T1_2mm_brain.nii.gz"   (optional -- see below)
# }
#
# Each field is a full, already-resolved absolute path -- no {subject}/
# {session} placeholders (those are per-file, not per-root) and no implicit
# shared-prefix magic the way the old hardcoded ANALYSIS_ROOT bash variable
# provided; if you want that convenience, build these paths with a shared
# prefix in whatever generates your config, not here.
#
# Only bids_hcp_root/hcppipe_root/group_gm_mask/output_dir/master_spreadsheet
# are ever validated as "required" -- and only the subset the *calling*
# stage script actually uses, not all five unconditionally. Each of 1_-4_
# sets REQUIRED_PIPELINE_FIELDS (space-separated) to exactly the fields it
# reads before sourcing this file -- e.g. 1_batch_resample_native_mask.sh
# and 4_sbatch_generate_report.sh's own native->MNI resample step are both
# optional (skippable if your data's already in MNI space -- see
# model.mnispace in README.md section 5), so a config that never runs stage
# 1 and has no native-space importance maps to resample doesn't need to
# supply hcppipe_root/group_gm_mask at all. Any field not in the caller's
# required list is exported anyway if the config happens to set it, but
# otherwise resolves to an empty string rather than erroring -- so a stage
# script that conditionally uses an optional field (e.g. stage 4's resample
# step gated on HCPPIPE_ROOT being non-empty) can just check for that.
# scripts_dir is always required for every stage and is validated/exported
# by the caller itself before this file is even sourced (see the header
# comment above), so it's not part of this mechanism.

if [ -z "$CONFIG_FILE" ]; then
    echo "resolve_pipeline_config.sh: CONFIG_FILE is not set -- each stage script requires a config path as its first argument (e.g. \`sbatch slurm/2_sbatch_generate_master_spreadsheet.sh configs/my-study.json\`)." >&2
    exit 1
fi

_pipeline_json=$(python3 - "$CONFIG_FILE" "${REQUIRED_PIPELINE_FIELDS:-}" <<'PYEOF'
import json
import shlex
import sys

config_path, required_arg = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)

pipeline = cfg.get("pipeline", {})
optional_fields = ["bids_hcp_root", "hcppipe_root", "group_gm_mask", "output_dir", "master_spreadsheet",
                    "full_frame_master_spreadsheet"]
required = required_arg.split()

missing = [k for k in required if k not in pipeline]
if missing:
    print(f"resolve_pipeline_config.sh: {config_path}'s \"pipeline\" section is missing "
          f"required field(s): {missing}", file=sys.stderr)
    sys.exit(1)

for key in optional_fields:
    print(f"{key.upper()}={shlex.quote(str(pipeline.get(key, '')))}")

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
