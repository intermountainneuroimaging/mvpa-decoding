#!/bin/bash
#
#SBATCH --job-name=mvpa_kfold_report
#SBATCH --qos=cpu-normal
#SBATCH --partition=acpu
#SBATCH --account=ucb-general
#SBATCH --time=00:45:00
#SBATCH --output=logs/mvpa_kfold_report_%j.out
#SBATCH --error=logs/mvpa_kfold_report_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Single (non-array) job: stage 4, the final stage of 0_submit_mvpa_pipeline.sh
# -- first resamples every subject's aggregated importance map into MNI space
# (native2mni via hcp_resample.py), so the group report can average them onto
# one shared grid instead of showing one per-subject page each (space not
# asserted) (see
# README.md section 7, "Importance maps are averaged across subjects only
# when they share a common grid"), then builds one group PDF report
# (report_<desc>.pdf) aggregating every subject the preceding
# 3_batch_run_mvpa_workflow.sh array job wrote output for. Expects that array
# job to have already completed for every subject (0_submit_mvpa_pipeline.sh's
# --dependency=afterok on the whole array job is what guarantees this --
# afterok on an array job id waits for every task in it, not just the first).
#
# Fully standalone -- no dependency on 0_ having run first or exported
# anything: submit directly with
# `sbatch slurm/4_sbatch_generate_report.sh configs/my-study.json` from
# anywhere. The config path (`$1`) is the only required input; its
# pipeline.scripts_dir is how this script locates the rest of the repo (see
# resolve_pipeline_config.sh) -- --output/--error log paths above are still
# plain SBATCH directives (no variable substitution happens in them), so
# they resolve relative to wherever `sbatch` was invoked from regardless.
#
# Optional second arg (`$2`): either a comma-separated subject ID list
# (e.g. "001,004,010") or a path to a text file listing them (one per line
# and/or comma-separated) -- forwarded as-is to generate_report.py's
# --subjects, which detects which form it got, to restrict the group report
# to just those subjects' already-computed results, instead of every
# subject folder found under $OUTPUT_DIR. IDs must match the subject folder
# names exactly as mvpa_workflow.py wrote them (e.g. zero-padded "001", not
# "1", if that's what's actually under $OUTPUT_DIR). Does not affect which
# subjects' importance maps get resampled to MNI below -- that loop still
# covers everyone under $OUTPUT_DIR regardless, since it's harmless/idempotent
# to resample a subject the report itself won't end up using.
#
# --time is a rough starting estimate (MNI resampling + PDF/plot rendering
# across every subject's output, no measured runtime yet) -- check the first
# run's actual wall time and adjust before relying on it.

set -e
umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

CONFIG_FILE="$1"
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: sbatch $(basename "$0") <config.json> [subject1,subject2,... | subject-list.txt]" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# optional -- see header comment above
SUBJECT_LIST="$2"
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

# the group report itself only needs output_dir -- hcppipe_root is optional
# here: it's only used by the native->MNI importance-map resample below,
# which is itself skipped when hcppipe_root isn't set (e.g. every classifier
# under this output_dir already used model.mnispace=true, so there's no
# native-space impa left to resample). full_frame_master_spreadsheet is also
# optional -- only needed for timecourse-page event annotations (see
# resolve_pipeline_config.sh's header comment); without it the report still
# renders, just without those annotations.
export REQUIRED_PIPELINE_FIELDS="output_dir"
source "$SCRIPTS_DIR/slurm/resolve_pipeline_config.sh"

# --------------------------------------------
# resample each subject's importance map(s) into MNI space (native2mni),
# writing <subject>_impa_mni.nii.gz alongside the original -- generate_report.py
# detects that suffix and averages across subjects into one group-mean page
# instead of one per-subject page each (space not asserted). model/ (k-fold
# CV) and test/ (independent test set) are two independent output families --
# a subject may have either, both, or neither -- so both are resampled
# whenever present; generate_report.py's resolve_group_impa_mni prefers the
# test/ (Full) family when both exist for the group average. Loops over
# every subject directory found under $OUTPUT_DIR (any desc), rather than
# hardcoding this pipeline's own desc, so it never drifts out of sync with
# model.desc in the config. MNI_TEMPLATE is used as the --reference grid so
# every subject's warped map lands on the exact same grid the group average
# needs (shape-mismatched subjects are otherwise excluded with a warning,
# not a failure -- see README.md section 7).
#
# A subject's own sessions are assumed to share one native/structural
# registration (the standard, non-longitudinal HCP Pipelines setup), so any
# one of a subject's sessions' warp fields applies to their (session-pooled)
# importance map -- the same "doesn't matter which one" reasoning
# 1_batch_resample_native_mask.sh already relies on within a single session's
# runs. If your subjects have genuinely distinct per-session registrations,
# this silently picks the wrong one -- check applywarp's output alignment.
#
# HCPPIPE_ROOT is optional (see pipeline.hcppipe_root in the config) --
# skipped entirely when unset, since every classifier under $OUTPUT_DIR
# might already be model.mnispace=true (writing *_impa_mni.nii.gz directly,
# with no plain *_impa.nii.gz ever produced to resample in the first place).
# --------------------------------------------
if [ -z "$HCPPIPE_ROOT" ]; then
    echo "pipeline.hcppipe_root not set -- skipping native->MNI importance-map resample (using whatever *_impa_mni.nii.gz files already exist, e.g. from model.mnispace=true)"
else
    for subject_dir in "$OUTPUT_DIR"/*/*/; do
        subject=$(basename "$subject_dir")

        session_dir=""
        for family in model test; do
            impa="${subject_dir}${family}/${subject}_impa.nii.gz"
            [ -f "$impa" ] || continue

            if [ -z "$session_dir" ]; then
                session_dir=$(find "$HCPPIPE_ROOT/sub-$subject" -maxdepth 1 -type d -name "ses-*" 2>/dev/null | sort | head -n 1)
                if [ -z "$session_dir" ]; then
                    echo "  (!) sub-$subject: no session directory found under $HCPPIPE_ROOT -- skipping MNI resample"
                    break
                fi
                session=$(basename "$session_dir" | cut -d"-" -f2)
            fi

            impa_mni="${subject_dir}${family}/${subject}_impa_mni.nii.gz"
            echo "  sub-$subject (ses-$session, $family): $impa -> $impa_mni"
            python "$SCRIPTS_DIR/utils/hcp_resample.py" \
                --input "$impa" \
                --output "$impa_mni" \
                --direction native2mni \
                --subject "$subject" --session "$session" \
                --derivatives-root "$HCPPIPE_ROOT" \
                --reference "$MNI_TEMPLATE" \
                --interp trilinear
        done
    done
fi

FULL_FRAME_ARGS=()
if [ -n "$FULL_FRAME_MASTER_SPREADSHEET" ]; then
    FULL_FRAME_ARGS=(--full-frame-spreadsheet "$FULL_FRAME_MASTER_SPREADSHEET")
fi

if [ -n "$SUBJECT_LIST" ]; then
    python "$SCRIPTS_DIR/workflows/generate_report.py" --analysis-output-dir $OUTPUT_DIR \
        --config $CONFIG_FILE --subjects "$SUBJECT_LIST" "${FULL_FRAME_ARGS[@]}"
else
    python "$SCRIPTS_DIR/workflows/generate_report.py" --analysis-output-dir $OUTPUT_DIR \
        --config $CONFIG_FILE "${FULL_FRAME_ARGS[@]}"
fi
