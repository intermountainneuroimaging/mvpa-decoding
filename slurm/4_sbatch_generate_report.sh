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
# Can also be run standalone -- submit from the repo root (`sbatch
# slurm/4_sbatch_generate_report.sh`) so slurm/pipeline_vars.sh's SCRIPTS_DIR
# fallback and the --output/--error log paths above (plain SBATCH
# directives, not variable-substituted) both resolve correctly -- or export
# SCRIPTS_DIR and pass --chdir yourself.
#
# --time is a rough starting estimate (MNI resampling + PDF/plot rendering
# across every subject's output, no measured runtime yet) -- check the first
# run's actual wall time and adjust before relying on it.

umask g+w

module use /projects/ics/modules
module load fsl/6.0.7

module load anaconda
conda activate incenv

source "${SCRIPTS_DIR:-.}/slurm/pipeline_vars.sh"

# --------------------------------------------
# resample each subject's aggregated importance map into MNI space
# (native2mni), writing <subject>_impa_mni.nii.gz alongside the original --
# generate_report.py detects that suffix and averages across subjects into
# one group-mean page instead of one per-subject page each (space not
# asserted). Loops
# over every subject directory found under $OUTPUT_DIR (any desc), rather
# than hardcoding this pipeline's own desc, so it never drifts out of sync
# with model.desc in the config. MNI_TEMPLATE is used as the --reference grid
# so every subject's warped map lands on the exact same grid the group
# average needs (shape-mismatched subjects are otherwise excluded with a
# warning, not a failure -- see README.md section 7).
#
# A subject's own sessions are assumed to share one native/structural
# registration (the standard, non-longitudinal HCP Pipelines setup), so any
# one of a subject's sessions' warp fields applies to their (session-pooled)
# importance map -- the same "doesn't matter which one" reasoning
# 1_batch_resample_native_mask.sh already relies on within a single session's
# runs. If your subjects have genuinely distinct per-session registrations,
# this silently picks the wrong one -- check applywarp's output alignment.
# --------------------------------------------
for subject_dir in "$OUTPUT_DIR"/*/*/; do
    subject=$(basename "$subject_dir")
    impa="${subject_dir}model/${subject}_impa.nii.gz"
    [ -f "$impa" ] || continue

    session_dir=$(find "$HCPPIPE_ROOT/sub-$subject" -maxdepth 1 -type d -name "ses-*" 2>/dev/null | sort | head -n 1)
    if [ -z "$session_dir" ]; then
        echo "  (!) sub-$subject: no session directory found under $HCPPIPE_ROOT -- skipping MNI resample"
        continue
    fi
    session=$(basename "$session_dir" | cut -d"-" -f2)
    impa_mni="${subject_dir}model/${subject}_impa_mni.nii.gz"

    echo "  sub-$subject (ses-$session): $impa -> $impa_mni"
    python "$SCRIPTS_DIR/utils/hcp_resample.py" \
        --input "$impa" \
        --output "$impa_mni" \
        --direction native2mni \
        --subject "$subject" --session "$session" \
        --derivatives-root "$HCPPIPE_ROOT" \
        --reference "$MNI_TEMPLATE" \
        --interp trilinear
done

python "$SCRIPTS_DIR/workflows/generate_report.py" --analysis-output-dir $OUTPUT_DIR \
    --config $CONFIG_FILE --master-spreadsheet $MASTER_SPREADSHEET
