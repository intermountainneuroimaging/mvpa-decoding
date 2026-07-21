#!/usr/bin/env bash
#
# EXPERIMENTAL -- not part of the stable pipeline (see experimental/README.md).
#
# Concatenates paired WMpos/WMneg runs for the example WM dataset: the Nth
# WMpos run (by ascending run number) is paired with the Nth WMneg run, and
# each pair is merged into one 4D file via fslmerge -- e.g. for
# examples/sample-data (WMpos runs 1,2,5,6; WMneg runs 3,4,7,8), pairs are
# (1,3), (2,4), (5,7), (6,8), giving 4 concatenated runs.
#
# Requires inputs already coregistered to a common space and linearly
# detrended (e.g. via a pipeline like tutorial/preprocess_haxby.sh, or
# fMRIPrep). This script VERIFIES that precondition -- same grid/space (dims,
# pixdims, sform) and a per-file sanity check that the data looks detrended
# (|global mean| < 1.0, since a proper linear detrend drives every voxel's
# own time-series mean to ~0) -- it does NOT perform the preprocessing
# itself. The detrended check is a heuristic, not a proof: it will not catch
# every possible way input could be wrong, only the common case of
# forgetting to detrend at all.
#
# Usage:
#   export FSLDIR=/path/to/fsl
#   bash experimental/concat_wm_runs.sh <preproc_func_dir> <out_dir> <sub-XXXX> [ses-YYYY]
#
# Expects filenames like:
#   <preproc_func_dir>/sub-XXXX[_ses-YYYY]_task-WMpos_..._run-NN_desc-preproc_bold.nii.gz
#   <preproc_func_dir>/sub-XXXX[_ses-YYYY]_task-WMneg_..._run-NN_desc-preproc_bold.nii.gz

set -euo pipefail

: "${FSLDIR:?set FSLDIR to your FSL install, e.g. export FSLDIR=/Users/you/fsl}"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <preproc_func_dir> <out_dir> <sub-XXXX> [ses-YYYY]" >&2
    exit 1
fi

FUNC_DIR="$1"
OUT_DIR="$2"
SUBJECT="$3"
SESSION="${4:-}"
ses_tag=""
[ -n "$SESSION" ] && ses_tag="*${SESSION}*"

mkdir -p "$OUT_DIR"

sort_by_run() {
    # stdin: filenames: stdout: "runnum<TAB>filename", sorted numerically by run number
    while read -r f; do
        run=$(basename "$f" | grep -oE 'run-[0-9]+' | grep -oE '[0-9]+')
        printf '%s\t%s\n' "$run" "$f"
    done | sort -n -k1,1
}

find_sorted() {
    # bash-3.2-compatible stand-in for `mapfile` (macOS ships bash 3.2, no mapfile/readarray)
    local pattern="$1"
    find "$FUNC_DIR" -maxdepth 1 -name "$pattern" | sort_by_run | cut -f2
}

pos_files=()
while IFS= read -r f; do pos_files+=("$f"); done < <(find_sorted "*${SUBJECT}*${ses_tag}task-WMpos*_desc-preproc_bold.nii.gz")
neg_files=()
while IFS= read -r f; do neg_files+=("$f"); done < <(find_sorted "*${SUBJECT}*${ses_tag}task-WMneg*_desc-preproc_bold.nii.gz")

n_pos=${#pos_files[@]}
n_neg=${#neg_files[@]}

if [ "$n_pos" -eq 0 ] || [ "$n_neg" -eq 0 ]; then
    echo "ERROR: found $n_pos WMpos and $n_neg WMneg preprocessed run(s) under $FUNC_DIR for ${SUBJECT}${ses_tag:+ $ses_tag} -- nothing to pair" >&2
    exit 1
fi
if [ "$n_pos" -ne "$n_neg" ]; then
    echo "ERROR: found $n_pos WMpos run(s) but $n_neg WMneg run(s) -- can't pair 1:1. Files found:" >&2
    printf '  pos: %s\n' "${pos_files[@]}" >&2
    printf '  neg: %s\n' "${neg_files[@]}" >&2
    exit 1
fi

same_grid() {
    # dims1-3 + pixdims1-3 (voxel grid) + sform (physical space/origin).
    # Matched by exact first-field name (not a \b-anchored regex -- macOS's
    # bundled awk doesn't support \b word boundaries; an unsupported \b just
    # silently never matches, which previously made this check a no-op).
    local a="$1" b="$2"
    local info_a info_b sform_a sform_b
    info_a=$(fslinfo "$a" | awk '$1=="dim1"||$1=="dim2"||$1=="dim3"||$1=="pixdim1"||$1=="pixdim2"||$1=="pixdim3" {print $2}')
    info_b=$(fslinfo "$b" | awk '$1=="dim1"||$1=="dim2"||$1=="dim3"||$1=="pixdim1"||$1=="pixdim2"||$1=="pixdim3" {print $2}')
    [ "$info_a" = "$info_b" ] || return 1
    sform_a=$(fslhd "$a" | awk '$1 ~ /^sto_xyz:/ {printf "%.3f %.3f %.3f %.3f ", $2, $3, $4, $5}')
    sform_b=$(fslhd "$b" | awk '$1 ~ /^sto_xyz:/ {printf "%.3f %.3f %.3f %.3f ", $2, $3, $4, $5}')
    [ "$sform_a" = "$sform_b" ]
}

same_tr() {
    local a="$1" b="$2"
    local tr_a tr_b
    tr_a=$(fslinfo "$a" | awk '$1=="pixdim4" {print $2}')
    tr_b=$(fslinfo "$b" | awk '$1=="pixdim4" {print $2}')
    [ "$tr_a" = "$tr_b" ]
}

looks_detrended() {
    local f="$1" mean_abs
    mean_abs=$(fslstats "$f" -m | awk '{v=$1; if (v<0) v=-v; print v}')
    awk -v m="$mean_abs" 'BEGIN{exit !(m < 1.0)}'
}

manifest="$OUT_DIR/${SUBJECT}${SESSION:+_$SESSION}_concat_manifest.json"
{
    echo "{"
} > "$manifest"
first=true

for i in "${!pos_files[@]}"; do
    pos="${pos_files[$i]}"
    neg="${neg_files[$i]}"
    run_out=$(printf "%02d" $((i + 1)))
    out="$OUT_DIR/${SUBJECT}${SESSION:+_$SESSION}_task-WMcombined_run-${run_out}_desc-preproc_bold.nii.gz"

    echo "=== pair $((i + 1))/${n_pos}: WMpos + WMneg -> run-${run_out} ==="
    echo "  pos: $pos"
    echo "  neg: $neg"

    if ! same_grid "$pos" "$neg"; then
        echo "ERROR: $pos and $neg are not on the same grid/space (dims/pixdims/sform differ) -- refusing to concatenate. Coregister them to a common space first." >&2
        exit 1
    fi
    if ! same_tr "$pos" "$neg"; then
        echo "ERROR: $pos and $neg have different TRs -- refusing to concatenate." >&2
        exit 1
    fi
    for f in "$pos" "$neg"; do
        if ! looks_detrended "$f"; then
            echo "  (!) $f doesn't look detrended (|global mean| >= 1.0, expected ~0 for a linearly detrended series) -- proceeding, but double-check your preprocessing" >&2
        fi
    done

    fslmerge -t "$out" "$pos" "$neg"
    echo "  wrote: $out"

    if [ "$first" = true ]; then first=false; else printf ',\n' >> "$manifest"; fi
    printf '  "run-%s": {"pos": "%s", "neg": "%s", "output": "%s"}' \
        "$run_out" "$pos" "$neg" "$out" >> "$manifest"
done

printf '\n}\n' >> "$manifest"
echo "wrote manifest: $manifest"
