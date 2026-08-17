#!/usr/bin/env python3

"""
Resample a NIfTI file between MNI and native space using the registration
warp fields HCP Pipelines writes for every subject -- a standalone utility,
independent of the MVPA classification/reporting scripts (they operate
entirely in native space; this is for moving data in or out of that space,
e.g. bringing an MNI-space preprocessed BOLD run into native space to match
this pipeline's native masks, or bringing a native-space importance map into
MNI space for group-level comparison).

Wraps FSL's `applywarp` (must be on PATH -- `module load fsl` on a cluster,
or `export FSLDIR=...`/`export PATH="$FSLDIR/bin:$PATH"` locally, same as
tutorial/preprocess_haxby.sh). Warp-file convention (absolute vs. relative
displacement) is left to applywarp's own auto-detection from the warp
file's header -- real fnirt-generated HCP warp fields carry that metadata,
so it's deliberately not hardcoded here; use --warp-convention to override
only if applywarp complains or misbehaves for a given file.

Expects the standard HCP Pipelines output layout under --derivatives-root:
    sub-{subject}/[ses-{session}/]MNINonLinear/xfms/standard2acpc_dc.nii.gz  (MNI -> native)
    sub-{subject}/[ses-{session}/]MNINonLinear/xfms/acpc_dc2standard.nii.gz  (native -> MNI)
Override --xfm-pattern if your layout differs.

Usage:
    python hcp_resample.py --input bold_mni.nii.gz --output bold_native.nii.gz \\
        --direction mni2native --subject 001 --session D1 \\
        --derivatives-root /pl/active/banich/studies/Clearvale/analysis/bids-hcp \\
        --reference sub-001_ses-D1_task-WMneg_run-03_bold.nii.gz --interp trilinear
"""

import argparse
import os
import shutil
import subprocess

DIRECTIONS = ("mni2native", "native2mni")
INTERP_CHOICES = ("nn", "trilinear", "sinc", "spline")

# Standard HCP Pipelines output layout -- {subject}/{session} placeholders,
# resolved relative to --derivatives-root. The with-session/without-session
# variants exist because -- same convention as mask_pattern elsewhere in this
# repo -- a sessionless dataset just omits the ses-{session}/ segment rather
# than needing a special "no session" mechanism.
DEFAULT_XFM_PATTERNS = {
    "mni2native": {
        True: "sub-{subject}/ses-{session}/MNINonLinear/xfms/standard2acpc_dc.nii.gz",
        False: "sub-{subject}/MNINonLinear/xfms/standard2acpc_dc.nii.gz",
    },
    "native2mni": {
        True: "sub-{subject}/ses-{session}/MNINonLinear/xfms/acpc_dc2standard.nii.gz",
        False: "sub-{subject}/MNINonLinear/xfms/acpc_dc2standard.nii.gz",
    },
}


# =====================================================
# Argument Parsing
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--input", required=True, help="NIfTI file to resample.")
    parser.add_argument("--output", required=True, help="Output path for the resampled NIfTI file.")
    parser.add_argument(
        "--direction", required=True, choices=DIRECTIONS,
        help="Which way to resample -- selects which HCP warp file to use."
    )
    parser.add_argument("--subject", required=True, help="Subject ID, used to resolve the warp file path (e.g. 001).")
    parser.add_argument(
        "--session", default=None,
        help="Session ID, used to resolve the warp file path. Omit for sessionless datasets."
    )
    parser.add_argument(
        "--derivatives-root", required=True,
        help="Root of the bids-hcp-style derivatives tree containing "
             "sub-{subject}/[ses-{session}/]MNINonLinear/xfms/."
    )
    parser.add_argument(
        "--reference", required=True,
        help="Target-space reference image (applywarp --ref). Not auto-templated -- pass explicitly, "
             "since this pipeline's native grid (EPI-resolution masks) and HCP's own native T1w grid "
             "differ, and guessing wrong here would silently misalign the output."
    )
    parser.add_argument(
        "--interp", default="trilinear", choices=INTERP_CHOICES,
        help="Interpolation method (applywarp --interp). Default trilinear -- use nn for masks/discrete labels."
    )
    parser.add_argument(
        "--datatype", default=None, choices=["char", "short", "int", "float", "double"],
        help="Force output data type (applywarp --datatype), e.g. int after --interp nn on a mask."
    )
    parser.add_argument(
        "--xfm-pattern", default=None,
        help="Override the warp-file path template ({subject}/{session} placeholders, resolved relative "
             "to --derivatives-root) for non-standard layouts. Defaults to the standard HCP Pipelines paths."
    )
    parser.add_argument(
        "--warp-convention", default=None, choices=["abs", "rel"],
        help="Force applywarp --abs/--rel. Leave unset (default) to let applywarp auto-detect the "
             "convention from the warp file's own header -- real HCP-generated warp fields carry this; "
             "only set this if applywarp complains or the result looks wrong."
    )

    return parser.parse_args()


# =====================================================
# Warp file resolution
# =====================================================

def resolve_xfm_path(derivatives_root: str, subject: str, session, direction: str, xfm_pattern: str = None) -> str:
    """Resolve and validate the HCP warp file for this subject/direction.
    Raises SystemExit with the resolved path if the file doesn't exist."""
    if direction not in DIRECTIONS:
        raise SystemExit(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    if xfm_pattern is None:
        xfm_pattern = DEFAULT_XFM_PATTERNS[direction][session is not None]

    xfm_path = os.path.join(derivatives_root, xfm_pattern.format(subject=subject, session=session))

    if not os.path.exists(xfm_path):
        raise SystemExit(
            f"HCP warp file not found for direction={direction!r}: {xfm_path}\n"
            f"  (resolved from derivatives_root={derivatives_root!r}, subject={subject!r}, "
            f"session={session!r}, xfm_pattern={xfm_pattern!r} -- override --xfm-pattern if your "
            f"layout differs from the standard HCP Pipelines convention)"
        )
    return xfm_path


# =====================================================
# applywarp invocation
# =====================================================

def build_applywarp_command(input_path: str, output_path: str, reference_path: str, warp_path: str,
                             interp: str = "trilinear", datatype: str = None, warp_convention: str = None) -> list:
    """Pure function -- returns the applywarp argv list, no subprocess call."""
    cmd = [
        "applywarp",
        f"--in={input_path}",
        f"--ref={reference_path}",
        f"--warp={warp_path}",
        f"--out={output_path}",
        f"--interp={interp}",
    ]
    if datatype:
        cmd.append(f"--datatype={datatype}")
    if warp_convention:
        cmd.append(f"--{warp_convention}")
    return cmd


def check_fsl_available() -> None:
    if shutil.which("applywarp") is None:
        raise SystemExit(
            "applywarp not found on PATH -- is FSL loaded? On a cluster: `module load fsl`. "
            "Locally: `export FSLDIR=/path/to/fsl` and `export PATH=\"$FSLDIR/bin:$PATH\"` "
            "(see tutorial/preprocess_haxby.sh for the full env setup, including FSLOUTPUTTYPE)."
        )


def run_applywarp(cmd: list) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"applywarp failed (exit {result.returncode}):\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout)


# =====================================================
# Main
# =====================================================

def main(args):
    if not os.path.exists(args.input):
        raise SystemExit(f"--input not found: {args.input}")
    if not os.path.exists(args.reference):
        raise SystemExit(f"--reference not found: {args.reference}")

    check_fsl_available()

    warp_path = resolve_xfm_path(args.derivatives_root, args.subject, args.session, args.direction, args.xfm_pattern)
    print(f"Using warp file: {warp_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    cmd = build_applywarp_command(
        args.input, args.output, args.reference, warp_path,
        interp=args.interp, datatype=args.datatype, warp_convention=args.warp_convention,
    )
    run_applywarp(cmd)

    print(f"Resampled ({args.direction}) -> {args.output}")


if __name__ == "__main__":
    main(parse_args())
