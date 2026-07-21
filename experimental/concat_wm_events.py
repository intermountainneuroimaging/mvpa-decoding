#!/usr/bin/env python3

"""
EXPERIMENTAL -- not part of the stable pipeline (see experimental/README.md).

Builds a new events.tsv for each WMcombined run written by concat_wm_runs.sh,
by concatenating that pair's WMpos + WMneg raw events.tsv tables and shifting
every WMneg row's onset by the WMpos run's actual duration -- n_volumes (its
NIfTI's dim4) * TR (its pixdim4), read straight from the WMpos preprocessed
NIfTI's header, not assumed or hardcoded, so this is correct regardless of
each run's real length. concat_wm_runs.sh's concatenation order is always
(pos, neg) -- see fslmerge there -- so the offset only ever needs to apply to
the WMneg side.

Without this, master_spreadsheet.csv would still contain each original run's
untouched onsets, which no longer correspond to where those events actually
land inside the new merged 4D file -- generate_master_spreadsheet.py would
compute volume windows against the wrong timeline.

Manifest format: the --manifest file is exactly what concat_wm_runs.sh writes
(a "run-NN" -> {"pos", "neg", "output"} mapping of file paths) -- see
experimental/concat_manifest.example.json for a worked example matching
examples/sample-data's sub-4057 pairing.

Usage:
    python experimental/concat_wm_events.py \\
        --manifest out/sub-4057_ses-A1_concat_manifest.json \\
        --bids-root examples/sample-data \\
        --out-dir out/combined_events
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mvpa_common
from mvpa_common import parse_bids_entities, get_bold_header_info


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest", required=True,
        help="concat_manifest.json written by concat_wm_runs.sh "
             "(see experimental/concat_manifest.example.json for the expected shape)"
    )
    parser.add_argument("--bids-root", required=True, help="Raw BIDS root to find each pair's WMpos/WMneg events.tsv")
    parser.add_argument("--out-dir", required=True, help="Where to write the new WMcombined events.tsv files")
    return parser.parse_args()


def find_events_file(bids_root: str, entities: dict) -> str:
    """Find the one raw events.tsv matching entities (sub/ses/task/run) -- the
    same BIDS-entity-driven lookup convention used throughout this repo."""
    tokens = [f"sub-{entities['sub']}", f"task-{entities['task']}", f"run-{entities['run']}"]
    if "ses" in entities:
        tokens.append(f"ses-{entities['ses']}")
    candidates = glob.glob(os.path.join(bids_root, "**", "*_events.tsv"), recursive=True)
    matches = [f for f in candidates if all(t in os.path.basename(f) for t in tokens)]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly 1 events.tsv for {entities}, found {len(matches)}: {matches}")
    return matches[0]


def main():
    args = parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    for run_label, pair in manifest.items():
        pos_bold, neg_bold, out_bold = pair["pos"], pair["neg"], pair["output"]

        pos_entities = parse_bids_entities(pos_bold)
        neg_entities = parse_bids_entities(neg_bold)

        pos_events_path = find_events_file(args.bids_root, pos_entities)
        neg_events_path = find_events_file(args.bids_root, neg_entities)

        # offset = the WMpos run's actual duration, computed from its own
        # preprocessed NIfTI header (n_volumes * TR) -- flexible to any run
        # length, not a hardcoded/assumed constant
        tr, n_frames = get_bold_header_info(pos_bold)
        offset_seconds = n_frames * tr

        pos_events = pd.read_csv(pos_events_path, sep="\t")
        neg_events = pd.read_csv(neg_events_path, sep="\t").copy()
        neg_events["onset"] = pd.to_numeric(neg_events["onset"], errors="coerce") + offset_seconds

        combined = (
            pd.concat([pos_events, neg_events], ignore_index=True)
            .sort_values("onset")
            .reset_index(drop=True)
        )

        out_name = os.path.basename(out_bold).replace("_desc-preproc_bold.nii.gz", "_events.tsv")
        out_path = os.path.join(args.out_dir, out_name)
        combined.to_csv(out_path, sep="\t", index=False)
        print(
            f"{run_label}: wrote {out_path} "
            f"(WMneg onsets shifted by {offset_seconds:.3f}s = {n_frames} volumes @ TR={tr}s, from {pos_bold})"
        )


if __name__ == "__main__":
    main()
