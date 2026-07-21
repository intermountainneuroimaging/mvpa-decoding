#!/usr/bin/env python3
"""
Build a searchable subject/session/task/run/volume table from BIDS events.tsv files.

Reads the "event_extraction" section of the mvpa config. For every row of every
events.tsv found under `bids_root`, locate the matching BOLD file (to read its TR
and frame count), compute which BOLD volumes were active during that event (onset
shifted by a configurable hemodynamic lag, spanning the event's full duration), and
emit one output row per volume.

BOLD files are searched under `derivatives_root` (defaults to `bids_root` if omitted) --
set this separately when your preprocessed/derivative data (e.g. fMRIPrep output)
lives in a different directory tree than the raw events.tsv files, or doesn't
follow the same naming convention (pair it with `bold_glob`).

Usage:
    python generate_master_spreadsheet.py --config mvpa_config.json
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import nibabel as nib

from mvpa_common import parse_bids_entities, compute_volume_range

# "ses" has a dedicated output column (session) whenever present, but -- per the
# BIDS spec -- is optional in filenames for single-session datasets, so it's not
# required for a file to be processed.
REQUIRED_ENTITIES = ("sub", "task", "run")
HANDLED_ENTITIES = ("sub", "ses", "task", "run")

# trial_type values considered administrative/non-trial (never relevant to any
# analysis) and always dropped, regardless of what any config asks for. Edit
# this list directly to add/remove exclusions -- not exposed as a config option
# on purpose, since it's a blanket policy rather than a per-dataset choice.
EXCLUDED_TRIAL_TYPE_EXACT = ("start_block", "end_block")
EXCLUDED_TRIAL_TYPE_SUBSTRINGS = ("fixation", "postrt")


def is_excluded_trial_type(trial_type) -> bool:
    tt = str(trial_type).lower()
    if tt in EXCLUDED_TRIAL_TYPE_EXACT:
        return True
    return any(s in tt for s in EXCLUDED_TRIAL_TYPE_SUBSTRINGS)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_events_files(bids_root: str, events_glob: str):
    return sorted(glob.glob(os.path.join(bids_root, events_glob), recursive=True))


def find_bold_file(derivatives_root: str, entities: dict, bold_glob: str = None):
    if bold_glob:
        pattern = bold_glob.format(
            subject=entities["sub"], session=entities.get("ses", ""),
            task=entities["task"], run=entities["run"],
        )
        matches = sorted(glob.glob(os.path.join(derivatives_root, pattern), recursive=True))
    else:
        tokens = [f"sub-{entities['sub']}", f"task-{entities['task']}", f"run-{entities['run']}"]
        if "ses" in entities:
            tokens.append(f"ses-{entities['ses']}")
        matches = [
            f for f in glob.glob(os.path.join(derivatives_root, "**", "*.nii.gz"), recursive=True)
            if "bold" in os.path.basename(f) and all(t in os.path.basename(f) for t in tokens)
        ]
        matches = sorted(matches)
    return matches


def load_expected_events(path: str) -> set:
    with open(path) as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        loaded = loaded.get("expected_trial_types", loaded.get("trial_types", []))
    return set(loaded)


def process_events_file(events_path: str, derivatives_root: str, hemodynamic_lag: float, bold_glob: str = None):
    entities = parse_bids_entities(events_path)
    missing = [e for e in REQUIRED_ENTITIES if e not in entities]
    if missing:
        print(f"  (!) skipping {events_path}: missing BIDS entities {missing} in filename")
        return None
    extra_entities = {k: v for k, v in entities.items() if k not in HANDLED_ENTITIES}

    bold_matches = find_bold_file(derivatives_root, entities, bold_glob)
    if len(bold_matches) == 0:
        print(f"  (!) skipping {events_path}: no matching BOLD file found")
        return None
    if len(bold_matches) > 1:
        print(f"  (!) skipping {events_path}: {len(bold_matches)} ambiguous BOLD matches: {bold_matches}")
        return None
    bold_path = bold_matches[0]

    header = nib.load(bold_path).header
    tr = float(header.get_zooms()[3])
    n_frames = int(header.get_data_shape()[-1])

    events = pd.read_csv(events_path, sep="\t")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
    events = events.sort_values("onset").reset_index(drop=True)

    # Drop excluded/invalid rows *before* assigning trial_index, so trial_index is a
    # contiguous 1..N over exactly the events that end up in the output table --
    # matching the final conditions of interest, not raw position in the source file.
    excluded_mask = events["trial_type"].apply(is_excluded_trial_type)
    invalid_mask = events["onset"].isna() | events["duration"].isna() | ~np.isfinite(events["duration"])

    excluded_count = int(excluded_mask.sum())
    if excluded_count:
        print(f"  (i) excluded {excluded_count} administrative/non-trial row(s) (fixation/block/postRT) from {events_path}")

    invalid_count = int((invalid_mask & ~excluded_mask).sum())
    if invalid_count:
        bad_types = events.loc[invalid_mask & ~excluded_mask, "trial_type"].tolist()
        print(f"  (!) skipping {invalid_count} row(s) with non-finite duration in {events_path}: trial_type(s)={bad_types}")

    events = events[~excluded_mask & ~invalid_mask].reset_index(drop=True)

    rows = []
    for i, row in events.iterrows():
        duration = row["duration"]
        start_time = row["onset"] + hemodynamic_lag
        stop_time = start_time + duration
        start_vol, stop_vol = compute_volume_range(start_time, stop_time, tr, n_frames)

        for vol in range(start_vol, stop_vol):
            rows.append({
                "subject": entities["sub"],
                "session": entities.get("ses", ""),
                "volume_of_interest": vol,
                "trial_type": row["trial_type"],
                "trial_index": i + 1,
                "onset": row["onset"],
                "duration": duration,
                "task": entities["task"],
                "run": int(entities["run"]),
                "boldfile": bold_path,
                "eventfile": events_path,
                **extra_entities,
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the mvpa config JSON (reads its event_extraction section)")
    parser.add_argument("--output", default=None, help="Override event_extraction's output_file")
    parser.add_argument("--hemodynamic-lag", type=float, default=None, help="Override event_extraction's hemodynamic_lag (seconds)")
    parser.add_argument("--expected-events", default=None, help="Override event_extraction's expected_events_file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "event_extraction" not in cfg:
        raise SystemExit("config missing required 'event_extraction' section")
    event_cfg = cfg["event_extraction"]

    bids_root = event_cfg["bids_root"]
    derivatives_root = event_cfg.get("derivatives_root", bids_root)
    hemodynamic_lag = args.hemodynamic_lag if args.hemodynamic_lag is not None else event_cfg.get("hemodynamic_lag", 0)
    output_file = args.output or event_cfg.get("output_file", "master_spreadsheet.csv")
    bold_glob = event_cfg.get("bold_glob")
    expected_events_file = args.expected_events or event_cfg.get("expected_events_file")

    events_files = find_events_files(bids_root, event_cfg.get("events_glob", "**/*_events.tsv"))
    print(f"Found {len(events_files)} events file(s) under {bids_root}")

    all_rows = []
    for events_path in events_files:
        df = process_events_file(events_path, derivatives_root, hemodynamic_lag, bold_glob)
        if df is not None and not df.empty:
            all_rows.append(df)

    if not all_rows:
        raise SystemExit("No events rows produced -- check bids_root/derivatives_root/events_glob/bold_glob in the config.")

    table = pd.concat(all_rows, ignore_index=True)
    table = table.sort_values(["subject", "task", "run", "volume_of_interest"])

    if expected_events_file:
        expected = load_expected_events(expected_events_file)
        observed = set(table["trial_type"].dropna().unique())
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        if missing:
            print(f"(!) expected trial_type(s) never observed in this dataset: {missing}")
        if unexpected:
            print(f"(!) observed trial_type(s) not in {expected_events_file} (possible typo?): {unexpected}")

    table.to_csv(output_file, index=False)
    print(f"Wrote {len(table)} rows to {output_file}")


if __name__ == "__main__":
    main()
