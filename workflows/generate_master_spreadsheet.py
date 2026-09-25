#!/usr/bin/env python3
"""
Build a searchable subject/session/task/run/volume table from BIDS events.tsv files.

Reads the "event_extraction" section of the mvpa config. For every events.tsv found
under `bids_root`, locates the matching BOLD file (to read its TR and frame count)
and emits one output row per BOLD volume, 0..n_frames-1, for every run -- nothing is
excluded and no hemodynamic_lag is applied here: each volume's trial_type/onset/
duration come from whichever real event (administrative rows like fixation/rest
included) covers it, verbatim from events.tsv, via `build_full_frame_table`
(utils/mvpa_common.py). A volume covered by no event at all gets a blank
trial_type/onset/duration/event_index rather than being dropped.

`event_extraction.hemodynamic_lag` is read here only to pass through to
downstream consumers via the config -- model_conditions.training/testing (and,
transitively, model.kfold_cv) apply it themselves when selecting which volumes
of a matched event to use (see mvpa_common.label_conditions_with_lag), shifting
forward from the event's own onset to where the BOLD response is expected to
peak. model_conditions.timecourse_decoding never applies it -- continuous
decoding always uses this table's real-time, unshifted labels.

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
import sys

import pandas as pd
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import parse_bids_entities, resolve_config_root, build_full_frame_table

# "ses" has a dedicated output column (session) whenever present, but -- per the
# BIDS spec -- is optional in filenames for single-session datasets, so it's not
# required for a file to be processed.
REQUIRED_ENTITIES = ("sub", "task", "run")
HANDLED_ENTITIES = ("sub", "ses", "task", "run")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def find_events_files(bids_root: str, events_glob: str):
    return sorted(glob.glob(os.path.join(bids_root, events_glob), recursive=True))


def find_bold_file(derivatives_root: str, entities: dict, bold_glob: str = None, verbose: bool = False):
    if bold_glob:
        # {subject}/{session}/{task}/{run} are friendly aliases for sub/ses/task/run;
        # any *other* BIDS entity found in the events filename (e.g. dir-pa -> {dir})
        # is also available under its own raw key, so bold_glob can reference whatever
        # entities your dataset actually has without any code change.
        fmt_entities = {
            **entities,
            "subject": entities["sub"], "session": entities.get("ses", ""),
            "task": entities["task"], "run": entities["run"],
        }
        try:
            pattern = bold_glob.format(**fmt_entities)
        except KeyError as exc:
            missing_key = exc.args[0]
            print(f"    (!) bold_glob references {{{missing_key}}}, which isn't among the entities parsed "
                  f"from this filename ({entities}) -- check for a typo in bold_glob, or confirm this entity "
                  f"actually appears in every events filename")
            return []
        search_path = os.path.join(derivatives_root, pattern)
        matches = sorted(glob.glob(search_path, recursive=True))
        search_desc = f"bold_glob={bold_glob!r} -> formatted={pattern!r} -> searched {search_path!r}"
    else:
        tokens = [f"sub-{entities['sub']}", f"task-{entities['task']}", f"run-{entities['run']}"]
        if "ses" in entities:
            tokens.append(f"ses-{entities['ses']}")
        all_nii = glob.glob(os.path.join(derivatives_root, "**", "*.nii.gz"), recursive=True)
        matches = sorted(
            f for f in all_nii
            if "bold" in os.path.basename(f) and all(t in os.path.basename(f) for t in tokens)
        )
        search_desc = (
            f"no bold_glob set -- scanned {len(all_nii)} *.nii.gz file(s) under "
            f"derivatives_root={derivatives_root!r} for a basename containing 'bold' + all of {tokens}"
        )

    if verbose or not matches:
        print(f"    bold search for entities={entities}: {search_desc} -> {len(matches)} match(es)")

    if not matches:
        near = sorted(glob.glob(os.path.join(derivatives_root, "**", f"*sub-{entities['sub']}*.nii.gz"), recursive=True))
        if near:
            print(f"    {len(near)} file(s) under derivatives_root do contain 'sub-{entities['sub']}' (showing up to 10) -- "
                  f"check bold_glob/naming against these:")
            for f in near[:10]:
                print(f"      {f}")
        else:
            print(f"    no files at all under derivatives_root ({derivatives_root!r}) contain 'sub-{entities['sub']}' -- "
                  f"derivatives_root is likely wrong (or this subject truly isn't there)")

    return matches


def load_expected_events(path: str) -> set:
    with open(path) as f:
        loaded = json.load(f)
    if isinstance(loaded, dict):
        loaded = loaded.get("expected_trial_types", loaded.get("trial_types", []))
    return set(loaded)


def process_events_file(events_path: str, derivatives_root: str, bold_glob: str = None, verbose: bool = False):
    """One row per BOLD volume 0..n_frames-1 for this events.tsv's run -- see
    build_full_frame_table for how each volume's trial_type/onset/duration/
    event_index is derived."""
    entities = parse_bids_entities(events_path)
    missing = [e for e in REQUIRED_ENTITIES if e not in entities]
    if missing:
        print(f"  (!) skipping {events_path}: missing BIDS entities {missing} in filename")
        return None
    extra_entities = {k: v for k, v in entities.items() if k not in HANDLED_ENTITIES}

    bold_matches = find_bold_file(derivatives_root, entities, bold_glob, verbose=verbose)
    if len(bold_matches) == 0:
        print(f"  (!) skipping {events_path}: no matching BOLD file found (see search details above)")
        return None
    if len(bold_matches) > 1:
        print(f"  (!) skipping {events_path}: {len(bold_matches)} ambiguous BOLD matches:")
        for m in bold_matches:
            print(f"      {m}")
        return None
    bold_path = bold_matches[0]

    header = nib.load(bold_path).header
    tr = float(header.get_zooms()[3])
    n_frames = int(header.get_data_shape()[-1])

    events = pd.read_csv(events_path, sep="\t")
    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events["duration"] = pd.to_numeric(events["duration"], errors="coerce")

    table = build_full_frame_table(events, tr, n_frames)
    table["subject"] = entities["sub"]
    table["session"] = entities.get("ses", "")
    table["task"] = entities["task"]
    table["run"] = int(entities["run"])
    table["boldfile"] = bold_path
    table["eventfile"] = events_path
    for k, v in extra_entities.items():
        table[k] = v

    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the mvpa config JSON (reads its event_extraction section)")
    parser.add_argument("--output", default=None, help="Override event_extraction's output_file")
    parser.add_argument("--expected-events", default=None, help="Override event_extraction's expected_events_file")
    parser.add_argument("--verbose", action="store_true", help="Print the BOLD-file search details for every events file, not just failures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if "event_extraction" not in cfg:
        raise SystemExit("config missing required 'event_extraction' section")
    event_cfg = cfg["event_extraction"]

    bids_root = event_cfg["bids_root"]
    derivatives_root = resolve_config_root(event_cfg, "derivatives_root", bids_root, "event_extraction.derivatives_root")
    output_file = args.output or event_cfg.get("output_file", "master_spreadsheet.csv")
    bold_glob = event_cfg.get("bold_glob")
    expected_events_file = args.expected_events or event_cfg.get("expected_events_file")

    events_files = find_events_files(bids_root, event_cfg.get("events_glob", "**/*_events.tsv"))
    print(f"Found {len(events_files)} events file(s) under {bids_root}")

    all_rows = []
    for events_path in events_files:
        df = process_events_file(events_path, derivatives_root, bold_glob, verbose=args.verbose)
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
