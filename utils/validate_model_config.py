#!/usr/bin/env python3
"""
Validate the "model_conditions" section of the mvpa config: which
volume_of_interest rows count as which classifier condition, for
training (required) / testing / timecourse_decoding (both optional -- omit
either section entirely to skip that step in mvpa_workflow.py: testing
skips the independent test-set evaluation, timecourse_decoding skips
decoding entirely, including generate_report.py's timecourse page).

Each section's "conditions" is a mapping of condition name -> query, where a
query is a small recursive boolean tree over the master_spreadsheet columns:

    {"column": "trial_type", "match": "exact", "value": "view_face"}
    {"column": "trial_type", "match": "in", "values": ["view_face", "view_place"]}
    {"column": "trial_type", "match": "regex", "value": ".*face.*"}
    {"and": [<query>, <query>, ...]}
    {"or":  [<query>, <query>, ...]}
    {"not": <query>}

"exact"/"in" compare the column's string value directly; "regex" uses
re.fullmatch. See mvpa_config.example.json for a full example.

timecourse_decoding also requires a "trial_start_event" -- a single query
(same shape as one entry of "conditions") identifying which real event marks
the start of a trial, e.g. to anchor on a "view_face" cue:

    "trial_start_event": {"column": "trial_type", "match": "exact", "value": "view_face"}

Every BOLD volume in the full-frame spreadsheet (event_extraction.
full_frame_output_file) is decoded continuously -- nothing is skipped --
grouped into trials at each occurrence of trial_start_event, with
window_index counting up from 0 at each anchor. "conditions" no longer
selects which volumes to decode; it only determines which real trial_type
values count as a scored category (regressor_label) -- rows matching none of
them are still decoded, just left unscored.

Usage:
    python validate_model_config.py --config mvpa_config.json \\
        [--master-spreadsheet master_spreadsheet.csv] \\
        [--full-frame-spreadsheet master_spreadsheet_full.csv]

Passing --master-spreadsheet additionally evaluates every "conditions" query
against the real table and reports empty-match and overlapping-condition
problems, not just structural JSON errors. Passing --full-frame-spreadsheet
does the same for timecourse_decoding's "trial_start_event", since that's
evaluated against the full-frame table, not master_spreadsheet.csv.
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import validate_query_node, evaluate_query_node

REQUIRED_SECTIONS = ("training",)
SECTIONS = REQUIRED_SECTIONS + ("testing", "timecourse_decoding")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_config(cfg: dict, valid_columns=None, df: pd.DataFrame = None, full_frame_valid_columns=None, full_frame_df: pd.DataFrame = None):
    errors = []
    warnings = []

    if "model_conditions" not in cfg:
        errors.append("config missing required 'model_conditions' section")
        return errors, warnings

    model_conditions = cfg["model_conditions"]

    missing_sections = [s for s in REQUIRED_SECTIONS if s not in model_conditions]
    if missing_sections:
        errors.append(f"model_conditions missing required section(s): {missing_sections}")

    section_condition_names = {}

    for section in SECTIONS:
        if section not in model_conditions:
            continue
        prefix = f"model_conditions.{section}"
        conditions = model_conditions[section].get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            errors.append(f"{prefix}.conditions must be a non-empty object of name -> query")
            continue

        section_condition_names[section] = set(conditions.keys())

        for name, query in conditions.items():
            errors.extend(validate_query_node(query, valid_columns, path=f"{prefix}.conditions[{name!r}]"))

        if section == "timecourse_decoding":
            trial_start_event = model_conditions[section].get("trial_start_event")
            if trial_start_event is None:
                errors.append(f"{prefix}.trial_start_event: required (identifies the real event that starts a trial)")
            else:
                errors.extend(validate_query_node(trial_start_event, full_frame_valid_columns, path=f"{prefix}.trial_start_event"))

    # cross-section condition-name consistency
    present = [s for s in SECTIONS if s in section_condition_names]
    for a, b in zip(present, present[1:]):
        if section_condition_names[a] != section_condition_names[b]:
            warnings.append(
                f"condition names differ between '{a}' {sorted(section_condition_names[a])} "
                f"and '{b}' {sorted(section_condition_names[b])}"
            )

    # data-driven checks. timecourse_decoding's "conditions" and
    # "trial_start_event" are evaluated at runtime against the full-frame
    # spreadsheet (real per-frame trial_type, no exclusions), not
    # master_spreadsheet.csv -- so they're checked against full_frame_df here
    # instead of df whenever it's available; without it, timecourse_decoding
    # is skipped in this data-driven pass (structural checks above still ran).
    if not errors:
        for section in SECTIONS:
            if section not in model_conditions:
                continue
            section_df = full_frame_df if section == "timecourse_decoding" else df
            if section_df is None:
                if section == "timecourse_decoding":
                    print(f"  [{section}] no --full-frame-spreadsheet given -- skipping data-driven checks for this section")
                continue

            conditions = model_conditions[section]["conditions"]
            masks = {}
            for name, query in conditions.items():
                mask = evaluate_query_node(query, section_df)
                masks[name] = mask
                n = int(mask.sum())
                if n == 0:
                    errors.append(f"model_conditions.{section}.conditions[{name!r}] matches 0 rows in the master_spreadsheet")
                else:
                    print(f"  [{section}] {name!r}: {n} rows")

            names = list(masks.keys())
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    overlap = int((masks[a] & masks[b]).sum())
                    if overlap > 0:
                        warnings.append(
                            f"{section}: conditions {a!r} and {b!r} overlap on {overlap} row(s) -- ambiguous label"
                        )

            if section == "timecourse_decoding":
                trial_start_event = model_conditions[section].get("trial_start_event")
                if trial_start_event is not None:
                    n = int(evaluate_query_node(trial_start_event, section_df).sum())
                    if n == 0:
                        errors.append(f"model_conditions.{section}.trial_start_event matches 0 rows in the full-frame spreadsheet")
                    else:
                        print(f"  [{section}] trial_start_event: {n} rows")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the mvpa config JSON (validates its model_conditions section)")
    parser.add_argument("--master-spreadsheet", default=None, help="Path to master_spreadsheet.csv (enables data-driven checks for training/testing)")
    parser.add_argument("--full-frame-spreadsheet", default=None, help="Path to the full-frame spreadsheet (enables data-driven checks for timecourse_decoding's conditions/trial_start_event)")
    args = parser.parse_args()

    cfg = load_json(args.config)

    valid_columns = None
    df = None
    if args.master_spreadsheet:
        df = pd.read_csv(args.master_spreadsheet, dtype=str)
        valid_columns = set(df.columns)

    full_frame_valid_columns = None
    full_frame_df = None
    if args.full_frame_spreadsheet:
        full_frame_df = pd.read_csv(args.full_frame_spreadsheet, dtype=str)
        full_frame_valid_columns = set(full_frame_df.columns)

    print(f"Validating {args.config}" + (f" against {args.master_spreadsheet}" if df is not None else " (structure only)"))
    errors, warnings = validate_config(
        cfg, valid_columns=valid_columns, df=df,
        full_frame_valid_columns=full_frame_valid_columns, full_frame_df=full_frame_df,
    )

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
