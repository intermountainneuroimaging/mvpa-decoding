#!/usr/bin/env python3
"""
Validate the "model_conditions" section of the mvpa config: which
volume_of_interest rows count as which classifier condition, for
training / testing / timecourse_decoding.

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

timecourse_decoding also requires a "window", describing the decode window
around each matched event independently of however volume_of_interest was
computed when the master_spreadsheet was built -- e.g. to decode from
stimulus onset (no hemodynamic lag) through 10s past the event's end:

    "window": {
        "start": {"reference": "onset", "offset_seconds": 0},
        "end":   {"reference": "offset_end", "offset_seconds": 10}
    }

"reference" is "onset" (the event's own onset) or "offset_end" (onset +
duration); "offset_seconds" is added to that reference time and may be
negative (e.g. to start before onset).

Usage:
    python validate_model_config.py --config mvpa_config.json \\
        [--master-spreadsheet master_spreadsheet.csv]

Passing --master-spreadsheet additionally evaluates every condition's query
against the real table and reports empty-match and overlapping-condition
problems, not just structural JSON errors.
"""

import argparse
import json
import sys

import pandas as pd

from mvpa_common import validate_query_node, evaluate_query_node, validate_window

SECTIONS = ("training", "testing", "timecourse_decoding")


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_config(cfg: dict, valid_columns=None, df: pd.DataFrame = None):
    errors = []
    warnings = []

    if "model_conditions" not in cfg:
        errors.append("config missing required 'model_conditions' section")
        return errors, warnings

    model_conditions = cfg["model_conditions"]

    missing_sections = [s for s in SECTIONS if s not in model_conditions]
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
            window = model_conditions[section].get("window")
            if window is None:
                errors.append(f"{prefix}.window: required (decode window around each matched event)")
            else:
                errors.extend(validate_window(window, path=f"{prefix}.window"))

    # cross-section condition-name consistency
    present = [s for s in SECTIONS if s in section_condition_names]
    for a, b in zip(present, present[1:]):
        if section_condition_names[a] != section_condition_names[b]:
            warnings.append(
                f"condition names differ between '{a}' {sorted(section_condition_names[a])} "
                f"and '{b}' {sorted(section_condition_names[b])}"
            )

    # data-driven checks
    if df is not None and not errors:
        for section in SECTIONS:
            if section not in model_conditions:
                continue
            conditions = model_conditions[section]["conditions"]
            masks = {}
            for name, query in conditions.items():
                mask = evaluate_query_node(query, df)
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

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the mvpa config JSON (validates its model_conditions section)")
    parser.add_argument("--master-spreadsheet", default=None, help="Path to master_spreadsheet.csv (enables data-driven checks)")
    args = parser.parse_args()

    cfg = load_json(args.config)

    valid_columns = None
    df = None
    if args.master_spreadsheet:
        df = pd.read_csv(args.master_spreadsheet, dtype=str)
        valid_columns = set(df.columns)

    print(f"Validating {args.config}" + (f" against {args.master_spreadsheet}" if df is not None else " (structure only)"))
    errors, warnings = validate_config(cfg, valid_columns=valid_columns, df=df)

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")

    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
