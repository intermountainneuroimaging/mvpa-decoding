#!/usr/bin/env python3
"""
Shared utilities for the mvpa_banich toolchain: BIDS filename parsing, the
onset/duration -> BOLD volume-range math, and the model_conditions query DSL
(used by generate_master_spreadsheet.py, validate_model_config.py, and
mvpa_workflow.py).
"""

import math
import re

import numpy as np
import pandas as pd

BIDS_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<val>[^_.]+)")

MATCH_TYPES = {"exact", "in", "regex"}
BOOL_KEYS = {"and", "or", "not"}
WINDOW_REFERENCES = {"onset", "offset_end"}


def parse_bids_entities(filename: str) -> dict:
    from pathlib import Path
    return {m.group("key"): m.group("val") for m in BIDS_ENTITY_RE.finditer(Path(filename).name)}


def compute_volume_range(start_time: float, stop_time: float, tr: float, n_frames: int):
    """Return [start_vol, stop_vol) covering start_time for a span of (stop_time - start_time)
    seconds, clipped to n_frames.

    The volume *count* is derived from the duration (rounded to the nearest whole TR), not from
    independently flooring start_time and ceiling stop_time -- that combination systematically
    rounds outward at both ends, so a real-world onset that doesn't fall exactly on a TR boundary
    (i.e. almost always) inflates the window by a full extra volume even when the duration is an
    exact multiple of TR. At least 1 volume is always kept, so a duration shorter than one TR still
    gets the single volume it overlaps rather than rounding down to zero.
    """
    start_vol = int(math.floor(start_time / tr))
    n_volumes = max(1, round((stop_time - start_time) / tr))
    stop_vol = min(start_vol + n_volumes, n_frames)
    return start_vol, max(stop_vol, start_vol)


def build_trial_pivot_table(df: pd.DataFrame, group_cols=("boldfile", "trial_index")) -> pd.DataFrame:
    """One row per source event (grouped by group_cols, matching the row count of the
    source events.tsv files), with that trial's volume_of_interest values spread across
    vol_of_interest_1..N columns (NaN-padded to the widest trial). Sanity-check table --
    not used for modeling."""
    id_cols = [c for c in df.columns if c != "volume_of_interest"]

    records = []
    max_vols = 0
    for _, group in df.groupby(list(group_cols), sort=False):
        first = group.iloc[0]
        vols = sorted(group["volume_of_interest"].tolist())
        max_vols = max(max_vols, len(vols))
        record = {col: first[col] for col in id_cols}
        record["_vols"] = vols
        records.append(record)

    for record in records:
        vols = record.pop("_vols")
        for i in range(max_vols):
            record[f"vol_of_interest_{i + 1}"] = vols[i] if i < len(vols) else np.nan

    return pd.DataFrame(records)


# =====================================================
# Query DSL: {"column", "match", "value"/"values"} leaves, {"and"/"or"/"not"} nodes
# =====================================================

def validate_query_node(node, valid_columns, path="query") -> list:
    errors = []
    if not isinstance(node, dict):
        return [f"{path}: must be an object, got {type(node).__name__}"]

    keys = set(node.keys())
    bool_keys_present = keys & BOOL_KEYS
    is_leaf = "column" in keys

    if bool_keys_present and is_leaf:
        return [f"{path}: cannot mix boolean operator {sorted(bool_keys_present)} with a leaf 'column' key"]
    if len(bool_keys_present) > 1:
        return [f"{path}: multiple boolean operators {sorted(bool_keys_present)}, expected exactly one"]

    if bool_keys_present:
        op = next(iter(bool_keys_present))
        if op == "not":
            errors.extend(validate_query_node(node["not"], valid_columns, f"{path}.not"))
        else:
            children = node[op]
            if not isinstance(children, list) or len(children) == 0:
                errors.append(f"{path}.{op}: must be a non-empty list")
            else:
                for i, child in enumerate(children):
                    errors.extend(validate_query_node(child, valid_columns, f"{path}.{op}[{i}]"))
        return errors

    if not is_leaf:
        return [f"{path}: must have one of 'and'/'or'/'not', or a leaf 'column' key"]

    column = node.get("column")
    match = node.get("match")

    if not isinstance(column, str) or not column:
        errors.append(f"{path}: 'column' must be a non-empty string")
    elif valid_columns is not None and column not in valid_columns:
        errors.append(f"{path}: unknown column {column!r} (not in master_spreadsheet columns: {sorted(valid_columns)})")

    if match not in MATCH_TYPES:
        errors.append(f"{path}: 'match' must be one of {sorted(MATCH_TYPES)}, got {match!r}")
    elif match == "exact":
        if not isinstance(node.get("value"), str):
            errors.append(f"{path}: match='exact' requires a string 'value'")
    elif match == "in":
        values = node.get("values")
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            errors.append(f"{path}: match='in' requires a non-empty list of strings 'values'")
    elif match == "regex":
        pattern = node.get("value")
        if not isinstance(pattern, str):
            errors.append(f"{path}: match='regex' requires a string 'value'")
        else:
            try:
                re.compile(pattern)
            except re.error as e:
                errors.append(f"{path}: invalid regex {pattern!r}: {e}")

    return errors


def evaluate_query_node(node, df: pd.DataFrame) -> pd.Series:
    if "not" in node:
        return ~evaluate_query_node(node["not"], df)
    if "and" in node:
        mask = pd.Series(True, index=df.index)
        for child in node["and"]:
            mask &= evaluate_query_node(child, df)
        return mask
    if "or" in node:
        mask = pd.Series(False, index=df.index)
        for child in node["or"]:
            mask |= evaluate_query_node(child, df)
        return mask

    column, match = node["column"], node["match"]
    series = df[column].astype(str)
    if match == "exact":
        return series == node["value"]
    if match == "in":
        return series.isin(node["values"])
    if match == "regex":
        pattern = re.compile(node["value"])
        return series.apply(lambda v: pattern.fullmatch(v) is not None)
    raise ValueError(f"Unknown match type: {match}")


# =====================================================
# timecourse_decoding window: {"start"/"end": {"reference", "offset_seconds"}}
# =====================================================

def validate_window_bound(bound, path) -> list:
    errors = []
    if not isinstance(bound, dict):
        return [f"{path}: must be an object with 'reference' and 'offset_seconds'"]

    reference = bound.get("reference")
    if reference not in WINDOW_REFERENCES:
        errors.append(f"{path}.reference: must be one of {sorted(WINDOW_REFERENCES)}, got {reference!r}")

    offset = bound.get("offset_seconds", 0)
    if not isinstance(offset, (int, float)) or isinstance(offset, bool):
        errors.append(f"{path}.offset_seconds: must be a number, got {offset!r}")

    return errors


def validate_window(window, path="timecourse_decoding.window") -> list:
    if not isinstance(window, dict):
        return [f"{path}: must be an object with 'start' and 'end'"]

    errors = []
    for bound_name in ("start", "end"):
        if bound_name not in window:
            errors.append(f"{path}.{bound_name}: required")
        else:
            errors.extend(validate_window_bound(window[bound_name], f"{path}.{bound_name}"))

    if not errors:
        start, end = window["start"], window["end"]
        if start["reference"] == end["reference"] and end["offset_seconds"] <= start["offset_seconds"]:
            errors.append(
                f"{path}: end offset ({end['offset_seconds']}s from {end['reference']}) must be later than "
                f"start offset ({start['offset_seconds']}s from {start['reference']})"
            )

    return errors


def resolve_window_times(window: dict, onset: float, duration: float):
    """Return (start_time, stop_time) in seconds for a window spec against one event."""
    def resolve(bound):
        base = onset if bound["reference"] == "onset" else onset + duration
        return base + bound.get("offset_seconds", 0)

    return resolve(window["start"]), resolve(window["end"])
