#!/usr/bin/env python3
"""
Shared utilities for the mvpa_banich toolchain: BIDS filename parsing, the
onset/duration -> BOLD volume-range math, the model_conditions query DSL, and
the config-loading/classification/decoding primitives used by
mvpa_workflow.py (independent train/test evaluation, k-fold cross-validation
within training, and timecourse decoding -- each an independent, optional
step) -- also used directly by generate_master_spreadsheet.py and
validate_model_config.py.
"""

import importlib
import json
import math
import os
import re
import time
import resource
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from nilearn.maskers import NiftiMasker
except Exception:
    from nilearn.input_data import NiftiMasker

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, permutation_test_score
from sklearn.pipeline import Pipeline

BIDS_ENTITY_RE = re.compile(r"(?:^|_)(?P<key>[a-zA-Z]+)-(?P<val>[^_.]+)")

MATCH_TYPES = {"exact", "in", "regex"}
BOOL_KEYS = {"and", "or", "not"}

# trial_type values considered administrative/non-trial (never a real training/
# testing candidate) -- edit these directly to add/remove exclusions; not
# exposed as a config option on purpose, since it's a blanket policy rather
# than a per-dataset choice. master_spreadsheet.csv itself no longer filters
# these out (generate_master_spreadsheet.py keeps every real event, admin
# rows included, for continuous timecourse decoding) -- label_conditions_with_lag
# is what actually excludes them, at training/testing selection time.
EXCLUDED_TRIAL_TYPE_EXACT = ("start_block", "end_block")
EXCLUDED_TRIAL_TYPE_SUBSTRINGS = ("fixation", "postrt")


def is_excluded_trial_type(trial_type) -> bool:
    tt = str(trial_type).lower()
    if tt in EXCLUDED_TRIAL_TYPE_EXACT:
        return True
    return any(s in tt for s in EXCLUDED_TRIAL_TYPE_SUBSTRINGS)


# reserved trial_type value: an events.tsv that emits an explicit row tagged
# exactly this (any onset/duration) marks trial starts unambiguously by
# construction -- partition_into_trials prefers these over trial_start_event
# for a boldfile whenever they're present, exactly as events.tsv's own
# fixation/postrt/start_block/end_block are a fixed, not-configurable
# convention (see EXCLUDED_TRIAL_TYPE_* above).
# events.tsv files that don't emit this tag are unaffected -- trial_start_event
# is the fallback for those, evaluated per boldfile independently of every
# other boldfile in the same dataset (a real mix of both is fine).
TRIAL_START_TAG = "trial_start"


def parse_bids_entities(filename: str) -> dict:
    from pathlib import Path
    return {m.group("key"): m.group("val") for m in BIDS_ENTITY_RE.finditer(Path(filename).name)}


def resolve_config_root(section: dict, key: str, default: str, label: str) -> str:
    """Read an optional root-path override (e.g. derivatives_root) from a config
    section. A missing key or explicit JSON null means "inherit `default`". An explicit empty
    string is honored literally -- it resolves to the current working directory once joined
    with a relative pattern -- since that's almost never what's intended, it's flagged with a
    warning rather than silently treated the same as "unset"."""
    value = section.get(key)
    if key not in section or value is None:
        return default
    if value == "":
        print(f"(!) {label} is explicitly set to \"\" in the config -- this is interpreted "
              f"literally as the current working directory, NOT as \"inherit {default!r}\". "
              f"If you meant to inherit the default, remove the {key!r} key or set it to null instead.")
    return value


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


def build_full_frame_table(events: pd.DataFrame, tr: float, n_frames: int) -> pd.DataFrame:
    """One row per BOLD volume 0..n_frames-1 for a single run -- unlike
    generate_master_spreadsheet.py's own windowed rows, nothing is excluded
    and no hemodynamic_lag is applied (verbatim events.tsv onset/duration
    throughout; lag only ever matters for training/testing volume selection).

    `events` is the raw events.tsv table for this run (needs onset/duration/
    trial_type; extra columns are ignored). Each volume's trial_type/onset/
    duration come from whichever event's [onset, onset+duration) span covers
    it -- on overlap, the later-onset event wins. event_index is a contiguous
    1..N id (in onset order) over every valid event, administrative rows
    included, so rows produced by the same real event can be grouped back
    together; a volume covered by no event gets trial_type/onset/duration/
    event_index = NaN rather than being dropped."""
    trial_type = np.full(n_frames, np.nan, dtype=object)
    onset_col = np.full(n_frames, np.nan)
    duration_col = np.full(n_frames, np.nan)
    event_index = np.full(n_frames, np.nan)

    valid = events["onset"].notna() & events["duration"].notna() & np.isfinite(events["duration"])
    ordered = events[valid].sort_values("onset").reset_index(drop=True)

    for i, row in ordered.iterrows():
        start_vol, stop_vol = compute_volume_range(row["onset"], row["onset"] + row["duration"], tr, n_frames)
        trial_type[start_vol:stop_vol] = row["trial_type"]
        onset_col[start_vol:stop_vol] = row["onset"]
        duration_col[start_vol:stop_vol] = row["duration"]
        event_index[start_vol:stop_vol] = i + 1

    return pd.DataFrame({
        "volume_of_interest": np.arange(n_frames),
        "trial_type": trial_type,
        "onset": onset_col,
        "duration": duration_col,
        "event_index": event_index,
    })


def partition_into_trials(full_frame_df: pd.DataFrame, trial_start_event: dict) -> pd.DataFrame:
    """full_frame_df (one or more boldfiles' full-frame rows, e.g. from
    build_full_frame_table) -> same rows plus trial_index/window_index.

    For a boldfile whose events.tsv includes an explicit TRIAL_START_TAG-tagged
    row per trial, those rows are used as the anchors directly --
    trial_start_event is only the fallback, evaluated per boldfile
    independently, for a boldfile with no TRIAL_START_TAG rows at all (so a
    dataset can freely mix events.tsv files that do and don't tag trial
    starts explicitly). trial_start_event is a query-DSL node (same shape as
    one entry of a `conditions` mapping) evaluated against each boldfile's
    real events (one row per event_index, so a multi-TR anchor event only
    counts once) to find trial-start anchors, in onset order. A maximal run
    of consecutive real events that are (a) back-to-back with nothing else
    in between and (b)
    the exact same trial_type collapses into a single anchor at the run's
    first event -- e.g. a block design's 12 repeated same-category stimulus
    flashes are one trial, not 12 -- while a *change* in trial_type (even to
    another value that also matches trial_start_event) still starts a fresh
    one, and so does the same trial_type recurring after something else (a
    non-matching real event) happened in between. Volumes are partitioned
    per boldfile into [anchor_i, anchor_{i+1}) spans (the last span runs to
    that boldfile's final volume); window_index counts up from 0 at each
    anchor. Volumes before a boldfile's first anchor get trial_index=0 and
    window_index = their own volume_of_interest (there's no trial start to
    count from yet)."""
    pieces = []
    for boldfile, group in full_frame_df.groupby("boldfile", sort=False):
        group = group.sort_values("volume_of_interest").reset_index(drop=True)
        vols = group["volume_of_interest"].to_numpy()

        events = group.dropna(subset=["event_index"]).drop_duplicates("event_index").sort_values("onset").reset_index(drop=True)
        explicit_starts = events["trial_type"] == TRIAL_START_TAG
        if explicit_starts.any():
            anchor_mask = explicit_starts.to_numpy()
        else:
            anchor_mask = evaluate_query_node(trial_start_event, events).to_numpy() if len(events) else np.zeros(0, dtype=bool)

        is_boundary = np.zeros(len(events), dtype=bool)
        if len(events):
            is_boundary[0] = anchor_mask[0]
            if len(events) > 1:
                trial_types = events["trial_type"].to_numpy()
                fresh = (~anchor_mask[:-1]) | (trial_types[1:] != trial_types[:-1])
                is_boundary[1:] = anchor_mask[1:] & fresh

        boundaries = sorted(events.loc[is_boundary, "volume_of_interest"].tolist())

        trial_index = np.zeros(len(group), dtype=int)
        window_index = vols.copy()

        for i, start_vol in enumerate(boundaries):
            end_vol = boundaries[i + 1] if i + 1 < len(boundaries) else vols.max() + 1
            in_span = (vols >= start_vol) & (vols < end_vol)
            trial_index[in_span] = i + 1
            window_index[in_span] = vols[in_span] - start_vol

        group["trial_index"] = trial_index
        group["window_index"] = window_index
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True) if pieces else full_frame_df.copy()


def label_rows_optional(df: pd.DataFrame, conditions: dict, label_column: str = "regressor_label") -> pd.DataFrame:
    """Like label_rows, but rows matching no condition are kept (label_column
    set to None) instead of dropped -- used where every row must survive
    regardless of whether it's a scored category (e.g. continuous timecourse
    decoding, where fixation/ITI/view-cue frames are still decoded, just not
    scored against any trained class)."""
    labels = pd.Series(pd.NA, index=df.index, dtype=object)
    for name, query in conditions.items():
        mask = evaluate_query_node(query, df) & labels.isna()
        labels[mask] = name
    result = df.copy()
    result[label_column] = labels
    return result


def build_trial_pivot_table(df: pd.DataFrame, group_cols=("boldfile", "trial_index")) -> pd.DataFrame:
    """One row per group_cols group (e.g. one row per real event, when df is
    master_spreadsheet.csv and group_cols is ("boldfile", "event_index")),
    with that group's volume_of_interest values spread across
    vol_of_interest_1..N columns (NaN-padded to the widest group). Sanity-check
    table -- not used for modeling."""
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
        # a missing value (e.g. a continuous-decoding gap/rest frame with no
        # real trial_type) never matches any pattern -- same as "exact"/"in"
        # above, where comparing/testing membership against NaN is already
        # safely False; astype(str) above can leave a real missing value as
        # an actual (non-str) NaN rather than stringifying it, depending on
        # the column's pandas dtype, so this can't just rely on fullmatch
        # raising on the wrong type
        return series.apply(lambda v: isinstance(v, str) and pattern.fullmatch(v) is not None)
    raise ValueError(f"Unknown match type: {match}")


# =====================================================
# Small shared helpers (no dependency on any script's module-level state --
# safe to import from mvpa_workflow.py, generate_report.py, or anywhere else)
# =====================================================

def quick_safe(name) -> str:
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))


def label_rows(df: pd.DataFrame, conditions: dict, label_column: str = "regressor_label") -> pd.DataFrame:
    """Tag rows matching any condition's query with a label_column column
    (first matching condition wins, in dict-insertion order), dropping rows
    that match none. label_column defaults to "regressor_label" (the
    classifier's own condition labels); pass a different name to tag rows
    with an independent category instead -- e.g. generate_report.py's
    timecourse overlay, which needs its own column since the input already
    has a real "regressor_label"."""
    labeled = []
    for name, query in conditions.items():
        mask = evaluate_query_node(query, df)
        subset = df[mask].copy()
        subset[label_column] = name
        labeled.append(subset)
    combined = pd.concat(labeled)
    return combined[~combined.index.duplicated(keep="first")]


def label_conditions_with_lag(full_frame_df: pd.DataFrame, conditions: dict, hemodynamic_lag: float,
                               label_column: str = "regressor_label") -> pd.DataFrame:
    """model_conditions.training/testing's (and, transitively, model.kfold_cv's)
    counterpart to build_timecourse_instructions: selects which BOLD volumes
    actually get used to train/test a classifier.

    Deliberately does NOT enumerate candidate events from full_frame_df's own
    (dense, one-real-event-per-volume) event_index/trial_type columns --
    whenever two real events are close enough together that a later one's
    span fully overwrites an earlier one's (e.g. Haxby's ~2s-apart stimulus
    flashes at a 2.5s TR, where several flashes in a row can compete for the
    same single volume), the earlier event's event_index never appears
    *anywhere* in full_frame_df, so it would be silently invisible here too.
    Real events overlapping in time is exactly the normal, expected case for
    training/testing purposes (a classifier is trained on a lag-shifted
    window per event regardless of what else is nominally happening at that
    moment) -- it must never cause an event to be dropped as a candidate.

    Instead, for each boldfile, the original events.tsv (found via its own
    `eventfile` column -- constant per boldfile in full_frame_df) is re-read
    from scratch: same administrative-row exclusion as
    generate_master_spreadsheet.py used to apply directly (EXCLUDED_TRIAL_TYPE_*
    above), same contiguous 1..N trial_index numbering (in onset order, over
    every retained event, assigned *before* matching conditions -- so it's
    consistent across training/testing/model.kfold_cv regardless of which
    condition, if any, a given event matches). `conditions` is then matched
    against each retained event's own values (first match wins, same as
    label_rows; task/run/subject/session/boldfile/eventfile/any extra BIDS
    entity are pulled from full_frame_df's own per-boldfile constant columns,
    since the raw events.tsv itself has none of those) and events matching
    none are dropped. For each matched event, volumes are selected shifted
    forward by hemodynamic_lag -- [onset + hemodynamic_lag, onset + duration +
    hemodynamic_lag) -- since the BOLD response to a real-world event peaks
    several seconds after it, not during it; independent of what
    full_frame_df's own (unshifted, single-winner-per-volume) trial_type says
    is "really" happening there at that moment."""
    non_event_cols = {"volume_of_interest", "trial_type", "onset", "duration", "event_index"}

    rows = []
    for boldfile, group in full_frame_df.groupby("boldfile", sort=False):
        first = group.iloc[0]
        extra_cols = {c: first[c] for c in group.columns if c not in non_event_cols}

        events = pd.read_csv(first["eventfile"], sep="\t")
        events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
        events["duration"] = pd.to_numeric(events["duration"], errors="coerce")
        events = events.sort_values("onset").reset_index(drop=True)

        valid = events["onset"].notna() & events["duration"].notna() & np.isfinite(events["duration"])
        excluded = events["trial_type"].apply(is_excluded_trial_type)
        events = events[valid & ~excluded].reset_index(drop=True)
        # contiguous 1..N over every retained event, in onset order -- assigned
        # before condition-matching, so it's the same regardless of which
        # condition (if any) ends up matching a given event
        events["trial_index"] = np.arange(1, len(events) + 1)

        for k, v in extra_cols.items():
            events[k] = v

        labels = pd.Series(pd.NA, index=events.index, dtype=object)
        for name, query in conditions.items():
            mask = evaluate_query_node(query, events) & labels.isna()
            labels[mask] = name
        events = events[labels.notna()].copy()
        events[label_column] = labels[labels.notna()]
        if events.empty:
            continue

        tr, n_frames = get_bold_header_info(boldfile)
        for _, ev in events.iterrows():
            start_time = ev["onset"] + hemodynamic_lag
            stop_time = start_time + ev["duration"]
            start_vol, stop_vol = compute_volume_range(start_time, stop_time, tr, n_frames)
            base = ev.to_dict()
            for vol in range(start_vol, stop_vol):
                row = dict(base)
                row["volume_of_interest"] = vol
                rows.append(row)

    return pd.DataFrame(rows)


def get_single_match(pattern: str) -> str:
    import glob
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected 1 file, found {len(matches)}:\n" +
            "\n".join(str(m) for m in matches)
        )

    return str(matches[0])


_bold_header_cache = {}


def get_bold_header_info(boldfile: str):
    """Return (tr, n_frames) for a boldfile, read once and cached."""
    if boldfile not in _bold_header_cache:
        header = nib.load(boldfile).header
        _bold_header_cache[boldfile] = (float(header.get_zooms()[3]), int(header.get_data_shape()[-1]))
    return _bold_header_cache[boldfile]


# =====================================================
# Performance Monitor
# =====================================================

@contextmanager
def track_runtime(label: str = "run"):
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        c1 = time.process_time()

        # ru_maxrss: on Linux it's KB; on macOS it's bytes.
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = r / 1024.0  # Linux -> MB (KB/1024). If on macOS, change to r/(1024**2).

        print(f"[{label}] wall={t1 - t0:.3f}s | cpu={c1 - c0:.3f}s | peak_rss≈{rss_mb:.1f} MB")


# =====================================================
# Config loading (event_extraction + model_conditions + model, with
# model defaults merged in)
# =====================================================

def import_from_path(path: str):
    module_name, cls_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def default_model_config() -> dict:
    return {
        "desc": "default_classifier",
        "featureSelection": {
            "model": "ANOVA",
            "feat_p": 0.05,
        },
        "classifier": {
            "name": "sklearn.linear_model.LogisticRegression",
            "params": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 5000,
                "class_weight": "balanced"
            }
        },
    }


def impa_tag(mnispace: bool) -> str:
    """The importance-map filename token shared by both workflow scripts and
    generate_report.py: "impa_mni" when model.mnispace confirms the input
    BOLD/mask are already in MNI space, "impa" otherwise (space left
    unasserted in the filename, since it isn't reliably knowable from the
    file itself). A subject whose workflow ran with mnispace=true therefore
    writes directly into the same {subject}_impa_mni.nii.gz filename that
    hcp_resample.py --direction native2mni would otherwise produce -- so
    generate_report.py's cross-subject group averaging (which keys off that
    exact filename, see resolve_group_impa_mni) works without any separate
    resampling step."""
    return "impa_mni" if mnispace else "impa"


def merge_with_defaults(user_cfg, base):
    def recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                recursive_update(d[k], v)
            else:
                d[k] = v
        return d

    return recursive_update(base, user_cfg)


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        full_cfg = json.load(f)

    if "event_extraction" not in full_cfg:
        raise SystemExit("config missing required 'event_extraction' section")
    if "model_conditions" not in full_cfg:
        raise SystemExit("config missing required 'model_conditions' section")

    full_cfg["model"] = merge_with_defaults(full_cfg.get("model", {}), default_model_config())
    return full_cfg


# =====================================================
# Row labeling / balancing / BOLD loading
# =====================================================

def apply_regressor_codes(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    df = df.copy()
    df["regressor"] = pd.Categorical(
        df["regressor_label"], categories=categories, ordered=True
    ).codes + 1
    return df


def balance(xdf: pd.DataFrame) -> pd.DataFrame:
    df = xdf.copy()
    df["ID"] = df.index  # keep original row id

    # target number of rows to keep per regressor (lowest common denominator)
    k = df.groupby("regressor").size().min()

    # count rows per (regressor, run, trial_index) to prioritize fuller trials
    pair_counts = (
        df.groupby(["regressor", "run", "trial_index"])
          .size()
          .rename("pair_n")
          .reset_index()
    )

    # merge counts back so each row knows how "full" its (run, trial_index) group is
    df2 = df.merge(pair_counts, on=["regressor", "run", "trial_index"], how="left")

    # sort so we:
    #  1) for each regressor, consider the most-populated (run, trial_index) pairs first
    #  2) within a pair, keep highest volume_of_interest first (then we'll cap total to k)
    df2 = df2.sort_values(
        ["regressor", "pair_n", "run", "trial_index", "volume_of_interest"],
        ascending=[True, False, True, True, False]
    )

    # take first k rows per regressor (after prioritization + within-pair trimming)
    out = df2.groupby("regressor", group_keys=False).head(k)

    # optional: final ordering for downstream use
    out = out.sort_values(["run", "trial_index", "volume_of_interest"])

    # drop helper column if you want
    out = out.drop(columns=["pair_n"])

    return out


def decision_evidence(clf, rawdata):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(rawdata)

    raw_scores = clf.decision_function(rawdata)

    if len(clf.classes_) == 2:
        raw_scores = raw_scores.reshape(-1, 1)
        prob1 = 1 / (1 + np.exp(-raw_scores))
        return np.hstack([1 - prob1, prob1])

    e = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class ShapeError(Exception):
    pass


_masker_cache = {}


def load_images_and_mask(labeled_df: pd.DataFrame, mask_pattern_template: str = None):
    """Load BOLD patterns for every (subject, session, boldfile) group in
    labeled_df, z-score, and slice to each row's volume_of_interest.

    mask_pattern_template is the full path to the mask -- absolute, or
    relative to wherever the workflow script is run from (same convention
    bids_root/derivatives_root already use) -- with optional {subject}/
    {session} placeholders. Include them for one mask per subject (the
    common native-space case); omit them entirely for a single shared mask
    used for every subject (e.g. one MNI-space group mask), since a template
    with no placeholders just formats to itself and every subject resolves
    to the same literal path.

    mask_pattern_template itself is also optional (None or ""): every voxel
    in the BOLD volume is then used (an explicit all-ones mask built from
    that boldfile's own grid, not nilearn's own auto-mask heuristic, so
    behavior is deterministic) -- a warning is printed, since a real
    analysis almost always wants a real mask (huge feature count otherwise,
    including background/non-brain voxels). A *configured* mask_pattern
    that matches no file is still a hard error (get_single_match raises) --
    optional-and-unset and configured-but-missing are different failure
    modes, only the former is a fallback."""

    matrices = []
    labels = []
    indices = []
    masker = None

    for boldfile, group in labeled_df.groupby("boldfile", sort=False):
        if not os.path.exists(boldfile):
            raise FileNotFoundError(f"boldfile referenced by master_spreadsheet does not exist: {boldfile}")

        subject = group["subject"].iloc[0]
        session = group["session"].iloc[0]
        mask_key = (subject, session)

        if mask_key not in _masker_cache:
            bold_tr, _ = get_bold_header_info(boldfile)
            if mask_pattern_template:
                mask_pattern = mask_pattern_template.format(subject=subject, session=session)
                mask_file = get_single_match(mask_pattern)
                print(f"Using Mask File: {mask_file}")
                mask_img = mask_file
            else:
                print(f"  (!) No model.mask.mask_pattern configured -- using every voxel (no masking) "
                      f"for subject={subject!r}, session={session!r}. This is rarely what you want for a "
                      f"real analysis (huge feature count, includes background/non-brain voxels) -- set "
                      f"model.mask.mask_pattern to restrict to real brain tissue.")
                ref_img = nib.load(boldfile)
                mask_img = nib.Nifti1Image(np.ones(ref_img.shape[:3], dtype=np.uint8), ref_img.affine)
            _masker_cache[mask_key] = NiftiMasker(mask_img=mask_img, standardize=False, detrend=False, t_r=bold_tr)
        masker = _masker_cache[mask_key]

        # apply mask
        masked_data = masker.fit_transform(boldfile)

        # apply z-transform
        z_patterns = StandardScaler().fit_transform(masked_data)
        z_patterns = np.nan_to_num(z_patterns)

        # crop data to selected volumes
        vols = (
            pd.to_numeric(group["volume_of_interest"], errors="raise")
            .astype(int)
            .to_numpy()
        )

        z_patterns = z_patterns[vols, :]
        # stack image data to match instructions
        matrices.append(z_patterns)

        # stack the regression labels as well to be 100% sure they data and labels align
        labels.extend(group["regressor"].to_numpy())

        # stack selected indices to later extract volume metadata
        indices.extend(group.index.to_numpy())

        print(f"Sucessfully loaded: {boldfile}")

    if not matrices:
        raise ValueError("No rows to load -- check that model_conditions' queries actually match this subject's data.")

    # all image data stacked
    X = np.vstack(matrices)
    Y = np.array(labels).reshape(-1, 1)
    idx = np.array(indices)

    # Check X and Y have same number of rows (observations)
    if X.shape[0] != Y.shape[0]:
        raise ShapeError("Image Data and Labels Do No Align... Can't Proceed!!")

    return X, Y, idx, masker


def qualifying_boldfiles(full_frame_df: pd.DataFrame, timecourse_conditions: dict) -> set:
    """A boldfile "qualifies" for continuous timecourse decoding if ANY of its
    real rows match ANY of timecourse_conditions' queries -- this is how a
    task/run restriction inside a condition (e.g. {"and": [{"column": "run",
    "match": "in", "values": [...]}, ...]}) still scopes *which runs* get
    decoded at all, even though conditions no longer gate individual frames
    within a qualifying boldfile (see build_timecourse_instructions). task/run
    are constant per boldfile, so this reproduces exactly the same run/task
    restriction a condition used to enforce per-row, just resolved once per
    boldfile instead of once per frame."""
    matches_any = pd.Series(False, index=full_frame_df.index)
    for query in timecourse_conditions.values():
        matches_any |= evaluate_query_node(query, full_frame_df)
    return set(full_frame_df.loc[matches_any, "boldfile"])


def build_timecourse_instructions(full_frame_df: pd.DataFrame, timecourse_conditions: dict, trial_start_event: dict) -> pd.DataFrame:
    """Every volume of every *qualifying* boldfile (qualifying_boldfiles) in
    full_frame_df gets decoded -- nothing is subset or skipped within a
    qualifying boldfile, and a boldfile with no matching row at all is
    excluded entirely (e.g. a training-only run in a same-task, run-split
    design). Rows are partitioned into trials by partition_into_trials, keyed
    on trial_start_event (window_index resets to 0 at each anchor and counts
    up until the next one). trial_type is left as the REAL event active at
    that frame (not the anchor's); regressor_label is filled in only for rows
    whose real trial_type matches one of timecourse_conditions (first match
    wins, same as label_rows) -- everything else still comes out (fixation/
    view-cue/ITI/... frames), just with regressor_label=None, so they're
    decoded but excluded from accuracy/summary scoring until the caller
    filters by regressor_label."""
    scoped = full_frame_df[full_frame_df["boldfile"].isin(qualifying_boldfiles(full_frame_df, timecourse_conditions))]
    trials = partition_into_trials(scoped, trial_start_event)
    trials = label_rows_optional(trials, timecourse_conditions, label_column="regressor_label")

    return trials[[
        "subject", "session", "task", "run", "trial_type", "trial_index",
        "regressor_label", "boldfile", "volume_of_interest", "window_index",
    ]].reset_index(drop=True)


# =====================================================
# Output saving / aggregation
# =====================================================

def save_model_results(output_pattern, results, categories):
    """
    Save a dict of model outputs to disk, one file per metric -- except
    plain scalar metrics (a single number, not one per category), which are
    all collected together into one "metadata" file instead of one
    single-line file apiece (total_scores/whole_voxels/selected_voxels/
    feature_percent were previously 4 near-empty files per subject/fold,
    each holding exactly one number).

    Parameters
    ----------
    output_pattern : str
        Format string used to build output filenames. Must include a '{metric}'
        placeholder, e.g.:
            "/path/to/out/{metric}.csv"
            "/path/to/out/sub-01_run-2_{metric}.csv"

    results : dict[str, array-like]
        Mapping from metric name -> value to save. Supported value shapes:

        1) Scalar (a plain number, or a 0-d/length-1 array)
           - Every scalar metric in `results` is collected into one row per
             metric ("value" column, indexed by metric name) and saved
             together as output_pattern.format(metric="metadata").

        2) Square matrix (C, C)
           - Interpreted as a class-by-class matrix (e.g., confusion matrix,
             importance matrix).
           - Saved as its own CSV with row/column labels from `categories`.

        3) Column vector (C,) or (C, 1)
           - Interpreted as one value per category/class.
           - Saved as its own single-column CSV indexed by `categories`.

        4) Anything else (e.g., (S, E), (n_features,))
           - Saved via np.savetxt as its own numeric CSV (no labels).

    categories : sequence of str
        Category/class labels in the same order used by the model outputs.
        Length defines C.

    Notes
    -----
    - All outputs are written as CSV files.
    - Parent directories are created automatically.
    """
    categories = list(categories)
    C = len(categories)

    scalar_metrics = {}

    for metric, x in results.items():
        x = np.asarray(x)

        # Case 1: plain scalar -- collected, not written per-metric (see below)
        if x.ndim == 0:
            scalar_metrics[metric] = float(x)
            continue

        # Build output path for this metric and ensure parent directory exists
        output_file = output_pattern.format(metric=metric)
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        # Case 2: category-by-category matrix
        if x.shape == (C, C):
            df = pd.DataFrame(x, index=categories, columns=categories)
            df.to_csv(output_file, index=True)

        # Case 3: one value per category (accept (C,) or (C,1))
        elif x.shape == (C,) or x.shape == (C, 1):
            x_vec = x.reshape(C)  # ensures 1D length-C
            df = pd.DataFrame({metric: x_vec}, index=categories)
            df.to_csv(output_file, index=True)

        # Case 4: everything else (no labels)
        else:
            np.savetxt(output_file, x, delimiter=",", fmt="%.6f")

        print(f"[{metric}] saved -> {output_file} (shape={x.shape})")

    if scalar_metrics:
        output_file = output_pattern.format(metric="metadata")
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"value": scalar_metrics}).to_csv(output_file, index=True)
        print(f"[metadata] saved -> {output_file} ({list(scalar_metrics)})")


def average_fold_results(fold_results: list) -> dict:
    """Average a list of model_performance-style result dicts (scalar or array-valued
    metrics, all sharing the same keys/shapes) elementwise across folds."""
    mean_results = {}
    for k in fold_results[0].keys():
        values = [res[k] for res in fold_results]
        if np.isscalar(values[0]):
            mean_results[k] = float(np.mean(values))
        else:
            mean_results[k] = np.mean(np.stack(values, axis=0), axis=0)
    return mean_results


# =====================================================
# Classification: feature selection, fitting, evaluation, significance
# =====================================================

def resolve_feature_selection_params(training_data, training_labels, feature_selection_cfg: dict) -> tuple:
    """(mode, param) for sklearn's GenericUnivariateSelect, from
    model.featureSelection:
      - "n_voxels" set: ("k_best", n_voxels) -- select exactly that many
        voxels by ANOVA F-score, regardless of significance.
      - otherwise: ("fpr", thr) -- the ANOVA p-value cutoff from feat_p,
        widened until it selects at least 5 voxels.
    Shared by model_classification() and permutation_significance() so the
    permutation test's null-distribution pipelines use the exact same
    selection rule the real model did, not a different unwidened threshold."""
    n_voxels = feature_selection_cfg.get("n_voxels")
    if n_voxels is not None:
        return "k_best", n_voxels

    _, xP = f_classif(training_data, training_labels)
    xP = np.nan_to_num(xP, nan=1.0)
    thr = feature_selection_cfg["feat_p"]
    while np.sum(xP < thr) < 5 and thr <= 1.0:
        thr *= 1.1
    return "fpr", thr


def build_classifier_pipeline(mode: str, param, classifier_name: str, classifier_params: dict) -> Pipeline:
    """An unfit Pipeline(ANOVA feature selection, classifier). mode/param are
    passed straight through to GenericUnivariateSelect -- "fpr" (param=a
    p-value threshold, features with p < param) or "k_best" (param=an exact
    voxel count, the top-scoring param features by ANOVA F-score) -- the
    built-in sklearn equivalent of the manual xP < thr mask the "fpr" path
    originally used (verified to select identical voxels, including NaN
    p-value handling for zero-variance voxels)."""
    Cls = import_from_path(classifier_name)
    return Pipeline([
        ("feature_selection", GenericUnivariateSelect(score_func=f_classif, mode=mode, param=param)),
        ("classifier", Cls(**classifier_params)),
    ])


# cross_validation
def model_classification(training_data, training_labels, feature_selection_cfg: dict, classifier_name: str, classifier_params: dict):
    """Fit an ANOVA-feature-selection + classifier Pipeline. Bundling both
    steps into one estimator -- rather than externally tracking a voxel
    boolean mask, as before -- means the whole thing can be refit as a
    single unit, which permutation_significance() below needs (it refits
    fresh on every permutation's shuffled training labels)."""

    print("Training classifier...")

    mode, param = resolve_feature_selection_params(training_data, training_labels, feature_selection_cfg)
    pipe = build_classifier_pipeline(mode, param, classifier_name, classifier_params)
    pipe.fit(training_data, training_labels)

    return pipe


def extract_importance_map(pipe, n_features: int) -> np.ndarray:
    """Classifier weights reshaped back into whole-brain (unsliced) voxel space --
    pure function of the fitted pipeline itself, no labeled data needed, so it can
    produce a model's importance map even when there's no test set to score it
    against (e.g. a training-only classifier used solely for timecourse decoding)."""
    clf = pipe.named_steps["classifier"]
    xfeat = pipe.named_steps["feature_selection"].get_support()
    n_class = len(clf.classes_)

    # special case where classifier is binary (yes/no) -- only codes one label
    if n_class == 2:
        # voxel weights -- volume 0 and volume 1 are mat*-1 of each other
        impa = np.vstack((clf.coef_, -clf.coef_))
    else:
        impa = clf.coef_

    impa_full = np.zeros((n_class, n_features), dtype=impa.dtype)
    impa_full[:, xfeat] = impa
    return impa_full


def model_performance(pipe, testing_data, testing_labels):

    print("Testing model performance...")

    clf = pipe.named_steps["classifier"]

    # classes the classifier was actually trained on -- not np.unique(testing_labels),
    # which would drift shape-to-shape if a given fold's held-out data happens to be
    # missing one of 3+ classes entirely, breaking cross-fold averaging in main().
    xclass = clf.classes_
    n_class = len(xclass)

    # keep original (whole-brain) feature count to size impa_full below
    n_samples, n_features = testing_data.shape

    # apply model -- the pipeline applies feature selection internally, so the
    # full (unsliced) testing_data goes in
    xpred = pipe.predict(testing_data)

    # total model accuracy
    ttl_score = accuracy_score(testing_labels, xpred)

    impa_full = extract_importance_map(pipe, n_features)

    # evidence: same normalized-probability definition timecourse_decoding()
    # uses (softmax/predict_proba, rows sum to 1) -- not an independent
    # per-class sigmoid, which would let "evidence" mean two different things
    # depending on which report page you're looking at
    xevi = decision_evidence(pipe, testing_data)

    # normalized confusion matrix, and evidence matrix
    acc_mx = np.zeros((n_class, n_class))
    evi_mx = np.zeros((n_class, n_class))
    for xx in range(n_class):
        cls = xclass[xx]
        idxs = np.where(testing_labels == cls)[0]
        if idxs.size == 0:
            continue
        pred_slice = xpred[idxs]
        evi_slice  = xevi[idxs, :]
        for yy in range(n_class):
            ycond = xclass[yy]
            acc_mx[xx, yy] = (pred_slice == ycond).sum() / len(pred_slice)
            evi_mx[xx, yy] = float(np.mean(evi_slice[:, yy])) if len(evi_slice) else 0.0

    # ROC/AUC per class for this fold

    # One-vs-rest indicator matrix
    Y = (testing_labels[:, None] == xclass[None, :]).astype(np.uint8)
    # AUC per class → returns 1D array length n_class
    auc = np.array([
        roc_auc_score(Y[:, j], xevi[:, j])
        if Y[:, j].min() != Y[:, j].max()  # avoid single-class error
        else np.nan
        for j in range(n_class)
    ], dtype=float)

    # feature-selection footprint of this particular fit -- same three values
    # timecourse_decoding() already reports, so a fold's/test-set's own
    # selected-voxel count is visible next to its accuracy/AUC rather than
    # only ever showing up for the timecourse page
    n_selected = int(pipe.named_steps["feature_selection"].get_support().sum())

    # record model results
    xout = {
        'total_scores': ttl_score,
        'accuracy': acc_mx,  #acc_mx
        'evidence': evi_mx,  #evi_mx

        'auc': auc,

        'whole_voxels': n_features,
        'selected_voxels': n_selected,
        'feature_percent': 100 * n_selected / n_features,
    }

    return xout, impa_full


def build_cv_raw_results(pipe, data, labels, held_out_df: pd.DataFrame, regressor_categories: list,
                          feature_selection_cfg: dict, model_descr: str, fold_id: int) -> pd.DataFrame:
    """One row per held-out sample in a single cross-validation fold -- same
    raw-table style as timecourse_decoding()'s output (predicted_label,
    correct, evidence_<category>, feature-selection footprint), just without
    a window_index (this is one prediction per trial-sample, not a decode
    window). held_out_df carries that sample's own metadata as-is (task,
    trial_type, run, boldfile, etc., whatever training_df/master_spreadsheet
    provided), row-aligned with data/labels, so a user can see exactly which
    run contributed each held-out row for a given fold. run_kfold
    concatenates this across every fold into a per-subject cross-validation
    raw results table (model/<subject>_cv_results.csv)."""
    predictions = pipe.predict(data)
    evidence = decision_evidence(pipe, data)
    n_sel = int(pipe.named_steps["feature_selection"].get_support().sum())
    n_features = data.shape[1]
    code_to_label = {i + 1: cat for i, cat in enumerate(regressor_categories)}

    raw = held_out_df.reset_index(drop=True).copy()
    raw.insert(1, "model_descr", model_descr)
    raw.insert(2, "fold", fold_id)
    raw["correct"] = predictions == labels
    raw["predicted_label"] = [code_to_label.get(p, p) for p in predictions]
    for i, cat in enumerate(regressor_categories):
        raw[f"evidence_{cat}"] = evidence[:, i]
    raw["threshold_p"] = feature_selection_cfg.get("feat_p") if feature_selection_cfg.get("n_voxels") is None else np.nan
    raw["selected_voxels"] = n_sel
    raw["whole_voxels"] = n_features
    raw["feature_percent"] = 100 * n_sel / n_features

    evidence_cols = [c for c in raw.columns if c.startswith("evidence")]
    other_cols = [c for c in raw.columns if not c.startswith("evidence")]
    return raw[other_cols + evidence_cols]


def _partial_roc_auc_ovr(estimator, X, y):
    """One-vs-rest macro-average AUC, tolerant of a y that doesn't contain
    every class the estimator was fit on -- e.g. model_conditions.testing
    deliberately querying fewer categories than model_conditions.training
    (a real, structural mismatch, not just an occasional permutation
    artifact). Classes absent from y (or present as only one label, making
    AUC undefined) are skipped rather than raising, mirroring the per-class
    tolerance model_performance() already has. sklearn's built-in
    "roc_auc_ovr" scorer has no such tolerance -- it hard-requires y's class
    count to equal the estimator's fitted class count, which this fixed
    train/test partition can never satisfy when testing is a strict subset
    of training's categories."""
    proba = estimator.predict_proba(X)
    aucs = []
    for j, cls in enumerate(estimator.classes_):
        yj = (y == cls).astype(int)
        if yj.min() == yj.max():
            continue
        aucs.append(roc_auc_score(yj, proba[:, j]))
    return float(np.mean(aucs)) if aucs else np.nan


def permutation_significance(training_data, training_labels, testing_data, testing_labels, n_permutations, random_state,
                              feature_selection_cfg: dict, classifier_name: str, classifier_params: dict):
    """Real-vs-null significance for the held-out test evaluation, via
    sklearn.model_selection.permutation_test_score (the tool nilearn's own
    decoding docs recommend for exactly this fMRI-classification case).

    A PredefinedSplit with test_fold=-1 for every training row and =0 for
    every testing row encodes today's fixed train/test partition as a single
    CV fold. permutation_test_score then reshuffles the combined label
    vector and reruns that same fit/score structure -- including feature
    selection, since the Pipeline from model_classification() is refit fresh
    each round -- which is the textbook-correct way to build a null
    distribution for a fixed train/test split (not a naive shuffle done
    outside the fit/CV structure, which would be optimistic).

    One permutation_test_score call per metric: accuracy (sklearn's builtin
    scorer), and a one-vs-rest macro AUC via _partial_roc_auc_ovr (not
    sklearn's builtin "roc_auc_ovr" scorer -- that one raises whenever y's
    class count doesn't equal the estimator's fitted class count, which a
    testing section covering fewer categories than training hits on every
    single permutation round)."""
    X = np.vstack([training_data, testing_data])
    y = np.concatenate([training_labels, testing_labels])
    test_fold = np.concatenate([
        np.full(len(training_labels), -1),
        np.zeros(len(testing_labels)),
    ])
    cv = PredefinedSplit(test_fold)

    # same selection params the real model fit used (see
    # resolve_feature_selection_params), resolved once from the real
    # (unpermuted) training data and held fixed as a pipeline hyperparameter
    # across every permutation round
    mode, param = resolve_feature_selection_params(training_data, training_labels, feature_selection_cfg)

    rows = []
    for metric_name, scoring in (("accuracy", "accuracy"), ("roc_auc_ovr", _partial_roc_auc_ovr)):
        pipe = build_classifier_pipeline(mode, param, classifier_name, classifier_params)
        score, _, p_value = permutation_test_score(
            pipe, X, y, cv=cv, scoring=scoring,
            n_permutations=n_permutations, random_state=random_state, n_jobs=-1,
        )
        print(f"  permutation test [{metric_name}]: real={score:.4f}, p={p_value:.4g} ({n_permutations} permutations)")
        rows.append({"metric": metric_name, "real_score": score, "p_value": p_value, "n_permutations": n_permutations})

    return pd.DataFrame(rows)


# =====================================================
# Timecourse decoding
# =====================================================

# grouping used for the timecourse decoding output -- the relative timepoint
# within each event's decode window, crossed with the classification label.
TIMECOURSE_GROUPING = ["window_index", "regressor_label"]


def timecourse_decoding(pipe, timecourse_data, timecourse_labels, timecourse_df, regressor_categories,
                         feature_selection_cfg: dict, subject_id: str, model_descr: str,
                         boldfile_to_pipe: dict = None):
    """Predict the trained classifier on every already-recomputed timecourse-decoding
    volume. Returns (raw, summary):
      - raw: one row per volume actually decoded, with its own prediction and
        evidence_<category> columns -- the actual per-TR data, not an average.
      - summary: raw grouped by (window_index, regressor_label) and averaged across
        every trial sharing that group -- the confusion-style timecourse view.

    boldfile_to_pipe, if given, maps a specific boldfile to an alternate fitted
    pipe that should decode that boldfile's rows instead of `pipe` --
    mvpa_workflow.py uses this to substitute each overlapping run's own
    held-out k-fold classifier when a run appears in both
    model_conditions.training and model_conditions.timecourse_decoding, so
    that run is never decoded by a classifier that was partly trained on it
    (see README.md's double-dipping warning). Rows whose boldfile isn't in
    the mapping (or when the mapping is omitted entirely) are decoded with
    `pipe` as before. Keyed by boldfile rather than run number, since two
    different tasks can reuse the same run number for genuinely different
    scans -- boldfile is the only unambiguous "same scan" identifier."""

    n_rows = timecourse_data.shape[0]
    if boldfile_to_pipe:
        boldfiles = timecourse_df["boldfile"].to_list()
        pipe_for_row = [boldfile_to_pipe.get(bf, pipe) for bf in boldfiles]
    else:
        pipe_for_row = [pipe] * n_rows

    # same dtype as the (numeric-coded) labels -- not dtype=object, which would
    # make accuracy_score's type_of_target see "unknown" instead of "binary"/
    # "multiclass" and reject the binary/multiclass comparison below
    predictions = np.empty(n_rows, dtype=timecourse_labels.dtype)
    evidence = np.zeros((n_rows, len(regressor_categories)))
    selected_voxels = np.zeros(n_rows, dtype=int)
    whole_voxels = np.zeros(n_rows, dtype=int)

    # dispatch each row to whichever pipe should decode it -- distinct pipe
    # objects are grouped (strictly by identity, not sklearn equality, which
    # Pipeline doesn't define anyway) into one batched predict()/
    # decision_evidence() call apiece, rather than one call per row
    for this_pipe in {id(p): p for p in pipe_for_row}.values():
        idx = np.array([i for i, p in enumerate(pipe_for_row) if p is this_pipe])
        sub_data = timecourse_data[idx]
        predictions[idx] = this_pipe.predict(sub_data)
        evidence[idx, :] = decision_evidence(this_pipe, sub_data)
        n_sel = int(this_pipe.named_steps["feature_selection"].get_support().sum())
        selected_voxels[idx] = n_sel
        whole_voxels[idx] = sub_data.shape[1]

    global_accuracy = accuracy_score(timecourse_labels, predictions)
    print(f"Global accuracy: {global_accuracy:.4f}")

    code_to_label = {i + 1: cat for i, cat in enumerate(regressor_categories)}

    raw = timecourse_df.reset_index(drop=True).copy()
    raw["predicted_label"] = [code_to_label.get(p, p) for p in predictions]
    raw["correct"] = predictions == timecourse_labels
    for i, cat in enumerate(regressor_categories):
        raw[f"evidence_{cat}"] = evidence[:, i]

    # only meaningful in "fpr" (p-threshold) mode -- NaN in "k_best" (n_voxels)
    # mode, where selected_voxels already says everything there is to say
    raw["threshold_p"] = feature_selection_cfg.get("feat_p") if feature_selection_cfg.get("n_voxels") is None else np.nan
    raw["selected_voxels"] = selected_voxels
    raw["whole_voxels"] = whole_voxels
    raw["feature_percent"] = 100 * selected_voxels / whole_voxels

    evidence_cols = [c for c in raw.columns if c.startswith("evidence")]
    other_cols = [c for c in raw.columns if not c.startswith("evidence")]
    raw = raw[other_cols + evidence_cols]
    raw.insert(1, "model_descr", model_descr)  # "subject" is already a column, from timecourse_df's own BIDS entity

    summary = summarize_decoding(raw, regressor_categories, subject_id, model_descr)

    return raw, summary


def summarize_decoding(raw: pd.DataFrame, regressor_categories: list, subject_id: str, model_descr: str) -> pd.DataFrame:
    """Collapse a raw (one-row-per-decoded-TR) decoding table down to one row per
    (window_index, regressor_label), averaging Accuracy/evidence across every trial
    sharing that group."""
    rows = []
    for (window_index, regressor_label), group in raw.groupby(TIMECOURSE_GROUPING, sort=False):
        row = {
            "subject": subject_id,
            "model_descr": model_descr,
            "window_index": window_index,
            "regressor_label": regressor_label,
            "trial_count": len(group),
            "Accuracy": group["correct"].mean(),
        }
        for cat in regressor_categories:
            row[f"evidence_{cat}"] = group[f"evidence_{cat}"].mean()
        row["threshold_p"] = group["threshold_p"].mean()
        row["selected_voxels"] = group["selected_voxels"].mean()
        row["whole_voxels"] = group["whole_voxels"].mean()
        row["feature_percent"] = group["feature_percent"].mean()
        rows.append(row)

    summary = pd.DataFrame(rows)
    evidence_cols = [c for c in summary.columns if c.startswith("evidence")]
    other_cols = [c for c in summary.columns if not c.startswith("evidence")]
    return summary[other_cols + evidence_cols]
