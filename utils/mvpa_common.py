#!/usr/bin/env python3
"""
Shared utilities for the mvpa_banich toolchain: BIDS filename parsing, the
onset/duration -> BOLD volume-range math, the model_conditions query DSL, and
the config-loading/classification/decoding primitives used by both
mvpa_generalization_workflow.py (independent train/test) and
mvpa_kfold_workflow.py (same-task, split-by-run k-fold) -- also used directly
by generate_master_spreadsheet.py and validate_model_config.py.
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
WINDOW_REFERENCES = {"onset", "offset_end"}


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


# =====================================================
# Small shared helpers (no dependency on any script's module-level state --
# safe to import from mvpa_generalization_workflow.py, generate_report.py, or anywhere else)
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


def build_timecourse_instructions(labeled_df: pd.DataFrame, window: dict) -> pd.DataFrame:
    """One row per source event in labeled_df (grouped by boldfile+trial_index),
    re-expanded into fresh volume_of_interest rows per `window`, tagged with a
    window_index (position within that event's recomputed window)."""

    rows = []
    for (boldfile, trial_index), group in labeled_df.groupby(["boldfile", "trial_index"], sort=False):
        first = group.iloc[0]
        tr, n_frames = get_bold_header_info(boldfile)

        start_time, stop_time = resolve_window_times(window, first["onset"], first["duration"])
        start_vol, stop_vol = compute_volume_range(start_time, stop_time, tr, n_frames)

        for vol in range(start_vol, stop_vol):
            rows.append({
                "subject": first["subject"],
                "session": first["session"],
                "task": first["task"],
                "run": first["run"],
                "trial_type": first["trial_type"],
                "trial_index": trial_index,
                "regressor_label": first["regressor_label"],
                "regressor": first["regressor"],
                "boldfile": boldfile,
                "volume_of_interest": vol,
                "window_index": vol - start_vol,
            })

    return pd.DataFrame(rows)


# =====================================================
# Output saving / aggregation
# =====================================================

def save_model_results(output_pattern, results, categories):
    """
    Save a dict of model outputs to disk, one file per metric.

    Parameters
    ----------
    output_pattern : str
        Format string used to build output filenames. Must include a '{metric}'
        placeholder, e.g.:
            "/path/to/out/{metric}.csv"
            "/path/to/out/sub-01_run-2_{metric}.csv"

    results : dict[str, array-like]
        Mapping from metric name -> value to save. Supported value shapes:

        1) Square matrix (C, C)
           - Interpreted as a class-by-class matrix (e.g., confusion matrix,
             importance matrix).
           - Saved as a CSV with row/column labels from `categories`.

        2) Column vector (C,) or (C, 1)
           - Interpreted as one value per category/class.
           - Saved as a single-column CSV indexed by `categories`.

        3) Anything else (e.g., (S, E), (n_features,), scalar)
           - Saved via np.savetxt as numeric CSV (no labels).
           - Scalars are promoted to 1D.

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

    for metric, x in results.items():

        # Build output path for this metric and ensure parent directory exists
        output_file = output_pattern.format(metric=metric)
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        # Coerce to numpy array (without forcing extra dims yet)
        x = np.asarray(x)

        # Promote scalars to shape (1,) so savetxt works
        if x.ndim == 0:
            x = np.atleast_1d(x)

        # Case 1: category-by-category matrix
        if x.shape == (C, C):
            df = pd.DataFrame(x, index=categories, columns=categories)
            df.to_csv(output_file, index=True)

        # Case 2: one value per category (accept (C,) or (C,1))
        elif x.shape == (C,) or x.shape == (C, 1):
            x_vec = x.reshape(C)  # ensures 1D length-C
            df = pd.DataFrame({metric: x_vec}, index=categories)
            df.to_csv(output_file, index=True)

        # Case 3: everything else (no labels)
        else:
            np.savetxt(output_file, x, delimiter=",", fmt="%.6f")

        print(f"[{metric}] saved -> {output_file} (shape={x.shape})")


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


def model_performance(pipe, testing_data, testing_labels):

    print("Testing model performance...")

    clf = pipe.named_steps["classifier"]
    xfeat = pipe.named_steps["feature_selection"].get_support()

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

    # special case where classifier is binary (yes/no) -- only codes one label
    if n_class == 2:

        # voxel weights
        impa = np.vstack((clf.coef_, -clf.coef_))
        ## important volume 0 and volume 1 are mat*-1 of eachother.. Compute 1 tail ttests always
        print(impa.shape)

        # evidence: sigmoid on decision function → class-1 prob; other is 1-p
        d = pipe.decision_function(testing_data)
        p1 = 1.0 / (1.0 + np.exp(-d))
        p0 = 1.0 - p1
        xevi = np.vstack([p0, p1]).T

    else:

        # voxel weights
        impa = clf.coef_

        # evidence: multinomial OV(A)R decision_function → pass through sigmoid per class
        d = pipe.decision_function(testing_data)  # shape: (n, n_class)
        xevi = 1.0 / (1.0 + np.exp(-d))

    # store importance values in original dataformat
    impa_full = np.zeros((n_class, n_features), dtype=impa.dtype)
    impa_full[:, xfeat] = impa

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

    # record model results
    xout = {
        'total_scores': ttl_score,
        'accuracy': acc_mx,  #acc_mx
        'evidence': evi_mx,  #evi_mx

        'auc': auc
    }

    return xout, impa_full


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

    One permutation_test_score call per metric (accuracy, roc_auc_ovr --
    the same one-vs-rest convention model_performance()'s own AUC already
    uses, just via sklearn's built-in scorer instead of a second hand-rolled
    computation)."""
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
    for metric in ("accuracy", "roc_auc_ovr"):
        pipe = build_classifier_pipeline(mode, param, classifier_name, classifier_params)
        score, _, p_value = permutation_test_score(
            pipe, X, y, cv=cv, scoring=metric,
            n_permutations=n_permutations, random_state=random_state, n_jobs=-1,
        )
        print(f"  permutation test [{metric}]: real={score:.4f}, p={p_value:.4g} ({n_permutations} permutations)")
        rows.append({"metric": metric, "real_score": score, "p_value": p_value, "n_permutations": n_permutations})

    return pd.DataFrame(rows)


# =====================================================
# Timecourse decoding
# =====================================================

# grouping used for the timecourse decoding output -- the relative timepoint
# within each event's decode window, crossed with the classification label.
TIMECOURSE_GROUPING = ["window_index", "regressor_label"]


def timecourse_decoding(pipe, timecourse_data, timecourse_labels, timecourse_df, regressor_categories,
                         feature_selection_cfg: dict, subject_id: str, model_descr: str):
    """Predict the trained classifier on every already-recomputed timecourse-decoding
    volume. Returns (raw, summary):
      - raw: one row per volume actually decoded, with its own prediction and
        evidence_<category> columns -- the actual per-TR data, not an average.
      - summary: raw grouped by (window_index, regressor_label) and averaged across
        every trial sharing that group -- the confusion-style timecourse view."""

    # pipeline applies feature selection internally -- full (unsliced) data goes in
    predictions = pipe.predict(timecourse_data)
    global_accuracy = accuracy_score(timecourse_labels, predictions)
    print(f"Global accuracy: {global_accuracy:.4f}")

    evidence = decision_evidence(pipe, timecourse_data)

    code_to_label = {i + 1: cat for i, cat in enumerate(regressor_categories)}

    raw = timecourse_df.reset_index(drop=True).copy()
    raw["predicted_label"] = [code_to_label.get(p, p) for p in predictions]
    raw["correct"] = predictions == timecourse_labels
    for i, cat in enumerate(regressor_categories):
        raw[f"evidence_{cat}"] = evidence[:, i]

    n_selected = int(pipe.named_steps["feature_selection"].get_support().sum())
    # only meaningful in "fpr" (p-threshold) mode -- NaN in "k_best" (n_voxels)
    # mode, where selected_voxels already says everything there is to say
    raw["threshold_p"] = feature_selection_cfg.get("feat_p") if feature_selection_cfg.get("n_voxels") is None else np.nan
    raw["selected_voxels"] = n_selected
    raw["whole_voxels"] = timecourse_data.shape[1]
    raw["feature_percent"] = 100 * n_selected / timecourse_data.shape[1]

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
