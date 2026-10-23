#!/usr/bin/env python3
"""
Optional step between generate_master_spreadsheet.py and mvpa_workflow.py:
force every subject's TRAINING trial set to be balanced across the conditions
of interest, and write the result as a trial list that
model_conditions.training.trial_restriction points at.

Nothing is rewritten: the events.tsv files and the master spreadsheet stay
exactly as they are. The balanced selection lives in a small separate CSV; the
workflow reads it only when the config names it, and uses it for training
(and so model.kfold_cv) only -- testing and timecourse decoding are left
unbalanced.

WHAT IT DOES (per subject, over all of that subject's runs together)

  The supplementary restriction config (--restriction-config, see
  examples/config-restriction.example.json) defines
    primary_groups    name -> query   e.g. face / place / item
    secondary_groups  name -> query   e.g. positive / negative   (optional)
  Queries are the same column/match/and/or/not trees used for
  model_conditions, evaluated against each trial (one retained events.tsv
  row -- administrative rows like fixation are never trials) with the usual
  events.tsv columns plus subject/session/task/run/boldfile/eventfile.
  Overlap is resolved first-match-wins in config order, as everywhere else.
  A trial matching no primary group is not governed and is listed unchanged.

  1. Trial counts. Trials are dropped at random until every primary group has
     the exact same number of trials. If secondary groups are given, that
     number is also split equally across the secondary groups inside every
     primary group (36 face = 18 pos + 18 neg, 36 place = 18 + 18, 12 item
     -> 12 face = 6 + 6, 12 place = 6 + 6, 12 item). The largest balanced
     set that exists is kept. `secondary_applies_to` can limit which
     primary groups get the secondary split (default: all of them).
     A trial in a secondary-split primary group that matches no secondary
     group cannot be balanced and is dropped.
  2. TR counts. Every kept trial is limited to the TR count of the shortest
     kept trial in that subject: the FIRST n TRs of its window are used and
     any extra TRs are cut off the end. TRs are counted the way training
     selects them (compute_volume_range over [onset + hemodynamic_lag,
     +duration)), so --config (or --hemodynamic-lag) must match the analysis
     config.

  With "balance_trial_counts": false in the restriction config, step 1 is
  skipped: no trial is dropped to equalize counts and only step 2 (the common TR
  count) is applied, so primary_groups only say which trials are governed.

  If the analysis config sets model_conditions.training.extend_events_until
  (pass --config), a trial's window is the event's own duration stretched to
  the onset of the next event matching that query (e.g. operation + ITI up to
  the next trial's cue), exactly as training selects it; a trial with no
  following match (the last of a run) has an open-ended window that runs to the end of the scan: it cannot set the shared
  TR count, it is kept if the scan leaves it at least that many TRs, and dropped (open_window_too_short) if not.

  A subject for whom no balanced set exists (a primary or secondary group is
  empty) is EXCLUDED -- it has no rows in the trial list, so mvpa_workflow.py
  refuses to run it rather than silently using an unbalanced set.

OUTPUTS (next to the input spreadsheet unless --output is given)
    <stem>_trial_restriction.csv  the list model_conditions.training.trial_restriction
                                  points at: one row per trial training may use
                                  (kept governed trials with their n_trs, and
                                  ungoverned trials with n_trs blank = whole window)
    <stem>_trial_restriction_summary.csv
                                  per subject x primary x secondary group: trials
                                  and TRs before and after, trs per trial
    <stem>_trial_restriction_audit.csv
                                  every governed trial and whether it was kept,
                                  with the reason if not

n_trs counts are volumes training/testing would select (lag-shifted window).
Dropped trials are chosen with a seeded generator (config `seed`, then the
subject ID), so the same inputs always give the same list. The list depends on
hemodynamic_lag and on the group definitions (the lag is stored in it and
checked by the workflow): rerun it if either, or the events files, change.

Usage:
    python restrict_master_spreadsheet.py --master-spreadsheet master_spreadsheet.csv \\
        --restriction-config restriction.json --config mvpa_config.json
"""

import argparse
import json
import math
import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for utils.mvpa_common
from utils.mvpa_common import (
    compute_volume_range, evaluate_query_node, extended_window_durations, get_bold_header_info,
    label_conditions_with_lag, load_retained_events, validate_query_node,
)

# the master spreadsheet's per-volume columns; every other column is a
# per-boldfile constant (subject/session/task/run/boldfile/eventfile/BIDS extras)
FRAME_COLUMNS = ("volume_of_interest", "trial_type", "onset", "duration", "event_index")
DROP_STRATEGIES = ("random", "spread_runs")

# trial status values in the audit table
KEPT = "kept"
DROPPED_OUTSIDE_SCAN = "outside_scan"            # lagged window falls entirely past the last volume
DROPPED_OPEN_SHORT = "open_window_too_short"     # extend_events_until is set, nothing follows the trial (last trial of a run) and the
                                                 # scan ends before it has the subject's common number of TRs
DROPPED_NO_SECONDARY = "no_secondary_label"      # in a secondary-split primary group, matches no secondary group
DROPPED_BALANCE = "balance_drop"                 # randomly dropped to equalize counts
EXCLUDED_SUBJECT = "subject_excluded"            # the subject has no balanced set at all


# =====================================================
# Config
# =====================================================

def load_restriction_config(path: str) -> dict:
    """The restriction JSON may be the bare object or wrapped in a top-level
    "restriction" key (so the same keys can also sit inside a main config)."""
    with open(path) as f:
        cfg = json.load(f)
    return cfg.get("restriction", cfg) if isinstance(cfg.get("restriction"), dict) else cfg


def validate_restriction_config(cfg: dict) -> list:
    """Returns a list of problems (empty if the config is usable)."""
    errors = []
    allowed = {"seed", "drop_strategy", "primary_groups", "secondary_groups", "secondary_applies_to", "balance_trial_counts",
               "notes", "created_by", "config_version"}
    for key in cfg:
        if key not in allowed:
            errors.append(f"unknown key {key!r} (expected: {sorted(allowed)})")

    primary = cfg.get("primary_groups")
    if not isinstance(primary, dict) or not primary:
        errors.append("primary_groups: must be a non-empty object of {name: query}")
        primary = {}
    secondary = cfg.get("secondary_groups", {})
    if not isinstance(secondary, dict):
        errors.append("secondary_groups: must be an object of {name: query}")
        secondary = {}

    for section, groups in (("primary_groups", primary), ("secondary_groups", secondary)):
        for name, query in groups.items():
            if not isinstance(name, str) or not name:
                errors.append(f"{section}: group names must be non-empty strings")
            errors.extend(validate_query_node(query, None, f"{section}[{name!r}]"))

    applies = cfg.get("secondary_applies_to")
    if applies is not None:
        if not secondary:
            errors.append("secondary_applies_to is set but secondary_groups is empty")
        if not isinstance(applies, list) or not all(isinstance(a, str) for a in applies):
            errors.append("secondary_applies_to: must be a list of primary group names")
        else:
            for a in applies:
                if a not in primary:
                    errors.append(f"secondary_applies_to: {a!r} is not a primary group")
    if "balance_trial_counts" in cfg and not isinstance(cfg["balance_trial_counts"], bool):
        errors.append("balance_trial_counts: must be true or false")
    if "seed" in cfg and not isinstance(cfg["seed"], int):
        errors.append("seed: must be an integer")
    if cfg.get("drop_strategy", "random") not in DROP_STRATEGIES:
        errors.append(f"drop_strategy: must be one of {list(DROP_STRATEGIES)}, got {cfg.get('drop_strategy')!r}")
    return errors


def groups_using_secondary(cfg: dict) -> set:
    if not cfg.get("secondary_groups"):
        return set()
    applies = cfg.get("secondary_applies_to")
    return set(cfg["primary_groups"]) if applies is None else set(applies)


# =====================================================
# Trials
# =====================================================

def first_match_labels(queries: dict, events: pd.DataFrame, what: str) -> pd.Series:
    """Label each event with the first query (config order) it matches; NA for none."""
    labels = pd.Series(pd.NA, index=events.index, dtype=object)
    for name, query in queries.items():
        try:
            mask = evaluate_query_node(query, events)
        except KeyError as exc:
            raise SystemExit(
                f"{what} {name!r}: query references column {exc.args[0]!r}, which this run's events don't have "
                f"(available: {sorted(events.columns)})"
            )
        labels[mask & labels.isna()] = name
    return labels


def collect_trials(master: pd.DataFrame, cfg: dict, hemodynamic_lag: float, extend_until: dict = None) -> pd.DataFrame:
    """One row per retained event (trial) of every run in the master spreadsheet,
    with its primary/secondary group (NA if none) and the number of volumes
    training/testing would select for it (n_trs). With `extend_until`
    (model_conditions.training.extend_events_until) a trial's window runs to the
    next matching event; one with none after it (`open_window` True) runs to the end of
    the scan, so its n_trs is just what the scan leaves."""
    rows = []
    for boldfile, group in master.groupby("boldfile", sort=False):
        first = group.iloc[0]
        extra_cols = {c: first[c] for c in group.columns if c not in FRAME_COLUMNS}
        if not os.path.isfile(first["eventfile"]):
            raise SystemExit(f"eventfile referenced by the master spreadsheet does not exist: {first['eventfile']}")
        events = load_retained_events(first["eventfile"], extra_cols, source_row_col="source_row")
        if events.empty:
            continue
        events["primary"] = first_match_labels(cfg["primary_groups"], events, "primary_groups")
        events["secondary"] = first_match_labels(cfg.get("secondary_groups", {}), events, "secondary_groups")

        tr, n_frames = get_bold_header_info(boldfile)
        window = extended_window_durations(events, extend_until)
        events["open_window"] = window.isna()
        n_trs = []
        for onset, duration in zip(events["onset"], window):
            if pd.isna(duration):
                duration = n_frames * tr                  # open-ended: to the end of the scan
            start = onset + hemodynamic_lag
            lo, hi = compute_volume_range(start, start + duration, tr, n_frames)
            n_trs.append(hi - lo)
        events["n_trs"] = n_trs
        events["tr"] = tr
        rows.append(events)

    if not rows:
        return pd.DataFrame(columns=["subject", "boldfile", "eventfile", "trial_index", "source_row", "primary", "secondary", "n_trs", "tr",
                                     "open_window"])
    return pd.concat(rows, ignore_index=True)


# =====================================================
# Selection
# =====================================================

def _lcm(values) -> int:
    out = 1
    for v in values:
        out = out * v // math.gcd(out, v)
    return out


def choose_kept(candidates: pd.DataFrame, n_keep: int, strategy: str, rng: np.random.Generator) -> list:
    """Row labels (of `candidates`) to keep, n_keep of them, without replacement.
    "random": uniform. "spread_runs": drop one at a time from whichever run
    currently holds the most of them (ties random), so the survivors stay as
    evenly spread over runs as the data allows -- keeps per-run k-fold usable."""
    labels = list(candidates.index)
    if n_keep >= len(labels):
        return labels
    if strategy == "random":
        return list(rng.choice(labels, size=n_keep, replace=False))

    by_run = {}
    for lab, ef in zip(candidates.index, candidates["eventfile"]):
        by_run.setdefault(ef, []).append(lab)
    for ef in by_run:
        rng.shuffle(by_run[ef])
    remaining = len(labels)
    while remaining > n_keep:
        biggest = max(len(v) for v in by_run.values())
        tied = sorted(ef for ef, v in by_run.items() if len(v) == biggest)
        by_run[tied[int(rng.integers(len(tied)))]].pop()
        remaining -= 1
    return [lab for v in by_run.values() for lab in v]


def common_trs(trials: pd.DataFrame, status: pd.Series, is_open: pd.Series) -> int:
    """The shared TR count of one subject's kept trials: the shortest kept trial whose window ends at a real
    following event. A kept trial with an OPEN-ENDED window (nothing follows it: its window runs to the end of
    the scan) cannot set that minimum -- it is cut to the shared count if the scan leaves it enough TRs, and
    otherwise dropped here (status open_window_too_short). Edits `status` in place."""
    kept = status == KEPT
    closed = kept & ~is_open
    trs = int((trials.loc[closed, "n_trs"] if closed.any() else trials.loc[kept, "n_trs"]).min())
    too_short = kept & is_open & (trials["n_trs"] < trs)
    status[too_short] = DROPPED_OPEN_SHORT
    return trs


def plan_subject(trials: pd.DataFrame, cfg: dict, subject: str) -> dict:
    """Decide which of one subject's trials to keep.

    Returns {"status": per-trial status Series (aligned with `trials`),
             "included": bool, "reason": str, "n_per_primary": int,
             "trs_per_trial": int|None}."""
    primary_names = list(cfg["primary_groups"])
    secondary_names = list(cfg.get("secondary_groups", {}))
    split = groups_using_secondary(cfg)
    strategy = cfg.get("drop_strategy", "random")
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg.get("seed", 0)), zlib.crc32(str(subject).encode())]))

    status = pd.Series(pd.NA, index=trials.index, dtype=object)
    governed = trials["primary"].notna()

    is_open = trials["open_window"].astype(bool) if "open_window" in trials else pd.Series(False, index=trials.index)
    status[governed & (trials["n_trs"] < 1)] = DROPPED_OUTSIDE_SCAN
    needs_sec = trials["primary"].isin(split)
    status[governed & status.isna() & needs_sec & trials["secondary"].isna()] = DROPPED_NO_SECONDARY
    eligible = governed & status.isna()

    def exclude(reason):
        out = status.copy()
        out[governed] = EXCLUDED_SUBJECT
        return {"status": out, "included": False, "reason": reason, "n_per_primary": 0, "trs_per_trial": None}

    if cfg.get("balance_trial_counts", True) is False:
        # no count balancing: every usable governed trial is kept, only the common TR count is applied
        if not eligible.any():
            return exclude("no usable trials")
        status[eligible] = KEPT
        trs = common_trs(trials, status, is_open)
        return {"status": status, "included": True, "reason": "", "n_per_primary": 0, "trs_per_trial": trs}

    # trials available per (primary, bucket); bucket is None for an unsplit primary group
    buckets = {p: (secondary_names if p in split else [None]) for p in primary_names}
    counts = {}
    for p in primary_names:
        for b in buckets[p]:
            sel = eligible & (trials["primary"] == p)
            if b is not None:
                sel &= trials["secondary"] == b
            counts[(p, b)] = int(sel.sum())
    empty = [f"{p}" + (f"/{b}" if b else "") for (p, b), n in counts.items() if n == 0]
    if empty:
        return exclude(f"no usable trials in: {', '.join(empty)}")

    # the most trials per primary group that every group can supply, with the
    # secondary groups inside it equal: group p holds k_p buckets of m trials,
    # so it can reach k_p * min(bucket counts); all primary groups must land on
    # the same total N, and N must divide into k_p equal buckets for every p
    k = {p: len(buckets[p]) for p in primary_names}
    cap = min(k[p] * min(counts[(p, b)] for b in buckets[p]) for p in primary_names)
    step = _lcm(k.values())
    n_per_primary = (cap // step) * step
    if n_per_primary == 0:
        return exclude(f"too few trials to balance: smallest cap {cap} < {step} "
                       f"(needed one trial per secondary group in every primary group)")

    kept = []
    for p in primary_names:
        for b in buckets[p]:
            sel = eligible & (trials["primary"] == p)
            if b is not None:
                sel &= trials["secondary"] == b
            kept += choose_kept(trials[sel], n_per_primary // k[p], strategy, rng)

    status[eligible] = DROPPED_BALANCE
    status[kept] = KEPT
    trs = common_trs(trials, status, is_open)
    return {"status": status, "included": True, "reason": "", "n_per_primary": n_per_primary, "trs_per_trial": trs}


# =====================================================
# Orchestration
# =====================================================

def restrict_trials(master: pd.DataFrame, cfg: dict, hemodynamic_lag: float, extend_until: dict = None):
    """Returns (trial list, summary table, audit table)."""
    trials = collect_trials(master, cfg, hemodynamic_lag, extend_until)
    trials["status"] = pd.NA
    trials["subject"] = trials["subject"].astype(str)

    primary_names = list(cfg["primary_groups"])
    secondary_names = list(cfg.get("secondary_groups", {}))
    split = groups_using_secondary(cfg)
    cells = [(p, b) for p in primary_names for b in (secondary_names if p in split else [None])]

    plans, listed = {}, []
    for subject in sorted(master["subject"].astype(str).unique()):
        sub_idx = trials.index[trials["subject"] == subject]
        plan = plan_subject(trials.loc[sub_idx], cfg, subject)
        plans[subject] = plan
        trials.loc[sub_idx, "status"] = plan["status"]
        if not plan["included"]:
            continue
        t = trials.loc[sub_idx]
        usable = t[t["primary"].isna() | (t["status"] == KEPT)].copy()
        # governed trials are capped at the subject's common TR count; ungoverned ones keep their whole window
        usable["n_trs"] = np.where(usable["primary"].notna(), plan["trs_per_trial"], np.nan)
        listed.append(usable)

    columns = ["subject", "boldfile", "trial_index", "onset", "trial_type", "primary_group", "secondary_group", "n_trs", "hemodynamic_lag"]
    if listed:
        out = pd.concat(listed, ignore_index=True).rename(columns={"primary": "primary_group", "secondary": "secondary_group"})
        out["hemodynamic_lag"] = hemodynamic_lag
        out = out[columns].sort_values(["subject", "boldfile", "trial_index"], kind="stable").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=columns)

    summary = build_summary(trials, plans, cells, split)
    audit = trials.loc[trials["primary"].notna(), [
        "subject", "boldfile", "eventfile", "trial_index", "source_row", "trial_type", "onset", "duration", "n_trs",
        "primary", "secondary", "status",
    ]].rename(columns={"primary": "primary_group", "secondary": "secondary_group"})
    return out, summary, audit


def build_summary(trials: pd.DataFrame, plans: dict, cells: list, split: set) -> pd.DataFrame:
    rows = []
    for subject, plan in plans.items():
        t = trials[trials["subject"] == subject]
        status = "included" if plan["included"] else "excluded"
        common = {"subject": subject, "status": status, "exclusion_reason": plan["reason"]}
        trs = plan["trs_per_trial"] if plan["included"] else np.nan
        sub_rows = []
        for p, b in cells:
            sel = t["primary"] == p
            if b is not None:
                sel &= t["secondary"] == b
            n_kept = int((sel & (t["status"] == KEPT)).sum())
            sub_rows.append({
                **common, "primary_group": p, "secondary_group": b if b is not None else "",
                "n_trials_original": int(sel.sum()), "n_trials_restricted": n_kept,
                "trs_per_trial_restricted": trs,
                "n_trs_original": int(t.loc[sel, "n_trs"].sum()),
                "n_trs_restricted": n_kept * int(plan["trs_per_trial"] or 0),
            })
        # trials a secondary-split primary group claimed but no secondary group matched
        unlabeled = t["primary"].isin(split) & t["secondary"].isna()
        for p in sorted(t.loc[unlabeled, "primary"].unique()):
            sel = (t["primary"] == p) & t["secondary"].isna()
            sub_rows.append({
                **common, "primary_group": p, "secondary_group": "(no secondary match)",
                "n_trials_original": int(sel.sum()), "n_trials_restricted": 0, "trs_per_trial_restricted": np.nan,
                "n_trs_original": int(t.loc[sel, "n_trs"].sum()), "n_trs_restricted": 0,
            })
        rows += sub_rows
        rows.append({
            **common, "primary_group": "(all)", "secondary_group": "",
            "n_trials_original": sum(x["n_trials_original"] for x in sub_rows),
            "n_trials_restricted": sum(x["n_trials_restricted"] for x in sub_rows),
            "trs_per_trial_restricted": trs,
            "n_trs_original": sum(x["n_trs_original"] for x in sub_rows),
            "n_trs_restricted": sum(x["n_trs_restricted"] for x in sub_rows),
        })
    return pd.DataFrame(rows)


def find_ungoverned_condition_trials(master: pd.DataFrame, trials: pd.DataFrame, model_conditions: dict, hemodynamic_lag: float,
                                     extend_until: dict = None) -> list:
    """Trials that model_conditions.training selects but no primary group
    governs, so they stay unbalanced: [(section, condition, n_ungoverned,
    n_total)], only for conditions with n_ungoverned > 0. The usual cause is a
    primary-group query narrower than the condition it is meant to balance
    (e.g. missing the condition's run restriction)."""
    governed = set(zip(trials.loc[trials["primary"].notna(), "boldfile"], trials.loc[trials["primary"].notna(), "trial_index"]))
    out = []
    for section in ("training",):
        conditions = (model_conditions.get(section) or {}).get("conditions") or {}
        if not conditions:
            continue
        labeled = label_conditions_with_lag(master, conditions, hemodynamic_lag, extend_until=extend_until)
        if labeled.empty:
            continue
        per_trial = labeled.drop_duplicates(["regressor_label", "boldfile", "trial_index"])
        per_trial = per_trial.assign(governed=[(b, i) in governed for b, i in zip(per_trial["boldfile"], per_trial["trial_index"])])
        for name, g in per_trial.groupby("regressor_label", sort=False):
            n_un = int((~g["governed"]).sum())
            if n_un:
                out.append((section, name, n_un, len(g)))
    return out


def default_paths(master_path: str):
    stem, _ = os.path.splitext(master_path)
    return f"{stem}_trial_restriction.csv", f"{stem}_trial_restriction_summary.csv", f"{stem}_trial_restriction_audit.csv"


def print_report(summary: pd.DataFrame):
    subjects = summary[summary["primary_group"] == "(all)"]
    n_in = int((subjects["status"] == "included").sum())
    print(f"\n{n_in} of {len(subjects)} subject(s) have a balanced trial set.")
    for _, s in subjects.iterrows():
        if s["status"] == "included":
            print(f"  {s['subject']}: {int(s['n_trials_original'])} -> {int(s['n_trials_restricted'])} governed trials, "
                  f"{int(s['trs_per_trial_restricted'])} TR(s)/trial, {int(s['n_trs_restricted'])} TRs")
        else:
            print(f"  (!) {s['subject']}: EXCLUDED -- {s['exclusion_reason']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-spreadsheet", required=True, help="master_spreadsheet.csv from generate_master_spreadsheet.py (read only)")
    parser.add_argument("--restriction-config", required=True, help="supplementary JSON defining primary_groups/secondary_groups")
    parser.add_argument("--config", default=None, help="the mvpa config: event_extraction.hemodynamic_lag is read from it, "
                                                       "and model_conditions.training is checked against the groups")
    parser.add_argument("--hemodynamic-lag", type=float, default=None, help="override/replace --config's event_extraction.hemodynamic_lag (seconds)")
    parser.add_argument("--output", default=None, help="trial list path (default: <master stem>_trial_restriction.csv)")
    args = parser.parse_args()

    cfg = load_restriction_config(args.restriction_config)
    errors = validate_restriction_config(cfg)
    if errors:
        raise SystemExit("Invalid restriction config:\n  " + "\n  ".join(errors))

    main_cfg = {}
    if args.config:
        with open(args.config) as f:
            main_cfg = json.load(f)
    lag = args.hemodynamic_lag
    if lag is None:
        if not args.config:
            raise SystemExit("pass --config (reads event_extraction.hemodynamic_lag) or --hemodynamic-lag: TR counts depend on the lag")
        lag = main_cfg.get("event_extraction", {}).get("hemodynamic_lag", 0)
    lag = float(lag)

    out_csv, summary_csv, audit_csv = default_paths(args.master_spreadsheet)
    if args.output:
        out_csv = args.output
        out_stem, _ = os.path.splitext(args.output)
        summary_csv, audit_csv = f"{out_stem}_summary.csv", f"{out_stem}_audit.csv"

    master = pd.read_csv(args.master_spreadsheet, dtype=str, keep_default_na=False)
    print(f"Restricting {args.master_spreadsheet} ({master['subject'].nunique()} subject(s)), hemodynamic_lag={lag}")

    extend_until = (main_cfg.get("model_conditions", {}).get("training") or {}).get("extend_events_until")
    if extend_until:
        print(f"Windows follow model_conditions.training.extend_events_until: each trial runs to the next matching event")
    trial_list, summary, audit = restrict_trials(master, cfg, lag, extend_until)
    summary.to_csv(summary_csv, index=False)
    audit.to_csv(audit_csv, index=False)
    print_report(summary)
    if trial_list.empty:
        raise SystemExit("No subject has a balanced trial set -- no trial list written (summary and audit saved).")

    if main_cfg.get("model_conditions"):
        ungoverned = find_ungoverned_condition_trials(master, collect_trials(master, cfg, lag, extend_until), main_cfg["model_conditions"], lag,
                                                      extend_until)
        for section, name, n_un, n_total in ungoverned:
            print(f"(!) model_conditions.{section}.{name}: {n_un} of its {n_total} trial(s) are in no primary group, so they are NOT "
                  f"balanced -- widen the primary_groups queries to cover everything this condition selects")

    trial_list.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(trial_list)} trials)\n      {summary_csv}\n      {audit_csv}")
    print(f"Use it with:  model_conditions.training.trial_restriction = \"{out_csv}\"")


if __name__ == "__main__":
    main()
