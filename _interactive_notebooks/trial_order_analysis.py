#!/usr/bin/env python3

"""
Trial-order (counterbalancing) analysis for the working-memory task, built from
the group timecourse-decoding table (<desc>_group_decoding_results.csv, one row
per decoded TR) that generate_report.py writes.

Questions it answers
  1. Is every trial type (e.g. maintain_face_pos, suppress_place_pos) preceded
     equally often by every trial type? -> transition count tables, rows = the
     PRECEDING trial, columns = the CURRENT trial, for the whole group and for
     every subject. Valence is part of the trial label and is a property of the
     acquisition (WMpos / WMneg), so a positive trial is never preceded by a
     negative one: there is one table for the pos acquisitions and one for the
     neg acquisitions.
     Every cell shows the number of trials and, in parentheses, how many of them
     had each inter-trial interval (ITI): "12 (8TR x3, 9TR x9)".
  2. Do subjects differ? -> which trial-order lists each acquisition used, and
     whether per-subject tables are identical.
  3. Could the preceding trial leak into a trial's baseline window? -> the
     self-evidence in the baseline TRs (timepoints 1-3 of the trial, the same
     TRs the notebooks subtract) split by the preceding trial, and how often each
     trial type follows a trial of its own category.
     The reference "start block" evidence is the last 5 TRs of the 13-TR fixation
     block that opens an acquisition (nothing shown yet).
  4. How much would that bias baseline-corrected comparisons, and what if the
     trials that follow a trial of the same category were removed? -> a
     per-trial-type decomposition of the baseline into the part due to a
     same-category predecessor and the part due to the ITI, pairwise bias
     matrices, the same checks on the restricted trial set, the trials that
     would be dropped per subject, and plots of all of it.

A "trial" is one view cue plus its operation event (trial type = the operation
event, e.g. maintain_face); trial 0 of every acquisition is the pre-trial
fixation and is not a trial. The first trial of an acquisition is preceded by
"<start>" (its "ITI" is then the lead-in before the first view cue).

ITI = the TRs between the last TR of the previous trial's operation event and
the first TR of this trial's view cue (the fixation, plus any unlabeled gap
volume). It is measured in whole TRs, the resolution of the decoding table
(--tr-seconds converts to seconds in the ITI distribution file). A trial whose operation event is missing (e.g. a truncated final
trial, trial_type view_nan) has no type: transitions into or out of it are
left out of the tables and counted in the summary.

Usage:
    python trial_order_analysis.py --decoding-csv <desc>_group_decoding_results.csv --output-dir trial_order_analysis
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

OPERATION_PATTERN = r"^(?:maintain|suppress|switch|clear)_(?:face|place)$"
START = "<start>"
VALENCE_BY_TASK = {"WMpos": "pos", "WMneg": "neg"}
TRIAL_KEYS = ["subject", "boldfile", "trial_index"]
DEFAULT_TR_SECONDS = 0.46


# =====================================================
# Trial table
# =====================================================

def build_trial_table(decoding: pd.DataFrame) -> pd.DataFrame:
    """One row per real trial (a trial_index with a view_* cue) of every
    acquisition, in time order, with the preceding trial attached.

    `decoding` needs subject, session, task, run, boldfile, trial_index,
    window_index, volume_of_interest and trial_type. Output columns: subject,
    session, task, valence, run, boldfile, trial_index, first_volume, view_cue,
    trial_type (the operation event, NaN if the trial has none), operation,
    category, previous_trial (the previous trial's type, "<start>" for the first
    trial of an acquisition, NaN if the previous trial has no type), trial_label
    and previous_label (the same with the valence appended, e.g.
    "maintain_face_pos"), and iti_trs (see the module docstring; for the first
    trial of an acquisition, the lead-in before it)."""
    df = decoding.copy()
    df["_tt"] = df["trial_type"].fillna("").astype(str)
    df = df.sort_values(["boldfile", "trial_index", "window_index"])

    base = df.groupby(TRIAL_KEYS, sort=False).agg(
        session=("session", "first"), task=("task", "first"), run=("run", "first"),
        first_volume=("volume_of_interest", "min"))
    view = df[df["_tt"].str.startswith("view_")].groupby(TRIAL_KEYS, sort=False)["_tt"].first().rename("view_cue")
    op_rows = df[df["_tt"].str.match(OPERATION_PATTERN)]
    op = op_rows.groupby(TRIAL_KEYS, sort=False)["_tt"].first().rename("trial_type")
    op_last = op_rows.groupby(TRIAL_KEYS, sort=False)["volume_of_interest"].max().rename("operation_last_volume")
    trials = base.join(view, how="inner").join(op).join(op_last).reset_index()   # inner: only trials that have a view cue

    trials["operation"] = trials["trial_type"].str.split("_").str[0]
    trials["category"] = trials["trial_type"].str.split("_").str[1]
    trials["valence"] = trials["task"].map(VALENCE_BY_TASK)
    trials = trials.sort_values(["boldfile", "first_volume"]).reset_index(drop=True)
    trials["previous_trial"] = trials.groupby("boldfile")["trial_type"].shift(1)
    first_of_acquisition = trials.groupby("boldfile").cumcount() == 0
    trials.loc[first_of_acquisition, "previous_trial"] = START

    previous_operation_end = trials.groupby("boldfile")["operation_last_volume"].shift(1)
    trials["iti_trs"] = trials["first_volume"] - previous_operation_end - 1
    trials.loc[first_of_acquisition, "iti_trs"] = trials.loc[first_of_acquisition, "first_volume"]   # the lead-in

    trials["trial_label"] = trials["trial_type"] + "_" + trials["valence"]
    trials["previous_label"] = np.where(trials["previous_trial"] == START, START,
                                        trials["previous_trial"] + "_" + trials["valence"])
    trials.loc[trials["previous_trial"].isna(), "previous_label"] = np.nan
    return trials[["subject", "session", "task", "valence", "run", "boldfile", "trial_index", "first_volume",
                   "view_cue", "trial_type", "operation", "category", "previous_trial", "trial_label",
                   "previous_label", "iti_trs"]]


def with_previous_attributes(trials: pd.DataFrame) -> pd.DataFrame:
    """Adds previous_operation / previous_category (NaN for the first trial of an acquisition)."""
    out = trials.copy()
    prev = out["previous_trial"].where(out["previous_trial"] != START).astype("string")
    out["previous_operation"] = prev.str.split("_").str[0].astype(object)
    out["previous_category"] = prev.str.split("_").str[1].astype(object)
    return out


def usable_transitions(trials: pd.DataFrame) -> pd.DataFrame:
    """Trials with a known type whose predecessor is known (or is <start>)."""
    return trials[trials["trial_type"].notna() & trials["previous_trial"].notna()]


# =====================================================
# Counts and tests
# =====================================================

def transition_counts(trials: pd.DataFrame, previous_col: str = "previous_trial", current_col: str = "trial_type",
                      previous_order=None, current_order=None, margins: bool = True) -> pd.DataFrame:
    """Rows = the preceding trial, columns = the current trial; cell = number of
    trials. Order the axes explicitly so tables from different subsets line up."""
    use = trials.dropna(subset=[previous_col, current_col])
    table = pd.crosstab(use[previous_col], use[current_col])
    if previous_order is not None:
        table = table.reindex(index=list(previous_order), fill_value=0)
    if current_order is not None:
        table = table.reindex(columns=list(current_order), fill_value=0)
    if margins:
        table["total"] = table.sum(axis=1)
    return table


def iti_cell(iti_values, unit: str = "TR") -> str:
    """"12 (8TR x3, 9TR x9)": the number of trials, then how many had each ITI
    length (shortest first). An ITI that could not be measured shows as "NA"."""
    values = list(iti_values)
    if not values:
        return "0"
    known = sorted(int(v) for v in values if pd.notna(v))
    parts = [f"{length}{unit} x{count}" for length, count in sorted(pd.Series(known).value_counts().items())]
    if len(known) < len(values):
        parts.append(f"NA x{len(values) - len(known)}")
    return f"{len(values)} ({', '.join(parts)})"


def transition_cells(trials: pd.DataFrame, previous_col: str = "previous_label", current_col: str = "trial_label",
                     iti_col: str = "iti_trs", previous_order=None, current_order=None, unit: str = "TR") -> pd.DataFrame:
    """The transition table with every cell as text: the trial count and, in
    parentheses, the count of each ITI length. The "total" column is the row's
    trial count as text."""
    use = trials.dropna(subset=[previous_col, current_col])
    previous_order = list(previous_order) if previous_order is not None else sorted(use[previous_col].unique())
    current_order = list(current_order) if current_order is not None else sorted(use[current_col].unique())
    grouped = {k: g[iti_col].tolist() for k, g in use.groupby([previous_col, current_col])}
    table = pd.DataFrame([[iti_cell(grouped.get((p, c), []), unit) for c in current_order] for p in previous_order],
                         index=pd.Index(previous_order, name=previous_col), columns=current_order)
    table["total"] = [str(sum(len(grouped.get((p, c), [])) for c in current_order)) for p in previous_order]
    return table


def row_percent(counts: pd.DataFrame) -> pd.DataFrame:
    """Each row as the % of trials that went on to each current type."""
    body = counts.drop(columns="total", errors="ignore")
    return body.div(body.sum(axis=1).replace(0, np.nan), axis=0) * 100


def independence_test(counts: pd.DataFrame) -> dict:
    """Chi-square test that the current trial type is independent of the
    preceding one (the <start> row is excluded: it is not a transition), plus
    Cramer's V and the spread of the cell counts. A perfectly counterbalanced
    design (every type preceded equally by every type) gives chi2 = 0."""
    body = counts.drop(columns="total", errors="ignore")
    body = body.drop(index=START, errors="ignore")
    body = body.loc[body.sum(axis=1) > 0, body.sum(axis=0) > 0]
    n = int(body.to_numpy().sum())
    if body.shape[0] < 2 or body.shape[1] < 2 or n == 0:
        return dict(n_transitions=n, chi2=np.nan, dof=np.nan, p=np.nan, cramers_v=np.nan,
                    min_cell=np.nan, max_cell=np.nan, expected_per_cell=np.nan)
    chi2, p, dof, expected = stats.chi2_contingency(body.to_numpy(), correction=False)
    v = float(np.sqrt(chi2 / (n * (min(body.shape) - 1))))
    return dict(n_transitions=n, chi2=float(chi2), dof=int(dof), p=float(p), cramers_v=v,
                min_cell=int(body.to_numpy().min()), max_cell=int(body.to_numpy().max()),
                expected_per_cell=float(n / body.size))


def sequence_table(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per acquisition with its trial order, and an id shared by every
    acquisition that used exactly the same order (S01, S02, ...). The order lists
    the trials that have a type; a trial without one (a malformed final trial) is
    left out of the order and counted in n_trials - n_typed_trials, and
    typeless_trial_not_last flags any such trial that is NOT the last one (it
    would break the adjacency of the trials around it)."""
    ordered = trials.sort_values(["boldfile", "first_volume"])
    order = ordered.groupby("boldfile")["trial_type"].apply(lambda s: "|".join(s.dropna())).rename("order")
    last_untyped_ok = ordered.groupby("boldfile")["trial_type"].apply(
        lambda s: bool(s.iloc[:-1].isna().any())).rename("typeless_trial_not_last")
    meta = ordered.groupby("boldfile", sort=False).agg(
        subject=("subject", "first"), session=("session", "first"), task=("task", "first"), valence=("valence", "first"),
        run=("run", "first"), n_trials=("trial_index", "size"), n_typed_trials=("trial_type", "count"))
    out = meta.join(order).join(last_untyped_ok).reset_index()
    # an order that is just the start of a longer one (its last trial lacks a type) is the same order:
    # group each order under the longest order it is a prefix of
    representatives = []
    for candidate in sorted(out["order"].unique(), key=len, reverse=True):
        if not any(rep == candidate or rep.startswith(candidate + "|") for rep in representatives):
            representatives.append(candidate)
    ids = {rep: f"S{i + 1:02d}" for i, rep in enumerate(sorted(representatives))}
    base = out["order"].map(lambda o: next(rep for rep in representatives if rep == o or rep.startswith(o + "|")))
    out.insert(0, "sequence_id", base.map(ids))
    out["base_order"] = base
    return out


def order_composition(sequences: pd.DataFrame, types) -> pd.DataFrame:
    """Per trial order: how often each trial type occurs in it (a design with 8
    types and 35 or 36 trials cannot have every type equally often)."""
    rows = []
    for (sid, valence), g in sequences.groupby(["sequence_id", "valence"]):
        counts = pd.Series(g["base_order"].iloc[0].split("|")).value_counts().reindex(list(types), fill_value=0)
        rows.append({"sequence_id": sid, "valence": valence, "n_acquisitions": len(g), "n_typed_trials": int(counts.sum()),
                     "runs": ",".join(str(r) for r in sorted(set(g["run"]))), **counts.to_dict()})
    return pd.DataFrame(rows)


def same_category_rate(trials: pd.DataFrame) -> pd.DataFrame:
    """For each current trial type: how many trials followed a trial of the SAME
    category (face after face, place after place), and the percentage. A
    previous-trial effect on a baseline window can only be balanced across trial
    types if these percentages are similar."""
    use = with_previous_attributes(usable_transitions(trials)).dropna(subset=["previous_category"])
    use = use.assign(same_category=(use["category"] == use["previous_category"]))
    out = use.groupby("trial_type").agg(n=("same_category", "size"), n_same_category=("same_category", "sum"))
    out["percent_same_category"] = out["n_same_category"] / out["n"] * 100
    return out


# =====================================================
# Baseline-window check
# =====================================================

def baseline_by_trial(decoding: pd.DataFrame, trials: pd.DataFrame, baseline_windows=(0, 1, 2)) -> pd.DataFrame:
    """Per trial: the mean evidence over the baseline TRs (window_index values),
    added to the trial table as
      baseline_evidence                 the trial's own category (self-evidence)
      baseline_off_category_evidence    the OTHER face/place category
      baseline_item_evidence            the item class (NaN if the table has no evidence_item)"""
    need = decoding[decoding["window_index"].isin(list(baseline_windows))]
    cols = [c for c in ("evidence_face", "evidence_place", "evidence_item") if c in decoding.columns]
    ev = need.groupby(TRIAL_KEYS)[cols].mean().reset_index()
    out = trials.merge(ev, on=TRIAL_KEYS, how="left")
    is_face = out["category"] == "face"
    out["baseline_evidence"] = np.where(is_face, out["evidence_face"], out["evidence_place"])
    out["baseline_off_category_evidence"] = np.where(is_face, out["evidence_place"], out["evidence_face"])
    out["baseline_item_evidence"] = out["evidence_item"] if "evidence_item" in out.columns else np.nan
    missing = out["category"].isna()
    out.loc[missing, ["baseline_evidence", "baseline_off_category_evidence", "baseline_item_evidence"]] = np.nan
    return out.drop(columns=[c for c in cols if c in out.columns])


def baseline_evidence_by_iti(trials_with_baseline: pd.DataFrame, min_trials: int = 10) -> pd.DataFrame:
    """Baseline-window evidence for the current category ("self"), the other
    face/place category ("off_category") and the item class, by ITI and by
    whether the preceding trial had a different or the same category as the
    current one. One row per (predecessor, ITI): the number of trials and
    subjects, and the mean (of per-subject means) and SE across subjects of each
    kind of evidence. ITIs with fewer than min_trials trials are left out."""
    use = with_predecessor_flags(trials_with_baseline)
    use = use[~use["is_first_trial"] & use["same_category"].notna() & use["iti_trs"].notna() & use["trial_type"].notna()
              & use["baseline_evidence"].notna()].copy()
    use["predecessor"] = np.where(use["same_category"].astype(bool), "same category", "different category")
    use["iti_trs"] = use["iti_trs"].astype(int)
    kinds = {"self": "baseline_evidence", "off_category": "baseline_off_category_evidence", "item": "baseline_item_evidence"}
    rows = []
    for (predecessor, iti), g in use.groupby(["predecessor", "iti_trs"]):
        if len(g) < min_trials:
            continue
        row = {"predecessor": predecessor, "iti_trs": iti, "n_trials": len(g), "n_subjects": int(g["subject"].nunique())}
        for name, col in kinds.items():
            per = g.groupby("subject")[col].mean().dropna()
            row[f"{name}_mean"] = per.mean()
            row[f"{name}_sem"] = per.sem() if len(per) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def start_block_evidence(decoding: pd.DataFrame, start_block_trs: int = 13, last_trs: int = 5) -> dict:
    """Evidence for every class in the last `last_trs` TRs of the start block: the
    fixation block that opens an acquisition (trial_index 0, before any trial),
    for acquisitions whose start block is exactly `start_block_trs` long. Nothing
    has been shown for a while there, so it is the reference for how much
    evidence each class has when no category is being driven.

    Each acquisition contributes the mean over those TRs; those are averaged
    within subject and then across subjects. Returns the mean per class
    ("face", "place", "item") and its SE across subjects, the numbers of
    acquisitions and subjects, and the settings used."""
    block = decoding[decoding["trial_index"] == 0]
    length = block.groupby("boldfile")["window_index"].transform("size")
    position_from_end = block.groupby("boldfile")["window_index"].transform("max") - block["window_index"]
    last = block[(length == start_block_trs) & (position_from_end < last_trs)]
    classes = {"face": "evidence_face", "place": "evidence_place", "item": "evidence_item"}
    out = {"block_trs": start_block_trs, "last_trs": last_trs, "n_acquisitions": int(last["boldfile"].nunique()),
           "n_subjects": int(last["subject"].nunique())}
    for name, col in classes.items():
        if col not in last.columns or last.empty:
            out[name], out[f"{name}_sem"] = np.nan, np.nan
            continue
        per_subject = last.groupby(["subject", "boldfile"])[col].mean().groupby("subject").mean()
        out[name] = float(per_subject.mean())
        out[f"{name}_sem"] = float(per_subject.sem()) if len(per_subject) > 1 else np.nan
    return out


def start_block_table(decoding: pd.DataFrame, last_trs: int = 5) -> pd.DataFrame:
    """start_block_evidence() for every start-block length found (the opening
    fixation block is 13 TRs in some acquisitions and longer in others), one row each."""
    lengths = sorted(decoding.loc[decoding["trial_index"] == 0].groupby("boldfile").size().unique())
    return pd.DataFrame([start_block_evidence(decoding, int(n), last_trs) for n in lengths])


def repeated_measures_anova(matrix: pd.DataFrame) -> dict:
    """One-way repeated-measures ANOVA on a (subjects x levels) table (rows with
    a missing value are dropped), F and p, with eta-squared (partial)."""
    m = matrix.dropna()
    n, k = m.shape
    if n < 2 or k < 2:
        return dict(n_subjects=n, levels=k, F=np.nan, df1=np.nan, df2=np.nan, p=np.nan, partial_eta_sq=np.nan)
    grand = m.to_numpy().mean()
    ss_level = n * ((m.mean(axis=0) - grand) ** 2).sum()
    ss_subject = k * ((m.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((m.to_numpy() - grand) ** 2).sum()
    ss_error = ss_total - ss_level - ss_subject
    df1, df2 = k - 1, (n - 1) * (k - 1)
    with np.errstate(divide="ignore", invalid="ignore"):       # a perfectly constant table has no error variance: F is NaN/inf
        f = (ss_level / df1) / (ss_error / df2)
        eta = ss_level / (ss_level + ss_error)
    return dict(n_subjects=n, levels=k, F=float(f), df1=df1, df2=df2, p=float(stats.f.sf(f, df1, df2)),
                partial_eta_sq=float(eta))


def baseline_summary(trials_with_baseline: pd.DataFrame):
    """(table, tests): the baseline-window evidence by preceding trial type (mean
    over subjects of per-subject means, with the number of trials), plus
    repeated-measures tests across preceding trial types, and same- vs
    different-category predecessor."""
    use = with_previous_attributes(usable_transitions(trials_with_baseline)).dropna(
        subset=["baseline_evidence", "previous_category"])
    per_subject = use.groupby(["subject", "previous_trial"])["baseline_evidence"].mean().unstack()
    table = pd.DataFrame({"n_trials": use.groupby("previous_trial").size(),
                          "mean_baseline_evidence": per_subject.mean(),
                          "sd_across_subjects": per_subject.std()})
    use = use.assign(predecessor=np.where(use["category"] == use["previous_category"], "same_category", "different_category"))
    by_pred = (use.groupby(["subject", "predecessor"])["baseline_evidence"].mean().unstack()
               .reindex(columns=["same_category", "different_category"]).dropna())
    diff = by_pred["same_category"] - by_pred["different_category"]
    if len(diff) >= 2:
        t = stats.ttest_1samp(diff, 0.0)
        t_stat, t_p, t_mean = float(t.statistic), float(t.pvalue), float(diff.mean())
    else:                                  # no subject has both kinds of predecessor: nothing to test
        t_stat, t_p, t_mean = np.nan, np.nan, np.nan
    tests = [
        {"test": "baseline evidence differs by preceding trial type (repeated-measures ANOVA)", **repeated_measures_anova(per_subject)},
        {"test": "same-category minus different-category predecessor (paired t-test across subjects)",
         "n_subjects": len(diff), "levels": 2, "F": t_stat ** 2, "df1": 1, "df2": len(diff) - 1, "p": t_p,
         "partial_eta_sq": np.nan, "mean_difference": t_mean},
    ]
    return table, pd.DataFrame(tests)


def baseline_by_current_type(trials_with_baseline: pd.DataFrame) -> pd.DataFrame:
    """Per CURRENT trial type: how many trials, the share that followed a trial of
    the same category, and the mean baseline-window self-evidence (mean of the
    per-subject means). If the baseline depends on the previous trial and the
    same-category share differs between trial types, the types start from
    different baselines for a reason that has nothing to do with the type."""
    use = with_previous_attributes(usable_transitions(trials_with_baseline)).dropna(
        subset=["baseline_evidence", "previous_category"])
    use = use.assign(same_category=(use["category"] == use["previous_category"]))
    per_subject = use.groupby(["subject", "trial_type"])["baseline_evidence"].mean().unstack()
    out = use.groupby("trial_type").agg(n_trials=("same_category", "size"), percent_same_category=("same_category", "mean"))
    out["percent_same_category"] *= 100
    out["mean_baseline_evidence"] = per_subject.mean()
    out["sd_across_subjects"] = per_subject.std()
    return out


# =====================================================
# Baseline bias and restricting to different-category predecessors
# =====================================================

def with_predecessor_flags(trials: pd.DataFrame) -> pd.DataFrame:
    """Adds previous_category, is_first_trial and same_category (a nullable
    boolean: True when the preceding trial has the same category as this one,
    <NA> for the first trial of an acquisition and for a trial whose predecessor
    has no type)."""
    out = with_previous_attributes(trials)
    out["is_first_trial"] = out["previous_trial"] == START
    known = out["previous_category"].notna() & out["category"].notna()
    out["same_category"] = pd.array(np.where(known, out["category"] == out["previous_category"], np.nan), dtype="boolean")
    return out


def _subject_label_means(frame: pd.DataFrame, value_col: str = "baseline_evidence") -> pd.DataFrame:
    return frame.groupby(["subject", "trial_label"])[value_col].mean().unstack()


def iti_slope_within(frame: pd.DataFrame, value_col: str = "baseline_evidence"):
    """Slope of the baseline on the ITI (evidence per TR) from OLS after removing each
    subject x trial-type mean from both, so neither subject nor trial-type
    differences leak into it. Returns (pooled_slope, per_subject_slopes)."""
    f = frame.dropna(subset=[value_col, "iti_trs"]).copy()
    keys = ["subject", "trial_label"]
    f["_dx"] = f["iti_trs"] - f.groupby(keys)["iti_trs"].transform("mean")
    f["_dy"] = f[value_col] - f.groupby(keys)[value_col].transform("mean")
    denom = (f["_dx"] ** 2).sum()
    pooled = float((f["_dx"] * f["_dy"]).sum() / denom) if denom > 0 else np.nan
    per_subject = f.groupby("subject").apply(
        lambda g: (g["_dx"] * g["_dy"]).sum() / (g["_dx"] ** 2).sum() if (g["_dx"] ** 2).sum() > 0 else np.nan,
        include_groups=False).dropna()
    return pooled, per_subject


def baseline_bias(trials_with_baseline: pd.DataFrame):
    """How much a trial type's baseline (the constant the baseline correction
    subtracts: the mean self-evidence over TRs 1-3, averaged within participant
    and then across participants) is shifted by what came before the trial.

    For every valence-labelled trial type (e.g. clear_face_pos), the baseline is
    computed on four sets of trials:
      notebook      every typed trial with a baseline (what the notebooks use)
      transitions   the same without each acquisition's first trial
      different     only trials whose predecessor has a DIFFERENT category
      different_iti_adjusted   `different`, with the ITI effect removed (a linear
                    effect of the ITI, estimated within subject x trial type,
                    evaluated at the mean ITI of that valence's trials)
    and the shift is split into
      bias_same  = transitions - different          (the same-category carry-over)
      bias_iti   = different - different_iti_adjusted   (the ITI schedule)
      bias_total = bias_same + bias_iti
    Because the corrected curve is raw - baseline, a type whose baseline is
    inflated by bias_total has its corrected curve lowered by that amount; a
    corrected comparison of type A with type B is therefore shifted by
    -(bias_total[A] - bias_total[B]).

    Returns (table, pairwise, tests, scenarios):
      table      one row per trial type
      pairwise   {valence: matrix}, [A, B] = bias_total[A] - bias_total[B]
      tests      per valence and scenario: repeated-measures ANOVA across trial types
                 (does the baseline differ by trial type?), the spread of the
                 type means, and the ITI-slope tests
      scenarios  {(valence, scenario): subject x trial-type matrix of baselines}"""
    use = with_predecessor_flags(trials_with_baseline)
    use = use[use["trial_type"].notna() & use["baseline_evidence"].notna()].copy()
    transitions = use[~use["is_first_trial"] & use["same_category"].notna()]
    different = transitions[transitions["same_category"] == False]          # noqa: E712
    rows, tests, scenarios, pairwise = [], [], {}, {}
    for valence in VALENCES:
        v_all, v_trans, v_diff = (x[x["valence"] == valence] for x in (use, transitions, different))
        pooled_slope, subject_slopes = iti_slope_within(v_diff)
        iti_ref = float(v_diff["iti_trs"].mean())
        adjusted = v_diff.assign(baseline_evidence=v_diff["baseline_evidence"] - pooled_slope * (v_diff["iti_trs"] - iti_ref))
        sets = {"notebook": v_all, "transitions": v_trans, "different": v_diff, "different_iti_adjusted": adjusted}
        matrices = {name: _subject_label_means(frame) for name, frame in sets.items()}
        for name, m in matrices.items():
            scenarios[(valence, name)] = m
            group_means = m.mean()
            r = repeated_measures_anova(m)
            tests.append({"valence": valence, "trial_set": name, "n_trials": int(len(sets[name])),
                          "type_mean_range": float(group_means.max() - group_means.min()),
                          "type_mean_sd": float(group_means.std()), **{k: r[k] for k in ("n_subjects", "F", "df1", "df2", "p", "partial_eta_sq")}})
        labels = sorted(use.loc[use["valence"] == valence, "trial_label"].unique())
        means = {name: m.mean().reindex(labels) for name, m in matrices.items()}
        share = v_trans.groupby("trial_label")["same_category"].apply(lambda s: float(s.astype(float).mean())).reindex(labels) * 100
        iti_diff = v_diff.groupby("trial_label")["iti_trs"].mean().reindex(labels)
        table = pd.DataFrame({
            "valence": valence, "n_trials_notebook": v_all.groupby("trial_label").size().reindex(labels),
            "n_transitions": v_trans.groupby("trial_label").size().reindex(labels),
            "n_different_category": v_diff.groupby("trial_label").size().reindex(labels),
            "percent_same_category_predecessor": share,
            "mean_iti_different_category_trials": iti_diff,
            "baseline_notebook": means["notebook"], "baseline_transitions": means["transitions"],
            "baseline_different": means["different"], "baseline_different_iti_adjusted": means["different_iti_adjusted"]})
        table["bias_same_category"] = table["baseline_transitions"] - table["baseline_different"]
        table["bias_iti"] = table["baseline_different"] - table["baseline_different_iti_adjusted"]
        table["bias_total"] = table["bias_same_category"] + table["bias_iti"]
        table["baseline_offset_vs_grand_mean"] = table["baseline_notebook"] - table["baseline_notebook"].mean()
        rows.append(table)
        pairwise[valence] = pd.DataFrame(table["bias_total"].to_numpy()[:, None] - table["bias_total"].to_numpy()[None, :],
                                         index=table.index, columns=table.index)
        pairwise[valence].index.name = "row_minus_column"
        within = v_diff.assign(_c=v_diff["baseline_evidence"])
        t = stats.ttest_1samp(subject_slopes, 0.0) if len(subject_slopes) > 1 else None
        same_slope, same_subject = iti_slope_within(v_trans[v_trans["same_category"] == True])   # noqa: E712
        t2 = stats.ttest_1samp(same_subject, 0.0) if len(same_subject) > 1 else None
        tests.append({"valence": valence, "trial_set": "ITI slope, different-category predecessor (evidence per TR)",
                      "n_trials": int(len(v_diff)), "type_mean_range": pooled_slope, "n_subjects": int(len(subject_slopes)),
                      "F": float(t.statistic) if t else np.nan, "p": float(t.pvalue) if t else np.nan})
        tests.append({"valence": valence, "trial_set": "ITI slope, same-category predecessor (evidence per TR)",
                      "n_trials": int((v_trans["same_category"] == True).sum()), "type_mean_range": same_slope, "n_subjects": int(len(same_subject)),   # noqa: E712
                      "F": float(t2.statistic) if t2 else np.nan, "p": float(t2.pvalue) if t2 else np.nan})
    table = pd.concat(rows).rename_axis("trial_label").reset_index()
    return table, pairwise, pd.DataFrame(tests), scenarios


def contrast_bias(table: pd.DataFrame) -> pd.DataFrame:
    """The bias the notebooks' own contrasts inherit from what preceded the trials,
    from baseline_bias()'s per-trial-type table.

    Contrasts: each operation minus maintain (within category and valence), and
    pos minus neg (within operation and category). For A - B the table gives the
    observed difference in baselines, the same-category / ITI / total parts of it
    that come from what preceded the trials, and the implied shift of the
    BASELINE-CORRECTED A - B difference, which is the negative of the total (the
    corrected curve is the raw curve minus the baseline)."""
    t = table.set_index("trial_label")
    rows = []

    def add(kind, valence, name, a, b):
        rows.append({"contrast_type": kind, "valence": valence, "contrast": name, "trial_type_a": a, "trial_type_b": b,
                     "observed_baseline_difference": t.at[a, "baseline_notebook"] - t.at[b, "baseline_notebook"],
                     "bias_same_category": t.at[a, "bias_same_category"] - t.at[b, "bias_same_category"],
                     "bias_iti": t.at[a, "bias_iti"] - t.at[b, "bias_iti"],
                     "bias_total": t.at[a, "bias_total"] - t.at[b, "bias_total"]})
    labels = set(t.index)
    for valence in VALENCES:
        for category in ("face", "place"):
            for op in ("clear", "suppress", "switch"):
                a, b = f"{op}_{category}_{valence}", f"maintain_{category}_{valence}"
                if a in labels and b in labels:
                    add("operation vs maintain", valence, f"{op} - maintain ({category}, {valence})", a, b)
    for op in ("maintain", "clear", "suppress", "switch"):
        for category in ("face", "place"):
            a, b = f"{op}_{category}_pos", f"{op}_{category}_neg"
            if a in labels and b in labels:
                add("pos vs neg", "pos - neg", f"{op} {category}: pos - neg", a, b)
    out = pd.DataFrame(rows)
    out["implied_shift_of_corrected_difference"] = -out["bias_total"]
    return out


def dropped_same_category(trials: pd.DataFrame):
    """Per subject and valence (plus an "all" row per subject): how many trials
    there are, how many would be dropped by removing every trial that follows a
    trial of the same category, and what is left. The first trial of an
    acquisition has no preceding trial and is kept.

    Returns (per_subject, kept_by_type): the per-subject table, and a subject x
    trial-type table of the trials that would remain."""
    use = with_predecessor_flags(trials[trials["trial_type"].notna()])
    use = use.assign(dropped=(use["same_category"] == True).fillna(False).astype(bool))   # noqa: E712  (<NA> = no predecessor: kept)
    rows = []
    for (subject, valence), g in use.groupby(["subject", "valence"]):
        rows.append({"subject": subject, "valence": valence, "n_trials": len(g), "n_first_trials": int(g["is_first_trial"].sum()),
                     "n_same_category_dropped": int(g["dropped"].sum())})
    per = pd.DataFrame(rows)
    total = per.groupby("subject")[["n_trials", "n_first_trials", "n_same_category_dropped"]].sum().reset_index().assign(valence="all")
    per = pd.concat([per, total], ignore_index=True)
    per["n_kept"] = per["n_trials"] - per["n_same_category_dropped"]
    per["percent_dropped"] = per["n_same_category_dropped"] / per["n_trials"] * 100
    per = per.sort_values(["subject", "valence"], key=lambda c: c if c.name == "subject" else c.map({"pos": 0, "neg": 1, "all": 2})).reset_index(drop=True)
    kept = use[~use["dropped"]].groupby(["subject", "trial_label"]).size().unstack(fill_value=0)
    kept["min_per_trial_type"] = kept.min(axis=1)
    return per, kept


def dropped_by_trial_type(trials: pd.DataFrame) -> pd.DataFrame:
    """Per valence-labelled trial type: trials in total, how many follow a trial of
    the same category (dropped), the percent, and how many are left per subject
    (mean and minimum over subjects, counting subjects with none as 0)."""
    use = with_predecessor_flags(trials[trials["trial_type"].notna()])
    use = use.assign(dropped=(use["same_category"] == True).fillna(False).astype(bool))   # noqa: E712
    counts = use.groupby("trial_label").agg(n_trials=("dropped", "size"), n_dropped=("dropped", "sum"))
    kept = use[~use["dropped"]].groupby(["subject", "trial_label"]).size().unstack(fill_value=0)
    kept = kept.reindex(columns=counts.index, fill_value=0).reindex(use["subject"].unique(), fill_value=0)
    out = counts.assign(percent_dropped=counts["n_dropped"] / counts["n_trials"] * 100,
                        mean_kept_per_subject=kept.mean(), min_kept_per_subject=kept.min())
    out["n_dropped"] = out["n_dropped"].astype(int)
    return out.reset_index()


# ---------------------------------------------------------------- plots

def _colors():
    return {"pos": "#d62728", "neg": "#1f77b4"}


def _fit_line(x, y):
    """(slope, intercept, r, p) of y on x over the finite pairs, or None if there are
    fewer than 3 pairs or x does not vary."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    r, p = stats.pearsonr(x[ok], y[ok])
    return float(slope), float(intercept), float(r), float(p)


def _type_marker(label: str) -> str:
    return {"maintain": "o", "suppress": "s", "switch": "^", "clear": "D"}.get(label.split("_")[0], "o")


MID_TRIAL_WINDOWS = (10, 11, 12)         # window_index of TRs 11-13 (1-based): the middle of the trial
MID_TRIAL_COLUMN = "evidence_tr11_13"


def _baseline_by_subject_predecessor(trials_with_baseline: pd.DataFrame, value_col: str = "baseline_evidence") -> pd.DataFrame:
    """Mean `value_col` per subject and valence after a different-category (False)
    and a same-category (True) predecessor; the acquisition's first trial is excluded."""
    use = with_predecessor_flags(trials_with_baseline)
    use = use[~use["is_first_trial"] & use["same_category"].notna() & use[value_col].notna()]
    return (use.groupby(["subject", "valence", use["same_category"].astype(bool)])[value_col].mean().unstack()
            .reindex(columns=[False, True]))


def baseline_by_subject_predecessor(trials_with_baseline: pd.DataFrame) -> pd.DataFrame:
    """The numbers behind plot 1a (and 1b, if the trial table has the TR 11-13
    column): one row per subject and valence."""
    out = None
    for col, tag in (("baseline_evidence", "tr1_3"), (MID_TRIAL_COLUMN, "tr11_13")):
        if col not in trials_with_baseline.columns:
            continue
        per = _baseline_by_subject_predecessor(trials_with_baseline, col)
        per = per.rename(columns={False: f"{tag}_after_different_category", True: f"{tag}_after_same_category"})
        per[f"{tag}_same_minus_different"] = per[f"{tag}_after_same_category"] - per[f"{tag}_after_different_category"]
        per.columns.name = None
        out = per if out is None else out.join(per, how="outer")
    return out.reset_index()


def _draw_predecessor_panel(ax, per: pd.DataFrame, title: str, ylabel: str):
    """Every subject's mean after a different- vs a same-category predecessor, plus group means."""
    for valence, color in _colors().items():
        sub = per.xs(valence, level="valence").dropna()
        for _, row in sub.iterrows():
            ax.plot([0, 1], [row[False], row[True]], color=color, alpha=0.18, lw=0.8)
        if len(sub):
            ax.plot([0, 1], [sub[False].mean(), sub[True].mean()], color=color, lw=3, marker="o", label=f"{valence} mean (n={len(sub)})")
    t = per.groupby("subject").mean()
    diff = (t[True] - t[False]).dropna()
    note = (f"same - different = {diff.mean():+.3f} (t-test across subjects p = {stats.ttest_1samp(diff, 0).pvalue:.1e})"
            if len(diff) > 1 else "no subject has both kinds of predecessor")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["predecessor:\ndifferent category", "predecessor:\nsame category"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{note}", fontsize=10)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)


def plot_carryover(trials_with_baseline: pd.DataFrame, table: pd.DataFrame):
    """Self-evidence vs the preceding trial's category: (a) every subject's mean over the
    baseline TRs 1-3 after a different- vs a same-category predecessor, (b) the same for
    TRs 11-13 in the middle of the trial (if the trial table has that column), (c) across
    trial types, the share of same-category predecessors vs the mean baseline."""
    import matplotlib.pyplot as plt
    has_mid = MID_TRIAL_COLUMN in trials_with_baseline.columns
    n = 3 if has_mid else 2
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.4), sharey=False)
    ax1, ax2 = axes[0], axes[-1]
    _draw_predecessor_panel(ax1, _baseline_by_subject_predecessor(trials_with_baseline),
                            "(a) TRs 1-3: each subject's baseline by predecessor", "self-evidence, TRs 1-3 (baseline window)")
    if has_mid:
        _draw_predecessor_panel(axes[1], _baseline_by_subject_predecessor(trials_with_baseline, MID_TRIAL_COLUMN),
                                "(b) TRs 11-13 (middle of the trial): the same, by predecessor", "self-evidence, TRs 11-13")
        lo = min(ax1.get_ylim()[0], axes[1].get_ylim()[0]); hi = max(ax1.get_ylim()[1], axes[1].get_ylim()[1])
        ax1.set_ylim(lo, hi); axes[1].set_ylim(lo, hi)
    panel_c = "(c)" if has_mid else "(b)"
    for valence, color in _colors().items():
        sub = table[table["valence"] == valence]
        for _, r in sub.iterrows():
            ax2.scatter(r["percent_same_category_predecessor"], r["baseline_transitions"], color=color, marker=_type_marker(r["trial_label"]), s=70, zorder=3)
            ax2.annotate(r["trial_label"].replace("_pos", "").replace("_neg", ""), (r["percent_same_category_predecessor"], r["baseline_transitions"]),
                         fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    fit = _fit_line(table["percent_same_category_predecessor"], table["baseline_transitions"])
    x = table["percent_same_category_predecessor"].to_numpy(dtype=float)
    if fit:
        slope, intercept, r, p = fit
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 20)
        ax2.plot(xs, intercept + slope * xs, color="gray", ls="--", zorder=1)
        note2 = f"r = {r:.2f}, p = {p:.1e}; slope {slope * 10:+.3f} per 10 percentage points"
    else:
        note2 = "too little variation to fit"
    ax2.set_xlabel("% of the type's trials that follow a same-category trial"); ax2.set_ylabel("mean baseline-window self-evidence (TRs 1-3)")
    ax2.set_title(f"{panel_c} TRs 1-3 across the 16 trial types (red = pos, blue = neg)\n{note2}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_iti(trials_with_baseline: pd.DataFrame, by_iti: pd.DataFrame = None, start_block: dict = None):
    """Why does the baseline self-evidence grow with the ITI after a different-category
    trial? Baseline-window (TR 1-3) evidence for the current category ("self"), the
    other face/place category ("off category") and the item class, against the ITI,
    when the preceding trial has a (a) different or (b) same category. Dotted lines
    mark the evidence in the start block (the last TRs of the fixation block that
    opens each acquisition, before any trial), if `start_block` is given."""
    import matplotlib.pyplot as plt
    by_iti = baseline_evidence_by_iti(trials_with_baseline) if by_iti is None else by_iti
    start_block = start_block or {}
    use = with_predecessor_flags(trials_with_baseline)
    use = use[~use["is_first_trial"] & use["same_category"].notna() & use["iti_trs"].notna() & use["baseline_evidence"].notna()]
    styles = {"self": ("#2ca02c", "-", "o", "current category (self)"), "off_category": ("#9467bd", "-", "s", "other face/place category (off category)"),
              "item": ("#7f7f7f", "--", "^", "item class")}
    cols = {"self": "baseline_evidence", "off_category": "baseline_off_category_evidence", "item": "baseline_item_evidence"}
    panels = [("different category", "(a) the preceding trial has a DIFFERENT category\n(the off category is the PREVIOUS trial's category)"),
              ("same category", "(b) the preceding trial has the SAME category\n(the off category is the OTHER category)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (predecessor, title) in zip(axes, panels):
        g = by_iti[by_iti["predecessor"] == predecessor] if len(by_iti) else by_iti
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ITI before the trial (TRs; 1 TR = 0.46 s)")
        if g.empty:                                         # no ITI has enough trials of this kind
            ax.text(0.5, 0.5, "not enough trials at any ITI", transform=ax.transAxes, ha="center", va="center", color="gray")
            continue
        trial_level = use[use["same_category"].astype(bool) == (predecessor == "same category")]
        for kind, (color, ls, marker, name) in styles.items():
            if g[f"{kind}_mean"].isna().all():
                continue
            ok = trial_level[[cols[kind], "iti_trs"]].dropna()
            rho = stats.spearmanr(ok["iti_trs"], ok[cols[kind]])[0] if len(ok) > 2 and ok["iti_trs"].nunique() > 1 else np.nan
            ax.errorbar(g["iti_trs"], g[f"{kind}_mean"], yerr=g[f"{kind}_sem"], color=color, ls=ls, marker=marker, capsize=3, lw=2,
                        label=f"{name}   (trial-level Spearman r with ITI = {rho:+.2f})")
        for label in ("face", "place", "item"):
            level = start_block.get(label, np.nan)
            if np.isfinite(level):
                ax.axhline(level, color="#bbbbbb", ls=":", lw=1)
                ax.text(g["iti_trs"].max() + 0.08, level, f"start block: {label} {level:.2f}", fontsize=7, va="center", color="#666666")
        ax.legend(frameon=False, fontsize=8, loc="upper left" if predecessor == "different category" else "lower right")
        ax.set_xlim(g["iti_trs"].min() - 0.3, g["iti_trs"].max() + 1.4)
    axes[0].set_ylabel("mean evidence in the baseline TRs (1-3) +/- SE across subjects")
    note = (f"; dotted = start block (last {start_block['last_trs']} of {start_block['block_trs']} TRs, {start_block['n_acquisitions']} acquisitions)"
            if start_block.get("n_acquisitions") else "")
    fig.suptitle("Baseline-window evidence vs ITI: the previous trial's category is still active at short ITIs and suppresses the other categories"
                 + note, fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_contrast_bias(contrasts: pd.DataFrame):
    """For each contrast the notebooks run, the shift its baseline-corrected
    difference inherits from what preceded the trials."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, kind in zip(axes, ("operation vs maintain", "pos vs neg")):
        c = contrasts[contrasts["contrast_type"] == kind].reset_index(drop=True)
        y = np.arange(len(c))
        colors = ["#d62728" if v == "pos" else "#1f77b4" if v == "neg" else "#555555" for v in c["valence"]]
        ax.barh(y, c["implied_shift_of_corrected_difference"], color=colors, alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(c["contrast"], fontsize=8); ax.invert_yaxis()
        vals = c["implied_shift_of_corrected_difference"].astype(float)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            ax.set_xlim(min(finite.min() * 1.25, -0.02), max(finite.max() * 1.35, 0.02))   # room for the value labels
        for i, v in enumerate(c["implied_shift_of_corrected_difference"].astype(float)):
            if not np.isfinite(v):
                continue
            ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
        ax.set_xlabel("shift of the baseline-corrected A - B difference caused by what preceded the trials (evidence units)")
        ax.set_title(f"{kind}" + (" (red = pos, blue = neg)" if kind == "operation vs maintain" else ""), fontsize=10)
    fig.suptitle("Bias the notebooks' own contrasts inherit from the preceding trials' category and the ITI "
                 "(TR 1-3 baseline correction)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_restricting(tests: pd.DataFrame, scenarios: dict):
    """Type-level baselines on every trial set: does restricting to different-category
    predecessors (and removing the ITI effect) flatten the differences between types?"""
    import matplotlib.pyplot as plt
    names = [("notebook", "all trials\n(as in the notebook)"), ("transitions", "all transitions"),
             ("different", "different-category\npredecessor only"), ("different_iti_adjusted", "different-category,\nITI-adjusted")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, valence in zip(axes, VALENCES):
        labels = list(scenarios[(valence, "notebook")].columns)
        for label in labels:
            ys = [scenarios[(valence, key)].mean().get(label, np.nan) for key, _ in names]
            ax.plot(range(len(names)), ys, marker=_type_marker(label), alpha=0.85, label=label.replace(f"_{valence}", ""))
        ax.set_xticks(range(len(names))); ax.set_xticklabels([n for _, n in names], fontsize=8)
        ax.set_title(f"{valence.upper()} acquisitions: mean baseline (TRs 1-3) of each trial type", fontsize=10)
        lines = []
        for key, label in names:
            r = tests[(tests["valence"] == valence) & (tests["trial_set"] == key)].iloc[0]
            lines.append(f"{label.replace(chr(10), ' ')}: range {r['type_mean_range']:.3f}, ANOVA F={r['F']:.1f}, p={r['p']:.1e}")
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=7, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    axes[0].set_ylabel("mean baseline-window self-evidence"); axes[-1].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig


def plot_dropped(by_type: pd.DataFrame):
    """For each trial type: the share of its trials that follow a trial of the same
    category (and would be dropped), and how many are left per subject."""
    import matplotlib.pyplot as plt
    t = by_type.sort_values("percent_dropped").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(t)), t["percent_dropped"], color=["#d62728" if i.endswith("_pos") else "#1f77b4" for i in t["trial_label"]], alpha=0.85)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t["trial_label"], fontsize=8)
    ax.set_xlim(0, max(100.0, float(t["percent_dropped"].max()) * 1.05) if len(t) else 100)
    for i, r in t.iterrows():
        ax.text(r["percent_dropped"] + 1, i, f"{r['percent_dropped']:.0f}%  ({r['n_dropped']:,} of {r['n_trials']:,}; "
                f"{r['mean_kept_per_subject']:.1f} kept per subject, min {int(r['min_kept_per_subject'])})", va="center", fontsize=7)
    ax.set_xlabel("% of the trial type's trials that follow a same-category trial (dropped)")
    total, dropped = int(t["n_trials"].sum()), int(t["n_dropped"].sum())
    ax.set_title(f"Trials dropped by removing same-category back-to-back trials: {dropped:,} of {total:,} ({100 * dropped / max(total, 1):.0f}%)"
                 "\n(red = pos, blue = neg)", fontsize=10)
    fig.tight_layout()
    return fig


# =====================================================
# Orchestration
# =====================================================

VALENCES = ("pos", "neg")


def iti_summary(trials: pd.DataFrame, tr_seconds: float = DEFAULT_TR_SECONDS):
    """(distribution, summary): the ITI of every between-trial transition (the
    lead-in before each acquisition's first trial is reported separately), by
    valence, in TRs and seconds."""
    between = trials[(trials["previous_trial"] != START) & trials["iti_trs"].notna() & trials["trial_type"].notna()]
    lead = trials[(trials["previous_trial"] == START) & trials["iti_trs"].notna()]
    rows = []
    for kind, data in (("iti", between), ("lead-in", lead)):
        dist = data.groupby(["valence", "iti_trs"]).size().rename("n").reset_index()
        dist.insert(0, "kind", kind)
        rows.append(dist)
    distribution = pd.concat(rows, ignore_index=True)
    distribution["iti_seconds"] = (distribution["iti_trs"] * tr_seconds).round(3)
    summaries = []
    for (kind, valence), g in pd.concat([between.assign(kind="iti"), lead.assign(kind="lead-in")]).groupby(["kind", "valence"]):
        summaries.append({"kind": kind, "valence": valence, "n": len(g), "mean_trs": g["iti_trs"].mean(), "sd_trs": g["iti_trs"].std(),
                          "min_trs": g["iti_trs"].min(), "max_trs": g["iti_trs"].max(),
                          "mean_seconds": g["iti_trs"].mean() * tr_seconds})
    return distribution, pd.DataFrame(summaries)


def iti_mean_by_transition(trials: pd.DataFrame, previous_order, current_order) -> pd.DataFrame:
    """Mean ITI (TRs) in every preceding-by-current cell (NaN for an empty cell)."""
    use = trials.dropna(subset=["previous_label", "trial_label", "iti_trs"])
    table = use.pivot_table(index="previous_label", columns="trial_label", values="iti_trs", aggfunc="mean")
    return table.reindex(index=list(previous_order), columns=list(current_order))


def iti_dependence_tests(trials: pd.DataFrame) -> pd.DataFrame:
    """Does the ITI depend on the trial type? Kruskal-Wallis of the between-trial
    ITI across the current trial's type, and across the preceding trial's type,
    separately for each valence."""
    use = trials[(trials["previous_trial"] != START) & trials["iti_trs"].notna() & trials["trial_type"].notna()
                 & trials["previous_label"].notna()]
    rows = []
    for valence in VALENCES:
        v = use[use["valence"] == valence]
        for name, col in (("current trial type", "trial_label"), ("preceding trial type", "previous_label")):
            groups = [g["iti_trs"].to_numpy() for _, g in v.groupby(col) if len(g) > 0]
            if len(groups) >= 2 and np.ptp(np.concatenate(groups)) > 0:
                h = stats.kruskal(*groups)
                rows.append({"valence": valence, "ITI differs by": name, "groups": len(groups), "H": float(h.statistic), "p": float(h.pvalue)})
            else:
                rows.append({"valence": valence, "ITI differs by": name, "groups": len(groups), "H": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def analyse(decoding: pd.DataFrame, baseline_windows=(0, 1, 2), tr_seconds: float = DEFAULT_TR_SECONDS,
            start_block_trs: int = 13, start_block_last_trs: int = 5) -> dict:
    """Every table as a dict of DataFrames (see save() for the file names). The
    transition tables are built separately for the pos and the neg acquisitions
    and use valence-labelled trial types ("maintain_face_pos")."""
    trials = build_trial_table(decoding)
    usable = usable_transitions(trials)
    n_unknown = int(trials["trial_type"].isna().sum())
    n_after_unknown = int(((trials["trial_type"].notna()) & trials["previous_trial"].isna()).sum())

    out = {"trials": trials, "types": sorted(trials["trial_type"].dropna().unique())}
    summary = {
        "acquisitions": int(trials["boldfile"].nunique()), "subjects": int(trials["subject"].nunique()),
        "trials_with_a_type": int(trials["trial_type"].notna().sum()), "trials_without_a_type": n_unknown,
        "transitions_counted": int(len(usable)), "trials_skipped_after_a_typeless_trial": n_after_unknown,
        "trial_types_per_valence": len(out["types"]),
    }
    long_tables, subject_test_rows = [], []
    for valence in VALENCES:
        v = usable[usable["valence"] == valence]
        labels = sorted(trials.loc[trials["valence"] == valence, "trial_label"].dropna().unique())
        order = [START] + labels
        counts = transition_counts(v, "previous_label", "trial_label", previous_order=order, current_order=labels)
        out[f"group_counts_{valence}"] = counts
        out[f"group_percent_{valence}"] = row_percent(counts)
        out[f"group_cells_{valence}"] = transition_cells(v, previous_order=order, current_order=labels)
        out[f"iti_mean_{valence}"] = iti_mean_by_transition(v, order, labels)

        subject_counts, subject_cells = {}, {}
        for subject, g in v.groupby("subject"):
            subject_counts[subject] = transition_counts(g, "previous_label", "trial_label", previous_order=order, current_order=labels)
            subject_cells[subject] = transition_cells(g, previous_order=order, current_order=labels)
            test = independence_test(subject_counts[subject])
            subject_test_rows.append({"subject": subject, "valence": valence, **test})
            long = subject_counts[subject].drop(columns="total").reset_index().melt(
                id_vars="previous_label", var_name="current_trial", value_name="count")
            long.insert(0, "valence", valence)
            long.insert(0, "subject", subject)
            long = long.rename(columns={"previous_label": "previous_trial"})
            long["iti_counts"] = [iti_cell(g.loc[(g["previous_label"] == p) & (g["trial_label"] == c), "iti_trs"])
                                  for p, c in zip(long["previous_trial"], long["current_trial"])]
            long_tables.append(long)
        out[f"by_subject_counts_{valence}"] = pd.concat(subject_counts, names=["subject", "previous_trial"]).reset_index()
        out[f"by_subject_cells_{valence}"] = pd.concat(subject_cells, names=["subject", "previous_trial"]).reset_index()

        distinct = len({t.drop(columns="total").to_numpy().tobytes() for t in subject_counts.values()})
        group_test = independence_test(counts)
        summary[f"{valence}_distinct_per_subject_tables"] = distinct
        summary.update({f"{valence}_group_{k}": val for k, val in group_test.items()})
    out["by_subject_long"] = pd.concat(long_tables, ignore_index=True)
    out["by_subject_tests"] = pd.DataFrame(subject_test_rows)
    summary["subject_valence_tables_with_p_below_0.05"] = int((out["by_subject_tests"]["p"] < 0.05).sum())
    summary["subject_valence_tables_total"] = int(len(out["by_subject_tests"]))

    sequences = sequence_table(trials)
    out["sequences"] = sequences
    out["order_composition"] = order_composition(sequences, out["types"])
    out["sequence_summary"] = (sequences.groupby(["sequence_id", "valence"]).agg(
        n_acquisitions=("boldfile", "size"), n_subjects=("subject", "nunique"), n_typed_trials=("n_typed_trials", "max"),
        n_acquisitions_missing_a_type=("n_typed_trials", lambda s: int((s < s.max()).sum())),
        runs=("run", lambda s: ",".join(str(r) for r in sorted(set(s)))),
        first_trials=("base_order", lambda s: " | ".join(s.iloc[0].split("|")[:6]))).reset_index())
    summary["distinct_trial_orders"] = int(sequences["sequence_id"].nunique())

    out["iti_distribution"], out["iti_summary"] = iti_summary(trials, tr_seconds)
    out["iti_tests"] = iti_dependence_tests(trials)
    out["summary"] = pd.Series(summary).rename("value").to_frame()
    out["same_category_rate"] = same_category_rate(trials)

    with_baseline = baseline_by_trial(decoding, trials, baseline_windows)
    mid = baseline_by_trial(decoding, trials, MID_TRIAL_WINDOWS)[TRIAL_KEYS + ["baseline_evidence"]].rename(columns={"baseline_evidence": MID_TRIAL_COLUMN})
    with_baseline = with_baseline.merge(mid, on=TRIAL_KEYS, how="left")
    out["trials_with_baseline"] = with_baseline
    out["baseline_by_previous"], out["baseline_tests"] = baseline_summary(with_baseline)
    out["baseline_by_current"] = baseline_by_current_type(with_baseline)
    out["baseline_by_iti"] = baseline_evidence_by_iti(with_baseline)
    out["start_block"] = start_block_evidence(decoding, start_block_trs, start_block_last_trs)
    out["start_block_table"] = start_block_table(decoding, start_block_last_trs)

    (out["bias_by_trial_type"], out["bias_pairwise"], out["bias_tests"], out["bias_scenarios"]) = baseline_bias(with_baseline)
    out["contrast_bias"] = contrast_bias(out["bias_by_trial_type"])
    out["dropped_per_subject"], out["kept_by_type"] = dropped_same_category(trials)
    out["dropped_by_type"] = dropped_by_trial_type(trials)
    dropped_all = out["dropped_per_subject"][out["dropped_per_subject"]["valence"] == "all"]
    kept_cells = out["kept_by_type"].drop(columns="min_per_trial_type")
    summary["same_category_trials_dropped_total"] = int(dropped_all["n_same_category_dropped"].sum())
    summary["same_category_dropped_per_subject_median"] = float(dropped_all["n_same_category_dropped"].median())
    summary["same_category_dropped_per_subject_min"] = int(dropped_all["n_same_category_dropped"].min())
    summary["same_category_dropped_per_subject_max"] = int(dropped_all["n_same_category_dropped"].max())
    summary["percent_dropped_median"] = float(dropped_all["percent_dropped"].median())
    summary["subject_x_type_cells_with_fewer_than_3_trials_kept"] = int((kept_cells < 3).to_numpy().sum())
    out["summary"] = pd.Series(summary).rename("value").to_frame()
    return out


def plot_transition_heatmap(counts: pd.DataFrame, path: str, title: str = "Preceding trial (rows) x current trial (columns)"):
    """Heatmap of a transition count table, each cell labelled with its count; the
    colour scale is the cell's difference from what independence would give
    (blue = fewer, red = more than expected), so an evenly counterbalanced design
    is uniformly white."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = counts.drop(columns="total", errors="ignore")
    transitions = body.drop(index=START, errors="ignore")
    expected = np.outer(transitions.sum(axis=1), transitions.sum(axis=0)) / transitions.to_numpy().sum()
    excess = (transitions.to_numpy() - expected) / np.sqrt(np.maximum(expected, 1e-9))     # Pearson residual
    limit = max(3.0, np.abs(excess).max())
    fig, ax = plt.subplots(figsize=(9, 7.2))
    im = ax.imshow(excess, cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(transitions.shape[1]))
    ax.set_xticklabels(transitions.columns, rotation=40, ha="right")
    ax.set_yticks(range(transitions.shape[0]))
    ax.set_yticklabels(transitions.index)
    for i in range(transitions.shape[0]):
        for j in range(transitions.shape[1]):
            ax.text(j, i, int(transitions.iat[i, j]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("current trial")
    ax.set_ylabel("preceding trial")
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="(observed - expected) / sqrt(expected)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


STALE_FILES = ("transitions_group_counts.csv", "transitions_group_row_percent.csv", "transitions_group_counts_pos.csv",
               "transitions_group_counts_neg.csv", "transitions_by_subject_wide.csv", "transitions_by_subject_valence_wide.csv",
               "transitions_group_heatmap.png",
               "plot_3_baseline_bias_by_trial_type.png")        # the pooled-over-valence tables of the first version


def save(results: dict, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    for stale in STALE_FILES:
        if os.path.exists(os.path.join(output_dir, stale)):
            os.remove(os.path.join(output_dir, stale))
    files = {
        "trial_sequence.csv": results["trials_with_baseline"],
        "transitions_by_subject_long.csv": results["by_subject_long"],
        "trial_orders_by_acquisition.csv": results["sequences"],
        "trial_orders_summary.csv": results["sequence_summary"],
        "trial_type_counts_by_order.csv": results["order_composition"],
        "balance_tests_by_subject.csv": results["by_subject_tests"],
        "summary.csv": results["summary"].reset_index(names="item"),
        "iti_distribution.csv": results["iti_distribution"],
        "iti_summary.csv": results["iti_summary"],
        "iti_dependence_tests.csv": results["iti_tests"],
        "same_category_predecessor_rate.csv": results["same_category_rate"].reset_index(),
        "baseline_window_by_previous_trial.csv": results["baseline_by_previous"].reset_index(),
        "baseline_window_by_current_trial_type.csv": results["baseline_by_current"].reset_index(),
        "baseline_window_tests.csv": results["baseline_tests"],
        "baseline_bias_by_trial_type.csv": results["bias_by_trial_type"],
        "baseline_bias_tests.csv": results["bias_tests"],
        "baseline_evidence_by_iti.csv": results["baseline_by_iti"],
        "baseline_by_subject_and_predecessor.csv": baseline_by_subject_predecessor(results["trials_with_baseline"]).round(4),
        "start_block_evidence.csv": results["start_block_table"],
        "baseline_bias_in_notebook_contrasts.csv": results["contrast_bias"],
        "dropped_same_category_per_subject.csv": results["dropped_per_subject"],
        "dropped_same_category_by_trial_type.csv": results["dropped_by_type"].round(2),
        "kept_trials_per_subject_trial_type.csv": results["kept_by_type"].reset_index(),
    }
    for valence in VALENCES:
        files[f"baseline_bias_pairwise_{valence}.csv"] = results["bias_pairwise"][valence].round(4).reset_index()
        files[f"transitions_group_{valence}_with_iti.csv"] = results[f"group_cells_{valence}"].reset_index()
        files[f"transitions_group_{valence}_counts.csv"] = results[f"group_counts_{valence}"].reset_index()
        files[f"transitions_group_{valence}_row_percent.csv"] = results[f"group_percent_{valence}"].round(2).reset_index()
        files[f"transitions_by_subject_{valence}_with_iti.csv"] = results[f"by_subject_cells_{valence}"]
        files[f"transitions_by_subject_{valence}_counts.csv"] = results[f"by_subject_counts_{valence}"]
        files[f"iti_mean_by_transition_{valence}.csv"] = results[f"iti_mean_{valence}"].round(3).reset_index()
    for name, table in files.items():
        table.to_csv(os.path.join(output_dir, name), index=False)
    for valence in VALENCES:
        plot_transition_heatmap(results[f"group_counts_{valence}"], os.path.join(output_dir, f"transitions_group_{valence}_heatmap.png"),
                                f"{valence.upper()} acquisitions, group: preceding trial (rows) x current trial (columns), counts")
    plot_files = save_plots(results, output_dir)
    return sorted(files) + [f"transitions_group_{v}_heatmap.png" for v in VALENCES] + plot_files


def save_plots(results: dict, output_dir: str) -> list:
    """The baseline-confound figures, as PNGs and as one multi-page PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    twb, table = results["trials_with_baseline"], results["bias_by_trial_type"]
    figures = {
        "plot_1_baseline_vs_same_category_predecessor.png": plot_carryover(twb, table),
        "plot_2_baseline_vs_iti.png": plot_iti(twb, results["baseline_by_iti"], results["start_block"]),
        "plot_4_restricting_to_different_category_predecessors.png": plot_restricting(results["bias_tests"], results["bias_scenarios"]),
        "plot_5_dropped_trials_same_category.png": plot_dropped(results["dropped_by_type"]),
        "plot_6_bias_in_the_notebook_contrasts.png": plot_contrast_bias(results["contrast_bias"]),
    }
    with PdfPages(os.path.join(output_dir, "baseline_confound_plots.pdf")) as pdf:
        for name, fig in figures.items():
            fig.savefig(os.path.join(output_dir, name), dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return sorted(figures) + ["baseline_confound_plots.pdf"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoding-csv", required=True, help="<desc>_group_decoding_results.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tr-seconds", type=float, default=DEFAULT_TR_SECONDS,
                        help="repetition time, only used to express the ITI in seconds in iti_distribution.csv (default 0.46)")
    parser.add_argument("--start-block-trs", type=int, default=13,
                        help="length of the acquisition-opening fixation block used as the 'nothing is evident' reference (default 13)")
    parser.add_argument("--start-block-last-trs", type=int, default=5,
                        help="how many TRs at the end of that block to average (default 5)")
    parser.add_argument("--baseline-timepoints", type=int, nargs="+", default=[1, 2, 3],
                        help="1-based TRs from the start of each trial used for the baseline-window check (default 1 2 3)")
    args = parser.parse_args()
    cols = ["subject", "session", "task", "run", "boldfile", "trial_index", "window_index", "volume_of_interest",
            "trial_type", "evidence_face", "evidence_place", "evidence_item"]
    decoding = pd.read_csv(args.decoding_csv, usecols=lambda c: c in cols, dtype={"subject": str, "session": str})
    results = analyse(decoding, baseline_windows=[t - 1 for t in args.baseline_timepoints], tr_seconds=args.tr_seconds,
                       start_block_trs=args.start_block_trs, start_block_last_trs=args.start_block_last_trs)
    files = save(results, args.output_dir)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(results["summary"].to_string())
    for valence in VALENCES:
        print(f"\n{valence.upper()} acquisitions -- trials (ITI length in TRs: count), rows = preceding trial, columns = current trial:")
        print(results[f"group_cells_{valence}"].to_string())
    print("\nITI summary:")
    print(results["iti_summary"].round(3).to_string(index=False))
    print(f"\nWrote {len(files)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
