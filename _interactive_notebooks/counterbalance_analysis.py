#!/usr/bin/env python3

"""
Counterbalancing of the working-memory task, read straight from a master spreadsheet (stage 2 output, one row per
volume), so it runs for any study (Clearvale, mindmem, ...) without decoding results.

A trial is one view cue (view_face / view_place) plus the operation event that follows it (e.g. maintain_face). The
operation is the class of the operation classifier; category (face / place) is the cue's image; valence is the task
(WMpos / WMneg). The first trial of an acquisition has no predecessor ("<start>").

Questions
  1. Operation x category x valence: does every trial type occur equally often (per run, per subject)?
  2. Order (first-order counterbalancing): is every operation preceded equally often by every operation (and the
     same split by category)? -> preceding x current tables with counts, expected counts under independence and
     standardised residuals; Cramer's V as a size summary (the orders are fixed lists, so p values are not inference).
  3. Back-to-back repeats: how often does an operation follow the same operation, per operation and per category,
     and how many trials would be lost by dropping them or by balancing the predecessors (the control of Kim et
     al. 2020, Nat Commun 11:6239).
  4. Fixed lists: how many distinct trial orders exist, which acquisitions use which, and is the pos / neg run
     layout the same for all subjects.
  5. Serial position and timing: does an operation tend to come early or late in a run; ITI by operation and by
     transition.
  6. Cross-validation folds (kfold held_out_runs): operation counts and repeat rate per fold, and how well the
     preceding trial alone predicts the current one across folds (order information without any brain data).

Usage:
    python counterbalance_analysis.py --master-spreadsheet master_spreadsheet.csv --study mindmem \
        --output-dir counterbalance_mindmem [--held-out-runs "1,3;2,4;5,7;6,8"]
"""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

START = "<start>"
ACQ = ["subject", "session", "task", "run"]
VALENCE_BY_TASK = {"WMpos": "pos", "WMneg": "neg"}
NOT_OPERATIONS = {"positive", "negative", "neutral", "view", "start", "end", "rest", "trial", "nan"}
DEFAULT_HELD_OUT_RUNS = [[1, 3], [2, 4], [5, 7], [6, 8]]


# =====================================================
# Trial table
# =====================================================

def read_events(path: str) -> pd.DataFrame:
    """One row per event of the working-memory tasks (the master spreadsheet repeats an event for each of its
    volumes), in time order within each acquisition."""
    cols = ["trial_type", "onset", "duration", "event_index", "subject", "session", "task", "run"]
    df = pd.read_csv(path, usecols=cols, dtype={"subject": str, "session": str})
    df = df[df["task"].isin(VALENCE_BY_TASK) & df["event_index"].notna() & df["trial_type"].notna()]
    df = df.drop_duplicates(ACQ + ["event_index"])
    return df.sort_values(ACQ + ["onset"]).reset_index(drop=True)


def infer_operations(events: pd.DataFrame):
    """Operation names: the `<name>` of every `<name>_face` / `<name>_place` event that is not an image class."""
    names = events["trial_type"].str.extract(r"^([A-Za-z]+)_(?:face|place)$")[0].dropna().unique()
    return sorted(n for n in names if n.lower() not in NOT_OPERATIONS)


def build_trials(events: pd.DataFrame, operations) -> pd.DataFrame:
    """One row per trial: acquisition keys, valence, trial_number (1-based within the acquisition), n_trials,
    operation, category, trial_type (e.g. maintain_face), onset, end, previous_* of the preceding trial
    ("<start>" for the first), iti_s (seconds from the end of the previous operation event to this view cue's
    onset; for the first trial the lead-in from the acquisition start) and same_operation / same_category."""
    pattern = r"^(%s)_(face|place)$" % "|".join(operations)
    rows = []
    for key, acq in events.groupby(ACQ, sort=False):
        acq = acq.reset_index(drop=True)
        trial_no, previous, prev_end = 0, None, 0.0
        for i in range(len(acq) - 1):
            cue, event = acq.iloc[i], acq.iloc[i + 1]
            if not str(cue["trial_type"]).startswith("view_"):
                continue
            match = pd.Series(str(event["trial_type"])).str.extract(pattern).iloc[0]
            if match.isna().any():                               # cue without an operation event (truncated run)
                previous = None
                continue
            trial_no += 1
            rows.append(dict(zip(ACQ, key), valence=VALENCE_BY_TASK[key[2]], trial_number=trial_no,
                             operation=match[0], category=match[1], trial_type=event["trial_type"],
                             cue=cue["trial_type"], onset=cue["onset"], end=event["onset"] + event["duration"],
                             previous_trial=START if previous is None and trial_no == 1 else previous,
                             iti_s=cue["onset"] - prev_end))
            previous, prev_end = event["trial_type"], event["onset"] + event["duration"]
    trials = pd.DataFrame(rows)
    trials["n_trials"] = trials.groupby(ACQ)["trial_number"].transform("max")
    trials["previous_operation"] = trials["previous_trial"].str.replace(r"_(face|place)$", "", regex=True)
    trials["previous_category"] = trials["previous_trial"].str.extract(r"_(face|place)$")[0]
    has_pred = trials["previous_trial"].notna() & (trials["previous_trial"] != START)
    trials["same_operation"] = np.where(has_pred, trials["operation"] == trials["previous_operation"], np.nan)
    trials["same_category"] = np.where(has_pred, trials["category"] == trials["previous_category"], np.nan)
    trials["cue_matches_event"] = trials["cue"].str.replace("view_", "") == trials["category"]
    return trials


# =====================================================
# Tables
# =====================================================

def transitions(trials: pd.DataFrame, prev_col: str, cur_col: str, prev_order, cur_order) -> pd.DataFrame:
    """Counts, preceding (rows) x current (columns); trials after a missing / untyped trial are left out, the first
    trial of an acquisition is its own `<start>` row."""
    t = trials.dropna(subset=[prev_col])
    return (pd.crosstab(t[prev_col], t[cur_col]).reindex(index=prev_order, columns=cur_order, fill_value=0))


def expected_and_residuals(counts: pd.DataFrame):
    """Expected counts under independence and standardised (Pearson) residuals, for the table without the
    `<start>` row; both shaped like `counts`."""
    c = counts.drop(index=START, errors="ignore")
    exp = np.outer(c.sum(axis=1), c.sum(axis=0)) / max(c.values.sum(), 1)
    exp = pd.DataFrame(exp, index=c.index, columns=c.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = (c - exp) / np.sqrt(exp)
    return exp, resid.fillna(0.0)


def independence(counts: pd.DataFrame) -> dict:
    c = counts.drop(index=START, errors="ignore")
    c = c.loc[c.sum(axis=1) > 0, c.sum(axis=0) > 0]
    n = c.values.sum()
    if min(c.shape) < 2 or n == 0:
        return dict(n=int(n), chi2=np.nan, df=np.nan, cramers_v=np.nan, empty_cells=0)
    chi2, p, dof, exp = stats.chi2_contingency(c.values, correction=False)
    v = np.sqrt(chi2 / (n * (min(c.shape) - 1)))
    return dict(n=int(n), chi2=round(float(chi2), 1), df=int(dof), cramers_v=round(float(v), 3),
                empty_cells=int((c.values == 0).sum()), cells=int(c.size),
                min_count=int(c.values.min()), max_count=int(c.values.max()), expected_per_cell=round(float(n / c.size), 1))


def type_counts(trials: pd.DataFrame) -> pd.DataFrame:
    """Trials per (valence, operation, category): total, per acquisition (mean, min, max) and number of subjects."""
    per_acq = trials.groupby(ACQ + ["valence", "operation", "category"]).size().rename("n").reset_index()
    out = per_acq.groupby(["valence", "operation", "category"]).agg(
        trials=("n", "sum"), per_acquisition_mean=("n", "mean"), per_acquisition_min=("n", "min"),
        per_acquisition_max=("n", "max"), subjects=("subject", "nunique")).reset_index()
    return out.round(2)


def repeat_rates(trials: pd.DataFrame) -> pd.DataFrame:
    """Per valence x operation x category: trials with a predecessor, the share that follows the SAME operation, the
    share that follows the same category, and the share that follows the same operation AND category."""
    t = trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)].copy()
    t["same_trial_type"] = t["trial_type"] == t["previous_trial"]
    g = t.groupby(["valence", "operation", "category"])
    out = g.agg(trials=("trial_type", "size"), same_operation=("same_operation", "mean"),
                same_category=("same_category", "mean"), same_trial_type=("same_trial_type", "mean")).reset_index()
    pooled = t.groupby(["valence", "operation"]).agg(trials=("trial_type", "size"), same_operation=("same_operation", "mean"),
                                                     same_category=("same_category", "mean"),
                                                     same_trial_type=("same_trial_type", "mean")).reset_index()
    pooled["category"] = "both"
    return pd.concat([out, pooled], ignore_index=True).round(3)


def dropping_cost(trials: pd.DataFrame) -> pd.DataFrame:
    """What the two controls cost, per valence x operation x category: trials, trials left after dropping
    back-to-back same-operation trials, and trials left if the predecessor operations are balanced by
    subsampling (each current operation keeps, within each subject and valence, the same number of trials after
    every predecessor operation: the smallest cell)."""
    t = trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)]
    rows = []
    for (val, op, cat), g in t.groupby(["valence", "operation", "category"]):
        kept = g[g["same_operation"] == 0]
        rows.append(dict(valence=val, operation=op, category=cat, trials=len(g), after_dropping_repeats=len(kept)))
    out = pd.DataFrame(rows)
    out["share_lost"] = (1 - out["after_dropping_repeats"] / out["trials"]).round(3)
    return out


def balanced_by_predecessor(trials: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Trials kept if, within each subject x valence x current operation, every preceding operation is represented by
    the same number of trials (random subsample down to the smallest non-empty cell; per category, so face and place
    keep the same predecessor mix). Rows: valence, operation, category, kept, of."""
    rng = np.random.default_rng(seed)
    t = trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)]
    rows = []
    for (sub, val, op, cat), g in t.groupby(["subject", "valence", "operation", "category"]):
        sizes = g.groupby("previous_operation").size()
        k = int(sizes.min()) if len(sizes) else 0
        rows.append(dict(subject=sub, valence=val, operation=op, category=cat, kept=k * len(sizes), of=len(g),
                         predecessors=len(sizes)))
    d = pd.DataFrame(rows)
    return d.groupby(["valence", "operation", "category"]).agg(
        kept=("kept", "sum"), of=("of", "sum"), subjects_with_all_predecessors=("predecessors", lambda s: int((s == s.max()).sum())),
        subjects=("subject", "nunique")).reset_index()


def order_lists(trials: pd.DataFrame) -> pd.DataFrame:
    """One row per acquisition with a hash of its trial sequence: list_id numbers the distinct sequences."""
    seq = (trials.sort_values(ACQ + ["trial_number"]).groupby(ACQ)["trial_type"].apply(lambda s: ">".join(s)).reset_index())
    seq["list_id"] = seq["trial_type"].map(lambda s: hashlib.md5(s.encode()).hexdigest()[:6])
    seq["n_trials"] = seq["trial_type"].str.count(">") + 1
    return seq.rename(columns={"trial_type": "sequence"})


def run_layout(trials: pd.DataFrame) -> pd.DataFrame:
    """Per subject x session: the valence of each run number, as a string (e.g. 'neg,neg,pos,pos,...') and the
    run numbers present."""
    acq = trials[ACQ + ["valence"]].drop_duplicates().sort_values(["subject", "session", "run"])
    out = acq.groupby(["subject", "session"]).agg(
        runs=("run", lambda s: ",".join(str(int(r)) for r in s)),
        layout=("valence", lambda s: ",".join(s))).reset_index()
    return out


def serial_position(trials: pd.DataFrame) -> pd.DataFrame:
    """Operation (rows) by position in the run (thirds), counts and the mean relative position (0 = first trial,
    1 = last)."""
    t = trials.copy()
    t["relative_position"] = (t["trial_number"] - 1) / (t["n_trials"] - 1).clip(lower=1)
    t["third"] = pd.cut(t["relative_position"], [-0.001, 1 / 3, 2 / 3, 1.001], labels=["first third", "middle third", "last third"])
    counts = pd.crosstab([t["valence"], t["operation"]], t["third"])
    counts["mean_relative_position"] = t.groupby(["valence", "operation"])["relative_position"].mean().round(3)
    return counts.reset_index()


def iti_by_operation(trials: pd.DataFrame) -> pd.DataFrame:
    t = trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)]
    return t.groupby(["valence", "operation"])["iti_s"].agg(["count", "mean", "std", "min", "max"]).round(2).reset_index()


def fold_balance(trials: pd.DataFrame, held_out_runs) -> pd.DataFrame:
    """Per cross-validation fold (the run numbers held out together): trials, share per operation, and the share of
    back-to-back same-operation trials."""
    rows = []
    for i, runs in enumerate(held_out_runs, 1):
        t = trials[trials["run"].isin(runs)]
        shares = t["operation"].value_counts(normalize=True)
        row = dict(fold=i, runs=",".join(map(str, runs)), trials=len(t), subjects=t["subject"].nunique(),
                   same_operation_rate=round(float(t["same_operation"].mean()), 3))
        row.update({f"share_{op}": round(float(shares.get(op, 0)), 3) for op in sorted(trials["operation"].unique())})
        row.update({f"pos_trials": int((t["valence"] == "pos").sum()), "neg_trials": int((t["valence"] == "neg").sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def order_predictability(trials: pd.DataFrame, held_out_runs) -> pd.DataFrame:
    """How well the PRECEDING trial alone predicts the current trial: for each cross-validation fold, the most
    frequent current class after each predecessor is learned from the other folds' trials and applied to the
    held-out fold (no brain data involved). Accuracy above `chance_majority` (always guessing the commonest class)
    is order information a classifier could pick up through the slow BOLD tail of the previous trial. Targets:
    operation, category, operation x category; per valence and pooled."""
    t = trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)].copy()
    t["fold"] = np.nan
    for i, runs in enumerate(held_out_runs, 1):
        t.loc[t["run"].isin(runs), "fold"] = i
    t = t.dropna(subset=["fold"])
    rows = []
    for valence, frame in [("pos", t[t["valence"] == "pos"]), ("neg", t[t["valence"] == "neg"]), ("pooled", t)]:
        for target, predecessor in (("operation", "previous_operation"), ("category", "previous_category"),
                                    ("operation_x_category", "previous_trial")):
            correct = n = 0
            for fold in sorted(frame["fold"].unique()):
                train, test = frame[frame["fold"] != fold], frame[frame["fold"] == fold]
                best = train.groupby(predecessor)[target if target != "operation_x_category" else "trial_type"].agg(lambda s: s.value_counts().index[0])
                fallback = train[target if target != "operation_x_category" else "trial_type"].value_counts().index[0]
                guess = test[predecessor].map(best).fillna(fallback)
                truth = test[target if target != "operation_x_category" else "trial_type"]
                correct += int((guess == truth).sum())
                n += len(test)
            col = target if target != "operation_x_category" else "trial_type"
            rows.append(dict(valence=valence, target=target, trials=n, accuracy_from_predecessor=round(correct / n, 3),
                             chance_even=round(1 / frame[col].nunique(), 3),
                             chance_majority=round(float(frame[col].value_counts(normalize=True).iloc[0]), 3)))
    return pd.DataFrame(rows)


def data_checks(events: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Things that make a subject's design differ from the others: acquisitions per subject, trials per acquisition,
    cue / event mismatches (view_face followed by a *_place event)."""
    acq = trials.groupby(ACQ).agg(n_trials=("trial_number", "max"), mismatches=("cue_matches_event", lambda s: int((~s).sum()))).reset_index()
    per_sub = acq.groupby("subject").agg(acquisitions=("run", "size"), pos_runs=("task", lambda s: int((s == "WMpos").sum())),
                                         neg_runs=("task", lambda s: int((s == "WMneg").sum())),
                                         min_trials=("n_trials", "min"), max_trials=("n_trials", "max"),
                                         cue_event_mismatches=("mismatches", "sum")).reset_index()
    return per_sub


# =====================================================
# Plots
# =====================================================

def heatmap(ax, counts, resid, title, fmt="{:.0f}"):
    import matplotlib.pyplot as plt
    ax.imshow(resid.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="equal")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i, fmt.format(counts.values[i, j]), ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(counts.shape[1]))
    ax.set_xticklabels(counts.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(counts.shape[0]))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("current trial")
    ax.set_ylabel("preceding trial")
    ax.set_title(title, fontsize=9)


def plot_transitions(op_tables: dict, type_tables: dict, rates: pd.DataFrame, study: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), gridspec_kw=dict(width_ratios=[1, 1.9, 1.2]))
    for r, val in enumerate(("pos", "neg")):
        counts = op_tables[val].drop(index=START, errors="ignore")
        _, resid = expected_and_residuals(op_tables[val])
        heatmap(axes[r, 0], counts, resid, f"{study} {val}: operation after operation\n(colour = standardised residual vs even mix)")
        tcounts = type_tables[val].drop(index=START, errors="ignore")
        _, tresid = expected_and_residuals(type_tables[val])
        heatmap(axes[r, 1], tcounts, tresid, f"{study} {val}: operation x category after operation x category")
        rr = rates[(rates["valence"] == val) & (rates["category"] != "both")]
        ops = sorted(rr["operation"].unique())
        x = np.arange(len(ops))
        for k, (cat, color) in enumerate((("face", "tab:blue"), ("place", "tab:orange"))):
            vals = [rr[(rr["operation"] == o) & (rr["category"] == cat)]["same_operation"].mean() for o in ops]
            axes[r, 2].bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=f"{cat} trial")
        axes[r, 2].axhline(1 / len(ops), color="k", ls=":", lw=1)
        axes[r, 2].set_xticks(x)
        axes[r, 2].set_xticklabels(ops, rotation=30, ha="right")
        axes[r, 2].set_ylim(0, 1)
        axes[r, 2].set_ylabel("share following the same operation")
        axes[r, 2].set_title(f"{study} {val}: back-to-back same operation\n(dotted = 1/{len(ops)}, an even mix)", fontsize=9)
        axes[r, 2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================
# PDF report
# =====================================================

FACE_COLOR, PLACE_COLOR = "#2b6cb0", "#dd8452"


def _with_predecessor(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[trials["previous_trial"].notna() & (trials["previous_trial"] != START)]


def draw_matrix(ax, values, rows, cols, labels, center, half_range, title, xlabel="", ylabel="", fontsize=8):
    """Heat map with pcolormesh (stays crisp in every PDF viewer): colour = values - center in a diverging scale of
    +-half_range, `labels` (same shape) printed in the cells."""
    colors = np.asarray(values, dtype=float) - center
    mesh = ax.pcolormesh(np.arange(len(cols) + 1), np.arange(len(rows) + 1), colors, cmap="RdBu_r",
                         vmin=-half_range, vmax=half_range)
    ax.set_ylim(len(rows), 0)
    for i in range(len(rows)):
        for j in range(len(cols)):
            strong = abs(colors[i, j]) > 0.65 * half_range
            ax.text(j + 0.5, i + 0.5, labels[i][j], ha="center", va="center", fontsize=fontsize,
                    color="white" if strong else "black")
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    return mesh


def _count_labels(counts):
    return [[f"{int(v)}" for v in row] for row in counts.values]


def _page_header(fig, text, sub=None):
    fig.suptitle(text, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.985)
    if sub:
        fig.text(0.02, 0.945, sub, fontsize=9, ha="left", va="top", color="#444444", wrap=True)


def page_overview(results):
    import matplotlib.pyplot as plt
    trials, study = results["trials"], results["study"]
    fig = plt.figure(figsize=(11, 8.5))
    _page_header(fig, f"{study}: counterbalancing of operation and category")
    per_list = results["order_lists"].groupby(["task", "list_id"]).size()
    lists = per_list[per_list >= 5].groupby("task").size()          # lists of a single odd acquisition are not designs
    lines = [
        f"{trials['subject'].nunique()} subjects, {len(results['order_lists'])} acquisitions, {len(trials)} trials "
        f"({', '.join(results['operations'])}).",
        "A trial is a view cue (face or place image) followed by one operation event. Valence (pos / neg) belongs to the "
        "acquisition, so every table is separate for WMpos and WMneg.",
        f"Distinct trial-order lists: WMpos {lists.get('WMpos', 0)}, WMneg {lists.get('WMneg', 0)} (the same lists for every "
        "subject, tied to the run number), so the tables below are essentially the same for every subject.",
        "",
        "How to read the tables: rows = what came BEFORE, columns = the CURRENT trial. Colour = counts above (red) or below "
        "(blue) what an even mix would give. If the order were counterbalanced every cell would be about equal.",
        "Cramer's V summarises the size of the association between the preceding and the current trial "
        "(0 = none; the orders are fixed lists, so no p values).",
    ]
    fig.text(0.04, 0.88, "\n".join(textwrap_lines(lines, 120)), fontsize=9.5, va="top")
    rows = []
    for key, label in (("pos_operation", "pos: operation after operation"), ("neg_operation", "neg: operation after operation"),
                       ("pos_operation_x_category", "pos: operation x category after operation x category"),
                       ("neg_operation_x_category", "neg: operation x category after operation x category")):
        t = results["tests"][key]
        rows.append([label, t["cells"], t["empty_cells"], f"{t['min_count']} - {t['max_count']}", t["expected_per_cell"], t["cramers_v"]])
    ax = fig.add_axes([0.04, 0.42, 0.92, 0.22])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["table", "cells", "empty cells", "cell range", "even mix", "Cramer's V"],
                     loc="upper center", cellLoc="center", colWidths=[0.42, 0.08, 0.1, 0.14, 0.1, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    pages = ("Pages: 2 operation order | 3 category order | 4 previous category by trial type | 5 previous operation by trial type | "
             "6 previous trial type by trial type | 7 position in the run and ITI")
    fig.text(0.04, 0.33, "\n".join(textwrap_lines([pages], 120)), fontsize=9, va="top", color="#444444")
    return fig


def textwrap_lines(lines, width):
    import textwrap
    out = []
    for line in lines:
        out.extend(textwrap.wrap(line, width) or [""])
    return out


def page_operation_order(results):
    import matplotlib.pyplot as plt
    ops, rates = results["operations"], results["repeat_rates"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw=dict(width_ratios=[1.1, 1], top=0.86, bottom=0.12, hspace=0.5))
    _page_header(fig, "Operation order: what preceded each operation",
                 "Rows = preceding operation, columns = current operation, all trials (face and place together). Counts summed over "
                 "subjects; colour = standardised residual against an even mix (an even mix would give every cell the expected count printed in the title).")
    for r, val in enumerate(("pos", "neg")):
        table = results["op_tables"][val].drop(index=START)
        exp, resid = expected_and_residuals(results["op_tables"][val])
        draw_matrix(axes[r, 0], resid.values, list(table.index), list(table.columns), _count_labels(table), 0, 8,
                    f"{val}: expected {exp.values.mean():.0f} per cell if counterbalanced", "current operation", "preceding operation", 10)
        ax = axes[r, 1]
        x = np.arange(len(ops))
        for k, (cat, color) in enumerate((("face", FACE_COLOR), ("place", PLACE_COLOR))):
            vals = [float(rates[(rates["valence"] == val) & (rates["operation"] == o) & (rates["category"] == cat)]["same_operation"].iloc[0]) for o in ops]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=f"current trial is {cat}")
            for xi, v in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(xi, v + 0.01, f"{100 * v:.0f}%", ha="center", fontsize=8)
        ax.axhline(1 / len(ops), color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ops)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("share that follow the same operation")
        ax.set_title(f"{val}: back-to-back same operation (dotted = even mix)", fontsize=10)
        ax.legend(fontsize=8)
    return fig


def page_category_order(results):
    import matplotlib.pyplot as plt
    trials, rates = results["trials"], results["repeat_rates"]
    ops = results["operations"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), gridspec_kw=dict(width_ratios=[0.8, 1, 1], top=0.86, bottom=0.12, hspace=0.5, wspace=0.4))
    _page_header(fig, "Category order: face and place, ignoring the operation",
                 "(left) preceding x current category counts; (middle) how many face and place trials each operation has (balance of the "
                 "cue image within an operation); (right) share of trials whose predecessor had the same category (dotted = 50%).")
    for r, val in enumerate(("pos", "neg")):
        t = _with_predecessor(trials[trials["valence"] == val])
        counts = pd.crosstab(t["previous_category"], t["category"]).reindex(index=["face", "place"], columns=["face", "place"], fill_value=0)
        expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / counts.values.sum()
        resid = (counts.values - expected) / np.sqrt(expected)
        draw_matrix(axes[r, 0], resid, ["face", "place"], ["face", "place"], _count_labels(counts), 0, 8,
                    f"{val}: preceding x current category", "current", "preceding", 11)
        ax = axes[r, 1]
        n = trials[trials["valence"] == val].groupby(["operation", "category"]).size().unstack().reindex(ops)
        x = np.arange(len(ops))
        ax.bar(x - 0.2, n["face"], 0.4, color=FACE_COLOR, label="face")
        ax.bar(x + 0.2, n["place"], 0.4, color=PLACE_COLOR, label="place")
        ax.set_xticks(x)
        ax.set_xticklabels(ops, fontsize=8)
        ax.set_ylim(0, n.values.max() * 1.2)
        ax.set_ylabel("trials")
        ax.set_title(f"{val}: trials per operation", fontsize=10)
        ax.legend(fontsize=8)
        ax = axes[r, 2]
        for k, (cat, color) in enumerate((("face", FACE_COLOR), ("place", PLACE_COLOR))):
            vals = [float(rates[(rates["valence"] == val) & (rates["operation"] == o) & (rates["category"] == cat)]["same_category"].iloc[0]) for o in ops]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=f"current is {cat}")
            for xi, v in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(xi, v + 0.01, f"{100 * v:.0f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ops, fontsize=8)
        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("share with a same-category predecessor")
        ax.set_title(f"{val}: same category as the preceding trial", fontsize=10)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    return fig


def _trial_type_order(ops):
    return [(op, cat) for cat in ("face", "place") for op in ops]


def page_previous_category_by_type(results):
    import matplotlib.pyplot as plt
    trials, ops = results["trials"], results["operations"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw=dict(top=0.86, bottom=0.08, wspace=0.12))
    _page_header(fig, "Previous trial's category, for every trial type",
                 "For each current trial type (operation x face/place; the first group is face trials, the second place trials): the share of its "
                 "trials that were preceded by a face trial (blue) or a place trial (orange). 50% everywhere would be balanced. n = trials with a predecessor.")
    order = _trial_type_order(ops)
    for ax, val in zip(axes, ("pos", "neg")):
        t = _with_predecessor(trials[trials["valence"] == val])
        y = np.arange(len(order)) + (np.arange(len(order)) >= len(ops)) * 0.8
        for yi, (op, cat) in zip(y, order):
            g = t[(t["operation"] == op) & (t["category"] == cat)]
            face = (g["previous_category"] == "face").mean()
            ax.barh(yi, face, color=FACE_COLOR, height=0.7)
            ax.barh(yi, 1 - face, left=face, color=PLACE_COLOR, height=0.7)
            ax.text(face / 2, yi, f"{100 * face:.0f}%", ha="center", va="center", color="white", fontsize=9)
            ax.text(face + (1 - face) / 2, yi, f"{100 * (1 - face):.0f}%", ha="center", va="center", color="white", fontsize=9)
            ax.text(1.02, yi, f"n={len(g)}", va="center", fontsize=8)
        ax.axvline(0.5, color="k", ls=":", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{op} {cat}" for op, cat in order], fontsize=9)
        ax.set_ylim(y[-1] + 0.6, -1.5)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.set_title(f"{val}: preceding trial was face / place", fontsize=11)
        if val == "neg":
            ax.set_yticklabels([])
        ax.text(0.5, y[0] - 0.8, "current trial = FACE", ha="center", fontsize=9, fontweight="bold", color=FACE_COLOR)
        ax.text(0.5, y[len(ops)] - 0.75, "current trial = PLACE", ha="center", fontsize=9, fontweight="bold", color=PLACE_COLOR)
    return fig


def page_previous_operation_by_type(results):
    import matplotlib.pyplot as plt
    trials, ops = results["trials"], results["operations"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw=dict(top=0.86, bottom=0.12, left=0.17, wspace=0.55))
    _page_header(fig, "Previous trial's operation, for every trial type",
                 "Rows = current trial type (operation x face/place), columns = the operation of the preceding trial; cells = % of that row's trials "
                 f"(n in the row label). An even mix is {100 / len(ops):.0f}% in every cell: red = more often than that, blue = less often.")
    order = _trial_type_order(ops)
    for ax, val in zip(axes, ("pos", "neg")):
        t = _with_predecessor(trials[trials["valence"] == val])
        values, labels, n_col = [], [], []
        for op, cat in order:
            g = t[(t["operation"] == op) & (t["category"] == cat)]
            share = g["previous_operation"].value_counts(normalize=True).reindex(ops, fill_value=0)
            values.append(100 * share.values)
            labels.append([f"{100 * v:.0f}%" for v in share.values])
            n_col.append(len(g))
        rows = [f"{op} {cat} (n={n})" for (op, cat), n in zip(order, n_col)]
        draw_matrix(ax, np.array(values), rows, ops, labels, 100 / len(ops), 25, f"{val}", "preceding operation", "", 10)
        ax.axhline(len(ops), color="k", lw=2)
    return fig


def page_previous_type_by_type(results):
    import matplotlib.pyplot as plt
    trials, ops = results["trials"], results["operations"]
    types = [f"{op}_{cat}" for op, cat in _trial_type_order(ops)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw=dict(top=0.86, bottom=0.14, wspace=0.12))
    _page_header(fig, "Previous trial type (operation and category), for every trial type",
                 "Rows = current trial type, columns = preceding trial type, cells = % of the row's trials. An even mix is 12.5% in every cell; "
                 "0% cells are transitions that never occur in these lists.")
    for ax, val in zip(axes, ("pos", "neg")):
        t = _with_predecessor(trials[trials["valence"] == val])
        counts = pd.crosstab(t["trial_type"], t["previous_trial"]).reindex(index=types, columns=types, fill_value=0)
        pct = 100 * counts.div(counts.sum(axis=1), axis=0)
        labels = [[f"{v:.0f}" for v in row] for row in pct.values]
        draw_matrix(ax, pct.values, types, types, labels, 12.5, 15, val, "preceding trial type", "", 8)
        ax.axhline(len(ops), color="k", lw=1.5)
        ax.axvline(len(ops), color="k", lw=1.5)
        if val == "neg":
            ax.set_yticklabels([])
    return fig


def page_position_and_iti(results):
    import matplotlib.pyplot as plt
    trials, ops = results["trials"], results["operations"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw=dict(top=0.84, bottom=0.14, hspace=0.55, wspace=0.25))
    _page_header(fig, "Position in the run and ITI, by trial type",
                 "(left) share of each operation's trials that fall in the first, middle and last third of a run (a third each would be balanced); "
                 "(right) mean ITI before each trial type: seconds from the end of the previous operation event to this view cue (error bars: SD across trials).")
    sp = results["serial_position"]
    thirds = ["first third", "middle third", "last third"]
    colors = ["#c6dbef", "#6baed6", "#08519c"]
    order = _trial_type_order(ops)
    for r, val in enumerate(("pos", "neg")):
        ax = axes[r, 0]
        s = sp[sp["valence"] == val].set_index("operation").reindex(ops)
        share = s[thirds].div(s[thirds].sum(axis=1), axis=0)
        left = np.zeros(len(ops))
        for third, color in zip(thirds, colors):
            ax.barh(np.arange(len(ops)), share[third], left=left, color=color, label=third)
            for i, (l, v) in enumerate(zip(left, share[third])):
                ax.text(l + v / 2, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9, color="white" if third != thirds[0] else "black")
            left += share[third].values
        ax.axvline(1 / 3, color="k", ls=":", lw=1)
        ax.axvline(2 / 3, color="k", ls=":", lw=1)
        ax.set_yticks(np.arange(len(ops)))
        ax.set_yticklabels(ops)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"])
        ax.set_xlabel("share of the operation's trials")
        ax.set_title(f"{val}: where in the run", fontsize=10)
        if r == 0:
            ax.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.1), frameon=False)
        ax = axes[r, 1]
        t = _with_predecessor(trials[trials["valence"] == val])
        means = [t[(t["operation"] == op) & (t["category"] == cat)]["iti_s"].agg(["mean", "std"]) for op, cat in order]
        x = np.arange(len(order))
        ax.bar(x, [m["mean"] for m in means], yerr=[m["std"] for m in means],
               color=[FACE_COLOR if c == "face" else PLACE_COLOR for _, c in order], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{op} {cat}" for op, cat in order], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("ITI (s)")
        ax.set_ylim(0, 6)
        ax.set_title(f"{val}: ITI before the trial", fontsize=10)
    return fig


def make_pdf(results: dict, path: str):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pages = (page_overview, page_operation_order, page_category_order, page_previous_category_by_type,
             page_previous_operation_by_type, page_previous_type_by_type, page_position_and_iti)
    with PdfPages(path) as pdf:
        for page in pages:
            fig = page(results)
            fig.text(0.98, 0.012, results["study"], ha="right", fontsize=7, color="#888888")
            pdf.savefig(fig)
            plt.close(fig)


# =====================================================
# Driver
# =====================================================

def parse_held_out(text: str):
    return [[int(r) for r in part.split(",")] for part in text.split(";") if part.strip()]


def analyse(events: pd.DataFrame, study: str, operations=None, held_out_runs=None) -> dict:
    operations = list(operations) if operations else infer_operations(events)
    trials = build_trials(events, operations)
    types = [f"{op}_{cat}" for op in operations for cat in ("face", "place")]
    out = dict(study=study, operations=operations, trials=trials)
    out["op_tables"], out["type_tables"], out["tests"] = {}, {}, {}
    for val in ("pos", "neg"):
        t = trials[trials["valence"] == val]
        out["op_tables"][val] = transitions(t, "previous_operation", "operation", [START] + operations, operations)
        out["type_tables"][val] = transitions(t, "previous_trial", "trial_type", [START] + types, types)
        out["tests"][f"{val}_operation"] = independence(out["op_tables"][val])
        out["tests"][f"{val}_operation_x_category"] = independence(out["type_tables"][val])
    pooled_ops = transitions(trials, "previous_operation", "operation", [START] + operations, operations)
    out["tests"]["all_operation"] = independence(pooled_ops)
    out["type_counts"] = type_counts(trials)
    out["repeat_rates"] = repeat_rates(trials)
    out["dropping_cost"] = dropping_cost(trials)
    out["balanced_by_predecessor"] = balanced_by_predecessor(trials)
    out["order_lists"] = order_lists(trials)
    out["run_layout"] = run_layout(trials)
    out["serial_position"] = serial_position(trials)
    out["iti"] = iti_by_operation(trials)
    out["folds"] = fold_balance(trials, held_out_runs or DEFAULT_HELD_OUT_RUNS)
    out["order_predictability"] = order_predictability(trials, held_out_runs or DEFAULT_HELD_OUT_RUNS)
    out["subject_checks"] = data_checks(events, trials)
    return out


def save(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    study = results["study"]
    results["trials"].to_csv(os.path.join(output_dir, "trials.csv"), index=False)
    for name in ("type_counts", "repeat_rates", "dropping_cost", "balanced_by_predecessor", "order_lists", "run_layout",
                 "serial_position", "iti", "folds", "order_predictability", "subject_checks"):
        results[name].to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)
    for val in ("pos", "neg"):
        for key, table in (("operation", results["op_tables"][val]), ("operation_x_category", results["type_tables"][val])):
            table.to_csv(os.path.join(output_dir, f"transitions_{val}_{key}_counts.csv"))
            exp, resid = expected_and_residuals(table)
            exp.round(1).to_csv(os.path.join(output_dir, f"transitions_{val}_{key}_expected.csv"))
            resid.round(2).to_csv(os.path.join(output_dir, f"transitions_{val}_{key}_residuals.csv"))
    with open(os.path.join(output_dir, "tests.json"), "w") as f:
        json.dump(results["tests"], f, indent=2)
    plot_transitions(results["op_tables"], results["type_tables"], results["repeat_rates"], study,
                     os.path.join(output_dir, "transitions_overview.png"))
    make_pdf(results, os.path.join(output_dir, f"counterbalance_{study.lower()}.pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master-spreadsheet", required=True)
    parser.add_argument("--study", required=True, help="label used in titles")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--operations", nargs="+", default=None, help="operation names (default: inferred)")
    parser.add_argument("--held-out-runs", default=None, help='cross-validation folds as run numbers, e.g. "1,3;2,4;5,7;6,8"')
    args = parser.parse_args()
    events = read_events(args.master_spreadsheet)
    results = analyse(events, args.study, args.operations, parse_held_out(args.held_out_runs) if args.held_out_runs else None)
    save(results, args.output_dir)
    print(json.dumps(results["tests"], indent=1))


if __name__ == "__main__":
    main()
