#!/usr/bin/env python3
"""
Everything the evidence notebooks share (valence_evidence_by_operation.ipynb for the category classifiers,
operation_evidence_by_category.ipynb for the operation classifier), so each notebook holds only its own analysis and plots:

  settings        configure(**settings) hands the notebook's settings to these functions (the footnotes, windows and
                  split-half plot read them); setting(name) reads one
  loading         prepare_evidence(...): the group decoding table -> one row per trial and TR over each trial's own extended
                  window, self-evidence as `value`, optional balancing of the preceding trial and baseline correction
  stat windows    make_windows, drop_baseline_bins, without_baseline_bins, pos_vs_neg_is_zero_in_baseline
  event markers   compute_marker_positions (the dotted lines for view / operation / fixation)
  figure layout   title_extra_in, title_scale, significance_note, figure_notes, baseline_note_pad, draw_footnotes,
                  add_shared_legend, plot_split_half_findings
  pages           importance_overlap_table, order_pages, add_image_page, combine_pages
  teaching        make_toy_data (the worked example at the end of both notebooks)
"""

import os
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repo root
from workflows.generate_report import broadcast_trial_label, resolve_marker_label, draw_event_annotations   # noqa: E402
from utils.mvpa_common import is_excluded_trial_type, is_trial_end_marker              # noqa: E402
from _interactive_notebooks.decoding_results_toolkit import (                           # noqa: E402
    select_evidence_value, bin_by_size, sliding_windows, max_reliable_window_index, baseline_correct_timecourses,
    baseline_only_bins, balance_preceding_category)

# =====================================================
# Settings
# =====================================================

SETTING_NAMES = ("MODEL_HEAD", "BASELINE_NOTE", "FDR_METHOD", "STAT_METHOD", "ALPHA", "BIN_SIZE", "SLIDING", "FDR_FIRST_TR",
                 "POS_VS_NEG_ALTERNATIVE", "OP_VS_MAINTAIN_ALTERNATIVE", "BASELINE_ONLY_BINS", "TEST_BASELINE_BIN_WHEN_NOT_ZERO",
                 "BASELINE_GROUP_BY", "TR_SECONDS", "EVENT_MARKERS", "N_SPLITS")
settings = {}


def configure(namespace=None, **kwargs):
    """Hand settings to the functions below: configure(globals()) takes every SETTING_NAMES entry the notebook has defined
    (call it after the configuration cell, and again once MODEL_HEAD / BASELINE_NOTE / EVENT_MARKERS exist)."""
    if namespace is not None:
        settings.update({n: namespace[n] for n in SETTING_NAMES if n in namespace})
    settings.update(kwargs)


def setting(name, default=None):
    return settings.get(name, default)


# =====================================================
# Loading
# =====================================================

def prepare_evidence(cfg: dict, primary: str, secondary: str, secondary_regex: str) -> SimpleNamespace:
    """The group decoding table -> the analysis table.

    `cfg` holds the notebook's settings (GROUP_CSV_PATH, DESC, MODEL_TITLE, TR_SECONDS, MIN_SUBJECT_FRACTION, EXTRA_TRAILING_TRS,
    BALANCE_PRECEDING_CATEGORY, BALANCE_WITHIN, BALANCE_MODE, BALANCE_SEED, BASELINE_CORRECTION, BASELINE_TIMEPOINTS,
    BASELINE_GROUP_BY, DEFAULT_BASELINE_GROUP_BY, OUTPUT_DIR, BIN_SIZE, SLIDING). `primary` is the column that holds the
    classifier's own target (its regressor_label: "object" for the category classifiers, "operation" for the operation
    classifier); `secondary` is the other label, read from trial_type with `secondary_regex` ("operation" / "category").

    Every trial gets one window of the same length from its own start (the reliable per-trial length plus
    EXTRA_TRAILING_TRS), the evidence is each row's self-evidence for the OWNING trial, and valence comes from the task.
    Returns a namespace with df, WEIGHT_COL, BASELINE_WINDOWS, BASELINE_ONLY_BINS, BASELINE_NOTE, MODEL_HEAD, MODEL_TITLE,
    OPERATION_HAS_OWN_BASELINE and the window lengths; it also configures the settings of this module."""
    path = cfg["GROUP_CSV_PATH"]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path!r} not found -- GROUP_CSV_PATH must be <model folder>/raw_data/group_decoding_results.csv "
            f"(parse a downloaded stage-4 folder into the layout with results_layout.py ingest)")
    df = pd.read_csv(path, dtype={"subject": str}, keep_default_na=True, na_values=[""])
    title = cfg.get("MODEL_TITLE") or cfg["DESC"]
    model_head = title + ("\n" if "\n" in title else "")        # a multi-line model title gets a blank line before the figure's own title
    print(f"Loaded {len(df)} rows from {path} -- {df['subject'].nunique()} subject(s)")

    # a blank CSV field reads back as a real NaN; replace() is a cheap safety net for a literal empty string
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # regressor_label is only set on the specific real event the classifier's `conditions` target; broadcasting it over every
    # row of its own trial (boldfile, trial_index) keeps "which trial type was this" defined for the whole trial
    df["regressor_label"] = broadcast_trial_label(df, "regressor_label")
    df["valence"] = df["task"].map({"WMpos": "pos", "WMneg": "neg"})
    df[primary] = df["regressor_label"]                                  # NaN on unscored frames
    # the other label is literally present only on part of a trial's frames ("maintain_face" / "view_face" ...): extract, then
    # broadcast it so a groupby keyed on it does not silently drop the rest of the trial (pandas drops NaN keys)
    df[secondary] = df["trial_type"].str.extract(secondary_regex, expand=False)
    df[secondary] = broadcast_trial_label(df, secondary)
    keys = [primary, secondary, "valence"]

    # self-evidence for each row's OWN natural trial: only to find the reliable per-trial cutoff; the plotted "value" is
    # recomputed for the OWNING trial after the windows are extended
    natural = select_evidence_value(df, value="self", new_col="value")
    max_window = max_reliable_window_index(natural, group_cols=keys, min_fraction=cfg["MIN_SUBJECT_FRACTION"])
    n_dropped = (natural["window_index"] > max_window).sum()
    print(f"reliable per-trial coverage extends to window_index <= {max_window} "
          f"({n_dropped} row(s) beyond it too sparse under each trial's own natural length)")
    window_len = max_window + 1 + cfg["EXTRA_TRAILING_TRS"]

    # real trial starts only: window_index == 0 AND a genuine anchor match
    owner = {k: f"owner_{k}" for k in keys}
    trial_starts = df.loc[(df["window_index"] == 0) & df["regressor_label"].notna(),
                          ["boldfile", "trial_index", "volume_of_interest", *keys]].copy()
    trial_starts = trial_starts.rename(columns={"trial_index": "owner_trial_index", "volume_of_interest": "start_vol", **owner})

    # optional: give trials after a same-category and after a different-category trial the same say
    balance = cfg.get("BALANCE_PRECEDING_CATEGORY", False)
    if balance:
        balanced_trials, balance_report = balance_preceding_category(df, within=cfg["BALANCE_WITHIN"], seed=cfg["BALANCE_SEED"], mode=cfg["BALANCE_MODE"])
        n_cells = len(balance_report)
        n_empty = int((balance_report["n_kept_per_group"] == 0).sum())
        n_all = int(balance_report[["n_same_category", "n_different_or_no_category"]].sum().sum())
        print(f"balancing the preceding trial within each participant x {' x '.join(cfg['BALANCE_WITHIN'])}: "
              f"{len(balanced_trials)} of {n_all} trials used ({cfg['BALANCE_MODE']}); "
              f"effective {2 * int(balance_report['n_kept_per_group'].sum())} trials' worth ({int(balance_report['n_kept_per_group'].sum())} after a same-category trial, the same number after a different/no-category trial); "
              f"{n_empty} of {n_cells} participant cells have no trial in one group and keep nothing; "
              f"{balance_report.attrs.get('n_trials_with_unknown_predecessor', 0)} trial(s) left out because the previous trial has no type")
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        balance_report.to_csv(os.path.join(cfg["OUTPUT_DIR"], "preceding_category_balance.csv"), index=False)
        trial_starts = trial_starts.merge(
            balanced_trials[["boldfile", "trial_index", "weight"]].rename(columns={"trial_index": "owner_trial_index"}),
            on=["boldfile", "owner_trial_index"], how="inner")
    else:
        print("preceding-trial balancing OFF: every trial used")
    weight_col = "weight" if (balance and cfg["BALANCE_MODE"] == "weighted") else None      # weighted mode: every mean is weighted

    # pull every row within window_len TRs of each trial's OWN start by absolute volume (a short trial's window keeps going into
    # the next trial's frames, which land in the output once per owner: the intended double-dip)
    frames = []
    for offset in range(window_len):
        probe = trial_starts.assign(volume_of_interest=trial_starts["start_vol"] + offset)
        hit = df.merge(probe, on=["boldfile", "volume_of_interest"])
        hit["window_index"] = offset
        frames.append(hit)
    df = pd.concat(frames, ignore_index=True)

    # every row is now keyed on the OWNING trial: a borrowed tail frame's evidence is "how much did the classifier still
    # believe this was the ORIGINATING trial's label"
    df["trial_index"] = df["owner_trial_index"]
    df["regressor_label"] = df[owner[primary]]
    for k in keys:
        df[k] = df[owner[k]]
    df = select_evidence_value(df, value="self", new_col="value")
    df["time_sec"] = df["window_index"] * cfg["TR_SECONDS"]

    # optional baseline correction
    baseline_windows = tuple(tp - 1 for tp in cfg["BASELINE_TIMEPOINTS"])               # window_index counts from 0
    only_bins = baseline_only_bins(baseline_windows, cfg["BIN_SIZE"], sliding=cfg.get("SLIDING", False)) if cfg["BASELINE_CORRECTION"] else set()
    group_by = tuple(cfg["BASELINE_GROUP_BY"])
    has_own_baseline = {primary, secondary, "valence"} <= set(group_by)                  # operation comparisons, on any subset of trials
    if cfg["BASELINE_CORRECTION"]:
        df, n_no_baseline = baseline_correct_timecourses(df, ["subject", *group_by], baseline_windows, weight_col=weight_col)
        print(f"baseline correction ON: timepoints {cfg['BASELINE_TIMEPOINTS']} of each participant's {' x '.join(group_by)} "
              f"time course subtracted from every timepoint ({n_no_baseline} row(s) dropped for lack of a baseline; "
              f"stat window(s) {sorted(only_bins)} lie wholly inside the baseline)")
        if cfg.get("TEST_BASELINE_BIN_WHEN_NOT_ZERO", True) and not has_own_baseline:
            print("  (coarser baseline groups than operation x category x valence: a window inside the baseline is skipped only for comparisons "
                  "whose curves average zero there by construction; the others are tested there too, outside the FDR family)")
    else:
        print("baseline correction OFF: raw evidence")

    # footnote for every figure when the baseline is not the default one-per-operation-x-category-x-valence
    note = None
    default_group = tuple(cfg["DEFAULT_BASELINE_GROUP_BY"])
    if cfg["BASELINE_CORRECTION"] and group_by != default_group:
        pooled = [n for n in default_group if n not in group_by]
        order = {"valence": 0, "operation": 1}
        by = sorted(group_by, key=lambda g: order.get(g, 2))
        word = {"pos": "positive", "neg": "negative"}
        groups = sorted({"-".join(word.get(str(row[g]), str(row[g])) for g in by)
                         for _, row in df[list(group_by)].dropna().drop_duplicates().iterrows()},
                        key=lambda name: (0 if name.startswith("positive") else 1, name))
        trs = cfg["BASELINE_TIMEPOINTS"]
        note = (f"Baseline correction independently calculated on item features (i.e. {', '.join(groups)})."
                + (f" Trials pooled by {' and '.join(pooled)} before baseline correction computed." if pooled else "")
                + f" Each participant's baseline = the mean over TRs {min(trs)}-{max(trs)} of all their trials in the group, subtracted from every TR.")
        print(note)

    out = SimpleNamespace(df=df, WEIGHT_COL=weight_col, BASELINE_WINDOWS=baseline_windows, BASELINE_ONLY_BINS=only_bins,
                          BASELINE_NOTE=note, MODEL_HEAD=model_head, MODEL_TITLE=title, OPERATION_HAS_OWN_BASELINE=has_own_baseline,
                          max_window=max_window, WINDOW_LEN=window_len)
    configure(MODEL_HEAD=model_head, BASELINE_NOTE=note, BASELINE_ONLY_BINS=only_bins, BASELINE_GROUP_BY=group_by)
    return out


# =====================================================
# Statistics windows
# =====================================================

def make_windows(wide):
    """(per_subject_bin, bins_meta) of a subject x TR table: the stat windows of the STAT_WINDOW setting (non-overlapping BIN_SIZE-TR
    bins, or a BIN_SIZE-TR window slid one TR at a time)."""
    if setting("SLIDING", False):
        return sliding_windows(wide, window=setting("BIN_SIZE", 3), tr_offset=0)
    return bin_by_size(wide, bin_size=setting("BIN_SIZE", 3), tr_offset=0)


def _skip_baseline(zero_by_construction):
    only_bins = setting("BASELINE_ONLY_BINS") or set()
    return bool(only_bins) and not (setting("TEST_BASELINE_BIN_WHEN_NOT_ZERO", True) and not zero_by_construction), only_bins


def drop_baseline_bins(per_subject_bin, zero_by_construction=True):
    """Stat bins wholly inside the baseline average to zero after correction, so they are not tested (unless
    TEST_BASELINE_BIN_WHEN_NOT_ZERO and the comparison does not average to zero there)."""
    skip, only_bins = _skip_baseline(zero_by_construction)
    if not skip: return per_subject_bin
    return per_subject_bin[~per_subject_bin.index.get_level_values("bin").isin(only_bins)]


def without_baseline_bins(table, zero_by_construction=True):
    skip, only_bins = _skip_baseline(zero_by_construction)
    if not skip: return table
    return table[~table["bin"].isin(only_bins)]


def pos_vs_neg_is_zero_in_baseline(operation, operations_pooled_ok=True):
    """pos and neg curves of one operation (operation=None: all operations pooled) are whole baseline groups (or pooled over
    whole groups), so their difference averages zero over the baseline TRs."""
    group_by = setting("BASELINE_GROUP_BY", ())
    return "valence" in group_by and ("operation" in group_by or operation is None)


# =====================================================
# Event markers
# =====================================================

def compute_marker_positions(data, annotation_labels=None, min_frequency=0.5, max_label_values=4):
    """Where the view / operation / fixation periods fall: one dotted-line marker per ordinal sub-event position that most trials
    have, in the format draw_event_annotations takes (from this table's own real, un-broadcast trial_type and window_index)."""
    events = data[["boldfile", "trial_index", "window_index", "trial_type"]].drop_duplicates()
    # a frame with no real event covering it (a brief, genuine gap in the
    # dense per-volume table -- e.g. between one event ending and the next
    # starting) must not count as its own ordinal position, or it shifts
    # every later position by one for whichever trials happen to have that
    # gap -- same reason compute_event_markers drops NaN event_index rows
    # before counting positions
    events = events[events["trial_type"].notna()]
    events = events.sort_values(["boldfile", "trial_index", "window_index"])

    # merge consecutive same-trial_type rows within a trial into one
    # sub-event -- mirrors compute_event_markers' own merge step (a
    # repeated run of the same real value is one occurrence, not one
    # per TR)
    same_trial = (events["boldfile"] == events["boldfile"].shift()) & (events["trial_index"] == events["trial_index"].shift())
    same_type = events["trial_type"] == events["trial_type"].shift()
    run_id = (~(same_trial & same_type)).cumsum()
    merged = events.groupby(run_id, sort=False).agg(
        boldfile=("boldfile", "first"), trial_index=("trial_index", "first"),
        trial_type=("trial_type", "first"), start=("window_index", "min"),
    ).reset_index(drop=True)

    merged["position"] = merged.groupby(["boldfile", "trial_index"]).cumcount() + 1

    # a trial_index whose own first frame is itself a block-boundary marker
    # (EndFixation/start_block/end_block) was created by partition_into_trials'
    # end_cut resetting the window, not by a real trial anchor -- it's
    # administrative filler, not a trial, and must not contribute its own
    # "position 1" here (mixing with a real trial's own position 1 and
    # corrupting resolve_marker_label's result there)
    admin_starts = merged[(merged["position"] == 1) & merged["trial_type"].apply(is_trial_end_marker)]
    admin_keys = set(admin_starts[["boldfile", "trial_index"]].itertuples(index=False, name=None))
    if admin_keys:
        merged = merged[~merged[["boldfile", "trial_index"]].apply(tuple, axis=1).isin(admin_keys)]

    n_trials = merged[["boldfile", "trial_index"]].drop_duplicates().shape[0]

    markers = []
    for position, group in merged.groupby("position"):
        frequency = group[["boldfile", "trial_index"]].drop_duplicates().shape[0] / n_trials
        if frequency < min_frequency:
            continue
        distinct_types = sorted(group["trial_type"].dropna().unique())
        # a "_nan"/"nan_..." trial_type (a malformed trial with a missing
        # item) is noise, not a real value worth covering or displaying
        real_types = [t for t in distinct_types if "nan" not in str(t).lower()] or distinct_types
        # a position that's only ever administrative/excluded content
        # (fixation/trial_fixation/EndFixation/...) is a rest/gap marker,
        # not a condition worth naming
        if all(is_excluded_trial_type(t) for t in real_types):
            label = ""
        else:
            label = resolve_marker_label(real_types, annotation_labels, max_label_values)
        markers.append({
            "trial_type": label,
            "mean_start": float(group["start"].mean()),
            "std_start": float(group["start"].std(ddof=0)) if len(group) > 1 else 0.0,
        })
    return sorted(markers, key=lambda m: m["mean_start"])


# =====================================================
# Figure layout
# =====================================================

def title_extra_in():
    """Inches ADDED to a figure's height for a multi-line MODEL_TITLE (the extra lines plus the blank line that
    separates the model details from the figure's own title), so the plots keep their size. 0 for a one-line title."""
    return 0.2 * str(setting("MODEL_HEAD") or "").count("\n")


def title_scale(fig):
    """Multiplier for the vertical margins that were designed for the figure without the extra title lines
    (fractions of the height): the same inches above the bottom edge, on the taller figure."""
    height = fig.get_figheight()
    return (height - title_extra_in()) / height


def significance_note(kind):
    """The footnote about significance for one kind of page: "valence" (pos vs. neg tests), "operation" (operation vs.
    maintain / vs. operation tests) or "split" (the descriptive split-half page). Pages without tests have no note."""
    fdr_method, first_tr, sliding, bin_size = setting("FDR_METHOD"), setting("FDR_FIRST_TR"), setting("SLIDING", False), setting("BIN_SIZE", 3)
    alpha = setting("ALPHA", 0.05)
    fdr_name = {"bh": "Benjamini-Hochberg", "by": "Benjamini-Yekutieli"}.get(fdr_method)
    corrected = f"FDR-corrected ({fdr_name}) " if fdr_method else "uncorrected "
    tests = {"ttest_1samp_diff": "one-sample t-test of the subjects' paired differences", "ttest_rel": "paired t-test",
             "wilcoxon": "Wilcoxon signed-rank test"}.get(setting("STAT_METHOD"), str(setting("STAT_METHOD")))
    window = (f"{bin_size}-TR windows slid one TR at a time (one test per TR, drawn at the centre TR)" if sliding
              else f"non-overlapping {bin_size}-TR windows")
    if first_tr is None:
        start = "Every window is tested and corrected together within its comparison"
        later = ""
    else:
        start = (f"The statistics start at TR {first_tr} (TRs 1-{first_tr - 1} are not tested), and each comparison's "
                 f"FDR family is its windows from TR {first_tr} on")
        later = f" from TR {first_tr} on (TRs 1-{first_tr - 1} not tested)"
    if kind == "split":
        return (f"No significance test on this page: split-half reliability is descriptive and covers every window. "
                f"The significance tests (other pages) are {corrected}p < {alpha}{later}.")
    alternative = setting("POS_VS_NEG_ALTERNATIVE") if kind == "valence" else setting("OP_VS_MAINTAIN_ALTERNATIVE")
    note = f"Significant (shaded cell, *) = {corrected}p < {alpha}, {alternative} {tests}, on {window}. {start}."
    if kind == "operation":
        note += " Rows 'vs. maintain' count only windows where the operation is below maintain; the other rows count either direction."
    if sliding:
        note += (" Neighbouring windows overlap, so their tests are correlated; the tables also give p_fdr_by "
                 "(Benjamini-Yekutieli, valid under any dependence).")
    return note


def figure_notes(fig, fontsize=7):
    """Footnote text of a figure, wrapped to its width: the baseline note (if any) and the significance note of
    the page kind the plot function set on it (fig._note_kind)."""
    paragraphs = [setting("BASELINE_NOTE")] if setting("BASELINE_NOTE") else []
    if getattr(fig, "_note_kind", None):
        paragraphs.append(significance_note(fig._note_kind))
    width = int(fig.get_figwidth() * 105 / fontsize)
    return "\n".join(textwrap.fill(p, width=width) for p in paragraphs)


def baseline_note_pad(fig, fontsize=7):
    """Fraction of the figure height reserved at the bottom for the footnotes (baseline note + significance note)."""
    text = figure_notes(fig, fontsize)
    return (0.06 + (text.count("\n") + 1) * fontsize * 1.25 / 72) / fig.get_figheight() if text else 0.0


def draw_footnotes(fig, fontsize=7):
    """The figure's footnotes, centred along its bottom edge (add_shared_legend puts its legend just above them)."""
    text = figure_notes(fig, fontsize)
    if text:
        fig.text(0.5, 0.006, text, ha="center", va="bottom", fontsize=fontsize, color="0.25")


def add_shared_legend(fig, color_map, title=None, ncol=None, extra_handles=None):
    """One legend for the whole figure, placed below every subplot -- built directly from color_map ({label: color}) rather
    than pulled from any one subplot's own handles, so it is correct whichever subplots have data for every hue level. A small
    footnote underneath gives the baseline groups (when unusual) and what 'significant' means on this page."""
    pad = baseline_note_pad(fig)
    handles = [Line2D([0], [0], color=color, lw=2, label=label) for label, color in color_map.items()] + list(extra_handles or [])
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
               title=title, bbox_to_anchor=(0.5, 0.02 + pad), fontsize=9)   # inside the figure, so a saved PDF page does not cut it off
    draw_footnotes(fig)


def plot_split_half_findings(table):
    """One panel per contrast of the split-half table (line = full-data group mean difference, band = 2.5-97.5th percentile of the
    half-split estimates, label = % of halves with the full-data direction), one row per family of contrasts."""
    tr_seconds, n_splits, markers = setting("TR_SECONDS"), setting("N_SPLITS"), setting("EVENT_MARKERS")
    families = list(dict.fromkeys(table["family"]))
    ncols = table.groupby("family")["contrast"].nunique().max()
    fig, axes = plt.subplots(len(families), ncols, figsize=(4.4 * ncols, 3.3 * len(families) + title_extra_in()), sharex=True, squeeze=False)
    for row, family in enumerate(families):
        contrasts = list(dict.fromkeys(table.loc[table["family"] == family, "contrast"]))
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(contrasts):
                ax.axis("off")
                continue
            t = table[table["contrast"] == contrasts[col]]
            x = (t["tr_start"] + t["tr_end"]) / 2 * tr_seconds
            ax.fill_between(x, t["half_ci_low"], t["half_ci_high"], color="gray", alpha=0.3, linewidth=0)
            ax.plot(x, t["full_mean_diff"], "o-", color="black", markersize=3)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            for xi, hi, p in zip(x, t["half_ci_high"], t["p_same_direction"]):
                if np.isfinite(p):
                    ax.annotate(f"{p:.0%}", (xi, hi), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6)
            draw_event_annotations(ax, markers, tr=tr_seconds, show_labels=False)
            ax.set_title(contrasts[col], fontsize=9)
            ax.margins(y=0.15)  # headroom for the % labels
            if row == len(families) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{family}\nmean difference (a - b)", fontsize=8)   # on every panel, so each can be used alone
    fig.suptitle(f"{setting('MODEL_HEAD')}\nsplit-half reliability of the findings ({n_splits} random splits of each participant's trials)\n"
                 "line = full-data group mean difference | band = 2.5-97.5th percentile of half-split estimates | "
                 "label = % of halves with the full-data direction", fontsize=10, fontweight="bold")
    fig._note_kind = "split"
    fig.tight_layout(rect=[0, baseline_note_pad(fig), 1, 0.93 * title_scale(fig)])
    draw_footnotes(fig)
    return fig


# =====================================================
# Pages and importance-map tables
# =====================================================

def importance_overlap_table(mean, significant, operations, top_fraction: float = 0.2) -> pd.DataFrame:
    """Per class: voxels in the top `top_fraction` of |mean weight| (threshold from all classes pooled, as in
    plot_importance_surface; `mean` is the group-mean map including the zeros), voxels significant across
    subjects, and their overlap (Dice)."""
    nonzero = np.abs(mean[mean != 0])
    cut = np.percentile(nonzero, 100 * (1 - top_fraction))
    rows = []
    for i, op in enumerate(operations):
        top = np.abs(mean[..., i]) >= cut
        sig = significant[..., i]
        both = int((top & sig).sum())
        denom = int(top.sum()) + int(sig.sum())
        rows.append(dict(operation=op, voxels_top=int(top.sum()), voxels_significant=int(sig.sum()), voxels_both=both,
                         dice=(2 * both / denom) if denom else np.nan,
                         significant_positive=int((sig & (mean[..., i] > 0)).sum()),
                         significant_negative=int((sig & (mean[..., i] < 0)).sum())))
    return pd.DataFrame(rows)


def order_pages(pages, order):
    """[(name, kind, item), ...] sorted by `order` (names not listed go last, in the order given) as [(kind, item), ...]."""
    rank = lambda page: order.index(page[0]) if page[0] in order else len(order)
    return [page[1:] for page in sorted(pages, key=rank)]


def add_image_page(pdf, path: str):
    """A saved image (the 3-D surface figures) as one full PDF page: a vector copy of the meshes would be ~90 MB."""
    image = plt.imread(path)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 130.0, height / 130.0))       # the surface figures are saved at 130 dpi
    fig.add_axes([0, 0, 1, 1]).imshow(image, interpolation="lanczos")
    fig.axes[0].axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def combine_pages(pages, path: str) -> int:
    """One multi-page PDF from a list of ("figure", matplotlib Figure) -- kept as vector -- or ("image", png path)."""
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for kind, item in pages:
            if kind == "figure":
                pdf.savefig(item)
            else:
                add_image_page(pdf, item)
    return len(pages)


# =====================================================
# Teaching example
# =====================================================

def make_toy_data(seed=39, n_subjects=12, n_trials=8, n_tr=12, bin_effect=(0.0, 0.03, 0.0, 0.09)):
    """Made-up trial-level data for the worked example at the end of the notebooks: 12 subjects, no true pos - neg effect in
    bins 0 and 2, a small one in bin 1 and a clear one in bin 3."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        offset = rng.normal(0.4, 0.08)  # each subject's own baseline
        for valence, sign in (("pos", +0.5), ("neg", -0.5)):
            for trial in range(n_trials):
                for tr in range(n_tr):
                    rows.append((f"s{s:02d}", valence, "run1", trial, tr,
                                 offset + sign * bin_effect[tr // 3] + rng.normal(0, 0.25)))
    return pd.DataFrame(rows, columns=["subject", "valence", "boldfile", "trial_index", "window_index", "value"])
