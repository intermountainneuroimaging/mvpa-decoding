"""Key figures for the operation (maintain / suppress / switch / clear) classifier.

Reads what stage 4 (generate_report.py) writes and draws one figure per call:

    confusion    cross-validated accuracy and evidence matrices (true x predicted), mean across subjects
    importance   group-mean importance map of each operation on the cortical surface (fsaverage)
    timecourse   one panel per operation; each panel overlays the operation classifier's evidence for
                 that panel's own operation on the trials of pos_face, pos_place, neg_face, neg_place
                 (mean +/- 1 SE across subjects)

Run:
    python operation_results.py --group-csv <desc>_group_decoding_results.csv --output-dir <folder> \
        [--title "Operation classifier ..."]

Evidence is the raw classifier output (no baseline correction, sigmoid scale so there is no 1/4 chance level); TR 0 of every trial is the view cue and
time is unshifted (the hemodynamic lag only selects the training/testing volumes).
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _interactive_notebooks.decoding_results_toolkit import select_evidence_value   # noqa: E402
from workflows.generate_report import draw_event_annotations                     # noqa: E402

OPERATIONS = ("maintain", "suppress", "switch", "clear")
TR_SECONDS = 0.460
CATEGORY_COLORS = {"face": "#7b3294", "place": "#008837"}              # colour = category ...
VALENCE_STYLES = {"pos": "-", "neg": "--"}                              # ... line style = valence (solid pos, dashed neg)
GROUP_ORDER = [("pos", "face"), ("pos", "place"), ("neg", "face"), ("neg", "place")]


TRIAL = ["subject", "boldfile", "trial_index"]
OPERATION_ROW = r"^(?:maintain|suppress|switch|clear)_(?:face|place)$"


def trial_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps every row of a trial that has an operation period and adds, for the WHOLE trial (view cue,
    operation, ITI), `regressor_label` (its operation), `valence` (pos / neg, from the task WMpos / WMneg)
    and `category` (face / place, from the operation row's trial_type, e.g. suppress_face).

    The decoding table labels only the operation rows (regressor_label is blank on the view and ITI rows),
    so the label is carried over to the other rows of the same trial."""
    out = df.drop(columns="regressor_label").copy()
    ops = df[df["trial_type"].astype(str).str.match(OPERATION_ROW) & df["regressor_label"].notna()]
    per_trial = (ops.groupby(TRIAL)
                 .agg(regressor_label=("regressor_label", "first"), _op_type=("trial_type", "first")).reset_index())
    out = out.merge(per_trial, on=TRIAL, how="inner")
    out["valence"] = out["task"].astype(str).str.extract(r"WM(pos|neg)", expand=False)
    out["category"] = out["_op_type"].str.extract(r"_(face|place)$", expand=False)
    return out.dropna(subset=["valence", "category"]).drop(columns="_op_type")


def reliable_windows(df: pd.DataFrame, min_fraction: float = 0.9) -> int:
    """Last window_index at which at least `min_fraction` of the trials still have a decoded TR (trials
    end at different lengths; later windows would rest on a shrinking, unrepresentative set of trials)."""
    n_trials = df[TRIAL].drop_duplicates().shape[0]
    present = df.drop_duplicates(TRIAL + ["window_index"]).groupby("window_index").size() / n_trials
    ok = present[present >= min_fraction]
    return int(ok.index.max())


def phase_markers(df: pd.DataFrame) -> list:
    """Vertical-line markers in the format generate_report.draw_event_annotations takes: the median first
    window_index of the view cue, the operation period and the ITI (trial_fixation after the operation)."""
    d = df[df["trial_type"].notna()].sort_values(TRIAL + ["window_index"])
    kinds = {"view": d["trial_type"].astype(str).str.match(r"^view_"),
             "operation": d["trial_type"].astype(str).str.match(OPERATION_ROW),
             "ITI": d["trial_type"].astype(str).eq("trial_fixation")}
    op_start = d[kinds["operation"]].groupby(TRIAL)["window_index"].min()
    markers = []
    for name, mask in kinds.items():
        first = d[mask].groupby(TRIAL)["window_index"].min()
        if name == "ITI":                                   # the ITI that follows the operation, not the one before the view
            after = d[mask].merge(op_start.rename("_op").reset_index(), on=TRIAL)
            after = after[after["window_index"] > after["_op"]]
            first = after.groupby(TRIAL)["window_index"].min()
        if len(first):
            markers.append({"trial_type": name, "mean_start": float(first.median()), "std_start": float(first.std(ddof=0))})
    return sorted(markers, key=lambda m: m["mean_start"])


def group_timecourses(df: pd.DataFrame, max_window: int = None) -> pd.DataFrame:
    """Long table (operation, valence, category, window_index, mean, se, n_subjects): each subject's
    own-operation evidence is averaged over that subject's trials of the group at every window_index, then
    the subjects are averaged (se = standard error across subjects). Windows after `max_window` (default:
    the last one still decoded for 90% of trials) are left out."""
    d = trial_groups(df)
    last = reliable_windows(d) if max_window is None else max_window
    d = select_evidence_value(d[d["window_index"] <= last], "self")
    per_subject = (d.groupby(["regressor_label", "valence", "category", "window_index", "subject"])["value"]
                   .mean().reset_index())
    g = per_subject.groupby(["regressor_label", "valence", "category", "window_index"])["value"]
    out = pd.DataFrame({"mean": g.mean(), "se": g.sem(), "n_subjects": g.count()}).reset_index()
    return out.rename(columns={"regressor_label": "operation"})


def plot_timecourses(curves: pd.DataFrame, title: str = "", markers=None, operations=OPERATIONS):
    """One panel per operation, the four valence x category groups overlaid."""
    fs = dict(suptitle=15, title=13, label=11, tick=11, legend=12)
    fig, axes = plt.subplots(1, len(operations), figsize=(5.6 * len(operations), 5.4), sharey=True)
    for ax, op in zip(np.atleast_1d(axes), operations):
        for valence, category in GROUP_ORDER:
            c = curves[(curves["operation"] == op) & (curves["valence"] == valence) & (curves["category"] == category)]
            c = c.sort_values("window_index")
            x = c["window_index"] * TR_SECONDS
            color = CATEGORY_COLORS[category]
            ax.plot(x, c["mean"], color=color, linestyle=VALENCE_STYLES[valence], linewidth=2)
            ax.fill_between(x, c["mean"] - c["se"], c["mean"] + c["se"], color=color, alpha=0.18, linewidth=0)
        ax.set_title(op.capitalize(), fontsize=fs["title"], fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=fs["label"])
        ax.tick_params(labelsize=fs["tick"])
    lo, hi = min(a.get_ylim()[0] for a in np.atleast_1d(axes)), max(a.get_ylim()[1] for a in np.atleast_1d(axes))
    for ax in np.atleast_1d(axes):
        ax.set_ylim(lo, hi)
        if markers:                                         # after the y range is final: labels hug the top
            draw_event_annotations(ax, markers, tr=TR_SECONDS, show_labels=True, fontsize=fs["tick"] - 2)
    np.atleast_1d(axes)[0].set_ylabel("Operation classifier evidence\n(for the trial's own operation)", fontsize=fs["label"])
    handles = [Line2D([], [], color=CATEGORY_COLORS[c], linestyle=VALENCE_STYLES[v], linewidth=2) for v, c in GROUP_ORDER]
    labels = [f"{v}_{c}" for v, c in GROUP_ORDER]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, fancybox=False, edgecolor="black",
               fontsize=fs["legend"], handlelength=2.6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title or "Operation classifier: time course of evidence", fontsize=fs["suptitle"], fontweight="bold")
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    return fig


def plot_confusion(summary_csv: str, title: str = "", operations=OPERATIONS):
    """Accuracy and evidence matrices (rows = true operation, columns = predicted / evidence for), the mean
    across subjects of the per-subject cells in <desc>_group_summary.csv (one figure row per family, CV
    and, when the config has a test set, held-out-test)."""
    summary = pd.read_csv(summary_csv, dtype={"subject": str})
    families = list(summary["family"].unique())
    fig, axes = plt.subplots(len(families), 2, figsize=(11, 5 * len(families)), squeeze=False)
    for r, family in enumerate(families):
        rows = summary[summary["family"] == family]
        for c, kind in enumerate(("accuracy", "evidence")):
            mat = np.array([[rows[f"{kind}_{t}_{p}"].mean() for p in operations] for t in operations])
            ax = axes[r][c]
            # pcolormesh (vector cells), not imshow: an imshow raster is drawn sheared by some PDF viewers
            im = ax.pcolormesh(mat, cmap="viridis", vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
            ax.set_xlim(0, len(operations))
            ax.set_ylim(len(operations), 0)
            ax.set_aspect("equal")
            ax.set_xticks(np.arange(len(operations)) + 0.5)
            ax.set_xticklabels(operations, rotation=45, ha="right", fontsize=11)
            ax.set_yticks(np.arange(len(operations)) + 0.5)
            ax.set_yticklabels(operations, fontsize=11)
            ax.set_xlabel("Predicted" if kind == "accuracy" else "Evidence for", fontsize=12)
            ax.set_ylabel("True operation", fontsize=12)
            ax.set_title(f"{family}: {kind.capitalize()} (n = {rows['subject'].nunique()})", fontsize=13, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046)
            for i in range(len(operations)):
                for j in range(len(operations)):
                    ax.text(j + 0.5, i + 0.5, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11,
                            color="white" if mat[i, j] < 0.7 else "black")
    fig.suptitle(title or "Operation classifier: confusion matrices", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_importance_surface(impa_nii: str, title: str = "", operations=OPERATIONS, top_fraction: float = 0.2):
    """Group-mean importance map of each operation (4-D image, one volume per class in `operations` order)
    projected onto the fsaverage inflated surface: left lateral / medial, right medial / lateral. Only the
    top `top_fraction` of |weight| is coloured; one colour scale for every operation. The projection from
    MNI152NLin6Asym onto the (MNI305-based) fsaverage surface is approximate, for display only."""
    import nibabel as nib
    from nilearn import datasets, plotting, surface
    img = nib.load(impa_nii)
    data = img.get_fdata()
    nonzero = np.abs(data[data != 0])
    vmax = np.percentile(nonzero, 99.5)
    threshold = np.percentile(nonzero, 100 * (1 - top_fraction))
    fsaverage = datasets.load_fsaverage("fsaverage")
    fig, axes = plt.subplots(len(operations), 4, figsize=(15, 3 * len(operations)), subplot_kw={"projection": "3d"})
    for r, op in enumerate(operations):
        vol = nib.Nifti1Image(data[..., r], img.affine)
        for c, (hemi, view) in enumerate([("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]):
            texture = surface.vol_to_surf(vol, fsaverage.pial.parts[hemi], radius=3.0, interpolation="linear")
            plotting.plot_surf_stat_map(fsaverage.inflated.parts[hemi], texture, hemi=hemi, view=view, axes=axes[r][c],
                                        figure=fig, colorbar=False, cmap="RdBu_r", threshold=threshold, vmax=vmax,
                                        vmin=-vmax, bg_map=None, darkness=None)
            if r == 0:
                axes[r][c].set_title(f"{hemi} {view}", fontsize=13)
        axes[r][0].text2D(-0.05, 0.5, op.capitalize(), transform=axes[r][0].transAxes, rotation=90, fontsize=15,
                          fontweight="bold", va="center")
    colorbar_axes = fig.add_axes([0.35, 0.04, 0.3, 0.015])
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax)), cax=colorbar_axes,
                 orientation="horizontal", label="mean classifier weight (one-vs-rest)")
    fig.suptitle(title or f"Operation classifier: group-mean importance maps, top {top_fraction:.0%} of |weight| shown",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.09, wspace=0, hspace=0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group-csv", default=None, help="<desc>_group_decoding_results.csv written by generate_report.py (time course)")
    parser.add_argument("--summary-csv", default=None, help="<desc>_group_summary.csv (confusion matrices)")
    parser.add_argument("--impa-nii", default=None, help="<desc>_group_mean_impa_mni.nii.gz (importance maps on the surface)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--prefix", default="operation", help="file name prefix for the saved figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    def save(fig, name, dpi=150, pdf=True):
        path = os.path.join(args.output_dir, f"{args.prefix}_{name}.png")
        fig.savefig(path, dpi=dpi)
        if pdf:                 # not for the surface maps: a vector copy of 3-D meshes is ~90 MB
            fig.savefig(path.replace(".png", ".pdf"))
        plt.close(fig)
        print("saved", path)

    if args.summary_csv:
        save(plot_confusion(args.summary_csv, args.title), "confusion_matrices")
    if args.impa_nii:
        save(plot_importance_surface(args.impa_nii), "importance_maps_surface", dpi=130, pdf=False)
    if args.group_csv:
        cols = ["subject", "task", "boldfile", "trial_index", "trial_type", "regressor_label", "window_index"] + [f"evidence_{o}" for o in OPERATIONS]
        df = pd.read_csv(args.group_csv, dtype={"subject": str}, usecols=cols)
        print(f"{len(df)} rows, {df['subject'].nunique()} subject(s)")
        print("phase markers (TR):", [(m["trial_type"], m["mean_start"]) for m in phase_markers(df)])
        curves = group_timecourses(df)
        curves.to_csv(os.path.join(args.output_dir, f"{args.prefix}_timecourse_curves.csv"), index=False)
        save(plot_timecourses(curves, args.title, markers=phase_markers(df)), "timecourse_decoding")


if __name__ == "__main__":
    main()
