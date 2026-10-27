"""Key figures for the operation (maintain / suppress / switch / clear) classifier.

Reads what stage 4 (generate_report.py) writes and draws one figure per call:

    confusion    cross-validated accuracy and evidence matrices (true x predicted), mean across subjects
    importance   group-mean importance map of each operation on the cortical surface (fsaverage), either the top
                 fraction of |weight| or only the voxels whose weight differs from zero across subjects
                 (sign-flip permutation test of the group mean per voxel, p < 0.001 and FDR-corrected; computed
                 by stage 4 (generate_report.py) or from the per-subject maps, see utils/importance_stats.py)
    timecourse   one panel per operation; each panel overlays the operation classifier's evidence for
                 that panel's own operation on the trials of pos_face, pos_place, neg_face, neg_place
                 (mean +/- 1 SE across subjects)

Run:
    python operation_results.py --group-csv <desc>_group_decoding_results.csv --output-dir <folder> \
        [--title "Operation classifier ..."]

Or, with the organized results folder (results_layout.py): python operation_results.py --model-dir
clearvale/operation/sigmoid/all_trials/gm_liblinear_onevrest_C50_kfold   reads group_summary.csv, raw_data/ and importance_maps/
of that model and writes confusion_matrices, timecourse_decoding, timecourse_curves.csv and stats_and_figures.pdf to its
decoding/uncorrected/, and the importance figures and tables to its importance_maps/.

Evidence is the raw classifier output (no baseline correction, sigmoid scale so there is no 1/4 chance level); TR 0 of every trial is the view cue and
time is unshifted (the hemodynamic lag only selects the training/testing volumes).
"""
import argparse
import json
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
from utils.importance_stats import (load_subject_maps, save_stats, sign_flip_test, significant_voxels,  # noqa: E402
                                    split_stats)
from workflows.generate_report import draw_event_annotations                     # noqa: E402
from _interactive_notebooks import results_layout as layout                      # noqa: E402

OPERATIONS = ("maintain", "suppress", "switch", "clear")
POLARITY_CMAP = "PuOr_r"        # orange = activation, purple = deactivation: not the blue-red of the signed weight / t maps
# page order of the combined PDF: confusion matrices, time courses, then the importance maps (significant voxels,
# polarity maps, top-fraction weights)
PAGE_ORDER = ("confusion_matrices", "timecourse_decoding", "importance_maps_surface_significant",
              "importance_maps_surface_polarity", "importance_maps_surface")
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


def importance_overlap_table(mean, significant, operations=OPERATIONS, top_fraction: float = 0.2) -> pd.DataFrame:
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


def _surface_grid(data, affine, classes, threshold, vmax, title, colorbar_label, colorbar_magnitude=False,
                  cmap="RdBu_r"):
    """One row per class (4-D `data`, last axis = class): left lateral / left medial / right medial / right lateral
    views of the volume projected onto the fsaverage inflated surface over sulcal-depth shading, diverging colours (`cmap`)
    on a common (-vmax, vmax) scale, values with |value| < threshold not drawn. The projection from
    MNI152NLin6Asym onto the (MNI305-based) fsaverage surface is approximate, for display only."""
    import nibabel as nib
    from nilearn import datasets, plotting, surface
    fsaverage = datasets.load_fsaverage("fsaverage")
    sulcal = datasets.load_fsaverage_data(mesh="fsaverage", mesh_type="inflated", data_type="sulcal").data.parts
    fig, axes = plt.subplots(len(classes), 4, figsize=(15, 3 * len(classes)), subplot_kw={"projection": "3d"})
    axes = np.atleast_2d(axes)
    for r, name in enumerate(classes):
        vol = nib.Nifti1Image(data[..., r], affine)
        for c, (hemi, view) in enumerate([("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]):
            texture = surface.vol_to_surf(vol, fsaverage.pial.parts[hemi], radius=3.0, interpolation="linear")
            plotting.plot_surf_stat_map(fsaverage.inflated.parts[hemi], texture, hemi=hemi, view=view, axes=axes[r][c],
                                        figure=fig, colorbar=False, cmap=cmap, threshold=threshold, vmax=vmax,
                                        vmin=-vmax, bg_map=sulcal[hemi], darkness=0.6)
            if r == 0:
                axes[r][c].set_title(f"{hemi} {view}", fontsize=13)
        axes[r][0].text2D(-0.05, 0.5, name.capitalize(), transform=axes[r][0].transAxes, rotation=90, fontsize=15,
                          fontweight="bold", va="center")
    colorbar_axes = fig.add_axes([0.35, 0.04, 0.3, 0.015])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax)), cax=colorbar_axes,
                            orientation="horizontal", label=colorbar_label)
    if colorbar_magnitude:       # the cold colour is a second kind of evidence, not a negative importance: label magnitudes
        from matplotlib.ticker import FuncFormatter
        colorbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.88 if "\n" in title else 0.93, bottom=0.09, wspace=0, hspace=0)
    return fig


def plot_importance_surface(impa_nii: str, title: str = "", operations=OPERATIONS, top_fraction: float = 0.2,
                            significant=None, vmax=None, label: str = "Operation classifier"):
    """Group-mean importance map (classifier weights) of each class (4-D image, one volume per class in
    `operations` order) on the cortical surface. By default only the top `top_fraction` of |weight| is
    coloured. With `significant` (a boolean array shaped like the map, see significant_voxels) only the
    weights of those voxels are coloured instead. One colour scale for every class (pass the same `vmax` to
    two calls to compare them)."""
    import nibabel as nib
    img = nib.load(impa_nii)
    data = img.get_fdata()
    nonzero = np.abs(data[data != 0])
    vmax = np.percentile(nonzero, 99.5) if vmax is None else vmax
    if significant is None:
        threshold = np.percentile(nonzero, 100 * (1 - top_fraction))
    else:
        data = np.where(significant, data, 0.0)
        threshold = 1e-12 * vmax                          # anything non-zero (a significant voxel) is drawn
    default_title = (f"{label}: group-mean importance maps, top {top_fraction:.0%} of |weight| shown"
                     if significant is None else
                     f"{label}: group-mean importance maps, voxels different from zero across subjects only")
    return _surface_grid(data, img.affine, operations, threshold, vmax, title or default_title,
                         "mean classifier weight (one-vs-rest)")


def plot_polarity_surface(polarity_nii: str, title: str = "", label: str = "Operation classifier",
                          display_fraction: float = 0.2):
    """The paper-style group importance maps (see utils/importance_maps.py, written by stage 4 as
    <desc>_group_impa_polarity_maps.nii.gz + .json) on the cortical surface: orange = BOLD activation evidence
    (positive importance where the voxel's BOLD activity for the class is above average), purple = BOLD
    deactivation evidence (positive importance where it is below average). Importance of the opposite sign
    (activity arguing against the class) is not shown. The smoothed maps are drawn where they are at least
    `display_fraction` of the largest smoothed value (the smoothing's faint tails are not); one scale for
    every class."""
    import nibabel as nib
    img = nib.load(polarity_nii)
    meta = json.load(open(polarity_nii.replace(".nii.gz", ".json")))
    stack = img.get_fdata()

    def volumes(name):
        idx = meta["volumes"][name]
        return stack[..., idx[0]:idx[-1] + 1]

    data = volumes("smooth_activation") - volumes("smooth_deactivation")
    vmax = float(np.abs(data).max())
    return _surface_grid(data, img.affine, meta["classes"], display_fraction * vmax, vmax,
                         title or (f"{label}: BOLD activation evidence (orange) and BOLD deactivation evidence (purple)\n"
                                   f"top {meta['top_fraction']:.0%} of the group importance map (n = {meta['n_subjects']}), "
                                   f"{meta['fwhm_mm']:g} mm smoothing"),
                         "importance (z-scored magnitude):  orange = BOLD activation evidence,  purple = BOLD deactivation evidence",
                         colorbar_magnitude=True, cmap=POLARITY_CMAP)


def order_pages(pages):
    """[(name, kind, item), ...] in PAGE_ORDER (names not listed go last, in the order given) as the
    [(kind, item), ...] that combine_pages takes."""
    rank = lambda page: PAGE_ORDER.index(page[0]) if page[0] in PAGE_ORDER else len(PAGE_ORDER)
    return [page[1:] for page in sorted(pages, key=rank)]


def combine_pages(pages, path: str) -> int:
    """Write one multi-page PDF. `pages` is a list of ("figure", matplotlib Figure) -- kept as vector -- or
    ("image", png path), which is placed full-page (the surface figures: a vector copy of the 3-D meshes is ~90 MB).
    Returns the number of pages."""
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for kind, item in pages:
            if kind == "figure":
                pdf.savefig(item)
                continue
            image = plt.imread(item)
            height, width = image.shape[:2]
            fig = plt.figure(figsize=(width / 130.0, height / 130.0))       # the surface figures are saved at 130 dpi
            fig.add_axes([0, 0, 1, 1]).imshow(image, interpolation="lanczos")
            fig.axes[0].axis("off")
            pdf.savefig(fig)
            plt.close(fig)
    return len(pages)


LAYOUT_FILES = {                    # figure name -> (folder: decoding / importance_maps, file stem) in the organized layout
    "confusion_matrices": ("decoding", "confusion_matrices"), "timecourse_decoding": ("decoding", "timecourse_decoding"),
    "importance_maps_surface": ("importance_maps", "surface_top20"),
    "importance_maps_surface_significant": ("importance_maps", "surface_significant"),
    "importance_maps_surface_polarity": ("importance_maps", "surface_polarity")}


def model_inputs(model_path: str) -> dict:
    """The inputs of an organized model folder that exist (summary_csv, group_csv, stats_nii, polarity_nii)."""
    candidates = {"summary_csv": "group_summary.csv", "group_csv": os.path.join("raw_data", "group_decoding_results.csv"),
                  "stats_nii": os.path.join("importance_maps", "importance_signflip_stats.nii.gz"),
                  "polarity_nii": os.path.join("importance_maps", "importance_polarity_maps.nii.gz")}
    return {key: os.path.join(model_path, rel) for key, rel in candidates.items() if os.path.isfile(os.path.join(model_path, rel))}


def layout_target(model_path: str, name: str, ext: str) -> str:
    """Where a figure goes in the organized layout (decoding/uncorrected/ or importance_maps/)."""
    folder, stem = LAYOUT_FILES[name]
    base = layout.operation_figures_dir(model_path) if folder == "decoding" else os.path.join(model_path, "importance_maps")
    return os.path.join(base, stem + ext)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=None,
                        help="a model folder of the organized results (absolute, or relative to --results-root): the inputs are found in it "
                             "and the outputs written to its decoding/uncorrected/ and importance_maps/ (no --output-dir needed)")
    parser.add_argument("--results-root", default=None, help="root of the organized results (default: results_layout.default_root())")
    parser.add_argument("--group-csv", default=None, help="<desc>_group_decoding_results.csv written by generate_report.py (time course)")
    parser.add_argument("--summary-csv", default=None, help="<desc>_group_summary.csv (confusion matrices)")
    parser.add_argument("--impa-nii", default=None, help="<desc>_group_mean_impa_mni.nii.gz (importance maps on the surface)")
    parser.add_argument("--subject-impa-glob", default=None,
                        help="glob of the per-subject importance maps (e.g. '<output_dir>/<desc>/*/model/*_impa_mni.nii.gz'): "
                             "runs the sign-flip test here instead of using stage 4's (which already saves it)")
    parser.add_argument("--n-perm", type=int, default=10000, help="sign flips for --subject-impa-glob")
    parser.add_argument("--min-subjects", type=int, default=10, help="a voxel is tested only if this many subjects selected it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-voxel", type=float, default=0.001, help="voxelwise (uncorrected) threshold, also the cluster-forming threshold")
    parser.add_argument("--fdr-q", type=float, default=0.05, help="FDR level (Benjamini-Hochberg, per operation)")
    parser.add_argument("--min-cluster", type=int, default=0, help="extent threshold: drop clusters smaller than this many voxels (0 = none)")
    parser.add_argument("--stats-only", action="store_true", help="with --subject-impa-glob: save the stat maps and the table, draw no surface figure")
    parser.add_argument("--polarity-nii", default=None,
                        help="<desc>_group_impa_polarity_maps.nii.gz written by stage 4 (the .json beside it is read too): "
                             "plot the paper-style BOLD activation / deactivation evidence maps")
    parser.add_argument("--stats-nii", default=None, help="<desc>_group_impa_mni_stats.nii.gz written by stage 4 (or by --subject-impa-glob): plot the significant voxels")
    parser.add_argument("--output-dir", default=None, help="where to write (not needed with --model-dir)")
    parser.add_argument("--title", default="")
    parser.add_argument("--prefix", default="operation", help="file name prefix for the saved figures")
    parser.add_argument("--no-combined-pdf", action="store_true",
                        help="do not also write <prefix>_results.pdf (every figure of this call, one per page)")
    parser.add_argument("--classes", default=",".join(OPERATIONS),
                        help="class names of the importance maps, in volume order (default: the four operations; the "
                             "category classifier has e.g. face,place,item). With --stats-nii the names stored in the "
                             "stats .json are used")
    parser.add_argument("--label", default="Operation classifier", help="start of the importance-map figure titles")
    args = parser.parse_args()

    model_path = None
    if args.model_dir:
        model_path = layout.resolve_model_dir(args.model_dir, args.results_root)
        for key, path in model_inputs(model_path).items():
            if getattr(args, key) is None and not (key == "stats_nii" and args.subject_impa_glob):
                setattr(args, key, path)
        args.output_dir = layout.operation_figures_dir(model_path)
        args.title = args.title or os.path.basename(model_path)
        os.makedirs(os.path.join(model_path, "importance_maps"), exist_ok=True)
    elif not args.output_dir:
        parser.error("give --output-dir, or --model-dir to use the organized layout")
    os.makedirs(args.output_dir, exist_ok=True)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    pages = []                  # every figure of this call, in order, for the combined PDF

    def save(fig, name, dpi=150, pdf=True):
        path = layout_target(model_path, name, ".png") if model_path else os.path.join(args.output_dir, f"{args.prefix}_{name}.png")
        fig.savefig(path, dpi=dpi)
        if pdf:                 # not for the surface maps: a vector copy of 3-D meshes is ~90 MB
            fig.savefig(path.replace(".png", ".pdf"))
            pages.append((name, "figure", fig))         # stays open (vector page) until the combined PDF is written
        else:
            pages.append((name, "image", path))
            plt.close(fig)
        print("saved", path)

    if args.summary_csv:
        save(plot_confusion(args.summary_csv, args.title), "confusion_matrices")
    if args.subject_impa_glob:
        import glob
        paths = sorted(glob.glob(args.subject_impa_glob))
        print(f"{len(paths)} subject importance maps")
        maps, affine = load_subject_maps(paths)
        result = sign_flip_test(maps, n_perm=args.n_perm, seed=args.seed, min_subjects=args.min_subjects)
        del maps
        stats_prefix = (os.path.join(model_path, "importance_maps", "importance_signflip") if model_path
                        else os.path.join(args.output_dir, f"{args.prefix}_group_impa_mni"))
        stats_path, rows = save_stats(stats_prefix, affine, result,
                                      classes, dict(n_subjects=len(paths), n_perm=args.n_perm, seed=args.seed,
                                                       min_subjects=args.min_subjects),
                                      p_voxel=args.p_voxel, fdr_q=args.fdr_q, min_cluster=args.min_cluster)
        print(pd.DataFrame(rows).to_string(index=False))
        args.stats_nii = stats_path
    if args.impa_nii:
        save(plot_importance_surface(args.impa_nii, operations=classes, label=args.label), "importance_maps_surface", dpi=130, pdf=False)
    if args.stats_nii and not args.stats_only:
        import nibabel as nib
        mean_all, mean_selected, p, sig, n_selected = split_stats(nib.load(args.stats_nii).get_fdata())
        meta_path = args.stats_nii.replace("_stats.nii.gz", "_stats.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        p_voxel, fdr_q = meta.get("p_voxel", args.p_voxel), meta.get("fdr_q", args.fdr_q)
        classes = meta.get("classes", classes)
        table = importance_overlap_table(mean_all, sig, operations=classes)
        importance_dir = os.path.join(model_path, "importance_maps") if model_path else args.output_dir
        table.to_csv(os.path.join(importance_dir, "importance_threshold_comparison.csv" if model_path
                                  else f"{args.prefix}_importance_threshold_comparison.csv"), index=False)
        print(table.to_string(index=False))
        import nibabel as nib
        impa_path = args.impa_nii
        if impa_path is None:                                           # draw the stats file's own mean map
            impa_path = os.path.join(importance_dir, "importance_group_mean_from_stats.nii.gz" if model_path
                                     else f"{args.prefix}_group_mean_from_stats.nii.gz")
            nib.save(nib.Nifti1Image(mean_all.astype(np.float32), nib.load(args.stats_nii).affine), impa_path)
        vmax = np.percentile(np.abs(mean_all[mean_all != 0]), 99.5)
        save(plot_importance_surface(impa_path, operations=classes, significant=sig, vmax=vmax, label=args.label,
                                     title=f"{args.label}: importance maps, voxels different from zero across subjects "
                                           f"(sign-flip test, p < {p_voxel:g}, FDR q < {fdr_q:g})"),
             "importance_maps_surface_significant", dpi=130, pdf=False)
    if args.polarity_nii:
        save(plot_polarity_surface(args.polarity_nii, label=args.label), "importance_maps_surface_polarity", dpi=130, pdf=False)
    if args.group_csv:
        cols = ["subject", "task", "boldfile", "trial_index", "trial_type", "regressor_label", "window_index"] + [f"evidence_{o}" for o in OPERATIONS]
        df = pd.read_csv(args.group_csv, dtype={"subject": str}, usecols=cols)
        print(f"{len(df)} rows, {df['subject'].nunique()} subject(s)")
        print("phase markers (TR):", [(m["trial_type"], m["mean_start"]) for m in phase_markers(df)])
        curves = group_timecourses(df)
        curves.to_csv(os.path.join(args.output_dir, "timecourse_curves.csv" if model_path else f"{args.prefix}_timecourse_curves.csv"), index=False)
        save(plot_timecourses(curves, args.title, markers=phase_markers(df)), "timecourse_decoding")
    if pages and not args.no_combined_pdf:
        combined = os.path.join(args.output_dir, "stats_and_figures.pdf" if model_path else f"{args.prefix}_results.pdf")
        print(f"saved {combined} ({combine_pages(order_pages(pages), combined)} pages)")
    for _, kind, item in pages:
        if kind == "figure":
            plt.close(item)


if __name__ == "__main__":
    main()
