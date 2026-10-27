#!/usr/bin/env python3
"""
The folder layout of the organized results, in one place: where a model's files go, where an analysis
(baseline correction x grouping x statistics window) writes its outputs, how its files are named, and how a
downloaded stage-4 results folder is parsed into the layout (ingest).

    <root>/                                   default: _interactive_notebooks/results_organized (or $MVPA_RESULTS_ROOT)
      <project: clearvale | mindmem>/
        <classifier: category | operation>/
          <evidence: norm_proba | sigmoid>/
            <sorting: all_trials | restricted>/
              <roi>_<classifier settings>/    e.g. vvps_liblinear_onevrest_C50, gm_liblinear_onevrest_C50_kfold
                group_summary.csv  group_cv_results.csv  report.pdf  model_config.json  README_model.md
                raw_data/group_decoding_results.csv
                importance_maps/importance_*.nii.gz ...
                decoding/
                  uncorrected/<stats window folder>/
                  corrected/<baseline-correction-...>/<grouping-...>/<stats window folder>/
                    stats_and_figures_<N>bins.pdf|.ipynb   pos_vs_neg_<N>bins.csv   ...

`<N>bins` (or `<N>windows` for sliding windows) is the size of the FDR family of a comparison: 7bins = the 3-TR
bins from TR 7 on, 9bins = every bin.

Command line (parse a downloaded results folder into the layout):
    python3 _interactive_notebooks/results_layout.py ingest <download_dir> [--config model_config.json] [--run]
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass

ROOT_ENV = "MVPA_RESULTS_ROOT"
PROJECTS = ("clearvale", "mindmem")
CLASSIFIERS = ("category", "operation")
EVIDENCE = ("norm_proba", "sigmoid")
SORTINGS = ("all_trials", "restricted")


def default_root() -> str:
    return os.environ.get(ROOT_ENV) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_organized")


# =====================================================
# Which model is it
# =====================================================

@dataclass(frozen=True)
class ModelInfo:
    project: str
    classifier: str
    evidence: str
    sorting: str
    model: str                      # <roi>_<classifier settings>

    def path(self, root: str = None) -> str:
        return os.path.join(root or default_root(), self.project, self.classifier, self.evidence, self.sorting, self.model)


def roi_label(mask_pattern: str) -> str:
    """vvps / gm from the mask file name (anything else: its first word)."""
    base = os.path.basename(str(mask_pattern)).lower()
    if "vvps" in base: return "vvps"
    if re.search(r"(^|[_\-.])gm([_\-.]|$)", base) or "_gm_" in base: return "gm"
    return re.sub(r"\.nii(\.gz)?$", "", base).split("_")[0] or "roi"


def classifier_label(cfg: dict) -> str:
    """e.g. liblinear_onevrest_C50, multinomial_lbfgs_C0.5; '_kfold' appended for a model that is only evaluated by k-fold
    cross-validation (no held-out 'testing' section, as for the operation classifier)."""
    model = cfg.get("model", {})
    clf = model.get("classifier", {})
    params = clf.get("params", {})
    name = str(clf.get("name", "classifier")).split(".")[-1].lower()
    solver = params.get("solver", "")
    c = params.get("C")
    c_text = f"C{c:g}" if isinstance(c, (int, float)) else ""
    if name == "logisticregression":
        parts = [solver, "onevrest"] if clf.get("one_vs_rest") else ["multinomial", solver]
    else:
        parts = [name, solver]
    label = "_".join(p for p in parts + [c_text] if p)
    if "testing" not in cfg.get("model_conditions", {}):
        label += "_kfold"
    return label


def model_info_from_config(cfg: dict, project: str = None, classifier: str = None, evidence: str = None,
                           sorting: str = None, roi: str = None, model_type: str = None) -> ModelInfo:
    """Everything the layout needs, read from the model's config (model section, model_conditions); any argument overrides
    what the config says (or fills what it cannot say)."""
    text = json.dumps(cfg).lower()
    desc = str(cfg.get("model", {}).get("desc", "")).lower()
    if project is None:
        found = [p for p in PROJECTS if f"/{p}/" in text]
        if len(found) != 1: raise ValueError(f"cannot tell the project from the config (found {found}): pass project=...")
        project = found[0]
    if classifier is None:
        found = [c for c in CLASSIFIERS if c in desc]
        if len(found) != 1: raise ValueError(f"cannot tell the classifier type from model.desc {desc!r}: pass classifier=...")
        classifier = found[0]
    if evidence is None:
        evidence = "sigmoid" if cfg.get("model", {}).get("classifier", {}).get("evidence") == "sigmoid" else "norm_proba"
    if sorting is None:
        sorting = "restricted" if cfg.get("model_conditions", {}).get("training", {}).get("trial_restriction") else "all_trials"
    if roi is None:
        roi = roi_label(cfg.get("model", {}).get("mask", {}).get("mask_pattern", ""))
    if model_type is None:
        model_type = classifier_label(cfg)
    return ModelInfo(project, classifier, evidence, sorting, f"{roi}_{model_type}")


def resolve_model_dir(model_dir: str, root: str = None) -> str:
    """A model folder given in full or relative to the results root (clearvale/category/sigmoid/all_trials/vvps_...)."""
    path = model_dir if os.path.isabs(model_dir) else os.path.join(root or default_root(), model_dir)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"model folder not found: {path}")
    return path


def resolve_path(path: str, root: str = None) -> str:
    """An absolute path as it is, a relative one taken relative to the results root."""
    return path if os.path.isabs(path) else os.path.join(root or default_root(), path)


def model_dir_from_raw_data(raw_data_path: str, root: str = None) -> str:
    """The model folder from the path of its input table (<model>/raw_data/group_decoding_results.csv) or of its raw_data folder."""
    path = resolve_path(raw_data_path, root)
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    if os.path.basename(os.path.normpath(folder)) != "raw_data":
        raise ValueError(f"{raw_data_path!r} is not inside a model's raw_data folder (<model>/raw_data/group_decoding_results.csv)")
    return os.path.dirname(os.path.normpath(folder))


# =====================================================
# Where an analysis writes its outputs, and what its files are called
# =====================================================

BASELINE_LABELS = {(1, 3): "viewonset", (4, 6): "viewoffset", (7, 9): "operationonset", (11, 13): "viewlag"}
GROUP_WORDS = {"object": "category", "category": "category", "operation": "operation", "valence": "valence"}
GROUP_ORDER = ("operation", "category", "valence")


def baseline_folder(trs) -> str:
    trs = sorted(int(t) for t in trs)
    first, last = trs[0], trs[-1]
    label = BASELINE_LABELS.get((first, last)) if trs == list(range(first, last + 1)) else None
    return f"baseline-correction{'-' + label if label else ''}_TR{first}-{last}"


def grouping_folder(group_by, balance: str = None) -> str:
    words = [w for w in GROUP_ORDER if w in {GROUP_WORDS[g] for g in group_by}]
    return "grouping-per-" + "-".join(words) + (f"_balanced-preceding-{balance}" if balance else "")


def window_folder(sliding: bool = False, bin_size: int = 3) -> str:
    return f"stats_sliding-{bin_size}TR_fdr" if sliding else f"stats_{bin_size}TR-bins_fdr"


def decoding_dir(model_dir: str, baseline_correction: bool = False, baseline_trs=(4, 5, 6),
                 group_by=("object", "operation", "valence"), balance: str = None, sliding: bool = False,
                 bin_size: int = 3) -> str:
    """The folder one run of the category / operation evidence notebooks writes to."""
    window = window_folder(sliding, bin_size)
    if not baseline_correction:
        base = os.path.join(model_dir, "decoding", "uncorrected")
        if balance: base = os.path.join(base, f"balanced-preceding-{balance}")
        return os.path.join(base, window)
    return os.path.join(model_dir, "decoding", "corrected", baseline_folder(baseline_trs), grouping_folder(group_by, balance), window)


def operation_figures_dir(model_dir: str) -> str:
    """Where operation_evidence_by_category.ipynb saves the confusion matrices and raw time courses (independent of the statistics settings)."""
    return os.path.join(model_dir, "decoding", "uncorrected")


def family_size(table, group_cols=("operation", "object", "category", "valence", "group")):
    """Size of the FDR family of a comparison: windows that have a p_fdr, per comparison (the largest, when they differ)."""
    if table is None or len(table) == 0 or "p_fdr" not in table.columns: return None
    keys = [c for c in group_cols if c in table.columns]
    counts = table.groupby(keys)["p_fdr"].apply(lambda s: int(s.notna().sum())) if keys else None
    n = int(counts.max()) if counts is not None else int(table["p_fdr"].notna().sum())
    return n or None


def stat_file(stem: str, n_family, sliding: bool = False, ext: str = ".csv") -> str:
    """pos_vs_neg + 7 -> pos_vs_neg_7bins.csv (sliding: _18windows); no family size -> pos_vs_neg.csv"""
    if n_family is None: return f"{stem}{ext}"
    return f"{stem}_{int(n_family)}{'windows' if sliding else 'bins'}{ext}"


def write_run_settings(output_dir: str, settings: dict, name: str = "run_settings.json") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    with open(path, "w") as f:
        json.dump(dict(settings, written=datetime.datetime.now().isoformat(timespec="seconds")), f, indent=1, default=str)
    return path


def copy_notebook(notebook_file: str, output_dir: str, name: str) -> str:
    """A copy of the (saved) notebook next to its results; the notebook has to be saved first for the copy to hold its outputs."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    shutil.copy2(notebook_file, path)
    return path


# =====================================================
# Ingest: a downloaded stage-4 results folder -> the layout
# =====================================================

DOWNLOAD_RULES = [        # (file name regex with {desc}, destination relative to the model folder)
    (r"^{desc}_group_summary\.csv$", "group_summary.csv"),
    (r"^{desc}_group_cv_results\.csv$", "group_cv_results.csv"),
    (r"^{desc}_group_decoding_results\.csv$", "raw_data/group_decoding_results.csv"),
    (r"^{desc}_group_mean_impa_mni\.nii\.gz$", "importance_maps/importance_group_mean.nii.gz"),
    (r"^{desc}_group_impa_mni_stats\.(nii\.gz|json)$", r"importance_maps/importance_signflip_stats.\1"),
    (r"^{desc}_group_impa_polarity_maps\.(nii\.gz|json)$", r"importance_maps/importance_polarity_maps.\1"),
    (r"^{desc}_group_impa_polarity_summary\.csv$", "importance_maps/importance_polarity_summary.csv"),
    (r"^{desc}_group_impa_signflip_summary\.csv$", "importance_maps/importance_signflip_summary.csv"),
    (r"^report_{desc}\.pdf$", "report.pdf"),
]


def find_config(download_dir: str):
    """A model config inside the download: a .json with 'model' and 'model_conditions' sections."""
    for f in sorted(os.listdir(download_dir)):
        if f.endswith(".json"):
            try:
                cfg = json.load(open(os.path.join(download_dir, f)))
            except Exception:
                continue
            if isinstance(cfg, dict) and "model" in cfg and "model_conditions" in cfg:
                return os.path.join(download_dir, f)
    return None


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def plan_ingest(download_dir: str, config_path: str = None, root: str = None, **overrides):
    """-> (ModelInfo, desc, [(source, destination, action)], [unmatched file names]); action is copy / same / differs."""
    config_path = config_path or find_config(download_dir)
    if config_path is None:
        raise ValueError("no model config found in the download: pass --config <the model's config .json>")
    cfg = json.load(open(config_path))
    info = model_info_from_config(cfg, **overrides)
    desc = cfg["model"]["desc"]
    model_dir = info.path(root)
    plan, unmatched = [(config_path, os.path.join(model_dir, "model_config.json"), None)], []
    for name in sorted(os.listdir(download_dir)):
        src = os.path.join(download_dir, name)
        if not os.path.isfile(src) or src == config_path or name == ".DS_Store": continue
        for pattern, dest in DOWNLOAD_RULES:
            m = re.match(pattern.format(desc=re.escape(desc)), name)
            if m:
                plan.append((src, os.path.join(model_dir, m.expand(dest)), None)); break
        else:
            unmatched.append(name)
    out = []
    for src, dst, _ in plan:
        if not os.path.exists(dst): action = "copy"
        elif os.path.getsize(dst) == os.path.getsize(src) and _md5(dst) == _md5(src): action = "same"
        else: action = "differs"
        out.append((src, dst, action))
    return info, desc, out, unmatched


def model_readme(info: ModelInfo, cfg: dict) -> str:
    clf = cfg.get("model", {}).get("classifier", {})
    return "\n".join([
        f"# {info.project} / {info.classifier} / {info.evidence} / {info.sorting} / {info.model}", "",
        f"- model.desc: {cfg['model'].get('desc')}",
        f"- classifier: {clf.get('name')} {json.dumps(clf.get('params', {}))}, one_vs_rest = {clf.get('one_vs_rest', False)}, evidence = {clf.get('evidence', 'normalised probability')}",
        f"- ROI mask: {cfg['model'].get('mask', {}).get('mask_pattern')}",
        f"- feature selection: {json.dumps(cfg['model'].get('featureSelection', {}))}",
        f"- trial restriction: {cfg.get('model_conditions', {}).get('training', {}).get('trial_restriction', 'none (all trials)')}", "",
        "Written by results_layout.py ingest from the model's config; add a description of the model here.", ""])


def ingest(download_dir: str, config_path: str = None, root: str = None, run: bool = False, move: bool = False,
           overwrite: bool = False, **overrides):
    """Parse a downloaded stage-4 folder into the layout. Without run=True nothing is changed. Existing identical files are
    left alone; a different file with the same destination is skipped unless overwrite=True. Returns the plan."""
    root = root or default_root()
    info, desc, plan, unmatched = plan_ingest(download_dir, config_path, root, **overrides)
    if run:
        model_dir = info.path(root)
        manifest_rows = []
        for src, dst, action in plan:
            if action == "same" or (action == "differs" and not overwrite): continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            (shutil.move if move else shutil.copy2)(src, dst)
            manifest_rows.append([os.path.relpath(dst, root), src, os.path.getsize(dst), f"ingested {datetime.date.today().isoformat()}"])
        readme = os.path.join(model_dir, "README_model.md")
        if not os.path.exists(readme):
            cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
            open(readme, "w").write(model_readme(info, cfg))
        manifest = os.path.join(root, "MANIFEST.csv")
        if manifest_rows:
            new = not os.path.exists(manifest)
            with open(manifest, "a", newline="") as f:
                w = csv.writer(f)
                if new: w.writerow(["destination (relative to this folder)", "source", "size_bytes", "note"])
                w.writerows(manifest_rows)
    return info, desc, plan, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="parse a downloaded stage-4 results folder into the layout")
    p.add_argument("download_dir")
    p.add_argument("--config", default=None, help="the model's config .json (default: one found in the download)")
    p.add_argument("--root", default=None, help=f"results root (default: ${ROOT_ENV} or _interactive_notebooks/results_organized)")
    p.add_argument("--run", action="store_true", help="do it (default: only show what would happen)")
    p.add_argument("--move", action="store_true", help="move instead of copy")
    p.add_argument("--overwrite", action="store_true", help="replace a different file at the same destination")
    for flag, choices in (("project", PROJECTS), ("classifier", CLASSIFIERS), ("evidence", EVIDENCE), ("sorting", SORTINGS)):
        p.add_argument(f"--{flag}", choices=choices, default=None, help=f"override what the config says")
    p.add_argument("--roi", default=None, help="override the ROI label (vvps, gm, ...)")
    p.add_argument("--model-type", default=None, help="override the classifier-settings part of the model folder name")
    args = parser.parse_args()
    overrides = {k: getattr(args, k) for k in ("project", "classifier", "evidence", "sorting", "roi", "model_type")}
    info, desc, plan, unmatched = ingest(args.download_dir, args.config, args.root, args.run, args.move, args.overwrite, **overrides)
    root = args.root or default_root()
    print(f"model: {info.project} / {info.classifier} / {info.evidence} / {info.sorting} / {info.model}   (model.desc {desc})")
    for src, dst, action in plan:
        print(f"  {action:7s} {os.path.basename(src)}  ->  {os.path.relpath(dst, root)}")
    if unmatched: print("not recognised (left where they are):", ", ".join(unmatched))
    if any(a == "differs" for _, _, a in plan) and not args.overwrite: print("'differs' files were not touched; pass --overwrite to replace them")
    print("done." if args.run else "dry run: nothing changed, add --run")


if __name__ == "__main__":
    main()
