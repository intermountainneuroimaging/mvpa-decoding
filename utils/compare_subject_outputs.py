#!/usr/bin/env python3
"""
Check that two runs of mvpa_workflow.py gave the same results for one subject -- e.g. to validate a code
change by re-running a subject you already have results for.

Point it at the two SUBJECT folders (<analysis-output-dir>/<model.desc>/<subject>), the original run and the new one.
Every CSV under their model/, test/ and decoding/ folders is compared: same files, same rows, identical text columns,
and numeric columns equal within --tol (absolute). Columns that legitimately differ between the runs
(`model_descr`, the model name written into decoding results) are ignored. Exit status 0 = all match, 1 = something differs.

Usage:
    python utils/compare_subject_outputs.py \\
        --original  /path/to/analysis/<desc>/<subject> \\
        --new       /path/to/validation/<desc>/<subject> [--tol 1e-8]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SUBFOLDERS = ("model", "test", "decoding")
IGNORED_COLUMNS = {"model_descr"}


def csv_files(folder: str) -> dict:
    """{relative path: absolute path} of every CSV under the subject folder's model/, test/ and decoding/."""
    found = {}
    for sub in SUBFOLDERS:
        root = os.path.join(folder, sub)
        for dirpath, _, names in os.walk(root):
            for name in names:
                if name.endswith(".csv"):
                    full = os.path.join(dirpath, name)
                    found[os.path.relpath(full, folder)] = full
    return found


def compare_csv(path_a: str, path_b: str, tol: float = 1e-8) -> list:
    """Problems found between two CSVs (empty list = they match)."""
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    a, b = a.drop(columns=[c for c in IGNORED_COLUMNS if c in a]), b.drop(columns=[c for c in IGNORED_COLUMNS if c in b])
    problems = []
    if list(a.columns) != list(b.columns):
        return [f"columns differ: {sorted(set(a.columns) ^ set(b.columns)) or 'same names, different order'}"]
    if len(a) != len(b):
        return [f"{len(a)} rows vs {len(b)} rows"]
    for col in a.columns:
        x, y = a[col], b[col]
        numeric = lambda col: pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)
        if numeric(x) and numeric(y):
            both_nan = x.isna() & y.isna()
            diff = (x - y).abs().where(~both_nan, 0.0)
            worst = float(diff.max()) if len(diff) else 0.0
            if not np.isfinite(worst) or worst > tol:
                problems.append(f"{col}: max |difference| {worst:.3g} (tolerance {tol:g})")
        elif not x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)):
            problems.append(f"{col}: {int((x.fillna('<NA>').astype(str) != y.fillna('<NA>').astype(str)).sum())} value(s) differ")
    return problems


def compare_subject_dirs(original: str, new: str, tol: float = 1e-8) -> dict:
    """{relative path: list of problems} for every file; a file in only one run is a problem too."""
    a, b = csv_files(original), csv_files(new)
    report = {}
    for rel in sorted(set(a) | set(b)):
        if rel not in b:
            report[rel] = ["missing from the new run"]
        elif rel not in a:
            report[rel] = ["missing from the original run"]
        else:
            report[rel] = compare_csv(a[rel], b[rel], tol)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, help="subject folder of the run to compare against: <analysis-output-dir>/<desc>/<subject>")
    parser.add_argument("--new", required=True, help="subject folder of the new run")
    parser.add_argument("--tol", type=float, default=1e-8, help="absolute tolerance for numeric columns (default 1e-8)")
    args = parser.parse_args()
    for label, folder in (("original", args.original), ("new", args.new)):
        if not os.path.isdir(folder):
            raise SystemExit(f"--{label}: not a folder: {folder}")
    report = compare_subject_dirs(args.original, args.new, args.tol)
    if not report:
        raise SystemExit("no CSV files found under model/, test/ or decoding/ in either folder")
    bad = {k: v for k, v in report.items() if v}
    for rel, problems in report.items():
        print(f"{'OK  ' if not problems else 'DIFF'} {rel}")
        for p in problems:
            print(f"       {p}")
    print(f"\n{len(report) - len(bad)} of {len(report)} file(s) match (tolerance {args.tol:g}).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
