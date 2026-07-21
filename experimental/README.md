# Experimental

Scripts here are **not part of the stable pipeline** documented in the
top-level [README.md](../README.md) -- they're one-off/exploratory tools that
haven't been generalized, validated across datasets, or committed to as a
stable interface. Expect rougher edges, narrower assumptions (e.g. specific
filename conventions), and no backward-compatibility guarantees between
changes.

## `concat_wm_runs.sh`

Concatenates paired WMpos/WMneg runs from the example WM dataset (the Nth
WMpos run with the Nth WMneg run, by ascending run number) into merged 4D
files via FSL's `fslmerge`. Requires already-preprocessed (coregistered +
linearly detrended) input, and verifies that precondition before merging
rather than performing the preprocessing itself -- see the header comment in
the script for exactly what's checked (and what isn't).

```bash
export FSLDIR=/path/to/fsl
bash experimental/concat_wm_runs.sh <preproc_func_dir> <out_dir> <sub-XXXX> [ses-YYYY]
```

Writes a `<sub-XXXX>[_ses-YYYY]_concat_manifest.json` alongside the merged
files, recording exactly which pos/neg file paths went into each output --
`concat_wm_events.py` (below) consumes this directly, so both scripts always
agree on the same pairing.

## `concat_wm_events.py`

Concatenating the BOLD data doesn't touch the events.tsv files -- each
original WMpos/WMneg run's events still carry onsets relative to *that run's
own* start, which no longer match where those events land inside the merged
4D file `concat_wm_runs.sh` wrote. This builds the matching combined
events.tsv per pair: WMpos events unchanged, WMneg events with every `onset`
shifted forward by the WMpos run's actual scanned duration (`n_volumes * TR`,
read from the WMpos preprocessed NIfTI's own header -- not a hardcoded
assumption, so it's correct for any run length).

```bash
python experimental/concat_wm_events.py --manifest <out_dir>/<sub-XXXX>[_ses-YYYY]_concat_manifest.json \
    --bids-root <raw BIDS root, to find each pair's original events.tsv> \
    --out-dir <where to write the new WMcombined events.tsv files>
```

Run this after `concat_wm_runs.sh` (it reads that script's manifest -- see
[`concat_manifest.example.json`](concat_manifest.example.json) for exactly
what that file looks like, using `examples/sample-data`'s sub-4057 pairing).
The output events.tsv files are named to match `concat_wm_runs.sh`'s BOLD
output (`..._task-WMcombined_run-NN_events.tsv`), so pointing
`generate_master_spreadsheet.py`'s `bids_root`/`derivatives_root` at these
two output directories should let the stable pipeline pick up the combined
runs like any other BIDS run.
