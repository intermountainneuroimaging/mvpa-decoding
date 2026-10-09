[← README](../README.md) · [all docs](README.md)

# Running the full pipeline end-to-end (`slurm/0_submit_mvpa_pipeline.sh`)

`slurm/0_submit_mvpa_pipeline.sh` chains every stage above into one SLURM
submission, so there's a single command that goes from HCP Pipelines output
to a group PDF report. The four stage scripts are numbered `1_`-`4_` to match
the order they run in (`0_` for the orchestrator itself, so it sorts first
in a directory listing too).

**The only required input is your config JSON.** Every stage script takes
it as a required first argument and reads all of its study-specific paths
(dataset roots, output/spreadsheet locations, the MNI template) from a
`"master"` section in that same file -- there's no separate bash file to
hand-edit for a new study/config anymore, and no risk of the four stage
scripts silently disagreeing about which study they're pointed at, since
they all read the exact same config path:

```json
"master": {
  "scripts_dir": "/path/to/mvpa-decoding",
  "bids_hcp_root": "/path/to/study/bids-hcp",
  "hcppipe_root": "/path/to/study/HCPPipe",
  "group_gm_mask": "/path/to/study/masks/group_gm_mask.nii.gz",
  "output_dir": "/path/to/study/mvpa-decoding",
  "master_spreadsheet": "/path/to/study/mvpa-decoding/master_spreadsheet.csv",
  "mni_template": "/path/to/MNI152_T1_2mm_brain.nii.gz"
}
```

> This section was called `pipeline` in earlier versions. A config that still has a
> `pipeline` section is refused with a message asking you to rename it to `master`;
> the fields are unchanged.

| Field | Required by | Meaning |
|---|---|---|
| `scripts_dir` | every stage | This repo's root on the cluster -- how every stage script locates `workflows/`, `utils/`, etc. See below for why this comes from the config rather than an exported variable or the job's own location. |
| `bids_hcp_root` | stages 1, 3 | Root of the BIDS-HCP derivatives tree -- where stage 1 writes each subject/session's resampled native-space mask ([resampling](resampling.md)'s `--derivatives-root`-adjacent layout: `sub-{subject}/ses-{session}/func/`), and where stage 3 lists subjects from. |
| `hcppipe_root` | stage 1; optional for stage 4 | Root of the raw HCP Pipelines output -- where stage 1 finds each session's first functional run (for its native reference grid) and resolves HCP warp fields ([resampling](resampling.md)). Stage 4 also uses it, but only to resample a plain `{subject}_impa.nii.gz` into `_impa_mni.nii.gz` -- skip it entirely if every classifier under `output_dir` uses `model.mnispace: true` (which writes `_impa_mni.nii.gz` directly, with no plain `_impa.nii.gz` ever produced), and stage 4 will just skip that resample step. |
| `group_gm_mask` | stage 1 only | The group-level MNI-space GM mask stage 1 resamples into each subject/session's native space. Not needed if your data is already MNI-space (`model.mnispace: true`, mask pointed straight at a shared MNI-space mask) and stage 1 never runs. |
| `output_dir` | stages 3, 4 | Where `mvpa_workflow.py`/`generate_report.py` write everything -- passed straight through as `--analysis-output-dir`. |
| `master_spreadsheet` | stages 2, 3, 4 | Where stage 2 writes `master_spreadsheet.csv` and every later stage reads it from. |
| `mni_template` | optional, stage 4 only | Reference grid for stage 4's importance-map resampling. Falls back to `$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz` (resolved after `module load fsl`) when omitted; moot entirely when `hcppipe_root` is also omitted, since that skips the resample step altogether. |

Every field is a full, already-resolved absolute path -- no `{subject}`/
`{session}` placeholders (those are per-file, not per-root) and no implicit
shared-prefix convenience the way a single hand-edited `ANALYSIS_ROOT` bash
variable used to provide; if you want that, build these paths with a
shared prefix in whatever generates your config, not in the pipeline
scripts themselves.

**Only the fields a given stage actually reads are required for it.**
`slurm/resolve_pipeline_config.sh` validates only the subset of fields the
*calling* stage script needs (each of `1_`-`4_` sets its own
`REQUIRED_PIPELINE_FIELDS` before sourcing it) -- an omitted field the
caller doesn't require just resolves to an empty string rather than
failing, so e.g. an all-MNI-space study ([model config](model-config.md)'s `model.mnispace`) that
never runs stage 1 and has nothing for stage 4 to resample can drop
`hcppipe_root`/`group_gm_mask` from its config entirely. A field a stage
*does* require still fails fast with a clear message if missing. There's
nothing left in `resolve_pipeline_config.sh` itself to hand-edit.

**Each of `1_`-`4_` is fully standalone given just its config-path
argument -- no dependency on `0_` having run first, an already-exported
`SCRIPTS_DIR`, or the job's working directory happening to be the repo
root.** That's *because* `scripts_dir` lives in the config: `sbatch` copies
a batch script into its own spool file before running it, so a stage
script's own location (`${BASH_SOURCE[0]}`) never points back at the real
repo the way it reliably does for `0_` (which is invoked directly via
`bash`, never through `sbatch`) -- and a stage script's working directory
is just wherever `sbatch` happened to be invoked from unless `--chdir` is
passed. Each of `1_`-`4_` therefore reads `master.scripts_dir` out of its
config argument *before* it can even locate `resolve_pipeline_config.sh`,
via a small bootstrap block at the top of the script (see any of `1_`-`4_`
for the exact sequence), and passes it along as a plain shell variable from
there. Run any stage on its own, from anywhere:

```
sbatch slurm/3_batch_run_mvpa_workflow.sh configs/my-study.json
```

`0_` still resolves its own location (reliable for the reason above) to
`cd` to the repo root and build the `sbatch` paths it submits, but that's
purely for its own use -- it doesn't export anything the child jobs
depend on. Run it the same way, from anywhere:

```
bash /any/path/to/slurm/0_submit_mvpa_pipeline.sh configs/my-study.json
```

`--output`/`--error` log paths are plain `#SBATCH` directives (no variable
substitution happens in them, since they're parsed before the script body
ever runs), so they always resolve relative to wherever `sbatch` was
actually invoked from -- `0_` `cd`s to the repo root before submitting so
its own jobs' logs land in the right place; submitting a stage script
directly from somewhere else just means its logs land there instead.

`0_` submits the four numbered jobs via `sbatch --parsable`, each depending
on the previous one via `--dependency=afterok` (which, for an array job,
only fires once *every* array task has succeeded):

| Stage | Script | Type |
|---|---|---|
| 1. Mask resample | `slurm/1_batch_resample_native_mask.sh` | per-subject/session array job ([resampling](resampling.md)) |
| 2. Master spreadsheet | `slurm/2_sbatch_generate_master_spreadsheet.sh` | single job ([input data](input-data.md)) |
| 3. Classifier/decoding | `slurm/3_batch_run_mvpa_workflow.sh` | per-subject array job ([running the workflow](running-the-workflow.md)) -- runs whichever of `model.kfold_cv`/`model_conditions.testing`/`model_conditions.timecourse_decoding` are configured, each independently; also writes each subject's own single-subject report |
| 4. Group report | `slurm/4_sbatch_generate_report.sh` | single job ([reports](reports.md)); resamples every subject's importance map to MNI ([resampling](resampling.md)) before building the group report |

Every stage script can still be run standalone (e.g. to rerun just one stage
after fixing a subject-specific failure) with plain
`sbatch slurm/<script>.sh configs/my-study.json` -- the orchestrator only
adds the dependency chaining on top, it doesn't own any logic itself.

`logs/` is created automatically; each stage's own `logs/<job>_%A_%a.{out,err}`
(or `_%j.{out,err}` for the two single jobs) is where to look first if a
stage fails, since `--dependency=afterok` will simply never submit the next
one -- no separate failure notification.

See **[tutorial/README.md](../tutorial/README.md)** for a full walkthrough
against this same real public data, downloaded fresh (not checked into this
repo) -- every command, the actual output and results, including a minimal
preprocessing pass (`tutorial/preprocess_haxby.sh`, built on FSL's `mcflirt`/
`flirt`/`applyxfm4D`/`fsl_glm`: motion correction, rigid coregistration to a
common run, linear detrending) written into a `derivatives/` folder and
picked up via `derivatives_root`/`bold_glob` -- and a detailed accounting of
where the tutorial still oversimplifies (rigid-only alignment with default
settings, no slice-timing correction, no normalization to a standard
template, a crude `bet` brain mask rather than an anatomical one). Its
sessionless filenames (no `ses-` entity at all) are also what surfaced a
real bug: `generate_master_spreadsheet.py` used to *require* `ses` and would
have silently rejected every file in a session-less dataset like this one --
fixed, since BIDS session labels are optional for single-session studies.
