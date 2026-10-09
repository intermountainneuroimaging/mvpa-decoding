[← README](../README.md) · [all docs](README.md)

# Resampling MNI <-> native space (`utils/hcp_resample.py`)

This pipeline's masks/classification are entirely native-space (see the
`native_*_mask.nii.gz` examples throughout), but data processed through HCP
Pipelines is commonly in MNI space and needs to move between the two --
e.g. bringing an MNI-space preprocessed BOLD run into native space to match
this pipeline's native masks, or bringing this pipeline's own native-space
importance map into MNI space for group-level comparison (save the result
as `{subject}_impa_mni.nii.gz`, right alongside the plain `impa` file under
`model/` or `test/`, to have `generate_report.py`'s group report pick it up
automatically -- see [reports](reports.md)).
`hcp_resample.py` is a standalone script for exactly that, independent of
the classification/reporting scripts (`model_conditions`/`model` don't
apply here at all -- it takes plain CLI flags, one file at a time, same as
`mvpa_workflow.py` leaves batching across subjects to your own SLURM array
wrapper).

It wraps FSL's `applywarp` -- **requires FSL on `PATH`** (`module load fsl`
on a cluster; locally, `export FSLDIR=...` and
`export PATH="$FSLDIR/bin:$PATH"`, same setup `tutorial/preprocess_haxby.sh`
already documents). This is the same external dependency the tutorial's
preprocessing script already requires; no new Python package is added.

```
python utils/hcp_resample.py --input bold_mni.nii.gz --output bold_native.nii.gz \
    --direction mni2native --subject 001 --session D1 \
    --derivatives-root /path/to/derivatives/bids-hcp \
    --reference sub-001_ses-D1_task-func_run-03_bold.nii.gz --interp trilinear
```

| Flag | Meaning |
|---|---|
| `--input`/`--output` | The file to resample and where to write the result. |
| `--direction` | `mni2native` or `native2mni` -- selects which HCP warp file to use. |
| `--subject`/`--session` | Used to resolve the warp file path. Omit `--session` for sessionless datasets (same convention as `mask_pattern` elsewhere in these docs -- just leave `{session}` out of the pattern). |
| `--derivatives-root` | Root of the `bids-hcp`-style derivatives tree containing `sub-{subject}/[ses-{session}/]MNINonLinear/xfms/`. |
| `--reference` | Target-space reference image (`applywarp --ref`). **Not auto-templated** -- pass it explicitly. This pipeline's native grid (EPI-resolution masks) and HCP's own native T1w grid differ, and guessing wrong here would silently misalign the output rather than error. |
| `--interp` | *(optional, default `trilinear`)* `nn`/`trilinear`/`sinc`/`spline` -- use `nn` for masks or other discrete-label images. |
| `--datatype` | *(optional)* Force the output data type, e.g. `int` after `--interp nn` on a mask. |
| `--xfm-pattern` | *(optional)* Override the warp-file path template (`{subject}`/`{session}` placeholders, resolved relative to `--derivatives-root`) for a non-standard layout. |
| `--warp-convention` | *(optional)* Force `applywarp --abs`/`--rel`. Leave unset by default -- real HCP-generated warp fields carry their own absolute/relative convention in the file header, which `applywarp` auto-detects; only set this if `applywarp` complains or the result looks wrong. |

## HCP warp file convention

Standard HCP Pipelines output layout, under each subject's
`MNINonLinear/xfms/`:

| Direction | File |
|---|---|
| `mni2native` | `standard2acpc_dc.nii.gz` |
| `native2mni` | `acpc_dc2standard.nii.gz` |

`--xfm-pattern` overrides this if your derivatives tree organizes these
differently.
