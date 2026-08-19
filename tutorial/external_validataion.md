## External validation (PyMVPA methodology)

[`pymvpa_style_validation.py`](pymvpa_style_validation.py) cross-checks the
result above against a genuinely independent implementation of **PyMVPA's
own documented methodology** for this exact dataset. PyMVPA itself
(`pymvpa2`, last released ~2020) isn't installable in a modern Python 3.12
environment -- its legacy `setup.py` requires `numpy.distutils`, which
depends on Python's stdlib `distutils`, removed in 3.12 (PEP 632); pinning
an older numpy/scipy doesn't help, since the blocker is the Python version,
not numpy's. So this reimplements PyMVPA's own canonical approach
(confirmed from their tutorial docs) using only `nibabel`/`numpy`/`pandas`/
`sklearn` -- deliberately **not** importing anything from this repo's own
`mvpa_common.py`/`generate_master_spreadsheet.py`/`mvpa_generalization_workflow.py`, so
feature extraction and classification are a genuinely separate code path:

- **Nearest-centroid correlation** -- PyMVPA's own tutorial describes this
  (kNN with correlation distance, k=1 against per-category training
  averages) as replicating "the original Haxby et al. (2001) study" --
  the same template-matching logic that paper itself used.
- **Linear SVM** -- PyMVPA's alternative canonical classifier
  (`LinearCSVMC`-equivalent).
- Both run under **two CV protocols**: the same matched train(1-9)/test(10-12)
  split this tutorial uses, and PyMVPA's own default **leave-one-run-out**
  (`NFoldPartitioner`).

It reuses the same preprocessed BOLD data and mask `mvpa_generalization_workflow.py` used
(so any numeric difference reflects classifier/CV methodology, not
different inputs), loaded directly via `nibabel`+`numpy` boolean indexing
rather than nilearn's `NiftiMasker`, and deliberately skips ANOVA feature
selection (unlike `mvpa_generalization_workflow.py`) to stay a simple, canonical baseline.

```bash
python tutorial/pymvpa_style_validation.py \
    --our-results-dir out/haxby_object_classifier/1/model  # optional, prints this repo's own numbers alongside
```

Real output against this tutorial's data:

| method | protocol | accuracy | AUC |
|---|---|---|---|
| `mvpa_generalization_workflow.py` (this repo, ANOVA + LogisticRegression) | matched split | 0.524 | 0.817 |
| nearest-centroid correlation | matched split | 0.236 | 0.634 |
| linear SVM | matched split | 0.410 | 0.773 |
| nearest-centroid correlation | leave-one-run-out | 0.266 | 0.602 |
| linear SVM | leave-one-run-out | 0.414 | 0.761 |

All four independently-computed numbers are well above chance (0.125
accuracy / 0.5 AUC), externally corroborating that this dataset genuinely
carries decodable category information after this tutorial's preprocessing
-- not an artifact of `mvpa_generalization_workflow.py`'s own code. The lower absolute
numbers here (vs. 0.524/0.817) are expected, not a discrepancy: this
validation classifies over the full ~31K-voxel mask with no feature
selection at all, while `mvpa_generalization_workflow.py` first narrows to an
ANOVA-selected subset -- the gap is a reasonable estimate of how much that
feature selection step alone is worth on this dataset.