# Theoretical background & use case

This pipeline (`generate_master_spreadsheet.py` → `validate_model_config.py` →
`mvpa_workflow.py`) is a config-driven replication of the multivariate
pattern analysis (MVPA) approach in:

> Kim, H., Smolker, H. R., Smith, L. L., Banich, M. T., & Lewis-Peacock, J. A.
> (2020). Changes to information in working memory depend on distinct removal
> operations. *Nature Communications*, 11, 6239.
> https://doi.org/10.1038/s41467-020-20085-4

That study (and the fMRI data it analyzed, collected at the Intermountain
Neuroimaging Consortium, CU Boulder) is the direct methodological ancestor of
this codebase: same TR (460 ms), the same 4.6 s / 10-TR hemodynamic-lag shift,
the same ANOVA feature selection (voxel-wise, p < 0.05), the same L2-penalized
logistic regression classifier, and the same leave-one-run-out cross-validation
scheme. `sample-data/` (the MINDMEM dataset) extends the same paradigm with a
different operation set (e.g. `suppress`/`switch`/`maintain`/`clear`/`breath`/
`track` rather than the paper's `maintain`/`replace`/`suppress`/`clear`) and
adds a positive/negative valence manipulation not present in the original
study -- the pipeline generalizes to whichever operations/categories a given
dataset actually has, via `model_conditions`, rather than assuming this
specific set.

## The scientific question

Working memory (WM) has limited capacity, so removing no-longer-relevant
information is as important as holding onto relevant information. Behavioral
report can't tell you whether a thought has actually been expunged from mind
versus just set aside -- so the paper's approach is to read the *representation*
of the to-be-removed item directly out of brain activity, using a classifier,
and watch that representation's strength change over time as different removal
strategies are applied.

## Two classifiers, two roles

The paper (and this pipeline) trains **two conceptually different classifiers**
from the same kind of data, which is why `model_conditions` has separate
`training`/`testing` sections that can point at entirely different tasks:

1. **A representation (category) classifier**, trained on a perceptual
   *localizer* task -- participants simply view images from each category
   (e.g. face/place, or face/fruit/scene in the original paper) with no
   working-memory demand. This teaches the classifier what each category
   "looks like" in brain activity, uncontaminated by any cognitive-control
   operation. In this repo: `event_extraction.bids_root`'s `loc`-task runs,
   selected by `model_conditions.training`.

2. **An operation (or valence) classifier**, trained on the *working-memory*
   task itself -- decoding which cognitive operation (maintain, suppress,
   switch, clear, ...) was being performed on a given trial, directly from
   whole-brain activity during that operation. In this repo:
   `gm_object_classifier.json`'s and `gm_valence_classifier.json`'s
   `model_conditions.testing`/`timecourse_decoding` sections, built from
   `WM*`-task runs.

Both classifiers are the same underlying tool
(`model.featureSelection`/`model.classifier` in the config) -- what differs is
*which* task's data trains/tests them, which is exactly what `model_conditions`
exists to express as data, not code.

## Timecourse decoding: the central logic of the paper

The paper's key analytic move -- and the reason `model_conditions.timecourse_decoding`
has its own independent `window` rather than reusing whatever window built
`master_spreadsheet.csv` -- is to apply the trained classifier at **every TR**
across a window locked to trial onset (the paper used 13.8 s / 30 TRs,
unshifted from onset), producing a trial-averaged time series of classifier
evidence for the originally-encoded item's category. Comparing this evidence
trajectory across operations against a `maintain` baseline is how the paper
answers its central question: an operation that drives classifier evidence
down faster or further than simply maintaining the item is evidence that the
item's representation is being actively removed from the focus of attention,
not just passively decaying.

This repo's `mvpa_workflow.py` reproduces that logic directly:
`build_timecourse_instructions()` recomputes, per trial, exactly this kind of
onset-locked window (via `onset`/`duration`/`trial_index`, independent of the
`hemodynamic_lag` used to build the table), predicts the trained classifier at
every resulting volume, and reports evidence/accuracy per relative
`window_index` -- the same shape of result as the paper's Fig. 4a/4b decoding
time series.

## Where each output maps back to the paper

| Output | Paper analog |
|---|---|
| `cv/*_cv_results_accuracy.csv`, `*_auc.csv` | Classifier confusion matrices / AUC per operation (Fig. 2a, 3a) -- "can this be reliably decoded at all" |
| `*_impa_native.nii.gz` (importance maps) | Positive/negative classifier importance maps (Fig. 2b) -- which voxels/regions drive the classification |
| `decoding/*_decoding_results.csv` | Trial-averaged decoding time series (Fig. 4a/4b) -- how classifier evidence for the removed item evolves over time under each operation |
| `<subject>_trial_pivot.csv` | Sanity check only -- no analog in the paper |

## Why this matters for interpreting results

A classifier that decodes an operation or category *above chance* on held-out
data tells you the brain state during that operation/category is distinguishable
from others -- it does not, by itself, tell you *why* (attention vs. genuine
removal vs. something else). The paper's own resolution of this ambiguity was
to additionally test whether removal operations reduced *proactive
interference* on subsequent encoding (their Fig. 5/RSA analysis) -- a
behavioral/representational-similarity check that this pipeline does not
implement. Treat decoding accuracy and timecourse evidence as evidence about
*representational status*, not conclusive proof that information has been
permanently removed from working memory, unless paired with converging
behavioral evidence as in the original study.
