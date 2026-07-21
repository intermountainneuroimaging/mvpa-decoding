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
scheme.

## The scientific question

Working memory (WM) removal -- the ability to actively clear no-longer-relevant
content from mind, rather than just setting it aside -- is thought to be a
point of disruption in individuals prone to repetitive negative thinking
(e.g. rumination). Behavioral report can't tell you whether a thought has
actually been expunged from mind versus just set aside -- so the paper's
approach is to read the *representation* of the to-be-removed item directly
out of brain activity, using a classifier, and watch that representation's
strength change over time as different removal strategies are applied.

## Building on Haxby's visual object recognition paradigm

This work extends the foundational MVPA finding that distinct, distributed
patterns of brain activity -- not just activity localized to specialized
regions -- encode which visual object category a person is currently
perceiving, and that these patterns can be reliably identified straight from
imaging data using a machine-learning classifier:

> Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L., &
> Pietrini, P. (2001). Distributed and overlapping representations of faces
> and objects in ventral temporal cortex. *Science*, 293(5539), 2425-2430.

Kim et al. take that idea into a working-memory context: viewing *and*
recalling faces and places each produce their own distinct, decodable brain
pattern -- and, critically, that pattern isn't fixed. Instructing a
participant to maintain, suppress, replace, or clear an item from memory
measurably changes the strength of its representation, which is exactly the
manipulation this pipeline's classifiers are built to detect.

This repo's distinctive contribution relative to a typical MVPA design is to
decode **frame by frame**, at every TR, rather than collapsing an entire
trial's hemodynamic response (the full HRF curve) into a single classifier
input the way a more traditional MVPA approach would. That's what lets it
track *how* a representation is manipulated or degraded as time progresses
within a trial, not just whether it can be decoded at all.

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
| `decoding/*_summary_decoding_results.csv` | Trial-averaged decoding time series (Fig. 4a/4b) -- how classifier evidence for the removed item evolves over time under each operation. (`*_decoding_results.csv` is the raw, per-TR data this is averaged from.) |
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
