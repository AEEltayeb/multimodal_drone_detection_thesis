"""Model MRI — a model-agnostic methodology for imaging a detector's internal
feature space.

Attach any ultralytics detector and two roles of image folders — a *positive*
(target-class) set and one or more *negative* (distractor) sets — and MRI runs
the detector, extracts the FPN features behind each detection, and produces:
PCA/LDA/ANOVA "brain statistics", figures, an optional FP-reduction MLP, and a
verdict on whether the detector needs that classifier at all. Feature dimensions
are read from the model at runtime, so nothing is tied to a specific
architecture, class set, or domain — drone detection is the worked case study,
not a requirement.

Module map
----------
Core pipeline:
  cli         entry point / orchestration (``python -m mri ...``)
  datasets    DatasetSpec, label auto-resolve, inline-spec + YAML parsing
  extract     model-agnostic Detect-head hook, ROI pool, FeatureSchema
  scan        detector sweep + positive/negative mining + bare-detector tallies
  stats       PCA, LDA separability, ANOVA F, per-feature AUROC, silhouette
  plots       all static figures (pca / lda / anova / heatmap / neuron-kde / fp)
  classifier  FocalLoss MLP + LogReg/RF/XGB + CV; saves a callable .pt artifact
  diagnose    the verdict (not_needed / recommended / wont_help / marginal)
  report      assembles report.md + the Delivered manifest
  examples    per-dataset spatial activation panels (crop + P3/P5 heatmaps)

Honest deployment gate (CV is optimistic):
  holdout     held-out per-surface bare-vs-verifier P/R/F1 of a *shipped* .pt

Cross-modal extension (case study: thermal vs grayscale-RGB):
  modality_probe  is a class's feature signature the same across two input modes?
  modality_align  rescue cross-modality transfer via per-feature affine alignment
  train_aligned   synthesize a verifier from cheap-modality negatives, aligned
  plot_alignment  the cross-modal transfer figure

Utilities:
  _stage_v5_corpus  stage a legacy feature corpus into an MRI run dir for --resume
"""

__all__ = ["cli", "extract", "datasets", "scan", "stats", "plots",
           "classifier", "diagnose", "report", "examples", "holdout",
           "modality_probe", "modality_align", "train_aligned", "plot_alignment"]
