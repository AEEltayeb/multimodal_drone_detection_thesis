# Fusion Classifier — Leakage-Aware Feature Statistics

**Date:** 2026-05-31 · **Script:** `classifier/fusion_feature_stats.py` (reuses `mri.stats` primitives) · **Data:** `classifier/fusion_models/optimal_v1/fusion_dataset_full56.csv` (65,192 rows, 56 features).

> ⚠️ **Old-detector features.** These features were mined from **selcom_1280 RGB + v3b IR** (lean19-era cache), *not* the production ft4 stack. This analysis is a **screen** to design the feature set; the chosen set must be re-validated on ft4-mined features. Scene features are detector-independent, but confidences/geometry are not.

## Purpose
Decide which features a lean trust/fusion classifier should use — *principled*, not by trial-and-error over feature counts (lean10/13/17/19). The lean line deprecated `lean13` (brightness) and `lean17` (`pos_x`) by hindsight for scene-fingerprint overfit. This computes a statistic that flags those features **up front**.

## Method
Reusing `mri.stats` (ANOVA-F, per-feature AUROC, LDA, PCA) on the tabular fusion table, plus one leakage statistic:
- **F_class** — ANOVA-F discriminating trust(≥1) vs reject(0). *Signal we want.*
- **AUROC-alone** — each feature alone as a classifier (direction-agnostic). *Practical separability.*
- **F_domain_inclass** — ANOVA-F discriminating *which dataset*, computed **within drone samples only** → isolates "varies by scene regardless of class" = fingerprint.
- **leakage_ratio = F_domain_inclass / (F_class + 1)** — low ⇒ robust; high ⇒ scene fingerprint.

## Result 1 — the space is cleanly separable
LDA linear separability (trust vs reject) = **0.982**. The trust signal is strongly, near-linearly present in the fusion features.

![Fusion LDA](images/fusion_lda_hist.png)
![Fusion PCA](images/fusion_pca_2d.png)

## Result 2 — the leakage statistic cleanly splits robust from fingerprint
The smoking gun: the highest-leakage features have **AUROC ≈ 0.50** (no class signal) but leakage in the **hundreds** — they encode *which scene*, nothing about *drone vs not*.

**Scene fingerprints — DROP (high leakage, ~chance AUROC):**
| feature | AUROC-alone | leakage_ratio |
|---|---|---|
| rgb_img_std | 0.502 | **349.6** |
| rgb_img_entropy | 0.510 | **307.4** |
| ir_blurriness | 0.795 | 11.7 |
| ir_img_entropy | 0.708 | 3.0 |
| scene_entropy_mean | 0.590 | 2.9 |
| rgb_edge_density | 0.525 | 1.75 |
| rgb/ir_img_dynamic_range | 0.55 / 0.61 | 1.25 / 0.95 |

This **empirically confirms the lean-line deprecations with a statistic** rather than hindsight: the `img_*` scalars are scene fingerprints.

![Leakage map](images/fusion_leakage_map.png)

## Result 3 — the robust, discriminative core
High AUROC **and** near-zero leakage:

![Per-feature AUROC](images/fusion_feature_auroc.png)

| feature | type | AUROC-alone | leakage |
|---|---|---|---|
| conf_sum | confidence | 0.983 | 0.002 |
| ir_mean_conf | confidence | 0.967 | 0.002 |
| ir_max_conf | confidence | 0.965 | 0.003 |
| ir_best_aspect_ratio | geometry | 0.952 | 0.002 |
| ir_best_log_bbox_area | geometry | 0.946 | 0.005 |
| xmodal_conf_ratio | confidence | 0.905 | 0.002 |
| xmodal_scale_ratio | geometry | 0.903 | 0.002 |
| xmodal_centroid_dist | cross-modal | 0.906 | 0.571 (borderline) |

## Two caveats that change the read
1. **Detection-presence flags are label-tautological, not visual.** `neither_detect`, `ir_detected`, `ir_n_dets` top the AUROC chart with ~0 leakage — but the trust label is *derived from* per-modality TPs, so these are near-tautological (the earlier 32-vs-40 ablation found `ir_detected` was a 35%-importance shortcut). leakage_ratio catches *scene* fingerprints, **not label leakage**. Exclude these from the honest feature core.
2. **The signal is IR-dominant on this corpus.** Top features are mostly IR (`ir_mean_conf`, `ir_max_conf`, IR geometry). This reflects the antiuav-heavy paired corpus; on RGB-fallback surfaces this will shift. Re-check after ft4 re-mine.

## Recommended feature set (to validate on ft4)
Robust + non-tautological + low-leakage:
`rgb_max_conf`, `ir_max_conf`, `rgb_mean_conf`, `ir_mean_conf`, `conf_sum`/`xmodal_conf_ratio`, `rgb/ir_best_log_bbox_area`, `rgb/ir_best_aspect_ratio`, `xmodal_scale_ratio`, `xmodal_centroid_dist` (~10 feats) — i.e. **meta5_geo extended**, ≈ a cleaned `lean10`. **Drop all `img_*`/entropy/blurriness/edge scene scalars.** This is the set worth re-mining from ft4 and scoring on OOD-confuser video.

## Delivered
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\docs\analysis\2026-05-31_fusion_feature_leakage_stats.md` (this doc)
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\classifier\fusion_feature_stats.py` (analysis script)
- `C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\classifier\fusion_models\optimal_v1\feature_stats_ranked.csv` (full ranked 56-feature table)
- `docs/analysis/images/fusion_lda_hist.png`, `fusion_pca_2d.png`, `fusion_feature_auroc.png`, `fusion_leakage_map.png`
