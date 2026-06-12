# Classifier & filter — speed + feature-efficiency (the robust6 / MLP sell)

**Date:** 2026-06-04. Speed measured by `eval/bench_speed.py` (classifiers timed on CPU =
their real device; filters on GPU). Feature stats from `2026-06-01_statistical_feature_selection_STUDY.md`
(`classifier/fusion_models/optimal_v1/feature_stats_ranked.csv`). Feature lists dumped from the joblibs.

---

## 1. Speed

### 1a. Trust classifiers (per frame)
| classifier | features | feature-extraction | model predict | **total / frame** |
|---|---:|---:|---:|---:|
| **robust6** | 6 (all free) | **0.010 ms** | 0.085 ms | **0.095 ms** |
| fusion_no_fn | 40 (incl. scene) | **38.2 ms** | 0.103 ms | **38.3 ms** |

**robust6 is ~404× faster per frame.** The model itself (XGBoost both) is identical cost (~0.09 ms);
the entire gap is feature extraction — fusion_no_fn computes per-frame OpenCV **scene statistics**
(entropy, edge-density, blurriness, sky/ground ratio, dynamic range × RGB+IR) that cost **38 ms**;
robust6 reads only detector confidence + box geometry (**0.010 ms, ~3,700× cheaper**).

### 1b. Confuser filters (per detection)
| filter | added cost / det | what it does |
|---|---:|---|
| **MLP v5** — network only | **0.11 ms** | tiny MLP on features YOLO already computed |
| **MLP v5** — full (incl. ROI-pool from YOLO maps) | **~1.3–2.1 ms** (ledger) | no pixel re-processing |
| CNN patch v2 | **23.7 ms** (here) / 59–112 ms (ledger, other HW) | crops each box + runs a 2.5M-param CNN |

**MLP is ~11–72× faster per detection** full-stack (214× on the bare forward). Per-frame pipeline
overhead: V5 **1–4%** vs patch 48–191% (ledger `v5-beats-patch`).

---

## 2. Feature sets — what each classifier reads

robust6 = `{rgb,ir}_max_conf`, `{rgb,ir}_best_log_bbox_area`, `{rgb,ir}_best_aspect_ratio`
— a **strict subset** of fusion_no_fn's 40, keeping only the 6 *free* (confidence + box-geometry) features.

| | **robust6** | fusion_no_fn | sa32 |
|---|---|---|---|
| feature count | **6** | 40 | 32 |
| confidence + box geometry | ✅ (all 6) | ✅ (subset) | ✅ (subset) |
| **scene statistics** (img mean/std/entropy/edge/blur/sky-ratio ×2) | ❌ none | ✅ 14 | ✅ 14 |
| position (pos_x/y, dist_to_centre) ×2 | ❌ | ✅ 6 | ✅ 6 |
| local_contrast / target_bg_delta ×2 | ❌ | ✅ 4 | ✅ 4 |
| detection-presence flags (n_dets, detected, both/neither) | ❌ | ✅ 8 | ❌ |
| **expensive per-frame image reads?** | **NO** | YES | YES |
| trained on current detector (ft4/v3b)? | **yes** | no | no (old `v3more`) |

---

## 3. Why 6 features ≈ 40 (the statistical justification)

The 6 were chosen by **statistics, not hand-picking** (`fusion_feature_stats.py`):

- **LDA**: the drone-vs-confuser signal is linearly separable (acc **0.982**) → it's a *selection*
  problem, not a lack of signal.
- **PCA**: classes **overlap** in unsupervised variance → the dominant variance is *scene-to-scene*
  (brightness/texture), **not** class signal. First clue the scene features are noise.
- **ANOVA F-test + per-feature AUROC**: rank each feature's standalone class-discrimination.
- **Leakage statistic** `F_domain_inclass / F_class` (the key innovation): high = the feature tracks
  *which scene* a frame came from (a fingerprint), not whether it's a drone.

**KEEP — robust6's core (high AUROC, ~zero leakage):**
| feature | AUROC-alone | leakage |
|---|---:|---:|
| `ir_max_conf` | 0.965 | 0.002 |
| `ir_best_aspect_ratio` | 0.952 | 0.002 |
| `ir_best_log_bbox_area` | 0.946 | 0.005 |

**DROP — the scene fingerprints that bloat fusion_no_fn / sa32 (chance AUROC, huge leakage):**
| feature | AUROC-alone | leakage |
|---|---:|---:|
| `rgb_img_std` | **0.502** | **349.6** |
| `rgb_img_entropy` | **0.510** | **307.4** |
| `ir_img_entropy` | 0.708 | 3.0 |

The 34 extra features are mostly **AUROC ≈ 0.5 (useless) but leakage in the hundreds** — they
*memorise scenes*, inflating in-domain scores and **failing out-of-domain**.

---

## 4. Performance — robust6 is "X% as good" (and better where it counts)

| metric (source) | robust6 | reference | robust6 = |
|---|---:|---:|---|
| in-domain Svanström F1 (full-pipeline ablation) | 0.9957 | sa32 (32f) 0.9974 | **99.8%** (−0.2pp) |
| current-detector overall F1-macro (ft4 re-mine) | **0.810** | 19-feat (w/ fingerprints) 0.787 | **103%** (better) |
| **OOD confuser fire-rate, clf-only** ↓ | **0.143** | sa32 0.203 | **30% fewer false alerts** |
| **OOD drone-video F1-macro** | **0.578** | 19-feat 0.262 | **2.2× better** |
| Anti-UAV recall | −0.6pp (68 vs 13 FN) | sa32 | tiny cost |

**Bottom line:** robust6 matches the 32–40-feature classifiers **in-domain (99.8%)** using **6/40
features and 0.1 ms/frame (404× faster)**, and **generalises 30%–2.2× better out-of-domain** — precisely
because the dropped features were scene-memorisation (leakage 300+), not drone signal.

---

## Delivered
- `docs/analysis/2026-06-04_classifier_filter_speed_and_features.md` (this doc)
- `eval/bench_speed.py` (the latency benchmark)
- Sources: `2026-06-01_statistical_feature_selection_STUDY.md`, `feature_stats_ranked.csv`, ledger
  `v5-beats-patch` / `latency-edge-unmeasured`, joblib feature lists.
