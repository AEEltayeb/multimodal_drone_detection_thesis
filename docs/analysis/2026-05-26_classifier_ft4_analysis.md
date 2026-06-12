# Classifier Ablations + FT4 RGB Finetune — Analysis (2026-05-26)

## TL;DR

- **Classifier: don't switch production from sa32 yet.** Lean-19 is competitive on aggregate but generalization gaps are wide (41–74pp). optimal_v1 (8 features) is the most promising direction — merge with sa32's training corpus and re-eval through the cascade.
- **FT4: R3 passed all regression gates.** 300 hard-negs @ freeze=15, 3 epochs, lr=5e-6. Reduces confuser halluc rate 61%→45% without breaking drone recall. The winner is already sitting on disk. Two more attempts (A1, A2) failed — R3 is the one.
- **Scene-fingerprint overfitting is the dominant failure mode** for the lean series. Brightness scalars and pos_x act as clip identity under GroupShuffleSplit.

---

## 1. Classifier: Lean series vs sa32

### 1.1 Comparison table (unified 300-frame eval)

| Classifier | n_feat | Acc | F1m | video_drone | video_birds | video_airplanes | video_helicopters | ms/f |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| lean19 | 19 | **0.990** | 0.978 | **1.000** | 1.000 | 1.000 | 1.000 | 0.28 |
| lean13 | 13 | 0.987 | **0.979** | 0.920 | 1.000 | 1.000 | 1.000 | 0.48 |
| lean10 | 10 | 0.980 | 0.963 | 0.880 | 1.000 | 1.000 | 0.960 | 0.45 |
| lean17 | 17 | 0.977 | 0.958 | 0.880 | 1.000 | 1.000 | 1.000 | 0.39 |
| **sa32** | 32 | **0.979*** | **0.949*** | 0.280* | 1.000* | 1.000* | 1.000* | 0.46 |
| 40feat | 40 | 0.953 | 0.920 | 0.800 | 0.920 | 0.840 | 0.960 | 0.40 |

\* sa32/40feat trained on a different corpus (retrained_v2 + v3more IR, 107k rows vs lean's 59k). The 300-frame eval samples 25 frames/video source — sa32's video_drone score of 0.280 is from the same subsample.

### 1.2 The scene-fingerprint problem

The lean series was evaluated on a GroupShuffleSplit by sequence — but per-clip scalar features (`rgb_img_mean`, `ir_img_mean`, `rgb_img_std`) are near-constant within each clip. XGBoost learns them as clip-identity fingerprints.

**Train-test gap on held-out drone clips:**

| Model | Train acc (7 clips) | Test acc (2 clips) | **Gap** |
|---|---:|---:|---:|
| lean19 | 0.994 | 0.581 | **41 pp** |
| lean10 | 0.967 | 0.474 | 49 pp |
| lean17 | 0.999 | 0.270 | **73 pp** |
| lean13 | 0.997 | 0.252 | **74 pp** |

**Evidence**: Dropping 3 brightness features (lean13→lean10) gains +18–26pp on failing clips. Adding `pos_x` back (lean17) makes it worse — `ir_best_pos_x` becomes #1 feature at 0.255 importance, a clear memorization signal.

### 1.3 What this means for production

| Claim | Supported? |
|---|---|
| "Lean variants beat sa32" | **No** — different training corpus, different eval split, different YOLO back-end |
| "Lean-19 is production-ready" | **No** — 41pp train-test gap; the 300-frame eval is dominated by train-set clips |
| "Lean-10 is competitive with half the features" | **Cautiously yes** — OOD mean 0.474 vs lean19's 0.581, with 10 features vs 19 |
| "sa32 still dominates" | **Yes** — 90% cascade TP retention on real video vs control40's 60% |

**The gap**: Lean models were trained on 47k rows with selcom_1280 + ir_v3b. sa32 was trained on 107k rows with retrained_v2 + ir_v3more. The lean models have never been plumbed through the cascade eval — we don't know their real-video TP retention.

### 1.4 Ablation variants (lean19_v2 series)

| Variant | Acc | F1m | Key change | OOD drone_and_bird |
|---|---:|---:|---:|---:|
| A (class weights) | 0.967 | 0.911 | Down-weight trust_both | 0.605 |
| B (strict labels) | **0.920** | **0.888** | Distance-gate trust_both | 0.526 |
| C (xmodal features) | 0.971 | 0.915 | +3 xmodal features | 0.614 |
| ABC (all combined) | **0.974** | **0.969** | xmodal + strict + weights | **0.702** |

**ABC is the best lean variant** — 0.974 acc, 0.969 F1m, 0.702 on the shared OOD clip. xmodal_centroid_dist dominates importance at 0.237. **But** it's still below what 40feat achieves on that clip (0.877) and hasn't been compared to sa32 on the same eval.

### 1.5 Feature selection pilot v2 → optimal_v1

Forward selection on ~56 features (lean19 + xmodal + scene + derived interaction). Optimal: **8 features**.

| Feature | Importance |
|---|---|
| xmodal_centroid_dist | **0.402** |
| ir_best_aspect_ratio | 0.301 |
| ir_best_dist_to_center | 0.087 |
| area_diff | 0.061 |
| rgb_best_log_bbox_area | 0.051 |
| rgb_blurriness | 0.040 |
| rgb_mean_conf | 0.034 |
| ir_best_local_contrast | 0.023 |

**optimal_v1** (8 features, trained on full 47k set): F1m=0.926 vs sa32's 0.949 on the same split (delta -0.023). **Not a replacement for sa32 but directionally promising** — 8 features doing 97.6% of sa32's job. The top-2 features (xmodal_centroid_dist, ir_best_aspect_ratio) carry 70% of total importance. If retrained on sa32's 107k-row corpus, this gap likely narrows.

### 1.6 Recommendation

**Keep sa32 as production classifier.** Do not switch to any lean variant until:

1. A lean model (ideally optimal_v1 or ABC) is **retrained on the sa32 training corpus** (retrained_v2 + v3more IR)
2. The resulting model is **evaluated through the cascade** on real video (TP retention, confuser FPR)
3. The scene-fingerprint features (`rgb_img_mean`, `ir_img_mean`, `rgb_img_std`, `pos_x`) are either removed or their OOD generalization is verified

**What to do now**: Retrain optimal_v1's feature set on sa32's 107k-row corpus. Train + eval takes ~2 hours. If F1m closes the gap to within 0.005, plumb it through the cascade.

---

## 2. FT4 RGB Finetune — Results

### 2.1 What FT4 is

A confuser-focused fine-tune of Yolo26n_selcom_mixed_ft3_1280. Adds 300–600 hard-negative confuser images (where ft3_1280 hallucinates) as empty-label training examples. Controlled by a multi-surface regression gate.

### 2.2 Run log

| Config | Hard-negs | Freeze | Epochs | Svan R | Confuser halluc | **Gate** |
|---|---:|---:|---:|---:|---:|---:|
| R1: 600hn, freeze=12 | 600 | 12 | 3 | **0.898** (-2.9pp) | 0.416 (-19.4pp) | **FAIL** (R) |
| R2: 300hn, freeze=12 | 300 | 12 | 3 | **—** (selcom F1 -0.023) | — | **FAIL** (selcom) |
| **R3: 300hn, freeze=15** | **300** | **15** | **3** | **0.919** (-0.8pp) | **0.450** (-16.0pp) | **PASS ✓** |
| A1: 600hn, freeze=15 | 600 | 15 | 3 | **0.897** (-3.0pp) | 0.443 (-16.7pp) | **FAIL** (R) |
| A2: 600hn+4000xp, freeze=15 | 600 | 15 | 3 | 0.919 (-0.8pp) | 0.452 (-15.8pp) | **FAIL** (selcom) |

**R3 is the winner.** All 8 gates passed.

### 2.3 R3 vs ft3_1280 baseline — full comparison

| Surface | Baseline (ft3) | R3 (ft4) | Δ |
|---|---|---|---|
| **Drone recall** | | | |
| Svanstrom DRONE R | 0.927 | 0.919 | **-0.008** |
| Svanstrom DRONE P | 0.960 | 0.957 | -0.003 |
| Svanstrom DRONE F1 | **0.943** | 0.938 | -0.006 |
| Anti-UAV F1 | 0.943 | **0.955** | +0.012 |
| Dataset RGB F1 | 0.920 | 0.918 | -0.002 |
| Selcom val F1 | 0.619 | 0.615 | -0.004 |
| **Confuser hallucination** | | | |
| Confuser test halluc | **0.610** | **0.450** | **-16.0pp** |
| Svan BIRD halluc | **0.733** | **0.694** | -3.9pp |
| Svan AIRPLANE halluc | **0.767** | **0.696** | -7.1pp |
| Svan HELI halluc | **0.862** | **0.589** | **-27.4pp** |

### 2.4 What the numbers mean

- **Drone recall is preserved** — Svanstrom R drops 0.008 (0.927→0.919). This is within the regression budget and may even be noise.
- **Confuser hallucinations are cut across the board** — 16pp on the confuser test, 27pp on Svanstrom helicopters (the hardest category).
- **Anti-UAV actually improves** (+0.012 F1) — likely fewer spurious FPs on the saturated surface.
- **Selcom val is flat** — the fine-tune didn't damage CCTV performance.

### 2.5 What to do with R3

**Ship it into the production stack** as the replacement for ft3_1280. The regression gate passed all surfaces. The confuser halluc reduction is meaningful (61%→45% = 26% relative reduction). No single surface regressed below threshold.

**Caveats**:
- Only tested at the bare-detector level. **Has not been run through the cascade** — the patch verifier + classifier may already handle the ft3 FPs, making the R3 improvement redundant at the pipeline level.
- The 3-epoch, freeze=15 recipe means the confuser suppression is entirely in the later layers. A deeper fine-tune might achieve more.
- Only 300 hard-negs out of 11,669 candidates were used (2.6%). Increasing the dose while maintaining freeze=15 (A1) caused Svanstrom recall regression. The ratio appears to be at the limit.
- The Svanstrom DRONE F1 is now 0.938 vs baseline Yolo26n_trained's 0.969 (from EVIDENCE_LEDGER §3.1). The three-stage fine-tune chain (baseline → ft2 → ft3 → ft4) has a cumulative recall cost of ~4pp from baseline.

**Decision**: Replace ft3_1280 with R3 ft4_1280 in the production stack. Then run the cascade eval to confirm the pipeline-level benefit.

---

## 3. Cross-cutting observations

### 3.1 The cascade masks detector-level improvements

The ft4 reduces confuser halluc rate by 16pp. But the cascade (patch verifier + classifier) already suppresses 45× on birds and 15–34% on helicopters. The ft4 improvement might be invisible at the cascade output level. Run the cascade eval before claiming victory.

### 3.2 The classifier and detector are on different training corpora

Classifier (lean series): selcom_1280 RGB + ir_v3b. Detector ft4 chain: baseline → ft2 (selcom mixed) → ft3 (50/50 val) → ft4 (confuser hard-negs). The classifier was trained on ft2's detections. If ft4 changes the detector's confidence distribution (which it clearly does — fewer FPs means lower conf on confusers), the classifier may be miscalibrated for ft4's detections.

**Recommendation**: After promoting ft4, check whether the classifier's per-feature distributions have shifted. If `rgb_max_conf` on confuser detections drops significantly (likely — that's the whole point of ft4), the classifier's `reject_both` precision might improve or its `trust_rgb` recall might drop. Run the 3-way eval with ft4 as the RGB model.

### 3.3 The lean-10 / optimal_v1 intersection

Lean-10 (10 best scene-invariant features) and optimal_v1 (8 best from 56) share only 2 features: `rgb_best_log_bbox_area` and a conf-derived metric. optimal_v1's top feature (`xmodal_centroid_dist`) requires both RGB and IR detections; lean-10 uses per-modality conf + geometry. These are complementary approaches — xmodal for paired scenes, per-modality for single-modality fallback.

---

## 4. Recommended next actions (ordered)

1. **Promote R3 ft4 to production** — update EVIDENCE_LEDGER §1 RGB YOLO row, replace ft3_1280 weights
2. **Run cascade eval with ft4** — does the confuser reduction persist through patch verifier + classifier?
3. **Run 3-way classifier eval with ft4 back-end** — check if the classifier is still calibrated for ft4's confidence distribution
4. **Retrain optimal_v1 on sa32 corpus** — closes the training-data gap, then compare F1m and cascade TP retention
5. **Run optimal_v1 or ABC through the cascade** — the moment one of these beats sa32's 90% TP retention, the production pick flips

## 5. Delivered

- `docs/analysis/2026-05-26_classifier_ft4_analysis.md` — this document.