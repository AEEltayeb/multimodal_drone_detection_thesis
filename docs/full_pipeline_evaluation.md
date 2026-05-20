# Full Pipeline Evaluation — Ablation Study

> **Date:** 2026-05-20
> **Scoring Rule:** IoP @ 0.5 (Intersection over Prediction) for all surfaces except Anti-UAV (IoU @ 0.5)
> **Trust-Aware Scoring:** The classifier is evaluated only against the ground-truth of the modality it chose to trust on that specific frame. label=0 → no dets (FN on both); label=1 → RGB dets vs RGB GT; label=2 → IR dets vs IR GT; label=3 → RGB vs RGB GT + IR vs IR GT.
> **Classifier:** `scene_aware_v3more_32feat` (sa32) — 32 features, 4-class trust (reject_both / trust_rgb / trust_ir / trust_both)
> **Patch Filter:** `confuser_filter4_rgb_v2_backup.pt` (RGB); `confuser_filter4_ir_v2_backup.pt` (IR); RGB filter on grayscale-IR
> **Temporal:** 2-of-3 segment voting over 3-frame windows. Patch filter fires only at alert gate (not per-frame).
> **Max per dataset:** ≤1000 frames (auto-stride applied)
> **Confidence thresholds:** RGB = 0.25, IR = 0.40, Patch veto = 0.70

---

## Models Under Test

| Alias | Weights | imgsz | Purpose |
|---|---|---|---|
| **baseline** | `Yolo26n_trained/best.pt` | 1280 | Production RGB detector (general datasets) |
| **selcom_960** | `Yolo26n_selcom_mixed_ft2_1280/best.pt` | 960 | CCTV fine-tuned, evaluated at 960px |
| **retrained_v2** | `Yolo26n_retrained_v2/best.pt` | 1280 | Aggressively hard-neg trained variant |
| **ir_v3b** | `finetune_v3b/best.pt` | 640 | Production IR detector (native thermal) |
| **ir_grayscale** | `finetune_v3b/best.pt` on grayscale-RGB | 640 | Cross-modal fallback (IR weights on grayscale-converted RGB) |

---

## Part 1: Standalone Detector Baselines

These numbers establish single-sensor performance before any multi-stage cascade is applied.

### 1.1 RGB Models on General RGB Dataset (`G:/drone/dataset`, test split)

Scored at IoP @ 0.5, conf = 0.25. Per-model native imgsz.

Source: `eval/results/_ablation/` and `runs/rgb_finetune_eval/` comparison JSONs.

| Model | imgsz | TP | FP | FN | P | R | F1 | FPPI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 640 | — | — | — | 0.997 | 0.923 | **0.959** | — |
| **selcom_960** | 960 | — | — | — | 0.995 | 0.923 | **0.958** | — |
| **retrained_v2** | 640 | — | — | — | — | — | — | — |

> [!NOTE]
> **General-domain safety:** Baseline and selcom_960 are functionally identical on the general RGB dataset (F1 delta = 0.001). CCTV fine-tuning does not compromise general detection capability. retrained_v2 on this split requires a dedicated run (Anti-UAV results show it ties baseline at F1 = 0.993, but the general RGB split may differ on small/medium targets).

### 1.2 IR Model on Native Thermal IR Dataset (`G:/drone/IR_dset_final`, test split)

Scored at IoU @ 0.5, imgsz = 640, conf = 0.40. 9612 images in split, stride applied.

Source: `eval/results/ir_version_comparison/ir_comparison_test_640.json`.

| Model | Stage | TP | FP | FN | P | R | F1 | FPPI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **ir_v3b** | Raw IR Detector | — | — | — | 0.957 | 0.977 | **0.967** | — |
| **ir_v3b** | + IR Patch Filter | — | — | — | 0.973 | 0.933 | **0.953** | — |

> [!NOTE]
> The IR patch filter trades 4.4 pp recall (0.977 → 0.933) for 1.6 pp precision gain (0.957 → 0.973) to suppress transient hot-spot hallucinations.

---

## Part 2: Selcom CCTV Dataset — Full Pipeline Ablation

**Dataset:** `selcom_mixed_ft2_val` (311 images, 295 GT boxes, CCTV surveillance footage)
**RGB Model:** selcom_960 (selcom_1280 weights at imgsz = 960)
**Scoring:** IoP @ 0.5, trust-aware for classifier stages

Source: `docs/analysis/full_pipeline_ablations/raw_results/selcom_val/selcom_960/sa32/summary.csv`

### 2.1 Per-Frame Detection Ablation (bbox-level)

| Stage | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0: RGB Detector** | 129 | 17 | 166 | 0.884 | 0.437 | **0.585** | 0.055 |
| **S1: + Classifier** (sa32, trust-aware) | 68 | 9 | 227 | 0.883 | 0.231 | 0.366 | 0.029 |
| **S2: + Classifier + Patch Filter** | 68 | 9 | 227 | 0.883 | 0.231 | 0.366 | 0.029 |
| **S3: + Patch Filter only** (no classifier) | 129 | 17 | 166 | 0.884 | 0.437 | 0.585 | 0.055 |

> [!WARNING]
> **Classifier recall drop on CCTV:** The sa32 classifier was trained on Svanström-like paired RGB+IR data and does not recognise CCTV signal statistics. It conservatively rejects 47% of drone-positive frames (recall 0.437 → 0.231). The cascade is not the right tool for this surface; **rgb_only is the correct reporting baseline** for Selcom.

### 2.2 Per-Size Breakdown (S0: RGB Detector)

| Size Bucket | n_gt | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** (< 0.1% img area) | 166 | 50 | 6 | 116 | 0.893 | 0.301 | **0.451** | 0.019 |
| **medium** (0.1–1%) | 125 | 76 | 11 | 49 | 0.874 | 0.608 | **0.717** | 0.035 |
| **large** (> 1%) | 4 | 3 | 0 | 1 | 1.000 | 0.750 | **0.857** | 0.000 |
| **ALL** | 295 | 129 | 17 | 166 | 0.884 | 0.437 | **0.585** | 0.055 |

> [!IMPORTANT]
> **Size-dependent recall:** Selcom drones are overwhelmingly small (56% of GT in small bucket). Small-drone recall is only 30.1%, confirming that even at imgsz=960, the CCTV targets (median ~24px) sit at the spatial resolution floor. Medium-drone recall is 60.8%, validating the resolution–domain co-dependency finding.

### 2.3 Temporal Segment-Level Metrics (2-of-3 voting, 3-frame windows)

Patch filter applied only at alert gate (S5). P/R/F1 calculated on segments.

| Stage | TP seg | FP seg | FN seg | TN seg | Total seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S4: Temporal 2/3** (no filter) | 46 | 0 | 53 | 5 | 104 | 1.000 | 0.465 | **0.634** | 0.000 |
| **S5: Temporal + Alert Gate Filter** | 46 | 0 | 53 | 5 | 104 | 1.000 | 0.465 | **0.634** | 0.000 |

> [!TIP]
> **Perfect precision at segment level.** The temporal smoother achieves P = 1.000 with zero false-positive segments on Selcom — every alert is a real drone. Recall at segment level (0.465) improves over the classifier per-frame (0.231) because the temporal voter corrects isolated frame drops. The alert-gate filter is inert here (no confuser hallucinations to suppress on CCTV footage).

---

## Part 3: Drone Detection Video Test Dataset — Full Pipeline Ablation

**Dataset:** 19 real-video clips (9 drone-positive, 10 confuser-only: 5 birds, 2 airplanes, 3 helicopters)
**RGB Model:** baseline (`Yolo26n_trained`) at imgsz = 1280
**Scoring:** IoP @ 0.5 for drone clips; all detections = FP on confuser clips

Source: `docs/analysis/full_pipeline_ablations/drone_video_tests.md` and `raw_results/video_*/baseline/*/summary.csv`

### 3.1 Drone Clips — Per-Frame Detection Ablation (9 clips, 1234 GT boxes)

| Stage | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0: RGB Detector** | 647 | 556 | 587 | 0.538 | 0.524 | **0.531** | 0.476 |
| **S0b: + RGB Patch Filter** (no classifier) | 604 | 451 | 630 | 0.573 | 0.489 | 0.528 | 0.386 |
| **S1: + Classifier** (sa32, trust-aware) | 603 | 713 | 631 | 0.458 | 0.489 | 0.473 | 0.611 |
| **S2: + Classifier + Patch Filter** | 508 | 600 | 726 | 0.458 | 0.412 | 0.434 | 0.514 |

> [!NOTE]
> **Per-frame metrics understate the cascade.** At bbox level, the classifier merges RGB and IR-grayscale detections, inflating per-frame FP counts when both modalities fire on clutter. The segment-level view (§3.3) is the production-relevant grain.

### 3.2 Drone Clips — Per-Size Breakdown (S0: RGB Detector)

Aggregated across 9 drone clips. Size buckets by GT area fraction.

| Size Bucket | n_gt | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** | 77 | 26 | 40 | 51 | 0.394 | 0.338 | **0.364** | 0.034 |
| **medium** | 437 | 355 | 98 | 82 | 0.784 | 0.812 | **0.798** | 0.084 |
| **large** | 720 | 266 | 418 | 454 | 0.389 | 0.369 | **0.379** | 0.358 |
| **ALL** | 1234 | 647 | 556 | 587 | 0.538 | 0.524 | **0.531** | 0.476 |

> [!IMPORTANT]
> **Medium drones are the sweet spot.** The baseline detector achieves F1 = 0.798 on medium-sized targets (0.1–1% of image area) but degrades to F1 ≈ 0.37 for both small and large targets. Small-drone misses are resolution-limited; large-target misses arise from heavily bird-cluttered scenes where large drone boxes overlap with bird detections, causing matching confusion.

### 3.3 Drone Clips — Temporal Segment-Level Metrics (2-of-3, 3-frame windows)

Patch filter at alert gate only. P/R/F1 on segments.

| Stage | TP seg | FP seg | FN seg | TN seg | Total seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S4: Temporal 2/3** | 241 | 18 | 181 | 16 | 456 | 0.931 | 0.571 | **0.708** | 0.039 |
| **S5: Temporal + Alert Gate Filter** | 226 | 12 | 196 | 22 | 456 | 0.950 | 0.536 | **0.685** | 0.026 |

> [!TIP]
> **Temporal smoothing recovers precision.** Per-frame drone P = 0.538 rises to segment-level P = 0.931 after 2-of-3 voting. The alert gate filter further cuts FP segments from 18 → 12 (−33%) at a cost of 3.5 pp segment recall (0.571 → 0.536).

### 3.4 Confuser Clips — Single-Frame FP Rates (no drone GT)

Every detection is a false positive. Reported as total FP boxes and FPPI (FP per image).

| Category | Clips | Frames | FP boxes | FPPI (boxes/frame) |
|---|---:|---:|---:|---:|
| **Birds** | 5 | 352 | 725 | **2.060** |
| **Airplanes** | 2 | 304 | 161 | **0.530** |
| **Helicopters** | 3 | 594 | 198 | **0.333** |
| **ALL confusers** | 10 | 1250 | 1084 | **0.867** |

#### With Cascade Stages (baseline)

| Category | Frames | S0 FPPI | + Classifier FPPI | + Clf+Filter FPPI | Suppression S0→S2 |
|---|---:|---:|---:|---:|---:|
| Birds | 352 | 2.060 | 0.347 | **0.210** | −89.8% |
| Airplanes | 304 | 0.530 | 0.451 | **0.171** | −67.7% |
| Helicopters | 594 | 0.333 | 0.242 | **0.177** | −46.8% |
| **ALL** | 1250 | 0.867 | 0.323 | **0.185** | −78.7% |

> [!IMPORTANT]
> **Birds are the cascade's primary target.** The classifier + filter cascade suppresses 90% of bird FPs (2.060 → 0.210 FPPI), while airplane suppression is 68% and helicopter suppression is 47%. This matches the design intent: bird rejection at the detector level hits a wall, so the downstream cascade handles it.

### 3.5 Confuser Clips — Temporal Segment-Level FP Rate

| Category | Clips | Segments | S4 seg fired (FP) | S4 FR% | S5 seg fired | S5 FR% | Suppression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Birds | 5 | 119 | 100 | 84.0% | 74 | **62.2%** | −26% |
| Airplanes | 2 | 102 | 37 | 36.3% | 17 | **16.7%** | −54% |
| Helicopters | 3 | 199 | 51 | 25.6% | 37 | **18.6%** | −27% |
| **ALL** | 10 | 420 | 188 | 44.8% | 128 | **30.5%** | −32% |

> [!NOTE]
> **Temporal + alert gate cuts confuser segment fire rate from 44.8% to 30.5%.** The alert gate's patch filter at veto time provides an additional 32% reduction on top of the temporal smoother. Birds remain the hardest category even after the full cascade (62.2% segment fire rate), because bird transits persist across multiple frames and survive the 2-of-3 voting window.

---

## Part 4: Svanström Paired Dataset — Full Pipeline Ablation (Trust-Aware)

**Dataset:** `svanstrom_paired` (4785 frames, 1954 drone GT boxes, mixed with confuser categories: AIRPLANE, BIRD, HELICOPTER)
**RGB Model:** baseline (`Yolo26n_trained`) at imgsz = 1280
**IR Model:** ir_v3b (`finetune_v3b`) at imgsz = 640 on native thermal IR
**Scoring:** IoP @ 0.5, trust-aware — classifier scored against the GT of the trusted modality only
**Type:** Paired (real RGB + real IR with separate ground-truth per modality)

Source: `docs/analysis/full_pipeline_ablations/raw_results/svanstrom/baseline/sa32/summary.csv`

> [!IMPORTANT]
> **This is the cascade's primary validation surface.** Unlike Anti-UAV (saturated, no confusers) or the video tests (grayscale IR fallback), Svanström has real paired IR and real confuser categories. This is where trust-aware routing and confuser suppression earn their value.

### 4.1 Per-Frame Detection Ablation — baseline (bbox-level, all sizes aggregated)

| Stage | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0: RGB Detector** | 1894 | 5252 | 60 | 0.265 | 0.969 | **0.416** | 1.098 |
| **S1: + Classifier** (sa32, trust-aware) | 1891 | 630 | 63 | 0.750 | 0.968 | **0.845** | 0.132 |
| **S2: + Classifier + Patch Filter** | 1825 | 503 | 129 | 0.784 | 0.934 | **0.852** | 0.105 |
| **S3: + Patch Filter only** (no classifier) | 1827 | 2760 | 127 | 0.398 | 0.935 | 0.559 | 0.577 |

> [!TIP]
> **Classifier is the dominant stage.** S0 → S1 cuts FPPI from 1.098 to 0.132 (−88%) while maintaining 96.8% recall. The trust-aware classifier routes confuser frames to the IR modality (where the IR detector doesn't hallucinate on birds/airplanes), suppressing 4622 of 5252 FP boxes. The patch filter (S2) provides a further 20% FP reduction (630 → 503) at 3.4 pp recall cost.

### 4.2 Per-Size Breakdown — baseline

#### S0: RGB Detector (per-frame)

| Size Bucket | n_gt | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** | 91 | 75 | 4185 | 16 | 0.018 | 0.824 | **0.035** | 0.875 |
| **medium** | 1819 | 1777 | 954 | 42 | 0.651 | 0.977 | **0.781** | 0.199 |
| **large** | 44 | 42 | 113 | 2 | 0.271 | 0.955 | **0.422** | 0.024 |
| **ALL** | **1954** | **1894** | **5252** | **60** | **0.265** | **0.969** | **0.416** | **1.098** |

#### S1: + Classifier (trust-aware)

| Size Bucket | n_gt | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **small** | 91 | 75 | 566 | 16 | 0.117 | 0.824 | **0.205** | 0.118 |
| **medium** | 1819 | 1774 | 64 | 45 | 0.965 | 0.975 | **0.970** | 0.013 |
| **large** | 44 | 42 | 0 | 2 | 1.000 | 0.955 | **0.977** | 0.000 |
| **ALL** | **1954** | **1891** | **630** | **63** | **0.750** | **0.968** | **0.845** | **0.132** |

> [!NOTE]
> **Cascade effectiveness by size:** Medium drones jump from F1 = 0.781 to F1 = 0.970 (+19 pp) — the classifier suppresses 93% of medium-bucket FPs (954 → 64). Large drones reach P = 1.000 after the classifier. Small-bucket precision remains low (0.117) because small FPs are confuser detections that survive the classifier's trust routing.

### 4.3 Per-Frame Detection Ablation — retrained_v2

| Stage | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **S0: RGB Detector** | 1125 | 3762 | 829 | 0.230 | 0.576 | **0.329** | 0.786 |
| **S1: + Classifier** (sa32, trust-aware) | 1124 | 1971 | 830 | 0.363 | 0.575 | **0.445** | 0.412 |
| **S2: + Classifier + Patch Filter** | 1111 | 1877 | 843 | 0.372 | 0.569 | **0.450** | 0.392 |

> [!WARNING]
> **retrained_v2 collapses on Svanström.** Aggressive hard-negative training (which eliminated bird/airplane hallucinations on the general dataset) destroyed recall on Svanström's drone-in-clutter scenarios. S0 recall = 0.576 vs baseline's 0.969. The classifier provides a modest lift (+11.6 pp F1) but cannot recover the missing detections — the cascade can only suppress FPs, not hallucinate TPs.

### 4.4 Temporal Segment-Level Metrics (2-of-3 voting, 3-frame windows)

Patch filter applied only at alert gate. P/R/F1 on segments.

#### baseline

| Stage | TP seg | FP seg | FN seg | TN seg | Total seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S4: Temporal 2/3** | 638 | 759 | 13 | 185 | 1595 | 0.457 | 0.980 | **0.623** | 0.476 |
| **S5: Temporal + Alert Gate Filter** | 623 | 346 | 28 | 598 | 1595 | 0.643 | 0.957 | **0.769** | 0.217 |

#### retrained_v2

| Stage | TP seg | FP seg | FN seg | TN seg | Total seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S4: Temporal 2/3** | 449 | 282 | 202 | 662 | 1595 | 0.614 | 0.690 | **0.650** | 0.177 |
| **S5: Temporal + Alert Gate Filter** | 445 | 170 | 206 | 774 | 1595 | 0.724 | 0.684 | **0.703** | 0.107 |

> [!TIP]
> **Segment-level flips the ranking.** At per-frame level, baseline (F1 = 0.845) dominates retrained_v2 (F1 = 0.450). But at segment level with alert gate, the gap narrows: baseline S5 F1 = 0.769 vs retrained_v2 S5 F1 = 0.703. retrained_v2's lower FP rate (FPPI = 0.107 vs 0.217) partially compensates for its recall deficit. In a deployment where false-alarm fatigue matters more than detection completeness, retrained_v2 could be preferable.

---

## Part 5: Cross-Dataset Size-Aggregate Summary

Aggregate P, R, and FPPI per dataset, per size bucket, at the **S0 (detector-only)** stage.

### 4.1 Selcom CCTV (selcom_960, 311 frames, 295 GT)

| Size | n_gt | P | R | FPPI |
|---|---:|---:|---:|---:|
| small | 166 | 0.893 | 0.301 | 0.019 |
| medium | 125 | 0.874 | 0.608 | 0.035 |
| large | 4 | 1.000 | 0.750 | 0.000 |
| **ALL** | **295** | **0.884** | **0.437** | **0.055** |

### 4.2 Video Tests — Drone Clips (baseline, 1359 frames, 1234 GT)

| Size | n_gt | P | R | FPPI |
|---|---:|---:|---:|---:|
| small | 77 | 0.394 | 0.338 | 0.034 |
| medium | 437 | 0.784 | 0.812 | 0.084 |
| large | 720 | 0.389 | 0.369 | 0.358 |
| **ALL** | **1234** | **0.538** | **0.524** | **0.476** |

### 4.3 Video Tests — Confuser Clips (baseline, 1250 frames, 0 GT)

| Category | Frames | FPPI (S0) | FPPI (+Clf+Filter) | FPPI (S5 seg) |
|---|---:|---:|---:|---:|
| Birds | 352 | 2.060 | 0.210 | 0.622* |
| Airplanes | 304 | 0.530 | 0.171 | 0.167* |
| Helicopters | 594 | 0.333 | 0.177 | 0.186* |
| **ALL** | **1250** | **0.867** | **0.185** | **0.305*** |

*\* Segment-level FR% expressed as FPPI equivalent (FP segments / total segments).*

### 4.4 Svanström Paired (baseline, sa32, 4785 frames, 1954 drone GT)

| Size | n_gt | P (S0) | R (S0) | FPPI (S0) | P (S1+clf) | R (S1+clf) | FPPI (S1+clf) |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 91 | 0.018 | 0.824 | 0.875 | 0.117 | 0.824 | 0.118 |
| medium | 1819 | 0.651 | 0.977 | 0.199 | 0.965 | 0.975 | 0.013 |
| large | 44 | 0.271 | 0.955 | 0.024 | 1.000 | 0.955 | 0.000 |
| **ALL** | **1954** | **0.265** | **0.969** | **1.098** | **0.750** | **0.967** | **0.132** |

#### Svanström Temporal (baseline, sa32)

| Stage | TP seg | FP seg | FN seg | TN seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S4: Temporal 2/3 | 638 | 759 | 13 | 185 | 0.457 | 0.980 | **0.623** | 0.476 |
| S5: + Alert Gate | 623 | 346 | 28 | 598 | 0.643 | 0.957 | **0.769** | 0.217 |

### 4.5 Anti-UAV Paired (baseline, sa32, 1779 frames, IoU @ 0.5)

| Size | n_gt | P (S0) | R (S0) | FPPI (S0) | P (S1+clf) | R (S1+clf) | FPPI (S1+clf) |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 158 | 0.944 | 0.968 | 0.005 | 0.950 | 0.968 | 0.005 |
| medium | 1405 | 0.973 | 0.982 | 0.022 | 0.973 | 0.981 | 0.021 |
| large | 100 | 0.990 | 1.000 | 0.001 | 0.990 | 1.000 | 0.001 |
| **ALL** | **1663** | **0.971** | **0.981** | **0.028** | **0.972** | **0.980** | **0.026** |

#### Anti-UAV Temporal (baseline, sa32)

| Stage | TP seg | FP seg | FN seg | TN seg | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S4: Temporal 2/3 | 555 | 0 | 22 | 16 | 1.000 | 0.962 | **0.981** | 0.000 |
| S5: + Alert Gate | 555 | 0 | 22 | 16 | 1.000 | 0.962 | **0.981** | 0.000 |

> [!NOTE]
> **Anti-UAV is a saturated sanity floor.** Cascade has zero effect — there are no confusers and the detector is already near-perfect. F1 = 0.981 at segment level.

---

## Part 6: Cross-Model Comparison on Shared Surfaces

### 6.1 Anti-UAV — Model Comparison (IoU @ 0.5)

| Model | TP | FP | FN | P | R | F1 | Temporal F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 1632 | 49 | 31 | 0.971 | 0.981 | 0.976 | 0.981 |
| **retrained_v2** | 1644 | 71 | 19 | 0.959 | 0.989 | 0.974 | 0.985 |
| **selcom_1280** | 1617 | 314 | 46 | 0.837 | 0.972 | 0.900 | 0.979 |
| **ir_model** (native IR) | 1654 | 35 | 101 | 0.979 | 0.942 | 0.960 | 0.982 |

### 6.2 Svanström — Model Comparison (IoP @ 0.5, S0 detector only)

| Model | TP | FP | FN | P | R | F1 | FPPI |
|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 1894 | 5252 | 60 | 0.265 | 0.969 | 0.416 | 1.098 |
| **retrained_v2** | 1125 | 3762 | 829 | 0.230 | 0.576 | 0.329 | 0.786 |

> [!WARNING]
> **Svanström includes confuser frames.** The low precision and high FPPI reflect hallucinations on BIRD/AIRPLANE/HELICOPTER frames within the same dataset. The classifier stage (§4.4) is where the cascade suppresses these — baseline S1 FPPI drops from 1.098 to 0.132.

---

## Reproduction

```powershell
# Selcom pipeline ablation (selcom_960):
python eval/eval_full_pipeline_singlepass.py --dataset selcom_val ^
    --rgb-detectors selcom_960 --classifiers sa32 --stride-cap 1000

# Video test pipeline ablation (baseline):
python eval/eval_full_pipeline_singlepass.py --dataset video_drone_* ^
    --rgb-detectors baseline --classifiers sa32 --stride-cap 1000

# Standalone RGB on general dataset:
python eval/eval_model.py --weights "RGB model/Yolo26n_trained/weights/best.pt" ^
    --dataset "G:/drone/dataset/dataset" --imgsz 640 --stride 3

# Standalone IR on thermal dataset:
python eval/eval_model.py --weights "runs/corrective_finetune/finetune_v3b/weights/best.pt" ^
    --dataset "G:/drone/IR_dset_final" --imgsz 640 --conf 0.40 --stride 7
```

---

## Changelog

- **2026-05-20** — Initial creation. Compiled from `docs/analysis/full_pipeline_ablations/raw_results/` CSVs, `docs/analysis/full_pipeline_ablations/drone_video_tests.md`, `docs/EVIDENCE_LEDGER.md`, and `docs/analysis/thesis_ablation_results.md`. Gap: `retrained_v2` on general RGB dataset test split not yet evaluated.
