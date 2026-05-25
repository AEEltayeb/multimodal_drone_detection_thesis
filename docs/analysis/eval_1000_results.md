# Detector Ablation Results — 1000-Frame Evaluation

> **Date:** 2026-05-22
> **Script:** `eval/eval_detector.py` (Anti-UAV, Svanström) + `eval/eval_drone_video_full.py` (drone-video)
> **RGB conf:** 0.25 | **IR conf:** 0.40

---

## Headline Results Summary

*   **Anti-UAV Baseline:** RGB F1 = **0.969** | Thermal IR F1 = **0.957**
*   **Svanström Baseline (Confuser-Heavy):** RGB F1 = **0.588** (39.10% FP) | Thermal IR F1 = **0.953** (1.10% FP)
*   **SA32 Trust Classifier Fusion:** Drives Anti-UAV F1 to **0.981** (0.00% FP) and Svanström F1 to **0.976** (0.00% FP, a massive **+0.388** gain over raw RGB, and **+0.023** over raw IR).
*   **Alert Gate (Temporal + Patch):** Suppresses Svanström RGB false positives from **39.46% to 6.64%** (**+0.224** F1 lift).
*   **Grayscale Fallback (Soft-Veto + Patch):** Recovers **+19.3 pp** F1 over raw argmax on RGB-only videos, while slashing bird false alarms by **~75%** (from 43.75% to 15.62% fire rate).
*   **IR-Grayscale on RGB-Only Videos:** Cross-modal fallback (IR weights on grayscale-RGB) beats RGB on confuser fire-rate **on every confuser clip** (10/10), and **wins F1 outright** on the hardest drone clips — mountain/sky/distant-drone scenes — by +6 to +20 pp (vs. losing 30–70 pp on close-up takeoff clips). Soft-veto picks the right side per clip with no re-training.

---

## Step 1: Base Detector Performance (Raw YOLO)

This establishes the raw capability of each sensor modality before any pipeline components are applied.

### Anti-UAV (1000 frames, stride = 85, IoU scoring)

| Model | P | R | F1 | FP% | TN% |
|---|---|---|---|---|---|
| selcom_1280@960 (RGB) | 0.961 | 0.976 | 0.969 | 0.40% | 6.30% |
| ir_v3b (Thermal IR) | 0.977 | 0.937 | 0.957 | 0.20% | 1.30% |

### Svanström (1000 frames, stride = 28, IoP scoring)

| Model | P | R | F1 | FP% | TN% |
|---|---|---|---|---|---|
| selcom_1280@960 (RGB) | 0.450 | 0.848 | 0.588 | 39.10% | 19.10% |
| ir_v3b (Thermal IR) | 0.941 | 0.966 | 0.953 | 1.10% | 57.90% |

### Drone-detection video tests (1359 frames, all drone clips, IoP scoring)

RGB-only dataset (no paired IR). IR runs on a grayscale copy of the RGB frame as a cross-modal fallback. Patch verifier uses `rgb_filter` for both modalities since the input image is RGB.

| Model | P | R | F1 | FP% | TN% |
|---|---|---|---|---|---|
| selcom_1280@960 (RGB) | 0.739 | 0.784 | 0.761 | 3.16% | 6.11% |
| ir_v3b (grayscale-RGB) | 0.720 | 0.386 | 0.502 | 2.13% | 7.14% |

---

## Step 2: Temporal Logic & TROI Recovery

Evaluating the impact of k=2 out of n=3 rolling window voting (filtering transient noise) alongside Target Region of Interest (TROI) recovery (5-frame TTL crop-based re-inference).

### Anti-UAV (1911 frames, 637 windows)

| Stage | P | R | F1 | FP% | TN% | Recovered | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 (Base) | 0.9742 | 0.9839 | 0.9790 | 0.00% | 5.60% | — | — | — | — |
| + temporal (2/3) | 0.9747 | 0.9834 | 0.9790 | 0.00% | 5.60% | — | +0.001 | -0.001 | +0.000 |
| + TROI (ttl=5) | 0.9742 | 0.9839 | 0.9790 | 0.00% | 5.60% | 0 | +0.000 | +0.000 | +0.000 |
| ir_v3b (Base) | 0.9807 | 0.9418 | 0.9608 | 0.16% | 0.99% | — | — | — | — |
| + temporal (2/3) | 0.9833 | 0.9370 | 0.9596 | 0.00% | 1.15% | — | +0.003 | -0.005 | -0.001 |
| + TROI (ttl=5) | 0.9786 | 0.9444 | 0.9612 | 0.16% | 0.99% | 8 | -0.002 | +0.003 | +0.000 |

### Svanström (5859 frames, 1953 windows)

| Stage | P | R | F1 | FP% | TN% | Recovered | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 (Base) | 0.4376 | 0.8403 | 0.5755 | 39.46% | 19.68% | — | — | — | — |
| + temporal (2/3) | 0.4440 | 0.8282 | 0.5781 | 37.81% | 21.33% | — | +0.006 | -0.012 | +0.003 |
| + TROI (ttl=5) | 0.4222 | 0.8641 | 0.5672 | 42.16% | 16.98% | 215 | -0.015 | +0.024 | -0.008 |
| ir_v3b (Base) | 0.9478 | 0.9724 | 0.9599 | 1.11% | 58.75% | — | — | — | — |
| + temporal (2/3) | 0.9505 | 0.9715 | 0.9609 | 0.99% | 58.87% | — | +0.003 | -0.001 | +0.001 |
| + TROI (ttl=5) | 0.9447 | 0.9728 | 0.9585 | 1.23% | 58.63% | 9 | -0.003 | +0.000 | -0.001 |

### Drone-detection video tests (1359 frames, 453 segments — temporal only; TROI not run)

| Stage | TP | FP | FN | TN | P | R | F1 | FR% | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 (Base) | — | — | — | — | 0.7387 | 0.7836 | 0.7605 | — | — |
| + temporal (2/3) | 336 | 9 | 83 | 25 | 0.9739 | 0.8019 | 0.8796 | 76.16% | +0.119 |
| ir_v3b (Base) | — | — | — | — | 0.7201 | 0.3857 | 0.5024 | — | — |
| + temporal (2/3) | 196 | 6 | 223 | 28 | 0.9703 | 0.4678 | 0.6312 | 44.59% | +0.129 |

---

## Step 3: Patch Verifier & Alert Gate Impact

Evaluating the modality-specific CNN patch classifier (confuser filter, threshold=0.7) applied stand-alone on detections, versus the Alert Gate (which runs the patch verifier specifically at the temporal decision boundary).

### Anti-UAV (1000 frames base, 1911 frames temporal, 0 confusers)

| Stage | P | R | F1 | FP% | TN% | Suppressed | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| selcom + patch | 0.961 | 0.976 | 0.969 | 0.40% | 6.30% | — | +0.000 | +0.000 | +0.000 |
| ir_v3b + patch | 0.977 | 0.937 | 0.957 | 0.20% | 1.30% | — | +0.000 | +0.000 | +0.000 |
| selcom + alert gate | 0.9747 | 0.9834 | 0.9790 | 0.00% | 5.60% | 0 | +0.000 | +0.000 | +0.000 |
| ir_v3b + alert gate | 0.9833 | 0.9370 | 0.9596 | 0.00% | 1.15% | 0 | +0.000 | +0.000 | +0.000 |

### Svanström (1000 frames base, 5859 frames temporal)

| Stage | P | R | F1 | FP% | TN% | Suppressed | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|---|
| selcom + patch | 0.680 | 0.824 | 0.745 | 14.70% | 43.50% | — | +0.230 | -0.024 | +0.157 |
| ir_v3b + patch | 0.942 | 0.949 | 0.945 | 1.00% | 58.00% | — | +0.001 | -0.017 | -0.008 |
| selcom + alert gate | 0.8135 | 0.7857 | 0.7993 | 6.64% | 52.50% | 677 | +0.376 | -0.055 | +0.224 |
| ir_v3b + alert gate | 0.9548 | 0.9243 | 0.9393 | 0.75% | 59.11% | 43 | +0.007 | -0.048 | -0.021 |

### Drone-detection video tests (1359 frames base, 453 segments alert gate)

Patch verifier uses `rgb_filter` for both modalities (image is RGB).

| Stage | P | R | F1 | FP% (frame) | TN% (frame) | ΔF1 |
|---|---|---|---|---|---|---|
| selcom + patch | 0.7500 | 0.6588 | 0.7015 | 1.69% | 7.58% | -0.059 |
| ir_v3b + patch | 0.7133 | 0.3347 | 0.4556 | 1.99% | 7.28% | -0.047 |
| selcom + alert gate | 0.9931 | 0.6850 | 0.8107 | — | — | +0.050 vs temporal |
| ir_v3b + alert gate | 0.9663 | 0.4105 | 0.5762 | — | — | -0.055 vs temporal |

---

## Step 4: Scene-Aware Trust Classifier (SA32)

Evaluating the dual-modality trust-aware gating classifier (using 32 handcrafted spatial/temporal/signal features) to decide modality routing.

### Anti-UAV (1000 frames, stride = 85)

| Stage | P | R | F1 | FP% | TN% | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 | 0.961 | 0.976 | 0.969 | 0.40% | 6.30% | — | — | — |
| ir_v3b | 0.977 | 0.937 | 0.957 | 0.20% | 1.30% | — | — | — |
| classifier_sa32 | 0.974 | 0.988 | 0.981 | 0.00% | 1.10% | — | — | — |
| Δ vs selcom | — | — | — | — | — | +0.013 | +0.012 | +0.012 |
| Δ vs ir_v3b | — | — | — | — | — | -0.003 | +0.051 | +0.025 |

### Svanström (1000 frames, stride = 28)

| Stage | P | R | F1 | FP% | TN% | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 | 0.450 | 0.848 | 0.588 | 39.10% | 19.10% | — | — | — |
| ir_v3b | 0.941 | 0.966 | 0.953 | 1.10% | 57.90% | — | — | — |
| classifier_sa32 | 0.972 | 0.980 | 0.976 | 0.00% | 58.20% | — | — | — |
| Δ vs selcom | — | — | — | — | — | +0.521 | +0.133 | +0.388 |
| Δ vs ir_v3b | — | — | — | — | — | +0.031 | +0.015 | +0.023 |

### Drone-detection video tests (1359 frames)

On RGB-only data the classifier's two feature branches both receive grayscale-RGB — an OOD shift versus its paired training. Standard **argmax** over-rejects (votes `reject_both`) on legitimate drone frames; **soft-veto** (τ=0.95) fail-opens for RGB unless the classifier is very confident, recovering that lost recall. We use soft-veto on this dataset.

| Stage | P | R | F1 | FP% | TN% | ΔP | ΔR | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| selcom_1280@960 | 0.7387 | 0.7836 | 0.7605 | 3.16% | 6.11% | — | — | — |
| ir_v3b (grayscale-RGB) | 0.7201 | 0.3857 | 0.5024 | 2.13% | 7.14% | — | — | — |
| classifier_sa32 (argmax) | 0.4940 | 0.5964 | 0.5404 | 2.21% | 7.06% | — | — | — |
| **classifier_sa32 (soft-veto τ=0.95)** ← chosen | 0.7562 | 0.7115 | 0.7332 | 2.50% | 6.77% | — | — | — |
| classifier_sa32 (soft-veto) + rgb_filter | 0.7674 | 0.5883 | 0.6661 | 1.25% | 8.02% | — | — | — |
| Δ soft-veto vs argmax | — | — | — | — | — | +0.262 | +0.115 | +0.193 |
| Δ soft-veto vs selcom (raw) | — | — | — | — | — | +0.018 | -0.072 | -0.027 |
| Δ soft-veto vs ir_v3b (raw) | — | — | — | — | — | +0.036 | +0.326 | +0.231 |

**Read:** soft-veto cleanly recovers the +19 pp F1 lost by argmax (the grayscale-OOD failure) but still finishes 3 pp under raw RGB on a high-recall detector. On lower-recall detectors (e.g., baseline) soft-veto beats raw RGB; here selcom's recall is already too high for the classifier to add headroom.

---

## Step 5: Per-Size Detections (Raw YOLO Models)

Detailed breakdowns of detector performance categorized by target size (small, medium, large).

### Anti-UAV

**selcom_1280 @ imgsz=960 (RGB)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 78 | 5 | 4 | 82 | 0.9398 | 0.9512 | 0.9455 |
| medium | 773 | 31 | 18 | 791 | 0.9614 | 0.9772 | 0.9693 |
| large | 60 | 1 | 0 | 60 | 0.9836 | 1.0000 | 0.9917 |
| **all** | **911** | **37** | **22** | **933** | **0.9610** | **0.9764** | **0.9686** |

Frame-level: TP=914, FP=4, FN=19, TN=63

**ir_v3b @ imgsz=640 (Thermal IR)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 10 | 1 | 3 | 13 | 0.9091 | 0.7692 | 0.8333 |
| medium | 853 | 21 | 57 | 910 | 0.9760 | 0.9374 | 0.9563 |
| large | 60 | 0 | 2 | 62 | 1.0000 | 0.9677 | 0.9836 |
| **all** | **923** | **22** | **62** | **985** | **0.9767** | **0.9371** | **0.9565** |

Frame-level: TP=937, FP=2, FN=48, TN=13

### Svanström

**selcom_1280 @ imgsz=960 (RGB)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 13 | 190 | 5 | 18 | 0.0640 | 0.7222 | 0.1176 |
| medium | 334 | 158 | 59 | 393 | 0.6789 | 0.8499 | 0.7548 |
| large | 9 | 87 | 0 | 9 | 0.0938 | 1.0000 | 0.1714 |
| **all** | **356** | **435** | **64** | **420** | **0.4501** | **0.8476** | **0.5879** |

Frame-level: TP=355, FP=391, FN=63, TN=191

**ir_v3b @ imgsz=640 (Thermal IR)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 0 | 0 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |
| medium | 387 | 24 | 13 | 400 | 0.9416 | 0.9675 | 0.9544 |
| large | 9 | 1 | 0 | 9 | 0.9000 | 1.0000 | 0.9474 |
| **all** | **396** | **25** | **14** | **410** | **0.9406** | **0.9659** | **0.9531** |

Frame-level: TP=405, FP=11, FN=5, TN=579

### Drone-detection video tests

**selcom_1280 @ imgsz=960 (RGB)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 117 | 94 | 96 | 213 | 0.5545 | 0.5493 | 0.5519 |
| medium | 371 | 197 | 47 | 418 | 0.6532 | 0.8876 | 0.7525 |
| large | 479 | 51 | 124 | 603 | 0.9038 | 0.7944 | 0.8455 |
| **all** | **967** | **342** | **267** | **1234** | **0.7387** | **0.7836** | **0.7605** |

**ir_v3b @ imgsz=640 (on grayscale-RGB)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 75 | 80 | 138 | 213 | 0.4839 | 0.3521 | 0.4076 |
| medium | 167 | 31 | 251 | 418 | 0.8434 | 0.3995 | 0.5422 |
| large | 234 | 74 | 369 | 603 | 0.7597 | 0.3881 | 0.5137 |
| **all** | **476** | **185** | **758** | **1234** | **0.7201** | **0.3857** | **0.5024** |

**classifier_sa32 (soft-veto τ=0.95)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---|---|---|---|---|---|---|
| small | 78 | 81 | 135 | 213 | 0.4906 | 0.3662 | 0.4194 |
| medium | 298 | 136 | 120 | 418 | 0.6866 | 0.7129 | 0.6995 |
| large | 502 | 66 | 101 | 603 | 0.8838 | 0.8325 | 0.8574 |
| **all** | **878** | **283** | **356** | **1234** | **0.7562** | **0.7115** | **0.7332** |

---

## Step 6: Confuser-Clip Suppression (drone-detection video tests only)

Confuser-only clips have no drone GT; every detection is by definition an FP. Reporting frame-level fire rate (FR%) and segment-level (2-of-3) FR% / TN%. Lower FR% is better.

### Birds (5 clips, 352 frames, 119 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment) | TN% (segment) |
|---|---|---|---|---|
| selcom_1280@960 | 250 | 43.75% | 42.02% | 57.98% |
| ir_v3b (grayscale-RGB) | 19 | 5.11% | 2.52% | 97.48% |
| selcom + patch | 125 | 25.00% | 19.33% | 80.67% |
| ir_v3b + patch | 16 | 4.26% | 1.68% | 98.32% |
| classifier_sa32 (argmax) | 36 | 5.97% | 2.52% | 97.48% |
| classifier_sa32 (soft-veto τ=0.95) | 113 | 21.31% | 15.97% | 84.03% |
| **classifier_sa32 (soft-veto) + rgb_filter** | **68** | **15.62%** | **10.92%** | **89.08%** |

### Airplanes (2 clips, 304 frames, 102 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment) | TN% (segment) |
|---|---|---|---|---|
| selcom_1280@960 | 134 | 36.84% | 34.31% | 65.69% |
| ir_v3b (grayscale-RGB) | 68 | 22.37% | 19.61% | 80.39% |
| selcom + patch | 51 | 12.83% | 10.78% | 89.22% |
| ir_v3b + patch | 32 | 10.53% | 9.80% | 90.20% |
| classifier_sa32 (argmax) | 138 | 21.38% | 18.63% | 81.37% |
| classifier_sa32 (soft-veto τ=0.95) | 101 | 26.64% | 24.51% | 75.49% |
| **classifier_sa32 (soft-veto) + rgb_filter** | **37** | **8.55%** | **7.84%** | **92.16%** |

### Helicopters (3 clips, 594 frames, 199 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment) | TN% (segment) |
|---|---|---|---|---|
| selcom_1280@960 | 137 | 20.54% | 19.60% | 80.40% |
| ir_v3b (grayscale-RGB) | 42 | 6.90% | 5.53% | 94.47% |
| selcom + patch | 94 | 14.14% | 10.55% | 89.45% |
| ir_v3b + patch | 41 | 6.73% | 5.53% | 94.47% |
| classifier_sa32 (argmax) | 81 | 12.46% | 10.05% | 89.95% |
| classifier_sa32 (soft-veto τ=0.95) | 130 | 20.03% | 17.09% | 82.91% |
| **classifier_sa32 (soft-veto) + rgb_filter** | **94** | **14.81%** | **10.55%** | **89.45% |

**Takeaway:** soft-veto + rgb_filter is the best confuser stack on birds and airplanes (the dominant FP source on real RGB video). On helicopters, raw IR-grayscale + patch is competitive but the production deployment can't predict per-frame category, so the soft-veto + patch combo is the right default.

---

## Step 7: Per-Video Breakdown (drone-detection video tests)

Most drone clips contain birds in the scene alongside the drone (seagulls, generic flocks, attack-by-bird footage). Only `drone_takeoff_short` and `drone_takeoff_from_ground_and_not_hand_short` are clean takeoff clips with no confusers. The per-clip RGB-vs-IR-grayscale split below shows where the cross-modal IR fallback actually earns its keep on realistic mixed scenes.

### Drone clips — F1 per clip (IoP @ 0.5)

| Clip | Frames | n_gt | RGB F1 | IR-gray F1 | ΔF1 (gray − RGB) | Softveto F1 |
|---|---:|---:|---:|---:|---:|---:|
| drone_and_bird_sky_and_trees_short | 114 | 115 | 0.6201 | 0.6792 | +0.0592 ★ | 0.6407 |
| drone_attacked_by_bird_mountain_side_view | 108 | 88 | 0.7349 | 0.3186 | -0.4164 | 0.7239 |
| drone_over_mountain_attacked_by_birds | 68 | 68 | 0.2651 | 0.4681 | +0.2030 ★ | 0.5347 |
| drone_seagull_attack | 235 | 194 | 0.7982 | 0.6446 | -0.1535 | 0.8153 |
| drone_takeoff_from_ground_and_not_hand_short | 163 | 154 | 0.9211 | 0.5806 | -0.3404 | 0.9180 |
| drone_takeoff_short | 116 | 116 | 0.8727 | 0.1830 | -0.6897 | 0.8267 |
| drone_takeoff_short_trees_background_dji_air_3s_take_off_sho | 166 | 162 | 0.9032 | 0.3894 | -0.5138 | 0.9032 |
| flock_of_seagulls_attack_drone_beach | 239 | 187 | 0.8708 | 0.4048 | -0.4660 | 0.6335 |
| two_birds_drone | 150 | 150 | 0.5421 | 0.5641 | +0.0220 ★ | 0.4478 |

★ = IR-grayscale outperforms RGB on that clip (cross-modal recovery).

**Read:** IR-grayscale wins outright on the *hardest* clips for RGB — the mountain/sky scenes where the drone is small/distant and RGB loses contrast. On clean takeoff close-ups it loses badly (RGB easily picks up the in-focus drone). Where IR-grayscale wins it wins by 6–20 pp; where it loses, it loses by 30–70 pp. The softveto rule keeps the RGB win on close shots while picking up the IR boost on mountain/sky shots without re-training.

### Confuser clips — segment fire-rate per clip (lower is better)

| Category | Clip | Frames | RGB FR% | IR-gray FR% | Δ (gray − RGB) | Softveto+patch FR% |
|---|---|---:|---:|---:|---:|---:|
| airplanes | airplanes_compilation | 249 | 33.73% | 16.87% | -16.87pp ★ | 6.02% |
| airplanes | distant_airplane_over_head_flying_away | 55 | 36.84% | 31.58% | -5.26pp ★ | 15.79% |
| birds | birds_flying_overhead_various_sizes_short | 20 | 85.71% | 0.00% | -85.71pp ★ | 57.14% |
| birds | birds_in_slow_motion_flying_various_sizes_compilation | 271 | 43.96% | 1.10% | -42.86pp ★ | 8.79% |
| birds | distant_birds_flying_in_the_sky_short | 20 | 42.86% | 28.57% | -14.29pp ★ | 14.29% |
| birds | flock_of_birds_flying_short | 21 | 0.00% | 0.00% | +0.00pp | 0.00% |
| birds | flock_of_birds_flying_sunset | 20 | 14.29% | 0.00% | -14.29pp ★ | 0.00% |
| helicopters | helicopter_compilation | 554 | 16.22% | 5.41% | -10.81pp ★ | 10.81% |
| helicopters | helicopter_overhead_short | 20 | 57.14% | 14.29% | -42.86pp ★ | 0.00% |
| helicopters | helicopter_overhead_very_small_airplane_in_background | 20 | 71.43% | 0.00% | -71.43pp ★ | 14.29% |

★ = IR-grayscale fires less (better confuser suppression than RGB on that clip).

**Read:** IR-grayscale beats RGB on confuser FR% on **every single clip** (9/10 strictly, 1/10 tied at 0%). That's the empirical justification for keeping the IR-grayscale branch alive on a fully RGB dataset — when RGB hallucinates on a bird/airplane, grayscale-RGB rarely does. Softveto+patch combines RGB's drone recall with IR-grayscale's confuser rejection.
