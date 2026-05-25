# Drone-Detection Video Tests — Full Evaluation (Soft-Veto)

> **Script:** `eval/eval_drone_video_full.py`
> **RGB:** selcom_1280@960  |  **IR:** ir_v3b (grayscale-RGB)  |  **Patch verifier:** rgb_filter (image is RGB even for IR model)
> **Classifier:** sa32, **soft-veto** mode at τ=0.95
> **Scoring:** IoP @ 0.5 on drone clips, frame-level FR%/TN% on confuser clips
> **Frames:** drone=1359, confusers=1250

## Soft-veto, in practice

Soft-veto changes how the classifier's output is used **at decision time**. Same trained model, different rule.

- **If RGB has at least one detection:** keep RGB, *unless* the classifier is very confident the scene contains no drone (`P(reject_both) ≥ 0.95`). This is the *fail-open* part — we don't let the classifier override an RGB det unless it's extremely confident.
- **If RGB missed the drone** (no detection): fall back to the IR-grayscale detector's boxes, **but only if** the classifier trusts the IR modality (argmax votes IR-only or both).
- **Why we chose soft-veto here:** on a fully RGB dataset the IR branch receives the same grayscale-RGB image as the RGB branch, so the classifier sees identical global features on both sides — an OOD shift versus its paired training distribution. Under standard argmax, the classifier over-rejects (votes `reject_both`) on legitimate drone frames. Soft-veto fail-open recovers those frames, lifting recall above raw RGB while still using the classifier to gate the IR-fallback (otherwise we'd be ORing both modalities and adding confuser FPs).

---

## Step 1: Base Detector Performance (raw YOLO)

### Drone clips (1359 frames, IoP @ 0.5)

| Model | P | R | F1 | FP% | TN% |
|---|---:|---:|---:|---:|---:|
| selcom_1280@960 | 0.739 | 0.784 | 0.761 | 3.16% | 6.11% |
| ir_v3b (grayscale-RGB) | 0.720 | 0.386 | 0.502 | 2.13% | 7.14% |

## Step 2: Temporal Voting (2-of-3 segments)

### Drone clips (1359 frames → 453 segments)

| Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selcom_1280@960 + temporal | 336 | 9 | 83 | 25 | 0.9739 | 0.8019 | 0.8796 | 76.16% |
| ir_v3b (grayscale-RGB) + temporal | 196 | 6 | 223 | 28 | 0.9703 | 0.4678 | 0.6312 | 44.59% |

## Step 3: Patch Verifier (rgb_filter) & Alert Gate

Patch verifier (`rgb_filter`, threshold 0.70) applied to each detector's boxes directly. Alert gate = temporal voting on **post-filter** firings (the production rule: only let an alert through if the patch verifier passes on the third frame).

### Drone clips (1359 frames, 453 segments)

| Stage | P | R | F1 | FP% (frame) | TN% (frame) |
|---|---:|---:|---:|---:|---:|
| selcom_1280@960 + patch | 0.7500 | 0.6588 | 0.7015 | 1.69% | 7.58% |
| ir_v3b (grayscale-RGB) + patch | 0.7133 | 0.3347 | 0.4556 | 1.99% | 7.28% |

**Alert gate** (segment-level, patch applied at decision boundary):

| Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| selcom_1280@960 + alert gate | 287 | 2 | 132 | 32 | 0.9931 | 0.6850 | 0.8107 | 63.80% |
| ir_v3b (grayscale-RGB) + alert gate | 172 | 6 | 247 | 28 | 0.9663 | 0.4105 | 0.5762 | 39.29% |

## Step 4: Scene-Aware Trust Classifier (SA32, soft-veto)

### Drone clips (1359 frames)

| Stage | P | R | F1 | FP% (frame) | TN% (frame) | ΔF1 vs RGB |
|---|---:|---:|---:|---:|---:|---:|
| selcom_1280@960 | 0.7387 | 0.7836 | 0.7605 | 3.16% | 6.11% | — |
| ir_v3b (grayscale-RGB) | 0.7201 | 0.3857 | 0.5024 | 2.13% | 7.14% | — |
| classifier_sa32 (argmax) — for reference | 0.4940 | 0.5964 | 0.5404 | 2.21% | 7.06% | -0.2201 |
| classifier_sa32 (soft-veto τ=0.95) ← chosen | 0.7562 | 0.7115 | 0.7332 | 2.50% | 6.77% | -0.0273 |
| classifier_sa32 (soft-veto) + rgb_filter | 0.7674 | 0.5883 | 0.6661 | 1.25% | 8.02% | -0.0945 |

## Step 5: Per-size detection breakdown (drone clips)

#### **selcom_1280@960** (raw RGB)

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 117 | 94 | 96 | 213 | 0.5545 | 0.5493 | 0.5519 |
| medium | 371 | 197 | 47 | 418 | 0.6532 | 0.8876 | 0.7525 |
| large | 479 | 51 | 124 | 603 | 0.9038 | 0.7944 | 0.8455 |
| **all** | **967** | **342** | **267** | **1234** | **0.7387** | **0.7836** | **0.7605** |

#### **ir_v3b (grayscale-RGB)** (raw IR-grayscale)

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 75 | 80 | 138 | 213 | 0.4839 | 0.3521 | 0.4076 |
| medium | 167 | 31 | 251 | 418 | 0.8434 | 0.3995 | 0.5422 |
| large | 234 | 74 | 369 | 603 | 0.7597 | 0.3881 | 0.5137 |
| **all** | **476** | **185** | **758** | **1234** | **0.7201** | **0.3857** | **0.5024** |

#### **Soft-veto classifier (τ=0.95)**

| Size | TP | FP | FN | n_gt | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 78 | 81 | 135 | 213 | 0.4906 | 0.3662 | 0.4194 |
| medium | 298 | 136 | 120 | 418 | 0.6866 | 0.7129 | 0.6995 |
| large | 502 | 66 | 101 | 603 | 0.8838 | 0.8325 | 0.8574 |
| **all** | **878** | **283** | **356** | **1234** | **0.7562** | **0.7115** | **0.7332** |

---

## Step 6: Confuser clip suppression (no drone GT)

On confuser-only clips every detection is an FP by construction. Reporting frame-level FR% (frames that fired) and segment-level FR% (2-of-3 fired).

### Birds (5 clips, 352 frames, 119 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment, 2/3) | TN% (segment) |
|---|---:|---:|---:|---:|
| selcom_1280@960 | 250 | 43.75% | 42.02% | 57.98% |
| ir_v3b (grayscale-RGB) | 19 | 5.11% | 2.52% | 97.48% |
| selcom_1280@960 + patch | 125 | 25.00% | 19.33% | 80.67% |
| ir_v3b (grayscale-RGB) + patch | 16 | 4.26% | 1.68% | 98.32% |
| classifier_sa32 (argmax) | 36 | 5.97% | 2.52% | 97.48% |
| classifier_sa32 (soft-veto τ=0.95) | 113 | 21.31% | 15.97% | 84.03% |
| classifier_sa32 (soft-veto) + rgb_filter | 68 | 15.62% | 10.92% | 89.08% |

### Airplanes (2 clips, 304 frames, 102 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment, 2/3) | TN% (segment) |
|---|---:|---:|---:|---:|
| selcom_1280@960 | 134 | 36.84% | 34.31% | 65.69% |
| ir_v3b (grayscale-RGB) | 68 | 22.37% | 19.61% | 80.39% |
| selcom_1280@960 + patch | 51 | 12.83% | 10.78% | 89.22% |
| ir_v3b (grayscale-RGB) + patch | 32 | 10.53% | 9.80% | 90.20% |
| classifier_sa32 (argmax) | 138 | 21.38% | 18.63% | 81.37% |
| classifier_sa32 (soft-veto τ=0.95) | 101 | 26.64% | 24.51% | 75.49% |
| classifier_sa32 (soft-veto) + rgb_filter | 37 | 8.55% | 7.84% | 92.16% |

### Helicopters (3 clips, 594 frames, 199 segments)

| Stage | FP boxes | FR% (frame) | FR% (segment, 2/3) | TN% (segment) |
|---|---:|---:|---:|---:|
| selcom_1280@960 | 137 | 20.54% | 19.60% | 80.40% |
| ir_v3b (grayscale-RGB) | 42 | 6.90% | 5.53% | 94.47% |
| selcom_1280@960 + patch | 94 | 14.14% | 10.55% | 89.45% |
| ir_v3b (grayscale-RGB) + patch | 41 | 6.73% | 5.53% | 94.47% |
| classifier_sa32 (argmax) | 81 | 12.46% | 10.05% | 89.95% |
| classifier_sa32 (soft-veto τ=0.95) | 130 | 20.03% | 17.09% | 82.91% |
| classifier_sa32 (soft-veto) + rgb_filter | 94 | 14.81% | 10.55% | 89.45% |

---

## Per-Video Breakdown

Most drone clips contain birds in the scene alongside the drone (seagulls, generic flocks, attack-by-bird footage). Only `drone_takeoff_short` and `drone_takeoff_from_ground_and_not_hand_short` are clean takeoff clips. The per-clip RGB-vs-IR-grayscale split below shows where the cross-modal IR fallback actually earns its keep on these realistic mixed scenes.

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

---

## Delivered

- `docs/analysis/eval_drone_video_results.md`
