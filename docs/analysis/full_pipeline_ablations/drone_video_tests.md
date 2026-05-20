# Drone-detection video tests — Full Pipeline Ablations

- **Category:** RGB videos, mixed drone + confuser scenes
- **Scoring:** IoP @ 0.5 (drone clips). Confuser clips use frame-level FR%/TN% (no GT).
- **Drone clips:** 9
- **Confuser clips:** 10 (birds=5, airplanes=2, helicopters=3)

**Read order:** drone-clip aggregate first (the headline numbers), then per-confuser-category aggregates, then per-clip drill-downs at the end.

## Drone clips (bbox-level scoring)

Aggregated across all drone clips (TP/FP/FN summed, P/R/F1 recomputed).

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | rgb | 647 | 556 | 587 | 0.5378 | 0.5243 | 0.5310 |
| baseline | +rgb_filter | 604 | 451 | 630 | 0.5725 | 0.4895 | 0.5277 |
| baseline | classifier | 598 | 678 | 636 | 0.4687 | 0.4846 | 0.4765 |
| baseline | classifier→filter | 526 | 575 | 708 | 0.4777 | 0.4263 | 0.4505 |
| retrained_v2 | rgb | 642 | 152 | 592 | 0.8086 | 0.5203 | 0.6331 |
| retrained_v2 | +rgb_filter | 596 | 139 | 638 | 0.8109 | 0.4830 | 0.6054 |
| retrained_v2 | classifier | 727 | 499 | 507 | 0.5930 | 0.5891 | 0.5911 |
| retrained_v2 | classifier→filter | 651 | 444 | 583 | 0.5945 | 0.5276 | 0.5590 |
| selcom_1280 | rgb | 1,002 | 542 | 232 | 0.6490 | 0.8120 | 0.7214 |
| selcom_1280 | +rgb_filter | 833 | 449 | 401 | 0.6498 | 0.6750 | 0.6622 |
| selcom_1280 | classifier | 774 | 887 | 460 | 0.4660 | 0.6272 | 0.5347 |
| selcom_1280 | classifier→filter | 667 | 708 | 567 | 0.4851 | 0.5405 | 0.5113 |
| ir_grayscale | ir_grayscale | 476 | 185 | 758 | 0.7201 | 0.3857 | 0.5024 |
| ir_grayscale | +rgb_filter | 413 | 166 | 821 | 0.7133 | 0.3347 | 0.4556 |

### Temporal (3-frame segments, 2-of-3)

| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 241 | 18 | 181 | 16 | 0.9305 | 0.5711 | 0.7078 | 56.80% |
| baseline | temporal+alert_gate | 226 | 12 | 196 | 22 | 0.9496 | 0.5355 | 0.6848 | 52.19% |
| retrained_v2 | temporal | 231 | 5 | 191 | 29 | 0.9788 | 0.5474 | 0.7021 | 51.75% |
| retrained_v2 | temporal+alert_gate | 218 | 3 | 204 | 31 | 0.9864 | 0.5166 | 0.6781 | 48.46% |
| selcom_1280 | temporal | 361 | 13 | 61 | 21 | 0.9652 | 0.8555 | 0.9070 | 82.02% |
| selcom_1280 | temporal+alert_gate | 299 | 6 | 123 | 28 | 0.9803 | 0.7085 | 0.8226 | 66.89% |
| ir_grayscale | temporal | 197 | 7 | 225 | 27 | 0.9657 | 0.4668 | 0.6294 | 44.74% |
| ir_grayscale | temporal+alert_gate | 174 | 6 | 248 | 28 | 0.9667 | 0.4123 | 0.5781 | 39.47% |

## Birds clips (no drone GT — every detection is FP)

Two views: single-frame stages report total FP box count and boxes-per-frame (a precision proxy when no positive frames exist); temporal stages report segment-level fire rate / true-negative rate.

### Single-frame stages

| Model | Stage | clips | frames | FP boxes | boxes/frame |
|---|---|---:|---:|---:|---:|
| baseline | rgb | 5 | 352 | 725 | 2.060 |
| baseline | +rgb_filter | 5 | 352 | 436 | 1.239 |
| baseline | classifier | 5 | 352 | 122 | 0.347 |
| baseline | classifier→filter | 5 | 352 | 74 | 0.210 |
| retrained_v2 | rgb | 5 | 352 | 18 | 0.051 |
| retrained_v2 | +rgb_filter | 5 | 352 | 15 | 0.043 |
| retrained_v2 | classifier | 5 | 352 | 23 | 0.065 |
| retrained_v2 | classifier→filter | 5 | 352 | 19 | 0.054 |
| selcom_1280 | rgb | 5 | 352 | 538 | 1.528 |
| selcom_1280 | +rgb_filter | 5 | 352 | 280 | 0.795 |
| selcom_1280 | classifier | 5 | 352 | 82 | 0.233 |
| selcom_1280 | classifier→filter | 5 | 352 | 54 | 0.153 |
| ir_grayscale | ir_grayscale | 5 | 352 | 19 | 0.054 |
| ir_grayscale | +rgb_filter | 5 | 352 | 16 | 0.045 |

### Temporal stages (3-frame segments, 2-of-3)

| Model | Stage | clips | segments | seg fired (FP) | seg quiet (TN) | FR% | TN% |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 5 | 119 | 100 | 19 | 84.03% | 15.97% |
| baseline | temporal+alert_gate | 5 | 119 | 74 | 45 | 62.18% | 37.82% |
| retrained_v2 | temporal | 5 | 119 | 3 | 116 | 2.52% | 97.48% |
| retrained_v2 | temporal+alert_gate | 5 | 119 | 2 | 117 | 1.68% | 98.32% |
| selcom_1280 | temporal | 5 | 119 | 81 | 38 | 68.07% | 31.93% |
| selcom_1280 | temporal+alert_gate | 5 | 119 | 59 | 60 | 49.58% | 50.42% |
| ir_grayscale | temporal | 5 | 119 | 3 | 116 | 2.52% | 97.48% |
| ir_grayscale | temporal+alert_gate | 5 | 119 | 2 | 117 | 1.68% | 98.32% |

## Airplanes clips (no drone GT — every detection is FP)

Two views: single-frame stages report total FP box count and boxes-per-frame (a precision proxy when no positive frames exist); temporal stages report segment-level fire rate / true-negative rate.

### Single-frame stages

| Model | Stage | clips | frames | FP boxes | boxes/frame |
|---|---|---:|---:|---:|---:|
| baseline | rgb | 2 | 304 | 161 | 0.530 |
| baseline | +rgb_filter | 2 | 304 | 94 | 0.309 |
| baseline | classifier | 2 | 304 | 137 | 0.451 |
| baseline | classifier→filter | 2 | 304 | 52 | 0.171 |
| retrained_v2 | rgb | 2 | 304 | 94 | 0.309 |
| retrained_v2 | +rgb_filter | 2 | 304 | 59 | 0.194 |
| retrained_v2 | classifier | 2 | 304 | 103 | 0.339 |
| retrained_v2 | classifier→filter | 2 | 304 | 36 | 0.118 |
| selcom_1280 | rgb | 2 | 304 | 150 | 0.493 |
| selcom_1280 | +rgb_filter | 2 | 304 | 63 | 0.207 |
| selcom_1280 | classifier | 2 | 304 | 141 | 0.464 |
| selcom_1280 | classifier→filter | 2 | 304 | 40 | 0.132 |
| ir_grayscale | ir_grayscale | 2 | 304 | 68 | 0.224 |
| ir_grayscale | +rgb_filter | 2 | 304 | 32 | 0.105 |

### Temporal stages (3-frame segments, 2-of-3)

| Model | Stage | clips | segments | seg fired (FP) | seg quiet (TN) | FR% | TN% |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 2 | 102 | 37 | 65 | 36.27% | 63.73% |
| baseline | temporal+alert_gate | 2 | 102 | 17 | 85 | 16.67% | 83.33% |
| retrained_v2 | temporal | 2 | 102 | 21 | 81 | 20.59% | 79.41% |
| retrained_v2 | temporal+alert_gate | 2 | 102 | 11 | 91 | 10.78% | 89.22% |
| selcom_1280 | temporal | 2 | 102 | 38 | 64 | 37.25% | 62.75% |
| selcom_1280 | temporal+alert_gate | 2 | 102 | 11 | 91 | 10.78% | 89.22% |
| ir_grayscale | temporal | 2 | 102 | 20 | 82 | 19.61% | 80.39% |
| ir_grayscale | temporal+alert_gate | 2 | 102 | 10 | 92 | 9.80% | 90.20% |

## Helicopters clips (no drone GT — every detection is FP)

Two views: single-frame stages report total FP box count and boxes-per-frame (a precision proxy when no positive frames exist); temporal stages report segment-level fire rate / true-negative rate.

### Single-frame stages

| Model | Stage | clips | frames | FP boxes | boxes/frame |
|---|---|---:|---:|---:|---:|
| baseline | rgb | 3 | 594 | 198 | 0.333 |
| baseline | +rgb_filter | 3 | 594 | 146 | 0.246 |
| baseline | classifier | 3 | 594 | 144 | 0.242 |
| baseline | classifier→filter | 3 | 594 | 105 | 0.177 |
| retrained_v2 | rgb | 3 | 594 | 109 | 0.184 |
| retrained_v2 | +rgb_filter | 3 | 594 | 105 | 0.177 |
| retrained_v2 | classifier | 3 | 594 | 87 | 0.146 |
| retrained_v2 | classifier→filter | 3 | 594 | 83 | 0.140 |
| selcom_1280 | rgb | 3 | 594 | 198 | 0.333 |
| selcom_1280 | +rgb_filter | 3 | 594 | 158 | 0.266 |
| selcom_1280 | classifier | 3 | 594 | 104 | 0.175 |
| selcom_1280 | classifier→filter | 3 | 594 | 84 | 0.141 |
| ir_grayscale | ir_grayscale | 3 | 594 | 42 | 0.071 |
| ir_grayscale | +rgb_filter | 3 | 594 | 41 | 0.069 |

### Temporal stages (3-frame segments, 2-of-3)

| Model | Stage | clips | segments | seg fired (FP) | seg quiet (TN) | FR% | TN% |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 3 | 199 | 51 | 148 | 25.63% | 74.37% |
| baseline | temporal+alert_gate | 3 | 199 | 37 | 162 | 18.59% | 81.41% |
| retrained_v2 | temporal | 3 | 199 | 26 | 173 | 13.07% | 86.93% |
| retrained_v2 | temporal+alert_gate | 3 | 199 | 24 | 175 | 12.06% | 87.94% |
| selcom_1280 | temporal | 3 | 199 | 50 | 149 | 25.13% | 74.87% |
| selcom_1280 | temporal+alert_gate | 3 | 199 | 40 | 159 | 20.10% | 79.90% |
| ir_grayscale | temporal | 3 | 199 | 11 | 188 | 5.53% | 94.47% |
| ir_grayscale | temporal+alert_gate | 3 | 199 | 11 | 188 | 5.53% | 94.47% |

## Per-clip drill-down

Each clip's summary table is in `raw_results/<clip_key>/<detector>/<classifier>/summary.csv`. Compact preview below — `frames` is the clip length; for drone clips `n_gt` is the total drone GT box count.

### airplanes

| Clip | Frames | n_gt | best F1 (drone) or lowest FR% (confuser) |
|---|---:|---:|---|
| video_airplanes_airplanes_compilation | 249 | 0 | ir_grayscale/no_classifier: 52 FP boxes (0.21/frame) |
| video_airplanes_distant_airplane_over_head_flying_away | 55 | 0 | retrained_v2/no_classifier: 7 FP boxes (0.13/frame) |

### birds

| Clip | Frames | n_gt | best F1 (drone) or lowest FR% (confuser) |
|---|---:|---:|---|
| video_birds_birds_flying_overhead_various_sizes_short | 20 | 0 | ir_grayscale/no_classifier: 1 FP boxes (0.05/frame) |
| video_birds_birds_in_slow_motion_flying_various_sizes_compilation | 271 | 0 | ir_grayscale/no_classifier: 6 FP boxes (0.02/frame) |
| video_birds_distant_birds_flying_in_the_sky_short | 20 | 0 | retrained_v2/no_classifier: 4 FP boxes (0.20/frame) |
| video_birds_flock_of_birds_flying_short | 21 | 0 | ir_grayscale/no_classifier: 1 FP boxes (0.05/frame) |
| video_birds_flock_of_birds_flying_sunset | 20 | 0 | ir_grayscale/no_classifier: 0 FP boxes (0.00/frame) |

### drone

| Clip | Frames | n_gt | best F1 (drone) or lowest FR% (confuser) |
|---|---:|---:|---|
| video_drone_drone_and_bird_sky_and_trees_short | 114 | 115 | ir_grayscale/no_classifier: F1=0.6792 |
| video_drone_drone_attacked_by_bird_mountain_side_view | 108 | 88 | selcom_1280/no_classifier: F1=0.7676 |
| video_drone_drone_over_mountain_attacked_by_birds | 68 | 68 | retrained_v2/no_classifier: F1=0.6897 |
| video_drone_drone_seagull_attack | 235 | 194 | selcom_1280/no_classifier: F1=0.8217 |
| video_drone_drone_takeoff_from_ground_and_not_hand_short | 163 | 154 | selcom_1280/no_classifier: F1=0.8839 |
| video_drone_drone_takeoff_short | 116 | 116 | selcom_1280/no_classifier: F1=0.9140 |
| video_drone_drone_takeoff_short_trees_background_dji_air_3s_take_off_sho | 166 | 162 | retrained_v2/no_classifier: F1=0.8876 |
| video_drone_flock_of_seagulls_attack_drone_beach | 239 | 187 | baseline/no_classifier: F1=0.7460 |
| video_drone_two_birds_drone | 150 | 150 | ir_grayscale/no_classifier: F1=0.5641 |

### helicopters

| Clip | Frames | n_gt | best F1 (drone) or lowest FR% (confuser) |
|---|---:|---:|---|
| video_helicopters_helicopter_compilation | 554 | 0 | ir_grayscale/no_classifier: 38 FP boxes (0.07/frame) |
| video_helicopters_helicopter_overhead_short | 20 | 0 | ir_grayscale/no_classifier: 4 FP boxes (0.20/frame) |
| video_helicopters_helicopter_overhead_very_small_airplane_in_background | 20 | 0 | ir_grayscale/no_classifier: 0 FP boxes (0.00/frame) |
