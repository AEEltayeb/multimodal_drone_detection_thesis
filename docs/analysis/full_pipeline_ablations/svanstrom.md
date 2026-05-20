# svanstrom — Full Pipeline Ablations

- **Category:** ?
- **Scoring:** IOP @ 0.5
- **Frames evaluated (per detector, post-stride):** ~4,785

## Overall summary (all sizes) — bbox-level

Detection-on-drone scoring. Any detection that does not match a drone GT box (IoU/IoP ≥ 0.5) is an FP — confuser hallucinations (birds, planes, helis sharing the frame) are already counted here.

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | rgb | 1,894 | 5,252 | 60 | 0.2650 | 0.9693 | 0.4163 |
| baseline | +rgb_filter | 1,827 | 2,760 | 127 | 0.3983 | 0.9350 | 0.5586 |
| baseline | classifier | 1,891 | 630 | 63 | 0.7501 | 0.9678 | 0.8451 |
| baseline | classifier→filter | 1,825 | 503 | 129 | 0.7839 | 0.9340 | 0.8524 |
| hardneg_v3more | rgb | 1,872 | 6,718 | 82 | 0.2179 | 0.9580 | 0.3551 |
| hardneg_v3more | +rgb_filter | 1,809 | 3,774 | 145 | 0.3240 | 0.9258 | 0.4800 |
| hardneg_v3more | classifier | 1,870 | 1,027 | 84 | 0.6455 | 0.9570 | 0.7710 |
| hardneg_v3more | classifier→filter | 1,807 | 788 | 147 | 0.6963 | 0.9248 | 0.7945 |
| retrained_v2 | rgb | 1,125 | 3,762 | 829 | 0.2302 | 0.5757 | 0.3289 |
| retrained_v2 | +rgb_filter | 1,112 | 2,964 | 842 | 0.2728 | 0.5691 | 0.3688 |
| retrained_v2 | classifier | 1,124 | 1,971 | 830 | 0.3632 | 0.5752 | 0.4452 |
| retrained_v2 | classifier→filter | 1,111 | 1,877 | 843 | 0.3718 | 0.5686 | 0.4496 |

## Temporal stages — segment-level (3-frame windows, 2-of-3)

Each row is one 3-frame segment scored as a single binary decision: fired ≥ 2 of 3 frames vs. any GT in the window.

| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 638 | 759 | 13 | 185 | 0.4567 | 0.9800 | 0.6230 | 87.59% |
| baseline | temporal+alert_gate | 623 | 346 | 28 | 598 | 0.6429 | 0.9570 | 0.7691 | 60.75% |
| hardneg_v3more | temporal | 637 | 643 | 14 | 301 | 0.4977 | 0.9785 | 0.6598 | 80.25% |
| hardneg_v3more | temporal+alert_gate | 623 | 324 | 28 | 620 | 0.6579 | 0.9570 | 0.7797 | 59.37% |
| retrained_v2 | temporal | 449 | 282 | 202 | 662 | 0.6142 | 0.6897 | 0.6498 | 45.83% |
| retrained_v2 | temporal+alert_gate | 445 | 170 | 206 | 774 | 0.7236 | 0.6836 | 0.7030 | 38.56% |

## Sanity flags

⚠️  `baseline`: classifier R=0.9678 below max(R_rgb=0.9693, R_ir=0.0000)
⚠️  `hardneg_v3more`: classifier R=0.9570 below max(R_rgb=0.9580, R_ir=0.0000)
⚠️  `retrained_v2`: classifier R=0.5752 below max(R_rgb=0.5757, R_ir=0.0000)

## Per-size breakdown

### baseline (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 91 | 75 | 4,185 | 16 | 0.0176 | 0.8242 | 0.0345 |
| rgb | medium | 1819 | 1,777 | 954 | 42 | 0.6507 | 0.9769 | 0.7811 |
| rgb | large | 44 | 42 | 113 | 2 | 0.2710 | 0.9545 | 0.4221 |
| rgb | all | 1954 | 1,894 | 5,252 | 60 | 0.2650 | 0.9693 | 0.4163 |
| +rgb_filter | small | 91 | 69 | 2,336 | 22 | 0.0287 | 0.7582 | 0.0553 |
| +rgb_filter | medium | 1819 | 1,719 | 402 | 100 | 0.8105 | 0.9450 | 0.8726 |
| +rgb_filter | large | 44 | 39 | 22 | 5 | 0.6393 | 0.8864 | 0.7429 |
| +rgb_filter | all | 1954 | 1,827 | 2,760 | 127 | 0.3983 | 0.9350 | 0.5586 |
| classifier | small | 91 | 75 | 566 | 16 | 0.1170 | 0.8242 | 0.2049 |
| classifier | medium | 1819 | 1,774 | 64 | 45 | 0.9652 | 0.9753 | 0.9702 |
| classifier | large | 44 | 42 | 0 | 2 | 1.0000 | 0.9545 | 0.9767 |
| classifier | all | 1954 | 1,891 | 630 | 63 | 0.7501 | 0.9678 | 0.8451 |
| classifier→filter | small | 91 | 69 | 445 | 22 | 0.1342 | 0.7582 | 0.2281 |
| classifier→filter | medium | 1819 | 1,717 | 58 | 102 | 0.9673 | 0.9439 | 0.9555 |
| classifier→filter | large | 44 | 39 | 0 | 5 | 1.0000 | 0.8864 | 0.9398 |
| classifier→filter | all | 1954 | 1,825 | 503 | 129 | 0.7839 | 0.9340 | 0.8524 |
| temporal | all | 651 | 638 | 759 | 13 | 0.4567 | 0.9800 | 0.6230 |
| temporal+alert_gate | all | 651 | 623 | 346 | 28 | 0.6429 | 0.9570 | 0.7691 |

### hardneg_v3more (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 91 | 77 | 5,761 | 14 | 0.0132 | 0.8462 | 0.0260 |
| rgb | medium | 1819 | 1,756 | 879 | 63 | 0.6664 | 0.9654 | 0.7885 |
| rgb | large | 44 | 39 | 78 | 5 | 0.3333 | 0.8864 | 0.4845 |
| rgb | all | 1954 | 1,872 | 6,718 | 82 | 0.2179 | 0.9580 | 0.3551 |
| +rgb_filter | small | 91 | 71 | 3,396 | 20 | 0.0205 | 0.7802 | 0.0399 |
| +rgb_filter | medium | 1819 | 1,700 | 360 | 119 | 0.8252 | 0.9346 | 0.8765 |
| +rgb_filter | large | 44 | 38 | 18 | 6 | 0.6786 | 0.8636 | 0.7600 |
| +rgb_filter | all | 1954 | 1,809 | 3,774 | 145 | 0.3240 | 0.9258 | 0.4800 |
| classifier | small | 91 | 77 | 938 | 14 | 0.0759 | 0.8462 | 0.1392 |
| classifier | medium | 1819 | 1,754 | 84 | 65 | 0.9543 | 0.9643 | 0.9593 |
| classifier | large | 44 | 39 | 5 | 5 | 0.8864 | 0.8864 | 0.8864 |
| classifier | all | 1954 | 1,870 | 1,027 | 84 | 0.6455 | 0.9570 | 0.7710 |
| classifier→filter | small | 91 | 71 | 700 | 20 | 0.0921 | 0.7802 | 0.1647 |
| classifier→filter | medium | 1819 | 1,698 | 83 | 121 | 0.9534 | 0.9335 | 0.9433 |
| classifier→filter | large | 44 | 38 | 5 | 6 | 0.8837 | 0.8636 | 0.8736 |
| classifier→filter | all | 1954 | 1,807 | 788 | 147 | 0.6963 | 0.9248 | 0.7945 |
| temporal | all | 651 | 637 | 643 | 14 | 0.4977 | 0.9785 | 0.6598 |
| temporal+alert_gate | all | 651 | 623 | 324 | 28 | 0.6579 | 0.9570 | 0.7797 |

### retrained_v2 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 91 | 50 | 1,636 | 41 | 0.0297 | 0.5495 | 0.0563 |
| rgb | medium | 1819 | 1,032 | 758 | 787 | 0.5765 | 0.5673 | 0.5719 |
| rgb | large | 44 | 43 | 1,368 | 1 | 0.0305 | 0.9773 | 0.0591 |
| rgb | all | 1954 | 1,125 | 3,762 | 829 | 0.2302 | 0.5757 | 0.3289 |
| +rgb_filter | small | 91 | 50 | 1,246 | 41 | 0.0386 | 0.5495 | 0.0721 |
| +rgb_filter | medium | 1819 | 1,020 | 576 | 799 | 0.6391 | 0.5607 | 0.5974 |
| +rgb_filter | large | 44 | 42 | 1,142 | 2 | 0.0355 | 0.9545 | 0.0684 |
| +rgb_filter | all | 1954 | 1,112 | 2,964 | 842 | 0.2728 | 0.5691 | 0.3688 |
| classifier | small | 91 | 50 | 756 | 41 | 0.0620 | 0.5495 | 0.1115 |
| classifier | medium | 1819 | 1,031 | 475 | 788 | 0.6846 | 0.5668 | 0.6202 |
| classifier | large | 44 | 43 | 740 | 1 | 0.0549 | 0.9773 | 0.1040 |
| classifier | all | 1954 | 1,124 | 1,971 | 830 | 0.3632 | 0.5752 | 0.4452 |
| classifier→filter | small | 91 | 50 | 667 | 41 | 0.0697 | 0.5495 | 0.1238 |
| classifier→filter | medium | 1819 | 1,019 | 472 | 800 | 0.6834 | 0.5602 | 0.6157 |
| classifier→filter | large | 44 | 42 | 738 | 2 | 0.0538 | 0.9545 | 0.1019 |
| classifier→filter | all | 1954 | 1,111 | 1,877 | 843 | 0.3718 | 0.5686 | 0.4496 |
| temporal | all | 651 | 449 | 282 | 202 | 0.6142 | 0.6897 | 0.6498 |
| temporal+alert_gate | all | 651 | 445 | 170 | 206 | 0.7236 | 0.6836 | 0.7030 |

## Per-stage commentary


## Delivered

- `docs/analysis/full_pipeline_ablations/svanstrom.md`
- `docs/analysis/full_pipeline_ablations/csv/svanstrom.csv`
