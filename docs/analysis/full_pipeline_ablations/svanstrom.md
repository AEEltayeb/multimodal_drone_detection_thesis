# Svanström — Full Pipeline Ablations

- **Category:** RGB+IR not-truly-paired, mixed confuser + drone
- **Scoring:** IOP @ 0.5
- **Frames evaluated (per detector, post-stride):** ~1,795

The classifier's keystone dataset. RGB collapses on confusers (birds, airplanes, helicopters) and hallucinates aggressively; IR is physically immune to feathers/wings and stays clean. Under trust-aware scoring the classifier routes RGB-confuser frames to the IR stream and lifts F1 from ~0.43 (RGB alone) to ~0.89 — the modality-arbitration rescue. The patch verifier (rgb_filter) only helps when applied before the classifier choice; once the classifier has picked the trustworthy modality the filter is largely redundant on this dataset.

## Overall summary (all sizes) — bbox-level

Detection-on-drone scoring. Any detection that does not match a drone GT box (IoU/IoP ≥ 0.5) is an FP — confuser hallucinations (birds, planes, helis sharing the frame) are already counted here.

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | rgb | 709 | 1,996 | 23 | 0.2621 | 0.9686 | 0.4126 |
| baseline | +rgb_filter | 682 | 1,083 | 50 | 0.3864 | 0.9317 | 0.5463 |
| baseline | classifier | 1,403 | 281 | 21 | 0.8331 | 0.9853 | 0.9028 |
| baseline | classifier→filter | 1,363 | 220 | 61 | 0.8610 | 0.9572 | 0.9066 |
| retrained_v2 | rgb | 425 | 1,449 | 307 | 0.2268 | 0.5806 | 0.3262 |
| retrained_v2 | +rgb_filter | 419 | 1,129 | 313 | 0.2707 | 0.5724 | 0.3675 |
| retrained_v2 | classifier | 1,119 | 765 | 96 | 0.5939 | 0.9210 | 0.7222 |
| retrained_v2 | classifier→filter | 1,100 | 728 | 115 | 0.6018 | 0.9053 | 0.7230 |
| selcom_1280 | rgb | 680 | 1,001 | 52 | 0.4045 | 0.9290 | 0.5636 |
| selcom_1280 | +rgb_filter | 652 | 388 | 80 | 0.6269 | 0.8907 | 0.7359 |
| selcom_1280 | classifier | 1,374 | 46 | 18 | 0.9676 | 0.9871 | 0.9772 |
| selcom_1280 | classifier→filter | 1,333 | 43 | 59 | 0.9688 | 0.9576 | 0.9632 |
| ir_model | ir_native | 696 | 37 | 20 | 0.9495 | 0.9721 | 0.9607 |
| ir_model | +ir_filter | 683 | 35 | 33 | 0.9513 | 0.9539 | 0.9526 |

## Temporal stages — segment-level (3-frame windows, 2-of-3)

Each row is one 3-frame segment scored as a single binary decision: fired ≥ 2 of 3 frames vs. any GT in the window.

| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 241 | 293 | 3 | 62 | 0.4513 | 0.9877 | 0.6195 | 89.15% |
| baseline | temporal+alert_gate | 238 | 139 | 6 | 216 | 0.6313 | 0.9754 | 0.7665 | 62.94% |
| retrained_v2 | temporal | 171 | 101 | 73 | 254 | 0.6287 | 0.7008 | 0.6628 | 45.41% |
| retrained_v2 | temporal+alert_gate | 170 | 61 | 74 | 294 | 0.7359 | 0.6967 | 0.7158 | 38.56% |
| selcom_1280 | temporal | 236 | 293 | 8 | 62 | 0.4461 | 0.9672 | 0.6106 | 88.31% |
| selcom_1280 | temporal+alert_gate | 229 | 95 | 15 | 260 | 0.7068 | 0.9385 | 0.8063 | 54.09% |
| ir_model | temporal | 242 | 4 | 0 | 353 | 0.9837 | 1.0000 | 0.9918 | 41.07% |
| ir_model | temporal+alert_gate | 242 | 3 | 0 | 354 | 0.9878 | 1.0000 | 0.9938 | 40.90% |

## Sanity flags

⚠️  `retrained_v2`: classifier R=0.9210 below max(R_rgb=0.5806, R_ir=0.9721)

## Per-size breakdown

### baseline (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 36 | 31 | 1,592 | 5 | 0.0191 | 0.8611 | 0.0374 |
| rgb | medium | 680 | 663 | 363 | 17 | 0.6462 | 0.9750 | 0.7773 |
| rgb | large | 16 | 15 | 41 | 1 | 0.2679 | 0.9375 | 0.4167 |
| rgb | all | 732 | 709 | 1,996 | 23 | 0.2621 | 0.9686 | 0.4126 |
| +rgb_filter | small | 36 | 29 | 917 | 7 | 0.0307 | 0.8056 | 0.0591 |
| +rgb_filter | medium | 680 | 639 | 154 | 41 | 0.8058 | 0.9397 | 0.8676 |
| +rgb_filter | large | 16 | 14 | 12 | 2 | 0.5385 | 0.8750 | 0.6667 |
| +rgb_filter | all | 732 | 682 | 1,083 | 50 | 0.3864 | 0.9317 | 0.5463 |
| classifier | small | 36 | 32 | 239 | 5 | 0.1181 | 0.8649 | 0.2078 |
| classifier | medium | 680 | 1,340 | 41 | 16 | 0.9703 | 0.9882 | 0.9792 |
| classifier | large | 16 | 31 | 1 | 0 | 0.9688 | 1.0000 | 0.9841 |
| classifier | all | 732 | 1,403 | 281 | 21 | 0.8331 | 0.9853 | 0.9028 |
| classifier→filter | small | 36 | 30 | 180 | 7 | 0.1429 | 0.8108 | 0.2429 |
| classifier→filter | medium | 680 | 1,303 | 39 | 53 | 0.9709 | 0.9609 | 0.9659 |
| classifier→filter | large | 16 | 30 | 1 | 1 | 0.9677 | 0.9677 | 0.9677 |
| classifier→filter | all | 732 | 1,363 | 220 | 61 | 0.8610 | 0.9572 | 0.9066 |
| temporal | all | 244 | 241 | 293 | 3 | 0.4513 | 0.9877 | 0.6195 |
| temporal+alert_gate | all | 244 | 238 | 139 | 6 | 0.6313 | 0.9754 | 0.7665 |

### retrained_v2 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 36 | 16 | 643 | 20 | 0.0243 | 0.4444 | 0.0460 |
| rgb | medium | 680 | 393 | 295 | 287 | 0.5712 | 0.5779 | 0.5746 |
| rgb | large | 16 | 16 | 511 | 0 | 0.0304 | 1.0000 | 0.0589 |
| rgb | all | 732 | 425 | 1,449 | 307 | 0.2268 | 0.5806 | 0.3262 |
| +rgb_filter | small | 36 | 16 | 485 | 20 | 0.0319 | 0.4444 | 0.0596 |
| +rgb_filter | medium | 680 | 389 | 223 | 291 | 0.6356 | 0.5721 | 0.6022 |
| +rgb_filter | large | 16 | 14 | 421 | 2 | 0.0322 | 0.8750 | 0.0621 |
| +rgb_filter | all | 732 | 419 | 1,129 | 313 | 0.2707 | 0.5724 | 0.3675 |
| classifier | small | 36 | 17 | 297 | 16 | 0.0541 | 0.5152 | 0.0980 |
| classifier | medium | 680 | 1,070 | 208 | 79 | 0.8372 | 0.9312 | 0.8817 |
| classifier | large | 16 | 32 | 260 | 1 | 0.1096 | 0.9697 | 0.1969 |
| classifier | all | 732 | 1,119 | 765 | 96 | 0.5939 | 0.9210 | 0.7222 |
| classifier→filter | small | 36 | 17 | 263 | 16 | 0.0607 | 0.5152 | 0.1086 |
| classifier→filter | medium | 680 | 1,053 | 205 | 96 | 0.8370 | 0.9164 | 0.8749 |
| classifier→filter | large | 16 | 30 | 260 | 3 | 0.1034 | 0.9091 | 0.1858 |
| classifier→filter | all | 732 | 1,100 | 728 | 115 | 0.6018 | 0.9053 | 0.7230 |
| temporal | all | 244 | 171 | 101 | 73 | 0.6287 | 0.7008 | 0.6628 |
| temporal+alert_gate | all | 244 | 170 | 61 | 74 | 0.7359 | 0.6967 | 0.7158 |

### selcom_1280 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 36 | 24 | 555 | 12 | 0.0415 | 0.6667 | 0.0780 |
| rgb | medium | 680 | 640 | 263 | 40 | 0.7087 | 0.9412 | 0.8086 |
| rgb | large | 16 | 16 | 183 | 0 | 0.0804 | 1.0000 | 0.1488 |
| rgb | all | 732 | 680 | 1,001 | 52 | 0.4045 | 0.9290 | 0.5636 |
| +rgb_filter | small | 36 | 24 | 254 | 12 | 0.0863 | 0.6667 | 0.1529 |
| +rgb_filter | medium | 680 | 613 | 97 | 67 | 0.8634 | 0.9015 | 0.8820 |
| +rgb_filter | large | 16 | 15 | 37 | 1 | 0.2885 | 0.9375 | 0.4412 |
| +rgb_filter | all | 732 | 652 | 388 | 80 | 0.6269 | 0.8907 | 0.7359 |
| classifier | small | 36 | 25 | 22 | 1 | 0.5319 | 0.9615 | 0.6849 |
| classifier | medium | 680 | 1,317 | 23 | 17 | 0.9828 | 0.9873 | 0.9850 |
| classifier | large | 16 | 32 | 1 | 0 | 0.9697 | 1.0000 | 0.9846 |
| classifier | all | 732 | 1,374 | 46 | 18 | 0.9676 | 0.9871 | 0.9772 |
| classifier→filter | small | 36 | 25 | 20 | 1 | 0.5556 | 0.9615 | 0.7042 |
| classifier→filter | medium | 680 | 1,277 | 22 | 57 | 0.9831 | 0.9573 | 0.9700 |
| classifier→filter | large | 16 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9688 |
| classifier→filter | all | 732 | 1,333 | 43 | 59 | 0.9688 | 0.9576 | 0.9632 |
| temporal | all | 244 | 236 | 293 | 8 | 0.4461 | 0.9672 | 0.6106 |
| temporal+alert_gate | all | 244 | 229 | 95 | 15 | 0.7068 | 0.9385 | 0.8063 |

### ir_model (ir_native)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ir_native | small | 2 | 1 | 0 | 1 | 1.0000 | 0.5000 | 0.6667 |
| ir_native | medium | 697 | 679 | 36 | 18 | 0.9497 | 0.9742 | 0.9618 |
| ir_native | large | 17 | 16 | 1 | 1 | 0.9412 | 0.9412 | 0.9412 |
| ir_native | all | 716 | 696 | 37 | 20 | 0.9495 | 0.9721 | 0.9607 |
| +ir_filter | small | 2 | 1 | 0 | 1 | 1.0000 | 0.5000 | 0.6667 |
| +ir_filter | medium | 697 | 666 | 34 | 31 | 0.9514 | 0.9555 | 0.9535 |
| +ir_filter | large | 17 | 16 | 1 | 1 | 0.9412 | 0.9412 | 0.9412 |
| +ir_filter | all | 716 | 683 | 35 | 33 | 0.9513 | 0.9539 | 0.9526 |
| temporal | all | 242 | 242 | 4 | 0 | 0.9837 | 1.0000 | 0.9918 |
| temporal+alert_gate | all | 242 | 242 | 3 | 0 | 0.9878 | 1.0000 | 0.9938 |

## Per-stage commentary

- **rgb** — Baseline RGB detector. Saturated by confusers — most FPs are birds. The R looks healthy (>0.9) but P is destroyed (<0.3).
- **ir_native** — IR detector on the paired IR frame. The most useful single signal on this dataset (F1≈0.95).
- **ir_grayscale** — IR weights on grayscale-RGB. Shown for symmetry with the RGB-only doc; not used in production on paired data.
- **+rgb_filter** — Patch verifier on RGB dets only. Catches some bird FPs but doesn't reach IR's confuser robustness.
- **+ir_filter** — Patch verifier on IR dets. Marginal effect since IR rarely hallucinates here.
- **classifier** — sa32 trust-aware: for each frame, classifier picks which modality (or both) to credit. The headline number — this is where modality arbitration shows up.
- **classifier→filter** — Classifier picks, then filter applied to the trusted side. So + filters out the residual after arbitration.
- **temporal** — 3-frame segments, 2-of-3 voting on the raw detector firing pattern. Caps the per-segment FR%.
- **temporal+alert_gate** — Production rule. The patch verifier runs only on the 3rd frame, gate-keeping the alert. So + temporal voting + confuser-veto on the decisive frame.

## Delivered

- `docs/analysis/full_pipeline_ablations/svanstrom.md`
- `docs/analysis/full_pipeline_ablations/csv/svanstrom.csv`
