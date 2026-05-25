# Selcom CCTV val (drone-only RGB) — Full Pipeline Ablations

- **Category:** RGB-only, drone-only (CCTV crops)
- **Scoring:** IOU @ 0.5
- **Frames evaluated (per detector, post-stride):** ~311

Held-out drone-only CCTV val split. RGB-only, so the IR side comes from ir_grayscale (cross-modal fallback). No confusers in this split, so the patch verifier and classifier have nothing to arbitrate — the interesting question here is whether the soft-veto / classifier harms RGB recall in the no-confuser case.

## Overall summary (all sizes) — bbox-level

Detection-on-drone scoring. Any detection that does not match a drone GT box (IoU/IoP ≥ 0.5) is an FP — confuser hallucinations (birds, planes, helis sharing the frame) are already counted here.

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | rgb | 26 | 37 | 269 | 0.4127 | 0.0881 | 0.1453 |
| baseline | +rgb_filter | 26 | 37 | 269 | 0.4127 | 0.0881 | 0.1453 |
| baseline | classifier | 5 | 34 | 290 | 0.1282 | 0.0169 | 0.0299 |
| baseline | classifier→filter | 5 | 34 | 290 | 0.1282 | 0.0169 | 0.0299 |
| retrained_v2 | rgb | 1 | 3 | 294 | 0.2500 | 0.0034 | 0.0067 |
| retrained_v2 | +rgb_filter | 1 | 3 | 294 | 0.2500 | 0.0034 | 0.0067 |
| retrained_v2 | classifier | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| retrained_v2 | classifier→filter | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| selcom_1280 | rgb | 137 | 44 | 158 | 0.7569 | 0.4644 | 0.5756 |
| selcom_1280 | +rgb_filter | 137 | 44 | 158 | 0.7569 | 0.4644 | 0.5756 |
| selcom_1280 | classifier | 66 | 29 | 229 | 0.6947 | 0.2237 | 0.3385 |
| selcom_1280 | classifier→filter | 66 | 29 | 229 | 0.6947 | 0.2237 | 0.3385 |
| selcom_960 | rgb | 129 | 17 | 166 | 0.8836 | 0.4373 | 0.5850 |
| selcom_960 | +rgb_filter | 129 | 17 | 166 | 0.8836 | 0.4373 | 0.5850 |
| selcom_960 | classifier | 68 | 9 | 227 | 0.8831 | 0.2305 | 0.3656 |
| selcom_960 | classifier→filter | 68 | 9 | 227 | 0.8831 | 0.2305 | 0.3656 |
| ir_grayscale | ir_grayscale | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| ir_grayscale | +rgb_filter | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |

## Temporal stages — segment-level (3-frame windows, 2-of-3)

Each row is one 3-frame segment scored as a single binary decision: fired ≥ 2 of 3 frames vs. any GT in the window.

| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 17 | 5 | 82 | 0 | 0.7727 | 0.1717 | 0.2810 | 21.15% |
| baseline | temporal+alert_gate | 17 | 5 | 82 | 0 | 0.7727 | 0.1717 | 0.2810 | 21.15% |
| retrained_v2 | temporal | 0 | 0 | 99 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| retrained_v2 | temporal+alert_gate | 0 | 0 | 99 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| selcom_1280 | temporal | 51 | 0 | 48 | 5 | 1.0000 | 0.5152 | 0.6800 | 49.04% |
| selcom_1280 | temporal+alert_gate | 51 | 0 | 48 | 5 | 1.0000 | 0.5152 | 0.6800 | 49.04% |
| selcom_960 | temporal | 46 | 0 | 53 | 5 | 1.0000 | 0.4646 | 0.6345 | 44.23% |
| selcom_960 | temporal+alert_gate | 46 | 0 | 53 | 5 | 1.0000 | 0.4646 | 0.6345 | 44.23% |
| ir_grayscale | temporal | 0 | 0 | 99 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.00% |
| ir_grayscale | temporal+alert_gate | 0 | 0 | 99 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.00% |

## Sanity flags

⚠️  `baseline`: classifier R=0.0169 below max(R_rgb=0.0881, R_ir=0.0000)
⚠️  `retrained_v2`: classifier R=0.0000 below max(R_rgb=0.0034, R_ir=0.0000)
⚠️  `selcom_1280`: classifier R=0.2237 below max(R_rgb=0.4644, R_ir=0.0000)
⚠️  `selcom_960`: classifier R=0.2305 below max(R_rgb=0.4373, R_ir=0.0000)

## Per-size breakdown

### baseline (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 166 | 12 | 5 | 154 | 0.7059 | 0.0723 | 0.1311 |
| rgb | medium | 125 | 14 | 32 | 111 | 0.3043 | 0.1120 | 0.1637 |
| rgb | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| rgb | all | 295 | 26 | 37 | 269 | 0.4127 | 0.0881 | 0.1453 |
| +rgb_filter | small | 166 | 12 | 5 | 154 | 0.7059 | 0.0723 | 0.1311 |
| +rgb_filter | medium | 125 | 14 | 32 | 111 | 0.3043 | 0.1120 | 0.1637 |
| +rgb_filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | all | 295 | 26 | 37 | 269 | 0.4127 | 0.0881 | 0.1453 |
| classifier | small | 166 | 4 | 3 | 162 | 0.5714 | 0.0241 | 0.0462 |
| classifier | medium | 125 | 1 | 31 | 124 | 0.0312 | 0.0080 | 0.0127 |
| classifier | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier | all | 295 | 5 | 34 | 290 | 0.1282 | 0.0169 | 0.0299 |
| classifier→filter | small | 166 | 4 | 3 | 162 | 0.5714 | 0.0241 | 0.0462 |
| classifier→filter | medium | 125 | 1 | 31 | 124 | 0.0312 | 0.0080 | 0.0127 |
| classifier→filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | all | 295 | 5 | 34 | 290 | 0.1282 | 0.0169 | 0.0299 |
| temporal | all | 99 | 17 | 5 | 82 | 0.7727 | 0.1717 | 0.2810 |
| temporal+alert_gate | all | 99 | 17 | 5 | 82 | 0.7727 | 0.1717 | 0.2810 |

### retrained_v2 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 166 | 0 | 1 | 166 | 0.0000 | 0.0000 | 0.0000 |
| rgb | medium | 125 | 1 | 2 | 124 | 0.3333 | 0.0080 | 0.0156 |
| rgb | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| rgb | all | 295 | 1 | 3 | 294 | 0.2500 | 0.0034 | 0.0067 |
| +rgb_filter | small | 166 | 0 | 1 | 166 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | medium | 125 | 1 | 2 | 124 | 0.3333 | 0.0080 | 0.0156 |
| +rgb_filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | all | 295 | 1 | 3 | 294 | 0.2500 | 0.0034 | 0.0067 |
| classifier | small | 166 | 0 | 1 | 166 | 0.0000 | 0.0000 | 0.0000 |
| classifier | medium | 125 | 0 | 1 | 125 | 0.0000 | 0.0000 | 0.0000 |
| classifier | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier | all | 295 | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | small | 166 | 0 | 1 | 166 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | medium | 125 | 0 | 1 | 125 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | all | 295 | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| temporal | all | 99 | 0 | 0 | 99 | 0.0000 | 0.0000 | 0.0000 |
| temporal+alert_gate | all | 99 | 0 | 0 | 99 | 0.0000 | 0.0000 | 0.0000 |

### selcom_1280 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 166 | 69 | 22 | 97 | 0.7582 | 0.4157 | 0.5370 |
| rgb | medium | 125 | 68 | 22 | 57 | 0.7556 | 0.5440 | 0.6326 |
| rgb | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| rgb | all | 295 | 137 | 44 | 158 | 0.7569 | 0.4644 | 0.5756 |
| +rgb_filter | small | 166 | 69 | 22 | 97 | 0.7582 | 0.4157 | 0.5370 |
| +rgb_filter | medium | 125 | 68 | 22 | 57 | 0.7556 | 0.5440 | 0.6326 |
| +rgb_filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | all | 295 | 137 | 44 | 158 | 0.7569 | 0.4644 | 0.5756 |
| classifier | small | 166 | 30 | 15 | 136 | 0.6667 | 0.1807 | 0.2844 |
| classifier | medium | 125 | 36 | 14 | 89 | 0.7200 | 0.2880 | 0.4114 |
| classifier | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier | all | 295 | 66 | 29 | 229 | 0.6947 | 0.2237 | 0.3385 |
| classifier→filter | small | 166 | 30 | 15 | 136 | 0.6667 | 0.1807 | 0.2844 |
| classifier→filter | medium | 125 | 36 | 14 | 89 | 0.7200 | 0.2880 | 0.4114 |
| classifier→filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | all | 295 | 66 | 29 | 229 | 0.6947 | 0.2237 | 0.3385 |
| temporal | all | 99 | 51 | 0 | 48 | 1.0000 | 0.5152 | 0.6800 |
| temporal+alert_gate | all | 99 | 51 | 0 | 48 | 1.0000 | 0.5152 | 0.6800 |

### selcom_960 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 166 | 50 | 6 | 116 | 0.8929 | 0.3012 | 0.4505 |
| rgb | medium | 125 | 76 | 11 | 49 | 0.8736 | 0.6080 | 0.7170 |
| rgb | large | 4 | 3 | 0 | 1 | 1.0000 | 0.7500 | 0.8571 |
| rgb | all | 295 | 129 | 17 | 166 | 0.8836 | 0.4373 | 0.5850 |
| +rgb_filter | small | 166 | 50 | 6 | 116 | 0.8929 | 0.3012 | 0.4505 |
| +rgb_filter | medium | 125 | 76 | 11 | 49 | 0.8736 | 0.6080 | 0.7170 |
| +rgb_filter | large | 4 | 3 | 0 | 1 | 1.0000 | 0.7500 | 0.8571 |
| +rgb_filter | all | 295 | 129 | 17 | 166 | 0.8836 | 0.4373 | 0.5850 |
| classifier | small | 166 | 26 | 3 | 140 | 0.8966 | 0.1566 | 0.2667 |
| classifier | medium | 125 | 42 | 6 | 83 | 0.8750 | 0.3360 | 0.4855 |
| classifier | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier | all | 295 | 68 | 9 | 227 | 0.8831 | 0.2305 | 0.3656 |
| classifier→filter | small | 166 | 26 | 3 | 140 | 0.8966 | 0.1566 | 0.2667 |
| classifier→filter | medium | 125 | 42 | 6 | 83 | 0.8750 | 0.3360 | 0.4855 |
| classifier→filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| classifier→filter | all | 295 | 68 | 9 | 227 | 0.8831 | 0.2305 | 0.3656 |
| temporal | all | 99 | 46 | 0 | 53 | 1.0000 | 0.4646 | 0.6345 |
| temporal+alert_gate | all | 99 | 46 | 0 | 53 | 1.0000 | 0.4646 | 0.6345 |

### ir_grayscale (ir_grayscale)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ir_grayscale | small | 166 | 0 | 2 | 166 | 0.0000 | 0.0000 | 0.0000 |
| ir_grayscale | medium | 125 | 0 | 0 | 125 | 0.0000 | 0.0000 | 0.0000 |
| ir_grayscale | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| ir_grayscale | all | 295 | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | small | 166 | 0 | 2 | 166 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | medium | 125 | 0 | 0 | 125 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | large | 4 | 0 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 |
| +rgb_filter | all | 295 | 0 | 2 | 295 | 0.0000 | 0.0000 | 0.0000 |
| temporal | all | 99 | 0 | 0 | 99 | 0.0000 | 0.0000 | 0.0000 |
| temporal+alert_gate | all | 99 | 0 | 0 | 99 | 0.0000 | 0.0000 | 0.0000 |

## Per-stage commentary

- **rgb** — Baseline RGB on the val split. Reference number.
- **ir_grayscale** — IR weights on the grayscale-RGB input — cross-modal fallback path. Low recall is expected (IR weights weren't trained on RGB-derived grayscale).
- **+rgb_filter** — Patch verifier on RGB dets. On a confuser-free dataset, this is a pure recall tax.
- **classifier** — sa32 trust-aware in grayscale mode (IR side = ir_grayscale).
- **classifier→filter** — Classifier-trusted dets passed through the patch verifier.
- **temporal** — 3-frame segments. Tight clip framing in CCTV makes temporal voting easy.
- **temporal+alert_gate** — Production rule, same as elsewhere.

## Delivered

- `docs/analysis/full_pipeline_ablations/selcom_val.md`
- `docs/analysis/full_pipeline_ablations/csv/selcom_val.csv`
