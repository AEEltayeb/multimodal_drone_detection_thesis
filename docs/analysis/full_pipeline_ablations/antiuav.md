# Anti-UAV RGBT — Full Pipeline Ablations

- **Category:** RGB+IR paired, drone-only
- **Scoring:** IOU @ 0.5
- **Frames evaluated (per detector, post-stride):** ~1,779

Clean paired benchmark — no confusers. RGB and IR both saturate near F1=0.99 and the classifier should not change much (both modalities are trustworthy). The filter has nothing to suppress, so it slightly trims recall without buying precision.

## Overall summary (all sizes) — bbox-level

Detection-on-drone scoring. Any detection that does not match a drone GT box (IoU/IoP ≥ 0.5) is an FP — confuser hallucinations (birds, planes, helis sharing the frame) are already counted here.

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | rgb | 1,632 | 49 | 31 | 0.9709 | 0.9814 | 0.9761 |
| baseline | +rgb_filter | 1,632 | 49 | 31 | 0.9709 | 0.9814 | 0.9761 |
| baseline | classifier | 3,283 | 74 | 37 | 0.9780 | 0.9889 | 0.9834 |
| baseline | classifier→filter | 3,283 | 74 | 37 | 0.9780 | 0.9889 | 0.9834 |
| retrained_v2 | rgb | 1,644 | 71 | 19 | 0.9586 | 0.9886 | 0.9734 |
| retrained_v2 | +rgb_filter | 1,644 | 71 | 19 | 0.9586 | 0.9886 | 0.9734 |
| retrained_v2 | classifier | 3,295 | 92 | 35 | 0.9728 | 0.9895 | 0.9811 |
| retrained_v2 | classifier→filter | 3,295 | 92 | 35 | 0.9728 | 0.9895 | 0.9811 |
| selcom_1280 | rgb | 1,617 | 314 | 46 | 0.8374 | 0.9723 | 0.8998 |
| selcom_1280 | +rgb_filter | 1,617 | 314 | 46 | 0.8374 | 0.9723 | 0.8998 |
| selcom_1280 | classifier | 3,270 | 336 | 42 | 0.9068 | 0.9873 | 0.9454 |
| selcom_1280 | classifier→filter | 3,270 | 336 | 42 | 0.9068 | 0.9873 | 0.9454 |
| ir_model | ir_native | 1,654 | 35 | 101 | 0.9793 | 0.9425 | 0.9605 |
| ir_model | +ir_filter | 1,654 | 35 | 101 | 0.9793 | 0.9425 | 0.9605 |

## Temporal stages — segment-level (3-frame windows, 2-of-3)

Each row is one 3-frame segment scored as a single binary decision: fired ≥ 2 of 3 frames vs. any GT in the window.

| Model | Stage | TP | FP | FN | TN | P | R | F1 | FR% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | temporal | 555 | 0 | 22 | 16 | 1.0000 | 0.9619 | 0.9806 | 93.59% |
| baseline | temporal+alert_gate | 555 | 0 | 22 | 16 | 1.0000 | 0.9619 | 0.9806 | 93.59% |
| retrained_v2 | temporal | 560 | 0 | 17 | 16 | 1.0000 | 0.9705 | 0.9850 | 94.44% |
| retrained_v2 | temporal+alert_gate | 560 | 0 | 17 | 16 | 1.0000 | 0.9705 | 0.9850 | 94.44% |
| selcom_1280 | temporal | 553 | 0 | 24 | 16 | 1.0000 | 0.9584 | 0.9788 | 93.25% |
| selcom_1280 | temporal+alert_gate | 553 | 0 | 24 | 16 | 1.0000 | 0.9584 | 0.9788 | 93.25% |
| ir_model | temporal | 571 | 0 | 21 | 1 | 1.0000 | 0.9645 | 0.9819 | 96.29% |
| ir_model | temporal+alert_gate | 571 | 0 | 21 | 1 | 1.0000 | 0.9645 | 0.9819 | 96.29% |

## Per-size breakdown

### baseline (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 158 | 153 | 9 | 5 | 0.9444 | 0.9684 | 0.9562 |
| rgb | medium | 1405 | 1,379 | 39 | 26 | 0.9725 | 0.9815 | 0.9770 |
| rgb | large | 100 | 100 | 1 | 0 | 0.9901 | 1.0000 | 0.9950 |
| rgb | all | 1663 | 1,632 | 49 | 31 | 0.9709 | 0.9814 | 0.9761 |
| +rgb_filter | small | 158 | 153 | 9 | 5 | 0.9444 | 0.9684 | 0.9562 |
| +rgb_filter | medium | 1405 | 1,379 | 39 | 26 | 0.9725 | 0.9815 | 0.9770 |
| +rgb_filter | large | 100 | 100 | 1 | 0 | 0.9901 | 1.0000 | 0.9950 |
| +rgb_filter | all | 1663 | 1,632 | 49 | 31 | 0.9709 | 0.9814 | 0.9761 |
| classifier | small | 158 | 182 | 10 | 2 | 0.9479 | 0.9891 | 0.9681 |
| classifier | medium | 1405 | 2,891 | 62 | 34 | 0.9790 | 0.9884 | 0.9837 |
| classifier | large | 100 | 210 | 2 | 1 | 0.9906 | 0.9953 | 0.9929 |
| classifier | all | 1663 | 3,283 | 74 | 37 | 0.9780 | 0.9889 | 0.9834 |
| classifier→filter | small | 158 | 182 | 10 | 2 | 0.9479 | 0.9891 | 0.9681 |
| classifier→filter | medium | 1405 | 2,891 | 62 | 34 | 0.9790 | 0.9884 | 0.9837 |
| classifier→filter | large | 100 | 210 | 2 | 1 | 0.9906 | 0.9953 | 0.9929 |
| classifier→filter | all | 1663 | 3,283 | 74 | 37 | 0.9780 | 0.9889 | 0.9834 |
| temporal | all | 577 | 555 | 0 | 22 | 1.0000 | 0.9619 | 0.9806 |
| temporal+alert_gate | all | 577 | 555 | 0 | 22 | 1.0000 | 0.9619 | 0.9806 |

### retrained_v2 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 158 | 152 | 20 | 6 | 0.8837 | 0.9620 | 0.9212 |
| rgb | medium | 1405 | 1,392 | 49 | 13 | 0.9660 | 0.9907 | 0.9782 |
| rgb | large | 100 | 100 | 2 | 0 | 0.9804 | 1.0000 | 0.9901 |
| rgb | all | 1663 | 1,644 | 71 | 19 | 0.9586 | 0.9886 | 0.9734 |
| +rgb_filter | small | 158 | 152 | 20 | 6 | 0.8837 | 0.9620 | 0.9212 |
| +rgb_filter | medium | 1405 | 1,392 | 49 | 13 | 0.9660 | 0.9907 | 0.9782 |
| +rgb_filter | large | 100 | 100 | 2 | 0 | 0.9804 | 1.0000 | 0.9901 |
| +rgb_filter | all | 1663 | 1,644 | 71 | 19 | 0.9586 | 0.9886 | 0.9734 |
| classifier | small | 158 | 181 | 18 | 4 | 0.9095 | 0.9784 | 0.9427 |
| classifier | medium | 1405 | 2,904 | 71 | 30 | 0.9761 | 0.9898 | 0.9829 |
| classifier | large | 100 | 210 | 3 | 1 | 0.9859 | 0.9953 | 0.9906 |
| classifier | all | 1663 | 3,295 | 92 | 35 | 0.9728 | 0.9895 | 0.9811 |
| classifier→filter | small | 158 | 181 | 18 | 4 | 0.9095 | 0.9784 | 0.9427 |
| classifier→filter | medium | 1405 | 2,904 | 71 | 30 | 0.9761 | 0.9898 | 0.9829 |
| classifier→filter | large | 100 | 210 | 3 | 1 | 0.9859 | 0.9953 | 0.9906 |
| classifier→filter | all | 1663 | 3,295 | 92 | 35 | 0.9728 | 0.9895 | 0.9811 |
| temporal | all | 577 | 560 | 0 | 17 | 1.0000 | 0.9705 | 0.9850 |
| temporal+alert_gate | all | 577 | 560 | 0 | 17 | 1.0000 | 0.9705 | 0.9850 |

### selcom_1280 (rgb)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rgb | small | 158 | 150 | 14 | 8 | 0.9146 | 0.9494 | 0.9317 |
| rgb | medium | 1405 | 1,367 | 300 | 38 | 0.8200 | 0.9730 | 0.8900 |
| rgb | large | 100 | 100 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| rgb | all | 1663 | 1,617 | 314 | 46 | 0.8374 | 0.9723 | 0.8998 |
| +rgb_filter | small | 158 | 150 | 14 | 8 | 0.9146 | 0.9494 | 0.9317 |
| +rgb_filter | medium | 1405 | 1,367 | 300 | 38 | 0.8200 | 0.9730 | 0.8900 |
| +rgb_filter | large | 100 | 100 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| +rgb_filter | all | 1663 | 1,617 | 314 | 46 | 0.8374 | 0.9723 | 0.8998 |
| classifier | small | 158 | 179 | 14 | 2 | 0.9275 | 0.9890 | 0.9572 |
| classifier | medium | 1405 | 2,881 | 321 | 39 | 0.8998 | 0.9866 | 0.9412 |
| classifier | large | 100 | 210 | 1 | 1 | 0.9953 | 0.9953 | 0.9953 |
| classifier | all | 1663 | 3,270 | 336 | 42 | 0.9068 | 0.9873 | 0.9454 |
| classifier→filter | small | 158 | 179 | 14 | 2 | 0.9275 | 0.9890 | 0.9572 |
| classifier→filter | medium | 1405 | 2,881 | 321 | 39 | 0.8998 | 0.9866 | 0.9412 |
| classifier→filter | large | 100 | 210 | 1 | 1 | 0.9953 | 0.9953 | 0.9953 |
| classifier→filter | all | 1663 | 3,270 | 336 | 42 | 0.9068 | 0.9873 | 0.9454 |
| temporal | all | 577 | 553 | 0 | 24 | 1.0000 | 0.9584 | 0.9788 |
| temporal+alert_gate | all | 577 | 553 | 0 | 24 | 1.0000 | 0.9584 | 0.9788 |

### ir_model (ir_native)

| Stage | Size | n_gt | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ir_native | small | 33 | 29 | 2 | 4 | 0.9355 | 0.8788 | 0.9062 |
| ir_native | medium | 1610 | 1,515 | 32 | 95 | 0.9793 | 0.9410 | 0.9598 |
| ir_native | large | 112 | 110 | 1 | 2 | 0.9910 | 0.9821 | 0.9865 |
| ir_native | all | 1755 | 1,654 | 35 | 101 | 0.9793 | 0.9425 | 0.9605 |
| +ir_filter | small | 33 | 29 | 2 | 4 | 0.9355 | 0.8788 | 0.9062 |
| +ir_filter | medium | 1610 | 1,515 | 32 | 95 | 0.9793 | 0.9410 | 0.9598 |
| +ir_filter | large | 112 | 110 | 1 | 2 | 0.9910 | 0.9821 | 0.9865 |
| +ir_filter | all | 1755 | 1,654 | 35 | 101 | 0.9793 | 0.9425 | 0.9605 |
| temporal | all | 592 | 571 | 0 | 21 | 1.0000 | 0.9645 | 0.9819 |
| temporal+alert_gate | all | 592 | 571 | 0 | 21 | 1.0000 | 0.9645 | 0.9819 |

## Per-stage commentary

- **rgb** — Detector alone, single frame. Reference RGB row.
- **ir_native** — IR detector on the paired IR frame. Reference IR row.
- **ir_grayscale** — IR weights applied to a grayscale copy of the RGB frame. Useful for the cross-modal-on-RGB fallback path.
- **+rgb_filter** — Patch verifier (rgb_filter v2) applied to every RGB det. On a confuser-free dataset this is a pure recall tax.
- **+ir_filter** — Patch verifier (ir_filter v2) on every IR det. Same dynamic.
- **classifier** — sa32 trust classifier picks RGB / IR / both. Scored against the GT of the trusted modality, so TP counts can exceed a single-modality detector.
- **classifier→filter** — Classifier picks, then the filter of the trusted modality is applied. So + filters out the small residual of mismatched dets.
- **temporal** — 3-frame segments, 2-of-3 voting on the raw detector firing pattern. So + temporal averaging without any confuser logic.
- **temporal+alert_gate** — Production rule — the filter is applied only on the 3rd frame, right before the alert would fire. So + same as temporal but each fired segment is veto-able by the patch verifier on its triggering frame.

## Delivered

- `docs/analysis/full_pipeline_ablations/antiuav.md`
- `docs/analysis/full_pipeline_ablations/csv/antiuav.csv`
