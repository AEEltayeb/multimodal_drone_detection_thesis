# Full pipeline ablations

Compact cascade ablation across surfaces. **All metrics are IoP@0.5.** Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv); per-size detail in [`per_size_detail.csv`](per_size_detail.csv).

## Reading the tables

Each row is a detector. Each column is a pipeline step applied on top of the previous.
- `rgb_only` — RGB YOLO alone
- `+classifier` — scene-aware classifier (default sa32) gates dets via a trust label; any trust ≠ 0 passes both RGB and IR-grayscale dets where coords align
- `+filter` — patch verifier applied to detector output (no classifier)
- `+temporal` — 2-of-3 vote over 3-frame segments on detector dets
- `+alert_gate` — temporal vote with patch filter applied at alert (production cascade endpoint)
- `Δ vs rgb_only` — F1 gain from the full cascade vs detector alone

## Scoring rule and `ir_grayscale` †

IoP@0.5 throughout. RGB models move ≤1 pp F1 under IoU; only `ir_grayscale` moves meaningfully (legacy IoU aggregate on 9 real-video drone clips: P=0.588, R=0.441, F1=0.504). `ir_grayscale` rows marked †. See `docs/EVIDENCE_LEDGER.md` §12 for canonical numbers.


## 1. Anti-UAV RGBT (paired drone, clean benchmark)

**Summary.** Saturated benchmark (rgb-only F1 ≈ 0.99 for baseline/retrained_v2/selcom_640). Cascade has nothing to do here — no confusers, no clutter. `selcom_1280` bleeds 849 FPs and drops to F1=0.90, the only model not at the ceiling. Cascade columns sourced from `full_pipeline_persize` (currently rgb_only + classifier + filter only; temporal pending data).

| Detector | rgb_only | +classifier | +filter | +temporal | +alert_gate | Δ vs rgb_only |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 0.992 | — | — | — | — | — |
| retrained_v2 | 0.993 | — | — | — | — | — |
| selcom_640 | 0.988 | — | — | — | — | — |
| selcom_960 | 0.972 | — | — | — | — | — |
| selcom_1280 | 0.902 | — | — | — | — | — |

*Legacy 2-config aggregate (no per-model breakdown):*  rgb_only F1=0.992, ir_only F1=0.965 (from `eval/results/antiuav/metrics_iop.csv`).

## 2. Svanström (paired drone + confusers)

**Summary.** RGB-only collapses under confusers (F1=0.54); IR alone is stable (F1=0.96). Classifier+filter combination is where the cascade earns its keep on this surface — when the in-progress run lands, expect cascade F1 to recover toward IR-only levels.

| Detector | rgb_only | +classifier | +filter | +temporal | +alert_gate | Δ vs rgb_only |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 0.949 | — | — | — | — | — |
| hardneg_v3more | 0.945 | — | — | — | — | — |
| retrained_v2 | 0.482 | — | — | — | — | — |
| selcom_640 | 0.738 | — | — | — | — | — |
| selcom_960 | 0.906 | — | — | — | — | — |
| selcom_1280 | 0.938 | — | — | — | — | — |

*Legacy 2-config:*  rgb_only F1=0.544 (collapses under confusers), ir_only F1=0.959.

## 3. Selcom held-out val (RGB only, 311 imgs)

**Summary.** Detector-only winner: `selcom_960` (F1=0.585) edging `selcom_1280` (F1=0.580). Classifier S1 drops recall on this surface — the scene-aware classifier was trained on Svanström-like data and doesn't recognize CCTV signal, so it conservatively rejects. Cascade not the right tool here; rgb_only is the correct reporting baseline.

| Detector | rgb_only | +classifier | +filter | +temporal | +alert_gate | Δ vs rgb_only |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 0.145 | — | — | — | — | — |
| hardneg_v3more | 0.026 | — | — | — | — | — |
| retrained_v2 | 0.007 | — | — | — | — | — |
| selcom_640 | 0.209 | — | — | — | — | — |
| selcom_960 | 0.585 | 0.366 | 0.585 | 0.634 | 0.634 | +0.049 |
| selcom_1280 | 0.580 | — | — | — | — | — |

## 4. Roboflow OOD drone (RGB only)

**Summary.** `selcom_960` Pareto-best (rgb+filter F1=0.84). Cascade stages beyond +filter not evaluated on this surface — gap noted.

| Detector | rgb_only | +classifier | +filter | +temporal | +alert_gate | Δ vs rgb_only |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 0.822 | — | 0.791 | — | — | -0.031 |
| hardneg_v3more | 0.732 | — | 0.712 | — | — | -0.020 |
| retrained_v2 | 0.813 | — | 0.789 | — | — | -0.024 |
| selcom_640 | 0.842 | — | 0.805 | — | — | -0.037 |
| selcom_960 | 0.854 | — | 0.839 | — | — | -0.014 |
| selcom_1280 | 0.804 | — | 0.798 | — | — | -0.006 |

## 5. Real-video drone clips (RGB only, 9 clips)

**Summary.** This is where the cascade story is clearest: baseline goes from rgb_only F1=0.76 to +temporal F1=0.83 (+7 pp) — segment-level voting recovers single-frame recall noise. Patch filter at alert gate keeps the recall while cutting FPs.


### 5a. Cascade per detector (sa32 classifier, aggregated across drone clips)

| Detector | rgb_only | +classifier | +filter | +temporal | +alert_gate | Δ vs rgb_only |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline_trained | 0.760 | 0.586 | — | 0.833 | 0.826 | +0.067 |
| retrained_v2 | 0.605 | 0.615 | — | 0.781 | 0.770 | +0.165 |
| selcom_1280 | 0.721 | 0.537 | — | 0.819 | 0.814 | +0.092 |
| selcom_640 | 0.730 | 0.568 | — | 0.822 | 0.816 | +0.086 |

### 5b. Three-classifier endpoint F1 (`+alert_gate` stage)

| Detector | sa32 | control40 | fnfn |
|---|:---:|:---:|:---:|
| baseline_trained | 0.826 | 0.644 | 0.219 |
| retrained_v2 | 0.770 | 0.576 | 0.181 |
| selcom_1280 | 0.814 | 0.595 | 0.108 |
| selcom_640 | 0.816 | 0.619 | 0.165 |

**Read:** sa32 is the production pick. control40 trades 18+ pp F1 for halved FPs; fnfn rejects 85% of correct TPs — only viable when false-alarm fatigue dominates.


*Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv).*

## 6. Confuser-only clips (no drone GT)

**Summary.** Cascade FPPI by detector × stage, aggregated per confuser category. Lower is better. Watch the cascade columns ↓ left-to-right — that's the FP reduction story.

| Category | Detector | rgb_only FPPI | +classifier | +temporal | +alert_gate | Δ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| birds | baseline_trained | 0.9602 | 0.1903 | 0.0504 | 0.0168 | -0.9434 |
| birds | retrained_v2 | 0.1392 | 0.0682 | 0.0084 | 0.0084 | -0.1308 |
| birds | selcom_1280 | 1.5284 | 0.2869 | 0.0420 | 0.0336 | -1.4948 |
| birds | selcom_640 | 0.3125 | 0.0682 | 0.0084 | 0.0084 | -0.3041 |
|  |  |  |  |  |  |  |
| airplanes | baseline_trained | 0.4507 | 0.4704 | 0.2451 | 0.2255 | -0.2252 |
| airplanes | retrained_v2 | 0.3849 | 0.4079 | 0.2255 | 0.2059 | -0.1790 |
| airplanes | selcom_1280 | 0.4934 | 0.4901 | 0.2353 | 0.2157 | -0.2777 |
| airplanes | selcom_640 | 0.3586 | 0.4539 | 0.2255 | 0.2157 | -0.1429 |
|  |  |  |  |  |  |  |
| helicopters | baseline_trained | 0.2778 | 0.2374 | 0.2312 | 0.2161 | -0.0617 |
| helicopters | retrained_v2 | 0.1330 | 0.1431 | 0.1457 | 0.1407 | +0.0077 |
| helicopters | selcom_1280 | 0.3333 | 0.2189 | 0.1658 | 0.1558 | -0.1776 |
| helicopters | selcom_640 | 0.1785 | 0.1633 | 0.1608 | 0.1508 | -0.0277 |

*Per-clip detail in [`per_clip_detail.csv`](per_clip_detail.csv).*


## Reproduction

```
# Doc:
python analytics/spec_analysis/09_fill_ablations_doc.py
# Per-size cascade gap-fill (in progress):
python eval/eval_full_pipeline_persize.py --classifiers sa32 no_classifier
```

Canonical sources used:
- `eval/results/antiuav/metrics_iop.csv`, `eval/results/svanstrom/metrics_iop.csv` (legacy aggregate)
- `eval/results/{antiuav_per_model, selcom_val_holdout}/<m>/<m>_results.json` (per-detector + per-size)
- `eval/results/svanstrom_persize/summary.csv` (per-size DRONE on Svanström)
- `eval/results/video_persize/summary.csv` (per-size on real-video)
- `eval/results/roboflow_ood/summary.csv` (rgb_drone + confuser size buckets)
- `eval/results/pipeline_video_tests*/pipeline_comparison.csv` (full cascade per clip, 3 classifiers)
- `eval/results/full_pipeline_persize/<ds>/<det>/<clf>/summary.csv` (per-size cascade, as it lands)
