# ft4 Lean Trust Classifier — Re-mine + Feature-Set Comparison

**Date:** 2026-05-31 · **Scripts:** `classifier/build_video_ft4_cache.py`, `classifier/generate_lean19_data.py`, `classifier/train_lean_ft4.py` · **Data:** `classifier/fusion_models/lean_ft4/fusion_dataset_lean19.csv` (8,871 rows, mined from **ft4 RGB + v3b IR** — the current production stack).

## Why
The production trust classifier `sa32` was mined from the **old** (v3more/selcom_1280) detector — input drift vs current ft4 — and the [[fusion-feature-leakage]] screen flagged scene-statistic features as fingerprints. This re-mines the fusion features from ft4+v3b (cheap features only — no expensive scene scalars) and tests, **on the current detector**, whether dropping the fingerprint features helps.

## Re-mine
- ft4 RGB video caches built for 19 clips (`build_video_ft4_cache.py` → `*_ft4_sz1280.json`); v3b IR-grayscale caches reused.
- `generate_lean19_data.py --rgb-weights <ft4> --ir-weights <v3b> --auv-stride 25 --svan-stride 10 --video-rgb-cache-tag ft4_sz1280` → 8,871 rows (antiuav 3391, svan 2871, 19 video clips). Labels: trust_both 4424 / reject 3373 / trust_rgb 700 / trust_ir 374.
- Only the 19 cheap lean features mined (confidences, box geometry, position, brightness) — the expensive scene scalars (entropy/edge/blur/sky/dynamic-range) deliberately skipped (they are the fingerprints and cost a full image read each).

## Result (GroupShuffleSplit test, F1-macro)
| variant | feats | overall | antiuav | svan | **video_drone (OOD)** | video_confuser (OOD) |
|---|---|---|---|---|---|---|
| all19 (incl. fingerprints) | 19 | 0.787 | 0.871 | 0.737 | **0.262** | 1.000 |
| no_fp (drop brightness+abs-pos) | 10 | 0.787 | 0.844 | 0.736 | 0.436 | 1.000 |
| **robust6** (conf + box geometry) | 6 | **0.810** | 0.827 | 0.725 | **0.578** | 1.000 |
| meta4 (conf + rgb geometry) | 4 | 0.787 | 0.769 | 0.716 | 0.569 | 0.222 |
| sa32 (ref, NOT comparable*) | 32 | 0.949 | — | — | — | — |

\* sa32 ref = old detector + 10 extra scene features + its own in-domain split; not an apples-to-apples number.

## Reads
1. **Leakage thesis confirmed on ft4.** Dropping the scene-fingerprint features does not hurt overall and **wins**: robust6 (6 feats) beats all19 (19 feats) on macro-F1.
2. **The OOD-drone-video column is the headline.** all19 (with brightness+position fingerprints) **collapses to 0.262**; removing fingerprints climbs it 0.262 → 0.436 → **0.578**. The fat feature set memorizes in-domain scenes and fails on unseen video — exactly the lean13/lean17 failure, now demonstrated quantitatively on the current detector.
3. **robust6 is the pick:** `rgb_max_conf, ir_max_conf, rgb_best_log_bbox_area, ir_best_log_bbox_area, rgb_best_aspect_ratio, ir_best_aspect_ratio`. Best overall, perfect confuser rejection, best OOD-drone, for ~4pp in-domain antiuav cost. Trade is favourable for deployment.
4. **meta4 is too lean** — drops IR geometry and loses confuser rejection (video_confuser 0.222). The IR box geometry is load-bearing.

## Caveats (recorded)
- `video_confuser(OOD)=1.000` surfaces are near-single-class (reject-only) → reads as "never falsely trusts a confuser," not a balanced-F1 triumph.
- This validates the **feature set** on ft4; the full-pipeline impact (classifier in-line with V5 verifier + temporal) is the next eval.
- Anti-UAV at stride 25 / Svanström stride 10 — a screen sample, not the full set.

## Delivered
- `C:\Users\User\...\ES_Drone_Detection\docs\analysis\2026-05-31_ft4_lean_trust_classifier.md` (this doc)
- `C:\Users\User\...\ES_Drone_Detection\classifier\train_lean_ft4.py`
- `C:\Users\User\...\ES_Drone_Detection\classifier\build_video_ft4_cache.py`
- `classifier/fusion_models/lean_ft4/` — `fusion_dataset_lean19.csv`, `trust_ft4_robust6.joblib`, `lean_ft4_compare.json`, `cache_{antiuav,svanstrom}.json`
- `docs/analysis/full_pipeline_ablations/cache/video_*_ft4_sz1280.json` (19 ft4 video RGB caches)
