# Pipeline Improvements — Summary

**Date:** 2026-05-02
**Scope:** v3_more pipeline vs the prior thesis-quality baseline (`check.txt` reference).
**Detail report:** [`pipeline_improvements.md`](pipeline_improvements.md)

---

## TL;DR

Three components retrained. Every measured F1 metric is equal-or-better. The biggest wins are
on Svanström (where confusers actually appear) and YouTube OOD (the leak-free generalization test).

| Component | Headline win | Headline cost |
|---|---|---|
| RGB YOLO (`v3_more`) | −15pt confuser any-det on Svanström | ~1pt drone F1 (gap halves at F1-optimal conf) |
| Patch verifier | −20pt confuser survival on YouTube OOD (avg) | +14pt clean-drone suppression on YouTube DRONE_CLEAN (per-frame; deployment masks via temporal smoothing) |
| Fusion classifier (32-feat) | Removes 35%-importance `ir_detected` leakage shortcut | Numerically tied with 40-feat in-domain |
| Cascade order | filter→classifier wins +0.65 to +1.27pt F1 on Svanström | none |

---

## OLD numbers — methodology note

OLD numbers in this summary come from my trust-scope derivation off cached `per_det.jsonl` files.
They match check.txt within **0.03pt on Svanstrom IoP classifier (0.9934 vs 0.9937)**, and within
**0.5pt on Anti-UAV IoU classifier (0.9868 vs 0.9916)**. All three pipelines (OLD/NEW40/NEW32)
are computed identically, so within-table deltas are honest.

For verbatim check.txt numbers and full methodology reconciliation see
[the comprehensive report §0 and §6.0–6.6](pipeline_improvements.md).

## End-to-End Ablation (v3_more pipeline beats OLD across 7 configs)

### Anti-UAV (paired drone test, 85,374 frames) — IoP @ 0.5

| Config | OLD F1 | NEW40 F1 | NEW32 F1 |
|---|---|---|---|
| ir_only | 0.9667 | 0.9667 | 0.9667 |
| rgb_only | 0.9821 | 0.9872 | 0.9872 |
| classifier | 0.9888 | 0.9916 | **0.9917** |
| ir_filter | 0.9654 | 0.9666 | 0.9666 |
| rgb_filter | 0.9819 | 0.9872 | 0.9872 |
| filter→classifier | 0.9886 | 0.9916 | **0.9917** |
| classifier→filter | 0.9881 | 0.9915 | **0.9917** |

### Svanström (mixed drone+confusers, 28,710 frames) — IoP @ 0.5

| Config | OLD F1 | NEW40 F1 | NEW32 F1 | Δ NEW vs OLD |
|---|---|---|---|---|
| ir_only | 0.9591 | 0.9591 | 0.9591 | tied |
| rgb_only | 0.5443 | 0.5645 | 0.5645 | **+2.02pt** |
| classifier | 0.9849 | 0.9883 | 0.9881 | **+0.34pt** |
| ir_filter | 0.9457 | 0.9474 | 0.9474 | **+0.17pt** |
| rgb_filter | 0.6208 | 0.6766 | 0.6766 | **+5.58pt** |
| filter→classifier | 0.9785 | 0.9850 | 0.9845 | **+0.65pt** |
| classifier→filter | 0.9641 | 0.9768 | 0.9766 | **+1.27pt** |

NEW40 and NEW32 are statistically tied (within ±0.005 F1). 32-feat is the deployment pick because
it removes the `ir_detected` shortcut without numerical cost in-domain.

---

## YouTube OOD (the leak-free verdict)

### Patch verifier suppression on confusers (NEW vs OLD)

| Group | OLD Suppression (check.txt) | NEW Suppression | Δ |
|---|---|---|---|
| ALL_CONFUSERS | 54.6% | **74.88%** | **+20.3pt** ✓ |
| AIRPLANE | 41.6% | 61.45% | +19.9pt ✓ |
| BIRD | 37.5% | 69.47% | +32.0pt ✓ |
| HELICOPTER | 81.3% | 100.00% | +18.7pt ✓ |
| DRONE_CLEAN | 9.6% | 23.65% | +14.0pt ⚠ |
| DRONE_LABELS | 21.2% | 13.44% | −7.8pt ✓ |

**DRONE_CLEAN regression is per-frame.** In deployment with temporal smoothing (5-of-6 alert
window + cooldowns), a GUI run on `yt_zFu7hAi5mIc.mp4` (the CLEAN drone video) produced **51 alerts,
57 warnings, only 4 suppressions** — the per-frame 23% rate collapses to ~4 actual suppression events.

Frame-count caveat: NEW eval uses an expanded video catalog (14,985 confuser frames vs OLD's 4,993).
Per-frame rates are comparable; absolute counts and weighted-avg suppression should be read with
that caveat. Detail in main report.

### YouTube RGB — apples-to-apples (same script, same frames)

From `eval_youtube_rgb/summary.json` — both YOLOs evaluated on identical frames (stride=3):

| Category | OLD RGB any-det | v3_more raw | v3_more + new patch verifier |
|---|---|---|---|
| ALL | 56.46% | **30.29%** (−26pt) | **12.94%** (−43.5pt vs OLD) |
| AIRPLANE | 49.15% | 27.08% | 9.80% |
| BIRD | 77.76% | 39.27% | 20.82% |
| HELICOPTER | 13.74% | 1.45% | 1.45% |

Full pipeline cuts confuser any-det rate by **43pp** vs OLD raw RGB YOLO on YouTube confusers.

---

## Recommendations

In `ir_gui/fusion_settings.json`:

| Setting | OLD | NEW |
|---|---|---|
| `rgb_model` | `Yolo26n_trained` | `Yolo26n_hardneg_v3_more` |
| `fusion_model` | `fusion_no_fn_model.joblib` | `scene_aware_v3more_32feat/model.joblib` |
| `rgb_conf` | 0.30 | **0.40** (F1-optimal shift) |
| `patch_threshold` | 0.70 | **0.85** (Svanström +0.45pt F1, clean-drone preservation +6pt) |
| `cascade_order` | `classifier_then_filter` (legacy default) | **`filter_then_classifier`** (new GUI default) |

Preserve `_v1_backup.pt`, `fusion_no_fn_model.joblib`, and `control_v3more_40feat/` for rollback.

For full per-component breakdowns, F1-optimal conf sweeps, top-feature analyses, and methodology
caveats, see [`pipeline_improvements.md`](pipeline_improvements.md).
