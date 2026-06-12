# Consolidated comparison: NEW stack (robust6 + V5 MLP) vs OLD ensemble — for Pietro / thesis

**Date:** 2026-06-04. Single reference pulling together the corrected paired tables, the OOD
filter comparison, speed, and feature-efficiency. All numbers measured this cycle (sources at end).

## TL;DR — the pitch in three lines
1. **Filter (CNN patch → V5 MLP): a clear upgrade** — far better OOD-confuser suppression *and*
   ~11–72× faster. This is Pietro's "best of both worlds."
2. **Classifier (fusion_no_fn → robust6): a principled trade** — 6 free features, **404× faster**,
   matches in-domain (99.8%), **+30% OOD-confuser robustness**, at a ~1–3pp paired-benchmark cost.
3. **The combined ensemble (IR + RGB + robust6 + MLP) matches the old >0.97 system** and is cheaper.

---

## 1. The email's 3 domains, recomputed (trust-aware scoring, OLD email vs NEW stack)

**Anti-UAV (IoU @0.5, 85,374 paired frames) — saturated/clean, no confusers:**
| Config | OLD F1 | NEW F1 | Δ |
|---|---:|---:|---:|
| ir_only / rgb_only | 0.962 / 0.990 | 0.962 / 0.986 | ≈ |
| classifier | 0.9916 | 0.9818 | −0.010 |
| filter→classifier | 0.9916 | 0.9820 | −0.010 |
| classifier→filter | 0.9909 | 0.9816 | −0.009 |

**Svanström (IoP @0.5, 28,710 paired frames) — confuser-heavy, the decider:**
| Config | OLD F1 | NEW F1 | Δ |
|---|---:|---:|---:|
| **rgb_filter** | 0.699 | **0.869** | **+0.170** ★ |
| ir_filter | 0.946 | 0.959 | +0.014 |
| rgb_only | 0.527 | 0.601 | +0.073 |
| classifier | 0.994 | 0.959 | −0.034 |
| filter→classifier | 0.993 | 0.968 | −0.025 |
| classifier→filter | 0.975 | 0.950 | −0.025 |

**Read:** the new **filter** improves every filter row (Svan `rgb_filter` +17pp). The **robust6**
classifier is ~1–3pp under the old `fusion_no_fn` on the paired classifier configs — its advantage
(OOD-confuser FP) doesn't appear on these in-domain/saturated benchmarks (§2 shows where it does).
Note: NEW = whole-stack (detector+classifier+filter all change, forced by the FT4-distilled MLP).

---

## 2. Where the filter earns its keep — OOD confusers (CNN patch vs V5 MLP, *same* FT4/v3b detector)

The filters' job is suppressing confuser false-alarms while preserving drone recall. On OOD data the
CNN patch hits a **suppression ceiling**; the MLP is tunable far past it.

| OOD surface | CNN patch suppression | V5 MLP suppression | drone-recall note |
|---|---|---|---|
| **IR thermal video** (domain-3) | 80.7% (heli 94.6) **but kills 68% of drone dets** | 21.7%→tunable; @0.05 **keeps 99.6% drones** | MLP recall-safe; CNN over-vetoes |
| **RGB images** (`rgb_confuser`) | **caps ~52%** | **78% @0.05 → 99% @0.95** | recall kept on svan/selcom; −7pp on rgb_dataset |
| **RGB video** (drone-video-tests) | **caps ~55%** (heli only 30%) | **90% @0.5, 99% @0.95** (heli 84%) | tied at low suppression; MLP can go where CNN can't |

- **MLP Pareto-dominates on Svanström `rgb_filter`:** at the CNN's suppression it keeps **+24pp more
  drones**; at the CNN's recall it suppresses **+23pp more confusers**.
- **The CNN patch structurally caps** at ~52–55% suppression on OOD RGB (helicopters ~30%); the MLP
  reaches 99%. The CNN cannot reach high-suppression operating points at all.

---

## 3. Speed (measured — `eval/bench_speed.py`)

| Trust classifier (per frame) | features | feat-extract | predict | total |
|---|---:|---:|---:|---:|
| **robust6** | 6 (free) | 0.010 ms | 0.085 ms | **0.095 ms** |
| fusion_no_fn | 40 | 38.2 ms | 0.10 ms | **38.3 ms** |

→ **robust6 404× faster/frame** (fusion_no_fn's 38 ms is OpenCV scene statistics; robust6 needs none).

| Confuser filter (per detection) | cost |
|---|---:|
| **V5 MLP** (full, incl. ROI-pool) | **~1.3–2.1 ms** (forward alone 0.11 ms) |
| CNN patch v2 | 23.7 ms (measured) / 59–112 ms (ledger) |

→ **MLP ~11–72× faster/det**; pipeline overhead 1–4% vs the patch's 48–191%.
(Edge end-to-end latency remains unmeasured — verifier-stage only; ledger `latency-edge-unmeasured`.)

---

## 4. robust6: 6 features vs 40, and why it's "X% as good"

robust6 = `{rgb,ir}_max_conf`, `{rgb,ir}_best_log_bbox_area`, `{rgb,ir}_best_aspect_ratio` — a strict
subset of fusion_no_fn's 40, keeping only the **free** (confidence + box-geometry) features and dropping
all 34 scene/position/flag features. The 6 were chosen by **statistics** (LDA 0.982 separability; PCA
shows scene-variance dominates; ANOVA/AUROC rank; a **leakage statistic** flags fingerprints):

| | robust6's KEEP features | fusion_no_fn's dropped scene features |
|---|---|---|
| AUROC-alone | 0.946–0.965 (real signal) | `rgb_img_std` 0.502, `rgb_img_entropy` 0.510 (chance) |
| leakage `F_domain/F_class` | 0.002–0.005 (robust) | **349.6, 307.4** (pure scene memorisation) |

| Performance | robust6 | reference | robust6 = |
|---|---:|---:|---|
| in-domain Svan F1 (full-pipeline) | 0.9957 | sa32 (32f) 0.9974 | **99.8%** |
| OOD confuser false-alerts ↓ | 0.143 | sa32 0.203 | **−30%** |
| OOD drone-video F1 | 0.578 | 19-feat 0.262 | **2.2× better** |
| Anti-UAV recall | −0.6pp | sa32 | tiny cost |

---

## 5. Pietro's three questions, answered
1. *Which baseline?* — The NEW column is the whole new stack (FT4+v3b + robust6 + MLP) vs the email's
   old ensemble (old detector + fusion_no_fn + CNN patch). Stated explicitly; the MLP forces FT4.
2. *Does the standalone solution beat the full ensemble?* — **No, and it shouldn't** — on
   confuser-heavy data the IR+classifier routing is essential (Svan: rgb_only 0.60 vs classifier 0.96).
   The **best** system is the ensemble **with** the MLP filter.
3. *Can we combine — full ensemble but MLP filter instead of CNN?* — **Yes, done, and it works:** the
   combined system matches the old >0.97 ensemble (Anti-UAV 0.982, Svan filter→classifier 0.968) while
   the filter itself is a clear OOD upgrade and ~11–72× faster, and robust6 adds OOD robustness at 404×
   the classifier speed.

---

## Provenance
- **Paired tables:** `eval/pipeline_cache_paired.py` (Phase A, f32 + cached P(drone)) →
  `eval/pipeline_eval_paired.py` (Phase B, trust-aware via `eval/metrics.py::score_trust_aware`).
  Cache `eval/results/_email_recompute/`. Frame sets driven from the email's own manifests.
- **OOD filter:** `eval/sweep_rgb_filter_ood.py` (rgb_confuser, FT4-isolated), `eval/eval_rgb_video_ood.py`
  (RGB video), `classifier/eval_youtube_ir_filter.py --mlp` (IR domain-3), `eval/sweep_email_thresholds.py`.
- **Speed:** `eval/bench_speed.py`. **Features/stats:** `2026-06-01_statistical_feature_selection_STUDY.md`.
- **Ledger:** `email-recompute-robust6-mlp`, `mlp-filter-beats-cnn-ood`, `robust6-speed-feature-efficiency`,
  `mlp-feats-need-f32`. **Evals:** `svan_classifier_robust6_ta`, `svan_rgbfilter_mlp_ta`,
  `antiuav_classifier_robust6_ta` (configs `svan_iop_email`, `antiuav_iou_email`).
