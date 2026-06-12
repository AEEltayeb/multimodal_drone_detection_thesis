# robust6 Reliability — Statistics-First Investigation Plan (incl. cross-modal features)

**Date:** 2026-06-01 · **Builds on:** `2026-06-01_robust6_state_and_improvement_plan.md`,
`2026-06-01_statistical_feature_selection_STUDY.md`
**Method discipline:** statistics first — every candidate is justified by a statistic *before* a training run, then
confirmed on held-out OOD with a bootstrap CI (memory `feedback_statistics_before_training`, `feedback_mri_always_plots`).
**Reuse, don't recreate:** extend `classifier/fusion_feature_stats.py` (LDA/PCA/ANOVA/leakage on `mri.stats`),
`classifier/train_lean_ft4.py` (trainer), `eval/overnight_ablation.py` (confirm). Record results to `evals.csv`.

---

## 0. The hook — what the ranked stats already tell us (verified from `feature_stats_ranked.csv`)

robust6 = `rgb_max_conf, ir_max_conf, rgb_best_log_bbox_area, ir_best_log_bbox_area, rgb_best_aspect_ratio,
ir_best_aspect_ratio`. Its **IR half is strong; its RGB half is the weak, leaky half** — and the **cross-modal
features that were left out are statistically better:**

| feature | in robust6? | AUROC-alone | leakage | robust_rank | note |
|---|---|---|---|---|---|
| `conf_sum` | ❌ | **0.983** | 0.0021 | **8** | best non-tautological feature in the whole set |
| `xmodal_scale_ratio` | ❌ | 0.903 | **0.0015** | 20 | lowest leakage of any geometry/cross-modal |
| `xmodal_conf_ratio` | ❌ | 0.905 | 0.0024 | 24 | encodes "which modality is winning" → regime-aware |
| `conf_product` | ❌ | 0.907 | 0.0030 | 26 | cross-modal agreement |
| `ir_max_conf` | ✅ | 0.965 | 0.0027 | 15 | strong (kept) |
| `ir_best_aspect_ratio` | ✅ | 0.952 | 0.0020 | 16 | strong (kept) |
| `rgb_max_conf` | ✅ | 0.816 | 0.0043 | 42 | weak |
| `rgb_best_aspect_ratio` | ✅ | 0.778 | 0.0111 | 50 | weak |
| `rgb_best_log_bbox_area` | ✅ | 0.806 | **0.0588** | 60 | **weakest + leakiest in robust6** |
| `xmodal_centroid_dist` | ❌ | 0.906 | **0.571** | 61 | high leakage → **exclude** |
| `neither_detect` / `ir_detected` / `rgb_detected` | ❌ | 0.96/0.95/0.55 | ~0 | 5/15/49 | **label-tautological → exclude by reasoning** |

**Two hypotheses fall straight out of this table:**
- **H1 (cross-modal adds reliability):** `conf_sum` / `xmodal_*_ratio` are stronger and less leaky than robust6's
  RGB geometry. Adding them should raise separability without adding fingerprints.
- **H2 (cross-modal fixes the grayscale recall hole):** a *ratio* like `xmodal_conf_ratio` intrinsically encodes
  "IR weak → trust RGB" — exactly the regime-awareness robust6 lacks (the root cause of its grayscale over-rejection).
  This is the most important thing to test: cross-modal ratios may be the principled fix for §4.1 of the state doc.

---

## 1. Phase 0 — Regime-aware & per-class re-ranking (the analysis robust6 never got)

The original ranking pooled an **IR-dominant** corpus, so RGB and regime-specific behaviour were averaged away.
Re-run the stats **conditioned**, before touching any model.

**Statistics to compute (extend `fusion_feature_stats.py`):**
1. **Per-regime ANOVA + AUROC + leakage:** split rows into `thermal-IR` vs `grayscale-IR` and rank every feature
   *within each regime*. Question: which features keep AUROC > 0.7 under grayscale, and which collapse?
   Prediction: IR-geometry collapses on gray; cross-modal *ratios* and `rgb_max_conf` hold.
2. **Per-trust-class discriminability:** the starved class is `trust_rgb` (700/8871 = 8%). One-vs-rest AUROC per
   feature for the `trust_rgb` and `trust_ir` decisions specifically — which features actually drive the
   single-modality routing (the decisions that fail on grayscale)?
3. **Gray-vs-thermal distribution shift (CORAL-style z-distance):** for each IR feature, standardized mean/scale
   shift between regimes — quantifies *how unreliable* each IR feature becomes on grayscale. Reuse `mri.stats`.
4. **Redundancy / collinearity:** Spearman correlation matrix among the candidate set — do not add a cross-modal
   feature that is collinear with one already in robust6 (e.g. `conf_sum` vs `rgb_max_conf+ir_max_conf`).
5. **Incremental separability:** LDA accuracy of `robust6` vs `robust6 + candidate` on a sequence-grouped split —
   does the candidate move the discriminant axis at all, or is it redundant?

**Plots (mandatory → `docs/analysis/images/`):** per-regime AUROC bars (thermal vs gray, side by side); gray-vs-thermal
z-shift bar; candidate-vs-robust6 correlation heatmap; LDA-accuracy delta bar. **Decision output of Phase 0:** a
shortlist of cross-modal/regime features that (a) hold AUROC on grayscale and (b) are non-redundant with robust6.

---

## 2. Phase 1 — Candidate feature sets (grounded, minimal, each tests one idea)

Train with `train_lean_ft4.py` (XGBoost 4-class, GroupShuffleSplit, ft4+v3b data). All candidates use **only free,
low-leakage** features; `xmodal_centroid_dist` and the detection-flags are **excluded** (leakage / tautology).

| set | features | tests |
|---|---|---|
| `robust6` (baseline) | the 6 | reference |
| `robust6 + conf_sum` | 7 | H1 — strongest single feature |
| `robust6 + xmodal_conf_ratio` | 7 | **H2 — regime-aware ratio (the key run)** |
| `robust6 + {xmodal_conf_ratio, xmodal_scale_ratio}` | 8 | H2 — both ratios |
| `xmodal_aug` = robust6 + {conf_sum, xmodal_conf_ratio, xmodal_scale_ratio} | 9 | H1+H2 combined |
| `xmodal_core` = {ir_max_conf, ir_best_aspect_ratio, ir_best_log_bbox_area, conf_sum, xmodal_conf_ratio, xmodal_scale_ratio} | 6 | **drop weak/leaky RGB geometry, swap in cross-modal** — same count, better stats |
| `robust6 + ir_is_thermal` | 7 | explicit regime flag (alt to ratio) |

`xmodal_core` is the most interesting structural test: it keeps the *count* at 6 but replaces robust6's weakest
member (`rgb_best_log_bbox_area`, leakage 0.059) and its weak RGB pair with the cross-modal trio — a head-to-head
of "raw RGB geometry" vs "cross-modal agreement" at equal complexity.

---

## 3. Phase 2 — Reliability / calibration statistics (not just F1)

Reliability is about *trustworthy probabilities*, not only point accuracy.
- **Reliability diagrams + Expected Calibration Error**, per surface and per regime — is robust6 over-confident
  when it rejects grayscale drones? (Likely yes — that's why it over-vetoes.)
- **Per-class operating-point sweep:** vary the `trust_rgb`/`trust_ir` decision threshold; plot recall-vs-fire
  trade curves. The antiuav −1.2pp recall and the grayscale recall hole may be partly threshold artifacts.
- **Isotonic/Platt recalibration** of the winning set's probabilities; re-measure.

---

## 4. Phase 3 — Confirmation ablation (the production decision)

Run the surviving 1–2 sets through `eval/overnight_ablation.py` (5000 strided frames, both cascade orders,
antiuav + svanström + rgb_confuser + **svan_gray + rgb_dataset + selcom** — the grayscale failure surfaces).

**Win condition (must hold simultaneously):**
1. **Recall up** on svan_gray / rgb_dataset / selcom vs robust6 — the whole point.
2. **No confuser-fire regression** — keep robust6's OOD-FP win (rgb_confuser fire ≤ 0.143 clf_only).
3. **OOD-video F1 ≥ 0.578** (don't reintroduce fingerprint memorisation).
4. **In-domain within bootstrap CI of sa32** (svan F1, antiuav R).

**Significance:** 1000-resample bootstrap CIs on every headline gap — decide ties statistically, not by eye
(closes the §3.2 gap in the state doc and pre-empts the "statistically indistinguishable" overclaim flagged in
`review.csv`).

---

## 5. Decision tree (what each outcome means)

- **H2 holds** (a cross-modal ratio lifts grayscale recall with no FP cost) → that set replaces robust6; the
  thesis story strengthens: *cross-modal agreement is the regime-robust signal, raw per-modality geometry is not.*
- **H2 fails but H1 holds** (cross-modal helps in-domain, not grayscale) → keep robust6 for confuser surfaces,
  **route to filter-only on grayscale** (Step 5 of the state doc; `filter_only[mlp]` already best on svan_gray).
- **`xmodal_core` matches robust6 at equal count** → adopt it anyway (lower leakage = more defensible, drift-proof).
- **Nothing beats robust6** → robust6 stands; the grayscale hole is then a *routing* problem, not a feature problem.

---

## 6. Sequence & cost

Phase 0 (stats only, zero training, minutes) → Phase 1 (≤7 XGBoost fits, fast) → Phase 2 (calibration, cheap) →
Phase 3 (one ablation run, the only GPU-ish cost — and the offline harness replays from cache). **Statistics gate
each training step**: no set goes to Phase 1 unless Phase 0 says its features hold AUROC and aren't redundant.

---

## Delivered
- `docs/analysis/2026-06-01_robust6_reliability_investigation_plan.md` (this plan)
- Verified inputs: `classifier/fusion_models/optimal_v1/feature_stats_ranked.csv` (the §0 table is from it),
  `fusion_dataset_lean19.csv` label counts.
- Runnable units: Phase 0–3 above; scripts to extend = `fusion_feature_stats.py`, `train_lean_ft4.py`,
  `eval/overnight_ablation.py`.
