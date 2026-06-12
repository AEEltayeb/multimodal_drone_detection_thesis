# robust6 Trust Classifier — State, Failure Modes & Statistics-First Improvement Plan

**Date:** 2026-06-01 · **Subject:** `trust_ft4_robust6.joblib` (`classifier/fusion_models/lean_ft4/`)
**Sources:** `2026-06-01_statistical_feature_selection_STUDY.md`, `2026-05-31_ft4_lean_trust_classifier.md`,
`eval/results/_overnight_ablation/ablation_results.{md,json}`, `eval/results/_overnight_ablation_full/ablation_full.md`,
training data `classifier/fusion_models/lean_ft4/fusion_dataset_lean19.csv` (8,871 rows).
Related ledger: `robust6-production-viable`, `ft4-lean-trust-classifier`, `grayscale-trust-classifier-degrades`,
`mlp-beats-patch-both-modalities`, `fusion-feature-leakage`.

---

## 1. What robust6 is (one paragraph)

A **4-class XGBoost trust router** (`reject_both=0 / trust_rgb=1 / trust_ir=2 / trust_both=3`) that fuses one
RGB + one IR frame into a trust decision. It uses **6 free features** — `rgb_max_conf, ir_max_conf,
rgb_best_log_bbox_area, ir_best_log_bbox_area, rgb_best_aspect_ratio, ir_best_aspect_ratio` — chosen by a
**leakage statistic** (`F_domain_inclass / F_class`) that flags scene-fingerprint features (AUROC≈0.5 but
leakage 300+) for removal. It was re-mined on the **current** detector (`ft4` RGB + `v3b` IR), fixing the drift
that made `sa32` (32 feats, old `v3more`/`selcom_1280` detector) stale. It is the statistically-principled
counter-proposal to `sa32`'s hand-picked 32.

---

## 2. Is it as good as sa32? (head-to-head, all verified from the 5000-frame ablation)

| surface (regime) | metric | sa32 | robust6 | verdict |
|---|---|---|---|---|
| svanström (in-domain thermal, IoP) | F1 best-composed | **0.9974** | 0.9957 | sa32 +0.2pp (negligible) |
| anti-UAV (in-domain thermal) | R clf_only | **0.9972** | 0.9854 | sa32 +1.2pp (55 more FN: 68 vs 13) |
| rgb_confuser (OOD, gray-IR) | fire clf_only ↓ | 0.2032 | **0.1428** | **robust6 −30%** false alerts |
| rgb_confuser | fire best-composed ↓ | 0.0277 | **0.0186** | **robust6 −33%** |
| OOD drone video (Level-2 F1m) | macro-F1 | 0.262 (all19) | **0.578** | **robust6 2.2×** |
| rgb_dataset_test (gray-IR drone) | R clf_only | **0.7134** | 0.607 | sa32 +11pp recall |
| svanström_gray (gray-IR drone, IoP) | R clf_only | **0.6305** | 0.5988 | sa32 +3pp recall |
| selcom_val (gray-IR, hard RGB, IoP) | R clf_only | **0.2653** | 0.0442 | sa32 **6× better** (outlier, §4) |

**Bottom line:** *In-domain they tie* (sa32 edges by ≤1.2pp, statistically untested). **OOD-confuser false-alert
suppression: robust6 wins clearly** — this is the thesis's central concern, so robust6 is the principled pick.
**But robust6 is more aggressive at rejecting, and on hard/grayscale *drone* surfaces it costs real recall** —
worse than sa32 everywhere a drone is hard to see. That asymmetry (better FP, worse hard-positive recall) is the
whole story and the target of the improvement plan.

---

## 3. Has it been thoroughly evaluated? — Mostly, with concrete gaps

**Done well:** 3-level ablation (cached feature-set screen → ft4 re-train w/ GroupShuffleSplit → 5000-frame
full-pipeline with both cascade orders); confuser + in-domain + OOD-video surfaces; statistical provenance
(LDA 0.982 separability, PCA, ANOVA/AUROC, leakage map) all plotted.

**Gaps (each is an eval to run):**
1. **NOT in `knowledge/evals.csv`** — `grep robust6 evals.csv` = 0 hits. The production-decision numbers live
   only in the ledger note + this study. **Recording gap** — must record the ablation cells as evals rows.
2. **No significance test / CI** on the close calls. The sa32-vs-robust6 in-domain verdict rests on a 0.2pp F1
   gap and a 1.2pp recall gap with **no bootstrap CI**. Per the review-table pattern (row on "statistically
   indistinguishable" overclaims) this must be a bootstrap CI, not an eyeballed tie.
3. **No threshold / operating-point sweep for robust6.** The 1.2pp anti-UAV recall cost may be a decision-cut
   artifact, not a capacity limit — untested.
4. **Trust-classifier swap absent from the offline verifier matrix** (`2026-06-01_full_pipeline_offline_eval.md`
   explicitly carves it out — needs paired RGB+IR caching).
5. **No per-confuser-category breakdown** — we know robust6 cuts aggregate confuser fire 30%, but not whether it
   helps birds and hurts airplanes (the known cascade asymmetry) or uniformly.
6. **selcom_val collapse unexplained** beyond "doubly-OOD" hand-wave (§4).

---

## 4. Where robust6 fails — root-caused

### 4.1 Hard-positive recall on grayscale / RGB-fallback drone surfaces (the real failure)
robust6 over-rejects drones whenever IR is the grayscale fallback (not true thermal): selcom R 0.044,
rgb_dataset R 0.607, svan_gray R 0.599 — all **below sa32**. `filter→clf` on grayscale **craters** (svan_gray
R 0.176). Root cause is structural and **statistically diagnosable**:

- **Training corpus is IR-dominant and minority-class-starved.** Verified label counts in `fusion_dataset_lean19.csv`:
  `trust_both=4424, reject=3373, trust_rgb=700, trust_ir=374`. The two single-modality-trust classes are **12%**
  of the data combined, and `trust_rgb` (the class you need when IR is unreliable gray) is only **8%**. The
  leakage/AUROC ranking that picked robust6 was therefore computed on an **IR-feature-favouring** pool (the study
  itself flags "corpus is IR-dominant, IR features rank above RGB"). So robust6 leans on IR-geometry features that
  **degrade on grayscale** → it stops trusting RGB exactly when it should.
- **No regime feature.** robust6 has nothing telling it "IR is grayscale fallback, downweight IR." sa32's extra
  (fingerprint) features accidentally proxy this on its training scenes, which is why sa32 holds recall slightly
  better on gray — but that's memorisation, not a fix.

### 4.2 In-domain anti-UAV recall (−1.2pp, 55 FN)
Small and possibly threshold-fixable (§3.3). Real but minor.

### 4.3 selcom_val (R 0.044) — outlier, not representative
selcom is **doubly-OOD**: hard RGB (bare detector R only 0.49, needs imgsz 960) **and** grayscale-IR fallback the
IR/aligned model never saw. Both robust6 and sa32 are bad here; robust6 worse. Treat as a stress case, not the
headline — but it is the most extreme instance of the §4.1 grayscale failure.

---

## 5. Improvement plan — statistics first, training second

> **Principle (also being added to memory/knowledge): we do not train-then-pray. Every change is justified by a
> statistic computed *before* training, and validated by a held-out OOD ablation *after*. Same MRI discipline as
> the YOLO model-MRI and this very feature-selection study.**

Ordered by expected value. Each step says the *statistic to compute first*, the *change*, and the *eval to confirm*.

### Step 1 — Fix the grayscale/RGB-fallback recall hole (highest value)
- **Statistic first:** (a) per-modality-regime ANOVA + leakage **recomputed on a regime-balanced pool** (split
  rows into thermal-IR vs gray-IR and rank features *within each*) — confirm which features flip from signal to
  noise under grayscale; (b) **z-distance** of each IR feature's distribution gray-vs-thermal (CORAL-style domain
  gap), reusing `mri.stats`. Plot both (per the always-plot rule).
- **Change A (data):** re-mine with **more `trust_rgb` / grayscale-drone positives** (selcom, svan_gray,
  rgb_dataset, drone videos on gray fallback) to lift the 8% `trust_rgb` class. Re-stratify.
- **Change B (feature):** add a **regime flag** `ir_is_thermal` (free, 1 bit) and/or a cross-modal feature that is
  robust under grayscale (`xmodal_conf_ratio` scored AUROC 0.905 / leakage 0.002 — statistically excellent and
  free). Test `robust6 + ir_is_thermal` and `robust6 + xmodal_conf_ratio` as 7-feature variants.
- **Confirm:** 5000-frame ablation, **recall on svan_gray / rgb_dataset / selcom must rise without raising
  confuser fire** — that's the win condition.

### Step 2 — Recover the anti-UAV 1.2pp via calibration, not features
- **Statistic first:** **reliability diagram** (predicted P(trust) vs empirical) per surface; check if robust6 is
  mis-calibrated on thermal (over-confident reject).
- **Change:** isotonic/Platt calibration of the 4-class probabilities, or per-class decision-threshold sweep.
- **Confirm:** anti-UAV recall → match sa32 (≤0.3pp) with no confuser-fire regression.

### Step 3 — Guided feature add-back (only if Steps 1–2 leave a gap)
- **Statistic first:** the ranked table already has the candidates — `conf_sum` (AUROC 0.983, leakage 0.002),
  `xmodal_scale_ratio` (0.903 / 0.002). These are **free and low-leakage** — they were left out for minimality,
  not because they failed the statistic.
- **Change:** test `robust7 = robust6 + conf_sum` and `+ xmodal` pair. meta4 proved IR geometry is load-bearing;
  this asks whether 1–2 vetted cross-modal features recover hard-positive recall without reintroducing fingerprints.
- **Confirm:** OOD-video F1 must not drop below robust6's 0.578; in-domain must not regress.

### Step 4 — Significance + the recording the thesis needs
- **Bootstrap CIs** (1000 resamples) on every sa32-vs-robust6 headline (svan F1, antiuav R, confuser fire). Decide
  ties *statistically*, kill the "0.2pp" eyeballing.
- **Per-confuser-category** fire-rate (bird/airplane/heli) for both classifiers — does robust6's 30% win hold per
  category or is it the known bird-vs-airplane asymmetry?
- **Record** the ablation cells into `knowledge/evals.csv` (`kb.py record`) + a `ledger` finding. Closes the §3.1 gap.

### Step 5 — Grayscale routing (architectural, if Step 1 insufficient)
If a single classifier can't serve both regimes, **route**: trust-classifier on thermal, **bypass to filter-only
on grayscale** (the ledger already shows `filter_only[mlp]` is *best* on svan_gray, F1 0.891 R 0.87 — beating
every classifier cell). This is a config/engine change, evidence already in hand.

---

## 6. Verdict on "ship robust6 or not"

**For the thesis's open-world / confuser-heavy framing: robust6 is the defensible pick today** (better OOD-FP,
cheaper, drift-proof, statistically justified). **For a recall-critical deployment on grayscale-fallback hardware:
not yet** — Step 1 must close the hard-positive recall hole first, or route per Step 5. sa32 is *not* the safer
fallback: it only wins where it memorised (in-domain + accidental gray proxy), which is the overfitting trap the
whole leakage analysis exists to avoid.

---

## Delivered
- `docs/analysis/2026-06-01_robust6_state_and_improvement_plan.md` (this doc)
- No new runs executed — this is a plan; Steps 1–5 are the runnable units.
- Verified inputs: `fusion_dataset_lean19.csv` label counts; ablation cells in
  `eval/results/_overnight_ablation{,_full}/`.
