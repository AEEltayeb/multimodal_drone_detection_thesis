# Filter (confuser-verifier) number map — thesis audit

**Date:** 2026-06-17
**Purpose:** Exhaustive registry of every numeric value in the thesis attributable to the two confuser **filters** (per-detection MLP verifiers), so they can be swapped for retrained versions.
**Filters in scope:**
- **RGB filter** = `mlp_v5` (distillation MLP verifier), op point P(drone) >= 0.25
- **IR filter** = `mlp_v5_ir_aligned` — ONE network, two scalers: thermal `mlp_aligned.pt` (conf 0.40, thr 0.05) and grayscale `mlp_aligned_gray.pt` (conf 0.25, thr 0.25)
- Dedicated grayscale-only `mlp_v5_gray` is the *superseded* per-mode model that the aligned IR filter replaces; its comparison numbers are filter-relevant and included (flagged `gray`).

**READ-ONLY.** No .tex edited.

## CRITICAL STRUCTURAL FINDINGS (read before swapping)

1. **`docs/thesis_working.tex` is the live thesis (2420 lines).** It contains the `mlp_v5` / `mlp_v5_ir_aligned` filters ONLY at the **component level**:
   - §`sec:distill_verifier` (`tab:distill_verifier`, lines ~1242-1349) — RGB filter
   - §`sec:ir_xmodal_verifier` + §`sec:grayscale_verifier` (`tab:ir_aligned`, `tab:ir_aligned_gray`, lines ~872-1815) — IR filter
   - inline mentions in Abstract / RQs / Contributions / Conclusion / Glossary

2. **This working file does NOT contain the `filt` / `filt_mlp` / `filt (mlp_v5,RGB)` / `clf->filt` / `filt->clf` paired-ablation tables, the `fig:filter_operating` per-filter sweep, the Part-C confuser FP-reduction tables driven by the mlp filter, or the `tab:robust8_operating` figure** described in the task brief. Those belong to a LATER (robust8 / robust8-nr) revision that is NOT present in this file. **The swap team must locate that other revision separately** — searching this file alone will miss every paired-ablation `filt` cell because they do not exist here yet.

3. **All "cascade" / "cumulative" / "Roboflow" / "patch-threshold" tables in this file use the MobileNetV3 PATCH verifier (v2), which the task explicitly EXCLUDES.** Affected (NOT filter numbers): `tab:cum_confuser`, `tab:cumulative_svanstrom`, `tab:ood_rgb_drone`, `tab:ood_rgb_confuser`, `tab:ood_ir`, `tab:patch_sweep`, `tab:patch_audit`, `tab:cascade_perframe`, `tab:cascade_segment`, `tab:cascade_percategory`, `tab:cascade_classifier_*`, `tab:robust6_pipeline`. They are listed in the EXCLUDED section at the bottom for completeness.

4. **`docs/thesis_chapters.tex` (older synced copy, 1675 lines) contains NO `mlp_v5` / `mlp_v5_ir_aligned` / distill-verifier / aligned-verifier content at all** — it predates the distilled filters and uses only the patch verifier. **Zero in-scope filter numbers in thesis_chapters.tex.** (Confirmed: no `tab:distill_verifier`, no `ir_aligned`, no `mlp_v5_ir_aligned` labels present.)

---

## Registry (thesis_working.tex) — grouped by surface

All `file` = `docs/thesis_working.tex`. `RGB/IR/gray` = which filter the number belongs to.

### Surface: Svanström (RGB filter, mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes / harness cell |
|---|---|---|---|---|---|---|
| 1252 | 0.869 | drone F1 (+mlp_v5) | Svanström IoP@0.5 | tab:distill_verifier | RGB | "+ mlp_v5" col; bare 0.596, +patch v2 0.768 |
| 1252 | 0.037 | halluc rate (mlp_v5) | Svanström | tab:distill_verifier | RGB | bare 0.470, patch v2 0.163 |
| 1269 | 0.768 -> 0.869 | drone F1 gain | Svanström | (prose) | RGB | "~10-pp gain" |
| 1269 | 0.163 -> 0.037 | halluc rate cut | Svanström | (prose) | RGB | |
| 1264 | 0.470 -> 0.037 | halluc rate (fig caption) | Svanström | fig:distill_verifier_bar | RGB | |
| 222 (contrib) | 0.768 -> 0.869 | drone F1 | Svanström | (Contributions) | RGB | |
| 1283 | conf=0.72 | drone crop conf (fig) | Svanström | fig:mri_activation | RGB | activation example, drone |

### Surface: SelCom CCTV (RGB filter, mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1254 | 0.607 | drone F1 (+mlp_v5) | SelCom CCTV IoP@0.5 | tab:distill_verifier | RGB | bare/patch both 0.591 |
| 1254 | 0.019 | halluc rate (mlp_v5) | SelCom | tab:distill_verifier | RGB | bare/patch 0.071 |
| 1269 | +1.5 pp | F1 gain over bare | SelCom | (prose) | RGB | "gains 1.5pp F1" |
| 222 (contrib) | +1.5 pp | F1 | SelCom | (Contributions) | RGB | |

### Surface: Anti-UAV (RGB filter, mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1253 | 0.985 | drone F1 (+mlp_v5) | Anti-UAV IoU@0.5 | tab:distill_verifier | RGB | bare/patch 0.986 (ties, slight regress) |
| 1253 | 0.010 | halluc rate (mlp_v5) | Anti-UAV | tab:distill_verifier | RGB | bare/patch 0.011 |

### Surface: rgb_dataset (photo-style held-out) (RGB filter, mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1255 | 0.792 | drone F1 (+mlp_v5) | rgb_dataset IoU | tab:distill_verifier | RGB | bare 0.929, patch v2 0.904 (carve-out, -11pp) |
| 1255 | 0.010 | halluc rate (mlp_v5) | rgb_dataset | tab:distill_verifier | RGB | bare 0.028, patch 0.026 |
| 1273 | 0.792 vs 0.904 | F1 (-11 pp) | rgb_dataset | (prose) | RGB | regression vs patch |
| 1273 | 0.896 -> 0.664 | drone R collapse | rgb_dataset | (prose) | RGB | recall ceiling |
| 1273 | 14,500-drone / -3 pp | coverage-boost recall | rgb_dataset | (prose) | RGB | failed fix |
| 222 (contrib) | -11 pp F1 | rgb_dataset | (Contributions) | RGB | carve-out |
| 1290-1296 | 0.896 -> 0.664 | drone R | rgb_dataset | sec:mlp_recall_drop | RGB | recall-drop diagnosis intro |
| 1293 | Δ=0.000 | conf-scalar mean (kept vs vetoed) | rgb_dataset | sec:mlp_recall_drop | RGB | refutes "kills marginal" |
| 1293 | -0.180 vs -0.075 (Δ+0.188) | mean log_area vetoed vs kept | Svanström | sec:mlp_recall_drop | RGB | vetoed drones smaller |
| 1293 | ~0.89 | single-neuron AUROC | rgb_dataset | sec:mlp_recall_drop | RGB | p3/p5 discriminators |
| 1293 | 517 feats / ~770 samples | feature/sample count | — | sec:mlp_recall_drop | RGB | LDA caveat |
| 1296 | 16.48 vs 11.05 | centroid dist to confuser (vetoed/kept) | rgb_dataset | sec:mlp_recall_drop | RGB | |
| 1296 | 0.876 vs 0.862 | mean top-20 AUROC | rgb_dataset | sec:mlp_recall_drop | RGB | |
| 1296 | 15.37 | centroid dist to training-drone cluster | rgb_dataset | sec:mlp_recall_drop | RGB | |
| 1300 | 34.4 vs 9.8 | median OOD-from-confuser dist (vetoed drones/confusers) | rgb_dataset | (fail-open prose) | RGB | |
| 1300 | 91.3% (157/172) | drones recovered @5% leak | rgb_dataset | (prose) | RGB | fail-open gate |
| 1300 | ~10 extra FP / 2,633 imgs | confuser cost @5% leak | rgb_dataset | (prose) | RGB | |
| 1300 | ~0.69 -> ~0.89 | recall restored | rgb_dataset | (prose) | RGB | |
| 1300 | ~9% | irreducible vetoed-drone tail | rgb_dataset | (prose) | RGB | overlaps confusers |
| 1304 | 0.887 -> 0.631 | precision (full veto -> fail-open) | Svanström | (prose) | RGB | fail-open backfires |
| 1306 | 0.486 -> 0.611 (vs 0.884) | precision @recall 0.90 | Svanström | (prose) | RGB | expanded reference |
| 1308 | 43 of 19,334 | under-scored training drones | — | (prose) | RGB | re-weight no-op |
| 1308 | ΔR=-0.008 | temporal-vote recall | photo-style frames | (prose) | RGB | temporal can't recover |
| 1311 | R 0.93 -> 0.84 | frame-level recall | rgb_dataset | (prose) | RGB | still-image carve-out |
| 1329 | τ=21.9, 83.1% (143/172) | fail-open recovery @<=2% leak | rgb_dataset | tab:failopen | RGB | |
| 1330 | τ=19.3, 91.3% (157/172) | fail-open recovery @<=5% leak | rgb_dataset | tab:failopen | RGB | bolded |
| 1331 | τ=15.8, 100% (172/172) | fail-open recovery @<=10% leak | rgb_dataset | tab:failopen | RGB | |
| 1343 | 34.4 / 9.8 ; 91.3% ; 100% | fig caption (dup of above) | rgb_dataset | fig:failopen | RGB | |
| 1348 | 91-100% | recovery range | rgb_dataset | (prose) | RGB | closing para |

### Surface: confuser-only / OOD zoo (RGB filter, mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1256 | 0.008 | halluc rate (mlp_v5) | confuser-only | tab:distill_verifier | RGB | bare 0.317, patch v2 0.107 |
| 1269 | 0.107 -> 0.008 | halluc rate (13x) | confuser-only | (prose) | RGB | |
| 1264 | 0.317 -> 0.008 | halluc rate (fig) | confuser-only | fig:distill_verifier_bar | RGB | |
| 222 (contrib) | 10.7% -> 0.8%, 13x | halluc rate | confuser-only | (Contributions) | RGB | |
| 1278 | 97% / 98.9% | confuser reject / drone retain (out-of-fold) | shipped corpus | (prose) | RGB | |
| 1278 | 0.9857 ± 0.0004 | 5-fold CV F1 | shipped corpus | (prose) | RGB | |
| 1278 | 517 feats | near-linear boundary | — | (prose) | RGB | |
| 1285 | conf=0.29 | confuser crop conf (fig) | rgb_confusers test | fig:mri_activation | RGB | |

### RGB filter — cross-surface / general (mlp_v5)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1269 | 46-72x faster; 1-4% overhead; <0.2 ms; 4.0 pp Svan F1 | latency / cost | — | (prose) | RGB | per-frame vs alert-gate |
| 222 (contrib) | 46-72x; 1-4% | latency/overhead | — | (Contributions) | RGB | |
| 222 (contrib) | ~95% single-threshold acc; 32,931-detection corpus | LDA separability | shipped corpus | (Contributions) | RGB | |
| 639 | 0.952 (RGB), 0.981 (IR) | LDA train accuracy | distilled corpus | sec:model_mri | RGB+IR | linear separability |
| 250 (P(drone) thr) | 0.25 | op-point threshold | — | (filter def, brief) | RGB | not a free-floating line; threshold = 0.25 |

### Surface: CBAM held-out + thermal (IR filter, mlp_v5_ir_aligned, thermal scaler)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1777 | 0.786/0.917/0.846 (15 FP) | P/R/F1 (FP) +aligned MLP | CBAM held-out | tab:ir_aligned | IR | bare 0.547/0.967/0.699 (48 FP) |
| 1777 | -0.050 / +0.147 | ΔR / ΔF1 | CBAM held-out | tab:ir_aligned | IR | |
| 1771 | 69% (48->15), +0.147 F1 | FP cut / F1 | CBAM held-out | tab:ir_aligned caption | IR | |
| 1771 | ΔR at most -0.007 | recall safety | in-distribution thermal | tab:ir_aligned caption | IR | |
| 1771 | -0.050 | held-out CBAM ΔR | CBAM | tab:ir_aligned caption | IR | residual cost |
| 1778 | 0.965/0.958/0.962 (108 FP) | P/R/F1 (FP) +aligned | ir_dset_final (n=4806) | tab:ir_aligned | IR | bare 0.965/0.965/0.965 (109); ΔR -0.007/ΔF1 -0.003 |
| 1779 | 0.909/0.977/0.942 (80 FP) | P/R/F1 (FP) +aligned | ir_video test (n=831) | tab:ir_aligned | IR | identical to bare (0/0) |
| 1780 | 0.983/0.942/0.962 (68 FP) | P/R/F1 (FP) +aligned | Anti-UAV test (n=4269) | tab:ir_aligned | IR | identical to bare (0/0) |
| 1782 | 0.564/0.883/0.688 (41 FP), cut 7 FP | patch verifier on CBAM | CBAM held-out | tab:ir_aligned | IR | patch baseline for comparison |
| 1810 | -0.050 | held-out CBAM drone R cost | CBAM | sec:grayscale_verifier | IR | residual cost restated |
| 1767 | 517-D | verifier dimensionality | — | sec:grayscale_verifier | IR | |
| 1767 | conf=0.40, thr=0.05 | thermal op point | thermal | tab:ir_aligned caption | IR | scaler mlp_aligned.pt |
| 1807 | ~30k thermal drones; ΔR -0.007 / 0.000 / 0.000 | re-mine recall safety | ir_dset_final/ir_video/Anti-UAV | sec:grayscale_verifier | IR | drone-diversity re-mine |
| 1807 | 4-6 pp recall (earlier attempt) | prior IR-verifier loss | held-out thermal | (prose) | IR | superseded attempt |
| 1807 | 0.986 (YOLO-feat-only CV) vs 0.987 (fused) | CV score | — | (prose) | IR | |

### IR filter — CBAM / thermal in Abstract / Contributions / Conclusion

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 222 (contrib) | 0.699 -> 0.846, FP 48->15 | held-out CBAM F1 / FP | CBAM | (Contributions) | IR | NOTE: 0.846 here vs 0.841 in glossary/memory — see AMBIGUOUS |
| 2356 (glossary) | 0.699 -> 0.841, FP 48->13 | CBAM held-out F1 / FP | CBAM | tab:models_evaluated row | IR | NOTE: 0.841/13 differ from tab:ir_aligned 0.846/15 — see AMBIGUOUS |

### Surface: grayscale (IR filter, aligned_gray scaler) + dedicated mlp_v5_gray

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1800 | 12 FP, 96% cut | confuser FP (aligned-gray) | rgb_confusers->gray (n=1317) | tab:ir_aligned_gray | IR(gray) | bare 325 FP |
| 1800 | 13 FP (96%), ΔR -0.113 | dedicated mlp_v5_gray | rgb_confusers->gray | tab:ir_aligned_gray | gray | superseded per-mode model |
| 1801 | 0.762/0.157/0.261 (723 FP) | P/R/F1 (FP) aligned-gray | rgb_dataset->gray (n=17209) | tab:ir_aligned_gray | IR(gray) | bare 0.704/0.210/0.324 (1307 FP) |
| 1801 | ΔR -0.053 | aligned-gray drone recall cost | rgb_dataset->gray | tab:ir_aligned_gray | IR(gray) | |
| 1789 | 96% (325->12) | grayscale confuser FP cut | rgb_confusers->gray | (prose) | IR(gray) | |
| 1789 | 13 FP; ΔR -0.053 vs -0.113 | aligned vs dedicated mlp_v5_gray | gray | (prose) | IR(gray)+gray | half the recall cost |
| 1794 | conf=0.25, thr=0.25 | grayscale op point | gray | tab:ir_aligned_gray caption | IR(gray) | scaler mlp_aligned_gray.pt |
| 1764 | 37.2% vs 1.8% (~20x) | v3b grayscale vs thermal halluc/image | confuser frames | sec:grayscale_verifier | (IR context) | motivates harvest; v3b detector not the filter, but defines the harvest surface |
| 1810 | +0.197 (Svan), +0.130 (rgb_dataset); 0.33-0.65 vs 0.79-0.98 | grayscale bare-v3b drone R | Svan/rgb_dataset | sec:grayscale_verifier | (IR context) | detection surface bound, not filter metric |

### IR filter — feature-space evidence (mlp_v5_ir_aligned underpinnings, §ir_xmodal_verifier)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 877 | 517-D; 14,697 drone / 1,386 confuser | corpus size | IR surfaces | sec:ir_xmodal_verifier | IR | MRI mining pool |
| 877 | 0.981 train acc | LDA separability | v3b IR | sec:ir_xmodal_verifier | IR | |
| 877 | F=5,370 | ANOVA single-neuron | v3b IR | sec:ir_xmodal_verifier | IR | |
| 877 | 1.8%/image | raw detector halluc | thermal | sec:ir_xmodal_verifier | IR | |
| 877 | 89% FP cut @ 10% recall cost | linear probe | thermal | sec:ir_xmodal_verifier | IR | |
| 884 | 0.981; F=5,370 | LDA / ANOVA (fig) | v3b IR | fig:ir_v3b_lda | IR | |
| 913 | Jaccard 0.71-0.88; corr 0.93-0.99; cosine 0.012 | gray<->thermal feature similarity | v3b IR | sec:ir_xmodal_verifier | IR | alignment basis |
| 913 | 0.500 -> 0.919 (ceiling 0.974); CORAL 0.707 | gray->thermal transfer AUROC | v3b IR | sec:ir_xmodal_verifier | IR | z-score alignment |
| 919 | 0.500 / 0.919 / 0.974 / 0.707 | transfer AUROC (fig) | v3b IR | fig:ir_gray_align | IR | dup of above |
| 639 | 0.952 (RGB) / 0.981 (IR) | LDA separability | distilled/IR corpus | sec:model_mri | RGB+IR | listed once above; cross-ref |

### Abstract / Intro / RQ inline filter mentions

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 158 (abstract) | (names mlp_v5, mlp_v5_ir_aligned) | — | — | Abstract | RGB+IR | no standalone metric; the 52.1->10.3 etc. on this line are CLASSIFIER+patch, NOT filter |
| 202 (RQ1) | (names mlp_v5, mlp_v5_ir_aligned) | — | — | RQ1 | RGB+IR | qualitative |
| 215 (contrib1) | (names mlp_v5, mlp_v5_ir_aligned) | — | — | Contributions | RGB+IR | the 52.1/10.3/0.8 numbers here are classifier-chain, not filter |
| 220 (contrib4) | grayscale F1 0.636, 0.837/0.840, 3.2x | cross-modal transfer | YouTube video | Contributions | (IR detector, not filter) | RQ4 transfer — EXCLUDE from filter swap (detector, not mlp filter) |

### Production-stack / conclusion inline (mlp_v5 / aligned named)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 1498 | (names mlp_v5 / mlp_v5_ir_aligned) | — | — | sec:cumulative | RGB+IR | states cumulative tables use PATCH, not these filters |
| 2091 (RQ2) | (names mlp_v5/aligned) | — | — | sec:rq_answers | RGB+IR | qualitative; numbers on line are patch |
| 2113 | (names mlp_v5/aligned, MLPVerifier) | — | — | sec:production_stack | RGB+IR | qualitative |
| 2116 | (carve-outs) | — | rgb_dataset/gray | sec:production_stack | RGB+IR | qualitative carve-out summary |

### Glossary / models table (filter rows)

| line | number/value | metric type | dataset/surface | table or \label | filter | notes |
|---|---|---|---|---|---|---|
| 2356 | F1 0.699->0.841, FP 48->13 | CBAM held-out | CBAM | tab:models_evaluated | IR | mlp_v5_ir_aligned row (values differ from tab:ir_aligned 0.846/15 — AMBIGUOUS) |
| 2357 | (held-out confuser FP cut, truncated) | gray | gray | tab:models_evaluated | gray | mlp_v5_gray row (text truncated in render) |
| 2364 | CV F1 0.880 | mlp_v4 CV | — | tab:models_evaluated | RGB | superseded predecessor (context only) |
| 2363 | F1 0.9955 CV-only | distill_mlp_261feat | — | tab:models_evaluated | RGB | superseded CV winner (context) |
| 2399 (glossary) | (p3+p5 ROI, supersedes patch) | — | — | glossary mlp_v5 | RGB | def, no metric |
| 2400 (glossary) | (two scalers) | — | — | glossary mlp_v5_ir_aligned | IR | def, no metric |
| 2250 (P(drone) thr defs) | 0.25 / 0.05 / 0.25 | thresholds | — | filter op points | RGB/IR | RGB 0.25; IR thermal 0.05; IR gray 0.25 |

---

## AMBIGUOUS / NEEDS-HUMAN

1. **CBAM held-out F1/FP discrepancy across the file.**
   - `tab:ir_aligned` (line 1777) and Abstract-caption say **F1 0.846, FP 15**.
   - Contribution 4 (line 222) says **0.699 -> 0.846, FP 48->15**.
   - Glossary `tab:models_evaluated` row (line 2356) AND user-memory say **0.699 -> 0.841, FP 48->13**.
   These are two different numbers (0.846/15 vs 0.841/13) for the SAME held-out CBAM result. When the filter is retrained, BOTH must be updated to the new value, and the existing inconsistency flagged. Cannot resolve which is canonical without the source eval cache.

2. **Line 1764 grayscale halluc (37.2% vs 1.8%, ~20x)** belongs to the bare `v3b` DETECTOR (the harvest surface), not to the filter itself. Included because it is the operating-point context the IR filter is calibrated against; a filter swap may or may not touch it. Human call on whether it must change.

3. **Contribution 4 (line 220) grayscale transfer numbers (0.636, 0.837/0.840, 3.2x)** are the IR DETECTOR's cross-modal transfer (RQ4), NOT the mlp filter. Excluded from the swap set, listed only so they are not confused with aligned-filter grayscale numbers. Confirm they stay fixed.

4. **Line 639 LDA 0.952 (RGB) / 0.981 (IR)** are separability of the detector feature space the filters read, measured pre-filter. They describe the substrate, not the trained filter's operating metrics — a retrained filter may not change them. Human call.

5. **The robust8 / robust8-nr revision (the `filt`/`clf->filt` paired ablation tables, `fig:filter_operating`, Part-C confuser FP tables, grayscale operating sweep) is NOT in this file.** Per user-memory it is the SHIPPED thesis state. **The swap team must find and audit that revision too** — it will contain the bulk of composed-pipeline `filt`-cell numbers that this working copy lacks. This is the single biggest coverage gap and needs human direction on which file is the true live thesis.

---

## EXCLUDED (patch verifier / classifier / detector — NOT the mlp filters), for completeness

These tables/numbers are confuser-related but driven by the **MobileNetV3 patch verifier v2** or the **trust classifier/robust6/robust8** or **bare detectors**, which the task excludes:
- `tab:cum_confuser` (1501), `tab:cumulative_svanstrom` (1523), `fig:cumulative_confuser`, `fig:svanstrom_by_cat` — patch v2 + fnfn classifier
- `tab:ood_rgb_drone` (1570), `tab:ood_rgb_confuser` (1591), `tab:ood_ir` (1611) — Roboflow + patch v2
- `tab:patch_sweep` (1146), `tab:patch_audit` (1180), `fig:patch_sweep`, `fig:patch_catchbar` — patch v2 thresholds/catch
- `tab:cascade_perframe` (1855), `tab:cascade_segment` (1888), `tab:cascade_percategory` (1929), `tab:cascade_classifier_drone` (1965), `tab:cascade_classifier_fpr` (1980) — temporal + patch veto
- `tab:robust6_pipeline` (1087), `tab:rgb_comparison` (805) — classifier / detector stance
- v1/v2 patch F1 0.9241 / 0.9311 (1128-1130), catch bird64/air52/heli71 @patch_thr=0.5, 5.4% veto — patch verifier
- robust8 grayscale recall 0.577->0.681->0.738, fire 0.006->0.046 (1107) — trust classifier, not filter

---

## TOTAL COUNT

- **thesis_working.tex in-scope filter numbers (RGB+IR+gray):** ~95 distinct numeric values across the rows above (RGB filter ~52, IR thermal filter ~24, IR/grayscale + dedicated mlp_v5_gray ~16, shared LDA/feature ~3).
- **thesis_chapters.tex in-scope filter numbers:** **0** (file predates the distilled filters; only patch verifier present).
- **AMBIGUOUS items:** 5.
- **Biggest gap:** the robust8/robust8-nr `filt`-cell paired-ablation revision is absent from both scanned files and must be located separately.

**Registry written to:** `docs/analysis/2026-06-17_filter_thesis_map_numbers.md`
