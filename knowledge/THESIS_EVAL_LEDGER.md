# Thesis Evaluation Ledger

**Purpose.** One traceable map of *every evaluation that appears (or should appear) in the thesis*,
organised by pipeline stage, so we know (a) what config/frame-count each eval must use, (b) which eval
rows / caches back it, and (c) what is still missing. This is the hand-maintained index; the measured
numbers live in `evals.csv`, the configs in `eval_configs.csv` (both written only via `kb.py`).

Pipeline stages (a detection survives left-to-right):
`detector → [trust classifier routes modality] → [confuser verifier filters] → alert`

- **Stage 1 — Model alone** (bare detector, no downstream)
- **Stage 2 — Model + trust classifier** (classifier routes; no verifier)
- **Stage 3 — Model + trust classifier + filter** (classifier → verifier = the full deployed pipeline)
- **Stage 4 — Model + filter** (detector → verifier, *no* classifier; isolates the verifier: patch vs mlp)

---

## §0 — EVAL STANDARD (the constraint: one config + frame budget per dataset)

> **⚠️ TWO-TIER RUN PLAN (2026-06-10).** **Tier 1 — 4k STRIDED screening (NOW):** ~4000 frames/surface
> sampled by **even stride (every Nth frame — NOT the first 4k)**; all frames if a surface is <4000.
> Yields per-frame P/R/F1 to decide whether the full run is warranted. **Temporal is a PLACEHOLDER in
> Tier 1 — segment-level voting needs CONSECUTIVE frames and cannot run on a strided sample.** **Tier 2 —
> FULL FRAMES (LATER, gated on Tier-1 results):** every consecutive frame → the FINAL thesis numbers, and
> the ONLY tier that produces the **temporal/segment-level** rows (C→F→T vs F→C→T).
>
> **⚠️ TIER-2 RE-SCOPED (2026-06-10, after measuring 0.77 fps paired on Tier-1 antiuav; I/O-bound, GPU ~2%).**
> Naïve full-frames over all 9 surfaces ≈ **52 GPU-hours**; Anti-UAV alone = 85,374 paired ≈ 31 h on a
> **saturated 0.994 surface** — indefensible. Therefore: **(a) Tier-1 strided n=4000 IS the final thesis
> number for every non-sequence surface** (CI ≤ ±1.5 pp at n=4000; selcom 311 / rgb-confuser 2,633 /
> gray-confuser 2,633 are already FULL below the cap; temporal is meaningless on photo-style splits and
> confuser image piles). **(b) Anti-UAV full is DROPPED** — Tier-1 4k + a saturation footnote is the
> thesis treatment. **(c) Tier 2 = SVANSTRÖM ONLY (paired + gray), consecutive frames** (~10.4 h + ~5 h
> full at the measured rate, or ~8 h total with complete-sequence subsetting — segment voting is valid on
> complete sequences) → yields the temporal rows + svanström's final digits. Levers: drop patch_v2 scoring
> at Tier-2 (its comparison is told at Tier-1), stage svanström to a local SSD. Per-frame Tier-1 numbers:
> ALWAYS print n and label "strided"; only svanström's per-frame digits may be re-stated by Tier 2.
>
> **Frame budget LOCKED (2026-06-10):** Tier 1 = **~4k cap, strided, all-if-smaller** (caps the n column
> below); Tier 2 = the **full n in the table below — svanström rows only** (see re-scope). ALWAYS print
> the actual n used.
>
> **IMGSZ STANDARD = 640.** `imgsz=1280` is used ONLY where a documented reason justifies it:
> • **Svanström-RGB = 1280** — *dimensions* (native 640×512; small drones fall below the resolvable floor
>   at 640 — `retrained_v2` collapses R 0.961→0.072 @640).
> • **SelCom-RGB = 1280** — *poor CCTV quality* (small, distant, low-fidelity targets).
> The rule is about **native source resolution**, not drone size: 1280 only where the source is small/low-res
> (Svanström 640×512, SelCom CCTV) so upscaling resolves sub-floor drones. Everything with normal high-res
> sources is **640**: **Anti-UAV (1920×1080; CHANGE: thesis used 1280 → 640)**, **video (normal 16:9, 720p/
> 1080p)**, and **all confuser surfaces (RGB + IR)**. The IR detector runs at **640 everywhere** (thermal
> native resolution). **Consequence:** the Anti-UAV-RGB baseline (0.9936 @1280) and the confuser-zoo fire
> rates must be **re-stated @640**; the IR side is already 640. **imgsz standard now fully locked.**

**Rule:** every table touching a dataset MUST use that dataset's single canonical config below. Numbers
at any other stride/imgsz/scoring are NOT comparable and may not sit in the same table (the 28-pp
scoring-audit lesson). Canonical configs are rows in `eval_configs.csv`.

| Dataset | Canonical `eval_config` | n_frames | imgsz | scoring | Status |
|---|---|---|---|---|---|
| Svanström (paired) | `svan_iop_1280_s9` | **3190** (stride 9) | 1280 | IoP@0.5 | LOCKED (thesis head-to-head) |
| Anti-UAV (paired) | `antiuav_iou_640_4k` (new) | **4000** (strided) | **640** | IoU@0.5 | LOCKED 2026-06-09 (n=4000; imgsz 640 — CHANGE from thesis 1280) |
| RGB in-domain test | `rgb_dataset_iou_640` (→ full) | **17209** (full test) | 640 | IoU@0.5 | LOCKED 2026-06-09 (full test split) |
| IR in-domain test | `ir_final_640` | 9612 (full test) | 640 | IoU@0.5 | LOCKED |
| RGB confusers | `confuser_zoo_640` (was `_1280`) | 2633 (full) | **640** | fire-rate | LOCKED 2026-06-09 (imgsz 640 standard; restate from 1280) |
| RGB bird confuser (NEW) | `rgb_bird_confuser` | 1731 | **640** | fire-rate | LOCKED 2026-06-09 (eval standalone; merge into RGB confusers only if it earns it) |
| IR confusers — thermal (NEW) | `IR_confusers` | 5938 (airplane 4281/bird 1200/heli 457) | 640 | fire-rate | LOCKED 2026-06-09 (replaces cbam-180: bigger + not in-domain) |
| IR confusers — grayscale | gray-confuser cfg (rgb\_confusers→gray→v3b) | 2633 | **640** | fire-rate | LOCKED 640 (confuser standard) |
| SelCom CCTV | `selcom_iop_1280` | 311 (full val) | 1280 | IoP@0.5 | LOCKED |
| Real-video drone | `video_drone_iop` / `pipe_video_drone_iop` | 1359 (full clips) | **640** | IoP@0.5 | LOCKED 2026-06-09 (normal 16:9 high-res; CHANGE from per-model 1280) |
| Real-video confuser | `video_confuser` / `pipe_video_confuser` | 1250 (full clips) | **640** | fire-rate | LOCKED 2026-06-09 (normal 16:9 high-res) |
| Roboflow OOD (RGB/IR) | `roboflow_{rgb,ir}_drone_640`, `..._confuser_640` | per-set | 640 | IoU/fire | LOCKED |

**⚠️ The 2026-06-09 routing run (`_routing_pipeline_cmp`, n=4000) does NOT match this standard** for
Svanström (it used stride≈7/n4000, the standard is stride-9/n3190) or Anti-UAV (stride≈21/n4000 vs
stride-5). It is *internally* comparable but must be **re-run at the standard** before its numbers enter
the thesis. Open decisions above (Anti-UAV n, RGB-confuser imgsz, RGB in-domain n) gate that re-run.

Production stack under test everywhere: **RGB detector `ft4`** (`Yolo26n_selcom_confuser_ft4_1280`, the
SelCom fine-tune) · **IR detector `v3b`** · **trust classifier `robust8`** · **verifiers `mlp_v5` (RGB) +
`mlp_v5_ir_aligned` (IR)**. Patch verifier (`patch_v2`) is the superseded comparison in Stage 4.

---

## § THESIS EVAL STRUCTURE (LOCKED 2026-06-09) — how the empirical chapter is organised

> Supersedes the loose surface set. The empirical chapter is **three parts on two distinct axes**
> (never mix the axes): Part A = **A/B across model versions**; Part B = **ablation across pipeline
> stages**; Part C = **the highlight finding**.

**Narrative spine (LOCKED 2026-06-09, UPDATED): A+D — the SYSTEM is the headline.** A deployable
dual-modality drone-detection pipeline **+ operator GUI** (A), produced and maintained by a
**self-improving loop** (D): a human-in-the-loop data engine + a statistics-first diagnostic (Model
MRI) that derive cheap, robust components and catch regressions. **Contributions: C1** pipeline+GUI ·
**C2** Model MRI · **C3** HITL data engine. **Findings (NOT the spine):** grayscale cross-modal
transfer · the 28-pp scoring-rule audit · the speed wins. *(Supersedes the earlier "grayscale =
headline" framing — grayscale is now a finding, not the spine.)*

- **Part A — Per-model, IN-DOMAIN (axis = A/B, model version).** BARE detectors only. Each model on
  its OWN test split + the public benchmarks it belongs on. **Three versions per model**; any other
  model is a *cameo*, cited only to make a point (e.g. "retrained_v2's recall collapse is why we
  built a filter, not a bigger detector"). No pipeline here — this is detector quality.
- **Part B — FULL-PIPELINE ABLATION (bare → +classifier → +verifier → +temporal).** Detector fixed at
  production. Full fusion pipeline is **PAIRED-ONLY (Anti-UAV, Svanström)** + grayscale-where-it-works;
  reported **per-frame** for the 4k screening run. ⚠️ **Temporal needs CONSECUTIVE frames → it CANNOT
  run on a STRIDED sample. Segment-level (3-frame 2-of-3 = the PySide GUI alert-gate; reuse
  `eval/temporal_ablation.py`, match `ir_gui/fusion/engine.py`) is a PLACEHOLDER until the FULL-FRAMES
  run.** **Order A/B (full run only): C→F→T vs F→C→T** (T always last; C,F per-frame, T aggregates).
  Two faces:
  - **B1 — drone-positive = COST:** Anti-UAV, Svanström → does the pipeline preserve recall?
  - **B2 — OOD confusers = BENEFIT:** RGB confusers, IR confusers → FP-reduction table (no P/R/F1).
  Component A/B inside B: router **sa32 + fusion_no_fn (sad; 32–40 feat, expensive) → robust8 (happy;
  8 feat, ~404× faster)**; verifier **bare / patch_v2 (5-class MobileNetV3, fail-open) / mlp_v5**.
  (Solo drone sets — RGB test, IR test, SelCom — get bare detector in Part A + verifier-only Stage-4;
  no fusion routing, single-modality. The −11pp in-domain RGB verifier carve-out is reported there.)
  - **SCORING (per-modality, NO UNION):** no-classifier → each modality vs its OWN GT separately;
    trust-RGB → RGB vs RGB GT only; trust-IR → IR vs IR GT only; trust-both → both; never score a
    rejected modality, never merge modalities into one number.
- **Part C — GRAYSCALE = BEST-RESULTS FINDING (highlight, not the spine).** Show ONLY the working
  config: **IR-on-grayscale + filter, classifier bypassed** (the router is thermal-trained → degrades
  on grayscale; one honest line, no collapse table). Evidence = svanström_gray P/R/F1 + the **MRI
  gray→thermal alignment** (AUROC 0.500→0.919, centroid cosine 0.012). Back with a **bootstrap CI**.

### Canonical thesis eval surfaces — the ONLY datasets in the empirical chapter (8)
1. RGB model test split — `rgb_dataset_iou_640` (≤4000, **4k cap** — supersedes "full 17209")
2. IR model test split — `ir_final_640` (≤4000, **4k cap** — supersedes "full 9612")
3. Anti-UAV — `antiuav_iou_640_4k` *(saturated → role = "no-harm" control, not an improvement story)*
4. Svanström — `svan_iop_1280_s9`
5. SelCom CCTV — `selcom_iop_1280` (minimal; deployability anchor)
6. RGB confusers — `confuser_zoo_640`
7. IR confusers — `IR_confusers` @640
8. Grayscale — `svanstrom_gray` + `gray_confuser` (Part C highlight)

**Internal / NOT a thesis surface** (eval-only or supporting, never headline): `rgb_bird_confuser`
(internal confuser probe), real-video drone/confuser (slow, frame-level redundant), Roboflow OOD
(optional "other OOD" citation only). **Birds are a CATEGORY** inside Svanström + RGB/IR confusers,
**not a standalone dataset** — tell the bird story per-category across the canonical confuser sets.

### Three versions per model (the A/B set; everything else is a narrative cameo)
| Model | reference | the lesson | production | Narrative use |
|---|---|---|---|---|
| RGB detector | baseline | retrained_v2 (hi-P / recall-collapse) | ft4 (selcom_1280) | recall↔precision continuum; retrainedv2 collapse → motivates the filter |
| IR detector | V2 | V5 (HITL regression) | v3b | HITL co-development; regression reported, not hidden *(confirm exact triple vs `tab:ir_evolution`)* |
| Trust classifier | sa32 | robust6 | robust8 | leakage-aware feature selection; grayscale-recall recovery |
| Confuser verifier | bare | patch_v2 (CNN) | mlp_v5 / mlp_v5_ir_aligned | feature-reuse MLP replaces the CNN at 46–72× speed |

### Where each model is SCORED (this is the fix for "evals all over the place")
- **Detectors** share test splits → Part A is a true **A/B** (comparable P/R/F1 across versions).
- **Classifier + verifier** have **NO comparable standalone split** (each version trained on
  different data → own-held-out accuracy is NOT cross-comparable). They are scored by **pipeline
  EFFECT in Part B** (OOD FP-reduction + recall kept), NOT a standalone accuracy A/B. Any
  standalone classifier/verifier accuracy lives in the appendix with the non-comparability caveat.

---

## §1 — Stage 1: MODEL ALONE (bare detector)

| Surface | Config | Eval id(s) | Thesis table | Have? |
|---|---|---|---|---|
| Svanström RGB (baseline/hardneg/retrained_v2) | `svan_iop_1280` | `rgb_svan_baseline`, `rgb_svan_hardneg`, `rgb_svan_retrainedv2` | `tab:rgb_comparison` | ✓ |
| Svanström RGB ft4 bare | `svan_iop_1280_s9` | `v5_svan_bare` | `tab:distill_verifier` | ✓ |
| Svanström IR (v3b) | `svan_ir_iop_640` | `ir_v3b_svan640_may10` | numerical-comparison | ✓ |
| Anti-UAV RGB (baseline) | `antiuav_iou_1280_s5` | `rgb_antiuav_baseline` | `tab:numerical_comparison` | ✓ |
| Anti-UAV IR (v3b) | `antiuav_iou_640_may10` | `ir_v3b_antiuav640_may10` | numerical-comparison | ✓ |
| RGB in-domain (ft4 bare) | `rgb_dataset_iou_640` | `v5_rgbds_*` (bare row) | `tab:distill_verifier` | ✓ |
| IR evolution V2→v3b | `ir_final_640` | `ir_final_*` (×8 versions) | `tab:ir_evolution` | ✓ |
| SelCom CCTV (ft2) | `selcom_iop_1280` | `selcom_*` | `tab:selcom` | ✓ |
| Roboflow OOD drone RGB | `roboflow_rgb_drone_640` | `rob_rgb_baseline`, `rob_rgb_retrainedv2` | `tab:ood_rgb_drone` | ✓ |
| Roboflow OOD drone IR | `roboflow_ir_drone_640` | `ir_ood_cbam_raw`, `ir_ood_night_raw` | `tab:ood_ir` | ✓ |
| Confuser zoo (bare fire 52.1%) | `confuser_zoo_1280` | (S1 row of `clfzoo_*`) | `tab:cum_confuser` | ✓ |
| Real-video drone (6 RGB/IR modes) | `video_drone_iop` | `vid_drone_*` (6) | `tab:realvideo_master` | ✓ |
| Real-video confuser (6 modes) | `video_confuser` | `vid_conf_*` (6) | `tab:realvideo_master` | ✓ |

---

## §2 — Stage 2: MODEL + TRUST CLASSIFIER (classifier routes; no verifier)

| Surface | Config | Eval id(s) | Thesis table | Have? |
|---|---|---|---|---|
| Classifier own held-out (acc / macro-F1) | `clf_own_holdout` | `own_phase1_*`, `own_rgb_reliability`, `own_fnfn_v3more`, `own_mlp_v5_p3p5` … (26) | `tab:classifiers` | ✓ (NOT cross-comparable) |
| 3-way drone/confuser/bg | `clf_3way_300` | `clf3_retrainedv2` | classifier eval | ✓ |
| Confuser zoo, classifier-only stage | `confuser_zoo_1280` | `clfzoo_sa32` (→20.5%), `clfzoo_fnfn` (→1.6%), `clfzoo_control40` | `tab:cum_confuser` | ✓ |
| Svan/Anti-UAV classifier-only (robust6, paired) | `svan_iop_email`, `antiuav_iou_email` | `svan_classifier_robust6_ta`, `antiuav_classifier_robust6_ta` | (email recompute) | ✓ |
| **robust8 classifier-only, all surfaces** | routing (n=4000 ⚠️) | cached `_routing_pipeline_cmp` `clf_only[robust8@τ]` cells | NEW master table | cached, **re-run at standard** |

---

## §3 — Stage 3: MODEL + TRUST CLASSIFIER + FILTER (the full deployed pipeline)

| Surface | Config | Eval id(s) | Thesis table | Have? |
|---|---|---|---|---|
| Svanström full cascade (sa32/control40 + patch) | `svan_iop_1280_s9` | `svan_s3_sa32_thr08`, `svan_s3_control40_thr09` | `tab:patch_sweep` | ✓ (comparison config) |
| Confuser zoo S3 (sa32 →10.3 / fnfn →0.8) | `confuser_zoo_1280` | `clfzoo_*` (S3 rows) | `tab:cum_confuser` | ✓ (comparison config) |
| Real-video FULL cascade, segment-level | `pipe_video_drone_iop`, `pipe_video_confuser` | `pipe_vid_*_seg`, `pipe_vid_*_pf` | `tab:cascade_segment` | ✓ but **sa32+patch, not production** |
| Paired Svan/Anti-UAV, new stack (robust6+mlp) | `svan_iop_email`, `antiuav_iou_email` | (email recompute `comparison_*.md`) | — | ✓ cached |
| **robust8 → mlp_v5 cascade, all surfaces** | routing (n=4000 ⚠️) | cached `_routing_pipeline_cmp` `clf->filter[robust8@τ]` | NEW master table | cached, **re-run at standard** |
| **ir_dset_final — full pipeline** | `ir_final_640` | — | — | **✗ TO RUN** |
| **rgb_dataset — full pipeline** | `rgb_dataset_iou_640` | — | — | **✗ TO RUN** |
| **IR-thermal confuser (cbam) — full pipeline** | `cbam_ir_640` | — | — | **✗ TO RUN** |

---

## §4 — Stage 4: MODEL + FILTER (detector + verifier, NO classifier — isolates patch vs mlp)

The offline verifier matrix (`eval/results/_offline_pipeline/offline_eval_results.md`,
`pipeline_eval_offline.py`, n=1000/surface ⚠️) already computes **bare / +patch / +mlp** for both
modalities on 12 surfaces. This is the cleanest verifier ablation.

| Surface | Config | Eval id(s) / cache | Thesis table | Have? |
|---|---|---|---|---|
| Svanström RGB ft4: bare/patch/mlp | `svan_iop_1280_s9` | `v5_svan_bare/patch/mlp` (+offline `svanstrom`) | `tab:distill_verifier` | ✓ |
| Anti-UAV RGB ft4: bare/patch/mlp | `antiuav_iou_640_s5` | `v5_antiuav_bare/patch/mlp` | `tab:distill_verifier` | ✓ |
| RGB in-domain: bare/patch/mlp | `rgb_dataset_iou_640` | `v5_rgbds_*`, `v52_rgbds_remine` (+offline `rgb_dataset_test`) | `tab:distill_verifier` | ✓ (recall carve-out) |
| RGB confusers: bare/patch/mlp (216→104→16 FP) | `confuser_test_640` | `v5_confuser_mlp` (+offline `rgb_confuser`) | `tab:distill_verifier` | ✓ |
| CBAM IR thermal: bare/patch/aligned (0.699→0.846) | `cbam_ir_640` | `ir_aligned_cbam_heldout` (+offline `cbam`) | `tab:ir_aligned` | ✓ |
| Grayscale confuser: bare/patch/aligned_gray (143→20 FP) | `rgb_gray_heldout_640` | `ir_aligned_gray_heldout` (+offline `gray_confuser`) | `tab:ir_aligned_gray` | ✓ |
| IR in-domain (ir_dset_final): bare/patch/aligned | `ir_final_640` | offline `ir_dset_final` | — | ✓ cache only (offline n=1000) |
| Patch version sweep v1–v4 | `svan_iop_640_may10` | `patch_v1/v2/v3/v4_svan640` | patch history | ✓ |

---

## §5 — What is missing / to run (after the §0 standard is locked)

1. **Full-pipeline (Stage 3) on `ir_dset_final`, `rgb_dataset`, `cbam`** — not in the routing harness's
   surface list. Add them + re-run. (Note: on single-modality sets fusion is degenerate — the classifier
   gates the one present modality / grayscale fallback; decide if Stage 3 there is meaningful vs Stage 4.)
2. **Re-run the routing pipeline at the §0 standard** (Svanström stride-9/n3190; Anti-UAV at the locked n)
   so Stage 2/3 robust8 numbers are comparable to the existing Stage 1/4 tables.
3. **Patch verifier in the routing harness** — it currently runs only `mlp` for Stage 3/2; to compare
   patch-vs-mlp *inside the full pipeline* the harness needs the patch CNN added (it already loads images).
4. **Unify n across Stage 4 (offline n=1000) and Stages 2/3 (routing n=4000)** — pick the §0 n and re-run
   both so the verifier-ablation and the pipeline tables share frames.

## Maintenance
- New eval → record it in `evals.csv` (`kb.py record evals ...`) with the canonical `config_id` from §0,
  then add/refresh its row here. Never put a number here that isn't backed by an `evals` row or a named cache.
- Config changes / a newly-locked standard → update §0 here AND log it in `knowledge/DECISIONS.md`.
