# Thesis integration checklist — production filter stack (v4 + thermal-only + balanced-gray)

**Date:** 2026-06-18 · **Status:** Part C prep (zero-GPU). The GUI already ships this stack; the thesis
still documents the OLD stack. This is the apply guide for the atomic integration pass. **Numbers source
= `ES_Drone_Detection/filter_PRODUCTION_swap_tables.md`** (shipped→final, every filter cell). Cell→chapter
line + audit CHECK label = `docs/analysis/filter_swap_registry_{empirical,methodology}.md`.

Production stack: RGB `mlp_v5_v4.pt` @0.25 · IR thermal `mlp_aligned_thermalonly.pt` @0.05 · IR gray
`mlp_aligned_gray_balanced.pt` @0.25. Manifest: `thesis_eval/_filter_swap/final/swap_manifest.json`.

> **Apply atomically** (freeze dirs + harness repoint + .tex numbers + audit constants together), else
> the audit goes red. Most filter cells are UN-AUDITED → the registries are the safety net, not the audit.

> ⚠️ **LEAKAGE — read `docs/analysis/2026-06-18_filter_provenance_train_heldout.md` FIRST.**
> **RULE (user, 2026-06-18): the existing thesis tables + eval surfaces DO NOT CHANGE.** The ablation
> tables keep their surfaces; cells just update to the new filter's value (cache-measured, as before).
> The **held-out results are ADDED only in the mlp filter's own sections** (§verifier_results,
> §ir_xmodal_verifier, §grayscale_verifier, `tab:ir_aligned`, `tab:distill_verifier`), where we also
> state each filter's **train dataset + held-out test set**. Two harness caches are the new filters'
> TRAINING data, so in the FILTER SECTION disclose this and present the held-out generalization:
> - `ir_confusers` cache = `IR_confusers/train` (thermal-only's data) → held-out val/test: **shipped 90 → cbam 22**.
> - `rgb_bird_confuser` cache = full bird.v1i incl. train → held-out bird.v1i TEST: **shipped 91 → v4 30**.
> These tell the same story as the cache numbers (user: fine since similar). Svanström RGB/IR = in-sample.

## 1. Regenerate canonical evidence (zero-GPU) — freeze `_filter_swap/final` → committed dirs
Re-run the bundle/replays into the committed dirs (per `runs/README.md`): `thesis_eval/results/`,
`results_noreject/`, `results_clean/`, `runs/results_dut/` + `temporal_results`, `notes_round1_results`,
`leakage_controlled`. Regenerate `fig_filter_operating` via `eval/filter_operating_sweep.py` (writes the
committed figure — run with the production weights live, i.e. after the harness-constant repoint in §4).

## 2. Audit CLAIMED constants to update — `thesis_eval/_audit_headline_numbers.py` (shipped → final)
| CHECK | old → new |
|---|---|
| NR svan filt->clf F1 (PROD) | 0.9439 → **0.9459** |
| NR svan composed F1 | 0.9302 → 0.9308 |
| svan composed F1 (r8) | 0.9485 → 0.9480 |
| NR antiuav composed F1 | 0.9841 → 0.9842 |
| rgbtest mlp F1 | 0.8092 → **0.9222** |
| irtest mlp F1 | 0.9578 → **0.9421** |
| selcom mlp F1 | 0.6115 → 0.6115 (unchanged) |
| NR dut composed F1 / filt->clf F1 | 0.792→0.790 / 0.8373→0.8351 |
| CLEAN svan / auv pipeline F1 | 0.9348→0.9340 / 0.9862→0.9861 |
| rgbconf mlp fire / FP | 0.0106→**0.0144** / 29→39 |
| grayconf mlp fire / FP | 0.0076→0.0053 / 21→**15** |
| NR ir_conf fire / FP | 0.237→**0.0278** / 968→**113** (table UNCHANGED — same cache surface; held-out 90→22 ADDED in filter section, not here) |
| irconf composed r8 fire / FP | 0.2167→0.0243 / 885→99 (table unchanged) |
| FIG rgb/gray recall@0.25, fire | **re-run filter_operating_sweep** (pending §1) |
| CBAM aligned (tab:ir_aligned) | **RESOLVED — no extra GPU**: cbam **R 0.967 / FP 6** @0.05 (held-out CBAM valid; was balanced 48→15). kb `cbam_heldout_thermalonly`. |
Plus the per-size `SZ rgbtest *` filt-recall CHECKs (from regenerated `notes_round1_results`).

## 3. CITED_PATHS additions — `_audit_headline_numbers.py`
Add the three versioned production weights: `models/verifiers/rgb_v5/mlp_v5_v4.pt`,
`models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt`, `.../mlp_aligned_gray_balanced.pt`.

## 4. Harness default repoint — `thesis_eval/pipeline_eval_unified.py` (lines ~52–54, 58)
`MLP_V5 → mlp_v5_v4.pt`, `ALIGNED → mlp_aligned_thermalonly.pt`, `ALIGNED_GRAY →
mlp_aligned_gray_balanced.pt`; `IR_THR_MLP` stays 0.05. Mirror in `gui` already done.

## 5. Chapter NUMBER swaps — `docs/thesis_working_distilling_overleaf/chapters/`
Apply every changed cell from the swap-map doc to its chapter:line (registry maps them). Headlines:
abstract/RQ/conclusion ir-confuser-fire (23.7%→**2.8%**), rgb_dataset_test recall (0.69→**0.89**),
svan production F1 (0.944→0.946). `tab:ablation_confusers`, `tab:ablation_solo`, `tab:ablation_svanstrom/
antiuav/dut`, `tab:per_size`, `tab:temporal_production`, `tab:ir_aligned`, `tab:distill_verifier`.

## 6. Chapter CLAIM reframes (prose that FLIPS — careful pass)
- **Retire "one network, two scalers"** → IR verifier is now **two heads**: thermal-native
  (`thermalonly`, CBAM) + grayscale-aligned (`balanced`). (methodology §sec:grayscale_verifier ~696,
  §sec:ir_xmodal_verifier; glossary; `fig_pipeline.tex` router label.)
- **Thermal-airplane hole largely CLOSED** by thermal-native training (thermal confuser fire 23.7%→2.8%,
  airplanes no longer "resist"). Rewrite §limits + the "thermal confusers resist ~39%" passages.
- **RGB bird carve-out fixed**: v4 (birdsplit) cuts rgb_bird_confuser 199→31; update the carve-out /
  fail-open-not-adopted narrative.
- **Unchanged (keep):** "IR filter inert on Svanström" (all heads identical there); recall-safe framing.

## 7. REMAINING regen (mostly resolved)
- **CBAM held-out — DONE** (in the provenance doc): cbam R 0.967 / FP 6 @0.05. No extra GPU.
- **Held-out re-mines (zero-GPU, must do):** IR_confusers val/test (`eval/eval_ir_heldout.py`) and
  bird.v1i TEST (`eval/eval_birdtest_heldout.py`) → the honest confuser numbers that replace the leaky
  caches. These are the IR/bird suppression figures the thesis should cite.
- `filter_operating_sweep.py` figure regen (writes committed figure; run after §4).

## 8b. RESOLVED — filters are FINAL (user, 2026-06-18)
No with-gray build. The thermal-only (`--no-gray`) + balanced-gray **two-net IR is production** and final.

## 8c. THESIS RULE (user, 2026-06-18) — tables unchanged; ADD held-out in filter sections only
- **Do NOT change the existing tables or eval surfaces.** Ablation cells just take the new filter's
  cache-measured value (the swap-map numbers). The `ir_confusers`/`rgb_bird_confuser` table cells stay
  as-is (cache surface).
- **ADD held-out results ONLY in the mlp filter sections** (§verifier_results, §ir_xmodal_verifier,
  §grayscale_verifier, `tab:ir_aligned`, `tab:distill_verifier`): IR confuser shipped 90 → cbam 22
  (val/test), bird shipped 91 → v4 30 (bird.v1i TEST), CBAM valid R 0.967/FP 6 — each labelled as the
  held-out **test split**, with the filter's **train dataset + test set** named (provenance doc §1–3).
- State that model numbers are reported on each dataset's **test split**; flag **Svanström RGB/IR** as
  the in-sample exception (no split — fair for Δ only).

## 8. PROVENANCE + LEAKAGE (per user, 2026-06-18 — do during the prose pass)
For **every part the mlp filter is mentioned**, state the **train dataset** and the **held-out/eval
dataset**. Seed + open gaps in `docs/analysis/2026-06-18_filter_provenance_train_heldout.md`.

## PARITY RESULTS — every old-filter number has a v4/thermal-only counterpart (ALL ZERO-GPU)
Confirmed 2026-06-18: the offline cache (`eval/results/_offline_pipeline/cache/*.pkl`) stores `feats`+`patch`
per detection for all 14 parity surfaces → score any filter zero-GPU. **No GPU run needed.** Tools:
`pipeline_eval_offline.py` (now THESIS_* env-overridable) + `filter_acceptance_eval.py --candidate`.
Saved: `thesis_eval/_filter_swap/final/offline_matrix_{shipped,v4}.txt`.

**`tab:distill_verifier` (RGB mlp vs patch), OLD → v4:**
| surface | OLD mlp | v4 mlp |
|---|---|---|
| svanstrom (iop) F1 | 0.869 | 0.861 |
| antiuav_rgb F1 | 0.987 | 0.984 |
| selcom_val F1 | 0.612 | 0.612 |
| rgb_dataset_test F1 (R) | 0.809 (0.694) | **0.916 (0.874)** |
| rgb_confuser FP | 16 | 21 |
| rgb_bird_confuser FP | 199 | **31** |

**`tab:ir_aligned` (IR aligned → thermal-only), OLD → new:**
| surface | OLD F1/FP | thermal-only F1/FP |
|---|---|---|
| CBAM | 0.846 / 15 | **0.935 / 6** (R 0.917→0.967) |
| ir_dset_final | 0.977 | 0.957 (R 0.965→0.928) |
| ir_video | 0.975 | 0.975 |
| antiuav_ir | 0.957 | 0.957 |
| svanstrom_ir | 0.953 | 0.953 |
| ir_confusers FP | 220 | **26** |
> Gray rows: the offline matrix scores gray with the THERMAL head (config quirk). Use the **production
> gray head (balanced @0.25)** numbers from the harness instead: gray_confuser 21→15, svanstrom_gray recall.

## Done already (provenance substrate)
kb models (`mlp_v5_balanced_v4`, `mlp_aligned_thermalonly`, `mlp_aligned_gray_balanced`; old two →
production=no), evals (3), ledger `filter-stack-v4-thermalonly-promoted`; run manifest with weight SHA-256.
