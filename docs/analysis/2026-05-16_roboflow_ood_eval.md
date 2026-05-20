# Roboflow OOD eval — RGB(baseline, retrained_v2) + IR + patch verifier

**Date:** 2026-05-16
**Run command:** `python eval/run_roboflow_eval.py --full`
**Stack tested:** YOLO + patch verifier (v2 backup) on alert-gate cascade. **No trust classifier.** **imgsz=640** (hardcoded in `run_roboflow_eval.py:204`). Temporal alert gate enabled but mostly inert on these still-image datasets.
**Source CSV:** `eval/results/roboflow_ood/summary.csv`

## TL;DR

1. **RGB baseline beats retrained_v2 on the drone OOD set even at imgsz=640.** Baseline R=0.746, retrained_v2 R=0.726 (raw, drone class, all splits). The Svanstrom-only retrained_v2 collapse does **not** generalize — on Roboflow drones (larger, more typical) retrained_v2 is fine; it only breaks on Svanstrom's tiny drones. This is consistent with the imgsz=1280 Svanstrom finding being a small-drone-resolution problem, not a generic model defect.
2. **Patch verifier costs net F1 on this OOD distribution.** On every drone setting (RGB baseline, RGB retrained_v2, IR), filtering reduces F1 by 1–5 pp because the drone-TP veto rate (~5–8%) outweighs the modest confuser catch on this distribution.
3. **RGB patch verifier under-catches OOD airplanes catastrophically.** Suppression on Roboflow airplane confusers is **~2–4%** (vs 52% on Svanstrom airplanes at imgsz=1280). Confirms the verifier is severely distribution-bound — Svanstrom airplane crops ≠ Roboflow airplane crops, and the verifier knows the difference.
4. **retrained_v2 RGB hallucinates more on airplanes (+25% raw FP vs baseline), less on helicopters/birds.** Net halluc count: baseline 1887 vs retrained_v2 2048 across the three RGB confuser datasets — retrained_v2 is **not** the universal confuser-suppressor that Svanstrom@1280 implied.
5. **Patch verifier suppresses retrained_v2's helicopter FPs at 1/4 the rate of baseline's (8% vs 36%).** Verifier was calibrated against baseline RGB's FP distribution; retrained_v2's helicopter false-positives look different and slip through.
6. **IR model has severe OOD recall problems.** On `ir_drone_night` R=0.264 (raw), on `ir_mixed_cbam` R=0.519 — vs Anti-UAV R=0.945 / Svanstrom R=0.973. The IR YOLO is **not** as transferable as it looks; its strong Anti-UAV/Svanstrom numbers reflect those two distributions specifically.

## Aggregated numbers

### RGB drone (all splits, drone class)

| Model | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| baseline | raw | 2492 | 240 | 849 | 0.912 | **0.746** | 0.820 |
| baseline | filtered (patch v2) | 2304 | 181 | 1037 | 0.927 | 0.690 | 0.792 |
| retrained_v2 | raw | 2424 | 199 | 917 | 0.924 | 0.726 | 0.813 |
| retrained_v2 | filtered | 2291 | 176 | 1050 | 0.929 | 0.686 | 0.790 |

Patch verifier removes ~7% of drone TPs (188 / 2492 baseline) for a ~25% confuser cut — net F1 loses 2.8 pp on baseline, 2.3 pp on retrained_v2.

### RGB confusers — total FPs across all splits (no GT, every det is a FP)

| Confuser | baseline raw FP | baseline filt FP | base supp | retrained_v2 raw | retrained_v2 filt | rv2 supp |
|---|---|---|---|---|---|---|
| airplane | 1327 | 1286 | **3.1%** | 1666 | 1634 | **1.9%** |
| bird | 103 | 51 | 50.5% | 85 | 41 | 51.8% |
| helicopter | 457 | 292 | 36.1% | 297 | 273 | **8.1%** |
| **total** | **1887** | **1629** | 13.7% | **2048** | **1948** | 4.9% |

retrained_v2 produces ~9% more raw FPs than baseline and patch verifier suppresses 3× less of them. On this OOD set, retrained_v2 is not a confuser-suppression upgrade.

### IR drone detection

| Dataset | Stage | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| ir_drone_night (all splits) | raw | 561 | 575 | 1567 | 0.494 | **0.264** | 0.344 |
| ir_drone_night (all splits) | filtered | 446 | 494 | 1682 | 0.474 | 0.210 | 0.291 |
| ir_mixed_cbam (train+valid) | raw | 1775 | 386 | 1646 | 0.821 | **0.519** | 0.636 |
| ir_mixed_cbam (train+valid) | filtered | 1678 | 242 | 1743 | 0.874 | 0.491 | 0.628 |

ir_drone_night recall=0.26 is a red flag. ir_mixed_cbam valid alone has R=0.98 (60 GT) but its train split has R=0.51 — distribution shift within the same dataset, or annotation density differences.

### IR confusers (no GT, all detections FP)

| Confuser | Raw FP | Filtered FP | Suppression |
|---|---|---|---|
| ir_airplane_hors2 (all splits) | 1128 | 1048 | 7.1% |
| ir_airplane_plane (train+valid) | 386 | 368 | 4.7% |
| ir_bird (train+valid) | 95 | 61 | 35.8% |

IR patch verifier suppression on OOD airplanes is essentially noise (~5%). Birds catch is moderate.

## Implications

### For the production stack pick (§1 of EVIDENCE_LEDGER)

- **Baseline RGB choice still holds.** Beats retrained_v2 on drone recall on this OOD set too (small effect, ~2 pp), in addition to the dominant Svanstrom@1280 win. retrained_v2's confuser-suppression story does not hold up off-Svanstrom.
- **The trust classifier is doing more work than realized.** The Svanstrom S2 cumulative-halluc results (BIRD 0.039, AIRPLANE 0.074, HELICOPTER 0.047 fire rates with `fusion_no_fn_v1.1`) are dramatically better than what YOLO+patch alone achieves here (baseline airplane FP suppression 3.1%, helicopter 36%). **The classifier — not the patch verifier — is the heavy lifter for confuser rejection.** The patch verifier is a fine *secondary* filter, not a standalone confuser screen.
- **imgsz=640 is the wrong operating point for confuser suppression.** Svanstrom@1280 showed retrained_v2 had 3.4–5.6% halluc on the same three confuser classes; here at 640 it has 60–90%+ halluc rates. Confirms the imgsz lesson: **never benchmark confuser suppression at 640**.

### For the thesis story

- **OOD generalization gap is real and measurable.** All three components (RGB baseline, RGB retrained_v2, patch verifier) degrade meaningfully off-Svanstrom. The thesis needs a paragraph explicitly stating: "Svanstrom-tuned metrics overstate open-world performance; here is the OOD floor."
- **Patch verifier needs OOD retraining.** Roboflow airplane suppression at 2–4% is the strongest single data point arguing for retraining the patch verifier on a more diverse confuser corpus (already a recommendation in `2026-05-11_path_forward.md` §3b; this strengthens it).
- **IR YOLO is more brittle than its Anti-UAV/Svanstrom numbers suggest.** The 0.945 → 0.264 recall collapse on `ir_drone_night` belongs in the limitations chapter.

### For next experiments

1. **Re-run this matrix at imgsz=1280** to separate the imgsz effect from the OOD effect. Hypothesis: confuser halluc drops dramatically, drone recall improves for both RGB models, IR may or may not improve.
2. **Re-run with trust classifier inserted** (`fusion_no_fn_v1.1`). Hypothesis: confuser FP collapses to single digits even on OOD data — would be a major thesis result confirming the classifier carries the cascade.
3. **`ir_drone_night` is a candidate dataset to flag as "model breaks here."** Worth a 10-image visual audit to confirm the recall floor isn't an annotation/format issue.

## Dataset-quality caveats (important — read before citing absolute numbers)

- **`ir_drone_night` is heavily sensor-augmented.** Synthetic noise/blur/contrast/inversion variants are applied per-image at dataset-build time, stacking multiple augmentations on the same frame. Many images are barely recognizable as IR drone footage. The R=0.264 floor reflects **augmentation-stacking robustness**, not a clean OOD generalization signal. This dataset is a worst-case robustness probe, not a fair benchmark — the IR YOLO's "OOD ceiling" claim should be softened accordingly.
- **`rgb_drone` has substantial missing labels.** Many frames contain visible drones with no GT box. A non-trivial fraction of the 240/199 "FP" counts and 849/917 "FN" counts is annotation noise rather than real model error. True P is higher than 0.912/0.924; true R is also higher (unlabeled drones can't be counted as TPs either). **Use for relative model comparison only** (baseline vs retrained_v2 ordering is still valid since both see the same labels) — not for absolute precision/recall claims in the thesis.

## Method caveats

- **imgsz=640 throughout.** All numbers above are at 640. Confuser halluc rates and small-drone recalls are not comparable to the Svanstrom@1280 numbers in the ledger.
- **No trust classifier in this run.** This is `YOLO → patch (alert-gate)`, a strictly less capable cascade than production (`YOLO → classifier → patch (alert-gate)`). Reported FPs are upper bounds on what the full stack would produce.
- **Temporal alert gate active but mostly inert.** The Roboflow datasets are independent images, not video; temporal windows rarely accumulate enough hits to gate. Alert counts in the CSV are low (≤50) and not directly comparable to video benchmarks.
- **No drone GT on confuser datasets**, so all detection events on `rgb_airplane/bird/helicopter` and `ir_airplane_*` / `ir_bird` are counted as FPs by construction.

## Delivered

- `C:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/es_drone_detection/eval/results/roboflow_ood/summary.csv` (aggregated CSV from `aggregate_results()`)
- `C:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/es_drone_detection/eval/results/roboflow_ood/{rgb,ir}_*/` (per-dataset per-model per-split `_results.json` + `per_detection.csv`)
- `C:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/es_drone_detection/docs/analysis/2026-05-16_roboflow_ood_eval.md` (this file)
- EVIDENCE_LEDGER row added: §10 Roboflow OOD eval.
