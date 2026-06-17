# FIX hand-off: balance the IR filter's confuser dataset to close the airplane hole (GPU — user runs)

**Date:** 2026-06-17 · Pairs with `2026-06-17_ir_filter_native_vs_aligned.md`. **Scope:** own-GT.
**You run the GPU job; the code + dataset plan are authored + syntax-checked here.**

## The fix in one line
The IR filter's hole is **thermal airplanes** (a confuser-side coverage gap — its confusers are
grayscale-harvested, OOD to thermal airplanes). Fix = add the **thermal-native `IR_confusers` TRAIN
split** (airplane-heavy) as thermal confusers and retrain the aligned filter, **holding out
val/test** for evaluation. This is hole-specific balancing, the IR analogue of the RGB drone re-mine
— but on the **confuser** axis.

## Why this is the right lever (evidence)
- The aligned filter only cuts thermal **airplane** fire 0.352→0.278 (−7.4 pp; bird/heli already low)
  — its confusers were grayscale-harvested, never thermal airplanes (`mri/train_aligned.py` thermal
  confusers = svan/IR_video only, capped ~1800).
- The thermal airplane data **exists and was unused**: `G:/drone/IR_confusers` has a train/val/test
  split — **train = 5237 (airplane 3984, bird 1140, heli 113)**, held-out val 536 + test 165.
- Feasibility confirms the signal is there: thermal drone-vs-confuser **held-out LDA 0.981**, univariate
  AUROC ≤ 0.966 (`2026-06-17_ir_thermal_native_feasibility.json`). So training on thermal airplanes
  should let the filter reject them.
- Keeps everything else: the grayscale-fallback path and the recall-safety shown in the sweep
  (thermal drone R 0.945–0.960) are preserved — we only **add** thermal confusers.

## Code changes (authored, syntax-checked, `--help` validated)
- `mri/train_aligned.py` — new **`--thermal-confusers`** flag: mines the thermal confuser pool
  **BALANCED by (category × size)** — for each (source, category prefix) it buckets every detector fire
  by short-side px (`<16/16–32/32–64/≥64`, from `log_area`) and subsamples each (category × size) **cell**
  to `--conf-cell-cap` (default 1000; scarce cells keep all). Sources: `IR_confusers` **train** split +
  svan IR + IR_video. This fixes BOTH axes — category (airplane was minority) **and** size (the thermal
  confuser FPs are 40% <16px, so large-only coverage wouldn't catch them). Prints the achieved kept/mined
  grid for verification. Writes `mri/results/ir_aligned_balanced/` (incl. `mlp_aligned_gray.pt`).
- `mri/modality_align.py` — `mine_multi` now **guards missing source dirs** (skips with a warning),
  fixing a reorg casualty: `train_aligned`'s `models/ir/corrective_finetune/dataset_v3/train/images`
  thermal-drone source no longer exists in this repo and would otherwise crash `mine()`.

## Workflow (GPU)
**1) Audit the mineable cells first** (so the cap is data-driven, ~10–15 min):
```powershell
py eval/ir_confuser_mine_audit.py        # -> exact thermal confuser category × size table
```
**2) Retrain** with the cap set from the smallest well-populated cell in that table:
```powershell
py -m mri.train_aligned --thermal-confusers --conf-cell-cap <N>   # default N=1000
#   -> mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt (+ mlp_aligned_gray.pt) — drop-in schema
```
(For the clean grayscale-vs-thermal A/B, `py -m mri.train_aligned --no-gray` still exists.)

## Held-out evaluation (no leakage) + acceptance bar
Evaluate on data **not** in training: `IR_confusers/images/val` + `test` (airplane 297 / heli 344 /
bird 60) and CBAM (already held out; `train_aligned` prints the `mri.cli --holdout-eval` command).
**Bar:** thermal **airplane fire down** vs the shipped aligned filter, **without** hurting thermal
drone recall (keep ≈0.945–0.960 from the sweep) or the grayscale-fallback path. Only promote
`ir_aligned_balanced/.../mlp_aligned.pt` → `models/verifiers/ir_aligned/` if the bar is met (reship is
a separate, user-gated step). I can author the held-out airplane-fire eval harness on request (cache
IR_confusers val/test once, then score airplane fire with/without the new filter, zero-GPU).

## Also fixed this pass
- `eval/distill_v5_p3p5_ft4.py` `MODEL_PATHS` repointed `RGB model/` → `models/rgb/` (the RGB
  re-mine `distill_v5_balanced_remine.py` failed to load the FT4 detector; **re-run it now**).

## Delivered
- `…\ES_Drone_Thesis\mri\train_aligned.py` (`--thermal-confusers` flag)
- `…\ES_Drone_Thesis\mri\modality_align.py` (missing-dir guard)
- `…\ES_Drone_Thesis\eval\distill_v5_p3p5_ft4.py` (detector path repoint)
- `…\ES_Drone_Thesis\docs\analysis\2026-06-17_ir_filter_airplane_balance_FIX.md` (this file)
