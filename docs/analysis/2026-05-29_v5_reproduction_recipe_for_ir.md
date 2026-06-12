# V5 Reproduction Recipe — IR Verifier (port from RGB)

*Companion to `docs/analysis/2026-05-28_distillation_v5_journey.md`.*
*Goal: reproduce the V5 feature-distillation MLP on IR features so it can replace the IR patch verifier at deployment time.*
*Status: PLAN. RGB-V5 is the reference; no IR-V5 has been trained yet.*

---

## 0. Why this doc exists

V5 RGB shipped as the production verifier on 2026-05-29 with:
- Svanstrom F1 0.869 (+10 pp vs patch v2), Anti-UAV F1 0.984 (tied), selcom F1 0.607 (+1.5 pp), confuser halluc 0.8% (13× lower), per-detection latency 1.5–2.1 ms (50× faster than patch v2).
- One carve-out: rgb_dataset_test recall ceiling ~0.77 (training-distribution issue, not threshold-tunable).

The IR side still ships **patch v2** (`classifier/runs/patches/confuser_filter4_ir_v2_backup.pt`) — a MobileNet-V3-Small on 224×224 IR crops, same latency profile as the RGB patch verifier. The opportunity is identical: replace it with an MLP on the IR YOLO's internal p3+p5 features for ~50× speedup and equivalent-or-better discrimination.

This recipe captures **every step** of the RGB-V5 process so it can be re-applied to the IR detector. It uses RGB-V5's *what works / what doesn't* as the prior; deviations from RGB-V5 are flagged explicitly with the rationale.

Outcomes that must be matched or improved on the IR side:
- IR-Svanstrom F1 ≥ patch v2 IR F1 (current baseline TBD; measure first).
- IR-Anti-UAV F1 ≥ patch v2 IR F1 (currently saturated, expect tied).
- IR confuser halluc/img reduction ≥ 5× over patch v2 IR.
- Per-detection latency ≤ 3 ms (vs ~70–110 ms patch v2 IR).

---

## 1. What changes vs RGB-V5 (the IR-specific deltas)

| Aspect | RGB-V5 | IR-V5 (this recipe) |
|---|---|---|
| Detector | `RGB model/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt` (FT4 R3) | **`runs/corrective_finetune/finetune_v3b/weights/best.pt`** — verified per [[production-stack-picks]] as the only IR-domain-trained model |
| Surfaces (eval) | 5: Svan, Anti-UAV, selcom_val, rgb_dataset_test, confuser_test | **4**: IR-Svan, IR-Anti-UAV, IR confuser-test, IR-video. **No IR selcom** — selcom is RGB CCTV only |
| Drone sources (training) | antiuav_val, svanstrom, selcom_train, rgb_dataset_{train,val}, rgb_video_*_drone | **antiuav_val IR, svanstrom IR, IR_dsetV9 (or IR_dset_gold) train+val, IR_video_ir_dataset_*_drone** |
| Confuser sources (training) | rgb_confusers_merged_{train,val}, rgb_video_*_conf | **IR_negative pool + Infrared_bird_drone_airplane_CBAM (bird/airplane/heli IR), IR_video_*_conf** |
| Match rule | Svan IoP, selcom IoP, others IoU | **IR-Svan IoP** (same reason — paired dataset shares GT boxes); **everything else IoU**. Verify on first 200-sample dry-run. |
| imgsz | Svan/selcom 1280, others 640 | **IR-Svan 1280** (svanstrom is 640×480 IR also), **others 640**. IR-Anti-UAV at 640 was the V3b training imgsz. |
| Patch verifier baseline | `confuser_filter4_rgb_v2_backup.pt` | **`confuser_filter4_ir_v2_backup.pt`** |
| Feature dim | 517 (5 meta + 256 p3 + 256 p5) | **517 — same** (the IR YOLO has the same Detect-head input shape) |
| Cascade decision | per-frame V5 (PF strictly > AG) | **expect PF**; verify by ablation in eval Phase 4 |
| §14 lesson applied | Svan effective drone-weight share = 40% (too high → rgb_dataset_test blind spot) | **cap any single source ≤ 30% effective weight before training**; rebalance quotas accordingly |

---

## 2. Phase 0 — Setup and inventory (1 hour)

### 2.1 Confirm IR detector path and Detect-head hook compatibility

```
runs/corrective_finetune/finetune_v3b/weights/best.pt
```

Run the V5 `DetectInputHook` (in `eval/distill_v5_p3p5_ft4.py` lines 191–210) against this checkpoint on **a single IR image**. Assert:
- p3 shape: `(1, 64, H, W)` with H,W ≈ imgsz/8
- p4 shape: `(1, 128, H, W)` with H,W ≈ imgsz/16
- p5 shape: `(1, 256, H, W)` with H,W ≈ imgsz/32

If any of those differ, the IR detector uses a different head — the recipe still applies, but `YOLO_FEAT_DIM` recomputes accordingly. Most likely they match (Yolo26n architecture is the same).

### 2.2 Inventory IR datasets

| Role | RGB-V5 source | Proposed IR source | Path | Notes |
|---|---|---|---|---|
| Primary drone (paired) | svanstrom RGB | **Svanstrom IR** | `G:/drone/svanstrom_paired/IR/images` | Same scenes, same GT boxes — IoP match rule confirmed transferable |
| Anti-UAV-style drone | Anti-UAV RGB val | **Anti-UAV IR val** | `G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images` | Test split is for eval — DO NOT train on it |
| General-IR drone benchmark | rgb_dataset (AirBird) | **IR_dsetV9 train + val** or **IR_dset_gold_duplicates_removed** | `G:/drone/IR_dsetV9/{train,val}` | Pick the largest, cleanest IR drone corpus available; mirror RGB's "use train+val for training, hold out test for eval" |
| Real-video drone | RGB_video V_DRONE_* | **IR_video_ir_dataset {train,val} V_DRONE_*** (if prefix split exists) | `G:/drone/IR_video_ir_dataset/{train,val}` | If no prefix splits, treat the entire video set as drone-positive and confuser-mining elsewhere |
| Confuser pool (IR-specific) | rgb_confusers_merged + V_BIRD/V_AIRPLANE/V_HELI | **Infrared_bird_drone_airplane_CBAM bird+airplane+heli classes**, **infrared_bird_*, infrared_drone_night** | `G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/train` etc. | Use class label 0 (drone) to skip; classes 1/2/3 (bird/airplane/heli) → confuser pool. Verify class indexing per dataset's `data.yaml` |
| Pure-IR negatives | confuser_train no-GT | **IR_negative_1_dropbox** (after unzip) | `G:/drone/IR_negative_1_dropbox.zip` extracted | Treat as `kind="image_no_gt"` like rgb_confusers_merged |

**Action item:** unzip `IR_negative_1_dropbox.zip` into `G:/drone/IR_negative_1/` before mining if not already done.

### 2.3 Confirm patch-v2 IR baseline numbers

Before training V5-IR, measure patch v2 IR on all four eval surfaces. This is the floor V5-IR must beat — without it, the decision gate has no anchor.

```
python eval/eval_v4_vs_patch.py \
    --modality ir \
    --datasets svanstrom_ir,antiuav_ir,confuser_ir_test,ir_video_eval \
    --patch-weights classifier/runs/patches/confuser_filter4_ir_v2_backup.pt \
    --out-suffix _ir_baseline
```

(The harness needs an `--modality ir` switch — see §6.1 for the diff.)

---

## 3. Phase 1 — Data mining (3–4 hours wall time)

### 3.1 Clone the V5 trainer

```
copy eval\distill_v5_p3p5_ft4.py eval\distill_v5_p3p5_ir.py
```

Keep both files. The RGB version stays the production reference; the IR version is independently iterable.

### 3.2 Edit the IR-V5 trainer — the *only* changes from RGB-V5

In `eval/distill_v5_p3p5_ir.py`:

1. **Detector path** (line ~68): `MODEL_PATHS["ir_v3b"] = REPO / "runs/corrective_finetune/finetune_v3b/weights/best.pt"`. Set `BASE_MODEL_PATH = MODEL_PATHS["ir_v3b"]`.

2. **Output dir** (line ~63): `OUT_DIR = EVAL_DIR / "results" / "_v5_ir_p3p5_v3b"`.

3. **Source paths** (lines 72–85) — replace with the inventory in §2.2:

   ```python
   ANTIUAV_VAL_IR    = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images")
   SVANSTROM_IR_DIR  = Path("G:/drone/svanstrom_paired/IR/images")
   IR_DSET_TRAIN     = Path("G:/drone/IR_dsetV9/train/images")     # or IR_dset_gold
   IR_DSET_VAL       = Path("G:/drone/IR_dsetV9/val/images")
   IR_VIDEO_TRAIN    = Path("G:/drone/IR_video_ir_dataset/train/images")
   IR_VIDEO_VAL      = Path("G:/drone/IR_video_ir_dataset/val/images")
   IR_CONF_BIRD      = Path("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/train/images")
   IR_NEGATIVE_POOL  = Path("G:/drone/IR_negative_1/images")       # after unzip
   ```

4. **SOURCES list** (lines 131–179) — keep the same 11-slot structure, drop selcom rows, add IR-specific rows. **Apply the §14 lesson: cap any single source's effective drone-grad weight at ≤ 30% of total.** Suggested quotas:

   | Source | target_drones | target_confusers | weight_drone | match_rule | imgsz | Effective drone-grad |
   |---|---:|---:|---:|---|---:|---:|
   | antiuav_val (IR) | 4,000 | 2,000 | 1.5 | iou | 640 | 6,000 (24%) |
   | svanstrom (IR) | 4,000 | 5,000 | **1.5** | iop | 1280 | 6,000 (24%) ← **was 2.5 in RGB; lowered per §14** |
   | ir_dset_train | 6,000 | 3,000 | 1.0 | iou | 640 | 6,000 (24%) |
   | ir_dset_val | 1,500 | 0 | 1.0 | iou | 640 | 1,500 (6%) |
   | ir_video_train_drone | 4,000 | 0 | 1.5 | iou | 640 | 6,000 (24%) ← **boosted because real-video is the most operationally valuable signal** |
   | ir_video_val_drone | 800 | 0 | 1.5 | iou | 640 | 1,200 (5%) |
   | ir_video_train_conf | 0 | 3,000 | 1.5 | iou | 640 | — |
   | ir_video_val_conf | 0 | 500 | 1.5 | iou | 640 | — |
   | ir_confuser_bird_etc | 0 | 8,000 | 1.0 | iou (no_gt for IR_negative) | 640 | — |
   | ir_negative_pool | 0 | 4,000 | 1.0 | iou (no_gt) | 640 | — |

   **Effective drone-grad total: ~25,000. Max single-source share: ~24% (antiuav, svanstrom, ir_dset, ir_video).** No source dominates — addresses the §14 root cause.

5. **Phase-1 mining check (smoke test before full run):** call `eval/distill_v5_p3p5_ir.py --phase 1 --max-samples 500` and verify each source actually yields drones. Expected failure modes:
   - `ir_video_train_drone` yields 0: same `V_DRONE_` prefix filter as RGB-video failed because the IR detector misses real-video drones (mirror of the RGB-V5 finding in [[v5-distillation-production]]). **If this happens, drop the IR_video rows and redistribute their drone quota to ir_dset_train.** Document explicitly — V5-IR's deployment on real IR-video will then be untested (same caveat as RGB).
   - Anti-UAV val IR yields << target: verify `Anti-UAV-RGBT_yolo_converted` labels-dir is sibling layout (`val/IR/labels/`) vs mirrored. Same labels-dir bug pattern as RGB-V5; same fix via `_resolve_labels_dir()`.

### 3.3 Run full Phase 1 collection

```
python eval/distill_v5_p3p5_ir.py --phase 1
```

Outputs `eval/results/_v5_ir_p3p5_v3b/training_data.npz` and `training_meta.json`. Verify:
- `n_drone >= 18000` (target was 20,300)
- `n_confuser >= 12000`
- `feature_dim == 517`
- Per-source counts roughly match targets (within 80% — drops indicate labels-dir or imgsz bug)
- **Re-compute effective drone-grad share from the realised yields — if any source exceeds 30%, lower its weight before Phase 2.**

---

## 4. Phase 2 — Training (45 minutes on GPU)

### 4.1 Reuse RGB-V5's MLP architecture exactly

`MLPWrapper` in `eval/distill_v5_p3p5_ft4.py` lines ~376–442. Constructor:

```python
MLPWrapper(input_dim=517, hidden_dims=(512, 256, 128, 64),
           use_batchnorm=True, dropout=0.3)
```

Total ~300k params. Loss / optimizer:
- **FocalLoss(α=0.75, γ=2.0, label_smoothing=0.1)** (the V5 default). `pos_weight` from class ratio.
- AdamW lr=1e-3, weight_decay=1e-4.
- **Cosine annealing LR** over 100 epochs, batch 64.
- K=5 stratified CV using `cross_val_score_f1` (line ~447), with sample weights from `weight_drone` / `weight_confuser`.

### 4.2 Phase 2 command

```
python eval/distill_v5_p3p5_ir.py --phase 2
```

Outputs `eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt`.

### 4.3 Sanity floor before continuing

- K=5 CV mean F1 ≥ **0.85** on held-out fold (RGB-V5 hit 0.9869 — IR is harder, but ≥ 0.85 is the cut). Below 0.85 → stop, re-examine quotas (§3.2 effective-weight table). Above 0.85 → proceed to Phase 3.

---

## 5. Phase 3 — Eval harness (1 hour eval per surface, ~4 hours end-to-end)

### 5.1 Adapt the head-to-head harness

`eval/eval_v4_vs_patch.py` (lines 71–84 `DATASETS`) currently has only RGB surfaces. Add an `--modality {rgb,ir}` switch that routes to either RGB-V5 + RGB patch v2 + RGB dataset paths or IR-V5 + IR patch v2 + IR dataset paths.

IR `DATASETS` registry:

```python
DATASETS_IR = {
    "svanstrom_ir":     (Path("G:/drone/svanstrom_paired/IR/images"), True,  "iop", 9, 1280),
    "antiuav_ir":       (Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"), True, "iou", 5, 640),
    "confuser_ir_test": (Path(...IR confuser test split...), False, "iou", 1, 640),
    "ir_video_eval":    (Path("G:/drone/IR_video_ir_dataset/test/images"), True, "iou", 5, 640),
}
```

### 5.2 Adapt the quick-pipeline harness

`eval/eval_pipeline_v5_quick.py` (lines 65–75) has the same `DATASETS` pattern; mirror the changes above. Add `--mlp-weights eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt` and `--patch-weights classifier/runs/patches/confuser_filter4_ir_v2_backup.pt`.

### 5.3 Run the 5-branch comparison

```
python eval/eval_pipeline_v5_quick.py --modality ir --n-images 500
```

Output: `eval/results/_v5_pipeline_quick_ir/comparison.md` with bare_v3b / patch_v2_pf / patch_v2_ag / v5_mlp_pf / v5_mlp_ag.

---

## 6. Phase 4 — Decision gates (apply RGB-V5 lessons)

V5-IR ships if **and only if** it clears these gates (mirrored from RGB-V5):

| Gate | RGB-V5 result | IR-V5 cut |
|---|---|---|
| 1. IR-Svanstrom F1 strictly > patch v2 IR | +10 pp | ≥ +3 pp |
| 2. IR-Anti-UAV F1 within ±1 pp of patch v2 IR | tied | tied |
| 3. IR confuser halluc/img ≤ patch v2 IR / 5 | 13× better | ≥ 5× better |
| 4. IR-video F1 ≥ patch v2 IR | untested at RGB | gate is informational, not blocking (RGB had the same caveat) |
| 5. Per-detection latency ≤ 3 ms | 1.5–2.1 ms | identical (same MLP, same forward) — should be automatic |
| 6. PF vs AG: PF F1 ≥ AG F1 | PF strictly better | confirm — if AG wins by > 0.5 pp, deploy AG (unlike RGB) |

**Gate-failure response:**
- Gate 1 fails: re-check Svanstrom IR imgsz=1280 and IoP match rule. If still fails, the IR YOLO features at p5 may be too saturated to discriminate; rebalance toward p3 by trying P3_GRID=(4,4) (raises YOLO_FEAT_DIM to 64×16 + 256 = 1280; recompute INPUT_DIM=1285).
- Gate 3 fails: confuser pool likely too narrow — IR_negative_1 alone won't cover bird/airplane/heli diversity. Add `Infrared_bird_*`, `Infrared_Helicopter`, `infrared_drone_night` zipped sources.
- Gate 4 informational fail: same as RGB carve-out — note in ledger, do not block production.

---

## 7. Phase 5 — Production swap (only after Phase 4 passes)

### 7.1 Save the V5-IR artifact

`eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt` — same dict layout as RGB:
```
{"state_dict", "scaler_mean", "scaler_scale",
 "input_dim", "hidden_dims", "use_batchnorm", "dropout", "threshold"}
```

Persist `scaler_mean` / `scaler_scale` as **torch tensors, not numpy arrays** (the RGB-V5 bug: numpy arrays broke `torch.load(weights_only=True)` at deploy).

### 7.2 GUI cascade swap

In `ir_gui/api.py`, replace the IR patch-loader weights path:
```
classifier/runs/patches/confuser_filter4_ir_v2_backup.pt
```
with
```
eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt
```

Verify the IR cascade in `ir_gui/api.py` accepts the V5 MLP interface (it currently expects a MobileNet-V3 image classifier — V5 takes pooled features). Either:
- Refactor the IR cascade slot to run V5 in feature space (preferred — mirrors RGB-V5's deploy path), OR
- Wrap V5 in a `compute_features(image, box) → MLP(features) → veto` shim.

### 7.3 Per-frame deploy

Same as RGB: **run V5-IR on every detection, not alert-gated.** Per [[v5-distillation-production]], the ~0.2 ms AG saving never compensates for the F1 loss.

### 7.4 Update memory and ledger

- Update `[[production-stack-picks]]` IR-verifier row from `confuser_filter4_ir_v2_backup.pt` to `mlp_v5_ir.pt`.
- Update `[[v5-distillation-production]]` to extend the IR section with realized numbers.
- Append `EVIDENCE_LEDGER` §14 (or whichever next section number) with the IR head-to-head table.

---

## 8. Pitfalls to avoid (from the RGB-V5 lessons)

These were debugged the hard way on the RGB side. **Do not repeat them on IR.**

1. **Wrong labels-dir layout.** Some datasets use `images/<split>/labels/` (mirrored), others use `<split>/labels/` (sibling). The `_resolve_labels_dir()` helper in V5 RGB tries both. Re-use as-is on IR.
2. **IoU under-counting on paired GT.** Svanstrom RGB GT boxes are larger than the drone → IoU=0.5 mining yielded 1,020 drones instead of 5,000. **IR-Svanstrom inherits the same GT boxes — use IoP. Same for any other paired-IR dataset.**
3. **Wrong imgsz collapses recall.** RGB-V5 selcom at imgsz=640 gave R=0.10; at 1280 R=0.45. **IR-Svanstrom must be 1280** (native 640×480). Test the IR detector's recall on Svanstrom IR at both imgszs *before* mining.
4. **Mixed-source pollution.** RGB-V5 first used `selcom_mixed_train` (80% general + 20% pure CCTV) — selcom F1 collapsed to 0.24. Fix was pure-source swap. **For IR**: when picking the IR_dset source, verify it is pure-distribution before quoting it as 6,000 drones. If `IR_dsetV9` is a mixed re-aggregation, prefer `IR_dset_gold_duplicates_removed`.
5. **Single-source dominance.** RGB-V5 Svanstrom dominated at 40% effective drone-grad → rgb_dataset_test recall ceiling. **§3.2 quota table caps every source at ≤30% effective weight.**
6. **Default eval subset too narrow.** RGB head-to-head originally ran on 3 surfaces; selcom_val and rgb_dataset_test got added later. **Set IR-V5 default eval surfaces to all 4 from day one** (or all 5 if an `ir_video_eval` test split exists).
7. **Quick-eval alphabetical bias.** Default first-N sampling on Svanstrom gave all confuser frames; rgb_dataset_test gave all blank frames. **Use stride sampling**: `stride = max(1, n_total // n_images)`. Already fixed in `eval/eval_pipeline_v5_quick.py`.
8. **Phase 3 in-script eval bug.** V5 RGB Phase 3 reported selcom F1=0 with FN=0 (impossible) due to the labels-dir bug. **Trust only the head-to-head harness numbers (`eval/eval_v4_vs_patch.py`), not the trainer's in-script Phase 3 verdict.**
9. **`weights_only=True` load fails on numpy-array scaler stats.** Save as torch tensors (§7.1).
10. **Single-Gaussian Mahalanobis prototype.** Tried as Lever 4 fallback; failed because drone class is multi-modal in feature space (per-source clusters). **Do not build single-Gaussian prototype on IR features either** — same multi-modality is expected.
11. **Alert-gating an MLP verifier.** RGB-V5 PF strictly beat AG (saves <0.2 ms, costs 0.8–4.0 pp F1). Same trade-off expected on IR; verify in Phase 4 gate 6.

---

## 9. End-to-end command sequence (copy-paste recipe)

```
# Phase 0 — setup
mkdir eval\results\_v5_ir_p3p5_v3b
copy eval\distill_v5_p3p5_ft4.py eval\distill_v5_p3p5_ir.py
# (edit per §3.2)

# Phase 0.5 — baseline measurement
python eval/eval_v4_vs_patch.py --modality ir --datasets svanstrom_ir,antiuav_ir,confuser_ir_test --patch-weights classifier/runs/patches/confuser_filter4_ir_v2_backup.pt --out-suffix _ir_baseline

# Phase 1 — mining (smoke test then full)
python eval/distill_v5_p3p5_ir.py --phase 1 --max-samples 500   # smoke
python eval/distill_v5_p3p5_ir.py --phase 1                      # full (3–4h)

# Phase 2 — training
python eval/distill_v5_p3p5_ir.py --phase 2                      # 45min on GPU

# Phase 3 — eval
python eval/eval_v4_vs_patch.py --modality ir --datasets svanstrom_ir,antiuav_ir,confuser_ir_test,ir_video_eval --mlp-weights eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt --patch-weights classifier/runs/patches/confuser_filter4_ir_v2_backup.pt --out-suffix _ir_v5_vs_patch
python eval/eval_pipeline_v5_quick.py --modality ir --n-images 500

# Phase 4 — decision gate
# (manual: review eval/results/_v5_pipeline_quick_ir/comparison.md against §6 table)

# Phase 5 — production swap (only if gates pass)
# (manual: edit ir_gui/api.py, update [[production-stack-picks]], append EVIDENCE_LEDGER)
```

---

## 10. Open questions / decisions deferred to execution time

These are **not blockers** for this recipe but must be answered before Phase 1 mining:

1. **Which IR drone dataset for the "general benchmark" role?** Candidates: `IR_dsetV9`, `IR_dset_gold_duplicates_removed`, `IR_dset_final`. Pick the largest non-overlapping with Svanstrom-IR / Anti-UAV-IR, and document the choice. Mirror the RGB lesson — train+val for mining, hold test out for eval.
2. **Which IR confuser pool?** RGB used `rgb_confusers_merged` (Roboflow bird/airplane/heli web-scraped). IR equivalent is union of `Infrared_bird_*`, `Infrared_Helicopter`, `infrared_drone_night` (negative class), and `IR_negative_1`. Verify each per-class index in their `data.yaml`.
3. **Does Anti-UAV IR have a test split we should hold out?** Confirm `G:/drone/Anti-UAV-RGBT_yolo_converted/{val,test}/IR/` are non-overlapping. If yes, train on `val/IR`, eval on `test/IR` (mirrors RGB). If `test` is empty or unsplit, use stride-sampled val for both (with explicit blocklist).
4. **PF vs AG on IR.** Default expectation: PF wins (mirror of RGB). But if IR detection rate is much lower (e.g., the IR detector fires on far fewer frames), AG cost saving may not exist either way. Decide empirically in Phase 4 gate 6.

---

## 11. Why this recipe is coherent with the RGB-V5 work

- Same feature pipeline (`DetectInputHook`, `roi_pool`, P3_GRID=(2,2), P5_GRID=(1,1), 517-D input).
- Same MLP architecture (`MLPWrapper(use_batchnorm=True, dropout=0.3, hidden_dims=(512,256,128,64))`).
- Same loss (FocalLoss α=0.75 γ=2.0 ls=0.1) and optimizer (AdamW + cosine).
- Same evaluation rules (IoP for paired-GT surfaces, IoU elsewhere; conf=0.25; per-surface imgsz).
- Same head-to-head harness pattern (one detector forward, both verifiers scored, identical sa32 trust).
- Same decision-gate philosophy (F1 + halluc beat raw recall; PF default unless ablation shows AG wins).
- Same memory + ledger update pattern at production swap time.

**The only invented elements for IR** are the dataset paths, the dropped selcom surface, and the §3.2 weight rebalance addressing §14's lesson. Everything else is a port, not a rewrite.

---

## Delivered

- `docs/analysis/2026-05-29_v5_reproduction_recipe_for_ir.md` (this file).
- Updates pending until V5-IR is actually trained:
  - `eval/distill_v5_p3p5_ir.py` (to be created from `eval/distill_v5_p3p5_ft4.py`).
  - `eval/results/_v5_ir_p3p5_v3b/` (to be populated by Phase 1+2 commands above).
  - `eval/eval_v4_vs_patch.py` + `eval/eval_pipeline_v5_quick.py` (`--modality ir` switch to be added).
  - Memory updates to `[[production-stack-picks]]` and `[[v5-distillation-production]]` once Phase 5 swap completes.
  - `docs/EVIDENCE_LEDGER.md` §14 (IR head-to-head row).

## References

- `docs/analysis/2026-05-28_distillation_v5_journey.md` — the RGB-V5 chapter this recipe mirrors (read §14 for the source-dominance lesson).
- `C:\Users\User\.claude\projects\...\memory\project_v5_distillation_production.md` — RGB-V5 production candidate state.
- `C:\Users\User\.claude\projects\...\memory\project_production_stack.md` — current stack pointing at RGB-V5 / IR patch v2.
- `eval/distill_v5_p3p5_ft4.py` — the trainer to clone.
- `eval/eval_v4_vs_patch.py` — the head-to-head harness to extend with `--modality ir`.
- `eval/eval_pipeline_v5_quick.py` — the speed+F1 harness to extend.
- `docs/EVIDENCE_LEDGER.md` §13.1–13.7 — RGB-V5 numbers to mirror at §14 once IR ships.
