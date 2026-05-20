# Phase 2 — gap-fill commands (READY TO RUN)

Run from repo root in Windows PowerShell. All scripts now exist and are wired up.
After each batch finishes, re-run `python analytics/spec_analysis/07_metrics_inventory.py` to merge new CSVs into the long-format inventory.

---

## A. Real-video per-size — `eval/eval_video_persize.py`

Runs each model on every clip under `G:/drone/drone detection video tests/rgb/{drone,birds,airplanes,helicopters}`. Buckets TP/FP/FN by GT box area. Confuser clips (birds/airplanes/helicopters) contribute FP-by-det-size only.

```powershell
python eval/eval_video_persize.py
```

Output: `eval/results/video_persize/<category>/<clip>/<model>_persize.csv` + `eval/results/video_persize/summary.csv`.

Subset variants:

```powershell
# Just drone-positive clips
python eval/eval_video_persize.py --categories drone

# Single model, all clips
python eval/eval_video_persize.py --models selcom_1280
```

Wall time: ~30–60 min on 1 GPU.

---

## B. Selcom held-out val (311 images) — uses `eval/eval_model.py`

```powershell
$weights = @{
  "baseline"        = "RGB model/Yolo26n_trained/weights/best.pt"
  "hardneg_v3more"  = "RGB model/Yolo26n_hardneg_v3_more/weights/best.pt"
  "retrained_v2"    = "RGB model/Yolo26n_retrained_v2/weights/best.pt"
  "selcom_1280"     = "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"
}

foreach ($name in $weights.Keys) {
  python eval/eval_model.py --weights $weights[$name] --model-name $name `
    --dataset G:/drone/_finetune_selcom_mixed_ft2/data.yaml `
    --imgsz 1280 --conf 0.25 `
    --output-dir eval/results/selcom_val_holdout/$name
}

# selcom_640 — same weights, imgsz=640
python eval/eval_model.py --weights "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt" --model-name selcom_640 `
  --dataset G:/drone/_finetune_selcom_mixed_ft2/data.yaml `
  --imgsz 640 --conf 0.25 `
  --output-dir eval/results/selcom_val_holdout/selcom_640
```

Wall time: <5 min total.

---

## C. Roboflow OOD — for selcom + hardneg_v3more (already patched into `run_roboflow_eval.py`)

`MODELS` dict now includes `rgb_hardneg_v3more`, `rgb_selcom_1280`, `rgb_selcom_640` with per-model `imgsz` override. Just run:

```powershell
python eval/run_roboflow_eval.py --full --skip-extract --datasets rgb_airplane rgb_bird rgb_helicopter rgb_drone
```

Outputs append to `eval/results/roboflow_ood/<dataset>/<model>/<split>/`. Re-aggregates `summary.csv` at the end.

Wall time: ~1–2 hours full RGB sweep.

---

## D. Svanström per-size — `eval/eval_svanstrom_persize.py`

```powershell
python eval/eval_svanstrom_persize.py --imgsz 1280
```

Output: `eval/results/svanstrom_persize/<model>_persize.csv` + `summary.csv`.
Per-category × per-size × per-model TP/FP/FN/P/R/F1.

Wall time: ~30 min (1299 drone frames + 1891 confuser frames × 5 models).

---

## E. Anti-UAV per-model RGB

```powershell
$models = @{
  "baseline"     = "RGB model/Yolo26n_trained/weights/best.pt"
  "retrained_v2" = "RGB model/Yolo26n_retrained_v2/weights/best.pt"
  "selcom_1280"  = "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"
}
foreach ($name in $models.Keys) {
  python eval/eval_model.py --weights $models[$name] --model-name $name `
    --dataset G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB `
    --imgsz 640 --conf 0.25 `
    --output-dir eval/results/antiuav_per_model/$name
}
```

Wall time: ~10–20 min.

---

## Recommended run order

Fastest-first so the inventory fills incrementally:

1. **B** — Selcom val, ~5 min. Run first.
2. **D** — Svanström per-size, ~30 min.
3. **A** — Real-video per-size, ~30–60 min.
4. **E** — Anti-UAV per-model, ~20 min.
5. **C** — Roboflow OOD selcom + hardneg, 1–2 hr (overnight if needed).

After each:
```powershell
python analytics/spec_analysis/07_metrics_inventory.py
```

Then proceed to Phase 3 (failure-mode sampling) once the gap matrix is mostly filled.

---

## Outputs feeding back into the doc

| Phase 2 batch | Output CSV | Inventory parser |
|---|---|---|
| A | `eval/results/video_persize/summary.csv` | needs new branch in `07_metrics_inventory.py` |
| B | `eval/results/selcom_val_holdout/<m>/*_results.json` | needs new branch |
| C | `eval/results/roboflow_ood/summary.csv` (re-aggregated) | already parsed |
| D | `eval/results/svanstrom_persize/summary.csv` | needs new branch |
| E | `eval/results/antiuav_per_model/<m>/*_results.json` | needs new branch |

I'll extend `07_metrics_inventory.py` to pick up A/B/D/E once you launch the runs.

---

## Hooks for the narrative doc

After Phase 2 lands the narrative will auto-fill these sections:

- §1 headline Q&A (best/worst surfaces per model, where no model detects)
- §2.2 Svanström per-size
- §4 Selcom val
- §5 Roboflow RGB per-model
- §7.2 / 8 / 9 real-video per-size
- §13 cross-surface findings
