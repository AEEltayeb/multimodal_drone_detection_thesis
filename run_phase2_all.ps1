# run_phase2_all.ps1 — Phase 2 gap-fill, all batches back-to-back.
#
# Logs everything to logs/phase2_<timestamp>/<batch>.log so a failure in one
# batch does not block the others. Run from repo root:
#   powershell -ExecutionPolicy Bypass -File run_phase2_all.ps1
#
# Override individual batches:
#   .\run_phase2_all.ps1 -SkipBatches B,E       # skip selcom-val and anti-uav
#   .\run_phase2_all.ps1 -OnlyBatches D         # only Svanstrom per-size

param(
    [string[]]$SkipBatches = @(),
    [string[]]$OnlyBatches = @()
)

$ErrorActionPreference = "Continue"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = "logs/phase2_$ts"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Should-Run([string]$batch) {
    if ($OnlyBatches.Count -gt 0) { return $OnlyBatches -contains $batch }
    return -not ($SkipBatches -contains $batch)
}

function Run-Batch([string]$batch, [string]$desc, [scriptblock]$body) {
    if (-not (Should-Run $batch)) {
        Write-Host "`n[SKIP] Batch $batch — $desc" -ForegroundColor Yellow
        return
    }
    $log = "$logRoot/batch_$batch.log"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "[BATCH $batch] $desc" -ForegroundColor Cyan
    Write-Host "  Log: $log" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    $t0 = Get-Date
    & $body *>&1 | Tee-Object -FilePath $log
    $dt = ((Get-Date) - $t0).TotalMinutes
    Write-Host "[BATCH $batch] Done in $([math]::Round($dt,1)) min" -ForegroundColor Green
}

# ── Batch B: Selcom held-out val (311 images) ──────────────────────
Run-Batch "B" "Selcom val (311 images) for 5 RGB models" {
    $datasetDir = "G:/drone/_finetune_selcom_mixed_ft2/images/val"
    $weights = @{
        "baseline"       = "RGB model/Yolo26n_trained/weights/best.pt"
        "hardneg_v3more" = "RGB model/Yolo26n_hardneg_v3_more/weights/best.pt"
        "retrained_v2"   = "RGB model/Yolo26n_retrained_v2/weights/best.pt"
        "selcom_1280"    = "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"
    }
    foreach ($name in $weights.Keys) {
        Write-Host "  -> $name (imgsz=1280)"
        python eval/eval_model.py --weights $weights[$name] --model-name $name `
            --dataset $datasetDir --imgsz 1280 --conf 0.25 `
            --output-dir "eval/results/selcom_val_holdout/$name"
    }
    Write-Host "  -> selcom_640 (imgsz=640)"
    python eval/eval_model.py `
        --weights "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt" `
        --model-name selcom_640 --dataset $datasetDir `
        --imgsz 640 --conf 0.25 `
        --output-dir "eval/results/selcom_val_holdout/selcom_640"
}

# ── Batch D: Svanstrom per-size ────────────────────────────────────
Run-Batch "D" "Svanstrom per-size for all RGB models (imgsz=1280)" {
    python eval/eval_svanstrom_persize.py --imgsz 1280
}

# ── Batch A: Real-video per-size ───────────────────────────────────
Run-Batch "A" "Real-video per-size across drone + confuser clips" {
    python eval/eval_video_persize.py
}

# ── Batch E: Anti-UAV per-model ────────────────────────────────────
Run-Batch "E" "Anti-UAV per-model (RGB)" {
    $au = @{
        "baseline"     = "RGB model/Yolo26n_trained/weights/best.pt"
        "retrained_v2" = "RGB model/Yolo26n_retrained_v2/weights/best.pt"
        "selcom_1280"  = "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"
    }
    foreach ($name in $au.Keys) {
        Write-Host "  -> $name"
        python eval/eval_model.py --weights $au[$name] --model-name $name `
            --dataset "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB" `
            --imgsz 640 --conf 0.25 `
            --output-dir "eval/results/antiuav_per_model/$name"
    }
}

# ── Batch C: Roboflow OOD for selcom + hardneg (longest) ───────────
Run-Batch "C" "Roboflow OOD full sweep — selcom + hardneg added to MODELS dict" {
    python eval/run_roboflow_eval.py --full --skip-extract `
        --datasets rgb_airplane rgb_bird rgb_helicopter rgb_drone
}

# ── Merge new CSVs into the long-format inventory ──────────────────
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Re-running metrics inventory consolidator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
python analytics/spec_analysis/07_metrics_inventory.py *>&1 |
    Tee-Object -FilePath "$logRoot/inventory_merge.log"

Write-Host "`nAll Phase 2 batches done. Logs in $logRoot" -ForegroundColor Green
Write-Host "Next: tell Claude in chat to extend 07_metrics_inventory.py" -ForegroundColor Green
Write-Host "      to parse the new CSV schemas (selcom_val, svanstrom_persize," -ForegroundColor Green
Write-Host "      video_persize, antiuav_per_model)." -ForegroundColor Green
