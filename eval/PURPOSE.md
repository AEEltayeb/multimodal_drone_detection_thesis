# Evaluation Directory — Purpose & Design

## What This Directory Does

The `eval/` directory is the **single, canonical place** to evaluate every component of the drone detection pipeline. It replaces ~20 scattered evaluation scripts that previously lived across `classifier/`, `analytics/eval/`, and the project root.

## Why It Exists

During development, each evaluation task spawned its own script:
- `eval_six_configs.py` for pipeline configs
- `eval_youtube_ir_filter.py` for OOD IR videos
- `eval_rgb_finetune.py` for model comparisons
- `eval_cross_model.py` for cross-dataset benchmarks
- ...and many more

This led to **duplicated logic** (matching, scoring, reporting) and **inconsistent interfaces** (different CLI flags, different output formats, different metric definitions). Finding "the right eval script" became a problem in itself.

## Design Principles

1. **Two scripts, two purposes**:
   - `eval_pipeline.py` — evaluates the **pipeline** (YOLO + classifier + filter)
   - `eval_model.py` — evaluates a **single YOLO model** on a dataset

2. **Shared modules**: All matching logic (`metrics.py`), dataset loading (`datasets.py`), and reporting (`reporting.py`) are shared. No duplicated code.

3. **Central config** (`config.yaml`): Dataset paths, model weights, and defaults live in one place.

4. **Cache reuse**: The `cache/` directory stores pre-computed YOLO detections. Run inference once, evaluate many times with different settings.

5. **Legacy compatibility**: The cache format is identical to the old `run_inference.py` output. Old caches from `classifier/runs/` are still loadable via `config.yaml` legacy paths.

## How Evaluation Works

### Pipeline Evaluation Flow

```
                    ┌──────────────┐
    Video frames →  │  YOLO (RGB)  │ → raw detections
                    │  YOLO (IR)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Classifier  │ → trust label (0-3)
                    │  (32 feats)  │   0=reject, 1=RGB, 2=IR, 3=both
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Patch Filter │ → confuser probability per detection
                    │ (RGB + IR)   │   P(confuser) ≥ threshold → reject
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Scoring     │ → TP/FP/FN vs GT boxes
                    │  IoU + IoP   │
                    └──────────────┘
```

Each "config" represents a different combination of these layers:
- `ir_only` / `rgb_only` — raw YOLO only
- `classifier` — YOLO + classifier
- `ir_filter` / `rgb_filter` — YOLO + patch filter
- `classifier_then_filter` — YOLO + classifier + filter
- `filter_then_classifier` — YOLO + filter + classifier

### Model Evaluation Flow

```
    Images + Labels → YOLO model → detections → match vs GT → metrics
```

Simple single-model benchmarking with optional per-source breakdown, confidence sweeps, and size distribution analysis.

## Metrics Computed

### Detection-Level (per bounding box)
- **TP**: Detection matches a GT box (IoU ≥ 0.5 or IoP ≥ 0.5)
- **FP**: Detection matches no GT box
- **FN**: GT box with no matching detection
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: harmonic mean of P and R

### Frame-Level (per image)
- **TP**: Frame has detection AND has GT
- **FP**: Frame has detection but NO GT
- **FN**: Frame has GT but NO detection
- **TN**: Frame has neither

### Size Distribution
Detections classified by bounding box area as fraction of image:
- **Small**: < 0.1% of image area (typical distant drones)
- **Medium**: 0.1% – 1%
- **Large**: > 1%

### FP by Category
For datasets with category labels (Svanström), FP counts are broken down by:
AIRPLANE, BIRD, DRONE, HELICOPTER, OTHER

## Output Structure

```
eval/results/{dataset_name}/
├── metrics_iou.csv          # P/R/F1 per config (IoU matching)
├── metrics_iop.csv          # P/R/F1 per config (IoP matching)
├── fp_by_category_iou.csv   # FP breakdown by confuser type
├── per_det.jsonl            # Per-detection records for offline analysis
├── patch_probs.json         # Cached filter probabilities
├── metrics_bars_iou.png     # Bar chart (if --plot)
├── confusion_iou.png        # Confusion matrices (if --plot)
├── pr_curves_iou.png        # PR curves (if --plot)
└── size_distribution.png    # Size dist chart (if --plot)
```
