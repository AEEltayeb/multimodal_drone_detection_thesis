# ES Drone Detection

Multi-sensor drone detection system using YOLOv26n, The system targets infrared (IR) drone detection with a human-in-the-loop dataset refinement pipeline that iteratively improves label quality using model predictions.

## Project Structure

```
ES_Drone_Detection/
├── configs/                    # Training and evaluation YAML configs
│   ├── base.yaml               # Shared hyperparameters
│   ├── ir_final_cleaned.yaml   # IR training config (final model)
│   ├── ir_final_cleaned_eval.yaml  # IR evaluation config
│   └── rgb_baseline.yaml       # RGB baseline config
│
├── drone_detector/             # RGB model
│   └── drone_detection.py
│
├── models/                     # Model weights (not tracked)
│   ├── pretrained/             # Base yolon26n pretrained weights
│   └── IR_final_cleaned/       # Final IR model (best.pt)
│
├── notebooks/
│   
│
├── runs/
│   └── IR_FT_final_cleaned_s0/  # Evaluation artifacts (metrics, curves)
│
├── scripts/
│   ├── train.py                # Training entry point
│   ├── eval.py                 # Evaluation (threshold sweep, size breakdown, TN)
│   ├── review_labels_gui.py    # Label reviewer GUI launcher
│   ├── label_reviewer/         # Human-in-the-loop label review toolkit
│   │   ├── core.py             # Review engine (4 modes)
│   │   ├── gui.py              # Tkinter GUI
│   │   └── predictor.py        # Model inference for review
│   ├── dataset_preparation/    # Dataset building & merging scripts
│   ├── eval/                   # Per-source evaluation tools
│   └── analysis/               # Audits, statistics, deduplication
│
└── requirements.txt
```

## Results — Final IR Model

**Dataset**: `ir_dset_final` — 129,130 images from 13 sources  
**Architecture**: YOLOv26n, fine-tuned 70 epochs  
**Optimal Threshold**: T* = 0.38 (F1-optimal, selected on val split)

### TEST Split — Overall Metrics

| Metric | Value | Note |
|--------|------:|------|
| Precision | 95.49% | YOLO .val() metrics |
| Recall | 97.95% | YOLO .val() metrics |
| F1 Score | 96.70% | YOLO .val() metrics |
| mAP@0.5 | 97.70% | threshold-agnostic |
| Precision @ T\*=0.38 | 96.52% | frozen operational threshold |
| Recall @ T\*=0.38 | 96.95% | frozen operational threshold |
| F1 @ T\*=0.38 | 96.73% | frozen operational threshold |
| True Negative Rate | 98.6% | 3,309 / 3,355 negative images |
| FPPI | 0.0137 | false positives per image |

### Size-Based Breakdown (TEST @ T\*=0.38)

| Size Bucket | Precision | Recall | TP | FP | FN | GT |
|-------------|----------:|-------:|---:|---:|---:|---:|
| Tiny | 94.81% | 95.53% | 3,380 | 185 | 158 | 3,720 |
| Medium | 98.92% | 98.80% | 2,467 | 27 | 30 | 2,315 |
| Large | 96.93% | 98.44% | 253 | 8 | 4 | 257 |

### Size-Based Breakdown (TEST @ T=0.40)

| Size Bucket | Precision | Recall | TP | FP | FN | GT |
|-------------|----------:|-------:|---:|---:|---:|---:|
| Tiny | 94.88% | 94.88% | 3,357 | 181 | 181 | 3,720 |
| Medium | 99.00% | 98.76% | 2,466 | 25 | 31 | 2,315 |
| Large | 96.93% | 98.44% | 253 | 8 | 4 | 257 |

### Model Progression

| Model | Sources | Images | TEST F1 | TEST mAP@0.5 |
|-------|--------:|-------:|--------:|--------------:|
| IR GoldV2 | 1 | 3.4K | 94.0% | 96.2% |
| IR dsetV4 | 6 | 33K | 92.7% | 95.2% |
| IR dsetV6 | 8 | 68K | 89.7% | 89.5% |
| **IR Final** | **13** | **129K** | **96.7%** | **97.7%** |

> See [`notebooks/ir_dset_final_results.ipynb`](notebooks/ir_dset_final_results.ipynb) for full visualizations, training curves, and per-source analysis.

## Label Reviewer — Human-in-the-Loop Dataset Cleaning

The label reviewer (`scripts/label_reviewer/`) is a GUI-based tool for iterative dataset refinement. It uses a **human-in-the-loop** approach where previous versions of the trained model are run on the dataset to surface labelling problems that the model struggles with:

- **FP Review**: Shows images where the model predicts a drone but no ground truth exists — surfaces missing annotations and false positive patterns.
- **FN Review**: Shows images where ground truth exists but the model fails to detect — reveals hard examples and incorrect labels.
- **GT Mismatch Review**: Shows images where predictions and ground truth overlap poorly — catches misaligned or incorrectly sized bounding boxes.
- **Full Review**: Manual inspection of all annotations for a given source.

This iterative cycle (train → evaluate → review errors → fix labels → retrain) was critical to scaling the dataset from 3K to 129K images while maintaining label quality. Each dataset version was cleaned using the previous version's model to identify and correct systematic labelling errors.

### Usage

```bash
python scripts/review_labels_gui.py
```

The GUI allows selecting a dataset path, a model weights file, and a review mode. Corrections are saved as updated YOLO-format label files.

## Quick Start

### Training
```bash
python scripts/train.py --config configs/ir_final_cleaned.yaml
```

### Evaluation
```bash
# VAL split — threshold sweep to find F1-optimal T*
python scripts/eval.py --config configs/ir_final_cleaned_eval.yaml --split val

# TEST split — frozen threshold from VAL
python scripts/eval.py --config configs/ir_final_cleaned_eval.yaml --split test --threshold 0.38
```

The eval script produces: `metrics.json`, `size_breakdown.json`, `confusion_matrix.json`, `threshold_sweep.csv`, and `pr_curve.json`. It computes per-size-bucket TP/FP/FN, true negative rates, and FPPI — metrics that YOLO's built-in `.val()` does not provide.

### Results Notebook
```bash
jupyter notebook notebooks/ir_dset_final_results.ipynb
```

## Key Concepts


- **Dataset iterations**: GoldV2 (1-src) → dsetV4 (6-src) → dsetV6 (8-src) → **ir_dset_final (13-src, 129K images)**
- **Human-in-the-loop cleaning**: Each dataset version was cleaned using the previous model's predictions to identify FP/FN/GT mismatches
- **Evaluation protocol**: F1-optimal threshold selection on VAL, frozen evaluation on TEST, with per-size and true-negative analysis
- **Run naming**: `{domain}_{type}_{dataset}_{seed}` (e.g., `IR_FT_final_cleaned_s0`)

## Dependencies

```bash
pip install -r requirements.txt
```

Core: `ultralytics`, `torch`, `opencv-python`, `numpy`, `matplotlib`, `pandas`, `tkinter` (stdlib)
