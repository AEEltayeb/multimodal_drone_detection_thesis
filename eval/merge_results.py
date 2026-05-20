import csv
import json
from pathlib import Path

out_dir = Path("eval/results/thesis_ablation")
out_dir.mkdir(parents=True, exist_ok=True)

# Define all aggregated rows
rows = [
    # --- Section A: Video Tests ---
    {"dataset": "vid_drone", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 1359, "P": 0.972, "R": 0.749, "F1": 0.846},
    {"dataset": "vid_drone", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 1359, "P": 0.959, "R": 0.491, "F1": 0.649},
    {"dataset": "vid_drone", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 1359, "P": 0.967, "R": 0.672, "F1": 0.793},
    {"dataset": "vid_drone", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 1359, "P": 0.970, "R": 0.578, "F1": 0.724},
    {"dataset": "vid_drone", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 1359, "P": 0.957, "R": 0.784, "F1": 0.862},
    {"dataset": "vid_drone", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 1359, "P": 0.959, "R": 0.491, "F1": 0.649},
    {"dataset": "vid_drone", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 1359, "P": 0.966, "R": 0.650, "F1": 0.777},
    {"dataset": "vid_drone", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 1359, "P": 0.969, "R": 0.553, "F1": 0.704},

    {"dataset": "vid_airplanes", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.398},
    {"dataset": "vid_airplanes", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.217},
    {"dataset": "vid_airplanes", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.224},
    {"dataset": "vid_airplanes", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.092},
    {"dataset": "vid_airplanes", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.368},
    {"dataset": "vid_airplanes", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.217},
    {"dataset": "vid_airplanes", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.220},
    {"dataset": "vid_airplanes", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 304, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.089},

    {"dataset": "vid_birds", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.537},
    {"dataset": "vid_birds", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.074},
    {"dataset": "vid_birds", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.099},
    {"dataset": "vid_birds", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.068},
    {"dataset": "vid_birds", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.438},
    {"dataset": "vid_birds", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.074},
    {"dataset": "vid_birds", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.077},
    {"dataset": "vid_birds", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 352, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.062},

    {"dataset": "vid_helicopters", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.254},
    {"dataset": "vid_helicopters", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.066},
    {"dataset": "vid_helicopters", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.175},
    {"dataset": "vid_helicopters", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.130},
    {"dataset": "vid_helicopters", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.205},
    {"dataset": "vid_helicopters", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.066},
    {"dataset": "vid_helicopters", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.125},
    {"dataset": "vid_helicopters", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 594, "P": 0.000, "R": 0.000, "F1": 0.000, "FP%": 0.091},

    # --- Section B: IR native ---
    {"dataset": "ir_dset_test", "model": "ir_v3b", "stage": "raw", "rule": "iou", "n": 1374, "P": 0.972, "R": 0.964, "F1": 0.968},
    {"dataset": "ir_dset_test", "model": "ir_v3b", "stage": "ir_filter", "rule": "iou", "n": 1374, "P": 0.973, "R": 0.933, "F1": 0.953},

    # --- Section C: RGB Dataset ---
    {"dataset": "rgb_dataset_test", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 1435, "P": 0.997, "R": 0.923, "F1": 0.959},
    {"dataset": "rgb_dataset_test", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 1435, "P": 0.988, "R": 0.358, "F1": 0.526},
    {"dataset": "rgb_dataset_test", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 1435, "P": 0.996, "R": 0.688, "F1": 0.814},
    {"dataset": "rgb_dataset_test", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 1435, "P": 0.997, "R": 0.677, "F1": 0.807},
    {"dataset": "rgb_dataset_test", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 1435, "P": 0.995, "R": 0.923, "F1": 0.958},
    {"dataset": "rgb_dataset_test", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 1435, "P": 0.988, "R": 0.358, "F1": 0.526},
    {"dataset": "rgb_dataset_test", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 1435, "P": 0.995, "R": 0.684, "F1": 0.811},
    {"dataset": "rgb_dataset_test", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 1435, "P": 0.996, "R": 0.675, "F1": 0.804},

    # --- Section D: SelCom footage val split ---
    {"dataset": "selcom_val_split", "model": "baseline", "stage": "rgb", "rule": "iop", "n": 311, "P": 1.000, "R": 0.007, "F1": 0.013},
    {"dataset": "selcom_val_split", "model": "baseline", "stage": "ir_gray", "rule": "iop", "n": 311, "P": 1.000, "R": 0.027, "F1": 0.053},
    {"dataset": "selcom_val_split", "model": "baseline", "stage": "classifier", "rule": "iop", "n": 311, "P": 1.000, "R": 0.003, "F1": 0.007},
    {"dataset": "selcom_val_split", "model": "baseline", "stage": "clf+filter", "rule": "iop", "n": 311, "P": 1.000, "R": 0.003, "F1": 0.007},
    {"dataset": "selcom_val_split", "model": "selcom_960", "stage": "rgb", "rule": "iop", "n": 311, "P": 1.000, "R": 0.439, "F1": 0.610},
    {"dataset": "selcom_val_split", "model": "selcom_960", "stage": "ir_gray", "rule": "iop", "n": 311, "P": 1.000, "R": 0.027, "F1": 0.053},
    {"dataset": "selcom_val_split", "model": "selcom_960", "stage": "classifier", "rule": "iop", "n": 311, "P": 1.000, "R": 0.245, "F1": 0.393},
    {"dataset": "selcom_val_split", "model": "selcom_960", "stage": "clf+filter", "rule": "iop", "n": 311, "P": 1.000, "R": 0.245, "F1": 0.393},
]

# Write CSV
csv_path = out_dir / "thesis_ablation.csv"
all_keys = ["dataset", "model", "stage", "rule", "n", "P", "R", "F1", "FP%"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"Merged CSV saved to: {csv_path}")

# Write JSON
json_path = out_dir / "thesis_ablation.json"
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Merged JSON saved to: {json_path}")
