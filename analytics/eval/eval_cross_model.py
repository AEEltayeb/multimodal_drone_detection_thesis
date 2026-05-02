"""
Cross-model evaluation benchmark.

Runs multiple models against multiple test sets and produces a comparison table.
Output: CSV + console summary with P, R, F1, mAP50, mAP50-95 per model × dataset.

Usage:
    python scripts/eval_cross_model.py
    python scripts/eval_cross_model.py --models dsetV4 ir_gold --datasets cst
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Configure models and datasets here ──
PROJ = Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection")

MODELS = {
    "dsetV4": PROJ / "models" / "IR_dsetV4_300ep" / "best.pt",
    "dsetV6": PROJ / "models" / "IR_dsetV6_118ep" / "best.pt",
    "dsetV9b1": PROJ / "IR_dsetV9b1_rgbcfg" / "weights" / "best.pt",
    "ir_gold": PROJ / "IR_gold_rgbcfg" / "IR_gold_rgbcfg" / "weights" / "best.pt",
}

DATASETS = {
    "cst_5k": {
        "images": Path(r"G:\drone\_cst_eval_subset\test\images"),
        "labels": Path(r"G:\drone\_cst_eval_subset\test\labels"),
    },
    "dsetV6_test": {
        "images": Path(r"G:\drone\IR_dsetV6\test\images"),
        "labels": Path(r"G:\drone\IR_dsetV6\test\labels"),
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Matching / metrics ──

def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    pw, ph = w * img_w, h * img_h
    x1 = cx * img_w - pw / 2
    y1 = cy * img_h - ph / 2
    return (x1, y1, x1 + pw, y1 + ph)


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def match_preds(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.5):
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)
    tp_flags = [False] * len(pred_boxes)
    matched = set()
    for i in order:
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = compute_iou(pred_boxes[i], gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp_flags[i] = True
            matched.add(best_j)
    return tp_flags, matched


def classify_size(area):
    if area < 1024:
        return "tiny"
    elif area < 9216:
        return "medium"
    return "large"


# ── Evaluation ──

def evaluate_model_on_dataset(model, img_dir, lbl_dir, conf_thresh=0.001,
                               iou_thresh=0.5, imgsz=640):
    """Run model on all images and compute metrics."""
    image_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    total = len(image_files)
    
    all_preds = []
    all_gt = {}
    
    t0 = time.time()
    for idx, img_file in enumerate(image_files):
        results = model.predict(
            source=str(img_file), conf=conf_thresh, iou=iou_thresh,
            imgsz=imgsz, verbose=False, save=False, max_det=300,
        )
        
        r = results[0]
        stem = img_file.stem
        img_h, img_w = r.orig_shape
        
        pred_boxes, pred_confs = [], []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                pred_boxes.append(tuple(float(v) for v in xyxy[i]))
                pred_confs.append(float(confs[i]))
        
        all_preds.append((stem, pred_boxes, pred_confs))
        
        # Load GT
        gt_boxes = []
        lbl_file = lbl_dir / f"{stem}.txt"
        if lbl_file.exists():
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes.append(yolo_to_pixel_box(cx, cy, w, h, img_w, img_h))
        all_gt[stem] = gt_boxes
        
        done = idx + 1
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            print(f"    [{done:>5}/{total}] {rate:.1f} img/s  ETA {eta:.0f}s", flush=True)
    
    # Sweep thresholds to find best F1
    best_f1, best_t, best_metrics = 0, 0.5, {}
    
    for t in np.arange(0.05, 0.95, 0.05):
        tp = fp = fn = 0
        size_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
        
        for stem, pboxes, pconfs in all_preds:
            filt_boxes = [b for b, c in zip(pboxes, pconfs) if c >= t]
            filt_confs = [c for c in pconfs if c >= t]
            gt = all_gt.get(stem, [])
            
            flags, matched_gt = match_preds(filt_boxes, filt_confs, gt)
            
            _tp = sum(flags)
            _fp = len(filt_boxes) - _tp
            _fn = len(gt) - len(matched_gt)
            tp += _tp
            fp += _fp
            fn += _fn
            
            # Size breakdown
            for i, (box, is_tp) in enumerate(zip(filt_boxes, flags)):
                sz = classify_size(box_area(box))
                if is_tp:
                    size_stats[sz]["tp"] += 1
                else:
                    size_stats[sz]["fp"] += 1
            for j, gb in enumerate(gt):
                sz = classify_size(box_area(gb))
                size_stats[sz]["gt"] += 1
                if j not in matched_gt:
                    size_stats[sz]["fn"] += 1
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            
            # Size breakdown
            size_br = {}
            for sz in ["tiny", "medium", "large"]:
                s = size_stats[sz]
                sp = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
                sr = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
                size_br[sz] = {"P": round(sp, 4), "R": round(sr, 4), "gt": s["gt"]}
            
            best_metrics = {
                "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
                "T*": round(best_t, 2),
                "TP": tp, "FP": fp, "FN": fn,
                "images": total,
                "gt_total": sum(len(v) for v in all_gt.values()),
                "size": size_br,
            }
    
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-model evaluation benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to evaluate (default: all). Choices: {list(MODELS.keys())}")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to test on (default: all). Choices: {list(DATASETS.keys())}")
    parser.add_argument("--device", default="0", help="CUDA device")
    args = parser.parse_args()
    
    from ultralytics import YOLO
    
    model_names = args.models or list(MODELS.keys())
    dataset_names = args.datasets or list(DATASETS.keys())
    
    # Validate
    for m in model_names:
        if m not in MODELS:
            print(f"ERROR: Unknown model '{m}'. Available: {list(MODELS.keys())}")
            sys.exit(1)
        if not MODELS[m].exists():
            print(f"WARNING: Model weights not found: {MODELS[m]}")
            print(f"         Skipping '{m}'.")
            model_names.remove(m)
    
    for d in dataset_names:
        if d not in DATASETS:
            print(f"ERROR: Unknown dataset '{d}'. Available: {list(DATASETS.keys())}")
            sys.exit(1)
    
    results = []  # (model, dataset, metrics)
    
    print(f"\n{'='*70}")
    print(f"  CROSS-MODEL EVALUATION BENCHMARK")
    print(f"  Models:   {model_names}")
    print(f"  Datasets: {dataset_names}")
    print(f"{'='*70}\n")
    
    for model_name in model_names:
        weights_path = MODELS[model_name]
        print(f"\n{'─'*60}")
        print(f"  Loading model: {model_name}")
        print(f"  Weights: {weights_path}")
        print(f"{'─'*60}")
        
        model = YOLO(str(weights_path))
        
        for ds_name in dataset_names:
            ds = DATASETS[ds_name]
            print(f"\n  ► Evaluating {model_name} on {ds_name}...")
            
            metrics = evaluate_model_on_dataset(
                model, ds["images"], ds["labels"]
            )
            
            print(f"    P={metrics['P']:.4f}  R={metrics['R']:.4f}  "
                  f"F1={metrics['F1']:.4f}  T*={metrics['T*']:.2f}")
            
            # Size breakdown
            for sz in ["tiny", "medium", "large"]:
                if sz in metrics.get("size", {}):
                    s = metrics["size"][sz]
                    print(f"      {sz:>6}: P={s['P']:.3f} R={s['R']:.3f} (n={s['gt']})")
            
            results.append((model_name, ds_name, metrics))
    
    # ── Summary table ──
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    
    header = f"{'Model':<15} {'Dataset':<15} {'P':>8} {'R':>8} {'F1':>8} {'T*':>6} {'TP':>6} {'FP':>6} {'FN':>6}"
    print(header)
    print("─" * len(header))
    for model_name, ds_name, m in results:
        print(f"{model_name:<15} {ds_name:<15} {m['P']:>8.4f} {m['R']:>8.4f} "
              f"{m['F1']:>8.4f} {m['T*']:>6.2f} {m['TP']:>6} {m['FP']:>6} {m['FN']:>6}")
    
    # ── Save CSV ──
    out_dir = PROJ / "evaluation"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "cross_model_benchmark.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "dataset", "P", "R", "F1", "T_star",
                         "TP", "FP", "FN", "images", "gt_total",
                         "tiny_P", "tiny_R", "tiny_gt",
                         "medium_P", "medium_R", "medium_gt",
                         "large_P", "large_R", "large_gt"])
        for model_name, ds_name, m in results:
            row = [model_name, ds_name, m["P"], m["R"], m["F1"], m["T*"],
                   m["TP"], m["FP"], m["FN"], m["images"], m["gt_total"]]
            for sz in ["tiny", "medium", "large"]:
                s = m.get("size", {}).get(sz, {"P": 0, "R": 0, "gt": 0})
                row.extend([s["P"], s["R"], s["gt"]])
            writer.writerow(row)
    
    print(f"\nResults saved to: {csv_path}")
    
    # Save JSON too
    json_path = out_dir / "cross_model_benchmark.json"
    with open(json_path, "w") as f:
        json.dump([{"model": m, "dataset": d, "metrics": met}
                   for m, d, met in results], f, indent=2)
    print(f"JSON saved to:   {json_path}")


if __name__ == "__main__":
    main()
