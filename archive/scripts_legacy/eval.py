"""
eval.py — Config-driven YOLO evaluation.

Primary metrics come from YOLO's model.val() (COCO evaluation protocol).
Optional threshold sweep for operational threshold selection.

Usage:
    # VAL threshold sweep (find T*)
    python scripts/eval.py --config configs/rgb_baseline.yaml --split val

    # TEST evaluation with frozen threshold
    python scripts/eval.py --config configs/rgb_baseline.yaml --split test --threshold 0.45

    # Negative-only FPPI evaluation
    python scripts/eval.py --config configs/rgb_baseline.yaml --split neg_test --threshold 0.45

Implements :
    - Precision-floor threshold selection (§7.2)
    - TP/FP matching at IoU ≥ matching_iou, one-to-one (§7.2a)
    - mAP@0.5 and mAP@0.5:0.95 via ultralytics model.val() (§7.1)
    - Size-bucket breakdown (§7.5)
    - FPPI on negative-only splits (§7.6)

Produces all required artifacts in runs/<run_name>/ per §7.3.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config loading (shared with train.py)
# ---------------------------------------------------------------------------

def load_config(config_path: str, device_profile_cli: str = None) -> dict:
    """Load base.yaml → experiment config → device profile (layered)."""
    config_dir = Path(config_path).parent
    base_path = config_dir / "base.yaml"

    cfg = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    with open(config_path, "r", encoding="utf-8") as f:
        override = yaml.safe_load(f) or {}
    cfg.update({k: v for k, v in override.items() if v is not None})

    device_profile = device_profile_cli or cfg.get("device_profile")
    if device_profile:
        dp_path = Path(device_profile)
        if not dp_path.exists():
            dp_path = Path(config_path).parent.parent / device_profile
        if dp_path.exists():
            with open(dp_path, "r", encoding="utf-8") as f:
                dp = yaml.safe_load(f) or {}
            cfg.update({k: v for k, v in dp.items() if v is not None})

    return cfg


def resolve_output_dir(cfg: dict) -> Path:
    template = cfg.get("output_dir", "runs/${run_name}")
    run_name = cfg["run_name"]
    return Path(template.replace("${run_name}", run_name))


# ---------------------------------------------------------------------------
# Dataset path resolution
# ---------------------------------------------------------------------------

def resolve_split_dirs(dataset_yaml_path: str, split: str):
    """
    Resolve image and label directories for a given split.

    For neg_test: requires explicit 'neg_test' key in dataset YAML.
    Never falls back silently to a different split.

    Returns:
        (image_dir: Path, label_dir: Path)
    """
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)

    ds_root = Path(dataset_yaml_path).parent
    if "path" in ds_cfg:
        ds_root = Path(ds_cfg["path"])

    # Negative splits must be explicit — no silent fallback
    if split == "neg_test":
        if "neg_test" not in ds_cfg:
            print("[ERROR] neg_test split requires an explicit 'neg_test' key in dataset YAML.")
            print("        Add  neg_test: images/neg_test  to your dataset YAML.")
            print("        DO NOT fall back to the regular test split — that would break FPPI.")
            sys.exit(1)
        split_rel = ds_cfg["neg_test"]
    elif split in ds_cfg:
        split_rel = ds_cfg[split]
    else:
        # Standard split name mapping
        split_map = {"val": "val", "test": "test", "train": "train"}
        mapped = split_map.get(split)
        if mapped and mapped in ds_cfg:
            split_rel = ds_cfg[mapped]
        else:
            # Final fallback: assume standard YOLO layout
            split_rel = f"images/{split_map.get(split, split)}"

    image_dir = ds_root / split_rel
    if not image_dir.exists():
        image_dir = Path(split_rel)  # try as absolute

    # Labels: mirror image path with 'images' → 'labels'
    label_dir = Path(str(image_dir).replace("images", "labels"))

    return image_dir, label_dir


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    pw = w * img_w
    ph = h * img_h
    x1 = (cx * img_w) - pw / 2
    y1 = (cy * img_h) - ph / 2
    return (x1, y1, x1 + pw, y1 + ph)


def box_area(box):
    """Area of (x1, y1, x2, y2) box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ---------------------------------------------------------------------------
# TP/FP matching (§7.2a)
# ---------------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(pred_boxes, pred_confs, gt_boxes, matching_iou=0.5):
    """
    One-to-one greedy matching per §7.2a.

    Returns:
        tp_flags: list[bool] — one per prediction, True if matched
        matched_gt_indices: set of GT indices that were matched
    """
    if not pred_boxes or not gt_boxes:
        return [False] * len(pred_boxes), set()

    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)

    tp_flags = [False] * len(pred_boxes)
    matched_gt = set()

    for pred_idx in order:
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx

        if best_iou >= matching_iou and best_gt >= 0:
            tp_flags[pred_idx] = True
            matched_gt.add(best_gt)

    return tp_flags, matched_gt


# ---------------------------------------------------------------------------
# Size bucket classification (§7.5)
# ---------------------------------------------------------------------------

def classify_size(area, tiny_max=1024, medium_max=9216):
    """Classify box area into tiny/medium/large bucket."""
    if area < tiny_max:
        return "tiny"
    elif area < medium_max:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# mAP computation via ultralytics (§7.1)
# ---------------------------------------------------------------------------

def compute_yolo_metrics(model, dataset_yaml: str, split: str, imgsz: int,
                         device: str, iou_threshold: float):
    """
    Compute all metrics using ultralytics model.val() (COCO evaluation protocol).

    This is the AUTHORITATIVE source for P, R, F1, mAP.
    Returns dict with precision, recall, f1, mAP50, mAP50_95.
    """
    # Map our split names to ultralytics split names
    ul_split = split

    print(f"  Running model.val(split='{ul_split}')...")

    val_results = model.val(
        data=dataset_yaml,
        split=ul_split,
        imgsz=imgsz,
        device=device,
        iou=iou_threshold,
        verbose=False,
        plots=False,
    )

    # Extract all metrics from ultralytics results
    box = val_results.box
    precision = float(box.mp)   # mean precision across classes
    recall = float(box.mr)      # mean recall across classes
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    map50 = float(box.map50)
    map50_95 = float(box.map)

    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {map50_95:.4f}")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }


# ---------------------------------------------------------------------------
# Evaluation at a single threshold
# ---------------------------------------------------------------------------

def evaluate_at_threshold(all_preds, all_gt, threshold, matching_iou=0.5,
                          size_buckets=None):
    """
    Evaluate predictions at a single confidence threshold.

    Returns dict with TP, FP, FN, precision, recall, F1, and size breakdown.
    """
    tiny_max = (size_buckets or {}).get("tiny_max", 1024)
    medium_max = (size_buckets or {}).get("medium_max", 9216)

    total_tp = total_fp = total_fn = 0
    total_neg_images = 0       # images with no GT
    total_tn = 0               # negatives correctly ignored (no preds above T)
    total_fp_on_neg = 0        # negatives with false alarms
    size_tp = defaultdict(int)
    size_fp = defaultdict(int)
    size_fn = defaultdict(int)
    size_gt_count = defaultdict(int)

    for image_stem, pred_boxes, pred_confs in all_preds:
        filtered_boxes = []
        filtered_confs = []
        for box, conf in zip(pred_boxes, pred_confs):
            if conf >= threshold:
                filtered_boxes.append(box)
                filtered_confs.append(conf)

        gt_boxes = all_gt.get(image_stem, [])

        # TN tracking: image has no GT
        if len(gt_boxes) == 0:
            total_neg_images += 1
            if len(filtered_boxes) == 0:
                total_tn += 1
            else:
                total_fp_on_neg += 1

        tp_flags, matched_gt = match_predictions_to_gt(
            filtered_boxes, filtered_confs, gt_boxes, matching_iou
        )

        tp = sum(tp_flags)
        fp = len(filtered_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for i, (box, is_tp) in enumerate(zip(filtered_boxes, tp_flags)):
            area = box_area(box)
            bucket = classify_size(area, tiny_max, medium_max)
            if is_tp:
                size_tp[bucket] += 1
            else:
                size_fp[bucket] += 1

        for gt_idx, gt_box in enumerate(gt_boxes):
            area = box_area(gt_box)
            bucket = classify_size(area, tiny_max, medium_max)
            size_gt_count[bucket] += 1
            if gt_idx not in matched_gt:
                size_fn[bucket] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    size_breakdown = {}
    for bucket in ["tiny", "medium", "large"]:
        b_tp = size_tp.get(bucket, 0)
        b_fp = size_fp.get(bucket, 0)
        b_fn = size_fn.get(bucket, 0)
        b_gt = size_gt_count.get(bucket, 0)
        b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        size_breakdown[bucket] = {
            "precision": round(b_prec, 4),
            "recall": round(b_rec, 4),
            "tp": b_tp, "fp": b_fp, "fn": b_fn, "gt_count": b_gt,
        }

    return {
        "threshold": round(threshold, 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "tn": total_tn,
        "neg_images": total_neg_images,
        "fp_on_neg": total_fp_on_neg,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "size_breakdown": size_breakdown,
    }


# ---------------------------------------------------------------------------
# Threshold sweep — F1-optimal
# ---------------------------------------------------------------------------

def threshold_sweep(all_preds, all_gt, cfg):
    """
    Sweep confidence thresholds and find T* (F1-optimal).

    Selects the threshold that maximizes F1 score.
    Tie-break: higher recall, then fewer FP.
    """
    sweep_start = cfg.get("sweep_start", 0.01)
    sweep_end = cfg.get("sweep_end", 0.99)
    sweep_step = cfg.get("sweep_step", 0.01)
    matching_iou = cfg.get("matching_iou", 0.5)
    size_buckets = cfg.get("size_buckets", {})

    thresholds = np.arange(sweep_start, sweep_end + sweep_step / 2, sweep_step)
    sweep_results = []

    print(f"  Sweeping {len(thresholds)} thresholds [{sweep_start:.2f} → {sweep_end:.2f}]...")

    for t in thresholds:
        result = evaluate_at_threshold(
            all_preds, all_gt, float(t), matching_iou, size_buckets
        )
        sweep_results.append(result)

    best = max(sweep_results, key=lambda r: (r["f1"], r["recall"], -r["fp"]))
    print(f"  T* = {best['threshold']:.2f}  (F1-optimal: F1={best['f1']:.4f}, "
          f"P={best['precision']:.4f}, R={best['recall']:.4f})")

    return sweep_results, best


# ---------------------------------------------------------------------------
# FPPI evaluation (§7.6)
# ---------------------------------------------------------------------------

def evaluate_fppi(all_preds, threshold):
    """Compute False Positives Per Image on a negative-only split."""
    total_images = len(all_preds)
    total_fp = 0
    fp_images = []  # images that triggered false positives

    for stem, pred_boxes, pred_confs in all_preds:
        fp_confs = [c for c in pred_confs if c >= threshold]
        n_fp = len(fp_confs)
        total_fp += n_fp
        if n_fp > 0:
            fp_images.append({
                "image": stem,
                "num_fp": n_fp,
                "confidences": sorted(fp_confs, reverse=True),
            })

    fppi = total_fp / total_images if total_images > 0 else 0.0

    # Sort by number of FP descending
    fp_images.sort(key=lambda x: x["num_fp"], reverse=True)

    return {
        "fppi": round(fppi, 6),
        "total_fp": total_fp,
        "total_images": total_images,
        "threshold": round(threshold, 4),
        "fp_image_count": len(fp_images),
        "fp_images": fp_images,
    }


# ---------------------------------------------------------------------------
# Inference + GT collection
# ---------------------------------------------------------------------------

def collect_predictions_and_gt(model, cfg, split, dataset_yaml_override=None):
    """
    Run inference and collect predictions + ground truth per image.

    Args:
        dataset_yaml_override: if set, use this dataset YAML instead of cfg["dataset_yaml"]

    Returns:
        all_preds: list of (image_stem, pred_boxes_px, pred_confs)
        all_gt: dict {image_stem: [gt_boxes_px]}
    """
    dataset_yaml_path = dataset_yaml_override or cfg["dataset_yaml"]
    print(f"  Dataset YAML: {dataset_yaml_path}")
    imgsz = cfg.get("image_size", 640)
    device = cfg.get("device", "0")
    iou_threshold = cfg.get("iou_threshold", 0.7)

    image_dir, label_dir = resolve_split_dirs(dataset_yaml_path, split)

    print(f"  Image dir: {image_dir}")
    print(f"  Label dir: {label_dir}")

    if not image_dir.exists():
        print(f"  [ERROR] Image directory not found: {image_dir}")
        sys.exit(1)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ])

    total_files = len(image_files)
    print(f"  Found {total_files} images in '{split}' split")

    all_preds = []
    all_gt = {}

    import time
    t_start = time.time()

    # Process one image at a time to avoid ultralytics autocast_list
    # (which pre-opens every image for EXIF data when given a list)
    for idx, img_file in enumerate(image_files):
        try:
            results = model.predict(
                source=str(img_file),
                conf=0.001,  # very low — we filter per-threshold later
                iou=iou_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
                save=False,
                max_det=300,
            )
        except Exception as e:
            print(f"  [WARN] Skipping corrupt image: {img_file.name} ({e})")
            continue

        result = results[0]
        stem = img_file.stem
        img_h, img_w = result.orig_shape

        pred_boxes = []
        pred_confs = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                pred_boxes.append(tuple(float(v) for v in xyxy[i]))
                pred_confs.append(float(confs[i]))

        all_preds.append((stem, pred_boxes, pred_confs))

        gt_boxes_px = []
        label_file = label_dir / f"{stem}.txt"
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        gt_boxes_px.append(yolo_to_pixel_box(cx, cy, w, h, img_w, img_h))

        all_gt[stem] = gt_boxes_px

        processed = idx + 1
        if processed % 100 == 0 or processed == total_files:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_files - processed) / rate if rate > 0 else 0
            print(f"  [{processed}/{total_files}] {rate:.1f} img/s — ETA {remaining:.0f}s", flush=True)

    elapsed_total = time.time() - t_start
    print(f"  Inference done in {elapsed_total:.1f}s ({total_files / elapsed_total:.1f} img/s)")

    return all_preds, all_gt


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_sweep_csv(sweep_results, out_path):
    """Save threshold_sweep.csv."""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        for r in sweep_results:
            writer.writerow([
                r["threshold"], r["precision"], r["recall"], r["f1"],
                r["tp"], r["fp"], r["fn"]
            ])


def save_pr_curve(sweep_results, out_path):
    """Save pr_curve.json — per-threshold precision/recall pairs."""
    pr_data = [
        {"threshold": r["threshold"], "precision": r["precision"], "recall": r["recall"]}
        for r in sweep_results
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pr_data, f, indent=2)


def load_json_if_exists(path):
    """Load existing JSON file or return empty dict."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(data, path):
    """Write dict to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_confusion_matrix(tp, fp, fn, tn, threshold, out_path):
    """Generate and save a 2x2 confusion matrix plot matching YOLO's exact style."""
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Build matrix: rows = Predicted, cols = True  (same as YOLO)
    # [pred=drone, true=drone]  [pred=drone, true=bg]
    # [pred=bg,    true=drone]  [pred=bg,    true=bg]
    array = np.array([[tp, fp],
                       [fn, tn]], dtype=float)

    # Match YOLO: set very small values to NaN so they aren't annotated
    array_display = array.copy()
    array_display[array_display < 0.005] = np.nan

    labels = ["drone", "background"]
    nc = len(labels)

    # Match YOLO: figsize=(12, 9)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    tick_fontsize = 15   # YOLO: max(6, 15 - 0.1 * nc) → 14.8 ≈ 15
    label_fontsize = 12  # YOLO: max(6, 12 - 0.1 * nc) → 11.8 ≈ 12
    title_fontsize = 12

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.imshow(array_display, cmap="Blues", vmin=0.0, interpolation="none")
        ax.xaxis.set_label_position("bottom")

        # Annotate cells — match YOLO exactly
        color_threshold = 0.45 * np.nanmax(array_display)
        for i in range(nc):
            for j in range(nc):
                val = array_display[i, j]
                if np.isnan(val):
                    continue
                ax.text(
                    j, i,
                    f"{int(val)}",
                    ha="center", va="center",
                    fontsize=10,
                    color="white" if val > color_threshold else "black",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)

    ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
    ax.set_title(f"Confusion Matrix (T* = {threshold})", fontsize=title_fontsize, pad=20)
    ax.set_xticks(np.arange(nc))
    ax.set_yticks(np.arange(nc))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=90, ha="center")
    ax.set_yticklabels(labels, fontsize=tick_fontsize)

    # Remove spines — match YOLO
    for s in {"left", "right", "bottom", "top", "outline"}:
        if s != "outline":
            ax.spines[s].set_visible(False)
        cbar.ax.spines[s].set_visible(False)

    fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=0.25)
    fig.savefig(str(out_path), dpi=250)
    plt.close(fig)
    print(f"  confusion_matrix_at_T.png ✓")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Config-driven YOLO evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--split", required=True,
                        help="Split to evaluate on (dev, val, test, neg_test, etc.)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Frozen confidence threshold (required for test/neg_test)")
    parser.add_argument("--device-profile", default=None,
                        help="Path to device profile YAML")
    parser.add_argument("--weights", default=None,
                        help="Override model weights path")
    parser.add_argument("--dataset-yaml", default=None,
                        help="Override dataset YAML (e.g. for FPPI with separate neg dataset)")
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(args.config, device_profile_cli=args.device_profile)
    out_dir = resolve_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["run_name"]

    # ── Split ──
    split = args.split

    if split in ("test", "neg_test") and args.threshold is None:
        print("[ERROR] --threshold is required for test/neg_test splits.")
        print("        Run on val first to determine T*, then pass it here.")
        sys.exit(1)

    # ── Load model ──
    from ultralytics import YOLO
    weights = args.weights or cfg.get("pretrained_weights", "yolo26n.pt")
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {run_name}")
    print(f"  Split:      {split}")
    print(f"  Weights:    {weights}")
    print(f"  Threshold:  {args.threshold or 'sweep (F1-optimal)'}")
    print(f"  Run grade:  {cfg.get('run_grade', 'PUB')}")
    if args.dataset_yaml:
        print(f"  Dataset:    {args.dataset_yaml} (override)")
    print(f"{'='*60}\n")

    model = YOLO(weights)

    # ── Collect predictions + GT ──
    print("[1/4] Running inference and collecting ground truth...")
    all_preds, all_gt = collect_predictions_and_gt(
        model, cfg, split, dataset_yaml_override=args.dataset_yaml
    )

    total_gt_boxes = sum(len(v) for v in all_gt.values())
    total_pred_boxes = sum(len(p[1]) for p in all_preds)
    print(f"  Total images: {len(all_preds)}")
    print(f"  Total GT boxes: {total_gt_boxes}")
    print(f"  Total predictions (conf>0.001): {total_pred_boxes}")

    # ── FPPI mode (negative-only split) ──
    if split == "neg_test":
        print("\n[2/4] Computing FPPI on negative-only split...")
        fppi_result = evaluate_fppi(all_preds, args.threshold)

        # Enrich with provenance metadata
        effective_yaml = args.dataset_yaml or cfg.get("dataset_yaml", "")
        fppi_output = {
            "neg_dataset_id": Path(effective_yaml).stem if effective_yaml else "unknown",
            "neg_dataset_yaml": effective_yaml,
            "model_weights": weights,
            "threshold": args.threshold,
            "total_fp": fppi_result["total_fp"],
            "total_images": fppi_result["total_images"],
            "fppi": fppi_result["fppi"],
        }

        # Try to load manifest hash
        manifest_path = Path(effective_yaml).parent / Path(effective_yaml).stem / "split_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                fppi_output["neg_dataset_manifest_sha256"] = manifest.get("manifest_sha256", "unknown")
        else:
            fppi_output["neg_dataset_manifest_sha256"] = "manifest_not_found"

        fppi_output["fp_image_count"] = fppi_result["fp_image_count"]
        fppi_output["fp_images"] = fppi_result["fp_images"]

        fppi_path = out_dir / "neg_test_fppi.json"
        save_json(fppi_output, fppi_path)

        print(f"  FPPI: {fppi_output['fppi']:.6f}")
        print(f"  Total FP: {fppi_output['total_fp']}")
        print(f"  Total images: {fppi_output['total_images']}")
        print(f"  Images with FP: {fppi_output['fp_image_count']}")
        print(f"  Dataset: {fppi_output['neg_dataset_id']}")

        if fppi_result["fp_images"]:
            print(f"\n  False positive detections:")
            for entry in fppi_result["fp_images"]:
                confs_str = ", ".join(f"{c:.3f}" for c in entry["confidences"])
                print(f"    {entry['image']}  ({entry['num_fp']} FP)  conf=[{confs_str}]")

            # Save annotated FP images with bounding boxes
            import cv2
            from datetime import datetime

            fp_dir = out_dir / f"fp_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            fp_dir.mkdir(parents=True, exist_ok=True)

            # Build stem → (pred_boxes, pred_confs) lookup
            pred_lookup = {}
            for stem, pred_boxes, pred_confs in all_preds:
                pred_lookup[stem] = (pred_boxes, pred_confs)

            # Resolve image directory from the dataset YAML
            effective_yaml = args.dataset_yaml or cfg["dataset_yaml"]
            image_dir, _ = resolve_split_dirs(effective_yaml, split)

            fp_stems = {entry["image"] for entry in fppi_result["fp_images"]}
            img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

            print(f"\n  Saving annotated FP images to: {fp_dir}")
            saved = 0
            for img_file in sorted(image_dir.iterdir()):
                if img_file.stem not in fp_stems:
                    continue
                if img_file.suffix.lower() not in img_extensions:
                    continue

                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                boxes, confs = pred_lookup.get(img_file.stem, ([], []))
                for box, conf in zip(boxes, confs):
                    if conf >= args.threshold:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"FP {conf:.3f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
                        cv2.putText(img, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                out_path = fp_dir / img_file.name
                cv2.imwrite(str(out_path), img)
                saved += 1

            print(f"  Saved {saved} annotated images")

        print(f"\n  Saved to: {fppi_path}")
        return

    # ── Compute YOLO metrics (authoritative P/R/F1/mAP) ──
    effective_yaml = args.dataset_yaml or cfg.get("dataset_yaml")
    print(f"\n[2/4] Computing metrics via YOLO model.val() (COCO protocol)...")
    yolo_metrics = compute_yolo_metrics(
        model, effective_yaml, split,
        cfg.get("image_size", 640),
        cfg.get("device", "0"),
        cfg.get("iou_threshold", 0.7),
    )

    # ── VAL: threshold sweep (for operational threshold selection) ──
    if split == "val" and args.threshold is None:
        print("\n[3/4] Running threshold sweep (for operational threshold selection)...")
        sweep_results, best = threshold_sweep(all_preds, all_gt, cfg)

        t_star = best["threshold"]
        print(f"\n  Operational threshold T* = {t_star:.4f}")

        # Save artifacts
        print("\n[4/4] Saving artifacts...")

        save_sweep_csv(sweep_results, out_dir / "threshold_sweep.csv")
        save_pr_curve(sweep_results, out_dir / "pr_curve.json")

        save_json(
            {"val": best["size_breakdown"], "threshold": t_star},
            out_dir / "size_breakdown.json"
        )

        save_json(
            {
                "val": {
                    "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
                    "tn": best["tn"], "neg_images": best["neg_images"], "fp_on_neg": best["fp_on_neg"]
                },
                "threshold": t_star
            },
            out_dir / "confusion_matrix.json"
        )
        plot_confusion_matrix(
            best["tp"], best["fp"], best["fn"], best["tn"],
            t_star, out_dir / "confusion_matrix_at_T.png"
        )

        # Metrics (val portion) — primary metrics from YOLO
        metrics = load_json_if_exists(out_dir / "metrics.json")
        metrics["val"] = {
            "precision": yolo_metrics["precision"],
            "recall": yolo_metrics["recall"],
            "f1": yolo_metrics["f1"],
            "mAP50": yolo_metrics["mAP50"],
            "mAP50_95": yolo_metrics["mAP50_95"],
            "threshold": t_star,
            "tp": best["tp"],
            "fp": best["fp"],
            "fn": best["fn"],
            "tn": best["tn"],
            "neg_images": best["neg_images"],
            "fp_on_neg": best["fp_on_neg"]
        }
        metrics["val_threshold"] = t_star
        metrics["run_name"] = run_name
        metrics["run_grade"] = cfg.get("run_grade", "PUB")
        save_json(metrics, out_dir / "metrics.json")

        # ── Print all metrics ──
        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  YOLO COCO Metrics (multi-threshold, best-point)       │")
        print(f"  │  P={yolo_metrics['precision']:.4f}  R={yolo_metrics['recall']:.4f}  "
              f"F1={yolo_metrics['f1']:.4f}  mAP50={yolo_metrics['mAP50']:.4f}  "
              f"mAP50-95={yolo_metrics['mAP50_95']:.4f}  │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  @ T*={t_star:.4f} (sweep-selected threshold)              │")
        print(f"  │  P={best['precision']:.4f}  R={best['recall']:.4f}  F1={best['f1']:.4f}"
              f"  TP={best['tp']}  FP={best['fp']}  FN={best['fn']:<6}│")
        print(f"  └─────────────────────────────────────────────────────────┘")

        # Per-size breakdown at T*
        sb = best["size_breakdown"]
        print(f"\n  PER-SIZE BREAKDOWN @ T*={t_star:.4f}")
        print(f"  {'Bucket':<12} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'─'*70}")
        med_large_tp = med_large_fp = med_large_fn = med_large_gt = 0
        for bucket in ["tiny", "medium", "large"]:
            if bucket in sb:
                b = sb[bucket]
                b_tp, b_fp, b_fn = b["tp"], b["fp"], b["fn"]
                b_gt = b["gt_count"]
                b_p = b["precision"]
                b_r = b["recall"]
                b_f1 = 2 * b_p * b_r / max(1e-9, b_p + b_r)
                print(f"  {bucket:<12} {b_gt:>6} {b_tp:>6} {b_fp:>6} {b_fn:>6} {b_p:>8.1%} {b_r:>8.1%} {b_f1:>8.1%}")
                if bucket in ("medium", "large"):
                    med_large_tp += b_tp
                    med_large_fp += b_fp
                    med_large_fn += b_fn
                    med_large_gt += b_gt

        ml_p = med_large_tp / max(1, med_large_tp + med_large_fp)
        ml_r = med_large_tp / max(1, med_large_tp + med_large_fn)
        ml_f1 = 2 * ml_p * ml_r / max(1e-9, ml_p + ml_r)
        print(f"  {'─'*70}")
        print(f"  {'medium+large':<12} {med_large_gt:>6} {med_large_tp:>6} {med_large_fp:>6} "
              f"{med_large_fn:>6} {ml_p:>8.1%} {ml_r:>8.1%} {ml_f1:>8.1%}")

        # Negative image stats
        neg_imgs = best["neg_images"]
        fp_on_neg = best["fp_on_neg"]
        tn = best["tn"]
        tn_rate = tn / max(1, neg_imgs)
        fppi = fp_on_neg / max(1, neg_imgs)
        print(f"\n  NEGATIVE IMAGE STATS @ T*={t_star:.4f}")
        print(f"  Total negatives: {neg_imgs:,}")
        print(f"  TN (correct):    {tn:,}  ({tn_rate:.1%})")
        print(f"  FP on negatives: {fp_on_neg:,}  ({fppi:.1%})")
        print(f"  FPPI rate:       {fppi:.4f}")

        print(f"\n  threshold_sweep.csv   ✓")
        print(f"  pr_curve.json         ✓")
        print(f"  size_breakdown.json   ✓")
        print(f"  confusion_matrix.json ✓")
        print(f"  metrics.json (val)    ✓")
        print(f"\n  Next: python scripts/eval.py --config {args.config} --split test --threshold {t_star:.4f}")

    # ── TEST or VAL with frozen threshold ──
    else:
        threshold = args.threshold
        print(f"\n[3/4] Frozen threshold T={threshold:.4f} (for operational use)")

        # Compute metrics at the frozen threshold FIRST
        frozen_result = evaluate_at_threshold(
            all_preds, all_gt, threshold,
            matching_iou=cfg.get("matching_iou", 0.5),
            size_buckets=cfg.get("size_buckets", {})
        )

        # ── Print all metrics ──
        fr = frozen_result
        fr_p = fr["precision"]
        fr_r = fr["recall"]
        fr_f1 = fr["f1"]

        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  YOLO COCO Metrics (multi-threshold, best-point)       │")
        print(f"  │  P={yolo_metrics['precision']:.4f}  R={yolo_metrics['recall']:.4f}  "
              f"F1={yolo_metrics['f1']:.4f}  mAP50={yolo_metrics['mAP50']:.4f}  "
              f"mAP50-95={yolo_metrics['mAP50_95']:.4f}  │")
        print(f"  ├─────────────────────────────────────────────────────────┤")
        print(f"  │  @ Frozen T={threshold:.4f} (deployment metrics)            │")
        print(f"  │  P={fr_p:.4f}  R={fr_r:.4f}  F1={fr_f1:.4f}"
              f"  TP={fr['tp']}  FP={fr['fp']}  FN={fr['fn']:<6}│")
        print(f"  └─────────────────────────────────────────────────────────┘")

        # Per-size breakdown
        sb = fr["size_breakdown"]
        print(f"\n  PER-SIZE BREAKDOWN @ T={threshold:.4f}")
        print(f"  {'Bucket':<12} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'─'*70}")
        med_large_tp = med_large_fp = med_large_fn = med_large_gt = 0
        for bucket in ["tiny", "medium", "large"]:
            if bucket in sb:
                b = sb[bucket]
                b_tp, b_fp, b_fn = b["tp"], b["fp"], b["fn"]
                b_gt = b["gt_count"]
                b_p = b["precision"]
                b_r = b["recall"]
                b_f1 = 2 * b_p * b_r / max(1e-9, b_p + b_r)
                print(f"  {bucket:<12} {b_gt:>6} {b_tp:>6} {b_fp:>6} {b_fn:>6} {b_p:>8.1%} {b_r:>8.1%} {b_f1:>8.1%}")
                if bucket in ("medium", "large"):
                    med_large_tp += b_tp
                    med_large_fp += b_fp
                    med_large_fn += b_fn
                    med_large_gt += b_gt

        ml_p = med_large_tp / max(1, med_large_tp + med_large_fp)
        ml_r = med_large_tp / max(1, med_large_tp + med_large_fn)
        ml_f1 = 2 * ml_p * ml_r / max(1e-9, ml_p + ml_r)
        print(f"  {'─'*70}")
        print(f"  {'medium+large':<12} {med_large_gt:>6} {med_large_tp:>6} {med_large_fp:>6} "
              f"{med_large_fn:>6} {ml_p:>8.1%} {ml_r:>8.1%} {ml_f1:>8.1%}")

        # Negative image stats / FPPI / TN
        neg_imgs = fr["neg_images"]
        fp_on_neg = fr["fp_on_neg"]
        tn = fr["tn"]
        tn_rate = tn / max(1, neg_imgs)
        fppi = fp_on_neg / max(1, neg_imgs)
        print(f"\n  NEGATIVE IMAGE STATS @ T={threshold:.4f}")
        print(f"  Total negatives: {neg_imgs:,}")
        print(f"  TN (correct):    {tn:,}  ({tn_rate:.1%})")
        print(f"  FP on negatives: {fp_on_neg:,}  ({fppi:.1%})")
        print(f"  FPPI rate:       {fppi:.4f}")

        # Save artifacts
        print(f"\n[4/4] Saving artifacts...")

        metrics = load_json_if_exists(out_dir / "metrics.json")
        metrics[split] = {
            "precision": yolo_metrics["precision"],
            "recall": yolo_metrics["recall"],
            "f1": yolo_metrics["f1"],
            "mAP50": yolo_metrics["mAP50"],
            "mAP50_95": yolo_metrics["mAP50_95"],
            "threshold": threshold,
            "frozen_precision": fr_p,
            "frozen_recall": fr_r,
            "frozen_f1": fr_f1,
            "tp": fr["tp"],
            "fp": fr["fp"],
            "fn": fr["fn"],
            "tn": fr["tn"],
            "neg_images": fr["neg_images"],
            "fp_on_neg": fr["fp_on_neg"],
            "fppi": round(fppi, 4),
        }
        metrics["run_name"] = run_name
        metrics["run_grade"] = cfg.get("run_grade", "PUB")
        save_json(metrics, out_dir / "metrics.json")

        save_json(
            {split: frozen_result["size_breakdown"], "threshold": threshold},
            out_dir / "size_breakdown.json"
        )

        save_json(
            {
                split: {
                    "tp": fr["tp"], "fp": fr["fp"], "fn": fr["fn"],
                    "tn": fr["tn"], "neg_images": fr["neg_images"], "fp_on_neg": fr["fp_on_neg"]
                },
                "threshold": threshold
            },
            out_dir / "confusion_matrix.json"
        )
        plot_confusion_matrix(
            fr["tp"], fr["fp"], fr["fn"], fr["tn"],
            threshold, out_dir / "confusion_matrix_at_T.png"
        )

        # Compute frozen threshold P/R/F1
        fr_tp, fr_fp, fr_fn = frozen_result["tp"], frozen_result["fp"], frozen_result["fn"]
        fr_p = fr_tp / max(1, fr_tp + fr_fp)
        fr_r = fr_tp / max(1, fr_tp + fr_fn)
        fr_f1 = 2 * fr_p * fr_r / max(1e-9, fr_p + fr_r)

        print(f"  metrics.json ({split})  ✓  [P={yolo_metrics['precision']:.4f} R={yolo_metrics['recall']:.4f} F1={yolo_metrics['f1']:.4f} mAP50={yolo_metrics['mAP50']:.4f}]")
        print(f"  @ T={threshold:.4f}:          P={fr_p:.4f} R={fr_r:.4f} F1={fr_f1:.4f}  (TP={fr_tp} FP={fr_fp} FN={fr_fn})")
        print(f"  size_breakdown.json     ✓  [T={threshold:.4f}]")
        print(f"  confusion_matrix.json   ✓")

    print(f"\nDone. All artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
