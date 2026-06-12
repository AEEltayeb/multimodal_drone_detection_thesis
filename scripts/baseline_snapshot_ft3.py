"""
baseline_snapshot_ft3.py — Record ft3_1280 metrics on all eval surfaces.

Saves a baseline_snapshot.json that becomes the regression gate reference
for subsequent confuser fine-tuning (ft4).

Surfaces:
  1. selcom_mixed_ft2_val (311 imgs, IoP@0.5)
  2. dataset_rgb_test (stride=34 → ~500 imgs, IoP@0.5)
  3. rgb_confusers_merged/test (2633 imgs, halluc rate)
  4. svanstrom DRONE (stride=9, IoP@0.5)
  5. svanstrom confusers BIRD/AIRPLANE/HELICOPTER (halluc rate)
  6. antiuav (stride=5, IoP@0.5)

Usage:
    python scripts/baseline_snapshot_ft3.py
    python scripts/baseline_snapshot_ft3.py --weights "models/rgb/Yolo26n_selcom_mixed_ft3_1280/weights/best.pt"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CONF = 0.25
IOP_THRESH = 0.5

# ── Dataset paths ────────────────────────────────────────────────
DATASETS = {
    "selcom_val": {
        "images": Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3/images/val"),
        "labels": Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3/labels/val"),
        "has_drones": True,
        "stride": 1,
        "fallback_images": Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val"),
        "fallback_labels": Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
    },
    "dataset_rgb_test": {
        "images": Path(r"G:/drone/dataset/dataset/images/test"),
        "labels": Path(r"G:/drone/dataset/dataset/labels/test"),
        "has_drones": True,
        "stride": 34,  # ~500 images from ~17K
    },
    "confuser_test": {
        "images": Path(r"G:/drone/rgb_confusers_merged/images/test"),
        "labels": None,
        "has_drones": False,
        "stride": 1,
    },
    "svanstrom": {
        "images": Path(r"G:/drone/svanstrom_paired/RGB/images"),
        "labels": Path(r"G:/drone/svanstrom_paired/RGB/labels"),
        "has_drones": True,  # mixed — has DRONE + confuser categories
        "stride": 9,
        "categorized": True,  # use filename-based category detection
    },
    "antiuav": {
        "images": Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "labels": Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "has_drones": True,
        "stride": 5,
    },
}


def detect_category(stem: str) -> str:
    """Svanström-style category detection from filename."""
    for cat in ("AIRPLANE", "BIRD", "HELICOPTER", "DRONE"):
        if f"_{cat}_" in stem:
            return cat
    return "OTHER"


def classify_confuser_source(stem: str) -> str:
    """Map confuser-zoo filename to a category bucket."""
    if stem.startswith("airplane_") or "_AIRPLANE_" in stem:
        return "AIRPLANE"
    if stem.startswith("helicopter_") or "_HELICOPTER_" in stem:
        return "HELICOPTER"
    if stem.startswith("bird_") or stem.startswith("raihanrsd_") or "_BIRD_" in stem:
        return "BIRD"
    return "OTHER"


def img_iter(d, stride=1):
    if not d.exists():
        return []
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return imgs[::stride] if stride > 1 else imgs


def load_gt(lbl_path):
    if lbl_path is None or not lbl_path.exists():
        return []
    boxes = []
    for ln in lbl_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        if int(parts[0]) != 0:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iop(pred, gt):
    ix1 = max(pred[0], gt[0]); iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2]); iy2 = min(pred[3], gt[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area = max(1e-9, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    return inter / pred_area


def eval_surface(model, ds_name, ds_cfg, imgsz):
    """Evaluate model on a single surface. Returns metrics dict."""
    img_dir = ds_cfg["images"]
    lbl_dir = ds_cfg.get("labels")
    has_drones = ds_cfg["has_drones"]
    stride = ds_cfg.get("stride", 1)
    categorized = ds_cfg.get("categorized", False)

    # Try fallback paths
    if not img_dir.exists() and "fallback_images" in ds_cfg:
        img_dir = ds_cfg["fallback_images"]
        lbl_dir = ds_cfg.get("fallback_labels", lbl_dir)

    imgs = img_iter(img_dir, stride)
    if not imgs:
        print(f"  [{ds_name}] SKIP — no images at {img_dir}")
        return {}

    print(f"  [{ds_name}] {len(imgs)} images  stride={stride}  imgsz={imgsz}")

    # Accumulators
    tp = fp = fn = 0
    total_dets = frames_with_det = 0
    cat_stats = defaultdict(lambda: {"n": 0, "fp_frames": 0, "tp": 0, "fp": 0, "fn": 0,
                                      "total_dets": 0})
    confuser_stats = defaultdict(lambda: {"n": 0, "fp_frames": 0, "total_dets": 0})
    t0 = time.perf_counter()

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        r = model.predict(frame, conf=CONF, iou=0.45, imgsz=imgsz,
                          verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds.append((x1 / w, y1 / h, x2 / w, y2 / h, float(r.boxes.conf[j])))

        total_dets += len(preds)
        if preds:
            frames_with_det += 1

        if categorized:
            cat = detect_category(img_path.stem)
            cs = cat_stats[cat]
            cs["n"] += 1

            if cat == "DRONE":
                gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt")) if lbl_dir else []
                matched_gt = set()
                for px1, py1, px2, py2, pc in preds:
                    best_iop, best_j = 0, -1
                    for j, gb in enumerate(gt_boxes):
                        s = iop((px1, py1, px2, py2), gb)
                        if s > best_iop:
                            best_iop, best_j = s, j
                    if best_iop >= IOP_THRESH and best_j not in matched_gt:
                        tp += 1; cs["tp"] += 1
                        matched_gt.add(best_j)
                    else:
                        fp += 1; cs["fp"] += 1
                fn_frame = len(gt_boxes) - len(matched_gt)
                fn += fn_frame; cs["fn"] += fn_frame
            else:
                # Confuser frame — any detection = FP
                cs["total_dets"] += len(preds)
                if preds:
                    cs["fp_frames"] += 1
                    fp += len(preds)

        elif not has_drones:
            # Pure confuser dataset
            cat = classify_confuser_source(img_path.stem)
            cs = confuser_stats[cat]
            cs["n"] += 1
            cs["total_dets"] += len(preds)
            if preds:
                cs["fp_frames"] += 1
            fp += len(preds)

        else:
            # Standard drone dataset
            gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt")) if lbl_dir else []
            matched_gt = set()
            for px1, py1, px2, py2, pc in preds:
                best_iop, best_j = 0, -1
                for j, gb in enumerate(gt_boxes):
                    s = iop((px1, py1, px2, py2), gb)
                    if s > best_iop:
                        best_iop, best_j = s, j
                if best_iop >= IOP_THRESH and best_j not in matched_gt:
                    tp += 1
                    matched_gt.add(best_j)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched_gt)

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {i + 1}/{len(imgs)} ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t0
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    result = {
        "n_images": len(imgs),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "total_dets": total_dets,
        "frames_with_det": frames_with_det,
        "halluc_rate": round(frames_with_det / max(len(imgs), 1), 4),
        "elapsed_s": round(elapsed, 1),
    }

    if categorized:
        result["by_category"] = {}
        for cat, cs in sorted(cat_stats.items()):
            n = max(cs["n"], 1)
            if cat == "DRONE":
                p = cs["tp"] / max(cs["tp"] + cs["fp"], 1)
                r = cs["tp"] / max(cs["tp"] + cs["fn"], 1)
                f = 2 * p * r / max(p + r, 1e-9)
                result["by_category"][cat] = {
                    "n": cs["n"], "tp": cs["tp"], "fp": cs["fp"], "fn": cs["fn"],
                    "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                }
            else:
                result["by_category"][cat] = {
                    "n": cs["n"],
                    "halluc_rate": round(cs["fp_frames"] / n, 4),
                    "total_dets": cs["total_dets"],
                    "fp_frames": cs["fp_frames"],
                }

    if confuser_stats:
        result["by_category"] = {}
        for cat, cs in sorted(confuser_stats.items()):
            n = max(cs["n"], 1)
            result["by_category"][cat] = {
                "n": cs["n"],
                "halluc_rate": round(cs["fp_frames"] / n, 4),
                "total_dets": cs["total_dets"],
                "fp_frames": cs["fp_frames"],
            }

    print(f"    P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
          f"TP={tp}  FP={fp}  FN={fn}  halluc={result['halluc_rate']:.2%}  ({elapsed:.0f}s)")

    return result


def main():
    ap = argparse.ArgumentParser(description="Baseline snapshot for ft3_1280 regression gating")
    ap.add_argument("--weights", default=str(ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--output", default=str(ROOT / "scripts" / "baseline_snapshot.json"))
    ap.add_argument("--surfaces", nargs="*", default=None,
                    help="Subset of surfaces to evaluate (default: all)")
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    surfaces = args.surfaces or list(DATASETS.keys())
    snapshot = {
        "model": args.weights,
        "imgsz": args.imgsz,
        "conf": CONF,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "surfaces": {},
    }

    for ds_name in surfaces:
        if ds_name not in DATASETS:
            print(f"  [{ds_name}] UNKNOWN — skipping")
            continue
        result = eval_surface(model, ds_name, DATASETS[ds_name], args.imgsz)
        snapshot["surfaces"][ds_name] = result

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2))
    print(f"\nBaseline snapshot saved: {out_path}")

    # Print summary table
    print(f"\n{'='*72}")
    print(f"BASELINE SNAPSHOT SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Surface':<25s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Halluc':>8s} {'N':>6s}")
    print(f"  {'-'*60}")
    for ds_name, r in snapshot["surfaces"].items():
        if not r:
            continue
        print(f"  {ds_name:<25s} {r.get('precision', 0):>7.4f} {r.get('recall', 0):>7.4f} "
              f"{r.get('f1', 0):>7.4f} {r.get('halluc_rate', 0):>7.2%} {r.get('n_images', 0):>6d}")
        if "by_category" in r:
            for cat, cs in r["by_category"].items():
                if "f1" in cs:
                    print(f"    {cat:<23s} {cs.get('precision', 0):>7.4f} {cs.get('recall', 0):>7.4f} "
                          f"{cs.get('f1', 0):>7.4f} {'---':>8s} {cs.get('n', 0):>6d}")
                else:
                    print(f"    {cat:<23s} {'---':>7s} {'---':>7s} {'---':>7s} "
                          f"{cs.get('halluc_rate', 0):>7.2%} {cs.get('n', 0):>6d}")


if __name__ == "__main__":
    main()
