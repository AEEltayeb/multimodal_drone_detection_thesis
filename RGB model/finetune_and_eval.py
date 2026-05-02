"""
finetune_and_eval.py — Build dataset → Train → Evaluate automatically.

End-to-end pipeline:
  Phase 1: Build dataset (with --no-svanstrom support)
  Phase 2: Fine-tune YOLO
  Phase 3: Evaluate on original test + Svanström + confuser sets
           (side-by-side with 'old' baseline)

Usage:
    python "RGB model/finetune_and_eval.py" --no-svanstrom --name Yolo26n_hardneg_v3
    python "RGB model/finetune_and_eval.py" --skip-build --skip-train --name Yolo26n_hardneg_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

ROOT       = Path(__file__).resolve().parents[1]
BUILDER    = ROOT / "RGB model" / "dataset preparation" / "build_finetune_dataset_v2.py"
DATA_YAML  = Path(r"G:/drone/finetune_dataset_v2/data.yaml")
BASE_MODEL = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"
OLD_MODEL  = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt"

# ── Test datasets ────────────────────────────────────────────────
DATASETS = {
    "dataset_rgb": {
        "images": Path(r"G:/drone/dataset/dataset/images/test"),
        "labels": Path(r"G:/drone/dataset/dataset/labels/test"),
        "has_drones": True,
        "stride": 5,  # ~27K → ~5.4K
    },
    "svanstrom": {
        "images": Path(r"G:/drone/svanstrom_paired/RGB/images"),
        "labels": Path(r"G:/drone/svanstrom_paired/RGB/labels"),
        "has_drones": True,
        "stride": 1,
        "categories": ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER"),
    },
    "airplane": {
        "images": Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb/test/images"),
        "labels": None,
        "has_drones": False,
        "stride": 1,
    },
    "helicopter": {
        "images": Path(r"G:/drone/finetune_dataset/images/test/helicopter"),
        "labels": None,
        "has_drones": False,
        "stride": 1,
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CONF = 0.25
IOP_THRESH = 0.5


# ── HELPERS ──────────────────────────────────────────────────────

def run_cmd(cmd, **kw):
    import subprocess
    print(">>", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        print(f"[fatal] command exited with {p.returncode}")
        sys.exit(p.returncode)


def img_iter(d, stride=1):
    if not d.exists():
        return []
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if stride > 1:
        imgs = imgs[::stride]
    return imgs


def load_gt(lbl_path):
    """Load GT boxes as [(x1,y1,x2,y2)] from YOLO label (normalised)."""
    if lbl_path is None or not lbl_path.exists():
        return []
    boxes = []
    for ln in lbl_path.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if cls != 0:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return boxes


def iop(pred, gt):
    """Intersection over Prediction area."""
    ix1 = max(pred[0], gt[0]); iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2]); iy2 = min(pred[3], gt[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    pred_area = max(1e-9, (pred[2]-pred[0]) * (pred[3]-pred[1]))
    return inter / pred_area


def eval_model(model_path, model_name, datasets, out_dir):
    """Run YOLO inference on each dataset and compute metrics."""
    print(f"\n{'='*72}")
    print(f"EVALUATING: {model_name}  ({model_path})")
    print(f"{'='*72}")

    model = YOLO(str(model_path))
    results = {}

    for ds_name, ds_cfg in datasets.items():
        img_dir = ds_cfg["images"]
        lbl_dir = ds_cfg.get("labels")
        has_drones = ds_cfg["has_drones"]
        stride = ds_cfg.get("stride", 1)

        imgs = img_iter(img_dir, stride)
        if not imgs:
            print(f"  [{ds_name}] no images found at {img_dir}")
            continue

        print(f"\n  [{ds_name}] {len(imgs)} images (stride={stride})")
        tp = fp = fn = 0
        total_dets = 0
        frames_with_det = 0
        t0 = time.perf_counter()

        for i, img_path in enumerate(imgs):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]

            # YOLO inference
            r = model.predict(frame, conf=CONF, iou=0.45, imgsz=640,
                             verbose=False, device=0)[0]
            preds = []
            if r.boxes is not None:
                for j in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                    preds.append((x1/w, y1/h, x2/w, y2/h, float(r.boxes.conf[j])))

            total_dets += len(preds)
            if preds:
                frames_with_det += 1

            if has_drones and lbl_dir:
                # Match preds to GT via IoP
                gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt"))
                matched_gt = set()
                for px1, py1, px2, py2, pc in preds:
                    best_iop = 0
                    best_j = -1
                    for j, gb in enumerate(gt_boxes):
                        s = iop((px1,py1,px2,py2), gb)
                        if s > best_iop:
                            best_iop = s
                            best_j = j
                    if best_iop >= IOP_THRESH and best_j not in matched_gt:
                        tp += 1
                        matched_gt.add(best_j)
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(matched_gt)
            else:
                # Confuser-only: any detection = FP
                fp += len(preds)

            if (i+1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    {i+1}/{len(imgs)} ({elapsed:.0f}s)")

        elapsed = time.perf_counter() - t0
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        any_det_rate = frames_with_det / max(len(imgs), 1)

        r_dict = {
            "n_images": len(imgs), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "total_dets": total_dets,
            "frames_with_det": frames_with_det,
            "any_det_rate": round(any_det_rate, 4),
            "elapsed_s": round(elapsed, 1),
        }
        results[ds_name] = r_dict

        print(f"    P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
              f"TPs={tp}  FPs={fp}  FNs={fn}  any_det={any_det_rate:.2%}  "
              f"({elapsed:.0f}s)")

    # Save
    out_file = out_dir / f"{model_name}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved: {out_file}")
    return results


def print_comparison(all_results, out_dir):
    """Print side-by-side comparison table."""
    models = list(all_results.keys())
    datasets = list(next(iter(all_results.values())).keys())

    print(f"\n{'='*72}")
    print("COMPARISON TABLE")
    print(f"{'='*72}")

    header = f"{'Dataset':<15s}"
    for m in models:
        header += f" | {m:>12s} F1  {m:>12s} R  {m:>6s} FP"
    # Simplified per-dataset table
    for ds in datasets:
        print(f"\n  {ds}:")
        print(f"    {'Model':<15s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TPs':>7s} {'FPs':>7s} {'FNs':>7s} {'any_det%':>9s}")
        print(f"    {'-'*70}")
        for m in models:
            r = all_results[m].get(ds, {})
            print(f"    {m:<15s} {r.get('precision',0):>7.4f} {r.get('recall',0):>7.4f} "
                  f"{r.get('f1',0):>7.4f} {r.get('tp',0):>7d} {r.get('fp',0):>7d} "
                  f"{r.get('fn',0):>7d} {r.get('any_det_rate',0):>8.2%}")

    # Save comparison
    comp_file = out_dir / "comparison.json"
    comp_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full comparison: {comp_file}")


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # Build
    ap.add_argument("--no-svanstrom", action="store_true",
                    help="exclude Svanström confusers from training negatives")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    # Train
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=10)
    ap.add_argument("--lr0", type=float, default=0.0001)
    ap.add_argument("--name", default="Yolo26n_hardneg_v3")
    ap.add_argument("--workers", type=int, default=2)
    # Eval
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--eval-datasets", nargs="+",
                    default=["dataset_rgb", "svanstrom", "airplane", "helicopter"])
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    new_model_path = ROOT / "RGB model" / args.name / "weights" / "best.pt"

    # ── Phase 1: Build ────────────────────────────────────────────
    if not args.skip_build:
        print("=" * 72)
        print("PHASE 1 — Build dataset")
        print("=" * 72)
        cmd = [sys.executable, str(BUILDER)]
        if args.rebuild:
            cmd.append("--clean")
        if args.no_svanstrom:
            cmd.append("--no-svanstrom")
        run_cmd(cmd, env=env)
    else:
        print("[skip] dataset build")

    # ── Phase 2: Train ────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 72)
        print("PHASE 2 — Fine-tune")
        print("=" * 72)
        if not DATA_YAML.exists():
            print(f"[fatal] {DATA_YAML} missing"); sys.exit(1)
        if not BASE_MODEL.exists():
            print(f"[fatal] {BASE_MODEL} missing"); sys.exit(1)

        model = YOLO(str(BASE_MODEL))
        train_kwargs = dict(
            data=str(DATA_YAML), epochs=args.epochs, patience=2,
            batch=args.batch, imgsz=640, device=0, amp=True,
            optimizer="AdamW", lr0=args.lr0, lrf=0.01,
            freeze=args.freeze, cos_lr=True, close_mosaic=2,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
            save_period=1, workers=args.workers, cache=False,
            project=str(ROOT / "RGB model"), name=args.name,
            pretrained=True, exist_ok=True, verbose=True,
        )
        print("Training config:")
        for k, v in train_kwargs.items():
            print(f"  {k} = {v}")
        model.train(**train_kwargs)
        print(f"\nBest checkpoint: {new_model_path}")
    else:
        print("[skip] training")

    # ── Phase 3: Evaluate ─────────────────────────────────────────
    if not args.skip_eval:
        print("\n" + "=" * 72)
        print("PHASE 3 — Evaluate (old vs new)")
        print("=" * 72)

        if not new_model_path.exists():
            print(f"[fatal] {new_model_path} missing"); sys.exit(1)

        out_dir = ROOT / "runs" / "rgb_finetune_eval" / args.name
        out_dir.mkdir(parents=True, exist_ok=True)

        eval_ds = {k: DATASETS[k] for k in args.eval_datasets if k in DATASETS}

        all_results = {}

        # Eval old baseline
        old_path = OLD_MODEL if OLD_MODEL.exists() else BASE_MODEL
        all_results["old"] = eval_model(old_path, "old", eval_ds, out_dir)

        # Eval new model
        all_results[args.name] = eval_model(new_model_path, args.name, eval_ds, out_dir)

        print_comparison(all_results, out_dir)
    else:
        print("[skip] evaluation")

    print("\n" + "=" * 72)
    print("ALL DONE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
