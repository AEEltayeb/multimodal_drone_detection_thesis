"""
finetune_unfrozen_run.py — Unfrozen-backbone fine-tune, continuing from the
frozen-head checkpoint. Pure experiment: see if backbone updates can shift
the trade-off (drone retention vs confuser rejection) further than head-only.

What this does:
  1. Snapshots Yolo26n_hardneg/best.pt -> Yolo26n_hardneg/best_pre_unfreeze.pt
  2. Rebuilds G:/drone/finetune_dataset with --drone-target 12000 --mixed-val
     (2:1 drone:confuser ratio + confuser frames in val for honest mAP).
  3. Trains with backbone UNFROZEN, low LR, augmentations on, batch=2 imgsz=512
     to fit the 1050 Ti / 4 GB ceiling.
  4. After each epoch, runs an inline 200-frame Anti-UAV stride to print a
     regression check; if F1 < 0.97 it prints "ABORT recommended" but does
     not auto-kill — caller decides.

Outputs:
    RGB model/Yolo26n_unfrozen/weights/best.pt
    RGB model/Yolo26n_unfrozen/weights/epoch{0..N}.pt
    RGB model/Yolo26n_unfrozen/inline_antiuav.csv

Usage:
    python "RGB model/finetune_unfrozen_run.py"
    python "RGB model/finetune_unfrozen_run.py" --skip-build
    python "RGB model/finetune_unfrozen_run.py" --epochs 5 --imgsz 512 --batch 2
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
BUILDER    = ROOT / "RGB model" / "dataset preparation" / "build_finetune_dataset.py"
DATA_YAML  = Path(r"G:/drone/finetune_dataset/data.yaml")

HARDNEG_BEST   = ROOT / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt"
PRE_UNFREEZE   = ROOT / "RGB model" / "Yolo26n_hardneg" / "weights" / "best_pre_unfreeze.pt"

ANTIUAV_RGB_IMG = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_RGB_LBL = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels")
ANTIUAV_INLINE_N = 200   # tiny stride sample per epoch

REGRESSION_FLOOR = 0.97   # Anti-UAV inline F1 below this → ABORT recommended


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        print(f"[fatal] command exited with {p.returncode}")
        sys.exit(p.returncode)


def stride(items, n):
    items = list(items)
    if n <= 0 or n >= len(items):
        return items
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def read_yolo_labels(path: Path, w: int, h: int):
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5 or p[0] != "0":
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def antiuav_inline_eval(weights_path: Path, n_frames=ANTIUAV_INLINE_N) -> dict:
    """Per-detection IoU @ 0.5, drones-only. Quick regression check."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    imgs = sorted(p for p in ANTIUAV_RGB_IMG.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = stride(imgs, n_frames)
    tp = fp = fn = 0
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_yolo_labels(ANTIUAV_RGB_LBL / (p.stem + ".txt"), w, h)
        res = model.predict(img, conf=0.30, iou=0.45, imgsz=640,
                            verbose=False, device=0, max_det=300)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xy = res.boxes.xyxy.cpu().numpy()
            for i in range(len(res.boxes)):
                dets.append((float(xy[i, 0]), float(xy[i, 1]),
                             float(xy[i, 2]), float(xy[i, 3])))
        used = set()
        for d in dets:
            best_i, best_iou = -1, 0.0
            for gi, g in enumerate(gts):
                v = iou(d, g)
                if v > best_iou:
                    best_iou, best_i = v, gi
            if best_iou >= 0.5 and best_i not in used:
                tp += 1
                used.add(best_i)
            else:
                fp += 1
        fn += len(gts) - len(used)
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "P": round(pr, 4), "R": round(rc, 4), "F1": round(f1, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2,
                    help="1050 Ti unfrozen ceiling; drop to 1 if OOM")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="reduced from 640 for memory; raise if you have headroom")
    ap.add_argument("--lr0", type=float, default=2e-5,
                    help="5x lower than frozen run — backbone is fragile")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 to avoid Windows page-file errors; 2 if memory allows")
    ap.add_argument("--name", default="Yolo26n_unfrozen")
    ap.add_argument("--drone-target", type=int, default=12000,
                    help="2:1 drone:confuser ratio")
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    if not HARDNEG_BEST.exists():
        print(f"[fatal] missing {HARDNEG_BEST} — run frozen finetune first")
        sys.exit(1)

    # ── Phase 0: snapshot the frozen-head checkpoint ──────────────
    if not PRE_UNFREEZE.exists():
        print(f"Snapshotting {HARDNEG_BEST.name} -> {PRE_UNFREEZE.name}")
        shutil.copy2(HARDNEG_BEST, PRE_UNFREEZE)
    else:
        print(f"[skip] {PRE_UNFREEZE.name} already exists")

    # ── Phase 1: rebuild dataset with 2:1 ratio + mixed val ───────
    if not args.skip_build:
        print("=" * 72)
        print(f"PHASE 1 - Rebuild dataset (drone={args.drone_target}, mixed val)")
        print("=" * 72)
        run([sys.executable, str(BUILDER),
             "--clean",
             f"--drone-target={args.drone_target}",
             "--mixed-val"], env=env)

    if not DATA_YAML.exists():
        print(f"[fatal] {DATA_YAML} missing")
        sys.exit(1)

    # ── Phase 2: unfrozen training ────────────────────────────────
    print("=" * 72)
    print("PHASE 2 - Unfrozen-backbone fine-tune")
    print("=" * 72)

    from ultralytics import YOLO
    model = YOLO(str(HARDNEG_BEST))
    out_proj = ROOT / "RGB model"
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        amp=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        freeze=0,                 # <<< the only structural change
        cos_lr=True,
        close_mosaic=3,
        # augmentations ON
        hsv_h=0.015, hsv_s=0.30, hsv_v=0.40,
        mosaic=0.5, scale=0.5,
        fliplr=0.5,
        mixup=0.0, copy_paste=0.0, erasing=0.0,
        save_period=1,
        workers=args.workers,
        cache=False,
        project=str(out_proj),
        name=args.name,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )
    print(">> ultralytics.YOLO.train(**) with:")
    for k, v in train_kwargs.items():
        print(f"     {k} = {v}")
    model.train(**train_kwargs)

    # ── Phase 3: per-epoch Anti-UAV inline regression check ───────
    print("=" * 72)
    print("PHASE 3 - Anti-UAV inline regression check (per epoch)")
    print("=" * 72)
    weights_dir = out_proj / args.name / "weights"
    rows = []
    for ckpt in sorted(weights_dir.glob("epoch*.pt")) + [weights_dir / "best.pt",
                                                          weights_dir / "last.pt"]:
        if not ckpt.exists():
            continue
        print(f"  scoring {ckpt.name} ...")
        m = antiuav_inline_eval(ckpt)
        verdict = "OK" if m["F1"] >= REGRESSION_FLOOR else "ABORT recommended"
        print(f"    {ckpt.name}: P={m['P']:.4f} R={m['R']:.4f} F1={m['F1']:.4f}  [{verdict}]")
        rows.append({"checkpoint": ckpt.name, **m, "verdict": verdict})

    csv_path = out_proj / args.name / "inline_antiuav.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["checkpoint", "TP", "FP", "FN",
                                            "P", "R", "F1", "verdict"])
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 72)
    print("DONE.")
    print(f"Best (per YOLO val mAP):    {weights_dir / 'best.pt'}")
    print(f"Per-epoch checkpoints:      {weights_dir}")
    print(f"Anti-UAV regression check:  {csv_path}")
    print()
    print("Use the inline_antiuav.csv to choose a checkpoint that holds")
    print("Anti-UAV F1 >= {REGRESSION_FLOOR} while improving Svanström. Then run")
    print("classifier/eval_rgb_finetune.py with that .pt to get full numbers.")
    print("=" * 72)


if __name__ == "__main__":
    main()
