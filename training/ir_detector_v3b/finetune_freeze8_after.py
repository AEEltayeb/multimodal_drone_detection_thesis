"""
finetune_freeze8_after.py — Wait for the currently-running unfrozen training
to finish, then launch a freeze=8 (partial-backbone) finetune.

Polls every 30s for any python process whose command line contains
'finetune_unfrozen_run' or 'ultralytics' on this machine. Once none remain,
kicks off the new training.

Approach:
  - Start from Yolo26n_hardneg/best.pt (the frozen-head winner) — NOT from
    the just-finished unfrozen run, which may be a regressed checkpoint.
  - freeze=8 (only top 2 backbone layers + neck + head trainable).
  - lr0=5e-5 (between 1e-4 frozen and 2e-5 fully unfrozen).
  - Same dataset that the unfrozen run rebuilt: 12k drones + mixed val.
  - imgsz=512 batch=2 workers=0 — proven survivable on the 1050 Ti.
  - Inline Anti-UAV regression check after every checkpoint.

Output:
    models/rgb/Yolo26n_freeze8/weights/{best,epoch0..N,last}.pt
    models/rgb/Yolo26n_freeze8/inline_antiuav.csv

Usage:
    python "training/finetune_freeze8_after.py"
    python "training/finetune_freeze8_after.py" --no-wait    # start now
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
DATA_YAML  = Path(r"G:/drone/finetune_dataset/data.yaml")
HARDNEG_BEST = ROOT / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt"

ANTIUAV_RGB_IMG = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_RGB_LBL = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels")
ANTIUAV_INLINE_N = 200
REGRESSION_FLOOR = 0.97


# ─── waiter ───────────────────────────────────────────────────────

def find_running_training_pids() -> list[int]:
    """Return PIDs of python processes whose command line shows we're
    currently training (unfrozen run still going, or any ultralytics worker)."""
    cmd = (
        'powershell -Command "Get-Process python -ErrorAction SilentlyContinue | '
        'ForEach-Object { '
        '  $cmd = (Get-CimInstance Win32_Process -Filter \\"ProcessId=$($_.Id)\\").CommandLine; '
        '  if ($cmd -and ($cmd -like \'*finetune_unfrozen_run*\' -or '
        '                  $cmd -like \'*finetune_run.py*\' -or '
        '                  $cmd -like \'*ultralytics*\' -or '
        '                  $cmd -like \'*finetune_freeze8*\')) { '
        '    Write-Output $_.Id '
        '  } '
        '}"'
    )
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=15)
        return [int(x.strip()) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def wait_for_training_to_finish(poll_seconds: int = 30):
    print("Waiting for current training to finish...")
    while True:
        pids = find_running_training_pids()
        # Filter out our own PID
        pids = [p for p in pids if p != os.getpid()]
        if not pids:
            print("  No active training processes detected. Proceeding.")
            return
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] still running: PIDs {pids} — sleep {poll_seconds}s")
        time.sleep(poll_seconds)


# ─── inline Anti-UAV eval (copied from finetune_unfrozen_run.py) ──

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


# ─── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="don't wait — start training immediately")
    ap.add_argument("--poll", type=int, default=30,
                    help="poll interval seconds while waiting")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--lr0", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--name", default="Yolo26n_freeze8")
    args = ap.parse_args()

    if not args.no_wait:
        wait_for_training_to_finish(poll_seconds=args.poll)

    if not HARDNEG_BEST.exists():
        print(f"[fatal] {HARDNEG_BEST} missing")
        sys.exit(1)
    if not DATA_YAML.exists():
        print(f"[fatal] {DATA_YAML} missing — unfrozen run was supposed to "
              "rebuild it; rebuild it manually if needed")
        sys.exit(1)

    # ── Train ────────────────────────────────────────────────────
    print("=" * 72)
    print(f"PHASE 1 - freeze=8 partial-backbone fine-tune (from "
          f"{HARDNEG_BEST.name})")
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
        freeze=8,                 # top 2 backbone layers + neck + head trainable
        cos_lr=True,
        close_mosaic=2,
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

    # ── Per-epoch Anti-UAV inline regression check ───────────────
    print("=" * 72)
    print("PHASE 2 - Anti-UAV inline regression check (per epoch)")
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
        print(f"    {ckpt.name}: P={m['P']:.4f} R={m['R']:.4f} "
              f"F1={m['F1']:.4f}  [{verdict}]")
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
    print("Compare side-by-side using classifier/eval_rgb_finetune.py with")
    print("each candidate (frozen / freeze=8 / unfrozen) plugged in as MODEL_NEW.")
    print("=" * 72)


if __name__ == "__main__":
    main()
