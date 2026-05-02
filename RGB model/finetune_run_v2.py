"""
finetune_run_v2.py — Fine-tune RGB YOLO on expanded hard negatives (v2).

Adds Svanström confusers (airplane/bird/helicopter) to the existing
negative pool for broader confuser coverage.

Usage:
    python "RGB model/finetune_run_v2.py"
    python "RGB model/finetune_run_v2.py" --skip-build
    python "RGB model/finetune_run_v2.py" --epochs 5 --batch 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
BUILDER    = ROOT / "RGB model" / "dataset preparation" / "build_finetune_dataset_v2.py"
DATA_YAML  = Path(r"G:/drone/finetune_dataset_v2/data.yaml")
BASE_MODEL = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"


def run(cmd, **kw):
    print(">>", " ".join(str(c) for c in cmd))
    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        print(f"[fatal] command exited with {p.returncode}")
        sys.exit(p.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe G:/drone/finetune_dataset_v2 and rebuild")
    ap.add_argument("--skip-build", action="store_true",
                    help="dataset already built, just train")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4,
                    help="GTX 1050 Ti / 4 GB ceiling is ~4 with AMP")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.0001)
    ap.add_argument("--freeze", type=int, default=10,
                    help="freeze first N layers (backbone) — 0 to unfreeze")
    ap.add_argument("--name", default="Yolo26n_hardneg_v2")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # ── Phase 1: dataset ──────────────────────────────────────────
    if not args.skip_build:
        print("=" * 72)
        print("PHASE 1 — Build expanded hard-negative dataset (v2)")
        print("=" * 72)
        cmd = [sys.executable, str(BUILDER)]
        if args.rebuild:
            cmd.append("--clean")
        run(cmd, env=env)
        if not DATA_YAML.exists():
            print(f"[fatal] expected {DATA_YAML} not produced by builder")
            sys.exit(1)
    else:
        print("[skip] dataset build (--skip-build set)")
        if not DATA_YAML.exists():
            print(f"[fatal] {DATA_YAML} missing — run without --skip-build")
            sys.exit(1)

    if not BASE_MODEL.exists():
        print(f"[fatal] base model missing: {BASE_MODEL}")
        sys.exit(1)

    # ── Phase 2: train ────────────────────────────────────────────
    print("=" * 72)
    print("PHASE 2 — Fine-tune RGB YOLO on expanded confuser hard negatives")
    print("=" * 72)

    from ultralytics import YOLO

    model = YOLO(str(BASE_MODEL))
    train_kwargs = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=2,
        batch=args.batch,
        imgsz=args.imgsz,
        device=0,
        amp=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        freeze=args.freeze,
        cos_lr=True,
        close_mosaic=2,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
        save_period=1,
        workers=args.workers,
        cache=False,
        project=str(ROOT / "RGB model"),
        name=args.name,
        pretrained=True,
        exist_ok=True,
        verbose=True,
    )
    print(">> ultralytics.YOLO.train(**) with:")
    for k, v in train_kwargs.items():
        print(f"     {k} = {v}")
    model.train(**train_kwargs)

    print()
    print("=" * 72)
    print("DONE.")
    print(f"Best checkpoint: {ROOT / 'RGB model' / args.name / 'weights' / 'best.pt'}")
    print(f"All checkpoints: {ROOT / 'RGB model' / args.name / 'weights'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
