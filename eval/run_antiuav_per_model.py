"""
run_antiuav_per_model.py — Per-model Anti-UAV RGB eval.

Usage:
  python eval/run_antiuav_per_model.py
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

SPECS = [
    ("baseline",     "models/rgb/Yolo26n_trained/weights/best.pt"),
    ("retrained_v2", "models/rgb/Yolo26n_retrained_v2/weights/best.pt"),
    ("selcom_1280",  "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"),
    ("selcom_960",   "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"),
    ("selcom_640",   "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"),
]

DATASET = "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB"


def auto_stride(n: int, cap: int = 5000, floor: int = 2000) -> int:
    if n < floor:
        return 1
    return max(1, -(-n // cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=0,
                    help="0 = auto (cap 5k, floor 2k)")
    args = ap.parse_args()

    # Count images for auto stride
    img_dir = Path(DATASET) / "images"
    if not img_dir.exists():
        img_dir = Path(DATASET)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n_imgs = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in exts)
    stride = args.stride if args.stride > 0 else auto_stride(n_imgs)
    print(f"Anti-UAV: {n_imgs} images, stride={stride} -> ~{n_imgs // stride} sampled")

    out_root = REPO / "eval" / "results" / "antiuav_per_model"
    out_root.mkdir(parents=True, exist_ok=True)
    todo = SPECS if not args.models else [s for s in SPECS if s[0] in args.models]

    PER_MODEL_IMGSZ = {"selcom_1280": 1280, "selcom_960": 960, "selcom_640": 640}
    for name, wrel in todo:
        imgsz = PER_MODEL_IMGSZ.get(name, args.imgsz)
        wpath = REPO / wrel
        if not wpath.exists():
            print(f"  SKIP missing weights: {wpath}")
            continue
        out_dir = out_root / name
        print(f"\n=== {name}  (imgsz={imgsz}) ===")
        cmd = [
            sys.executable, str(EVAL_DIR / "eval_model.py"),
            "--weights", str(wpath),
            "--model-name", name,
            "--dataset", DATASET,
            "--imgsz", str(imgsz),
            "--conf", str(args.conf),
            "--stride", str(stride),
            "--output-dir", str(out_dir),
        ]
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f"  FAILED rc={rc}")


if __name__ == "__main__":
    main()
