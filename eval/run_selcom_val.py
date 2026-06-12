"""
run_selcom_val.py — Run all 5 RGB models on the selcom held-out val (311 imgs).

Wraps eval_model.py to avoid PowerShell loop quoting issues. Each model gets
its own output dir under eval/results/selcom_val_holdout/<model>/.

Usage:
  python eval/run_selcom_val.py
  python eval/run_selcom_val.py --models baseline selcom_1280
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

# (model_name, weights_relpath, imgsz)
SPECS = [
    ("baseline",       "models/rgb/Yolo26n_trained/weights/best.pt",                  1280),
    ("hardneg_v3more", "models/rgb/Yolo26n_hardneg_v3_more/weights/best.pt",          1280),
    ("retrained_v2",   "models/rgb/Yolo26n_retrained_v2/weights/best.pt",             1280),
    ("selcom_1280",    "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt",    1280),
    ("selcom_960",     "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt",    960),
    ("selcom_640",     "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt",    640),
]

DATASET = "G:/drone/_finetune_selcom_mixed_ft2/images/val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of model names (default: all)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dataset", default=DATASET)
    args = ap.parse_args()

    out_root = REPO / "eval" / "results" / "selcom_val_holdout"
    out_root.mkdir(parents=True, exist_ok=True)

    todo = SPECS if not args.models else [s for s in SPECS if s[0] in args.models]
    print(f"Running {len(todo)} model(s) on {args.dataset}")

    for name, wrel, imgsz in todo:
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
            "--dataset", args.dataset,
            "--imgsz", str(imgsz),
            "--conf", str(args.conf),
            "--output-dir", str(out_dir),
        ]
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f"  FAILED rc={rc}")


if __name__ == "__main__":
    main()
