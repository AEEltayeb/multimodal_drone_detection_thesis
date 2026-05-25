"""
compare_selcom_ft.py — Score old_baseline + ft2_1280 + ft3_1280 on the same
two splits (selcom-only val, baseline test) and print one diff table.

Uses the eval_model helper from finetune_selcom.py so scoring rules
(IoP=0.5, conf=0.25, imgsz=1280) match the original finetune-time eval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RGB model"))

from finetune_selcom import eval_model, print_comparison  # noqa: E402

IMGSZ  = 1280
STRIDE = 5  # over dataset_rgb test split; bump to 1 for full eval

MODELS = {
    "Yolo26n_trained_baseline":      ROOT / "RGB model" / "Yolo26n_trained"             / "weights" / "best.pt",
    "Yolo26n_selcom_mixed_ft3_1280": ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt",
}

# Selcom-only val (the ft2 staging) — both finetuned models will be judged here
SELCOM_VAL_ONLY = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft2")

DATASETS = {
    # Baseline regression check: 17209 images, stride=34 -> ~506 sampled.
    "dataset_rgb_test_500": dict(
        images=Path(r"G:/drone/dataset/dataset/images/test"),
        labels=Path(r"G:/drone/dataset/dataset/labels/test"),
        has_drones=True,
        stride=34,
        imgsz=IMGSZ,
    ),
}


def main():
    out_dir = ROOT / "runs" / "rgb_finetune_eval" / "compare_selcom_ft2_vs_ft3"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, path in MODELS.items():
        if not path.exists():
            print(f"[skip] missing weights for {name}: {path}")
            continue
        all_results[name] = eval_model(path, name, DATASETS, out_dir)

    print_comparison(all_results, out_dir)
    print(f"\nWrote: {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
