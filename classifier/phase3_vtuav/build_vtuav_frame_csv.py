"""
build_vtuav_frame_csv.py — Convert cached VTUAV detections to a frame-level
CSV matching the curated Phase 1 feature schema.

Reads:
    runs/vtuav_detections.json

Writes:
    runs/vtuav_frame_dataset.csv

Every frame is a negative for drone detection (VTUAV contains no drones).
All rows get label=0. Feature set matches curate_dataset.FEATURE_COLS exactly
so the existing classifiers can score these rows without retraining.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


FEATURE_COLS = [
    "max_conf_rgb", "max_conf_ir",
    "conf_max", "conf_min", "conf_mean", "conf_delta",
    "both_detected",
    "n_dets_rgb", "n_dets_ir", "n_dets_total",
    "conf_rgb_2nd", "conf_ir_2nd",
    "rgb_area_norm", "ir_area_norm",
]

META_COLS = ["stem", "sequence", "source", "category", "lighting", "label"]


def bbox_area(b):
    x1, y1, x2, y2 = b[:4]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_features(det):
    """Compute the 14 honest features from a VTUAV detection record."""
    rgb_dets = det["rgb_dets"]
    ir_dets = det["ir_dets"]
    rgb_w, rgb_h = det["rgb_w"], det["rgb_h"]
    ir_w, ir_h = det["ir_w"], det["ir_h"]

    # Top confidences (0 if no detection)
    rgb_confs = sorted([d[4] for d in rgb_dets], reverse=True) if rgb_dets else []
    ir_confs = sorted([d[4] for d in ir_dets], reverse=True) if ir_dets else []

    max_conf_rgb = rgb_confs[0] if rgb_confs else 0.0
    max_conf_ir = ir_confs[0] if ir_confs else 0.0
    conf_rgb_2nd = rgb_confs[1] if len(rgb_confs) >= 2 else 0.0
    conf_ir_2nd = ir_confs[1] if len(ir_confs) >= 2 else 0.0

    # Derived (matches build_dataset.py)
    conf_max = max(max_conf_rgb, max_conf_ir)
    conf_min = min(max_conf_rgb, max_conf_ir)
    conf_mean = (max_conf_rgb + max_conf_ir) / 2.0
    conf_delta = abs(max_conf_rgb - max_conf_ir)
    both_detected = 1 if (max_conf_rgb > 0 and max_conf_ir > 0) else 0

    n_dets_rgb = len(rgb_dets)
    n_dets_ir = len(ir_dets)
    n_dets_total = n_dets_rgb + n_dets_ir

    # Area of the top box, normalized by image area
    rgb_img_area = max(1.0, rgb_w * rgb_h)
    ir_img_area = max(1.0, ir_w * ir_h)
    rgb_area_norm = (bbox_area(rgb_dets[0]) / rgb_img_area) if rgb_dets else 0.0
    ir_area_norm = (bbox_area(ir_dets[0]) / ir_img_area) if ir_dets else 0.0

    return {
        "max_conf_rgb": round(max_conf_rgb, 6),
        "max_conf_ir": round(max_conf_ir, 6),
        "conf_max": round(conf_max, 6),
        "conf_min": round(conf_min, 6),
        "conf_mean": round(conf_mean, 6),
        "conf_delta": round(conf_delta, 6),
        "both_detected": both_detected,
        "n_dets_rgb": n_dets_rgb,
        "n_dets_ir": n_dets_ir,
        "n_dets_total": n_dets_total,
        "conf_rgb_2nd": round(conf_rgb_2nd, 6),
        "conf_ir_2nd": round(conf_ir_2nd, 6),
        "rgb_area_norm": round(rgb_area_norm, 8),
        "ir_area_norm": round(ir_area_norm, 8),
    }


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    classifier_dir = script_dir.parent
    parser.add_argument("--input",
                        default=str(classifier_dir / "runs" / "vtuav_detections.json"))
    parser.add_argument("--output",
                        default=str(classifier_dir / "runs" / "vtuav_frame_dataset.csv"))
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    print("=" * 60)
    print("Building VTUAV frame CSV")
    print("=" * 60)
    print(f"  input:  {in_path}")
    print(f"  output: {out_path}")

    if not in_path.exists():
        print(f"  ERROR: {in_path} not found. Run run_inference_vtuav.py first.")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        detections = json.load(f)
    print(f"  Loaded {len(detections)} frames")

    rows = []
    for stem, det in detections.items():
        feats = compute_features(det)
        row = {
            "stem": stem,
            "sequence": det.get("sequence", ""),
            "source": "vtuav",
            "category": det.get("category", "unknown"),
            "lighting": "unknown",
            "label": 0,
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[META_COLS + FEATURE_COLS]  # enforce column order

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n  Saved {len(df)} rows (all label=0) to {out_path}")
    print("\n  Per category:")
    cat_counts = df.groupby("category").size().sort_values(ascending=False)
    for cat, n in cat_counts.items():
        print(f"    {cat:<12s} {n:>5d}")

    # Sanity: how often does each model fire on VTUAV?
    rgb_fire_rate = (df["max_conf_rgb"] > 0).mean()
    ir_fire_rate = (df["max_conf_ir"] > 0).mean()
    both_fire_rate = df["both_detected"].mean()
    print(f"\n  Detection rates (fraction of frames with any detection):")
    print(f"    RGB model fired:  {rgb_fire_rate:.3f}")
    print(f"    IR model fired:   {ir_fire_rate:.3f}")
    print(f"    Both fired:       {both_fire_rate:.3f}")
    print("  (These are raw YOLO detections — classifier still has to promote them.)")


if __name__ == "__main__":
    main()
