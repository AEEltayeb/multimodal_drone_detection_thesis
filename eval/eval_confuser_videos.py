"""
eval_confuser_videos.py — Evaluate RGB models on negative-only confuser video datasets.

Runs 3 RGB models (baseline, retrained_v2, selcom_1280) on each per-video
dataset created by scripts/extract_confuser_datasets.py.

For confuser categories (airplanes, birds, helicopters): all frames are negative,
every detection is a false positive. Reports FP count, FPPI, FP-frame rate.

Usage:
    python eval/eval_confuser_videos.py
    python eval/eval_confuser_videos.py --conf 0.25 --device 0
    python eval/eval_confuser_videos.py --categories airplanes birds
    python eval/eval_confuser_videos.py --models baseline_trained retrained_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

sys.path.insert(0, str(EVAL_DIR))
from metrics import compute_prf, score_detections, compute_frame_metrics, classify_size
from datasets import ImageDataset

# ── Model registry ───────────────────────────────────────────────
MODELS = {
    "baseline_trained": {
        "weights": str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
        "imgsz": 640,
    },
    "retrained_v2": {
        "weights": str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
        "imgsz": 640,
    },
    "selcom_1280": {
        "weights": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"),
        "imgsz": 1280,
    },
}

DATASET_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"

# Only evaluate on negative-only categories (confusers)
CONFUSER_CATEGORIES = ["airplanes", "birds", "helicopters"]


def eval_model_on_dataset(model_name: str, model_info: dict,
                          ds_path: Path, conf: float,
                          device: str) -> dict:
    """Evaluate a single model on a single negative-only dataset."""
    from ultralytics import YOLO

    weights = model_info["weights"]
    imgsz = model_info["imgsz"]
    model = YOLO(weights)

    img_dir = ds_path / "images" / "test"
    lbl_dir = ds_path / "labels" / "test"

    if not img_dir.exists():
        print(f"    SKIP: {img_dir} not found")
        return {}

    ds = ImageDataset(img_dir, lbl_dir)
    images = ds.list_images()

    if not images:
        print(f"    SKIP: No images in {img_dir}")
        return {}

    total_frames = len(images)
    total_dets = 0
    fp_frames = 0
    sizes = {"small": 0, "medium": 0, "large": 0}
    max_conf_seen = 0.0
    conf_values = []

    t0 = time.time()
    for idx, img_path in enumerate(images):
        frame = ds.load_frame(img_path)
        if frame is None:
            continue
        img = frame["img"]
        w, h = frame["w"], frame["h"]

        res = model.predict(img, conf=conf, verbose=False, imgsz=imgsz,
                            device=device)
        boxes = res[0].boxes
        n_dets = len(boxes)
        total_dets += n_dets

        if n_dets > 0:
            fp_frames += 1
            for i in range(n_dets):
                c = float(boxes.conf[i])
                conf_values.append(c)
                max_conf_seen = max(max_conf_seen, c)
                xyxy = boxes.xyxy[i].cpu().numpy()
                box = (float(xyxy[0]), float(xyxy[1]),
                       float(xyxy[2]), float(xyxy[3]))
                sz = classify_size(box, w, h)
                sizes[sz] += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"      {idx+1:>5d}/{total_frames}  {fps:.1f} fps")

    elapsed = time.time() - t0

    fppi = total_dets / max(total_frames, 1)
    fp_rate = fp_frames / max(total_frames, 1)

    return {
        "model": model_name,
        "total_frames": total_frames,
        "total_fp_dets": total_dets,
        "fp_frames": fp_frames,
        "tn_frames": total_frames - fp_frames,
        "fppi": round(fppi, 4),
        "fp_frame_rate": round(fp_rate, 4),
        "max_conf": round(max_conf_seen, 4),
        "mean_conf": round(np.mean(conf_values), 4) if conf_values else 0.0,
        "sizes": sizes,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate RGB models on confuser videos")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (default: 0.25)")
    ap.add_argument("--device", type=str, default="0",
                    help="Device (default: 0)")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Specific categories to evaluate (default: airplanes,birds,helicopters)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Specific video dataset names to evaluate (default: all)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Specific model names to evaluate (default: all)")
    args = ap.parse_args()

    if not DATASET_ROOT.exists():
        print(f"ERROR: {DATASET_ROOT} not found. Run extract_confuser_datasets.py first.")
        return

    # Load manifest for metadata
    manifest_path = DATASET_ROOT / "extraction_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for entry in json.load(f):
                manifest[entry["dataset_name"]] = entry

    categories = args.categories or CONFUSER_CATEGORIES
    models_to_run = MODELS
    if args.models:
        models_to_run = {k: v for k, v in MODELS.items() if k in args.models}

    # Discover datasets organized by category
    datasets = []
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if ds_dir.is_dir() and (ds_dir / "images" / "test").exists():
                if args.videos and ds_dir.name not in args.videos:
                    continue
                datasets.append((cat, ds_dir))

    print(f"Confuser Video Evaluation")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Models:   {len(models_to_run)}")
    print(f"  Conf:     {args.conf}")
    print(f"  Device:   {args.device}")

    out_dir = EVAL_DIR / "results" / "confuser_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for cat, ds_dir in datasets:
        ds_name = ds_dir.name
        n_frames = manifest.get(ds_name, {}).get("extracted", "?")
        video_file = manifest.get(ds_name, {}).get("video", "")

        print(f"\n{'='*70}")
        print(f"  Dataset: {cat}/{ds_name}")
        print(f"  Category: {cat}  |  Frames: {n_frames}  |  Video: {video_file}")
        print(f"{'='*70}")

        # Per-video output dir
        vid_out = out_dir / cat / ds_name
        vid_out.mkdir(parents=True, exist_ok=True)

        for model_name, model_info in models_to_run.items():
            print(f"\n    Model: {model_name} (imgsz={model_info['imgsz']})")
            result = eval_model_on_dataset(model_name, model_info,
                                           ds_dir, args.conf, args.device)
            if not result:
                continue

            result["dataset"] = ds_name
            result["category"] = cat

            print(f"    >> FP dets: {result['total_fp_dets']:>5d}  "
                  f"FPPI: {result['fppi']:.4f}  "
                  f"FP frames: {result['fp_frames']}/{result['total_frames']} "
                  f"({result['fp_frame_rate']:.1%})  "
                  f"max_conf: {result['max_conf']:.3f}")

            # Save per-video per-model JSON
            json_path = vid_out / f"{model_name}.json"
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2)

            all_rows.append(result)

    # ── Aggregate Summary ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'='*70}")

    hdr = (f"  {'Dataset':<40s} {'Model':<18s} {'Frames':>7s} "
           f"{'FP_det':>7s} {'FPPI':>7s} {'FP_fr':>6s} {'FPR':>7s} {'MaxC':>6s}")
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for row in all_rows:
        print(f"  {row['dataset']:<40s} {row['model']:<18s} "
              f"{row['total_frames']:>7d} "
              f"{row['total_fp_dets']:>7d} {row['fppi']:>7.4f} "
              f"{row['fp_frames']:>6d} {row['fp_frame_rate']:>7.4f} "
              f"{row['max_conf']:>6.3f}")

    # Per-model aggregate
    print(f"\n  PER-MODEL AGGREGATE (across all videos):")
    print(f"  {'Model':<18s} {'Frames':>7s} {'FP_det':>7s} {'FPPI':>7s} "
          f"{'FP_fr':>7s} {'FPR':>7s}")
    print(f"  {'-'*55}")

    for model_name in models_to_run:
        model_rows = [r for r in all_rows if r["model"] == model_name]
        if not model_rows:
            continue
        tot_frames = sum(r["total_frames"] for r in model_rows)
        tot_fp = sum(r["total_fp_dets"] for r in model_rows)
        tot_fp_fr = sum(r["fp_frames"] for r in model_rows)
        agg_fppi = tot_fp / max(tot_frames, 1)
        agg_fpr = tot_fp_fr / max(tot_frames, 1)
        print(f"  {model_name:<18s} {tot_frames:>7d} {tot_fp:>7d} "
              f"{agg_fppi:>7.4f} {tot_fp_fr:>7d} {agg_fpr:>7.4f}")

    # Per-category aggregate
    cats = sorted(set(r["category"] for r in all_rows))
    for cat in cats:
        print(f"\n  CATEGORY: {cat.upper()}")
        print(f"  {'Model':<18s} {'Frames':>7s} {'FP_det':>7s} {'FPPI':>7s} "
              f"{'FP_fr':>7s} {'FPR':>7s}")
        print(f"  {'-'*55}")
        for model_name in models_to_run:
            cat_rows = [r for r in all_rows
                        if r["model"] == model_name and r["category"] == cat]
            if not cat_rows:
                continue
            tot_frames = sum(r["total_frames"] for r in cat_rows)
            tot_fp = sum(r["total_fp_dets"] for r in cat_rows)
            tot_fp_fr = sum(r["fp_frames"] for r in cat_rows)
            agg_fppi = tot_fp / max(tot_frames, 1)
            agg_fpr = tot_fp_fr / max(tot_frames, 1)
            print(f"  {model_name:<18s} {tot_frames:>7d} {tot_fp:>7d} "
                  f"{agg_fppi:>7.4f} {tot_fp_fr:>7d} {agg_fpr:>7.4f}")

    # Save aggregate CSV
    csv_path = out_dir / "confuser_comparison.csv"
    if all_rows:
        fields = ["dataset", "category", "model", "total_frames",
                   "total_fp_dets", "fppi", "fp_frames", "fp_frame_rate",
                   "max_conf", "mean_conf", "elapsed_s"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Saved: {csv_path}")

    # Save full JSON
    json_path = out_dir / "confuser_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n[eval_confuser_videos] Done.")


if __name__ == "__main__":
    main()
