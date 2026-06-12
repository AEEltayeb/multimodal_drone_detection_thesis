"""
mine_confuser_hardnegs.py — Find images where ft3_1280 hallucinates on confusers.

Runs the model on G:/drone/rgb_confusers_merged/images/train/ at imgsz=1280
and saves a list of images where the model fires (n_dets > 0).

These are the hard-negative candidates for the confuser fine-tune (ft4).

Output:
  scripts/confuser_hardnegs_ft3.csv   — full results for every image
  scripts/confuser_hardnegs_ft3.txt   — just the paths of hard-neg candidates
  scripts/confuser_mining_summary.json — summary stats

Usage:
    python scripts/mine_confuser_hardnegs.py
    python scripts/mine_confuser_hardnegs.py --weights "..." --min-conf 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFUSER_TRAIN = Path(r"G:/drone/rgb_confusers_merged/images/train")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def classify_confuser_source(stem: str) -> str:
    if stem.startswith("airplane_") or "_AIRPLANE_" in stem:
        return "AIRPLANE"
    if stem.startswith("helicopter_") or "_HELICOPTER_" in stem:
        return "HELICOPTER"
    if stem.startswith("bird_") or stem.startswith("raihanrsd_") or "_BIRD_" in stem:
        return "BIRD"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser(description="Mine confuser hard-negatives for ft4")
    ap.add_argument("--weights", default=str(ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Only keep hard-negs where max det conf >= this (0=all)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Limit number of images to process (0=all)")
    ap.add_argument("--output-dir", default=str(ROOT / "scripts"))
    args = ap.parse_args()

    from ultralytics import YOLO
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    imgs = sorted(p for p in CONFUSER_TRAIN.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.max_images > 0:
        imgs = imgs[:args.max_images]
    print(f"Processing {len(imgs)} confuser train images at imgsz={args.imgsz}, conf={args.conf}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    hardneg_paths = []
    cat_stats = defaultdict(lambda: {"total": 0, "fired": 0, "total_dets": 0,
                                      "confs": []})
    t0 = time.time()

    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        cat = classify_confuser_source(img_path.stem)
        cs = cat_stats[cat]
        cs["total"] += 1

        r = model.predict(frame, conf=args.conf, iou=0.45, imgsz=args.imgsz,
                          verbose=False, device=0)[0]
        n_dets = len(r.boxes) if r.boxes is not None else 0
        confs = [float(r.boxes.conf[j]) for j in range(n_dets)] if n_dets > 0 else []
        max_conf = max(confs) if confs else 0.0
        mean_conf = float(np.mean(confs)) if confs else 0.0

        row = {
            "image_path": str(img_path),
            "category": cat,
            "n_dets": n_dets,
            "max_conf": round(max_conf, 4),
            "mean_conf": round(mean_conf, 4),
            "is_hardneg": int(n_dets > 0 and max_conf >= args.min_conf),
        }
        all_rows.append(row)

        if n_dets > 0:
            cs["fired"] += 1
            cs["total_dets"] += n_dets
            cs["confs"].extend(confs)
            if max_conf >= args.min_conf:
                hardneg_paths.append(str(img_path))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            fired = sum(cs["fired"] for cs in cat_stats.values())
            print(f"  {i+1:>6d}/{len(imgs)}  {fps:.1f} fps  "
                  f"fired={fired}/{i+1} ({fired/(i+1):.1%})")

    elapsed = time.time() - t0
    total_fired = sum(cs["fired"] for cs in cat_stats.values())

    # Save full CSV
    csv_path = out_dir / "confuser_hardnegs_ft3.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "category", "n_dets",
                                           "max_conf", "mean_conf", "is_hardneg"])
        w.writeheader()
        w.writerows(all_rows)

    # Save hard-neg paths
    txt_path = out_dir / "confuser_hardnegs_ft3.txt"
    txt_path.write_text("\n".join(hardneg_paths) + "\n")

    # Summary
    summary = {
        "model": args.weights,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "min_conf_filter": args.min_conf,
        "total_images": len(imgs),
        "total_fired": total_fired,
        "total_hardnegs": len(hardneg_paths),
        "fire_rate": round(total_fired / max(len(imgs), 1), 4),
        "elapsed_s": round(elapsed, 1),
        "by_category": {},
    }
    for cat, cs in sorted(cat_stats.items()):
        n = max(cs["total"], 1)
        summary["by_category"][cat] = {
            "total": cs["total"],
            "fired": cs["fired"],
            "fire_rate": round(cs["fired"] / n, 4),
            "total_dets": cs["total_dets"],
            "mean_conf_of_fps": round(float(np.mean(cs["confs"])), 4) if cs["confs"] else 0.0,
            "median_conf_of_fps": round(float(np.median(cs["confs"])), 4) if cs["confs"] else 0.0,
        }

    summary_path = out_dir / "confuser_mining_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print results
    print(f"\n{'='*72}")
    print(f"CONFUSER MINING RESULTS")
    print(f"{'='*72}")
    print(f"  Total images:   {len(imgs):,}")
    print(f"  Total fired:    {total_fired:,} ({total_fired/max(len(imgs),1):.1%})")
    print(f"  Hard-neg candidates (min_conf>={args.min_conf}): {len(hardneg_paths):,}")
    print(f"  Elapsed:        {elapsed:.0f}s")
    print()
    print(f"  {'Category':<15s} {'Total':>7s} {'Fired':>7s} {'Rate':>7s} {'Dets':>7s} {'MedConf':>8s}")
    print(f"  {'-'*55}")
    for cat, cs in sorted(cat_stats.items()):
        n = max(cs["total"], 1)
        med = float(np.median(cs["confs"])) if cs["confs"] else 0
        print(f"  {cat:<15s} {cs['total']:>7d} {cs['fired']:>7d} "
              f"{cs['fired']/n:>6.1%} {cs['total_dets']:>7d} {med:>8.3f}")

    print(f"\n  Saved:")
    print(f"    {csv_path}")
    print(f"    {txt_path}")
    print(f"    {summary_path}")


if __name__ == "__main__":
    main()
