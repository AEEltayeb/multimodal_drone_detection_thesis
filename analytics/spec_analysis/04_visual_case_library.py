"""
04_visual_case_library.py — Save annotated worst-FN, worst-FP, hardest-TP frames.

For each (model, dataset) in failures.csv, picks:
  - 5 worst FN: highest clutter or smallest sqrt_area_px among FN rows
  - 5 worst FP: highest conf among FP rows
  - 5 hardest TP: smallest sqrt_area_px among TP rows

Writes JPGs with overlays under
analytics/spec_analysis/visual_cases/<model>/<dataset>/.

CPU only.
"""

from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "analytics" / "spec_analysis" / "results"
OUT_DIR = ROOT / "analytics" / "spec_analysis" / "visual_cases"

# Dataset images dir lookup
IMG_DIRS = {
    "selcom_val":  Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val"),
    "dataset_rgb": Path(r"G:/drone/dataset/dataset/images/test"),
    "antiuav":     Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
}

N_PER_BUCKET = 5


def overlay(img, row, color):
    h, w = img.shape[:2]
    cx, cy = float(row["gt_cx"]) * w, float(row["gt_cy"]) * h
    bw, bh = float(row["gt_w_px"]), float(row["gt_h_px"])
    x1, y1 = int(cx - bw/2), int(cy - bh/2)
    x2, y2 = int(cx + bw/2), int(cy + bh/2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    lbl = f"{row['status']} sqrt={float(row['sqrt_area_px']):.0f}px clut={float(row['clutter']):.0f} conf={float(row['conf']):.2f}"
    cv2.putText(img, lbl, (x1, max(y1-6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def pick(rows, key, n, reverse=True):
    return sorted(rows, key=lambda r: float(r[key]), reverse=reverse)[:n]


def _dist(r1, r2):
    return ((float(r1["gt_cx"]) - float(r2["gt_cx"])) ** 2
            + (float(r1["gt_cy"]) - float(r2["gt_cy"])) ** 2) ** 0.5


def classify_fps(all_rows_for_pair, radius=0.10):
    """Return (dup_fps, near_miss_fps, real_fps).

    - dup: FP within `radius` of a TP center (duplicate box on a detected drone)
    - near-miss: FP within `radius` of an FN center (model saw the drone but
      bbox was off; IoP<0.5 rejected the match → both an unmatched pred and
      an unmatched GT exist near each other)
    - real: FP far from any GT center (genuine hallucination)
    """
    by_frame = defaultdict(list)
    for r in all_rows_for_pair:
        by_frame[r["frame"]].append(r)
    dup, near_miss, real = [], [], []
    for fr, recs in by_frame.items():
        tps = [r for r in recs if r["status"] == "TP"]
        fns = [r for r in recs if r["status"] == "FN"]
        fps = [r for r in recs if r["status"] == "FP"]
        for fp in fps:
            near_tp = tps and min(_dist(fp, t) for t in tps) < radius
            near_fn = fns and min(_dist(fp, n) for n in fns) < radius
            if near_tp:
                dup.append(fp)
            elif near_fn:
                near_miss.append(fp)
            else:
                real.append(fp)
    return dup, near_miss, real


def main():
    csv_path = RESULTS / "failures.csv"
    if not csv_path.exists():
        print(f"[fatal] {csv_path} not found — run 03_per_model_failures first")
        return

    grouped = defaultdict(list)   # (model,dataset) -> [all rows]
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            grouped[(row["model"], row["dataset"])].append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for (m, d), all_rows in grouped.items():
        out = OUT_DIR / m / d
        # Wipe any stale cases from earlier runs so the duplicate-mislabeled
        # FP_high_conf jpgs do not linger alongside the corrected ones.
        if out.exists():
            for f in out.glob("*.jpg"):
                try: f.unlink()
                except Exception: pass
        out.mkdir(parents=True, exist_ok=True)
        img_dir = IMG_DIRS.get(d)
        if img_dir is None:
            print(f"[skip] no image dir for {d}")
            continue

        tps = [r for r in all_rows if r["status"] == "TP"]
        fns = [r for r in all_rows if r["status"] == "FN"]
        dup_fps, near_miss_fps, real_fps = classify_fps(all_rows)

        picks = {
            "FN_worst_clutter":  pick(fns, "clutter", N_PER_BUCKET, reverse=True),
            "FN_smallest":       pick(fns, "sqrt_area_px", N_PER_BUCKET, reverse=False),
            "FP_real":           pick(real_fps,      "conf", N_PER_BUCKET, reverse=True),
            "FP_near_miss":      pick(near_miss_fps, "conf", N_PER_BUCKET, reverse=True),
            "FP_duplicate":      pick(dup_fps,       "conf", N_PER_BUCKET, reverse=True),
            "TP_hardest_small":  pick(tps, "sqrt_area_px", N_PER_BUCKET, reverse=False),
        }
        colors = {"FN_worst_clutter":  (0, 0, 255),     # red
                  "FN_smallest":       (0, 128, 255),    # orange
                  "FP_real":           (0, 255, 255),    # yellow — genuine hallucination
                  "FP_near_miss":      (255, 128, 255),  # pink — drone hit, bbox sloppy
                  "FP_duplicate":      (128, 128, 128),  # grey — duplicate box
                  "TP_hardest_small":  (0, 255, 0)}      # green

        for tag, rows in picks.items():
            for i, row in enumerate(rows):
                src = img_dir / row["frame"]
                if not src.exists(): continue
                img = cv2.imread(str(src))
                if img is None: continue
                overlay(img, row, colors[tag])
                fname = f"{tag}_{i:02d}_{row['frame']}"
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fname += ".jpg"
                cv2.imwrite(str(out / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[done] {m} x {d}: real={len(real_fps)} near_miss={len(near_miss_fps)} "
              f"dup={len(dup_fps)}  wrote {sum(len(v) for v in picks.values())} cases to {out}")

    print(f"\nVisual case library: {OUT_DIR}")


if __name__ == "__main__":
    main()
