"""
eval_video_persize.py — Per-size drone P/R/F1 on real-video test clips.

Runs each model on each clip frame-by-frame, then buckets TP/FP/FN by GT box
area. Mirrors eval_video_tests.py but emits per-size CSVs.

Output:
  eval/results/video_persize/<clip>/<model>_persize.csv
  eval/results/video_persize/summary.csv

Usage:
  python eval/eval_video_persize.py
  python eval/eval_video_persize.py --models baseline_trained selcom_1280
  python eval/eval_video_persize.py --clips drone_takeoff_short two_birds_drone
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import SIZE_BUCKETS, classify_size, score_per_size  # noqa: E402

VIDEO_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"

MODELS = {
    "baseline_trained": (REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt", 640),
    "retrained_v2":     (REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt", 640),
    "selcom_1280":      (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 1280),
    "selcom_960":       (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 960),
    "selcom_640":       (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 640),
    "ir_final_gray":    (REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt", 640),
    "ir_final_rgb":     (REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt", 640),
}

# Models that need BGR->gray->BGR conversion before inference (IR-on-grayscale path)
GRAYSCALE_MODELS = {"ir_final_gray"}

# Categories with drone GT (positive clips) vs confuser-only
CATEGORY_DIRS = {
    "drone": True,        # has GT
    "birds": False,
    "airplanes": False,
    "helicopters": False,
}


def read_labels_for_frame(label_path: Path, w: int, h: int) -> list[tuple]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append((
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ))
    return boxes


def precision(tp, fp): return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def recall(tp, fn):    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def f1(p, r):          return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def list_clips(category_root: Path) -> list[Path]:
    """A clip directory has either video files + labels/, or images/ + labels/."""
    out = []
    if not category_root.exists():
        return out
    for child in sorted(category_root.iterdir()):
        if child.is_dir():
            out.append(child)
    return out


def iter_frames(clip_dir: Path):
    """Yield (stem, image, gt_label_path) for a clip.
    Supports layouts:
      (a) clip/images/test/*.jpg + clip/labels/test/*.txt   (YOLO split)
      (b) clip/images/*.jpg     + clip/labels/*.txt          (flat split)
      (c) clip/*.jpg            + clip/*.txt                 (flat)"""
    exts = (".jpg", ".jpeg", ".png", ".bmp")

    candidates: list[tuple[Path, Path]] = []
    img_root = clip_dir / "images"
    lbl_root = clip_dir / "labels"
    if img_root.exists():
        # If images/ contains split subdirs (test/train/val), use those
        split_subs = [p for p in img_root.iterdir() if p.is_dir()]
        if split_subs:
            for sub in split_subs:
                candidates.append((sub, lbl_root / sub.name))
        else:
            candidates.append((img_root, lbl_root))
    else:
        candidates.append((clip_dir, clip_dir))

    for img_dir, lbl_dir in candidates:
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            try:
                img = cv2.imread(str(p))
            except cv2.error:
                continue
            if img is None:
                continue
            yield p.stem, img, lbl_dir / f"{p.stem}.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--clips", nargs="+", default=None,
                    help="Subset of clip names. If omitted runs all clips under all categories.")
    ap.add_argument("--categories", nargs="+", default=list(CATEGORY_DIRS.keys()))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iop-thr", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--output-dir", type=str,
                    default=str(EVAL_DIR / "results" / "video_persize"))
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Enumerate clips
    clip_specs: list[tuple[str, str, bool, Path]] = []  # (category, clip_name, has_gt, clip_dir)
    for cat in args.categories:
        cat_root = VIDEO_ROOT / cat
        for cdir in list_clips(cat_root):
            if args.clips and cdir.name not in args.clips:
                continue
            clip_specs.append((cat, cdir.name, CATEGORY_DIRS[cat], cdir))

    print(f"Clips to process: {len(clip_specs)}")

    summary_rows: list[dict] = []

    for mname in args.models:
        if mname not in MODELS:
            continue
        wpath, imgsz = MODELS[mname]
        if not wpath.exists():
            print(f"SKIP missing weights: {wpath}")
            continue
        print(f"\n=== Model: {mname}  imgsz={imgsz} ===")
        model = YOLO(str(wpath))

        is_grayscale = mname in GRAYSCALE_MODELS
        if is_grayscale:
            print(f"  (grayscale mode: BGR->gray->BGR before inference)")

        for cat, clip, has_gt, cdir in clip_specs:
            counts = {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}
            t0 = time.time()
            n_frames = 0
            for stem, img, lbl in iter_frames(cdir):
                h, w = img.shape[:2]
                gts = read_labels_for_frame(lbl, w, h) if has_gt else []
                for g in gts:
                    counts[classify_size(g, w, h)]["n_gt"] += 1
                if is_grayscale:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    inp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    inp = img
                res = model.predict(inp, imgsz=imgsz, conf=args.conf,
                                    device=args.device, verbose=False)
                r0 = res[0]
                dets = []
                if r0.boxes is not None and len(r0.boxes) > 0:
                    xyxy = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()
                    dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
                ps = score_per_size(dets, gts, w, h, iou_thr=0.5, iop_thr=args.iop_thr)
                for b in SIZE_BUCKETS:
                    counts[b]["tp"] += ps["iop"][b]["tp"]
                    counts[b]["fp"] += ps["iop"][b]["fp"]
                    counts[b]["fn"] += ps["iop"][b]["fn"]
                n_frames += 1
            dt = time.time() - t0
            print(f"  {cat}/{clip}: {n_frames} frames in {dt:.1f}s")

            rows = []
            for b in SIZE_BUCKETS:
                c = counts[b]
                p = precision(c["tp"], c["fp"])
                r = recall(c["tp"], c["fn"])
                rows.append({
                    "model": mname, "category": cat, "clip": clip,
                    "has_drone_gt": has_gt, "size_bucket": b,
                    "TP": c["tp"], "FP": c["fp"], "FN": c["fn"], "n_gt": c["n_gt"],
                    "n_frames": n_frames, "fppi_bucket": (c["fp"] / n_frames) if n_frames else 0.0,
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1(p, r), 4),
                })

            clip_out = out_root / cat / clip
            clip_out.mkdir(parents=True, exist_ok=True)
            cp = clip_out / f"{mname}_persize.csv"
            with cp.open("w", newline="") as f:
                w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w_.writeheader()
                w_.writerows(rows)
            summary_rows.extend(rows)

    sp = out_root / "summary.csv"
    if summary_rows:
        with sp.open("w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w_.writeheader()
            w_.writerows(summary_rows)
        print(f"\nWrote {sp} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
