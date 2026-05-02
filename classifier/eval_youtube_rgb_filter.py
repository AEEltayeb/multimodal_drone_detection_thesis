"""
eval_youtube_rgb_filter.py — Evaluate the RGB pipeline on OOD YouTube
confuser videos. Mirrors `eval_youtube_ir_filter.py` but for RGB.

Tests whether the RGB hard-negative fine-tune (Yolo26n_hardneg) generalises
its confuser rejection to out-of-distribution YouTube footage that the
training pipeline never saw.

Configs (per video, per frame):
  rgb_only_old     — best_pre_finetune.pt at conf 0.30
  rgb_only_new     — Yolo26n_hardneg/best.pt at conf 0.30
  rgb_filter_new   — Yolo26n_hardneg + RGB patch verifier veto

Each frame is assumed to contain the labeled object (these are tracking-
style confuser videos). Any det = FP. Lower any-det-rate = better.

The video corpus is confuser-only — see VIDEO_LABELS below.

Usage:
  python classifier/eval_youtube_rgb_filter.py
  python classifier/eval_youtube_rgb_filter.py --stride 1     # every frame
  python classifier/eval_youtube_rgb_filter.py --models new   # skip old/filter
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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "ir_gui"))

OUT_ROOT = SCRIPT_DIR / "runs" / "eval_youtube_rgb"

YT_RGB_DIR = Path(r"D:/Downloads/youtube_classifier_videos")

# ── Video catalogue (confuser-only, RGB) ────────────────────────────
VIDEO_LABELS = {
    # Airplanes
    "airplane_rgb.mp4":             "AIRPLANE",
    "airplane_rgb_2.mp4":           "AIRPLANE",
    "airplane_rgb_3.mp4":           "AIRPLANE",
    "airplane_rgb_compilation.mp4": "AIRPLANE",
    # Helicopters
    "heli_rgb.mp4":                 "HELICOPTER",
    "heli_rgb_2.mp4":               "HELICOPTER",
    # Birds
    "bird_rgb.mp4":                 "BIRD",
    "birds_flock_rgb.mp4":          "BIRD",
}

MODELS = {
    "old":      REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt",
    "new":      REPO / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt",
    "v3_more":  REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt",
}

PATCH_RGB = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_rgb.pt"

RGB_CONF      = 0.30   # the post-fine-tune sweet spot from ablation
PATCH_THR     = 0.70


# ── Helpers ─────────────────────────────────────────────────────────

def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))


def load_patch():
    sys.path.insert(0, str(REPO / "classifier"))
    from patch_verifier import PatchVerifier
    return PatchVerifier(str(PATCH_RGB), device="cuda:0")


def yolo_predict(model, frame, conf):
    res = model.predict(frame, conf=conf, iou=0.45, imgsz=640,
                        verbose=False, device=0, max_det=300)[0]
    out = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i in range(len(confs)):
            out.append(((float(xyxy[i, 0]), float(xyxy[i, 1]),
                         float(xyxy[i, 2]), float(xyxy[i, 3])),
                        float(confs[i])))
    return out


# ── Main eval ───────────────────────────────────────────────────────

def eval_video(video_path: Path, category: str, models: dict, verifier,
               stride: int, want_filter: bool):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [error] cannot open {video_path.name}")
        return None

    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Per-config counters
    det_frames = defaultdict(int)
    veto_label_counts = defaultdict(lambda: defaultdict(int))  # filter labels
    n_processed = 0

    t0 = time.time()
    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        n_processed += 1

        # ── rgb_only per model ──
        per_model_dets = {}
        for name, model in models.items():
            dets = yolo_predict(model, frame, RGB_CONF)
            per_model_dets[name] = dets
            if dets:
                det_frames[f"rgb_only_{name}"] += 1

        # ── rgb_filter (new model only) ──
        if want_filter and "new" in per_model_dets:
            dets = per_model_dets["new"]
            survives = True
            if dets and verifier is not None:
                boxes = [d[0] for d in dets]
                probs = verifier.predict_boxes(frame, boxes)
                # Track which labels the filter assigns (for diagnostics)
                if hasattr(verifier, "last_labels"):
                    for lab in verifier.last_labels:
                        veto_label_counts["rgb_filter_new"][lab] += 1
                # Veto if ANY box has P(confuser) >= threshold
                if probs.size > 0 and float(probs.max()) >= PATCH_THR:
                    survives = False
            else:
                survives = bool(dets)
            if survives and dets:
                det_frames["rgb_filter_new"] += 1

    cap.release()
    elapsed = time.time() - t0

    rows = {
        "video": video_path.name,
        "category": category,
        "n_frames_total": n_frames_total,
        "n_processed": n_processed,
        "fps_processed": round(n_processed / max(elapsed, 1e-6), 2),
    }
    for cfg, cnt in det_frames.items():
        rate = cnt / n_processed if n_processed > 0 else 0.0
        rows[f"{cfg}_det_count"]   = cnt
        rows[f"{cfg}_det_rate"]    = round(rate, 4)
    # Filter top labels (diagnostic)
    if "rgb_filter_new" in veto_label_counts:
        top = sorted(veto_label_counts["rgb_filter_new"].items(),
                     key=lambda x: -x[1])[:3]
        rows["rgb_filter_top_labels"] = ",".join(f"{lab}:{c}" for lab, c in top)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3,
                    help="evaluate every Nth frame (default 3 — like IR eval)")
    ap.add_argument("--models", nargs="+", default=["old", "new"],
                    choices=list(MODELS.keys()))
    ap.add_argument("--no-filter", action="store_true",
                    help="skip rgb_filter_new config")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading YOLO models + RGB patch verifier...")
    loaded = {}
    for m in args.models:
        if not MODELS[m].exists():
            print(f"[fatal] {m} weights missing at {MODELS[m]}")
            sys.exit(1)
        loaded[m] = load_yolo(MODELS[m])
        print(f"  {m}: {MODELS[m].name}")

    verifier = None
    if not args.no_filter and "new" in loaded:
        if PATCH_RGB.exists():
            verifier = load_patch()
            print(f"  filter: {PATCH_RGB.name} (thr={PATCH_THR})")
        else:
            print(f"  [warn] patch verifier missing at {PATCH_RGB} — filter disabled")

    # Run
    all_rows = []
    for video_name, category in VIDEO_LABELS.items():
        path = YT_RGB_DIR / video_name
        if not path.exists():
            print(f"[skip] {video_name} not found")
            continue
        print(f"\n[{category}] {video_name}")
        row = eval_video(path, category, loaded, verifier,
                         stride=args.stride,
                         want_filter=(verifier is not None))
        if row is None:
            continue
        all_rows.append(row)
        # quick print
        rate_old = row.get("rgb_only_old_det_rate")
        rate_new = row.get("rgb_only_new_det_rate")
        rate_flt = row.get("rgb_filter_new_det_rate")
        print(f"  frames processed: {row['n_processed']}")
        if rate_old is not None:
            print(f"  rgb_only_old:    {rate_old*100:>6.2f}%")
        if rate_new is not None:
            print(f"  rgb_only_new:    {rate_new*100:>6.2f}%"
                  + ("    [improved]" if (rate_old is not None and rate_new < rate_old)
                     else ""))
        if rate_flt is not None:
            print(f"  rgb_filter_new:  {rate_flt*100:>6.2f}%")
        if row.get("rgb_filter_top_labels"):
            print(f"  filter labels:   {row['rgb_filter_top_labels']}")

    # Save
    csv_path = OUT_ROOT / "youtube_per_video.csv"
    if all_rows:
        keys = sorted({k for r in all_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved per-video: {csv_path}")

    # Aggregate by category
    by_cat = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [det_count, n_processed]
    for r in all_rows:
        cat = r["category"]
        n = r["n_processed"]
        for cfg in ("rgb_only_old", "rgb_only_new", "rgb_filter_new"):
            cnt = r.get(f"{cfg}_det_count")
            if cnt is None:
                continue
            by_cat[cfg][cat][0] += cnt
            by_cat[cfg][cat][1] += n
            by_cat[cfg]["ALL"][0] += cnt
            by_cat[cfg]["ALL"][1] += n

    print("\n" + "=" * 72)
    print("AGGREGATE — confuser any-det rate (lower = better)")
    print("=" * 72)
    cats = ["ALL", "AIRPLANE", "HELICOPTER", "BIRD"]
    print(f"  {'config':<20s} " + "".join(f"{c:>14s}" for c in cats))
    for cfg in ("rgb_only_old", "rgb_only_new", "rgb_filter_new"):
        if cfg not in by_cat:
            continue
        row = f"  {cfg:<20s} "
        for c in cats:
            cnt, n = by_cat[cfg].get(c, [0, 0])
            row += f"{(cnt/n*100 if n>0 else 0):>13.2f}%"
        print(row)

    summary = {
        "stride": args.stride,
        "rgb_conf": RGB_CONF,
        "patch_threshold": PATCH_THR,
        "by_category": {
            cfg: {cat: {"dets": cnt, "frames": n,
                        "rate": (cnt / n if n > 0 else None)}
                  for cat, (cnt, n) in cats.items()}
            for cfg, cats in by_cat.items()
        },
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved summary:  {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
