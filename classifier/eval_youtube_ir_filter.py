"""
eval_youtube_ir_filter.py  —  Evaluate IR filter on OOD YouTube thermal videos.

Proves that the confuser filter suppresses IR hallucinations on OOD
helicopters / airplanes / birds, while preserving recall on OOD drones.

Two configs:
  ir_only   — raw IR YOLO detections
  ir_filter — IR YOLO + patch verifier

Every frame is assumed to contain the labeled object (tracking videos).
  - Confuser videos: any detection = False Positive → filter should suppress
  - Drone videos:    any detection = True Positive  → filter should preserve

Usage:
    python classifier/eval_youtube_ir_filter.py
    python classifier/eval_youtube_ir_filter.py --stride 3
    python classifier/eval_youtube_ir_filter.py --also-copy-external
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
DEMO_OUT   = REPO / "ir_gui" / "demo_outputs"
OUT_ROOT   = SCRIPT_DIR / "runs" / "eval_youtube_ir"

sys.path.insert(0, str(REPO / "ir_gui"))

# ── Video → category mapping ────────────────────────────────────────
# Only videos with a known category are included.
# Duplicates (YTDown versions of same video ID) removed.
# yt_4JlDTYiyoIk removed — contains no drones (misleading title).
VIDEO_LABELS = {
    # ── CONFUSERS ──
    "yt_EdOX8tJZDzw.mp4": "HELICOPTER",
    "yt_gg0Da0AtWJk.mp4": "AIRPLANE",
    "yt_LflkvbKEEr8.mp4": "AIRPLANE",
    "yt_UwOMwAGVwvs.mp4": "AIRPLANE",
    "yt_oon2AjhmAE8.mp4": "AIRPLANE",
    "yt_vfLc8n8mcKo.mp4": "AIRPLANE",     # Flight-Land-Process-Tracking
    "yt_r5tBDvY7MrA.mp4": "AIRPLANE",     # 10km-Flight-Tracking-For-Airport-Safety
    "yt_5BYnJQfMvrg.mp4": "AIRPLANE",     # airplane_landing
    "yt_omoX_2UYb0s.mp4": "BIRD",         # Flock-of-birds-captured-on-thermal
    "yt_NEANQ74oTew.mp4": "BIRD",         # RGS1000-Software-Bird-Identification
    # ── DRONES ──
    "yt_zFu7hAi5mIc.mp4": "DRONE",        # Drone-Tracking-Complex-Background (CLEAN)
    "yt_oA8Bfc_bjFk.mp4": "DRONE",        # Thermal-Tracking-Identification (LABELS)
    "yt_Y0epqCI7muk.mp4": "DRONE",        # Drone-Tracking-by-Thermal-Camera (LABELS)
    "yt_nqk0NsTBlFI.mp4": "DRONE",        # Drone-Identification-By-Thermal (LABELS/UI)
}

# Quality tag for drone videos: CLEAN = pure thermal footage, LABELS = has
# big text overlays / software UI covering the drone on many frames
DRONE_QUALITY = {
    "yt_zFu7hAi5mIc.mp4": "CLEAN",
    "yt_oA8Bfc_bjFk.mp4": "LABELS",
    "yt_Y0epqCI7muk.mp4": "LABELS",
    "yt_nqk0NsTBlFI.mp4": "LABELS",
}

# External directories to scan for additional videos
EXTERNAL_DIRS = [
    Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\youtube negatives"),
    Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\video_saved"),
]

# Keywords for auto-categorization from filenames
CATEGORY_KEYWORDS = {
    "DRONE":      ["drone", "uav"],
    "BIRD":       ["bird", "flock", "hummingbird"],
    "HELICOPTER": ["helicopter", "heli"],
    "AIRPLANE":   ["airplane", "plane", "flight", "airport"],
}


def auto_categorize(filename: str) -> str | None:
    """Try to guess category from filename keywords."""
    lower = filename.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return cat
    return None


def copy_external_videos(target_dir: Path):
    """Copy classifiable videos from external dirs into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for ext_dir in EXTERNAL_DIRS:
        if not ext_dir.exists():
            print(f"  [skip] {ext_dir} not found")
            continue
        for f in ext_dir.iterdir():
            if not f.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
                continue
            cat = auto_categorize(f.name)
            if cat is None:
                continue
            dest = target_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                copied += 1
                print(f"  [copy] {f.name} → {cat}")
                # Add to VIDEO_LABELS
                VIDEO_LABELS[f.name] = cat
    print(f"  Copied {copied} new videos from external directories")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (1 = all frames)")
    parser.add_argument("--ir-conf", type=float, default=0.40,
                        help="IR YOLO confidence threshold")
    parser.add_argument("--patch-thr", type=float, default=0.70,
                        help="Patch verifier threshold (below = reject)")
    parser.add_argument("--also-copy-external", action="store_true",
                        help="Copy classifiable videos from external dirs first")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per video (0 = all)")
    parser.add_argument("--video-dir", type=str, default=str(DEMO_OUT),
                        help="Directory containing yt_*.mp4 videos")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)

    if args.also_copy_external:
        print("Copying classifiable videos from external directories...")
        copy_external_videos(video_dir)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Discover videos ──────────────────────────────────────────────
    available = []
    for fname, cat in VIDEO_LABELS.items():
        fpath = video_dir / fname
        if fpath.exists():
            available.append((fname, cat, fpath))

    if not available:
        print("No labeled videos found! Check VIDEO_LABELS and video_dir.")
        return

    # Sort by category then name
    available.sort(key=lambda x: (x[1], x[0]))

    print(f"\nFound {len(available)} labeled videos:")
    by_cat = defaultdict(list)
    for fname, cat, _ in available:
        by_cat[cat].append(fname)
    for cat in sorted(by_cat):
        print(f"  {cat}: {len(by_cat[cat])} videos")

    # ── Load models ──────────────────────────────────────────────────
    print("\nLoading IR model...")
    from ultralytics import YOLO
    with open(REPO / "ir_gui" / "fusion_settings.json") as f:
        settings = json.load(f)
    ir_model = YOLO(settings["ir_model"])

    print("Loading patch verifier...")
    from patch_verifier import PatchVerifier
    patch_ir_path = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"
    ir_verifier = PatchVerifier(str(patch_ir_path))

    ir_conf = args.ir_conf
    patch_thr = args.patch_thr
    stride = args.stride

    print(f"\nConfig: ir_conf={ir_conf}  patch_thr={patch_thr}  stride={stride}")

    # ── Per-video results ────────────────────────────────────────────
    results_per_video = []
    # Aggregate by category
    cat_ir_only = defaultdict(lambda: {"det_frames": 0, "total_frames": 0, "total_dets": 0})
    cat_ir_filter = defaultdict(lambda: {"det_frames": 0, "total_frames": 0, "total_dets": 0})

    t_start = time.time()

    for vid_idx, (fname, cat, fpath) in enumerate(available):
        cap = cv2.VideoCapture(str(fpath))
        if not cap.isOpened():
            print(f"  [{vid_idx+1}] SKIP {fname} — cannot open")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue

        frame_idx = 0
        processed = 0
        ir_only_dets = 0        # total detection count (ir_only)
        ir_only_frames = 0      # frames with ≥1 detection (ir_only)
        ir_filter_dets = 0      # total detection count after filter
        ir_filter_frames = 0    # frames with ≥1 detection after filter
        all_filter_labels = []  # per-detection filter verdicts

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if stride > 1 and (frame_idx % stride != 0):
                continue
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            processed += 1

            # Run IR YOLO
            res = ir_model.predict(frame, conf=ir_conf, verbose=False, imgsz=640)
            boxes = res[0].boxes
            n_raw = len(boxes)

            # Count ir_only
            ir_only_dets += n_raw
            if n_raw > 0:
                ir_only_frames += 1

            # Run patch filter on all detections at once
            # predict_boxes returns P(confuser): high = confuser, low = drone
            n_survived = 0
            frame_labels = []
            if n_raw > 0:
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                probs = ir_verifier.predict_boxes(frame, boxes_xyxy)
                labels = ir_verifier.last_labels  # e.g. "airplane:0.85" or "pass(bird:0.12)"
                # Detection SURVIVES filter when P(confuser) < threshold
                n_survived = int((probs < patch_thr).sum())
                frame_labels = list(zip(probs.tolist(), labels))
                # Track what the filter thinks confusers are
                for p, lbl in zip(probs.tolist(), labels):
                    all_filter_labels.append({"frame": frame_idx, "prob": p, "label": lbl,
                                              "rejected": p >= patch_thr})

                ir_filter_dets += n_survived
                if n_survived > 0:
                    ir_filter_frames += 1

        cap.release()

        det_rate_raw = ir_only_frames / max(processed, 1)
        det_rate_filt = ir_filter_frames / max(processed, 1)

        # Summarize filter verdicts for this video
        n_rejected = sum(1 for x in all_filter_labels if x["rejected"])
        n_passed = sum(1 for x in all_filter_labels if not x["rejected"])
        # Top labels among rejected detections
        rejected_labels = [x["label"] for x in all_filter_labels if x["rejected"]]
        from collections import Counter
        top_rejected = Counter(rejected_labels).most_common(3)
        top_rej_str = ", ".join(f"{lbl}({cnt})" for lbl, cnt in top_rejected) if top_rejected else "-"

        results_per_video.append({
            "video": fname,
            "category": cat,
            "frames": processed,
            "ir_only_det_frames": ir_only_frames,
            "ir_only_dets": ir_only_dets,
            "ir_only_det_rate": det_rate_raw,
            "ir_filter_det_frames": ir_filter_frames,
            "ir_filter_dets": ir_filter_dets,
            "ir_filter_det_rate": det_rate_filt,
            "filter_suppression": 1.0 - (ir_filter_frames / max(ir_only_frames, 1)) if ir_only_frames > 0 else 0.0,
            "filter_rejected": n_rejected,
            "filter_passed": n_passed,
            "top_reject_labels": top_rej_str,
        })

        # Aggregate
        cat_ir_only[cat]["det_frames"] += ir_only_frames
        cat_ir_only[cat]["total_frames"] += processed
        cat_ir_only[cat]["total_dets"] += ir_only_dets
        cat_ir_filter[cat]["det_frames"] += ir_filter_frames
        cat_ir_filter[cat]["total_frames"] += processed
        cat_ir_filter[cat]["total_dets"] += ir_filter_dets

        elapsed = time.time() - t_start
        print(f"  [{vid_idx+1}/{len(available)}] {cat:12s} {fname:35s}  "
              f"frames={processed:5d}  ir_only={det_rate_raw:.1%}  "
              f"ir_filter={det_rate_filt:.1%}  rejected={n_rejected} passed={n_passed}  "
              f"top_reject: {top_rej_str}")

    elapsed_total = time.time() - t_start

    # ── Print per-video detail ───────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"YOUTUBE OOD IR FILTER EVALUATION  ({elapsed_total:.0f}s)")
    print(f"{'='*100}")

    print(f"\n{'Video':<35s} {'Cat':<12s} {'Quality':<8s} {'Frames':>6s}  "
          f"{'ir_only':>8s}  {'ir_filt':>8s}  {'Suppr':>7s}")
    print("-" * 95)
    for r in results_per_video:
        quality = DRONE_QUALITY.get(r["video"], "") if r["category"] == "DRONE" else ""
        print(f"{r['video']:<35s} {r['category']:<12s} {quality:<8s} {r['frames']:>6d}  "
              f"{r['ir_only_det_rate']:>7.1%}  {r['ir_filter_det_rate']:>7.1%}  "
              f"{r['filter_suppression']:>6.1%}")

    # ── Category summary ─────────────────────────────────────────────
    print(f"\n{'─'*95}")
    print(f"\n{'Category':<14} {'Frames':>7}  {'ir_only det%':>13}  {'ir_filter det%':>15}  {'Suppression':>12}")
    print("-" * 70)
    all_cats = sorted(set(list(cat_ir_only.keys()) + list(cat_ir_filter.keys())))
    for cat in all_cats:
        raw = cat_ir_only[cat]
        filt = cat_ir_filter[cat]
        total = raw["total_frames"]
        raw_rate = raw["det_frames"] / max(total, 1)
        filt_rate = filt["det_frames"] / max(total, 1)
        suppression = 1.0 - (filt["det_frames"] / max(raw["det_frames"], 1)) if raw["det_frames"] > 0 else 0.0
        is_confuser = cat != "DRONE"
        marker = " ← CONFUSER" if is_confuser else ""
        print(f"{cat:<14} {total:>7}  {raw_rate:>12.1%}  {filt_rate:>14.1%}  {suppression:>11.1%}{marker}")

    # ── Drone quality breakdown ──────────────────────────────────────
    print(f"\n{'─'*95}")
    print("DRONE BREAKDOWN BY VIDEO QUALITY:")
    for quality_tag in ["CLEAN", "LABELS"]:
        q_frames = 0
        q_raw = 0
        q_filt = 0
        for r in results_per_video:
            if r["category"] != "DRONE":
                continue
            if DRONE_QUALITY.get(r["video"]) != quality_tag:
                continue
            q_frames += r["frames"]
            q_raw += r["ir_only_det_frames"]
            q_filt += r["ir_filter_det_frames"]
        if q_frames > 0:
            raw_rate = q_raw / q_frames
            filt_rate = q_filt / q_frames
            supp = 1.0 - (q_filt / max(q_raw, 1)) if q_raw > 0 else 0.0
            print(f"  DRONE_{quality_tag:<7s}  frames={q_frames:>5d}  "
                  f"ir_only={raw_rate:.1%}  ir_filter={filt_rate:.1%}  suppressed={supp:.1%}")

    # ── Confuser-only aggregate ──────────────────────────────────────
    print(f"\n{'─'*95}")
    confuser_frames = 0
    confuser_raw = 0
    confuser_filt = 0
    for r in results_per_video:
        if r["category"] == "DRONE":
            continue
        confuser_frames += r["frames"]
        confuser_raw += r["ir_only_det_frames"]
        confuser_filt += r["ir_filter_det_frames"]
    if confuser_frames > 0:
        raw_rate = confuser_raw / confuser_frames
        filt_rate = confuser_filt / confuser_frames
        supp = 1.0 - (confuser_filt / max(confuser_raw, 1)) if confuser_raw > 0 else 0.0
        print(f"ALL CONFUSERS   frames={confuser_frames:>5d}  "
              f"ir_only={raw_rate:.1%}  ir_filter={filt_rate:.1%}  suppressed={supp:.1%}")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = OUT_ROOT / "youtube_per_video.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(results_per_video[0].keys()) + ["quality"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results_per_video:
            row = dict(r)
            row["quality"] = DRONE_QUALITY.get(r["video"], "")
            w.writerow(row)
    print(f"\nSaved per-video results to {csv_path}")

    # Build category summary with DRONE split into CLEAN/LABELS
    cat_csv = OUT_ROOT / "category_summary.csv"
    with open(cat_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "total_frames", "ir_only_det_frames", "ir_only_det_rate",
                     "ir_filter_det_frames", "ir_filter_det_rate", "suppression"])

        def write_cat_row(label, frames, raw_det, filt_det):
            raw_rate = raw_det / max(frames, 1)
            filt_rate = filt_det / max(frames, 1)
            supp = 1.0 - (filt_det / max(raw_det, 1)) if raw_det > 0 else 0.0
            w.writerow([label, frames, raw_det, f"{raw_rate:.4f}",
                         filt_det, f"{filt_rate:.4f}", f"{supp:.4f}"])

        # ALL_CONFUSERS aggregate
        write_cat_row("ALL_CONFUSERS", confuser_frames, confuser_raw, confuser_filt)

        # Per-confuser category
        for cat in sorted(c for c in all_cats if c != "DRONE"):
            raw = cat_ir_only[cat]
            filt = cat_ir_filter[cat]
            write_cat_row(cat, raw["total_frames"], raw["det_frames"], filt["det_frames"])

        # DRONE split by quality (CLEAN / LABELS)
        for quality_tag in ["CLEAN", "LABELS"]:
            q_frames = q_raw = q_filt = 0
            for r in results_per_video:
                if r["category"] != "DRONE":
                    continue
                if DRONE_QUALITY.get(r["video"]) != quality_tag:
                    continue
                q_frames += r["frames"]
                q_raw += r["ir_only_det_frames"]
                q_filt += r["ir_filter_det_frames"]
            if q_frames > 0:
                write_cat_row(f"DRONE_{quality_tag}", q_frames, q_raw, q_filt)

    print(f"Saved category summary to {cat_csv}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()

