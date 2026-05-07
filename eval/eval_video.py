"""
eval_video.py — Evaluate YOLO model(s) on video files.

No ground truth needed — reports detection rate, total detections,
and per-frame statistics. Supports multi-model comparison.

Usage:
    # Single model, single video
    python eval/eval_video.py --weights best.pt --video path/to/video.mp4

    # Compare two models
    python eval/eval_video.py --weights old.pt new.pt --video path/to/video.mp4

    # Multiple videos
    python eval/eval_video.py --weights best.pt --video vid1.mp4 vid2.mp4 vid3.mp4

    # All videos in a directory
    python eval/eval_video.py --weights best.pt --video-dir path/to/videos/

    # With filter (patch verifier)
    python eval/eval_video.py --weights best.pt --video vid.mp4 --filter rgb

    # Label videos (for suppression analysis)
    python eval/eval_video.py --weights best.pt --video vid.mp4 --label AIRPLANE

    # Custom settings
    python eval/eval_video.py --weights best.pt --video vid.mp4 --stride 5 --conf 0.1 --imgsz 640
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


def eval_video(model, video_path: Path, args, model_name: str = "",
               verifier=None, patch_thr: float = 0.70) -> dict:
    """Evaluate a single model on a single video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open {video_path.name}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed = 0
    det_frames = 0
    total_dets = 0
    filt_frames = 0
    filt_dets = 0
    conf_values = []
    dets_per_frame = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if args.stride > 1 and (idx % args.stride != 0):
            continue
        if args.max_frames > 0 and processed >= args.max_frames:
            break
        processed += 1

        # Optionally convert to grayscale (3-channel)
        if args.grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        res = model.predict(frame, conf=args.conf, verbose=False, imgsz=args.imgsz)
        boxes = res[0].boxes
        n_raw = len(boxes)
        total_dets += n_raw
        dets_per_frame.append(n_raw)
        if n_raw > 0:
            det_frames += 1
            for i in range(n_raw):
                conf_values.append(float(boxes.conf[i]))

        # Filter
        n_survived = n_raw
        if verifier is not None and n_raw > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            probs = verifier.predict_boxes(frame, xyxy)
            n_survived = int((probs < patch_thr).sum())
        filt_dets += n_survived
        if n_survived > 0:
            filt_frames += 1

    cap.release()

    det_rate = det_frames / max(processed, 1)
    filt_rate = filt_frames / max(processed, 1)
    suppression = 1.0 - (filt_frames / max(det_frames, 1)) if det_frames > 0 else 0.0

    result = {
        "model": model_name,
        "video": video_path.name,
        "label": args.label or "",
        "total_frames": total_frames,
        "processed": processed,
        "stride": args.stride,
        "conf": args.conf,
        "resolution": f"{w}x{h}",
        "det_frames": det_frames,
        "det_rate": round(det_rate, 4),
        "total_dets": total_dets,
        "avg_dets_per_frame": round(total_dets / max(processed, 1), 2),
        "avg_conf": round(float(np.mean(conf_values)), 4) if conf_values else 0.0,
        "max_conf": round(float(max(conf_values)), 4) if conf_values else 0.0,
        "min_conf": round(float(min(conf_values)), 4) if conf_values else 0.0,
    }

    if verifier is not None:
        result.update({
            "filt_frames": filt_frames,
            "filt_rate": round(filt_rate, 4),
            "filt_dets": filt_dets,
            "suppression": round(suppression, 4),
        })

    return result


def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO model(s) on video files")
    ap.add_argument("--weights", nargs="+", required=True,
                    help="Model weight file(s) — pass multiple for comparison")
    ap.add_argument("--video", nargs="*", default=[],
                    help="Video file(s) to evaluate")
    ap.add_argument("--video-dir", type=str, default="",
                    help="Directory of videos to evaluate (all mp4/avi/mkv)")
    ap.add_argument("--stride", type=int, default=3,
                    help="Process every Nth frame (default: 3)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Max frames per video (0=all)")
    ap.add_argument("--label", type=str, default="",
                    help="Category label for the video(s): DRONE, AIRPLANE, BIRD, HELICOPTER")
    ap.add_argument("--filter", choices=["rgb", "ir", ""],  default="",
                    help="Apply patch verifier filter")
    ap.add_argument("--patch-thr", type=float, default=0.70,
                    help="Patch verifier threshold")
    ap.add_argument("--grayscale", action="store_true",
                    help="Convert frames to grayscale before inference (for IR model)")
    ap.add_argument("--output-dir", type=str, default=str(EVAL_DIR / "results" / "video"))
    args = ap.parse_args()

    # Collect videos
    videos = [Path(v) for v in args.video]
    if args.video_dir:
        vdir = Path(args.video_dir)
        if vdir.exists():
            for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
                videos.extend(vdir.glob(ext))
    videos = sorted(set(videos))

    if not videos:
        print("No videos specified. Use --video or --video-dir.")
        return

    print(f"Videos: {len(videos)}, Models: {len(args.weights)}, "
          f"stride={args.stride}, conf={args.conf}")

    # Load verifier if needed
    verifier = None
    if args.filter:
        sys.path.insert(0, str(REPO / "classifier"))
        from patch_verifier import PatchVerifier
        from datasets import load_config, resolve_path
        cfg = load_config()
        key = "patch_rgb_weights" if args.filter == "rgb" else "patch_ir_weights"
        verifier = PatchVerifier(str(resolve_path(cfg[key])))
        print(f"  Loaded {args.filter} patch verifier (thr={args.patch_thr})")

    # Run
    all_results = []
    from ultralytics import YOLO

    for w_path in args.weights:
        model_name = Path(w_path).parent.name  # e.g. "Yolo26n_trained"
        if model_name == "weights":
            model_name = Path(w_path).parent.parent.name
        print(f"\n{'='*70}")
        print(f"Model: {model_name}  ({Path(w_path).name})")
        print(f"{'='*70}")

        model = YOLO(w_path)

        for vpath in videos:
            if not vpath.exists():
                print(f"  [SKIP] {vpath} not found")
                continue
            t0 = time.time()
            result = eval_video(model, vpath, args, model_name, verifier, args.patch_thr)
            elapsed = time.time() - t0
            if not result:
                continue
            all_results.append(result)

            # Print
            lbl = f" [{result['label']}]" if result["label"] else ""
            filt_str = ""
            if "filt_rate" in result:
                filt_str = (f"  filt={result['filt_rate']:.1%} "
                            f"supp={result['suppression']:.1%}")
            print(f"  {vpath.name}{lbl}  {result['processed']} frames  "
                  f"det={result['det_rate']:.1%} ({result['total_dets']} dets)  "
                  f"avg_conf={result['avg_conf']:.3f}{filt_str}  "
                  f"[{elapsed:.0f}s]")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*90}")
        print("COMPARISON SUMMARY")
        print(f"{'='*90}")
        hdr = f"  {'model':<25s} {'video':<30s} {'det%':>6s} {'dets':>6s} {'avg_c':>6s}"
        if verifier:
            hdr += f" {'filt%':>6s} {'supp':>6s}"
        print(hdr)
        print("  " + "-" * 88)
        for r in all_results:
            row = (f"  {r['model']:<25s} {r['video']:<30s} "
                   f"{r['det_rate']:>5.1%} {r['total_dets']:>6d} "
                   f"{r['avg_conf']:>6.3f}")
            if "filt_rate" in r:
                row += f" {r['filt_rate']:>5.1%} {r['suppression']:>5.1%}"
            print(row)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_results:
        csv_path = out_dir / "video_eval.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\n  Saved: {csv_path}")

        json_path = out_dir / "video_eval.json"
        json_path.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved: {json_path}")

    print("\n[eval_video] Done.")


if __name__ == "__main__":
    main()
