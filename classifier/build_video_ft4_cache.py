"""build_video_ft4_cache.py — mine RGB detection caches for the video-test clips
with a chosen detector, in the exact format generate_lean19_data.process_video_tests
consumes:  {ctag}_{tag}.json  ->  {"__manifest__": {...}, "dets": {stem: [[x1,y1,x2,y2,conf],...]}}

Default builds ft4 @ imgsz=1280 -> video_{cat}_{clip}_ft4_sz1280.json, matching the
existing v3b IR-grayscale caches (which are reused as-is). Then run:

  py classifier/generate_lean19_data.py --rgb-weights <ft4> --ir-weights <v3b> \
     --video-rgb-cache-tag ft4_sz1280 ...

Usage:
  py classifier/build_video_ft4_cache.py
  py classifier/build_video_ft4_cache.py --overwrite        # re-mine existing
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
VIDEO_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
CACHE_DIR = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
CATS = ("drone", "birds", "airplanes", "helicopters")
DEFAULT_FT4 = REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"


def list_imgs(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []


def run_yolo(model, img, conf, imgsz):
    r = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for i in range(len(xy)):
            out.append([float(xy[i][0]), float(xy[i][1]), float(xy[i][2]),
                        float(xy[i][3]), float(cf[i])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_FT4))
    ap.add_argument("--tag", default="ft4_sz1280")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--require-ir-cache", action="store_true",
                    help="only build clips that have a matching v3b ir-grayscale cache "
                         "(the run skips clips lacking it anyway)")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"RGB weights: {args.weights}")
    print(f"tag={args.tag}  imgsz={args.imgsz}  conf={args.conf}")
    model = YOLO(args.weights)

    n_clips = n_skip = n_frames = 0
    t0 = time.time()
    for cat in CATS:
        cd = VIDEO_ROOT / cat
        if not cd.exists():
            continue
        for clip in sorted(p for p in cd.iterdir() if p.is_dir()):
            ctag = f"video_{cat}_{clip.name}"
            img_d = clip / "images" / "test"
            if not img_d.exists():
                img_d = clip / "images"
            out = CACHE_DIR / f"{ctag}_{args.tag}.json"
            ir_cache = CACHE_DIR / f"{ctag}_ir_grayscale_sz640.json"

            if args.require_ir_cache and not ir_cache.exists():
                print(f"  [skip:no-ir-cache] {ctag}")
                n_skip += 1
                continue
            if out.exists() and not args.overwrite:
                print(f"  [skip:exists] {out.name}")
                n_skip += 1
                continue
            imgs = list_imgs(img_d)
            if not imgs:
                print(f"  [skip:no-imgs] {ctag}")
                n_skip += 1
                continue

            dets = {}
            for ip in imgs:
                im = cv2.imread(str(ip))
                if im is None:
                    continue
                dets[ip.stem] = run_yolo(model, im, args.conf, args.imgsz)
            out.write_text(json.dumps({
                "__manifest__": {"weights_path": str(args.weights), "imgsz": args.imgsz,
                                 "conf": args.conf},
                "dets": dets,
            }))
            n_clips += 1
            n_frames += len(dets)
            print(f"  wrote {out.name}  ({len(dets)} frames)")

    dt = time.time() - t0
    print(f"\nDONE: {n_clips} clips ({n_frames} frames) built, {n_skip} skipped, {dt:.0f}s")
    print(f"Cache tag '{args.tag}' ready in {CACHE_DIR}")


if __name__ == "__main__":
    main()
