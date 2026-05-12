"""
cache_inference.py — Pre-cache YOLO detections for a dataset.

Runs both RGB and IR models on every frame and saves raw detections to JSON.
This is the slow step — run once, then use eval_pipeline.py to evaluate.

Compatible with the legacy cache format from run_inference.py.

Usage:
    python eval/cache_inference.py --dataset antiuav
    python eval/cache_inference.py --dataset svanstrom --stride 3
    python eval/cache_inference.py --dataset antiuav --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from datasets import load_config, resolve_path, img_from_label
from run_manifest import cache_identity_tag, write_manifest, weights_short_hash


def cache_paired_dataset(ds_name: str, cfg: dict, args):
    """Cache YOLO detections for a paired RGB+IR dataset."""
    from ultralytics import YOLO

    ds_cfg = cfg["datasets"][ds_name]
    root = Path(ds_cfg["root"])
    rgb_img_dir = root / ds_cfg.get("rgb_images", "RGB/images")
    rgb_lbl_dir = root / ds_cfg.get("rgb_labels", "RGB/labels")
    ir_img_dir = root / ds_cfg.get("ir_images", "IR/images")
    ir_lbl_dir = root / ds_cfg.get("ir_labels", "IR/labels")
    rgb_suffix = ds_cfg.get("rgb_stem_suffix", "")
    ir_suffix = ds_cfg.get("ir_stem_suffix", "")

    # Resolve weights up front so we can hash-tag the cache filename.
    rgb_weights = args.rgb_weights or str(resolve_path(cfg["rgb_weights"]))
    ir_weights = args.ir_weights or str(resolve_path(cfg["ir_weights"]))

    # Output path. If --tag is given, use it verbatim (back-compat). Otherwise
    # auto-tag with a deterministic identity that captures (weights, imgsz,
    # stride) so different model versions can't silently share a cache file.
    cache_dir = EVAL_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.tag:
        tag_suffix = f"_{args.tag}"
    else:
        auto = cache_identity_tag(
            rgb_weights=rgb_weights, ir_weights=ir_weights,
            imgsz=args.imgsz, stride=args.stride,
        )
        tag_suffix = f"_{auto}"
    out_path = cache_dir / f"raw_detections_{ds_name}{tag_suffix}.json"

    # Print cache identity banner
    print(f"[{ds_name}] RGB weights: {Path(rgb_weights).name}  "
          f"({weights_short_hash(rgb_weights)})")
    print(f"[{ds_name}] IR weights:  {Path(ir_weights).name}  "
          f"({weights_short_hash(ir_weights)})")
    print(f"[{ds_name}] cache file:  {out_path.name}")
    # Manifest goes next to the cache file (one per cache, named by cache stem)
    write_manifest(
        out_dir=cache_dir,
        args=args,
        cfg=cfg,
        weights_paths={"rgb_weights": rgb_weights, "ir_weights": ir_weights},
        cache_paths=[out_path],
        extra={"dataset": ds_name, "stage": "cache_inference"},
        filename=f"{out_path.stem}.manifest.json",
    )

    rgb_model = YOLO(rgb_weights)
    ir_model = YOLO(ir_weights)
    conf = cfg["defaults"].get("conf", 0.001) if not args.conf else args.conf
    imgsz = args.imgsz

    # List all label stems
    stems = sorted(p.stem for p in rgb_lbl_dir.glob("*.txt"))
    if args.stride > 1:
        stems = stems[::args.stride]
    if args.limit:
        stems = stems[:args.limit]

    # Resume
    data = {}
    if args.resume and out_path.exists():
        data = json.loads(out_path.read_text())
        print(f"[{ds_name}] Resuming: {len(data):,} already cached")

    print(f"[{ds_name}] Processing {len(stems):,} frames (conf={conf}, imgsz={imgsz})")
    t0 = time.time()
    n_new = 0

    for idx, stem in enumerate(stems):
        if stem in data:
            continue

        ir_stem = stem
        if rgb_suffix and ir_suffix:
            ir_stem = stem.replace(rgb_suffix, ir_suffix)

        rgb_lbl = rgb_lbl_dir / f"{stem}.txt"
        ir_lbl = ir_lbl_dir / f"{ir_stem}.txt"
        rgb_path = img_from_label(rgb_lbl, rgb_img_dir)
        ir_path = img_from_label(ir_lbl, ir_img_dir)
        if rgb_path is None or ir_path is None:
            continue

        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        rh, rw = rgb_img.shape[:2]
        ih, iw = ir_img.shape[:2]

        # Run YOLO
        rgb_res = rgb_model.predict(rgb_img, conf=conf, verbose=False, imgsz=imgsz)
        ir_res = ir_model.predict(ir_img, conf=conf, verbose=False, imgsz=imgsz)

        def _extract(res):
            boxes = res[0].boxes
            dets = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                c = float(boxes.conf[i])
                dets.append([round(float(xyxy[0]), 1), round(float(xyxy[1]), 1),
                             round(float(xyxy[2]), 1), round(float(xyxy[3]), 1),
                             round(c, 4)])
            return dets

        data[stem] = {
            "rgb_dets": _extract(rgb_res),
            "ir_dets": _extract(ir_res),
            "rgb_w": rw, "rgb_h": rh,
            "ir_w": iw, "ir_h": ih,
            "rgb_lbl": str(rgb_lbl),
            "ir_lbl": str(ir_lbl),
        }
        n_new += 1

        if n_new % 200 == 0:
            # Checkpoint
            out_path.write_text(json.dumps(data))
            elapsed = time.time() - t0
            fps = n_new / elapsed
            remaining = (len(stems) - idx - 1) / max(fps, 1e-6)
            print(f"  [{ds_name}] {idx + 1:>6,}/{len(stems):,}  "
                  f"{fps:.1f} fps  ETA {remaining / 60:.1f} min  "
                  f"(checkpoint: {len(data):,} total)")

    # Final save
    out_path.write_text(json.dumps(data))
    elapsed = time.time() - t0
    print(f"[{ds_name}] Done: {len(data):,} frames cached in {elapsed:.0f}s → {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Cache YOLO detections for eval")
    ap.add_argument("--dataset", required=True,
                    choices=["antiuav", "svanstrom", "both"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rgb-weights", type=str, default="")
    ap.add_argument("--ir-weights", type=str, default="")
    ap.add_argument("--conf", type=float, default=0.0,
                    help="YOLO conf threshold (0 = use config default)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", type=str, default="",
                    help="Output filename tag, e.g. 'v3more'")
    args = ap.parse_args()

    cfg = load_config()

    datasets = []
    if args.dataset in ("antiuav", "both"):
        datasets.append("antiuav")
    if args.dataset in ("svanstrom", "both"):
        datasets.append("svanstrom")

    for ds in datasets:
        cache_paired_dataset(ds, cfg, args)

    print("\n[cache_inference] All done.")


if __name__ == "__main__":
    main()
