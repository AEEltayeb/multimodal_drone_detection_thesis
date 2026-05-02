"""
run_inference_vtuav.py — Run both YOLO models on VTUAV paired frames and
cache detections to JSON. Same format as run_inference_svanstrom.py output.

VTUAV structure (e.g. G:/drone/test_LT_001/):
    bus_025/
        rgb/000000.jpg, 000001.jpg, ...
        ir/000000.jpg,  000001.jpg, ...
        rgb.txt, ir.txt  (we ignore — annotations track buses/cars, not drones)
    bus_032/ ...
    car_001/ ...

Every frame in VTUAV is a negative for drone detection (neither model was
trained on this data; it's aerial footage OF ground objects, not drones).
We subsample every Nth frame to keep inference time manageable.

Usage:
    python run_inference_vtuav.py
    python run_inference_vtuav.py --resume
    python run_inference_vtuav.py --subsample 20
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_vtuav_frames(dataset_root, subsample_every=10):
    """
    Discover all paired RGB+IR frames in a VTUAV test folder.

    Returns a list of dicts:
        {stem: "bus_025_000000", sequence: "bus_025",
         category: "bus", rgb_img: Path, ir_img: Path}

    Subsampling: takes every `subsample_every`-th frame per sequence
    (indexing over the sorted list of frames that exist in BOTH modalities).
    """
    root = Path(dataset_root)
    pairs = []
    for seq_dir in sorted(root.iterdir()):
        if not seq_dir.is_dir():
            continue
        rgb_dir = seq_dir / "rgb"
        ir_dir = seq_dir / "ir"
        if not (rgb_dir.exists() and ir_dir.exists()):
            continue

        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        ir_files = sorted(ir_dir.glob("*.jpg"))
        rgb_map = {f.stem: f for f in rgb_files}
        ir_map = {f.stem: f for f in ir_files}
        shared = sorted(set(rgb_map) & set(ir_map))

        # Category = everything before the trailing _NNN, e.g. "bus_025" -> "bus"
        name = seq_dir.name
        category = name.rsplit("_", 1)[0] if "_" in name else name

        n_kept = 0
        for i, stem in enumerate(shared):
            if i % subsample_every != 0:
                continue
            pairs.append({
                "stem": f"{name}_{stem}",
                "sequence": name,
                "category": category,
                "rgb_img": rgb_map[stem],
                "ir_img": ir_map[stem],
            })
            n_kept += 1

        print(f"  {name:<15s} total={len(shared):>5d}  kept={n_kept:>4d}  "
              f"(category={category})")
    return pairs


def run_inference(model, img_path, cfg, flip_polarity=False):
    """Run YOLO, return list of [x1, y1, x2, y2, conf] and image dims.

    If flip_polarity=True, inverts the image (255-img) before inference.
    This converts black-hot IR to white-hot (or vice versa).

    Returns (None, None, None) if the image is unreadable — caller should
    treat this as "model emitted no detection, frame is corrupt".
    """
    import cv2
    import numpy as np

    source = str(img_path)
    if flip_polarity:
        img = cv2.imread(source)
        if img is None:
            print(f"  [WARN] cannot read {img_path}")
            return None, None, None
        img = 255 - img  # invert polarity
        source = img  # pass array directly to YOLO

    try:
        results = model.predict(
            source=source,
            conf=cfg["conf"],
            iou=cfg["iou_nms"],
            imgsz=cfg["imgsz"],
            device=cfg["device"],
            verbose=False,
            save=False,
            max_det=cfg["max_det"],
        )
    except (ValueError, OSError, RuntimeError) as e:
        print(f"  [WARN] inference failed on {img_path}: {e}")
        return None, None, None

    if not results:
        return None, None, None
    r = results[0]
    img_h, img_w = r.orig_shape
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append([float(v) for v in xyxy[i]] + [float(confs[i])])
    return dets, img_w, img_h


def atomic_json_write(path, data):
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    try:
        os.replace(tmp_path, str(path))
    except OSError:
        if os.path.exists(str(path)):
            os.remove(str(path))
        os.rename(tmp_path, str(path))


def load_checkpoint_safe(path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARN] Corrupt checkpoint, starting fresh: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    parser.add_argument("--dataset", default="G:/drone/test_LT_001")
    parser.add_argument("--output",
                        default=str(Path(__file__).resolve().parent.parent / "runs" / "vtuav_detections.json"))
    parser.add_argument("--subsample", type=int, default=10,
                        help="Keep every Nth frame per sequence (default 10)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flip-ir", action="store_true",
                        help="Invert IR image polarity (black-hot → white-hot) before inference")
    args = parser.parse_args()

    # If flipping IR, use separate output file
    if args.flip_ir:
        out_base = Path(args.output)
        args.output = str(out_base.with_stem(out_base.stem + "_flipped"))
        print("  ** IR POLARITY FLIP ENABLED (black-hot → white-hot) **")

    cfg = load_config(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".checkpoint.json")

    print("=" * 60)
    print(f"VTUAV inference (Phase 3){' [IR FLIPPED]' if args.flip_ir else ''}")
    print("=" * 60)
    print(f"  dataset:   {args.dataset}")
    print(f"  output:    {out_path}")
    print(f"  subsample: every {args.subsample}th frame")
    print(f"  flip IR:   {args.flip_ir}")

    print("\nLoading YOLO models...")
    from ultralytics import YOLO
    # Resolve weight paths relative to classifier/ if not absolute
    classifier_dir = Path(__file__).resolve().parent.parent
    def resolve_weight(p):
        p = Path(p)
        return p if p.is_absolute() else classifier_dir / p
    rgb_model = YOLO(str(resolve_weight(cfg["rgb_weights"])))
    ir_model = YOLO(str(resolve_weight(cfg["ir_weights"])))
    print("  Models loaded.")

    print("\nDiscovering paired frames...")
    pairs = discover_vtuav_frames(args.dataset, subsample_every=args.subsample)
    print(f"  Total kept: {len(pairs)} paired frames")

    if args.resume:
        detections = load_checkpoint_safe(ckpt_path)
        done_stems = set(detections.keys())
        print(f"  Resuming: {len(done_stems)} already done")
    else:
        detections = {}
        done_stems = set()

    remaining = [p for p in pairs if p["stem"] not in done_stems]
    print(f"  Remaining: {len(remaining)}")

    if not remaining:
        print("  All frames already processed!")
        if not out_path.exists():
            atomic_json_write(out_path, detections)
        return

    t_start = time.time()
    n_skipped = 0
    for idx, pair in enumerate(remaining):
        rgb_dets, rgb_w, rgb_h = run_inference(rgb_model, pair["rgb_img"], cfg)
        ir_dets, ir_w, ir_h = run_inference(
            ir_model, pair["ir_img"], cfg, flip_polarity=args.flip_ir)

        # If either image was unreadable, skip this frame entirely —
        # we can't fairly compute paired features when one side is missing.
        if rgb_dets is None or ir_dets is None:
            n_skipped += 1
            continue

        detections[pair["stem"]] = {
            "rgb_dets": rgb_dets,
            "ir_dets": ir_dets,
            "rgb_w": rgb_w, "rgb_h": rgb_h,
            "ir_w": ir_w, "ir_h": ir_h,
            "sequence": pair["sequence"],
            "category": pair["category"],
        }

        processed = idx + 1
        if processed % 50 == 0 or processed == len(remaining):
            elapsed = time.time() - t_start
            fps = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / fps if fps > 0 else 0
            print(f"  [{processed}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta / 60:.1f}min")
            atomic_json_write(ckpt_path, detections)

    atomic_json_write(out_path, detections)
    print(f"\nDone. Saved {len(detections)} frames to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} frames due to unreadable images")

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
