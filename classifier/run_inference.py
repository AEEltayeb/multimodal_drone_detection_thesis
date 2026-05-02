"""
run_inference.py — Phase 1: Run both YOLO models on all paired frames
and save raw detections to disk.

This is the slow step (inference). Run once, then use build_dataset.py
(Phase 2) to align/label/extract features — that step is fast and can
be re-run with different parameters without re-doing inference.

Usage:
    python run_inference.py
    python run_inference.py --resume
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_paired_frames(dataset_root, rgb_subdir, ir_subdir,
                           rgb_suffix, ir_suffix):
    """Find all paired RGB/IR frames."""
    rgb_img_dir = Path(dataset_root) / rgb_subdir / "images"
    ir_img_dir = Path(dataset_root) / ir_subdir / "images"
    rgb_lbl_dir = Path(dataset_root) / rgb_subdir / "labels"
    ir_lbl_dir = Path(dataset_root) / ir_subdir / "labels"
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    def stem_map(img_dir, suffix):
        out = {}
        if not img_dir.exists():
            return out
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in img_exts:
                key = f.stem.replace(suffix, "") if suffix else f.stem
                out[key] = f
        return out

    rgb_map = stem_map(rgb_img_dir, rgb_suffix)
    ir_map = stem_map(ir_img_dir, ir_suffix)
    shared = sorted(set(rgb_map) & set(ir_map))

    pairs = []
    for stem in shared:
        rgb_lbl_name = rgb_map[stem].stem + ".txt"
        ir_lbl_name = ir_map[stem].stem + ".txt"
        pairs.append({
            "stem": stem,
            "rgb_img": rgb_map[stem],
            "ir_img": ir_map[stem],
            "rgb_lbl": rgb_lbl_dir / rgb_lbl_name,
            "ir_lbl": ir_lbl_dir / ir_lbl_name,
        })
    return pairs


def run_inference(model, img_path, cfg):
    """Run YOLO, return list of [x1, y1, x2, y2, conf] and image dims."""
    results = model.predict(
        source=str(img_path),
        conf=cfg["conf"],
        iou=cfg["iou_nms"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        verbose=False,
        save=False,
        max_det=cfg["max_det"],
    )
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
    """Write JSON atomically: write to .tmp then replace.
    Uses os.replace() which is atomic on Windows."""
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    # os.replace is atomic and overwrites the target on Windows
    for attempt in range(5):
        try:
            os.replace(tmp_path, str(path))
            return
        except PermissionError:
            # Another process has the file open — wait and retry
            import time as _time
            _time.sleep(1)
    # Last resort: just leave the .tmp file
    print(f"  [WARN] Could not replace {path}, saved as {tmp_path}")


def load_checkpoint_safe(path):
    """Load checkpoint, handling corruption gracefully."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARN] Corrupt checkpoint, starting fresh: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Run inference and save raw detections")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / "raw_detections.json"
    ckpt_path = out_dir / "inference_checkpoint.json"

    print("Loading models...")
    from ultralytics import YOLO
    rgb_model = YOLO(cfg["rgb_weights"])
    ir_model = YOLO(cfg["ir_weights"])
    print("  Models loaded.")

    print("Discovering paired frames...")
    pairs = discover_paired_frames(
        cfg["dataset_root"], cfg["rgb_subdir"], cfg["ir_subdir"],
        cfg["rgb_stem_suffix"], cfg["ir_stem_suffix"],
    )
    print(f"  Found {len(pairs)} paired frames")

    # Resume support
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
        if not det_path.exists():
            atomic_json_write(det_path, detections)
            print(f"  Wrote {det_path}")
        return

    t_start = time.time()
    for idx, pair in enumerate(remaining):
        rgb_dets, rgb_w, rgb_h = run_inference(rgb_model, pair["rgb_img"], cfg)
        ir_dets, ir_w, ir_h = run_inference(ir_model, pair["ir_img"], cfg)

        detections[pair["stem"]] = {
            "rgb_dets": rgb_dets,  # [[x1,y1,x2,y2,conf], ...]
            "ir_dets": ir_dets,
            "rgb_w": rgb_w, "rgb_h": rgb_h,
            "ir_w": ir_w, "ir_h": ir_h,
            "rgb_lbl": str(pair["rgb_lbl"]),
            "ir_lbl": str(pair["ir_lbl"]),
        }

        processed = idx + 1
        if processed % 50 == 0 or processed == len(remaining):
            elapsed = time.time() - t_start
            fps = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / fps if fps > 0 else 0
            print(f"  [{processed}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta / 60:.1f}min")
            # Atomic checkpoint
            atomic_json_write(ckpt_path, detections)

    # Save final
    atomic_json_write(det_path, detections)
    print(f"\nDone. Saved {len(detections)} frames to {det_path}")
    print(f"  File size: {det_path.stat().st_size / 1024 / 1024:.1f} MB")

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
