"""
run_inference_svanstrom.py — Run both YOLO models on Svanstrom paired dataset
and cache detections to JSON (same format as run_inference.py output).

Usage:
    python run_inference_svanstrom.py
    python run_inference_svanstrom.py --resume
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


def discover_paired_frames(dataset_root):
    """Find all paired IR/RGB frames in Svanstrom dataset."""
    ir_img_dir = Path(dataset_root) / "IR" / "images"
    rgb_img_dir = Path(dataset_root) / "RGB" / "images"
    ir_lbl_dir = Path(dataset_root) / "IR" / "labels"
    rgb_lbl_dir = Path(dataset_root) / "RGB" / "labels"

    # Build stem map: IR_DRONE_001_f000000 from IR_DRONE_001_f000000_infrared.jpg
    ir_files = {}
    for f in sorted(ir_img_dir.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stem = f.stem.replace("_infrared", "")
            ir_files[stem] = f

    rgb_files = {}
    for f in sorted(rgb_img_dir.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stem = f.stem.replace("_visible", "")
            rgb_files[stem] = f

    shared = sorted(set(ir_files) & set(rgb_files))

    pairs = []
    for stem in shared:
        ir_lbl = ir_lbl_dir / (ir_files[stem].stem + ".txt")
        rgb_lbl = rgb_lbl_dir / (rgb_files[stem].stem + ".txt")
        pairs.append({
            "stem": stem,
            "rgb_img": rgb_files[stem],
            "ir_img": ir_files[stem],
            "rgb_lbl": rgb_lbl,
            "ir_lbl": ir_lbl,
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
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="G:/drone/svanstrom_paired")
    parser.add_argument("--output", default="runs/svanstrom_detections.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".checkpoint.json")

    print("Loading models...")
    from ultralytics import YOLO
    rgb_model = YOLO(cfg["rgb_weights"])
    ir_model = YOLO(cfg["ir_weights"])
    print("  Models loaded.")

    print("Discovering paired frames...")
    pairs = discover_paired_frames(args.dataset)
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
        if not out_path.exists():
            atomic_json_write(out_path, detections)
        return

    t_start = time.time()
    for idx, pair in enumerate(remaining):
        rgb_dets, rgb_w, rgb_h = run_inference(rgb_model, pair["rgb_img"], cfg)
        ir_dets, ir_w, ir_h = run_inference(ir_model, pair["ir_img"], cfg)

        detections[pair["stem"]] = {
            "rgb_dets": rgb_dets,
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
            atomic_json_write(ckpt_path, detections)

    # Save final
    atomic_json_write(out_path, detections)
    print(f"\nDone. Saved {len(detections)} frames to {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
