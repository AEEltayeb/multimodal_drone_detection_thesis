"""
process_youtube_videos.py — Run both YOLO models on YouTube aerial-negative
videos and emit fusion-format rows (label=0) for training + eval.

Filename convention (in --input-dir):
    {category}_{modality}[_N].mp4       airplane_rgb.mp4, heli_ir_2.mp4, ...
    where category ∈ {airplane, helicopter, heli, bird, birds}
          modality ∈ {rgb, ir}

For RGB-source videos, each frame is also fed to the IR YOLO as a 3ch
grayscale replicate (BGR→GRAY→merge([g,g,g])). This recreates the GUI's
no-thermal-camera regime and populates the (ir_is_real_thermal=0, aerial_neg,
both_fire=1) cell that is currently absent from training.

Output CSV columns (superset of svanstrom_frame_dataset.csv, with extras):
    stem, max_conf_rgb, max_conf_ir, conf_max, conf_min, conf_mean,
    conf_delta, both_detected, n_dets_rgb, n_dets_ir, n_dets_total,
    conf_rgb_2nd, conf_ir_2nd, rgb_area_norm, ir_area_norm, hour,
    time_of_day, rgb_brightness, ir_brightness, label, negative_class,
    source_modality, ir_is_real_thermal, video

Stems use the IR_{CATEGORY}_... prefix so eval_aerial_negatives.py's regex
picks up the category without changes.
"""

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


CATEGORY_MAP = {
    "airplane": "AIRPLANE",
    "plane": "AIRPLANE",
    "heli": "HELICOPTER",
    "helicopter": "HELICOPTER",
    "bird": "BIRD",
    "birds": "BIRD",
}


def parse_video_name(path: Path):
    """Return (category_upper, modality, tag) or None if unparseable."""
    stem = path.stem.lower()
    parts = stem.split("_")
    cat_token = parts[0]
    cat = CATEGORY_MAP.get(cat_token)
    if cat is None:
        return None
    modality = None
    for tok in parts[1:]:
        if tok in ("rgb", "visible", "vis"):
            modality = "rgb"
            break
        if tok == "ir":
            modality = "ir"
            break
    if modality is None:
        return None
    return cat, modality, stem


def run_yolo(model, img, cfg):
    """Return sorted list of confs (desc) and top-conf box (x1,y1,x2,y2) or None."""
    results = model.predict(
        source=img,
        conf=cfg["conf"],
        iou=cfg["iou_nms"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        verbose=False,
        save=False,
        max_det=cfg["max_det"],
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], None
    confs = r.boxes.conf.cpu().numpy().tolist()
    xyxy = r.boxes.xyxy.cpu().numpy()
    order = np.argsort(confs)[::-1]
    confs_sorted = [float(confs[i]) for i in order]
    top_box = tuple(float(v) for v in xyxy[order[0]])
    return confs_sorted, top_box


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def features_for_frame(rgb_confs, rgb_top_box, rgb_wh,
                       ir_confs, ir_top_box, ir_wh,
                       rgb_brightness, ir_brightness):
    max_conf_rgb = rgb_confs[0] if rgb_confs else 0.0
    max_conf_ir = ir_confs[0] if ir_confs else 0.0
    conf_max = max(max_conf_rgb, max_conf_ir)
    conf_min = min(max_conf_rgb, max_conf_ir)
    conf_mean = (max_conf_rgb + max_conf_ir) / 2.0
    conf_delta = abs(max_conf_rgb - max_conf_ir)
    both_detected = 1 if (max_conf_rgb > 0 and max_conf_ir > 0) else 0
    conf_rgb_2nd = rgb_confs[1] if len(rgb_confs) > 1 else 0.0
    conf_ir_2nd = ir_confs[1] if len(ir_confs) > 1 else 0.0

    if rgb_top_box is not None and rgb_wh[0] > 0 and rgb_wh[1] > 0:
        rgb_area_norm = box_area(rgb_top_box) / (rgb_wh[0] * rgb_wh[1])
    else:
        rgb_area_norm = 0.0
    if ir_top_box is not None and ir_wh[0] > 0 and ir_wh[1] > 0:
        ir_area_norm = box_area(ir_top_box) / (ir_wh[0] * ir_wh[1])
    else:
        ir_area_norm = 0.0

    return {
        "max_conf_rgb": round(max_conf_rgb, 4),
        "max_conf_ir": round(max_conf_ir, 4),
        "conf_max": round(conf_max, 4),
        "conf_min": round(conf_min, 4),
        "conf_mean": round(conf_mean, 4),
        "conf_delta": round(conf_delta, 4),
        "both_detected": both_detected,
        "n_dets_rgb": len(rgb_confs),
        "n_dets_ir": len(ir_confs),
        "n_dets_total": len(rgb_confs) + len(ir_confs),
        "conf_rgb_2nd": round(conf_rgb_2nd, 4),
        "conf_ir_2nd": round(conf_ir_2nd, 4),
        "rgb_area_norm": round(rgb_area_norm, 6),
        "ir_area_norm": round(ir_area_norm, 6),
        "rgb_brightness": round(rgb_brightness, 2),
        "ir_brightness": round(ir_brightness, 2),
    }


def process_video(video_path, category, modality, tag, rgb_model, ir_model,
                  cfg, sample_fps, max_frames):
    """Yield (row_dict, detection_summary_per_frame)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERR] cannot open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(src_fps / sample_fps)))
    print(f"  src_fps={src_fps:.1f} total_frames={total} stride={stride}")

    is_real_thermal = 1 if modality == "ir" else 0
    ir_mode_str = "real_thermal" if modality == "ir" else "grayscale_replicate"

    frame_idx = 0
    emitted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        if max_frames and emitted >= max_frames:
            break

        h, w = frame.shape[:2]

        if modality == "rgb":
            rgb_img = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ir_img = cv2.merge([gray, gray, gray])
            rgb_wh = (w, h)
            ir_wh = (w, h)
            rgb_brightness = float(np.mean(gray))
            ir_brightness = rgb_brightness
        else:
            ir_img = frame
            rgb_img = None
            rgb_wh = (0, 0)
            ir_wh = (w, h)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ir_brightness = float(np.mean(gray))
            rgb_brightness = 0.0

        if rgb_img is not None:
            rgb_confs, rgb_top = run_yolo(rgb_model, rgb_img, cfg)
        else:
            rgb_confs, rgb_top = [], None
        ir_confs, ir_top = run_yolo(ir_model, ir_img, cfg)

        feats = features_for_frame(rgb_confs, rgb_top, rgb_wh,
                                   ir_confs, ir_top, ir_wh,
                                   rgb_brightness, ir_brightness)

        stem = f"IR_{category}_YT_{tag}_f{frame_idx:06d}"
        row = {
            "stem": stem,
            **feats,
            "hour": 12,
            "time_of_day": "unknown",
            "label": 0,
            "negative_class": category.lower(),
            "source_modality": modality,
            "ir_is_real_thermal": is_real_thermal,
            "ir_mode": ir_mode_str,
            "video": video_path.name,
        }
        yield row

        frame_idx += 1
        emitted += 1

    cap.release()
    print(f"  emitted {emitted} frames from {video_path.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="D:/Downloads/youtube_classifier_videos")
    p.add_argument("--config", default="classifier/config.yaml")
    p.add_argument("--output-csv", default="classifier/runs/youtube_aerial_dataset.csv")
    p.add_argument("--output-summary", default="classifier/runs/youtube_aerial_summary.json")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="Frames per second to sample from each video")
    p.add_argument("--max-frames-per-video", type=int, default=0,
                   help="0 = no cap")
    p.add_argument("--skip", nargs="*", default=["airplane_rgb_2.mp4",
                                                  "heli_ir_and_rgb_software.mp4"],
                   help="Filenames to skip (duplicates / mixed content)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists():
        raise SystemExit(f"input dir not found: {in_dir}")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg_repo = Path(args.config).resolve().parent
    rgb_weights = (cfg_repo / cfg["rgb_weights"]).resolve()
    ir_weights = (cfg_repo / cfg["ir_weights"]).resolve()
    print(f"RGB weights: {rgb_weights}")
    print(f"IR  weights: {ir_weights}")

    from ultralytics import YOLO
    rgb_model = YOLO(str(rgb_weights))
    ir_model = YOLO(str(ir_weights))
    print("Models loaded.")

    videos = sorted([p for p in in_dir.iterdir()
                     if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
                     and p.name not in args.skip])
    print(f"Found {len(videos)} video(s) (skipping {args.skip})")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "stem", "max_conf_rgb", "max_conf_ir", "conf_max", "conf_min",
        "conf_mean", "conf_delta", "both_detected", "n_dets_rgb", "n_dets_ir",
        "n_dets_total", "conf_rgb_2nd", "conf_ir_2nd", "rgb_area_norm",
        "ir_area_norm", "hour", "time_of_day", "rgb_brightness",
        "ir_brightness", "label", "negative_class", "source_modality",
        "ir_is_real_thermal", "ir_mode", "video",
    ]

    summary = defaultdict(lambda: {"n_frames": 0, "rgb_fire": 0, "ir_fire": 0,
                                    "both_fire": 0, "rgb_fire_25": 0,
                                    "ir_fire_25": 0, "both_fire_25": 0})

    t0 = time.time()
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for v in videos:
            parsed = parse_video_name(v)
            if parsed is None:
                print(f"[SKIP] unparseable name: {v.name}")
                continue
            category, modality, tag = parsed
            print(f"\n--- {v.name}  cat={category}  modality={modality}  tag={tag}")
            for row in process_video(v, category, modality, tag,
                                     rgb_model, ir_model, cfg,
                                     args.sample_fps,
                                     args.max_frames_per_video):
                writer.writerow(row)
                key = v.name
                s = summary[key]
                s["n_frames"] += 1
                s["category"] = category
                s["modality"] = modality
                if row["max_conf_rgb"] > 0:
                    s["rgb_fire"] += 1
                if row["max_conf_ir"] > 0:
                    s["ir_fire"] += 1
                if row["both_detected"]:
                    s["both_fire"] += 1
                if row["max_conf_rgb"] >= 0.25:
                    s["rgb_fire_25"] += 1
                if row["max_conf_ir"] >= 0.25:
                    s["ir_fire_25"] += 1
                if row["max_conf_rgb"] >= 0.25 and row["max_conf_ir"] >= 0.25:
                    s["both_fire_25"] += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min.  CSV -> {out_csv}")

    report = {"summary": dict(summary), "config": {
        "input_dir": str(in_dir), "sample_fps": args.sample_fps,
        "rgb_weights": str(rgb_weights), "ir_weights": str(ir_weights),
        "conf": cfg["conf"], "imgsz": cfg["imgsz"],
    }}
    Path(args.output_summary).write_text(json.dumps(report, indent=2))
    print(f"Summary -> {args.output_summary}")

    print("\n=== Per-video fire rates (conf>=0.25) ===")
    print(f"{'video':40s} {'cat':11s} {'mod':4s} {'n':>5s} {'rgb%':>7s} {'ir%':>7s} {'both%':>7s}")
    for name, s in sorted(summary.items()):
        n = max(1, s["n_frames"])
        print(f"{name:40s} {s.get('category','?'):11s} {s.get('modality','?'):4s} "
              f"{s['n_frames']:5d} {100*s['rgb_fire_25']/n:6.1f}% "
              f"{100*s['ir_fire_25']/n:6.1f}% {100*s['both_fire_25']/n:6.1f}%")


if __name__ == "__main__":
    main()
