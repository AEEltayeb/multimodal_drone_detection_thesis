"""
process_youtube_videos_4class.py — Generate rows in the 4-class fusion
dataset schema for the YouTube aerial-negative videos.

The GUI uses fusion_no_fn_model.joblib (4-class trust classifier) with
40 features. Its training CSV lives at
    classifier/runs/reliability/fusion/fusion_dataset.csv
This script appends YouTube grayscale-replicate rows (trust_label=0,
drone_present=0) into that CSV so the classifier learns to reject the
exact GUI failure mode: RGB airplane frame → both YOLOs fire → reject.

Both RGB-source and IR-source videos are processed:
  * RGB-source: RGB YOLO on the BGR frame, IR YOLO on gray-replicate
    (exact match to the GUI's grayscale-replicate failure mode).
  * IR-source: both YOLOs on gray-3ch of the thermal frame. RGB YOLO
    rarely fires on pure thermal (which is fine); IR YOLO fires
    natively. Scene features come from the thermal gray on both sides.
Svanström also supplies real-thermal aerial rows, but its IR YOLO never
fires on those scenes (ir_detected=0 in all 6090 airplane rows), so
YouTube IR videos are the first training rows where the IR side
actually fires on aerial negatives.

Output:
    classifier/runs/reliability/fusion/fusion_dataset_v2.csv   (merged)
    classifier/runs/reliability/fusion/fusion_dataset.csv.bak_v1
"""

import argparse
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


# ── Feature extraction (verbatim from ir_gui/fusion/features.py) ──

def compute_global_features(img_gray):
    h, w = img_gray.shape[:2]
    img_area = h * w
    img_f = img_gray.astype(np.float32)
    img_mean = float(img_f.mean())
    img_std = float(img_f.std())
    p2 = float(np.percentile(img_gray, 2))
    p98 = float(np.percentile(img_gray, 98))
    img_dynamic_range = p98 - p2
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    img_entropy = float(-np.sum(p * np.log2(p)))
    top_mean = float(img_f[:h // 2].mean())
    bot_mean = float(img_f[h // 2:].mean())
    sky_ground_ratio = top_mean / max(bot_mean, 1.0)
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(edges.sum()) / (img_area * 255.0)
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    blurriness = float(lap.var())
    return {
        "img_mean": round(img_mean, 3),
        "img_std": round(img_std, 3),
        "img_dynamic_range": round(img_dynamic_range, 3),
        "img_entropy": round(img_entropy, 4),
        "sky_ground_ratio": round(sky_ground_ratio, 4),
        "edge_density": round(edge_density, 6),
        "blurriness": round(blurriness, 3),
    }


def compute_target_features(img_gray, bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = pw * ph
    log_bbox_area = float(np.log(area + 1.0))
    aspect_ratio = float(pw / ph)
    pos_x = float(cx / img_w) if img_w > 0 else 0.5
    pos_y = float(cy / img_h) if img_h > 0 else 0.5
    dist_to_center = float(np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2))
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        local_contrast = 0.0
        target_bg_delta = 0.0
    else:
        target = img_gray[yi1:yi2, xi1:xi2].astype(np.float32)
        target_mean = float(target.mean())
        mx, my = int(pw), int(ph)
        bx1, by1 = max(0, xi1 - mx), max(0, yi1 - my)
        bx2, by2 = min(img_w, xi2 + mx), min(img_h, yi2 + my)
        bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        bg_mean = float(bg.mean())
        bg_std = float(bg.std())
        target_bg_delta = target_mean - bg_mean
        local_contrast = target_bg_delta / bg_std if bg_std >= 1.0 else 0.0
    return {
        "log_bbox_area": round(log_bbox_area, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "pos_x": round(pos_x, 4),
        "pos_y": round(pos_y, 4),
        "dist_to_center": round(dist_to_center, 4),
        "local_contrast": round(local_contrast, 4),
        "target_bg_delta": round(target_bg_delta, 3),
    }


TARGET_NAMES = ["log_bbox_area", "aspect_ratio", "pos_x", "pos_y",
                "dist_to_center", "local_contrast", "target_bg_delta"]


CATEGORY_MAP = {
    "airplane": "AIRPLANE", "plane": "AIRPLANE",
    "heli": "HELICOPTER", "helicopter": "HELICOPTER",
    "bird": "BIRD", "birds": "BIRD",
}


def parse_video_name(path):
    stem = path.stem.lower()
    parts = stem.split("_")
    cat = CATEGORY_MAP.get(parts[0])
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
    r = model.predict(
        source=img, conf=cfg["conf"], iou=cfg["iou_nms"],
        imgsz=cfg["imgsz"], device=cfg["device"],
        verbose=False, save=False, max_det=cfg["max_det"],
    )[0]
    boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(confs)):
            boxes.append((tuple(float(v) for v in xyxy[i]), float(confs[i])))
    return boxes


def det_stats(dets, prefix):
    n = len(dets)
    if n == 0:
        return {f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.0,
                f"{prefix}_mean_conf": 0.0, f"{prefix}_detected": 0}
    confs = [c for _, c in dets]
    return {
        f"{prefix}_n_dets": n,
        f"{prefix}_max_conf": round(float(max(confs)), 6),
        f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
        f"{prefix}_detected": 1,
    }


def best_det_target(dets, img_gray, img_w, img_h, prefix):
    if not dets:
        return {f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES}
    best_box, _ = max(dets, key=lambda d: d[1])
    tf = compute_target_features(img_gray, best_box, img_w, img_h)
    return {f"{prefix}_best_{k}": v for k, v in tf.items()}


def frame_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}
    feats.update(det_stats(rgb_dets, "rgb"))
    feats.update(det_stats(ir_dets, "ir"))
    rgb_global = compute_global_features(rgb_gray)
    ir_global = compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
    feats.update({f"ir_{k}": v for k, v in ir_global.items()})
    feats.update(best_det_target(rgb_dets, rgb_gray, rgb_w, rgb_h, "rgb"))
    feats.update(best_det_target(ir_dets, ir_gray, ir_w, ir_h, "ir"))
    rgb_detected = len(rgb_dets) > 0
    ir_detected = len(ir_dets) > 0
    feats["both_detect"] = int(rgb_detected and ir_detected)
    feats["neither_detect"] = int(not rgb_detected and not ir_detected)
    feats["rgb_only_detect"] = int(rgb_detected and not ir_detected)
    feats["ir_only_detect"] = int(not rgb_detected and ir_detected)
    feats["rgb_max_fn"] = 0.0
    feats["rgb_mean_fn"] = 0.0
    feats["rgb_min_fn"] = 0.0
    feats["ir_max_fn"] = 0.0
    feats["ir_mean_fn"] = 0.0
    feats["ir_min_fn"] = 0.0
    feats["rgb_n_gt"] = 0
    feats["ir_n_gt"] = 0
    feats["rgb_tp"] = 0
    feats["rgb_fp"] = len(rgb_dets)
    feats["ir_tp"] = 0
    feats["ir_fp"] = len(ir_dets)
    feats["rgb_has_tp"] = 0
    feats["ir_has_tp"] = 0
    feats["drone_present"] = 0
    feats["trust_label"] = 0
    return feats


def process_video(vpath, category, tag, modality, rgb_model, ir_model, cfg,
                  sample_fps):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        return
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / sample_fps)))
    print(f"  {vpath.name}: src_fps={src_fps:.1f} stride={stride} modality={modality}")
    frame_idx = 0
    source_dataset = f"youtube_aerial_{modality}"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if modality == "rgb":
            rgb_dets = run_yolo(rgb_model, frame, cfg)
            ir_dets = run_yolo(ir_model, gray_3ch, cfg)
        else:
            rgb_dets = run_yolo(rgb_model, gray_3ch, cfg)
            ir_dets = run_yolo(ir_model, gray_3ch, cfg)
        feats = frame_features(rgb_dets, ir_dets, gray, gray)
        feats["base_stem"] = f"IR_{category}_YT_{modality}_{tag}_f{frame_idx:06d}"
        feats["source_dataset"] = source_dataset
        yield feats
        frame_idx += 1
    cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="D:/Downloads/youtube_classifier_videos")
    p.add_argument("--config", default="classifier/config.yaml")
    p.add_argument("--base-csv",
                   default="classifier/runs/reliability/fusion/fusion_dataset.csv")
    p.add_argument("--merged-csv",
                   default="classifier/runs/reliability/fusion/fusion_dataset_v2.csv")
    p.add_argument("--youtube-only-csv",
                   default="classifier/runs/reliability/fusion/fusion_youtube_rows.csv")
    p.add_argument("--sample-fps", type=float, default=2.0)
    p.add_argument("--skip", nargs="*",
                   default=["airplane_rgb_2.mp4", "heli_ir_and_rgb_software.mp4"])
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_repo = Path(args.config).resolve().parent
    rgb_weights = (cfg_repo / cfg["rgb_weights"]).resolve()
    ir_weights = (cfg_repo / cfg["ir_weights"]).resolve()

    from ultralytics import YOLO
    print(f"Loading YOLO models...")
    rgb_model = YOLO(str(rgb_weights))
    ir_model = YOLO(str(ir_weights))

    videos = sorted([v for v in in_dir.iterdir()
                     if v.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
                     and v.name not in args.skip])
    todo = []
    for v in videos:
        parsed = parse_video_name(v)
        if parsed is None:
            print(f"  [SKIP unparseable] {v.name}")
            continue
        cat, modality, tag = parsed
        todo.append((v, cat, modality, tag))
    by_mod = {"rgb": 0, "ir": 0}
    for _, _, m, _ in todo:
        by_mod[m] += 1
    print(f"Processing {len(todo)} videos: "
          f"{by_mod['rgb']} RGB-source, {by_mod['ir']} IR-source.")

    t0 = time.time()
    rows = []
    for v, cat, modality, tag in todo:
        print(f"\n-- {v.name}  cat={cat}  modality={modality}")
        for row in process_video(v, cat, tag, modality, rgb_model, ir_model,
                                 cfg, args.sample_fps):
            rows.append(row)
    print(f"\nYouTube rows: {len(rows)} in {(time.time() - t0) / 60:.1f} min")

    yt_df = pd.DataFrame(rows)
    Path(args.youtube_only_csv).parent.mkdir(parents=True, exist_ok=True)
    yt_df.to_csv(args.youtube_only_csv, index=False)
    print(f"Saved YouTube-only rows -> {args.youtube_only_csv}")

    print("\nMerging with base CSV...")
    base = pd.read_csv(args.base_csv)
    print(f"  base rows: {len(base)}")
    merged = pd.concat([base, yt_df], ignore_index=True, sort=False)
    for col in base.columns:
        if col not in merged.columns:
            merged[col] = 0
    merged = merged[list(base.columns) +
                    [c for c in merged.columns if c not in base.columns]]
    merged.to_csv(args.merged_csv, index=False)
    print(f"  merged rows: {len(merged)}  ({len(merged) - len(base)} new)")
    print(f"  saved -> {args.merged_csv}")


if __name__ == "__main__":
    main()
