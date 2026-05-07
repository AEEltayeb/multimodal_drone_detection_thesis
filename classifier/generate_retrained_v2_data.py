"""
generate_retrained_v2_data.py — Generate 32-feature fusion dataset for
retrained_v2 RGB model + IR model.

Sources:
  1. Anti-UAV paired (RGB + IR) — positives + some negatives
  2. Confuser videos (RGB color + grayscale-as-IR) — all negatives

Produces the same 32 features as scene_aware_v3more_32feat:
  - 4 detection confidence features (rgb/ir max/mean conf)
  - 14 scene features (7 per modality: mean, std, dynamic_range, entropy,
    sky_ground_ratio, edge_density, blurriness)
  - 14 target features (7 per modality: log_bbox_area, aspect_ratio,
    pos_x, pos_y, dist_to_center, local_contrast, target_bg_delta)

Usage:
    python generate_retrained_v2_data.py
    python generate_retrained_v2_data.py --auv-stride 2 --confuser-stride 3
    python generate_retrained_v2_data.py --neg-keep 0.20
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np

# ── Feature computation (same as build_fusion_dataset.py) ────────

def compute_global_features(img_gray):
    """Scene-level features from grayscale image."""
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
    """Per-detection features for the best detection."""
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
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


# The 32 feature columns (must match model exactly)
FEATURE_COLS = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
    "rgb_img_entropy", "rgb_sky_ground_ratio", "rgb_edge_density",
    "rgb_blurriness",
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
    "ir_img_entropy", "ir_sky_ground_ratio", "ir_edge_density",
    "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]


def run_yolo(model, img, conf=0.25, imgsz=640):
    """Run YOLO inference, return list of [x1,y1,x2,y2,conf]."""
    results = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append([float(xyxy[i][0]), float(xyxy[i][1]),
                         float(xyxy[i][2]), float(xyxy[i][3]),
                         float(confs[i])])
    return dets


# ── Detection cache ──────────────────────────────────────────────

class DetectionCache:
    """Cache YOLO detections to disk as JSON. Keyed by stem."""

    def __init__(self, cache_path):
        self.path = Path(cache_path)
        self.data = {}
        self.dirty = False
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = json.load(f)
            print(f"  Cache loaded: {self.path.name} ({len(self.data)} entries)")

    def has(self, stem):
        return stem in self.data

    def get(self, stem):
        entry = self.data[stem]
        return entry["rgb_dets"], entry["ir_dets"]

    def put(self, stem, rgb_dets, ir_dets):
        self.data[stem] = {"rgb_dets": rgb_dets, "ir_dets": ir_dets}
        self.dirty = True

    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.data, f)
            print(f"  Cache saved: {self.path.name} ({len(self.data)} entries)")

    def __len__(self):
        return len(self.data)


def build_row(rgb_dets, ir_dets, rgb_gray, ir_gray, rgb_wh, ir_wh,
              label, stem, source, conf_thresh=0.25):
    """Build one feature row from RGB and IR detections + images."""
    rgb_w, ir_w = rgb_wh[0], ir_wh[0]
    rgb_h, ir_h = rgb_wh[1], ir_wh[1]

    # Filter by confidence
    rgb_dets = [d for d in rgb_dets if d[4] >= conf_thresh]
    ir_dets = [d for d in ir_dets if d[4] >= conf_thresh]

    # Detection confidence features
    rgb_confs = [d[4] for d in rgb_dets]
    ir_confs = [d[4] for d in ir_dets]

    row = {
        "rgb_max_conf": max(rgb_confs) if rgb_confs else 0.0,
        "rgb_mean_conf": round(float(np.mean(rgb_confs)), 6) if rgb_confs else 0.0,
        "ir_max_conf": max(ir_confs) if ir_confs else 0.0,
        "ir_mean_conf": round(float(np.mean(ir_confs)), 6) if ir_confs else 0.0,
    }

    # Scene features
    rgb_global = compute_global_features(rgb_gray)
    ir_global = compute_global_features(ir_gray)
    for k, v in rgb_global.items():
        row[f"rgb_{k}"] = v
    for k, v in ir_global.items():
        row[f"ir_{k}"] = v

    # Best detection target features (RGB)
    if rgb_dets:
        best_rgb = max(rgb_dets, key=lambda d: d[4])
        tf = compute_target_features(rgb_gray, best_rgb[:4], rgb_w, rgb_h)
        for k, v in tf.items():
            row[f"rgb_best_{k}"] = v
    else:
        for k in ["log_bbox_area", "aspect_ratio", "pos_x", "pos_y",
                   "dist_to_center", "local_contrast", "target_bg_delta"]:
            row[f"rgb_best_{k}"] = 0.0

    # Best detection target features (IR)
    if ir_dets:
        best_ir = max(ir_dets, key=lambda d: d[4])
        tf = compute_target_features(ir_gray, best_ir[:4], ir_w, ir_h)
        for k, v in tf.items():
            row[f"ir_best_{k}"] = v
    else:
        for k in ["log_bbox_area", "aspect_ratio", "pos_x", "pos_y",
                   "dist_to_center", "local_contrast", "target_bg_delta"]:
            row[f"ir_best_{k}"] = 0.0

    # Metadata
    row["trust_label"] = label
    row["stem"] = stem
    row["source"] = source

    return row


def has_gt(label_path):
    """Check if YOLO label file has any ground truth."""
    p = Path(label_path)
    if not p.exists():
        return False
    text = p.read_text().strip()
    return any(len(line.split()) >= 5 for line in text.split("\n") if line)


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def compute_iop(det_box, gt_box):
    """Intersection over prediction area — robust to oversized GT."""
    x1 = max(det_box[0], gt_box[0]); y1 = max(det_box[1], gt_box[1])
    x2 = min(det_box[2], gt_box[2]); y2 = min(det_box[3], gt_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    det_area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
    return inter / det_area if det_area > 0 else 0.0


def parse_yolo_gt(label_path, img_w, img_h):
    """Parse YOLO label to pixel (x1,y1,x2,y2) boxes."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(((cx - w/2)*img_w, (cy - h/2)*img_h,
                       (cx + w/2)*img_w, (cy + h/2)*img_h))
    return boxes


def has_tp(dets, gt_boxes, thresh=0.5, mode="iou"):
    """Check if any detection matches any GT box. Returns bool."""
    if not dets or not gt_boxes:
        return False
    score_fn = compute_iou if mode == "iou" else compute_iop
    for d in dets:
        for g in gt_boxes:
            if score_fn(d[:4], g) >= thresh:
                return True
    return False


def compute_trust_label(rgb_dets, ir_dets, rgb_lbl, ir_lbl,
                        rgb_wh, ir_wh, rgb_match_mode="iou"):
    """Compute 4-class trust label: 0=reject, 1=trust_rgb, 2=trust_ir, 3=trust_both."""
    rgb_gt = parse_yolo_gt(rgb_lbl, rgb_wh[0], rgb_wh[1])
    ir_gt = parse_yolo_gt(ir_lbl, ir_wh[0], ir_wh[1])
    rgb_has = has_tp(rgb_dets, rgb_gt, mode=rgb_match_mode)
    ir_has = has_tp(ir_dets, ir_gt, mode="iou")
    if rgb_has and ir_has:
        return 3
    elif rgb_has:
        return 1
    elif ir_has:
        return 2
    else:
        return 0


# ── Anti-UAV processing ──────────────────────────────────────────

def discover_antiuav_pairs(dataset_root):
    """Find paired RGB+IR frames in Anti-UAV dataset."""
    import re

    rgb_img_dir = Path(dataset_root) / "RGB" / "images"
    ir_img_dir = Path(dataset_root) / "IR" / "images"
    rgb_lbl_dir = Path(dataset_root) / "RGB" / "labels"
    ir_lbl_dir = Path(dataset_root) / "IR" / "labels"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    def strip_suffix(stem):
        return re.sub(r"_(visible|infrared)", "", stem, flags=re.IGNORECASE)

    rgb_map = {}
    for f in sorted(rgb_img_dir.iterdir()):
        if f.suffix.lower() in img_exts:
            base = strip_suffix(f.stem)
            rgb_map[base] = f

    ir_map = {}
    for f in sorted(ir_img_dir.iterdir()):
        if f.suffix.lower() in img_exts:
            base = strip_suffix(f.stem)
            ir_map[base] = f

    shared = sorted(set(rgb_map) & set(ir_map))
    print(f"  Anti-UAV: {len(rgb_map)} RGB, {len(ir_map)} IR, {len(shared)} paired")

    pairs = []
    for base in shared:
        rgb_img = rgb_map[base]
        ir_img = ir_map[base]
        rgb_lbl = rgb_lbl_dir / (rgb_img.stem + ".txt")
        ir_lbl = ir_lbl_dir / (ir_img.stem + ".txt")
        is_positive = has_gt(rgb_lbl) or has_gt(ir_lbl)
        pairs.append({
            "base_stem": base,
            "rgb_img": rgb_img,
            "ir_img": ir_img,
            "rgb_lbl": rgb_lbl,
            "ir_lbl": ir_lbl,
            "is_positive": is_positive,
        })
    return pairs


def process_antiuav(rgb_model, ir_model, dataset_root, stride, neg_keep,
                    conf_thresh, imgsz, cache=None):
    """Process Anti-UAV paired frames."""
    pairs = discover_antiuav_pairs(dataset_root)

    # Apply stride
    pairs = pairs[::stride]

    # Split positives and negatives
    positives = [p for p in pairs if p["is_positive"]]
    negatives = [p for p in pairs if not p["is_positive"]]

    # Subsample negatives
    n_neg_keep = int(len(negatives) * neg_keep)
    random.seed(42)
    negatives = random.sample(negatives, min(n_neg_keep, len(negatives)))

    frames = positives + negatives
    random.shuffle(frames)

    n_cached = sum(1 for p in frames if cache and cache.has(p["base_stem"]))
    print(f"  Processing: {len(positives)} pos + {len(negatives)} neg "
          f"= {len(frames)} frames ({n_cached} cached)")

    rows = []
    t0 = time.time()
    for idx, pair in enumerate(frames):
        stem = pair["base_stem"]

        rgb_img = cv2.imread(str(pair["rgb_img"]))
        ir_img = cv2.imread(str(pair["ir_img"]))
        if rgb_img is None or ir_img is None:
            continue

        rgb_h, rgb_w = rgb_img.shape[:2]
        ir_h, ir_w = ir_img.shape[:2]

        # Use cache or run inference
        if cache and cache.has(stem):
            rgb_dets, ir_dets = cache.get(stem)
        else:
            rgb_dets = run_yolo(rgb_model, rgb_img, conf=conf_thresh, imgsz=imgsz)
            ir_dets = run_yolo(ir_model, ir_img, conf=conf_thresh, imgsz=imgsz)
            if cache:
                cache.put(stem, rgb_dets, ir_dets)

        # Grayscale for features
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape) == 3 else ir_img

        # Compute 4-class trust label via GT matching
        trust = compute_trust_label(
            rgb_dets, ir_dets,
            pair["rgb_lbl"], pair["ir_lbl"],
            (rgb_w, rgb_h), (ir_w, ir_h),
            rgb_match_mode="iou",
        )
        row = build_row(
            rgb_dets, ir_dets, rgb_gray, ir_gray,
            (rgb_w, rgb_h), (ir_w, ir_h),
            trust, pair["base_stem"], "antiuav",
            conf_thresh=conf_thresh,
        )
        rows.append(row)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            eta = (len(frames) - idx - 1) / fps / 60
            print(f"    [{idx + 1}/{len(frames)}] {fps:.1f} fps, "
                  f"ETA {eta:.1f}min, {len(rows)} rows")

    if cache:
        cache.save()
    print(f"  Anti-UAV done: {len(rows)} rows in {(time.time()-t0)/60:.1f}min")
    return rows


# ── Svanström processing ─────────────────────────────────────────

def discover_svanstrom_pairs(dataset_root):
    """Find paired RGB+IR frames in Svanström dataset."""
    import re

    rgb_img_dir = Path(dataset_root) / "RGB" / "images"
    ir_img_dir = Path(dataset_root) / "IR" / "images"
    rgb_lbl_dir = Path(dataset_root) / "RGB" / "labels"
    ir_lbl_dir = Path(dataset_root) / "IR" / "labels"

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    def strip_suffix(stem):
        return re.sub(r"_(visible|infrared)", "", stem, flags=re.IGNORECASE)

    rgb_map = {}
    for f in sorted(rgb_img_dir.iterdir()):
        if f.suffix.lower() in img_exts:
            rgb_map[strip_suffix(f.stem)] = f

    ir_map = {}
    for f in sorted(ir_img_dir.iterdir()):
        if f.suffix.lower() in img_exts:
            ir_map[strip_suffix(f.stem)] = f

    shared = sorted(set(rgb_map) & set(ir_map))
    print(f"  Svanstrom: {len(rgb_map)} RGB, {len(ir_map)} IR, {len(shared)} paired")

    pairs = []
    for base in shared:
        rgb_img = rgb_map[base]
        ir_img = ir_map[base]
        rgb_lbl = rgb_lbl_dir / (rgb_img.stem + ".txt")
        ir_lbl = ir_lbl_dir / (ir_img.stem + ".txt")
        pairs.append({
            "base_stem": base,
            "rgb_img": rgb_img,
            "ir_img": ir_img,
            "rgb_lbl": rgb_lbl,
            "ir_lbl": ir_lbl,
        })
    return pairs


# ── Confuser video processing ────────────────────────────────────

def process_confuser_videos(rgb_model, ir_model, video_configs, stride,
                            conf_thresh, imgsz):
    """Process confuser videos (RGB color + grayscale-as-IR)."""
    rows = []

    for vconf in video_configs:
        vpath = Path(vconf["path"])
        label_name = vconf["label"]
        print(f"\n  Video: {vpath.name} [{label_name}]")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"    [SKIP] Cannot open")
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"    {total} frames, {w}x{h}, stride={stride}")

        t0 = time.time()
        idx = 0
        processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if stride > 1 and (idx % stride != 0):
                continue
            processed += 1

            # RGB model on color frame
            rgb_dets = run_yolo(rgb_model, frame, conf=conf_thresh, imgsz=imgsz)

            # IR model on grayscale version of same frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            ir_dets = run_yolo(ir_model, gray_3ch, conf=conf_thresh, imgsz=imgsz)

            # Features: RGB scene from color (as grayscale), IR scene from grayscale
            rgb_gray = gray  # same underlying image
            ir_gray = gray   # grayscale version = "IR"

            stem = f"{vpath.stem}_f{idx:06d}"
            row = build_row(
                rgb_dets, ir_dets, rgb_gray, ir_gray,
                (w, h), (w, h),
                0,  # always reject — confuser
                stem, f"confuser_{label_name}",
                conf_thresh=conf_thresh,
            )
            rows.append(row)

            if processed % 200 == 0:
                elapsed = time.time() - t0
                fps = processed / elapsed
                print(f"    [{processed}] {fps:.1f} fps, {len(rows)} rows")

        cap.release()
        elapsed = time.time() - t0
        print(f"    Done: {processed} frames in {elapsed:.0f}s")

    print(f"\n  Confuser total: {len(rows)} rows")
    return rows


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 32-feature fusion dataset for retrained_v2")
    parser.add_argument("--rgb-weights", required=True,
                        help="RGB model weights (retrained_v2)")
    parser.add_argument("--ir-weights", required=True,
                        help="IR model weights")
    parser.add_argument("--auv-root",
                        default="G:/drone/Anti-UAV-RGBT_yolo_converted/test",
                        help="Anti-UAV paired dataset root")
    parser.add_argument("--svan-root",
                        default="G:/drone/svanstrom_paired",
                        help="Svanstrom paired dataset root")
    parser.add_argument("--auv-stride", type=int, default=2)
    parser.add_argument("--svan-stride", type=int, default=2)
    parser.add_argument("--neg-keep", type=float, default=0.20,
                        help="Fraction of Anti-UAV negatives to keep")
    parser.add_argument("--confuser-stride", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output-dir", type=str,
                        default="fusion_models/retrained_v2_32feat")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Confuser video configs
    base = Path(__file__).resolve().parent.parent
    demo_dir = base / "ir_gui" / "demo_outputs"
    confuser_videos = [
        {"path": str(demo_dir / "yt_Z8HJNypu_1Y.mp4"), "label": "AIRPLANE"},
        {"path": str(demo_dir / "yt_1U7Bu2pSUwU.mp4"), "label": "HELICOPTER"},
        {"path": str(demo_dir / "yt_ZO5lV0gh5i4.mp4"), "label": "BIRD"},
    ]

    print("=" * 70)
    print("Generate 32-feature fusion dataset (retrained_v2)")
    print("=" * 70)
    print(f"  RGB weights: {args.rgb_weights}")
    print(f"  IR weights:  {args.ir_weights}")
    print(f"  AUV stride:  {args.auv_stride}, neg_keep: {args.neg_keep}")
    print(f"  Confuser stride: {args.confuser_stride}")
    print(f"  Output: {out_dir}")
    print()

    # Load models
    print("Loading models...")
    from ultralytics import YOLO
    rgb_model = YOLO(args.rgb_weights)
    ir_model = YOLO(args.ir_weights)
    print("  Models loaded.\n")

    # Phase 1: Anti-UAV paired data
    print("=" * 60)
    print("Phase 1: Anti-UAV paired data")
    print("=" * 60)
    auv_cache = DetectionCache(out_dir / "cache_antiuav.json")
    auv_rows = process_antiuav(
        rgb_model, ir_model, args.auv_root,
        args.auv_stride, args.neg_keep,
        args.conf, args.imgsz, cache=auv_cache,
    )

    # Phase 2: Svanstrom paired data (IoP for RGB)
    print("\n" + "=" * 60)
    print("Phase 2: Svanstrom paired data")
    print("=" * 60)
    svan_root = Path(args.svan_root)
    if svan_root.exists():
        svan_cache = DetectionCache(out_dir / "cache_svanstrom.json")
        svan_pairs = discover_svanstrom_pairs(svan_root)
        svan_pairs = svan_pairs[::args.svan_stride]
        n_cached = sum(1 for p in svan_pairs if svan_cache.has(p["base_stem"]))
        print(f"  Processing: {len(svan_pairs)} frames (stride {args.svan_stride}, {n_cached} cached)")
        svan_rows = []
        t0 = time.time()
        for idx, pair in enumerate(svan_pairs):
            stem = pair["base_stem"]
            rgb_img = cv2.imread(str(pair["rgb_img"]))
            ir_img = cv2.imread(str(pair["ir_img"]))
            if rgb_img is None or ir_img is None:
                continue
            rgb_h, rgb_w = rgb_img.shape[:2]
            ir_h, ir_w = ir_img.shape[:2]
            if svan_cache.has(stem):
                rgb_dets, ir_dets = svan_cache.get(stem)
            else:
                rgb_dets = run_yolo(rgb_model, rgb_img, conf=args.conf, imgsz=args.imgsz)
                ir_dets = run_yolo(ir_model, ir_img, conf=args.conf, imgsz=args.imgsz)
                svan_cache.put(stem, rgb_dets, ir_dets)
            rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape) == 3 else ir_img
            trust = compute_trust_label(
                rgb_dets, ir_dets,
                pair["rgb_lbl"], pair["ir_lbl"],
                (rgb_w, rgb_h), (ir_w, ir_h),
                rgb_match_mode="iop",  # IoP for Svanstrom RGB
            )
            row = build_row(
                rgb_dets, ir_dets, rgb_gray, ir_gray,
                (rgb_w, rgb_h), (ir_w, ir_h),
                trust, pair["base_stem"], "svanstrom",
                conf_thresh=args.conf,
            )
            svan_rows.append(row)
            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                fps = (idx + 1) / elapsed
                eta = (len(svan_pairs) - idx - 1) / fps / 60
                print(f"    [{idx+1}/{len(svan_pairs)}] {fps:.1f} fps, ETA {eta:.1f}min")
        svan_cache.save()
        print(f"  Svanstrom done: {len(svan_rows)} rows in {(time.time()-t0)/60:.1f}min")
    else:
        print(f"  [SKIP] {svan_root} not found")
        svan_rows = []

    # Phase 3: Confuser videos
    print("\n" + "=" * 60)
    print("Phase 3: Confuser videos (RGB + grayscale-as-IR)")
    print("=" * 60)
    confuser_rows = process_confuser_videos(
        rgb_model, ir_model, confuser_videos,
        args.confuser_stride, args.conf, args.imgsz,
    )

    # Combine
    all_rows = auv_rows + svan_rows + confuser_rows
    from collections import Counter
    trust_dist = Counter(r["trust_label"] for r in all_rows)
    TRUST_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

    print(f"\n{'=' * 70}")
    print("DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total rows: {len(all_rows):,}")
    for t in sorted(trust_dist):
        print(f"  {TRUST_NAMES[t]:15s} (class {t}): {trust_dist[t]:,}")
    print(f"  Sources:")
    sources = {}
    for r in all_rows:
        src = r["source"]
        sources[src] = sources.get(src, 0) + 1
    for src, n in sorted(sources.items()):
        print(f"    {src}: {n:,}")

    # Save CSV
    csv_path = out_dir / "fusion_dataset.csv"
    fieldnames = FEATURE_COLS + ["trust_label", "stem", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")

    # Save config
    config = {
        "rgb_weights": args.rgb_weights,
        "ir_weights": args.ir_weights,
        "auv_stride": args.auv_stride,
        "neg_keep": args.neg_keep,
        "confuser_stride": args.confuser_stride,
        "conf": args.conf,
        "trust_distribution": {TRUST_NAMES[k]: v for k, v in sorted(trust_dist.items())},
        "total_rows": len(all_rows),
        "feature_cols": FEATURE_COLS,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved: {out_dir / 'config.json'}")
    print("\nDone. Run train_classifier.py on this CSV next.")


if __name__ == "__main__":
    main()
