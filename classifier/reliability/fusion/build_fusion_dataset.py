"""
build_fusion_dataset.py — Build frame-level paired RGB+IR fusion dataset.

Each row = one paired frame. Each modality's detections matched to its OWN GT
(no cross-modal spatial matching — images are not co-registered).

Per frame computes:
  - RGB side: n_dets, max/mean conf, P(rgb_fn) stats, scene features
  - IR side:  n_dets, max/mean conf, P(ir_fn) stats, scene features
  - Frame-level agreement signals
  - Labels: rgb_has_tp, ir_has_tp, drone_present

Paired datasets: Anti-UAV val/test (123K), Svanstrom (29K) = 152K frames.

Outputs:
  runs/reliability/fusion/fusion_dataset.csv

Usage:
    python build_fusion_dataset.py
    python build_fusion_dataset.py --conf-thresh 0.4
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

# ── PATHS ──────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
RELIABILITY   = SCRIPT_DIR.parent
CLASSIFIER    = RELIABILITY.parent
INFERENCE_DIR_DEFAULT = CLASSIFIER / "runs" / "reliability" / "inference"
FM_DIR        = CLASSIFIER / "runs" / "reliability" / "failure_models"
OUTPUT_DIR    = CLASSIFIER / "runs" / "reliability" / "fusion"
CHECKPOINT_DIR_DEFAULT = OUTPUT_DIR / "checkpoints"

# Module-level globals overridden by CLI in main()
INFERENCE_DIR     = INFERENCE_DIR_DEFAULT  # RGB inference dir
IR_INFERENCE_DIR  = INFERENCE_DIR_DEFAULT  # IR inference dir (often same)
CHECKPOINT_DIR    = CHECKPOINT_DIR_DEFAULT

sys.path.insert(0, str(RELIABILITY))
from run_all_inference import DATASETS  # noqa: E402

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
CHECKPOINT_EVERY = 500


# ── PAIRED DATASET CONFIG ─────────────────────────────────────────

def build_paired_config():
    ds_map = {}
    for tag, img_dir, lbl_dir, modality in DATASETS:
        ds_map[tag] = (img_dir, lbl_dir, modality)

    pairs = [
        ("antiuav_val",  "antiuav_val_rgb",  "antiuav_val_ir"),
        ("antiuav_test", "antiuav_test_rgb", "antiuav_test_ir"),
        ("svanstrom",    "svanstrom_rgb",    "svanstrom_ir"),
    ]

    config = []
    for name, rgb_tag, ir_tag in pairs:
        if rgb_tag not in ds_map or ir_tag not in ds_map:
            continue
        config.append({
            "name": name,
            "rgb_tag": rgb_tag,
            "ir_tag": ir_tag,
            "rgb_img_dir": ds_map[rgb_tag][0],
            "ir_img_dir": ds_map[ir_tag][0],
        })
    return config


# ── STEM PAIRING ──────────────────────────────────────────────────

def strip_modality_suffix(stem):
    return re.sub(r"_(visible|infrared)", "", stem, flags=re.IGNORECASE)


def pair_frames(rgb_data, ir_data):
    rgb_map = {strip_modality_suffix(s): s for s in rgb_data}
    ir_map = {strip_modality_suffix(s): s for s in ir_data}
    paired = sorted(set(rgb_map.keys()) & set(ir_map.keys()))
    return [(base, rgb_map[base], ir_map[base]) for base in paired]


# ── IMAGE LOADING ─────────────────────────────────────────────────

def locate_image(img_dir, stem):
    img_dir = Path(img_dir)
    for ext in IMG_EXTENSIONS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for ext in IMG_EXTENSIONS:
        p = img_dir / f"{stem}{ext.upper()}"
        if p.exists():
            return p
    return None


# ── FEATURE COMPUTATION ──────────────────────────────────────────

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


# ── IoU + GT ─────────────────────────────────────────────────────

def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_gt(gt_text, img_w, img_h):
    boxes = []
    if not gt_text.strip():
        return boxes
    for line in gt_text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iop(det_box, gt_box):
    """Intersection over prediction area — robust to oversized GT."""
    x1 = max(det_box[0], gt_box[0]); y1 = max(det_box[1], gt_box[1])
    x2 = min(det_box[2], gt_box[2]); y2 = min(det_box[3], gt_box[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    det_area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
    return inter / det_area if det_area > 0 else 0.0


def match_dets_to_gt(dets, gt_boxes, iou_thresh, mode="iou"):
    """Returns (n_tp, n_fp, matched_det_indices). mode in {iou, iop}."""
    n_det = len(dets)
    n_gt = len(gt_boxes)
    det_matched = [False] * n_det
    gt_matched = [False] * n_gt

    if n_det == 0 or n_gt == 0:
        return 0, n_det, []

    score_fn = compute_iou if mode == "iou" else compute_iop
    pairs = []
    for di in range(n_det):
        for gi in range(n_gt):
            s = score_fn(dets[di][:4], gt_boxes[gi])
            if s >= iou_thresh:
                pairs.append((s, di, gi))
    pairs.sort(reverse=True)

    matched_dets = []
    for _, di, gi in pairs:
        if not det_matched[di] and not gt_matched[gi]:
            det_matched[di] = True
            gt_matched[gi] = True
            matched_dets.append(di)

    n_tp = sum(det_matched)
    n_fp = n_det - n_tp
    return n_tp, n_fp, matched_dets


# ── FN MODEL ─────────────────────────────────────────────────────

GLOBAL_NAMES = [
    "img_mean", "img_std", "img_dynamic_range", "img_entropy",
    "sky_ground_ratio", "edge_density", "blurriness",
]
TARGET_NAMES = [
    "log_bbox_area", "aspect_ratio", "pos_x", "pos_y", "dist_to_center",
    "local_contrast", "target_bg_delta",
]


def load_fn_models():
    models = {}
    for tag in ["rgb_fn", "ir_fn"]:
        path = FM_DIR / f"{tag}_model.joblib"
        if not path.exists():
            print(f"  [ERROR] {path} not found")
            sys.exit(1)
        bundle = joblib.load(path)
        models[tag] = bundle
        print(f"  Loaded {tag}: thresh={bundle['threshold']:.4f}")
    return models


def score_detections(dets, img_gray, img_w, img_h, fn_bundle, global_feats):
    """Score each detection with FN model. Returns list of P(FN) values."""
    if not dets:
        return []
    model = fn_bundle["model"]
    feat_names = fn_bundle["features"]
    scores = []
    for det in dets:
        target_feats = compute_target_features(img_gray, det[:4], img_w, img_h)
        merged = {**global_feats, **target_feats}
        X = np.array([[merged[f] for f in feat_names]])
        p_fn = float(model.predict_proba(X)[0, 1])
        scores.append(p_fn)
    return scores


# ── CHECKPOINT ────────────────────────────────────────────────────

def atomic_pickle_write(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(5):
        try:
            os.replace(tmp, str(path))
            return
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    if os.path.exists(str(path)):
        os.remove(str(path))
    os.rename(tmp, str(path))


def load_ckpt(ckpt_path):
    if not ckpt_path.exists():
        return [], set()
    try:
        with open(ckpt_path, "rb") as f:
            d = pickle.load(f)
        return d.get("rows", []), set(d.get("processed", []))
    except Exception as e:
        print(f"    [WARN] corrupt checkpoint: {e}")
        return [], set()


# ── PER-FRAME PROCESSING ─────────────────────────────────────────

def process_frame(
    rgb_frame, ir_frame, rgb_img, ir_img,
    fn_models, conf_thresh, gt_iou,
    rgb_match_mode="iou", ir_match_mode="iou",
):
    """Process one paired frame. Returns one row dict."""
    rgb_w, rgb_h = rgb_frame["w"], rgb_frame["h"]
    ir_w, ir_h = ir_frame["w"], ir_frame["h"]

    # Filter dets
    rgb_dets = [d for d in rgb_frame["dets"] if d[4] >= conf_thresh]
    ir_dets = [d for d in ir_frame["dets"] if d[4] >= conf_thresh]

    # Parse GT — each modality uses its OWN GT
    rgb_gt = parse_yolo_gt(rgb_frame.get("gt", ""), rgb_w, rgb_h)
    ir_gt = parse_yolo_gt(ir_frame.get("gt", ""), ir_w, ir_h)

    # Drone present = GT exists in either modality
    drone_present = 1 if (rgb_gt or ir_gt) else 0

    # Match each modality's dets to its own GT (svanstrom RGB GT is oversized → IoP)
    rgb_tp, rgb_fp, rgb_tp_idx = match_dets_to_gt(rgb_dets, rgb_gt, gt_iou, mode=rgb_match_mode)
    ir_tp, ir_fp, ir_tp_idx = match_dets_to_gt(ir_dets, ir_gt, gt_iou, mode=ir_match_mode)

    rgb_has_tp = 1 if rgb_tp > 0 else 0
    ir_has_tp = 1 if ir_tp > 0 else 0

    # Global features per modality
    rgb_global = compute_global_features(rgb_img)
    ir_global = compute_global_features(ir_img)

    # FN scores per detection
    rgb_fn_scores = score_detections(
        rgb_dets, rgb_img, rgb_w, rgb_h, fn_models["rgb_fn"], rgb_global
    )
    ir_fn_scores = score_detections(
        ir_dets, ir_img, ir_w, ir_h, fn_models["ir_fn"], ir_global
    )

    # Aggregate detection features per modality
    def det_stats(dets, fn_scores, prefix):
        """Aggregate detection-level info into frame-level features."""
        n = len(dets)
        if n == 0:
            return {
                f"{prefix}_n_dets": 0,
                f"{prefix}_max_conf": 0.0,
                f"{prefix}_mean_conf": 0.0,
                f"{prefix}_max_fn": 0.0,
                f"{prefix}_mean_fn": 0.0,
                f"{prefix}_min_fn": 0.0,
                f"{prefix}_detected": 0,
            }
        confs = [d[4] for d in dets]
        return {
            f"{prefix}_n_dets": n,
            f"{prefix}_max_conf": round(max(confs), 6),
            f"{prefix}_mean_conf": round(np.mean(confs), 6),
            f"{prefix}_max_fn": round(max(fn_scores), 6),
            f"{prefix}_mean_fn": round(np.mean(fn_scores), 6),
            f"{prefix}_min_fn": round(min(fn_scores), 6),
            f"{prefix}_detected": 1,
        }

    rgb_stats = det_stats(rgb_dets, rgb_fn_scores, "rgb")
    ir_stats = det_stats(ir_dets, ir_fn_scores, "ir")

    # Target features for highest-conf detection per modality
    def best_det_target(dets, img_gray, img_w, img_h, prefix):
        if not dets:
            return {f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES}
        best = max(dets, key=lambda d: d[4])
        tf = compute_target_features(img_gray, best[:4], img_w, img_h)
        return {f"{prefix}_best_{k}": v for k, v in tf.items()}

    rgb_best_target = best_det_target(rgb_dets, rgb_img, rgb_w, rgb_h, "rgb")
    ir_best_target = best_det_target(ir_dets, ir_img, ir_w, ir_h, "ir")

    # Frame-level agreement (no spatial matching — just "both detect something?")
    both_detect = 1 if (rgb_dets and ir_dets) else 0
    neither_detect = 1 if (not rgb_dets and not ir_dets) else 0
    rgb_only_detect = 1 if (rgb_dets and not ir_dets) else 0
    ir_only_detect = 1 if (not rgb_dets and ir_dets) else 0

    # Trust label: what SHOULD we trust?
    # 0=reject_both, 1=trust_rgb, 2=trust_ir, 3=trust_both
    if rgb_has_tp and ir_has_tp:
        trust_label = 3  # both correct
    elif rgb_has_tp and not ir_has_tp:
        trust_label = 1  # only RGB correct
    elif not rgb_has_tp and ir_has_tp:
        trust_label = 2  # only IR correct
    else:
        trust_label = 0  # neither correct (both missed or no drone)

    row = {
        # RGB detection aggregates
        **rgb_stats,
        # IR detection aggregates
        **ir_stats,
        # RGB scene features
        **{f"rgb_{k}": v for k, v in rgb_global.items()},
        # IR scene features
        **{f"ir_{k}": v for k, v in ir_global.items()},
        # RGB best-detection target features
        **rgb_best_target,
        # IR best-detection target features
        **ir_best_target,
        # Frame-level agreement
        "both_detect": both_detect,
        "neither_detect": neither_detect,
        "rgb_only_detect": rgb_only_detect,
        "ir_only_detect": ir_only_detect,
        # Raw counts for analysis
        "rgb_n_gt": len(rgb_gt),
        "ir_n_gt": len(ir_gt),
        "rgb_tp": rgb_tp,
        "rgb_fp": rgb_fp,
        "ir_tp": ir_tp,
        "ir_fp": ir_fp,
        # Labels
        "drone_present": drone_present,
        "rgb_has_tp": rgb_has_tp,
        "ir_has_tp": ir_has_tp,
        "trust_label": trust_label,
    }
    return row


# ── FEATURE COLUMN DEFINITIONS ────────────────────────────────────

FUSION_FEATURE_COLS = [
    # RGB detection aggregates (7)
    "rgb_n_dets", "rgb_max_conf", "rgb_mean_conf",
    "rgb_max_fn", "rgb_mean_fn", "rgb_min_fn", "rgb_detected",
    # IR detection aggregates (7)
    "ir_n_dets", "ir_max_conf", "ir_mean_conf",
    "ir_max_fn", "ir_mean_fn", "ir_min_fn", "ir_detected",
    # RGB scene (7)
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
    "rgb_img_entropy", "rgb_sky_ground_ratio", "rgb_edge_density",
    "rgb_blurriness",
    # IR scene (7)
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
    "ir_img_entropy", "ir_sky_ground_ratio", "ir_edge_density",
    "ir_blurriness",
    # RGB best-det target (7)
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    # IR best-det target (7)
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
    # Frame-level agreement (4)
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
]

# 7+7+7+7+7+7+4 = 46 features


# ── PER-DATASET PROCESSING ───────────────────────────────────────

def process_dataset(cfg, fn_models, conf_thresh, gt_iou):
    name = cfg["name"]
    # Svanstrom RGB GT is oversized — use IoP for RGB matching there only.
    rgb_match_mode = "iop" if name == "svanstrom" else "iou"
    ir_match_mode  = "iou"

    parquet_path = CHECKPOINT_DIR / f"{name}.parquet"
    if parquet_path.exists():
        print(f"  [CACHED] {name}")
        return pd.read_parquet(parquet_path)

    rgb_json = INFERENCE_DIR / f"{cfg['rgb_tag']}.json"
    ir_json = IR_INFERENCE_DIR / f"{cfg['ir_tag']}.json"
    if not rgb_json.exists() or not ir_json.exists():
        print(f"  [SKIP] {name}: missing JSONs")
        return None

    print(f"  Loading JSONs...", end="", flush=True)
    with open(rgb_json, "r", encoding="utf-8") as f:
        rgb_data = json.load(f)
    with open(ir_json, "r", encoding="utf-8") as f:
        ir_data = json.load(f)
    print(f" {len(rgb_data):,} + {len(ir_data):,}")

    frame_pairs = pair_frames(rgb_data, ir_data)
    print(f"    Paired: {len(frame_pairs):,}")

    ckpt_path = CHECKPOINT_DIR / f"{name}.pkl"
    rows, processed = load_ckpt(ckpt_path)
    remaining = [(b, r, i) for b, r, i in frame_pairs if b not in processed]
    print(f"    Cached: {len(processed):,}, remaining: {len(remaining):,}")

    if not remaining:
        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, index=False)
        if ckpt_path.exists():
            ckpt_path.unlink()
        return df

    n_missing = 0
    n_session = 0
    t0 = time.time()

    for idx, (base_stem, rgb_stem, ir_stem) in enumerate(remaining):
        rgb_path = locate_image(cfg["rgb_img_dir"], rgb_stem)
        ir_path = locate_image(cfg["ir_img_dir"], ir_stem)

        if rgb_path is None or ir_path is None:
            n_missing += 1
            processed.add(base_stem)
            n_session += 1
            continue

        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or ir_img is None:
            n_missing += 1
            processed.add(base_stem)
            n_session += 1
            continue

        rgb_frame = rgb_data[rgb_stem]
        ir_frame = ir_data[ir_stem]

        row = process_frame(
            rgb_frame, ir_frame, rgb_img, ir_img,
            fn_models, conf_thresh, gt_iou,
            rgb_match_mode=rgb_match_mode, ir_match_mode=ir_match_mode,
        )
        row["base_stem"] = base_stem
        row["source_dataset"] = name
        rows.append(row)

        processed.add(base_stem)
        n_session += 1

        if n_session % CHECKPOINT_EVERY == 0 or (idx + 1) == len(remaining):
            elapsed = time.time() - t0
            fps = n_session / elapsed if elapsed > 0 else 0
            left = len(remaining) - (idx + 1)
            eta_min = left / fps / 60 if fps > 0 else 0
            print(f"    [{idx + 1}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta_min:.1f}min | rows: {len(rows):,}")
            atomic_pickle_write(
                ckpt_path,
                {"rows": rows, "processed": list(processed)},
            )

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"  Done {name}: {n_session:,} frames in {elapsed / 60:.1f}min, "
          f"{n_missing:,} missing")
    return df


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-thresh", type=float, default=0.4)
    parser.add_argument("--gt-iou", type=float, default=0.5,
                        help="IoU for matching dets to own-modality GT")
    parser.add_argument("--inference-dir", type=str, default=str(INFERENCE_DIR_DEFAULT),
                        help="Dir containing RGB inference JSONs")
    parser.add_argument("--ir-inference-dir", type=str, default=None,
                        help="Dir containing IR inference JSONs (defaults to --inference-dir)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for fusion_dataset and checkpoints, e.g. _v3more")
    args = parser.parse_args()

    global INFERENCE_DIR, IR_INFERENCE_DIR, CHECKPOINT_DIR
    INFERENCE_DIR     = Path(args.inference_dir)
    IR_INFERENCE_DIR  = Path(args.ir_inference_dir) if args.ir_inference_dir else INFERENCE_DIR
    CHECKPOINT_DIR    = OUTPUT_DIR / f"checkpoints{args.out_suffix}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build frame-level fusion dataset (paired RGB+IR)")
    print("=" * 70)
    print(f"  Conf threshold: {args.conf_thresh}")
    print(f"  GT IoU:         {args.gt_iou}")
    print(f"  Features:       {len(FUSION_FEATURE_COLS)}")
    print()

    print("Loading FN models...")
    fn_models = load_fn_models()
    print()

    paired_configs = build_paired_config()
    all_dfs = []

    for cfg in paired_configs:
        print(f"\n{'=' * 60}")
        print(f"Processing {cfg['name']}")
        print(f"{'=' * 60}")
        df = process_dataset(cfg, fn_models, args.conf_thresh, args.gt_iou)
        if df is not None and len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo data. Check paired datasets.")
        return

    master = pd.concat(all_dfs, ignore_index=True)
    csv_path = OUTPUT_DIR / f"fusion_dataset{args.out_suffix}.csv"
    master.to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("FUSION DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total frames: {len(master):,}")
    print(f"Drone present: {(master['drone_present'] == 1).sum():,} "
          f"({(master['drone_present'] == 1).mean() * 100:.1f}%)")
    print()

    # Trust label distribution
    labels = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
    print("Trust label distribution:")
    for val, name in labels.items():
        n = (master["trust_label"] == val).sum()
        pct = n / len(master) * 100
        print(f"  {val} ({name:<12s}): {n:>8,} ({pct:>5.1f}%)")

    print()
    print("Per-dataset:")
    print(f"  {'dataset':<15s} {'frames':>8s} {'drone':>7s} {'rgb_tp':>7s} "
          f"{'ir_tp':>7s} {'both':>7s} {'neither':>7s}")
    for ds in master["source_dataset"].unique():
        s = master[master["source_dataset"] == ds]
        n_drone = (s["drone_present"] == 1).sum()
        n_rgb_tp = (s["rgb_has_tp"] == 1).sum()
        n_ir_tp = (s["ir_has_tp"] == 1).sum()
        n_both_tp = ((s["rgb_has_tp"] == 1) & (s["ir_has_tp"] == 1)).sum()
        n_neither = ((s["rgb_has_tp"] == 0) & (s["ir_has_tp"] == 0) & (s["drone_present"] == 1)).sum()
        print(f"  {ds:<15s} {len(s):>8,} {n_drone:>7,} {n_rgb_tp:>7,} "
              f"  {n_ir_tp:>7,} {n_both_tp:>7,} {n_neither:>7,}")

    print(f"\nFeatures: {len(FUSION_FEATURE_COLS)}")
    print(f"Saved: {csv_path.name}")
    print("Done. Ready for train_fusion.py")


if __name__ == "__main__":
    main()
