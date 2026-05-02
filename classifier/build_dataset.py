"""
build_dataset.py — Phase 2: Read raw detections from Phase 1 (run_inference.py),
align cross-modality detections, extract features, label against GT, and save CSV.

This is the FAST step. It reads pre-computed detections from raw_detections.json
and can be re-run in seconds with different alignment/feature parameters.

Features extracted:
  Core (always):  conf_max, conf_min, agreement, bbox_area_norm
  Extended:       conf_delta, aspect_ratio, n_dets_total
  Environmental:  hour, time_of_day (day/night/dusk_dawn)
  Thermal:        ir_contrast_ratio (drone-to-background thermal contrast)

Usage:
    python build_dataset.py
    python build_dataset.py --mode detection   # force detection-level
    python build_dataset.py --mode frame       # force frame-level
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils import (
    align_detections,
    align_detections_by_distance,
    box_area,
    compute_iou,
    extract_features,
    label_candidates,
    representative_box,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GT parsing helpers
# ---------------------------------------------------------------------------

def parse_gt_norm(label_path):
    """Parse YOLO label → list of (cx, cy, w, h) normalized."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    text = p.read_text().strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def parse_norm_xyxy(label_path):
    """Parse YOLO label → normalized (x1, y1, x2, y2) boxes."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    text = p.read_text().strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def has_gt(label_path):
    p = Path(label_path)
    if not p.exists():
        return False
    text = p.read_text().strip()
    return any(len(line.split()) >= 5 for line in text.split("\n") if line)


def parse_yolo_labels_px(label_path, img_w, img_h):
    """Parse YOLO label → pixel (x1, y1, x2, y2) boxes."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


# ---------------------------------------------------------------------------
# Alignment check (auto-detect mode)
# ---------------------------------------------------------------------------

def check_alignment(detections, sample_size=500, iou_threshold=0.3,
                    aligned_fraction_needed=0.5):
    """
    Sample frames and check if RGB & IR GT boxes overlap in
    normalized coords. Returns ('detection', stats) or ('frame', stats).
    """
    candidates = []
    for stem, d in detections.items():
        rgb_boxes = parse_norm_xyxy(d["rgb_lbl"])
        ir_boxes = parse_norm_xyxy(d["ir_lbl"])
        if rgb_boxes and ir_boxes:
            candidates.append((rgb_boxes, ir_boxes))

    if not candidates:
        return "frame", {"reason": "no frames with GT in both modalities"}

    sample = random.sample(candidates, min(sample_size, len(candidates)))

    ious = []
    for rgb_boxes, ir_boxes in sample:
        for rb in rgb_boxes:
            best = max((compute_iou(rb, ib) for ib in ir_boxes), default=0.0)
            ious.append(best)

    if not ious:
        return "frame", {"reason": "no GT boxes to compare"}

    ious = np.array(ious)
    frac_above = float((ious >= iou_threshold).mean())

    stats = {
        "sample_size": len(sample),
        "gt_pairs_compared": len(ious),
        "mean_iou": round(float(ious.mean()), 4),
        "median_iou": round(float(np.median(ious)), 4),
        "fraction_above": round(frac_above, 4),
        "threshold": iou_threshold,
    }

    if frac_above >= aligned_fraction_needed:
        return "detection", stats
    else:
        return "frame", stats


# ---------------------------------------------------------------------------
# Per-SEQUENCE GT calibration (replaces broken per-frame approach)
# ---------------------------------------------------------------------------

def extract_sequence_id(stem):
    """
    Extract sequence ID from stem.

    Stem pattern: 20190925_111757_1_10_f000000
    Sequence ID:  20190925_111757_1_10 (everything before _fNNNNNN)
    """
    parts = stem.rsplit("_f", 1)
    return parts[0] if len(parts) == 2 else stem


def compute_per_frame_offset(gt_rgb_norm, gt_ir_norm):
    """
    Compute single-frame offset from GT correspondence.
    Returns (dcx, dcy, sw, sh) or None.
    """
    if not gt_rgb_norm or not gt_ir_norm:
        return None
    r = gt_rgb_norm[0]  # (cx, cy, w, h)
    i = gt_ir_norm[0]
    if i[2] <= 0 or i[3] <= 0:
        return None
    dcx = r[0] - i[0]
    dcy = r[1] - i[1]
    sw = r[2] / i[2]
    sh = r[3] / i[3]
    return (dcx, dcy, sw, sh)


def compute_sequence_offsets(detections):
    """
    Pre-compute a per-sequence (dcx, dcy, sw, sh) offset by averaging
    all GT correspondences within each tracking sequence.

    Returns: dict { seq_id: (dcx, dcy, sw, sh) }
    """
    from collections import defaultdict

    seq_offsets_raw = defaultdict(list)

    for stem, d in detections.items():
        seq_id = extract_sequence_id(stem)
        gt_rgb = parse_gt_norm(d["rgb_lbl"])
        gt_ir = parse_gt_norm(d["ir_lbl"])
        offset = compute_per_frame_offset(gt_rgb, gt_ir)
        if offset is not None:
            seq_offsets_raw[seq_id].append(offset)

    # Average offsets per sequence
    seq_offsets = {}
    for seq_id, offsets in seq_offsets_raw.items():
        dcx = np.median([o[0] for o in offsets])
        dcy = np.median([o[1] for o in offsets])
        sw = np.median([o[2] for o in offsets])
        sh = np.median([o[3] for o in offsets])
        seq_offsets[seq_id] = (float(dcx), float(dcy), float(sw), float(sh))

    # Global fallback: median of all per-frame offsets
    all_offsets = [o for offsets in seq_offsets_raw.values() for o in offsets]
    if all_offsets:
        global_offset = (
            float(np.median([o[0] for o in all_offsets])),
            float(np.median([o[1] for o in all_offsets])),
            float(np.median([o[2] for o in all_offsets])),
            float(np.median([o[3] for o in all_offsets])),
        )
    else:
        global_offset = (0.0, 0.0, 1.0, 1.0)  # identity transform

    return seq_offsets, global_offset


def get_offset_for_stem(stem, seq_offsets, global_offset):
    """Get the best calibration offset for a given stem."""
    seq_id = extract_sequence_id(stem)
    return seq_offsets.get(seq_id, global_offset)


def calibrate_ir_box(box_xyxy, offset, ir_w, ir_h, rgb_w, rgb_h):
    """Apply offset to transform an IR detection into RGB pixel space."""
    dcx, dcy, sw, sh = offset
    x1, y1, x2, y2 = box_xyxy
    cx_n = ((x1 + x2) / 2) / ir_w
    cy_n = ((y1 + y2) / 2) / ir_h
    w_n = (x2 - x1) / ir_w
    h_n = (y2 - y1) / ir_h
    new_cx = cx_n + dcx
    new_cy = cy_n + dcy
    new_w = w_n * sw
    new_h = h_n * sh
    rx1 = (new_cx - new_w / 2) * rgb_w
    ry1 = (new_cy - new_h / 2) * rgb_h
    rx2 = (new_cx + new_w / 2) * rgb_w
    ry2 = (new_cy + new_h / 2) * rgb_h
    return (rx1, ry1, rx2, ry2)


# ---------------------------------------------------------------------------
# Environmental / thermal feature extraction
# ---------------------------------------------------------------------------

def extract_time_features(stem):
    """
    Extract time-of-day from Anti-UAV-RGBT stem name.

    Stem pattern: 20190925_111757_1_10_f000000
                  YYYYMMDD_HHMMSS_...

    Returns: (hour: int, period: str)
        period is one of: "night", "dusk_dawn", "day"
    """
    match = re.match(r"(\d{8})_(\d{6})", stem)
    if not match:
        return None, "unknown"

    time_str = match.group(2)  # "111757"
    hour = int(time_str[:2])

    if 7 <= hour < 17:
        period = "day"
    elif (5 <= hour < 7) or (17 <= hour < 20):
        period = "dusk_dawn"
    else:
        period = "night"

    return hour, period


def resolve_ir_image_path(stem, cfg):
    """
    Reconstruct the IR image path from stem + config.
    Suffix goes before _fNNNNNN: stem '..._1_f000020' + '_infrared' -> '..._1_infrared_f000020'
    """
    ir_img_dir = Path(cfg["dataset_root"]) / cfg["ir_subdir"] / "images"
    ir_suffix = cfg["ir_stem_suffix"]

    m = re.match(r"(.+)(_f\d+)$", stem)
    if m:
        filename_base = m.group(1) + ir_suffix + m.group(2)
    else:
        filename_base = stem + ir_suffix
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = ir_img_dir / f"{filename_base}{ext}"
        if candidate.exists():
            return candidate
    return None


def compute_ir_contrast(ir_img_path, det_box, ir_w, ir_h):
    """
    Compute thermal contrast ratio: how different the detection region is
    from the surrounding background in the IR image.

    Returns ir_contrast_ratio:
        > 1.0 = detection region is BRIGHTER (hotter) than background
        < 1.0 = detection region is DARKER (cooler) than background
        ≈ 1.0 = low contrast (thermal crossover — IR struggles here)

    Also returns ir_local_contrast:
        Absolute contrast = |det_mean - bg_mean| / frame_std
        Higher = more visible to IR sensor
        Lower = harder to detect (thermal crossover)
    """
    if ir_img_path is None:
        return None, None

    img = cv2.imread(str(ir_img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None

    h, w = img.shape[:2]

    # Clamp detection box to image bounds
    x1 = max(0, int(det_box[0]))
    y1 = max(0, int(det_box[1]))
    x2 = min(w, int(det_box[2]))
    y2 = min(h, int(det_box[3]))

    if x2 <= x1 or y2 <= y1:
        return None, None

    det_region = img[y1:y2, x1:x2]
    det_mean = float(det_region.mean())
    frame_mean = float(img.mean())
    frame_std = float(img.std())

    # Contrast ratio: how different detection is from frame average
    contrast_ratio = det_mean / frame_mean if frame_mean > 0 else 1.0

    # Local contrast: standardized absolute difference
    local_contrast = abs(det_mean - frame_mean) / frame_std if frame_std > 0 else 0.0

    return round(contrast_ratio, 4), round(local_contrast, 4)


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def resolve_rgb_image_path(stem, cfg):
    """Reconstruct the RGB image path from stem + config."""
    rgb_img_dir = Path(cfg["dataset_root"]) / cfg["rgb_subdir"] / "images"
    rgb_suffix = cfg["rgb_stem_suffix"]
    # Suffix goes before _fNNNNNN: stem '..._1_f000020' + '_visible' -> '..._1_visible_f000020'
    m = re.match(r"(.+)(_f\d+)$", stem)
    if m:
        filename_base = m.group(1) + rgb_suffix + m.group(2)
    else:
        filename_base = stem + rgb_suffix
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = rgb_img_dir / f"{filename_base}{ext}"
        if candidate.exists():
            return candidate
    return None


def compute_frame_brightness(img_path):
    """Compute mean pixel intensity of an image (0-255 scale)."""
    if img_path is None:
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(img.mean())


def process_frame_level(stem, d, cfg, rgb_brightness=None, ir_brightness=None):
    """Frame-level: one row per frame with features from both models."""
    rgb_confs = sorted([det[4] for det in d["rgb_dets"]], reverse=True)
    ir_confs = sorted([det[4] for det in d["ir_dets"]], reverse=True)
    gt_present = 1 if (has_gt(d["rgb_lbl"]) or has_gt(d["ir_lbl"])) else 0

    max_rgb = rgb_confs[0] if rgb_confs else 0.0
    max_ir = ir_confs[0] if ir_confs else 0.0

    hour, time_period = extract_time_features(stem)

    row = {
        "stem": stem,
        # Core: both models always represented
        "max_conf_rgb": max_rgb,
        "max_conf_ir": max_ir,
        "conf_max": max(max_rgb, max_ir),
        "conf_min": min(max_rgb, max_ir),
        "conf_mean": (max_rgb + max_ir) / 2,
        "conf_delta": abs(max_rgb - max_ir),
        # Agreement: both models detected something
        "both_detected": 1.0 if (rgb_confs and ir_confs) else 0.0,
        # Detection counts
        "n_dets_rgb": len(rgb_confs),
        "n_dets_ir": len(ir_confs),
        "n_dets_total": len(rgb_confs) + len(ir_confs),
    }

    # Second-best confidence (clutter indicator)
    row["conf_rgb_2nd"] = rgb_confs[1] if len(rgb_confs) > 1 else 0.0
    row["conf_ir_2nd"] = ir_confs[1] if len(ir_confs) > 1 else 0.0

    # Best detection area (normalized) from each model
    img_w, img_h = d["rgb_w"], d["rgb_h"]
    ir_w, ir_h = d["ir_w"], d["ir_h"]
    img_area = img_w * img_h
    ir_area = ir_w * ir_h

    if d["rgb_dets"]:
        best_rgb = max(d["rgb_dets"], key=lambda x: x[4])
        rgb_box_area = (best_rgb[2] - best_rgb[0]) * (best_rgb[3] - best_rgb[1])
        row["rgb_area_norm"] = rgb_box_area / img_area if img_area > 0 else 0
    else:
        row["rgb_area_norm"] = 0.0

    if d["ir_dets"]:
        best_ir = max(d["ir_dets"], key=lambda x: x[4])
        ir_box_area = (best_ir[2] - best_ir[0]) * (best_ir[3] - best_ir[1])
        row["ir_area_norm"] = ir_box_area / ir_area if ir_area > 0 else 0
    else:
        row["ir_area_norm"] = 0.0

    # Time features
    if cfg.get("use_time_features", True):
        row["hour"] = hour if hour is not None else -1
        row["time_of_day"] = time_period

    # Brightness features
    if rgb_brightness is not None:
        row["rgb_brightness"] = round(rgb_brightness, 2)
    if ir_brightness is not None:
        row["ir_brightness"] = round(ir_brightness, 2)

    row["label"] = gt_present
    return row


def process_detection_level(stem, d, cfg, ir_img_cache, seq_offsets, global_offset):
    """Detection-level: per-bbox features with sequence-calibrated alignment."""
    img_w, img_h = d["rgb_w"], d["rgb_h"]
    ir_w, ir_h = d["ir_w"], d["ir_h"]

    # Convert raw detections back to (box_tuple, conf) format
    rgb_dets = [(tuple(det[:4]), det[4]) for det in d["rgb_dets"]]
    ir_dets = [(tuple(det[:4]), det[4]) for det in d["ir_dets"]]

    # Get pre-computed sequence offset (ALWAYS available, never None)
    offset = get_offset_for_stem(stem, seq_offsets, global_offset)

    # Calibrate ALL IR detections into RGB pixel space
    ir_dets_calibrated = [
        (calibrate_ir_box(box, offset, ir_w, ir_h, img_w, img_h), conf)
        for box, conf in ir_dets
    ]

    # Align calibrated IR with RGB detections
    matched, rgb_only, ir_only = align_detections(
        rgb_dets, ir_dets_calibrated, iou_thresh=cfg["alignment_iou"]
    )

    # --- Second-pass rescue ---
    # If high-conf detections from both models went unmatched (due to bad GT
    # causing bad offset), re-align using prediction-based correspondence.
    # Only rescue when the high-conf prediction is NEAR a GT box (prevents
    # matching two unrelated false positives).
    HIGH_CONF = 0.5
    GT_PROXIMITY = 0.3  # RGB prediction must overlap RGB GT by at least this IoU
    gt_rgb_px_early = parse_yolo_labels_px(d["rgb_lbl"], img_w, img_h)

    hc_rgb = [d_r for d_r in rgb_only if d_r["rgb_conf"] >= HIGH_CONF]
    hc_ir = [d_i for d_i in ir_only if d_i["ir_conf"] >= HIGH_CONF]
    if hc_rgb and hc_ir and gt_rgb_px_early:
        # Filter: only consider RGB predictions near a GT box
        hc_rgb_near_gt = []
        for d_r in hc_rgb:
            best_gt_iou = max(
                (compute_iou(d_r["rgb_box"], gt) for gt in gt_rgb_px_early),
                default=0.0)
            if best_gt_iou >= GT_PROXIMITY:
                hc_rgb_near_gt.append(d_r)

        if hc_rgb_near_gt:
            best_rgb = max(hc_rgb_near_gt, key=lambda x: x["rgb_conf"])
            best_ir = max(hc_ir, key=lambda x: x["ir_conf"])
            rb = best_rgb["rgb_box"]
            ib_cal = best_ir["ir_box"]  # already in RGB space from first pass
            rgb_cx = (rb[0] + rb[2]) / 2
            rgb_cy = (rb[1] + rb[3]) / 2
            ir_cx = (ib_cal[0] + ib_cal[2]) / 2
            ir_cy = (ib_cal[1] + ib_cal[3]) / 2
            # Only rescue if in same general area (within 30% of image)
            dx = abs(rgb_cx - ir_cx) / img_w
            dy = abs(rgb_cy - ir_cy) / img_h
            if dx < 0.3 and dy < 0.3:
                correction = (rgb_cx - ir_cx, rgb_cy - ir_cy)
                ir_only_corrected = []
                for d_ir in ir_only:
                    box = d_ir["ir_box"]
                    new_box = (box[0] + correction[0], box[1] + correction[1],
                               box[2] + correction[0], box[3] + correction[1])
                    ir_only_corrected.append((new_box, d_ir["ir_conf"]))
                rgb_only_dets = [(d_r["rgb_box"], d_r["rgb_conf"]) for d_r in rgb_only]
                matched2, rgb_only2, ir_only2 = align_detections(
                    rgb_only_dets, ir_only_corrected, iou_thresh=cfg["alignment_iou"]
                )
                if matched2:
                    matched.extend(matched2)
                    rgb_only = rgb_only2
                    ir_only = ir_only2

    sources = (["both"] * len(matched) +
               ["rgb_only"] * len(rgb_only) +
               ["ir_only"] * len(ir_only))
    candidates = matched + rgb_only + ir_only

    # Reuse GT already parsed above for labeling
    gt_rgb_px = gt_rgb_px_early

    if not candidates:
        return [], len(gt_rgb_px)

    # Label matched + rgb_only against RGB GT (they're in RGB pixel space)
    non_ir_only = matched + rgb_only
    if non_ir_only:
        labels_non_ir, n_missed = label_candidates(non_ir_only, gt_rgb_px,
                                                    matching_iou=cfg["matching_iou"])
    else:
        labels_non_ir = []
        n_missed = len(gt_rgb_px)

    # Label ir_only against IR GT in native IR pixel space
    # Use ir_raw_idx to look up original uncalibrated IR boxes
    gt_ir_px = parse_yolo_labels_px(d["ir_lbl"], ir_w, ir_h)
    ir_only_native = []
    for d_ir in ir_only:
        raw_idx = d_ir.get("ir_raw_idx")
        if raw_idx is not None and raw_idx < len(ir_dets):
            raw_box = ir_dets[raw_idx][0]
            ir_only_native.append({"ir_box": raw_box, "ir_conf": d_ir["ir_conf"]})
        else:
            ir_only_native.append(d_ir)  # fallback

    if ir_only_native and gt_ir_px:
        labels_ir, _ = label_candidates(ir_only_native, gt_ir_px,
                                         matching_iou=cfg["matching_iou"])
    else:
        labels_ir = [0] * len(ir_only)

    labels = labels_non_ir + labels_ir

    img_area = img_w * img_h
    n_total = len(candidates)

    # Time features (same for all candidates in this frame)
    hour, time_period = extract_time_features(stem)

    # IR contrast: load image once per frame (use cache)
    use_ir_contrast = cfg.get("use_ir_contrast", False)
    ir_img_path = None
    if use_ir_contrast:
        if stem not in ir_img_cache:
            ir_img_cache[stem] = resolve_ir_image_path(stem, cfg)
        ir_img_path = ir_img_cache[stem]

    rows = []
    for cand, label, source in zip(candidates, labels, sources):
        # Core features
        feats = extract_features(cand, img_area, n_dets_total=n_total)

        conf_rgb = cand.get("rgb_conf", 0.0)
        conf_ir = cand.get("ir_conf", 0.0)

        row = {
            "stem": stem,
            "source": source,
            "conf_rgb": conf_rgb,
            "conf_ir": conf_ir,
            **feats,
        }

        # Extended features
        if cfg.get("use_conf_delta", False):
            row["conf_delta"] = round(abs(conf_rgb - conf_ir), 6)

        if cfg.get("use_aspect_ratio", False):
            box = representative_box(cand)
            w = box[2] - box[0]
            h = box[3] - box[1]
            row["aspect_ratio"] = round(w / h, 4) if h > 0 else 1.0

        if cfg.get("use_n_dets_total", False):
            row["n_dets_total"] = n_total

        # Environmental features
        if cfg.get("use_time_features", True):
            row["hour"] = hour if hour is not None else -1
            row["time_of_day"] = time_period

        # Thermal contrast feature
        if use_ir_contrast:
            # Use the IR detection box (in IR pixel space, not calibrated)
            ir_box = cand.get("ir_box", None)
            if ir_box is not None:
                contrast_ratio, local_contrast = compute_ir_contrast(
                    ir_img_path, ir_box, ir_w, ir_h)
                row["ir_contrast_ratio"] = contrast_ratio if contrast_ratio is not None else 1.0
                row["ir_local_contrast"] = local_contrast if local_contrast is not None else 0.0
            else:
                # RGB-only detection — no IR box to measure contrast
                row["ir_contrast_ratio"] = 1.0  # neutral
                row["ir_local_contrast"] = 0.0  # no signal

        row["label"] = label
        rows.append(row)

    return rows, n_missed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_detection_distance(stem, d, cfg, ir_img_cache):
    """Detection-level using center-distance matching (no calibration needed)."""
    img_w, img_h = d["rgb_w"], d["rgb_h"]
    ir_w, ir_h = d["ir_w"], d["ir_h"]

    rgb_dets = [(tuple(det[:4]), det[4]) for det in d["rgb_dets"]]
    ir_dets = [(tuple(det[:4]), det[4]) for det in d["ir_dets"]]

    # Match by normalized center distance — no calibration offset needed
    max_dist = cfg.get("max_center_dist", 0.15)
    matched, rgb_only, ir_only = align_detections_by_distance(
        rgb_dets, ir_dets,
        rgb_wh=(img_w, img_h), ir_wh=(ir_w, ir_h),
        max_dist=max_dist,
    )

    sources = (["both"] * len(matched) +
               ["rgb_only"] * len(rgb_only) +
               ["ir_only"] * len(ir_only))
    candidates = matched + rgb_only + ir_only

    # Label against RGB GT (IR boxes are already scaled to RGB pixel space by align fn)
    gt_rgb_px = parse_yolo_labels_px(d["rgb_lbl"], img_w, img_h)
    # Also check IR GT for ir_only detections
    gt_ir_px = parse_yolo_labels_px(d["ir_lbl"], ir_w, ir_h)

    if not candidates:
        return [], len(gt_rgb_px)

    # Label: for each candidate, check against RGB GT first,
    # then for ir_only also check against IR GT (scaled to RGB space)
    labels, n_missed = label_candidates(candidates, gt_rgb_px,
                                        matching_iou=cfg["matching_iou"])

    # Second chance: ir_only candidates that didn't match RGB GT —
    # check against IR GT in IR native space
    for ci in range(len(candidates)):
        if labels[ci] == 0 and sources[ci] == "ir_only":
            ir_box_native = candidates[ci].get("ir_box", None)
            if ir_box_native and gt_ir_px:
                # Scale ir_box back to IR space for comparison
                ir_box_native_px = (
                    ir_box_native[0] * ir_w / img_w,
                    ir_box_native[1] * ir_h / img_h,
                    ir_box_native[2] * ir_w / img_w,
                    ir_box_native[3] * ir_h / img_h,
                )
                best_iou = max((compute_iou(ir_box_native_px, gt) for gt in gt_ir_px), default=0)
                if best_iou >= cfg["matching_iou"]:
                    labels[ci] = 1

    img_area = img_w * img_h
    n_total = len(candidates)
    hour, time_period = extract_time_features(stem)

    use_ir_contrast = cfg.get("use_ir_contrast", False)
    ir_img_path = None
    if use_ir_contrast:
        if stem not in ir_img_cache:
            ir_img_cache[stem] = resolve_ir_image_path(stem, cfg)
        ir_img_path = ir_img_cache[stem]

    rows = []
    for cand, label, source in zip(candidates, labels, sources):
        feats = extract_features(cand, img_area, n_dets_total=n_total)
        conf_rgb = cand.get("rgb_conf", 0.0)
        conf_ir = cand.get("ir_conf", 0.0)

        row = {
            "stem": stem,
            "source": source,
            "conf_rgb": conf_rgb,
            "conf_ir": conf_ir,
            **feats,
        }

        if cfg.get("use_conf_delta", False):
            row["conf_delta"] = round(abs(conf_rgb - conf_ir), 6)
        if cfg.get("use_aspect_ratio", False):
            box = representative_box(cand)
            w = box[2] - box[0]
            h = box[3] - box[1]
            row["aspect_ratio"] = round(w / h, 4) if h > 0 else 1.0
        if cfg.get("use_n_dets_total", False):
            row["n_dets_total"] = n_total
        if cfg.get("use_time_features", True):
            row["hour"] = hour if hour is not None else -1
            row["time_of_day"] = time_period
        if use_ir_contrast:
            ir_box = cand.get("ir_box", None)
            if ir_box is not None:
                contrast_ratio, local_contrast = compute_ir_contrast(
                    ir_img_path, ir_box, ir_w, ir_h)
                row["ir_contrast_ratio"] = contrast_ratio if contrast_ratio is not None else 1.0
                row["ir_local_contrast"] = local_contrast if local_contrast is not None else 0.0
            else:
                row["ir_contrast_ratio"] = 1.0
                row["ir_local_contrast"] = 0.0

        row["label"] = label
        rows.append(row)

    return rows, n_missed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Build fusion dataset from raw detections")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["auto", "frame", "detection"],
                        default="auto", help="Fusion level (default: auto-detect)")
    parser.add_argument("--calibration", choices=["gt_offset", "distance"],
                        default="gt_offset",
                        help="Calibration approach: gt_offset (per-sequence GT) or distance (normalized center distance, no calibration)")
    parser.add_argument("--detections", default=None,
                        help="Path to raw_detections.json (default: runs/raw_detections.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw detections from Phase 1
    det_path = Path(args.detections) if args.detections else out_dir / "raw_detections.json"
    if not det_path.exists():
        # Also check for checkpoint from interrupted Phase 1
        ckpt_path = out_dir / "inference_checkpoint.json"
        if ckpt_path.exists():
            print(f"No raw_detections.json found, but inference checkpoint exists.")
            print(f"Using checkpoint ({ckpt_path.stat().st_size / 1024 / 1024:.1f} MB)...")
            det_path = ckpt_path
        else:
            print("ERROR: No raw detections found. Run 'python run_inference.py' first (Phase 1).")
            return

    print(f"Loading detections from {det_path}...")
    t_load = time.time()
    with open(det_path, "r", encoding="utf-8") as f:
        detections = json.load(f)
    print(f"  Loaded {len(detections)} frames in {time.time() - t_load:.1f}s")

    # --- Feature config summary ---
    print(f"\nFeatures enabled:")
    print(f"  Core:       conf_max, conf_min, agreement, bbox_area_norm (always)")
    for feat in ["use_conf_delta", "use_aspect_ratio", "use_n_dets_total",
                 "use_time_features", "use_ir_contrast"]:
        status = "✓" if cfg.get(feat, feat == "use_time_features") else "✗"
        print(f"  {feat:<22s} {status}")

    # --- Decide fusion mode ---
    if args.mode == "auto":
        print("\nChecking GT alignment (sampling up to 500 frames)...")
        mode, stats = check_alignment(detections)
        print(f"  Alignment stats: {json.dumps(stats, indent=2)}")
        print(f"  → Auto-selected mode: {mode.upper()}-level fusion")
    else:
        mode = args.mode
        stats = {"forced": True}
        print(f"\n  → Forced mode: {mode.upper()}-level fusion")

    # Save mode metadata
    meta = {"mode": mode, "alignment_stats": stats, "n_frames": len(detections),
            "calibration": args.calibration}
    with open(out_dir / "fusion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Pre-compute calibration offsets (only for detection + gt_offset mode) ---
    seq_offsets, global_offset = None, None
    if mode == "frame":
        print("\n  Skipping calibration (frame-level mode)")
    elif args.calibration == "gt_offset":
        print("\nComputing per-sequence calibration offsets...")
        seq_offsets, global_offset = compute_sequence_offsets(detections)
        print(f"  Computed offsets for {len(seq_offsets)} sequences")
        print(f"  Global fallback offset: dcx={global_offset[0]:.4f}, dcy={global_offset[1]:.4f}, "
              f"sw={global_offset[2]:.4f}, sh={global_offset[3]:.4f}")
    else:
        print(f"\nUsing center-distance matching (max_dist={cfg.get('max_center_dist', 0.15)})")

    # --- Process all frames ---
    print(f"\nProcessing {len(detections)} frames...")
    t_start = time.time()
    all_rows = []
    total_missed_gt = 0
    ir_img_cache = {}  # stem → ir_img_path (avoid repeated disk lookups)

    use_brightness = cfg.get("use_brightness", False) and mode == "frame"
    if use_brightness:
        print("  Computing brightness features (loading images from disk)...")

    for idx, (stem, d) in enumerate(sorted(detections.items())):
        if mode == "frame":
            rgb_bright, ir_bright = None, None
            if use_brightness:
                rgb_path = resolve_rgb_image_path(stem, cfg)
                ir_path = resolve_ir_image_path(stem, cfg)
                rgb_bright = compute_frame_brightness(rgb_path)
                ir_bright = compute_frame_brightness(ir_path)
            row = process_frame_level(stem, d, cfg, rgb_bright, ir_bright)
            all_rows.append(row)
        elif args.calibration == "distance":
            rows, n_missed = process_detection_distance(stem, d, cfg, ir_img_cache)
            all_rows.extend(rows)
            total_missed_gt += n_missed
        else:
            rows, n_missed = process_detection_level(
                stem, d, cfg, ir_img_cache, seq_offsets, global_offset)
            all_rows.extend(rows)
            total_missed_gt += n_missed

        if (idx + 1) % 5000 == 0 or (idx + 1) == len(detections):
            elapsed = time.time() - t_start
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx + 1}/{len(detections)}] {fps:.0f} frames/s, "
                  f"{len(all_rows)} rows so far")

    elapsed = time.time() - t_start
    print(f"  Processed in {elapsed:.1f}s ({len(detections) / elapsed:.0f} frames/s)")

    # --- Save CSV ---
    if not all_rows:
        print("WARNING: No candidate rows generated.")
        return

    csv_path = out_dir / "fusion_dataset.csv"
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Summary ---
    n_pos = sum(1 for r in all_rows if r["label"] == 1)
    n_neg = len(all_rows) - n_pos
    print(f"\nDone. Saved {len(all_rows)} rows to {csv_path}")
    print(f"  Mode: {mode}-level | Positive: {n_pos} | Negative: {n_neg}")

    if mode == "frame":
        pos_rows = [r for r in all_rows if r["label"] == 1]
        if pos_rows:
            rgb_det = sum(1 for r in pos_rows if r["max_conf_rgb"] > 0.25)
            ir_det = sum(1 for r in pos_rows if r["max_conf_ir"] > 0.25)
            both = sum(1 for r in pos_rows
                       if r["max_conf_rgb"] > 0.25 and r["max_conf_ir"] > 0.25)
            neither = sum(1 for r in pos_rows
                          if r["max_conf_rgb"] <= 0.25 and r["max_conf_ir"] <= 0.25)
            n = len(pos_rows)
            print(f"\n  Detection on positive frames (conf > 0.25):")
            print(f"    RGB:     {rgb_det}/{n} ({rgb_det/n*100:.1f}%)")
            print(f"    IR:      {ir_det}/{n} ({ir_det/n*100:.1f}%)")
            print(f"    Both:    {both}/{n} ({both/n*100:.1f}%)")
            print(f"    Neither: {neither}/{n} ({neither/n*100:.1f}%)")
    else:
        print(f"  Missed GT (irrecoverable FN): {total_missed_gt}")
        if n_pos + total_missed_gt > 0:
            print(f"  OR-union recall ceiling: "
                  f"{n_pos / (n_pos + total_missed_gt) * 100:.2f}%")
        for src in ["both", "rgb_only", "ir_only"]:
            src_rows = [r for r in all_rows if r.get("source") == src]
            s_tp = sum(1 for r in src_rows if r["label"] == 1)
            s_total = len(src_rows)
            if s_total:
                print(f"    {src:<10} {s_total:>7} total, {s_tp:>7} TP "
                      f"({s_tp/s_total*100:.1f}%)")

        # Time-of-day breakdown
        if cfg.get("use_time_features", True):
            print(f"\n  Time-of-day breakdown:")
            for period in ["day", "dusk_dawn", "night", "unknown"]:
                p_rows = [r for r in all_rows if r.get("time_of_day") == period]
                if not p_rows:
                    continue
                p_tp = sum(1 for r in p_rows if r["label"] == 1)
                p_total = len(p_rows)
                print(f"    {period:<10} {p_total:>7} total, {p_tp:>7} TP "
                      f"({p_tp/p_total*100:.1f}%)")

        # IR contrast breakdown
        if cfg.get("use_ir_contrast", False):
            ir_contrast_rows = [r for r in all_rows
                                if r.get("ir_local_contrast") is not None
                                and r.get("ir_local_contrast", 0) > 0]
            if ir_contrast_rows:
                contrasts = [r["ir_local_contrast"] for r in ir_contrast_rows]
                print(f"\n  IR local contrast stats (non-zero):")
                print(f"    Mean: {np.mean(contrasts):.3f}")
                print(f"    Median: {np.median(contrasts):.3f}")
                tp_contrasts = [r["ir_local_contrast"] for r in ir_contrast_rows
                                if r["label"] == 1]
                fp_contrasts = [r["ir_local_contrast"] for r in ir_contrast_rows
                                if r["label"] == 0]
                if tp_contrasts:
                    print(f"    TP mean contrast: {np.mean(tp_contrasts):.3f}")
                if fp_contrasts:
                    print(f"    FP mean contrast: {np.mean(fp_contrasts):.3f}")


if __name__ == "__main__":
    main()
