"""
Shared utilities for the fusion classifier.

Provides: IoU computation, cross-modality detection alignment,
feature extraction, and GT label matching.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_area(box):
    """Area of an (x1, y1, x2, y2) box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def compute_iou(box_a, box_b):
    """IoU between two (x1, y1, x2, y2) boxes."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Cross-modality detection alignment
# ---------------------------------------------------------------------------

def align_detections_by_distance(rgb_dets, ir_dets, rgb_wh, ir_wh, max_dist=0.15):
    """
    Match RGB and IR detections by center distance in normalized coords.

    No calibration needed — each detection is normalized by its own image size,
    then matched by proximity of centers.

    Parameters
    ----------
    rgb_dets : list of (box_px, conf)
    ir_dets  : list of (box_px, conf)
    rgb_wh   : (width, height) of RGB image
    ir_wh    : (width, height) of IR image
    max_dist : maximum Euclidean distance in normalized [0,1] space

    Returns
    -------
    matched, rgb_only, ir_only — same format as align_detections
    """
    if not rgb_dets and not ir_dets:
        return [], [], []

    def center_norm(box, wh):
        cx = (box[0] + box[2]) / 2 / wh[0]
        cy = (box[1] + box[3]) / 2 / wh[1]
        return cx, cy

    # Build distance pairs, sort by combined confidence (prefer high-conf matches)
    pairs = []
    for ri, (rb, rc) in enumerate(rgb_dets):
        rcx, rcy = center_norm(rb, rgb_wh)
        for ii, (ib, ic) in enumerate(ir_dets):
            icx, icy = center_norm(ib, ir_wh)
            dist = ((rcx - icx) ** 2 + (rcy - icy) ** 2) ** 0.5
            if dist <= max_dist:
                combined_conf = rc + ic
                pairs.append((combined_conf, -dist, ri, ii))
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    matched_rgb = set()
    matched_ir = set()
    matched = []

    for combined_conf, neg_dist, ri, ii in pairs:
        if ri in matched_rgb or ii in matched_ir:
            continue
        rb, rc = rgb_dets[ri]
        ib, ic = ir_dets[ii]
        # Store IR box scaled to RGB pixel space for downstream compatibility
        ir_scaled = (
            ib[0] * rgb_wh[0] / ir_wh[0], ib[1] * rgb_wh[1] / ir_wh[1],
            ib[2] * rgb_wh[0] / ir_wh[0], ib[3] * rgb_wh[1] / ir_wh[1],
        )
        matched.append({
            "rgb_box": rb, "rgb_conf": rc,
            "ir_box": ir_scaled, "ir_conf": ic,
            "dist": -neg_dist,
        })
        matched_rgb.add(ri)
        matched_ir.add(ii)

    rgb_only = [{"rgb_box": b, "rgb_conf": c}
                for i, (b, c) in enumerate(rgb_dets) if i not in matched_rgb]
    # Scale ir_only boxes to RGB pixel space too
    ir_only = [{"ir_box": (b[0] * rgb_wh[0] / ir_wh[0], b[1] * rgb_wh[1] / ir_wh[1],
                           b[2] * rgb_wh[0] / ir_wh[0], b[3] * rgb_wh[1] / ir_wh[1]),
                "ir_conf": c}
               for i, (b, c) in enumerate(ir_dets) if i not in matched_ir]

    return matched, rgb_only, ir_only


def align_detections(rgb_dets, ir_dets, iou_thresh=0.3):
    """
    Match RGB detections to IR detections by IoU (greedy, highest-first).

    Parameters
    ----------
    rgb_dets : list of (box, conf)  — box is (x1, y1, x2, y2)
    ir_dets  : list of (box, conf)
    iou_thresh : float

    Returns
    -------
    matched   : list of dict with keys rgb_box, rgb_conf, ir_box, ir_conf, iou
    rgb_only  : list of dict with keys rgb_box, rgb_conf
    ir_only   : list of dict with keys ir_box, ir_conf
    """
    if not rgb_dets and not ir_dets:
        return [], [], []

    # Build IoU matrix and sort by combined confidence (break ties by IoU)
    # This ensures high-confidence detections are matched first, preventing
    # low-confidence duplicates from stealing matches.
    pairs = []
    for ri, (rb, rc) in enumerate(rgb_dets):
        for ii, (ib, ic) in enumerate(ir_dets):
            iou = compute_iou(rb, ib)
            if iou >= iou_thresh:
                combined_conf = rc + ic
                pairs.append((combined_conf, iou, ri, ii))
    pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    matched_rgb = set()
    matched_ir = set()
    matched = []

    for combined_conf, iou, ri, ii in pairs:
        if ri in matched_rgb or ii in matched_ir:
            continue
        rb, rc = rgb_dets[ri]
        ib, ic = ir_dets[ii]
        matched.append({
            "rgb_box": rb, "rgb_conf": rc,
            "ir_box": ib, "ir_conf": ic,
            "iou": iou,
        })
        matched_rgb.add(ri)
        matched_ir.add(ii)

    rgb_only = [{"rgb_box": b, "rgb_conf": c}
                for i, (b, c) in enumerate(rgb_dets) if i not in matched_rgb]
    ir_only = [{"ir_box": b, "ir_conf": c, "ir_raw_idx": i}
               for i, (b, c) in enumerate(ir_dets) if i not in matched_ir]

    return matched, rgb_only, ir_only


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def representative_box(candidate):
    """Return a single representative (x1,y1,x2,y2) for a candidate."""
    if "rgb_box" in candidate and "ir_box" in candidate:
        rb = candidate["rgb_box"]
        ib = candidate["ir_box"]
        return tuple((a + b) / 2 for a, b in zip(rb, ib))
    return candidate.get("rgb_box") or candidate.get("ir_box")


def extract_features(candidate, img_area, n_dets_total=0, frame_brightness=None,
                     use_phase2=False):
    """
    Extract feature vector from a candidate detection.

    Phase 1 (always): conf_max, conf_min, agreement, bbox_area_norm
    Phase 2 (opt):    frame_brightness, n_dets_total, aspect_ratio, conf_delta

    Returns dict of feature name → value.
    """
    conf_rgb = candidate.get("rgb_conf", 0.0)
    conf_ir = candidate.get("ir_conf", 0.0)
    agreement = 1.0 if ("rgb_box" in candidate and "ir_box" in candidate) else 0.0

    box = representative_box(candidate)
    area = box_area(box)
    area_norm = area / img_area if img_area > 0 else 0.0

    feats = {
        "conf_max": max(conf_rgb, conf_ir),
        "conf_min": min(conf_rgb, conf_ir),
        "agreement": agreement,
        "bbox_area_norm": area_norm,
    }

    if use_phase2:
        w = box[2] - box[0]
        h = box[3] - box[1]
        feats["aspect_ratio"] = w / h if h > 0 else 1.0
        feats["conf_delta"] = abs(conf_rgb - conf_ir)
        feats["n_dets_total"] = float(n_dets_total)
        if frame_brightness is not None:
            feats["frame_brightness"] = frame_brightness

    return feats


# ---------------------------------------------------------------------------
# GT matching (label candidates as TP / FP)
# ---------------------------------------------------------------------------

def label_candidates(candidates, gt_boxes, matching_iou=0.5):
    """
    Label each candidate detection as 1 (TP) or 0 (FP) via greedy GT matching.

    Parameters
    ----------
    candidates : list of dicts (output of align_detections, flattened)
    gt_boxes   : list of (x1, y1, x2, y2)
    matching_iou : float

    Returns
    -------
    labels : list of int (1=TP, 0=FP), same length as candidates
    n_missed_gt : int — GT boxes not matched by any candidate (irrecoverable FN)
    """
    if not candidates:
        return [], len(gt_boxes)

    # Score each candidate by its max confidence for priority ordering
    def _score(c):
        return max(c.get("rgb_conf", 0.0), c.get("ir_conf", 0.0))

    order = sorted(range(len(candidates)), key=lambda i: _score(candidates[i]),
                   reverse=True)

    labels = [0] * len(candidates)
    matched_gt = set()

    for ci in order:
        box = representative_box(candidates[ci])
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= matching_iou and best_gi >= 0:
            labels[ci] = 1
            matched_gt.add(best_gi)

    n_missed_gt = len(gt_boxes) - len(matched_gt)
    return labels, n_missed_gt


# ---------------------------------------------------------------------------
# YOLO label file parsing
# ---------------------------------------------------------------------------

def parse_yolo_labels(label_path, img_w, img_h):
    """
    Parse a YOLO label .txt file into pixel-coordinate boxes.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
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
