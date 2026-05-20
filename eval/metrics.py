"""
metrics.py — Shared detection matching and metric computation.

Provides IoU/IoP matching, per-detection TP/FP/FN scoring,
precision/recall/F1 computation, and size-bucketed statistics.

Used by eval_pipeline.py and eval_model.py.
"""

from __future__ import annotations
import numpy as np


# ── Box geometry ──────────────────────────────────────────────────

def iou_iop(a: tuple, b: tuple) -> tuple[float, float]:
    """Compute (IoU, IoP) between two (x1,y1,x2,y2) boxes.
    IoP = intersection / prediction(a) area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0, 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    iop = inter / area_a if area_a > 0 else 0.0
    return iou, iop


def box_area_fraction(box: tuple, img_w: int, img_h: int) -> float:
    """Box area as fraction of image area."""
    bw = max(0, box[2] - box[0])
    bh = max(0, box[3] - box[1])
    img_area = img_w * img_h
    return (bw * bh) / img_area if img_area > 0 else 0.0


# ── Detection matching ───────────────────────────────────────────

def score_detections(
    dets: list[tuple[tuple, float]],
    gts: list[tuple],
    rule: str = "iou",
    iou_thr: float = 0.5,
    iop_thr: float = 0.5,
) -> tuple[int, int, int]:
    """Match detections to ground truths.

    Args:
        dets: list of ((x1,y1,x2,y2), confidence)
        gts: list of (x1,y1,x2,y2) ground-truth boxes
        rule: "iou" or "iop"
        iou_thr: IoU threshold for match
        iop_thr: IoP threshold for match

    Returns:
        (tp, fp, fn) counts
    """
    tp = fp = 0
    matched_gt: set[int] = set()
    for d_box, _conf in dets:
        best_idx, best_score = -1, 0.0
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(d_box, g)
            s = iu if rule == "iou" else ip
            if s > best_score:
                best_score, best_idx = s, gi
        thr = iou_thr if rule == "iou" else iop_thr
        if best_score >= thr and best_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def score_detections_detailed(
    dets: list[tuple[tuple, float]],
    gts: list[tuple],
    iou_thr: float = 0.5,
    iop_thr: float = 0.5,
) -> list[dict]:
    """Per-detection scoring returning match details for both IoU and IoP.

    Returns list of dicts per detection:
        {conf, box, best_iou, matched_iou, best_iop, matched_iop}
    """
    used_iou: set[int] = set()
    used_iop: set[int] = set()
    results = []
    for d_box, conf in dets:
        best_iu, best_ip = 0.0, 0.0
        bi_u, bi_p = -1, -1
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(d_box, g)
            if iu > best_iu:
                best_iu, bi_u = iu, gi
            if ip > best_ip:
                best_ip, bi_p = ip, gi
        m_iou = int(best_iu >= iou_thr and bi_u not in used_iou)
        m_iop = int(best_ip >= iop_thr and bi_p not in used_iop)
        if m_iou:
            used_iou.add(bi_u)
        if m_iop:
            used_iop.add(bi_p)
        results.append({
            "conf": round(conf, 4),
            "box": d_box,
            "best_iou": round(best_iu, 4),
            "matched_iou": m_iou,
            "best_iop": round(best_ip, 4),
            "matched_iop": m_iop,
        })
    return results


# ── Aggregate metrics ────────────────────────────────────────────

def compute_prf(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, F1 from TP/FP/FN."""
    tn = 0  # detection-level eval has no TN concept, but we include for API
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def compute_frame_metrics(
    has_det: bool, has_gt: bool
) -> tuple[int, int, int, int]:
    """Frame-level TP/FP/FN/TN (binary: any-detection vs any-GT).

    Returns (tp, fp, fn, tn) as 0/1 for a single frame.
    """
    tp = int(has_det and has_gt)
    fp = int(has_det and not has_gt)
    fn = int(not has_det and has_gt)
    tn = int(not has_det and not has_gt)
    return tp, fp, fn, tn


# ── Size distribution ────────────────────────────────────────────

SIZE_BUCKETS = {
    "small": (0.0, 0.001),
    "medium": (0.001, 0.01),
    "large": (0.01, 1.0),
}


def classify_size(box: tuple, img_w: int, img_h: int) -> str:
    """Classify detection into small/medium/large by area fraction."""
    frac = box_area_fraction(box, img_w, img_h)
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= frac < hi:
            return name
    return "large"


def size_distribution(
    dets: list[tuple[tuple, float]], img_w: int, img_h: int
) -> dict[str, int]:
    """Count detections per size bucket."""
    dist = {k: 0 for k in SIZE_BUCKETS}
    for box, _conf in dets:
        dist[classify_size(box, img_w, img_h)] += 1
    return dist


def score_per_size(
    dets: list[tuple[tuple, float]],
    gts: list[tuple],
    img_w: int,
    img_h: int,
    iou_thr: float = 0.5,
    iop_thr: float = 0.5,
) -> dict:
    """Attribute TP/FP/FN to size buckets.

    TP and FN are bucketed by GT box size (the "true" target size).
    FP are bucketed by the predicted box size (no matching GT).

    Returns:
        {
          "iou": {"small": {tp,fp,fn}, "medium": {...}, "large": {...}},
          "iop": {...},
        }
    """
    out = {rule: {b: {"tp": 0, "fp": 0, "fn": 0} for b in SIZE_BUCKETS}
           for rule in ("iou", "iop")}
    for rule in ("iou", "iop"):
        thr = iou_thr if rule == "iou" else iop_thr
        matched_gt: set[int] = set()
        for d_box, _conf in dets:
            best_idx, best_score = -1, 0.0
            for gi, g in enumerate(gts):
                iu, ip = iou_iop(d_box, g)
                s = iu if rule == "iou" else ip
                if s > best_score:
                    best_score, best_idx = s, gi
            if best_score >= thr and best_idx not in matched_gt:
                gb = gts[best_idx]
                out[rule][classify_size(gb, img_w, img_h)]["tp"] += 1
                matched_gt.add(best_idx)
            else:
                out[rule][classify_size(d_box, img_w, img_h)]["fp"] += 1
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                out[rule][classify_size(g, img_w, img_h)]["fn"] += 1
    return out


# ── Trust-aware scoring (the only scoring rule used in the
#    full_pipeline_ablations docs) ─────────────────────────────────
#
# Per check.txt's "Rule A" (the email rule):
#   - label=0 (reject_both): kept=[] on both sides; both GTs become FN.
#   - label=1 (trust RGB):   score RGB dets vs RGB GT; IR GT *excluded* (the
#                            system explicitly chose to not look at IR).
#   - label=2 (trust IR):    score IR dets vs IR GT; RGB GT *excluded*.
#   - label=3 (trust both):  RGB dets vs RGB GT + IR dets vs IR GT (TPs sum
#                            across modalities, hence absolute counts can
#                            exceed any single-modality run).
#
# For RGB-only datasets (no separate IR GT), `ir_dets` share the RGB coord
# frame (e.g. ir_grayscale on RGB-only video). Trust-aware then collapses to:
#   label=0: kept=[]; label=1: rgb_dets; label=2: ir_dets;
#   label=3: rgb_dets + ir_dets   (all scored vs the single RGB GT)

def score_trust_aware(
    label: int,
    rgb_dets: list, ir_dets: list,
    gts: list, ir_gts: list,
    w: int, h: int, iw: int, ih: int,
    is_paired: bool,
    rule: str = "iop",
    iou_thr: float = 0.5, iop_thr: float = 0.5,
) -> dict:
    """Return per-size {bucket: {tp, fp, fn}} under trust-aware scoring.

    Sums RGB-side and IR-side scores when both modalities are trusted (label=3).
    For RGB-only datasets, treats ir_dets as additional RGB-coord-frame
    candidate detections.
    """
    out = {b: {"tp": 0, "fp": 0, "fn": 0} for b in SIZE_BUCKETS}

    def _add(s):
        for b in s:
            out[b]["tp"] += s[b]["tp"]
            out[b]["fp"] += s[b]["fp"]
            out[b]["fn"] += s[b]["fn"]

    if is_paired:
        # RGB side: score if label ∈ {0, 1, 3}; label=2 ignores RGB GT entirely
        if label in (0, 1, 3):
            kept_rgb = rgb_dets if label in (1, 3) else []
            s = score_per_size(kept_rgb, gts, w, h,
                               iou_thr=iou_thr, iop_thr=iop_thr)[rule]
            _add(s)
        # IR side: score if label ∈ {0, 2, 3}; label=1 ignores IR GT entirely
        if label in (0, 2, 3):
            kept_ir = ir_dets if label in (2, 3) else []
            s = score_per_size(kept_ir, ir_gts, iw, ih,
                               iou_thr=iou_thr, iop_thr=iop_thr)[rule]
            _add(s)
    else:
        # RGB-only dataset: single GT side
        if label == 0:
            kept = []
        elif label == 1:
            kept = rgb_dets
        elif label == 2:
            kept = ir_dets
        else:  # label == 3
            kept = rgb_dets + ir_dets
        s = score_per_size(kept, gts, w, h,
                           iou_thr=iou_thr, iop_thr=iop_thr)[rule]
        _add(s)
    return out


# ── PR curve helpers ─────────────────────────────────────────────

def pr_sweep(
    records: list[tuple[float, int]],
    total_gt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep confidence threshold for PR curve.

    Args:
        records: list of (confidence, is_tp) sorted by descending conf
        total_gt: total ground-truth count

    Returns:
        (precision_array, recall_array, threshold_array)
    """
    records = sorted(records, key=lambda x: -x[0])
    tp = fp = 0
    precs, recs, threshs = [], [], []
    for conf, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / total_gt if total_gt > 0 else 0.0
        precs.append(p)
        recs.append(r)
        threshs.append(conf)
    return np.array(precs), np.array(recs), np.array(threshs)
