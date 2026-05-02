"""
generate_fusion_data.py — Generate fusion classifier training data.

Auto-detects whether RGB/IR ground truth is spatially aligned:
  - If aligned (IoU > threshold): detection-level fusion (per-bbox features)
  - If misaligned: frame-level fusion (per-frame features)

NOTE: For large datasets, prefer the two-phase approach:
  Phase 1: python run_inference.py       (slow, run once)
  Phase 2: python build_dataset.py       (fast, re-runnable)

Usage:
    python generate_fusion_data.py
    python generate_fusion_data.py --mode frame    # force frame-level
    python generate_fusion_data.py --mode detection # force detection-level
    python generate_fusion_data.py --resume
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import yaml

from utils import (
    align_detections,
    box_area,
    compute_iou,
    extract_features,
    label_candidates,
    parse_yolo_labels,
)


# ---------------------------------------------------------------------------
# Config / discovery
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_paired_frames(dataset_root, rgb_subdir, ir_subdir,
                           rgb_suffix, ir_suffix):
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
                s = f.stem
                if suffix:
                    s = s.replace(suffix, "")
                out[s] = f
        return out

    rgb_map = stem_map(rgb_img_dir, rgb_suffix)
    ir_map = stem_map(ir_img_dir, ir_suffix)
    shared = sorted(set(rgb_map) & set(ir_map))
    print(f"  Found {len(rgb_map)} RGB, {len(ir_map)} IR, {len(shared)} paired frames")

    pairs = []
    for stem in shared:
        rgb_img = rgb_map[stem]
        ir_img = ir_map[stem]
        pairs.append({
            "stem": stem,
            "rgb_img": rgb_img,
            "ir_img": ir_img,
            "rgb_lbl": rgb_lbl_dir / (rgb_img.stem + ".txt"),
            "ir_lbl": ir_lbl_dir / (ir_img.stem + ".txt"),
        })
    return pairs


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def _parse_norm(label_path):
    """Parse YOLO label file → normalized (x1,y1,x2,y2) boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def check_alignment(pairs, sample_size=500, iou_threshold=0.3,
                    aligned_fraction_needed=0.5):
    """
    Sample paired frames and check if RGB & IR GT boxes overlap in
    normalized coordinates. Returns ('detection', stats) or ('frame', stats).
    """
    # Only sample frames where both modalities have GT
    candidates = []
    for p in pairs:
        rgb_boxes = _parse_norm(p["rgb_lbl"])
        ir_boxes = _parse_norm(p["ir_lbl"])
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

    import numpy as np
    ious = np.array(ious)
    frac_above = float((ious >= iou_threshold).mean())
    mean_iou = float(ious.mean())
    median_iou = float(np.median(ious))

    stats = {
        "sample_size": len(sample),
        "gt_pairs_compared": len(ious),
        "mean_iou": round(mean_iou, 4),
        "median_iou": round(median_iou, 4),
        "fraction_above_threshold": round(frac_above, 4),
        "threshold_used": iou_threshold,
    }

    if frac_above >= aligned_fraction_needed:
        return "detection", stats
    else:
        return "frame", stats


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARN] Corrupt checkpoint ({e}), starting fresh.")
    return {"processed_stems": [], "rows": []}


def save_checkpoint(path, state):
    """Atomic write: write to .tmp then rename to prevent corruption."""
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    if os.path.exists(str(path)):
        os.remove(str(path))
    os.rename(tmp_path, str(path))


# ---------------------------------------------------------------------------
# YOLO inference helpers
# ---------------------------------------------------------------------------

def has_gt(label_path):
    if not label_path.exists():
        return False
    text = label_path.read_text().strip()
    return any(len(line.split()) >= 5 for line in text.split("\n") if line)


def run_inference(model, img_path, cfg):
    """Run YOLO, return list of (box, conf) and image dims."""
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
            dets.append((tuple(float(v) for v in xyxy[i]), float(confs[i])))
    return dets, img_w, img_h


# ---------------------------------------------------------------------------
# Per-frame GT calibration
# ---------------------------------------------------------------------------


def compute_per_frame_offset(gt_rgb_norm, gt_ir_norm):
    """
    Given GT boxes in normalized coords for both modalities (same frame),
    compute the per-frame offset: delta_cx, delta_cy, scale_w, scale_h.

    Uses the first GT box pair (single-drone assumption).
    Returns (dcx, dcy, sw, sh) or None if can't compute.
    """
    if not gt_rgb_norm or not gt_ir_norm:
        return None
    r = gt_rgb_norm[0]  # (cx, cy, w, h)
    i = gt_ir_norm[0]
    if i[2] <= 0 or i[3] <= 0:
        return None
    dcx = r[0] - i[0]  # shift in normalized cx
    dcy = r[1] - i[1]  # shift in normalized cy
    sw = r[2] / i[2]   # width scale ratio
    sh = r[3] / i[3]   # height scale ratio
    return (dcx, dcy, sw, sh)


def calibrate_ir_box(box_xyxy, offset, ir_w, ir_h, rgb_w, rgb_h):
    """Apply per-frame offset to transform an IR detection box into RGB pixel space."""
    dcx, dcy, sw, sh = offset
    x1, y1, x2, y2 = box_xyxy
    # IR pixel -> IR normalized
    cx_n = ((x1 + x2) / 2) / ir_w
    cy_n = ((y1 + y2) / 2) / ir_h
    w_n = (x2 - x1) / ir_w
    h_n = (y2 - y1) / ir_h
    # Apply offset
    new_cx = cx_n + dcx
    new_cy = cy_n + dcy
    new_w = w_n * sw
    new_h = h_n * sh
    # RGB normalized -> RGB pixel
    rx1 = (new_cx - new_w / 2) * rgb_w
    ry1 = (new_cy - new_h / 2) * rgb_h
    rx2 = (new_cx + new_w / 2) * rgb_w
    ry2 = (new_cy + new_h / 2) * rgb_h
    return (rx1, ry1, rx2, ry2)


def parse_gt_norm(label_path):
    """Parse YOLO label -> list of (cx, cy, w, h) normalized."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(tuple(map(float, parts[1:5])))
    return boxes


# ---------------------------------------------------------------------------
# Frame-level processing
# ---------------------------------------------------------------------------

def process_frame_level(pair, rgb_model, ir_model, cfg):
    """One row per frame: max confidence + detection count per model."""
    rgb_dets, _, _ = run_inference(rgb_model, pair["rgb_img"], cfg)
    ir_dets, _, _ = run_inference(ir_model, pair["ir_img"], cfg)

    rgb_confs = [c for _, c in rgb_dets]
    ir_confs = [c for _, c in ir_dets]
    gt_present = 1 if (has_gt(pair["rgb_lbl"]) or has_gt(pair["ir_lbl"])) else 0

    return {
        "stem": pair["stem"],
        "max_conf_rgb": max(rgb_confs) if rgb_confs else 0.0,
        "max_conf_ir": max(ir_confs) if ir_confs else 0.0,
        "n_dets_rgb": len(rgb_confs),
        "n_dets_ir": len(ir_confs),
        "label": gt_present,
    }


# ---------------------------------------------------------------------------
# Detection-level processing
# ---------------------------------------------------------------------------

def process_detection_level(pair, rgb_model, ir_model, cfg):
    """
    Per-frame GT-calibrated detection-level fusion.

    Uses GT boxes from both modalities to compute the exact per-frame offset,
    then transforms IR detections into RGB pixel space for alignment.
    All candidates are matched against RGB GT (now in the same coord space).
    """
    rgb_dets, img_w, img_h = run_inference(rgb_model, pair["rgb_img"], cfg)
    ir_dets, ir_w, ir_h = run_inference(ir_model, pair["ir_img"], cfg)

    gt_rgb_norm = parse_gt_norm(pair["rgb_lbl"])
    gt_ir_norm = parse_gt_norm(pair["ir_lbl"])

    # Compute per-frame offset from GT correspondence
    offset = compute_per_frame_offset(gt_rgb_norm, gt_ir_norm)

    # Calibrate IR detections into RGB pixel space
    if offset:
        ir_dets_calibrated = [
            (calibrate_ir_box(box, offset, ir_w, ir_h, img_w, img_h), conf)
            for box, conf in ir_dets
        ]
    else:
        # No offset available (only one modality has GT) — keep as-is
        ir_dets_calibrated = ir_dets

    # Align calibrated IR detections with RGB detections
    matched, rgb_only, ir_only = align_detections(
        rgb_dets, ir_dets_calibrated, iou_thresh=cfg["alignment_iou"]
    )
    sources = (["both"] * len(matched) +
               ["rgb_only"] * len(rgb_only) +
               ["ir_only"] * len(ir_only))
    candidates = matched + rgb_only + ir_only

    # GT in RGB pixel space for labeling
    gt_rgb_px = parse_yolo_labels(pair["rgb_lbl"], img_w, img_h)

    if not candidates:
        return [], len(gt_rgb_px)

    labels, n_missed = label_candidates(candidates, gt_rgb_px,
                                        matching_iou=cfg["matching_iou"])

    img_area = img_w * img_h
    n_total = len(candidates)

    rows = []
    for cand, label, source in zip(candidates, labels, sources):
        feats = extract_features(cand, img_area, n_dets_total=n_total)
        rows.append({
            "stem": pair["stem"],
            "source": source,
            "conf_rgb": cand.get("rgb_conf", 0.0),
            "conf_ir": cand.get("ir_conf", 0.0),
            **feats,
            "label": label,
        })
    return rows, n_missed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["auto", "frame", "detection"],
                        default="auto", help="Fusion level (default: auto-detect)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if not pairs:
        print("ERROR: No paired frames found.")
        return

    # --- Decide fusion mode ---
    if args.mode == "auto":
        print("Checking GT alignment (sampling up to 500 frames)...")
        mode, stats = check_alignment(pairs)
        print(f"  Alignment stats: {json.dumps(stats, indent=2)}")
        print(f"  → Auto-selected mode: {mode.upper()}-level fusion")
    else:
        mode = args.mode
        stats = {"forced": True}
        print(f"  → Forced mode: {mode.upper()}-level fusion")

    # Save mode choice for downstream scripts
    meta = {"mode": mode, "alignment_stats": stats}
    with open(out_dir / "fusion_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # --- Checkpoint ---
    ckpt_path = out_dir / "generate_checkpoint.json"
    if args.resume:
        ckpt = load_checkpoint(ckpt_path)
        done_stems = set(ckpt["processed_stems"])
        all_rows = ckpt["rows"]
        print(f"  Resuming: {len(done_stems)} already done")
    else:
        done_stems = set()
        all_rows = []

    total_missed_gt = 0
    t_start = time.time()
    remaining = [p for p in pairs if p["stem"] not in done_stems]

    for idx, pair in enumerate(remaining):
        if mode == "frame":
            row = process_frame_level(pair, rgb_model, ir_model, cfg)
            all_rows.append(row)
        else:
            rows, n_missed = process_detection_level(pair, rgb_model, ir_model, cfg)
            all_rows.extend(rows)
            total_missed_gt += n_missed

        done_stems.add(pair["stem"])
        processed = idx + 1
        if processed % 50 == 0 or processed == len(remaining):
            elapsed = time.time() - t_start
            fps = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / fps if fps > 0 else 0
            print(f"  [{processed}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta / 60:.1f}min, {len(all_rows)} rows")
            save_checkpoint(ckpt_path, {
                "processed_stems": list(done_stems),
                "rows": all_rows,
            })

    # --- Save CSV ---
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

    if ckpt_path.exists():
        ckpt_path.unlink()


if __name__ == "__main__":
    main()
