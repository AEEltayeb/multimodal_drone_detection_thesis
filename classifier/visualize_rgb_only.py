"""
visualize_rgb_only.py — Show RGB-only TP cases side by side (RGB + IR)
with GT boxes and model predictions.

Picks 20 diverse frames (spread across sequences) where RGB detected
a drone but IR didn't, and the detection was a true positive.

Usage:
    python visualize_rgb_only.py
"""

import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO

from utils import parse_yolo_labels


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_seq(stem):
    m = re.match(r"(.+)_f\d+$", stem)
    return m.group(1) if m else stem


def draw_boxes(img, boxes, color, label_text, thickness=2):
    """Draw bounding boxes on image."""
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label_text:
            conf_str = ""
            if len(box) > 4:
                conf_str = f" {box[4]:.2f}"
            text = f"{label_text}{conf_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    cfg = load_config()
    root = Path(cfg["dataset_root"])
    rgb_img_dir = root / cfg["rgb_subdir"] / "images"
    ir_img_dir = root / cfg["ir_subdir"] / "images"
    rgb_lbl_dir = root / cfg["rgb_subdir"] / "labels"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"
    rgb_suffix = cfg["rgb_stem_suffix"]
    ir_suffix = cfg["ir_stem_suffix"]

    RGB_W, RGB_H = 1920, 1080
    IR_W, IR_H = 640, 512

    # Load fusion data
    csv_path = Path("runs/fusion_dataset.csv")
    df = pd.read_csv(csv_path)

    # Filter to rgb_only TPs that are genuinely rgb_only
    # (no "both" TP on the same frame — those would mean calibration matched fine)
    rgb_tp_all = df[(df["source"] == "rgb_only") & (df["label"] == 1)].copy()
    both_tp_stems = set(df[(df["source"] == "both") & (df["label"] == 1)]["stem"])
    rgb_tp = rgb_tp_all[~rgb_tp_all["stem"].isin(both_tp_stems)].copy()
    rgb_tp["seq"] = rgb_tp["stem"].apply(extract_seq)
    print(f"RGB-only TPs (genuine, no both-TP on frame): {len(rgb_tp)} "
          f"from {rgb_tp['seq'].nunique()} sequences")
    print(f"  (filtered out {len(rgb_tp_all) - len(rgb_tp)} that had both-TP on same frame)")

    # Sample 20 diverse frames — spread across sequences
    random.seed(42)
    seqs = rgb_tp["seq"].unique().tolist()
    random.shuffle(seqs)

    selected_stems = []
    # Round-robin from each sequence
    seq_frames = {s: rgb_tp[rgb_tp["seq"] == s]["stem"].tolist() for s in seqs}
    for s in seq_frames:
        random.shuffle(seq_frames[s])

    while len(selected_stems) < 20:
        added = False
        for s in seqs:
            if seq_frames[s] and len(selected_stems) < 20:
                selected_stems.append(seq_frames[s].pop(0))
                added = True
        if not added:
            break

    print(f"Selected {len(selected_stems)} frames for visualization")

    # Load models
    print(f"Loading RGB model: {cfg['rgb_weights']}")
    rgb_model = YOLO(cfg["rgb_weights"])
    print(f"Loading IR model: {cfg['ir_weights']}")
    ir_model = YOLO(cfg["ir_weights"])

    conf_thresh = 0.25  # reasonable visual threshold

    out_dir = Path("runs/rgb_only_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    def stem_to_filename(stem, suffix):
        """Insert suffix before _fNNNNNN: stem '..._1_f000020' + '_visible' -> '..._1_visible_f000020'."""
        m = re.match(r"(.+)(_f\d+)$", stem)
        if m:
            return m.group(1) + suffix + m.group(2) + ".jpg"
        return stem + suffix + ".jpg"

    for idx, stem in enumerate(selected_stems):
        rgb_img_name = stem_to_filename(stem, rgb_suffix)
        ir_img_name = stem_to_filename(stem, ir_suffix)
        rgb_img_path = rgb_img_dir / rgb_img_name
        ir_img_path = ir_img_dir / ir_img_name

        if not rgb_img_path.exists() or not ir_img_path.exists():
            print(f"  [{idx+1}] SKIP {stem} — image not found")
            continue

        # Load images
        rgb_img = cv2.imread(str(rgb_img_path))
        ir_img = cv2.imread(str(ir_img_path))
        if rgb_img is None or ir_img is None:
            print(f"  [{idx+1}] SKIP {stem} — failed to read")
            continue

        # Parse GT labels
        rgb_lbl_name = stem_to_filename(stem, rgb_suffix).replace(".jpg", ".txt")
        ir_lbl_name = stem_to_filename(stem, ir_suffix).replace(".jpg", ".txt")
        rgb_lbl_path = rgb_lbl_dir / rgb_lbl_name
        ir_lbl_path = ir_lbl_dir / ir_lbl_name
        rgb_gt = parse_yolo_labels(rgb_lbl_path, RGB_W, RGB_H)
        ir_gt = parse_yolo_labels(ir_lbl_path, IR_W, IR_H)

        # Run models
        rgb_results = rgb_model(str(rgb_img_path), imgsz=cfg["imgsz"],
                                conf=conf_thresh, iou=cfg["iou_nms"],
                                max_det=cfg["max_det"], verbose=False)
        ir_results = ir_model(str(ir_img_path), imgsz=cfg["imgsz"],
                              conf=conf_thresh, iou=cfg["iou_nms"],
                              max_det=cfg["max_det"], verbose=False)

        # Extract predictions
        rgb_preds = []
        for r in rgb_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                rgb_preds.append((*xyxy, conf))

        ir_preds = []
        for r in ir_results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                ir_preds.append((*xyxy, conf))

        # Draw on RGB image
        rgb_vis = rgb_img.copy()
        draw_boxes(rgb_vis, [(x1, y1, x2, y2) for x1, y1, x2, y2 in rgb_gt],
                   (0, 255, 0), "GT", thickness=2)
        draw_boxes(rgb_vis, rgb_preds, (0, 0, 255), "Pred", thickness=2)

        # Title on RGB
        cv2.putText(rgb_vis, f"RGB — {len(rgb_preds)} dets, {len(rgb_gt)} GT",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw on IR image
        ir_vis = ir_img.copy()
        draw_boxes(ir_vis, [(x1, y1, x2, y2) for x1, y1, x2, y2 in ir_gt],
                   (0, 255, 0), "GT", thickness=2)
        draw_boxes(ir_vis, ir_preds, (0, 0, 255), "Pred", thickness=2)

        cv2.putText(ir_vis, f"IR — {len(ir_preds)} dets, {len(ir_gt)} GT",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resize IR to match RGB height for side-by-side
        scale = RGB_H / IR_H
        ir_resized = cv2.resize(ir_vis, (int(IR_W * scale), RGB_H))

        # Combine side by side
        combined = np.hstack([rgb_vis, ir_resized])

        # Add stem label at bottom
        h, w = combined.shape[:2]
        cv2.putText(combined, stem, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out_path = out_dir / f"{idx+1:02d}_{stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  [{idx+1}/{len(selected_stems)}] Saved {out_path.name}")

    print(f"\nDone — images saved to {out_dir}/")


if __name__ == "__main__":
    main()
