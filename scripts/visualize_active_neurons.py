"""
Generate spatial activation heatmaps overlaid on actual drone/confuser images.

Shows WHERE in the image the top discriminative neurons fire for a drone
vs a confuser detection. Produces:

    docs/analysis/images/v5_activation_drone_example.png
    docs/analysis/images/v5_activation_confuser_example.png

Each output is a 2x2 panel: top-left = original crop, top-right = P3 heatmap,
bottom-left = P5 heatmap, bottom-right = fused activation intensity.

Usage:
    python scripts/visualize_active_neurons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "scripts"))
from distill_v5_p3p5_ft4 import (
    DetectInputHook, _extract_detection_features, _match_det_to_gt,
    _resolve_labels_dir, INPUT_DIM, META_DIM, _P3_DIM, _P5_DIM,
    MODEL_PATHS, IOU_THR, IOP_THR, CONF_THR,
)
from visualize_v5_features import neuron_anova_rank, _layer_name

OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Load V5 cache to get neuron rankings
DATA = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_data.npz"

# Sample image sources
SVANSTROM_DIR = Path("G:/drone/svanstrom_paired/RGB/images")
CONFUSER_DIR = Path("G:/drone/rgb_confusers_merged/images/test")


def get_top_neurons(n_top: int = 8):
    """Load V5 cache, compute ANOVA, return indices of top neurons."""
    z = np.load(DATA)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    F = neuron_anova_rank(X, y)
    top = np.argsort(F)[::-1][:n_top]
    print(f"Top {n_top} neurons by ANOVA F-stat: {top.tolist()}")
    for i, idx in enumerate(top):
        print(f"  #{i+1}: Feature {idx} ({_layer_name(idx)}) F={F[idx]:.0f}")
    return top, F


def make_spatial_heatmap(hook, det_box, img_shape, top_neurons, layer="p3"):
    """
    Build a spatial heatmap for one detection box by averaging the top
    discriminative channels of the P3 or P5 feature map in the ROI region.
    """
    ih, iw = img_shape[:2]
    x1, y1, x2, y2 = det_box

    if layer == "p3":
        fmap = hook.p3  # (1, 64, H/8, W/8)
        stride = 8
        # Top neuron indices in p3 space: feat_idx in [5, 261) → channel = (feat_idx - 5) % 64
        channels = []
        for idx in top_neurons:
            if META_DIM <= idx < META_DIM + _P3_DIM:
                ch = (idx - META_DIM) % 64
                channels.append(ch)
    else:  # p5
        fmap = hook.p5  # (1, 256, H/32, W/32)
        stride = 32
        channels = []
        for idx in top_neurons:
            if idx >= META_DIM + _P3_DIM:
                ch = idx - META_DIM - _P3_DIM
                channels.append(ch)

    if fmap is None or len(channels) == 0:
        return None

    fmap_np = fmap[0].detach().cpu().numpy()  # (C, H_f, W_f)
    fh, fw = fmap_np.shape[1], fmap_np.shape[2]

    # Map detection box to feature map coordinates
    fx1 = max(0, int(x1 / (iw / fw)))
    fy1 = max(0, int(y1 / (ih / fh)))
    fx2 = min(fw, int(x2 / (iw / fw)) + 1)
    fy2 = min(fh, int(y2 / (ih / fh)) + 1)

    # Average activation of discriminative channels
    selected = fmap_np[channels]  # (n_channels, H_f, W_f)
    heat = np.abs(selected).mean(axis=0)  # (H_f, W_f) — absolute activation

    # Resize to image dimensions
    heat_resized = cv2.resize(heat, (iw, ih), interpolation=cv2.INTER_LINEAR)
    return heat_resized


def find_best_example(model, hook, img_dir, is_drone: bool, max_scan: int = 200):
    """
    Scan images to find the detection with the highest MLP-discriminative
    neuron activation (for drones) or lowest (for confusers).
    """
    if "svanstrom" in str(img_dir).lower():
        labels_dir = img_dir.parent / "labels"
    else:
        labels_dir = _resolve_labels_dir(img_dir)
    all_images = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    np.random.RandomState(42).shuffle(all_images)
    images = all_images[:max_scan]

    best_score = -1e9
    best_result = None

    z = np.load(DATA)
    X_all = z["X"].astype(np.float32)
    y_all = z["y"].astype(np.int64)
    F = neuron_anova_rank(X_all, y_all)
    top = np.argsort(F)[::-1][:8]

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        ih, iw = img_bgr.shape[:2]

        # Load GT if available
        gt_boxes = []
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * iw
                    y1 = (yc - bh / 2) * ih
                    x2 = (xc + bw / 2) * iw
                    y2 = (yc + bh / 2) * ih
                    gt_boxes.append((x1, y1, x2, y2))

        hook.clear()
        results = model.predict(img_bgr, imgsz=1280, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            det_box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            det_conf = float(boxes.conf[i])
            feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)

            # Check if this is a TP (drone) or FP (confuser)
            if gt_boxes:
                is_tp = _match_det_to_gt(det_box, gt_boxes, "iop")
            else:
                is_tp = False  # No GT = confuser dataset, all dets are FPs

            if is_drone and not is_tp:
                continue
            if not is_drone and is_tp:
                continue

            # Score: sum of absolute activation on top neurons
            score = sum(abs(feat[idx]) for idx in top)
            if is_drone:
                # For drones, we want high positive activation
                score = sum(feat[idx] for idx in top if feat[idx] > 0)
            else:
                # For confusers, we want high confuser-pattern activation
                score = sum(abs(feat[idx]) for idx in top)

            if score > best_score:
                best_score = score
                x1, y1, x2, y2 = det_box
                pad = 20
                cx1 = max(0, int(x1) - pad)
                cy1 = max(0, int(y1) - pad)
                cx2 = min(iw, int(x2) + pad)
                cy2 = min(ih, int(y2) + pad)
                crop = img_bgr[cy1:cy2, cx1:cx2].copy()
                best_result = {
                    "img_bgr": img_bgr.copy(),
                    "crop": crop,
                    "det_box": det_box,
                    "conf": det_conf,
                    "feat": feat,
                    "path": str(img_path),
                    "score": score,
                }

    return best_result


def generate_activation_panel(model, hook, result, top_neurons, title, out_path):
    """Generate a 2x2 panel: crop + P3 heatmap + P5 heatmap + fused."""
    img_bgr = result["img_bgr"]
    det_box = result["det_box"]
    ih, iw = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det_box]

    # Re-run inference to get fresh feature maps
    hook.clear()
    model.predict(img_bgr, imgsz=1280, conf=CONF_THR, verbose=False, device="cuda")

    # Build heatmaps
    heat_p3 = make_spatial_heatmap(hook, det_box, img_bgr.shape, top_neurons, "p3")
    heat_p5 = make_spatial_heatmap(hook, det_box, img_bgr.shape, top_neurons, "p5")

    # Crop region for display
    pad = 40
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(iw, x2 + pad)
    cy2 = min(ih, y2 + pad)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = img_rgb[cy1:cy2, cx1:cx2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Original crop with bounding box
    axes[0].imshow(crop_rgb)
    # Draw box relative to crop
    bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    rect = plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                          linewidth=2, edgecolor="lime", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].set_title(f"Detection (conf={result['conf']:.2f})")
    axes[0].axis("off")

    # Panel 2: P3 heatmap (high-res spatial)
    if heat_p3 is not None:
        heat_p3_crop = heat_p3[cy1:cy2, cx1:cx2]
        axes[1].imshow(crop_rgb)
        axes[1].imshow(heat_p3_crop, cmap="jet", alpha=0.5,
                       vmin=0, vmax=np.percentile(heat_p3, 95))
        axes[1].set_title("P3 activation (stride 8 — spatial detail)")
    else:
        axes[1].text(0.5, 0.5, "No P3 neurons\nin top-8", ha="center", va="center")
    axes[1].axis("off")

    # Panel 3: P5 heatmap (semantic depth)
    if heat_p5 is not None:
        heat_p5_crop = heat_p5[cy1:cy2, cx1:cx2]
        axes[2].imshow(crop_rgb)
        axes[2].imshow(heat_p5_crop, cmap="jet", alpha=0.5,
                       vmin=0, vmax=np.percentile(heat_p5, 95))
        axes[2].set_title("P5 activation (stride 32 — semantic depth)")
    else:
        axes[2].text(0.5, 0.5, "No P5 neurons\nin top-8", ha="center", va="center")
    axes[2].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")


def main():
    print("=" * 72)
    print("  Active Neuron Visualization — Drone vs Confuser")
    print("=" * 72)

    # Load model
    yolo = YOLO(MODEL_PATHS["ft4_r3"])
    hook = DetectInputHook()
    handle = hook.register(yolo)

    top_neurons, F = get_top_neurons(n_top=8)

    try:
        # Find best drone example
        print("\n[1/4] Scanning Svanström for best drone example ...")
        drone_result = find_best_example(yolo, hook, SVANSTROM_DIR,
                                         is_drone=True, max_scan=300)
        if drone_result:
            print(f"  Best drone: {drone_result['path']} "
                  f"(conf={drone_result['conf']:.2f}, score={drone_result['score']:.1f})")
        else:
            print("  WARNING: No drone example found!")

        # Find best confuser example
        print("\n[2/4] Scanning confuser test for best confuser example ...")
        confuser_result = find_best_example(yolo, hook, CONFUSER_DIR,
                                            is_drone=False, max_scan=300)
        if confuser_result:
            print(f"  Best confuser: {confuser_result['path']} "
                  f"(conf={confuser_result['conf']:.2f}, score={confuser_result['score']:.1f})")
        else:
            print("  WARNING: No confuser example found!")

        # Generate panels
        if drone_result:
            print("\n[3/4] Generating drone activation panel ...")
            generate_activation_panel(
                yolo, hook, drone_result, top_neurons,
                "DRONE detection — top discriminative neuron activation",
                OUT / "v5_activation_drone_example.png")

        if confuser_result:
            print("\n[4/4] Generating confuser activation panel ...")
            generate_activation_panel(
                yolo, hook, confuser_result, top_neurons,
                "CONFUSER detection — top discriminative neuron activation",
                OUT / "v5_activation_confuser_example.png")

    finally:
        handle.remove()

    print("\nDone.")


if __name__ == "__main__":
    main()
