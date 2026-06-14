"""
Generate a compilation grid of spatial activation heatmaps across multiple
datasets (drones vs confusers). Handles both image directories and video files.

For video sources without GT labels, uses heuristics to ensure the detected
object actually matches what we expect (e.g. drone videos -> drone detections,
airplane videos -> airplane detections).

Usage:
    python scripts/visualize_activation_compilation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import random

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
    _resolve_labels_dir, META_DIM, _P3_DIM, CONF_THR
)
from visualize_v5_features import neuron_anova_rank
from visualize_active_neurons import get_top_neurons, make_spatial_heatmap

OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

DATA = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_data.npz"

# ── Video file filters ──────────────────────────────────────────────────
# For the drone split, only use "pure drone" videos (takeoffs, no bird attacks)
DRONE_VIDEO_WHITELIST = [
    "drone takeoff",
    "drone takeoff short trees",
    "drone takeoff short (dji",
    "drone takeoff from ground",
    "IMG_1519",
]

# For the drone split, explicitly exclude videos that have birds
DRONE_VIDEO_BLACKLIST = [
    "bird", "seagull", "flock",
]

# Define the targets we want to collect
TARGETS = [
    {"name": "Anti-UAV (Drone)", "path": "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images", "is_drone": True, "type": "img"},
    {"name": "Svanström (Drone)", "path": "G:/drone/svanstrom_paired/RGB/images", "is_drone": True, "type": "img"},
    {"name": "Selcom (Drone)", "path": "G:/drone/selcom_dataset/images", "is_drone": True, "type": "img"},
    {"name": "Video Test (Drone)", "path": "G:/drone/drone detection video tests/rgb/drone", "is_drone": True, "type": "vid"},
    
    {"name": "Svanström (Confuser)", "path": "G:/drone/svanstrom_paired/RGB/images", "is_drone": False, "type": "img"},
    {"name": "Merged Confusers", "path": "G:/drone/rgb_confusers_merged/images/test", "is_drone": False, "type": "img"},
    {"name": "Video Test (Airplane)", "path": "G:/drone/drone detection video tests/rgb/airplanes", "is_drone": False, "type": "vid"},
    {"name": "Video Test (Bird)", "path": "G:/drone/drone detection video tests/rgb/birds", "is_drone": False, "type": "vid"}
]


def load_gt_for_image(img_path: Path, labels_dir: Path, img_shape: tuple) -> list:
    ih, iw = img_shape
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
    return gt_boxes


def score_detection(feat, is_drone, top):
    if is_drone:
        return sum(feat[idx] for idx in top if feat[idx] > 0)
    else:
        return sum(abs(feat[idx]) for idx in top)


def is_sky_detection(det_box, ih, iw):
    """Check if detection is in the upper 2/3 of the frame (likely sky, not ground/tree)."""
    _, y1, _, y2 = det_box
    cy = (y1 + y2) / 2.0
    return cy < ih * 0.7


def scan_images(model, hook, img_dir: Path, is_drone: bool, top_neurons: np.ndarray,
                max_scan: int = 2000):
    img_dir = Path(img_dir)
    if "svanstrom" in str(img_dir).lower():
        labels_dir = img_dir.parent / "labels"
    elif "selcom_dataset" in str(img_dir).lower():
        labels_dir = img_dir.parent / "labels"
    else:
        labels_dir = _resolve_labels_dir(img_dir)

    all_images = [p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    np.random.RandomState(42).shuffle(all_images)
    images = all_images[:max_scan]

    best_score = -1e9
    best_result = None

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        ih, iw = img_bgr.shape[:2]

        gt_boxes = load_gt_for_image(img_path, labels_dir, (ih, iw))

        hook.clear()
        results = model.predict(img_bgr, imgsz=1280, conf=CONF_THR, verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            det_box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            det_conf = float(boxes.conf[i])
            feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)

            if gt_boxes:
                is_tp = _match_det_to_gt(det_box, gt_boxes, "iop")
            else:
                is_tp = False

            if is_drone and not is_tp:
                continue
            if not is_drone and is_tp:
                continue

            score = score_detection(feat, is_drone, top_neurons)
            if score > best_score:
                best_score = score
                best_result = {
                    "img_bgr": img_bgr.copy(),
                    "det_box": det_box,
                    "conf": det_conf,
                    "source": str(img_path.name),
                }

    return best_result


def _filter_drone_videos(videos: list[Path]) -> list[Path]:
    """Only keep videos that are pure drone footage (no bird attack videos)."""
    filtered = []
    for v in videos:
        name_lower = v.name.lower()
        # Exclude any video with bird-related keywords
        if any(kw in name_lower for kw in DRONE_VIDEO_BLACKLIST):
            continue
        filtered.append(v)
    if not filtered:
        # Fallback: return all if filter was too aggressive
        return videos
    return filtered


def scan_videos(model, hook, vid_dir: Path, is_drone: bool, top_neurons: np.ndarray,
                target_name: str = "", max_frames_total: int = 1500):
    vid_dir = Path(vid_dir)
    videos = [p for p in vid_dir.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".avi")]

    # For drone split, filter out bird-attack videos
    if is_drone and "drone" in str(vid_dir).lower():
        videos = _filter_drone_videos(videos)
        print(f"    Filtered drone videos: {[v.name for v in videos]}")

    random.seed(42)
    random.shuffle(videos)

    best_score = -1e9
    best_result = None
    frames_processed = 0

    for vid_path in videos:
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue

        # Sample frames uniformly from the video
        num_to_sample = min(200, total_frames)
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)

        for idx in indices:
            if frames_processed >= max_frames_total:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frames_processed += 1
            ih, iw = frame.shape[:2]

            hook.clear()
            results = model.predict(frame, imgsz=1280, conf=CONF_THR, verbose=False,
                                    device="cuda")
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                det_box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
                det_conf = float(boxes.conf[i])

                # ── Verification heuristics for video detections ──
                x1, y1, x2, y2 = det_box
                bw = x2 - x1
                bh = y2 - y1
                box_area = bw * bh
                img_area = ih * iw

                # For airplane videos: require detection is in the sky (upper portion)
                # and is a reasonable size (not a tiny edge artifact or huge ground blob)
                if "airplane" in target_name.lower():
                    if not is_sky_detection(det_box, ih, iw):
                        continue
                    # Filter out very small detections (likely noise) and very large
                    # (likely ground/tree)
                    area_ratio = box_area / img_area
                    if area_ratio < 0.001 or area_ratio > 0.15:
                        continue

                # For bird videos: prefer sky detections, reasonable size
                if "bird" in target_name.lower():
                    area_ratio = box_area / img_area
                    if area_ratio > 0.3:  # Too large = probably not a bird
                        continue

                # For drone videos: prefer confident detections with reasonable size
                if is_drone and "drone" in target_name.lower():
                    if det_conf < 0.35:  # Low conf drones in drone-only videos are suspect
                        continue

                feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)
                score = score_detection(feat, is_drone, top_neurons)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "img_bgr": frame.copy(),
                        "det_box": det_box,
                        "conf": det_conf,
                        "source": f"{vid_path.name} frame={idx}",
                    }

        cap.release()
        if frames_processed >= max_frames_total:
            break

    return best_result


def build_compilation(results: dict, top_neurons: np.ndarray, out_path: Path):
    """Build a 8-row x 3-col grid of crops, p3 heatmaps, and p5 heatmaps."""
    fig, axes = plt.subplots(8, 3, figsize=(14, 28))

    # Column headers
    for j, title in enumerate(["Detection Crop", "P3 Activation (Spatial)", "P5 Activation (Semantic)"]):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=12)

    yolo = YOLO(str(REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"))
    hook = DetectInputHook()
    handle = hook.register(yolo)

    # Draw separator between drones and confusers
    drone_count = sum(1 for t in TARGETS if t["is_drone"])

    for row, target in enumerate(TARGETS):
        name = target["name"]
        res = results.get(name)

        ax_crop = axes[row, 0]
        ax_p3 = axes[row, 1]
        ax_p5 = axes[row, 2]

        if res is None:
            ax_crop.text(0.5, 0.5, f"No detection found\nfor {name}",
                         ha="center", va="center", fontsize=10)
            ax_crop.axis("off")
            ax_p3.axis("off")
            ax_p5.axis("off")
            continue

        img_bgr = res["img_bgr"]
        det_box = res["det_box"]
        x1, y1, x2, y2 = [int(v) for v in det_box]
        ih, iw = img_bgr.shape[:2]

        hook.clear()
        yolo.predict(img_bgr, imgsz=1280, conf=CONF_THR, verbose=False, device="cuda")

        heat_p3 = make_spatial_heatmap(hook, det_box, img_bgr.shape, top_neurons, "p3")
        heat_p5 = make_spatial_heatmap(hook, det_box, img_bgr.shape, top_neurons, "p5")

        pad = 50
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(iw, x2 + pad), min(ih, y2 + pad)
        crop_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[cy1:cy2, cx1:cx2]

        # Color-code: green border for drones, red border for confusers
        border_color = "lime" if target["is_drone"] else "red"
        label_color = "#2ecc71" if target["is_drone"] else "#e74c3c"
        category = "DRONE" if target["is_drone"] else "CONFUSER"

        # Plot Crop
        ax_crop.imshow(crop_rgb)
        bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
        rect = plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                              linewidth=2.5, edgecolor=border_color, facecolor="none")
        ax_crop.add_patch(rect)
        ax_crop.set_ylabel(f"{name}\nconf={res['conf']:.2f}", fontsize=9, rotation=0,
                           labelpad=80, ha="right", va="center",
                           color=label_color, fontweight="bold")
        ax_crop.axis("off")

        # Plot P3
        if heat_p3 is not None:
            heat_p3_crop = heat_p3[cy1:cy2, cx1:cx2]
            ax_p3.imshow(crop_rgb)
            ax_p3.imshow(heat_p3_crop, cmap="jet", alpha=0.5,
                         vmin=0, vmax=np.percentile(heat_p3, 95))
        ax_p3.axis("off")

        # Plot P5
        if heat_p5 is not None:
            heat_p5_crop = heat_p5[cy1:cy2, cx1:cx2]
            ax_p5.imshow(crop_rgb)
            ax_p5.imshow(heat_p5_crop, cmap="jet", alpha=0.5,
                         vmin=0, vmax=np.percentile(heat_p5, 95))
        ax_p5.axis("off")

    handle.remove()

    # Add a horizontal line between drones and confusers
    # Using figure-level annotation
    fig.text(0.02, 0.52, "─── DRONES (above) ───  vs  ─── CONFUSERS (below) ───",
             ha="left", fontsize=10, color="gray", style="italic")

    plt.tight_layout(h_pad=1.5)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved compilation to {out_path}")


def main():
    print("=" * 72)
    print("  Activation Compilation — Drone vs Confuser across datasets")
    print("=" * 72)

    yolo = YOLO(str(REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"))
    hook = DetectInputHook()
    handle = hook.register(yolo)

    top_neurons, _ = get_top_neurons(n_top=8)

    results = {}

    for t in TARGETS:
        print(f"\nScanning {t['name']} ...")
        if t["type"] == "img":
            res = scan_images(yolo, hook, t["path"], t["is_drone"], top_neurons)
        else:
            res = scan_videos(yolo, hook, t["path"], t["is_drone"], top_neurons,
                              target_name=t["name"])

        if res:
            print(f"  ✓ Found: {res['source']} (conf={res['conf']:.2f})")
            results[t["name"]] = res
        else:
            print("  ✗ WARNING: No example found!")

    handle.remove()

    print("\n" + "=" * 72)
    print("Building compilation grid...")
    build_compilation(results, top_neurons, OUT / "v5_activation_compilation.png")
    print("Done.")


if __name__ == "__main__":
    main()
