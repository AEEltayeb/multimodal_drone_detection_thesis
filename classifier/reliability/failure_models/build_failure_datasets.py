"""
build_failure_datasets.py — Build per-modality FN and FP datasets for image-
content-based failure prediction classifiers.

Unlike build_reliability_dataset.py (which uses confidence-derived features and
produces a glorified conf gate), this script extracts features ENTIRELY from the
image content + target geometry. No confidence features.

For each cached inference frame:
  1. Locate and load the image (grayscale)
  2. Compute 7 global image features (mean, std, dynamic range, entropy,
     sky/ground ratio, edge density, blurriness)
  3. Parse GT boxes from cached label text
  4. IoU-match detections to GT at --iou-thresh (default 0.2)
  5. Emit rows to two datasets per modality:
     - FN dataset: one row per GT object, label = 1 if model missed it
       Uses global (7) + target (7) features including local_contrast and
       target_bg_delta (raw target-minus-background brightness).
     - FP dataset: one row per frame, label = 1 if any unmatched detection
       Uses only global (7) features — scene-level hallucination signal.

Crash-safe: per-dataset pickle checkpoints, resumes mid-dataset on restart.

Usage:
    python build_failure_datasets.py
    python build_failure_datasets.py --iou-thresh 0.2 --only rgb
    python build_failure_datasets.py --skip-overlap-check
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Import dataset registry from the inference script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from run_all_inference import DATASETS  # noqa: E402

# ── PATHS ───────────────────────────────────────────────────────────
INFERENCE_DIR  = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "inference"
OUTPUT_DIR     = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "failure_models"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

CHECKPOINT_EVERY = 1000  # save + log progress every N frames


# ── GLOBAL IMAGE FEATURES ──────────────────────────────────────────

def compute_global_features(img_gray):
    """Compute 7 global image features from a grayscale uint8 image."""
    h, w = img_gray.shape[:2]
    img_area = h * w
    img_f = img_gray.astype(np.float32)

    img_mean = float(img_f.mean())
    img_std  = float(img_f.std())
    p2  = float(np.percentile(img_gray, 2))
    p98 = float(np.percentile(img_gray, 98))
    img_dynamic_range = p98 - p2

    # Shannon entropy of intensity histogram
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    p = hist / hist.sum()
    img_entropy = float(-np.sum(p * np.log2(p)))

    # Sky vs ground brightness (aerial-shot detector)
    top_mean = float(img_f[:h // 2].mean())
    bot_mean = float(img_f[h // 2:].mean())
    sky_ground_ratio = top_mean / max(bot_mean, 1.0)

    # Edge density via Canny
    edges = cv2.Canny(img_gray, 50, 150)
    edge_density = float(edges.sum()) / (img_area * 255.0)

    # Blurriness: variance of Laplacian (higher = sharper)
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


GLOBAL_FEATURE_COLS = [
    "img_mean", "img_std", "img_dynamic_range", "img_entropy",
    "sky_ground_ratio", "edge_density", "blurriness",
]


# ── TARGET FEATURES (FOR FN DATASET) ───────────────────────────────

def compute_target_features(img_gray, bbox_xyxy, img_w, img_h):
    """Compute 7 target-specific features for one GT box (pixel coords)."""
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = pw * ph

    log_bbox_area  = float(np.log(area + 1.0))
    aspect_ratio   = float(pw / ph)
    pos_x          = float(cx / img_w) if img_w > 0 else 0.5
    pos_y          = float(cy / img_h) if img_h > 0 else 0.5
    dist_to_center = float(np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2))

    # Clamp to image bounds for cropping
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))

    if xi2 <= xi1 or yi2 <= yi1:
        local_contrast  = 0.0
        target_bg_delta = 0.0
    else:
        target = img_gray[yi1:yi2, xi1:xi2].astype(np.float32)
        target_mean = float(target.mean())

        # Background = 1x box margin ring around the target
        mx = int(pw)
        my = int(ph)
        bx1 = max(0, xi1 - mx)
        by1 = max(0, yi1 - my)
        bx2 = min(img_w, xi2 + mx)
        by2 = min(img_h, yi2 + my)
        bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        bg_mean = float(bg.mean())
        bg_std  = float(bg.std())

        target_bg_delta = target_mean - bg_mean  # raw signed delta
        local_contrast  = target_bg_delta / bg_std if bg_std >= 1.0 else 0.0

    return {
        "log_bbox_area": round(log_bbox_area, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "pos_x": round(pos_x, 4),
        "pos_y": round(pos_y, 4),
        "dist_to_center": round(dist_to_center, 4),
        "local_contrast": round(local_contrast, 4),
        "target_bg_delta": round(target_bg_delta, 3),
    }


TARGET_FEATURE_COLS = [
    "log_bbox_area", "aspect_ratio", "pos_x", "pos_y", "dist_to_center",
    "local_contrast", "target_bg_delta",
]


# ── IoU + GT PARSING ───────────────────────────────────────────────

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
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


def match_gt_to_dets(gt_boxes, dets, iou_thresh):
    """
    Greedy 1:1 matching between GT and detections (by max IoU).
    Returns (gt_matched, det_matched) as boolean lists.
    """
    n_gt = len(gt_boxes)
    n_det = len(dets)
    gt_matched  = [False] * n_gt
    det_matched = [False] * n_det

    if n_gt == 0 or n_det == 0:
        return gt_matched, det_matched

    pairs = []
    for gi in range(n_gt):
        for di in range(n_det):
            iou = compute_iou(gt_boxes[gi], dets[di][:4])
            if iou >= iou_thresh:
                pairs.append((iou, gi, di))
    pairs.sort(reverse=True)

    for _, gi, di in pairs:
        if not gt_matched[gi] and not det_matched[di]:
            gt_matched[gi]  = True
            det_matched[di] = True

    return gt_matched, det_matched


# ── IMAGE LOCATION ─────────────────────────────────────────────────

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


# ── CHECKPOINT I/O ─────────────────────────────────────────────────

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
        return [], [], set()
    try:
        with open(ckpt_path, "rb") as f:
            d = pickle.load(f)
        return d.get("fn", []), d.get("fp", []), set(d.get("processed", []))
    except Exception as e:
        print(f"    [WARN] corrupt checkpoint: {e}")
        return [], [], set()


# ── PER-DATASET PROCESSING ─────────────────────────────────────────

def process_dataset(tag, img_dir, iou_thresh):
    """
    Process one dataset. Returns (fn_df, fp_df) or (None, None) if unavailable.
    Resumes from mid-dataset checkpoint if one exists.
    """
    json_path = INFERENCE_DIR / f"{tag}.json"
    if not json_path.exists():
        print(f"  [SKIP] {tag}: no inference JSON")
        return None, None

    fn_out = CHECKPOINT_DIR / f"{tag}_fn.parquet"
    fp_out = CHECKPOINT_DIR / f"{tag}_fp.parquet"
    if fn_out.exists() and fp_out.exists():
        print(f"  [CACHED] {tag}: loading existing parquets")
        return pd.read_parquet(fn_out), pd.read_parquet(fp_out)

    ckpt_path = CHECKPOINT_DIR / f"{tag}.pkl"
    fn_rows, fp_rows, processed = load_ckpt(ckpt_path)

    print(f"  Loading {tag} inference JSON...", end="", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f" {len(data)} frames")

    stems = sorted(data.keys())
    n_total = len(stems)
    remaining = [s for s in stems if s not in processed]

    print(f"    {n_total} total, {len(processed)} cached, "
          f"{len(remaining)} remaining")
    if not remaining:
        # Convert checkpoint to final parquet
        fn_df = pd.DataFrame(fn_rows)
        fp_df = pd.DataFrame(fp_rows)
        fn_df.to_parquet(fn_out, index=False)
        fp_df.to_parquet(fp_out, index=False)
        if ckpt_path.exists():
            ckpt_path.unlink()
        return fn_df, fp_df

    n_image_missing = 0
    n_image_unreadable = 0
    n_processed_session = 0
    t0 = time.time()

    for idx, stem in enumerate(remaining):
        frame = data[stem]
        dets = frame["dets"]
        img_w = frame["w"]
        img_h = frame["h"]
        gt_text = frame.get("gt", "")

        img_path = locate_image(img_dir, stem)
        if img_path is None:
            n_image_missing += 1
            processed.add(stem)
            n_processed_session += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_image_unreadable += 1
            processed.add(stem)
            n_processed_session += 1
            continue

        gf = compute_global_features(img)
        gt_boxes = parse_yolo_gt(gt_text, img_w, img_h)
        gt_matched, det_matched = match_gt_to_dets(gt_boxes, dets, iou_thresh)

        # FN rows — one per GT object
        for gi, gt in enumerate(gt_boxes):
            tf = compute_target_features(img, gt, img_w, img_h)
            row = {
                "stem": stem,
                "source_dataset": tag,
                "gt_idx": gi,
                "label": 0 if gt_matched[gi] else 1,  # 1 = FN (missed)
            }
            row.update(gf)
            row.update(tf)
            fn_rows.append(row)

        # FP row — one per frame
        n_fp = sum(1 for m in det_matched if not m)
        frame_row = {
            "stem": stem,
            "source_dataset": tag,
            "label": 1 if n_fp > 0 else 0,
            "n_dets_in_frame": len(dets),
            "n_fp_in_frame": n_fp,
            "n_gt_in_frame": len(gt_boxes),
        }
        frame_row.update(gf)
        fp_rows.append(frame_row)

        processed.add(stem)
        n_processed_session += 1

        if n_processed_session % CHECKPOINT_EVERY == 0 or (idx + 1) == len(remaining):
            elapsed = time.time() - t0
            fps = n_processed_session / elapsed if elapsed > 0 else 0
            left = len(remaining) - (idx + 1)
            eta_sec = left / fps if fps > 0 else 0
            print(f"    [{idx + 1}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta_sec/60:.1f}min | "
                  f"FN rows: {len(fn_rows)}, FP rows: {len(fp_rows)}")
            # Checkpoint (cheap — pickle is fast)
            atomic_pickle_write(
                ckpt_path,
                {"fn": fn_rows, "fp": fp_rows, "processed": list(processed)},
            )

    # Final output
    fn_df = pd.DataFrame(fn_rows)
    fp_df = pd.DataFrame(fp_rows)
    fn_df.to_parquet(fn_out, index=False)
    fp_df.to_parquet(fp_out, index=False)
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"  ✓ {tag}: processed {n_processed_session} frames this session "
          f"({elapsed/60:.1f}min) | "
          f"missing={n_image_missing}, unreadable={n_image_unreadable}")
    return fn_df, fp_df


# ── ANTI-UAV VAL/TEST OVERLAP DIAGNOSTIC ───────────────────────────

def check_antiuav_overlap():
    for modality in ["rgb", "ir"]:
        val_tag  = f"antiuav_val_{modality}"
        test_tag = f"antiuav_test_{modality}"
        val_json  = INFERENCE_DIR / f"{val_tag}.json"
        test_json = INFERENCE_DIR / f"{test_tag}.json"
        if not val_json.exists() or not test_json.exists():
            print(f"  [antiuav {modality}] inference JSONs missing, skipping")
            continue
        with open(val_json, "r") as f:
            val_stems = set(json.load(f).keys())
        with open(test_json, "r") as f:
            test_stems = set(json.load(f).keys())
        overlap = val_stems & test_stems
        pct_of_val = len(overlap) / len(val_stems) * 100 if val_stems else 0
        status = "DISJOINT" if len(overlap) == 0 else "OVERLAP"
        print(f"  [antiuav {modality}] val={len(val_stems)}, "
              f"test={len(test_stems)}, overlap={len(overlap)} "
              f"({pct_of_val:.1f}% of val) -> {status}")
        if pct_of_val > 50:
            print(f"    [WARN] val largely inside test — consider dropping val")


# ── MAIN ───────────────────────────────────────────────────────────

def get_datasets_for_modality(modality):
    return [(tag, img_dir) for tag, img_dir, lbl_dir, m in DATASETS if m == modality]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-thresh", type=float, default=0.2,
                        help="IoU threshold for matching (default: 0.2)")
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Process only one modality")
    parser.add_argument("--skip-overlap-check", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build failure datasets (FN + FP) for image-content reliability models")
    print("=" * 70)
    print(f"  IoU threshold: {args.iou_thresh}")
    print(f"  Inference dir: {INFERENCE_DIR}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print()

    if not args.skip_overlap_check:
        print("Anti-UAV val vs test overlap check:")
        check_antiuav_overlap()
        print()

    modalities = [args.only] if args.only else ["rgb", "ir"]

    for modality in modalities:
        print(f"\n{'=' * 70}")
        print(f"Building {modality.upper()} failure datasets")
        print(f"{'=' * 70}")

        dataset_list = get_datasets_for_modality(modality)
        all_fn, all_fp = [], []

        for tag, img_dir in dataset_list:
            print(f"\n  → {tag}")
            fn_df, fp_df = process_dataset(tag, img_dir, args.iou_thresh)
            if fn_df is not None:
                all_fn.append(fn_df)
            if fp_df is not None:
                all_fp.append(fp_df)

        if all_fn:
            master_fn = pd.concat(all_fn, ignore_index=True)
            out = OUTPUT_DIR / f"{modality}_fn_dataset.csv"
            master_fn.to_csv(out, index=False)
            fn_rate = (master_fn["label"] == 1).mean() * 100
            print(f"\n  ✓ {modality.upper()} FN dataset: {len(master_fn):,} rows "
                  f"(FN rate: {fn_rate:.1f}%) -> {out}")

            print(f"    Per-dataset FN rate:")
            for tag in master_fn["source_dataset"].unique():
                s = master_fn[master_fn["source_dataset"] == tag]
                r = (s["label"] == 1).mean() * 100
                print(f"      {tag:<25s} {len(s):>7,} objects, FN {r:5.1f}%")

        if all_fp:
            master_fp = pd.concat(all_fp, ignore_index=True)
            out = OUTPUT_DIR / f"{modality}_fp_dataset.csv"
            master_fp.to_csv(out, index=False)
            fp_rate = (master_fp["label"] == 1).mean() * 100
            print(f"\n  ✓ {modality.upper()} FP dataset: {len(master_fp):,} rows "
                  f"(FP frame rate: {fp_rate:.1f}%) -> {out}")

            print(f"    Per-dataset FP frame rate:")
            for tag in master_fp["source_dataset"].unique():
                s = master_fp[master_fp["source_dataset"] == tag]
                r = (s["label"] == 1).mean() * 100
                print(f"      {tag:<25s} {len(s):>7,} frames, FP {r:5.1f}%")

    print("\nDone. Ready for train_failure_models.py")


if __name__ == "__main__":
    main()
