"""
run_all_inference.py — Run RGB and IR YOLO models across ALL datasets,
cache predictions to per-(model, dataset) JSON files for reuse.

Each (model, dataset) pair gets its own checkpoint file under
    runs/reliability/inference/{model_tag}_{dataset_tag}.json

Crash-safe: checkpoints every N frames. Resume by re-running the same command.
Completed jobs are skipped entirely.

Usage:
    python run_all_inference.py                 # run everything
    python run_all_inference.py --only rgb      # run RGB model only
    python run_all_inference.py --only ir       # run IR model only
    python run_all_inference.py --dry-run       # list jobs without running
"""

import argparse
import json
import os
import time
from pathlib import Path

# ── PATHS ───────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parents[2]   # ES_Drone_Thesis/ (resident; was ES_Drone_Detection)

RGB_WEIGHTS_DEFAULT = WORKSPACE / "models" / "rgb" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"
IR_WEIGHTS_DEFAULT  = WORKSPACE / "models" / "ir" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

OUTPUT_DIR_BASE = WORKSPACE / "classifier" / "runs" / "reliability"

# Inference settings (matching existing pipeline)
CONF_THRESH = 0.001   # low threshold — let the classifier decide
IOU_NMS     = 0.45
IMGSZ       = 640
MAX_DET     = 20
DEVICE      = 0       # GPU 0
CHECKPOINT_EVERY = 200

# ── DATASET REGISTRY ────────────────────────────────────────────────
# Each entry: (tag, image_dir, label_dir, modality)
#   tag:       unique name for caching
#   image_dir: folder with images
#   label_dir: folder with YOLO .txt labels (for GT comparison later)
#   modality:  "rgb" or "ir" — determines which model to use

DATASETS = [
    # --- RGB model datasets ---
    ("rgb_dataset_val",
     r"G:\drone\dataset\dataset\images\val",
     r"G:\drone\dataset\dataset\labels\val",
     "rgb"),
    ("rgb_dataset_test",
     r"G:\drone\dataset\dataset\images\test",
     r"G:\drone\dataset\dataset\labels\test",
     "rgb"),
    ("antiuav_val_rgb",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\RGB\images",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\RGB\labels",
     "rgb"),
    ("antiuav_test_rgb",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\images",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\labels",
     "rgb"),
    ("svanstrom_rgb",
     r"G:\drone\svanstrom_paired\RGB\images",
     r"G:\drone\svanstrom_paired\RGB\labels",
     "rgb"),

    # --- IR model datasets ---
    ("ir_dset_final_val",
     r"G:\drone\IR_dset_final\val\images",
     r"G:\drone\IR_dset_final\val\labels",
     "ir"),
    ("ir_dset_final_test",
     r"G:\drone\IR_dset_final\test\images",
     r"G:\drone\IR_dset_final\test\labels",
     "ir"),
    ("cst_antiuav_test",
     r"G:\drone\CST-AntiUAV_YOLO\test\images",
     r"G:\drone\CST-AntiUAV_YOLO\test\labels",
     "ir"),
    ("antiuav_val_ir",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\IR\images",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\IR\labels",
     "ir"),
    ("antiuav_test_ir",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\IR\images",
     r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\IR\labels",
     "ir"),
    ("svanstrom_ir",
     r"G:\drone\svanstrom_paired\IR\images",
     r"G:\drone\svanstrom_paired\IR\labels",
     "ir"),
]


# ── UTILITIES ───────────────────────────────────────────────────────
def atomic_json_write(path, data):
    """Write JSON atomically with os.replace (Windows-safe)."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    for attempt in range(5):
        try:
            os.replace(tmp, str(path))
            return
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    # Last resort
    if os.path.exists(str(path)):
        os.remove(str(path))
    os.rename(tmp, str(path))


def load_checkpoint(path):
    """Load existing checkpoint, or return empty dict."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    [WARN] Corrupt checkpoint {path.name}: {e}")
        return {}


def count_images_fast(img_dir):
    """Fast count of images without building full list."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.suffix.lower() in exts)


def discover_images(img_dir):
    """Return sorted list of image paths in a directory."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return []
    print(f"      Scanning {img_dir.name}...", end="", flush=True)
    result = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    print(f" {len(result)} images")
    return result


def run_model_on_image(model, img_path):
    """Run YOLO on one image. Returns (dets, w, h) or (None, None, None)."""
    try:
        results = model.predict(
            source=str(img_path),
            conf=CONF_THRESH,
            iou=IOU_NMS,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
            save=False,
            max_det=MAX_DET,
        )
    except Exception as e:
        # Catch ALL exceptions: cv2.error, ValueError, OSError, etc.
        return None, None, None

    if not results:
        return None, None, None

    r = results[0]
    h, w = r.orig_shape
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append([float(v) for v in xyxy[i]] + [float(confs[i])])
    return dets, w, h


def run_dataset(model, tag, img_dir, label_dir, out_dir):
    """Run model on one dataset, with checkpoint/resume."""
    images = discover_images(img_dir)
    if not images:
        print(f"    [SKIP] No images found in {img_dir}")
        return

    out_path = out_dir / f"{tag}.json"
    ckpt_path = out_dir / f"{tag}.checkpoint.json"

    # Check if already complete
    if out_path.exists():
        existing = load_checkpoint(out_path)
        if len(existing) >= len(images):
            print(f"    [DONE] {tag}: {len(existing)} frames already cached → skip")
            return
        # Partial final file — treat as checkpoint
        detections = existing
    else:
        detections = load_checkpoint(ckpt_path)

    done_stems = set(detections.keys())
    remaining = [p for p in images if p.stem not in done_stems]

    print(f"    {tag}: {len(images)} total, {len(done_stems)} cached, {len(remaining)} remaining")
    if not remaining:
        # Write final output
        atomic_json_write(out_path, detections)
        if ckpt_path.exists():
            ckpt_path.unlink()
        return

    t0 = time.time()
    n_skipped = 0

    for idx, img_path in enumerate(remaining):
        dets, w, h = run_model_on_image(model, img_path)
        if dets is None:
            n_skipped += 1
            continue

        # Find matching label file
        lbl_path = Path(label_dir) / (img_path.stem + ".txt")
        gt_text = ""
        if lbl_path.exists():
            gt_text = lbl_path.read_text().strip()

        detections[img_path.stem] = {
            "dets": dets,
            "w": w,
            "h": h,
            "gt": gt_text,  # raw YOLO label text — parse later
        }

        processed = idx + 1
        if processed % CHECKPOINT_EVERY == 0 or processed == len(remaining):
            elapsed = time.time() - t0
            fps = processed / elapsed if elapsed > 0 else 0
            eta = (len(remaining) - processed) / fps if fps > 0 else 0
            print(f"      [{processed}/{len(remaining)}] {fps:.1f} fps, "
                  f"ETA {eta/60:.1f}min, {len(detections)} total cached")
            atomic_json_write(ckpt_path, detections)

    # Write final
    atomic_json_write(out_path, detections)
    if ckpt_path.exists():
        ckpt_path.unlink()

    elapsed = time.time() - t0
    print(f"    ✓ {tag}: {len(detections)} frames saved "
          f"({elapsed/60:.1f}min, {n_skipped} skipped)")


# ── MAIN ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run RGB/IR YOLO models on all datasets, cache predictions")
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Run only one modality (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without executing")
    parser.add_argument("--device", type=str, default=str(DEVICE),
                        help="CUDA device (default: 0)")
    parser.add_argument("--rgb-weights", type=str, default=str(RGB_WEIGHTS_DEFAULT),
                        help="Path to RGB YOLO weights")
    parser.add_argument("--ir-weights", type=str, default=str(IR_WEIGHTS_DEFAULT),
                        help="Path to IR YOLO weights")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output dir; e.g. '_v3more' -> inference_v3more/")
    args = parser.parse_args()

    device = args.device
    RGB_WEIGHTS = Path(args.rgb_weights)
    IR_WEIGHTS  = Path(args.ir_weights)
    OUTPUT_DIR  = OUTPUT_DIR_BASE / f"inference{args.suffix}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for tag, img_dir, lbl_dir, modality in DATASETS:
        if args.only and modality != args.only:
            continue
        if not Path(img_dir).exists():
            print(f"  [WARN] {img_dir} does not exist — skipping {tag}")
            continue
        jobs.append((tag, img_dir, lbl_dir, modality))

    print("=" * 70)
    print("Bulk inference for reliability classifiers")
    print("=" * 70)
    print(f"  RGB weights: {RGB_WEIGHTS}")
    print(f"  IR weights:  {IR_WEIGHTS}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  Device:      {device}")
    print(f"  Conf thresh: {CONF_THRESH}")
    print()

    # Group by modality
    rgb_jobs = [(t, i, l) for t, i, l, m in jobs if m == "rgb"]
    ir_jobs  = [(t, i, l) for t, i, l, m in jobs if m == "ir"]

    print(f"  RGB jobs ({len(rgb_jobs)}):")
    for tag, img_dir, _ in rgb_jobs:
        done = OUTPUT_DIR / f"{tag}.json"
        if done.exists():
            status = "✓ CACHED"
        else:
            print(f"    {tag:<25s} counting...", end="\r", flush=True)
            n = count_images_fast(img_dir)
            status = f"{n:,} images"
        print(f"    {tag:<25s} {status}")

    print(f"\n  IR jobs ({len(ir_jobs)}):")
    for tag, img_dir, _ in ir_jobs:
        done = OUTPUT_DIR / f"{tag}.json"
        if done.exists():
            status = "✓ CACHED"
        else:
            print(f"    {tag:<25s} counting...", end="\r", flush=True)
            n = count_images_fast(img_dir)
            status = f"{n:,} images"
        print(f"    {tag:<25s} {status}")

    if args.dry_run:
        print("\n  [DRY RUN] Exiting.")
        return

    # Load RGB model and run RGB jobs
    if rgb_jobs:
        print(f"\n{'='*70}")
        print("Loading RGB model...")
        print("="*70)
        from ultralytics import YOLO
        rgb_model = YOLO(str(RGB_WEIGHTS))
        print("  Loaded.\n")

        for tag, img_dir, lbl_dir in rgb_jobs:
            run_dataset(rgb_model, tag, img_dir, lbl_dir, OUTPUT_DIR)

        # Free GPU memory before loading IR model
        del rgb_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load IR model and run IR jobs
    if ir_jobs:
        print(f"\n{'='*70}")
        print("Loading IR model...")
        print("="*70)
        from ultralytics import YOLO
        ir_model = YOLO(str(IR_WEIGHTS))
        print("  Loaded.\n")

        for tag, img_dir, lbl_dir in ir_jobs:
            run_dataset(ir_model, tag, img_dir, lbl_dir, OUTPUT_DIR)

        del ir_model

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    total_frames = 0
    total_bytes = 0
    for tag, _, _, _ in jobs:
        p = OUTPUT_DIR / f"{tag}.json"
        if p.exists():
            sz = p.stat().st_size
            with open(p, "r") as f:
                n = len(json.load(f))
            total_frames += n
            total_bytes += sz
            print(f"  {tag:<25s} {n:>7d} frames  ({sz/1024/1024:.1f} MB)")
        else:
            print(f"  {tag:<25s} MISSING")

    print(f"\n  Total: {total_frames:,} frames, {total_bytes/1024/1024:.1f} MB")
    print("  All inference cached. Ready for build_reliability_dataset.py")


if __name__ == "__main__":
    main()
