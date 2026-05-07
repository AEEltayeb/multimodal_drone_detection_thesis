"""
consensus_filter.py — Run on vast.ai AFTER training.
Uses two models (original + retrained) to identify noisy/mislabeled images
via consensus: frames that BOTH models struggle with are likely bad data.

Logic:
  POSITIVES (has label):
    - Both miss → noisy/mislabeled → REMOVE
    - One detects, one misses → hard but learnable → KEEP
    - Both detect → good → KEEP

  NEGATIVES (empty label):
    - Both fire high conf (>0.5) → suspect unlabeled drone → REMOVE
    - Both fire low conf → valuable hard negative → KEEP
    - One fires, one doesn't → KEEP
    - Neither fires → easy negative → KEEP

Usage:
    python consensus_filter.py \
        --dataset /workspace/retrain_dataset \
        --model-a /workspace/best.pt \
        --model-b /workspace/runs/Yolo26n_retrained_v2/weights/best.pt \
        --output /workspace/retrain_dataset_clean

    # Dry run (just report, don't create new dataset):
    python consensus_filter.py \
        --dataset /workspace/retrain_dataset \
        --model-a /workspace/best.pt \
        --model-b /workspace/runs/Yolo26n_retrained_v2/weights/best.pt \
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def img_iter(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def is_empty_label(lbl_path: Path) -> bool:
    """Check if a label file is empty (= negative / no objects)."""
    if not lbl_path.exists():
        return True
    content = lbl_path.read_text().strip()
    return len(content) == 0


def has_detection(results, conf_thresh: float = 0.25) -> tuple[bool, float]:
    """Check if model produced any detection above threshold.
    Returns (has_det, max_conf)."""
    if len(results.boxes) == 0:
        return False, 0.0
    max_conf = float(results.boxes.conf.max())
    return max_conf >= conf_thresh, max_conf


def run_consensus(model_a: YOLO, model_b: YOLO, img_dir: Path, lbl_dir: Path,
                  conf_thresh: float = 0.25, suspect_neg_conf: float = 0.5,
                  batch_size: int = 64):
    """Run both models on all images and classify each frame.

    Returns dict with keys: 'keep', 'remove_noisy_pos', 'remove_suspect_neg'
    Each value is a list of (img_path, lbl_path, reason).
    """
    all_images = img_iter(img_dir)
    if not all_images:
        return {"keep": [], "remove_noisy_pos": [], "remove_suspect_neg": []}

    keep = []
    remove_noisy_pos = []
    remove_suspect_neg = []

    for i in range(0, len(all_images), batch_size):
        batch_paths = all_images[i:i+batch_size]
        str_paths = [str(p) for p in batch_paths]

        # Run both models
        results_a = model_a(str_paths, conf=conf_thresh, imgsz=640,
                            verbose=False, device=0)
        results_b = model_b(str_paths, conf=conf_thresh, imgsz=640,
                            verbose=False, device=0)

        for img_path, res_a, res_b in zip(batch_paths, results_a, results_b):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            is_neg = is_empty_label(lbl_path)

            det_a, conf_a = has_detection(res_a, conf_thresh)
            det_b, conf_b = has_detection(res_b, conf_thresh)

            if is_neg:
                # NEGATIVE frame logic
                if det_a and det_b and conf_a >= suspect_neg_conf and conf_b >= suspect_neg_conf:
                    remove_suspect_neg.append((
                        img_path, lbl_path,
                        f"both_fire_high (A={conf_a:.2f}, B={conf_b:.2f})"
                    ))
                else:
                    keep.append((img_path, lbl_path, "negative_ok"))
            else:
                # POSITIVE frame logic
                if not det_a and not det_b:
                    remove_noisy_pos.append((
                        img_path, lbl_path,
                        f"both_miss (A={conf_a:.2f}, B={conf_b:.2f})"
                    ))
                else:
                    keep.append((img_path, lbl_path, "positive_ok"))

        done = min(i + batch_size, len(all_images))
        if done % (batch_size * 10) == 0 or done == len(all_images):
            print(f"      {done:>7,}/{len(all_images):,}  "
                  f"keep={len(keep):,}  "
                  f"noisy_pos={len(remove_noisy_pos):,}  "
                  f"suspect_neg={len(remove_suspect_neg):,}")

    return {
        "keep": keep,
        "remove_noisy_pos": remove_noisy_pos,
        "remove_suspect_neg": remove_suspect_neg,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Consensus-based dataset cleaning using two models"
    )
    ap.add_argument("--dataset", type=Path, required=True,
                    help="Path to dataset (images/{train,val,test}, labels/{train,val,test})")
    ap.add_argument("--model-a", type=str, required=True,
                    help="Path to first model (e.g. original best.pt)")
    ap.add_argument("--model-b", type=str, required=True,
                    help="Path to second model (e.g. retrained best.pt)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory for cleaned dataset (default: dry run only)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    ap.add_argument("--suspect-conf", type=float, default=0.5,
                    help="Conf threshold for suspect mislabeled negatives (default: 0.5)")
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "val", "test"],
                    help="Which splits to clean (default: train only)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for inference (default: 64)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report only, don't create cleaned dataset")
    args = ap.parse_args()

    print("=" * 60)
    print("CONSENSUS DATASET FILTER")
    print("=" * 60)
    print(f"  Model A: {args.model_a}")
    print(f"  Model B: {args.model_b}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Splits:  {args.splits}")
    print(f"  Conf threshold: {args.conf}")
    print(f"  Suspect neg conf: {args.suspect_conf}")

    model_a = YOLO(args.model_a)
    model_b = YOLO(args.model_b)

    total_stats = Counter()
    all_removals = []  # For logging

    for split in ("train", "val", "test"):
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split

        if not img_dir.exists():
            continue

        n_images = len(img_iter(img_dir))

        if split in args.splits:
            print(f"\n{'='*60}")
            print(f"  CLEANING: {split} ({n_images:,} images)")
            print(f"{'='*60}")

            results = run_consensus(
                model_a, model_b, img_dir, lbl_dir,
                conf_thresh=args.conf,
                suspect_neg_conf=args.suspect_conf,
                batch_size=args.batch_size,
            )

            n_keep = len(results["keep"])
            n_noisy = len(results["remove_noisy_pos"])
            n_suspect = len(results["remove_suspect_neg"])
            n_total = n_keep + n_noisy + n_suspect

            print(f"\n  Results for {split}:")
            print(f"    Keep:              {n_keep:>7,} ({n_keep/n_total*100:.1f}%)")
            print(f"    Remove noisy pos:  {n_noisy:>7,} ({n_noisy/n_total*100:.1f}%)")
            print(f"    Remove suspect neg:{n_suspect:>7,} ({n_suspect/n_total*100:.1f}%)")

            total_stats["keep"] += n_keep
            total_stats["noisy_pos"] += n_noisy
            total_stats["suspect_neg"] += n_suspect

            # Show some examples
            if results["remove_noisy_pos"]:
                print(f"\n    Sample noisy positives (both models miss):")
                for img, lbl, reason in results["remove_noisy_pos"][:5]:
                    print(f"      {img.name} — {reason}")
                if n_noisy > 5:
                    print(f"      ... and {n_noisy-5} more")

            if results["remove_suspect_neg"]:
                print(f"\n    Sample suspect negatives (both fire high conf):")
                for img, lbl, reason in results["remove_suspect_neg"][:5]:
                    print(f"      {img.name} — {reason}")
                if n_suspect > 5:
                    print(f"      ... and {n_suspect-5} more")

            for img, lbl, reason in results["remove_noisy_pos"] + results["remove_suspect_neg"]:
                all_removals.append([img.name, split, reason, str(img)])

            # Build cleaned dataset
            if args.output and not args.dry_run:
                out_img = args.output / "images" / split
                out_lbl = args.output / "labels" / split
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                for img, lbl, _ in results["keep"]:
                    dst_img = out_img / img.name
                    dst_lbl = out_lbl / lbl.name
                    if not dst_img.exists():
                        try:
                            dst_img.symlink_to(img.resolve())
                        except OSError:
                            shutil.copy2(img, dst_img)
                    if not dst_lbl.exists():
                        if lbl.exists():
                            try:
                                dst_lbl.symlink_to(lbl.resolve())
                            except OSError:
                                shutil.copy2(lbl, dst_lbl)
                        else:
                            dst_lbl.write_text("")
        else:
            # Non-cleaned splits: copy as-is
            print(f"\n  PASSTHROUGH: {split} ({n_images:,} images, not cleaned)")
            total_stats["keep"] += n_images
            if args.output and not args.dry_run:
                out_img = args.output / "images" / split
                out_lbl = args.output / "labels" / split
                out_img.mkdir(parents=True, exist_ok=True)
                out_lbl.mkdir(parents=True, exist_ok=True)
                for img in img_iter(img_dir):
                    lbl = lbl_dir / (img.stem + ".txt")
                    dst_img = out_img / img.name
                    dst_lbl = out_lbl / lbl.name
                    if not dst_img.exists():
                        try:
                            dst_img.symlink_to(img.resolve())
                        except OSError:
                            shutil.copy2(img, dst_img)
                    if not dst_lbl.exists():
                        if lbl.exists():
                            try:
                                dst_lbl.symlink_to(lbl.resolve())
                            except OSError:
                                shutil.copy2(lbl, dst_lbl)
                        else:
                            dst_lbl.write_text("")

    # Write data.yaml for cleaned dataset
    if args.output and not args.dry_run:
        data_yaml = f"""\
# Consensus-cleaned dataset
path: {args.output}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['drone']
"""
        (args.output / "data.yaml").write_text(data_yaml)

    # Write removal log
    if all_removals:
        log_path = args.output / "removed.csv" if args.output and not args.dry_run else args.dataset / "consensus_removed.csv"
        if not args.dry_run or True:  # Always write the log
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["filename", "split", "reason", "original_path"])
                w.writerows(all_removals)
            print(f"\n  Removal log: {log_path} ({len(all_removals)} rows)")

    # Final summary
    print(f"\n{'='*60}")
    print(f"CONSENSUS FILTER SUMMARY")
    print(f"{'='*60}")
    total = sum(total_stats.values())
    print(f"  Keep:              {total_stats['keep']:>7,}")
    print(f"  Remove noisy pos:  {total_stats['noisy_pos']:>7,}")
    print(f"  Remove suspect neg:{total_stats['suspect_neg']:>7,}")
    print(f"  Total removed:     {total_stats['noisy_pos']+total_stats['suspect_neg']:>7,} "
          f"({(total_stats['noisy_pos']+total_stats['suspect_neg'])/total*100:.1f}%)")

    if args.dry_run:
        print("\n  *** DRY RUN — no dataset created ***")
        print("  Re-run without --dry-run and with --output to create cleaned dataset")
    elif args.output:
        print(f"\n  Cleaned dataset: {args.output}")
        print(f"  Next: python train_rgb_v2.py --data {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
