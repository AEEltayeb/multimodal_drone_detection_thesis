"""
merge_and_mine.py — Run on vast.ai. Mines hard negatives from pre-split
confusers and merges them into the original dataset, split-by-split.

The confusers_merged/ folder is ALREADY split into train/val/test
(by collect_confusers.py with video-level splitting to prevent leakage).
This script simply:
    1. Runs the original model on ALL confuser images
    2. Keeps only hallucination frames (or all, with --skip-mining)
    3. Merges train->train, val->val, test->test

Usage:
    python merge_and_mine.py \
        --drone-dataset /workspace/dataset/dataset \
        --confusers /workspace/confusers_merged \
        --model /workspace/best.pt \
        --output /workspace/retrain_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def short_name(prefix: str, src: Path) -> str:
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{h}{src.suffix.lower()}"


def img_iter(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


# ─── HARD NEGATIVE MINING ────────────────────────────────────────

def mine_split(model: YOLO, img_dir: Path, conf_thresh: float,
               batch_size: int = 64) -> tuple[list[Path], list[Path]]:
    """Run model on images in a directory.
    Returns (hallucinated, passed).
    - hallucinated: model fires (conf >= conf_thresh) -> hard negatives
    - passed: model doesn't fire -> easy negatives, not used
    """
    all_images = img_iter(img_dir)
    if not all_images:
        return [], []

    hallucinated = []
    passed = []

    for i in range(0, len(all_images), batch_size):
        batch_paths = all_images[i:i+batch_size]
        results = model(
            [str(p) for p in batch_paths],
            conf=conf_thresh,
            imgsz=640,
            verbose=False,
            device=0,
        )
        for img_path, result in zip(batch_paths, results):
            if len(result.boxes) > 0:
                hallucinated.append(img_path)
            else:
                passed.append(img_path)

        done = min(i + batch_size, len(all_images))
        if done % (batch_size * 10) == 0 or done == len(all_images):
            print(f"      {done:>7,}/{len(all_images):,}  "
                  f"hard_neg={len(hallucinated):,}  "
                  f"passed={len(passed):,}")

    return hallucinated, passed


# ─── MERGE ────────────────────────────────────────────────────────

def merge_dataset(drone_dataset: Path, confuser_dir: Path,
                  model_path: str, output: Path,
                  mine_conf: float, batch_size: int, skip_mining: bool):
    """Merge original drone dataset with confuser negatives, split-by-split."""

    print(f"\n{'='*60}")
    print(f"MERGE AND MINE")
    print(f"{'='*60}")

    # Create output directories
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = [["filename", "split", "type", "source"]]

    # Load model for mining
    model = None
    if not skip_mining:
        print(f"\n  Loading model for mining: {model_path}")
        model = YOLO(model_path)

    total_pos = Counter()
    total_neg = Counter()

    for split in ("train", "val", "test"):
        print(f"\n{'='*60}")
        print(f"  SPLIT: {split}")
        print(f"{'='*60}")

        # 1. Link original positives
        orig_img_dir = drone_dataset / "images" / split
        orig_lbl_dir = drone_dataset / "labels" / split
        orig_imgs = img_iter(orig_img_dir)
        print(f"  Original positives: {len(orig_imgs):,}")

        for img in orig_imgs:
            dst_img = output / "images" / split / img.name
            dst_lbl = output / "labels" / split / (img.stem + ".txt")
            src_lbl = orig_lbl_dir / (img.stem + ".txt")

            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst_img)

            if not dst_lbl.exists():
                if src_lbl.exists():
                    try:
                        dst_lbl.symlink_to(src_lbl.resolve())
                    except OSError:
                        shutil.copy2(src_lbl, dst_lbl)
                else:
                    dst_lbl.write_text("")

            manifest.append([img.name, split, "positive", str(img)])
        total_pos[split] = len(orig_imgs)

        # 2. Mine and merge negatives for this split
        confuser_split_dir = confuser_dir / "images" / split
        if not confuser_split_dir.exists():
            print(f"  No confuser {split} split found, skipping negatives")
            continue

        if skip_mining:
            # Include ALL confuser images
            neg_imgs = img_iter(confuser_split_dir)
            print(f"  Confuser negatives (all, no mining): {len(neg_imgs):,}")
        else:
            # Mine: keep only hallucinations
            print(f"  Mining confuser {split} images (conf > {mine_conf})...")
            hallucinated, passed = mine_split(
                model, confuser_split_dir, mine_conf, batch_size=batch_size)
            neg_imgs = hallucinated
            total_scanned = len(hallucinated) + len(passed)
            print(f"  Hard negatives: {len(hallucinated):,} / {total_scanned:,} "
                  f"({len(hallucinated)/total_scanned*100:.1f}%)")

        # Copy negatives with empty labels
        n_neg = 0
        for img in neg_imgs:
            dst_name = short_name("neg", img)
            dst_img = output / "images" / split / dst_name
            dst_lbl = output / "labels" / split / (Path(dst_name).stem + ".txt")

            if not dst_img.exists():
                shutil.copy2(img, dst_img)
                dst_lbl.write_text("")

            manifest.append([dst_name, split, "negative", str(img)])
            n_neg += 1

        total_neg[split] = n_neg
        print(f"  -> Merged: {len(orig_imgs):,} pos + {n_neg:,} neg = "
              f"{len(orig_imgs)+n_neg:,} total  "
              f"(neg={n_neg/(len(orig_imgs)+n_neg)*100:.1f}%)")

    # Write data.yaml
    data_yaml = f"""\
# Auto-generated by merge_and_mine.py
# Original drone dataset + mined hard negatives
path: {output}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['drone']
"""
    (output / "data.yaml").write_text(data_yaml)

    # Write manifest
    mf_path = output / "manifest.csv"
    with mf_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(manifest)

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL DATASET")
    print(f"{'='*60}")
    grand_pos = sum(total_pos.values())
    grand_neg = sum(total_neg.values())
    grand_total = grand_pos + grand_neg
    for split in ("train", "val", "test"):
        p = total_pos[split]
        n = total_neg[split]
        t = p + n
        pct = n / t * 100 if t > 0 else 0
        split_pct = t / grand_total * 100 if grand_total > 0 else 0
        print(f"  {split:<6s}: {t:>8,}  (pos={p:>7,}  neg={n:>6,}  "
              f"neg%={pct:5.1f}%  split%={split_pct:5.1f}%)")
    print(f"  {'TOTAL':<6s}: {grand_total:>8,}  "
          f"(pos={grand_pos:>7,}  neg={grand_neg:>6,})")
    print(f"\n  data.yaml: {output / 'data.yaml'}")
    print(f"  manifest:  {mf_path} ({len(manifest)-1:,} rows)")


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Mine hard negatives from pre-split confusers and merge into dataset"
    )
    ap.add_argument("--drone-dataset", type=Path, required=True,
                    help="Path to original drone dataset (images/{train,val,test})")
    ap.add_argument("--confusers", type=Path, required=True,
                    help="Path to confusers_merged/ (pre-split by collect_confusers.py)")
    ap.add_argument("--model", type=str, required=True,
                    help="Path to Yolo26n_trained best.pt for mining")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output directory for merged dataset")
    ap.add_argument("--mine-conf", type=float, default=0.25,
                    help="Confidence threshold for hallucination (default: 0.25)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch size for mining inference (default: 64)")
    ap.add_argument("--skip-mining", action="store_true",
                    help="Skip mining, include ALL confuser images")
    args = ap.parse_args()

    merge_dataset(
        drone_dataset=args.drone_dataset,
        confuser_dir=args.confusers,
        model_path=args.model,
        output=args.output,
        mine_conf=args.mine_conf,
        batch_size=args.batch_size,
        skip_mining=args.skip_mining,
    )

    print("\nDone! Next: python train_rgb_v2.py")


if __name__ == "__main__":
    main()
