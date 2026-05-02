"""
Usage:
    python merge_datasets.py \
        --gold /workspace/IR_dset_gold_duplicates_removed \
        --source /workspace/IR_dsetV9b1 \
        --output /workspace/IR_dset_final \
        --flagged /workspace/consensus_results/flagged_images.csv \
        --duplicates /workspace/IR_dsetV9b1/duplicates.csv/duplicates.csv \
        --dry-run
"""

import argparse
import csv
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_flagged(csv_path):
    """Load flagged filenames from consensus cleaning."""
    flagged = set()
    if not csv_path or not csv_path.exists():
        return flagged
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flagged.add(row["filename"])
    return flagged


def load_duplicates(csv_path):
    """Load duplicate filenames to remove."""
    dups = set()
    if not csv_path or not csv_path.exists():
        return dups
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("action") == "remove":
                dups.add(row["filename"])
    return dups


def scan_dataset(root, splits):
    """Scan dataset, return {filename: (img_path, lbl_path, split)}."""
    images = {}
    root = Path(root)
    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                lbl = lbl_dir / f"{f.stem}.txt"
                images[f.name] = (f, lbl, split)
    return images


def main():
    parser = argparse.ArgumentParser(description="Merge dsetV9b1 into gold")
    parser.add_argument("--gold", required=True, type=Path,
                        help="Gold dataset root (base)")
    parser.add_argument("--source", required=True, type=Path,
                        help="Source dataset to merge from (dsetV9b1)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output merged dataset")
    parser.add_argument("--flagged", type=Path, default=None,
                        help="flagged_images.csv from consensus cleaning")
    parser.add_argument("--duplicates", type=Path, default=None,
                        help="duplicates.csv from dedup")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load exclusion lists
    flagged = load_flagged(args.flagged)
    print(f"Flagged (consensus): {len(flagged):,}")

    duplicates = load_duplicates(args.duplicates)
    print(f"Duplicates to skip:  {len(duplicates):,}")

    exclude = flagged | duplicates
    print(f"Total excluded:      {len(exclude):,}")

    # Scan gold dataset
    print(f"\nScanning gold: {args.gold}")
    gold_images = scan_dataset(args.gold, args.splits)
    print(f"  Gold images: {len(gold_images):,}")

    # Scan source dataset
    print(f"\nScanning source: {args.source}")
    source_images = scan_dataset(args.source, args.splits)
    print(f"  Source images: {len(source_images):,}")

    # Find new images: in source but not in gold, and not excluded
    new_images = {}
    skipped_exists = 0
    skipped_flagged = 0
    skipped_dup = 0

    for fname, (img_path, lbl_path, split) in source_images.items():
        if fname in gold_images:
            skipped_exists += 1
            continue
        if fname in flagged:
            skipped_flagged += 1
            continue
        if fname in duplicates:
            skipped_dup += 1
            continue
        new_images[fname] = (img_path, lbl_path, split)

    print(f"\n  Already in gold:     {skipped_exists:,}")
    print(f"  Flagged (consensus): {skipped_flagged:,}")
    print(f"  Duplicates:          {skipped_dup:,}")
    print(f"  New to add:          {len(new_images):,}")

    # Create output dataset
    if not args.dry_run:
        # First copy gold entirely
        print(f"\nCopying gold to output: {args.output}")
        for split in args.splits:
            (args.output / split / "images").mkdir(parents=True, exist_ok=True)
            (args.output / split / "labels").mkdir(parents=True, exist_ok=True)

        copied = 0
        for fname, (img_path, lbl_path, split) in gold_images.items():
            # Skip gold images that were flagged
            if fname in flagged:
                continue
            dst_img = args.output / split / "images" / fname
            dst_lbl = args.output / split / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(str(img_path), str(dst_img))
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(dst_lbl))
            else:
                dst_lbl.write_text("")
            copied += 1
            if copied % 10000 == 0:
                print(f"    Copied {copied:,} gold images...")

        print(f"  Copied {copied:,} gold images")

        # Then add new source images
        added = 0
        for fname, (img_path, lbl_path, split) in new_images.items():
            dst_img = args.output / split / "images" / fname
            dst_lbl = args.output / split / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(str(img_path), str(dst_img))
            if lbl_path.exists():
                shutil.copy2(str(lbl_path), str(dst_lbl))
            else:
                dst_lbl.write_text("")
            added += 1
            if added % 10000 == 0:
                print(f"    Added {added:,} new images...")

        print(f"  Added {added:,} new images from source")

        # Write dataset.yaml
        yaml_content = f"""path: {args.output}
train: train/images
val: val/images
test: test/images

names:
  0: drone
"""
        (args.output / "dataset.yaml").write_text(yaml_content)

    # Summary
    per_split = defaultdict(lambda: {"gold": 0, "new": 0})
    for fname, (_, _, split) in gold_images.items():
        if fname not in flagged:
            per_split[split]["gold"] += 1
    for fname, (_, _, split) in new_images.items():
        per_split[split]["new"] += 1

    total = sum(v["gold"] + v["new"] for v in per_split.values())

    print(f"\n{'='*60}")
    print(f"MERGED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {total:,}")
    for split in args.splits:
        g = per_split[split]["gold"]
        n = per_split[split]["new"]
        print(f"  {split:>5}: {g + n:>7,}  (gold: {g:,}  new: {n:,})")

    if args.dry_run:
        print(f"\n  Dry run. Use without --dry-run to build dataset.")
    else:
        print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
