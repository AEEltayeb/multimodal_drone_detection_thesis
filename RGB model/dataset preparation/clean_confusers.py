"""
clean_confusers.py — Run on vast.ai AFTER extracting the confusers archive.
Removes:
  1. ALL YouTube eval images (IR, not RGB) — prefix: yt_*
  2. Specific mislabeled images (real drones in confuser sources)

Run BEFORE merge_and_mine.py.

Usage:
    python clean_confusers.py --confusers /workspace/rgb_confusers_merged
    python clean_confusers.py --confusers /workspace/rgb_confusers_merged --dry-run
"""

from __future__ import annotations
import argparse
from pathlib import Path

# ── Known mislabeled images (real drones in confuser sources) ─────
# Identified by manual review of the collected dataset.
MISLABELED_FILES = [
    # Test split — Airplane Roboflow
    ("test", "airplane_test__ae740e9591.jpg"),
    # Train split — Airplane Roboflow
    ("train", "airplane_train__1501aedd1e.jpg"),
    ("train", "airplane_train__23f355889c.jpg"),
    ("train", "airplane_train__830950fb67.jpg"),
    ("train", "airplane_train__86eccd271b.jpg"),
    ("train", "airplane_train__b9c07fe3aa.jpg"),
    ("train", "airplane_train__c17241d610.jpg"),
    ("train", "airplane_train__e491f92a24.jpg"),
    # Train split — New_Dataset
    ("train", "newds_train__4f34be8cae.jpg"),
    ("train", "newds_train__7cff1b86eb.jpg"),
]


def remove_file(path: Path, dry_run: bool) -> bool:
    if path.exists():
        if not dry_run:
            path.unlink()
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Clean confuser archive on vast.ai")
    ap.add_argument("--confusers", type=Path, required=True,
                    help="Path to extracted confusers directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be deleted without deleting")
    args = ap.parse_args()

    root = args.confusers
    if not root.exists():
        print(f"ERROR: {root} does not exist")
        return

    print("=" * 60)
    print(f"Cleaning confusers: {root}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    print("=" * 60)

    total_removed = 0

    # 1. Remove ALL YouTube eval images (IR, not RGB)
    print("\n[1] Removing YouTube eval images (yt_* prefix, IR modality)...")
    yt_removed = 0
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.name.startswith("yt_"):
                remove_file(f, args.dry_run)
                # Also remove corresponding label
                lbl = lbl_dir / (f.stem + ".txt")
                remove_file(lbl, args.dry_run)
                yt_removed += 1
    print(f"    Removed {yt_removed} YouTube images + labels")
    total_removed += yt_removed

    # 2. Remove specific mislabeled images (real drones)
    print("\n[2] Removing known mislabeled images (real drones)...")
    ml_removed = 0
    for split, filename in MISLABELED_FILES:
        img_path = root / "images" / split / filename
        lbl_path = root / "labels" / split / (Path(filename).stem + ".txt")
        if remove_file(img_path, args.dry_run):
            remove_file(lbl_path, args.dry_run)
            ml_removed += 1
            print(f"    [{split}] {filename}")
        else:
            print(f"    [{split}] {filename} — NOT FOUND (already removed?)")
    total_removed += ml_removed

    # 3. Recount
    print(f"\n{'='*60}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*60}")
    print(f"  Total removed: {total_removed}")
    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        if img_dir.exists():
            count = len([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
            print(f"  {split}: {count:,} images remaining")

    if args.dry_run:
        print("\n  *** Nothing was actually deleted (dry run) ***")
    print(f"\nNext: python merge_and_mine.py ...")


if __name__ == "__main__":
    main()
