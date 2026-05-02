

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(source_root: Path):
    """Discover images and return list of (img_path, lbl_path, split).
    Handles both split layout (train/images/) and flat layout (images/)."""
    results = []

    # Try split layout first
    for split in ["train", "val", "test"]:
        img_dir = source_root / split / "images"
        lbl_dir = source_root / split / "labels"
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                lbl = lbl_dir / f"{f.stem}.txt"
                results.append((f, lbl if lbl.exists() else None, split))

    # Flat layout fallback
    if not results:
        img_dir = source_root / "images"
        lbl_dir = source_root / "labels"
        if img_dir.exists():
            all_imgs = sorted([f for f in img_dir.iterdir()
                               if f.suffix.lower() in IMG_EXTS])
            # Split 80/10/10
            random.seed(42)
            random.shuffle(all_imgs)
            n = len(all_imgs)
            n_val = max(1, n // 10)
            n_test = max(1, n // 10)
            n_train = n - n_val - n_test

            for i, f in enumerate(all_imgs):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"
                lbl = lbl_dir / f"{f.stem}.txt"
                results.append((f, lbl if lbl.exists() else None, split))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Merge supplementary datasets into gold dataset"
    )
    parser.add_argument("--target", type=Path, required=True,
                        help="Target gold dataset root")
    parser.add_argument("--sources", type=Path, nargs="+", required=True,
                        help="Source dataset directories to merge")
    parser.add_argument("--prefixes", nargs="+", required=True,
                        help="Filename prefixes for each source (to avoid collisions)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if len(args.sources) != len(args.prefixes):
        print("ERROR: --sources and --prefixes must have same length")
        return

    target = args.target
    print(f"Target: {target}")
    print(f"Dry run: {args.dry_run}\n")

    # Count existing
    for split in ["train", "val", "test"]:
        d = target / split / "images"
        if d.exists():
            n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
            print(f"  Existing {split}: {n:,}")

    total_copied = 0
    split_counts = defaultdict(int)

    for source, prefix in zip(args.sources, args.prefixes):
        print(f"\n{'='*60}")
        print(f"Source: {source}")
        print(f"Prefix: {prefix}_")
        print(f"{'='*60}")

        images = discover_images(source)
        print(f"  Found {len(images)} images")

        src_splits = defaultdict(int)
        for img, lbl, split in images:
            src_splits[split] += 1
        for s, c in sorted(src_splits.items()):
            print(f"    {s}: {c}")

        if args.dry_run:
            for s, c in src_splits.items():
                split_counts[s] += c
            total_copied += len(images)
            continue

        for img, lbl, split in images:
            # Add prefix to avoid name collisions
            new_name = f"{prefix}_{img.name}"
            dst_img = target / split / "images" / new_name
            dst_lbl = target / split / "labels" / f"{prefix}_{img.stem}.txt"

            # Ensure dirs exist
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(str(img), str(dst_img))
            if lbl and lbl.exists():
                shutil.copy2(str(lbl), str(dst_lbl))
            else:
                dst_lbl.write_text("")  # negative

            split_counts[split] += 1
            total_copied += 1

        print(f"  Copied {len(images)} images")

    print(f"\n{'='*60}")
    print(f"MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total added: {total_copied:,}")
    for s in ["train", "val", "test"]:
        print(f"    {s}: +{split_counts[s]:,}")

    if args.dry_run:
        print(f"\n  Dry run — no files copied.")
    else:
        # Count final totals
        print(f"\n  Final dataset:")
        for split in ["train", "val", "test"]:
            d = target / split / "images"
            if d.exists():
                n = sum(1 for f in d.iterdir() if f.suffix.lower() in IMG_EXTS)
                print(f"    {split}: {n:,}")


if __name__ == "__main__":
    main()
