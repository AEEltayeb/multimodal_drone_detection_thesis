r"""
Merge IR_dsetV5 + Svanström IR_video_ir_dataset → IR_dsetV6.

Strategy:
  - dsetV5: keep existing train/val/test splits as-is (already leak-free)
  - Svanström: keep existing train/val/test splits as-is (already video-level split)
  - Copy images+labels into G:\drone\IR_dsetV6\{train,val,test}\{images,labels}
  - Prefix filenames: dv5_<name> and svan_<name> to avoid collisions

Usage:
  python scripts/dataset/merge_dsetV6.py --dry-run     # preview counts
  python scripts/dataset/merge_dsetV6.py               # actually copy
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

# ── Paths ──
DSETV5_DIR = Path(r"G:\drone\IR_dsetV5")
SVANSTROM_DIR = Path(r"G:\drone\IR_video_ir_dataset")
OUTPUT_DIR = Path(r"G:\drone\IR_dsetV6")

SEED = 42


def collect_dataset_files(dataset_dir: Path, prefix: str, source_name: str) -> dict:
    """Collect files from a dataset organized as {split}/images + {split}/labels."""
    result = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        img_dir = dataset_dir / split / "images"

        # Also check images/{split} layout (dsetV4-style)
        if not img_dir.exists():
            img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            print(f"  [WARN] {source_name} {split} images not found")
            continue

        # Find the corresponding labels directory
        lbl_dir = dataset_dir / split / "labels"
        if not lbl_dir.exists():
            lbl_dir = dataset_dir / "labels" / split

        # Pre-scan labels for fast lookup
        existing_labels = set()
        if lbl_dir.exists():
            existing_labels = {f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt"}

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                result[split].append({
                    "img_src": img_file,
                    "lbl_src": lbl_file if img_file.stem in existing_labels else None,
                    "out_name": f"{prefix}_{img_file.stem}",
                    "source": source_name,
                })

    return result


def copy_files(file_list: list, split: str, output_dir: Path, dry_run: bool):
    """Copy image+label pairs to output directory."""
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for entry in file_list:
        out_name = entry["out_name"]
        img_src = entry["img_src"]
        lbl_src = entry["lbl_src"]

        if not dry_run:
            # Copy image
            shutil.copy2(str(img_src), str(img_out / f"{out_name}{img_src.suffix}"))

            # Copy or create empty label
            if lbl_src and lbl_src.exists():
                shutil.copy2(str(lbl_src), str(lbl_out / f"{out_name}.txt"))
            else:
                (lbl_out / f"{out_name}.txt").write_text("")

        copied += 1

    return copied


def count_positives(file_list: list) -> int:
    """Count files that have non-empty labels (positive samples)."""
    count = 0
    for entry in file_list:
        if entry["lbl_src"] and entry["lbl_src"].exists():
            content = entry["lbl_src"].read_text().strip()
            if content:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Merge dsetV5 + Svanström → IR_dsetV6")
    parser.add_argument("--dsetv5", type=Path, default=DSETV5_DIR)
    parser.add_argument("--svanstrom", type=Path, default=SVANSTROM_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute stats, don't copy files")
    args = parser.parse_args()

    print(f"dsetV5:    {args.dsetv5}")
    print(f"Svanström: {args.svanstrom}")
    print(f"Output:    {args.output}")

    # Step 1: Collect dsetV5 files (keep existing splits)
    print("\n[1/3] Collecting dsetV5 files...")
    dv5_files = collect_dataset_files(args.dsetv5, "dv5", "dsetV5")
    for split, files in dv5_files.items():
        print(f"  dsetV5 {split}: {len(files):,} files")

    # Step 2: Collect Svanström files (keep existing splits)
    print("\n[2/3] Collecting Svanström files...")
    svan_files = collect_dataset_files(args.svanstrom, "svan", "svanstrom")
    for split, files in svan_files.items():
        print(f"  Svanström {split}: {len(files):,} files")

    # Step 3: Merge and copy
    action = "DRY RUN — " if args.dry_run else ""
    print(f"\n[3/3] {action}Copying files to {args.output}...")
    total_stats = {}

    for split in ["train", "val", "test"]:
        merged = dv5_files[split] + svan_files[split]
        n_dv5 = len(dv5_files[split])
        n_svan = len(svan_files[split])

        copied = copy_files(merged, split, args.output, args.dry_run)
        total_stats[split] = {
            "total": copied,
            "dsetV5": n_dv5,
            "svanstrom": n_svan,
        }

        print(f"  {split}: {copied:,} files (dsetV5={n_dv5:,}, svanstrom={n_svan:,})")

    # Write dataset.yaml
    if not args.dry_run:
        yaml_content = f"""# IR_dsetV6 — dsetV5 + Svanström merged dataset
# Sources: dsetV5 (7-source IR) + Svanström must-cite (IR videos, 8th source)
# Svanström non-drone classes (airplane, bird, helicopter) included as hard negatives
# Seed: {args.seed}
# Class 0 = drone (UAV)

path: {args.output}

train: train/images
val: val/images
test: test/images

nc: 1
names: ['drone']
"""
        (args.output / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
        print(f"\n  dataset.yaml written")

    # Write split manifest
    if not args.dry_run:
        manifest = {
            "seed": args.seed,
            "sources": {
                "dsetV5": {
                    "path": str(args.dsetv5),
                    "strategy": "preserve_existing_splits",
                    "description": "7-source IR dataset (Gold V2, May22, Roboflow, VarEnv, "
                                   "Small Objects, DroneDetect IR, Bird negatives, 3rd Anti-UAV)",
                },
                "svanstrom": {
                    "path": str(args.svanstrom),
                    "strategy": "preserve_existing_splits",
                    "description": "Svanström must-cite dataset. IR_DRONE = positive, "
                                   "IR_AIRPLANE/BIRD/HELICOPTER = hard negatives (empty labels)",
                },
            },
            "splits": total_stats,
        }
        (args.output / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"  split_manifest.json written")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — IR_dsetV6")
    print(f"{'='*60}")
    grand_total = sum(s["total"] for s in total_stats.values())
    for split, s in total_stats.items():
        pct = s["total"] / grand_total * 100
        print(f"  {split:5s}: {s['total']:>7,} ({pct:5.1f}%)  "
              f"[dsetV5={s['dsetV5']:>6,}, svanstrom={s['svanstrom']:>6,}]")
    print(f"  {'total':5s}: {grand_total:>7,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
