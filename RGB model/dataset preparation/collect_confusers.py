"""
collect_confusers.py — Gather ALL confuser / hard-negative sources from G: drive
into a single output folder, already split into train/val/test.

Sources WITH existing splits (Airplane, New_Dataset):
    -> merge train->train, valid->val, test->test

Sources WITHOUT splits (Svanstrom, Helicopter, raihanrsd):
    -> split by video/sequence ID into 80/10/10

NOTE: YouTube eval confusers EXCLUDED -- those are IR images, not RGB.

Output layout:
    <output>/
        images/{train,val,test}/<prefix>__<hash>.jpg
        labels/{train,val,test}/<prefix>__<hash>.txt   (ALL empty)
        manifest.csv

Usage:
    python "RGB model/dataset preparation/collect_confusers.py" --dry-run
    python "RGB model/dataset preparation/collect_confusers.py"
    python "RGB model/dataset preparation/collect_confusers.py" --raihanrsd-limit 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict

# ─── SOURCE PATHS (G: drive) ─────────────────────────────────────

AIRPLANE_DS  = Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb")
NEW_DS       = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb")
HELI_DS      = Path(r"G:/drone/Helicopter-kaggle-dataset/Helicopter Class 1")
SVAN_RGB     = Path(r"G:/drone/svanstrom_paired/RGB/images")
RAIHANRSD    = Path(r"G:/drone/raihanrsd_drone-bird-frames_kaggle/New_Data/RGB")
# YouTube eval confusers EXCLUDED -- they are IR images, not RGB

OUT_ROOT     = Path(r"G:/drone/confusers_merged")

# raihanrsd has 20K bird frames — way too many, original dataset already
# has bird negatives. Cap at 1-2K.
RAIHANRSD_LIMIT = 1500

# New_Dataset class mapping: 0=Airplane, 1=Bird, 2=Drone, 3=Helicopter, 4=tree
NEW_DS_DRONE_CLASS = 2

SVAN_CONFUSER_CATS = {"AIRPLANE", "BIRD", "HELICOPTER"}
SPLIT_RATIO = {"train": 0.80, "val": 0.10, "test": 0.10}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)


# ─── HELPERS ──────────────────────────────────────────────────────

def short_name(prefix: str, src: Path) -> str:
    """Compact destination filename: prefix__<10-char hash>.ext."""
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}__{h}{src.suffix.lower()}"


def img_iter(d: Path):
    if not d.exists():
        print(f"  [SKIP] {d} does not exist")
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def img_iter_recursive(d: Path):
    if not d.exists():
        print(f"  [SKIP] {d} does not exist")
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def label_has_class(lbl_path: Path, target_cls: int) -> bool:
    if not lbl_path or not lbl_path.exists():
        return False
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            if int(parts[0]) == target_cls:
                return True
        except ValueError:
            continue
    return False


def svan_video_id(stem: str) -> str:
    """Extract video sequence ID from Svanstrom stem.
    e.g. IR_AIRPLANE_001_f000000_visible -> AIRPLANE_001"""
    # Pattern: IR_<CATEGORY>_<NNN>_f<FFFFFF>_visible
    m = re.match(r"IR_([A-Z]+_\d+)_f\d+", stem)
    return m.group(1) if m else stem





def raihanrsd_video_id(path: Path) -> str:
    """Extract video folder name from raihanrsd path."""
    # Path structure: .../RGB/video_00160_001_054/frame.jpg
    for parent in path.parents:
        if parent.name.startswith("video_"):
            return parent.name
    return "unknown"


def split_by_groups(groups: dict[str, list[Path]], ratios=SPLIT_RATIO,
                    seed=42) -> dict[str, list[Path]]:
    """Split a dict of {group_id: [images]} into train/val/test by GROUP.
    Entire groups go to one split to prevent leakage."""
    rng = random.Random(seed)
    group_ids = sorted(groups.keys())
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = max(1, int(n * ratios["train"]))
    n_val = max(1, int(n * ratios["val"]))

    train_ids = set(group_ids[:n_train])
    val_ids = set(group_ids[n_train:n_train + n_val])
    test_ids = set(group_ids[n_train + n_val:])

    result = {"train": [], "val": [], "test": []}
    for gid, imgs in groups.items():
        if gid in train_ids:
            result["train"].extend(imgs)
        elif gid in val_ids:
            result["val"].extend(imgs)
        else:
            result["test"].extend(imgs)

    return result


def copy_image(src: Path, dst_img: Path, dst_lbl: Path, dry_run: bool) -> bool:
    if dst_img.exists():
        return False
    if dry_run:
        return True
    shutil.copy2(src, dst_img)
    dst_lbl.write_text("")  # empty label = no objects
    return True


# ─── COLLECTORS ───────────────────────────────────────────────────

def collect_airplane(out_root: Path, manifest: list, dry_run: bool) -> Counter:
    """Airplane Roboflow: has train/valid/test splits -> respect them."""
    counts = Counter()
    split_map = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in split_map.items():
        img_dir = AIRPLANE_DS / src_split / "images"
        out_img = out_root / "images" / dst_split
        out_lbl = out_root / "labels" / dst_split
        for img in img_iter(img_dir):
            dst_name = short_name(f"airplane_{src_split}", img)
            if copy_image(img, out_img / dst_name,
                          out_lbl / (Path(dst_name).stem + ".txt"), dry_run):
                counts[dst_split] += 1
                manifest.append([dst_name, "airplane", dst_split, str(img)])
    return counts


def collect_new_dataset(out_root: Path, manifest: list, dry_run: bool) -> Counter:
    """New_Dataset: has train/valid/test splits, exclude drone class (2)."""
    counts = Counter()
    split_map = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in split_map.items():
        img_dir = NEW_DS / src_split / "images"
        lbl_dir = NEW_DS / src_split / "labels"
        for img in img_iter(img_dir):
            lbl = lbl_dir / (img.stem + ".txt")
            if label_has_class(lbl, NEW_DS_DRONE_CLASS):
                continue
            dst_name = short_name(f"newds_{src_split}", img)
            out_img = out_root / "images" / dst_split
            out_lbl = out_root / "labels" / dst_split
            if copy_image(img, out_img / dst_name,
                          out_lbl / (Path(dst_name).stem + ".txt"), dry_run):
                counts[dst_split] += 1
                manifest.append([dst_name, "new_dataset", dst_split, str(img)])
    return counts


def collect_helicopter(out_root: Path, manifest: list, dry_run: bool) -> Counter:
    """Helicopter Kaggle: flat folder -> split by image hash (80/10/10)."""
    counts = Counter()
    all_imgs = img_iter(HELI_DS)
    # No video structure — use image hash to deterministically split
    groups = defaultdict(list)
    # Create ~20 pseudo-groups to split by (gives decent 80/10/10)
    for img in all_imgs:
        bucket = int(hashlib.md5(img.name.encode()).hexdigest()[:2], 16) % 20
        groups[str(bucket)].append(img)
    splits = split_by_groups(groups)
    for split, imgs in splits.items():
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        for img in imgs:
            dst_name = short_name("heli", img)
            if copy_image(img, out_img / dst_name,
                          out_lbl / (Path(dst_name).stem + ".txt"), dry_run):
                counts[split] += 1
                manifest.append([dst_name, "helicopter_kaggle", split, str(img)])
    return counts


def collect_svanstrom(out_root: Path, manifest: list, dry_run: bool) -> Counter:
    """Svanstrom RGB: AIRPLANE/BIRD/HELICOPTER only. Split by VIDEO SEQUENCE ID
    to prevent data leakage (eval uses all Svanstrom frames)."""
    counts = Counter()
    # Group by video sequence
    groups = defaultdict(list)
    for img in img_iter(SVAN_RGB):
        # Only confuser categories
        cat = None
        for c in SVAN_CONFUSER_CATS:
            if f"_{c}_" in img.stem:
                cat = c
                break
        if cat is None:
            continue
        vid = svan_video_id(img.stem)
        groups[vid].append(img)

    print(f"    {len(groups)} video sequences across {sum(len(v) for v in groups.values()):,} frames")
    splits = split_by_groups(groups)

    for split, imgs in splits.items():
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        for img in imgs:
            cat = "unknown"
            for c in SVAN_CONFUSER_CATS:
                if f"_{c}_" in img.stem:
                    cat = c.lower()
                    break
            dst_name = short_name(f"svan_{cat}", img)
            if copy_image(img, out_img / dst_name,
                          out_lbl / (Path(dst_name).stem + ".txt"), dry_run):
                counts[split] += 1
                manifest.append([dst_name, f"svanstrom_{cat}", split, str(img)])
    return counts





def collect_raihanrsd(out_root: Path, manifest: list, dry_run: bool,
                      limit: int = RAIHANRSD_LIMIT) -> Counter:
    """raihanrsd bird frames: split by video folder. Stride-sample to limit."""
    counts = Counter()
    # Group by video folder
    groups = defaultdict(list)
    for img in img_iter_recursive(RAIHANRSD):
        if "labels" in str(img):
            continue
        vid = raihanrsd_video_id(img)
        groups[vid].append(img)

    # Stride-sample within each group to hit total limit
    total_available = sum(len(v) for v in groups.items() if isinstance(v, list))
    total_available = sum(len(v) for v in groups.values())
    if limit > 0 and total_available > limit:
        # Sample proportionally from each group
        scale = limit / total_available
        for vid in groups:
            n_keep = max(1, int(len(groups[vid]) * scale))
            step = len(groups[vid]) / float(n_keep) if n_keep < len(groups[vid]) else 1
            groups[vid] = [groups[vid][int(i * step)] for i in range(n_keep)]

    print(f"    {len(groups)} video folders, {sum(len(v) for v in groups.values()):,} frames after sampling")
    splits = split_by_groups(groups)

    for split, imgs in splits.items():
        out_img = out_root / "images" / split
        out_lbl = out_root / "labels" / split
        for img in imgs:
            dst_name = short_name("raihanbird", img)
            if copy_image(img, out_img / dst_name,
                          out_lbl / (Path(dst_name).stem + ".txt"), dry_run):
                counts[split] += 1
                manifest.append([dst_name, "raihanrsd_bird", split, str(img)])
    return counts


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Collect all confuser sources into a single split folder"
    )
    ap.add_argument("--output", type=Path, default=OUT_ROOT,
                    help=f"Output root directory (default: {OUT_ROOT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count files only, don't copy anything")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["airplane", "new_dataset", "helicopter", "svanstrom",
                             "raihanrsd"],
                    help="Sources to skip")
    ap.add_argument("--raihanrsd-limit", type=int, default=RAIHANRSD_LIMIT,
                    help=f"Max raihanrsd bird frames to include (default: {RAIHANRSD_LIMIT})")
    args = ap.parse_args()

    out_root = args.output

    if not args.dry_run:
        for split in ("train", "val", "test"):
            (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = [["filename", "source", "split", "original_path"]]
    all_counts = Counter()  # {split: count}

    collectors = [
        ("airplane",    "Airplane Roboflow (uses existing splits)",
         lambda: collect_airplane(out_root, manifest, args.dry_run)),
        ("new_dataset", "New_Dataset (uses existing splits, drone-excluded)",
         lambda: collect_new_dataset(out_root, manifest, args.dry_run)),
        ("helicopter",  "Helicopter Kaggle (split by hash 80/10/10)",
         lambda: collect_helicopter(out_root, manifest, args.dry_run)),
        ("svanstrom",   "Svanstrom RGB confusers (split by video sequence)",
         lambda: collect_svanstrom(out_root, manifest, args.dry_run)),
        ("raihanrsd",   f"raihanrsd bird frames (limit={args.raihanrsd_limit}, split by video)",
         lambda: collect_raihanrsd(out_root, manifest, args.dry_run, args.raihanrsd_limit)),
    ]

    print("=" * 72)
    print(f"Collecting confuser sources -> {out_root}")
    if args.dry_run:
        print("  *** DRY RUN -- no files will be copied ***")
    print("  Split strategy: 80% train / 10% val / 10% test")
    print("  Leakage prevention: split by video/sequence ID")
    print("=" * 72)

    source_totals = {}
    for key, label, fn in collectors:
        if key in args.skip:
            print(f"\n  [{key}] SKIPPED (--skip)")
            continue
        print(f"\n  [{key}] {label}...")
        counts = fn()
        for split, n in counts.items():
            all_counts[split] += n
        total = sum(counts.values())
        source_totals[key] = total
        print(f"    -> {total:,} total  (train={counts.get('train',0):,}  "
              f"val={counts.get('val',0):,}  test={counts.get('test',0):,})")

    # Summary
    grand_total = sum(all_counts.values())
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n  By source:")
    for key, n in sorted(source_totals.items(), key=lambda x: -x[1]):
        pct = n / grand_total * 100 if grand_total > 0 else 0
        print(f"    {key:<20s} {n:>7,}  ({pct:5.1f}%)")

    print(f"\n  By split:")
    for split in ("train", "val", "test"):
        n = all_counts.get(split, 0)
        pct = n / grand_total * 100 if grand_total > 0 else 0
        print(f"    {split:<20s} {n:>7,}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<20s} {grand_total:>7,}")

    est_gb = grand_total * 100 * 1024 / (1024**3)
    print(f"\n  Estimated zip size: ~{est_gb:.1f} GB (at ~100KB/image avg)")

    if not args.dry_run:
        mf_path = out_root / "manifest.csv"
        with mf_path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(manifest)
        print(f"\n  Wrote {mf_path} ({len(manifest)-1:,} rows)")

    print(f"\nDone. {'(dry run)' if args.dry_run else f'Output: {out_root}'}")
    if not args.dry_run:
        print(f"\nNext step: zip and upload to vast.ai:")
        print(f"  cd {out_root.parent}")
        print(f"  tar -czf confusers_merged.tar.gz {out_root.name}/")


if __name__ == "__main__":
    main()
