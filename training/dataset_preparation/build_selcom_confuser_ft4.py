"""
build_selcom_confuser_ft4.py — Build ft4 dataset: ft3 base + confuser hard-negs.

Reuses the ft3 train directory (no re-copy) and injects a small number of
confuser hard-negatives mined by mine_confuser_hardnegs.py.

Dataset composition:
  - Train: ft3 train images (~8,825) + confuser hard-negs (500-800, empty labels)
  - Val: ft3 val (50/50 selcom + baseline, ~622) + confuser val sample (~100)

Key safeguards vs retrained_v2:
  - Confuser ratio ≤ 8% (vs retrained_v2's 30%)
  - Only images where the model actually hallucinates
  - Confidence-stratified sampling (high-conf FPs first)
  - Category-balanced (equal birds/airplanes/helicopters)
  - No Svanström data in training

Output:
  C:/drone_cache/_finetune_selcom_confuser_ft4/
    data.yaml
    images/train/  (symlinks or copies of confuser hard-negs)
    images/val/    (confuser val samples)
    labels/train/  (empty .txt files for hard-negs)
    labels/val/    (empty .txt files for confuser val)
    manifest.csv

Usage:
    python "training/dataset_preparation/build_selcom_confuser_ft4.py"
    python "training/dataset_preparation/build_selcom_confuser_ft4.py" --n-hardnegs 800 --clean
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Input paths
FT3_TRAIN_DIR = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft2/images/train")
FT3_TRAIN_LABELS = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft2/labels/train")
FT3_VAL_TXT = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3/val.txt")
HARDNEG_CSV = ROOT / "scripts" / "confuser_hardnegs_ft3.csv"
CONFUSER_VAL_DIR = Path(r"G:/drone/rgb_confusers_merged/images/val")

# Extra positives pool (same source as build_selcom_mixed_ft2.py)
GENERAL_IMAGES = Path(r"G:/drone/dataset/dataset/images/train")
GENERAL_LABELS = Path(r"G:/drone/dataset/dataset/labels/train")

# Output
OUT_ROOT = Path(r"C:/drone_cache/_finetune_selcom_confuser_ft4")

# Fallbacks if C: cache not available
FT3_TRAIN_DIR_FALLBACK = Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/train")
FT3_TRAIN_LABELS_FALLBACK = Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/train")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)


def classify_confuser_source(stem: str) -> str:
    if stem.startswith("airplane_") or "_AIRPLANE_" in stem:
        return "AIRPLANE"
    if stem.startswith("helicopter_") or "_HELICOPTER_" in stem:
        return "HELICOPTER"
    if stem.startswith("bird_") or stem.startswith("raihanrsd_") or "_BIRD_" in stem:
        return "BIRD"
    return "OTHER"


def load_hardnegs(csv_path: Path, n_target: int, min_conf: float = 0.0) -> list[dict]:
    """Load and select hard-negatives from the mining CSV.

    Strategy: confidence-stratified, category-balanced sampling.
    - First, filter to images where model fired (is_hardneg=1)
    - Sort by max_conf descending (highest-confidence FPs are most damaging)
    - Balance across categories (equal-ish from each)
    - Take top N
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if int(r["is_hardneg"]) == 1 and float(r["max_conf"]) >= min_conf:
                rows.append(r)

    print(f"  Total hard-neg candidates: {len(rows)}")

    # Group by category
    by_cat = {}
    for r in rows:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r)

    # Sort each category by max_conf descending
    for cat in by_cat:
        by_cat[cat].sort(key=lambda x: -float(x["max_conf"]))

    # Category-balanced sampling
    cats = sorted(by_cat.keys())
    per_cat = max(1, n_target // len(cats)) if cats else 0
    selected = []
    for cat in cats:
        pool = by_cat[cat]
        take = min(per_cat, len(pool))
        selected.extend(pool[:take])
        print(f"    {cat}: {take}/{len(pool)} (top conf: "
              f"{float(pool[0]['max_conf']):.3f}–{float(pool[min(take,len(pool))-1]['max_conf']):.3f})")

    # If we haven't reached target, fill from remaining (round-robin)
    if len(selected) < n_target:
        used = set(r["image_path"] for r in selected)
        remaining = [r for r in rows if r["image_path"] not in used]
        remaining.sort(key=lambda x: -float(x["max_conf"]))
        selected.extend(remaining[:n_target - len(selected)])

    selected = selected[:n_target]
    print(f"  Selected {len(selected)} hard-negs for training")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-hardnegs", type=int, default=600,
                    help="Number of confuser hard-negs to inject into train (default: 600)")
    ap.add_argument("--n-extra-positives", type=int, default=0,
                    help="Number of extra drone-positive images to sample from general pool (default: 0)")
    ap.add_argument("--n-confuser-val", type=int, default=100,
                    help="Number of confuser images to add to val (default: 100)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Minimum max-det confidence to qualify as hard-neg (default: 0)")
    ap.add_argument("--clean", action="store_true",
                    help="Remove output directory before building")
    ap.add_argument("--confusers-only", action="store_true",
                    help="Only swap variable images (confusers + extra positives), "
                         "keep base intact. Assumes base exists from prior full build.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without copying")
    args = ap.parse_args()

    if not HARDNEG_CSV.exists():
        print(f"[fatal] Hard-neg CSV not found: {HARDNEG_CSV}")
        print("  Run mine_confuser_hardnegs.py first.")
        sys.exit(1)

    # Resolve ft3 train dir
    train_dir = FT3_TRAIN_DIR if FT3_TRAIN_DIR.exists() else FT3_TRAIN_DIR_FALLBACK
    train_lbl_dir = FT3_TRAIN_LABELS if FT3_TRAIN_LABELS.exists() else FT3_TRAIN_LABELS_FALLBACK
    if not train_dir.exists():
        print(f"[fatal] ft3 train directory not found at {FT3_TRAIN_DIR} or {FT3_TRAIN_DIR_FALLBACK}")
        sys.exit(1)

    if args.clean and not args.confusers_only and OUT_ROOT.exists():
        print(f"[!] Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    # Create output dirs
    ft4_train_img = OUT_ROOT / "images" / "train"
    ft4_train_lbl = OUT_ROOT / "labels" / "train"
    ft4_val_img = OUT_ROOT / "images" / "val"
    ft4_val_lbl = OUT_ROOT / "labels" / "val"
    for d in [ft4_train_img, ft4_train_lbl, ft4_val_img, ft4_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # In confusers-only mode, purge old variable files before re-injecting
    if args.confusers_only:
        print("[swap] Purging old confuser_*, extrapos_*, confuserval_* files...")
        n_purged = 0
        for d in [ft4_train_img, ft4_train_lbl]:
            for f in d.iterdir():
                if f.name.startswith("confuser_") or f.name.startswith("extrapos_"):
                    f.unlink()
                    n_purged += 1
        for d in [ft4_val_img, ft4_val_lbl]:
            for f in d.iterdir():
                if f.name.startswith("confuserval_"):
                    f.unlink()
                    n_purged += 1
        # Delete label caches so YOLO re-scans
        for cache_file in (OUT_ROOT / "labels" / "train.cache",
                           OUT_ROOT / "labels" / "val.cache"):
            if cache_file.exists():
                cache_file.unlink()
                print(f"  Deleted stale cache: {cache_file.name}")
        print(f"  Purged {n_purged} old variable files")

    manifest = [["dst_image", "split", "kind", "src_image"]]
    n_added = Counter()

    # ── Step 1: Link/copy ft3 base training images ──────────────────
    if args.confusers_only:
        # Count existing base images (exclude confuser_* and extrapos_*)
        existing_base = [p for p in ft4_train_img.iterdir()
                         if p.suffix.lower() in IMG_EXTS
                         and not p.name.startswith("confuser_")
                         and not p.name.startswith("extrapos_")]
        n_added["base_train"] = len(existing_base)
        print(f"[swap] Step 1 SKIP -- {n_added['base_train']} base images already present")
    else:
        print("="*72)
        print("Step 1: Linking ft3 base training images")
        print("="*72)

        base_imgs = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"  ft3 base train: {len(base_imgs)} images at {train_dir}")

        if not args.dry_run:
            for img_path in base_imgs:
                dst_img = ft4_train_img / img_path.name
                dst_lbl = ft4_train_lbl / (img_path.stem + ".txt")
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                # Copy label if exists
                src_lbl = train_lbl_dir / (img_path.stem + ".txt")
                if not dst_lbl.exists():
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, dst_lbl)
                    else:
                        dst_lbl.write_text("")
                n_added["base_train"] += 1
        else:
            n_added["base_train"] = len(base_imgs)
        print(f"  Added {n_added['base_train']} base training images")

    # ── Step 2: Inject confuser hard-negatives ──────────────────────
    print(f"\n{'='*72}")
    print(f"Step 2: Injecting {args.n_hardnegs} confuser hard-negatives")
    print(f"{'='*72}")

    hardnegs = load_hardnegs(HARDNEG_CSV, args.n_hardnegs, args.min_conf)

    if not args.dry_run:
        for hn in hardnegs:
            src = Path(hn["image_path"])
            if not src.exists():
                continue
            dst_img = ft4_train_img / f"confuser_{src.name}"
            dst_lbl = ft4_train_lbl / f"confuser_{src.stem}.txt"
            if not dst_img.exists():
                shutil.copy2(src, dst_img)
            if not dst_lbl.exists():
                dst_lbl.write_text("")  # Empty label = no objects
            n_added["confuser_hardneg"] += 1
            manifest.append([str(dst_img), "train", "confuser_hardneg", str(src)])
    else:
        n_added["confuser_hardneg"] = len(hardnegs)
    print(f"  Added {n_added['confuser_hardneg']} confuser hard-negatives")

    # ── Step 2.5: Inject extra positives from general pool ────────
    n_added["extra_positives"] = 0
    if args.n_extra_positives > 0:
        print(f"\n{'='*72}")
        print(f"Step 2.5: Injecting {args.n_extra_positives} extra positives from general pool")
        print(f"{'='*72}")

        if not GENERAL_IMAGES.exists():
            print(f"  [WARN] General images dir not found: {GENERAL_IMAGES}")
            print(f"         Skipping extra positives.")
        else:
            # Find stems already used by gen_* base images (from ft2)
            used_stems = set()
            for f in ft4_train_img.iterdir():
                if f.name.startswith("gen_"):
                    used_stems.add(f.stem.removeprefix("gen_"))
            print(f"  Already used gen_* stems: {len(used_stems)}")

            # Scan general pool and exclude already-used images
            all_general = sorted(
                p for p in GENERAL_IMAGES.iterdir()
                if p.suffix.lower() in IMG_EXTS and p.stem not in used_stems
            )
            print(f"  Available extra positives: {len(all_general)}")

            # Deterministic sample with a different seed than confusers
            rng_extra = random.Random(99)
            rng_extra.shuffle(all_general)
            extra_picks = all_general[:args.n_extra_positives]

            if len(extra_picks) < args.n_extra_positives:
                print(f"  [WARN] Requested {args.n_extra_positives} but only "
                      f"{len(extra_picks)} available after dedup")

            if not args.dry_run:
                for i, ep in enumerate(extra_picks):
                    dst_img = ft4_train_img / f"extrapos_{ep.name}"
                    dst_lbl = ft4_train_lbl / f"extrapos_{ep.stem}.txt"
                    if not dst_img.exists():
                        shutil.copy2(ep, dst_img)
                    src_lbl = GENERAL_LABELS / (ep.stem + ".txt")
                    if not dst_lbl.exists():
                        if src_lbl.exists():
                            shutil.copy2(src_lbl, dst_lbl)
                        else:
                            dst_lbl.write_text("")
                    n_added["extra_positives"] += 1
                    manifest.append([str(dst_img), "train", "extra_positive", str(ep)])
                    if (i + 1) % 2000 == 0:
                        print(f"    {i+1}/{len(extra_picks)} copied...")
            else:
                n_added["extra_positives"] = len(extra_picks)
            print(f"  Added {n_added['extra_positives']} extra positives")

    # ── Step 3: Build val set ──────────────────────────────────────
    if args.confusers_only:
        # Count existing base val images instead of copying
        existing_val = [p for p in ft4_val_img.iterdir()
                        if p.suffix.lower() in IMG_EXTS and not p.name.startswith("confuserval_")]
        n_added["val_base"] = len(existing_val)
        print(f"[confusers-only] Step 3 base val SKIP — {n_added['val_base']} base val images already present")
    else:
        print(f"\n{'='*72}")
        print("Step 3: Building val set (ft3 val + confuser val sample)")
        print("="*72)

        # Copy ft3 val images (from val.txt path list or from directory)
        ft3_val_count = 0
        if FT3_VAL_TXT.exists():
            # ft3 uses a val.txt path list
            val_paths = [Path(ln.strip()) for ln in FT3_VAL_TXT.read_text().splitlines() if ln.strip()]
            print(f"  ft3 val.txt: {len(val_paths)} paths")
            if not args.dry_run:
                for vp in val_paths:
                    if not vp.exists():
                        continue
                    dst_img = ft4_val_img / vp.name
                    # Find label — check multiple possible locations
                    lbl_name = vp.stem + ".txt"
                    src_lbl = None
                    for lbl_dir in [vp.parent.parent / "labels" / "val",
                                    vp.parent.parent.parent / "labels" / "val",
                                    Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
                                    Path(r"G:/drone/dataset/dataset/labels/val")]:
                        candidate = lbl_dir / lbl_name
                        if candidate.exists():
                            src_lbl = candidate
                            break
                    dst_lbl = ft4_val_lbl / lbl_name
                    if not dst_img.exists():
                        shutil.copy2(vp, dst_img)
                    if not dst_lbl.exists():
                        if src_lbl and src_lbl.exists():
                            shutil.copy2(src_lbl, dst_lbl)
                        else:
                            dst_lbl.write_text("")
                    ft3_val_count += 1
            else:
                ft3_val_count = len(val_paths)
        else:
            # Fallback: copy from directory
            ft2_val_img = Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val")
            ft2_val_lbl = Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val")
            if ft2_val_img.exists():
                val_imgs = sorted(p for p in ft2_val_img.iterdir() if p.suffix.lower() in IMG_EXTS)
                print(f"  ft2 val dir: {len(val_imgs)} images")
                if not args.dry_run:
                    for vp in val_imgs:
                        dst_img = ft4_val_img / vp.name
                        dst_lbl = ft4_val_lbl / (vp.stem + ".txt")
                        src_lbl = ft2_val_lbl / (vp.stem + ".txt")
                        if not dst_img.exists():
                            shutil.copy2(vp, dst_img)
                        if not dst_lbl.exists():
                            if src_lbl.exists():
                                shutil.copy2(src_lbl, dst_lbl)
                            else:
                                dst_lbl.write_text("")
                        ft3_val_count += 1
                else:
                    ft3_val_count = len(val_imgs)

        n_added["val_base"] = ft3_val_count
        print(f"  Added {ft3_val_count} ft3 val images")

    # Add confuser val samples
    if CONFUSER_VAL_DIR.exists():
        confuser_val_imgs = sorted(p for p in CONFUSER_VAL_DIR.iterdir()
                                   if p.suffix.lower() in IMG_EXTS)
        random.shuffle(confuser_val_imgs)
        confuser_val_picks = confuser_val_imgs[:args.n_confuser_val]
        print(f"  Confuser val candidates: {len(confuser_val_imgs)}, picking {len(confuser_val_picks)}")
        if not args.dry_run:
            for vp in confuser_val_picks:
                dst_img = ft4_val_img / f"confuserval_{vp.name}"
                dst_lbl = ft4_val_lbl / f"confuserval_{vp.stem}.txt"
                if not dst_img.exists():
                    shutil.copy2(vp, dst_img)
                if not dst_lbl.exists():
                    dst_lbl.write_text("")  # Empty = no drones
                n_added["val_confuser"] += 1
                manifest.append([str(dst_img), "val", "confuser_val", str(vp)])
        else:
            n_added["val_confuser"] = len(confuser_val_picks)
        print(f"  Added {n_added['val_confuser']} confuser val images")
    else:
        print(f"  [WARN] Confuser val dir not found: {CONFUSER_VAL_DIR}")

    # ── Step 4: Write data.yaml ──────────────────────────────────
    data_yaml_content = f"""\
# Auto-generated by build_selcom_confuser_ft4.py
# ft3 base + confuser hard-negatives — single-class drone.

path: {OUT_ROOT}
train: images/train
val: images/val

nc: 1
names: ['drone']
"""
    if not args.dry_run:
        (OUT_ROOT / "data.yaml").write_text(data_yaml_content)

    # Write manifest
    if not args.dry_run:
        mf_path = OUT_ROOT / "manifest.csv"
        with open(mf_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(manifest)

    # ── Summary ──────────────────────────────────────────────────
    n_extra = n_added.get("extra_positives", 0)
    total_train = n_added["base_train"] + n_extra + n_added["confuser_hardneg"]
    total_val = n_added["val_base"] + n_added.get("val_confuser", 0)
    confuser_ratio = n_added["confuser_hardneg"] / max(total_train, 1)
    positive_total = n_added["base_train"] + n_extra

    print(f"\n{'='*72}")
    print(f"DATASET SUMMARY {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'='*72}")
    print(f"  Output: {OUT_ROOT}")
    print(f"  Train:")
    print(f"    Base images:      {n_added['base_train']:>7,}")
    print(f"    Extra positives:  {n_extra:>7,}")
    print(f"    Confuser hardnegs:{n_added['confuser_hardneg']:>7,}")
    print(f"    TOTAL train:      {total_train:>7,}")
    print(f"    Positives total:  {positive_total:>7,}")
    print(f"    Confuser ratio:   {confuser_ratio:>7.1%}")
    print(f"  Val:")
    print(f"    Base val:         {n_added['val_base']:>7,}")
    print(f"    Confuser val:     {n_added.get('val_confuser', 0):>7,}")
    print(f"    TOTAL val:        {total_val:>7,}")

    if confuser_ratio > 0.10:
        print(f"\n  [WARN] Confuser ratio {confuser_ratio:.1%} > 10%!")
        print(f"  Consider reducing --n-hardnegs to stay under 8%.")
    if n_extra > 0:
        print(f"\n  Ratio analysis:")
        print(f"    Without extras: {n_added['confuser_hardneg']}/{n_added['base_train']+n_added['confuser_hardneg']} = "
              f"{n_added['confuser_hardneg']/max(n_added['base_train']+n_added['confuser_hardneg'],1):.1%}")
        print(f"    With extras:    {n_added['confuser_hardneg']}/{total_train} = {confuser_ratio:.1%}")


if __name__ == "__main__":
    main()
