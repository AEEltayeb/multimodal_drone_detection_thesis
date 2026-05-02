"""
build_finetune_dataset_v2.py — Assemble an expanded hard-negative fine-tune
dataset for the RGB drone detector.

Changes vs v1:
  - Adds Svanström confuser frames (AIRPLANE, BIRD, HELICOPTER — NO DRONES)
  - Larger drone positive pool (21K stride-sampled from 137K originals)
  - Target 70/30 drone/confuser ratio → ~30K total
  - Outputs to G:/drone/finetune_dataset_v2/

Output layout (single-class drone, same label space as current model):
    G:/drone/finetune_dataset_v2/
        data.yaml
        manifest.csv
        images/{train,val}/...
        labels/{train,val}/...

Usage:
    python "RGB model/dataset preparation/build_finetune_dataset_v2.py"
    python "RGB model/dataset preparation/build_finetune_dataset_v2.py" --clean
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import random
from pathlib import Path
from collections import Counter


# ─── INPUT PATHS ──────────────────────────────────────────────────

DRONE_DSET   = Path(r"G:/drone/dataset/dataset")
AIRPLANE_DS  = Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb")
NEW_DS       = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb")
HELI_DS      = Path(r"G:/drone/Helicopter-kaggle-dataset/Helicopter Class 1")
SVAN_RGB     = Path(r"G:/drone/svanstrom_paired/RGB/images")

OUT_ROOT     = Path(r"G:/drone/finetune_dataset_v2")

NEW_DS_DRONE_CLASS = 2   # 0=Airplane, 1=Bird, 2=Drone, 3=Helicopter, 4=tree

# ─── KNOBS ────────────────────────────────────────────────────────

DRONE_TRAIN_TARGET = 21_000
DRONE_VAL_TARGET   = 700
NEG_TOTAL_TARGET   = 9_000  # 30% of 30K
SVAN_CONFUSER_CATS = {"AIRPLANE", "BIRD", "HELICOPTER"}
INCLUDE_SVANSTROM = True  # toggled by --no-svanstrom

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)


# ─── HELPERS ──────────────────────────────────────────────────────

def short_name(prefix: str, src: Path) -> str:
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}{src.suffix.lower()}"


def stride_pick(items, n_target):
    items = list(items)
    if n_target <= 0 or n_target >= len(items):
        return items
    step = len(items) / float(n_target)
    return [items[int(i * step)] for i in range(n_target)]


def img_iter(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def label_for_image(img_path: Path, labels_dir: Path) -> Path | None:
    return labels_dir / (img_path.stem + ".txt")


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


def svan_category(stem: str) -> str | None:
    for c in SVAN_CONFUSER_CATS:
        if f"_{c}_" in stem:
            return c
    if "_DRONE_" in stem:
        return "DRONE"
    return None


def ensure_dirs():
    for split in ("train", "val"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ─── COPY / WRITE PRIMITIVES ──────────────────────────────────────

def copy_with_label(src_img: Path, src_lbl: Path | None,
                    dst_img: Path, dst_lbl: Path,
                    empty_label: bool):
    if dst_img.exists():
        return False
    shutil.copy2(src_img, dst_img)
    if empty_label or src_lbl is None or not src_lbl.exists():
        dst_lbl.write_text("")
    else:
        out_lines = []
        for ln in src_lbl.read_text().splitlines():
            parts = ln.strip().split()
            if len(parts) < 5:
                continue
            try:
                _ = float(parts[1]); _ = float(parts[2])
                _ = float(parts[3]); _ = float(parts[4])
            except ValueError:
                continue
            out_lines.append(" ".join(["0", *parts[1:5]]))
        dst_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return True


# ─── BUILD STAGES ─────────────────────────────────────────────────

def collect_all_negatives():
    """Gather ALL available negatives, then we'll sample to NEG_TOTAL_TARGET."""
    all_negs = []  # list of (source_name, img_path)

    # 1. Airplane (train+valid)
    for split in ("train", "valid"):
        img_dir = AIRPLANE_DS / split / "images"
        for img in img_iter(img_dir):
            all_negs.append(("airplane", img))

    # 2. New_Dataset (train+valid) — exclude drone-class
    for split in ("train", "valid"):
        img_dir = NEW_DS / split / "images"
        lbl_dir = NEW_DS / split / "labels"
        for img in img_iter(img_dir):
            lbl = label_for_image(img, lbl_dir)
            if label_has_class(lbl, NEW_DS_DRONE_CLASS):
                continue
            all_negs.append(("newds", img))

    # 3. Helicopter-Kaggle (all)
    for img in img_iter(HELI_DS):
        all_negs.append(("heli", img))

    # 4. Svanström confusers (AIRPLANE, BIRD, HELICOPTER — NO DRONES)
    if INCLUDE_SVANSTROM:
        for img in img_iter(SVAN_RGB):
            cat = svan_category(img.stem)
            if cat in SVAN_CONFUSER_CATS:
                all_negs.append((f"svan_{cat.lower()}", img))
    else:
        print("    [skip] Svanström confusers (--no-svanstrom)")

    return all_negs


def build_train(manifest):
    n_added = Counter()

    # ── Collect and sample negatives ────────────────────────────
    print("  Collecting all negative sources...")
    all_negs = collect_all_negatives()
    print(f"  Total negatives available: {len(all_negs):,}")

    # Count by source
    source_counts = Counter(src for src, _ in all_negs)
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:<20s} {cnt:>7,}")

    # Stride-sample to target
    random.shuffle(all_negs)
    sampled_negs = all_negs[:NEG_TOTAL_TARGET] if len(all_negs) > NEG_TOTAL_TARGET else all_negs
    print(f"  Sampled {len(sampled_negs):,} negatives (target: {NEG_TOTAL_TARGET:,})")

    # Count sampled by source
    sampled_counts = Counter(src for src, _ in sampled_negs)
    for src, cnt in sorted(sampled_counts.items()):
        print(f"    {src:<20s} {cnt:>7,}")

    # Copy negatives
    for source, img in sampled_negs:
        dst_img = OUT_ROOT / "images" / "train" / short_name(source, img)
        dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added[f"{source}_neg"] += 1
            manifest.append([str(dst_img), "train", f"{source}_neg",
                             str(img), "empty"])

    # ── Drone positives: stride-sample from originals ───────────
    print(f"\n  Sampling {DRONE_TRAIN_TARGET:,} drone positives...")
    drone_train_imgs = img_iter(DRONE_DSET / "images" / "train")
    drone_picks = stride_pick(drone_train_imgs, DRONE_TRAIN_TARGET)
    drone_lbl_dir = DRONE_DSET / "labels" / "train"
    for img in drone_picks:
        lbl = label_for_image(img, drone_lbl_dir)
        if lbl is None or not lbl.exists():
            continue
        dst_img = OUT_ROOT / "images" / "train" / short_name("drone", img)
        dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
        if copy_with_label(img, lbl, dst_img, dst_lbl, empty_label=False):
            n_added["drone_pos"] += 1
            manifest.append([str(dst_img), "train", "drone_pos",
                             str(img), str(lbl)])

    return n_added


def build_val(manifest):
    n_added = Counter()

    # Drone val
    drone_val_imgs = img_iter(DRONE_DSET / "images" / "val")
    drone_picks = stride_pick(drone_val_imgs, DRONE_VAL_TARGET)
    drone_lbl_dir = DRONE_DSET / "labels" / "val"
    for img in drone_picks:
        lbl = label_for_image(img, drone_lbl_dir)
        if lbl is None or not lbl.exists():
            continue
        dst_img = OUT_ROOT / "images" / "val" / short_name("droneval", img)
        dst_lbl = OUT_ROOT / "labels" / "val" / (dst_img.stem + ".txt")
        if copy_with_label(img, lbl, dst_img, dst_lbl, empty_label=False):
            n_added["drone_val"] += 1
            manifest.append([str(dst_img), "val", "drone_val",
                             str(img), str(lbl)])

    # Mixed val: add 300 confusers (gives YOLO feedback on FP during training)
    n_confuser_val = 300
    all_negs = collect_all_negatives()
    random.shuffle(all_negs)
    val_negs = all_negs[:n_confuser_val]
    for source, img in val_negs:
        dst_img = OUT_ROOT / "images" / "val" / short_name(f"{source}val", img)
        dst_lbl = OUT_ROOT / "labels" / "val" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added["confuser_val"] += 1
            manifest.append([str(dst_img), "val", "confuser_val",
                             str(img), "empty"])

    return n_added


# ─── data.yaml ────────────────────────────────────────────────────

DATA_YAML_CONTENT = """\
# Auto-generated by build_finetune_dataset_v2.py
# Single-class drone — same label space as the source RGB model.

path: G:/drone/finetune_dataset_v2
train: images/train
val: images/val

nc: 1
names: ['drone']
"""


# ─── ENTRYPOINT ───────────────────────────────────────────────────

def main():
    global DRONE_TRAIN_TARGET, NEG_TOTAL_TARGET, INCLUDE_SVANSTROM

    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="rm -rf the output dir before building")
    ap.add_argument("--drone-target", type=int, default=DRONE_TRAIN_TARGET)
    ap.add_argument("--neg-target", type=int, default=NEG_TOTAL_TARGET)
    ap.add_argument("--no-svanstrom", action="store_true",
                    help="exclude Svanström confusers from negatives")
    args = ap.parse_args()

    DRONE_TRAIN_TARGET = args.drone_target
    NEG_TOTAL_TARGET = args.neg_target
    INCLUDE_SVANSTROM = not args.no_svanstrom

    if args.clean and OUT_ROOT.exists():
        print(f"[!] Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    ensure_dirs()
    manifest = [["dst_image", "split", "kind", "src_image", "src_label"]]

    print("=" * 72)
    print(f"Building TRAIN split (target: {DRONE_TRAIN_TARGET:,} drones + {NEG_TOTAL_TARGET:,} negatives)")
    print("=" * 72)
    n_train = build_train(manifest)
    for k, v in sorted(n_train.items()):
        print(f"  {k:<20s} {v:>7,}")
    print(f"  {'TOTAL train':<20s} {sum(n_train.values()):>7,}")

    print()
    print("=" * 72)
    print("Building VAL split (mixed: drones + confusers)...")
    print("=" * 72)
    n_val = build_val(manifest)
    for k, v in sorted(n_val.items()):
        print(f"  {k:<20s} {v:>7,}")
    print(f"  {'TOTAL val':<20s} {sum(n_val.values()):>7,}")

    # data.yaml
    (OUT_ROOT / "data.yaml").write_text(DATA_YAML_CONTENT)
    print(f"\nWrote {OUT_ROOT / 'data.yaml'}")

    # manifest.csv
    mf_path = OUT_ROOT / "manifest.csv"
    with mf_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(manifest)
    print(f"Wrote {mf_path} ({len(manifest)-1:,} rows)")

    total = sum(n_train.values()) + sum(n_val.values())
    drone_total = n_train.get("drone_pos", 0) + n_val.get("drone_val", 0)
    neg_total = total - drone_total
    print(f"\nTotal: {total:,} images ({drone_total:,} drones = {drone_total/total*100:.1f}%, "
          f"{neg_total:,} negatives = {neg_total/total*100:.1f}%)")
    print("\nDone.")


if __name__ == "__main__":
    main()
