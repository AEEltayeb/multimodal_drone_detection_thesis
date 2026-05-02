"""
build_finetune_dataset.py — Assemble a hard-negative fine-tune dataset for the
RGB drone detector.

Output layout (single-class drone, same label space as current model):
    G:/drone/finetune_dataset/
        data.yaml
        manifest.csv
        images/{train,val,test}/<source_prefix>__<original_name>.jpg
        labels/{train,val,test}/<source_prefix>__<original_name>.txt

Composition:
    TRAIN
      Confuser negatives (empty .txt labels)
        - Airplane.v1...roboflow-rgb         train + valid       (~2,300)
        - New_Dataset.v1i...rgb              train + valid, but ANY frame
                                              whose label contains class id 2
                                              (Drone) is excluded.            (~3,000)
        - Helicopter-kaggle (Helicopter Class 1) — all minus 200 carved out   (~1,550)
      Drone positives (real labels copied)
        - G:/drone/dataset/dataset/train     stride-sampled to ~7,000

    VAL  (drones only — regression monitor for YOLO during training)
        - G:/drone/dataset/dataset/val       stride-sampled to ~500

    TEST (held out, used by eval scripts)
        - test_anti_uav      Anti-UAV-RGBT test/RGB                (existing)
        - test_dataset_rgb   G:/drone/dataset/dataset/test
        - test_airplane      Airplane test split                   (~100)
        - test_new_dataset   New_Dataset test split,
                             excluding drone-containing frames     (filtered)
        - test_helicopter    200 carved frames from Helicopter-kaggle
        - test_svanstrom     handled by eval_six_configs (not copied here)

The manifest captures provenance per output file so we can audit what came
from where.
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


def short_name(prefix: str, src: Path) -> str:
    """Compact destination filename: prefix_<10-char hash>.ext.

    Windows MAX_PATH (260 chars) bites when source filenames already carry
    50+ char roboflow hashes; we shorten ours to keep dst paths well under
    the limit.
    """
    h = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}{src.suffix.lower()}"

# ─── INPUT PATHS ──────────────────────────────────────────────────

DRONE_DSET   = Path(r"G:/drone/dataset/dataset")  # original RGB drone training corpus
AIRPLANE_DS  = Path(r"G:/drone/Airplane.v1-2025-04-19-5-35am.yolo26-roboflow-rgb")
NEW_DS       = Path(r"G:/drone/New_Dataset.v1i.yolo26_airplane-drone-heli-rgb")
HELI_DS      = Path(r"G:/drone/Helicopter-kaggle-dataset/Helicopter Class 1")
ANTIUAV_TEST = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB")

OUT_ROOT     = Path(r"G:/drone/finetune_dataset")

NEW_DS_DRONE_CLASS = 2   # in New_Dataset: 0=Airplane, 1=Bird, 2=Drone, 3=Helicopter, 4=tree

# ─── KNOBS ────────────────────────────────────────────────────────

DRONE_TRAIN_TARGET = 7000
DRONE_VAL_TARGET   = 500
HELI_TEST_HOLDOUT  = 200
# Mixed-val mode: number of confuser frames to add to val (empty labels).
# Sourced 50/50 from New_Dataset/valid (random pick after drone-class filter)
# and Helicopter-kaggle (random subset distinct from train + heli test).
MIXED_VAL_CONFUSERS = 200

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(42)


# ─── HELPERS ──────────────────────────────────────────────────────

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
    """Yolo convention: same stem, .txt extension, in parallel labels dir."""
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


def ensure_dirs():
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    # Only the helicopter test bucket is local (we carve it out of Kaggle).
    # All other test sets are referenced in-place via test_paths.json.
    (OUT_ROOT / "images" / "test" / "helicopter").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "labels" / "test" / "helicopter").mkdir(parents=True, exist_ok=True)


# ─── COPY / WRITE PRIMITIVES ──────────────────────────────────────

def copy_with_label(src_img: Path, src_lbl: Path | None,
                    dst_img: Path, dst_lbl: Path,
                    empty_label: bool):
    if dst_img.exists():
        return False
    shutil.copy2(src_img, dst_img)
    if empty_label or src_lbl is None or not src_lbl.exists():
        dst_lbl.write_text("")  # empty: model learns "no drone here"
    else:
        # Sanitize: keep only valid YOLO lines. We're a single-class (drone)
        # detector — preserve as-is when it's already class 0; otherwise
        # rewrite class id to 0 (caller is responsible for ensuring this is
        # actually a drone-positive frame).
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
            # Force class 0 for our single-class label space.
            out_lines.append(" ".join(["0", *parts[1:5]]))
        dst_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return True


# ─── BUILD STAGES ─────────────────────────────────────────────────

def build_train(manifest):
    n_added = Counter()

    # ── Confuser negatives: Airplane (train+valid) ──────────────
    for split in ("train", "valid"):
        img_dir = AIRPLANE_DS / split / "images"
        for img in img_iter(img_dir):
            dst_img = OUT_ROOT / "images" / "train" / short_name(f"airplane_{split}", img)
            dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
            if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
                n_added["airplane_neg"] += 1
                manifest.append([str(dst_img), "train", "airplane_neg",
                                 str(img), "empty"])

    # ── Confuser negatives: New_Dataset (train+valid) — exclude drone-class
    for split in ("train", "valid"):
        img_dir = NEW_DS / split / "images"
        lbl_dir = NEW_DS / split / "labels"
        for img in img_iter(img_dir):
            lbl = label_for_image(img, lbl_dir)
            if label_has_class(lbl, NEW_DS_DRONE_CLASS):
                continue   # drop frames containing real drones
            dst_img = OUT_ROOT / "images" / "train" / short_name(f"newds_{split}", img)
            dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
            if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
                n_added["newds_neg"] += 1
                manifest.append([str(dst_img), "train", "newds_neg",
                                 str(img), "empty"])

    # ── Confuser negatives: Helicopter-kaggle (carve a 200-frame test split)
    heli_imgs = img_iter(HELI_DS)
    if not heli_imgs:
        print(f"[WARN] no helicopter images at {HELI_DS}")
    test_heli_picks = set(stride_pick(heli_imgs, HELI_TEST_HOLDOUT))
    train_heli = [p for p in heli_imgs if p not in test_heli_picks]
    for img in train_heli:
        dst_img = OUT_ROOT / "images" / "train" / short_name("heli", img)
        dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added["heli_neg"] += 1
            manifest.append([str(dst_img), "train", "heli_neg",
                             str(img), "empty"])
    # ...and stash the heli test picks for later
    for img in test_heli_picks:
        dst_img = OUT_ROOT / "images" / "test" / "helicopter" / short_name("helitest", img)
        dst_lbl = OUT_ROOT / "labels" / "test" / "helicopter" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added["heli_test"] += 1
            manifest.append([str(dst_img), "test", "heli_test",
                             str(img), "empty"])

    # ── Drone positives: stride-sample G:/drone/dataset/dataset/train ──
    drone_train_imgs = img_iter(DRONE_DSET / "images" / "train")
    drone_picks = stride_pick(drone_train_imgs, DRONE_TRAIN_TARGET)
    drone_lbl_dir = DRONE_DSET / "labels" / "train"
    for img in drone_picks:
        lbl = label_for_image(img, drone_lbl_dir)
        if lbl is None or not lbl.exists():
            continue   # skip drone images without labels (rare)
        dst_img = OUT_ROOT / "images" / "train" / short_name("drone", img)
        dst_lbl = OUT_ROOT / "labels" / "train" / (dst_img.stem + ".txt")
        if copy_with_label(img, lbl, dst_img, dst_lbl, empty_label=False):
            n_added["drone_pos"] += 1
            manifest.append([str(dst_img), "train", "drone_pos",
                             str(img), str(lbl)])

    return n_added


def build_val(manifest, mixed: bool):
    n_added = Counter()
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

    if not mixed:
        return n_added

    # Mixed val: add confuser frames (empty labels) so val mAP penalises
    # both missed drones and fired-on-confusers. Half from New_Dataset/valid
    # (drone-class filtered), half from Helicopter-kaggle.
    n_each = MIXED_VAL_CONFUSERS // 2

    nd_imgs = img_iter(NEW_DS / "valid" / "images")
    nd_lbl_dir = NEW_DS / "valid" / "labels"
    nd_safe = [p for p in nd_imgs
               if not label_has_class(label_for_image(p, nd_lbl_dir),
                                      NEW_DS_DRONE_CLASS)]
    for img in stride_pick(nd_safe, n_each):
        dst_img = OUT_ROOT / "images" / "val" / short_name("ndval", img)
        dst_lbl = OUT_ROOT / "labels" / "val" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added["confuser_val"] += 1
            manifest.append([str(dst_img), "val", "confuser_val",
                             str(img), "empty"])

    # Heli-kaggle: avoid frames already used in train (model has trained
    # heli/* in finetune_dataset/images/train) — but those were copied not
    # symlinked, so we just stride from a different region. Easiest: take
    # the first N that are NOT already in our train dir by hash.
    used_hashes = {p.stem for p in (OUT_ROOT / "images" / "train").iterdir()}
    heli_pool = [p for p in img_iter(HELI_DS)
                 if short_name("heli", p).rsplit(".", 1)[0] not in used_hashes]
    for img in stride_pick(heli_pool, n_each):
        dst_img = OUT_ROOT / "images" / "val" / short_name("helival", img)
        dst_lbl = OUT_ROOT / "labels" / "val" / (dst_img.stem + ".txt")
        if copy_with_label(img, None, dst_img, dst_lbl, empty_label=True):
            n_added["confuser_val"] += 1
            manifest.append([str(dst_img), "val", "confuser_val",
                             str(img), "empty"])

    return n_added


def build_test(manifest):
    """Test sets are NOT copied — eval scripts read from original paths.
    We just record their locations in test_paths.json so downstream eval
    knows where to look. Heli test split was carved into images/test/helicopter
    during build_train (small, kept local since it doesn't exist elsewhere).
    """
    paths = {
        "test_anti_uav":      str(ANTIUAV_TEST),
        "test_dataset_rgb":   str(DRONE_DSET / "images" / "test"),
        "test_airplane":      str(AIRPLANE_DS / "test"),
        "test_new_dataset":   str(NEW_DS / "test"),
        "test_helicopter":    str(OUT_ROOT / "images" / "test" / "helicopter"),
    }
    import json
    (OUT_ROOT / "test_paths.json").write_text(json.dumps(paths, indent=2))
    print(f"  wrote test_paths.json (5 references, no frames copied)")
    return Counter()


# ─── data.yaml ────────────────────────────────────────────────────

DATA_YAML = """\
# Auto-generated by build_finetune_dataset.py
# Single-class drone — same label space as the source RGB model.

path: G:/drone/finetune_dataset
train: images/train
val: images/val
# YOLO can only point at one test path; we keep buckets under images/test/<sub>
# and let our eval scripts iterate them. The single key here is just the
# parent so YOLO's `mode=val split=test` works as a smoke check.
test: images/test

nc: 1
names: ['drone']
"""


# ─── ENTRYPOINT ───────────────────────────────────────────────────

def main():
    global DRONE_TRAIN_TARGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="rm -rf the output dir before building")
    ap.add_argument("--drone-target", type=int, default=DRONE_TRAIN_TARGET,
                    help="drone positives to include in train (default 7000)")
    ap.add_argument("--mixed-val", action="store_true",
                    help="add confuser frames (empty labels) to val so mAP "
                         "penalises both missed drones AND fired-on-confusers")
    args = ap.parse_args()

    DRONE_TRAIN_TARGET = args.drone_target

    if args.clean and OUT_ROOT.exists():
        print(f"[!] Removing existing {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    ensure_dirs()
    manifest = [["dst_image", "split", "kind", "src_image", "src_label"]]

    print("-" * 72)
    print("Building TRAIN split…")
    print("-" * 72)
    n_train = build_train(manifest)
    for k, v in sorted(n_train.items()):
        print(f"  {k:<20s} {v:>7,}")
    print(f"  {'TOTAL train':<20s} {sum(n_train.values()):>7,}")

    print("-" * 72)
    print(f"Building VAL split{' (MIXED)' if args.mixed_val else ''}...")
    print("-" * 72)
    n_val = build_val(manifest, mixed=args.mixed_val)
    for k, v in sorted(n_val.items()):
        print(f"  {k:<20s} {v:>7,}")

    print("-" * 72)
    print("Building TEST splits…")
    print("-" * 72)
    n_test = build_test(manifest)
    for k, v in sorted(n_test.items()):
        print(f"  {k:<20s} {v:>7,}")

    # data.yaml
    (OUT_ROOT / "data.yaml").write_text(DATA_YAML)
    print(f"\nWrote {OUT_ROOT / 'data.yaml'}")

    # manifest.csv
    mf_path = OUT_ROOT / "manifest.csv"
    with mf_path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(manifest)
    print(f"Wrote {mf_path} ({len(manifest)-1:,} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
