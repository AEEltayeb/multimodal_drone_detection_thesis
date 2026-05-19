"""
build_selcom_ft1.py — Stage the selcom TestDrone dataset for fine-tuning.

Source: G:/drone/RGB_TestDrone_dense_0_6440_selcom_footage_ft1/
        RGB_TestDrone_dense_0_6440_selcom_footage_ft1/
Output: G:/drone/_finetune_selcom_ft1/

Steps:
  1. Walk all 691 images in Images/.
  2. For each image, copy its paired label if it exists; otherwise write an
     empty .txt (true negative, per user decision).
  3. Ignore orphan labels (label file with no matching image).
  4. Deterministic 85/15 stratified split: positives and negatives are
     shuffled independently then merged, so the val set contains both kinds.
  5. Write data.yaml + manifest.csv.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

SRC_ROOT = Path(r"G:/drone/RGB_TestDrone_dense_0_6440_selcom_footage_ft1"
                r"/RGB_TestDrone_dense_0_6440_selcom_footage_ft1")
SRC_IMAGES = SRC_ROOT / "Images"
SRC_LABELS = SRC_ROOT / "labels"

DST_ROOT = Path(r"G:/drone/_finetune_selcom_ft1")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 0
VAL_FRAC = 0.15


def build(clean: bool = False) -> Path:
    if clean and DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    for split in ("train", "val"):
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in SRC_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images found in {SRC_IMAGES}")

    # Separate into positives (have a non-empty label) and negatives
    positives, negatives = [], []
    for img in imgs:
        lbl = SRC_LABELS / (img.stem + ".txt")
        has_label = lbl.exists() and lbl.stat().st_size > 0
        if has_label:
            positives.append(img)
        else:
            negatives.append(img)

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_list(lst):
        n_val = max(1, round(len(lst) * VAL_FRAC))
        return lst[n_val:], lst[:n_val]   # train, val

    pos_train, pos_val = split_list(positives)
    neg_train, neg_val = split_list(negatives)

    print(f"Positives  train={len(pos_train)}  val={len(pos_val)}")
    print(f"Negatives  train={len(neg_train)}  val={len(neg_val)}")

    manifest_rows = []
    for split, items in (("train", pos_train + neg_train),
                         ("val",   pos_val   + neg_val)):
        for img in items:
            lbl_src = SRC_LABELS / (img.stem + ".txt")
            has_paired = lbl_src.exists() and lbl_src.stat().st_size > 0

            dst_img = DST_ROOT / "images" / split / img.name
            dst_lbl = DST_ROOT / "labels" / split / (img.stem + ".txt")

            shutil.copy2(img, dst_img)
            if has_paired:
                shutil.copy2(lbl_src, dst_lbl)
            else:
                dst_lbl.write_text("")   # true negative stub

            manifest_rows.append({
                "split": split,
                "image": img.name,
                "has_label": has_paired,
            })

    # data.yaml
    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {DST_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n"
    )

    # manifest.csv
    manifest_path = DST_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "has_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_train = len(pos_train) + len(neg_train)
    n_val = len(pos_val) + len(neg_val)
    print(f"Staged {n_train} train / {n_val} val  →  {DST_ROOT}")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true", help="Wipe DST_ROOT before staging")
    args = ap.parse_args()
    build(clean=args.clean)
