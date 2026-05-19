"""
build_selcom_mixed_ft2.py — Stage the *labeled* selcom dataset for fine-tuning 2.

Source: G:/drone/selcom_dataset/  (2076 images, 1953 positives + 123 labeled negatives)
Mixed 80/20 with the general RGB training set to prevent forgetting.

Output: G:/drone/_finetune_selcom_mixed_ft2/
  images/{train,val}/
  labels/{train,val}/
  data.yaml
  manifest.csv

Default ratio: 80% general / 20% selcom (--ratio 0.20 means selcom is 20%).
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

# ── Sources ──────────────────────────────────────────────────────────────────

SELCOM_ROOT   = Path(r"G:/drone/selcom_dataset")
SELCOM_IMAGES = SELCOM_ROOT / "images"
SELCOM_LABELS = SELCOM_ROOT / "labels"

GENERAL_IMAGES = Path(r"G:/drone/dataset/dataset/images/train")
GENERAL_LABELS = Path(r"G:/drone/dataset/dataset/labels/train")

DST_ROOT = Path(r"G:/drone/_finetune_selcom_mixed_ft2")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED     = 0
VAL_FRAC = 0.15


def _selcom_items():
    """Yield (img_path, lbl_path_or_None_if_empty, is_positive) for all selcom images.
    Orphan labels (no paired image) are ignored automatically because we iterate images."""
    imgs = sorted(p for p in SELCOM_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)
    out = []
    for img in imgs:
        lbl = SELCOM_LABELS / (img.stem + ".txt")
        has = lbl.exists() and lbl.stat().st_size > 0
        out.append((img, lbl if has else lbl if lbl.exists() else None, has))
    return out


def _general_sample(n: int, rng: random.Random):
    imgs = sorted(p for p in GENERAL_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)
    rng.shuffle(imgs)
    imgs = imgs[:n]
    out = []
    for img in imgs:
        lbl = GENERAL_LABELS / (img.stem + ".txt")
        out.append((img, lbl if lbl.exists() else None))
    return out


def build(selcom_ratio: float = 0.20, clean: bool = False) -> Path:
    if not 0.05 <= selcom_ratio <= 0.95:
        raise ValueError("selcom_ratio must be in [0.05, 0.95]")

    if clean and DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    for split in ("train", "val"):
        (DST_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    print("Scanning selcom images ...", flush=True)
    selcom_all = _selcom_items()
    n_pos = sum(1 for _, _, p in selcom_all if p)
    n_neg = len(selcom_all) - n_pos
    print(f"  selcom: {len(selcom_all)} images  positives={n_pos}  negatives={n_neg}", flush=True)

    rng_s = random.Random(SEED)
    rng_s.shuffle(selcom_all)
    n_val = max(1, round(len(selcom_all) * VAL_FRAC))
    selcom_val   = selcom_all[:n_val]
    selcom_train = selcom_all[n_val:]

    n_selcom_train = len(selcom_train)
    n_general = round(n_selcom_train * (1.0 - selcom_ratio) / selcom_ratio)
    print(f"Scanning general train dir ({GENERAL_IMAGES}) for {n_general} samples ...", flush=True)
    general_train = _general_sample(n_general, rng)

    print(f"selcom_ratio={selcom_ratio:.0%}  -> "
          f"selcom_train={n_selcom_train}  general_train={n_general}  "
          f"total_train={n_selcom_train+n_general}")
    print(f"selcom_val={len(selcom_val)}  (general excluded from val)")

    manifest_rows = []

    def copy_pair(img_src, lbl_src_or_none, split, source_tag, is_positive):
        stem = f"{source_tag}_{img_src.stem}"
        dst_img = DST_ROOT / "images" / split / (stem + img_src.suffix)
        dst_lbl = DST_ROOT / "labels" / split / (stem + ".txt")
        shutil.copy2(img_src, dst_img)
        if (lbl_src_or_none is not None
                and lbl_src_or_none.exists()
                and lbl_src_or_none.stat().st_size > 0):
            shutil.copy2(lbl_src_or_none, dst_lbl)
        else:
            dst_lbl.write_text("")   # explicit true-negative stub
        manifest_rows.append(dict(
            split=split, source=source_tag,
            image=dst_img.name, has_label=is_positive,
        ))

    total = len(selcom_train) + len(general_train) + len(selcom_val)
    done = 0

    def _progress():
        nonlocal done
        done += 1
        if done % 200 == 0 or done == total:
            print(f"  copying {done}/{total} ...", flush=True)

    # selcom train
    for img, lbl, pos in selcom_train:
        copy_pair(img, lbl, "train", "selcom", pos); _progress()

    # general train
    for img, lbl in general_train:
        pos = lbl is not None and lbl.exists() and lbl.stat().st_size > 0
        copy_pair(img, lbl, "train", "gen", pos); _progress()

    # selcom val (only selcom)
    for img, lbl, pos in selcom_val:
        copy_pair(img, lbl, "val", "selcom", pos); _progress()

    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {DST_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n"
    )

    manifest_path = DST_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "image", "has_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_train = n_selcom_train + n_general
    print(f"Staged  train={n_train}  val={len(selcom_val)}  ->  {DST_ROOT}")
    print(f"data.yaml: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.20)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    build(selcom_ratio=args.ratio, clean=args.clean)
