"""
build_selcom_mixed_ft3.py — Same train as ft2, but val is 50/50 baseline+selcom.

Train: REUSES ft2 staging at G:/drone/_finetune_selcom_mixed_ft2/images/train
       (no re-copy — data.yaml points there directly)
Val:   freshly staged at G:/drone/_finetune_selcom_mixed_ft3/images/val
       = selcom_val (same as ft2, deterministic SEED=0) + matching general_val
No test split.
"""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

SELCOM_ROOT   = Path(r"C:/drone_cache/selcom_dataset")
SELCOM_IMAGES = SELCOM_ROOT / "images"
SELCOM_LABELS = SELCOM_ROOT / "labels"

GENERAL_VAL_IMAGES = Path(r"C:/drone_cache/dataset/images/val")
GENERAL_VAL_LABELS = Path(r"C:/drone_cache/dataset/labels/val")

FT2_TRAIN_IMAGES = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft2/images/train")
FT2_TRAIN_LABELS = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft2/labels/train")

DST_ROOT = Path(r"C:/drone_cache/_finetune_selcom_mixed_ft3")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED     = 0
VAL_FRAC = 0.15


def _selcom_items():
    imgs = sorted(p for p in SELCOM_IMAGES.iterdir() if p.suffix.lower() in IMG_EXTS)
    out = []
    for img in imgs:
        lbl = SELCOM_LABELS / (img.stem + ".txt")
        has = lbl.exists() and lbl.stat().st_size > 0
        out.append((img, lbl if lbl.exists() else None, has))
    return out


def _general_sample(img_dir: Path, lbl_dir: Path, n: int, rng: random.Random):
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    rng.shuffle(imgs)
    imgs = imgs[:n]
    out = []
    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        out.append((img, lbl if lbl.exists() else None))
    return out


def build(selcom_ratio: float = 0.20, clean: bool = False) -> Path:
    # selcom_ratio kept for CLI compatibility but ignored — train comes from ft2.
    if not FT2_TRAIN_IMAGES.exists():
        raise SystemExit(
            f"[fatal] ft2 train staging missing at {FT2_TRAIN_IMAGES}\n"
            f"        Stage ft2 first (build_selcom_mixed_ft2.py) — ft3 reuses its train split."
        )

    if clean and DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print("Scanning selcom images ...", flush=True)
    selcom_all = _selcom_items()
    rng_s = random.Random(SEED)
    rng_s.shuffle(selcom_all)
    n_selcom_val = max(1, round(len(selcom_all) * VAL_FRAC))
    selcom_val   = selcom_all[:n_selcom_val]  # same set as ft2 val (same SEED)

    rng_g = random.Random(SEED + 1)
    general_val = _general_sample(GENERAL_VAL_IMAGES, GENERAL_VAL_LABELS,
                                  n_selcom_val, rng_g)

    print(f"selcom_val={len(selcom_val)}  general_val={len(general_val)}  "
          f"total_val={len(selcom_val)+len(general_val)} (50/50)")
    print(f"train: reusing {FT2_TRAIN_IMAGES} (no copy)")
    print(f"val: writing path list (no copy) — Ultralytics finds labels via /images/ -> /labels/ swap")

    val_txt = DST_ROOT / "val.txt"
    lines = []
    manifest_rows = []
    for img, _, pos in selcom_val:
        lines.append(img.as_posix())
        manifest_rows.append(dict(split="val", source="selcom",
                                  image=img.as_posix(), has_label=pos))
    for img, lbl in general_val:
        pos = lbl is not None and lbl.exists() and lbl.stat().st_size > 0
        lines.append(img.as_posix())
        manifest_rows.append(dict(split="val", source="gen",
                                  image=img.as_posix(), has_label=pos))
    val_txt.write_text("\n".join(lines) + "\n")

    yaml_path = DST_ROOT / "data.yaml"
    yaml_path.write_text(
        f"train: {FT2_TRAIN_IMAGES.as_posix()}\n"
        f"val: {val_txt.as_posix()}\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n"
    )

    manifest_path = DST_ROOT / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "image", "has_label"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    total = len(selcom_val) + len(general_val)
    print(f"Staged  val={total} via path list  (train reused from ft2)  ->  {DST_ROOT}")
    print(f"data.yaml: {yaml_path}")
    print(f"val.txt:   {val_txt}")
    return yaml_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.20,
                    help="(kept for CLI compatibility; train is reused from ft2)")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    build(selcom_ratio=args.ratio, clean=args.clean)
