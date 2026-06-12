#!/usr/bin/env python3
"""
synth_yolo_for_patches.py

Build a synthetic YOLO-format dataset from a classifier patch tree so the
existing label_reviewer GUI can be used to review/relabel/delete patches.

Input layout (e.g. models/patches/ir):
    <patches_root>/<class_name>/<image>.jpg

Output layout:
    <out>/images/<link_name>           # hardlink/symlink/copy of each patch
    <out>/labels/<link_name>.txt       # "<class_idx> 0.5 0.5 1.0 1.0"
    <out>/labels/classes.txt           # one class name per line (sorted)
    <out>/manifest.csv                 # link_name,orig_class,orig_path

Usage:
    python scripts/dataset_preparation/synth_yolo_for_patches.py models/patches/ir
    python scripts/dataset_preparation/synth_yolo_for_patches.py models/patches/rgb --mode copy

Then in the label reviewer (review_labels_gui.py) point:
    Images Dir -> <out>/images
    Labels Dir -> <out>/labels

After reviewing, run apply_patch_review.py (sibling script) to push class
changes / deletions back to the original patch tree.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_file(src: Path, dst: Path, mode: str) -> str:
    """Create dst pointing at src using requested mode. Returns mode actually used."""
    if dst.exists():
        return mode
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass  # fall through to copy
    elif mode == "symlink":
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("patches_root", type=Path,
                    help="Root containing per-class subfolders (e.g. models/patches/ir)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <patches_root>_synth_review)")
    ap.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink",
                    help="How to materialize images in <out>/images. Falls back to copy on failure.")
    args = ap.parse_args()

    root: Path = args.patches_root.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"ERROR: no class subfolders under {root}", file=sys.stderr)
        return 2
    class_names = [p.name for p in class_dirs]
    cls_idx = {name: i for i, name in enumerate(class_names)}

    out = (args.out or root.parent / f"{root.name}_synth_review").resolve()
    images_out = out / "images"
    labels_out = out / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"Patches root : {root}")
    print(f"Output       : {out}")
    print(f"Classes      : {class_names}")
    print(f"Link mode    : {args.mode}")

    # First pass: detect filename collisions across classes.
    seen: dict[str, str] = {}  # stem -> class_name (first seen)
    collisions: set[str] = set()
    per_class_files: dict[str, list[Path]] = {}
    for cdir in class_dirs:
        files = [f for f in cdir.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        per_class_files[cdir.name] = files
        for f in files:
            if f.stem in seen and seen[f.stem] != cdir.name:
                collisions.add(f.stem)
            else:
                seen[f.stem] = cdir.name
    if collisions:
        print(f"  Detected {len(collisions)} stem collisions across classes -> prefixing those with class name.")

    manifest_rows: list[tuple[str, str, str]] = []
    used_modes: dict[str, int] = {}
    total = 0
    for cdir in class_dirs:
        cls = cdir.name
        idx = cls_idx[cls]
        label_line = f"{idx} 0.5 0.5 1.0 1.0\n"
        files = per_class_files[cls]
        for f in files:
            link_name = f"{cls}__{f.name}" if f.stem in collisions else f.name
            img_dst = images_out / link_name
            lbl_dst = labels_out / (Path(link_name).stem + ".txt")
            mode_used = link_file(f, img_dst, args.mode)
            used_modes[mode_used] = used_modes.get(mode_used, 0) + 1
            if not lbl_dst.exists():
                lbl_dst.write_text(label_line)
            manifest_rows.append((link_name, cls, str(f)))
            total += 1
        print(f"  {cls:12s} -> {len(files):,} patches (idx={idx})")

    (labels_out / "classes.txt").write_text("\n".join(class_names) + "\n")

    manifest_path = out / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["link_name", "orig_class", "orig_path"])
        w.writerows(manifest_rows)

    print(f"\nDone. {total:,} patches materialized.")
    print(f"  modes used : {used_modes}")
    print(f"  manifest   : {manifest_path}")
    print(f"\nIn review_labels_gui.py set:")
    print(f"  Images Dir -> {images_out}")
    print(f"  Labels Dir -> {labels_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
