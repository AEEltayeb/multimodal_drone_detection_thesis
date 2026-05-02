"""
convert_coco_to_yolo_ir_full.py — Convert full cleaned IR Roboflow COCO export to YOLO.

Roboflow exports everything into a single 'train/' folder with no split info.
We recover split membership by matching image stems against the original
IR_dsetV1_ironly dataset structure.

Usage:
    python scripts/convert_coco_to_yolo_ir_full.py \
        --src "G:\drone\IR_dsetV1_ironly.coco" \
        --dst datasets/IR_dsetV1_gold \
        --orig datasets/IR_dsetV1_ironly
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to Roboflow COCO export root")
    p.add_argument("--dst", required=True, help="Output YOLO dataset path")
    p.add_argument("--orig", default="datasets/IR_dsetV1_ironly",
                   help="Original IR dataset for split recovery")
    return p.parse_args()


def clean_stem(filename: str) -> str:
    """Remove Roboflow .rf.HASH suffix and _png/_jpg extension artifacts."""
    stem = Path(filename).stem
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    for ext in ["_png", "_jpg", "_jpeg", "_bmp"]:
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    return stem


def build_orig_split_index(orig_root: Path) -> dict:
    """Build stem -> split mapping from original dataset."""
    index = {}
    for split in ["train", "val", "test"]:
        img_dir = orig_root / "images" / split
        if not img_dir.exists():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                index[f.stem] = split
    return index


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    orig = Path(args.orig)

    # Find COCO JSON
    coco_json = src / "train" / "_annotations.coco.json"
    if not coco_json.exists():
        coco_json = src / "_annotations.coco.json"
    if not coco_json.exists():
        print("[ERROR] _annotations.coco.json not found")
        return

    img_src_dir = coco_json.parent
    print(f"  COCO JSON: {coco_json}")
    print(f"  Image dir: {img_src_dir}")

    # Load COCO
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    cat_names = [c["name"] for c in coco.get("categories", [])]
    print(f"  Categories: {cat_names} -> all mapped to class 0 (drone)")

    # Build image index
    img_index = {}
    for img in coco["images"]:
        img_index[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Build annotation index
    ann_index = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_index[ann["image_id"]].append(ann)

    print(f"  Total images in COCO: {len(img_index)}")
    print(f"  Total annotations: {len(coco.get('annotations', []))}")

    # Build split recovery index
    print(f"\n  Building split index from: {orig}")
    split_index = build_orig_split_index(orig)
    print(f"  Original stems indexed: {len(split_index)}")

    # Process each image
    split_counts = defaultdict(lambda: {"images": 0, "labels": 0, "empty": 0, "boxes": 0})
    manifest = defaultdict(list)
    unmatched = []

    for img_id, img_info in img_index.items():
        fname = img_info["file_name"]
        w, h = float(img_info["width"]), float(img_info["height"])
        stem = clean_stem(fname)

        # Recover split
        split = split_index.get(stem)
        if split is None:
            unmatched.append((stem, fname))
            continue

        # Create dirs
        img_dst_dir = dst / "images" / split
        lbl_dst_dir = dst / "labels" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        lbl_dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy image
        src_img = img_src_dir / fname
        ext = Path(fname).suffix
        dst_img = img_dst_dir / f"{stem}{ext}"
        if src_img.exists():
            shutil.copy2(str(src_img), str(dst_img))
        else:
            print(f"  [WARN] Image not found: {src_img}")
            continue

        # Convert annotations
        anns = ann_index.get(img_id, [])
        lbl_path = lbl_dst_dir / f"{stem}.txt"
        lines = []
        for ann in anns:
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        split_counts[split]["images"] += 1
        split_counts[split]["labels"] += 1
        split_counts[split]["boxes"] += len(lines)
        if not lines:
            split_counts[split]["empty"] += 1
        manifest[split].append(stem)

    # Create YAML
    yaml_path = dst / "IR_dsetV1_gold.yaml"
    yaml_content = (
        "# IR_dsetV1_gold - Full IR dataset with cleaned/gold labels\n"
        f"path: {dst}\n\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: drone\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Create manifest
    manifest_path = dst / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dict(manifest), f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print(f"  Conversion Complete")
    print(f"{'='*55}")

    total_imgs = 0
    total_boxes = 0
    total_empty = 0
    for split in ["train", "val", "test"]:
        c = split_counts[split]
        print(f"\n  {split.upper()}:")
        print(f"    Images:       {c['images']}")
        print(f"    Label files:  {c['labels']}")
        print(f"    Empty labels: {c['empty']}")
        print(f"    Total boxes:  {c['boxes']}")
        total_imgs += c["images"]
        total_boxes += c["boxes"]
        total_empty += c["empty"]

    print(f"\n  TOTAL:")
    print(f"    Images:       {total_imgs}")
    print(f"    Boxes:        {total_boxes}")
    print(f"    Empty labels: {total_empty}")
    print(f"    Unmatched:    {len(unmatched)}")
    print(f"    YAML:         {yaml_path}")
    print(f"    Manifest:     {manifest_path}")

    if unmatched:
        print(f"\n  Unmatched stems (first 10):")
        for stem, fname in unmatched[:10]:
            print(f"    {stem} ({fname})")

    # Comparison with original
    print(f"\n{'='*55}")
    print(f"  Original vs Cleaned Comparison")
    print(f"{'='*55}")
    orig_empty = {"train": 251, "val": 220, "test": 152}
    for split in ["train", "val", "test"]:
        c = split_counts[split]
        oe = orig_empty.get(split, "?")
        print(f"  {split}: orig_empty={oe}, gold_empty={c['empty']}, "
              f"newly_annotated={oe - c['empty'] if isinstance(oe, int) else '?'}")


if __name__ == "__main__":
    main()
