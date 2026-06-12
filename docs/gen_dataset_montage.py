#!/usr/bin/env python
"""gen_dataset_montage.py - 'small silhouette against sky' montage: GT-boxed drone
crops across RGB/IR and Svanstrom/Anti-UAV, illustrating the visual smallness that
motivates the cascade. No model inference (GT boxes only). CPU.

  py docs/gen_dataset_montage.py
Outputs: docs/figures/fig_dataset_montage.{pdf,png}
"""
import os, glob, random
from PIL import Image, ImageDraw, ImageFont
random.seed(1)
OUT = "docs/figures/fig_dataset_montage"

# (label, labels_dir, images_dir, n_cells)
SRC = [
    ("Svanström RGB", "G:/drone/svanstrom_paired/RGB/labels", "G:/drone/svanstrom_paired/RGB/images", 2),
    ("Svanström IR",  "G:/drone/svanstrom_paired/IR/labels",  "G:/drone/svanstrom_paired/IR/images",  2),
    ("Anti-UAV RGB",  "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels", "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images", 2),
    ("Anti-UAV IR",   "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels",  "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images",  2),
]
CELL = 256


def find_image(images_dir, stem):
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def cell_from(labels_dir, images_dir, label, used):
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    random.shuffle(files)
    for fp in files:
        if fp in used:
            continue
        try:
            box = None
            for ln in open(fp, encoding="utf-8"):
                p = ln.split()
                if len(p) >= 5 and p[0] == "0":
                    box = tuple(map(float, p[1:5])); break
            if not box:
                continue
            img_p = find_image(images_dir, os.path.splitext(os.path.basename(fp))[0])
            if not img_p:
                continue
            im = Image.open(img_p).convert("RGB"); W, H = im.size
            cx, cy, bw, bh = box[0]*W, box[1]*H, box[2]*W, box[3]*H
            # window ~ 4x box, min 180px, centered on box
            win = max(bw, bh) * 4; win = max(win, 180); win = min(win, min(W, H))
            x0 = max(0, min(W - win, cx - win/2)); y0 = max(0, min(H - win, cy - win/2))
            crop = im.crop((int(x0), int(y0), int(x0+win), int(y0+win))).resize((CELL, CELL))
            d = ImageDraw.Draw(crop)
            sc = CELL / win
            bx0 = (cx - bw/2 - x0)*sc; by0 = (cy - bh/2 - y0)*sc
            d.rectangle([bx0, by0, bx0 + bw*sc, by0 + bh*sc], outline=(255, 40, 40), width=3)
            d.rectangle([0, 0, CELL-1, 18], fill=(0, 0, 0))
            d.text((4, 4), f"{label}  (drone {int(max(bw,bh))}px)", fill=(255, 255, 255))
            used.add(fp)
            return crop
        except Exception:
            continue
    return Image.new("RGB", (CELL, CELL), (60, 60, 60))


def main():
    cells, used = [], set()
    for label, ld, idd, n in SRC:
        if not os.path.isdir(ld):
            print("  skip (no dir):", label); continue
        for _ in range(n):
            cells.append(cell_from(ld, idd, label, used))
    cols = 4; rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols*CELL, rows*CELL), (255, 255, 255))
    for i, c in enumerate(cells):
        grid.paste(c, ((i % cols)*CELL, (i // cols)*CELL))
    os.makedirs("docs/figures", exist_ok=True)
    grid.save(OUT + ".png")
    grid.save(OUT + ".pdf")
    print(f"wrote {OUT}.{{png,pdf}}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
