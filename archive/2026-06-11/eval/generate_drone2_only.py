"""Sanity dataset: paste drone 2 (only pair both models detect) onto 10 NIRScene bgs."""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys; sys.path.insert(0, str(REPO / "eval"))
from generate_cutpaste_v4 import (
    soft_elliptical_alpha, paste, ensure_3ch, NIRSCENE, NIR_CATS,
    MIN_PASTE_PX, SIZE_TIERS, TIER_WEIGHTS, DRONES_PER_IMAGE, SCENE_CONTEXTS,
)

DRONE_DIR = Path(r"G:/drone/drone assets")
# Detector-derived bboxes (from probe run):
RGB_BBOX = (167, 154, 374, 292)   # drone 2 rgb
IR_BBOX  = (123, 111, 432, 310)   # drone 2 ir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed)

    for sub in ["RGB/images","RGB/labels","IR/images","IR/labels"]:
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    rgb_src = cv2.imread(str(DRONE_DIR / "drone 2 rgb.png"))
    ir_src  = cv2.imread(str(DRONE_DIR / "drone 2 ir.png"))
    assert rgb_src is not None and ir_src is not None
    rgb_crop = rgb_src[RGB_BBOX[1]:RGB_BBOX[3], RGB_BBOX[0]:RGB_BBOX[2]].copy()
    ir_crop  = ir_src[IR_BBOX[1]:IR_BBOX[3],  IR_BBOX[0]:IR_BBOX[2]].copy()
    rgb_alpha = soft_elliptical_alpha(*rgb_crop.shape[:2])
    ir_alpha  = soft_elliptical_alpha(*ir_crop.shape[:2])
    print(f"[crops] rgb {rgb_crop.shape}, ir {ir_crop.shape}", flush=True)

    # NIRScene paired bgs (only outdoor 4 categories)
    nir_pool = []
    for cat in NIR_CATS:
        for f in sorted((NIRSCENE / cat).glob("*_rgb.tiff")):
            nir_pool.append(f)
    rng.shuffle(nir_pool)
    bgs = []
    for rgb_p in nir_pool:
        if len(bgs) >= args.n: break
        nir_p = rgb_p.parent / rgb_p.name.replace("_rgb", "_nir")
        rgb_bg = ensure_3ch(cv2.imread(str(rgb_p), cv2.IMREAD_UNCHANGED))
        ir_bg  = ensure_3ch(cv2.imread(str(nir_p), cv2.IMREAD_UNCHANGED))
        if rgb_bg is None or ir_bg is None: continue
        bgs.append((rgb_p.stem, rgb_bg, ir_bg))
    print(f"[bg] {len(bgs)} pairs", flush=True)

    meta = []
    for i, (stem, rgb_bg, ir_bg) in enumerate(bgs):
        rgb_out = rgb_bg.copy(); ir_out = ir_bg.copy()
        Hr,Wr = rgb_out.shape[:2]; Hi,Wi = ir_out.shape[:2]
        boxes_rgb=[]; boxes_ir=[]; placements=[]
        for k in range(DRONES_PER_IMAGE):
            tier = rng.choices(list(TIER_WEIGHTS), weights=list(TIER_WEIGHTS.values()))[0]
            ctx  = SCENE_CONTEXTS[k % len(SCENE_CONTEXTS)]
            lo, hi = SIZE_TIERS[tier]
            target_w_rgb = rng.randint(lo, hi)
            # cap upscale at 1.0x source crop width
            target_w_rgb = min(target_w_rgb, rgb_crop.shape[1])
            target_w_rgb = max(target_w_rgb, MIN_PASTE_PX)
            target_w_ir = max(MIN_PASTE_PX, int(target_w_rgb * (Wi / Wr)))
            target_w_ir = min(target_w_ir, ir_crop.shape[1])

            if ctx == "sky":     cy_n = rng.uniform(0.05, 0.35)
            elif ctx == "clutter": cy_n = rng.uniform(0.30, 0.65)
            else:                cy_n = rng.uniform(0.55, 0.90)
            cx_n = rng.uniform(0.10, 0.90)
            cx_r,cy_r = int(cx_n*Wr), int(cy_n*Hr)
            cx_i,cy_i = int(cx_n*Wi), int(cy_n*Hi)
            bb_r = paste(rgb_out, rgb_crop, rgb_alpha, cx_r, cy_r, target_w_rgb)
            bb_i = paste(ir_out,  ir_crop,  ir_alpha,  cx_i, cy_i, target_w_ir)
            boxes_rgb.append(bb_r); boxes_ir.append(bb_i)
            placements.append({"tier":tier, "ctx":ctx, "target_w_rgb":target_w_rgb})

        def lbl(boxes, W, H):
            lines=[]
            for (x1,y1,x2,y2) in boxes:
                lines.append(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
            return "\n".join(lines)+"\n"
        name = f"d2_{i:03d}"
        cv2.imwrite(str(args.out / "RGB/images" / f"{name}_visible.jpg"),  rgb_out, [cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(args.out / "IR/images"  / f"{name}_infrared.jpg"), ir_out,  [cv2.IMWRITE_JPEG_QUALITY,95])
        (args.out / "RGB/labels" / f"{name}_visible.txt").write_text(lbl(boxes_rgb, Wr, Hr))
        (args.out / "IR/labels"  / f"{name}_infrared.txt").write_text(lbl(boxes_ir, Wi, Hi))
        meta.append({"name":name, "bg":stem, "placements":placements})
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {len(meta)} samples -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
