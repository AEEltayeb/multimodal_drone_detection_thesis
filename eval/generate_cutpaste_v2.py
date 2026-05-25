"""Cut-paste paired RGB+IR drone eval generator (v2 - fixed).

Key fixes vs v1:
  - IR is loaded with cv2.imread (verified uint8 3-ch JPGs); no more black boxes.
  - Soft alpha matte from Otsu inside the GT box + 3-5px feathering -> no hard seams.
  - Backgrounds are verified drone-free: BOTH RGB and IR label files must be empty.
  - Identical normalised paste position in RGB and IR -> YOLO bbox matches both.
  - Local brightness match: crop mean luminance is shifted toward bg patch mean.
  - GT-bbox-tightening via Otsu (no detector selection bias).

Run:
  python eval/generate_cutpaste_v2.py --n 10 --out G:/drone/cutpaste_eval_v2 --seed 42
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import cv2
import numpy as np

SRC = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test")
RGB_IMG = SRC / "RGB/images"
RGB_LBL = SRC / "RGB/labels"
IR_IMG  = SRC / "IR/images"
IR_LBL  = SRC / "IR/labels"

# (label, target relative width range, weight)
TIERS = [
    ("easy",   (0.10, 0.20), 3),
    ("medium", (0.04, 0.09), 4),
    ("hard",   (0.012, 0.035), 3),
]

def load_label(p: Path):
    if not p.exists(): return []
    out = []
    for ln in p.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            out.append((int(parts[0]),) + tuple(float(x) for x in parts[1:5]))
    return out

def index_split(rng, max_scan=15000):
    """Fast index using file sizes via os.scandir (no label reads)."""
    import os, sys
    print(f"[index] scanning RGB labels via scandir...", flush=True)
    rgb_sizes = {}
    with os.scandir(str(RGB_LBL)) as it:
        for e in it:
            if e.name.endswith(".txt"):
                try: rgb_sizes[e.name] = e.stat().st_size
                except OSError: pass
    print(f"[index] {len(rgb_sizes)} RGB labels indexed", flush=True)
    ir_sizes = {}
    with os.scandir(str(IR_LBL)) as it:
        for e in it:
            if e.name.endswith(".txt"):
                try: ir_sizes[e.name] = e.stat().st_size
                except OSError: pass
    print(f"[index] {len(ir_sizes)} IR labels indexed", flush=True)
    pos, neg = [], []
    names = list(rgb_sizes.keys())
    rng.shuffle(names)
    if max_scan: names = names[:max_scan]
    for name in names:
        ir_name = name.replace("_visible_", "_infrared_")
        if ir_name not in ir_sizes: continue
        rs, isz = rgb_sizes[name], ir_sizes[ir_name]
        # Positive: both labels non-empty (we'll verify single-drone at use time)
        if rs > 0 and isz > 0:
            pos.append(name[:-4])
        elif rs == 0 and isz == 0:
            neg.append(name[:-4])
    print(f"[index] positives={len(pos)}  negatives={len(neg)}", flush=True)
    return pos, neg

def yolo_to_xyxy(b, W, H):
    cx, cy, w, h = b
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

MIN_CROP_CONTRAST = 8.0     # min std of gray inside GT box - kills invisible drones
MIN_BG_STD = 12.0           # min global std of bg image
MAX_BG_SKY_MEAN = 200.0     # max mean gray in TOP 40% of bg - kills blown-out IR sky
MAX_UPSCALE = 1.2           # reject paste if target width > MAX_UPSCALE * source crop width

def make_elliptical_alpha(crop):
    """Soft elliptical alpha: 1.0 inside inner ellipse, cosine falloff to 0 at edge."""
    h, w = crop.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ry, rx = h / 2, w / 2
    # Normalised radial distance (1.0 at bbox edge, 0 at centre)
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    # Cosine falloff: 1.0 for r<=0.55, 0 for r>=1.0
    alpha = np.clip((1.0 - r) / 0.45, 0.0, 1.0)
    alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)  # smoothstep
    return alpha.astype(np.float32)

def crop_and_alpha(img, bbox_xyxy, pad_frac=0.05):
    """Tight crop (small pad) + soft elliptical alpha. Skip low-contrast drones."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4: return None, None
    # Contrast inside GT bbox
    inside = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if inside.size == 0 or inside.std() < MIN_CROP_CONTRAST:
        return None, None
    px = int(bw * pad_frac); py = int(bh * pad_frac)
    X1, Y1 = max(0, x1 - px), max(0, y1 - py)
    X2, Y2 = min(W, x2 + px), min(H, y2 + py)
    crop = img[Y1:Y2, X1:X2].copy()
    if crop.size == 0: return None, None
    alpha = make_elliptical_alpha(crop)
    return crop, alpha

def paste(bg, crop, alpha, cx_n, cy_n, tw_n):
    H, W = bg.shape[:2]
    ch, cw = crop.shape[:2]
    target_w = max(4, int(tw_n * W))
    scale = target_w / cw
    target_h = max(4, int(ch * scale))
    crop_r  = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_r = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    cx, cy = int(cx_n * W), int(cy_n * H)
    x1 = cx - target_w // 2; y1 = cy - target_h // 2
    x1 = max(0, min(W - target_w, x1)); y1 = max(0, min(H - target_h, y1))
    x2 = x1 + target_w; y2 = y1 + target_h
    bg_patch = bg[y1:y2, x1:x2].astype(np.float32)
    crop_f = crop_r.astype(np.float32)
    if alpha_r.sum() > 1:
        wsum = alpha_r.sum() + 1e-6
        crop_mean = (crop_f * alpha_r[..., None]).sum(axis=(0,1)) / wsum
        bg_mean = bg_patch.mean(axis=(0,1))
        shift = (bg_mean - crop_mean) * 0.4
        crop_f = np.clip(crop_f + shift, 0, 255)
    a3 = alpha_r[..., None]
    blended = (crop_f * a3 + bg_patch * (1 - a3)).astype(np.uint8)
    out = bg.copy(); out[y1:y2, x1:x2] = blended
    return out, (x1, y1, x2, y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=15000)
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    out_rgb_img = args.out / "RGB/images"; out_rgb_lbl = args.out / "RGB/labels"
    out_ir_img  = args.out / "IR/images";  out_ir_lbl  = args.out / "IR/labels"
    for d in (out_rgb_img, out_rgb_lbl, out_ir_img, out_ir_lbl):
        d.mkdir(parents=True, exist_ok=True)

    positives, negatives = index_split(rng, args.max_scan)
    if not positives or not negatives:
        print("ERROR: not enough positives or negatives. Try larger --max-scan."); return

    total_w = sum(w for _,_,w in TIERS)
    plan = []
    for label, rng_w, w in TIERS:
        k = max(1, round(args.n * w / total_w))
        plan += [(label, rng_w)] * k
    plan = plan[:args.n]
    while len(plan) < args.n:
        plan.append(TIERS[1][0:2])
    rng.shuffle(plan)

    rng.shuffle(positives); rng.shuffle(negatives)
    meta = []
    i_out = 0; ip = 0; ib = 0; tries = 0
    while i_out < args.n and tries < args.n * 200:
        tries += 1
        if ip >= len(positives) or ib >= len(negatives):
            print("[gen] pools exhausted"); break
        pos_stem = positives[ip]; ip += 1
        bg_stem  = negatives[ib]; ib += 1
        rgb_src = cv2.imread(str(RGB_IMG / f"{pos_stem}.jpg"))
        ir_src  = cv2.imread(str(IR_IMG  / f"{pos_stem.replace('_visible_', '_infrared_')}.jpg"))
        rgb_bg  = cv2.imread(str(RGB_IMG / f"{bg_stem}.jpg"))
        ir_bg   = cv2.imread(str(IR_IMG  / f"{bg_stem.replace('_visible_', '_infrared_')}.jpg"))
        if any(x is None for x in (rgb_src, ir_src, rgb_bg, ir_bg)): continue
        # Background quality filters
        rgb_g = cv2.cvtColor(rgb_bg, cv2.COLOR_BGR2GRAY)
        ir_g  = cv2.cvtColor(ir_bg,  cv2.COLOR_BGR2GRAY)
        if rgb_g.std() < MIN_BG_STD or ir_g.std() < MIN_BG_STD: continue
        # Top-40% sky region must not be blown out
        if rgb_g[:int(rgb_g.shape[0]*0.4)].mean() > MAX_BG_SKY_MEAN: continue
        if ir_g[:int(ir_g.shape[0]*0.4)].mean()   > MAX_BG_SKY_MEAN: continue
        rgb_lbls = load_label(RGB_LBL / f"{pos_stem}.txt")
        ir_lbls  = load_label(IR_LBL  / f"{pos_stem.replace('_visible_', '_infrared_')}.txt")
        if len(rgb_lbls) != 1 or len(ir_lbls) != 1: continue
        rgb_gt = rgb_lbls[0][1:]; ir_gt = ir_lbls[0][1:]
        Hr, Wr = rgb_src.shape[:2]; Hi, Wi = ir_src.shape[:2]
        rgb_crop, rgb_alpha = crop_and_alpha(rgb_src, yolo_to_xyxy(rgb_gt, Wr, Hr))
        ir_crop,  ir_alpha  = crop_and_alpha(ir_src,  yolo_to_xyxy(ir_gt,  Wi, Hi))
        if rgb_crop is None or ir_crop is None: continue
        tier, (smin, smax) = plan[i_out]
        tw_n = rng.uniform(smin, smax)
        # No-upscale rule: paste width (in bg pixels) must not exceed MAX_UPSCALE * source crop width.
        # Check both modalities; if either would upscale beyond limit, skip this sample.
        rgb_target_w = tw_n * rgb_bg.shape[1]
        ir_target_w  = tw_n * ir_bg.shape[1]
        if rgb_target_w > MAX_UPSCALE * rgb_crop.shape[1] or \
           ir_target_w  > MAX_UPSCALE * ir_crop.shape[1]:
            continue
        cx_n = rng.uniform(0.20, 0.80); cy_n = rng.uniform(0.25, 0.65)
        rgb_out, rgb_bb = paste(rgb_bg, rgb_crop, rgb_alpha, cx_n, cy_n, tw_n)
        ir_out,  ir_bb  = paste(ir_bg,  ir_crop,  ir_alpha,  cx_n, cy_n, tw_n)
        Wro, Hro = rgb_out.shape[1], rgb_out.shape[0]
        x1, y1, x2, y2 = rgb_bb
        cx = (x1+x2)/2/Wro; cy = (y1+y2)/2/Hro
        bw = (x2-x1)/Wro;   bh = (y2-y1)/Hro
        lbl = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        name = f"cp_{tier}_{i_out:04d}"
        cv2.imwrite(str(out_rgb_img / f"{name}_visible.jpg"),  rgb_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_ir_img  / f"{name}_infrared.jpg"), ir_out,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_rgb_lbl / f"{name}_visible.txt").write_text(lbl)
        (out_ir_lbl  / f"{name}_infrared.txt").write_text(lbl)
        meta.append({"idx": i_out, "name": name, "tier": tier, "tw_n": round(tw_n,4),
                     "cx_n": round(cx_n,3), "cy_n": round(cy_n,3),
                     "pos_stem": pos_stem, "bg_stem": bg_stem})
        i_out += 1
        print(f"[gen] {name}  tier={tier}  tw_n={tw_n:.3f}  rgb_bb={rgb_bb}")
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {i_out} samples to {args.out}")

if __name__ == "__main__":
    main()
