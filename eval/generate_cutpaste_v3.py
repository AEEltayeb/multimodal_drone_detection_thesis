"""Cut-paste paired RGB+IR drone eval generator (v3 - real matting + day/night match).

vs v2:
  - True alpha matting via rembg (isnet-general-use) on each source crop -> real
    drone silhouette, no halo from elliptical mask.
  - Day/night brightness match: skip if mean luminance of source crop region
    differs from target bg sky region by more than MAX_BRIGHTNESS_DIFF.

Run:
  python eval/generate_cutpaste_v3.py --n 10 --out G:/drone/cutpaste_eval_v3 --seed 1
"""
from __future__ import annotations
import argparse, json, random, os
from pathlib import Path
import cv2
import numpy as np
from rembg import remove, new_session

SRC = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test")
RGB_IMG = SRC / "RGB/images"; RGB_LBL = SRC / "RGB/labels"
IR_IMG  = SRC / "IR/images";  IR_LBL  = SRC / "IR/labels"

TIERS = [
    ("easy",   (0.10, 0.18), 3),
    ("medium", (0.04, 0.09), 4),
    ("hard",   (0.012, 0.035), 3),
]

MIN_CROP_CONTRAST = 8.0
MIN_BG_STD = 12.0
MAX_BG_SKY_MEAN = 200.0
MAX_UPSCALE = 1.5
MAX_BRIGHTNESS_DIFF = 60.0   # max abs difference between crop-region mean and bg-sky mean (0-255)
MIN_ALPHA_AREA = 9            # min foreground pixels in matte to accept

_session = None
def get_session():
    global _session
    if _session is None:
        print("[rembg] loading isnet-general-use session...", flush=True)
        _session = new_session("isnet-general-use")
    return _session

def load_label(p):
    if not p.exists(): return []
    out = []
    for ln in p.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            out.append((int(parts[0]),) + tuple(float(x) for x in parts[1:5]))
    return out

def index_split(rng, max_scan=0):
    print("[index] scandir...", flush=True)
    rgb_sizes = {e.name: e.stat().st_size for e in os.scandir(str(RGB_LBL)) if e.name.endswith(".txt")}
    ir_sizes  = {e.name: e.stat().st_size for e in os.scandir(str(IR_LBL))  if e.name.endswith(".txt")}
    pos, neg = [], []
    names = list(rgb_sizes.keys())
    rng.shuffle(names)
    if max_scan: names = names[:max_scan]
    for name in names:
        ir_name = name.replace("_visible_", "_infrared_")
        if ir_name not in ir_sizes: continue
        rs, isz = rgb_sizes[name], ir_sizes[ir_name]
        if rs > 0 and isz > 0:   pos.append(name[:-4])
        elif rs == 0 and isz == 0: neg.append(name[:-4])
    print(f"[index] pos={len(pos)} neg={len(neg)}", flush=True)
    return pos, neg

def yolo_to_xyxy(b, W, H):
    cx, cy, w, h = b
    return (max(0,int((cx-w/2)*W)), max(0,int((cy-h/2)*H)),
            min(W,int((cx+w/2)*W)), min(H,int((cy+h/2)*H)))

def crop_and_matte(img, bbox_xyxy, pad_frac=0.15, is_ir=False):
    """Tight crop around GT box + rembg alpha matte. Returns (crop, alpha 0..1) or (None,None)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4: return None, None
    inside = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if inside.size == 0 or inside.std() < MIN_CROP_CONTRAST: return None, None
    px = int(bw * pad_frac); py = int(bh * pad_frac)
    X1, Y1 = max(0, x1 - px), max(0, y1 - py)
    X2, Y2 = min(W, x2 + px), min(H, y2 + py)
    crop = img[Y1:Y2, X1:X2].copy()
    if crop.size == 0: return None, None
    # rembg wants RGBA-like input; pass BGR -> rembg returns BGRA
    matted = remove(crop, session=get_session())  # ndarray with alpha
    if matted.ndim != 3 or matted.shape[2] != 4: return None, None
    alpha = matted[..., 3].astype(np.float32) / 255.0
    if alpha.sum() < MIN_ALPHA_AREA: return None, None
    # Tighten to alpha support bbox
    ys, xs = np.where(alpha > 0.1)
    if len(xs) < 3: return None, None
    tx1, ty1, tx2, ty2 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    crop = crop[ty1:ty2, tx1:tx2]; alpha = alpha[ty1:ty2, tx1:tx2]
    # Slight feather
    alpha = cv2.GaussianBlur(alpha, (3,3), 0)
    return crop, alpha

def sky_mean(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(g[:int(g.shape[0]*0.4)].mean())

def crop_region_mean(img, bbox_xyxy, pad=0.3):
    x1,y1,x2,y2 = bbox_xyxy
    bw,bh = x2-x1, y2-y1
    px,py = int(bw*pad), int(bh*pad)
    H,W = img.shape[:2]
    X1,Y1 = max(0,x1-px), max(0,y1-py)
    X2,Y2 = min(W,x2+px), min(H,y2+py)
    g = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2GRAY)
    return float(g.mean()) if g.size else 0.0

def paste(bg, crop, alpha, cx_n, cy_n, tw_n):
    H, W = bg.shape[:2]
    ch, cw = crop.shape[:2]
    target_w = max(4, int(tw_n * W))
    target_h = max(4, int(ch * target_w / cw))
    crop_r  = cv2.resize(crop,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_r = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    cx, cy = int(cx_n * W), int(cy_n * H)
    x1 = max(0, min(W - target_w, cx - target_w // 2))
    y1 = max(0, min(H - target_h, cy - target_h // 2))
    x2 = x1 + target_w; y2 = y1 + target_h
    bg_patch = bg[y1:y2, x1:x2].astype(np.float32)
    crop_f = crop_r.astype(np.float32)
    if alpha_r.sum() > 1:
        wsum = alpha_r.sum() + 1e-6
        crop_mean = (crop_f * alpha_r[..., None]).sum(axis=(0,1)) / wsum
        bg_mean = bg_patch.mean(axis=(0,1))
        shift = (bg_mean - crop_mean) * 0.3
        crop_f = np.clip(crop_f + shift, 0, 255)
    a3 = alpha_r[..., None]
    blended = (crop_f * a3 + bg_patch * (1 - a3)).astype(np.uint8)
    out = bg.copy(); out[y1:y2, x1:x2] = blended
    return out, (x1, y1, x2, y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-scan", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    out_rgb_img = args.out / "RGB/images"; out_rgb_lbl = args.out / "RGB/labels"
    out_ir_img  = args.out / "IR/images";  out_ir_lbl  = args.out / "IR/labels"
    for d in (out_rgb_img, out_rgb_lbl, out_ir_img, out_ir_lbl): d.mkdir(parents=True, exist_ok=True)

    positives, negatives = index_split(rng, args.max_scan)
    if not positives or not negatives: print("ERROR: empty pools"); return

    total_w = sum(w for _,_,w in TIERS)
    plan = []
    for label, rng_w, w in TIERS:
        plan += [(label, rng_w)] * max(1, round(args.n * w / total_w))
    plan = plan[:args.n]
    while len(plan) < args.n: plan.append(TIERS[1][0:2])
    rng.shuffle(plan)
    rng.shuffle(positives); rng.shuffle(negatives)

    # Build CROP BANK: pre-extract & matte a target number of drone crops.
    crop_bank = []   # list of dicts: {stem, rgb_crop, rgb_alpha, ir_crop, ir_alpha, rgb_region_mean, src_w_rgb, src_w_ir}
    bg_bank = []     # list of dicts: {stem, rgb_bg, ir_bg, sky_mean_rgb}
    target_crops = max(8, args.n // 4)   # 250 crops for 1000 samples, etc.
    target_bgs   = max(8, args.n // 8)
    print(f"[bank] targeting {target_crops} crops / {target_bgs} bgs", flush=True)

    rejects = {"bg_std":0,"bg_sky":0,"matte":0,"crop":0,"crop_lowcontrast":0}
    # ---- Fill crop bank ----
    ip = 0
    while len(crop_bank) < target_crops and ip < len(positives):
        pos_stem = positives[ip]; ip += 1
        rgb_src = cv2.imread(str(RGB_IMG / f"{pos_stem}.jpg"))
        ir_src  = cv2.imread(str(IR_IMG  / f"{pos_stem.replace('_visible_', '_infrared_')}.jpg"))
        if rgb_src is None or ir_src is None: continue
        rgb_lbls = load_label(RGB_LBL / f"{pos_stem}.txt")
        ir_lbls  = load_label(IR_LBL  / f"{pos_stem.replace('_visible_', '_infrared_')}.txt")
        if len(rgb_lbls) != 1 or len(ir_lbls) != 1: continue
        Hr, Wr = rgb_src.shape[:2]; Hi, Wi = ir_src.shape[:2]
        rgb_bbox = yolo_to_xyxy(rgb_lbls[0][1:], Wr, Hr)
        ir_bbox  = yolo_to_xyxy(ir_lbls[0][1:],  Wi, Hi)
        rgb_region = crop_region_mean(rgb_src, rgb_bbox)
        rgb_crop, rgb_alpha = crop_and_matte(rgb_src, rgb_bbox, is_ir=False)
        ir_crop,  ir_alpha  = crop_and_matte(ir_src,  ir_bbox,  is_ir=True)
        if rgb_crop is None or ir_crop is None:
            rejects["matte"] += 1; continue
        crop_bank.append({"stem":pos_stem, "rgb_crop":rgb_crop, "rgb_alpha":rgb_alpha,
                          "ir_crop":ir_crop, "ir_alpha":ir_alpha, "rgb_region_mean":rgb_region})
        if len(crop_bank) % 5 == 0:
            print(f"[bank] crops {len(crop_bank)}/{target_crops}", flush=True)
    print(f"[bank] crops final={len(crop_bank)} (scanned {ip}, rejects={rejects})", flush=True)
    if not crop_bank: print("ERROR: no crops"); return

    # ---- Fill bg bank ----
    ib = 0
    while len(bg_bank) < target_bgs and ib < len(negatives):
        bg_stem = negatives[ib]; ib += 1
        rgb_bg = cv2.imread(str(RGB_IMG / f"{bg_stem}.jpg"))
        ir_bg  = cv2.imread(str(IR_IMG  / f"{bg_stem.replace('_visible_', '_infrared_')}.jpg"))
        if rgb_bg is None or ir_bg is None: continue
        rgb_g = cv2.cvtColor(rgb_bg, cv2.COLOR_BGR2GRAY)
        ir_g  = cv2.cvtColor(ir_bg,  cv2.COLOR_BGR2GRAY)
        if rgb_g.std() < MIN_BG_STD or ir_g.std() < MIN_BG_STD: rejects["bg_std"]+=1; continue
        sm = sky_mean(rgb_bg)
        if sm > MAX_BG_SKY_MEAN or sky_mean(ir_bg) > MAX_BG_SKY_MEAN: rejects["bg_sky"]+=1; continue
        bg_bank.append({"stem":bg_stem, "rgb_bg":rgb_bg, "ir_bg":ir_bg, "sky_mean_rgb":sm})
    print(f"[bank] bgs final={len(bg_bank)} (scanned {ib})", flush=True)
    if not bg_bank: print("ERROR: no bgs"); return

    # ---- Sample compositing ----
    meta = []; i_out = 0; tries = 0
    upscale_rej = brightness_rej = 0
    while i_out < args.n and tries < args.n * 60:
        tries += 1
        c = rng.choice(crop_bank); b = rng.choice(bg_bank)
        if abs(c["rgb_region_mean"] - b["sky_mean_rgb"]) > MAX_BRIGHTNESS_DIFF:
            brightness_rej += 1; continue
        tier, (smin, smax) = plan[i_out]
        tw_n = rng.uniform(smin, smax)
        if (tw_n * b["rgb_bg"].shape[1] > MAX_UPSCALE * c["rgb_crop"].shape[1] or
            tw_n * b["ir_bg"].shape[1]  > MAX_UPSCALE * c["ir_crop"].shape[1]):
            upscale_rej += 1; continue
        cx_n = rng.uniform(0.20, 0.80); cy_n = rng.uniform(0.25, 0.65)
        rgb_out, rgb_bb = paste(b["rgb_bg"], c["rgb_crop"], c["rgb_alpha"], cx_n, cy_n, tw_n)
        ir_out,  ir_bb  = paste(b["ir_bg"],  c["ir_crop"],  c["ir_alpha"],  cx_n, cy_n, tw_n)
        Wro, Hro = rgb_out.shape[1], rgb_out.shape[0]
        x1, y1, x2, y2 = rgb_bb
        lbl = f"0 {(x1+x2)/2/Wro:.6f} {(y1+y2)/2/Hro:.6f} {(x2-x1)/Wro:.6f} {(y2-y1)/Hro:.6f}\n"
        name = f"cp_{tier}_{i_out:04d}"
        cv2.imwrite(str(out_rgb_img / f"{name}_visible.jpg"),  rgb_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_ir_img  / f"{name}_infrared.jpg"), ir_out,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_rgb_lbl / f"{name}_visible.txt").write_text(lbl)
        (out_ir_lbl  / f"{name}_infrared.txt").write_text(lbl)
        meta.append({"idx": i_out, "name": name, "tier": tier, "tw_n": round(tw_n,4),
                     "pos_stem": c["stem"], "bg_stem": b["stem"]})
        i_out += 1
        if i_out % 10 == 0 or i_out <= 5:
            print(f"[gen] {name} tier={tier} tw_n={tw_n:.3f}  ({i_out}/{args.n})", flush=True)

    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {i_out} samples; brightness_rej={brightness_rej} upscale_rej={upscale_rej} matte_rejects={rejects['matte']}", flush=True)

if __name__ == "__main__":
    main()
