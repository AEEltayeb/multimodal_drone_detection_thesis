"""Cut-paste paired RGB+IR drone eval generator (v4).

Pipeline:
  1. Detector-validated drone crop bank
     - Iterate Anti-UAV test positives.
     - Run RGB detector + IR detector on the source frame (cropped 2x GT region).
     - Accept if either modality fires conf>=0.5 inside the GT box.
     - Tighten the crop to the detector bbox (per modality) if available; else GT.
     - Skip if either modality's tightened bbox < 8x8 px in source.
     - Build soft elliptical alpha.
  2. Background bank (mixed)
     - Anti-UAV paired negatives (LWIR IR + RGB).
     - NIRScene1 outdoor categories (RGB + NIR-as-IR).
     - Filter: bg std and sky-mean.
  3. Composite N images, each with 3 drones (size tiers small/medium/large in
     scene contexts sky/clutter/ground) -> multi-bbox YOLO label.

Usage:
  python eval/generate_cutpaste_v4.py --n 1000 --out G:/drone/cutpaste_drone_v4 --seed 1
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────
ANTIUAV = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test")
AU_RGB_IMG = ANTIUAV / "RGB/images"; AU_RGB_LBL = ANTIUAV / "RGB/labels"
AU_IR_IMG  = ANTIUAV / "IR/images";  AU_IR_LBL  = ANTIUAV / "IR/labels"

NIRSCENE = Path(r"G:/drone/nirscene1/nirscene1")
NIR_CATS = ["water", "street", "field", "mountain"]

# ── Models for detector-validated cropping ────────────────────────────
RGB_WEIGHTS = REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"
IR_WEIGHTS  = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

# ── Composition parameters ────────────────────────────────────────────
DRONES_PER_IMAGE = 3
SIZE_TIERS = {
    "small":  (8, 20),     # paste width in pixels
    "medium": (20, 80),
    "large":  (80, 400),
}
TIER_WEIGHTS = {"small": 0.25, "medium": 0.50, "large": 0.25}
SCENE_CONTEXTS = ["sky", "clutter", "ground"]  # not enforced strictly; used as metadata
MIN_PASTE_PX = 8
MAX_BRIGHTNESS_DIFF = 60.0
MAX_BG_SKY_MEAN = 200.0
MIN_BG_STD = 12.0

# ── Helpers ───────────────────────────────────────────────────────────
def load_label(p: Path):
    if not p.exists(): return []
    out = []
    for ln in p.read_text().strip().splitlines():
        parts = ln.split()
        if len(parts) >= 5:
            out.append((int(parts[0]),) + tuple(float(x) for x in parts[1:5]))
    return out

def yolo_to_xyxy(b, W, H):
    cx, cy, w, h = b
    return (max(0,int((cx-w/2)*W)), max(0,int((cy-h/2)*H)),
            min(W,int((cx+w/2)*W)), min(H,int((cy+h/2)*H)))

def soft_elliptical_alpha(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ry, rx = h / 2, w / 2
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    alpha = np.clip((1.0 - r) / 0.45, 0.0, 1.0)
    return (0.5 - 0.5 * np.cos(np.pi * alpha)).astype(np.float32)

def sky_mean(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(g[:int(g.shape[0]*0.4)].mean())

def ensure_3ch(img):
    if img is None: return None
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1: return cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# ── Crop bank with detector validation ────────────────────────────────
def build_crop_bank(target_n, rng, max_scan=4000, conf_thr=0.5, stem_allowlist=None):
    print(f"[bank] loading detectors RGB={RGB_WEIGHTS.name}, IR={IR_WEIGHTS.name}", flush=True)
    from ultralytics import YOLO
    rgb_model = YOLO(str(RGB_WEIGHTS)); ir_model = YOLO(str(IR_WEIGHTS))

    print("[bank] scanning Anti-UAV positives...", flush=True)
    rgb_sizes = {e.name: e.stat().st_size for e in os.scandir(str(AU_RGB_LBL)) if e.name.endswith(".txt")}
    ir_sizes  = {e.name: e.stat().st_size for e in os.scandir(str(AU_IR_LBL))  if e.name.endswith(".txt")}
    positives = [n[:-4] for n in rgb_sizes if rgb_sizes[n] > 0
                 and n.replace("_visible_", "_infrared_") in ir_sizes
                 and ir_sizes[n.replace("_visible_", "_infrared_")] > 0]
    if stem_allowlist:
        allow = set(stem_allowlist)
        positives = [p for p in positives if p in allow]
        print(f"[bank] allowlist filtered -> {len(positives)} candidates", flush=True)
    else:
        rng.shuffle(positives); positives = positives[:max_scan]
        print(f"[bank] {len(positives)} candidates", flush=True)

    bank = []
    for i, stem in enumerate(positives):
        if len(bank) >= target_n: break
        if i % 50 == 0:
            print(f"[bank] scanned {i}/{len(positives)} -> {len(bank)} kept", flush=True)
        rgb_src = cv2.imread(str(AU_RGB_IMG / f"{stem}.jpg"))
        ir_src  = cv2.imread(str(AU_IR_IMG  / f"{stem.replace('_visible_', '_infrared_')}.jpg"))
        if rgb_src is None or ir_src is None: continue
        rl = load_label(AU_RGB_LBL / f"{stem}.txt")
        il = load_label(AU_IR_LBL  / f"{stem.replace('_visible_', '_infrared_')}.txt")
        if len(rl) != 1 or len(il) != 1: continue
        Hr,Wr = rgb_src.shape[:2]; Hi,Wi = ir_src.shape[:2]
        rgt = yolo_to_xyxy(rl[0][1:], Wr, Hr)
        igt = yolo_to_xyxy(il[0][1:], Wi, Hi)

        # Run RGB model on full frame at imgsz=1280, look for box overlapping GT
        rgb_box = None
        try:
            res = rgb_model.predict(rgb_src, imgsz=1280, conf=conf_thr, verbose=False)[0]
            if len(res.boxes) > 0:
                # best box whose IoU with GT > 0.3
                boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
                gx1,gy1,gx2,gy2 = rgt
                ious = []
                for (x1,y1,x2,y2) in boxes:
                    ix1,iy1 = max(x1,gx1), max(y1,gy1); ix2,iy2 = min(x2,gx2), min(y2,gy2)
                    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                    union = (x2-x1)*(y2-y1) + (gx2-gx1)*(gy2-gy1) - inter
                    ious.append(inter/union if union>0 else 0)
                ious = np.array(ious)
                ok = np.where(ious > 0.3)[0]
                if len(ok):
                    j = ok[confs[ok].argmax()]
                    rgb_box = tuple(int(v) for v in boxes[j])
        except Exception: pass

        ir_box = None
        try:
            res = ir_model.predict(ir_src, imgsz=640, conf=conf_thr, verbose=False)[0]
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
                gx1,gy1,gx2,gy2 = igt
                ious = []
                for (x1,y1,x2,y2) in boxes:
                    ix1,iy1 = max(x1,gx1), max(y1,gy1); ix2,iy2 = min(x2,gx2), min(y2,gy2)
                    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
                    union = (x2-x1)*(y2-y1) + (gx2-gx1)*(gy2-gy1) - inter
                    ious.append(inter/union if union>0 else 0)
                ious = np.array(ious)
                ok = np.where(ious > 0.3)[0]
                if len(ok):
                    j = ok[confs[ok].argmax()]
                    ir_box = tuple(int(v) for v in boxes[j])
        except Exception: pass

        # Require BOTH to fire (so the bank only has high-quality drones in both modalities)
        if rgb_box is None or ir_box is None: continue
        # Choose the detector box where available, else GT
        rx1,ry1,rx2,ry2 = rgb_box
        ix1,iy1,ix2,iy2 = ir_box
        if (rx2-rx1) < 8 or (ry2-ry1) < 8: continue
        if (ix2-ix1) < 8 or (iy2-iy1) < 8: continue

        # Tight crops + elliptical alpha (no pad - detector bbox is already tight)
        pad = 0.0
        def tight(img, bx):
            x1,y1,x2,y2 = bx; bw,bh = x2-x1, y2-y1
            px,py = int(bw*pad), int(bh*pad)
            H,W = img.shape[:2]
            X1,Y1 = max(0,x1-px), max(0,y1-py)
            X2,Y2 = min(W,x2+px), min(H,y2+py)
            crop = img[Y1:Y2, X1:X2].copy()
            return crop
        rgb_crop = tight(rgb_src, rgb_box)
        ir_crop  = tight(ir_src,  ir_box)
        if rgb_crop.size == 0 or ir_crop.size == 0: continue
        rgb_alpha = soft_elliptical_alpha(*rgb_crop.shape[:2])
        ir_alpha  = soft_elliptical_alpha(*ir_crop.shape[:2])

        # Per-modality region mean for brightness match
        rgb_mean = float(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2GRAY).mean())
        bank.append({
            "stem": stem, "rgb_crop": rgb_crop, "rgb_alpha": rgb_alpha,
            "ir_crop": ir_crop, "ir_alpha": ir_alpha,
            "rgb_mean": rgb_mean,
            "rgb_src_w": rgb_crop.shape[1], "ir_src_w": ir_crop.shape[1],
        })
    print(f"[bank] crops final={len(bank)}", flush=True)
    return bank

# ── Background bank (mixed Anti-UAV + NIRScene) ───────────────────────
def build_bg_bank(target_n, rng):
    print("[bg] scanning Anti-UAV negatives + NIRScene...", flush=True)
    bgs = []

    # Anti-UAV paired negatives
    rgb_sizes = {e.name: e.stat().st_size for e in os.scandir(str(AU_RGB_LBL)) if e.name.endswith(".txt")}
    ir_sizes  = {e.name: e.stat().st_size for e in os.scandir(str(AU_IR_LBL))  if e.name.endswith(".txt")}
    au_negs = []
    for n, sz in rgb_sizes.items():
        if sz != 0: continue
        ir_n = n.replace("_visible_", "_infrared_")
        if ir_n in ir_sizes and ir_sizes[ir_n] == 0:
            au_negs.append(n[:-4])
    rng.shuffle(au_negs)
    for stem in au_negs:
        if len(bgs) >= target_n // 2: break
        rgb = cv2.imread(str(AU_RGB_IMG / f"{stem}.jpg"))
        ir  = cv2.imread(str(AU_IR_IMG  / f"{stem.replace('_visible_', '_infrared_')}.jpg"))
        if rgb is None or ir is None: continue
        rgb = ensure_3ch(rgb); ir = ensure_3ch(ir)
        if cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).std() < MIN_BG_STD: continue
        if cv2.cvtColor(ir,  cv2.COLOR_BGR2GRAY).std() < MIN_BG_STD: continue
        if sky_mean(rgb) > MAX_BG_SKY_MEAN or sky_mean(ir) > MAX_BG_SKY_MEAN: continue
        bgs.append({"source": "antiuav", "stem": stem, "rgb": rgb, "ir": ir,
                    "sky_mean": sky_mean(rgb)})
    print(f"[bg] anti-uav: {len(bgs)}", flush=True)

    # NIRScene1
    nir_pool = []
    for cat in NIR_CATS:
        cdir = NIRSCENE / cat
        if not cdir.exists(): continue
        for f in sorted(cdir.glob("*_rgb.tiff")):
            nir_pool.append((cat, f))
    rng.shuffle(nir_pool)
    before = len(bgs)
    for cat, rgb_p in nir_pool:
        if len(bgs) >= target_n: break
        nir_p = rgb_p.parent / rgb_p.name.replace("_rgb", "_nir")
        rgb = cv2.imread(str(rgb_p), cv2.IMREAD_UNCHANGED)
        nir = cv2.imread(str(nir_p), cv2.IMREAD_UNCHANGED)
        if rgb is None or nir is None: continue
        rgb = ensure_3ch(rgb); nir = ensure_3ch(nir)
        if cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).std() < MIN_BG_STD: continue
        bgs.append({"source": f"nirscene_{cat}", "stem": rgb_p.stem, "rgb": rgb, "ir": nir,
                    "sky_mean": sky_mean(rgb)})
    print(f"[bg] +nirscene: {len(bgs) - before}", flush=True)
    print(f"[bg] total={len(bgs)}", flush=True)
    return bgs

# ── Paste ─────────────────────────────────────────────────────────────
def paste(bg, crop, alpha, cx, cy, target_w):
    # Both bg and crop must be 3-channel; caller ensures via ensure_3ch.
    H, W = bg.shape[:2]
    ch, cw = crop.shape[:2]
    target_w = max(MIN_PASTE_PX, int(target_w))
    target_h = max(MIN_PASTE_PX, int(ch * target_w / cw))
    crop_r  = cv2.resize(crop,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_r = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, min(W - target_w, cx - target_w // 2))
    y1 = max(0, min(H - target_h, cy - target_h // 2))
    x2 = x1 + target_w; y2 = y1 + target_h
    bg_patch = bg[y1:y2, x1:x2].astype(np.float32)
    crop_f = crop_r.astype(np.float32)
    if alpha_r.sum() > 1:
        wsum = alpha_r.sum() + 1e-6
        cm = (crop_f * alpha_r[..., None]).sum(axis=(0,1)) / wsum
        bm = bg_patch.mean(axis=(0,1))
        crop_f = np.clip(crop_f + (bm - cm) * 0.3, 0, 255)
    a3 = alpha_r[..., None]
    bg[y1:y2, x1:x2] = (crop_f * a3 + bg_patch * (1 - a3)).astype(np.uint8)
    return (x1, y1, x2, y2)

# ── Main ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-crops", type=int, default=120)
    ap.add_argument("--target-bgs", type=int, default=200)
    ap.add_argument("--crop-stems", type=str, default=None,
                    help="Comma-separated list of source crop stems to use exclusively")
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    out = args.out
    for sub in ["RGB/images", "RGB/labels", "IR/images", "IR/labels"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    allow = args.crop_stems.split(",") if args.crop_stems else None
    bank = build_crop_bank(args.target_crops, rng, stem_allowlist=allow)
    if not bank: print("ERROR: empty crop bank"); sys.exit(1)
    bgs = build_bg_bank(args.target_bgs, rng)
    if not bgs: print("ERROR: empty bg bank"); sys.exit(1)

    meta = []; rejects = 0
    for i in range(args.n):
        b = rng.choice(bgs)
        # Day/night brightness match: pick a crop whose mean is within MAX_BRIGHTNESS_DIFF
        candidates = [c for c in bank if abs(c["rgb_mean"] - b["sky_mean"]) <= MAX_BRIGHTNESS_DIFF]
        if not candidates:
            rejects += 1; continue
        c = rng.choice(candidates)

        rgb_out = ensure_3ch(b["rgb"].copy())
        ir_out  = ensure_3ch(b["ir"].copy())
        Hr, Wr = rgb_out.shape[:2]; Hi, Wi = ir_out.shape[:2]
        boxes_rgb = []; boxes_ir = []; placements = []

        # Pick tier per drone weighted by overall distribution
        tiers = []
        for _ in range(DRONES_PER_IMAGE):
            r = rng.random(); acc = 0; chosen = "medium"
            for t, w in TIER_WEIGHTS.items():
                acc += w
                if r <= acc: chosen = t; break
            tiers.append(chosen)

        for k, tier in enumerate(tiers):
            ctx = SCENE_CONTEXTS[k % len(SCENE_CONTEXTS)]
            lo, hi = SIZE_TIERS[tier]
            # Width in target pixels for the OUTPUT (use RGB output res)
            target_w_rgb = rng.randint(lo, hi)
            # Constrain so it never upscales beyond 1.5x source
            target_w_rgb = min(target_w_rgb, int(1.5 * c["rgb_src_w"]))
            target_w_rgb = max(target_w_rgb, MIN_PASTE_PX)
            # IR target width: scale so same NORMALISED width as RGB
            target_w_ir = max(MIN_PASTE_PX, int(target_w_rgb * (Wi / Wr)))

            # Context-driven y placement bias
            if ctx == "sky":     cy_n = rng.uniform(0.05, 0.35)
            elif ctx == "clutter": cy_n = rng.uniform(0.30, 0.65)
            else:                cy_n = rng.uniform(0.55, 0.90)
            cx_n = rng.uniform(0.10, 0.90)

            cx_r, cy_r = int(cx_n * Wr), int(cy_n * Hr)
            cx_i, cy_i = int(cx_n * Wi), int(cy_n * Hi)
            bb_r = paste(rgb_out, c["rgb_crop"], c["rgb_alpha"], cx_r, cy_r, target_w_rgb)
            bb_i = paste(ir_out,  c["ir_crop"],  c["ir_alpha"],  cx_i, cy_i, target_w_ir)
            boxes_rgb.append(bb_r); boxes_ir.append(bb_i)
            placements.append({"tier": tier, "ctx": ctx, "target_w_rgb": target_w_rgb})

        # Write images + multi-bbox labels
        name = f"cp_{i:05d}"
        cv2.imwrite(str(out / "RGB/images" / f"{name}_visible.jpg"),  rgb_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "IR/images"  / f"{name}_infrared.jpg"), ir_out,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        def yolo_lbl(boxes, W, H):
            lines = []
            for (x1,y1,x2,y2) in boxes:
                cx = (x1+x2)/2/W; cy = (y1+y2)/2/H; bw=(x2-x1)/W; bh=(y2-y1)/H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            return "\n".join(lines) + "\n"
        (out / "RGB/labels" / f"{name}_visible.txt").write_text(yolo_lbl(boxes_rgb, Wr, Hr))
        (out / "IR/labels"  / f"{name}_infrared.txt").write_text(yolo_lbl(boxes_ir,  Wi, Hi))
        meta.append({"name": name, "bg_source": b["source"], "bg_stem": b["stem"],
                     "crop_stem": c["stem"], "placements": placements})
        if (i+1) % 50 == 0: print(f"[gen] {i+1}/{args.n}", flush=True)

    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {len(meta)} samples (rejects={rejects}) to {out}", flush=True)

if __name__ == "__main__":
    main()
