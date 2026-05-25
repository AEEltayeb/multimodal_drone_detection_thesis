"""Confuser-only cut-paste paired RGB+IR eval set.

Confuser crops come from DETECTOR FALSE-POSITIVES on Svanstrom BIRD + HELICOPTER
frames: we run the RGB model on RGB frames and the IR model on IR frames, save
crops at the detection bboxes. This is intentionally selection-biased - these
ARE the confusers the cascade is supposed to reject. The labels are empty
(drone-FP test). RGB and IR confuser banks are independent (Svanstrom RGB/IR
confusers come from different source videos anyway).

Usage:
  python eval/generate_cutpaste_confusers_v4.py --n 500 --out G:/drone/cutpaste_confusers_v4 --seed 2
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path
import cv2
import numpy as np

# Reuse helpers from v4
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cutpaste_v4 import (
    build_bg_bank, soft_elliptical_alpha, paste, ensure_3ch, load_label,
    DRONES_PER_IMAGE, SIZE_TIERS, TIER_WEIGHTS, SCENE_CONTEXTS, MIN_PASTE_PX,
    REPO, RGB_WEIGHTS, IR_WEIGHTS,
)

SV = Path(r"G:/drone/svanstrom_paired")
SV_RGB_IMG = SV / "RGB/images"
SV_IR_IMG  = SV / "IR/images"

CONFUSER_CATS = ("BIRD", "HELICOPTER")

def scan_confuser_fp_crops(side_dir: Path, model, imgsz, cats, rng, target,
                            conf_thr=0.4, max_scan=4000):
    """Run detector on confuser frames; save crops at detection bboxes (false positives)."""
    files = [f for f in os.listdir(side_dir) if any(c in f for c in cats)]
    rng.shuffle(files); files = files[:max_scan]
    kept = []
    for i, f in enumerate(files):
        if len(kept) >= target: break
        if i % 200 == 0:
            print(f"[fp-scan] {side_dir.name}: scanned {i}/{len(files)} -> {len(kept)} kept", flush=True)
        img = cv2.imread(str(side_dir / f))
        if img is None: continue
        img = ensure_3ch(img)
        try:
            res = model.predict(img, imgsz=imgsz, conf=conf_thr, verbose=False)[0]
        except Exception: continue
        if len(res.boxes) == 0: continue
        boxes = res.boxes.xyxy.cpu().numpy(); confs = res.boxes.conf.cpu().numpy()
        # Best (highest conf) FP detection
        j = int(confs.argmax())
        x1, y1, x2, y2 = (int(v) for v in boxes[j])
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8: continue
        crop = img[max(0,y1):y2, max(0,x1):x2].copy()
        if crop.size == 0: continue
        alpha = soft_elliptical_alpha(*crop.shape[:2])
        cat = next((c for c in cats if c in f), "UNKNOWN")
        kept.append({"crop": crop, "alpha": alpha, "cat": cat, "stem": Path(f).stem,
                     "src_w": crop.shape[1], "conf": float(confs[j])})
    print(f"[fp-scan] {side_dir.name}: final {len(kept)} crops "
          + ", ".join(f"{c}={sum(1 for x in kept if x['cat']==c)}" for c in cats), flush=True)
    return kept

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--target-bgs", type=int, default=200)
    ap.add_argument("--target-crops", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed); np.random.seed(args.seed)
    out = args.out
    for sub in ["RGB/images", "RGB/labels", "IR/images", "IR/labels"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    print(f"[confuser] loading detectors...", flush=True)
    rgb_model = YOLO(str(RGB_WEIGHTS)); ir_model = YOLO(str(IR_WEIGHTS))

    print("[confuser] scanning RGB FP detections...", flush=True)
    rgb_bank = scan_confuser_fp_crops(SV_RGB_IMG, rgb_model, 1280, CONFUSER_CATS, rng, args.target_crops)
    print("[confuser] scanning IR FP detections...", flush=True)
    ir_bank  = scan_confuser_fp_crops(SV_IR_IMG,  ir_model,  640,  CONFUSER_CATS, rng, args.target_crops)
    if not rgb_bank or not ir_bank: print("ERROR: empty confuser bank"); sys.exit(1)
    bgs = build_bg_bank(args.target_bgs, rng)
    if not bgs: print("ERROR: empty bg bank"); sys.exit(1)

    meta = []
    for i in range(args.n):
        b = rng.choice(bgs)
        rgb_out = b["rgb"].copy(); ir_out = b["ir"].copy()
        Hr,Wr = rgb_out.shape[:2]; Hi,Wi = ir_out.shape[:2]
        placements = []
        for k in range(DRONES_PER_IMAGE):
            # Pick category, then a crop in that category for EACH modality independently
            cat = rng.choice(CONFUSER_CATS)
            rgb_cands = [x for x in rgb_bank if x["cat"] == cat] or rgb_bank
            ir_cands  = [x for x in ir_bank  if x["cat"] == cat] or ir_bank
            cr = rng.choice(rgb_cands); ci = rng.choice(ir_cands)

            tier = rng.choices(list(TIER_WEIGHTS), weights=list(TIER_WEIGHTS.values()))[0]
            ctx  = SCENE_CONTEXTS[k % len(SCENE_CONTEXTS)]
            lo, hi = SIZE_TIERS[tier]
            target_w_rgb = rng.randint(lo, hi)
            target_w_rgb = min(target_w_rgb, int(1.5 * cr["src_w"]))
            target_w_rgb = max(target_w_rgb, MIN_PASTE_PX)
            target_w_ir = max(MIN_PASTE_PX, int(target_w_rgb * (Wi / Wr)))

            if   ctx == "sky":     cy_n = rng.uniform(0.05, 0.35)
            elif ctx == "clutter": cy_n = rng.uniform(0.30, 0.65)
            else:                  cy_n = rng.uniform(0.55, 0.90)
            cx_n = rng.uniform(0.10, 0.90)

            cx_r,cy_r = int(cx_n*Wr), int(cy_n*Hr)
            cx_i,cy_i = int(cx_n*Wi), int(cy_n*Hi)
            paste(rgb_out, cr["crop"], cr["alpha"], cx_r, cy_r, target_w_rgb)
            paste(ir_out,  ci["crop"], ci["alpha"], cx_i, cy_i, target_w_ir)
            placements.append({"cat": cat, "tier": tier, "ctx": ctx,
                               "rgb_stem": cr["stem"], "ir_stem": ci["stem"]})

        name = f"cf_{i:05d}"
        cv2.imwrite(str(out / "RGB/images" / f"{name}_visible.jpg"),  rgb_out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / "IR/images"  / f"{name}_infrared.jpg"), ir_out,  [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "RGB/labels" / f"{name}_visible.txt").write_text("")   # empty - confusers are negatives for DRONE
        (out / "IR/labels"  / f"{name}_infrared.txt").write_text("")
        meta.append({"name": name, "bg_source": b["source"], "bg_stem": b["stem"], "placements": placements})
        if (i+1) % 50 == 0: print(f"[gen] {i+1}/{args.n}", flush=True)

    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {len(meta)} confuser samples to {out}", flush=True)

if __name__ == "__main__":
    main()
