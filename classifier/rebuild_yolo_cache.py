"""
rebuild_yolo_cache.py — Re-run RGB+IR YOLO inference on Anti-UAV test
and Svanström paired, with the CURRENT models in fusion_settings.json.

Why: eval_six_configs.py reuses cached detections from
    classifier/runs/raw_detections.json        (Anti-UAV test, 85,374 pairs)
    classifier/runs/svanstrom_detections.json  (Svanström paired,  28,710 pairs)
The cache is keyed by base stem with rgb_dets, ir_dets, sizes, label paths.
When RGB or IR weights change, the cache is invalid and must be rebuilt.

This script:
  1. Backs up the existing JSONs to *.old.json (one-time per dataset).
  2. Re-runs RGB + IR YOLO at conf=0 (so eval_six_configs can re-threshold
     downstream — same convention as the existing cache).
  3. Writes a fresh JSON in the same format.
  4. Resume-safe: re-running picks up where it left off via per-key checks.

Usage:
    python classifier/rebuild_yolo_cache.py                # both datasets
    python classifier/rebuild_yolo_cache.py --dataset antiuav
    python classifier/rebuild_yolo_cache.py --dataset svanstrom
    python classifier/rebuild_yolo_cache.py --resume       # don't backup, append
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent

ANTIUAV_RGB_IMG = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_IR_IMG  = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images")
ANTIUAV_RGB_LBL = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels")
ANTIUAV_IR_LBL  = Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels")

SVAN_RGB_IMG    = Path(r"G:/drone/svanstrom_paired/RGB/images")
SVAN_IR_IMG     = Path(r"G:/drone/svanstrom_paired/IR/images")
SVAN_RGB_LBL    = Path(r"G:/drone/svanstrom_paired/RGB/labels")
SVAN_IR_LBL     = Path(r"G:/drone/svanstrom_paired/IR/labels")

ANTIUAV_OUT_BASE = SCRIPT_DIR / "runs" / "raw_detections"
SVAN_OUT_BASE    = SCRIPT_DIR / "runs" / "svanstrom_detections"

FUSION_SETTINGS = REPO / "ir_gui" / "fusion_settings.json"

CKPT_EVERY = 200


def load_settings():
    return json.loads(FUSION_SETTINGS.read_text())


def run_yolo_all(model, frame, imgsz=640):
    """Run at conf=0 so eval_six_configs can apply its own threshold later."""
    res = model.predict(frame, conf=0.001, iou=0.45, imgsz=imgsz,
                        verbose=False, device=0, max_det=300)[0]
    out = []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i in range(len(confs)):
            out.append([float(xyxy[i, 0]), float(xyxy[i, 1]),
                        float(xyxy[i, 2]), float(xyxy[i, 3]),
                        float(confs[i])])
    return out


def derive_pair_key(rgb_stem: str, suffix_visible="_visible") -> str:
    if suffix_visible in rgb_stem:
        return rgb_stem.replace(suffix_visible, "")
    return rgb_stem


def iter_antiuav_pairs():
    rgb_imgs = sorted(p for p in ANTIUAV_RGB_IMG.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for rgb_path in rgb_imgs:
        stem = rgb_path.stem
        if "_visible" in stem:
            ir_stem = stem.replace("_visible", "_infrared")
            base = stem.replace("_visible", "")
        else:
            ir_stem = stem
            base = stem
        ir_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = ANTIUAV_IR_IMG / (ir_stem + ext)
            if c.exists():
                ir_path = c
                break
        if ir_path is None:
            continue
        yield {
            "key": base,
            "rgb_img": rgb_path, "ir_img": ir_path,
            "rgb_lbl": str(ANTIUAV_RGB_LBL / (stem + ".txt")),
            "ir_lbl":  str(ANTIUAV_IR_LBL  / (ir_stem + ".txt")),
        }


def iter_svanstrom_pairs():
    rgb_imgs = sorted(p for p in SVAN_RGB_IMG.iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for rgb_path in rgb_imgs:
        stem = rgb_path.stem
        if "_visible" in stem:
            ir_stem = stem.replace("_visible", "_infrared")
            base = stem.replace("_visible", "")
        else:
            ir_stem = stem
            base = stem
        ir_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            c = SVAN_IR_IMG / (ir_stem + ext)
            if c.exists():
                ir_path = c
                break
        if ir_path is None:
            continue
        yield {
            "key": base,
            "rgb_img": rgb_path, "ir_img": ir_path,
            "rgb_lbl": str(SVAN_RGB_LBL / (stem + ".txt")),
            "ir_lbl":  str(SVAN_IR_LBL  / (ir_stem + ".txt")),
        }


def build_cache(name, pairs_iter, out_path: Path,
                rgb_model, ir_model, resume=False):
    pairs = list(pairs_iter)
    total = len(pairs)
    print(f"[{name}] {total:,} pairs total")

    cache: dict = {}
    if resume and out_path.exists():
        try:
            cache = json.loads(out_path.read_text())
            print(f"[{name}] resumed: {len(cache):,} keys already in cache")
        except Exception:
            print(f"[{name}] failed to load existing cache; starting fresh")
            cache = {}

    if not resume and out_path.exists():
        backup = out_path.with_suffix(".old.json")
        if not backup.exists():
            shutil.copy2(out_path, backup)
            print(f"[{name}] backed up existing cache -> {backup.name}")
        out_path.unlink()
        cache = {}

    t0 = time.time()
    n_session = 0

    for idx, pair in enumerate(pairs):
        if pair["key"] in cache:
            continue
        rgb_img = cv2.imread(str(pair["rgb_img"]))
        ir_img  = cv2.imread(str(pair["ir_img"]))
        if rgb_img is None or ir_img is None:
            continue

        rgb_dets = run_yolo_all(rgb_model, rgb_img)
        ir_dets  = run_yolo_all(ir_model,  ir_img)

        cache[pair["key"]] = {
            "rgb_dets": rgb_dets,
            "ir_dets":  ir_dets,
            "rgb_w": int(rgb_img.shape[1]),
            "rgb_h": int(rgb_img.shape[0]),
            "ir_w":  int(ir_img.shape[1]),
            "ir_h":  int(ir_img.shape[0]),
            "rgb_lbl": pair["rgb_lbl"],
            "ir_lbl":  pair["ir_lbl"],
        }
        n_session += 1

        if n_session % CKPT_EVERY == 0:
            out_path.write_text(json.dumps(cache))
            elapsed = time.time() - t0
            fps = n_session / elapsed
            done_total = len(cache)
            remaining = (total - done_total) / max(fps, 1e-6)
            print(f"[{name}] {done_total:>6,}/{total:,}  {fps:.1f} fps  "
                  f"ETA {remaining/60:.1f} min")

    out_path.write_text(json.dumps(cache))
    print(f"[{name}] done: {len(cache):,} pairs cached at {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["antiuav", "svanstrom", "both"],
                    default="both")
    ap.add_argument("--resume", action="store_true",
                    help="don't back up; append to existing cache")
    ap.add_argument("--tag", type=str, default="",
                    help="tag suffix for output, e.g. 'v3more' -> raw_detections_v3more.json")
    ap.add_argument("--rgb-weights", type=str, default=None,
                    help="override RGB weights (defaults to fusion_settings.json)")
    ap.add_argument("--ir-weights", type=str, default=None,
                    help="override IR weights (defaults to fusion_settings.json)")
    args = ap.parse_args()

    sfx = f"_{args.tag}" if args.tag else ""
    antiuav_out = ANTIUAV_OUT_BASE.with_name(ANTIUAV_OUT_BASE.name + sfx + ".json")
    svan_out    = SVAN_OUT_BASE.with_name(SVAN_OUT_BASE.name + sfx + ".json")

    settings = load_settings()
    rgb_w = args.rgb_weights or settings["rgb_model"]
    ir_w  = args.ir_weights  or settings["ir_model"]
    print(f"RGB model: {rgb_w}")
    print(f"IR  model: {ir_w}")

    rgb_model = YOLO(rgb_w)
    ir_model  = YOLO(ir_w)

    if args.dataset in ("antiuav", "both"):
        build_cache("antiuav", iter_antiuav_pairs(), antiuav_out,
                    rgb_model, ir_model, resume=args.resume)
    if args.dataset in ("svanstrom", "both"):
        build_cache("svanstrom", iter_svanstrom_pairs(), svan_out,
                    rgb_model, ir_model, resume=args.resume)


if __name__ == "__main__":
    main()
