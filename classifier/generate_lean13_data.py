"""
generate_lean13_data.py - 13-feature fusion dataset for the Lean-13 classifier.

Same trust-label logic + same paired sources as generate_retrained_v2_data.py
(Anti-UAV + Svanstrom), but:
  * 13 features only: 2 detection-conf max + 8 best-target geometry/contrast
    + 3 cheap scene scalars (img_mean RGB & IR, img_std RGB).
  * Replaces the 3 unlabeled yt_*.mp4 confuser clips with the labeled
    drone-video-tests confusers (birds, airplanes, helicopters) which have
    real YOLO GT.
  * Adds drone-video-tests `drone/` clips as grayscale-positive rows
    (RGB->grayscale->IR pathway, mirrors step1_build_real_grayscale.py).

Usage:
    python classifier/generate_lean13_data.py \
        --rgb-weights "RGB model/weights/retrained_v2.pt" \
        --ir-weights "IR_dataset_runs/.../best.pt"
"""

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np


# ---- Lean-13 feature computation -------------------------------------------

def compute_global_features_lean(img_gray):
    """Only the scalars we actually use: img_mean (per modality) and img_std (RGB)."""
    img_f = img_gray.astype(np.float32)
    return {
        "img_mean": round(float(img_f.mean()), 3),
        "img_std": round(float(img_f.std()), 3),
    }


def compute_target_features_lean(img_gray, bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = pw * ph

    log_bbox_area = float(np.log(area + 1.0))
    aspect_ratio = float(pw / ph)
    pos_y = float(cy / img_h) if img_h > 0 else 0.5

    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        local_contrast = 0.0
    else:
        target = img_gray[yi1:yi2, xi1:xi2].astype(np.float32)
        target_mean = float(target.mean())
        mx, my = int(pw), int(ph)
        bx1, by1 = max(0, xi1 - mx), max(0, yi1 - my)
        bx2, by2 = min(img_w, xi2 + mx), min(img_h, yi2 + my)
        bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        bg_mean = float(bg.mean())
        bg_std = float(bg.std())
        local_contrast = (target_mean - bg_mean) / bg_std if bg_std >= 1.0 else 0.0

    return {
        "log_bbox_area": round(log_bbox_area, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "pos_y": round(pos_y, 4),
        "local_contrast": round(local_contrast, 4),
    }


FEATURE_COLS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean",
    "rgb_img_std",
]

TARGET_NAMES = ["log_bbox_area", "aspect_ratio", "pos_y", "local_contrast"]


# ---- IoU/IoP + GT helpers (copied from generate_retrained_v2_data.py) ------

def has_gt(label_path):
    p = Path(label_path)
    if not p.exists():
        return False
    text = p.read_text().strip()
    return any(len(line.split()) >= 5 for line in text.split("\n") if line)


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def compute_iop(d, g):
    x1 = max(d[0], g[0]); y1 = max(d[1], g[1])
    x2 = min(d[2], g[2]); y2 = min(d[3], g[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    da = max(0.0, d[2] - d[0]) * max(0.0, d[3] - d[1])
    return inter / da if da > 0 else 0.0


def parse_yolo_gt(label_path, img_w, img_h):
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append(((cx - w/2)*img_w, (cy - h/2)*img_h,
                       (cx + w/2)*img_w, (cy + h/2)*img_h))
    return boxes


def has_tp(dets, gt_boxes, thresh=0.5, mode="iou"):
    if not dets or not gt_boxes:
        return False
    score_fn = compute_iou if mode == "iou" else compute_iop
    for d in dets:
        for g in gt_boxes:
            if score_fn(d[:4], g) >= thresh:
                return True
    return False


def trust_label(rgb_dets, ir_dets, rgb_gt, ir_gt, rgb_mode="iou", ir_mode="iou"):
    rgb_has = has_tp(rgb_dets, rgb_gt, mode=rgb_mode)
    ir_has = has_tp(ir_dets, ir_gt, mode=ir_mode)
    if rgb_has and ir_has: return 3
    if rgb_has: return 1
    if ir_has: return 2
    return 0


# ---- Row builder ----------------------------------------------------------

def build_row(rgb_dets, ir_dets, rgb_gray, ir_gray, rgb_wh, ir_wh,
              label, stem, source, conf_thresh=0.25):
    rgb_dets = [d for d in rgb_dets if d[4] >= conf_thresh]
    ir_dets = [d for d in ir_dets if d[4] >= conf_thresh]

    rgb_confs = [d[4] for d in rgb_dets]
    ir_confs = [d[4] for d in ir_dets]

    row = {
        "rgb_max_conf": float(max(rgb_confs)) if rgb_confs else 0.0,
        "ir_max_conf": float(max(ir_confs)) if ir_confs else 0.0,
    }

    rg = compute_global_features_lean(rgb_gray)
    ig = compute_global_features_lean(ir_gray)
    row["rgb_img_mean"] = rg["img_mean"]
    row["rgb_img_std"] = rg["img_std"]
    row["ir_img_mean"] = ig["img_mean"]

    rgb_w, rgb_h = rgb_wh
    ir_w, ir_h = ir_wh
    if rgb_dets:
        best = max(rgb_dets, key=lambda d: d[4])
        tf = compute_target_features_lean(rgb_gray, best[:4], rgb_w, rgb_h)
        for k, v in tf.items():
            row[f"rgb_best_{k}"] = v
    else:
        for k in TARGET_NAMES:
            row[f"rgb_best_{k}"] = 0.0
    if ir_dets:
        best = max(ir_dets, key=lambda d: d[4])
        tf = compute_target_features_lean(ir_gray, best[:4], ir_w, ir_h)
        for k, v in tf.items():
            row[f"ir_best_{k}"] = v
    else:
        for k in TARGET_NAMES:
            row[f"ir_best_{k}"] = 0.0

    row["trust_label"] = label
    row["stem"] = stem
    row["source"] = source
    return row


# ---- YOLO inference + cache ------------------------------------------------

def run_yolo(model, img, conf=0.25, imgsz=640):
    results = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)
    r = results[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            dets.append([float(xyxy[i][0]), float(xyxy[i][1]),
                         float(xyxy[i][2]), float(xyxy[i][3]),
                         float(confs[i])])
    return dets


class DetectionCache:
    """Cache used for Anti-UAV / Svanstrom inference."""
    def __init__(self, cache_path):
        self.path = Path(cache_path)
        self.data = {}
        self.dirty = False
        if self.path.exists():
            with open(self.path, "r") as f:
                self.data = json.load(f)
            print(f"  Cache loaded: {self.path.name} ({len(self.data)} entries)")
    def has(self, stem): return stem in self.data
    def get(self, stem):
        e = self.data[stem]; return e["rgb_dets"], e["ir_dets"]
    def put(self, stem, r, i):
        self.data[stem] = {"rgb_dets": r, "ir_dets": i}; self.dirty = True
    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f: json.dump(self.data, f)
            print(f"  Cache saved: {self.path.name}")


# ---- Anti-UAV ---------------------------------------------------------------

def discover_paired(dataset_root):
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rgb_img_dir = Path(dataset_root) / "RGB" / "images"
    ir_img_dir = Path(dataset_root) / "IR" / "images"
    rgb_lbl_dir = Path(dataset_root) / "RGB" / "labels"
    ir_lbl_dir = Path(dataset_root) / "IR" / "labels"

    def strip_suffix(stem):
        return re.sub(r"_(visible|infrared)", "", stem, flags=re.IGNORECASE)

    rgb_map = {strip_suffix(f.stem): f for f in sorted(rgb_img_dir.iterdir())
               if f.suffix.lower() in img_exts}
    ir_map = {strip_suffix(f.stem): f for f in sorted(ir_img_dir.iterdir())
              if f.suffix.lower() in img_exts}
    shared = sorted(set(rgb_map) & set(ir_map))
    pairs = []
    for base in shared:
        ri, ii = rgb_map[base], ir_map[base]
        pairs.append({
            "base_stem": base, "rgb_img": ri, "ir_img": ii,
            "rgb_lbl": rgb_lbl_dir / (ri.stem + ".txt"),
            "ir_lbl": ir_lbl_dir / (ii.stem + ".txt"),
            "is_positive": has_gt(rgb_lbl_dir / (ri.stem + ".txt"))
                          or has_gt(ir_lbl_dir / (ii.stem + ".txt")),
        })
    print(f"  paired: {len(rgb_map)} RGB, {len(ir_map)} IR, {len(shared)} shared")
    return pairs


def process_paired(rgb_model, ir_model, dataset_root, stride, neg_keep,
                   conf_thresh, imgsz, ir_imgsz, cache, source_name,
                   rgb_match_mode="iou"):
    pairs = discover_paired(dataset_root)
    pairs = pairs[::stride]
    positives = [p for p in pairs if p["is_positive"]]
    negatives = [p for p in pairs if not p["is_positive"]]
    if neg_keep is not None and neg_keep < 1.0 and negatives:
        n = int(len(negatives) * neg_keep)
        random.seed(42)
        negatives = random.sample(negatives, min(n, len(negatives)))
    frames = positives + negatives
    random.shuffle(frames)
    print(f"  {source_name}: {len(positives)} pos + {len(negatives)} neg = {len(frames)}")

    rows = []
    t0 = time.time()
    for idx, pair in enumerate(frames):
        stem = pair["base_stem"]
        rgb_img = cv2.imread(str(pair["rgb_img"]))
        ir_img = cv2.imread(str(pair["ir_img"]))
        if rgb_img is None or ir_img is None: continue
        rh, rw = rgb_img.shape[:2]
        ih, iw = ir_img.shape[:2]

        if cache and cache.has(stem):
            rgb_dets, ir_dets = cache.get(stem)
        else:
            rgb_dets = run_yolo(rgb_model, rgb_img, conf=conf_thresh, imgsz=imgsz)
            ir_dets = run_yolo(ir_model, ir_img, conf=conf_thresh, imgsz=ir_imgsz)
            if cache: cache.put(stem, rgb_dets, ir_dets)

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape) == 3 else ir_img

        rgb_gt = parse_yolo_gt(pair["rgb_lbl"], rw, rh)
        ir_gt = parse_yolo_gt(pair["ir_lbl"], iw, ih)
        label = trust_label(rgb_dets, ir_dets, rgb_gt, ir_gt,
                            rgb_mode=rgb_match_mode, ir_mode="iou")
        rows.append(build_row(rgb_dets, ir_dets, rgb_gray, ir_gray,
                              (rw, rh), (iw, ih), label, stem, source_name,
                              conf_thresh=conf_thresh))

        if (idx + 1) % 500 == 0:
            fps = (idx + 1) / max(1e-3, time.time() - t0)
            print(f"    [{idx+1}/{len(frames)}] {fps:.1f} fps  rows={len(rows)}")

    if cache: cache.save()
    print(f"  {source_name} done: {len(rows)} rows in {(time.time()-t0)/60:.1f}min")
    return rows


# ---- Drone-video-tests (cached, RGB+grayscale-as-IR) ------------------------

def list_imgs(d):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []


def process_video_tests(repo_root, cache_dir, conf_thresh, rgb_cache_tag="selcom_1280_sz1280"):
    """Iterate drone-video-tests clips using cached detections.

    Confuser clips (birds/airplanes/helicopters) -> source=video_<cat>.
    Drone clips -> source=video_drone (grayscale positives).
    """
    vid_root = repo_root / "datasets" / "drone detection video tests" / "rgb"
    rows = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cat_dir = vid_root / cat
        if not cat_dir.exists():
            print(f"  [skip] {cat_dir} missing"); continue
        for clip in sorted(cat_dir.iterdir()):
            if not clip.is_dir(): continue
            img_dir = clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
            lbl_dir = clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
            tag = f"video_{cat}_{clip.name}"
            rgb_cache = cache_dir / f"{tag}_{rgb_cache_tag}.json"
            ir_cache = cache_dir / f"{tag}_ir_grayscale_sz640.json"
            if not (img_dir.exists() and rgb_cache.exists() and ir_cache.exists()):
                print(f"  [skip] {tag}: cache or images missing"); continue

            rgb_c = json.load(open(rgb_cache))["dets"]
            ir_c = json.load(open(ir_cache))["dets"]
            imgs = list_imgs(img_dir)
            n0 = len(rows)
            for img_path in imgs:
                stem = img_path.stem
                rgb_dets = rgb_c.get(stem, [])
                ir_dets = ir_c.get(stem, [])
                img = cv2.imread(str(img_path))
                if img is None: continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                gt = parse_yolo_gt(lbl_dir / f"{stem}.txt", w, h)
                # Grayscale-mode: both branches see the same grayscale image,
                # both compared against the same GT with IoP-0.5 (drone-video
                # GT can be tight, IoP is more forgiving and matches step1).
                label = trust_label(rgb_dets, ir_dets, gt, gt,
                                    rgb_mode="iop", ir_mode="iop")
                rows.append(build_row(rgb_dets, ir_dets, gray, gray,
                                      (w, h), (w, h), label,
                                      f"{tag}_{stem}", tag,
                                      conf_thresh=conf_thresh))
            print(f"  {tag}: +{len(rows)-n0} rows")
    return rows


# ---- Legacy yt_*.mp4 confusers (optional via --include-yt) ----------------

def process_yt_confusers(rgb_model, ir_model, demo_dir, stride,
                         conf_thresh, imgsz, ir_imgsz):
    """Replay the legacy yt_*.mp4 confuser block from generate_retrained_v2_data.
    All frames blanket-labeled class-0 (reject_both). Skips missing files."""
    configs = [
        {"path": demo_dir / "yt_Z8HJNypu_1Y.mp4", "label": "AIRPLANE"},
        {"path": demo_dir / "yt_1U7Bu2pSUwU.mp4", "label": "HELICOPTER"},
        {"path": demo_dir / "yt_ZO5lV0gh5i4.mp4", "label": "BIRD"},
    ]
    rows = []
    for cfg in configs:
        vpath = cfg["path"]
        if not vpath.exists():
            print(f"  [skip yt] {vpath.name} missing"); continue
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"  [skip yt] cannot open {vpath.name}"); continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  yt {vpath.name} [{cfg['label']}]: {total} frames, stride={stride}")
        idx, processed, n0 = 0, 0, len(rows)
        while True:
            ret, frame = cap.read()
            if not ret: break
            idx += 1
            if stride > 1 and (idx % stride != 0): continue
            processed += 1
            rgb_dets = run_yolo(rgb_model, frame, conf=conf_thresh, imgsz=imgsz)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            ir_dets = run_yolo(ir_model, gray_3ch, conf=conf_thresh, imgsz=ir_imgsz)
            stem = f"{vpath.stem}_f{idx:06d}"
            rows.append(build_row(rgb_dets, ir_dets, gray, gray,
                                  (w, h), (w, h), 0, stem,
                                  f"confuser_{cfg['label']}",
                                  conf_thresh=conf_thresh))
        cap.release()
        print(f"    +{len(rows)-n0} rows")
    return rows


# ---- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-weights", required=False, default=None,
                    help="RGB weights (only needed if Anti-UAV/Svanstrom caches are not populated)")
    ap.add_argument("--ir-weights", required=False, default=None)
    ap.add_argument("--auv-root", default="G:/drone/Anti-UAV-RGBT_yolo_converted/test")
    ap.add_argument("--svan-root", default="G:/drone/svanstrom_paired")
    ap.add_argument("--auv-stride", type=int, default=2)
    ap.add_argument("--svan-stride", type=int, default=2)
    ap.add_argument("--neg-keep", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--auv-imgsz", type=int, default=640,
                    help="RGB imgsz for Anti-UAV inference")
    ap.add_argument("--svan-imgsz", type=int, default=1280,
                    help="RGB imgsz for Svanstrom (1280 per project memory)")
    ap.add_argument("--ir-imgsz", type=int, default=640,
                    help="IR imgsz everywhere")
    ap.add_argument("--video-rgb-cache-tag", default="selcom_1280_sz1280",
                    help="Cache filename tag for drone-video-tests RGB detections")
    ap.add_argument("--skip-auv", action="store_true")
    ap.add_argument("--skip-svan", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--include-yt", action="store_true",
                    help="Also process legacy yt_*.mp4 confuser clips at ir_gui/demo_outputs/")
    ap.add_argument("--yt-stride", type=int, default=3)
    ap.add_argument("--output-dir", default="classifier/fusion_models/lean13")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo_root / "docs" / "analysis" / "full_pipeline_ablations" / "cache"

    print("=" * 70)
    print("Lean-13 fusion dataset generator")
    print("=" * 70)

    rgb_model = ir_model = None
    need_models = (not args.skip_auv) or (not args.skip_svan) or args.include_yt
    if need_models:
        if not args.rgb_weights or not args.ir_weights:
            print("  [warn] --rgb-weights / --ir-weights not given; will rely on caches only.")
        else:
            from ultralytics import YOLO
            print(f"  Loading RGB: {args.rgb_weights}")
            rgb_model = YOLO(args.rgb_weights)
            print(f"  Loading IR : {args.ir_weights}")
            ir_model = YOLO(args.ir_weights)

    auv_rows = []
    if not args.skip_auv and Path(args.auv_root).exists():
        print("\n--- Anti-UAV (RGB imgsz=%d, IR imgsz=%d) ---" % (args.auv_imgsz, args.ir_imgsz))
        cache = DetectionCache(out_dir / "cache_antiuav.json")
        auv_rows = process_paired(rgb_model, ir_model, args.auv_root,
                                   args.auv_stride, args.neg_keep,
                                   args.conf, args.auv_imgsz, args.ir_imgsz,
                                   cache, "antiuav", rgb_match_mode="iou")

    svan_rows = []
    if not args.skip_svan and Path(args.svan_root).exists():
        print("\n--- Svanstrom (RGB imgsz=%d, IR imgsz=%d) ---" % (args.svan_imgsz, args.ir_imgsz))
        cache = DetectionCache(out_dir / "cache_svanstrom.json")
        svan_rows = process_paired(rgb_model, ir_model, args.svan_root,
                                    args.svan_stride, None,
                                    args.conf, args.svan_imgsz, args.ir_imgsz,
                                    cache, "svanstrom", rgb_match_mode="iop")

    vid_rows = []
    if not args.skip_video:
        print("\n--- Drone-video-tests (cached, tag=%s) ---" % args.video_rgb_cache_tag)
        vid_rows = process_video_tests(repo_root, cache_dir, args.conf,
                                       rgb_cache_tag=args.video_rgb_cache_tag)

    yt_rows = []
    if args.include_yt:
        if rgb_model is None or ir_model is None:
            print("\n[skip yt] --include-yt requires --rgb-weights and --ir-weights")
        else:
            print("\n--- Legacy yt_*.mp4 confusers (stride=%d, RGB imgsz=%d) ---"
                  % (args.yt_stride, args.auv_imgsz))
            demo_dir = repo_root / "ir_gui" / "demo_outputs"
            yt_rows = process_yt_confusers(rgb_model, ir_model, demo_dir,
                                            args.yt_stride, args.conf,
                                            args.auv_imgsz, args.ir_imgsz)

    all_rows = auv_rows + svan_rows + vid_rows + yt_rows
    if not all_rows:
        print("No rows produced. Check --skip flags and dataset paths."); return

    from collections import Counter
    trust_dist = Counter(r["trust_label"] for r in all_rows)
    src_dist = Counter(r["source"] for r in all_rows)
    print(f"\n{'=' * 70}\nDATASET SUMMARY\n{'=' * 70}")
    print(f"  Total rows: {len(all_rows):,}")
    TN = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
    for t in sorted(trust_dist):
        print(f"  {TN[t]:15s}: {trust_dist[t]:,}")
    print("  Sources:")
    for s, n in sorted(src_dist.items()):
        print(f"    {s:50s}: {n:,}")

    csv_path = out_dir / "fusion_dataset_lean13.csv"
    fieldnames = FEATURE_COLS + ["trust_label", "stem", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Saved: {csv_path}")

    cfg = {
        "rgb_weights": args.rgb_weights, "ir_weights": args.ir_weights,
        "auv_stride": args.auv_stride, "svan_stride": args.svan_stride,
        "neg_keep": args.neg_keep, "conf": args.conf,
        "trust_distribution": {TN[k]: v for k, v in sorted(trust_dist.items())},
        "source_counts": dict(src_dist),
        "total_rows": len(all_rows),
        "feature_cols": FEATURE_COLS,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
