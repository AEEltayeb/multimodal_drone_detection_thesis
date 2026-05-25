"""generate_lean19_data.py - 19-feature fusion dataset.

Superset of Lean-13: same 13 features + 6 geometry features that the
original 32-feat set had (pos_x, dist_to_center, target_bg_delta for both
RGB and IR). Used to test whether the dropped geometry features were
genuinely uninformative or just shadowed by scene-global noise.

Reuses Anti-UAV / Svanstrom YOLO detection caches if present at the
output directory; otherwise runs selcom YOLO. Drone-video-tests caches
read from docs/analysis/full_pipeline_ablations/cache/.
"""
import argparse, csv, json, random, re, time
from pathlib import Path
from collections import Counter

import cv2, numpy as np


FEATURE_COLS = [
    # Lean-13 base (13)
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
    # +6 geometry features
    "rgb_best_pos_x", "ir_best_pos_x",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
    "rgb_best_target_bg_delta", "ir_best_target_bg_delta",
]
TARGET_NAMES_19 = ["log_bbox_area", "aspect_ratio", "pos_y", "local_contrast",
                   "pos_x", "dist_to_center", "target_bg_delta"]


def compute_global_features_lean(img_gray):
    img_f = img_gray.astype(np.float32)
    return {
        "img_mean": round(float(img_f.mean()), 3),
        "img_std": round(float(img_f.std()), 3),
    }


def compute_target_features_19(img_gray, bbox_xyxy, img_w, img_h):
    x1, y1, x2, y2 = bbox_xyxy
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = pw * ph
    log_bbox_area = float(np.log(area + 1.0))
    aspect_ratio = float(pw / ph)
    pos_x = float(cx / img_w) if img_w > 0 else 0.5
    pos_y = float(cy / img_h) if img_h > 0 else 0.5
    dist_to_center = float(np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2))

    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        local_contrast = 0.0
        target_bg_delta = 0.0
    else:
        target = img_gray[yi1:yi2, xi1:xi2].astype(np.float32)
        target_mean = float(target.mean())
        mx, my = int(pw), int(ph)
        bx1, by1 = max(0, xi1 - mx), max(0, yi1 - my)
        bx2, by2 = min(img_w, xi2 + mx), min(img_h, yi2 + my)
        bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
        bg_mean = float(bg.mean())
        bg_std = float(bg.std())
        target_bg_delta = target_mean - bg_mean
        local_contrast = target_bg_delta / bg_std if bg_std >= 1.0 else 0.0
    return {
        "log_bbox_area": round(log_bbox_area, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "pos_y": round(pos_y, 4),
        "local_contrast": round(local_contrast, 4),
        "pos_x": round(pos_x, 4),
        "dist_to_center": round(dist_to_center, 4),
        "target_bg_delta": round(target_bg_delta, 3),
    }


# IoU/IoP/GT helpers (same logic as lean13 generator)

def has_gt(p):
    p = Path(p)
    return p.exists() and any(len(l.split()) >= 5 for l in p.read_text().strip().split("\n") if l)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    i = max(0., x2 - x1) * max(0., y2 - y1)
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    ab = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    u = aa + ab - i
    return i / u if u > 0 else 0.


def iop(d, g):
    x1 = max(d[0], g[0]); y1 = max(d[1], g[1])
    x2 = min(d[2], g[2]); y2 = min(d[3], g[3])
    i = max(0., x2 - x1) * max(0., y2 - y1)
    da = max(0., d[2]-d[0]) * max(0., d[3]-d[1])
    return i / da if da > 0 else 0.


def parse_yolo_gt(p, w, h):
    out = []
    p = Path(p)
    if not p.exists(): return out
    for l in p.read_text().strip().split("\n"):
        parts = l.strip().split()
        if len(parts) < 5: continue
        cx, cy, bw, bh = map(float, parts[1:5])
        out.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return out


def has_tp(dets, gts, thr=0.5, mode="iou"):
    if not dets or not gts: return False
    score = iou if mode == "iou" else iop
    for d in dets:
        for g in gts:
            if score(d[:4], g) >= thr: return True
    return False


def trust_label(rgb_dets, ir_dets, rgb_gt, ir_gt, rgb_mode="iou", ir_mode="iou"):
    r = has_tp(rgb_dets, rgb_gt, mode=rgb_mode)
    i = has_tp(ir_dets, ir_gt, mode=ir_mode)
    if r and i: return 3
    if r: return 1
    if i: return 2
    return 0


def build_row(rgb_dets, ir_dets, rgb_gray, ir_gray, rgb_wh, ir_wh,
              label, stem, source, conf_thresh=0.25):
    rgb_dets = [d for d in rgb_dets if d[4] >= conf_thresh]
    ir_dets = [d for d in ir_dets if d[4] >= conf_thresh]
    rgb_confs = [d[4] for d in rgb_dets]; ir_confs = [d[4] for d in ir_dets]
    row = {
        "rgb_max_conf": float(max(rgb_confs)) if rgb_confs else 0.0,
        "ir_max_conf": float(max(ir_confs)) if ir_confs else 0.0,
    }
    rg = compute_global_features_lean(rgb_gray)
    ig = compute_global_features_lean(ir_gray)
    row["rgb_img_mean"] = rg["img_mean"]
    row["rgb_img_std"] = rg["img_std"]
    row["ir_img_mean"] = ig["img_mean"]

    rw, rh = rgb_wh; iw, ih = ir_wh
    if rgb_dets:
        b = max(rgb_dets, key=lambda d: d[4])
        tf = compute_target_features_19(rgb_gray, b[:4], rw, rh)
        for k, v in tf.items(): row[f"rgb_best_{k}"] = v
    else:
        for k in TARGET_NAMES_19: row[f"rgb_best_{k}"] = 0.0
    if ir_dets:
        b = max(ir_dets, key=lambda d: d[4])
        tf = compute_target_features_19(ir_gray, b[:4], iw, ih)
        for k, v in tf.items(): row[f"ir_best_{k}"] = v
    else:
        for k in TARGET_NAMES_19: row[f"ir_best_{k}"] = 0.0

    row["trust_label"] = label
    row["stem"] = stem
    row["source"] = source
    return row


def run_yolo(model, img, conf, imgsz):
    r = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for i in range(len(xy)):
            out.append([float(xy[i][0]), float(xy[i][1]), float(xy[i][2]),
                        float(xy[i][3]), float(cf[i])])
    return out


class DetectionCache:
    def __init__(self, path):
        self.path = Path(path); self.data = {}; self.dirty = False
        if self.path.exists():
            self.data = json.load(open(self.path))
            print(f"  cache loaded: {self.path.name} ({len(self.data)} entries)")
    def has(self, k): return k in self.data
    def get(self, k):
        e = self.data[k]; return e["rgb_dets"], e["ir_dets"]
    def put(self, k, r, i):
        self.data[k] = {"rgb_dets": r, "ir_dets": i}; self.dirty = True
    def save(self):
        if self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(self.data, open(self.path, "w"))
            print(f"  cache saved: {self.path.name}")


def discover_paired(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rgb_d = Path(root) / "RGB" / "images"
    ir_d = Path(root) / "IR" / "images"
    rgb_l = Path(root) / "RGB" / "labels"; ir_l = Path(root) / "IR" / "labels"
    strip = lambda s: re.sub(r"_(visible|infrared)", "", s, flags=re.IGNORECASE)
    rmap = {strip(f.stem): f for f in sorted(rgb_d.iterdir()) if f.suffix.lower() in exts}
    imap = {strip(f.stem): f for f in sorted(ir_d.iterdir()) if f.suffix.lower() in exts}
    shared = sorted(set(rmap) & set(imap))
    pairs = []
    for b in shared:
        ri, ii = rmap[b], imap[b]
        rl = rgb_l/(ri.stem + ".txt"); il = ir_l/(ii.stem + ".txt")
        pairs.append({"base_stem": b, "rgb_img": ri, "ir_img": ii,
                       "rgb_lbl": rl, "ir_lbl": il,
                       "is_positive": has_gt(rl) or has_gt(il)})
    print(f"  paired: {len(rmap)} RGB, {len(imap)} IR, {len(shared)} shared")
    return pairs


def process_paired(rgb_m, ir_m, root, stride, neg_keep, conf, rgb_sz, ir_sz,
                   cache, src, rgb_mode="iou"):
    pairs = discover_paired(root)[::stride]
    pos = [p for p in pairs if p["is_positive"]]
    neg = [p for p in pairs if not p["is_positive"]]
    if neg_keep is not None and neg_keep < 1.0 and neg:
        random.seed(42); neg = random.sample(neg, int(len(neg)*neg_keep))
    frames = pos + neg; random.shuffle(frames)
    print(f"  {src}: {len(pos)} pos + {len(neg)} neg = {len(frames)}")
    rows = []; t0 = time.time()
    for idx, p in enumerate(frames):
        stem = p["base_stem"]
        rimg = cv2.imread(str(p["rgb_img"])); iimg = cv2.imread(str(p["ir_img"]))
        if rimg is None or iimg is None: continue
        rh, rw = rimg.shape[:2]; ih, iw = iimg.shape[:2]
        if cache and cache.has(stem):
            rd, id_ = cache.get(stem)
        elif rgb_m is not None:
            rd = run_yolo(rgb_m, rimg, conf, rgb_sz)
            id_ = run_yolo(ir_m, iimg, conf, ir_sz)
            if cache: cache.put(stem, rd, id_)
        else:
            continue
        rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        igray = cv2.cvtColor(iimg, cv2.COLOR_BGR2GRAY) if len(iimg.shape) == 3 else iimg
        rgt = parse_yolo_gt(p["rgb_lbl"], rw, rh)
        igt = parse_yolo_gt(p["ir_lbl"], iw, ih)
        lab = trust_label(rd, id_, rgt, igt, rgb_mode, "iou")
        rows.append(build_row(rd, id_, rgray, igray, (rw, rh), (iw, ih),
                              lab, stem, src, conf))
        if (idx+1) % 500 == 0:
            print(f"    [{idx+1}/{len(frames)}] {(idx+1)/(time.time()-t0):.1f} fps")
    if cache: cache.save()
    print(f"  {src} done: {len(rows)} rows")
    return rows


def list_imgs(d):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []


def process_video_tests(repo, cache_dir, conf, tag="selcom_1280_sz1280"):
    root = repo / "datasets" / "drone detection video tests" / "rgb"
    rows = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cd = root / cat
        if not cd.exists(): continue
        for clip in sorted(cd.iterdir()):
            if not clip.is_dir(): continue
            img_d = clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
            lbl_d = clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
            ctag = f"video_{cat}_{clip.name}"
            rc = cache_dir / f"{ctag}_{tag}.json"
            ic = cache_dir / f"{ctag}_ir_grayscale_sz640.json"
            if not (img_d.exists() and rc.exists() and ic.exists()): continue
            rd_c = json.load(open(rc))["dets"]; id_c = json.load(open(ic))["dets"]
            n0 = len(rows)
            for ip in list_imgs(img_d):
                stem = ip.stem
                img = cv2.imread(str(ip))
                if img is None: continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gt = parse_yolo_gt(lbl_d / f"{stem}.txt", w, h)
                lab = trust_label(rd_c.get(stem, []), id_c.get(stem, []),
                                  gt, gt, "iop", "iop")
                rows.append(build_row(rd_c.get(stem, []), id_c.get(stem, []),
                                      gray, gray, (w, h), (w, h), lab,
                                      f"{ctag}_{stem}", ctag, conf))
            print(f"  {ctag}: +{len(rows)-n0}")
    return rows


def process_yt(rgb_m, ir_m, demo_dir, stride, conf, rgb_sz, ir_sz):
    cfg = [
        {"path": demo_dir/"yt_Z8HJNypu_1Y.mp4", "label": "AIRPLANE"},
        {"path": demo_dir/"yt_1U7Bu2pSUwU.mp4", "label": "HELICOPTER"},
        {"path": demo_dir/"yt_ZO5lV0gh5i4.mp4", "label": "BIRD"},
    ]
    rows = []
    for c in cfg:
        v = c["path"]
        if not v.exists(): print(f"  [skip yt] {v.name}"); continue
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened(): continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  yt {v.name} [{c['label']}]: {total} frames stride={stride}")
        idx, n0 = 0, len(rows)
        while True:
            ret, fr = cap.read()
            if not ret: break
            idx += 1
            if stride > 1 and (idx % stride != 0): continue
            rd = run_yolo(rgb_m, fr, conf, rgb_sz)
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            id_ = run_yolo(ir_m, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), conf, ir_sz)
            rows.append(build_row(rd, id_, gray, gray, (w, h), (w, h),
                                  0, f"{v.stem}_f{idx:06d}",
                                  f"confuser_{c['label']}", conf))
        cap.release()
        print(f"    +{len(rows)-n0} rows")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-weights", default=None)
    ap.add_argument("--ir-weights", default=None)
    ap.add_argument("--auv-root", default="G:/drone/Anti-UAV-RGBT_yolo_converted/test")
    ap.add_argument("--svan-root", default="G:/drone/svanstrom_paired")
    ap.add_argument("--auv-stride", type=int, default=2)
    ap.add_argument("--svan-stride", type=int, default=2)
    ap.add_argument("--neg-keep", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--auv-imgsz", type=int, default=640)
    ap.add_argument("--svan-imgsz", type=int, default=1280)
    ap.add_argument("--ir-imgsz", type=int, default=640)
    ap.add_argument("--video-rgb-cache-tag", default="selcom_1280_sz1280")
    ap.add_argument("--skip-auv", action="store_true")
    ap.add_argument("--skip-svan", action="store_true")
    ap.add_argument("--skip-video", action="store_true")
    ap.add_argument("--include-yt", action="store_true")
    ap.add_argument("--yt-stride", type=int, default=3)
    ap.add_argument("--output-dir", default="classifier/fusion_models/lean19")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    out_dir = (repo / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = repo / "docs" / "analysis" / "full_pipeline_ablations" / "cache"

    print("Lean-19 dataset generator")

    rgb_m = ir_m = None
    need = (not args.skip_auv) or (not args.skip_svan) or args.include_yt
    if need:
        if args.rgb_weights and args.ir_weights:
            from ultralytics import YOLO
            print(f"  loading RGB: {args.rgb_weights}")
            rgb_m = YOLO(args.rgb_weights)
            print(f"  loading IR : {args.ir_weights}")
            ir_m = YOLO(args.ir_weights)
        else:
            print("  [warn] no weights given; will rely on caches only")

    rows = []
    if not args.skip_auv and Path(args.auv_root).exists():
        print(f"\n-- Anti-UAV (RGB sz={args.auv_imgsz}, IR sz={args.ir_imgsz}) --")
        c = DetectionCache(out_dir / "cache_antiuav.json")
        rows += process_paired(rgb_m, ir_m, args.auv_root, args.auv_stride,
                                args.neg_keep, args.conf, args.auv_imgsz, args.ir_imgsz,
                                c, "antiuav", "iou")
    if not args.skip_svan and Path(args.svan_root).exists():
        print(f"\n-- Svanstrom (RGB sz={args.svan_imgsz}, IR sz={args.ir_imgsz}) --")
        c = DetectionCache(out_dir / "cache_svanstrom.json")
        rows += process_paired(rgb_m, ir_m, args.svan_root, args.svan_stride,
                                None, args.conf, args.svan_imgsz, args.ir_imgsz,
                                c, "svanstrom", "iop")
    if not args.skip_video:
        print(f"\n-- Drone-video-tests (cached) --")
        rows += process_video_tests(repo, cache_dir, args.conf, args.video_rgb_cache_tag)
    if args.include_yt and rgb_m is not None:
        print(f"\n-- yt confusers --")
        rows += process_yt(rgb_m, ir_m, repo/"ir_gui"/"demo_outputs",
                           args.yt_stride, args.conf, args.auv_imgsz, args.ir_imgsz)

    if not rows:
        raise SystemExit("No rows produced.")
    td = Counter(r["trust_label"] for r in rows)
    sd = Counter(r["source"] for r in rows)
    TN = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
    print(f"\n=== {len(rows):,} rows ===")
    for t in sorted(td): print(f"  {TN[t]}: {td[t]}")
    csv_path = out_dir / "fusion_dataset_lean19.csv"
    fields = FEATURE_COLS + ["trust_label", "stem", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  saved -> {csv_path}")
    json.dump({"feature_cols": FEATURE_COLS, "total_rows": len(rows),
               "source_counts": dict(sd),
               "trust_distribution": {TN[k]: v for k, v in sorted(td.items())}},
              open(out_dir/"config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
