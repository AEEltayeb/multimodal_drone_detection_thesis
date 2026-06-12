"""generate_routing_data.py — Phase 1b: verifier-augmented fusion dataset.

Re-mines the SAME paired/video frames as generate_lean19_data, but runs the detectors
WITH the P3/P5 DetectInputHook so it can score each detection through the verifier MLP
("hooked to YOLO's brain") and add three columns to the lean-19 features:

  rgb_verifier_pdrone : max P(drone) from mlp_v5 over RGB detections   (0 if none)
  ir_verifier_pdrone  : max P(drone) from the IR verifier over IR dets (0 if none)
                        thermal IR (antiuav/svanstrom) -> aligned_thr ; grayscale (video) -> aligned_gray
  conf_sum            : sum of ALL detection confidences (both modalities)  [the true sum, not max+max]

Trust label identical logic to generate_lean19_data (has_tp per modality). Output feeds
classifier/train_routing_robust.py (which auto-detects the new columns).

  py -u classifier/generate_routing_data.py            # full re-mine (GPU)
  py -u classifier/generate_routing_data.py --limit 40 # smoke test (few frames/source)

GPU. ~15-30 min depending on machine. Resumable per-source via the box caches it shares
with generate_lean19_data (backbone feats are NOT cached -> the detector still runs).
"""
from __future__ import annotations
import argparse, csv, random, time
from pathlib import Path
from collections import Counter

import cv2, numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from generate_lean19_data import (discover_paired, parse_yolo_gt, trust_label, build_row,  # noqa: E402
                                   list_imgs, FEATURE_COLS)
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features, INPUT_DIM    # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                                                 # noqa: E402

FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
MLP_V5 = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
ALIGNED_THR = REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"
ALIGNED_GRAY = REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt"
NEW_COLS = ["conf_sum", "rgb_verifier_pdrone", "ir_verifier_pdrone"]


def hooked_detect(yolo, hook, img, conf, imgsz):
    """Return (dets[[x1,y1,x2,y2,conf]], feats[N,517]) with backbone features."""
    hook.clear()
    res = yolo.predict(img, imgsz=imgsz, conf=conf, verbose=False, device="cuda")
    b = res[0].boxes
    if b is None or len(b) == 0:
        return [], np.zeros((0, INPUT_DIM), np.float32)
    ih, iw = img.shape[:2]
    boxes = [tuple(b.xyxy[i].cpu().numpy().tolist()) for i in range(len(b))]
    confs = [float(b.conf[i]) for i in range(len(b))]
    feats = np.stack([_extract_detection_features(hook, bx, (ih, iw), c) for bx, c in zip(boxes, confs)])
    dets = [list(bx) + [c] for bx, c in zip(boxes, confs)]
    return dets, feats.astype(np.float32)


def pdrone_max(verif, feats):
    return float(verif.predict_drone_probs(feats).max()) if len(feats) else 0.0


def augment(row, rgb_dets, ir_dets, rgb_feats, ir_feats, rgb_v, ir_v, conf_thresh=0.25):
    """Add conf_sum + verifier columns to a lean-19 row (apply same conf filter as build_row)."""
    rf = [(d, f) for d, f in zip(rgb_dets, rgb_feats) if d[4] >= conf_thresh]
    iff = [(d, f) for d, f in zip(ir_dets, ir_feats) if d[4] >= conf_thresh]
    rgb_kept_feats = np.stack([f for _, f in rf]) if rf else np.zeros((0, INPUT_DIM), np.float32)
    ir_kept_feats = np.stack([f for _, f in iff]) if iff else np.zeros((0, INPUT_DIM), np.float32)
    row["conf_sum"] = round(sum(d[4] for d, _ in rf) + sum(d[4] for d, _ in iff), 4)
    row["rgb_verifier_pdrone"] = round(pdrone_max(rgb_v, rgb_kept_feats), 4)
    row["ir_verifier_pdrone"] = round(pdrone_max(ir_v, ir_kept_feats), 4)
    return row


def process_paired(rgb_m, rhook, ir_m, ihook, rgb_v, ir_v, root, stride, neg_keep,
                   conf, rgb_sz, ir_sz, src, rgb_mode, limit):
    pairs = discover_paired(root)[::stride]
    pos = [p for p in pairs if p["is_positive"]]; neg = [p for p in pairs if not p["is_positive"]]
    if neg_keep is not None and neg_keep < 1.0 and neg:
        random.seed(42); neg = random.sample(neg, int(len(neg) * neg_keep))
    frames = pos + neg; random.shuffle(frames)
    if limit:
        frames = frames[:limit]
    print(f"  {src}: {len(frames)} frames")
    rows, t0 = [], time.time()
    for idx, p in enumerate(frames):
        rimg = cv2.imread(str(p["rgb_img"])); iimg = cv2.imread(str(p["ir_img"]))
        if rimg is None or iimg is None:
            continue
        rh, rw = rimg.shape[:2]; ih, iw = iimg.shape[:2]
        rdets, rfeats = hooked_detect(rgb_m, rhook, rimg, conf, rgb_sz)
        idets, ifeats = hooked_detect(ir_m, ihook, iimg, conf, ir_sz)
        rgray = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
        igray = cv2.cvtColor(iimg, cv2.COLOR_BGR2GRAY) if iimg.ndim == 3 else iimg
        rgt = parse_yolo_gt(p["rgb_lbl"], rw, rh); igt = parse_yolo_gt(p["ir_lbl"], iw, ih)
        lab = trust_label(rdets, idets, rgt, igt, rgb_mode, "iou")
        row = build_row(rdets, idets, rgray, igray, (rw, rh), (iw, ih), lab, p["base_stem"], src, conf)
        rows.append(augment(row, rdets, idets, rfeats, ifeats, rgb_v, ir_v, conf))
        if (idx + 1) % 200 == 0:
            print(f"    [{idx+1}/{len(frames)}] {(idx+1)/(time.time()-t0):.1f} fps")
    print(f"  {src} done: {len(rows)} rows")
    return rows


def process_video(rgb_m, rhook, ir_m, ihook, rgb_v, ir_v_gray, conf, rgb_sz, ir_sz, limit):
    """Video clips: RGB image -> ft4; IR = grayscale(RGB) -> v3b; gray IR verifier."""
    root = REPO / "datasets" / "drone detection video tests" / "rgb"
    rows = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cd = root / cat
        if not cd.exists():
            continue
        for clip in sorted(cd.iterdir()):
            if not clip.is_dir():
                continue
            img_d = clip / "images" / "test" if (clip / "images" / "test").exists() else clip / "images"
            lbl_d = clip / "labels" / "test" if (clip / "labels" / "test").exists() else clip / "labels"
            ctag = f"video_{cat}_{clip.name}"; n0 = len(rows)
            imgs = list_imgs(img_d)
            if limit:
                imgs = imgs[:max(2, limit // 4)]
            for ip in imgs:
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                rdets, rfeats = hooked_detect(rgb_m, rhook, img, conf, rgb_sz)
                idets, ifeats = hooked_detect(ir_m, ihook, gray_bgr, conf, ir_sz)
                gt = parse_yolo_gt(lbl_d / f"{ip.stem}.txt", w, h)
                lab = trust_label(rdets, idets, gt, gt, "iop", "iop")
                row = build_row(rdets, idets, gray, gray, (w, h), (w, h), lab, f"{ctag}_{ip.stem}", ctag, conf)
                rows.append(augment(row, rdets, idets, rfeats, ifeats, rgb_v, ir_v_gray, conf))
            if len(rows) > n0:
                print(f"  {ctag}: +{len(rows)-n0}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auv-root", default="G:/drone/Anti-UAV-RGBT_yolo_converted/test")
    ap.add_argument("--svan-root", default="G:/drone/svanstrom_paired")
    ap.add_argument("--auv-stride", type=int, default=25)
    ap.add_argument("--svan-stride", type=int, default=10)
    ap.add_argument("--neg-keep", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="cap frames/source for a smoke test")
    ap.add_argument("--output-dir", default="models/routers/routing_robust")
    args = ap.parse_args()

    out_dir = REPO / args.output_dir; out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    print(f"loading detectors + verifiers (INPUT_DIM={INPUT_DIM})")
    rgb_m = YOLO(FT4); rhook = DetectInputHook(); rhook.register(rgb_m)
    ir_m = YOLO(V3B); ihook = DetectInputHook(); ihook.register(ir_m)
    rgb_v = MLPv4Verifier(MLP_V5, device="cpu")
    ir_v_thr = MLPv4Verifier(ALIGNED_THR, device="cpu")
    ir_v_gray = MLPv4Verifier(ALIGNED_GRAY, device="cpu")

    rows = []
    if Path(args.auv_root).exists():
        print("\n-- Anti-UAV (thermal IR -> aligned_thr) --")
        rows += process_paired(rgb_m, rhook, ir_m, ihook, rgb_v, ir_v_thr, args.auv_root,
                               args.auv_stride, args.neg_keep, args.conf, 640, 640, "antiuav", "iou", args.limit)
    if Path(args.svan_root).exists():
        print("\n-- Svanstrom (thermal IR -> aligned_thr) --")
        rows += process_paired(rgb_m, rhook, ir_m, ihook, rgb_v, ir_v_thr, args.svan_root,
                               args.svan_stride, None, args.conf, 1280, 640, "svanstrom", "iop", args.limit)
    print("\n-- Drone/confuser videos (grayscale IR -> aligned_gray) --")
    rows += process_video(rgb_m, rhook, ir_m, ihook, rgb_v, ir_v_gray, args.conf, 1280, 640, args.limit)

    if not rows:
        raise SystemExit("No rows produced.")
    td = Counter(r["trust_label"] for r in rows)
    TN = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
    print(f"\n=== {len(rows):,} rows ===")
    for t in sorted(td):
        print(f"  {TN[t]}: {td[t]}")
    csv_path = out_dir / "fusion_dataset_routing.csv"
    fields = FEATURE_COLS + NEW_COLS + ["trust_label", "stem", "source"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"  saved -> {csv_path}\n  then: py classifier/train_routing_robust.py --csv {csv_path}")


if __name__ == "__main__":
    main()
