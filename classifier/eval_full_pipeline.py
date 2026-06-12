"""
eval_full_pipeline.py — End-to-end per-layer metrics across every dataset.

Stages (per dataset, only those that apply):
    1. RGB YOLO        frame-level TP/TN/FP/FN vs drone GT
    2. IR  YOLO        frame-level TP/TN/FP/FN vs drone GT
    3. Fusion classif. frame-level binary (paired data only — Anti-UAV-RGBT)
    4. Patch verifier  per-detection TP/TN/FP/FN (per modality)

No temporal logic. One pass per dataset, sequential.

Datasets handled:
    - Anti-UAV-RGBT test split    (paired)                → all 4 layers
    - Svanström RGB / IR tracks   (unpaired, separate)    → YOLO + patch
    - Training test splits:
        RGB dir = G:/drone/dataset/dataset
        IR  dir = G:/drone/IR_dset_final
      (unpaired)                                          → YOLO + patch
    - YouTube classifier videos   (frame-sampled first)   → RGB YOLO + patch
    - YouTube negatives           (frame-sampled first)   → RGB YOLO + patch

Outputs:
    classifier/runs/full_pipeline_eval/
        {dataset}/metrics.csv    layer,TP,TN,FP,FN,precision,recall,n
        summary.json             every dataset × every layer
        youtube_samples/         reusable frame dumps
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

CLASSIFIER_DIR = Path(__file__).resolve().parent
REPO = CLASSIFIER_DIR.parent

# --- hard-coded paths (per user spec) -------------------------------------

FUSION_SETTINGS = REPO / "ir_gui" / "fusion_settings.json"
CLASSIFIER_PATH = CLASSIFIER_DIR / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib"
PATCH_RGB_PATH  = CLASSIFIER_DIR / "runs" / "patches" / "confuser_filter4_rgb.pt"
PATCH_IR_PATH   = CLASSIFIER_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"

ANTIUAV_ROOT      = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test")
SVANSTROM_ROOT    = Path("G:/drone/svanstrom_paired")
TRAIN_RGB_ROOT    = Path("G:/drone/dataset/dataset")
TRAIN_IR_ROOT     = Path("G:/drone/IR_dset_final")
YT_CLASSIFIER_DIR = Path("D:/Downloads/youtube_classifier_videos")
YT_NEGATIVES_DIR  = Path("C:/Users/User/Desktop/UNISA projects/Drone detection/youtube negatives")

OUT_ROOT = CLASSIFIER_DIR / "runs" / "full_pipeline_eval"
YT_SAMPLES_ROOT = OUT_ROOT / "youtube_samples"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Classifier operational defaults
PATCH_THRESHOLD = 0.70
RGB_CONF = 0.25
IR_CONF = 0.40
IOU_MATCH = 0.5

# COCO-style size buckets (in px² of the box)
SIZE_BUCKETS = [("small", 0.0, 1024.0),
                ("medium", 1024.0, 9216.0),
                ("large", 9216.0, float("inf"))]


def size_bucket(area: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def safe_div(a, b):
    return float(a) / b if b > 0 else float("nan")


def roll_confusion(name, tp, tn, fp, fn):
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    n = tp + tn + fp + fn
    return {"layer": name, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": p, "recall": r, "n": n}


def write_metrics_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer", "TP", "TN", "FP", "FN",
                                            "precision", "recall", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


# --------------------------------------------------------------------------- #
# YOLO + patch loaders
# --------------------------------------------------------------------------- #

def load_settings():
    with FUSION_SETTINGS.open() as fh:
        return json.load(fh)


def load_yolo(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))


def run_yolo_image(model, img, conf=0.25, iou_nms=0.45, imgsz=640, device=0):
    res = model.predict(source=img, conf=conf, iou=iou_nms, imgsz=imgsz,
                        device=device, verbose=False, save=False, max_det=300)
    r = res[0]
    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i in range(len(confs)):
            dets.append((tuple(float(v) for v in xyxy[i]), float(confs[i])))
    dets.sort(key=lambda bc: bc[1], reverse=True)
    return dets


def load_patch(path, device="cuda"):
    sys.path.insert(0, str(CLASSIFIER_DIR))
    from patch_verifier import PatchVerifier
    return PatchVerifier(path, device=device)


# --------------------------------------------------------------------------- #
# YOLO label I/O
# --------------------------------------------------------------------------- #

def read_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """Return list of (x1,y1,x2,y2) in pixel coords. Empty if file missing/empty."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        _cls, cx, cy, w, h = parts[:5]
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((x1, y1, x2, y2))
    return boxes


# --------------------------------------------------------------------------- #
# Frame-level confusion (detection layer)
# --------------------------------------------------------------------------- #

def frame_confusion(has_drone_gt: bool, gt_boxes: list, dets: list):
    """Return ('TP'|'TN'|'FP'|'FN') — frame-level."""
    any_det = len(dets) > 0
    if has_drone_gt:
        matched = False
        for (db, _dc) in dets:
            for gb in gt_boxes:
                if iou(db, gb) >= IOU_MATCH:
                    matched = True
                    break
            if matched:
                break
        return "TP" if matched else "FN"
    else:
        return "FP" if any_det else "TN"


def object_counts_by_size(has_drone_gt: bool, gt_boxes: list, dets: list):
    """Object-level per-size counts: {bucket: {TP, FN, FP}}.
    TP bucketed by GT box size; FN bucketed by missed GT; FP bucketed by det size.
    (TN not meaningful at object level.)"""
    per = {b[0]: {"TP": 0, "FN": 0, "FP": 0} for b in SIZE_BUCKETS}
    # Recall view: iterate drone GTs
    if has_drone_gt:
        for gb in gt_boxes:
            b = size_bucket(box_area(gb))
            hit = any(iou(db, gb) >= IOU_MATCH for db, _ in dets)
            if hit:
                per[b]["TP"] += 1
            else:
                per[b]["FN"] += 1
    # Precision view: iterate dets for FPs
    for (db, _dc) in dets:
        matched = has_drone_gt and any(iou(db, gb) >= IOU_MATCH for gb in gt_boxes)
        if not matched:
            per[size_bucket(box_area(db))]["FP"] += 1
    return per


def add_obj(accum, per):
    for b, d in per.items():
        accum[b]["TP"] += d["TP"]
        accum[b]["FN"] += d["FN"]
        accum[b]["FP"] += d["FP"]


# --------------------------------------------------------------------------- #
# Per-detection confusion (patch verifier layer)
# --------------------------------------------------------------------------- #

def detection_patch_confusion(img_bgr, dets, gt_boxes, has_drone_gt, verifier):
    """Per-det TP/TN/FP/FN (overall + per size bucket).
    Verifier emits P(CONFUSER) — high prob = airplane/bird/heli. Matches
    gui/fusion/engine.py:286: reject when p >= threshold.
    Prediction here: pred=1 (drone) iff p < PATCH_THRESHOLD.
    Returns ((tp,tn,fp,fn), per_bucket_dict)."""
    per = {b[0]: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for b in SIZE_BUCKETS}
    if not dets:
        return (0, 0, 0, 0), per
    boxes = [d[0] for d in dets]
    probs = verifier.predict_boxes(img_bgr, boxes)
    tp = tn = fp = fn = 0
    for (b, _c), p in zip(dets, probs):
        pred = 1 if p < PATCH_THRESHOLD else 0   # accept as drone iff not confuser
        if has_drone_gt:
            label = 1 if any(iou(b, gb) >= IOU_MATCH for gb in gt_boxes) else 0
        else:
            label = 0
        bucket = size_bucket(box_area(b))
        if pred == 1 and label == 1:
            tp += 1; per[bucket]["TP"] += 1
        elif pred == 0 and label == 0:
            tn += 1; per[bucket]["TN"] += 1
        elif pred == 1 and label == 0:
            fp += 1; per[bucket]["FP"] += 1
        else:
            fn += 1; per[bucket]["FN"] += 1
    return (tp, tn, fp, fn), per


def add_patch(accum, per):
    for b, d in per.items():
        for k in ("TP", "TN", "FP", "FN"):
            accum[b][k] += d[k]


# --------------------------------------------------------------------------- #
# Classifier stage (paired only) — matches features used by classifier.joblib
# --------------------------------------------------------------------------- #

def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def parse_hour_from_stem(stem):
    m = re.match(r"\d{8}_(\d{2})", stem)
    if m:
        return int(m.group(1))
    return 12


def time_of_day_onehots(hour):
    if 6 <= hour < 18:
        cat = "day"
    elif hour in (5, 18, 19):
        cat = "dusk_dawn"
    elif hour >= 20 or hour < 5:
        cat = "night"
    else:
        cat = "unknown"
    return {
        "time_of_day_day":       1 if cat == "day" else 0,
        "time_of_day_dusk_dawn": 1 if cat == "dusk_dawn" else 0,
        "time_of_day_night":     1 if cat == "night" else 0,
        "time_of_day_unknown":   1 if cat == "unknown" else 0,
    }


def build_frame_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
    """Build the same 40-feature dict used by gui/fusion/engine.py."""
    # -- add fusion/features.py to path --
    fusion_pkg = REPO / "ir_gui" / "fusion"
    if str(fusion_pkg.parent) not in sys.path:
        sys.path.insert(0, str(fusion_pkg.parent))
    from fusion.features import TARGET_NAMES, compute_global_features, compute_target_features

    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}

    # Detection aggregates (matches engine._det_stats)
    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        n = len(confs)
        if n == 0:
            feats.update({f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.0,
                          f"{prefix}_mean_conf": 0.0, f"{prefix}_detected": 0})
        else:
            feats.update({f"{prefix}_n_dets": n,
                          f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
                          f"{prefix}_detected": 1})

    # Scene features
    rgb_global = compute_global_features(rgb_gray)
    ir_global = compute_global_features(ir_gray)
    feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
    feats.update({f"ir_{k}": v for k, v in ir_global.items()})

    # Best-detection target features
    for prefix, dets, gray, gw, gh in [
        ("rgb", rgb_dets, rgb_gray, rgb_w, rgb_h),
        ("ir", ir_dets, ir_gray, ir_w, ir_h),
    ]:
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})

    # Agreement flags
    rgb_detected = len(rgb_dets) > 0
    ir_detected = len(ir_dets) > 0
    feats["both_detect"] = int(rgb_detected and ir_detected)
    feats["neither_detect"] = int(not rgb_detected and not ir_detected)
    feats["rgb_only_detect"] = int(rgb_detected and not ir_detected)
    feats["ir_only_detect"] = int(not rgb_detected and ir_detected)

    return feats


# --------------------------------------------------------------------------- #
# YouTube sampling
# --------------------------------------------------------------------------- #

def sample_youtube(src_dir: Path, out_dir: Path, stride_seconds: float = 0.5):
    """Dump frames from every video in src_dir to out_dir/<stem>/f######.jpg.
    Idempotent — skip videos whose out dir already has ≥1 jpg.
    Writes out_dir/frames.csv with (path, video, label=0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "frames.csv"

    videos = sorted(p for p in src_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTS
                    and "_fusion" not in p.stem.lower())
    print(f"  [youtube-sample] {src_dir.name}: {len(videos)} videos")

    # Collect all frame rows (re-emit csv each run to keep it in sync).
    rows = []
    for v in videos:
        sub = out_dir / v.stem
        sub.mkdir(parents=True, exist_ok=True)
        existing = sorted(sub.glob("f*.jpg"))
        if existing:
            for f in existing:
                rows.append({"path": str(f), "video": v.name, "label": 0})
            continue
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"    [err] cannot open {v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * stride_seconds)))
        idx = 0
        kept = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                out_path = sub / f"f{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                rows.append({"path": str(out_path), "video": v.name, "label": 0})
                kept += 1
            idx += 1
        cap.release()
        print(f"    {v.name}: kept {kept} / {idx} frames (stride {step})")

    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "video", "label"])
        w.writeheader()
        w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- #
# Single-modality evaluation (YOLO + patch verifier) — used by every unpaired set
# --------------------------------------------------------------------------- #

def eval_single_modality(name, frames, yolo, verifier, modality_label, conf=0.25):
    """frames: iterable of dicts with keys {img_path, label_path_or_none, has_drone?}.
    Returns per-layer rollups, including size-bucketed object-level rows."""
    frames = list(frames)
    total = len(frames)
    yt = yp = yn = yfn = 0
    pt = pn = pf = pfn = 0
    obj_yolo = {b[0]: {"TP": 0, "FN": 0, "FP": 0} for b in SIZE_BUCKETS}
    obj_patch = {b[0]: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for b in SIZE_BUCKETS}
    n_frames = 0
    n_dets = 0
    t0 = time.time()
    for fi, f in enumerate(frames):
        img = cv2.imread(str(f["img_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = []
        if f.get("label_path"):
            gt = read_yolo_labels(f["label_path"], w, h)
        hd = f.get("has_drone")
        has_drone = (len(gt) > 0) if hd is None else bool(hd)
        dets = run_yolo_image(yolo, img, conf=conf)
        kind = frame_confusion(has_drone, gt, dets)
        if   kind == "TP": yt += 1
        elif kind == "TN": yn += 1
        elif kind == "FP": yp += 1
        elif kind == "FN": yfn += 1
        add_obj(obj_yolo, object_counts_by_size(has_drone, gt, dets))
        (ptp, ptn, pfp, pfnn), per = detection_patch_confusion(
            img, dets, gt, has_drone, verifier)
        pt += ptp; pn += ptn; pf += pfp; pfn += pfnn
        add_patch(obj_patch, per)
        n_dets += len(dets)
        n_frames += 1
        if (fi + 1) % 200 == 0:
            print(f"    [{name}] {fi + 1}/{total} frames  "
                  f"({(fi + 1) / (time.time() - t0):.1f} fps)")
    print(f"    [{name}] done: {n_frames}/{total} frames, {n_dets} dets, "
          f"{time.time() - t0:.1f}s")
    rows = [roll_confusion(f"{modality_label}_yolo", yt, yn, yp, yfn)]
    for bname, d in obj_yolo.items():
        rows.append(roll_confusion(f"{modality_label}_yolo_{bname}",
                                    d["TP"], 0, d["FP"], d["FN"]))
    rows.append(roll_confusion(f"{modality_label}_patch", pt, pn, pf, pfn))
    for bname, d in obj_patch.items():
        rows.append(roll_confusion(f"{modality_label}_patch_{bname}",
                                    d["TP"], d["TN"], d["FP"], d["FN"]))
    return rows


# --------------------------------------------------------------------------- #
# Paired evaluation (Anti-UAV-RGBT) — RGB + IR + classifier + patch (both)
# --------------------------------------------------------------------------- #

def iter_antiuav_pairs(sample=None):
    rgb_img_dir = ANTIUAV_ROOT / "RGB" / "images"
    ir_img_dir  = ANTIUAV_ROOT / "IR"  / "images"
    rgb_lab_dir = ANTIUAV_ROOT / "RGB" / "labels"
    ir_lab_dir  = ANTIUAV_ROOT / "IR"  / "labels"
    rgb_imgs = sorted(p for p in rgb_img_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS)
    if sample and sample > 0:
        rgb_imgs = stride_pick(rgb_imgs, sample)
    for rgb_path in rgb_imgs:
        stem = rgb_path.stem
        # Filename format: <seq>_visible_f<nnn>  →  IR: <seq>_infrared_f<nnn>
        if "_visible" in stem:
            ir_stem = stem.replace("_visible", "_infrared")
            base = stem.replace("_visible", "")
        else:
            ir_stem = stem
            base = stem
        ir_path = None
        for ext in IMG_EXTS:
            c = ir_img_dir / (ir_stem + ext)
            if c.exists():
                ir_path = c
                break
        if ir_path is None:
            continue
        yield {
            "base": base,
            "rgb_img": rgb_path,
            "ir_img":  ir_path,
            "rgb_lab": rgb_lab_dir / (stem + ".txt"),
            "ir_lab":  ir_lab_dir  / (ir_stem + ".txt"),
        }


def eval_antiuav(rgb_yolo, ir_yolo, patch_rgb, patch_ir, clf_bundle, sample=None):
    feat_cols = clf_bundle["features"]
    model     = clf_bundle["model"]
    pairs = list(iter_antiuav_pairs(sample=sample))
    total = len(pairs)
    print(f"    [antiuav] {total} paired frames to process")

    r_yt=r_yn=r_yp=r_yfn=0
    i_yt=i_yn=i_yp=i_yfn=0
    c_tp=c_tn=c_fp=c_fn=0
    rpt=rpn=rpf=rpfn=0
    ipt=ipn=ipf=ipfn=0
    obj_rgb  = {b[0]: {"TP":0,"FN":0,"FP":0} for b in SIZE_BUCKETS}
    obj_ir   = {b[0]: {"TP":0,"FN":0,"FP":0} for b in SIZE_BUCKETS}
    obj_rgbp = {b[0]: {"TP":0,"TN":0,"FP":0,"FN":0} for b in SIZE_BUCKETS}
    obj_irp  = {b[0]: {"TP":0,"TN":0,"FP":0,"FN":0} for b in SIZE_BUCKETS}
    n = 0
    t0 = time.time()

    for pair in pairs:
        rgb_img = cv2.imread(str(pair["rgb_img"]))
        ir_img  = cv2.imread(str(pair["ir_img"]))
        if rgb_img is None or ir_img is None:
            continue
        rh, rw = rgb_img.shape[:2]
        ih, iw = ir_img.shape[:2]
        rgb_gt = read_yolo_labels(pair["rgb_lab"], rw, rh)
        ir_gt  = read_yolo_labels(pair["ir_lab"],  iw, ih)
        rgb_has = len(rgb_gt) > 0
        ir_has  = len(ir_gt) > 0
        frame_has_drone = rgb_has or ir_has

        rgb_dets = run_yolo_image(rgb_yolo, rgb_img, conf=RGB_CONF)
        ir_dets  = run_yolo_image(ir_yolo,  ir_img, conf=IR_CONF)

        # per-modality YOLO frame confusion
        rk = frame_confusion(rgb_has, rgb_gt, rgb_dets)
        ik = frame_confusion(ir_has,  ir_gt,  ir_dets)
        for k, bump in [(rk, "rgb"), (ik, "ir")]:
            pass
        if rk == "TP": r_yt += 1
        elif rk == "TN": r_yn += 1
        elif rk == "FP": r_yp += 1
        elif rk == "FN": r_yfn += 1
        if ik == "TP": i_yt += 1
        elif ik == "TN": i_yn += 1
        elif ik == "FP": i_yp += 1
        elif ik == "FN": i_yfn += 1
        add_obj(obj_rgb, object_counts_by_size(rgb_has, rgb_gt, rgb_dets))
        add_obj(obj_ir,  object_counts_by_size(ir_has,  ir_gt,  ir_dets))

        # classifier features -> 4-class decision
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray  = cv2.cvtColor(ir_img,  cv2.COLOR_BGR2GRAY)
        feats = build_frame_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
        x = np.array([[feats.get(c, 0) for c in feat_cols]], dtype=np.float32)
        label = int(model.predict(x)[0])
        # label: 0=reject_both, 1=trust_rgb, 2=trust_ir, 3=trust_both
        pred = 1 if label != 0 else 0  # any trust = drone detected
        gt = 1 if frame_has_drone else 0
        if pred == 1 and gt == 1: c_tp += 1
        elif pred == 0 and gt == 0: c_tn += 1
        elif pred == 1 and gt == 0: c_fp += 1
        else: c_fn += 1

        # per-modality patch verifier
        (a, b, c, d), per_r = detection_patch_confusion(
            rgb_img, rgb_dets, rgb_gt, rgb_has, patch_rgb)
        rpt += a; rpn += b; rpf += c; rpfn += d
        add_patch(obj_rgbp, per_r)
        (a, b, c, d), per_i = detection_patch_confusion(
            ir_img, ir_dets, ir_gt, ir_has, patch_ir)
        ipt += a; ipn += b; ipf += c; ipfn += d
        add_patch(obj_irp, per_i)

        n += 1
        if n % 200 == 0:
            print(f"    [antiuav] {n}/{total} pairs  "
                  f"({n / (time.time() - t0):.1f} fps)")

    print(f"    [antiuav] done: {n}/{total} pairs, {time.time() - t0:.1f}s")
    rows = [
        roll_confusion("rgb_yolo",   r_yt, r_yn, r_yp, r_yfn),
    ]
    for bname, d in obj_rgb.items():
        rows.append(roll_confusion(f"rgb_yolo_{bname}",
                                    d["TP"], 0, d["FP"], d["FN"]))
    rows.append(roll_confusion("ir_yolo", i_yt, i_yn, i_yp, i_yfn))
    for bname, d in obj_ir.items():
        rows.append(roll_confusion(f"ir_yolo_{bname}",
                                    d["TP"], 0, d["FP"], d["FN"]))
    rows.append(roll_confusion("classifier", c_tp, c_tn, c_fp, c_fn))
    rows.append(roll_confusion("rgb_patch", rpt, rpn, rpf, rpfn))
    for bname, d in obj_rgbp.items():
        rows.append(roll_confusion(f"rgb_patch_{bname}",
                                    d["TP"], d["TN"], d["FP"], d["FN"]))
    rows.append(roll_confusion("ir_patch", ipt, ipn, ipf, ipfn))
    for bname, d in obj_irp.items():
        rows.append(roll_confusion(f"ir_patch_{bname}",
                                    d["TP"], d["TN"], d["FP"], d["FN"]))
    return rows


# --------------------------------------------------------------------------- #
# Dataset frame iterators (single-modality)
# --------------------------------------------------------------------------- #

def iter_yolo_split(images_dir: Path, labels_dir: Path, limit=None):
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS)
    if limit:
        imgs = imgs[:limit]
    for p in imgs:
        yield {
            "img_path": p,
            "label_path": labels_dir / (p.stem + ".txt"),
            "has_drone": None,  # resolved via GT file contents
        }


SVAN_CAT_RE = re.compile(r"^(?:IR|RGB)_([A-Z]+)_")


def svan_category(stem: str) -> str:
    m = SVAN_CAT_RE.match(stem)
    return m.group(1) if m else "UNKNOWN"


def stride_pick(items, n):
    if n <= 0 or n >= len(items):
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def iter_svanstrom_stratified(images_dir: Path, labels_dir: Path, per_class: int):
    """Stratified by {DRONE, AIRPLANE, BIRD, HELICOPTER}: stride-sample per_class each."""
    all_imgs = sorted(p for p in images_dir.iterdir()
                      if p.suffix.lower() in IMG_EXTS)
    buckets = defaultdict(list)
    for p in all_imgs:
        buckets[svan_category(p.stem)].append(p)
    print(f"    [svanstrom] available per class: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(buckets.items())))
    chosen = []
    for cat, items in sorted(buckets.items()):
        picks = stride_pick(items, per_class)
        print(f"      {cat}: sampled {len(picks)} / {len(items)}")
        chosen.extend(picks)
    chosen.sort()
    for p in chosen:
        yield {
            "img_path": p,
            "label_path": labels_dir / (p.stem + ".txt"),
            "has_drone": None,
        }


def iter_youtube_frames(rows):
    for r in rows:
        yield {
            "img_path": Path(r["path"]),
            "label_path": None,
            "has_drone": False,
        }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--antiuav-sample", type=int, default=3000,
                    help="Stride-sample N paired Anti-UAV frames (0 = all)")
    ap.add_argument("--svan-per-class", type=int, default=500,
                    help="Stride-sample N frames per Svanström category (0 = all)")
    ap.add_argument("--yt-stride-seconds", type=float, default=0.5,
                    help="Seconds between sampled YouTube frames")
    ap.add_argument("--datasets", nargs="*", default=[
        "antiuav_rgbt", "svanstrom_rgb", "svanstrom_ir",
        "train_test_rgb", "train_test_ir",
        "youtube_classifier", "youtube_negatives",
    ])
    ap.add_argument("--patch-threshold", type=float, default=None,
                    help="Override PATCH_THRESHOLD (default: use code constant 0.70)")
    args = ap.parse_args()

    global PATCH_THRESHOLD
    if args.patch_threshold is not None:
        PATCH_THRESHOLD = args.patch_threshold
        print(f"[override] PATCH_THRESHOLD = {PATCH_THRESHOLD}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    antiuav_sample = args.antiuav_sample or 0
    svan_per_class = args.svan_per_class or 0

    settings = load_settings()
    rgb_weights = settings["rgb_model"]
    ir_weights  = settings["ir_model"]
    print(f"RGB weights: {rgb_weights}")
    print(f"IR  weights: {ir_weights}")
    rgb_yolo = load_yolo(rgb_weights)
    ir_yolo  = load_yolo(ir_weights)
    patch_rgb = load_patch(PATCH_RGB_PATH)
    patch_ir  = load_patch(PATCH_IR_PATH)
    clf = joblib.load(CLASSIFIER_PATH)
    print(f"Classifier: {CLASSIFIER_PATH.name}  "
          f"features={len(clf['features'])}, classes={list(clf['model'].classes_)}")

    all_results = {}

    def run(name, fn):
        if name not in args.datasets:
            return
        print(f"\n=== {name} ===")
        rows = fn()
        out_csv = OUT_ROOT / name / "metrics.csv"
        write_metrics_csv(out_csv, rows)
        all_results[name] = rows
        print(f"    → {out_csv}")

    # -- paired -----------------------------------------------------------
    run("antiuav_rgbt",
        lambda: eval_antiuav(rgb_yolo, ir_yolo, patch_rgb, patch_ir, clf,
                             sample=antiuav_sample))

    # -- svanstrom (stratified sampling per category) ---------------------
    run("svanstrom_rgb", lambda: eval_single_modality(
        "svanstrom_rgb",
        iter_svanstrom_stratified(SVANSTROM_ROOT / "RGB" / "images",
                                   SVANSTROM_ROOT / "RGB" / "labels",
                                   per_class=svan_per_class),
        rgb_yolo, patch_rgb, "rgb", conf=RGB_CONF))
    run("svanstrom_ir", lambda: eval_single_modality(
        "svanstrom_ir",
        iter_svanstrom_stratified(SVANSTROM_ROOT / "IR" / "images",
                                   SVANSTROM_ROOT / "IR" / "labels",
                                   per_class=svan_per_class),
        ir_yolo, patch_ir, "ir", conf=IR_CONF))

    # -- training test splits (FULL) --------------------------------------
    run("train_test_rgb", lambda: eval_single_modality(
        "train_test_rgb",
        iter_yolo_split(TRAIN_RGB_ROOT / "images" / "test",
                        TRAIN_RGB_ROOT / "labels" / "test"),
        rgb_yolo, patch_rgb, "rgb", conf=RGB_CONF))
    run("train_test_ir", lambda: eval_single_modality(
        "train_test_ir",
        iter_yolo_split(TRAIN_IR_ROOT / "test" / "images",
                        TRAIN_IR_ROOT / "test" / "labels"),
        ir_yolo, patch_ir, "ir", conf=IR_CONF))

    # -- youtube ---------------------------------------------------------
    if "youtube_classifier" in args.datasets or "youtube_negatives" in args.datasets:
        print("\n--- sampling YouTube frames ---")
    if "youtube_classifier" in args.datasets:
        yt_c_rows = sample_youtube(YT_CLASSIFIER_DIR,
                                   YT_SAMPLES_ROOT / "classifier_videos",
                                   stride_seconds=args.yt_stride_seconds)
        run("youtube_classifier", lambda: eval_single_modality(
            "youtube_classifier",
            iter_youtube_frames(yt_c_rows),
            rgb_yolo, patch_rgb, "rgb", conf=RGB_CONF))
    if "youtube_negatives" in args.datasets:
        yt_n_rows = sample_youtube(YT_NEGATIVES_DIR,
                                   YT_SAMPLES_ROOT / "negatives",
                                   stride_seconds=args.yt_stride_seconds)
        run("youtube_negatives", lambda: eval_single_modality(
            "youtube_negatives",
            iter_youtube_frames(yt_n_rows),
            rgb_yolo, patch_rgb, "rgb", conf=RGB_CONF))

    # -- summary ----------------------------------------------------------
    summary_path = OUT_ROOT / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSummary → {summary_path}")

    print("\n== per-layer summary ==")
    for ds, rows in all_results.items():
        print(f"\n[{ds}]")
        for r in rows:
            p = "nan" if r["precision"] != r["precision"] else f"{r['precision']:.3f}"
            rec = "nan" if r["recall"] != r["recall"] else f"{r['recall']:.3f}"
            print(f"  {r['layer']:<14}  TP={r['TP']:>6}  TN={r['TN']:>6}  "
                  f"FP={r['FP']:>6}  FN={r['FN']:>6}  P={p}  R={rec}  n={r['n']}")


if __name__ == "__main__":
    main()
