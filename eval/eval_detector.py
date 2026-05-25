"""
eval_detector.py — Reusable single-detector evaluation with per-size breakdown.

Evaluates one or more YOLO detectors on any dataset defined in config.yaml
(or ad-hoc paths). Outputs aggregate + per-size CSV with TP, FP, FN, P, R, F1,
FP%, TN% (frame-level).

Usage:
  # Anti-UAV, 1000 frames, selcom_1280@960 + ir_v3b@640:
  python eval/eval_detector.py --dataset antiuav --max-frames 1000 \
      --models selcom_1280_960imgsz ir_v3b

  # Svanström, 500 frames:
  python eval/eval_detector.py --dataset svanstrom --max-frames 500 \
      --models selcom_1280_960imgsz ir_v3b

  # Custom paths:
  python eval/eval_detector.py --dataset custom \
      --img-dir G:/drone/my_dataset/images --lbl-dir G:/drone/my_dataset/labels \
      --max-frames 1000 --models selcom_1280_960imgsz

  # All registered models on antiuav:
  python eval/eval_detector.py --dataset antiuav --max-frames 1000 --all-models
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import (  # noqa: E402
    SIZE_BUCKETS, classify_size, score_per_size, compute_prf,
    score_detections,
)
from datasets import read_yolo_labels  # noqa: E402

sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

# ── Classifier paths ─────────────────────────────────────────────────
CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
IR_WEIGHTS_PATH = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
PATCH_RGB_PATH = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
PATCH_IR_PATH  = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"


# ── Model Registry ───────────────────────────────────────────────────
# Each entry: (weights_path, imgsz, modality, conf_threshold)
# modality: "rgb" | "ir"

MODEL_REGISTRY = {
    # RGB detectors
    "baseline":       (REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt",
                       640, "rgb", 0.25),
    "retrained_v2":   (REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt",
                       640, "rgb", 0.25),
    "selcom_640":     (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
                       640, "rgb", 0.25),
    "selcom_1280_960imgsz":     (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
                       960, "rgb", 0.25),
    "selcom_1280":    (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
                       1280, "rgb", 0.25),
    "baseline_1280":  (REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt",
                       1280, "rgb", 0.25),
    "ft3_1280":       (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt",
                       1280, "rgb", 0.25),
    # IR detector
    "ir_v3b":         (REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt",
                       640, "ir", 0.40),
}


# ── Dataset Registry ─────────────────────────────────────────────────
# type: "paired" (has separate RGB/IR dirs) or "image" (single dir)

DATASET_REGISTRY = {
    "antiuav": {
        "type": "paired",
        "rgb_img": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "rgb_lbl": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_img":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "ir_lbl":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "scoring": "iou",
    },
    "svanstrom": {
        "type": "paired",
        "rgb_img": Path("G:/drone/svanstrom_paired/RGB/images"),
        "rgb_lbl": Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_img":  Path("G:/drone/svanstrom_paired/IR/images"),
        "ir_lbl":  Path("G:/drone/svanstrom_paired/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "scoring": "iop",
    },
    "rgb_test": {
        "type": "image",
        "img_dir": Path("G:/drone/dataset/dataset/images/test"),
        "lbl_dir": Path("G:/drone/dataset/dataset/labels/test"),
        "scoring": "iop",
    },
    "ir_test": {
        "type": "image",
        "img_dir": Path("G:/drone/IR_dset_final/test/images"),
        "lbl_dir": Path("G:/drone/IR_dset_final/test/labels"),
        "scoring": "iou",
    },
    "selcom_val": {
        "type": "image",
        "img_dir": Path("G:/drone/_finetune_selcom_mixed_ft2/images/val"),
        "lbl_dir": Path("G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
        "scoring": "iou",
    },
}


# ── Frame enumeration ────────────────────────────────────────────────

def list_paired_frames(ds: dict) -> list[dict]:
    """List all frame pairs from a paired dataset."""
    img_dir = ds["rgb_img"]
    lbl_dir = ds["rgb_lbl"]
    ir_img_dir = ds["ir_img"]
    ir_lbl_dir = ds["ir_lbl"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem
        rgb_lbl = lbl_dir / f"{stem}.txt"
        # Map to IR stem
        ir_stem = stem
        if ds.get("rgb_suffix") and ds.get("ir_suffix"):
            ir_stem = stem.replace(ds["rgb_suffix"], ds["ir_suffix"])
        ir_path = None
        for ext in exts:
            cand = ir_img_dir / f"{ir_stem}{ext}"
            if cand.exists():
                ir_path = cand
                break
        ir_lbl = ir_lbl_dir / f"{ir_stem}.txt"
        frames.append({
            "stem": stem,
            "rgb_path": p, "rgb_lbl": rgb_lbl,
            "ir_path": ir_path, "ir_lbl": ir_lbl,
        })
    return frames


def list_image_frames(ds: dict) -> list[dict]:
    """List all frames from a single-modality dataset."""
    img_dir = ds["img_dir"]
    lbl_dir = ds["lbl_dir"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        frames.append({
            "stem": p.stem,
            "rgb_path": p, "rgb_lbl": lbl_dir / f"{p.stem}.txt",
            "ir_path": None, "ir_lbl": None,
        })
    return frames


def list_frames(ds: dict) -> list[dict]:
    if ds["type"] == "paired":
        return list_paired_frames(ds)
    else:
        return list_image_frames(ds)


def _video_prefix(stem: str) -> str:
    """Extract video prefix from a frame stem.
    Handles: 'video_prefix_f000123' -> 'video_prefix'
             'video_prefix_000123' -> 'video_prefix'
             'IR_AIRPLANE_001_f000000_visible' -> 'IR_AIRPLANE_001'
    Falls back to the full stem if no numeric suffix found."""
    import re
    # Strip known modality suffixes
    clean = re.sub(r'_(visible|infrared)$', '', stem)
    m = re.match(r'^(.+?)_f?(\d{4,})$', clean)
    return m.group(1) if m else stem


def group_frames_by_video(frames: list[dict]) -> dict[str, list[dict]]:
    """Group frames by video prefix, preserving order."""
    from collections import OrderedDict
    videos: dict[str, list[dict]] = OrderedDict()
    for fr in frames:
        vp = _video_prefix(fr["stem"])
        videos.setdefault(vp, []).append(fr)
    return videos


def sample_temporal_frames(
    all_frames: list[dict],
    segment_size: int = 3,
    target_windows: int = 7,
    min_consec: int = 15,
) -> list[dict]:
    """Sample consecutive runs of frames from each video for temporal eval.

    For each video:
      - Need at least min_consec consecutive frames
      - Target target_windows temporal windows (= target_windows * segment_size frames)
      - Pick a centered block of consecutive frames
      - Videos too short are skipped

    Returns flat list of frames, ordered by video then frame.
    """
    videos = group_frames_by_video(all_frames)
    run_length = max(target_windows * segment_size, min_consec)
    sampled = []
    skipped = 0

    for vp, vframes in videos.items():
        n = len(vframes)
        if n < min_consec:
            skipped += 1
            continue
        # Adapt run length to video size
        actual_run = min(run_length, n)
        # At least min_consec
        actual_run = max(actual_run, min(min_consec, n))
        # Center the block
        offset = (n - actual_run) // 2
        sampled.extend(vframes[offset:offset + actual_run])

    n_videos = len(videos) - skipped
    n_windows = sum(
        max(0, min(run_length, len(vf)) // segment_size)
        for vf in videos.values() if len(vf) >= min_consec
    )
    print(f"  Temporal sampling: {len(sampled)} frames from {n_videos} videos "
          f"(~{n_windows} windows of {segment_size}, skipped {skipped} short videos)")
    return sampled


# ── Core evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    weights: Path,
    imgsz: int,
    modality: str,
    conf: float,
    frames: list[dict],
    scoring_rule: str,
    device: str,
    out_dir: Path | None = None,
    cache_suffix: str = "",
) -> dict:
    """Run YOLO inference + per-size scoring on a list of frames.

    Returns dict with:
      - per_size: {bucket: {tp, fp, fn, n_gt}}
      - frame_level: {bucket: {tp, fp, fn, tn}}  (binary: any-det vs any-gt)
      - n_frames: int
      - det_cache_path: Path to saved per-frame detections JSON
    """
    from ultralytics import YOLO
    yolo = YOLO(str(weights))

    per_size = {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}
    frame_level = {b: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for b in list(SIZE_BUCKETS) + ["all"]}
    det_cache = {}
    n_frames = 0
    t0 = time.time()
    total = len(frames)

    for i, fr in enumerate(frames):
        # Choose input image based on modality
        if modality == "ir" and fr["ir_path"] is not None:
            img = cv2.imread(str(fr["ir_path"]))
            lbl_path = fr["ir_lbl"]
        else:
            img = cv2.imread(str(fr["rgb_path"]))
            lbl_path = fr["rgb_lbl"]

        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground truth
        gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []

        # Track GT counts per size
        for g in gts:
            per_size[classify_size(g, w, h)]["n_gt"] += 1

        # Run inference
        res = yolo.predict(img, imgsz=imgsz, conf=conf, device=device, verbose=False)
        r0 = res[0]
        dets = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]

        # Cache detections
        det_cache[fr["stem"]] = [[b[0], b[1], b[2], b[3], c] for b, c in dets]

        # Per-size detection scoring
        s = score_per_size(dets, gts, w, h, iou_thr=0.5, iop_thr=0.5)[scoring_rule]
        for b in SIZE_BUCKETS:
            per_size[b]["tp"] += s[b]["tp"]
            per_size[b]["fp"] += s[b]["fp"]
            per_size[b]["fn"] += s[b]["fn"]

        # Frame-level binary scoring
        has_det = len(dets) > 0
        has_gt = len(gts) > 0
        if has_gt:
            largest = max(gts, key=lambda g: (g[2] - g[0]) * (g[3] - g[1]))
            bucket = classify_size(largest, w, h)
        else:
            bucket = "all"

        if has_gt and has_det:
            frame_level[bucket]["tp"] += 1
            if bucket != "all":
                frame_level["all"]["tp"] += 1
        elif has_gt and not has_det:
            frame_level[bucket]["fn"] += 1
            if bucket != "all":
                frame_level["all"]["fn"] += 1
        elif not has_gt and has_det:
            frame_level[bucket]["fp"] += 1
            if bucket != "all":
                frame_level["all"]["fp"] += 1
        else:
            frame_level[bucket]["tn"] += 1
            if bucket != "all":
                frame_level["all"]["tn"] += 1

        n_frames += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / fps if fps > 0 else 0
            print(f"  [{model_key}] {i+1}/{total}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0
    if dt > 0:
        print(f"  [{model_key}] Done: {n_frames} frames in {dt:.1f}s ({n_frames/dt:.1f} fps)")
    else:
        print(f"  [{model_key}] Done: 0 frames")

    # Save detection cache
    det_cache_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        det_cache_path = out_dir / f"{model_key}{cache_suffix}_detections.json"
        det_cache_path.write_text(json.dumps(det_cache, indent=1))
        print(f"  -> det cache: {det_cache_path.name} ({len(det_cache)} frames)")

    return {
        "per_size": per_size,
        "frame_level": frame_level,
        "n_frames": n_frames,
        "det_cache_path": det_cache_path,
    }


# ── Classifier evaluation ────────────────────────────────────────────

def evaluate_classifier(
    frames: list[dict],
    ds: dict,
    scoring_rule: str,
    device: str,
    rgb_cache_path: Path | None = None,
    ir_cache_path: Path | None = None,
    ir_conf: float = 0.40,
) -> dict:
    """Run the sa32 trust classifier using cached detections."""
    from ultralytics import YOLO
    import joblib
    from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

    obj = joblib.load(str(CLASSIFIER_PATH))
    clf_model = obj["model"]
    feat_cols = obj.get("features") or obj.get("feat_cols") or []

    # Load cached detections
    rgb_det_cache = json.loads(rgb_cache_path.read_text()) if rgb_cache_path and rgb_cache_path.exists() else {}
    ir_det_cache = json.loads(ir_cache_path.read_text()) if ir_cache_path and ir_cache_path.exists() else {}
    print(f"  Loaded caches: RGB={len(rgb_det_cache)}f  IR={len(ir_det_cache)}f")

    # Only load IR YOLO if no IR cache
    ir_yolo = YOLO(str(IR_WEIGHTS_PATH)) if not ir_det_cache else None

    is_paired = ds["type"] == "paired"
    tp_total = fp_total = fn_total = 0
    # Frame-level counters
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = 0
    t0 = time.time()
    total = len(frames)

    for i, fr in enumerate(frames):
        rgb_img = cv2.imread(str(fr["rgb_path"]))
        if rgb_img is None:
            continue
        h, w = rgb_img.shape[:2]
        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        # RGB GT
        rgb_gts = read_yolo_labels(fr["rgb_lbl"], w, h, drone_classes={0}) if fr["rgb_lbl"] else []

        # IR image + GT
        if is_paired and fr["ir_path"] is not None:
            ir_img = cv2.imread(str(fr["ir_path"]))
            if ir_img is None:
                ir_img = rgb_img
            ih_px, iw_px = ir_img.shape[:2]
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
            ir_gts = read_yolo_labels(fr["ir_lbl"], iw_px, ih_px, drone_classes={0}) if fr["ir_lbl"] else []
        else:
            ir_img = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR)
            ih_px, iw_px = h, w
            ir_gray = rgb_gray
            ir_gts = rgb_gts  # same GT for grayscale fallback

        # Load detections from cache
        stem = fr["stem"]
        if stem in rgb_det_cache:
            rgb_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in rgb_det_cache[stem]]
        else:
            rgb_dets = []

        if stem in ir_det_cache:
            ir_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in ir_det_cache[stem]]
        elif ir_yolo is not None:
            ir_res = ir_yolo.predict(ir_img, imgsz=640, conf=ir_conf, device=device, verbose=False)
            ir_dets = []
            if ir_res[0].boxes is not None and len(ir_res[0].boxes) > 0:
                xyxy = ir_res[0].boxes.xyxy.cpu().numpy()
                confs = ir_res[0].boxes.conf.cpu().numpy()
                ir_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
        else:
            ir_dets = []

        # Build features
        feats = {}
        for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
            confs_list = [c for _, c in dets]
            if not confs_list:
                feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
            else:
                feats.update({f"{prefix}_max_conf": float(max(confs_list)),
                              f"{prefix}_mean_conf": float(np.mean(confs_list))})
        feats.update({f"rgb_{k}": v for k, v in compute_global_features(rgb_gray).items()})
        feats.update({f"ir_{k}": v for k, v in compute_global_features(ir_gray).items()})
        for prefix, dets, gray, gw, gh in (
            ("rgb", rgb_dets, rgb_gray, w, h),
            ("ir", ir_dets, ir_gray, iw_px, ih_px),
        ):
            if not dets:
                feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
            else:
                best_box = max(dets, key=lambda d: d[1])[0]
                tf = compute_target_features(gray, best_box, gw, gh)
                feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
        x = np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

        # Predict trust label
        try:
            label = int(clf_model.predict(x)[0])
        except Exception:
            label = 3

        # Trust-aware scoring
        if is_paired:
            # Score RGB side when trusted
            if label in (1, 3):
                tp_r, fp_r, fn_r = score_detections(rgb_dets, rgb_gts, rule=scoring_rule)
            elif label == 0:
                tp_r, fp_r, fn_r = 0, 0, len(rgb_gts)
            else:  # label==2, RGB not trusted, exclude RGB GT
                tp_r, fp_r, fn_r = 0, 0, 0

            # Score IR side when trusted
            if label in (2, 3):
                tp_i, fp_i, fn_i = score_detections(ir_dets, ir_gts, rule=scoring_rule)
            elif label == 0:
                tp_i, fp_i, fn_i = 0, 0, len(ir_gts)
            else:  # label==1, IR not trusted, exclude IR GT
                tp_i, fp_i, fn_i = 0, 0, 0

            tp_total += tp_r + tp_i
            fp_total += fp_r + fp_i
            fn_total += fn_r + fn_i

            has_det = (label != 0) and (len(rgb_dets) + len(ir_dets) > 0)
            has_gt = len(rgb_gts) + len(ir_gts) > 0
        else:
            # Single-modality
            if label == 0:
                kept = []
            elif label == 1:
                kept = rgb_dets
            elif label == 2:
                kept = ir_dets
            else:
                kept = rgb_dets + ir_dets
            tp, fp, fn = score_detections(kept, rgb_gts, rule=scoring_rule)
            tp_total += tp; fp_total += fp; fn_total += fn
            has_det = len(kept) > 0
            has_gt = len(rgb_gts) > 0

        # Frame-level
        if has_gt and has_det: frm_tp += 1
        elif has_gt and not has_det: frm_fn += 1
        elif not has_gt and has_det: frm_fp += 1
        else: frm_tn += 1

        n_frames += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / fps if fps > 0 else 0
            print(f"  [classifier] {i+1}/{total}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"  [classifier] Done: {n_frames} frames in {dt:.1f}s")

    prf = compute_prf(tp_total, fp_total, fn_total)
    total_frm = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / total_frm * 100, 2) if total_frm > 0 else 0.0
    tn_pct = round(frm_tn / total_frm * 100, 2) if total_frm > 0 else 0.0

    return {
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct,
        "n_frames": n_frames,
        "frame_tp": frm_tp, "frame_fp": frm_fp,
        "frame_fn": frm_fn, "frame_tn": frm_tn,
    }


# ── Patch verifier evaluation ───────────────────────────────────

def evaluate_patch_verifier(
    model_key: str,
    frames: list[dict],
    ds: dict,
    scoring_rule: str,
    det_cache_path: Path,
    modality: str,
    patch_thr: float = 0.70,
) -> dict:
    """Apply patch verifier to cached detections and re-score.

    Uses the RGB patch filter for RGB/grayscale models,
    IR patch filter for IR models."""
    from patch_verifier import PatchVerifier

    patch_path = PATCH_IR_PATH if modality == "ir" else PATCH_RGB_PATH
    pv = PatchVerifier(str(patch_path))
    det_cache = json.loads(det_cache_path.read_text())
    print(f"  Loaded {len(det_cache)} cached dets for {model_key}, patch={patch_path.name}")

    tp_total = fp_total = fn_total = 0
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = 0
    t0 = time.time()
    total = len(frames)

    for i, fr in enumerate(frames):
        # Read image for patch cropping
        if modality == "ir" and fr["ir_path"] is not None:
            img = cv2.imread(str(fr["ir_path"]))
            lbl_path = fr["ir_lbl"]
        else:
            img = cv2.imread(str(fr["rgb_path"]))
            lbl_path = fr["rgb_lbl"]
        if img is None:
            continue
        h, w = img.shape[:2]

        gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []

        # Load cached dets
        stem = fr["stem"]
        raw = det_cache.get(stem, [])
        dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]

        # Apply patch verifier
        if dets:
            boxes = [d[0] for d in dets]
            probs = pv.predict_boxes(img, boxes)
            filtered = [d for d, p in zip(dets, probs) if p < patch_thr]
        else:
            filtered = []

        # Score filtered detections
        tp, fp, fn = score_detections(filtered, gts, rule=scoring_rule)
        tp_total += tp; fp_total += fp; fn_total += fn

        # Frame-level
        has_det = len(filtered) > 0
        has_gt = len(gts) > 0
        if has_gt and has_det: frm_tp += 1
        elif has_gt and not has_det: frm_fn += 1
        elif not has_gt and has_det: frm_fp += 1
        else: frm_tn += 1

        n_frames += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / fps if fps > 0 else 0
            print(f"  [patch/{model_key}] {i+1}/{total}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0
    print(f"  [patch/{model_key}] Done: {n_frames} frames in {dt:.1f}s")

    prf = compute_prf(tp_total, fp_total, fn_total)
    total_frm = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / total_frm * 100, 2) if total_frm > 0 else 0.0
    tn_pct = round(frm_tn / total_frm * 100, 2) if total_frm > 0 else 0.0

    return {
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct,
        "n_frames": n_frames,
        "frame_tp": frm_tp, "frame_fp": frm_fp,
        "frame_fn": frm_fn, "frame_tn": frm_tn,
    }


# ── Temporal evaluation ─────────────────────────────────────────

def evaluate_temporal(
    model_key: str,
    frames: list[dict],
    ds: dict,
    det_cache: dict,
    scoring_rule: str,
    modality: str,
    segment_size: int = 3,
    k: int = 2,
) -> dict:
    """Apply k-out-of-n temporal voting on cached detections.

    Groups consecutive frames (within each video) into segments of
    segment_size. A frame's detections survive only if at least k
    frames in the segment have any detections (frame-level vote).

    Returns aggregate metrics dict."""
    videos = group_frames_by_video(frames)
    tp_total = fp_total = fn_total = 0
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = 0
    n_windows = 0

    for vp, vframes in videos.items():
        # Build per-frame data: dets + GT
        frame_data = []
        for fr in vframes:
            if modality == "ir" and fr["ir_path"] is not None:
                lbl_path = fr["ir_lbl"]
                # Need image dims for GT parsing
                img = cv2.imread(str(fr["ir_path"]))
            else:
                lbl_path = fr["rgb_lbl"]
                img = cv2.imread(str(fr["rgb_path"]))
            if img is None:
                continue
            h, w = img.shape[:2]
            gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []
            raw = det_cache.get(fr["stem"], [])
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]
            frame_data.append({"stem": fr["stem"], "dets": dets, "gts": gts,
                               "has_det": len(dets) > 0, "has_gt": len(gts) > 0})

        # Slide segments of segment_size
        for seg_start in range(0, len(frame_data) - segment_size + 1, segment_size):
            seg = frame_data[seg_start:seg_start + segment_size]
            det_count = sum(1 for f in seg if f["has_det"])
            confirmed = det_count >= k  # temporal vote
            n_windows += 1

            for fd in seg:
                # If temporal vote says "not confirmed", suppress all dets
                kept = fd["dets"] if confirmed else []
                tp, fp, fn = score_detections(kept, fd["gts"], rule=scoring_rule)
                tp_total += tp; fp_total += fp; fn_total += fn

                has_det = len(kept) > 0
                has_gt = fd["has_gt"]
                if has_gt and has_det: frm_tp += 1
                elif has_gt and not has_det: frm_fn += 1
                elif not has_gt and has_det: frm_fp += 1
                else: frm_tn += 1
                n_frames += 1

    prf = compute_prf(tp_total, fp_total, fn_total)
    total_frm = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / total_frm * 100, 2) if total_frm > 0 else 0.0
    tn_pct = round(frm_tn / total_frm * 100, 2) if total_frm > 0 else 0.0

    print(f"  [temporal/{model_key}] {n_frames} frames, {n_windows} windows, "
          f"P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}")

    return {
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct,
        "n_frames": n_frames, "n_windows": n_windows,
        "frame_tp": frm_tp, "frame_fp": frm_fp,
        "frame_fn": frm_fn, "frame_tn": frm_tn,
    }


# ── TROI evaluation ────────────────────────────────────────────

def evaluate_troi(
    model_key: str,
    frames: list[dict],
    ds: dict,
    det_cache: dict,
    scoring_rule: str,
    modality: str,
    weights: Path,
    imgsz: int,
    conf: float,
    device: str,
    roi_ttl: int = 5,
    roi_expand: float = 1.5,
    troi_conf_factor: float = 0.8,
) -> dict:
    """Evaluate TROI (Temporal ROI Recovery) on consecutive frames.

    Uses cached full-frame detections. When a frame has no dets but a
    recent previous frame did (within roi_ttl), crops around last known
    position and re-runs YOLO on the crop to recover missed drones.
    """
    from ultralytics import YOLO
    yolo = YOLO(str(weights))

    videos = group_frames_by_video(frames)
    tp_total = fp_total = fn_total = 0
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = 0
    n_recovered = 0  # frames where TROI found something
    t0 = time.time()

    for vp, vframes in videos.items():
        last_roi = None  # (x1,y1,x2,y2) of best det
        roi_age = 0

        for fr in vframes:
            # Read image
            if modality == "ir" and fr["ir_path"] is not None:
                img = cv2.imread(str(fr["ir_path"]))
                lbl_path = fr["ir_lbl"]
            else:
                img = cv2.imread(str(fr["rgb_path"]))
                lbl_path = fr["rgb_lbl"]
            if img is None:
                continue
            h, w = img.shape[:2]
            gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []

            # Cached full-frame dets
            raw = det_cache.get(fr["stem"], [])
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]

            # TROI recovery: if no dets but last_roi exists
            if not dets and last_roi is not None and roi_age > 0 and roi_age <= roi_ttl:
                x1r, y1r, x2r, y2r = last_roi
                bw, bh = x2r - x1r, y2r - y1r
                cx, cy = (x1r + x2r) / 2, (y1r + y2r) / 2
                nw = max(bw * roi_expand, 128)
                nh = max(bh * roi_expand, 128)
                rx1 = int(max(0, cx - nw / 2))
                ry1 = int(max(0, cy - nh / 2))
                rx2 = int(min(w, cx + nw / 2))
                ry2 = int(min(h, cy + nh / 2))
                if rx2 - rx1 >= 32 and ry2 - ry1 >= 32:
                    crop = img[ry1:ry2, rx1:rx2]
                    troi_conf = conf * troi_conf_factor
                    res = yolo.predict(crop, imgsz=imgsz, conf=troi_conf,
                                       device=device, verbose=False)
                    r0 = res[0]
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs_arr = r0.boxes.conf.cpu().numpy()
                        # Remap to full frame coords
                        for b, c in zip(xyxy, confs_arr):
                            fb = (float(b[0]) + rx1, float(b[1]) + ry1,
                                  float(b[2]) + rx1, float(b[3]) + ry1)
                            dets.append((fb, float(c)))
                        n_recovered += 1

            # Update ROI state
            if dets:
                best = max(dets, key=lambda d: d[1])
                last_roi = best[0]  # (x1,y1,x2,y2)
                roi_age = 0
            else:
                roi_age += 1
                if roi_age > roi_ttl:
                    last_roi = None

            # Score
            tp, fp, fn = score_detections(dets, gts, rule=scoring_rule)
            tp_total += tp; fp_total += fp; fn_total += fn

            has_det = len(dets) > 0
            has_gt = len(gts) > 0
            if has_gt and has_det: frm_tp += 1
            elif has_gt and not has_det: frm_fn += 1
            elif not has_gt and has_det: frm_fp += 1
            else: frm_tn += 1
            n_frames += 1

    dt = time.time() - t0
    prf = compute_prf(tp_total, fp_total, fn_total)
    total_frm = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / total_frm * 100, 2) if total_frm > 0 else 0.0
    tn_pct = round(frm_tn / total_frm * 100, 2) if total_frm > 0 else 0.0

    print(f"  [troi/{model_key}] {n_frames} frames, {n_recovered} recovered, "
          f"{dt:.1f}s  P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}")

    return {
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct,
        "n_frames": n_frames, "n_recovered": n_recovered,
        "frame_tp": frm_tp, "frame_fp": frm_fp,
        "frame_fn": frm_fn, "frame_tn": frm_tn,
    }


# ── Alert gate evaluation ────────────────────────────────────────────

def evaluate_alert_gate(
    model_key: str,
    frames: list[dict],
    ds: dict,
    det_cache: dict,
    scoring_rule: str,
    modality: str,
    segment_size: int = 3,
    k: int = 2,
    patch_thr: float = 0.70,
) -> dict:
    """Evaluate alert gate: temporal voting + confuser filter at decision point.

    For each segment of consecutive frames:
      1. Check temporal vote (k/n frames have detections)
      2. If confirmed, apply patch verifier to detection crops
      3. If any frame's detections are flagged as confuser, suppress entire segment
    
    This simulates the production pipeline's alert gate behavior.
    """
    from patch_verifier import PatchVerifier

    patch_path = PATCH_IR_PATH if modality == "ir" else PATCH_RGB_PATH
    pv = PatchVerifier(str(patch_path))

    videos = group_frames_by_video(frames)
    tp_total = fp_total = fn_total = 0
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = 0
    n_suppressed = 0  # segments suppressed by confuser filter
    t0 = time.time()

    for vp, vframes in videos.items():
        # Build per-frame data
        frame_data = []
        for fr in vframes:
            if modality == "ir" and fr["ir_path"] is not None:
                img = cv2.imread(str(fr["ir_path"]))
                lbl_path = fr["ir_lbl"]
            else:
                img = cv2.imread(str(fr["rgb_path"]))
                lbl_path = fr["rgb_lbl"]
            if img is None:
                continue
            h, w = img.shape[:2]
            gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []
            raw = det_cache.get(fr["stem"], [])
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]
            frame_data.append({"stem": fr["stem"], "dets": dets, "gts": gts,
                               "has_det": len(dets) > 0, "has_gt": len(gts) > 0,
                               "img": img})

        # Process segments
        for seg_start in range(0, len(frame_data) - segment_size + 1, segment_size):
            seg = frame_data[seg_start:seg_start + segment_size]
            det_count = sum(1 for f in seg if f["has_det"])
            confirmed = det_count >= k

            # Alert gate: if temporal vote confirms, check confuser filter
            gate_suppressed = False
            if confirmed:
                # Check patch verifier on frames that have detections
                confuser_votes = 0
                det_frames = [f for f in seg if f["has_det"]]
                for fd in det_frames:
                    boxes = [d[0] for d in fd["dets"]]
                    probs = pv.predict_boxes(fd["img"], boxes)
                    if hasattr(probs, '__len__') and len(probs) > 0:
                        max_prob = float(np.max(probs))
                    else:
                        max_prob = 0.0
                    if max_prob >= patch_thr:
                        confuser_votes += 1
                # If any frame in segment is flagged as confuser, suppress
                if confuser_votes > 0:
                    gate_suppressed = True
                    n_suppressed += 1

            for fd in seg:
                if not confirmed or gate_suppressed:
                    kept = []
                else:
                    kept = fd["dets"]
                tp, fp, fn = score_detections(kept, fd["gts"], rule=scoring_rule)
                tp_total += tp; fp_total += fp; fn_total += fn

                has_det = len(kept) > 0
                has_gt = fd["has_gt"]
                if has_gt and has_det: frm_tp += 1
                elif has_gt and not has_det: frm_fn += 1
                elif not has_gt and has_det: frm_fp += 1
                else: frm_tn += 1
                n_frames += 1

    dt = time.time() - t0
    prf = compute_prf(tp_total, fp_total, fn_total)
    total_frm = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / total_frm * 100, 2) if total_frm > 0 else 0.0
    tn_pct = round(frm_tn / total_frm * 100, 2) if total_frm > 0 else 0.0

    print(f"  [alert_gate/{model_key}] {n_frames} frames, {n_suppressed} segments suppressed, "
          f"{dt:.1f}s  P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}")

    return {
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct,
        "n_frames": n_frames, "n_suppressed": n_suppressed,
        "frame_tp": frm_tp, "frame_fp": frm_fp,
        "frame_fn": frm_fn, "frame_tn": frm_tn,
    }


# ── CSV output ───────────────────────────────────────────────────────

def write_results(results: dict, model_key: str, dataset_key: str,
                  scoring_rule: str, out_dir: Path):
    """Write per-size + aggregate CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Detection-level CSV (per-size + all) ---
    det_path = out_dir / f"{model_key}_{dataset_key}_detection.csv"
    fieldnames = ["model", "dataset", "scoring", "size", "TP", "FP", "FN",
                  "n_gt", "precision", "recall", "f1", "n_frames"]
    rows = []
    total = {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0}
    for b in SIZE_BUCKETS:
        c = results["per_size"][b]
        prf = compute_prf(c["tp"], c["fp"], c["fn"])
        rows.append({
            "model": model_key, "dataset": dataset_key, "scoring": scoring_rule,
            "size": b,
            "TP": c["tp"], "FP": c["fp"], "FN": c["fn"], "n_gt": c["n_gt"],
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
            "n_frames": results["n_frames"],
        })
        total["tp"] += c["tp"]; total["fp"] += c["fp"]
        total["fn"] += c["fn"]; total["n_gt"] += c["n_gt"]

    prf_all = compute_prf(total["tp"], total["fp"], total["fn"])
    rows.append({
        "model": model_key, "dataset": dataset_key, "scoring": scoring_rule,
        "size": "all",
        "TP": total["tp"], "FP": total["fp"], "FN": total["fn"],
        "n_gt": total["n_gt"],
        "precision": prf_all["precision"], "recall": prf_all["recall"],
        "f1": prf_all["f1"],
        "n_frames": results["n_frames"],
    })

    with det_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {det_path}")

    # --- Frame-level CSV (per-size + all) ---
    frm_path = out_dir / f"{model_key}_{dataset_key}_frame_level.csv"
    fieldnames_f = ["model", "dataset", "size", "TP", "FP", "FN", "TN",
                    "total", "FP_pct", "TN_pct", "precision", "recall", "f1"]
    rows_f = []
    for b in list(SIZE_BUCKETS) + ["all"]:
        c = results["frame_level"][b]
        total_f = c["tp"] + c["fp"] + c["fn"] + c["tn"]
        prf = compute_prf(c["tp"], c["fp"], c["fn"])
        fp_pct = round(c["fp"] / total_f * 100, 2) if total_f > 0 else 0.0
        tn_pct = round(c["tn"] / total_f * 100, 2) if total_f > 0 else 0.0
        rows_f.append({
            "model": model_key, "dataset": dataset_key, "size": b,
            "TP": c["tp"], "FP": c["fp"], "FN": c["fn"], "TN": c["tn"],
            "total": total_f,
            "FP_pct": fp_pct, "TN_pct": tn_pct,
            "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
        })

    with frm_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_f)
        w.writeheader()
        w.writerows(rows_f)
    print(f"  -> {frm_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {model_key} on {dataset_key} ({scoring_rule} @ 0.5, {results['n_frames']} frames)")
    print(f"{'='*70}")
    print(f"  {'Size':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'n_gt':>6}  {'P':>7} {'R':>7} {'F1':>7}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r['size']:<10} {r['TP']:>6} {r['FP']:>6} {r['FN']:>6} {r['n_gt']:>6}"
              f"  {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
    print()
    print(f"  Frame-level:")
    print(f"  {'Size':<10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} {'FP%':>7} {'TN%':>7}")
    print(f"  {'-'*50}")
    for r in rows_f:
        print(f"  {r['size']:<10} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['TN']:>5}"
              f"  {r['FP_pct']:>6.2f}% {r['TN_pct']:>6.2f}%")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Reusable YOLO detector evaluation")
    ap.add_argument("--dataset", required=True,
                    help=f"Dataset key: {', '.join(DATASET_REGISTRY)} or 'custom'")
    ap.add_argument("--models", nargs="+", default=["selcom_960", "ir_v3b"],
                    help=f"Model keys: {', '.join(MODEL_REGISTRY)}")
    ap.add_argument("--all-models", action="store_true",
                    help="Run all registered models")
    ap.add_argument("--max-frames", type=int, default=1000,
                    help="Max frames to evaluate (uniform stride to hit this target)")
    ap.add_argument("--scoring", type=str, default=None,
                    help="Scoring rule override: iou or iop (default: dataset-specific)")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for CSVs (default: eval/results/detector_eval/)")
    ap.add_argument("--with-classifier", action="store_true",
                    help="Also evaluate the sa32 trust classifier (trust-aware scoring)")
    ap.add_argument("--with-patch", action="store_true",
                    help="Also evaluate patch verifier (confuser filter) on each model")
    ap.add_argument("--patch-thr", type=float, default=0.70,
                    help="Patch verifier confuser threshold (default: 0.70)")
    ap.add_argument("--with-temporal", action="store_true",
                    help="Evaluate temporal voting (k/n) on consecutive frames")
    ap.add_argument("--temporal-seg", type=int, default=3,
                    help="Temporal segment size (default: 3)")
    ap.add_argument("--temporal-k", type=int, default=2,
                    help="Temporal k-out-of-n voting threshold (default: 2)")
    ap.add_argument("--temporal-windows", type=int, default=7,
                    help="Target temporal windows per video (default: 7)")
    ap.add_argument("--temporal-min-frames", type=int, default=15,
                    help="Minimum consecutive frames per video (default: 15)")
    ap.add_argument("--with-troi", action="store_true",
                    help="Evaluate TROI (Temporal ROI Recovery) on consecutive frames")
    ap.add_argument("--troi-ttl", type=int, default=5,
                    help="TROI time-to-live in frames (default: 5)")
    ap.add_argument("--troi-expand", type=float, default=1.5,
                    help="TROI ROI expansion factor (default: 1.5)")
    # Custom dataset paths
    ap.add_argument("--img-dir", type=str, help="Custom image directory")
    ap.add_argument("--lbl-dir", type=str, help="Custom label directory")
    ap.add_argument("--ir-img-dir", type=str, help="Custom IR image directory (paired)")
    ap.add_argument("--ir-lbl-dir", type=str, help="Custom IR label directory (paired)")
    args = ap.parse_args()

    # Resolve dataset
    if args.dataset == "custom":
        if not args.img_dir or not args.lbl_dir:
            ap.error("--img-dir and --lbl-dir required for custom dataset")
        ds = {
            "type": "paired" if args.ir_img_dir else "image",
            "scoring": args.scoring or "iou",
        }
        if ds["type"] == "paired":
            ds.update({
                "rgb_img": Path(args.img_dir), "rgb_lbl": Path(args.lbl_dir),
                "ir_img": Path(args.ir_img_dir), "ir_lbl": Path(args.ir_lbl_dir),
            })
        else:
            ds.update({"img_dir": Path(args.img_dir), "lbl_dir": Path(args.lbl_dir)})
        dataset_key = "custom"
    elif args.dataset in DATASET_REGISTRY:
        ds = DATASET_REGISTRY[args.dataset]
        dataset_key = args.dataset
    else:
        ap.error(f"Unknown dataset '{args.dataset}'. Available: {', '.join(DATASET_REGISTRY)}")
        return

    scoring_rule = args.scoring or ds["scoring"]

    # List all frames and apply stride
    all_frames = list_frames(ds)
    n_total = len(all_frames)
    if n_total == 0:
        print(f"ERROR: No frames found for dataset '{dataset_key}'")
        return

    stride = max(1, n_total // args.max_frames) if n_total > args.max_frames else 1
    frames = all_frames[::stride][:args.max_frames]
    print(f"Dataset: {dataset_key}  |  Total frames: {n_total}  |  Stride: {stride}  |  Evaluating: {len(frames)}")

    # Resolve models
    if args.all_models:
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = args.models
        for mk in model_keys:
            if mk not in MODEL_REGISTRY:
                print(f"ERROR: Unknown model '{mk}'. Available: {', '.join(MODEL_REGISTRY)}")
                return

    # Output dir
    out_dir = Path(args.output_dir) if args.output_dir else (EVAL_DIR / "results" / "detector_eval")

    # Run each model
    model_results = {}
    for mk in model_keys:
        weights, imgsz, modality, conf = MODEL_REGISTRY[mk]
        if not weights.exists():
            print(f"SKIP {mk}: weights not found at {weights}")
            continue

        # For IR model on non-paired datasets, skip (no IR images)
        if modality == "ir" and ds["type"] != "paired":
            print(f"SKIP {mk}: IR model requires paired dataset with IR images")
            continue

        print(f"\n{'#'*70}")
        print(f"# Evaluating: {mk}  (imgsz={imgsz}, modality={modality}, conf={conf})")
        print(f"# Scoring: {scoring_rule} @ 0.5")
        print(f"{'#'*70}")

        results = evaluate_model(mk, weights, imgsz, modality, conf,
                                 frames, scoring_rule, args.device,
                                 out_dir=out_dir)
        write_results(results, mk, dataset_key, scoring_rule, out_dir)
        model_results[mk] = results

    # ── Classifier evaluation ──
    if args.with_classifier:
        print(f"\n{'#'*70}")
        print(f"# Evaluating: sa32 trust classifier (using cached dets)")
        print(f"{'#'*70}")
        # Find RGB and IR caches
        rgb_cache = ir_cache = None
        for mk, res in model_results.items():
            cp = res.get("det_cache_path")
            if cp is None: continue
            _, _, mod, _ = MODEL_REGISTRY[mk]
            if mod == "ir" and ir_cache is None: ir_cache = cp
            elif mod == "rgb" and rgb_cache is None: rgb_cache = cp
        clf_results = evaluate_classifier(frames, ds, scoring_rule, args.device,
                                          rgb_cache_path=rgb_cache, ir_cache_path=ir_cache)

        # Write classifier CSV
        clf_path = out_dir / f"classifier_sa32_{dataset_key}.csv"
        with clf_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["stage", "TP", "FP", "FN", "P", "R", "F1",
                                              "FP_pct", "TN_pct", "n_frames",
                                              "frame_tp", "frame_fp", "frame_fn", "frame_tn"])
            w.writeheader()
            w.writerow({"stage": "classifier_sa32", **clf_results})
        print(f"  -> {clf_path}")

        # Print classifier impact table
        print(f"\n{'='*70}")
        print(f"  Classifier Impact (sa32, trust-aware, {scoring_rule} @ 0.5)")
        print(f"{'='*70}")
        print(f"  {'Stage':<25} {'P':>7} {'R':>7} {'F1':>7} {'FP%':>7} {'TN%':>7}  ΔP      ΔR      ΔF1")
        print(f"  {'-'*90}")

        # Print each standalone model first, then classifier with deltas
        for mk, res in model_results.items():
            ps = res["per_size"]
            t_tp = sum(ps[b]["tp"] for b in SIZE_BUCKETS)
            t_fp = sum(ps[b]["fp"] for b in SIZE_BUCKETS)
            t_fn = sum(ps[b]["fn"] for b in SIZE_BUCKETS)
            prf = compute_prf(t_tp, t_fp, t_fn)
            fl = res["frame_level"]["all"]
            total_f = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
            fp_p = round(fl["fp"] / total_f * 100, 2) if total_f else 0
            tn_p = round(fl["tn"] / total_f * 100, 2) if total_f else 0
            print(f"  {mk:<25} {prf['precision']:>7.4f} {prf['recall']:>7.4f} {prf['f1']:>7.4f}"
                  f" {fp_p:>6.2f}% {tn_p:>6.2f}%")

        # Classifier row with deltas vs each model
        c = clf_results
        print(f"  {'classifier_sa32':<25} {c['P']:>7.4f} {c['R']:>7.4f} {c['F1']:>7.4f}"
              f" {c['FP_pct']:>6.2f}% {c['TN_pct']:>6.2f}%")
        for mk, res in model_results.items():
            ps = res["per_size"]
            t_tp = sum(ps[b]["tp"] for b in SIZE_BUCKETS)
            t_fp = sum(ps[b]["fp"] for b in SIZE_BUCKETS)
            t_fn = sum(ps[b]["fn"] for b in SIZE_BUCKETS)
            prf = compute_prf(t_tp, t_fp, t_fn)
            dp = c["P"] - prf["precision"]
            dr = c["R"] - prf["recall"]
            df = c["F1"] - prf["f1"]
            print(f"    Δ vs {mk:<18} {dp:>+7.4f} {dr:>+7.4f} {df:>+7.4f}")
        print()

    # ── Patch verifier evaluation ──
    if args.with_patch:
        print(f"\n{'#'*70}")
        print(f"# Evaluating: patch verifier (confuser filter, thr={args.patch_thr})")
        print(f"{'#'*70}")
        patch_results = {}
        for mk, res in model_results.items():
            cp = res.get("det_cache_path")
            if cp is None:
                print(f"  SKIP {mk}: no det cache")
                continue
            _, _, mod, _ = MODEL_REGISTRY[mk]
            pr = evaluate_patch_verifier(mk, frames, ds, scoring_rule, cp, mod,
                                         patch_thr=args.patch_thr)
            patch_results[mk] = pr

            # Write CSV
            pv_path = out_dir / f"patch_{mk}_{dataset_key}.csv"
            with pv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["stage", "TP", "FP", "FN", "P", "R", "F1",
                                                  "FP_pct", "TN_pct", "n_frames",
                                                  "frame_tp", "frame_fp", "frame_fn", "frame_tn"])
                w.writeheader()
                w.writerow({"stage": f"patch_{mk}", **pr})
            print(f"  -> {pv_path}")

        # Print patch impact table
        print(f"\n{'='*70}")
        print(f"  Patch Verifier Impact (thr={args.patch_thr}, {scoring_rule} @ 0.5)")
        print(f"{'='*70}")
        print(f"  {'Stage':<30} {'P':>7} {'R':>7} {'F1':>7} {'FP%':>7} {'TN%':>7}  \u0394P      \u0394R      \u0394F1")
        print(f"  {'-'*95}")
        for mk in patch_results:
            # Standalone row
            res = model_results[mk]
            ps = res["per_size"]
            t_tp = sum(ps[b]["tp"] for b in SIZE_BUCKETS)
            t_fp = sum(ps[b]["fp"] for b in SIZE_BUCKETS)
            t_fn = sum(ps[b]["fn"] for b in SIZE_BUCKETS)
            prf = compute_prf(t_tp, t_fp, t_fn)
            fl = res["frame_level"]["all"]
            total_f = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
            fp_p = round(fl["fp"] / total_f * 100, 2) if total_f else 0
            tn_p = round(fl["tn"] / total_f * 100, 2) if total_f else 0
            print(f"  {mk:<30} {prf['precision']:>7.4f} {prf['recall']:>7.4f} {prf['f1']:>7.4f}"
                  f" {fp_p:>6.2f}% {tn_p:>6.2f}%")
            # Patch row with delta
            pr = patch_results[mk]
            dp = pr["P"] - prf["precision"]
            dr = pr["R"] - prf["recall"]
            df = pr["F1"] - prf["f1"]
            print(f"  {('+ patch'):>30} {pr['P']:>7.4f} {pr['R']:>7.4f} {pr['F1']:>7.4f}"
                  f" {pr['FP_pct']:>6.2f}% {pr['TN_pct']:>6.2f}%  {dp:>+7.4f} {dr:>+7.4f} {df:>+7.4f}")
        print()

    # ── Temporal evaluation ──
    if args.with_temporal:
        print(f"\n{'#'*70}")
        print(f"# Temporal Voting: {args.temporal_k}/{args.temporal_seg} "
              f"(target {args.temporal_windows} windows/video, min {args.temporal_min_frames} frames)")
        print(f"{'#'*70}")

        # Sample consecutive frames for temporal eval
        all_frames_full = list_frames(ds)
        temporal_frames = sample_temporal_frames(
            all_frames_full,
            segment_size=args.temporal_seg,
            target_windows=args.temporal_windows,
            min_consec=args.temporal_min_frames,
        )

        # Run inference on temporal frames for each model
        temporal_results_raw = {}
        temporal_results_voted = {}
        for mk in model_keys:
            weights, imgsz, modality, conf = MODEL_REGISTRY[mk]
            if not weights.exists():
                continue
            if modality == "ir" and ds["type"] != "paired":
                continue

            print(f"\n  Running inference: {mk} on {len(temporal_frames)} temporal frames...")
            temp_res = evaluate_model(mk, weights, imgsz, modality, conf,
                                      temporal_frames, scoring_rule, args.device,
                                      out_dir=out_dir, cache_suffix=f"_temporal_{dataset_key}")
            temporal_results_raw[mk] = temp_res

            # Load the det cache we just wrote
            cache_path = temp_res.get("det_cache_path")
            if cache_path and cache_path.exists():
                det_cache = json.loads(cache_path.read_text())
            else:
                print(f"  SKIP temporal vote for {mk}: no cache")
                continue

            # Apply temporal voting
            voted = evaluate_temporal(
                mk, temporal_frames, ds, det_cache, scoring_rule, modality,
                segment_size=args.temporal_seg, k=args.temporal_k,
            )
            temporal_results_voted[mk] = voted

            # Write CSV
            tv_path = out_dir / f"temporal_{mk}_{dataset_key}.csv"
            with tv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["stage", "TP", "FP", "FN", "P", "R", "F1",
                                                  "FP_pct", "TN_pct", "n_frames", "n_windows",
                                                  "frame_tp", "frame_fp", "frame_fn", "frame_tn"])
                w.writeheader()
                w.writerow({"stage": f"temporal_{mk}", **voted})
            print(f"  -> {tv_path}")

        # Print temporal impact table
        if temporal_results_voted:
            print(f"\n{'='*70}")
            print(f"  Temporal Voting Impact ({args.temporal_k}/{args.temporal_seg}, {scoring_rule} @ 0.5)")
            print(f"{'='*70}")
            print(f"  {'Stage':<30} {'P':>7} {'R':>7} {'F1':>7} {'FP%':>7} {'TN%':>7}  \u0394P      \u0394R      \u0394F1")
            print(f"  {'-'*95}")
            for mk in temporal_results_voted:
                # Raw (no voting) on same temporal frames
                res = temporal_results_raw[mk]
                ps = res["per_size"]
                t_tp = sum(ps[b]["tp"] for b in SIZE_BUCKETS)
                t_fp = sum(ps[b]["fp"] for b in SIZE_BUCKETS)
                t_fn = sum(ps[b]["fn"] for b in SIZE_BUCKETS)
                prf = compute_prf(t_tp, t_fp, t_fn)
                fl = res["frame_level"]["all"]
                total_f = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
                fp_p = round(fl["fp"] / total_f * 100, 2) if total_f else 0
                tn_p = round(fl["tn"] / total_f * 100, 2) if total_f else 0
                print(f"  {mk:<30} {prf['precision']:>7.4f} {prf['recall']:>7.4f} {prf['f1']:>7.4f}"
                      f" {fp_p:>6.2f}% {tn_p:>6.2f}%")
                # Voted row with delta
                v = temporal_results_voted[mk]
                dp = v["P"] - prf["precision"]
                dr = v["R"] - prf["recall"]
                df = v["F1"] - prf["f1"]
                print(f"  {'+ temporal':>30} {v['P']:>7.4f} {v['R']:>7.4f} {v['F1']:>7.4f}"
                      f" {v['FP_pct']:>6.2f}% {v['TN_pct']:>6.2f}%  {dp:>+7.4f} {dr:>+7.4f} {df:>+7.4f}")
            print()

    # ── TROI evaluation ──
    if args.with_troi:
        print(f"\n{'#'*70}")
        print(f"# TROI Recovery (ttl={args.troi_ttl}, expand={args.troi_expand})")
        print(f"{'#'*70}")

        # Need consecutive frames — reuse temporal frames if available, else sample
        if 'temporal_frames' not in dir():
            all_frames_full = list_frames(ds)
            temporal_frames = sample_temporal_frames(
                all_frames_full,
                segment_size=getattr(args, 'temporal_seg', 3),
                target_windows=getattr(args, 'temporal_windows', 7),
                min_consec=getattr(args, 'temporal_min_frames', 15),
            )

        troi_results = {}
        for mk in model_keys:
            weights, imgsz, modality, conf = MODEL_REGISTRY[mk]
            if not weights.exists():
                continue
            if modality == "ir" and ds["type"] != "paired":
                continue

            # Need det cache for this model on temporal frames
            # Check if temporal already ran it
            cache_path = out_dir / f"{mk}_temporal_{dataset_key}_detections.json"
            if not cache_path.exists():
                print(f"  Running inference for {mk} to build det cache...")
                temp_res = evaluate_model(mk, weights, imgsz, modality, conf,
                                          temporal_frames, scoring_rule, args.device,
                                          out_dir=out_dir, cache_suffix=f"_temporal_{dataset_key}")
            det_cache_data = json.loads(cache_path.read_text())

            troi_res = evaluate_troi(
                mk, temporal_frames, ds, det_cache_data, scoring_rule, modality,
                weights, imgsz, conf, args.device,
                roi_ttl=args.troi_ttl, roi_expand=args.troi_expand,
            )
            troi_results[mk] = troi_res

            # Write CSV
            tr_path = out_dir / f"troi_{mk}_{dataset_key}.csv"
            with tr_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["stage", "TP", "FP", "FN", "P", "R", "F1",
                                                  "FP_pct", "TN_pct", "n_frames", "n_recovered",
                                                  "frame_tp", "frame_fp", "frame_fn", "frame_tn"])
                w.writeheader()
                w.writerow({"stage": f"troi_{mk}", **troi_res})
            print(f"  -> {tr_path}")

        # Print TROI impact table
        if troi_results:
            print(f"\n{'='*70}")
            print(f"  TROI Recovery Impact (ttl={args.troi_ttl}, {scoring_rule} @ 0.5)")
            print(f"{'='*70}")
            print(f"  {'Stage':<30} {'P':>7} {'R':>7} {'F1':>7} {'FP%':>7} {'TN%':>7} {'Recov':>6}  \u0394P      \u0394R      \u0394F1")
            print(f"  {'-'*100}")
            for mk in troi_results:
                # Raw baseline from temporal results or re-read cache
                raw_key = f"{mk}_raw"
                if 'temporal_results_raw' in dir() and mk in temporal_results_raw:
                    res = temporal_results_raw[mk]
                else:
                    # Compute from cache
                    cache_path = out_dir / f"{mk}_detections.json"
                    det_cache_data = json.loads(cache_path.read_text())
                    # Quick score from cache
                    t_tp = t_fp = t_fn = 0
                    f_tp = f_fp = f_fn = f_tn = 0
                    for fr in temporal_frames:
                        if modality == "ir" and fr["ir_path"] is not None:
                            img = cv2.imread(str(fr["ir_path"]))
                            lbl_path = fr["ir_lbl"]
                        else:
                            img = cv2.imread(str(fr["rgb_path"]))
                            lbl_path = fr["rgb_lbl"]
                        if img is None: continue
                        h_px, w_px = img.shape[:2]
                        gts = read_yolo_labels(lbl_path, w_px, h_px, drone_classes={0}) if lbl_path else []
                        raw = det_cache_data.get(fr["stem"], [])
                        dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]
                        tp, fp, fn = score_detections(dets, gts, rule=scoring_rule)
                        t_tp += tp; t_fp += fp; t_fn += fn
                        hd = len(dets) > 0; hg = len(gts) > 0
                        if hg and hd: f_tp += 1
                        elif hg and not hd: f_fn += 1
                        elif not hg and hd: f_fp += 1
                        else: f_tn += 1
                    prf_raw = compute_prf(t_tp, t_fp, t_fn)
                    tot = f_tp + f_fp + f_fn + f_tn
                    res = {"per_size": {b: {"tp": 0, "fp": 0, "fn": 0} for b in SIZE_BUCKETS},
                           "frame_level": {"all": {"tp": f_tp, "fp": f_fp, "fn": f_fn, "tn": f_tn}}}
                    # Fake per_size with totals for the print logic
                    res["per_size"]["small"] = {"tp": t_tp, "fp": t_fp, "fn": t_fn}
                    res["_prf"] = prf_raw

                # Get raw P/R/F1
                if "_prf" in res:
                    prf = res["_prf"]
                else:
                    ps = res["per_size"]
                    rt_tp = sum(ps[b]["tp"] for b in SIZE_BUCKETS)
                    rt_fp = sum(ps[b]["fp"] for b in SIZE_BUCKETS)
                    rt_fn = sum(ps[b]["fn"] for b in SIZE_BUCKETS)
                    prf = compute_prf(rt_tp, rt_fp, rt_fn)
                fl = res["frame_level"]["all"]
                total_f = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
                fp_p = round(fl["fp"] / total_f * 100, 2) if total_f else 0
                tn_p = round(fl["tn"] / total_f * 100, 2) if total_f else 0
                print(f"  {mk:<30} {prf['precision']:>7.4f} {prf['recall']:>7.4f} {prf['f1']:>7.4f}"
                      f" {fp_p:>6.2f}% {tn_p:>6.2f}%")

                # TROI row
                tr = troi_results[mk]
                dp = tr["P"] - prf["precision"]
                dr = tr["R"] - prf["recall"]
                df = tr["F1"] - prf["f1"]
                print(f"  {'+ troi':>30} {tr['P']:>7.4f} {tr['R']:>7.4f} {tr['F1']:>7.4f}"
                      f" {tr['FP_pct']:>6.2f}% {tr['TN_pct']:>6.2f}% {tr['n_recovered']:>6}"
                      f"  {dp:>+7.4f} {dr:>+7.4f} {df:>+7.4f}")
            print()

    print(f"\nAll results written to: {out_dir}")


if __name__ == "__main__":
    main()
