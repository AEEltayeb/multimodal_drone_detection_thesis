"""
eval_full_pipeline_persize.py — Full cascade × per-size on every dataset.

For each (dataset, detector, classifier), evaluates the stack:
  S0 detector only
  S1 + classifier (sa32 / control40 / fnfn)
  S2 + classifier + patch verifier
  S3 + patch verifier only (no classifier)

Patch filter rule:
  RGB detectors        -> confuser_filter4_rgb_v2_backup.pt
  IR detector on IR    -> confuser_filter4_ir_v2_backup.pt
  IR detector on gray  -> confuser_filter4_rgb_v2_backup.pt  (RGB filter applied to grayscale-RGB input)

Output:
  eval/results/full_pipeline_persize/<dataset>/<detector>/<classifier>/summary.csv
      Rows: stage, size_bucket, TP, FP, FN, n_gt, n_frames, P, R, F1, FPPI
  eval/results/full_pipeline_persize/manifest.json — tracks completed (dataset, detector, classifier) tuples for --resume

Built to run overnight. Skips combos whose summary.csv already exists.

Usage:
  python eval/eval_full_pipeline_persize.py
  python eval/eval_full_pipeline_persize.py --datasets svanstrom selcom_val
  python eval/eval_full_pipeline_persize.py --classifiers sa32
  python eval/eval_full_pipeline_persize.py --redo                  # ignore existing
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np
import joblib
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import SIZE_BUCKETS, classify_size, score_per_size, iou_iop  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402
from patch_verifier import PatchVerifier  # noqa: E402
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES  # noqa: E402
from det_cache import DetCache  # noqa: E402


# ── Catalogue ────────────────────────────────────────────────────────

RGB_MODELS = {
    "baseline":       (REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt", 1280),
    "hardneg_v3more": (REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt", 1280),
    "retrained_v2":   (REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt", 1280),
    "selcom_1280":    (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 1280),
    "selcom_960":     (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 960),
    "selcom_640":     (REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt", 640),
}

IR_WEIGHTS = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

# Detector configurations: (key, weights, imgsz, modality, patch_filter)
# modality: "rgb" | "ir" | "ir_grayscale" (IR weights on grayscale-RGB input)
DETECTORS = []
for k, (w, sz) in RGB_MODELS.items():
    DETECTORS.append((k, w, sz, "rgb", "rgb_filter"))
DETECTORS.append(("ir_model", IR_WEIGHTS, 640, "ir", "ir_filter"))
DETECTORS.append(("ir_grayscale", IR_WEIGHTS, 640, "ir_grayscale", "rgb_filter"))

CLASSIFIERS = {
    "sa32":      REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib",
    "control40": REPO / "classifier" / "fusion_models" / "control_v3more_40feat" / "model.joblib",
    "fnfn":      REPO / "classifier" / "runs" / "reliability" / "fusion" / "fusion_no_fn_model_v1.1.joblib",
}

PATCH_RGB = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
PATCH_IR  = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"

# Datasets — modality determines image source; has_drone_gt controls scoring.
# Each entry: dict(key, type, modality, root, ...)
DATASETS = [
    {
        "key": "svanstrom", "type": "paired", "modality": "rgb",
        "img_dir": Path("G:/drone/svanstrom_paired/RGB/images"),
        "lbl_dir": Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_img_dir": Path("G:/drone/svanstrom_paired/IR/images"),
        "ir_lbl_dir": Path("G:/drone/svanstrom_paired/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "has_drone_gt": True, "drone_class": 0,
        "category_from_name": True,  # split BIRD/AIRPLANE/HELI by stem
        "is_sequence": True,  # Svanstrom is sequential video frames
        "scoring": "iop",  # GT boxes larger than drone -> IoU under-counts
    },
    {
        "key": "antiuav", "type": "paired", "modality": "rgb",
        "img_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "lbl_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_img_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "ir_lbl_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "has_drone_gt": True, "drone_class": 0,
        "is_sequence": True,  # Anti-UAV is sequential video frames
        "scoring": "iou",  # clean paired benchmark, IoU @ 0.5
    },
    {
        "key": "selcom_val", "type": "image", "modality": "rgb",
        "img_dir": Path("G:/drone/_finetune_selcom_mixed_ft2/images/val"),
        "lbl_dir": Path("G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
        "has_drone_gt": True, "drone_class": 0,
        "is_sequence": True,  # selcom val is CCTV footage frames
    },
    # Roboflow OOD: drone + 3 confuser categories (RGB only)
    {
        "key": "roboflow_rgb_drone_test", "type": "image", "modality": "rgb",
        "img_dir": Path("G:/drone/roboflow_eval/rgb_drone/test/images"),
        "lbl_dir": Path("G:/drone/roboflow_eval/rgb_drone/test/labels"),
        "has_drone_gt": True, "drone_class": 0,
    },
    {
        "key": "roboflow_rgb_bird_test", "type": "image", "modality": "rgb",
        "img_dir": Path("G:/drone/roboflow_eval/rgb_bird/test/images"),
        "lbl_dir": Path("G:/drone/roboflow_eval/rgb_bird/test/labels"),
        "has_drone_gt": False,  # negatives-only
    },
    {
        "key": "roboflow_rgb_airplane_test", "type": "image", "modality": "rgb",
        "img_dir": Path("G:/drone/roboflow_eval/rgb_airplane/test/images"),
        "lbl_dir": Path("G:/drone/roboflow_eval/rgb_airplane/test/labels"),
        "has_drone_gt": False,
    },
    {
        "key": "roboflow_rgb_helicopter_test", "type": "image", "modality": "rgb",
        "img_dir": Path("G:/drone/roboflow_eval/rgb_helicopter/test/images"),
        "lbl_dir": Path("G:/drone/roboflow_eval/rgb_helicopter/test/labels"),
        "has_drone_gt": False,
    },
    # Real-video clip family — enumerated dynamically below
]


def enumerate_video_clips():
    root = REPO / "datasets" / "drone detection video tests" / "rgb"
    out = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        croot = root / cat
        if not croot.exists():
            continue
        for cdir in sorted(croot.iterdir()):
            if not cdir.is_dir(): continue
            img_split = next((d for d in (cdir/"images"/"test", cdir/"images"/"train", cdir/"images") if d.exists()), None)
            lbl_split = next((d for d in (cdir/"labels"/"test", cdir/"labels"/"train", cdir/"labels") if d.exists()), None)
            if not img_split or not lbl_split:
                continue
            out.append({
                "key": f"video_{cat}_{cdir.name}",
                "type": "image", "modality": "rgb",
                "img_dir": img_split, "lbl_dir": lbl_split,
                "has_drone_gt": (cat == "drone"), "drone_class": 0,
            })
    return out


DATASETS.extend(enumerate_video_clips())


# ── Auto-stride ──────────────────────────────────────────────────────

def auto_stride(n: int, cap: int = 5000, floor: int = 2000) -> int:
    if n < floor: return 1
    return max(1, -(-n // cap))


# ── Patch verifier cache ─────────────────────────────────────────────

_PATCH_CACHE: dict[str, PatchVerifier] = {}

def get_patch(name: str) -> PatchVerifier:
    if name in _PATCH_CACHE:
        return _PATCH_CACHE[name]
    if name == "rgb_filter":
        _PATCH_CACHE[name] = PatchVerifier(str(PATCH_RGB))
    elif name == "ir_filter":
        _PATCH_CACHE[name] = PatchVerifier(str(PATCH_IR))
    else:
        raise ValueError(name)
    return _PATCH_CACHE[name]


# ── Classifier feature build (mirrors eval_pipeline_video_tests.py) ─

def build_clf_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
    feats = {}
    for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": float(max(confs)),
                          f"{prefix}_mean_conf": float(np.mean(confs))})
    feats.update({f"rgb_{k}": v for k, v in compute_global_features(rgb_gray).items()})
    feats.update({f"ir_{k}": v for k, v in compute_global_features(ir_gray).items()})
    for prefix, dets, gray, gw, gh in (
        ("rgb", rgb_dets, rgb_gray, rw, rh),
        ("ir", ir_dets, ir_gray, iw, ih),
    ):
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    return np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)


def load_classifier(path: Path):
    obj = joblib.load(str(path))
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feat_cols = obj.get("features") or obj.get("feat_cols") or []
    else:
        # Raw classifier — look for sibling metrics json
        model = obj
        meta_path = path.parent / f"{path.stem}_metrics.json"
        if not meta_path.exists():
            meta_path = path.parent / f"{path.stem.replace('_model', '')}_metrics.json"
        feat_cols = []
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                feat_cols = meta.get("features") or meta.get("feat_cols") or []
            except Exception:
                pass
    if not feat_cols:
        raise RuntimeError(f"No feature list found for {path}")
    return model, feat_cols


# ── Stage scoring ────────────────────────────────────────────────────

def precision(tp, fp): return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def recall(tp, fn):    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def f1(p, r):          return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def trust_label_to_dets(label, rgb_dets, ir_dets):
    if label == 0: return []
    if label == 1: return rgb_dets
    if label == 2: return ir_dets
    return rgb_dets + ir_dets


def trust_label_to_dets_for_modality(label, my_dets, other_dets):
    """RGB-GT-only scoring: return my_dets if classifier trusts my modality
    (label 1 or 3) else []. Avoids the cross-modality coord mismatch where
    IR boxes were scored against RGB GT."""
    if label == 0: return []
    if label == 1: return my_dets   # trust RGB only -> my (RGB) dets
    if label == 2: return []        # trust IR only -> RGB side rejected
    if label == 3: return my_dets   # trust both -> my dets count toward my GT
    return my_dets


# ── Frame iteration ──────────────────────────────────────────────────

def list_frames(ds):
    """Returns list of (stem, rgb_img_path, ir_img_path_or_None, rgb_lbl_path, ir_lbl_path_or_None)."""
    img_dir = ds["img_dir"]
    lbl_dir = ds["lbl_dir"]
    if not img_dir.exists():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    out = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts: continue
        stem = p.stem
        rgb_lbl = lbl_dir / f"{stem}.txt"
        ir_img = None; ir_lbl = None
        if ds["type"] == "paired":
            ir_stem = stem
            if ds.get("rgb_suffix") and ds.get("ir_suffix"):
                ir_stem = stem.replace(ds["rgb_suffix"], ds["ir_suffix"])
            for ext in exts:
                cand = ds["ir_img_dir"] / f"{ir_stem}{ext}"
                if cand.exists():
                    ir_img = cand; break
            ir_lbl = ds["ir_lbl_dir"] / f"{ir_stem}.txt"
        out.append((stem, p, ir_img, rgb_lbl, ir_lbl))
    return out


# ── Main per-combo eval ──────────────────────────────────────────────

def run_combo(ds, det_key, det_weights, det_imgsz, det_modality, det_patch,
              classifier_key, classifier, feat_cols, ir_yolo, out_root,
              conf, patch_thr, ir_conf, device, det_cache_obj=None):
    out_dir = out_root / ds["key"] / det_key / (classifier_key or "no_classifier")
    summary_path = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset scoring rule (iou | iop). Default iop preserves prior behavior.
    score_rule = ds.get("scoring", "iop")

    # If we want patch verifier suppression: get patch for this detector
    patch_verifier = get_patch(det_patch) if det_patch else None

    yolo = YOLO(str(det_weights))

    frames = list_frames(ds)
    if not frames:
        print(f"  SKIP empty dataset: {ds['key']}")
        return
    stride = auto_stride(len(frames))
    if stride > 1:
        frames = frames[::stride]

    has_gt = ds["has_drone_gt"]
    drone_cls = {ds.get("drone_class", 0)}

    # Stage counters: {stage: {size: {tp,fp,fn,n_gt}}}
    stages = ["S0_detector", "S1_+classifier", "S2_+classifier+patch", "S3_+patch_only"]
    counts = {st: {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS} for st in stages}

    # Temporal logic: only meaningful on sequence (video) datasets.
    # Video clips + Anti-UAV + Svanstrom + Selcom val are all sequential.
    is_sequence = ds.get("is_sequence", False) or ds["key"].startswith("video_")
    # Per-frame booleans for segment voting later
    fired_raw = []        # any det survived from detector
    fired_post_filter = [] # any det survived patch filter
    gt_present = []        # frame has at least one GT drone

    n_frames = 0
    t0 = time.time()
    n_total = len(frames)
    combo_tag = f"{ds['key']}/{det_key}/{classifier_key or 'no_classifier'}"
    print(f"  [{combo_tag}] {n_total} frames begin")

    for stem, rgb_path, ir_path, rgb_lbl, ir_lbl in frames:
        img = cv2.imread(str(rgb_path))
        if img is None: continue
        h, w = img.shape[:2]
        # GT
        gts = read_yolo_labels(rgb_lbl, w, h, drone_classes=drone_cls) if has_gt else []
        for g in gts:
            for st in stages:
                counts[st][classify_size(g, w, h)]["n_gt"] += 1

        # ── Detector inference (cached when possible) ──
        if det_modality == "ir_grayscale":
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            inp = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        elif det_modality == "ir" and ir_path is not None:
            ir_img = cv2.imread(str(ir_path))
            inp = ir_img if ir_img is not None else img
        else:
            inp = img

        # Try cache first
        cached = det_cache_obj.get_dets(ds["key"], det_key, det_weights, det_imgsz, stem,
                                         ir_weights_path=IR_WEIGHTS if ds["type"] == "paired" else None)
        if cached is not None:
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
        else:
            this_conf = ir_conf if det_modality.startswith("ir") else conf
            res = yolo.predict(inp, imgsz=det_imgsz, conf=this_conf,
                               device=device, verbose=False)
            r0 = res[0]
            dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
            # Save to self-managed cache for future runs
            flat = [(b[0], b[1], b[2], b[3], c) for (b, c) in dets]
            det_cache_obj.put_dets(ds["key"], det_key, det_weights, det_imgsz, stem, flat)

        # S0: detector only
        s0 = score_per_size(dets, gts, w, h, iop_thr=0.5)["iop"]
        for b in SIZE_BUCKETS:
            counts["S0_detector"][b]["tp"] += s0[b]["tp"]
            counts["S0_detector"][b]["fp"] += s0[b]["fp"]
            counts["S0_detector"][b]["fn"] += s0[b]["fn"]

        # Classifier path needs IR features too. If we have IR detector + paired IR image, use it.
        # Otherwise, use grayscale of the RGB image as a stand-in.
        if classifier is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Classifier needs BOTH RGB and IR features.
            # Paired: run IR YOLO on IR image.
            # RGB-only: run IR YOLO on grayscale-RGB (same coord frame as RGB image).
            if ds["type"] == "paired" and ir_path is not None and ir_yolo is not None:
                ir_im = cv2.imread(str(ir_path))
                ir_gray = cv2.cvtColor(ir_im, cv2.COLOR_BGR2GRAY) if ir_im is not None else gray
                ir_input = ir_im if ir_im is not None else inp
                ir_coords_match_rgb = False  # IR camera != RGB camera
            else:
                # RGB-only dataset: IR side comes from ir_grayscale path
                ir_gray = gray
                ir_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ir_coords_match_rgb = True   # same image, just channel-flipped
            # Cache IR-side dets. Use ds-specific keys:
            #   paired:   (ds, "ir_native", 640)
            #   rgb-only: (ds, "ir_grayscale_side", 640)
            ir_cache_key = "ir_native" if ir_coords_match_rgb is False else "ir_grayscale_side"
            ir_cached = (det_cache_obj.get_dets(ds["key"], ir_cache_key, IR_WEIGHTS, 640, stem)
                         if det_cache_obj else None)
            if ir_cached is not None:
                ir_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in ir_cached]
            else:
                ir_res = ir_yolo.predict(ir_input, imgsz=640, conf=ir_conf,
                                         device=device, verbose=False) if ir_yolo else None
                ir_dets = []
                if ir_res:
                    ir_boxes = ir_res[0].boxes
                    if ir_boxes is not None and len(ir_boxes) > 0:
                        xyxy = ir_boxes.xyxy.cpu().numpy()
                        confs = ir_boxes.conf.cpu().numpy()
                        ir_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
                if det_cache_obj:
                    flat = [(b[0], b[1], b[2], b[3], c) for (b, c) in ir_dets]
                    det_cache_obj.put_dets(ds["key"], ir_cache_key, IR_WEIGHTS, 640, stem, flat)

            # Classifier features (always uses RGB + IR-side, never empty IR)
            rgb_dets = dets if det_modality.startswith("rgb") or det_modality == "ir_grayscale" else []
            x = build_clf_features(rgb_dets, ir_dets, gray, ir_gray, feat_cols)
            try:
                label = int(classifier.predict(x)[0])
            except Exception:
                label = 3

            # ── BBox-level S1: "any trust = detection" semantics ──
            # If classifier rejects (label=0), no dets pass.
            # Otherwise the alert fires; pass ALL dets we have access to:
            #   - RGB-only datasets (coords match): union of rgb_dets + ir_grayscale_dets
            #   - Paired (coords differ): rgb_dets only when RGB is trusted (label∈{1,3});
            #     label=2 (IR-only-trust on paired) is a real alert but IR coords don't
            #     map to RGB GT, so RGB-side scoring sees no boxes (alert tracked via S1b).
            if label == 0:
                kept_dets = []
            elif ir_coords_match_rgb:
                kept_dets = rgb_dets + ir_dets   # any trust → union (matches user spec)
            else:  # paired
                kept_dets = rgb_dets if label in (1, 3) else []

            s1 = score_per_size(kept_dets, gts, w, h, iop_thr=0.5)[score_rule]
            for b in SIZE_BUCKETS:
                counts["S1_+classifier"][b]["tp"] += s1[b]["tp"]
                counts["S1_+classifier"][b]["fp"] += s1[b]["fp"]
                counts["S1_+classifier"][b]["fn"] += s1[b]["fn"]

            # ── Frame-level S1b: alert-fired vs GT-present ──
            fired = (label != 0) and (len(kept_dets) > 0 or (label == 2 and not ir_coords_match_rgb and len(ir_dets) > 0))
            frame_has_gt = len(gts) > 0   # local — DO NOT shadow outer has_gt
            # Bucket by largest GT box size when present (frame-level metric per-size approximation)
            frame_bucket = "all"
            if frame_has_gt:
                largest = max(gts, key=lambda g: (g[2]-g[0])*(g[3]-g[1]))
                frame_bucket = classify_size(largest, w, h)
            if frame_has_gt and fired:
                counts.setdefault("S1b_classifier_frame", {b: {"tp":0,"fp":0,"fn":0,"tn":0,"n_gt":0} for b in list(SIZE_BUCKETS)+["all"]})
                counts["S1b_classifier_frame"][frame_bucket].setdefault("tn", 0)
                counts["S1b_classifier_frame"][frame_bucket]["tp"] += 1
                counts["S1b_classifier_frame"][frame_bucket]["n_gt"] += 1
            elif frame_has_gt and not fired:
                counts.setdefault("S1b_classifier_frame", {b: {"tp":0,"fp":0,"fn":0,"tn":0,"n_gt":0} for b in list(SIZE_BUCKETS)+["all"]})
                counts["S1b_classifier_frame"][frame_bucket]["fn"] += 1
                counts["S1b_classifier_frame"][frame_bucket]["n_gt"] += 1
            elif (not frame_has_gt) and fired:
                counts.setdefault("S1b_classifier_frame", {b: {"tp":0,"fp":0,"fn":0,"tn":0,"n_gt":0} for b in list(SIZE_BUCKETS)+["all"]})
                counts["S1b_classifier_frame"]["all"]["fp"] += 1
            else:
                counts.setdefault("S1b_classifier_frame", {b: {"tp":0,"fp":0,"fn":0,"tn":0,"n_gt":0} for b in list(SIZE_BUCKETS)+["all"]})
                counts["S1b_classifier_frame"]["all"]["tn"] += 1

            # ── S2: + patch on kept_dets ──
            if patch_verifier is not None and kept_dets:
                boxes = [d[0] for d in kept_dets]
                probs = patch_verifier.predict_boxes(inp, boxes)
                kept_after_patch = [d for d, p in zip(kept_dets, probs) if p < patch_thr]
            else:
                kept_after_patch = kept_dets
            s2 = score_per_size(kept_after_patch, gts, w, h, iop_thr=0.5)[score_rule]
            for b in SIZE_BUCKETS:
                counts["S2_+classifier+patch"][b]["tp"] += s2[b]["tp"]
                counts["S2_+classifier+patch"][b]["fp"] += s2[b]["fp"]
                counts["S2_+classifier+patch"][b]["fn"] += s2[b]["fn"]

        # S3: + patch only (no classifier)
        if patch_verifier is not None and dets:
            boxes = [d[0] for d in dets]
            probs = patch_verifier.predict_boxes(inp, boxes)
            dets_patch_only = [d for d, p in zip(dets, probs) if p < patch_thr]
        else:
            dets_patch_only = dets
        s3 = score_per_size(dets_patch_only, gts, w, h, iop_thr=0.5)[score_rule]
        for b in SIZE_BUCKETS:
            counts["S3_+patch_only"][b]["tp"] += s3[b]["tp"]
            counts["S3_+patch_only"][b]["fp"] += s3[b]["fp"]
            counts["S3_+patch_only"][b]["fn"] += s3[b]["fn"]

        # Track per-frame booleans for temporal voting (sequence data only)
        if is_sequence:
            fired_raw.append(len(dets) > 0)
            fired_post_filter.append(len(dets_patch_only) > 0)
            gt_present.append(len(gts) > 0)

        n_frames += 1
        if n_frames % 200 == 0:
            elapsed = time.time() - t0
            fps = n_frames / elapsed if elapsed > 0 else 0
            eta = (n_total - n_frames) / fps if fps > 0 else 0
            print(f"  [{combo_tag}] {n_frames}/{n_total}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    dt = time.time() - t0

    # --- Segment-level temporal scoring (S4, S5) ---
    seg_rows = []
    if is_sequence and n_frames > 0:
        SEG = 3
        def seg_vote(per_frame_bool):
            """Return list of (segment_fired) booleans, 2-of-3 voting over SEG-frame windows."""
            out = []
            for i in range(0, len(per_frame_bool), SEG):
                w_ = per_frame_bool[i:i+SEG]
                fired = sum(1 for x in w_ if x) >= 2
                out.append(fired)
            return out
        def seg_gt(per_frame_gt):
            return [any(per_frame_gt[i:i+SEG]) for i in range(0, len(per_frame_gt), SEG)]
        gtl = seg_gt(gt_present)
        for st_name, fire_list in (
            ("S4_temporal_no_filter", fired_raw),
            ("S5_alert_gate_filter", fired_post_filter),
        ):
            sfire = seg_vote(fire_list)
            tp = fp = fn = tn = 0
            for fg, gt in zip(sfire, gtl):
                if gt and fg: tp += 1
                elif gt and not fg: fn += 1
                elif (not gt) and fg: fp += 1
                else: tn += 1
            n_seg = len(sfire)
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            seg_rows.append({
                "dataset": ds["key"], "detector": det_key, "classifier": classifier_key or "none",
                "stage": st_name, "size_bucket": "all", "scoring": score_rule,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn, "n_gt": sum(gtl), "n_frames": n_seg,
                "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
                "fppi": round(fp / n_seg, 4) if n_seg else 0.0,
            })

    # Write summary
    rows = []
    for st in stages:
        for b in SIZE_BUCKETS:
            c = counts[st][b]
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            rows.append({
                "dataset": ds["key"], "detector": det_key, "classifier": classifier_key or "none",
                "stage": st, "size_bucket": b, "scoring": score_rule,
                "TP": tp, "FP": fp, "FN": fn, "TN": "", "n_gt": c["n_gt"], "n_frames": n_frames,
                "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
                "fppi": round(fp / n_frames, 4) if n_frames else 0.0,
            })
    rows.extend(seg_rows)
    # ── Ledger regression check: S0_detector must reproduce Phase 2 results.json if available ──
    p2_paths = {
        ("antiuav", det_key):    REPO / "eval" / "results" / "antiuav_per_model" / det_key / f"{det_key}_results.json",
        ("selcom_val", det_key): REPO / "eval" / "results" / "selcom_val_holdout" / det_key / f"{det_key}_results.json",
    }
    p2_path = p2_paths.get((ds["key"], det_key))
    if p2_path and p2_path.exists():
        try:
            p2 = json.loads(p2_path.read_text())
            dm = p2.get("detection_metrics", [])
            expected = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
            actual_tp = sum(counts["S0_detector"][b]["tp"] for b in SIZE_BUCKETS)
            actual_fp = sum(counts["S0_detector"][b]["fp"] for b in SIZE_BUCKETS)
            actual_fn = sum(counts["S0_detector"][b]["fn"] for b in SIZE_BUCKETS)
            exp = (expected.get("TP",0), expected.get("FP",0), expected.get("FN",0))
            got = (actual_tp, actual_fp, actual_fn)
            if exp != got:
                print(f"  ⚠️ REGRESSION {ds['key']}/{det_key}: ledger says TP/FP/FN={exp}, got {got}")
            else:
                print(f"  ✓ {ds['key']}/{det_key} matches ledger ({exp})")
        except Exception as e:
            print(f"  (could not verify against ledger: {e})")

    # Use widest fieldnames (segment rows may have TN that the per-size rows lack)
    fieldnames = list({k for r in rows for k in r.keys()})
    # Stable order
    preferred = ["dataset","detector","classifier","stage","size_bucket","scoring",
                 "TP","FP","FN","TN","n_gt","n_frames",
                 "precision","recall","f1","fppi"]
    fieldnames = [k for k in preferred if k in fieldnames] + [k for k in fieldnames if k not in preferred]
    with summary_path.open("w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=fieldnames)
        w_.writeheader()
        w_.writerows(rows)
    print(f"  -> {ds['key']}/{det_key}/{classifier_key or 'no_classifier'}  {n_frames}f in {dt:.0f}s")
    if det_cache_obj is not None:
        n_flushed = det_cache_obj.flush()
        if n_flushed:
            print(f"     flushed {n_flushed} cache file(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--classifiers", nargs="*", default=list(CLASSIFIERS.keys()) + [None])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--output-dir", type=str,
                    default=str(REPO / "docs" / "analysis" /
                                "full_pipeline_ablations" / "raw_results"))
    ap.add_argument("--redo", action="store_true", help="Re-run combos with existing summary.csv")
    args = ap.parse_args()

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Resolve dataset & detector subsets
    datasets = DATASETS if not args.datasets else [d for d in DATASETS if d["key"] in args.datasets]
    detectors = DETECTORS if not args.detectors else [d for d in DETECTORS if d[0] in args.detectors]

    # Pre-load classifier objects once
    loaded_clfs: dict[str, tuple] = {}
    for ck in args.classifiers:
        if ck is None or ck == "none" or ck == "no_classifier":
            loaded_clfs[None] = (None, None); continue
        if ck not in CLASSIFIERS:
            print(f"  SKIP unknown classifier {ck}")
            continue
        cpath = CLASSIFIERS[ck]
        if not cpath.exists():
            print(f"  SKIP missing classifier weights {cpath}")
            continue
        try:
            model, feats = load_classifier(cpath)
            loaded_clfs[ck] = (model, feats)
            print(f"  Loaded classifier {ck}: {len(feats)} features")
        except Exception as e:
            print(f"  FAIL load {ck}: {e}")

    ir_yolo = YOLO(str(IR_WEIGHTS)) if IR_WEIGHTS.exists() else None
    det_cache_obj = DetCache(REPO)
    print(f"Datasets: {len(datasets)}   Detectors: {len(detectors)}   Classifier combos: {len(loaded_clfs)}")
    total = len(datasets) * len(detectors) * len(loaded_clfs)
    print(f"Total combos: {total}")

    done = skipped = 0
    for ds in datasets:
        for det_key, det_w, det_imgsz, det_mod, det_patch in detectors:
            # Run IR detector on paired datasets (which have a real IR image dir)
            # but not on RGB-only image datasets.
            if det_mod == "ir" and ds["type"] != "paired":
                continue
            # ir_grayscale is OK on RGB datasets (that's its whole point)
            if not det_w.exists():
                print(f"  SKIP missing weights for {det_key}")
                continue
            for clf_key, (clf, feats) in loaded_clfs.items():
                out_dir = out_root / ds["key"] / det_key / (clf_key or "no_classifier")
                if (out_dir / "summary.csv").exists() and not args.redo:
                    skipped += 1
                    continue
                try:
                    run_combo(ds, det_key, det_w, det_imgsz, det_mod, det_patch,
                              clf_key, clf, feats, ir_yolo, out_root,
                              args.conf, args.patch_thr, args.ir_conf, args.device,
                              det_cache_obj=det_cache_obj)
                    done += 1
                except Exception as e:
                    print(f"  FAILED {ds['key']}/{det_key}/{clf_key}: {e}")
    print(f"\nDone {done}, skipped {skipped} (already exist)")
    print(f"Cache stats: {det_cache_obj.stats()}")


if __name__ == "__main__":
    main()
