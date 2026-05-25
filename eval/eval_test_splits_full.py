"""
eval_test_splits_full.py — Full pipeline eval on two test splits:

  rgb_test   = G:/drone/dataset/dataset/{images,labels}/test
               RGB-only mixed dataset (drone-positive + confuser categories).
               Pipeline: selcom_1280@960 + ir_v3b on grayscale-RGB,
                         classifier in **soft-veto** τ=0.95, rgb_filter patch.

  ir_test    = G:/drone/IR_dset_final/test/{images,labels}
               IR-only mixed dataset.
               Pipeline: ir_v3b on IR + selcom on IR-as-RGB (synthetic RGB
                         to drive the classifier's RGB branch — should route
                         to IR), classifier in **argmax**, ir_filter patch.

Outputs (under docs/analysis/full_pipeline_ablations/csv/):
  eval_<ds>_aggregate.csv  — per-stage P/R/F1 + per-size + frame-level for drone-positive subset
  eval_<ds>_confuser.csv   — per-stage FR% per confuser sub-category (drone-negative subset)

Frame classification:
  label file non-empty -> drone-positive
  label file empty     -> confuser (sub-category from filename prefix)

Usage:
  python -u eval/eval_test_splits_full.py --dataset rgb_test
  python -u eval/eval_test_splits_full.py --dataset ir_test
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import SIZE_BUCKETS, classify_size, score_per_size  # noqa: E402


def _p(tp, fp): return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def _r(tp, fn): return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def _f(p, r): return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
from datasets import read_yolo_labels  # noqa: E402
from det_cache import DetCache  # noqa: E402
from ultralytics import YOLO  # noqa: E402
import joblib  # noqa: E402
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────
RGB_WEIGHTS = REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"
RGB_IMGSZ = 960
RGB_CONF = 0.25
RGB_LABEL = "selcom_1280@960"

IR_WEIGHTS = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
IR_IMGSZ = 640
IR_CONF = 0.40
IR_LABEL = "ir_v3b"

CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
PATCH_RGB_PATH = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"
PATCH_IR_PATH  = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"
PATCH_THR = 0.70
SOFTVETO_TAU = 0.95
N_TARGET = 1000

DATASETS = {
    "rgb_test": {
        "img_dir": Path("G:/drone/dataset/dataset/images/test"),
        "lbl_dir": Path("G:/drone/dataset/dataset/labels/test"),
        "primary_modality": "rgb",
        "classifier_mode": "softveto",
        "patch": "rgb",
        "scoring": "iop",
    },
    "ir_test": {
        "img_dir": Path("G:/drone/IR_dset_final/test/images"),
        "lbl_dir": Path("G:/drone/IR_dset_final/test/labels"),
        "primary_modality": "ir",
        "classifier_mode": "argmax",
        "patch": "ir",
        "scoring": "iou",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────

def soft_veto_effective_label(rgb_dets, ir_dets, probs, threshold) -> int:
    p_reject = float(probs[0]); argmax = int(np.argmax(probs))
    if rgb_dets:
        return 0 if p_reject >= threshold else 1
    if argmax in (2, 3) and ir_dets:
        return argmax
    return 0


def build_clf_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
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
    rh, rw = rgb_gray.shape[:2]; ih, iw = ir_gray.shape[:2]
    for prefix, dets, gray, gw, gh in (
        ("rgb", rgb_dets, rgb_gray, rw, rh),
        ("ir",  ir_dets,  ir_gray,  iw, ih),
    ):
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
    return np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)


def parse_category(stem: str, fallback: str) -> str:
    """Categorise a frame by semantic content from its filename.

    For drone-positive frames the result is 'drone'.
    For drone-negative (empty-label) frames we look for the canonical confuser
    keywords first; everything else gets bucketed as 'other_negative' so
    confuser-specific FR% plots can filter cleanly."""
    if fallback == "drone":
        return "drone"
    s = stem.lower()
    if "bird" in s: return "bird"
    if "airplane" in s: return "airplane"
    if "helicopter" in s: return "helicopter"
    return "other_negative"


def list_frames(ds: dict, stride_to: int) -> tuple[list[dict], int]:
    img_dir = ds["img_dir"]; lbl_dir = ds["lbl_dir"]
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_frames = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in exts: continue
        all_frames.append({"stem": p.stem, "img_path": p,
                            "lbl_path": lbl_dir / f"{p.stem}.txt"})
    n_total = len(all_frames)
    stride = max(1, n_total // stride_to) if n_total > stride_to else 1
    return all_frames[::stride], n_total


# ── Counters ─────────────────────────────────────────────────────────

def _empty_counts(): return {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}
def _empty_frame(): return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
def _add_ps(into, s):
    for b in SIZE_BUCKETS:
        into[b]["tp"] += s[b]["tp"]; into[b]["fp"] += s[b]["fp"]; into[b]["fn"] += s[b]["fn"]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    primary = ds["primary_modality"]; clf_mode = ds["classifier_mode"]
    score_rule = ds["scoring"]; patch_kind = ds["patch"]
    print(f"Dataset {args.dataset}: primary={primary}, classifier={clf_mode}, "
          f"patch={patch_kind}, scoring={score_rule.upper()}")

    yolo_rgb = YOLO(str(RGB_WEIGHTS)); yolo_ir = YOLO(str(IR_WEIGHTS))
    obj = joblib.load(str(CLASSIFIER_PATH))
    classifier = obj["model"]; feat_cols = obj.get("features") or obj.get("feat_cols") or []
    print(f"Loaded classifier ({len(feat_cols)} features)")

    from patch_verifier import PatchVerifier
    patch_rgb = PatchVerifier(str(PATCH_RGB_PATH))
    patch_ir = PatchVerifier(str(PATCH_IR_PATH))
    patch_for_alert = patch_rgb if patch_kind == "rgb" else patch_ir

    det_cache = DetCache(REPO)

    stages = [
        "S0_rgb", "S0_ir",                          # base detectors (in their own coord)
        "S2_rgb_patch", "S2_ir_patch",              # +patch (ablation)
        "S4_clf",                                   # classifier (mode per dataset)
        "S4_clf_patch",                             # classifier + patch = alert-gate basis
        "S4_clf_other_mode",                        # ablation: opposite classifier mode
    ]
    # Drone-positive counters
    drone_counts = {s: _empty_counts() for s in stages}
    drone_frame_lvl = {s: _empty_frame() for s in stages}
    drone_fired = {s: [] for s in stages}
    drone_gt_present: list[bool] = []
    drone_n_frames = 0
    # Confuser counters: per-category per-stage frame-level + segment fired list per (cat, stage)
    conf_per_cat: dict[str, dict[str, dict]] = defaultdict(
        lambda: {s: {"fp_boxes": 0, "fired_frames": 0, "n_frames": 0} for s in stages})
    conf_clip_fired: dict[tuple, list[bool]] = defaultdict(list)  # (cat, stage) -> per-frame fired

    frames, n_total_orig = list_frames(ds, N_TARGET)
    print(f"Frames: {n_total_orig} total -> {len(frames)} sampled (stride {max(1, n_total_orig // N_TARGET)})")

    t0 = time.time()
    for i, fr in enumerate(frames):
        img = cv2.imread(str(fr["img_path"]))
        if img is None: continue
        h, w = img.shape[:2]

        # Decide what "RGB" and "IR" channels see based on dataset's primary
        if primary == "rgb":
            rgb_img = img
            rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR)
            ir_input = gray3
            ir_gray_for_clf = rgb_gray
        else:  # ir
            # IR image is the primary; selcom runs on IR-as-RGB (replicate channels)
            ir_img = img
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if ir_img.ndim == 3 else ir_img
            ir3 = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR) if ir_gray.ndim == 2 else ir_img
            rgb_img = ir3  # selcom input
            rgb_gray = ir_gray
            ir_input = ir3
            ir_gray_for_clf = ir_gray

        # GT (single GT list — coords are in the primary image frame)
        has_lbl = fr["lbl_path"].exists()
        gts = read_yolo_labels(fr["lbl_path"], w, h, drone_classes={0}) if has_lbl else []
        is_drone = bool(gts)
        category = parse_category(fr["stem"], "drone" if is_drone else "confuser")

        # ── Inference: RGB detector ──
        cached = det_cache.get_dets(args.dataset, "selcom_960",
                                    RGB_WEIGHTS, RGB_IMGSZ, fr["stem"])
        if cached is not None:
            rgb_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
        else:
            r0 = yolo_rgb.predict(rgb_img, imgsz=RGB_IMGSZ, conf=RGB_CONF,
                                   device=args.device, verbose=False)[0]
            rgb_dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy(); confs = r0.boxes.conf.cpu().numpy()
                rgb_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
            det_cache.put_dets(args.dataset, "selcom_960", RGB_WEIGHTS, RGB_IMGSZ, fr["stem"],
                                [(b[0], b[1], b[2], b[3], c) for b, c in rgb_dets])

        # ── Inference: IR detector ──
        ir_det_key = "ir_native" if primary == "ir" else "ir_grayscale"
        cached = det_cache.get_dets(args.dataset, ir_det_key, IR_WEIGHTS, IR_IMGSZ, fr["stem"])
        if cached is not None:
            ir_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
        else:
            r0 = yolo_ir.predict(ir_input, imgsz=IR_IMGSZ, conf=IR_CONF,
                                  device=args.device, verbose=False)[0]
            ir_dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy(); confs = r0.boxes.conf.cpu().numpy()
                ir_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]
            det_cache.put_dets(args.dataset, ir_det_key, IR_WEIGHTS, IR_IMGSZ, fr["stem"],
                                [(b[0], b[1], b[2], b[3], c) for b, c in ir_dets])

        # Patch verifier on each side (per-frame ablation)
        rgb_keep = rgb_dets
        if rgb_dets:
            probs_p = patch_rgb.predict_boxes(rgb_img, [b for b, _ in rgb_dets])
            rgb_keep = [d for d, p in zip(rgb_dets, probs_p) if p < PATCH_THR]
        ir_keep = ir_dets
        if ir_dets:
            patch_obj = patch_ir if primary == "ir" else patch_rgb
            probs_p = patch_obj.predict_boxes(ir_input, [b for b, _ in ir_dets])
            ir_keep = [d for d, p in zip(ir_dets, probs_p) if p < PATCH_THR]

        # Classifier features + probs
        x = build_clf_features(rgb_dets, ir_dets, rgb_gray, ir_gray_for_clf, feat_cols)
        try:
            probs = classifier.predict_proba(x)[0]; argmax = int(np.argmax(probs))
        except Exception:
            probs = np.array([0.0, 0.0, 0.0, 1.0]); argmax = 3

        def route(label):
            if label == 0: return []
            if label == 1: return rgb_dets
            if label == 2: return ir_dets
            return rgb_dets + ir_dets

        # Pipeline classifier mode (default for this dataset)
        if clf_mode == "softveto":
            clf_label = soft_veto_effective_label(rgb_dets, ir_dets, probs, SOFTVETO_TAU)
            other_mode_label = argmax  # ablation = argmax
        else:  # argmax
            clf_label = argmax
            other_mode_label = soft_veto_effective_label(rgb_dets, ir_dets, probs, SOFTVETO_TAU)

        kept_clf = route(clf_label)
        kept_clf_other = route(other_mode_label)
        # Alert-gate kept: patch on the classifier-kept dets
        if kept_clf:
            patch_obj_alert = patch_for_alert
            p2 = patch_obj_alert.predict_boxes(img, [b for b, _ in kept_clf])
            kept_clf_patch = [d for d, p in zip(kept_clf, p2) if p < PATCH_THR]
        else:
            kept_clf_patch = []

        stage_dets = {
            "S0_rgb": rgb_dets, "S0_ir": ir_dets,
            "S2_rgb_patch": rgb_keep, "S2_ir_patch": ir_keep,
            "S4_clf": kept_clf,
            "S4_clf_patch": kept_clf_patch,
            "S4_clf_other_mode": kept_clf_other,
        }

        if is_drone:
            # GT bookkeeping
            for s in stages:
                for g in gts:
                    drone_counts[s][classify_size(g, w, h)]["n_gt"] += 1
            for s, dets in stage_dets.items():
                ps = score_per_size(dets, gts, w, h, iop_thr=0.5)[score_rule]
                _add_ps(drone_counts[s], ps)
                fired = len(dets) > 0
                # Frame-level
                if fired: drone_frame_lvl[s]["tp"] += 1
                else: drone_frame_lvl[s]["fn"] += 1
                drone_fired[s].append(fired)
            drone_gt_present.append(True)
            drone_n_frames += 1
        else:
            for s, dets in stage_dets.items():
                fired = len(dets) > 0
                c = conf_per_cat[category][s]
                c["fp_boxes"] += len(dets); c["fired_frames"] += int(fired); c["n_frames"] += 1
                conf_clip_fired[(category, s)].append(fired)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0; fps = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(frames) - i - 1) / fps if fps > 0 else 0
            print(f"  {i+1}/{len(frames)}  {fps:.1f} fps  ETA {eta:.0f}s", flush=True)

    det_cache.flush()
    dt = time.time() - t0
    print(f"\nProcessed {drone_n_frames} drone + {sum(c['S0_rgb']['n_frames'] for c in conf_per_cat.values())} confuser frames in {dt:.0f}s")

    # Temporal voting (2/3 segments)
    def seg_vote(per_frame, seg=3, k=2):
        return [sum(per_frame[i:i+seg]) >= k for i in range(0, len(per_frame), seg)]

    seg_gt = seg_vote(drone_gt_present, seg=3, k=1)
    drone_temp = {}
    for s in stages:
        sfire = seg_vote(drone_fired[s], seg=3, k=2)
        tp = fp = fn = tn = 0
        for fg, gg in zip(sfire, seg_gt):
            if gg and fg: tp += 1
            elif gg: fn += 1
            elif fg: fp += 1
            else: tn += 1
        drone_temp[s] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_seg": len(sfire)}

    conf_temp_per_cat: dict[str, dict[str, dict]] = defaultdict(
        lambda: {s: {"fired_segs": 0, "n_segs": 0} for s in stages})
    for (cat, s), per_frame in conf_clip_fired.items():
        sfire = seg_vote(per_frame, seg=3, k=2)
        conf_temp_per_cat[cat][s]["fired_segs"] += sum(sfire)
        conf_temp_per_cat[cat][s]["n_segs"] += len(sfire)

    # ── Write CSVs ──
    csv_dir = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    agg = csv_dir / f"eval_{args.dataset}_aggregate.csv"
    with agg.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "size", "TP", "FP", "FN", "n_gt", "P", "R", "F1",
                    "frame_TP", "frame_FN", "n_frames",
                    "seg_TP", "seg_FP", "seg_FN", "seg_TN", "seg_FR_pct", "n_seg"])
        for s in stages:
            for b in list(SIZE_BUCKETS) + ["all"]:
                if b == "all":
                    tp = sum(drone_counts[s][bb]["tp"] for bb in SIZE_BUCKETS)
                    fp = sum(drone_counts[s][bb]["fp"] for bb in SIZE_BUCKETS)
                    fn = sum(drone_counts[s][bb]["fn"] for bb in SIZE_BUCKETS)
                    n_gt = sum(drone_counts[s][bb]["n_gt"] for bb in SIZE_BUCKETS)
                else:
                    c = drone_counts[s][b]
                    tp, fp, fn, n_gt = c["tp"], c["fp"], c["fn"], c["n_gt"]
                P = _p(tp, fp); R = _r(tp, fn); F = _f(P, R)
                if b == "all":
                    fl = drone_frame_lvl[s]
                    m = drone_temp[s]
                    nseg = m["n_seg"]
                    fr_seg = 100 * (m["tp"] + m["fp"]) / nseg if nseg else 0.0
                    w.writerow([s, b, tp, fp, fn, n_gt,
                                round(P, 4), round(R, 4), round(F, 4),
                                fl["tp"], fl["fn"], drone_n_frames,
                                m["tp"], m["fp"], m["fn"], m["tn"],
                                round(fr_seg, 2), nseg])
                else:
                    w.writerow([s, b, tp, fp, fn, n_gt,
                                round(P, 4), round(R, 4), round(F, 4),
                                "", "", drone_n_frames, "", "", "", "", "", ""])
    print(f"Wrote {agg}")

    conf = csv_dir / f"eval_{args.dataset}_confuser.csv"
    with conf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "stage", "n_frames", "fp_boxes",
                    "fired_frames", "fr_frame_pct",
                    "fired_segs", "n_segs", "fr_seg_pct", "tn_seg_pct"])
        for cat in sorted(conf_per_cat.keys()):
            for s in stages:
                c = conf_per_cat[cat][s]
                tc = conf_temp_per_cat[cat][s]
                fr_f = 100 * c["fired_frames"] / c["n_frames"] if c["n_frames"] else 0.0
                fr_s = 100 * tc["fired_segs"] / tc["n_segs"] if tc["n_segs"] else 0.0
                w.writerow([cat, s, c["n_frames"], c["fp_boxes"],
                            c["fired_frames"], round(fr_f, 2),
                            tc["fired_segs"], tc["n_segs"],
                            round(fr_s, 2), round(100 - fr_s, 2)])
    print(f"Wrote {conf}")


if __name__ == "__main__":
    main()
