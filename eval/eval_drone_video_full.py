"""
eval_drone_video_full.py — Full evaluation on drone-detection video tests.

Mirrors the 5-step structure of `eval/eval_detector.py` outputs (which
produced `docs/analysis/eval_1000_results.md`), specialised for this dataset:

  - Dataset: drone-detection video tests
        (9 drone-positive clips + 10 confuser-only clips, all under
         `datasets/drone detection video tests/rgb/`)
  - Frames: ALL (no stride)
  - RGB:    `selcom_1280` weights at imgsz=960 (= "selcom_1280@960")
  - IR:     `ir_v3b` weights at imgsz=640 on grayscale-RGB
            (cross-modal fallback, since the dataset has no real IR)
  - Classifier: sa32 in **soft-veto** mode at τ=0.95 (not argmax)
  - Patch verifier: `rgb_filter` for both modalities (the image is RGB)

Soft-veto rule:
  - If RGB has ≥1 detection: keep RGB, unless `P(reject_both) ≥ τ`.
  - If RGB missed (no detection) AND classifier trusts IR (argmax ∈ {2,3}):
        fall back to the IR (grayscale) detections.
  - Else: empty.
In plain English: "use RGB when it fires; fall back to IR-grayscale only
when RGB is silent and the classifier doesn't think the scene is empty."

Output:
  docs/analysis/eval_drone_video_results.md
"""
from __future__ import annotations
import argparse
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

from metrics import (  # noqa: E402
    SIZE_BUCKETS, classify_size, score_per_size, compute_prf, score_detections,
)
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
IR_LABEL = "ir_v3b (grayscale-RGB)"

CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
PATCH_RGB_PATH = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt"

SOFTVETO_TAU = 0.95
SCORING = "iop"
DRONE_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"


# ── Frame enumeration ───────────────────────────────────────────────

def enumerate_clips() -> dict[str, dict[str, list[dict]]]:
    """Returns {"drone": [..., ...], "birds": [...], ...} where each entry is
    a per-clip list of frame dicts.
    Frame dict: {"clip": clip_key, "stem": ..., "rgb_path": ..., "rgb_lbl": ...}
    """
    out: dict[str, list[dict]] = {}
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cat_root = DRONE_ROOT / cat
        if not cat_root.exists():
            continue
        for cdir in sorted(cat_root.iterdir()):
            if not cdir.is_dir(): continue
            img_dir = cdir / "images" / "test"
            lbl_dir = cdir / "labels" / "test"
            if not img_dir.exists(): continue
            frames = []
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                frames.append({
                    "clip": cdir.name, "category": cat,
                    "stem": p.stem, "rgb_path": p,
                    "rgb_lbl": (lbl_dir / f"{p.stem}.txt"),
                })
            out.setdefault(cat, [])
            out[cat].append({"clip": cdir.name, "frames": frames})
    return out


# ── Pipeline ────────────────────────────────────────────────────────

def soft_veto_effective_label(rgb_dets, ir_dets, probs, threshold) -> int:
    """Soft-veto rule -> effective trust label (0/1/2/3)."""
    p_reject = float(probs[0])
    argmax = int(np.argmax(probs))
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
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
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


# ── Counters ─────────────────────────────────────────────────────────

def _empty_counts():
    return {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}


def _add_per_size(into, s):
    for b in SIZE_BUCKETS:
        into[b]["tp"] += s[b]["tp"]
        into[b]["fp"] += s[b]["fp"]
        into[b]["fn"] += s[b]["fn"]


def precision(tp, fp): return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def recall(tp, fn):    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def f1(p, r):          return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def aggregate_total(counts):
    tp = sum(counts[b]["tp"] for b in counts)
    fp = sum(counts[b]["fp"] for b in counts)
    fn = sum(counts[b]["fn"] for b in counts)
    n_gt = sum(counts[b]["n_gt"] for b in counts)
    return tp, fp, fn, n_gt


# ── Main run ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    print(f"RGB: {RGB_LABEL}, IR: {IR_LABEL}, classifier: sa32 soft-veto (τ={SOFTVETO_TAU})")

    # Load models
    yolo_rgb = YOLO(str(RGB_WEIGHTS))
    yolo_ir = YOLO(str(IR_WEIGHTS))
    obj = joblib.load(str(CLASSIFIER_PATH))
    classifier = obj["model"]
    feat_cols = obj.get("features") or obj.get("feat_cols") or []
    print(f"Loaded classifier ({len(feat_cols)} features)")

    sys.path.insert(0, str(REPO / "classifier"))
    from patch_verifier import PatchVerifier
    patch_rgb = PatchVerifier(str(PATCH_RGB_PATH))
    PATCH_THR = 0.70
    det_cache = DetCache(REPO)

    # Per-clip counters (used for per-video tables).
    # Drone clips: per-clip P/R/F1 for {rgb_only, ir_grayscale, softveto}.
    # Confuser clips: per-clip FR% for {rgb_only, ir_grayscale, softveto, softveto+patch}.
    per_clip_drone: dict[str, dict] = {}   # clip -> {stage -> {tp, fp, fn, n_gt}}
    per_clip_conf: dict[str, dict] = {}    # clip -> {"cat":..., stage -> {fp_boxes, fired_frames, n_frames}}

    # Counters for each (stage). Drone-positive frames: per-size + frame-level.
    # Confuser frames: per-clip frame-level.
    drone_stages = [
        "S0_rgb", "S0_ir_grayscale",          # Step 1 base
        "S2_rgb_patch", "S2_ir_patch",        # Step 3 patch only
        "S4_clf_argmax", "S4_clf_softveto",   # Step 4 classifier
        "S4_softveto_patch",                  # Step 4 + filter
    ]
    drone_counts = {s: _empty_counts() for s in drone_stages}
    drone_frame_level = {s: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for s in drone_stages}
    drone_fired = {s: [] for s in drone_stages}  # per-frame fired booleans, in clip order
    drone_gt_present: list[bool] = []
    drone_n_frames = 0

    # Confuser counters: per-category per-stage frame-level + segment-level
    conf_stages = [
        "S0_rgb", "S0_ir_grayscale",
        "S2_rgb_patch", "S2_ir_patch",
        "S4_clf_argmax", "S4_clf_softveto", "S4_softveto_patch",
    ]
    conf_per_cat: dict[str, dict[str, dict]] = defaultdict(
        lambda: {s: {"fp_boxes": 0, "fired_frames": 0, "n_frames": 0,
                     "fired_segs": 0, "n_segs": 0} for s in conf_stages})
    conf_clip_fired: dict[tuple, list[bool]] = defaultdict(list)  # (cat, clip, stage) -> per-frame bool

    clips_by_cat = enumerate_clips()

    # Process each clip
    t0 = time.time()
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        if cat not in clips_by_cat:
            continue
        for clip_entry in clips_by_cat[cat]:
            clip = clip_entry["clip"]
            frames = clip_entry["frames"]
            is_drone = (cat == "drone")
            print(f"[{cat}/{clip}] {len(frames)} frames")

            # Per-stage in-clip booleans (for temporal voting later on drone clips)
            clip_fired_by_stage = {s: [] for s in drone_stages}
            clip_gt_in_clip: list[bool] = []

            for fr in frames:
                img = cv2.imread(str(fr["rgb_path"]))
                if img is None: continue
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gts = (read_yolo_labels(fr["rgb_lbl"], w, h, drone_classes={0})
                       if is_drone and fr["rgb_lbl"].exists() else [])

                # ── RGB inference (cached) ──
                cached = det_cache.get_dets(f"video_{cat}_{clip}", "selcom_960",
                                            RGB_WEIGHTS, RGB_IMGSZ, fr["stem"])
                if cached is not None:
                    rgb_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
                else:
                    res = yolo_rgb.predict(img, imgsz=RGB_IMGSZ, conf=RGB_CONF,
                                            device=args.device, verbose=False)
                    r0 = res[0]
                    rgb_dets = []
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs = r0.boxes.conf.cpu().numpy()
                        rgb_dets = [(tuple(map(float, b)), float(c))
                                    for b, c in zip(xyxy, confs)]
                    det_cache.put_dets(f"video_{cat}_{clip}", "selcom_960",
                                       RGB_WEIGHTS, RGB_IMGSZ, fr["stem"],
                                       [(b[0], b[1], b[2], b[3], c) for b, c in rgb_dets])

                # ── IR-grayscale inference (cached) ──
                cached = det_cache.get_dets(f"video_{cat}_{clip}", "ir_grayscale",
                                            IR_WEIGHTS, IR_IMGSZ, fr["stem"])
                if cached is not None:
                    ir_dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in cached]
                else:
                    res = yolo_ir.predict(gray3, imgsz=IR_IMGSZ, conf=IR_CONF,
                                          device=args.device, verbose=False)
                    r0 = res[0]
                    ir_dets = []
                    if r0.boxes is not None and len(r0.boxes) > 0:
                        xyxy = r0.boxes.xyxy.cpu().numpy()
                        confs = r0.boxes.conf.cpu().numpy()
                        ir_dets = [(tuple(map(float, b)), float(c))
                                   for b, c in zip(xyxy, confs)]
                    det_cache.put_dets(f"video_{cat}_{clip}", "ir_grayscale",
                                       IR_WEIGHTS, IR_IMGSZ, fr["stem"],
                                       [(b[0], b[1], b[2], b[3], c) for b, c in ir_dets])

                # ── Patch verifier (rgb_filter for both, since image is RGB) ──
                rgb_keep = rgb_dets
                if rgb_dets:
                    probs_p = patch_rgb.predict_boxes(img, [b for b, _ in rgb_dets])
                    rgb_keep = [d for d, p in zip(rgb_dets, probs_p) if p < PATCH_THR]
                ir_keep = ir_dets
                if ir_dets:
                    probs_p = patch_rgb.predict_boxes(gray3, [b for b, _ in ir_dets])
                    ir_keep = [d for d, p in zip(ir_dets, probs_p) if p < PATCH_THR]

                # ── Classifier features + predict_proba ──
                x = build_clf_features(rgb_dets, ir_dets, gray, gray, feat_cols)
                try:
                    probs = classifier.predict_proba(x)[0]
                    argmax = int(np.argmax(probs))
                except Exception:
                    probs = np.array([0.0, 0.0, 0.0, 1.0])
                    argmax = 3

                # Argmax routing (RGB-only path: kept dets per label)
                if argmax == 0:   argmax_kept = []
                elif argmax == 1: argmax_kept = rgb_dets
                elif argmax == 2: argmax_kept = ir_dets
                else:             argmax_kept = rgb_dets + ir_dets

                # Soft-veto routing
                sv_label = soft_veto_effective_label(rgb_dets, ir_dets, probs, SOFTVETO_TAU)
                if sv_label == 0:   sv_kept = []
                elif sv_label == 1: sv_kept = rgb_dets
                elif sv_label == 2: sv_kept = ir_dets
                else:               sv_kept = rgb_dets + ir_dets

                # Patch on soft-veto kept dets
                if sv_kept:
                    p_sv = patch_rgb.predict_boxes(img, [b for b, _ in sv_kept])
                    sv_kept_patch = [d for d, p in zip(sv_kept, p_sv) if p < PATCH_THR]
                else:
                    sv_kept_patch = []

                # ── Score (drone-positive: per-size + frame-level; confuser: frame-level) ──
                stage_dets = {
                    "S0_rgb": rgb_dets,
                    "S0_ir_grayscale": ir_dets,
                    "S2_rgb_patch": rgb_keep,
                    "S2_ir_patch": ir_keep,
                    "S4_clf_argmax": argmax_kept,
                    "S4_clf_softveto": sv_kept,
                    "S4_softveto_patch": sv_kept_patch,
                }
                if is_drone:
                    # Per-clip drone counters init
                    if clip not in per_clip_drone:
                        per_clip_drone[clip] = {
                            s: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0}
                            for s in drone_stages}
                    # n_gt per size
                    for s in drone_stages:
                        for g in gts:
                            drone_counts[s][classify_size(g, w, h)]["n_gt"] += 1
                            per_clip_drone[clip][s]["n_gt"] += 1
                    for s, dets in stage_dets.items():
                        ps = score_per_size(dets, gts, w, h, iop_thr=0.5)[SCORING]
                        _add_per_size(drone_counts[s], ps)
                        # Per-clip aggregate (sum across size buckets)
                        for b in ps:
                            per_clip_drone[clip][s]["tp"] += ps[b]["tp"]
                            per_clip_drone[clip][s]["fp"] += ps[b]["fp"]
                            per_clip_drone[clip][s]["fn"] += ps[b]["fn"]
                        fired = len(dets) > 0
                        has_gt = len(gts) > 0
                        # Frame-level
                        if has_gt and fired: drone_frame_level[s]["tp"] += 1
                        elif has_gt:         drone_frame_level[s]["fn"] += 1
                        elif fired:          drone_frame_level[s]["fp"] += 1
                        else:                drone_frame_level[s]["tn"] += 1
                        drone_fired[s].append(fired)
                        clip_fired_by_stage[s].append(fired)
                    clip_gt_in_clip.append(len(gts) > 0)
                    drone_gt_present.append(len(gts) > 0)
                    drone_n_frames += 1
                else:
                    # Per-clip confuser counters init
                    clip_key = (cat, clip)
                    if clip_key not in per_clip_conf:
                        per_clip_conf[clip_key] = {
                            "cat": cat,
                            **{s: {"fp_boxes": 0, "fired_frames": 0, "n_frames": 0}
                               for s in drone_stages}
                        }
                    for s, dets in stage_dets.items():
                        fired = len(dets) > 0
                        c = conf_per_cat[cat][s]
                        c["fp_boxes"] += len(dets)
                        c["fired_frames"] += int(fired)
                        c["n_frames"] += 1
                        conf_clip_fired[(cat, clip, s)].append(fired)
                        # Per-clip
                        pc = per_clip_conf[clip_key][s]
                        pc["fp_boxes"] += len(dets)
                        pc["fired_frames"] += int(fired)
                        pc["n_frames"] += 1

    det_cache.flush()
    dt = time.time() - t0
    print(f"\nProcessed {drone_n_frames} drone frames + confusers in {dt:.0f}s")

    # ── Temporal voting (2/3 segments) for drone-positive stages ──
    def seg_vote(per_frame, seg=3, k=2):
        return [sum(per_frame[i:i+seg]) >= k for i in range(0, len(per_frame), seg)]

    seg_gt = seg_vote(drone_gt_present, seg=3, k=1)  # any GT in window
    drone_temporal_metrics = {}
    drone_alert_gate_metrics = {}
    for s in drone_stages:
        sfire = seg_vote(drone_fired[s], seg=3, k=2)
        tp = fp = fn = tn = 0
        for fg, gg in zip(sfire, seg_gt):
            if gg and fg: tp += 1
            elif gg: fn += 1
            elif fg: fp += 1
            else: tn += 1
        drone_temporal_metrics[s] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                     "n_seg": len(sfire)}

    # Confuser temporal (FR%/TN% at segment level)
    conf_temporal_per_cat: dict[str, dict[str, dict]] = defaultdict(
        lambda: {s: {"fired_segs": 0, "n_segs": 0} for s in conf_stages})
    for (cat, clip, s), per_frame in conf_clip_fired.items():
        sfire = seg_vote(per_frame, seg=3, k=2)
        conf_temporal_per_cat[cat][s]["fired_segs"] += sum(sfire)
        conf_temporal_per_cat[cat][s]["n_segs"] += len(sfire)

    # ── Write doc ──
    out_path = REPO / "docs" / "analysis" / "eval_drone_video_results.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    L: list[str] = []
    L.append("# Drone-Detection Video Tests — Full Evaluation (Soft-Veto)")
    L.append("")
    L.append(f"> **Script:** `eval/eval_drone_video_full.py`")
    L.append(f"> **RGB:** {RGB_LABEL}  |  **IR:** {IR_LABEL}  |  **Patch verifier:** rgb_filter (image is RGB even for IR model)")
    L.append(f"> **Classifier:** sa32, **soft-veto** mode at τ={SOFTVETO_TAU}")
    L.append(f"> **Scoring:** IoP @ 0.5 on drone clips, frame-level FR%/TN% on confuser clips")
    L.append(f"> **Frames:** drone={drone_n_frames}, confusers={sum(c['S0_rgb']['n_frames'] for c in conf_per_cat.values())}")
    L.append("")
    L.append("## Soft-veto, in practice")
    L.append("")
    L.append("Soft-veto changes how the classifier's output is used **at decision time**. Same trained model, different rule.")
    L.append("")
    L.append("- **If RGB has at least one detection:** keep RGB, *unless* the classifier is very confident the scene contains no drone (`P(reject_both) ≥ 0.95`). This is the *fail-open* part — we don't let the classifier override an RGB det unless it's extremely confident.")
    L.append("- **If RGB missed the drone** (no detection): fall back to the IR-grayscale detector's boxes, **but only if** the classifier trusts the IR modality (argmax votes IR-only or both).")
    L.append("- **Why we chose soft-veto here:** on a fully RGB dataset the IR branch receives the same grayscale-RGB image as the RGB branch, so the classifier sees identical global features on both sides — an OOD shift versus its paired training distribution. Under standard argmax, the classifier over-rejects (votes `reject_both`) on legitimate drone frames. Soft-veto fail-open recovers those frames, lifting recall above raw RGB while still using the classifier to gate the IR-fallback (otherwise we'd be ORing both modalities and adding confuser FPs).")
    L.append("")
    L.append("---")
    L.append("")

    # Step 1 — Base Detector
    L.append("## Step 1: Base Detector Performance (raw YOLO)")
    L.append("")
    L.append(f"### Drone clips ({drone_n_frames} frames, IoP @ 0.5)")
    L.append("")
    L.append("| Model | P | R | F1 | FP% | TN% |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s, label in (("S0_rgb", RGB_LABEL), ("S0_ir_grayscale", IR_LABEL)):
        tp, fp, fn, _ = aggregate_total(drone_counts[s])
        P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
        fl = drone_frame_level[s]
        total = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
        fp_pct = 100 * fl["fp"] / total if total else 0.0
        tn_pct = 100 * fl["tn"] / total if total else 0.0
        L.append(f"| {label} | {P:.3f} | {R:.3f} | {F:.3f} | {fp_pct:.2f}% | {tn_pct:.2f}% |")
    L.append("")

    # Step 2 — Temporal
    L.append("## Step 2: Temporal Voting (2-of-3 segments)")
    L.append("")
    L.append(f"### Drone clips ({drone_n_frames} frames → {drone_temporal_metrics['S0_rgb']['n_seg']} segments)")
    L.append("")
    L.append("| Stage | TP | FP | FN | TN | P | R | F1 | FR% |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, label in (
        ("S0_rgb", f"{RGB_LABEL} + temporal"),
        ("S0_ir_grayscale", f"{IR_LABEL} + temporal"),
    ):
        m = drone_temporal_metrics[s]
        P = precision(m["tp"], m["fp"]); R = recall(m["tp"], m["fn"]); F = f1(P, R)
        nseg = m["n_seg"]
        fr_pct = 100 * (m["tp"] + m["fp"]) / nseg if nseg else 0.0
        L.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                 f"{P:.4f} | {R:.4f} | {F:.4f} | {fr_pct:.2f}% |")
    L.append("")

    # Step 3 — Patch Verifier & Alert Gate
    L.append("## Step 3: Patch Verifier (rgb_filter) & Alert Gate")
    L.append("")
    L.append("Patch verifier (`rgb_filter`, threshold 0.70) applied to each detector's boxes "
             "directly. Alert gate = temporal voting on **post-filter** firings (the production "
             "rule: only let an alert through if the patch verifier passes on the third frame).")
    L.append("")
    L.append(f"### Drone clips ({drone_n_frames} frames, {drone_temporal_metrics['S2_rgb_patch']['n_seg']} segments)")
    L.append("")
    L.append("| Stage | P | R | F1 | FP% (frame) | TN% (frame) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s, label in (
        ("S2_rgb_patch", f"{RGB_LABEL} + patch"),
        ("S2_ir_patch", f"{IR_LABEL} + patch"),
    ):
        tp, fp, fn, _ = aggregate_total(drone_counts[s])
        P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
        fl = drone_frame_level[s]
        total = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
        fp_pct = 100 * fl["fp"] / total if total else 0.0
        tn_pct = 100 * fl["tn"] / total if total else 0.0
        L.append(f"| {label} | {P:.4f} | {R:.4f} | {F:.4f} | {fp_pct:.2f}% | {tn_pct:.2f}% |")
    L.append("")
    L.append("**Alert gate** (segment-level, patch applied at decision boundary):")
    L.append("")
    L.append("| Stage | TP | FP | FN | TN | P | R | F1 | FR% |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, label in (
        ("S2_rgb_patch", f"{RGB_LABEL} + alert gate"),
        ("S2_ir_patch", f"{IR_LABEL} + alert gate"),
    ):
        m = drone_temporal_metrics[s]
        P = precision(m["tp"], m["fp"]); R = recall(m["tp"], m["fn"]); F = f1(P, R)
        nseg = m["n_seg"]
        fr_pct = 100 * (m["tp"] + m["fp"]) / nseg if nseg else 0.0
        L.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | "
                 f"{P:.4f} | {R:.4f} | {F:.4f} | {fr_pct:.2f}% |")
    L.append("")

    # Step 4 — Classifier (with soft-veto)
    L.append("## Step 4: Scene-Aware Trust Classifier (SA32, soft-veto)")
    L.append("")
    L.append(f"### Drone clips ({drone_n_frames} frames)")
    L.append("")
    L.append("| Stage | P | R | F1 | FP% (frame) | TN% (frame) | ΔF1 vs RGB |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    # Base reference
    tp0, fp0, fn0, _ = aggregate_total(drone_counts["S0_rgb"])
    F0 = f1(precision(tp0, fp0), recall(tp0, fn0))
    for s, label in (
        ("S0_rgb", RGB_LABEL),
        ("S0_ir_grayscale", IR_LABEL),
        ("S4_clf_argmax", "classifier_sa32 (argmax) — for reference"),
        ("S4_clf_softveto", f"classifier_sa32 (soft-veto τ={SOFTVETO_TAU}) ← chosen"),
        ("S4_softveto_patch", f"classifier_sa32 (soft-veto) + rgb_filter"),
    ):
        tp, fp, fn, _ = aggregate_total(drone_counts[s])
        P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
        fl = drone_frame_level[s]
        total = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
        fp_pct = 100 * fl["fp"] / total if total else 0.0
        tn_pct = 100 * fl["tn"] / total if total else 0.0
        delta = F - F0 if s.startswith("S4_") else 0.0
        delta_str = f"{delta:+.4f}" if s.startswith("S4_") else "—"
        L.append(f"| {label} | {P:.4f} | {R:.4f} | {F:.4f} | {fp_pct:.2f}% | {tn_pct:.2f}% | {delta_str} |")
    L.append("")

    # Step 5 — Per-Size
    L.append("## Step 5: Per-size detection breakdown (drone clips)")
    L.append("")
    for s, label in (
        ("S0_rgb", f"**{RGB_LABEL}** (raw RGB)"),
        ("S0_ir_grayscale", f"**{IR_LABEL}** (raw IR-grayscale)"),
        ("S4_clf_softveto", f"**Soft-veto classifier (τ={SOFTVETO_TAU})**"),
    ):
        L.append(f"#### {label}")
        L.append("")
        L.append("| Size | TP | FP | FN | n_gt | P | R | F1 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        tp_t = fp_t = fn_t = ngt_t = 0
        for b in SIZE_BUCKETS:
            c = drone_counts[s][b]
            tp = c["tp"]; fp = c["fp"]; fn = c["fn"]; n_gt = c["n_gt"]
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            tp_t += tp; fp_t += fp; fn_t += fn; ngt_t += n_gt
            L.append(f"| {b} | {tp} | {fp} | {fn} | {n_gt} | {P:.4f} | {R:.4f} | {F:.4f} |")
        P = precision(tp_t, fp_t); R = recall(tp_t, fn_t); F = f1(P, R)
        L.append(f"| **all** | **{tp_t}** | **{fp_t}** | **{fn_t}** | **{ngt_t}** | **{P:.4f}** | **{R:.4f}** | **{F:.4f}** |")
        L.append("")
    L.append("---")
    L.append("")

    # Step 6 — Confuser suppression
    L.append("## Step 6: Confuser clip suppression (no drone GT)")
    L.append("")
    L.append("On confuser-only clips every detection is an FP by construction. Reporting "
             "frame-level FR% (frames that fired) and segment-level FR% (2-of-3 fired).")
    L.append("")
    for cat in ("birds", "airplanes", "helicopters"):
        if cat not in conf_per_cat: continue
        n_clips = sum(1 for (c, _, _) in conf_clip_fired if c == cat)
        n_clips = len({clip for (c, clip, _) in conf_clip_fired if c == cat})
        n_frames = conf_per_cat[cat]["S0_rgb"]["n_frames"]
        n_segs = conf_temporal_per_cat[cat]["S0_rgb"]["n_segs"]
        L.append(f"### {cat.title()} ({n_clips} clips, {n_frames} frames, {n_segs} segments)")
        L.append("")
        L.append("| Stage | FP boxes | FR% (frame) | FR% (segment, 2/3) | TN% (segment) |")
        L.append("|---|---:|---:|---:|---:|")
        for s, label in (
            ("S0_rgb", RGB_LABEL),
            ("S0_ir_grayscale", IR_LABEL),
            ("S2_rgb_patch", f"{RGB_LABEL} + patch"),
            ("S2_ir_patch", f"{IR_LABEL} + patch"),
            ("S4_clf_argmax", "classifier_sa32 (argmax)"),
            ("S4_clf_softveto", f"classifier_sa32 (soft-veto τ={SOFTVETO_TAU})"),
            ("S4_softveto_patch", "classifier_sa32 (soft-veto) + rgb_filter"),
        ):
            c = conf_per_cat[cat][s]
            tc = conf_temporal_per_cat[cat][s]
            fr_frame = 100 * c["fired_frames"] / c["n_frames"] if c["n_frames"] else 0.0
            fr_seg = 100 * tc["fired_segs"] / tc["n_segs"] if tc["n_segs"] else 0.0
            tn_seg = 100 - fr_seg
            L.append(f"| {label} | {c['fp_boxes']} | {fr_frame:.2f}% | {fr_seg:.2f}% | {tn_seg:.2f}% |")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## Per-Video Breakdown")
    L.append("")
    L.append("Most drone clips contain birds in the scene alongside the drone "
             "(seagulls, generic flocks, attack-by-bird footage). Only "
             "`drone_takeoff_short` and `drone_takeoff_from_ground_and_not_hand_short` "
             "are clean takeoff clips. The per-clip RGB-vs-IR-grayscale split below "
             "shows where the cross-modal IR fallback actually earns its keep on "
             "these realistic mixed scenes.")
    L.append("")
    L.append("### Drone clips — F1 per clip (IoP @ 0.5)")
    L.append("")
    L.append("| Clip | Frames | n_gt | RGB F1 | IR-gray F1 | ΔF1 (gray − RGB) | Softveto F1 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for clip in sorted(per_clip_drone.keys()):
        st = per_clip_drone[clip]
        c_rgb = st["S0_rgb"]
        c_ir = st["S0_ir_grayscale"]
        c_sv = st["S4_clf_softveto"]
        n_gt = c_rgb["n_gt"]
        F_rgb = f1(precision(c_rgb["tp"], c_rgb["fp"]), recall(c_rgb["tp"], c_rgb["fn"]))
        F_ir = f1(precision(c_ir["tp"], c_ir["fp"]), recall(c_ir["tp"], c_ir["fn"]))
        F_sv = f1(precision(c_sv["tp"], c_sv["fp"]), recall(c_sv["tp"], c_sv["fn"]))
        delta = F_ir - F_rgb
        delta_str = f"{delta:+.4f}" + (" ★" if delta > 0 else "")
        # Count frames from confuser path? actually we know from clips_by_cat
        n_frames = sum(1 for e in clips_by_cat["drone"] if e["clip"] == clip
                       for _ in e["frames"])
        L.append(f"| {clip} | {n_frames} | {n_gt} | {F_rgb:.4f} | {F_ir:.4f} | {delta_str} | {F_sv:.4f} |")
    L.append("")
    L.append("★ = IR-grayscale outperforms RGB on that clip (cross-modal recovery).")
    L.append("")
    L.append("### Confuser clips — segment fire-rate per clip (lower is better)")
    L.append("")
    L.append("| Category | Clip | Frames | RGB FR% | IR-gray FR% | Δ (gray − RGB) | Softveto+patch FR% |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    # Segment-level FR% per clip
    def clip_seg_fr(per_frame):
        sfire = seg_vote(per_frame, seg=3, k=2)
        n = len(sfire)
        return 100 * sum(sfire) / n if n else 0.0
    for (cat, clip), st in sorted(per_clip_conf.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_frames = st["S0_rgb"]["n_frames"]
        fr_rgb = clip_seg_fr(conf_clip_fired.get((cat, clip, "S0_rgb"), []))
        fr_ir = clip_seg_fr(conf_clip_fired.get((cat, clip, "S0_ir_grayscale"), []))
        fr_sv_patch = clip_seg_fr(conf_clip_fired.get((cat, clip, "S4_softveto_patch"), []))
        delta = fr_ir - fr_rgb
        delta_str = f"{delta:+.2f}pp" + (" ★" if delta < 0 else "")
        L.append(f"| {cat} | {clip} | {n_frames} | {fr_rgb:.2f}% | {fr_ir:.2f}% | "
                 f"{delta_str} | {fr_sv_patch:.2f}% |")
    L.append("")
    L.append("★ = IR-grayscale fires less (better confuser suppression than RGB on that clip).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Delivered")
    L.append("")
    L.append(f"- `{out_path.relative_to(REPO).as_posix()}`")
    L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {out_path}")

    # ── Per-clip CSV dumps (used by the dashboard notebook) ──
    import csv as _csv
    csv_dir = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    drone_csv = csv_dir / "eval_drone_video_per_clip.csv"
    with drone_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["clip", "frames", "n_gt", "stage", "TP", "FP", "FN", "P", "R", "F1"])
        for clip, st in sorted(per_clip_drone.items()):
            n_frames = sum(1 for e in clips_by_cat["drone"]
                           if e["clip"] == clip for _ in e["frames"])
            for s in drone_stages:
                c = st[s]
                P = precision(c["tp"], c["fp"])
                R = recall(c["tp"], c["fn"])
                F = f1(P, R)
                w.writerow([clip, n_frames, c["n_gt"], s, c["tp"], c["fp"], c["fn"],
                            round(P, 4), round(R, 4), round(F, 4)])
    print(f"Wrote {drone_csv}")

    # Aggregate drone-stage CSV (per-stage, all sizes + each size bucket).
    # Mirrors what the .md tables show so the notebook can read selcom@960 numbers.
    agg_csv = csv_dir / "eval_drone_video_aggregate.csv"
    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["stage", "size", "TP", "FP", "FN", "n_gt", "P", "R", "F1",
                    "frame_TP", "frame_FP", "frame_FN", "frame_TN",
                    "FP_pct_frame", "TN_pct_frame", "n_frames"])
        for s in drone_stages:
            for b in list(SIZE_BUCKETS) + ["all"]:
                if b == "all":
                    tp = sum(drone_counts[s][bb]["tp"] for bb in SIZE_BUCKETS)
                    fp = sum(drone_counts[s][bb]["fp"] for bb in SIZE_BUCKETS)
                    fn = sum(drone_counts[s][bb]["fn"] for bb in SIZE_BUCKETS)
                    n_gt = sum(drone_counts[s][bb]["n_gt"] for bb in SIZE_BUCKETS)
                else:
                    c = drone_counts[s][b]
                    tp, fp, fn, n_gt = c["tp"], c["fp"], c["fn"], c["n_gt"]
                P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
                fl = drone_frame_level[s] if b == "all" else None
                if fl:
                    total = fl["tp"] + fl["fp"] + fl["fn"] + fl["tn"]
                    fpp = 100 * fl["fp"] / total if total else 0.0
                    tnp = 100 * fl["tn"] / total if total else 0.0
                    w.writerow([s, b, tp, fp, fn, n_gt,
                                round(P, 4), round(R, 4), round(F, 4),
                                fl["tp"], fl["fp"], fl["fn"], fl["tn"],
                                round(fpp, 2), round(tnp, 2), drone_n_frames])
                else:
                    w.writerow([s, b, tp, fp, fn, n_gt,
                                round(P, 4), round(R, 4), round(F, 4),
                                "", "", "", "", "", "", drone_n_frames])
        # Temporal/alert-gate (segment-level)
        for st_name in ("S0_rgb", "S0_ir_grayscale", "S2_rgb_patch", "S2_ir_patch"):
            m = drone_temporal_metrics.get(st_name)
            if not m: continue
            tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
            P = precision(tp, fp); R = recall(tp, fn); F = f1(P, R)
            kind = "temporal" if st_name.startswith("S0_") else "alert_gate"
            w.writerow([f"{kind}__{st_name}", "segment",
                        tp, fp, fn, sum([tp, fn]),
                        round(P, 4), round(R, 4), round(F, 4),
                        "", "", "", tn, "", "", m["n_seg"]])
    print(f"Wrote {agg_csv}")

    # Per-category confuser aggregate (selcom@960) — replaces the
    # drone_video_tests.csv confuser rows which were generated at imgsz=1280.
    conf_agg_csv = csv_dir / "eval_drone_video_confuser_aggregate.csv"
    with conf_agg_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["category", "stage", "n_clips", "n_frames",
                    "fp_boxes", "fired_frames", "fired_segs", "n_segs",
                    "fr_frame_pct", "fr_seg_pct", "tn_seg_pct"])
        for cat in ("birds", "airplanes", "helicopters"):
            if cat not in conf_per_cat: continue
            n_clips = len({clip for (c, clip, _) in conf_clip_fired if c == cat})
            for s in drone_stages:
                c = conf_per_cat[cat][s]
                tc = conf_temporal_per_cat[cat][s]
                fr_f = 100 * c["fired_frames"] / c["n_frames"] if c["n_frames"] else 0.0
                fr_s = 100 * tc["fired_segs"] / tc["n_segs"] if tc["n_segs"] else 0.0
                tn_s = 100 - fr_s
                w.writerow([cat, s, n_clips, c["n_frames"],
                            c["fp_boxes"], c["fired_frames"],
                            tc["fired_segs"], tc["n_segs"],
                            round(fr_f, 2), round(fr_s, 2), round(tn_s, 2)])
    print(f"Wrote {conf_agg_csv}")

    conf_csv = csv_dir / "eval_drone_video_confuser_per_clip.csv"
    with conf_csv.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["category", "clip", "frames", "stage", "fp_boxes",
                    "fired_frames", "fired_segs", "n_segs", "fr_frame_pct", "fr_seg_pct"])
        for (cat, clip), st in sorted(per_clip_conf.items()):
            for s in drone_stages:
                c = st[s]
                seg_fire = sum(seg_vote(conf_clip_fired.get((cat, clip, s), []), seg=3, k=2))
                seg_n = len(seg_vote(conf_clip_fired.get((cat, clip, s), []), seg=3, k=2))
                fr_f = 100 * c["fired_frames"] / c["n_frames"] if c["n_frames"] else 0.0
                fr_s = 100 * seg_fire / seg_n if seg_n else 0.0
                w.writerow([cat, clip, c["n_frames"], s,
                            c["fp_boxes"], c["fired_frames"], seg_fire, seg_n,
                            round(fr_f, 2), round(fr_s, 2)])
    print(f"Wrote {conf_csv}")


if __name__ == "__main__":
    main()
