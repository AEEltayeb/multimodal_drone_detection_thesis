"""
compare_grayscale_clf_modes.py — Compare classifier grayscale feature strategies.

Three modes for IR-side global features when running in grayscale fallback:
  A) baseline:   compute globals from grayscale RGB (current behavior — identical to RGB globals)
  B) mean_fill:  replace IR globals with real IR training-set means (from _TRAIN_MEANS)
  C) zero_scene: zero out ALL scene globals (both RGB and IR), keep only det+target features

Runs on video test drone clips with baseline RGB detector. Reports per-mode:
  - trust label distribution (how often the classifier says trust_neither)
  - drone TP/FP/FN and F1 at the classifier stage (trust-aware scoring)
"""
from __future__ import annotations
import sys, time, csv, json
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import joblib
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import score_detections, score_per_size, SIZE_BUCKETS
from datasets import read_yolo_labels
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES, _TRAIN_MEANS

# ── Paths ──
RGB_WEIGHTS = REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"
IR_WEIGHTS  = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
CLF_PATH    = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"

# ── Load classifier ──
def load_clf():
    obj = joblib.load(str(CLF_PATH))
    model = obj["model"]
    feat_cols = obj.get("features") or obj.get("feat_cols") or []
    return model, feat_cols

# ── Feature builder with mode control ──
def build_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols, mode="baseline"):
    """
    mode:
      baseline  — current behavior (ir globals from grayscale RGB)
      mean_fill — ir globals replaced with _TRAIN_MEANS["ir"]
      zero_scene — all scene globals zeroed for both modalities
    """
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
    feats = {}

    # Detection confidence features (unchanged across modes)
    for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": float(max(confs)),
                          f"{prefix}_mean_conf": float(np.mean(confs))})

    # Global scene features — mode-dependent
    rgb_globals = compute_global_features(rgb_gray)
    if mode == "baseline":
        ir_globals = compute_global_features(ir_gray)  # same as rgb_gray in grayscale mode
    elif mode == "mean_fill":
        ir_globals = dict(_TRAIN_MEANS["ir"])  # real IR training means
    elif mode == "zero_scene":
        rgb_globals = {k: 0.0 for k in rgb_globals}
        ir_globals = {k: 0.0 for k in rgb_globals}
    else:
        raise ValueError(mode)

    if mode == "zero_scene":
        feats.update({f"rgb_{k}": 0.0 for k in compute_global_features(rgb_gray)})
        feats.update({f"ir_{k}": 0.0 for k in compute_global_features(ir_gray)})
    else:
        feats.update({f"rgb_{k}": v for k, v in rgb_globals.items()})
        feats.update({f"ir_{k}": v for k, v in ir_globals.items()})

    # Target features (unchanged — from detection bboxes)
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500, help="Max frames total")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model, feat_cols = load_clf()
    print(f"Classifier: {len(feat_cols)} features")

    rgb_yolo = YOLO(str(RGB_WEIGHTS))
    ir_yolo  = YOLO(str(IR_WEIGHTS))

    # Enumerate video test drone clips
    root = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone"
    clips = []
    if root.exists():
        for cdir in sorted(root.iterdir()):
            if not cdir.is_dir(): continue
            img_dir = next((d for d in (cdir/"images"/"test", cdir/"images") if d.exists()), None)
            lbl_dir = next((d for d in (cdir/"labels"/"test", cdir/"labels") if d.exists()), None)
            if img_dir and lbl_dir:
                clips.append((cdir.name, img_dir, lbl_dir))
    print(f"Found {len(clips)} drone clips")

    # Collect all frames
    all_frames = []
    for name, img_dir, lbl_dir in clips:
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() in exts:
                all_frames.append((name, p, lbl_dir / f"{p.stem}.txt"))
    print(f"Total frames: {len(all_frames)}")

    # Stride if needed
    if args.limit and len(all_frames) > args.limit:
        stride = max(1, len(all_frames) // args.limit)
        all_frames = all_frames[::stride]
    print(f"Evaluating: {len(all_frames)} frames")

    modes = ["baseline", "mean_fill", "zero_scene"]
    # Per-mode counters
    label_counts = {m: Counter() for m in modes}
    tp_fp_fn = {m: {"tp": 0, "fp": 0, "fn": 0} for m in modes}
    # Also track raw RGB detector as reference
    raw_rgb = {"tp": 0, "fp": 0, "fn": 0}

    t0 = time.time()
    for idx, (clip, img_path, lbl_path) in enumerate(all_frames):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        gts = read_yolo_labels(lbl_path, w, h, drone_classes={0})

        # RGB detections
        res_rgb = rgb_yolo.predict(img, imgsz=640, conf=0.25, device=args.device, verbose=False)
        rgb_dets = []
        if res_rgb[0].boxes is not None and len(res_rgb[0].boxes) > 0:
            xyxy = res_rgb[0].boxes.xyxy.cpu().numpy()
            confs = res_rgb[0].boxes.conf.cpu().numpy()
            rgb_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]

        # IR detections on grayscale
        res_ir = ir_yolo.predict(gray3, imgsz=640, conf=0.40, device=args.device, verbose=False)
        ir_dets = []
        if res_ir[0].boxes is not None and len(res_ir[0].boxes) > 0:
            xyxy = res_ir[0].boxes.xyxy.cpu().numpy()
            confs = res_ir[0].boxes.conf.cpu().numpy()
            ir_dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]

        # Raw RGB score (reference)
        tp, fp, fn = score_detections(rgb_dets, gts, rule="iop", iop_thr=0.5)
        raw_rgb["tp"] += tp
        raw_rgb["fp"] += fp
        raw_rgb["fn"] += fn

        # Test each classifier mode
        for mode in modes:
            x = build_features(rgb_dets, ir_dets, gray, gray, feat_cols, mode=mode)
            try:
                label = int(model.predict(x)[0])
            except Exception:
                label = 3
            label_counts[mode][label] += 1

            # Trust-aware: which dets to keep
            if label == 0:
                kept = []
            elif label == 1:
                kept = rgb_dets
            elif label == 2:
                kept = ir_dets
            else:  # 3
                kept = rgb_dets + ir_dets

            tp, fp, fn = score_detections(kept, gts, rule="iop", iop_thr=0.5)
            tp_fp_fn[mode]["tp"] += tp
            tp_fp_fn[mode]["fp"] += fp
            tp_fp_fn[mode]["fn"] += fn

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(all_frames)}  {fps:.1f} fps")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_frames)} frames in {elapsed:.0f}s\n")

    # ── Results ──
    def prf(d):
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        return tp, fp, fn, p, r, f

    print("=" * 80)
    print("TRUST LABEL DISTRIBUTION (% of frames)")
    print(f"{'Mode':<15s} {'reject(0)':>10s} {'rgb(1)':>10s} {'ir(2)':>10s} {'both(3)':>10s}")
    n = len(all_frames)
    for mode in modes:
        c = label_counts[mode]
        print(f"{mode:<15s} {c[0]/n:>10.1%} {c[1]/n:>10.1%} {c[2]/n:>10.1%} {c[3]/n:>10.1%}")

    print()
    print("DETECTION METRICS (IoP @ 0.5, trust-aware)")
    print(f"{'Mode':<15s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'P':>8s} {'R':>8s} {'F1':>8s}")
    tp, fp, fn, p, r, f = prf(raw_rgb)
    print(f"{'rgb_only':<15s} {tp:>6d} {fp:>6d} {fn:>6d} {p:>8.4f} {r:>8.4f} {f:>8.4f}")
    for mode in modes:
        tp, fp, fn, p, r, f = prf(tp_fp_fn[mode])
        print(f"{mode:<15s} {tp:>6d} {fp:>6d} {fn:>6d} {p:>8.4f} {r:>8.4f} {f:>8.4f}")

    print()
    print("REJECT RATE COMPARISON")
    for mode in modes:
        rej = label_counts[mode][0]
        print(f"  {mode:<15s}: {rej}/{n} frames rejected ({rej/n:.1%})")

    # Save results
    out = REPO / "eval" / "results" / "grayscale_clf_comparison.json"
    results = {
        "n_frames": n,
        "modes": {},
        "rgb_only": {"tp": raw_rgb["tp"], "fp": raw_rgb["fp"], "fn": raw_rgb["fn"]},
    }
    for mode in modes:
        tp, fp, fn, p, r, f = prf(tp_fp_fn[mode])
        results["modes"][mode] = {
            "labels": dict(label_counts[mode]),
            "tp": tp_fp_fn[mode]["tp"], "fp": tp_fp_fn[mode]["fp"], "fn": tp_fp_fn[mode]["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "reject_rate": round(label_counts[mode][0] / n, 4),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
