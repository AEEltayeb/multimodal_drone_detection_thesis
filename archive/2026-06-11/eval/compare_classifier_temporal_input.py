"""
compare_classifier_temporal_input.py — A/B classifier input variants on TRULY sequential frames.

Variant A: classifier(rgb_dets, ir_dets) per-frame  (current production)
Variant B: classifier(rgb_dets', ir_dets') per-frame, where the prime is
           temporally-smoothed via IoU-continuity in adjacent frames.

For this test we use sequences of CONSECUTIVE frames (no stride) so the
"adjacent frame" notion is real:
  - drone_video drone clips (already per-clip, consecutive)
  - antiuav grouped by video prefix
  - svanstrom grouped by video prefix
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from metrics import score_detections  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES  # noqa: E402

CACHE = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
CLASSIFIER_PATH = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
SOFTVETO_TAU = 0.95
IOU_THR_TEMP = 0.2   # looser — drones move


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); bb = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + bb - inter) if (aa + bb - inter) > 0 else 0.0


def temporal_smooth(per_frame_dets, iou_thr=IOU_THR_TEMP):
    """Each det survives if a co-located det fires in f-1 OR f+1."""
    out = []
    for i, dets in enumerate(per_frame_dets):
        survivors = []
        for d in dets:
            b = d[0]
            ok = False
            for j in (i - 1, i + 1):
                if 0 <= j < len(per_frame_dets):
                    for d2 in per_frame_dets[j]:
                        if iou(b, d2[0]) >= iou_thr:
                            ok = True; break
                if ok: break
            if ok: survivors.append(d)
        out.append(survivors)
    return out


def soft_veto_label(rgb_dets, ir_dets, probs, tau=SOFTVETO_TAU):
    p_reject = float(probs[0]); argmax = int(np.argmax(probs))
    if rgb_dets:
        return 0 if p_reject >= tau else 1
    if argmax in (2, 3) and ir_dets:
        return argmax
    return 0


def build_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
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


def load_cache(path):
    if not path.exists(): return {}
    return json.loads(path.read_text())["dets"]


def to_dets(raw):
    return [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]


def group_by_video_prefix(stems):
    """Group stems by their video prefix (e.g., 'dv5_auv_04_8618_1215-2714')."""
    groups = defaultdict(list)
    for s in stems:
        # Drop a trailing _frame_NNN / _NNNN suffix; group on the rest.
        m = re.match(r'^(.+?)_f?\d{4,}', s)
        key = m.group(1) if m else s
        groups[key].append(s)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 30}


def eval_sequence(stems, rgb_dets_map, ir_dets_map, img_dir, lbl_dir,
                   mode, score_rule, ir_kind, classifier, feat_cols):
    """Process a single consecutive-frame sequence and return TP/FP/FN for both variants."""
    rgb_seq = [to_dets(rgb_dets_map.get(s, [])) for s in stems]
    ir_seq = [to_dets(ir_dets_map.get(s, [])) for s in stems]
    rgb_seq_t = temporal_smooth(rgb_seq)
    ir_seq_t = temporal_smooth(ir_seq)

    results = {}
    for variant, rgb_for, ir_for in [
        ("no-temp", rgb_seq, ir_seq),
        ("+temp-in", rgb_seq_t, ir_seq_t),
    ]:
        tp = fp = fn = 0
        for i, stem in enumerate(stems):
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = img_dir / f"{stem}{ext}"
                if p.exists(): img_path = p; break
            if img_path is None: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            rgb_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ir_gray = rgb_gray
            if ir_kind == "ir_paired":
                ir_stem = stem.replace("_visible", "_infrared") if "_visible" in stem else stem
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    ir_p = img_dir.parent.parent / "IR" / "images" / f"{ir_stem}{ext}"
                    if ir_p.exists():
                        ir_img = cv2.imread(str(ir_p))
                        if ir_img is not None:
                            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
                        break
            lbl = lbl_dir / f"{stem}.txt"
            gts = read_yolo_labels(lbl, w, h, drone_classes={0}) if lbl.exists() else []
            rgb_d = rgb_for[i]; ir_d = ir_for[i]
            x = build_features(rgb_d, ir_d, rgb_gray, ir_gray, feat_cols)
            try:
                probs = classifier.predict_proba(x)[0]
            except Exception:
                probs = np.array([0.0, 0.0, 0.0, 1.0])
            label = soft_veto_label(rgb_d, ir_d, probs) if mode == "softveto" else int(np.argmax(probs))
            if label == 0: kept = []
            elif label == 1: kept = rgb_d
            elif label == 2: kept = ir_d
            else: kept = rgb_d + ir_d
            t, f_, n = score_detections(kept, gts, rule=score_rule)
            tp += t; fp += f_; fn += n
        results[variant] = (tp, fp, fn)
    return results


def main():
    obj = joblib.load(str(CLASSIFIER_PATH))
    classifier = obj["model"]; feat_cols = obj.get("features") or []

    print(f"{'Dataset / sequence':<50s} {'Mode':<8s} {'Variant':<9s} "
          f"{'Frames':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
          f"{'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 120)

    # ── drone_video clips (truly sequential, one cache per clip) ──
    drone_clips = sorted({p.name.replace("_baseline_sz1280.json", "")
                          for p in CACHE.glob("video_drone_*_baseline_sz1280.json")})
    DRONE_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone"
    for clip_key in drone_clips[:4]:  # 4 drone clips
        # Extract clip subdir name (strip 'video_drone_' prefix)
        subname = clip_key.replace("video_drone_", "")
        clip_dir = DRONE_ROOT / subname
        if not clip_dir.exists(): continue
        img_dir = clip_dir / "images" / "test"
        lbl_dir = clip_dir / "labels" / "test"
        rgb_map = load_cache(CACHE / f"{clip_key}_baseline_sz1280.json")
        ir_map = load_cache(CACHE / f"{clip_key}_ir_grayscale_sz640.json")
        stems = sorted(set(rgb_map) & set(ir_map))
        if not stems or not img_dir.exists(): continue
        res = eval_sequence(stems, rgb_map, ir_map, img_dir, lbl_dir,
                             "softveto", "iop", "ir_gray", classifier, feat_cols)
        for variant, (tp, fp, fn) in res.items():
            P = tp/(tp+fp) if (tp+fp) else 0.0
            R = tp/(tp+fn) if (tp+fn) else 0.0
            F = 2*P*R/(P+R) if (P+R) else 0.0
            print(f"{('drone_video / ' + subname[:36]):<50s} softveto {variant:<9s} "
                  f"{len(stems):>6d} {tp:>5d} {fp:>5d} {fn:>5d} "
                  f"{P:>6.3f} {R:>6.3f} {F:>6.3f}")

    # ── antiuav grouped by video prefix ──
    for ds_name, rgb_c, ir_c, lbl_dir, img_dir, ir_kind, score_rule in [
        ("antiuav", "antiuav_baseline_sz1280.json", "antiuav_ir_model_sz640.json",
         Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
         Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
         "ir_paired", "iou"),
        ("svanstrom", "svanstrom_baseline_sz1280.json", "svanstrom_ir_model_sz640.json",
         Path("G:/drone/svanstrom_paired/RGB/labels"),
         Path("G:/drone/svanstrom_paired/RGB/images"),
         "ir_paired", "iop"),
    ]:
        rgb_map = load_cache(CACHE / rgb_c); ir_map = load_cache(CACHE / ir_c)
        stems = sorted(set(rgb_map) & set(ir_map))
        groups = group_by_video_prefix(stems)
        # take the 2 longest sequences
        chosen = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:2]
        for vid, vstems in chosen:
            # cap to 200 consecutive frames for speed
            vstems = vstems[:200]
            res = eval_sequence(vstems, rgb_map, ir_map, img_dir, lbl_dir,
                                 "argmax", score_rule, ir_kind, classifier, feat_cols)
            for variant, (tp, fp, fn) in res.items():
                P = tp/(tp+fp) if (tp+fp) else 0.0
                R = tp/(tp+fn) if (tp+fn) else 0.0
                F = 2*P*R/(P+R) if (P+R) else 0.0
                label = f"{ds_name} / {vid[:36]}"
                print(f"{label:<50s} argmax   {variant:<9s} "
                      f"{len(vstems):>6d} {tp:>5d} {fp:>5d} {fn:>5d} "
                      f"{P:>6.3f} {R:>6.3f} {F:>6.3f}")


if __name__ == "__main__":
    main()
