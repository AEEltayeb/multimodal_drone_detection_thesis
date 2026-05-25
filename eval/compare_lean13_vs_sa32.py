"""Fast head-to-head: lean13 vs sa32 on all 5 dashboard datasets.

- Uses cached YOLO detections (no GPU work for the detector).
- Computes classifier features per-frame fresh, since they're not cached.
- Caps each dataset to 300 frames via uniform stride for speed.
- Applies the production routing rule per dataset (argmax / softveto).
- Reports P / R / F1 side-by-side + Δ + a verdict per dataset.

Outputs CSV → docs/analysis/full_pipeline_ablations/csv/lean13_vs_sa32.csv
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(REPO / "classifier"))

from datasets import read_yolo_labels  # noqa
from metrics import score_detections   # noqa
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES  # noqa

CACHE = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
OUT_CSV = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "csv" / "lean13_vs_sa32.csv"

SA32 = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"
LEAN13 = REPO / "classifier" / "fusion_models" / "lean13" / "model.joblib"

SOFTVETO_TAU = 0.95
MAX_FRAMES = 300

DATASETS = {
    "antiuav": {
        "rgb_cache": "antiuav_selcom_1280_sz1280.json",
        "ir_cache":  "antiuav_ir_model_sz640.json",
        "rgb_img":   Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "rgb_lbl":   Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_img":    Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "ir_lbl":    Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "mode": "argmax", "score": "iou", "ir_kind": "paired",
    },
    "svanstrom": {
        "rgb_cache": "svanstrom_selcom_1280_sz1280.json",
        "ir_cache":  "svanstrom_ir_model_sz640.json",
        "rgb_img":   Path("G:/drone/svanstrom_paired/RGB/images"),
        "rgb_lbl":   Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_img":    Path("G:/drone/svanstrom_paired/IR/images"),
        "ir_lbl":    Path("G:/drone/svanstrom_paired/IR/labels"),
        "rgb_suffix": "_visible", "ir_suffix": "_infrared",
        "mode": "argmax", "score": "iop", "ir_kind": "paired",
    },
    "rgb_test": {
        "rgb_cache": "rgb_test_selcom_960_sz960.json",
        "ir_cache":  "rgb_test_ir_grayscale_sz640.json",
        "rgb_img":   Path("G:/drone/dataset/dataset/images/test"),
        "rgb_lbl":   Path("G:/drone/dataset/dataset/labels/test"),
        "mode": "softveto", "score": "iou", "ir_kind": "grayscale",
    },
    "ir_test": {
        "rgb_cache": "ir_test_selcom_960_sz960.json",   # selcom-on-IR (synthetic) — kept for fairness, classifier will see it
        "ir_cache":  "ir_test_ir_native_sz640.json",
        "rgb_img":   Path("G:/drone/IR_dset_final/test/images"),
        "rgb_lbl":   Path("G:/drone/IR_dset_final/test/labels"),
        "mode": "argmax", "score": "iou", "ir_kind": "native_only",
    },
    # drone_video has many clips — pick the same clip used in dashboard headline
    "drone_video": {
        "_clips": [
            ("drone_and_bird_sky_and_trees_short", "drone"),
            ("drone_seagull_attack", "drone"),
            ("drone_attacked_by_bird_mountain_side", "drone"),
        ],
        "mode": "softveto", "score": "iop", "ir_kind": "grayscale",
    },
}


def _load_cache(p):
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8")).get("dets", {})


def _to_dets(raw):
    if not raw: return []
    return [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]


def _build_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
    feats = {}
    for prefix, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [c for _, c in dets]
        feats[f"{prefix}_max_conf"]  = float(max(confs)) if confs else 0.0
        feats[f"{prefix}_mean_conf"] = float(np.mean(confs)) if confs else 0.0
    feats.update({f"rgb_{k}": v for k, v in compute_global_features(rgb_gray).items()})
    feats.update({f"ir_{k}":  v for k, v in compute_global_features(ir_gray).items()})
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
    return feats


def _vec(feats, cols):
    return np.array([[feats.get(c, 0.0) for c in cols]], dtype=np.float32)


def _label(rgb_dets, ir_dets, probs, mode):
    if mode == "argmax":
        return int(np.argmax(probs))
    # softveto τ=0.95
    p_reject = float(probs[0]); argmax = int(np.argmax(probs))
    if rgb_dets:
        return 0 if p_reject >= SOFTVETO_TAU else 1
    if argmax in (2, 3) and ir_dets: return argmax
    return 0


def _route(label, rgb_dets, ir_dets):
    if label == 0: return []
    if label == 1: return rgb_dets
    if label == 2: return ir_dets
    return rgb_dets + ir_dets


def _eval(stems, rgb_cache, ir_cache, img_dir, lbl_dir, mode, score, ir_kind, info, sa, le, sa_cols, le_cols):
    """Returns ((tp,fp,fn) for sa, for lean13)."""
    tp_sa = fp_sa = fn_sa = 0
    tp_le = fp_le = fn_le = 0
    rgb_suf = info.get("rgb_suffix", ""); ir_suf = info.get("ir_suffix", "")
    ir_img_dir = info.get("ir_img"); ir_lbl_dir = info.get("ir_lbl")
    for stem in stems:
        rgb_d = _to_dets(rgb_cache.get(stem, []))
        ir_d  = _to_dets(ir_cache.get(stem, []))
        # find RGB image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = img_dir / f"{stem}{ext}"
            if p.exists(): img_path = p; break
        if img_path is None: continue
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        h, w = img.shape[:2]
        rgb_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # IR side
        if ir_kind == "paired" and rgb_suf:
            ir_stem = stem.replace(rgb_suf, ir_suf)
            ir_img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = ir_img_dir / f"{ir_stem}{ext}"
                if p.exists(): ir_img_path = p; break
            if ir_img_path is not None:
                ir_im = cv2.imdecode(np.fromfile(str(ir_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                ir_gray = cv2.cvtColor(ir_im, cv2.COLOR_BGR2GRAY) if ir_im is not None else rgb_gray
            else:
                ir_gray = rgb_gray
        else:
            ir_gray = rgb_gray  # grayscale-RGB fallback OR ir_test where image IS IR
        lbl = lbl_dir / f"{stem}.txt"
        gts = read_yolo_labels(lbl, w, h, drone_classes={0}) if (lbl.exists() and lbl.stat().st_size > 0) else []

        feats = _build_features(rgb_d, ir_d, rgb_gray, ir_gray)
        try: p_sa = sa.predict_proba(_vec(feats, sa_cols))[0]
        except Exception: p_sa = np.array([0,0,0,1.0])
        try: p_le = le.predict_proba(_vec(feats, le_cols))[0]
        except Exception: p_le = np.array([0,0,0,1.0])
        kept_sa = _route(_label(rgb_d, ir_d, p_sa, mode), rgb_d, ir_d)
        kept_le = _route(_label(rgb_d, ir_d, p_le, mode), rgb_d, ir_d)
        t, f_, n = score_detections(kept_sa, gts, rule=score); tp_sa+=t; fp_sa+=f_; fn_sa+=n
        t, f_, n = score_detections(kept_le, gts, rule=score); tp_le+=t; fp_le+=f_; fn_le+=n
    return (tp_sa, fp_sa, fn_sa), (tp_le, fp_le, fn_le)


def _pr(tp, fp, fn):
    P = tp/(tp+fp) if (tp+fp) else 0.0
    R = tp/(tp+fn) if (tp+fn) else 0.0
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return P, R, F


def main():
    sa_obj = joblib.load(str(SA32)); sa = sa_obj["model"]; sa_cols = sa_obj["features"]
    le_obj = joblib.load(str(LEAN13)); le = le_obj["model"]; le_cols = le_obj["features"]
    print(f"sa32:   {len(sa_cols)} feat   |  lean13: {len(le_cols)} feat   |  cap {MAX_FRAMES} frames/dataset\n")
    print(f"{'dataset':<22s} {'mode':<9s} {'n':>5s}  "
          f"{'sa32 P':>7s} {'sa32 R':>7s} {'sa32 F1':>8s}  "
          f"{'l13 P':>7s} {'l13 R':>7s} {'l13 F1':>8s}  {'ΔF1':>7s}  verdict")
    print("-"*120)
    rows = []
    for ds_name, info in DATASETS.items():
        if "_clips" in info:
            # drone_video: combine across clips
            stems_all = []
            rgb_all = {}; ir_all = {}
            for clip, kind in info["_clips"]:
                rc = CACHE / f"video_drone_{clip}_selcom_960_sz960.json"
                ic = CACHE / f"video_drone_{clip}_ir_grayscale_sz640.json"
                rgb_all.update(_load_cache(rc))
                ir_all.update(_load_cache(ic))
                # NOTE: img/lbl dirs differ per clip — handle inside loop
            # Per-clip walk
            tp_s=fp_s=fn_s=0; tp_l=fp_l=fn_l=0; total_n = 0
            for clip, kind in info["_clips"]:
                rc = _load_cache(CACHE / f"video_drone_{clip}_selcom_960_sz960.json")
                ic = _load_cache(CACHE / f"video_drone_{clip}_ir_grayscale_sz640.json")
                stems = sorted(set(rc) & set(ic))
                if len(stems) > MAX_FRAMES:
                    stride = max(1, len(stems) // MAX_FRAMES)
                    stems = stems[::stride][:MAX_FRAMES]
                total_n += len(stems)
                img_dir = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone" / clip / "images" / "test"
                lbl_dir = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone" / clip / "labels" / "test"
                clip_info = {"ir_img": None, "ir_lbl": None}
                a, b = _eval(stems, rc, ic, img_dir, lbl_dir, info["mode"], info["score"],
                             info["ir_kind"], clip_info, sa, le, sa_cols, le_cols)
                tp_s+=a[0]; fp_s+=a[1]; fn_s+=a[2]
                tp_l+=b[0]; fp_l+=b[1]; fn_l+=b[2]
            Ps, Rs, Fs = _pr(tp_s, fp_s, fn_s); Pl, Rl, Fl = _pr(tp_l, fp_l, fn_l)
            d = Fl - Fs
            verdict = "lean13 WINS" if d > 0.005 else ("sa32 wins" if d < -0.005 else "tie")
            print(f"{ds_name:<22s} {info['mode']:<9s} {total_n:>5d}  "
                  f"{Ps:>7.3f} {Rs:>7.3f} {Fs:>8.3f}  {Pl:>7.3f} {Rl:>7.3f} {Fl:>8.3f}  {d:+7.3f}  {verdict}")
            rows.append((ds_name, info["mode"], total_n, Ps, Rs, Fs, Pl, Rl, Fl, d, verdict))
            continue
        rgb_cache = _load_cache(CACHE / info["rgb_cache"])
        ir_cache  = _load_cache(CACHE / info["ir_cache"])
        stems = sorted(set(rgb_cache) & set(ir_cache))
        if not stems:
            print(f"{ds_name:<22s} (no overlapping stems — skipping)"); continue
        if len(stems) > MAX_FRAMES:
            stride = max(1, len(stems) // MAX_FRAMES)
            stems = stems[::stride][:MAX_FRAMES]
        a, b = _eval(stems, rgb_cache, ir_cache, info["rgb_img"], info["rgb_lbl"],
                     info["mode"], info["score"], info["ir_kind"], info, sa, le, sa_cols, le_cols)
        Ps, Rs, Fs = _pr(*a); Pl, Rl, Fl = _pr(*b); d = Fl - Fs
        verdict = "lean13 WINS" if d > 0.005 else ("sa32 wins" if d < -0.005 else "tie")
        print(f"{ds_name:<22s} {info['mode']:<9s} {len(stems):>5d}  "
              f"{Ps:>7.3f} {Rs:>7.3f} {Fs:>8.3f}  {Pl:>7.3f} {Rl:>7.3f} {Fl:>8.3f}  {d:+7.3f}  {verdict}")
        rows.append((ds_name, info["mode"], len(stems), Ps, Rs, Fs, Pl, Rl, Fl, d, verdict))

    # Cross-modal: also try lean13 with ARGMAX on the softveto datasets (the key
    # hypothesis: lean13 should let argmax work on grayscale modes)
    print()
    print("Hypothesis check: does lean13 enable argmax on softveto datasets?")
    print("-"*120)
    for ds_name in ("rgb_test", "drone_video"):
        info = DATASETS[ds_name]
        info2 = dict(info); info2["mode"] = "argmax"
        if "_clips" in info:
            tp_s=fp_s=fn_s=0; tp_l=fp_l=fn_l=0; total_n=0
            for clip, kind in info["_clips"]:
                rc = _load_cache(CACHE / f"video_drone_{clip}_selcom_960_sz960.json")
                ic = _load_cache(CACHE / f"video_drone_{clip}_ir_grayscale_sz640.json")
                stems = sorted(set(rc) & set(ic))
                if len(stems) > MAX_FRAMES:
                    stride = max(1, len(stems) // MAX_FRAMES)
                    stems = stems[::stride][:MAX_FRAMES]
                total_n += len(stems)
                img_dir = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone" / clip / "images" / "test"
                lbl_dir = REPO / "datasets" / "drone detection video tests" / "rgb" / "drone" / clip / "labels" / "test"
                a, b = _eval(stems, rc, ic, img_dir, lbl_dir, "argmax", info["score"],
                             info["ir_kind"], {"ir_img": None, "ir_lbl": None}, sa, le, sa_cols, le_cols)
                tp_s+=a[0]; fp_s+=a[1]; fn_s+=a[2]; tp_l+=b[0]; fp_l+=b[1]; fn_l+=b[2]
            Ps,Rs,Fs = _pr(tp_s,fp_s,fn_s); Pl,Rl,Fl = _pr(tp_l,fp_l,fn_l); d = Fl - Fs
            verdict = "lean13 WINS argmax" if d > 0.005 else ("sa32 wins" if d < -0.005 else "tie")
            print(f"{ds_name+' @argmax':<22s} {'argmax':<9s} {total_n:>5d}  "
                  f"{Ps:>7.3f} {Rs:>7.3f} {Fs:>8.3f}  {Pl:>7.3f} {Rl:>7.3f} {Fl:>8.3f}  {d:+7.3f}  {verdict}")
            rows.append((ds_name+"@argmax", "argmax", total_n, Ps, Rs, Fs, Pl, Rl, Fl, d, verdict))
            continue
        rgb_cache = _load_cache(CACHE / info["rgb_cache"])
        ir_cache  = _load_cache(CACHE / info["ir_cache"])
        stems = sorted(set(rgb_cache) & set(ir_cache))
        if len(stems) > MAX_FRAMES:
            stride = max(1, len(stems) // MAX_FRAMES)
            stems = stems[::stride][:MAX_FRAMES]
        a, b = _eval(stems, rgb_cache, ir_cache, info["rgb_img"], info["rgb_lbl"],
                     "argmax", info["score"], info["ir_kind"], info2, sa, le, sa_cols, le_cols)
        Ps, Rs, Fs = _pr(*a); Pl, Rl, Fl = _pr(*b); d = Fl - Fs
        verdict = "lean13 WINS argmax" if d > 0.005 else ("sa32 wins" if d < -0.005 else "tie")
        print(f"{ds_name+' @argmax':<22s} {'argmax':<9s} {len(stems):>5d}  "
              f"{Ps:>7.3f} {Rs:>7.3f} {Fs:>8.3f}  {Pl:>7.3f} {Rl:>7.3f} {Fl:>8.3f}  {d:+7.3f}  {verdict}")
        rows.append((ds_name+"@argmax", "argmax", len(stems), Ps, Rs, Fs, Pl, Rl, Fl, d, verdict))

    # Save CSV
    import csv
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["dataset","mode","n","sa32_P","sa32_R","sa32_F1","lean13_P","lean13_R","lean13_F1","dF1","verdict"])
        for r in rows: w.writerow(r)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
