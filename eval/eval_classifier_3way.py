"""
eval_classifier_3way.py - Head-to-head eval of 3 fusion classifiers
(retrained_v2 32-feat, control_v3more 40-feat, lean13 13-feat) on a unified
100-frames-per-source eval set: Anti-UAV, Svanstrom, drone-video-tests.

Computes the SUPERSET of features once per frame (40 columns: 32 standard
features + 8 detection-flag features), then each classifier picks its
subset from the model bundle's `features` list.

Outputs:
  docs/analysis/<today>_classifier_3way_eval.md
  docs/analysis/full_pipeline_ablations/csv/classifier_3way.csv

Usage:
    python eval/eval_classifier_3way.py --n-per-source 100
"""

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))

# Reuse the full feature computations from generate_retrained_v2_data
from generate_retrained_v2_data import (  # noqa: E402
    compute_global_features, compute_target_features,
    parse_yolo_gt, has_tp, has_gt,
)


# 40-feature column order = 32 standard + 8 detection flags.
STANDARD_32 = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
    "rgb_img_entropy", "rgb_sky_ground_ratio", "rgb_edge_density",
    "rgb_blurriness",
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
    "ir_img_entropy", "ir_sky_ground_ratio", "ir_edge_density",
    "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]
DET_FLAGS_8 = [
    "rgb_n_dets", "ir_n_dets",
    "rgb_detected", "ir_detected",
    "both_detect", "neither_detect",
    "rgb_only_detect", "ir_only_detect",
]
SUPERSET = STANDARD_32 + DET_FLAGS_8


LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_superset_row(rgb_dets, ir_dets, rgb_gray, ir_gray, rgb_wh, ir_wh,
                       conf_thresh=0.25):
    rgb_dets = [d for d in rgb_dets if d[4] >= conf_thresh]
    ir_dets = [d for d in ir_dets if d[4] >= conf_thresh]
    rgb_w, rgb_h = rgb_wh
    ir_w, ir_h = ir_wh

    rgb_confs = [d[4] for d in rgb_dets]
    ir_confs = [d[4] for d in ir_dets]
    row = {
        "rgb_max_conf": max(rgb_confs) if rgb_confs else 0.0,
        "rgb_mean_conf": float(np.mean(rgb_confs)) if rgb_confs else 0.0,
        "ir_max_conf": max(ir_confs) if ir_confs else 0.0,
        "ir_mean_conf": float(np.mean(ir_confs)) if ir_confs else 0.0,
    }
    for k, v in compute_global_features(rgb_gray).items():
        row[f"rgb_{k}"] = v
    for k, v in compute_global_features(ir_gray).items():
        row[f"ir_{k}"] = v
    if rgb_dets:
        b = max(rgb_dets, key=lambda d: d[4])
        for k, v in compute_target_features(rgb_gray, b[:4], rgb_w, rgb_h).items():
            row[f"rgb_best_{k}"] = v
    else:
        for k in ["log_bbox_area","aspect_ratio","pos_x","pos_y",
                  "dist_to_center","local_contrast","target_bg_delta"]:
            row[f"rgb_best_{k}"] = 0.0
    if ir_dets:
        b = max(ir_dets, key=lambda d: d[4])
        for k, v in compute_target_features(ir_gray, b[:4], ir_w, ir_h).items():
            row[f"ir_best_{k}"] = v
    else:
        for k in ["log_bbox_area","aspect_ratio","pos_x","pos_y",
                  "dist_to_center","local_contrast","target_bg_delta"]:
            row[f"ir_best_{k}"] = 0.0

    # 8 detection flags
    rd = 1 if rgb_dets else 0
    idet = 1 if ir_dets else 0
    row["rgb_n_dets"] = len(rgb_dets)
    row["ir_n_dets"] = len(ir_dets)
    row["rgb_detected"] = rd
    row["ir_detected"] = idet
    row["both_detect"] = 1 if (rd and idet) else 0
    row["neither_detect"] = 1 if (not rd and not idet) else 0
    row["rgb_only_detect"] = 1 if (rd and not idet) else 0
    row["ir_only_detect"] = 1 if (not rd and idet) else 0
    return row


def trust_from_gt(rgb_dets, ir_dets, rgb_gt, ir_gt, rgb_mode="iou", ir_mode="iou"):
    r = has_tp(rgb_dets, rgb_gt, mode=rgb_mode)
    i = has_tp(ir_dets, ir_gt, mode=ir_mode)
    if r and i: return 3
    if r: return 1
    if i: return 2
    return 0


# ---- Source samplers --------------------------------------------------------

def sample_antiuav(root, n, conf_thresh, rgb_model, ir_model, rgb_imgsz, ir_imgsz, cache_path):
    """Sample n frames from Anti-UAV test split."""
    root = Path(root)
    if not root.exists():
        print(f"  [skip antiuav] {root} not found"); return []
    rgb_dir = root / "RGB" / "images"
    ir_dir = root / "IR" / "images"
    rgb_lbl_dir = root / "RGB" / "labels"
    ir_lbl_dir = root / "IR" / "labels"
    strip = lambda s: re.sub(r"_(visible|infrared)", "", s, flags=re.IGNORECASE)
    rgb_map = {strip(f.stem): f for f in sorted(rgb_dir.iterdir())
               if f.suffix.lower() in IMG_EXTS}
    ir_map = {strip(f.stem): f for f in sorted(ir_dir.iterdir())
              if f.suffix.lower() in IMG_EXTS}
    shared = sorted(set(rgb_map) & set(ir_map))
    random.seed(42)
    sample = random.sample(shared, min(n, len(shared)))
    print(f"  antiuav: sampled {len(sample)} / {len(shared)} pairs")

    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = json.load(open(cache_path))

    rows = []
    for base in sample:
        ri, ii = rgb_map[base], ir_map[base]
        rgb_img = cv2.imread(str(ri)); ir_img = cv2.imread(str(ii))
        if rgb_img is None or ir_img is None: continue
        rh, rw = rgb_img.shape[:2]; ih, iw = ir_img.shape[:2]
        rl = rgb_lbl_dir / (ri.stem + ".txt"); il = ir_lbl_dir / (ii.stem + ".txt")

        key = base
        if key in cache:
            rgb_dets = cache[key]["rgb_dets"]
            ir_dets = cache[key]["ir_dets"]
        elif rgb_model is not None:
            rgb_dets = _yolo(rgb_model, rgb_img, conf_thresh, rgb_imgsz)
            ir_dets = _yolo(ir_model, ir_img, conf_thresh, ir_imgsz)
        else:
            print(f"  [skip] {key}: no cache + no model"); continue

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape) == 3 else ir_img
        rgb_gt = parse_yolo_gt(rl, rw, rh)
        ir_gt = parse_yolo_gt(il, iw, ih)
        label = trust_from_gt(rgb_dets, ir_dets, rgb_gt, ir_gt, "iou", "iou")
        row = build_superset_row(rgb_dets, ir_dets, rgb_gray, ir_gray,
                                  (rw, rh), (iw, ih), conf_thresh)
        row.update({"trust_label": label, "stem": base, "source": "antiuav"})
        rows.append(row)
    return rows


def sample_svanstrom(root, n, conf_thresh, rgb_model, ir_model, rgb_imgsz, ir_imgsz, cache_path):
    root = Path(root)
    if not root.exists():
        print(f"  [skip svanstrom] {root} not found"); return []
    rgb_dir = root / "RGB" / "images"
    ir_dir = root / "IR" / "images"
    rgb_lbl_dir = root / "RGB" / "labels"
    ir_lbl_dir = root / "IR" / "labels"
    strip = lambda s: re.sub(r"_(visible|infrared)", "", s, flags=re.IGNORECASE)
    rgb_map = {strip(f.stem): f for f in sorted(rgb_dir.iterdir())
               if f.suffix.lower() in IMG_EXTS}
    ir_map = {strip(f.stem): f for f in sorted(ir_dir.iterdir())
              if f.suffix.lower() in IMG_EXTS}
    shared = sorted(set(rgb_map) & set(ir_map))
    random.seed(43)
    sample = random.sample(shared, min(n, len(shared)))
    print(f"  svanstrom: sampled {len(sample)} / {len(shared)} pairs")

    cache = json.load(open(cache_path)) if (cache_path and Path(cache_path).exists()) else {}

    rows = []
    for base in sample:
        ri, ii = rgb_map[base], ir_map[base]
        rgb_img = cv2.imread(str(ri)); ir_img = cv2.imread(str(ii))
        if rgb_img is None or ir_img is None: continue
        rh, rw = rgb_img.shape[:2]; ih, iw = ir_img.shape[:2]
        rl = rgb_lbl_dir / (ri.stem + ".txt"); il = ir_lbl_dir / (ii.stem + ".txt")

        if base in cache:
            rgb_dets = cache[base]["rgb_dets"]; ir_dets = cache[base]["ir_dets"]
        elif rgb_model is not None:
            rgb_dets = _yolo(rgb_model, rgb_img, conf_thresh, rgb_imgsz)
            ir_dets = _yolo(ir_model, ir_img, conf_thresh, ir_imgsz)
        else:
            print(f"  [skip] {base}: no cache + no model"); continue

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY) if len(ir_img.shape) == 3 else ir_img
        rgb_gt = parse_yolo_gt(rl, rw, rh)
        ir_gt = parse_yolo_gt(il, iw, ih)
        label = trust_from_gt(rgb_dets, ir_dets, rgb_gt, ir_gt, "iop", "iou")
        row = build_superset_row(rgb_dets, ir_dets, rgb_gray, ir_gray,
                                  (rw, rh), (iw, ih), conf_thresh)
        row.update({"trust_label": label, "stem": base, "source": "svanstrom"})
        rows.append(row)
    return rows


def sample_video_tests(repo, n_per_source, conf_thresh, rgb_cache_tag="selcom_1280_sz1280"):
    """Sample n_per_source TOTAL from drone-video-tests across all categories."""
    vid_root = repo / "datasets" / "drone detection video tests" / "rgb"
    cache_dir = repo / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
    if not vid_root.exists():
        print(f"  [skip] {vid_root} missing"); return []

    all_frames = []  # (cat, clip, tag, stem, img_path, lbl_path, rgb_dets, ir_dets)
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cat_dir = vid_root / cat
        if not cat_dir.exists(): continue
        for clip in sorted(cat_dir.iterdir()):
            if not clip.is_dir(): continue
            img_dir = clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
            lbl_dir = clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
            tag = f"video_{cat}_{clip.name}"
            rgb_cache = cache_dir / f"{tag}_{rgb_cache_tag}.json"
            ir_cache = cache_dir / f"{tag}_ir_grayscale_sz640.json"
            if not (img_dir.exists() and rgb_cache.exists() and ir_cache.exists()):
                continue
            rgb_c = json.load(open(rgb_cache))["dets"]
            ir_c = json.load(open(ir_cache))["dets"]
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS: continue
                stem = img_path.stem
                all_frames.append((cat, clip.name, tag, stem, img_path,
                                   lbl_dir / f"{stem}.txt",
                                   rgb_c.get(stem, []), ir_c.get(stem, [])))

    # Stratify roughly: equal share per category present.
    by_cat = defaultdict(list)
    for f in all_frames: by_cat[f[0]].append(f)
    cats = [c for c in ("drone","birds","airplanes","helicopters") if by_cat[c]]
    per_cat = max(1, n_per_source // len(cats)) if cats else 0
    random.seed(44)
    sample = []
    for c in cats:
        sample.extend(random.sample(by_cat[c], min(per_cat, len(by_cat[c]))))
    # pad/trim to exactly n_per_source
    if len(sample) > n_per_source:
        sample = random.sample(sample, n_per_source)
    elif len(sample) < n_per_source:
        remaining = [f for f in all_frames if f not in sample]
        if remaining:
            sample.extend(random.sample(remaining, min(n_per_source - len(sample), len(remaining))))
    print(f"  drone-video-tests: sampled {len(sample)} frames (cats: {cats})")

    rows = []
    for cat, clip_name, tag, stem, img_path, lbl_path, rgb_dets, ir_dets in sample:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = parse_yolo_gt(lbl_path, w, h)
        label = trust_from_gt(rgb_dets, ir_dets, gt, gt, "iop", "iop")
        row = build_superset_row(rgb_dets, ir_dets, gray, gray,
                                  (w, h), (w, h), conf_thresh)
        row.update({"trust_label": label, "stem": f"{tag}_{stem}",
                    "source": f"video_{cat}"})
        rows.append(row)
    return rows


def _yolo(model, img, conf, imgsz):
    r = model.predict(img, conf=conf, verbose=False, imgsz=imgsz)[0]
    out = []
    if r.boxes is not None and len(r.boxes) > 0:
        xy = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        for i in range(len(xy)):
            out.append([float(xy[i][0]), float(xy[i][1]),
                        float(xy[i][2]), float(xy[i][3]), float(cf[i])])
    return out


# ---- Eval -------------------------------------------------------------------

def load_classifier(path):
    b = joblib.load(path)
    # Bundles vary: some store "features", some "feature_names".
    feats = b.get("features") or b.get("feature_names")
    if feats is None:
        raise SystemExit(f"Bundle at {path} has no feature list.")
    return b["model"], list(feats)


def eval_one(model, feats, df):
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required features in eval frame: {missing}")
    X = df[feats].values
    y = df["trust_label"].values
    t0 = time.time()
    pred = model.predict(X)
    latency_ms = 1000.0 * (time.time() - t0) / max(1, len(X))
    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro", zero_division=0))
    f1w = float(f1_score(y, pred, average="weighted", zero_division=0))
    per_src = {}
    for s in sorted(df["source"].unique()):
        m = df["source"].values == s
        per_src[s] = {
            "n": int(m.sum()),
            "acc": float(accuracy_score(y[m], pred[m])),
            "f1_macro": float(f1_score(y[m], pred[m], average="macro", zero_division=0)),
        }
    return {
        "acc": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "latency_ms_per_frame": latency_ms,
        "per_source": per_src,
        "report": classification_report(
            y, pred,
            target_names=[LABEL_NAMES[i] for i in range(4)],
            zero_division=0),
    }, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-source", type=int, default=100)
    ap.add_argument("--auv-root", default="G:/drone/Anti-UAV-RGBT_yolo_converted/test")
    ap.add_argument("--svan-root", default="G:/drone/svanstrom_paired")
    ap.add_argument("--rgb-weights", default=None,
                    help="needed only if antiuav/svanstrom caches are not populated")
    ap.add_argument("--ir-weights", default=None)
    ap.add_argument("--auv-imgsz", type=int, default=640)
    ap.add_argument("--svan-imgsz", type=int, default=1280)
    ap.add_argument("--ir-imgsz", type=int, default=640)
    ap.add_argument("--video-rgb-cache-tag", default="selcom_1280_sz1280")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--clf-32",
                    default="classifier/fusion_models/retrained_v2_32feat/model.joblib")
    ap.add_argument("--clf-40",
                    default="classifier/fusion_models/control_v3more_40feat/model.joblib")
    ap.add_argument("--clf-13",
                    default="classifier/fusion_models/lean13/model.joblib")
    ap.add_argument("--clf-10",
                    default="classifier/fusion_models/lean10/model.joblib")
    ap.add_argument("--clf-19",
                    default="classifier/fusion_models/lean19/model.joblib")
    ap.add_argument("--auv-cache",
                    default="classifier/fusion_models/retrained_v2_32feat/cache_antiuav.json")
    ap.add_argument("--svan-cache",
                    default="classifier/fusion_models/retrained_v2_32feat/cache_svanstrom.json")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent

    rgb_model = ir_model = None
    if args.rgb_weights and args.ir_weights:
        from ultralytics import YOLO
        rgb_model = YOLO(args.rgb_weights); ir_model = YOLO(args.ir_weights)

    print("Building eval set...")
    rows = []
    print(" Anti-UAV (RGB sz=%d, IR sz=%d):" % (args.auv_imgsz, args.ir_imgsz))
    rows += sample_antiuav(args.auv_root, args.n_per_source, args.conf,
                            rgb_model, ir_model, args.auv_imgsz, args.ir_imgsz,
                            repo / args.auv_cache if args.auv_cache else None)
    print(" Svanstrom (RGB sz=%d, IR sz=%d):" % (args.svan_imgsz, args.ir_imgsz))
    rows += sample_svanstrom(args.svan_root, args.n_per_source, args.conf,
                              rgb_model, ir_model, args.svan_imgsz, args.ir_imgsz,
                              repo / args.svan_cache if args.svan_cache else None)
    print(" Drone-video-tests (cache tag=%s):" % args.video_rgb_cache_tag)
    rows += sample_video_tests(repo, args.n_per_source, args.conf,
                                rgb_cache_tag=args.video_rgb_cache_tag)

    if not rows:
        raise SystemExit("No rows sampled.")

    import pandas as pd
    df = pd.DataFrame(rows)
    src_dist = Counter(df["source"]); trust_dist = Counter(df["trust_label"])
    print(f"\nEval set: {len(df)} rows")
    print("  Sources:", dict(src_dist))
    print("  Trust :", {LABEL_NAMES[k]: v for k, v in trust_dist.items()})

    results = {}
    preds_by_clf = {}
    for name, path in [("lean10", args.clf_10),
                       ("lean13", args.clf_13),
                       ("lean19", args.clf_19),
                       ("32feat", args.clf_32),
                       ("40feat", args.clf_40)]:
        p = repo / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            print(f"  [skip {name}] {p} missing"); continue
        print(f"\n--- {name} ({p.name}) ---")
        model, feats = load_classifier(p)
        res, pred = eval_one(model, feats, df)
        results[name] = res; preds_by_clf[name] = pred
        print(f"  acc={res['acc']:.4f}  f1m={res['f1_macro']:.4f}  "
              f"latency={res['latency_ms_per_frame']:.3f} ms/frame")
        for s, m in res["per_source"].items():
            print(f"    {s:35s} n={m['n']:3d}  acc={m['acc']:.4f}  f1m={m['f1_macro']:.4f}")

    # Write CSV
    csv_dir = repo / "docs" / "analysis" / "full_pipeline_ablations" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "classifier_3way.csv"
    sources = sorted(df["source"].unique())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["classifier", "metric"] + sources + ["overall"])
        for name, res in results.items():
            w.writerow([name, "n"] + [res["per_source"][s]["n"] for s in sources] + [len(df)])
            w.writerow([name, "acc"] +
                       [f"{res['per_source'][s]['acc']:.4f}" for s in sources] +
                       [f"{res['acc']:.4f}"])
            w.writerow([name, "f1_macro"] +
                       [f"{res['per_source'][s]['f1_macro']:.4f}" for s in sources] +
                       [f"{res['f1_macro']:.4f}"])
            w.writerow([name, "latency_ms_per_frame", "", "", "",
                        f"{res['latency_ms_per_frame']:.4f}"])
    print(f"\n  CSV: {csv_path}")

    # Markdown
    md_dir = repo / "docs" / "analysis"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{date.today().isoformat()}_classifier_3way_eval.md"
    lines = []
    lines.append(f"# Classifier 3-way eval - {date.today().isoformat()}\n")
    lines.append(f"Eval set: {len(df)} frames (target ~{args.n_per_source}/source).\n")
    lines.append("Sources: " + ", ".join(f"`{s}`={src_dist[s]}" for s in sources) + "\n")
    lines.append("Trust labels: " +
                 ", ".join(f"`{LABEL_NAMES[k]}`={v}" for k, v in sorted(trust_dist.items())) + "\n")

    lines.append("\n## Overall\n")
    lines.append("| classifier | n_features | acc | F1m | F1w | ms/frame |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    nfeats = {"lean10": 10, "lean13": 13, "lean19": 19, "32feat": 32, "40feat": 40}
    for name, res in results.items():
        lines.append(f"| {name} | {nfeats.get(name,'?')} | {res['acc']:.4f} | "
                     f"{res['f1_macro']:.4f} | {res['f1_weighted']:.4f} | "
                     f"{res['latency_ms_per_frame']:.3f} |")

    lines.append("\n## Per-source accuracy\n")
    lines.append("| classifier | " + " | ".join(sources) + " |")
    lines.append("|---|" + "|".join("---:" for _ in sources) + "|")
    for name, res in results.items():
        lines.append(f"| {name} | " +
                     " | ".join(f"{res['per_source'][s]['acc']:.4f}" for s in sources) + " |")

    lines.append("\n## Per-source F1-macro\n")
    lines.append("| classifier | " + " | ".join(sources) + " |")
    lines.append("|---|" + "|".join("---:" for _ in sources) + "|")
    for name, res in results.items():
        lines.append(f"| {name} | " +
                     " | ".join(f"{res['per_source'][s]['f1_macro']:.4f}" for s in sources) + " |")

    for name, res in results.items():
        lines.append(f"\n## Classification report - {name}\n```\n{res['report']}```\n")

    lines.append("\n## Delivered\n")
    lines.append(f"- CSV: `{csv_path.as_posix()}`")
    lines.append(f"- This MD: `{md_path.as_posix()}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD : {md_path}")


if __name__ == "__main__":
    main()
