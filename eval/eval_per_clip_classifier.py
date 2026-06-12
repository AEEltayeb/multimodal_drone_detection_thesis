"""eval_per_clip_classifier.py - Per-clip evaluation of every trained
classifier on every drone-video-tests clip.

For each (classifier, clip) pair:
  - Compute features for ALL frames in the clip
  - Build trust labels via GT IoP-0.5 matching
  - Run classifier
  - Report acc, F1m, per-trust-class counts
  - Flag whether this clip was in that classifier's TRAIN or TEST split
    (determined by the classifier's training CSV + seed=42 sequence split)

Uses cached detections from docs/analysis/full_pipeline_ablations/cache/
so no YOLO inference is needed.
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import cv2, joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))
from generate_retrained_v2_data import (  # noqa: E402
    compute_global_features, compute_target_features,
    parse_yolo_gt, has_tp,
)

VID_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
CACHE_DIR = REPO / "docs/analysis/full_pipeline_ablations/cache"
VIDEO_RGB_TAG = "selcom_1280_sz1280"
IR_TAG = "ir_grayscale_sz640"
LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASSIFIERS = [
    ("lean10",  "models/routers/lean10/model.joblib",
                "models/routers/lean13/fusion_dataset_lean13.csv"),
    ("lean13",  "models/routers/lean13/model.joblib",
                "models/routers/lean13/fusion_dataset_lean13.csv"),
    ("lean17",  "models/routers/lean17/model.joblib",
                "models/routers/lean17/fusion_dataset_lean17.csv"),
    ("lean19",  "models/routers/lean19/model.joblib",
                "models/routers/lean19/fusion_dataset_lean19.csv"),
    ("32feat",  "models/routers/retrained_v2_32feat/model.joblib", None),
    ("40feat",  "models/routers/control_v3more_40feat/model.joblib", None),
]
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def seq_id(stem, source):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{source}::{base}"


def split_membership(csv_path, source_tag):
    """Return 'train' or 'test' for a given source_tag based on seed-42 split of csv."""
    if csv_path is None: return "n/a"
    df = pd.read_csv(csv_path)
    df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))
    test_sources = set(df.iloc[test_idx]["source"].unique())
    if source_tag in test_sources:
        return "test"
    if source_tag in set(df["source"].unique()):
        return "train"
    return "absent"


def build_superset_row(rgb_dets, ir_dets, rgb_gray, ir_gray, rgb_wh, ir_wh, conf=0.25):
    rgb_dets = [d for d in rgb_dets if d[4] >= conf]
    ir_dets = [d for d in ir_dets if d[4] >= conf]
    rw, rh = rgb_wh; iw, ih = ir_wh
    rgb_confs = [d[4] for d in rgb_dets]; ir_confs = [d[4] for d in ir_dets]
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
        for k, v in compute_target_features(rgb_gray, b[:4], rw, rh).items():
            row[f"rgb_best_{k}"] = v
    else:
        for k in ["log_bbox_area","aspect_ratio","pos_x","pos_y","dist_to_center","local_contrast","target_bg_delta"]:
            row[f"rgb_best_{k}"] = 0.0
    if ir_dets:
        b = max(ir_dets, key=lambda d: d[4])
        for k, v in compute_target_features(ir_gray, b[:4], iw, ih).items():
            row[f"ir_best_{k}"] = v
    else:
        for k in ["log_bbox_area","aspect_ratio","pos_x","pos_y","dist_to_center","local_contrast","target_bg_delta"]:
            row[f"ir_best_{k}"] = 0.0
    rd = 1 if rgb_dets else 0; idet = 1 if ir_dets else 0
    row["rgb_n_dets"] = len(rgb_dets); row["ir_n_dets"] = len(ir_dets)
    row["rgb_detected"] = rd; row["ir_detected"] = idet
    row["both_detect"] = 1 if (rd and idet) else 0
    row["neither_detect"] = 1 if (not rd and not idet) else 0
    row["rgb_only_detect"] = 1 if (rd and not idet) else 0
    row["ir_only_detect"] = 1 if (not rd and idet) else 0
    return row


def trust_label(rgb_dets, ir_dets, rgb_gt, ir_gt, mode="iop"):
    r = has_tp(rgb_dets, rgb_gt, mode=mode)
    i = has_tp(ir_dets, ir_gt, mode=mode)
    if r and i: return 3
    if r: return 1
    if i: return 2
    return 0


def list_clips():
    out = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cd = VID_ROOT / cat
        if not cd.exists(): continue
        for clip in sorted(cd.iterdir()):
            if not clip.is_dir(): continue
            img_d = clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
            lbl_d = clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
            tag = f"video_{cat}_{clip.name}"
            rgb_c = CACHE_DIR / f"{tag}_{VIDEO_RGB_TAG}.json"
            ir_c = CACHE_DIR / f"{tag}_{IR_TAG}.json"
            if not (img_d.exists() and rgb_c.exists() and ir_c.exists()): continue
            out.append((cat, clip.name, tag, img_d, lbl_d, rgb_c, ir_c))
    return out


def build_clip_dataframe(tag, img_d, lbl_d, rgb_c_path, ir_c_path):
    rgb_dets_by_stem = json.load(open(rgb_c_path))["dets"]
    ir_dets_by_stem = json.load(open(ir_c_path))["dets"]
    rows = []
    for ip in sorted(img_d.iterdir()):
        if ip.suffix.lower() not in IMG_EXTS: continue
        stem = ip.stem
        img = cv2.imread(str(ip))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = parse_yolo_gt(lbl_d / f"{stem}.txt", w, h)
        rd = rgb_dets_by_stem.get(stem, [])
        id_ = ir_dets_by_stem.get(stem, [])
        label = trust_label(rd, id_, gt, gt, "iop")
        row = build_superset_row(rd, id_, gray, gray, (w, h), (w, h))
        row["trust_label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    clips = list_clips()
    print(f"Found {len(clips)} clips with caches\n")

    # Load all classifiers
    bundles = {}
    membership = {}  # (clf_name, source_tag) -> 'train'/'test'/'absent'
    for name, mp, csvp in CLASSIFIERS:
        p = REPO / mp
        if not p.exists():
            print(f"  [skip {name}] {p} missing"); continue
        b = joblib.load(p)
        feats = b.get("features") or b.get("feature_names")
        bundles[name] = (b["model"], list(feats))
        for cat, clip, tag, *_ in clips:
            membership[(name, tag)] = split_membership(REPO / csvp, tag) if csvp else "n/a"

    # Build per-clip features once
    results = defaultdict(dict)  # results[clf][tag] = {acc, f1m, n, pred_dist}
    clip_meta = {}  # tag -> {category, n_frames, gt_dist}
    for cat, clip, tag, img_d, lbl_d, rgb_c, ir_c in clips:
        df = build_clip_dataframe(tag, img_d, lbl_d, rgb_c, ir_c)
        if df.empty:
            print(f"  [skip {tag}] no frames"); continue
        clip_meta[tag] = {
            "category": cat, "n_frames": len(df),
            "gt_dist": dict(df["trust_label"].value_counts().sort_index()),
        }
        for name, (model, feats) in bundles.items():
            missing = [c for c in feats if c not in df.columns]
            if missing:
                print(f"  [skip {name} on {tag}] missing feature cols: {missing}"); continue
            X = df[feats].values
            y = df["trust_label"].values
            pred = model.predict(X)
            acc = float(accuracy_score(y, pred))
            f1m = float(f1_score(y, pred, average="macro", zero_division=0))
            results[name][tag] = {
                "n": len(df), "acc": acc, "f1m": f1m,
                "pred_dist": dict(pd.Series(pred).value_counts().sort_index()),
                "split": membership.get((name, tag), "n/a"),
            }
        print(f"  {tag}: n={len(df)} gt={clip_meta[tag]['gt_dist']}  done")

    # Save CSV
    out_csv = REPO / "docs/analysis/full_pipeline_ablations/csv/classifier_per_clip.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in sorted(clip_meta.keys()):
        m = clip_meta[tag]
        for clf in bundles.keys():
            r = results[clf].get(tag, {})
            rows.append({
                "clip": tag, "category": m["category"], "n_frames": m["n_frames"],
                "gt_drone_pos": m["gt_dist"].get(3, 0) + m["gt_dist"].get(1, 0) + m["gt_dist"].get(2, 0),
                "classifier": clf,
                "split": r.get("split", "?"),
                "acc": f"{r.get('acc', float('nan')):.4f}",
                "f1m": f"{r.get('f1m', float('nan')):.4f}",
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n  CSV: {out_csv}")

    # Markdown summary
    out_md = REPO / "docs/analysis" / f"{pd.Timestamp.today().strftime('%Y-%m-%d')}_classifier_per_clip.md"
    lines = [f"# Per-clip classifier evaluation ({pd.Timestamp.today().strftime('%Y-%m-%d')})\n"]
    lines.append("Each classifier evaluated on every drone-video-tests clip. `split` column shows whether that clip was in TRAIN or TEST for that specific classifier (per seed=42 GroupShuffleSplit of its own training CSV); `n/a` for 32-feat/40-feat which used a different training corpus.\n")
    # Acc table
    lines.append("\n## Per-clip accuracy\n")
    header = "| clip | category | n | gt_pos | " + " | ".join(bundles.keys()) + " |"
    sep = "|---|---|---:|---:|" + "|".join(":---:" for _ in bundles) + "|"
    lines.append(header); lines.append(sep)
    for tag in sorted(clip_meta.keys()):
        m = clip_meta[tag]
        gp = m["gt_dist"].get(3, 0) + m["gt_dist"].get(1, 0) + m["gt_dist"].get(2, 0)
        cells = []
        for clf in bundles.keys():
            r = results[clf].get(tag, {})
            split_mark = {"train": "🅣", "test": "🅗", "absent": "—", "n/a": ""}.get(r.get("split", "?"), "?")
            cells.append(f"{r.get('acc', float('nan')):.3f}{split_mark}")
        lines.append(f"| `{tag}` | {m['category']} | {m['n_frames']} | {gp} | " + " | ".join(cells) + " |")
    lines.append("\nLegend: 🅣 = clip in classifier's TRAIN split  🅗 = clip in classifier's TEST split  — = clip not in classifier's training data  (blank) = classifier trained on a different corpus\n")

    # F1m table
    lines.append("\n## Per-clip F1-macro\n")
    lines.append(header); lines.append(sep)
    for tag in sorted(clip_meta.keys()):
        m = clip_meta[tag]
        gp = m["gt_dist"].get(3, 0) + m["gt_dist"].get(1, 0) + m["gt_dist"].get(2, 0)
        cells = []
        for clf in bundles.keys():
            r = results[clf].get(tag, {})
            split_mark = {"train": "🅣", "test": "🅗", "absent": "—", "n/a": ""}.get(r.get("split", "?"), "?")
            cells.append(f"{r.get('f1m', float('nan')):.3f}{split_mark}")
        lines.append(f"| `{tag}` | {m['category']} | {m['n_frames']} | {gp} | " + " | ".join(cells) + " |")

    # Held-out summary
    lines.append("\n## Drone clips where the model never saw any frame (true OOD)\n")
    lines.append("| clip | n | " + " | ".join(bundles.keys()) + " |")
    lines.append("|---|---:|" + "|".join(":---:" for _ in bundles) + "|")
    for tag in sorted(clip_meta.keys()):
        if clip_meta[tag]["category"] != "drone": continue
        cells, ood_any = [], False
        for clf in bundles.keys():
            r = results[clf].get(tag, {})
            if r.get("split") in ("test", "absent"):
                ood_any = True
                cells.append(f"{r.get('acc', float('nan')):.3f} ({r.get('split','?')})")
            else:
                cells.append(f"_train_")
        if ood_any:
            lines.append(f"| `{tag}` | {clip_meta[tag]['n_frames']} | " + " | ".join(cells) + " |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD : {out_md}")


if __name__ == "__main__":
    main()
