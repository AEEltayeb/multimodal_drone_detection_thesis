"""Build real grayscale training rows from video test cached detections,
append to v3more dataset, retrain, and evaluate."""
import sys, json, time, re
from pathlib import Path
import cv2, numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ir_gui"))
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

CACHE = REPO / "docs" / "analysis" / "full_pipeline_ablations" / "cache"
VID_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
DATA = REPO / "classifier" / "runs" / "reliability" / "fusion"
EXTS = {".jpg",".jpeg",".png",".bmp"}

def list_imgs(d):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []

def read_gt(lbl_path, w, h):
    boxes = []
    if not lbl_path.exists(): return boxes
    for line in lbl_path.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5: continue
        cx,cy,bw,bh = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
        boxes.append([
            (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h
        ])
    return boxes

def iop(det, gt):
    x1=max(det[0],gt[0]); y1=max(det[1],gt[1])
    x2=min(det[2],gt[2]); y2=min(det[3],gt[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    da = max(1e-6,(det[2]-det[0])*(det[3]-det[1]))
    return inter/da

def has_tp(dets, gt_boxes, thr=0.5):
    for d in dets:
        for g in gt_boxes:
            if iop(d[:4], g) >= thr:
                return True
    return False

# Collect clips
clips = []
for cat in ("drone","birds","airplanes","helicopters"):
    cat_dir = VID_ROOT / cat
    if not cat_dir.exists(): continue
    for clip in sorted(cat_dir.iterdir()):
        if not clip.is_dir(): continue
        img_dir = clip/"images"/"test" if (clip/"images"/"test").exists() else clip/"images"
        lbl_dir = clip/"labels"/"test" if (clip/"labels"/"test").exists() else clip/"labels"
        tag = f"video_{cat}_{clip.name}"
        rgb_cache = CACHE / f"{tag}_baseline_sz1280.json"
        ir_cache = CACHE / f"{tag}_ir_grayscale_sz640.json"
        if img_dir.exists() and rgb_cache.exists() and ir_cache.exists():
            clips.append((tag, img_dir, lbl_dir, rgb_cache, ir_cache))
print(f"Found {len(clips)} clips with caches")

rows = []
for tag, img_dir, lbl_dir, rgb_cache_path, ir_cache_path in clips:
    rgb_c = json.load(open(rgb_cache_path))["dets"]
    ir_c = json.load(open(ir_cache_path))["dets"]
    imgs = list_imgs(img_dir)
    n = 0
    for img_path in imgs:
        stem = img_path.stem
        rgb_dets_raw = rgb_c.get(stem, [])
        ir_dets_raw = ir_c.get(stem, [])
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gt = read_gt(lbl_dir / f"{stem}.txt", w, h)
        rgb_dets = [d for d in rgb_dets_raw if d[4] >= 0.25]
        ir_dets = [d for d in ir_dets_raw if d[4] >= 0.40]
        
        # Trust label (grayscale: both modalities see same image)
        rgb_tp = has_tp(rgb_dets, gt)
        ir_tp = has_tp(ir_dets, gt)
        drone_present = 1 if gt else 0
        if rgb_tp and ir_tp: trust = 3
        elif rgb_tp: trust = 1
        elif ir_tp: trust = 2
        else: trust = 0
        
        # Features (same 32 as sa32)
        row = {}
        # Detection conf
        rgb_confs = [d[4] for d in rgb_dets]
        ir_confs = [d[4] for d in ir_dets]
        row["rgb_max_conf"] = max(rgb_confs) if rgb_confs else 0.0
        row["rgb_mean_conf"] = float(np.mean(rgb_confs)) if rgb_confs else 0.0
        row["ir_max_conf"] = max(ir_confs) if ir_confs else 0.0
        row["ir_mean_conf"] = float(np.mean(ir_confs)) if ir_confs else 0.0
        # Scene globals (both from grayscale — this IS the grayscale reality)
        g = compute_global_features(gray)
        row.update({f"rgb_{k}": v for k, v in g.items()})
        row.update({f"ir_{k}": v for k, v in g.items()})  # same image!
        # Target features
        for pfx, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
            if not dets:
                row.update({f"{pfx}_best_{k}": 0.0 for k in TARGET_NAMES})
            else:
                best = max(dets, key=lambda d: d[4])
                tf = compute_target_features(gray, best[:4], w, h)
                row.update({f"{pfx}_best_{k}": v for k, v in tf.items()})
        # Metadata
        row["base_stem"] = stem
        row["source_dataset"] = tag
        row["drone_present"] = drone_present
        row["rgb_has_tp"] = int(rgb_tp)
        row["ir_has_tp"] = int(ir_tp)
        row["trust_label"] = trust
        rows.append(row)
        n += 1
    print(f"  {tag}: {n} rows (trust dist: " + 
          ", ".join(f"{l}={sum(1 for r in rows[-n:] if r['trust_label']==l)}" for l in range(4)) + ")")

gray_df = pd.DataFrame(rows)
print(f"\nTotal grayscale rows: {len(gray_df)}")
print(f"Trust distribution:\n{gray_df['trust_label'].value_counts().sort_index().to_string()}")

# Load original paired data and combine
orig = pd.read_csv(DATA / "fusion_dataset_v3more.csv")
print(f"Original paired: {len(orig):,}")

# Ensure columns match (fill missing with 0)
for c in orig.columns:
    if c not in gray_df.columns:
        gray_df[c] = 0
gray_df = gray_df[orig.columns]  # same column order

combined = pd.concat([orig, gray_df], ignore_index=True)
out_path = DATA / "fusion_dataset_v3more_realgray.csv"
combined.to_csv(out_path, index=False)
print(f"Combined: {len(combined):,} rows -> {out_path.name}")
