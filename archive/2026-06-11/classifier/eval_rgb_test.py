"""eval_rgb_test.py — Evaluate grayscale vs sa32 on RGB dataset test (G drive)"""
import json, sys, os, cv2
import numpy as np, pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\ir_gui")))
from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

REPO = Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection")
CACHE_DIR = REPO / "docs/analysis/full_pipeline_ablations/cache"
IMG_DIR = Path(r"G:\drone\dataset\dataset\images\test")
LBL_DIR = Path(r"G:\drone\dataset\dataset\labels\test")

SA32_PATH = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
GRAY_PATH = REPO / "models/routers/split_v3/classifier_grayscale.joblib"

SA32_FEATURES = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio", "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio", "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]

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
            if iop(d[:4], g) >= thr: return True
    return False

def main():
    print("Loading models...")
    sa32_data = joblib.load(SA32_PATH)
    sa32_model = sa32_data.get("model", sa32_data)
    
    gray_data = joblib.load(GRAY_PATH)
    gray_model = gray_data["model"]
    gray_feats = gray_data["features"]

    rgb_cache = json.load(open(CACHE_DIR / "rgb_test_selcom_960_sz960.json"))["dets"]
    ir_cache = json.load(open(CACHE_DIR / "rgb_test_ir_grayscale_sz640.json"))["dets"]

    rows = []
    stems = list(rgb_cache.keys())
    print(f"Processing {len(stems)} cached stems...")
    
    for i, stem in enumerate(stems):
        if i % 100 == 0: print(f"  {i}/{len(stems)}")
        img_path = IMG_DIR / f"{stem}.jpg"
        if not img_path.exists():
            img_path = IMG_DIR / f"{stem}.png"
            if not img_path.exists(): continue
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gt = read_gt(LBL_DIR / f"{stem}.txt", w, h)
        rgb_dets_raw = rgb_cache.get(stem, [])
        ir_dets_raw = ir_cache.get(stem, [])
        
        rgb_dets = [d for d in rgb_dets_raw if d[4] >= 0.25]
        ir_dets = [d for d in ir_dets_raw if d[4] >= 0.40]
        
        rgb_tp = has_tp(rgb_dets, gt)
        ir_tp = has_tp(ir_dets, gt)
        
        if rgb_tp and ir_tp: trust = 3
        elif rgb_tp: trust = 1
        elif ir_tp: trust = 2
        else: trust = 0
        
        row = {"stem": stem, "trust_label": trust}
        
        rgb_confs = [d[4] for d in rgb_dets]
        ir_confs = [d[4] for d in ir_dets]
        row["rgb_max_conf"] = max(rgb_confs) if rgb_confs else 0.0
        row["rgb_mean_conf"] = float(np.mean(rgb_confs)) if rgb_confs else 0.0
        row["ir_max_conf"] = max(ir_confs) if ir_confs else 0.0
        row["ir_mean_conf"] = float(np.mean(ir_confs)) if ir_confs else 0.0
        
        g = compute_global_features(gray)
        row.update({f"rgb_{k}": v for k, v in g.items()})
        row.update({f"ir_{k}": v for k, v in g.items()})
        
        for pfx, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
            if not dets:
                row.update({f"{pfx}_best_{k}": 0.0 for k in TARGET_NAMES})
            else:
                best = max(dets, key=lambda d: d[4])
                tf = compute_target_features(gray, best[:4], w, h)
                row.update({f"{pfx}_best_{k}": v for k, v in tf.items()})
                
        row["area_diff"] = abs(row["rgb_best_log_bbox_area"] - row["ir_best_log_bbox_area"])
        row["xmodal_centroid_dist"] = 0.0
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df["source"] = "rgb_dataset_test"
    df.to_csv(CACHE_DIR / "fusion_dataset_rgb_test.csv", index=False)
    print(f"\nExtracted features for {len(df)} images and saved to {CACHE_DIR / 'fusion_dataset_rgb_test.csv'}")
    print(f"Trust dist:\n{df['trust_label'].value_counts().sort_index()}")
    
    y_true = df["trust_label"].values
    
    # SA32
    y_sa32 = sa32_model.predict(df[SA32_FEATURES].values)
    sa32_f1 = f1_score(y_true, y_sa32, average="macro", zero_division=0)
    
    # New Grayscale
    y_new = gray_model.predict(df[gray_feats].values)
    new_f1 = f1_score(y_true, y_new, average="macro", zero_division=0)
    
    print("\n" + "="*60)
    print("Grayscale fallback on RGB Test Dataset (G Drive)")
    print("="*60)
    print(f"Dataset Size: {len(df)} images")
    print(f"SA32 (production) F1-Macro:  {sa32_f1:.4f}")
    print(f"New Grayscale F1-Macro:      {new_f1:.4f}")
    print(f"Delta:                       {new_f1 - sa32_f1:+.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
