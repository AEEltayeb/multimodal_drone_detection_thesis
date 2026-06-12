"""Step 3: Compare sa32 vs gray_aug classifier on all datasets.
Trust-aware scoring, stride-sampled to <=1500 images per dataset."""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from collections import Counter
import cv2, numpy as np, joblib
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "ir_gui"))
from metrics import score_detections, score_per_size, SIZE_BUCKETS, compute_prf
from datasets import read_yolo_labels
from fusion.features import (compute_global_features, compute_target_features,
                              TARGET_NAMES, _TRAIN_MEANS)

MAX_PER_DS = 1500

# ── Load both classifiers ──
def load_clf(path):
    obj = joblib.load(str(path))
    return obj["model"], obj.get("features") or obj.get("feat_cols") or []

CLF_DIR = REPO / "classifier" / "fusion_models" / "scene_aware_v3more_32feat"
GRAY_DIR = REPO / "classifier" / "runs" / "reliability" / "fusion"
sa32_model, sa32_feats = load_clf(CLF_DIR / "model.joblib")
# Find the gray_aug model
gray_model_path = GRAY_DIR / "fusion_no_fn_v3more_realgray_model.joblib"
if not gray_model_path.exists():
    print(f"[ERROR] {gray_model_path} not found. Run step2 first.")
    sys.exit(1)
gray_model, gray_feats = load_clf(gray_model_path)
print(f"sa32: {len(sa32_feats)} feats | gray_aug: {len(gray_feats)} feats")

# ── Feature builder ──
def build_feats(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    rh, rw = rgb_gray.shape[:2]
    ih, iw = ir_gray.shape[:2]
    feats = {}
    for pfx, dets in (("rgb", rgb_dets), ("ir", ir_dets)):
        confs = [c for _, c in dets]
        feats[f"{pfx}_max_conf"] = float(max(confs)) if confs else 0.0
        feats[f"{pfx}_mean_conf"] = float(np.mean(confs)) if confs else 0.0
    feats.update({f"rgb_{k}": v for k, v in compute_global_features(rgb_gray).items()})
    feats.update({f"ir_{k}": v for k, v in compute_global_features(ir_gray).items()})
    for pfx, dets, gray, gw, gh in (("rgb", rgb_dets, rgb_gray, rw, rh),
                                      ("ir", ir_dets, ir_gray, iw, ih)):
        if not dets:
            feats.update({f"{pfx}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best, gw, gh)
            feats.update({f"{pfx}_best_{k}": v for k, v in tf.items()})
    return feats

def feats_to_x(feats, feat_cols):
    return np.array([[feats.get(c, 0.0) for c in feat_cols]], dtype=np.float32)

def trust_kept(label, rgb_dets, ir_dets):
    if label == 0: return []
    if label == 1: return rgb_dets
    if label == 2: return ir_dets
    return rgb_dets + ir_dets

# ── Dataset configs ──
DATASETS = {
    "svanstrom": {
        "type": "paired",
        "rgb_dir": Path("G:/drone/svanstrom_paired/RGB/images"),
        "rgb_lbl": Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_dir": Path("G:/drone/svanstrom_paired/IR/images"),
        "ir_lbl": Path("G:/drone/svanstrom_paired/IR/labels"),
    },
    "antiuav": {
        "type": "paired",
        "rgb_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "rgb_lbl": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_dir": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "ir_lbl": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
    },
}
# Add video test clips as grayscale datasets
VID_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
for cat in ("drone", "birds", "airplanes", "helicopters"):
    cat_dir = VID_ROOT / cat
    if not cat_dir.exists(): continue
    for clip in sorted(cat_dir.iterdir()):
        if not clip.is_dir(): continue
        img_dir = clip / "images" / "test" if (clip / "images" / "test").exists() else clip / "images"
        lbl_dir = clip / "labels" / "test" if (clip / "labels" / "test").exists() else clip / "labels"
        if img_dir.exists():
            DATASETS[f"video_{cat}_{clip.name}"] = {
                "type": "grayscale", "rgb_dir": img_dir, "rgb_lbl": lbl_dir,
            }

def list_images(d):
    exts = {".jpg",".jpeg",".png",".bmp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []

def strip_suffix(stem):
    import re
    return re.sub(r"_(visible|infrared)", "", stem)

# ── YOLO models (for live inference) ──
from ultralytics import YOLO
RGB_YOLO = YOLO(str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"))
IR_YOLO = YOLO(str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"))

# ── Main evaluation loop ──
results = {}
for ds_name, ds_cfg in DATASETS.items():
    is_paired = ds_cfg["type"] == "paired"
    rgb_imgs = list_images(ds_cfg["rgb_dir"])
    if not rgb_imgs:
        continue
    # Stride
    if len(rgb_imgs) > MAX_PER_DS:
        stride = max(1, len(rgb_imgs) // MAX_PER_DS)
        rgb_imgs = rgb_imgs[::stride]
    
    # For paired: build stem map
    ir_map = {}
    if is_paired:
        ir_imgs = list_images(ds_cfg["ir_dir"])
        ir_map = {strip_suffix(p.stem): p for p in ir_imgs}

    counters = {clf_name: {"labels": Counter(), "tp":0,"fp":0,"fn":0}
                for clf_name in ("sa32", "gray_aug")}
    n_frames = 0
    t0 = time.time()
    
    for idx, rgb_path in enumerate(rgb_imgs):
        img = cv2.imread(str(rgb_path))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # GT
        lbl = ds_cfg["rgb_lbl"] / f"{rgb_path.stem}.txt"
        gts = read_yolo_labels(lbl, w, h, drone_classes={0})
        
        # RGB dets
        res = RGB_YOLO.predict(img, imgsz=640, conf=0.25, device="0", verbose=False)
        rgb_dets = []
        if res[0].boxes is not None and len(res[0].boxes) > 0:
            for i in range(len(res[0].boxes)):
                b = res[0].boxes.xyxy[i].cpu().numpy()
                c = float(res[0].boxes.conf[i])
                rgb_dets.append((tuple(map(float, b)), c))
        
        # IR dets + IR gray
        if is_paired:
            base = strip_suffix(rgb_path.stem)
            ir_path = ir_map.get(base)
            if ir_path is None: continue
            ir_img = cv2.imread(str(ir_path))
            if ir_img is None: continue
            ir_gray_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
            ir_input = ir_img
            ir_gts = read_yolo_labels(
                ds_cfg["ir_lbl"] / f"{ir_path.stem}.txt",
                ir_img.shape[1], ir_img.shape[0], drone_classes={0})
        else:
            ir_gray_img = gray
            ir_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            ir_gts = gts  # same GT for grayscale
        
        res_ir = IR_YOLO.predict(ir_input, imgsz=640, conf=0.40, device="0", verbose=False)
        ir_dets = []
        if res_ir[0].boxes is not None and len(res_ir[0].boxes) > 0:
            for i in range(len(res_ir[0].boxes)):
                b = res_ir[0].boxes.xyxy[i].cpu().numpy()
                c = float(res_ir[0].boxes.conf[i])
                ir_dets.append((tuple(map(float, b)), c))
        
        feats = build_feats(rgb_dets, ir_dets, gray, ir_gray_img, sa32_feats)
        n_frames += 1
        
        for clf_name, model, fc in [("sa32", sa32_model, sa32_feats),
                                      ("gray_aug", gray_model, gray_feats)]:
            x = feats_to_x(feats, fc)
            try:
                label = int(model.predict(x)[0])
            except:
                label = 3
            counters[clf_name]["labels"][label] += 1
            
            # Trust-aware scoring
            if is_paired:
                # Score RGB side if label in {0,1,3}
                if label in (0, 1, 3):
                    kept_rgb = rgb_dets if label in (1,3) else []
                    tp,fp,fn = score_detections(kept_rgb, gts, rule="iop", iop_thr=0.5)
                    counters[clf_name]["tp"] += tp
                    counters[clf_name]["fp"] += fp
                    counters[clf_name]["fn"] += fn
                if label in (0, 2, 3):
                    kept_ir = ir_dets if label in (2,3) else []
                    tp,fp,fn = score_detections(kept_ir, ir_gts, rule="iop", iop_thr=0.5)
                    counters[clf_name]["tp"] += tp
                    counters[clf_name]["fp"] += fp
                    counters[clf_name]["fn"] += fn
            else:
                kept = trust_kept(label, rgb_dets, ir_dets)
                tp,fp,fn = score_detections(kept, gts, rule="iop", iop_thr=0.5)
                counters[clf_name]["tp"] += tp
                counters[clf_name]["fp"] += fp
                counters[clf_name]["fn"] += fn
        
        if (idx+1) % 200 == 0:
            print(f"  [{ds_name}] {idx+1}/{len(rgb_imgs)} {(idx+1)/(time.time()-t0):.1f} fps")
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DATASET: {ds_name} ({n_frames} frames, {elapsed:.0f}s)")
    print(f"{'='*70}")
    
    ds_result = {"n_frames": n_frames, "type": ds_cfg["type"]}
    for clf_name in ("sa32", "gray_aug"):
        c = counters[clf_name]
        m = compute_prf(c["tp"], c["fp"], c["fn"])
        rej = c["labels"][0]
        rej_pct = rej/n_frames*100 if n_frames > 0 else 0
        print(f"  {clf_name:12s} TP={c['tp']:>5d} FP={c['fp']:>5d} FN={c['fn']:>5d} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
              f"rej={rej}/{n_frames} ({rej_pct:.1f}%)")
        ds_result[clf_name] = {
            "tp":c["tp"],"fp":c["fp"],"fn":c["fn"],
            "P":m["precision"],"R":m["recall"],"F1":m["f1"],
            "reject_rate": round(rej_pct/100,4),
            "labels": dict(c["labels"]),
        }
    results[ds_name] = ds_result

# ── Aggregate summaries ──
print(f"\n{'='*70}")
print("AGGREGATE SUMMARY")
print(f"{'='*70}")
for group, prefix in [("paired", ""), ("grayscale", "video_")]:
    ds_list = [k for k,v in results.items()
               if (group=="paired" and v["type"]=="paired") or
                  (group=="grayscale" and v["type"]=="grayscale")]
    if not ds_list: continue
    for clf_name in ("sa32", "gray_aug"):
        tp = sum(results[d][clf_name]["tp"] for d in ds_list)
        fp = sum(results[d][clf_name]["fp"] for d in ds_list)
        fn = sum(results[d][clf_name]["fn"] for d in ds_list)
        m = compute_prf(tp, fp, fn)
        n = sum(results[d]["n_frames"] for d in ds_list)
        print(f"  {group:10s} {clf_name:12s} n={n:>5d} TP={tp:>5d} FP={fp:>5d} FN={fn:>5d} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

out = REPO / "eval" / "results" / "gray_aug_clf_comparison.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
print("Step 3 done.")
