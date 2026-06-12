import joblib
import cv2
import numpy as np
from pathlib import Path
import sys

REPO = Path("c:/Users/User/Desktop/UNISA projects/Drone detection/es proj 3 thesis workspace/es_drone_detection")
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "ir_gui"))

from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES
from datasets import read_yolo_labels

# Load model
clf_data = joblib.load(REPO / "models/routers/scene_aware_v3more_32feat/model.joblib")
classifier = clf_data["model"]
feat_cols = clf_data["features"]

# Load YOLO models
from ultralytics import YOLO
m_base = YOLO(str(REPO / "models/rgb/Yolo26n_trained/weights/best.pt"))
m_ir = YOLO(str(REPO / "models/ir/IR_final_cleaned/weights/best.pt"))

# Find drone-positive frames
test_img_dir = Path("G:/drone/dataset/dataset/images/test")
test_lbl_dir = Path("G:/drone/dataset/dataset/labels/test")

if not test_img_dir.exists():
    print("Test directory not found")
    sys.exit(0)

all_lbls = sorted(test_lbl_dir.iterdir())
pos_pairs = []
exts = {".jpg", ".jpeg", ".png", ".bmp"}

for lbl_path in all_lbls:
    if lbl_path.is_file() and lbl_path.stat().st_size > 0:
        # Check if corresponding image exists
        for ext in exts:
            img_path = test_img_dir / f"{lbl_path.stem}{ext}"
            if img_path.exists():
                pos_pairs.append((img_path, lbl_path))
                break
    if len(pos_pairs) >= 50:
        break

def build_classifier_features(rgb_dets, ir_dets, rgb_gray, ir_gray, feat_cols):
    rgb_h, rgb_w = rgb_gray.shape[:2]
    ir_h, ir_w = ir_gray.shape[:2]
    feats = {}

    for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
        confs = [c for _, c in dets]
        if not confs:
            feats.update({f"{prefix}_max_conf": 0.0, f"{prefix}_mean_conf": 0.0})
        else:
            feats.update({f"{prefix}_max_conf": round(max(confs), 6),
                          f"{prefix}_mean_conf": round(float(np.mean(confs)), 6)})

    rgb_global = compute_global_features(rgb_gray, modality="rgb")
    ir_global = compute_global_features(ir_gray, modality="ir")
    feats.update({f"rgb_{k}": v for k, v in rgb_global.items()})
    feats.update({f"ir_{k}": v for k, v in ir_global.items()})

    for prefix, dets, gray, gw, gh in [
        ("rgb", rgb_dets, rgb_gray, rgb_w, rgb_h),
        ("ir", ir_dets, ir_gray, ir_w, ir_h),
    ]:
        if not dets:
            feats.update({f"{prefix}_best_{k}": 0.0 for k in TARGET_NAMES})
        else:
            best_box = max(dets, key=lambda d: d[1])[0]
            tf = compute_target_features(gray, best_box, gw, gh)
            feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})

    return np.array([[feats.get(c, 0) for c in feat_cols]], dtype=np.float32)

print(f"Analyzing classifier predictions in grayscale mode on {len(pos_pairs)} DRONE-POSITIVE frames...")
counts = {0: 0, 1: 0, 2: 0, 3: 0}
prob_sums = np.zeros(4)
n_pos = 0

for img_path, lbl_path in pos_pairs:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    # Run YOLO RGB
    res_rgb = m_base.predict(img, conf=0.25, verbose=False, imgsz=640)
    rgb_dets = []
    for b in res_rgb[0].boxes:
        rgb_dets.append((b.xyxy[0].cpu().numpy(), float(b.conf[0])))
    
    # Run YOLO IR Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    res_ir = m_ir.predict(img_gray_bgr, conf=0.40, verbose=False, imgsz=640)
    ir_dets = []
    for b in res_ir[0].boxes:
        ir_dets.append((b.xyxy[0].cpu().numpy(), float(b.conf[0])))
        
    x = build_classifier_features(rgb_dets, ir_dets, gray, gray, feat_cols)
    label = int(classifier.predict(x)[0])
    probs = classifier.predict_proba(x)[0]
    
    counts[label] += 1
    prob_sums += probs
    n_pos += 1
    
    print(f"Frame {img_path.name}: Predicted Label={label} (Probs: reject_both={probs[0]:.3f}, trust_rgb={probs[1]:.3f}, trust_ir={probs[2]:.3f}, trust_both={probs[3]:.3f})")

print("\n--- Summary (Drone-Positive Frames) ---")
print(f"Total analyzed: {n_pos}")
print("Prediction counts:")
for l, c in counts.items():
    print(f"  Class {l}: {c} times ({c/n_pos:.1%})")
print("Average class probabilities:")
labels_str = ["reject_both", "trust_rgb", "trust_ir", "trust_both"]
for i, name in enumerate(labels_str):
    print(f"  {name}: {prob_sums[i]/n_pos:.3f}")
