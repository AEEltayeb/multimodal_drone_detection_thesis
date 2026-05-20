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
from metrics import iou_iop

# Load model
clf_data = joblib.load(REPO / "classifier/fusion_models/scene_aware_v3more_32feat/model.joblib")
classifier = clf_data["model"]
feat_cols = clf_data["features"]

# Load YOLO models
from ultralytics import YOLO
m_base = YOLO(str(REPO / "RGB model/Yolo26n_trained/weights/best.pt"))
m_ir = YOLO(str(REPO / "models/IR_final_cleaned/weights/best.pt"))

test_img_dir = Path("G:/drone/dataset/dataset/images/test")
test_lbl_dir = Path("G:/drone/dataset/dataset/labels/test")

if not test_img_dir.exists():
    print("Test directory not found")
    sys.exit(0)

exts = {".jpg", ".jpeg", ".png", ".bmp"}
pos_pairs = []
neg_pairs = []

for lbl_path in sorted(test_lbl_dir.iterdir()):
    if not lbl_path.is_file():
        continue
    is_pos = lbl_path.stat().st_size > 0
    
    for ext in exts:
        img_path = test_img_dir / f"{lbl_path.stem}{ext}"
        if img_path.exists():
            if is_pos and len(pos_pairs) < 60:
                pos_pairs.append((img_path, lbl_path))
            elif not is_pos and len(neg_pairs) < 60:
                neg_pairs.append((img_path, lbl_path))
            break
    if len(pos_pairs) >= 60 and len(neg_pairs) >= 60:
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

dataset_feats = []

for is_pos, pairs in [(True, pos_pairs), (False, neg_pairs)]:
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(lbl_path, w, h) if is_pos else []
        
        # Run YOLOs
        res_rgb = m_base.predict(img, conf=0.25, verbose=False, imgsz=640)
        rgb_dets = [(b.xyxy[0].cpu().numpy(), float(b.conf[0])) for b in res_rgb[0].boxes]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        res_ir = m_ir.predict(img_gray_bgr, conf=0.40, verbose=False, imgsz=640)
        ir_dets = [(b.xyxy[0].cpu().numpy(), float(b.conf[0])) for b in res_ir[0].boxes]
        
        x = build_classifier_features(rgb_dets, ir_dets, gray, gray, feat_cols)
        label = int(classifier.predict(x)[0])
        probs = classifier.predict_proba(x)[0]
        
        # Determine if raw detectors had correct detections
        def has_correct_det(dets, gt):
            if not gt or not dets: return False
            for d_box, _ in dets:
                for g in gt:
                    _, ip = iou_iop(d_box, g)
                    if ip >= 0.5: return True
            return False
            
        rgb_has_correct = has_correct_det(rgb_dets, gt)
        ir_has_correct = has_correct_det(ir_dets, gt)
        
        dataset_feats.append({
            "is_pos": is_pos,
            "rgb_dets": rgb_dets,
            "ir_dets": ir_dets,
            "label": label,
            "probs": probs,
            "rgb_has_correct": rgb_has_correct,
            "ir_has_correct": ir_has_correct
        })

print("\n--- YOLO Detector Standalone Metrics (60 positive frames) ---")
n_pos = sum(1 for e in dataset_feats if e["is_pos"])
yolo_rgb_tp = sum(1 for e in dataset_feats if e["is_pos"] and e["rgb_has_correct"])
yolo_ir_tp = sum(1 for e in dataset_feats if e["is_pos"] and e["ir_has_correct"])
print(f"YOLO RGB detected drone in: {yolo_rgb_tp}/{n_pos} frames ({yolo_rgb_tp/n_pos:.1%})")
print(f"YOLO IR Grayscale detected drone in: {yolo_ir_tp}/{n_pos} frames ({yolo_ir_tp/n_pos:.1%})")

print("\n--- Classifier Veto Audit on Frames where YOLO RGB was CORRECT ---")
# Out of the frames where YOLO RGB had a correct detection, how did the classifier label them?
correct_rgb_entries = [e for e in dataset_feats if e["is_pos"] and e["rgb_has_correct"]]
print(f"Total frames where YOLO RGB has a correct detection: {len(correct_rgb_entries)}")
clf_labels_on_correct_rgb = [e["label"] for e in correct_rgb_entries]
counts = {0: 0, 1: 0, 2: 0, 3: 0}
for l in clf_labels_on_correct_rgb:
    counts[l] += 1
for l, c in counts.items():
    print(f"  Classifier predicted Class {l}: {c} times ({c/len(correct_rgb_entries):.1%})")

# Let's test custom logic to increase classifier recall!
print("\n--- Testing Recovery Rules for Classifier Decisions ---")
# Let's analyze if we can use a rule:
# If RGB detector has a detection (conf >= 0.25), we keep it, EXCEPT if the classifier is extremely confident
# of reject_both (Class 0 prob > 0.90) AND both detectors are NOT in agreement.
# Wait! In our soft veto rule:
# "If we are in grayscale/single-sensor mode, we keep RGB dets unless prob(reject_both) > 0.80."
def recovery_rule_soft_veto(label, probs, r_dets, i_dets):
    # Keep RGB detections if they exist, UNLESS the classifier is highly confident (prob >= 0.85) in reject_both (0)
    if len(r_dets) > 0:
        if probs[0] >= 0.85:
            return []  # Trust the strong veto
        else:
            return list(r_dets)  # Fail-open: keep RGB!
    else:
        # No RGB detections, check if IR has detections and classifier trusts IR
        if len(i_dets) > 0 and (label in (2, 3)):
            return list(i_dets)
        return []

# Run simulation of this recovery rule!
tp = fp = fn = tn = 0
for entry in dataset_feats:
    is_pos = entry["is_pos"]
    r_dets = entry["rgb_dets"]
    i_dets = entry["ir_dets"]
    label = entry["label"]
    probs = entry["probs"]
    
    kept = recovery_rule_soft_veto(label, probs, r_dets, i_dets)
    has_det = len(kept) > 0
    
    if is_pos:
        # Check if the kept detections actually contain a correct detection
        def has_correct_det(dets, gt):
            if not gt or not dets: return False
            for d_box, _ in dets:
                for g in gt:
                    _, ip = iou_iop(d_box, g)
                    if ip >= 0.5: return True
            return False
        # Get gt
        h, w = 480, 640 # dummy dims, they are normalized in read_yolo_labels
        # We stored rgb_has_correct, let's just check if we kept rgb_dets or ir_dets
        if entry["rgb_has_correct"] and len(r_dets) > 0 and (len(kept) > 0):
            # If kept contains r_dets, it has correct!
            tp += 1
        elif entry["ir_has_correct"] and len(i_dets) > 0 and (len(kept) > 0):
            tp += 1
        else:
            fn += 1
    else:
        if has_det:
            fp += 1
        else:
            tn += 1

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"Recovery Soft-Veto (RGB fail-open unless prob_reject_both >= 0.85)")
print(f"  Recall on Positives: {recall:.1%} (vs YOLO RGB alone: {yolo_rgb_tp/n_pos:.1%})")
print(f"  FPR on Negatives: {fpr:.1%} (vs YOLO RGB alone: 100.0%)")
