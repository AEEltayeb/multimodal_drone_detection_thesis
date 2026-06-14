import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Load ultralytics without printing header
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
from ultralytics import YOLO

# Add eval dir to path so we can import hook
sys.path.append(str(Path('eval').resolve()))
from overnight_confuser_distill import DetectInputHook, _extract_detection_features, IOU_THR

def get_class_samples(model, hook, img_paths, max_samples=50):
    features = []
    for p in img_paths:
        if len(features) >= max_samples:
            break
        img_bgr = cv2.imread(str(p))
        if img_bgr is None: continue
        ih, iw = img_bgr.shape[:2]
        
        hook.clear()
        results = model.predict(img_bgr, imgsz=1280, conf=0.25, verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0: continue
        
        # Take the highest conf detection
        best_idx = torch.argmax(boxes.conf).item()
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        conf = float(boxes.conf[best_idx])
        
        feat = _extract_detection_features(hook, tuple(xyxy), (ih, iw), conf)
        # return just the YOLO features (last 256 dims)
        features.append(feat[5:])
    
    if not features:
        return np.zeros(256)
    return np.mean(features, axis=0)

def main():
    print("Loading model...")
    model_path = "models/rgb/Yolo26n_selcom_mixed_ft3_1280/weights/best.pt"
    model = YOLO(model_path)
    hook = DetectInputHook()
    hook.register(model)
    
    print("Collecting images...")
    confuser_dir = Path("G:/drone/rgb_confusers_merged/images/train")
    drone_dir = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images")
    
    airplanes = list(confuser_dir.glob("airplane_*.jpg"))
    birds = list(confuser_dir.glob("raihanbird_*.jpg"))
    helis = list(confuser_dir.glob("heli_*.jpg"))
    drones = list(drone_dir.glob("*.jpg"))
    
    print(f"Found {len(airplanes)} airplanes, {len(birds)} birds, {len(helis)} helis, {len(drones)} drones.")
    
    print("Extracting features (Airplane)...")
    air_feat = get_class_samples(model, hook, airplanes)
    print("Extracting features (Bird)...")
    bird_feat = get_class_samples(model, hook, birds)
    print("Extracting features (Helicopter)...")
    heli_feat = get_class_samples(model, hook, helis)
    print("Extracting features (Drone)...")
    drone_feat = get_class_samples(model, hook, drones)
    
    features = [drone_feat, air_feat, bird_feat, heli_feat]
    labels = ["Drone", "Airplane", "Bird", "Helicopter"]
    
    # Sort neurons so the pattern is clearer (sort by drone vs average of confusers)
    avg_confuser = np.mean([air_feat, bird_feat, heli_feat], axis=0)
    diff = drone_feat - avg_confuser
    sort_idx = np.argsort(diff)[::-1]
    
    # Scale for visualization
    all_feats = np.array(features)[:, sort_idx]
    # Normalize each neuron to [0, 1] across classes for better contrast, or global?
    # Global scaling is better to show true magnitude
    vmin = np.min(all_feats)
    vmax = np.max(all_feats)
    all_feats = (all_feats - vmin) / (vmax - vmin)
    
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    
    # Draw neurons
    import matplotlib.cm as cm
    cmap = cm.get_cmap('coolwarm')
    
    # We will draw 4 columns
    x_coords = [1, 2, 3, 4]
    y_coords = np.linspace(0, 1, 256)
    
    # Draw connections (faint lines from left to right like in user image, but just visual flair)
    # We'll just draw the nodes to keep it clean.
    
    for i, (label, feat_vec) in enumerate(zip(labels, all_feats)):
        x = x_coords[i]
        for j, y in enumerate(y_coords):
            color = cmap(feat_vec[j])
            circle = plt.Circle((x, y), 0.15, color=color, ec='#333333', lw=0.5, zorder=2)
            ax.add_patch(circle)
            
    # Styling
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.02, 1.05)
    
    ax.set_xticks(x_coords)
    ax.set_xticklabels(labels, color='white', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.title("YOLO p5 Layer Activation (Mean of 50 samples)", color='white', fontsize=16, pad=20)
    
    out_path = Path("docs/analysis/images/layer_activation_classes.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, facecolor='#1e1e1e')
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
