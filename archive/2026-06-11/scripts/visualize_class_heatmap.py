import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def main():
    # Load the data we already extracted
    features = []
    # I'll just re-extract or I could have saved them. I'll just re-extract quickly.
    import cv2
    import torch
    import logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    from ultralytics import YOLO
    
    sys.path.append(str(Path('eval').resolve()))
    from overnight_confuser_distill import DetectInputHook, _extract_detection_features
    
    model_path = "models/rgb/Yolo26n_selcom_mixed_ft3_1280/weights/best.pt"
    model = YOLO(model_path)
    hook = DetectInputHook()
    hook.register(model)
    
    confuser_dir = Path("G:/drone/rgb_confusers_merged/images/train")
    drone_dir = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images")
    
    airplanes = list(confuser_dir.glob("airplane_*.jpg"))[:50]
    birds = list(confuser_dir.glob("raihanbird_*.jpg"))[:50]
    helis = list(confuser_dir.glob("heli_*.jpg"))[:50]
    drones = list(drone_dir.glob("*.jpg"))[:50]
    
    def get_feats(paths):
        feats = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is None: continue
            ih, iw = img.shape[:2]
            hook.clear()
            res = model.predict(img, imgsz=1280, conf=0.25, verbose=False, device="cuda")
            boxes = res[0].boxes
            if len(boxes) == 0: continue
            best = torch.argmax(boxes.conf).item()
            xyxy = boxes.xyxy[best].cpu().numpy()
            conf = float(boxes.conf[best])
            f = _extract_detection_features(hook, tuple(xyxy), (ih, iw), conf)[5:]
            feats.append(f)
        return np.mean(feats, axis=0) if feats else np.zeros(256)
    
    print("Extracting...")
    air_f = get_feats(airplanes)
    bird_f = get_feats(birds)
    heli_f = get_feats(helis)
    drone_f = get_feats(drones)
    
    # We want to find the top 20 most discriminative neurons between Drone and Confusers
    avg_conf = np.mean([air_f, bird_f, heli_f], axis=0)
    diff = np.abs(drone_f - avg_conf)
    top_indices = np.argsort(diff)[::-1][:20]
    
    data = np.vstack([drone_f[top_indices], air_f[top_indices], bird_f[top_indices], heli_f[top_indices]])
    labels = ["Drone", "Airplane", "Bird", "Helicopter"]
    
    # Normalize for better visualization (z-score across the 4 classes per neuron)
    data_norm = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-9)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(data_norm, cmap="coolwarm", center=0, 
                yticklabels=labels, xticklabels=[f"N{i}" for i in top_indices],
                cbar_kws={'label': 'Relative Activation (Z-Score)'},
                linewidths=1, linecolor='black')
    
    plt.title("Top 20 Discriminative Neurons Across Classes", pad=20, fontsize=14)
    plt.xlabel("Neuron Index", fontsize=12)
    plt.ylabel("Object Class", fontsize=12)
    
    plt.tight_layout()
    out_path = Path("docs/analysis/images/class_heatmap_clean.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
