import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch

# Append repo root and eval directory so we can import eval modules
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / "eval"))

from ultralytics import YOLO
from eval.distill_v2_domain_mixed import DetectInputHook, collect_predictions, MODEL_PATHS, ANTIUAV_VAL, SVANSTROM_DIR, SELCOM_VAL, CONFUSER_TRAIN

def main():
    print("Loading Base Detector...")
    model = YOLO(MODEL_PATHS["selcom_ft3_1280"])
    hook = DetectInputHook()
    handle = hook.register(model)

    datasets = [
        ("Anti-UAV", ANTIUAV_VAL, True),
        ("Svanstrom", SVANSTROM_DIR, True),
        ("Selcom", SELCOM_VAL, True),
        ("Web Confusers", CONFUSER_TRAIN, False)
    ]

    features = []
    labels = []
    domains = []

    print("\nExtracting features for visualization (max 200 per domain)...")
    for name, path, has_gt in datasets:
        if not path.exists():
            print(f"Skipping {name}, path not found.")
            continue
            
        X_tp, y_tp, _, X_fp, y_fp, meta_fp = collect_predictions(
            model, hook, path, stride=15, max_samples=200, has_gt=has_gt, category=name
        )
        
        # We only plot the YOLO features (columns 5:261), ignoring the metadata
        if len(X_tp) > 0:
            features.append(X_tp[:, 5:])
            labels.extend(["Drone"] * len(X_tp))
            domains.extend([name] * len(X_tp))
        if len(X_fp) > 0:
            features.append(X_fp[:, 5:])
            domains.extend([name] * len(X_fp))
            # Determine fine-grained class for Confusers
            for m in meta_fp:
                fname = Path(m["img"]).name.lower()
                if "bird" in fname:
                    labels.append("Bird")
                elif "airplane" in fname:
                    labels.append("Airplane")
                elif "helicopter" in fname:
                    labels.append("Helicopter")
                else:
                    labels.append("Background/Clutter")

    if not features:
        print("No features extracted.")
        return

    X_all = np.concatenate(features, axis=0)
    
    print("\nRunning PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    # Plot 1: Colored by Dataset Domain
    plt.figure(figsize=(10, 8))
    unique_domains = list(set(domains))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    
    for i, domain in enumerate(unique_domains):
        idx = [j for j, d in enumerate(domains) if d == domain]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=domain, color=colors[i], alpha=0.7, edgecolors='none')

    plt.title("YOLO Internal Features: Colored by Domain", fontsize=14)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out1 = REPO / "docs" / "analysis" / "domain_shift_pca.png"
    plt.savefig(out1, bbox_inches='tight', dpi=300)
    print(f"Saved domain plot to {out1}")
    
    # Plot 2: Colored by Fine-Grained Semantic Class
    plt.figure(figsize=(10, 8))
    
    class_colors = {
        "Drone": "green",
        "Bird": "blue",
        "Airplane": "red",
        "Helicopter": "orange",
        "Background/Clutter": "gray"
    }
    class_markers = {
        "Drone": "o",
        "Bird": "v",
        "Airplane": "^",
        "Helicopter": "s",
        "Background/Clutter": "x"
    }
    
    unique_labels = list(set(labels))
    for label in unique_labels:
        idx = [j for j, l in enumerate(labels) if l == label]
        c = class_colors.get(label, "black")
        m = class_markers.get(label, ".")
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, color=c, alpha=0.7, marker=m)

    plt.title("YOLO Internal Features: Colored by Semantic Class", fontsize=14)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out2 = REPO / "docs" / "analysis" / "class_shift_pca.png"
    plt.savefig(out2, bbox_inches='tight', dpi=300)
    print(f"Saved class plot to {out2}")

if __name__ == "__main__":
    main()
