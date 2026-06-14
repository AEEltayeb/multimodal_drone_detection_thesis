import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Append repo root and eval directory so we can import eval modules
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / "eval"))

from ultralytics import YOLO
# Import from the V3 script to ensure we get p3 features (64-D)
from eval.distill_v3_p3_features import DetectInputHook, collect_predictions, MODEL_PATHS, ANTIUAV_VAL, SVANSTROM_DIR, SELCOM_VAL, CONFUSER_TRAIN

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

    print("\nExtracting P3 features for visualization (max 200 per domain)...")
    for name, path, has_gt in datasets:
        if not path.exists():
            continue
            
        X_tp, y_tp, _, X_fp, y_fp, meta_fp = collect_predictions(
            model, hook, path, stride=15, max_samples=200, has_gt=has_gt, category=name
        )
        
        # We only plot the YOLO features (columns 5:69), ignoring the metadata
        if len(X_tp) > 0:
            features.append(X_tp[:, 5:])
            labels.extend(["Drone"] * len(X_tp))
        if len(X_fp) > 0:
            features.append(X_fp[:, 5:])
            for m in meta_fp:
                fname = Path(m["img"]).name.lower()
                if "bird" in fname:
                    labels.append("Bird")
                elif "airplane" in fname:
                    labels.append("Airplane")
                elif "helicopter" in fname:
                    labels.append("Helicopter")
                else:
                    labels.append("Background")

    if not features:
        print("No features extracted.")
        return

    X_all = np.concatenate(features, axis=0)
    y_labels = np.array(labels)
    
    # ── Diagnostic 1: Centroids & Cosine Similarity ──
    print("\n=== P3 SEMANTIC ENTANGLEMENT TEST (COSINE SIMILARITY) ===")
    unique_classes = [c for c in ["Drone", "Bird", "Airplane", "Helicopter", "Background"] if c in labels]
    centroids = {}
    
    for c in unique_classes:
        mask = (y_labels == c)
        if mask.sum() > 0:
            centroids[c] = X_all[mask].mean(axis=0)
    
    for i, c1 in enumerate(unique_classes):
        for j, c2 in enumerate(unique_classes):
            if i < j and c1 in centroids and c2 in centroids:
                sim = cosine_similarity(centroids[c1].reshape(1, -1), centroids[c2].reshape(1, -1))[0][0]
                print(f"Cosine Similarity [{c1} vs {c2}]: {sim:.4f}")

    # ── Diagnostic 2: PCA vs LDA ──
    print("\nRunning PCA (Max Variance)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all)

    print("Running LDA (Max Class Separation)...")
    # For LDA, we need to map labels to integers. We will ask LDA to separate Drone vs Confusers.
    y_binary = np.where(y_labels == "Drone", 1, 0)
    lda = LDA(n_components=1) # Binary classification = 1 LDA component
    
    # Since LDA is 1D for binary, we'll plot it as a histogram
    X_lda = lda.fit_transform(X_all, y_binary)
    
    # ── Plotting ──
    class_colors = {"Drone": "green", "Bird": "blue", "Airplane": "red", "Helicopter": "orange", "Background": "gray"}
    class_markers = {"Drone": "o", "Bird": "v", "Airplane": "^", "Helicopter": "s", "Background": "x"}
    
    # PCA Plot
    plt.figure(figsize=(10, 8))
    for label in unique_classes:
        idx = (y_labels == label)
        if idx.sum() > 0:
            c, m = class_colors[label], class_markers[label]
            plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, color=c, alpha=0.7, marker=m)
    plt.title("P3 Features (Layer 3): PCA (Colored by Class)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_pca = REPO / "docs" / "analysis" / "p3_class_shift_pca.png"
    plt.savefig(out_pca, bbox_inches='tight', dpi=300)
    
    # LDA Histogram
    plt.figure(figsize=(10, 6))
    for label in unique_classes:
        idx = (y_labels == label)
        if idx.sum() > 0:
            plt.hist(X_lda[idx], bins=50, alpha=0.5, label=label, color=class_colors[label])
    plt.title("P3 Features (Layer 3): LDA Drone vs Confuser Separation", fontsize=14)
    plt.xlabel("LDA Component 1 (Discriminant Axis)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_lda = REPO / "docs" / "analysis" / "p3_class_shift_lda.png"
    plt.savefig(out_lda, bbox_inches='tight', dpi=300)
    
    print(f"\nSaved PCA plot to {out_pca}")
    print(f"Saved LDA plot to {out_lda}")

if __name__ == "__main__":
    main()
