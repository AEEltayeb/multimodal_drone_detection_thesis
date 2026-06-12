import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path

def main():
    # Load data
    data_path = Path('eval/results/_overnight_distill/training_data.npz')
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return
        
    z = np.load(data_path)
    X, y = z['X'], z['y']
    
    drone_X = X[y==1]
    conf_X = X[y==0]
    
    # YOLO features are the last 256 dimensions
    yolo_drone = drone_X[:, 5:]
    yolo_conf = conf_X[:, 5:]
    yolo_X = X[:, 5:]
    
    out_dir = Path('docs/analysis/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. PCA Visualization
    print("Generating PCA plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(yolo_X)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.5, label='Confusers', s=15, c='red')
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.7, label='Drones', s=15, c='blue')
    plt.title(f'PCA of YOLO p5 Features (256D \u2192 2D)\nExplained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'yolo_features_pca.png', dpi=300)
    plt.close()
    
    # 2. Top Discriminative Neurons
    print("Finding top neurons and plotting distributions...")
    t_stats = []
    for j in range(256):
        t, p = stats.ttest_ind(yolo_drone[:, j], yolo_conf[:, j])
        t_stats.append((abs(t), j))
    t_stats.sort(reverse=True)
    
    top_neurons = [j for _, j in t_stats[:4]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, neuron_idx in enumerate(top_neurons):
        ax = axes[i]
        sns.kdeplot(yolo_conf[:, neuron_idx], ax=ax, color='red', fill=True, label='Confusers', alpha=0.4)
        sns.kdeplot(yolo_drone[:, neuron_idx], ax=ax, color='blue', fill=True, label='Drones', alpha=0.4)
        ax.set_title(f'Neuron {neuron_idx} Activation Distribution')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'top_neuron_activations.png', dpi=300)
    plt.close()
    
    # 3. Mean Activation Signature (Heatmap)
    print("Generating mean signature heatmap...")
    mean_drone = yolo_drone.mean(axis=0)
    mean_conf = yolo_conf.mean(axis=0)
    
    # Sort neurons by absolute difference for better visualization
    diff = np.abs(mean_drone - mean_conf)
    sort_idx = np.argsort(diff)[::-1]
    
    # Take top 50 neurons for clean heatmap
    top_50_idx = sort_idx[:50]
    
    heatmap_data = np.vstack((mean_drone[top_50_idx], mean_conf[top_50_idx]))
    
    plt.figure(figsize=(15, 4))
    sns.heatmap(heatmap_data, cmap='coolwarm', center=0, 
                yticklabels=['Drones', 'Confusers'],
                xticklabels=top_50_idx)
    plt.title('Mean Activation Signature (Top 50 Most Discriminative Neurons)')
    plt.xlabel('Neuron Index')
    plt.tight_layout()
    plt.savefig(out_dir / 'mean_activation_signature.png', dpi=300)
    plt.close()

    print(f"Visualizations saved to {out_dir}")

if __name__ == '__main__':
    main()
