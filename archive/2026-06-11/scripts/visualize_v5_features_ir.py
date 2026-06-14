"""
Generate the V5-IR feature-space figures for docs/analysis/mlp_v5_report_ir.md.

Loads eval/results/_v5_ir_p3p5_v3b/training_data.npz (X: n x 517,
y: n, w: n) and produces:

    docs/analysis/images/v5_ir_pca_p3.png
    docs/analysis/images/v5_ir_pca_p5.png
    docs/analysis/images/v5_ir_pca_fused.png
    docs/analysis/images/v5_ir_lda_fused.png
    docs/analysis/images/v5_ir_top_neuron_activations.png  (top-4 neurons KDE)

Same plots as the RGB version (scripts/visualize_v5_features.py), adapted
for the IR cache path and output filenames.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval" / "results" / "_v5_ir_p3p5_v3b" / "training_data.npz"
OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# Feature layout: 5 metadata + 256 p3 (=64ch * 2x2 grid) + 256 p5 (=256ch * 1x1)
META_DIM = 5
P3_DIM = 256
P5_DIM = 256


# ── PCA plots per subspace ──────────────────────────────────────────────────

def pca_plot(X: np.ndarray, y: np.ndarray, title: str, out_path: Path,
              subsample: int = 5000):
    """2-D PCA of X colored by y. Subsamples for legibility."""
    if len(X) > subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), subsample, replace=False)
        X, y = X[idx], y[idx]
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=0.4, s=12, c="red",
                label=f"Confusers (n={(y == 0).sum()})")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=0.6, s=12, c="blue",
                label=f"Drones (n={(y == 1).sum()})")
    plt.title(f"{title}\nExplained variance: {sum(var) * 100:.1f}%")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── LDA histogram ───────────────────────────────────────────────────────────

def lda_plot(X: np.ndarray, y: np.ndarray, title: str, out_path: Path):
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        print("  SKIP LDA (insufficient per-class samples)")
        return None
    lda = LinearDiscriminantAnalysis(n_components=1).fit(X, y)
    Z = lda.transform(X).ravel()
    acc = lda.score(X, y)
    plt.figure(figsize=(12, 5))
    plt.hist(Z[y == 1], bins=80, alpha=0.6, label=f"Drones (n={(y == 1).sum()})",
             color="green")
    plt.hist(Z[y == 0], bins=80, alpha=0.6, label=f"Confusers (n={(y == 0).sum()})",
             color="red")
    plt.title(f"{title}\nTrain-set accuracy: {acc:.4f}")
    plt.xlabel("LDA Component 1 (drone-confuser discriminant axis)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")
    return acc


# ── Neuron-level analysis ────────────────────────────────────────────────────

def neuron_anova_rank(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature ANOVA F-stat. Higher = more discriminative."""
    F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def top_neuron_distribution_plot(X: np.ndarray, y: np.ndarray, top_idx: np.ndarray,
                                  out_path: Path, n_show: int = 4):
    """Show 2x2 grid of top-N discriminative neuron activation KDE curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(n_show):
        ax = axes[i // 2, i % 2]
        feat_idx = int(top_idx[i])
        x_d = X[y == 1, feat_idx]
        x_c = X[y == 0, feat_idx]
        # Build a shared x-range
        all_vals = np.concatenate([x_d, x_c])
        lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
        margin = (hi - lo) * 0.05
        xs = np.linspace(lo - margin, hi + margin, 500)
        # KDE for each class
        kde_c = gaussian_kde(x_c, bw_method=0.15)
        kde_d = gaussian_kde(x_d, bw_method=0.15)
        ax.fill_between(xs, kde_c(xs), alpha=0.45, color="red", label="Confusers")
        ax.fill_between(xs, kde_d(xs), alpha=0.45, color="blue", label="Drones")
        ax.plot(xs, kde_c(xs), color="red", linewidth=1.2)
        ax.plot(xs, kde_d(xs), color="blue", linewidth=1.2)
        ax.set_title(f"Neuron {feat_idx} Activation Distribution")
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    fig.suptitle("IR V5: Top 4 Discriminative Neurons (by ANOVA F-stat)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not DATA.exists():
        print(f"FATAL: {DATA} not found. Run `python eval/distill_v5_p3p5_ir.py --phase 1` first.")
        return

    z = np.load(DATA)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    n = X.shape[0]
    n_d = int((y == 1).sum())
    n_c = int((y == 0).sum())
    print(f"Loaded IR V5 cache: {n} samples ({n_d} drones, {n_c} confusers), dim={X.shape[1]}")
    assert X.shape[1] == META_DIM + P3_DIM + P5_DIM == 517, \
        f"expected 517-D vectors, got {X.shape[1]}"

    sns.set_theme(style="whitegrid")

    p3 = X[:, META_DIM:META_DIM + P3_DIM]
    p5 = X[:, META_DIM + P3_DIM:]
    fused = X[:, META_DIM:]

    # PCA plots
    pca_plot(p3, y, "IR V5 PCA: p3 (256-D, 2x2 ROI grid x 64 ch) on V3b features",
              OUT / "v5_ir_pca_p3.png")
    pca_plot(p5, y, "IR V5 PCA: p5 (256-D, 1x1 ROI x 256 ch) on V3b features",
              OUT / "v5_ir_pca_p5.png")
    pca_plot(fused, y, "IR V5 PCA: p3 + p5 fused (512-D) on V3b features",
              OUT / "v5_ir_pca_fused.png")

    # LDA
    lda_acc = lda_plot(fused, y, "IR V5 LDA: p3+p5 fused on IR features",
                        OUT / "v5_ir_lda_fused.png")

    # Neuron-level analysis
    print("\nComputing ANOVA F-rank per feature ...")
    F = neuron_anova_rank(X, y)
    top = np.argsort(F)[::-1]
    print(f"  Top 10 features (by F-stat): {top[:10].tolist()}")
    print(f"  Top 10 F values: {F[top[:10]].round(0).tolist()}")
    print(f"  Median F = {np.median(F):.1f}, max F = {F.max():.1f}")

    top_neuron_distribution_plot(X, y, top, OUT / "v5_ir_top_neuron_activations.png")

    print(f"\nDone. All IR figures in docs/analysis/images/v5_ir_*.png")
    if lda_acc is not None:
        print(f"IR V5 LDA train accuracy on fused features: {lda_acc:.4f}")
        if lda_acc > 0.99:
            print("  -> features are essentially linearly separable")
        elif lda_acc > 0.95:
            print("  -> very strong class separation; nonlinear classifier needed for last few %")
        elif lda_acc > 0.85:
            print("  -> good separation; MLP should work well")
        else:
            print("  -> moderate separation; may need more diverse training data")


if __name__ == "__main__":
    main()
