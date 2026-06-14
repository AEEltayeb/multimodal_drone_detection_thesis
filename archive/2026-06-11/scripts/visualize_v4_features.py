"""
Validate the V4 hypothesis: FT4 R3 features should show tighter drone/confuser
separation than baseline features, AND fused p3+p5 should out-cluster either
layer alone.

Loads eval/results/_v4_p3p5_ft4_distill/training_data.npz (325-D vectors:
5 metadata + 64 p3 + 256 p5) and produces:

    docs/analysis/images/v4_pca_p3.png
    docs/analysis/images/v4_pca_p5.png
    docs/analysis/images/v4_pca_fused.png
    docs/analysis/images/v4_lda_fused.png

If the fused FT4 plot shows tighter drone clustering than the baseline plot
in docs/analysis/class_shift_pca.png, the V4 hypothesis is validated and the
full 30k-sample training run is justified.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval" / "results" / "_v4_p3p5_ft4_distill" / "training_data.npz"
OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _pca_plot(X: np.ndarray, y: np.ndarray, title: str, out_path: Path):
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=0.5, label="Confusers",
                s=15, c="red")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=0.7, label="Drones",
                s=15, c="blue")
    plt.title(f"{title}\nExplained variance: {sum(var) * 100:.1f}%")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Wrote {out_path}")


def _lda_plot(X: np.ndarray, y: np.ndarray, title: str, out_path: Path):
    if (y == 1).sum() < 2 or (y == 0).sum() < 2:
        print(f"  SKIP LDA: not enough samples per class")
        return None
    lda = LinearDiscriminantAnalysis(n_components=1)
    Z = lda.fit_transform(X, y).ravel()
    plt.figure(figsize=(12, 5))
    bins = 60
    plt.hist(Z[y == 1], bins=bins, alpha=0.55, label="Drones", color="green")
    plt.hist(Z[y == 0], bins=bins, alpha=0.55, label="Confusers", color="red")
    plt.title(title)
    plt.xlabel("LDA Component 1")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Wrote {out_path}")
    return float(lda.score(X, y))


def main():
    if not DATA.exists():
        print(f"FATAL: {DATA} not found. Run V4 distill Phase 1 first.")
        return

    z = np.load(DATA)
    X, y = z["X"], z["y"]
    n_d = int((y == 1).sum())
    n_c = int((y == 0).sum())
    print(f"Loaded {X.shape[0]} samples ({n_d} drones, {n_c} confusers), "
          f"shape={X.shape}")

    # Schema check
    expected = 325
    if X.shape[1] != expected:
        print(f"WARN: expected feature dim {expected}, got {X.shape[1]}")
    else:
        print(f"OK: feature dim = {expected} (5 metadata + 64 p3 + 256 p5)")

    # Slice into components
    meta = X[:, :5]
    p3 = X[:, 5:5 + 64]
    p5 = X[:, 5 + 64:5 + 64 + 256]
    fused = X[:, 5:]  # 320-D = p3 + p5

    sns.set_theme(style="whitegrid")

    _pca_plot(p3, y, "V4 PCA: p3 (64-D) on FT4 R3 features", OUT / "v4_pca_p3.png")
    _pca_plot(p5, y, "V4 PCA: p5 (256-D) on FT4 R3 features", OUT / "v4_pca_p5.png")
    _pca_plot(fused, y, "V4 PCA: p3+p5 fused (320-D) on FT4 R3 features",
              OUT / "v4_pca_fused.png")

    acc = _lda_plot(fused, y, "V4 LDA: p3+p5 fused on FT4 R3 features",
                     OUT / "v4_lda_fused.png")
    if acc is not None:
        print(f"\n  LDA train-set accuracy on fused features: {acc:.4f}")
        print(f"  (Compare to V2/V3: V2 CV F1 0.62, V3 CV F1 0.55.")
        print(f"   LDA acc > 0.85 -> FT4 features are well-separated -> "
              f"proceed with full training.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
