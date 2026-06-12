"""
Generate the V5 feature-space figures used in the thesis chapter
docs/analysis/2026-05-28_distillation_v5_journey.md.

Loads eval/results/_v5_p3p5_ft4_distill/training_data.npz (X: n x 517,
y: n, w: n) and produces:

    docs/analysis/images/v5_pca_p3.png
    docs/analysis/images/v5_pca_p5.png
    docs/analysis/images/v5_pca_fused.png
    docs/analysis/images/v5_lda_fused.png
    docs/analysis/images/v5_class_heatmap.png        (top-20 neurons)
    docs/analysis/images/v5_mean_signature.png       (top-50 neurons)
    docs/analysis/images/v5_top_neuron_activations.png  (top-4 neurons)
    docs/analysis/images/v5_per_layer_anova.png      (p3 vs p5 anova rank)
    docs/analysis/images/v5_metric_evolution.png     (V1 -> V5 progression)

Designed to mirror the analysis in
docs/analysis/domain_shift_and_feature_distillation.md so the thesis
chapter can show the same neuron-signature evidence on the 35k V5 cache.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import f_classif


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_data.npz"
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


# ── LDA histogram (generic K-class) ─────────────────────────────────────────

_LDA_DEFAULT_COLORS = ["red", "green", "blue", "orange", "gray", "purple", "brown"]


def lda_plot(X: np.ndarray, y: np.ndarray, title: str, out_path: Path,
             class_names: dict | None = None,
             class_colors: dict | None = None):
    classes = sorted(int(c) for c in np.unique(y))
    if len(classes) < 2:
        print("  SKIP LDA (fewer than 2 classes)")
        return None
    for c in classes:
        if (y == c).sum() < 2:
            print(f"  SKIP LDA (class {c} has <2 samples)")
            return None

    names = class_names or {c: str(c) for c in classes}
    colors = class_colors or {
        c: _LDA_DEFAULT_COLORS[i % len(_LDA_DEFAULT_COLORS)]
        for i, c in enumerate(classes)
    }

    n_components = min(len(classes) - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y)
    Z = lda.transform(X)
    acc = lda.score(X, y)

    if n_components == 1:
        Z1d = Z.ravel()
        plt.figure(figsize=(12, 5))
        for c in classes:
            m = y == c
            plt.hist(Z1d[m], bins=80, alpha=0.6,
                     label=f"{names.get(c, c)} (n={m.sum()})",
                     color=colors[c])
        plt.title(f"{title}\nTrain-set accuracy: {acc:.4f}")
        plt.xlabel("LDA Component 1")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        for c in classes:
            m = y == c
            ax.hist(Z[m, 0], bins=80, alpha=0.5,
                    label=f"{names.get(c, c)} (n={m.sum()})",
                    color=colors[c])
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("Count")
        ax.set_title("LD1 histogram")
        ax.legend(fontsize=8)
        ax = axes[1]
        for c in classes:
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], alpha=0.3, s=8,
                       c=colors[c], label=names.get(c, c))
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("LDA Component 2")
        ax.set_title("LD1 vs LD2 scatter")
        ax.legend(fontsize=8, markerscale=2)
        fig.suptitle(
            f"{title}\nTrain acc: {acc:.4f}  |  {len(y)} samples  |  {len(classes)} classes",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()

    print(f"  Wrote {out_path}")
    return acc


def lda_multiclass_plot() -> None:
    """Run multi-class LDA (Drone/Bird/Airplane/Helicopter/Other) by dynamically
    loading the sibling visualize_v5_lda_multiclass.py script.
    Writes docs/analysis/images/v5_lda_multiclass.png.
    """
    import importlib.util

    sibling = Path(__file__).resolve().parent / "visualize_v5_lda_multiclass.py"
    if not sibling.exists():
        print(f"  SKIP multiclass LDA: {sibling} not found")
        return
    spec = importlib.util.spec_from_file_location("_mc_lda", sibling)
    mc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mc)
    except Exception as e:
        print(f"  SKIP multiclass LDA (import error): {e}")
        return
    if not mc.BASE_MODEL.exists():
        print(f"  SKIP multiclass LDA: YOLO model not found at {mc.BASE_MODEL}")
        return
    print("\nRunning multi-class LDA (mines Svanström + video confusers) ...")
    mc.main()


# ── Neuron-level analysis (the thesis core) ─────────────────────────────────

def neuron_anova_rank(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature ANOVA F-stat. Higher = more discriminative."""
    F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def top_neuron_distribution_plot(X: np.ndarray, y: np.ndarray, top_idx: np.ndarray,
                                  out_path: Path, n_show: int = 4):
    """Show 2x2 grid of top-N discriminative neuron activation KDE curves."""
    from scipy.stats import gaussian_kde
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
    plt.suptitle("Top discriminative neurons: drone vs confuser activation distributions",
                 y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Wrote {out_path}")


def _layer_name(feat_idx: int) -> str:
    """Map a flat 517-D index to its source layer for plot labels."""
    if feat_idx < META_DIM:
        names = ["conf", "log_area", "aspect", "rel_cx", "rel_cy"]
        return f"metadata: {names[feat_idx]}"
    if feat_idx < META_DIM + P3_DIM:
        local = feat_idx - META_DIM
        cell = local // 64
        chan = local % 64
        return f"p3 cell={cell} ch={chan}"
    local = feat_idx - META_DIM - P3_DIM
    return f"p5 ch={local}"


def class_heatmap_plot(X: np.ndarray, y: np.ndarray, top_idx: np.ndarray,
                        out_path: Path, n_neurons: int = 20):
    """Z-score-normalized mean activation across top-N neurons, drones vs confusers."""
    sel = top_idx[:n_neurons]
    X_sel = X[:, sel]
    # Z-score per feature
    mu = X_sel.mean(axis=0)
    sd = X_sel.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Z = (X_sel - mu) / sd
    mean_drone = Z[y == 1].mean(axis=0)
    mean_conf = Z[y == 0].mean(axis=0)
    heat = np.stack([mean_drone, mean_conf], axis=0)
    fig, ax = plt.subplots(figsize=(max(12, n_neurons * 0.5), 4))
    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Drone", "Confuser"])
    ax.set_xticks(range(n_neurons))
    ax.set_xticklabels([f"N{int(i)}" for i in sel], rotation=45)
    ax.set_xlabel("Neuron index in V5 517-D feature vector")
    ax.set_title(f"V5 top-{n_neurons} discriminative neurons (Z-score, by class)")
    plt.colorbar(im, ax=ax, label="Relative activation (Z-score)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


def mean_signature_plot(X: np.ndarray, y: np.ndarray, top_idx: np.ndarray,
                         out_path: Path, n_neurons: int = 50):
    """Mean activation signature heatmap, top-N neurons."""
    sel = top_idx[:n_neurons]
    X_sel = X[:, sel]
    mu_d = X_sel[y == 1].mean(axis=0)
    mu_c = X_sel[y == 0].mean(axis=0)
    heat = np.stack([mu_d, mu_c], axis=0)
    fig, ax = plt.subplots(figsize=(max(16, n_neurons * 0.4), 3))
    im = ax.imshow(heat, cmap="Reds", aspect="auto")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Drones", "Confusers"])
    ax.set_xticks(range(n_neurons))
    ax.set_xticklabels([f"{int(i)}" for i in sel], rotation=90, fontsize=7)
    ax.set_xlabel("Neuron index (top-50 by ANOVA F-stat)")
    ax.set_title("V5 mean activation signature: top-50 discriminative neurons")
    plt.colorbar(im, ax=ax, label="Mean activation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


def per_layer_anova_plot(F: np.ndarray, out_path: Path):
    """Bar chart of ANOVA F-stat distribution per source layer (metadata / p3 / p5)."""
    meta_F = F[:META_DIM]
    p3_F = F[META_DIM:META_DIM + P3_DIM]
    p5_F = F[META_DIM + P3_DIM:]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot([meta_F, p3_F, p5_F], labels=["metadata\n(5)", "p3\n(256)", "p5\n(256)"],
                     showmeans=True, meanline=True)
    ax.set_yscale("log")
    ax.set_ylabel("ANOVA F-statistic (log scale)")
    ax.set_title("Per-layer discriminative power: ANOVA F-stat distribution")
    ax.axhline(np.median(F), color="orange", linestyle="--", alpha=0.5,
               label=f"Global median F={np.median(F):.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")
    return {
        "meta_mean_F": float(meta_F.mean()),
        "p3_mean_F":   float(p3_F.mean()),
        "p5_mean_F":   float(p5_F.mean()),
        "meta_max_F":  float(meta_F.max()),
        "p3_max_F":    float(p3_F.max()),
        "p5_max_F":    float(p5_F.max()),
    }


def metric_evolution_plot(out_path: Path):
    """Bar chart: V1 -> V5 progression on CV F1 + Svan F1."""
    versions = ["V1\n(catastrophe)", "V2\n(p5, dm)", "V3\n(p3)",
                "V4\n(fused, FT4, 1k)", "V5\n(56k, focal)"]
    cv_f1 = [0.30, 0.62, 0.55, 0.88, 0.99]
    svan_f1 = [0.00, 0.57, 0.50, 0.25, 0.87]
    confuser_halluc = [None, 0.0167, 0.0121, 0.0015, 0.0103]
    x = np.arange(len(versions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, cv_f1, width=w, label="CV F1 (training fit)",
                color="steelblue")
    b2 = ax.bar(x + w/2, svan_f1, width=w, label="Svanstrom F1 (deploy)",
                color="darkorange")
    for rects in (b1, b2):
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Distillation evolution: training fit vs deployment generalization")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not DATA.exists():
        print(f"FATAL: {DATA} not found. Run `python eval/distill_v5_p3p5_ft4.py` first.")
        return

    z = np.load(DATA)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    n = X.shape[0]
    n_d = int((y == 1).sum())
    n_c = int((y == 0).sum())
    print(f"Loaded V5 cache: {n} samples ({n_d} drones, {n_c} confusers), dim={X.shape[1]}")
    assert X.shape[1] == META_DIM + P3_DIM + P5_DIM == 517, \
        f"expected 517-D vectors, got {X.shape[1]}"

    sns.set_theme(style="whitegrid")

    p3 = X[:, META_DIM:META_DIM + P3_DIM]
    p5 = X[:, META_DIM + P3_DIM:]
    fused = X[:, META_DIM:]

    # PCA plots
    pca_plot(p3, y, "V5 PCA: p3 (256-D, 2x2 ROI grid x 64 ch) on FT4 R3 features",
              OUT / "v5_pca_p3.png")
    pca_plot(p5, y, "V5 PCA: p5 (256-D, 1x1 ROI x 256 ch) on FT4 R3 features",
              OUT / "v5_pca_p5.png")
    pca_plot(fused, y, "V5 PCA: p3 + p5 fused (512-D) on FT4 R3 features",
              OUT / "v5_pca_fused.png")

    # LDA — binary (drone vs confuser) from NPZ
    lda_acc = lda_plot(fused, y, "V5 LDA: p3+p5 fused on 35k samples",
                        OUT / "v5_lda_fused.png",
                        class_names={0: "Confuser", 1: "Drone"},
                        class_colors={0: "red", 1: "green"})

    # Neuron-level analysis
    print("\nComputing ANOVA F-rank per feature ...")
    F = neuron_anova_rank(X, y)
    top = np.argsort(F)[::-1]
    print(f"  Top 10 features (by F-stat): {top[:10].tolist()}")
    print(f"  Top 10 F values: {F[top[:10]].round(0).tolist()}")
    print(f"  Median F = {np.median(F):.1f}, max F = {F.max():.1f}")

    top_neuron_distribution_plot(X, y, top, OUT / "v5_top_neuron_activations.png")
    class_heatmap_plot(X, y, top, OUT / "v5_class_heatmap.png", n_neurons=20)
    mean_signature_plot(X, y, top, OUT / "v5_mean_signature.png", n_neurons=50)
    per_layer_stats = per_layer_anova_plot(F, OUT / "v5_per_layer_anova.png")

    print("\nPer-layer ANOVA summary:")
    for k, v in per_layer_stats.items():
        print(f"  {k}: {v:.2f}")

    # Metric evolution
    metric_evolution_plot(OUT / "v5_metric_evolution.png")

    if lda_acc is not None:
        print(f"V5 LDA train accuracy on fused features: {lda_acc:.4f}")
        if lda_acc > 0.99:
            print("  -> features are essentially linearly separable on 35k samples")
        elif lda_acc > 0.95:
            print("  -> very strong class separation; nonlinear classifier needed for last few %")

    # Multi-class LDA (Drone / Bird / Airplane / Helicopter / Other)
    lda_multiclass_plot()

    print("\nDone. All figures in docs/analysis/images/v5_*.png")


if __name__ == "__main__":
    main()
