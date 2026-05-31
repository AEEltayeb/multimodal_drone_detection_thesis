"""
mri.stats — the "brain statistics" computed on the extracted feature space.

Pure-numeric layer (no plotting): PCA projection, LDA separability, per-feature
ANOVA F-statistic and AUROC, silhouette score, and a compact separability
summary the diagnosis layer consumes. Everything here is deterministic given a
seed and returns plain dicts/arrays so it serializes to stats.json.
"""
from __future__ import annotations

import numpy as np


def subsample(X, y, n_max, seed=0):
    """Return (X, y, idx) subsampled to at most n_max rows (stratify-agnostic)."""
    if len(X) <= n_max:
        return X, y, np.arange(len(X))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n_max, replace=False)
    return X[idx], y[idx], idx


def pca_2d(X, n_components=2, seed=0):
    """2-D PCA. Returns (Z, explained_variance_ratio)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)
    return Z, pca.explained_variance_ratio_


def lda_separability(X, y):
    """Fit LDA drone(1) vs confuser(0). Returns (Z, train_accuracy, n_components).

    Train accuracy is the headline separability number: ~1.0 means the classes
    are (near-)linearly separable in this feature space.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    classes = np.unique(y)
    if len(classes) < 2 or any((y == c).sum() < 2 for c in classes):
        return None, None, 0
    n_comp = min(len(classes) - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_comp).fit(X, y)
    return lda.transform(X), float(lda.score(X, y)), n_comp


def anova_f(X, y):
    """Per-feature ANOVA F-statistic. Higher = more discriminative."""
    from sklearn.feature_selection import f_classif
    F, _ = f_classif(X, y)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


def per_feature_auroc(X, y, top_k=None):
    """AUROC of each feature taken alone as a drone score. Returns array of AUROC.

    Cheap rank-based AUROC (no sklearn loop). Guards against the metadata
    shortcut: if only conf separates, that shows up here.
    """
    y = np.asarray(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return np.zeros(X.shape[1], dtype=np.float32)
    aurocs = np.empty(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        ranks = np.empty(len(X), dtype=np.float64)
        ranks[order] = np.arange(1, len(X) + 1)
        # average ties
        sum_ranks_pos = ranks[y == 1].sum()
        auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        aurocs[j] = max(auc, 1 - auc)  # direction-agnostic separability
    return aurocs


def silhouette(X, y, n_max=4000, seed=0):
    """Silhouette score of the two-class labelling in feature space."""
    from sklearn.metrics import silhouette_score
    Xs, ys, _ = subsample(X, y, n_max, seed)
    if len(np.unique(ys)) < 2:
        return None
    try:
        return float(silhouette_score(Xs, ys))
    except Exception:
        return None


def separability_summary(X, y, schema, seed=0) -> dict:
    """One-call bundle of the numbers the diagnosis layer needs."""
    F = anova_f(X, y)
    auroc = per_feature_auroc(X, y)
    _, lda_acc, _ = lda_separability(X, y)
    sil = silhouette(X, y, seed=seed)
    top = np.argsort(F)[::-1]

    # Per-block (meta vs each layer) mean discriminative power.
    block_F = {}
    for name, sl in schema.layer_slices().items():
        seg = F[sl]
        block_F[name] = {
            "mean_F": float(seg.mean()) if seg.size else 0.0,
            "max_F": float(seg.max()) if seg.size else 0.0,
            "dim": int(seg.size),
        }

    # How much of the signal is metadata-only (esp. conf)? Guards the shortcut.
    meta_sl = schema.layer_slices()["meta"]
    meta_max_auroc = float(auroc[meta_sl].max()) if auroc[meta_sl].size else 0.0
    yolo_max_auroc = float(np.delete(auroc, np.arange(meta_sl.start, meta_sl.stop)).max())

    return {
        "lda_train_accuracy": lda_acc,
        "silhouette": sil,
        "median_anova_F": float(np.median(F)),
        "max_anova_F": float(F.max()),
        "top_feature_indices": top[:20].tolist(),
        "top_feature_labels": [schema.column_label(int(i)) for i in top[:20]],
        "per_block_F": block_F,
        "meta_max_auroc": meta_max_auroc,
        "yolo_max_auroc": yolo_max_auroc,
        "n_drone": int((y == 1).sum()),
        "n_confuser": int((y == 0).sum()),
    }, F, auroc, top
