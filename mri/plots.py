"""
mri.plots — figure generation for the MRI report.

Every function takes already-extracted features (X, y) plus a FeatureSchema and
writes one .png to the run's images/ dir. Ported from
scripts/visualize_v5_features.py; titles and tick labels are derived from the
schema instead of hardcoded constants, so any model/feature layout renders.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Headless backend so runs work over SSH / in CI.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import stats  # noqa: E402

_DRONE_C = "#1f77b4"
_CONF_C = "#d62728"
_LAYER_C = {"meta": "#7f7f7f", "p3": "#2ca02c", "p5": "#9467bd"}


def _ensure(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def pca_plot(X, y, title, out_path, subsample=5000, seed=0):
    Xs, ys, _ = stats.subsample(X, y, subsample, seed)
    Z, var = stats.pca_2d(Xs, seed=seed)
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[ys == 0, 0], Z[ys == 0, 1], alpha=0.4, s=12, c=_CONF_C,
                label=f"Confusers (n={(ys == 0).sum()})")
    plt.scatter(Z[ys == 1, 0], Z[ys == 1, 1], alpha=0.6, s=12, c=_DRONE_C,
                label=f"Drones (n={(ys == 1).sum()})")
    plt.title(f"{title}\nExplained variance: {var.sum() * 100:.1f}%")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=160); plt.close()


def lda_plot(X, y, title, out_path):
    Z, acc, n_comp = stats.lda_separability(X, y)
    if Z is None:
        return None
    classes = sorted(int(c) for c in np.unique(y))
    names = {0: "Confuser", 1: "Drone"}
    colors = {0: _CONF_C, 1: _DRONE_C}
    if n_comp == 1:
        Z1 = Z.ravel()
        plt.figure(figsize=(12, 5))
        for c in classes:
            m = y == c
            plt.hist(Z1[m], bins=80, alpha=0.6, color=colors.get(c),
                     label=f"{names.get(c, c)} (n={m.sum()})")
        plt.title(f"{title}\nTrain-set accuracy: {acc:.4f}")
        plt.xlabel("LDA component 1"); plt.ylabel("Count")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_path, dpi=160); plt.close()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for c in classes:
            m = y == c
            axes[0].hist(Z[m, 0], bins=80, alpha=0.5, color=colors.get(c),
                         label=f"{names.get(c, c)} (n={m.sum()})")
            axes[1].scatter(Z[m, 0], Z[m, 1], alpha=0.3, s=8, color=colors.get(c),
                            label=names.get(c, c))
        axes[0].set(xlabel="LD1", ylabel="Count", title="LD1 histogram")
        axes[1].set(xlabel="LD1", ylabel="LD2", title="LD1 vs LD2")
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8, markerscale=2)
        fig.suptitle(f"{title}\nTrain acc: {acc:.4f} | {len(y)} samples", fontsize=13)
        plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()
    return acc


def top_neuron_kde(X, y, top_idx, schema, out_path, n_show=4):
    from scipy.stats import gaussian_kde
    n_show = min(n_show, len(top_idx))
    rows = (n_show + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_show):
        ax = axes[i]
        j = int(top_idx[i])
        x_d, x_c = X[y == 1, j], X[y == 0, j]
        allv = np.concatenate([x_d, x_c])
        lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        m = (hi - lo) * 0.05 + 1e-6
        xs = np.linspace(lo - m, hi + m, 400)
        try:
            ax.fill_between(xs, gaussian_kde(x_c, 0.15)(xs), alpha=0.45, color=_CONF_C, label="Confusers")
            ax.fill_between(xs, gaussian_kde(x_d, 0.15)(xs), alpha=0.45, color=_DRONE_C, label="Drones")
        except Exception:
            ax.hist(x_c, bins=40, alpha=0.5, color=_CONF_C, density=True)
            ax.hist(x_d, bins=40, alpha=0.5, color=_DRONE_C, density=True)
        ax.set_title(schema.column_label(j), fontsize=9)
        ax.legend(fontsize=8)
    for k in range(n_show, len(axes)):
        axes[k].axis("off")
    plt.suptitle("Top discriminative features: drone vs confuser activation", y=1.01)
    plt.tight_layout(); plt.savefig(out_path, dpi=160, bbox_inches="tight"); plt.close()


def class_heatmap(X, y, top_idx, schema, out_path, n=20):
    sel = top_idx[:n]
    Xs = X[:, sel]
    sd = np.where(Xs.std(0) < 1e-6, 1.0, Xs.std(0))
    Z = (Xs - Xs.mean(0)) / sd
    heat = np.stack([Z[y == 1].mean(0), Z[y == 0].mean(0)], 0)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 4))
    im = ax.imshow(heat, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Drone", "Confuser"])
    ax.set_xticks(range(len(sel)))
    ax.set_xticklabels([schema.column_label(int(i)) for i in sel], rotation=90, fontsize=6)
    ax.set_title(f"Top-{n} discriminative features (Z-score by class)")
    plt.colorbar(im, ax=ax, label="Relative activation (Z)")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def per_block_anova(F, schema, out_path):
    slices = schema.layer_slices()
    data = [F[sl] for sl in slices.values() if F[sl].size]
    labels = [f"{k}\n({F[sl].size})" for k, sl in slices.items() if F[sl].size]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=labels, showmeans=True, meanline=True)
    ax.set_yscale("log"); ax.set_ylabel("ANOVA F-statistic (log)")
    ax.set_title("Per-block discriminative power")
    ax.axhline(np.median(F), color="orange", ls="--", alpha=0.5,
               label=f"median F={np.median(F):.0f}")
    ax.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def fp_rate_bars(diag, out_path):
    """Raw vs post-classifier confuser FP rate (if a classifier was trained)."""
    raw = diag.get("raw_halluc_rate")
    post = diag.get("projected_fp_rate")
    if raw is None:
        return
    labels, vals = ["raw detector"], [raw]
    if post is not None:
        labels.append("+ MLP classifier"); vals.append(post)
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=[_CONF_C, _DRONE_C][:len(vals)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set_ylabel("Confuser FP rate (per detection)")
    ax.set_title("Confuser FP reduction")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def auroc_by_layer(auroc, schema, out_path):
    """Per-feature AUROC (drone vs confuser) as a jittered strip per layer."""
    slices = schema.layer_slices()
    layers = [(k, sl) for k, sl in slices.items() if auroc[sl].size]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, sl) in enumerate(layers):
        vals = auroc[sl]
        x = i + (rng.random(vals.size) - 0.5) * 0.55
        ax.scatter(x, vals, s=14, alpha=0.45, color=_LAYER_C.get(name, "#555"))
        ax.scatter([i], [vals.max()], s=140, marker="*", color="black", zorder=5)
        ax.text(i, vals.max() + 0.012, f"max {vals.max():.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.7, label="chance (0.5)")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"{n}\n({auroc[sl].size} feats)" for n, sl in layers])
    ax.set_ylabel("Per-feature AUROC (drone vs confuser)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Per-feature discriminative power by layer (AUROC)")
    ax.legend(loc="lower right"); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def top_feature_auroc_bar(auroc, schema, out_path, n=20):
    """Top-n individual features by AUROC, colored by layer."""
    from matplotlib.patches import Patch
    n = min(n, auroc.size)
    order = np.argsort(auroc)[::-1][:n]
    vals = auroc[order]
    labels = [schema.column_label(int(i)) for i in order]
    colors = [_LAYER_C.get(schema.locate(int(i))[0], "#555") for i in order]
    fig, ax = plt.subplots(figsize=(8.5, max(5, n * 0.32)))
    yp = np.arange(n)[::-1]
    ax.barh(yp, vals, color=colors)
    ax.set_yticks(yp); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.5, min(1.0, float(vals.max()) + 0.05)); ax.set_xlabel("AUROC (drone vs confuser)")
    ax.set_title(f"Top-{n} discriminative features by AUROC")
    handles = [Patch(color=c, label=l) for l, c in _LAYER_C.items()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def roc_curve_plot(oof, y, out_path):
    """Out-of-fold MLP ROC curve with AUC."""
    from sklearn.metrics import roc_curve, auc
    oof = np.asarray(oof, dtype=float); y = np.asarray(y)
    valid = ~np.isnan(oof)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return None
    fpr, tpr, _ = roc_curve(y[valid], oof[valid])
    a = auc(fpr, tpr)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color=_DRONE_C, lw=2.5, label=f"MLP verifier (AUC = {a:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    plt.xlabel("False positive rate (confusers passed as drone)")
    plt.ylabel("True positive rate (real drones kept)")
    plt.title("Distilled verifier — out-of-fold ROC")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()
    return a


def fp_cut_vs_recall(oof, y, out_path, op_thr=0.5):
    """Threshold sweep: confuser FP cut vs drone recall retained."""
    oof = np.asarray(oof, dtype=float); y = np.asarray(y)
    valid = ~np.isnan(oof); oof, y = oof[valid], y[valid]
    drones, confs = y == 1, y == 0
    if drones.sum() == 0 or confs.sum() == 0:
        return None
    taus = np.linspace(0, 1, 200)
    keep = np.array([(oof[drones] >= t).mean() for t in taus])
    cut = np.array([(oof[confs] < t).mean() for t in taus])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(keep, cut, color=_DRONE_C, lw=2.5)
    ok, oc = (oof[drones] >= op_thr).mean(), (oof[confs] < op_thr).mean()
    ax.scatter([ok], [oc], s=150, marker="*", color="black", zorder=5,
               label=f"op @{op_thr:g}: keep {ok:.1%} drones, cut {oc:.1%} confusers")
    ax.set_xlabel("Drone recall retained"); ax.set_ylabel("Confuser FP cut")
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
    ax.set_title("Verifier operating curve (threshold sweep)")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def generate_all(X, y, schema, F, top, out_dir, want=None, diag=None, auroc=None, oof=None):
    """Render the requested figures. Returns list of written paths."""
    out_dir = _ensure(Path(out_dir))
    want = set(want or ["pca", "lda", "heatmap", "neurons", "auroc", "roc", "opcurve"])
    yolo_sl = slice(schema.layer_slices()["meta"].stop, X.shape[1])
    written = []

    def _emit(name, fn):
        p = out_dir / name
        fn(p); written.append(p)

    if "pca" in want:
        _emit("pca_fused.png", lambda p: pca_plot(X[:, yolo_sl], y, "PCA: YOLO features", p))
    if "lda" in want:
        _emit("lda_fused.png", lambda p: lda_plot(X[:, yolo_sl], y, "LDA: YOLO features", p))
    if "anova" in want:
        _emit("per_block_anova.png", lambda p: per_block_anova(F, schema, p))
    if "heatmap" in want:
        _emit("class_heatmap.png", lambda p: class_heatmap(X, y, top, schema, p))
    if "neurons" in want:
        _emit("top_neuron_kde.png", lambda p: top_neuron_kde(X, y, top, schema, p))
    # Only emit the FP-reduction bar chart when there is a hallucination rate to
    # plot (i.e. a real --neg image stream); otherwise the figure would be empty.
    if "auroc" in want and auroc is not None:
        _emit("auroc_by_layer.png", lambda p: auroc_by_layer(auroc, schema, p))
        _emit("top_feature_auroc.png", lambda p: top_feature_auroc_bar(auroc, schema, p))
    if "roc" in want and oof is not None:
        _emit("verifier_roc.png", lambda p: roc_curve_plot(oof, y, p))
    if "opcurve" in want and oof is not None:
        _emit("operating_curve.png", lambda p: fp_cut_vs_recall(oof, y, p))
    if diag is not None and diag.get("raw_halluc_rate") is not None:
        _emit("fp_reduction.png", lambda p: fp_rate_bars(diag, p))
    return written
