"""vet_blurriness_stats.py — statistically vet rgb_blurriness BEFORE paying for it.

The forward sweep picked rgb_blurriness as the biggest single lift. Before adopting an
expensive hand-crafted scene statistic, answer three questions with statistics only
(no training, ~seconds, full56 CSV):

  1. ARTIFACT TEST — is its discrimination a corpus/clip-ID proxy? Compare POOLED AUROC
     (drone vs reject) to WITHIN-SOURCE AUROC. If pooled is high but within-clip ~0.5,
     blurriness just encodes which video it came from (won't generalize).
  2. CHEAP SUBSTITUTE — does a FREE / ROI-cheap feature carry the same signal? Spearman
     |r| of blurriness vs {confidences, ROI local_contrast, target_bg_delta, geometry}.
  3. NONLINEAR / UNEXHAUSTED STATS — mutual information MI(feature; label) for ALL features
     (catches nonlinear signal AUROC/ANOVA miss); rank, tag by cost.

  py classifier/vet_blurriness_stats.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
IMG = REPO / "docs/analysis/images"
META = {"trust_label", "stem", "source", "_seq"}
TARGET = "rgb_blurriness"
CHEAP = ["rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "rgb_best_local_contrast",
         "rgb_best_target_bg_delta", "rgb_best_log_bbox_area", "rgb_best_aspect_ratio"]
# cost tags for the MI plot
SCENE = ("img_mean", "img_std", "img_entropy", "blurriness", "edge_density",
         "sky_ground", "dynamic_range", "scene_entropy")
ROI = ("local_contrast", "target_bg_delta", "pos_x", "pos_y", "dist_to_center")


def cost(name):
    if any(h in name for h in SCENE):
        return "scene/expensive"
    if any(h in name for h in ROI):
        return "roi-cheap"
    return "free"


def main():
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META]
    X = df[feats].to_numpy(float); X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, 0); nm = np.isnan(X); X[nm] = np.take(med, np.where(nm)[1])
    df[feats] = X
    y = (df["trust_label"] >= 1).astype(int).to_numpy()
    if TARGET not in df.columns:
        print(f"[abort] {TARGET} not in dataset"); return

    # ── 1. ARTIFACT TEST ─────────────────────────────────────────────
    pooled = roc_auc_score(y, df[TARGET])
    print(f"=== 1. ARTIFACT TEST — {TARGET} ===")
    print(f"POOLED AUROC (drone vs reject): {pooled:.3f}")
    rows = []
    for s, g in df.groupby("source"):
        yy = (g["trust_label"] >= 1).astype(int).to_numpy()
        if yy.sum() >= 15 and (yy == 0).sum() >= 15:
            rows.append((s, len(g), int(yy.sum()), roc_auc_score(yy, g[TARGET])))
    per_src = pd.DataFrame(rows, columns=["source", "n", "pos", "within_auroc"]).sort_values("within_auroc")
    print(per_src.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(per_src):
        wmean = float(per_src.within_auroc.mean())
        print(f"mean WITHIN-source AUROC: {wmean:.3f}  (pooled {pooled:.3f})  "
              f"-> {'ARTIFACT: pooled signal is mostly cross-clip' if pooled - wmean > 0.1 else 'holds within-clip (real signal)'}")

    # ── 2. CHEAP SUBSTITUTE ──────────────────────────────────────────
    print(f"\n=== 2. CHEAP SUBSTITUTE — |Spearman| of {TARGET} vs cheaper features ===")
    for c in CHEAP:
        if c in df.columns:
            rho = abs(spearmanr(df[TARGET], df[c]).correlation)
            print(f"  {c:<28} |r|={rho:.3f}  ({cost(c)})")

    # ── 3. MUTUAL INFORMATION (nonlinear) ────────────────────────────
    print(f"\n=== 3. MUTUAL INFORMATION MI(feature; trust-vs-reject) — top 18 ===")
    mi = mutual_info_classif(df[feats].to_numpy(), y, random_state=0)
    mitab = pd.DataFrame({"feature": feats, "MI": mi, "cost": [cost(f) for f in feats]}).sort_values("MI", ascending=False)
    print(mitab.head(18).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  {TARGET} MI rank: {int(mitab.reset_index(drop=True).index[mitab.reset_index(drop=True).feature==TARGET][0])+1} / {len(feats)}")

    # ── PLOTS ────────────────────────────────────────────────────────
    if len(per_src):
        plt.figure(figsize=(9, 5))
        plt.barh(per_src.source.str.slice(0, 34), per_src.within_auroc, color="#9467bd")
        plt.axvline(pooled, color="r", ls="--", label=f"pooled {pooled:.3f}")
        plt.axvline(0.5, color="k", lw=0.8, ls=":")
        plt.xlabel("within-source AUROC"); plt.legend(fontsize=8)
        plt.title(f"{TARGET}: within-source vs pooled AUROC (artifact test)")
        plt.tight_layout(); plt.savefig(IMG / "blurriness_artifact_test.png", dpi=160); plt.close()

    top = mitab.head(18)[::-1]
    cmap = {"free": "#1f77b4", "roi-cheap": "#2ca02c", "scene/expensive": "#d62728"}
    plt.figure(figsize=(8.5, 7))
    plt.barh(top.feature, top.MI, color=[cmap[c] for c in top.cost])
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in cmap.values()]
    plt.legend(handles, cmap.keys(), fontsize=8)
    plt.xlabel("mutual information with trust-vs-reject"); plt.title("MI ranking (nonlinear) — color = compute cost")
    plt.tight_layout(); plt.savefig(IMG / "fusion_mutual_info.png", dpi=160); plt.close()
    print("\nPlots: blurriness_artifact_test.png, fusion_mutual_info.png")


if __name__ == "__main__":
    main()
