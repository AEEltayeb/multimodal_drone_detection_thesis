"""fusion_feature_stats.py — leakage-aware statistical analysis of the 56-D fusion
feature space, reusing mri.stats primitives (ANOVA-F, per-feature AUROC, LDA, PCA).

The point: standard single-pool stats rank a scene-fingerprint feature (img_mean,
pos_x) HIGH because it correlates with which scene a sample came from. To separate
*robust signal* from *fingerprint*, we compute two ANOVA F-statistics per feature:

  F_class          : discriminates trust(>=1) vs reject(0)         -> signal we want
  F_domain_inclass : discriminates which dataset, WITHIN drones    -> fingerprint signal
  leakage_ratio    = F_domain_inclass / F_class
        low  -> robust feature  (keep)
        high -> scene fingerprint (drop; this is what sank lean13/lean17)

Outputs a ranked table + LDA/PCA/AUROC/leakage plots. Runs on the cached CSV
(no detector); ~1-2 min.

  py classifier/fusion_feature_stats.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mri.stats import anova_f, per_feature_auroc, pca_2d  # reuse mri primitives

CSV = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
IMG = REPO / "docs/analysis/images"
OUT_CSV = REPO / "models/routers/optimal_v1/feature_stats_ranked.csv"
PHASE0_CSV = REPO / "models/routers/optimal_v1/feature_stats_regime.csv"
IMG.mkdir(parents=True, exist_ok=True)

# robust6 = the production 6-feature set; the cross-modal candidates left out of it.
ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
XMODAL_CANDS = ["conf_sum", "xmodal_conf_ratio", "xmodal_scale_ratio", "conf_product"]
# thermal-IR sources; everything else (video_*, confuser_*) is grayscale-IR fallback.
THERMAL_SRC = {"antiuav", "svanstrom"}

META_COLS = {"trust_label", "stem", "source", "_seq"}

# Feature type tags (for the leakage interpretation / coloring)
FINGERPRINT_HINTS = ("img_mean", "img_std", "pos_x", "pos_y", "dist_to_center",
                     "sky_ground", "brightness", "img_dynamic_range", "img_entropy",
                     "scene_entropy")


def feature_type(name: str) -> str:
    if any(h in name for h in ("max_conf", "mean_conf", "conf_", "_conf")):
        return "confidence"
    if any(h in name for h in ("bbox_area", "aspect", "area_diff", "scale_ratio")):
        return "geometry"
    if name.startswith("xmodal") or name in ("both_detect", "neither_detect",
            "rgb_only_detect", "ir_only_detect", "det_count_diff", "total_dets",
            "pos_euclidean_diff"):
        return "cross-modal"
    if any(h in name for h in FINGERPRINT_HINTS):
        return "scene/fingerprint"
    return "other"


def main():
    print(f"Loading {CSV.name} ...")
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META_COLS]
    print(f"  {len(df):,} rows, {len(feats)} features")

    X = df[feats].to_numpy(dtype=float)
    # impute column medians for NaN (undefined geometry when a modality didn't fire)
    col_med = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

    y_multi = df["trust_label"].to_numpy()
    y_bin = (y_multi >= 1).astype(int)          # 1 = trust (drone seen), 0 = reject
    src = df["source"].to_numpy()
    print(f"  binary class: trust={int(y_bin.sum()):,}  reject={int((y_bin==0).sum()):,}")

    # ── per-feature stats ────────────────────────────────────────
    F_class = anova_f(X, y_bin)
    auroc = per_feature_auroc(X, y_bin)

    # within-class (drone) domain F: how much does the feature vary by SCENE
    # among samples of the SAME class -> isolates fingerprint from real signal
    drone = y_bin == 1
    F_domain = anova_f(X[drone], src[drone])
    leakage = F_domain / (F_class + 1.0)        # +1 guards tiny denominators

    tbl = pd.DataFrame({
        "feature": feats,
        "type": [feature_type(f) for f in feats],
        "F_class": F_class,
        "auroc_alone": auroc,
        "F_domain_inclass": F_domain,
        "leakage_ratio": leakage,
    }).sort_values("auroc_alone", ascending=False).reset_index(drop=True)

    # robust score: discriminative AND not a fingerprint
    tbl["robust_rank"] = (tbl["auroc_alone"].rank(ascending=False)
                          + tbl["leakage_ratio"].rank(ascending=True))
    tbl = tbl.sort_values("robust_rank").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT_CSV, index=False)

    pd.set_option("display.width", 160, "display.max_rows", 60)
    print("\n=== TOP 15 by AUROC-alone (most discriminative single features) ===")
    print(tbl.sort_values("auroc_alone", ascending=False)
            [["feature", "type", "auroc_alone", "F_class", "leakage_ratio"]].head(15).to_string(index=False))
    print("\n=== HIGHEST leakage (scene fingerprints — drop these) ===")
    print(tbl.sort_values("leakage_ratio", ascending=False)
            [["feature", "type", "auroc_alone", "leakage_ratio"]].head(12).to_string(index=False))
    print("\n=== ROBUST shortlist (top by combined robust_rank) ===")
    print(tbl[["feature", "type", "auroc_alone", "leakage_ratio"]].head(12).to_string(index=False))

    # ── LDA separability (standardized) ──────────────────────────
    Xs = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xs, y_bin)
    z = lda.transform(Xs)[:, 0]
    lda_acc = lda.score(Xs, y_bin)
    print(f"\nLDA train accuracy (linear separability): {lda_acc:.4f}")

    # ── PLOTS ────────────────────────────────────────────────────
    # 1. LDA histogram
    plt.figure(figsize=(7, 4))
    plt.hist(z[y_bin == 1], bins=80, alpha=0.6, color="green", label=f"trust/drone (n={int(drone.sum())})", density=True)
    plt.hist(z[y_bin == 0], bins=80, alpha=0.6, color="red", label=f"reject (n={int((~drone).sum())})", density=True)
    plt.title(f"Fusion LDA — 56-D trust vs reject (linear acc {lda_acc:.3f})")
    plt.xlabel("LDA component 1"); plt.ylabel("density"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "fusion_lda_hist.png", dpi=160); plt.close()

    # 2. PCA 2D
    Z, evr = pca_2d(Xs, 2)
    plt.figure(figsize=(6, 5))
    s = np.random.RandomState(0).permutation(len(Z))[:8000]
    plt.scatter(Z[s][y_bin[s] == 0, 0], Z[s][y_bin[s] == 0, 1], s=4, alpha=0.3, c="red", label="reject")
    plt.scatter(Z[s][y_bin[s] == 1, 0], Z[s][y_bin[s] == 1, 1], s=4, alpha=0.3, c="green", label="trust/drone")
    plt.title(f"Fusion PCA (unsupervised) — PC1 {evr[0]:.0%} PC2 {evr[1]:.0%}")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "fusion_pca_2d.png", dpi=160); plt.close()

    # 3. AUROC bar (top 20)
    top = tbl.sort_values("auroc_alone", ascending=False).head(20)[::-1]
    cmap = {"confidence": "#1f77b4", "geometry": "#2ca02c", "cross-modal": "#9467bd",
            "scene/fingerprint": "#d62728", "other": "#7f7f7f"}
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top["auroc_alone"], color=[cmap[t] for t in top["type"]])
    plt.axvline(0.5, color="k", lw=0.8, ls="--")
    plt.title("Per-feature AUROC-alone (top 20)  —  color = feature type")
    plt.xlabel("AUROC (direction-agnostic)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    plt.legend(handles, cmap.keys(), fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(IMG / "fusion_feature_auroc.png", dpi=160); plt.close()

    # 4. Leakage scatter: F_class (x) vs F_domain_inclass (y)
    plt.figure(figsize=(7.5, 6))
    for t, c in cmap.items():
        m = tbl["type"] == t
        plt.scatter(tbl[m]["F_class"], tbl[m]["F_domain_inclass"], c=c, label=t, s=40, alpha=0.8)
    # annotate notable points
    for _, r in tbl.iterrows():
        if r["leakage_ratio"] > tbl["leakage_ratio"].quantile(0.85) or r["auroc_alone"] > 0.7:
            plt.annotate(r["feature"], (r["F_class"], r["F_domain_inclass"]),
                         fontsize=6, alpha=0.7)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("F_class  (discriminates drone vs reject — WANT high)")
    plt.ylabel("F_domain_inclass  (varies by scene within drones — fingerprint, WANT low)")
    plt.title("Leakage map: lower-right = robust signal, upper-left = scene fingerprint")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(IMG / "fusion_leakage_map.png", dpi=160); plt.close()

    print(f"\nSaved ranked table: {OUT_CSV}")
    print("Saved plots: fusion_lda_hist.png, fusion_pca_2d.png, fusion_feature_auroc.png, fusion_leakage_map.png")


def _load_imputed():
    """Load full56, impute NaN/inf with column medians, return (df, feats, X, regime)."""
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META_COLS]
    X = df[feats].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    col_med = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])
    regime = np.where(df["source"].isin(THERMAL_SRC).to_numpy(), "thermal", "grayscale")
    return df, feats, X, regime


def phase0():
    """Phase 0 — regime-aware & per-class re-ranking (statistics only, zero training).

    Answers: (1) which features keep their drone-vs-reject AUROC under GRAYSCALE-IR vs
    thermal-IR; (2) what drives the starved single-modality trust_rgb/trust_ir routing;
    (3) how far each IR feature's distribution shifts gray-vs-thermal; (4) redundancy of
    the cross-modal candidates with robust6; (5) does a candidate add held-out separability.
    """
    df, feats, X, regime = _load_imputed()
    fi = {f: i for i, f in enumerate(feats)}
    y_multi = df["trust_label"].to_numpy()
    y_bin = (y_multi >= 1).astype(int)
    src = df["source"].to_numpy()
    th, gr = regime == "thermal", regime == "grayscale"
    print(f"Loaded {len(df):,} rows | thermal={th.sum():,} grayscale={gr.sum():,}")
    print(f"  trust labels: " + ", ".join(f"{k}={int((y_multi==k).sum())}" for k in (0,1,2,3)))

    cand = [f for f in (ROBUST6 + XMODAL_CANDS) if f in fi]

    # ── (1) per-regime drone-vs-reject AUROC + leakage ───────────────
    rows = []
    for f in feats:
        i = fi[f]
        a_th = per_feature_auroc(X[th, i:i+1], y_bin[th])[0] if y_bin[th].min() != y_bin[th].max() else np.nan
        a_gr = per_feature_auroc(X[gr, i:i+1], y_bin[gr])[0] if y_bin[gr].min() != y_bin[gr].max() else np.nan
        # one-vs-rest AUROC for the single-modality routing decisions
        a_trgb = per_feature_auroc(X[:, i:i+1], (y_multi == 1).astype(int))[0]
        a_tir  = per_feature_auroc(X[:, i:i+1], (y_multi == 2).astype(int))[0]
        # gray-vs-thermal standardized distribution shift (CORAL-style)
        m_th, m_gr = X[th, i].mean(), X[gr, i].mean()
        sd = X[:, i].std() + 1e-9
        zshift = abs(m_th - m_gr) / sd
        rows.append(dict(feature=f, auroc_thermal=a_th, auroc_grayscale=a_gr,
                         auroc_drop=a_th - a_gr, auroc_trust_rgb=a_trgb,
                         auroc_trust_ir=a_tir, gray_thermal_zshift=zshift,
                         in_robust6=f in ROBUST6))
    reg = pd.DataFrame(rows)
    reg.to_csv(PHASE0_CSV, index=False)

    def show(d, cols, title, n=12):
        print(f"\n=== {title} ===")
        print(d[cols].head(n).to_string(index=False))

    show(reg.sort_values("auroc_drop", ascending=False),
         ["feature", "auroc_thermal", "auroc_grayscale", "auroc_drop", "in_robust6"],
         "Largest AUROC COLLAPSE thermal->grayscale (these stop working on gray)")
    show(reg[reg.feature.isin(cand)].sort_values("auroc_grayscale", ascending=False),
         ["feature", "auroc_thermal", "auroc_grayscale", "auroc_drop", "in_robust6"],
         "Candidate features by GRAYSCALE AUROC (which hold up)", n=20)
    show(reg.sort_values("auroc_trust_rgb", ascending=False),
         ["feature", "auroc_trust_rgb", "auroc_trust_ir", "in_robust6"],
         "Top drivers of trust_rgb (the starved 8% class)")
    show(reg[reg.feature.str.startswith("ir_")].sort_values("gray_thermal_zshift", ascending=False),
         ["feature", "gray_thermal_zshift", "auroc_thermal", "auroc_grayscale"],
         "IR features by gray-vs-thermal distribution shift (high = unreliable on gray)")

    # ── (4) redundancy of candidates with robust6 (Spearman) ─────────
    cdf = df[cand].apply(lambda s: s.replace([np.inf, -np.inf], np.nan).fillna(s.median()))
    corr = cdf.corr(method="spearman")
    print("\n=== Candidate vs robust6 max |Spearman| (redundancy; high = collinear, skip) ===")
    for c in XMODAL_CANDS:
        if c not in corr:
            continue
        mx = corr.loc[c, ROBUST6].abs()
        print(f"  {c:20s} max|r| with robust6 = {mx.max():.3f} (vs {mx.idxmax()})")

    # ── (5) incremental held-out LDA separability (grouped by source) ─
    print("\n=== Incremental separability: held-out LDA acc, grouped by source ===")
    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    def cv_lda(cols):
        idx = [fi[c] for c in cols]
        accs = []
        for tr, te in gss.split(X, y_bin, groups=src):
            Xs = StandardScaler().fit(X[tr][:, idx])
            m = LinearDiscriminantAnalysis().fit(Xs.transform(X[tr][:, idx]), y_bin[tr])
            accs.append(m.score(Xs.transform(X[te][:, idx]), y_bin[te]))
        return np.mean(accs), np.std(accs)
    base_m, base_s = cv_lda(ROBUST6)
    print(f"  robust6                         {base_m:.4f} +/- {base_s:.4f}")
    for c in XMODAL_CANDS:
        if c not in fi:
            continue
        m, s = cv_lda(ROBUST6 + [c])
        print(f"  robust6 + {c:20s} {m:.4f} +/- {s:.4f}   (delta {m-base_m:+.4f})")
    xcore = ["ir_max_conf", "ir_best_aspect_ratio", "ir_best_log_bbox_area",
             "conf_sum", "xmodal_conf_ratio", "xmodal_scale_ratio"]
    if all(c in fi for c in xcore):
        m, s = cv_lda(xcore)
        print(f"  xmodal_core (6, no RGB geom)    {m:.4f} +/- {s:.4f}   (delta {m-base_m:+.4f})")

    # ── PLOTS ────────────────────────────────────────────────────────
    sub = reg[reg.feature.isin(cand)].sort_values("auroc_grayscale")
    yy = np.arange(len(sub)); h = 0.4
    plt.figure(figsize=(9, 7))
    plt.barh(yy + h/2, sub.auroc_thermal, h, color="#d62728", label="thermal-IR")
    plt.barh(yy - h/2, sub.auroc_grayscale, h, color="#1f77b4", label="grayscale-IR")
    plt.yticks(yy, [f + ("*" if r else "") for f, r in zip(sub.feature, sub.in_robust6)])
    plt.axvline(0.5, color="k", lw=0.8, ls="--")
    plt.xlabel("AUROC (drone vs reject)"); plt.legend()
    plt.title("Phase 0: per-feature AUROC, thermal vs grayscale IR  (* = in robust6)")
    plt.tight_layout(); plt.savefig(IMG / "fusion_regime_auroc.png", dpi=160); plt.close()

    irf = reg[reg.feature.str.startswith("ir_")].sort_values("gray_thermal_zshift").tail(14)
    plt.figure(figsize=(8, 6))
    plt.barh(irf.feature, irf.gray_thermal_zshift,
             color=["#2ca02c" if r else "#9467bd" for r in irf.in_robust6])
    plt.xlabel("gray-vs-thermal standardized shift (high = unreliable on grayscale)")
    plt.title("Phase 0: IR feature distribution shift gray vs thermal")
    plt.tight_layout(); plt.savefig(IMG / "fusion_regime_zshift.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.xticks(range(len(cand)), cand, rotation=90, fontsize=7)
    plt.yticks(range(len(cand)), cand, fontsize=7)
    plt.colorbar(im, fraction=0.046); plt.title("Phase 0: candidate Spearman redundancy")
    plt.tight_layout(); plt.savefig(IMG / "fusion_candidate_corr.png", dpi=160); plt.close()

    print(f"\nSaved: {PHASE0_CSV}")
    print("Plots: fusion_regime_auroc.png, fusion_regime_zshift.png, fusion_candidate_corr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase0", action="store_true",
                    help="regime-aware & per-class re-ranking (cross-modal investigation)")
    args = ap.parse_args()
    if args.phase0:
        phase0()
    else:
        main()
