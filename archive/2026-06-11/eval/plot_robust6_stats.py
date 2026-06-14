"""plot_robust6_stats.py - fresh MRI-style statistical figures for robust6, on the
CURRENT ft4-mined trust dataset (models/routers/lean_ft4/fusion_dataset_lean19.csv).
Reuses mri.stats primitives (same layer as the YOLO model-MRI). Three figures:
  email_robust6_lda.png    - LDA separability histogram (is the signal there)
  email_robust6_auroc.png  - per-feature AUROC bars (robust6's 6 vs the rest)
  email_robust6_anova.png  - per-feature ANOVA F-statistic bars
robust6's 6 features are marked (★). Writes NEW filenames (does not touch the 56-D study figs).
  py eval/plot_robust6_stats.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mri.stats import anova_f, per_feature_auroc  # MRI stats layer

CSV = REPO / "models/routers/lean_ft4/fusion_dataset_lean19.csv"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
ROBUST6 = {"rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"}
META = {"trust_label", "stem", "source", "_seq"}
GREEN, GREY = "#27ae60", "#95a5a6"


def main():
    df = pd.read_csv(CSV)
    feats = [c for c in df.columns if c not in META]
    X = df[feats].to_numpy(float)
    m = np.nanmedian(X, axis=0); nm = np.isnan(X); X[nm] = np.take(m, np.where(nm)[1])
    y = (df["trust_label"].to_numpy() >= 1).astype(int)   # trust(drone) vs reject
    n_src = df["source"].nunique()
    print(f"{CSV.name}: {len(df):,} rows, {len(feats)} feats, sources={n_src}, "
          f"trust={int(y.sum())} reject={int((y==0).sum())}")

    F = anova_f(X, y); A = per_feature_auroc(X, y)
    # leakage: ANOVA-F by SOURCE/scene WITHIN the drone class (isolates fingerprint)
    src = df["source"].to_numpy(); drone = y == 1
    Fdom = anova_f(X[drone], src[drone])
    leak = Fdom / (F + 1.0)
    tbl = pd.DataFrame({"feature": feats, "F": F, "auroc": A, "leak": leak,
                        "r6": [f in ROBUST6 for f in feats]})

    def label(f):
        return ("★ " + f) if f in ROBUST6 else f

    # ── Fig 1: LDA separability histogram ──
    Xs = StandardScaler().fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1).fit(Xs, y)
    z = lda.transform(Xs)[:, 0]; acc = lda.score(Xs, y)
    plt.figure(figsize=(7, 4))
    plt.hist(z[y == 1], bins=80, alpha=0.6, color=GREEN, density=True, label=f"trust / drone (n={int(y.sum()):,})")
    plt.hist(z[y == 0], bins=80, alpha=0.6, color="#e74c3c", density=True, label=f"reject (n={int((y==0).sum()):,})")
    plt.title(f"LDA separability — ft4 trust features (linear acc {acc:.3f})")
    plt.xlabel("LDA component 1"); plt.ylabel("density"); plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "email_robust6_lda.png", dpi=160); plt.close()

    # ── Fig 2: per-feature AUROC bars ──
    t = tbl.sort_values("auroc", ascending=True)
    plt.figure(figsize=(8, 7))
    plt.barh([label(f) for f in t["feature"]], t["auroc"],
             color=[GREEN if r else GREY for r in t["r6"]])
    plt.axvline(0.5, color="k", lw=0.8, ls="--")
    plt.title("Per-feature AUROC (drone-vs-confuser signal) — ★ = robust6")
    plt.xlabel("AUROC-alone  (0.5 = no signal)"); plt.xlim(0.45, 1.0)
    plt.legend([plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=GREY)],
               ["robust6 (kept)", "dropped (scene/position)"], loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(IMG / "email_robust6_auroc.png", dpi=160); plt.close()

    # ── Fig 3: per-feature ANOVA F-statistic bars (log x) ──
    t = tbl.sort_values("F", ascending=True)
    plt.figure(figsize=(8, 7))
    plt.barh([label(f) for f in t["feature"]], t["F"],
             color=[GREEN if r else GREY for r in t["r6"]])
    plt.title("Per-feature ANOVA F-statistic (class discrimination) — ★ = robust6")
    plt.xlabel("ANOVA F  (higher = more discriminative)"); plt.xscale("log")
    plt.legend([plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=GREY)],
               ["robust6 (kept)", "dropped (scene/position)"], loc="lower right", fontsize=9)
    plt.tight_layout(); plt.savefig(IMG / "email_robust6_anova.png", dpi=160); plt.close()

    # ── Fig 4: leakage vs signal (why the high-AUROC position features are dropped) ──
    plt.figure(figsize=(8, 5.6))
    plt.axhspan(0.5, 1.0, xmin=0, xmax=1, color="none")
    for r6f, col, mk, sz, lbl in [(True, GREEN, "*", 320, "robust6 (kept)"),
                                  (False, "#e74c3c", "X", 90, "dropped")]:
        mm = tbl["r6"] == r6f
        plt.scatter(tbl[mm]["leak"], tbl[mm]["auroc"], c=col, marker=mk, s=sz,
                    edgecolor="k", linewidth=0.6, zorder=5, label=lbl)
    for _, r in tbl.iterrows():
        if r["r6"] or (r["auroc"] > 0.80):           # label robust6 + the high-AUROC drops
            plt.annotate(r["feature"], (r["leak"], r["auroc"]), fontsize=7,
                         xytext=(4, 3), textcoords="offset points")
    plt.axhline(0.5, color="gray", ls="--", lw=0.8)
    plt.axvline(1.0, color="gray", ls=":", lw=0.8)
    plt.xscale("log")
    plt.xlabel("leakage  F_domain-in-class / F_class   (→ scene memorisation; want LOW)")
    plt.ylabel("AUROC-alone   (→ drone signal; want HIGH)")
    plt.title("Why robust6 drops high-AUROC features: leakage\n"
              "keep = high signal + LOW leakage (left) · drop = scene fingerprints (right)")
    plt.ylim(0.45, 1.0); plt.legend(loc="lower left"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(IMG / "email_robust6_leakage.png", dpi=160); plt.close()

    # ── Fig 5: feature-count ablation (the real robust6 justification) ──
    # Source: 2026-06-01_statistical_feature_selection_STUDY.md §7.2 (ft4 re-train, GroupShuffleSplit)
    sets = ["meta4 (4)", "robust6 (6)", "no_fp (10)", "all19 (19)"]
    indom = [0.716, 0.725, 0.736, 0.737]   # in-domain Svanström F1
    ood = [0.569, 0.578, 0.436, 0.262]     # OOD drone-video F1-macro
    xx = np.arange(len(sets)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    b1 = ax.bar(xx - w/2, indom, w, color="#3498db", label="in-domain F1 (Svanström)")
    b2 = ax.bar(xx + w/2, ood, w, color="#e67e22", label="OOD F1 (drone-video)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f"{b.get_height():.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(sets)
    ax.set_ylabel("F1"); ax.set_ylim(0, 0.85)
    ax.set_title("Feature-count ablation: fewer features → same in-domain, BETTER OOD\n"
                 "(robust6's 6 ≈ 19 in-domain, 2.2× better OOD — more features memorise & fail)")
    ax.axvspan(0.5, 1.5, color="#27ae60", alpha=0.08)
    ax.legend(loc="upper right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(IMG / "email_robust6_ablation.png", dpi=160); plt.close(fig)

    print("\n  feature                    AUROC   ANOVA-F   leakage  in_robust6")
    for _, r in tbl.sort_values("auroc", ascending=False).iterrows():
        print(f"  {r['feature']:<26} {r['auroc']:.3f}  {r['F']:>8.0f}  {r['leak']:>8.2f}   {'*R6*' if r['r6'] else ''}")
    print("\nSaved: email_robust6_{lda,auroc,anova,leakage}.png")


if __name__ == "__main__":
    main()
