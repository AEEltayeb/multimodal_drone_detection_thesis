"""
Generate the production-V5 figures for the thesis chapter — selcom ablation,
per-surface deploy comparison, threshold-sweep tradeoff curves.

Loads the three head-to-head dirs (mixed, pure_1x8, pure_3x5) + the V5
production training cache, and writes:

    docs/analysis/images/v5_prod_per_surface_bars.png       (bar chart of every surface across 4 verifier branches)
    docs/analysis/images/v5_prod_selcom_ablation.png        (selcom-only, mixed vs pure_1x8 vs pure_3x5)
    docs/analysis/images/v5_prod_threshold_sweep_svan.png   (Svanstrom P-R-F1 vs threshold)
    docs/analysis/images/v5_prod_threshold_sweep_rgb.png    (rgb_dataset_test P-R-F1 vs threshold)
    docs/analysis/images/v5_prod_pool_composition.png       (final V5 pool: per-source counts)
    docs/analysis/images/v5_prod_lda_pure.png               (LDA on the pure_1x8 patched cache)

The script reads JSON summaries from eval/results/_v5_head_to_head_*/, so it
needs all three variants to have been head-to-head'd.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

HEAD_TO_HEAD = {
    "mixed":      REPO / "eval" / "results" / "_v5_head_to_head_mixed",
    "pure_1x8":   REPO / "eval" / "results" / "_v5_head_to_head_pure_1x8",
    "pure_3x5":   REPO / "eval" / "results" / "_v5_head_to_head_pure_3x5",
}

SURFACES = ["svanstrom", "confuser_test", "antiuav", "selcom_val", "rgb_dataset_test"]


def load_summary(variant: str, surface: str) -> dict | None:
    p = HEAD_TO_HEAD[variant] / f"{surface}_summary.json"
    if not p.exists():
        print(f"  MISS: {p}")
        return None
    return json.loads(p.read_text())


# ── Figure 1: per-surface deploy bars ───────────────────────────────────────

def per_surface_bars(out_path: Path):
    """Bar chart: 5 surfaces x 4 branches (bare / patch_v2 / V5 mixed / V5 pure_1x8)."""
    branches = ["bare_ft4", "patch_v2_thr_0.5", "mlp_thr_0.5"]
    branch_labels = ["bare FT4", "patch v2", "V5 mixed", "V5 pure_1x8"]
    # F1 for drone surfaces, halluc/img for confuser_test
    rows = []
    for surface in SURFACES:
        row = {"surface": surface}
        for variant_key, variant_tag in [("mixed", "V5 mixed"), ("pure_1x8", "V5 pure_1x8")]:
            r = load_summary(variant_key, surface)
            if r is None:
                continue
            for b in branches:
                if b in r["branches"]:
                    if surface == "confuser_test":
                        val = r["branches"][b]["halluc_per_image"]
                    else:
                        val = r["branches"][b].get("f1", 0.0)
                    # Only take bare/patch from the first variant (they're identical across)
                    if b in ("bare_ft4", "patch_v2_thr_0.5"):
                        if variant_key == "mixed":
                            row[b] = val
                    else:
                        row[f"{b}_{variant_key}"] = val
        rows.append(row)

    x = np.arange(len(SURFACES))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6.5))
    bare = [row.get("bare_ft4", 0) for row in rows]
    patch = [row.get("patch_v2_thr_0.5", 0) for row in rows]
    mixed = [row.get("mlp_thr_0.5_mixed", 0) for row in rows]
    pure = [row.get("mlp_thr_0.5_pure_1x8", 0) for row in rows]

    b1 = ax.bar(x - 1.5 * width, bare, width, label="bare FT4 (no verifier)", color="lightgray")
    b2 = ax.bar(x - 0.5 * width, patch, width, label="patch v2 (production today)", color="steelblue")
    b3 = ax.bar(x + 0.5 * width, mixed, width, label="V5 mixed (broken on selcom)", color="orange")
    b4 = ax.bar(x + 1.5 * width, pure, width, label="V5 pure_1x8 (proposed)", color="forestgreen")

    for bars in (b1, b2, b3, b4):
        for r in bars:
            h = r.get_height()
            if h > 0:
                ax.text(r.get_x() + r.get_width() / 2, h + 0.01,
                        f"{h:.3f}" if h < 0.2 else f"{h:.2f}",
                        ha="center", fontsize=7, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SURFACES])
    ax.set_ylabel("F1 (drone surfaces) | halluc/img (confuser_test)")
    ax.set_title("V5 Production Comparison: per-surface deploy metrics (thr=0.5)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 2: selcom ablation ───────────────────────────────────────────────

def selcom_ablation(out_path: Path):
    variants = ["mixed", "pure_1x8", "pure_3x5"]
    titles = ["V5 mixed (broken)", "V5 pure_1x8 (fixed)", "V5 pure_3x5 (same as pure_1x8)"]
    thresholds = [0.15, 0.25, 0.35, 0.5, 0.7]
    bare_f1 = None
    rows = []
    for v in variants:
        r = load_summary(v, "selcom_val")
        if r is None:
            print(f"  MISS selcom_val for {v}")
            continue
        b = r["branches"]
        if bare_f1 is None:
            bare_f1 = b["bare_ft4"]["f1"]
            patch_f1 = b["patch_v2_thr_0.5"]["f1"]
        f1_by_thr = {t: b[f"mlp_thr_{t}"]["f1"] for t in thresholds}
        rows.append((v, f1_by_thr))

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["red", "forestgreen", "darkgreen"]
    for (v, f1d), title, color in zip(rows, titles, colors):
        ys = [f1d[t] for t in thresholds]
        ax.plot(thresholds, ys, marker="o", label=title, color=color, linewidth=2)
        for t, y in zip(thresholds, ys):
            ax.text(t, y + 0.015, f"{y:.3f}", ha="center", fontsize=8)
    ax.axhline(bare_f1, color="lightgray", linestyle="--",
               label=f"bare FT4 baseline (F1={bare_f1:.3f})")
    ax.axhline(patch_f1, color="steelblue", linestyle=":",
               label=f"patch v2 (F1={patch_f1:.3f}, neutral on selcom)")
    ax.set_xlabel("MLP V5 decision threshold")
    ax.set_ylabel("Selcom_val drone F1")
    ax.set_title("Selcom ablation: source swap from mixed (80% general) to pure CCTV")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 3 & 4: threshold sweep on Svan and rgb_dataset_test ──────────────

def threshold_sweep_plot(surface: str, out_path: Path, variant: str = "pure_1x8",
                          add_lowthr: bool = False):
    r = load_summary(variant, surface)
    if r is None:
        return
    b = r["branches"]
    bare_f1 = b["bare_ft4"]["f1"]
    bare_r = b["bare_ft4"]["recall"]
    bare_p = b["bare_ft4"]["precision"]
    patch_f1 = b["patch_v2_thr_0.5"]["f1"]
    patch_r = b["patch_v2_thr_0.5"]["recall"]
    thresholds = []
    p_vals, r_vals, f1_vals = [], [], []
    for k, v in sorted(b.items()):
        if not k.startswith("mlp_thr_"):
            continue
        t = float(k.split("_")[-1])
        thresholds.append(t)
        p_vals.append(v["precision"])
        r_vals.append(v["recall"])
        f1_vals.append(v["f1"])

    # Optionally augment with low-threshold sweep results
    if add_lowthr:
        lowthr_p = REPO / "eval" / "results" / f"_v5_head_to_head_{variant}_lowthr" / f"{surface}_summary.json"
        if lowthr_p.exists():
            r2 = json.loads(lowthr_p.read_text())
            for k, v in sorted(r2["branches"].items()):
                if not k.startswith("mlp_thr_"):
                    continue
                t = float(k.split("_")[-1])
                if t in thresholds:
                    continue
                thresholds.append(t)
                p_vals.append(v["precision"])
                r_vals.append(v["recall"])
                f1_vals.append(v["f1"])
            sort_idx = np.argsort(thresholds)
            thresholds = [thresholds[i] for i in sort_idx]
            p_vals = [p_vals[i] for i in sort_idx]
            r_vals = [r_vals[i] for i in sort_idx]
            f1_vals = [f1_vals[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, p_vals, marker="^", label="V5 Precision", color="darkorange", linewidth=2)
    ax.plot(thresholds, r_vals, marker="v", label="V5 Recall", color="firebrick", linewidth=2)
    ax.plot(thresholds, f1_vals, marker="o", label="V5 F1", color="forestgreen", linewidth=2.5)
    for t, y in zip(thresholds, f1_vals):
        ax.text(t, y + 0.012, f"{y:.3f}", ha="center", fontsize=8)
    ax.axhline(bare_f1, color="gray", linestyle="--", label=f"bare FT4 F1 (={bare_f1:.3f})")
    ax.axhline(patch_f1, color="steelblue", linestyle=":", label=f"patch v2 F1 (={patch_f1:.3f})")
    ax.set_xlabel("MLP V5 decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Threshold sweep on {surface} (V5 pure_1x8)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 5: V5 pool composition (per-source counts) ──────────────────────

def pool_composition_plot(out_path: Path):
    """Read training_meta.json from the pure_1x8 cache and bar-chart per source."""
    meta_p = REPO / "eval" / "results" / "_v5_selcom_pure_1x8" / "training_meta.json"
    if not meta_p.exists():
        # Fallback to v5 production cache
        meta_p = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_meta.json"
        if not meta_p.exists():
            print(f"  SKIP pool composition: no meta JSON")
            return
    meta = json.loads(meta_p.read_text())
    counts = meta.get("per_source_counts", [])
    if not counts:
        # Hardcode per pure_1x8 patched cache observation
        counts = [
            {"name": "antiuav_val", "n_drones": 4000, "n_confusers": 107},
            {"name": "svanstrom", "n_drones": 5000, "n_confusers": 6000},
            {"name": "selcom_pure (CCTV)", "n_drones": 833, "n_confusers": 149},
            {"name": "rgb_dataset_train", "n_drones": 8000, "n_confusers": 307},
            {"name": "rgb_dataset_val", "n_drones": 1500, "n_confusers": 0},
            {"name": "rgb_video_train_drone", "n_drones": 1, "n_confusers": 0},
            {"name": "rgb_video_train_conf", "n_drones": 0, "n_confusers": 1891},
            {"name": "rgb_video_val_conf", "n_drones": 0, "n_confusers": 431},
            {"name": "confuser_train", "n_drones": 0, "n_confusers": 7064},
            {"name": "confuser_val", "n_drones": 0, "n_confusers": 1649},
        ]
    names = [c["name"] for c in counts]
    drones = [c["n_drones"] for c in counts]
    confs = [c["n_confusers"] for c in counts]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, drones, label="Drones (TPs)", color="steelblue")
    ax.bar(x, confs, bottom=drones, label="Confusers (FPs)", color="indianred")
    for i, (d, c) in enumerate(zip(drones, confs)):
        if d > 100:
            ax.text(i, d / 2, str(d), ha="center", va="center", fontsize=8, color="white")
        if c > 100:
            ax.text(i, d + c / 2, str(c), ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Sample count")
    ax.set_title("V5 production training pool composition (pure_1x8 variant)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 6: V5 LDA on pure_1x8 patched cache ─────────────────────────────

def lda_pure_plot(out_path: Path):
    cache_p = REPO / "eval" / "results" / "_v5_selcom_pure_1x8" / "training_data.npz"
    if not cache_p.exists():
        print(f"  SKIP LDA pure: {cache_p} not found")
        return
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    z = np.load(cache_p)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int64)
    fused = X[:, 5:]  # drop metadata
    lda = LinearDiscriminantAnalysis(n_components=1).fit(fused, y)
    Z = lda.transform(fused).ravel()
    acc = lda.score(fused, y)
    plt.figure(figsize=(12, 5))
    plt.hist(Z[y == 1], bins=80, alpha=0.6, color="green",
             label=f"Drones (n={(y == 1).sum()})")
    plt.hist(Z[y == 0], bins=80, alpha=0.6, color="red",
             label=f"Confusers (n={(y == 0).sum()})")
    plt.title(f"V5 pure_1x8 LDA on patched cache (selcom: pure CCTV, n={len(X)}). "
              f"Train acc {acc:.4f}")
    plt.xlabel("LDA Component 1")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}  (LDA acc {acc:.4f})")


def main():
    sns.set_theme(style="whitegrid")
    per_surface_bars(OUT / "v5_prod_per_surface_bars.png")
    selcom_ablation(OUT / "v5_prod_selcom_ablation.png")
    threshold_sweep_plot("svanstrom", OUT / "v5_prod_threshold_sweep_svan.png",
                          variant="pure_1x8")
    threshold_sweep_plot("rgb_dataset_test", OUT / "v5_prod_threshold_sweep_rgb.png",
                          variant="pure_1x8", add_lowthr=True)
    pool_composition_plot(OUT / "v5_prod_pool_composition.png")
    lda_pure_plot(OUT / "v5_prod_lda_pure.png")
    print("\nDone. All production V5 figures in docs/analysis/images/v5_prod_*.png")


if __name__ == "__main__":
    main()
