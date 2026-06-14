"""
Generate latency / speed figures for the V5 quick pipeline eval (§13).

Reads eval/results/_v5_pipeline_quick/*_summary.json and writes:

    docs/analysis/images/v5_prod_latency_per_det.png   (V5 vs patch v2 per-detection)
    docs/analysis/images/v5_prod_pipeline_overhead.png (per-frame pipeline ms vs bare baseline)
    docs/analysis/images/v5_prod_pf_vs_ag.png          (per-frame vs alert-gated F1 + ms)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "eval" / "results" / "_v5_pipeline_quick"
OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def load_all():
    results = {}
    for p in sorted(IN_DIR.glob("*_summary.json")):
        ds = p.stem.replace("_summary", "")
        results[ds] = json.loads(p.read_text())
    return results


# ── Figure 1: per-detection latency bar chart ──────────────────────────────

def per_det_latency_plot(results: dict, out_path: Path):
    surfaces = list(results.keys())
    patch_ms = []
    v5_ms = []
    for ds in surfaces:
        r = results[ds]
        patch_ms.append(r["latency"]["patch_v2_per_detection"]["mean_ms"])
        v5_ms.append(r["latency"]["v5_mlp_per_detection"]["mean_ms"])

    x = np.arange(len(surfaces))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_p = ax.bar(x - w / 2, patch_ms, w, label="Patch v2 (MobileNet-V3 on 224×224 crop)", color="steelblue")
    bars_v = ax.bar(x + w / 2, v5_ms, w, label="V5 MLP (forward on 517-D features)", color="forestgreen")
    for bars in (bars_p, bars_v):
        for r in bars:
            h = r.get_height()
            if h > 0:
                ax.text(r.get_x() + r.get_width() / 2, h + max(patch_ms) * 0.01,
                        f"{h:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(surfaces)
    ax.set_ylabel("Mean per-detection latency (ms)")
    ax.set_title("Per-detection verifier latency — V5 MLP vs patch v2 (lower is better)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    # Annotate speedup ratios
    for i, (p, v) in enumerate(zip(patch_ms, v5_ms)):
        if v > 0:
            ratio = p / v
            ax.text(i, max(patch_ms) * 1.05, f"V5 is {ratio:.1f}× faster",
                    ha="center", fontsize=10, color="darkgreen", weight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 2: per-frame pipeline ms overhead ──────────────────────────────

def pipeline_overhead_plot(results: dict, out_path: Path):
    surfaces = list(results.keys())
    branches = ["bare_ft4", "patch_v2_pf", "patch_v2_ag", "v5_mlp_pf", "v5_mlp_ag"]
    branch_colors = {"bare_ft4": "lightgray", "patch_v2_pf": "steelblue",
                     "patch_v2_ag": "lightblue", "v5_mlp_pf": "forestgreen",
                     "v5_mlp_ag": "lightgreen"}
    data = {b: [] for b in branches}
    for ds in surfaces:
        for b in branches:
            data[b].append(results[ds]["latency"]["pipeline_per_frame"][b]["mean_ms"])

    x = np.arange(len(surfaces))
    w = 0.16
    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, b in enumerate(branches):
        offset = (i - 2) * w
        ax.bar(x + offset, data[b], w, label=b, color=branch_colors[b])
    ax.set_xticks(x)
    ax.set_xticklabels(surfaces)
    ax.set_ylabel("Mean per-frame pipeline latency (ms)")
    ax.set_title("Per-frame pipeline latency per branch (YOLO + verifier compute)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


# ── Figure 3: PF vs AG comparison ─────────────────────────────────────────

def pf_vs_ag_plot(results: dict, out_path: Path):
    """Per surface: V5 PF F1, V5 AG F1, patch v2 PF F1, patch v2 AG F1 grouped bars."""
    surfaces = [ds for ds, r in results.items()
                if "precision" in r["branches"].get("v5_mlp_pf", {})]
    if not surfaces:
        print("  SKIP PF vs AG: no drone surfaces measured")
        return

    v5_pf_f1 = [results[ds]["branches"]["v5_mlp_pf"]["f1"] for ds in surfaces]
    v5_ag_f1 = [results[ds]["branches"]["v5_mlp_ag"]["f1"] for ds in surfaces]
    pa_pf_f1 = [results[ds]["branches"]["patch_v2_pf"]["f1"] for ds in surfaces]
    pa_ag_f1 = [results[ds]["branches"]["patch_v2_ag"]["f1"] for ds in surfaces]

    x = np.arange(len(surfaces))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * w, pa_pf_f1, w, label="Patch v2 per-frame", color="steelblue")
    ax.bar(x - 0.5 * w, pa_ag_f1, w, label="Patch v2 alert-gated", color="lightblue")
    ax.bar(x + 0.5 * w, v5_pf_f1, w, label="V5 per-frame", color="forestgreen")
    ax.bar(x + 1.5 * w, v5_ag_f1, w, label="V5 alert-gated", color="lightgreen")
    for i, (a, b, c, d) in enumerate(zip(pa_pf_f1, pa_ag_f1, v5_pf_f1, v5_ag_f1)):
        for offset, v in zip([-1.5, -0.5, 0.5, 1.5], [a, b, c, d]):
            ax.text(i + offset * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(surfaces)
    ax.set_ylabel("Drone F1")
    ax.set_title("Per-frame vs Alert-gated: F1 across surfaces, V5 vs patch v2")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Wrote {out_path}")


def main():
    sns.set_theme(style="whitegrid")
    results = load_all()
    if not results:
        print(f"FATAL: no summaries found in {IN_DIR}")
        return
    per_det_latency_plot(results, OUT / "v5_prod_latency_per_det.png")
    pipeline_overhead_plot(results, OUT / "v5_prod_pipeline_overhead.png")
    pf_vs_ag_plot(results, OUT / "v5_prod_pf_vs_ag.png")
    print("\nDone. Figures in docs/analysis/images/v5_prod_*.png")


if __name__ == "__main__":
    main()
