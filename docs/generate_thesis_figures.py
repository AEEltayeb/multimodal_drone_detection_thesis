"""
generate_thesis_figures.py — Generate all data-backed thesis figures.

Reads from eval/results/_cumulative_halluc/ and other result dirs,
produces publication-quality matplotlib figures saved as PDF.

Usage:
    python docs/generate_thesis_figures.py
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "ES_Drone_Detection" / "eval" / "results"
CH_DIR  = RESULTS / "_cumulative_halluc"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_summary(run_name):
    """Load summary.json from a cumulative_halluc run."""
    p = CH_DIR / run_name / "summary.json"
    if not p.exists():
        print(f"  WARNING: {p} not found")
        return None
    with open(p) as f:
        return json.load(f)


# ── Fig 6.1: Cumulative Confuser Suppression ──────────────────────────
def fig_cumulative_confuser():
    """Bar chart: S1 → S2 → S3 fire rates for confuser zoo (fusion_no_fn)."""
    data = load_summary("confuser_fusion_no_fn_model_v1.1")
    if not data: return

    overall = data["overall"]
    stages = ["S1 (YOLO)", "S2 (+Classifier)", "S3 (+Patch)"]
    rates  = [overall["s1_fire_rate"], overall["s2_fire_rate"], overall["s3_fire_rate"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stages, [r * 100 for r in rates], color=["#e74c3c", "#f39c12", "#27ae60"],
                  edgecolor="white", linewidth=1.5, width=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate*100:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.set_ylabel("Confuser Fire Rate (%)")
    ax.set_title(f"Cumulative Confuser Suppression (n={data['n_frames']} OOD images)")
    ax.set_ylim(0, 65)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add suppression annotation
    ax.annotate(f"98.4% suppressed",
                xy=(2, rates[2]*100), xytext=(1.5, 40),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
                fontsize=11, color="#27ae60", fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_1_cumulative_confuser.pdf")
    fig.savefig(OUT_DIR / "fig6_1_cumulative_confuser.png")
    print("  ✓ fig6_1_cumulative_confuser.pdf")
    plt.close(fig)


# ── Fig 6.2: Svanström by-category suppression ───────────────────────
def fig_svanstrom_by_category():
    """Grouped bar: per-category × per-stage fire rates on Svanström."""
    data = load_summary("svanstrom_fusion_no_fn_model_v1.1")
    if not data: return

    cats = ["BIRD", "AIRPLANE", "HELICOPTER"]
    s1 = [data["by_category"][c]["s1_fire_rate"] * 100 for c in cats]
    s2 = [data["by_category"][c]["s2_fire_rate"] * 100 for c in cats]
    s3 = [data["by_category"][c]["s3_fire_rate"] * 100 for c in cats]

    x = np.arange(len(cats))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w, s1, w, label="S1 (YOLO)", color="#e74c3c", edgecolor="white")
    ax.bar(x,     s2, w, label="S2 (+Classifier)", color="#f39c12", edgecolor="white")
    ax.bar(x + w, s3, w, label="S3 (+Patch)", color="#27ae60", edgecolor="white")

    # Add value labels on S1 bars
    for i, v in enumerate(s1):
        ax.text(i - w, v + 1, f"{v:.0f}%", ha='center', fontsize=9, color="#e74c3c")
    for i, v in enumerate(s3):
        ax.text(i + w, v + 1, f"{v:.1f}%", ha='center', fontsize=9, color="#27ae60")

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Fire Rate (%)")
    ax.set_title("Confuser Suppression by Category (Svanström, fusion_no_fn)")
    ax.legend()
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_2_svanstrom_by_category.pdf")
    fig.savefig(OUT_DIR / "fig6_2_svanstrom_by_category.png")
    print("  ✓ fig6_2_svanstrom_by_category.pdf")
    plt.close(fig)


# ── Fig 6.3: Patch Threshold Sweep ───────────────────────────────────
def fig_threshold_sweep():
    """Dual-axis: Drone F1 vs patch_thr, confuser FPs on secondary."""
    thresholds = [0.6, 0.7, 0.8, 0.9]
    drone_f1 = []
    confuser_fps = []

    for thr in thresholds:
        thr_str = f"0{int(thr*10)}"
        run = f"svanstrom_fnfn_thr{thr_str}"
        data = load_summary(run)
        if data and "by_category" in data:
            d = data["by_category"].get("DRONE", {})
            drone_f1.append(d.get("s3_F1", 0))
            # Sum confuser FPs across non-drone categories
            fps = sum(data["by_category"][c].get("s3_FP", 0)
                     for c in data["by_category"] if c != "DRONE")
            confuser_fps.append(fps)

    if not drone_f1:
        print("  WARNING: threshold sweep data missing")
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1, color2 = "#2980b9", "#e74c3c"

    ax1.plot(thresholds, drone_f1, 'o-', color=color1, linewidth=2, markersize=8, label="Drone F1")
    ax1.set_xlabel("Patch Threshold")
    ax1.set_ylabel("Drone F1", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.80, 0.95)

    ax2 = ax1.twinx()
    ax2.bar(thresholds, confuser_fps, width=0.05, alpha=0.4, color=color2, label="Confuser FPs")
    ax2.set_ylabel("Confuser FPs", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark optimal
    best_idx = np.argmax(drone_f1)
    ax1.annotate(f"Optimal: thr={thresholds[best_idx]}",
                xy=(thresholds[best_idx], drone_f1[best_idx]),
                xytext=(thresholds[best_idx]-0.15, drone_f1[best_idx]+0.02),
                arrowprops=dict(arrowstyle="->", color=color1),
                fontsize=10, color=color1, fontweight='bold')

    ax1.set_title("Patch Threshold Operating Point Selection (Svanström)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_3_threshold_sweep.pdf")
    fig.savefig(OUT_DIR / "fig6_3_threshold_sweep.png")
    print("  ✓ fig6_3_threshold_sweep.pdf")
    plt.close(fig)


# ── Fig 7.1: OOD Classifier Comparison ───────────────────────────────
def fig_ood_classifier():
    """Bar chart: 3 classifiers on confuser zoo (S2 fire rate)."""
    runs = {
        "control40":    load_summary("confuser_c40"),
        "scene_aware":  load_summary("confuser_sa32"),
        "fusion_no_fn": load_summary("confuser_fusion_no_fn_model_v1.1"),
    }

    names = list(runs.keys())
    s2 = [runs[n]["overall"]["s2_fire_rate"] * 100 for n in names if runs[n]]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3498db", "#9b59b6", "#27ae60"]
    bars = ax.bar(names, s2, color=colors, edgecolor="white", linewidth=1.5, width=0.5)

    for bar, val in zip(bars, s2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel("OOD Confuser S2 Fire Rate (%)")
    ax.set_title("Classifier OOD Performance (confuser zoo, n=2633)")
    ax.set_ylim(0, 30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_1_ood_classifier.pdf")
    fig.savefig(OUT_DIR / "fig7_1_ood_classifier.png")
    print("  ✓ fig7_1_ood_classifier.pdf")
    plt.close(fig)


# ── Fig 4.x: IR Model Evolution ──────────────────────────────────────
def fig_ir_evolution():
    """Line plot: IR model mAP50 across dataset versions."""
    stages = ["v3", "v4", "v5", "final", "v3b"]
    map50  = [0.900, 0.955, 0.867, 0.958, 0.946]
    labels = [
        "Model-assisted\nlabeling",
        "Human FP\nreview",
        "New data\n(regressed)",
        "Full cleanup +\nreviewer",
        "2-epoch\ncorrective"
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(stages, map50, 'o-', color="#2c3e50", linewidth=2, markersize=10, zorder=5)

    # Color markers by improvement/regression
    for i, (s, m) in enumerate(zip(stages, map50)):
        color = "#27ae60" if i == 0 or m >= map50[i-1] else "#e74c3c"
        ax.plot(s, m, 'o', color=color, markersize=12, zorder=6)
        ax.text(i, m + 0.008, f"{m:.3f}", ha='center', fontweight='bold', fontsize=10)

    # Add intervention labels below
    for i, label in enumerate(labels):
        ax.text(i, m - 0.055 if i != 2 else map50[2] - 0.055, label,
                ha='center', fontsize=8, color="#7f8c8d", style='italic')

    ax.set_ylabel("Validation mAP50")
    ax.set_title("IR Model Evolution Through Human-in-the-Loop Data Curation")
    ax.set_ylim(0.82, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_ir_evolution.pdf")
    fig.savefig(OUT_DIR / "fig4_ir_evolution.png")
    print("  ✓ fig4_ir_evolution.pdf")
    plt.close(fig)


# ── Fig 6.6: Resolution Sensitivity ──────────────────────────────────
def fig_resolution():
    """Simple bar: imgsz 640 vs 1280 recall on Svanström."""
    imgsz   = ["640", "1280"]
    recall  = [0.072, 0.959]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors  = ["#e74c3c", "#27ae60"]
    bars = ax.bar(imgsz, recall, color=colors, edgecolor="white", width=0.4)

    for bar, r in zip(bars, recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{r:.3f}", ha='center', fontweight='bold', fontsize=13)

    ax.set_xlabel("Inference Resolution (imgsz)")
    ax.set_ylabel("Drone Recall")
    ax.set_title("Svanström Drone Recall vs Resolution\n(native 640×512)")
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation
    ax.annotate("13.3× improvement",
                xy=(1, 0.959), xytext=(0.3, 0.7),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
                fontsize=12, color="#27ae60", fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_6_resolution.pdf")
    fig.savefig(OUT_DIR / "fig6_6_resolution.png")
    print("  ✓ fig6_6_resolution.pdf")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output: {OUT_DIR}")
    print(f"Data:   {CH_DIR}")
    print()

    fig_cumulative_confuser()
    fig_svanstrom_by_category()
    fig_threshold_sweep()
    fig_ood_classifier()
    fig_ir_evolution()
    fig_resolution()

    print(f"\nAll figures saved to {OUT_DIR}")
