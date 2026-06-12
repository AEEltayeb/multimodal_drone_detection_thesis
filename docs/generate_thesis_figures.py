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
REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "eval" / "results"
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
    """Dual-axis: Drone F1 vs patch_thr, confuser FPs on secondary.
    thr=0.5 values from thesis table tab:patch_sweep (Drone R=0.817, F1=0.868, FP≈29).
    thr=0.6–0.9 loaded from svanstrom_fnfn_thr* JSON summaries where available."""
    # thr=0.5 datum from thesis table tab:patch_sweep
    thresholds = [0.5]
    drone_f1 = [0.868]
    confuser_fps = [29]

    for thr in [0.6, 0.7, 0.8, 0.9]:
        thr_str = f"0{int(thr*10)}"
        run = f"svanstrom_fnfn_thr{thr_str}"
        data = load_summary(run)
        if data and "by_category" in data:
            d = data["by_category"].get("DRONE", {})
            thresholds.append(thr)
            drone_f1.append(d.get("s3_F1", 0))
            # Sum confuser FPs across non-drone categories
            fps = sum(data["by_category"][c].get("s3_FP", 0)
                     for c in data["by_category"] if c != "DRONE")
            confuser_fps.append(fps)

    if len(drone_f1) < 2:
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
    """P/R trajectory of the IR detector across HITL revisions, on the fixed
    IR_dset_final test split @640 (tab:ir_evolution). V2 from ir_v2_eval_test_640.csv;
    V3-v3b from ir_comparison_test_640. Highlights the V5 precision regression."""
    stages = ["V2", "V3", "V4", "V5", "V6", "Final", "v3b"]
    prec   = [0.661, 0.648, 0.895, 0.768, 0.921, 0.955, 0.957]
    rec    = [0.406, 0.579, 0.669, 0.709, 0.941, 0.980, 0.977]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, prec, 'o-', color="#1f77b4", linewidth=2, markersize=8, label="Precision", zorder=5)
    ax.plot(x, rec,  's-', color="#2ca02c", linewidth=2, markersize=8, label="Recall", zorder=5)
    # mark the V5 precision regression
    v5 = stages.index("V5")
    ax.annotate("V5 regression\n(bulk-ingest, bypassed review)", (v5, prec[v5]),
                textcoords="offset points", xytext=(0, -42), ha="center", fontsize=8,
                color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7))
    ax.plot(v5, prec[v5], 'o', color="#d62728", markersize=11, zorder=6)
    for xi, p in zip(x, prec):
        ax.text(xi, p + 0.012, f"{p:.3f}", ha='center', fontsize=7.5, color="#1f77b4")

    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Precision / Recall")
    ax.set_title("IR detector P/R across HITL revisions (fixed test split, imgsz=640)")
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_ir_evolution.pdf")
    fig.savefig(OUT_DIR / "fig4_ir_evolution.png")
    print("  fig4_ir_evolution.pdf")
    plt.close(fig)


# ── Fig 6.6: Resolution Sensitivity ──────────────────────────────────
def fig_resolution():
    """Bar: Svanström drone recall at imgsz 640 vs 1280. 640 measured on retrained_v2
    (baseline@640 pending); 1280 is baseline. See caption for attribution."""
    imgsz   = ["640\n(retrained_v2)", "1280\n(baseline)"]
    recall  = [0.072, 0.961]

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
                xy=(1, 0.961), xytext=(0.3, 0.7),
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=2),
                fontsize=12, color="#27ae60", fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_6_resolution.pdf")
    fig.savefig(OUT_DIR / "fig6_6_resolution.png")
    print("  ✓ fig6_6_resolution.pdf")
    plt.close(fig)


# ── Fig 5.realvideo: Pareto frontier (drone F1 vs all-confuser FPPI) ──
def fig_realvideo_pareto():
    """Scatter of six detector modes on the joint (drone F1, all-confuser FPPI) axis.
    Hard-coded from Ledger §9.4 because the source CSVs are unified per-run aggregates."""
    detectors = [
        # (label, F1, FPPI, color, marker)
        ("baseline RGB",        0.760, 0.512, "#1f77b4", "o"),
        ("retrained_v2",        0.605, 0.196, "#ff7f0e", "s"),
        ("selcom_1280",         0.721, 0.709, "#2ca02c", "^"),
        ("selcom_640",          0.730, 0.260, "#d62728", "D"),
        ("IR on grayscale-RGB", 0.636, 0.158, "#9467bd", "P"),
        ("IR on raw RGB",       0.295, 0.150, "#8c564b", "X"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, f1, fppi, color, marker in detectors:
        ax.scatter(fppi, f1, s=140, c=color, marker=marker, edgecolor="black",
                   linewidth=1.0, label=label, zorder=3)

    # Pareto frontier: baseline, selcom_640, IR-grayscale
    front = sorted(
        [(fppi, f1, lbl) for lbl, f1, fppi, *_ in detectors
         if lbl in ("baseline RGB", "selcom_640", "IR on grayscale-RGB")],
        key=lambda r: r[0])
    fx = [p[0] for p in front]; fy = [p[1] for p in front]
    ax.plot(fx, fy, "--", color="gray", linewidth=1.4, alpha=0.7,
            label="Pareto frontier", zorder=2)

    ax.set_xlabel("All-confuser FPPI (lower is better)")
    ax.set_ylabel("Aggregate drone $F1$ (higher is better)")
    ax.set_title("Real-video detector Pareto frontier (9 drone + 10 confuser videos)")
    ax.set_xlim(0, 0.78)
    ax.set_ylim(0.25, 0.82)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_realvideo_pareto.pdf")
    fig.savefig(OUT_DIR / "fig5_realvideo_pareto.png")
    print("  ✓ fig5_realvideo_pareto.pdf")
    plt.close(fig)


# ── Fig 5.cascade: Per-category cascade FPR (real video) ──────────────
def fig_cascade_percategory():
    """Grouped bars: per-category confuser FPR before vs after cascade, by RGB model.
    Sourced from eval/results/pipeline_video_tests/{cat}/{video}/{model}.json (sa32)."""
    import glob, collections
    PIPE_DIR = REPO / "eval" / "results" / "pipeline_video_tests"
    if not PIPE_DIR.exists():
        print(f"  WARNING: {PIPE_DIR} not found; skipping per-category figure")
        return

    # Aggregate: per (rgb_model, category) sum of seg_final FP / segments
    seg = collections.defaultdict(lambda: collections.defaultdict(lambda: {"FP": 0, "Total": 0}))
    raw = collections.defaultdict(lambda: collections.defaultdict(lambda: {"frames": 0, "FP": 0}))
    for f in glob.glob(str(PIPE_DIR / "**" / "*.json"), recursive=True):
        if "pipeline_comparison" in f: continue
        with open(f) as fh:
            d = json.load(fh)
        if not d["is_negative"]:
            continue
        cat = d["category"]; m = d["rgb_model"]
        seg[m][cat]["FP"] += d["seg_final"]["FP"]
        seg[m][cat]["Total"] += d["seg_final"]["segments"]
        raw[m][cat]["frames"] += d["total_frames"]
        raw[m][cat]["FP"] += d["rgb_yolo"]["FP"]

    models = ["baseline_trained", "retrained_v2", "selcom_1280", "selcom_640"]
    cats = ["birds", "airplanes", "helicopters"]
    cat_labels = ["Birds", "Airplanes", "Helicopters"]
    model_labels = ["baseline", "retrained_v2", "selcom_1280", "selcom_640"]

    x = np.arange(len(cats))
    w = 0.20

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (m, lbl, col) in enumerate(zip(models, model_labels, colors)):
        fprs = []
        for cat in cats:
            v = seg[m][cat]
            fprs.append(v["FP"] / v["Total"] * 100 if v["Total"] else 0)
        offset = (i - 1.5) * w
        bars = ax.bar(x + offset, fprs, w, label=lbl, color=col, edgecolor="white")
        for bar, v in zip(bars, fprs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", fontsize=8, color=col)

    ax.set_xticks(x); ax.set_xticklabels(cat_labels)
    ax.set_ylabel("Segment-level FPR (%)")
    ax.set_title("Per-category cascade confuser FPR on real-video diagnostic (sa32)")
    ax.set_ylim(0, max(0.255, ax.get_ylim()[1]) * 100 * 1.1 if ax.get_ylim()[1] < 1 else ax.get_ylim()[1])
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y", linestyle=":")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_cascade_percategory.pdf")
    fig.savefig(OUT_DIR / "fig5_cascade_percategory.png")
    print("  ✓ fig5_cascade_percategory.pdf")
    plt.close(fig)


def fig_distill_verifier():
    """Grouped bars: distilled feature-space verifier (mlp_v5) vs patch v2 vs bare FT4.
    Reads drone F1 and hallucination rate straight from knowledge/evals.csv (no drift)."""
    import csv as _csv
    EVALS = REPO / "knowledge" / "evals.csv"
    rows = {r["id"]: r for r in _csv.DictReader(open(EVALS, encoding="utf-8"))}

    # (surface label, bare id, patch id, mlp id)
    surf = [
        ("Svanström", "v5_svan_bare",   "v5_svan_patch",   "v5_svan_mlp"),
        ("Anti-UAV",       "v5_antiuav_bare","v5_antiuav_patch","v5_antiuav_mlp"),
        ("SelCom",         "v5_selcom_bare", "v5_selcom_patch", "v5_selcom_mlp"),
        ("rgb_dataset",    "v5_rgbds_bare",  "v5_rgbds_patch",  "v5_rgbds_mlp"),
        ("confuser",       "v5_confuser_bare","v5_confuser_patch","v5_confuser_mlp"),
    ]
    def val(eid, col):
        v = rows[eid].get(col, "")
        return float(v) if v not in ("", None) else None

    stages = [("bare FT4", 1), ("+ patch v2", 2), ("+ mlp_v5", 3)]
    colors = ["#9e9e9e", "#1f77b4", "#2ca02c"]
    w = 0.26

    fig, (axF, axH) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A — drone F1 (skip confuser surface: no GT)
    fsurf = [s for s in surf if s[0] != "confuser"]
    xF = np.arange(len(fsurf))
    for j, (lbl, col) in enumerate(zip([s[0] for s in stages], colors)):
        ys = [val(s[1 + j], "f1") for s in fsurf]
        ys = [y if y is not None else 0 for y in ys]
        off = (j - 1) * w
        bars = axF.bar(xF + off, ys, w, label=lbl, color=col, edgecolor="white")
        for b, y in zip(bars, ys):
            if y: axF.text(b.get_x() + b.get_width()/2, y + 0.01, f"{y:.2f}",
                           ha="center", fontsize=7.5, color=col)
    axF.set_xticks(xF); axF.set_xticklabels([s[0] for s in fsurf])
    axF.set_ylabel("Drone $F1$"); axF.set_ylim(0, 1.08)
    axF.set_title("(a) Drone $F1$ by surface")
    axF.legend(loc="lower left", framealpha=0.95, fontsize=9)
    axF.grid(True, alpha=0.3, axis="y", linestyle=":")

    # Panel B — hallucination rate (all surfaces incl. confuser)
    xH = np.arange(len(surf))
    for j, (lbl, col) in enumerate(zip([s[0] for s in stages], colors)):
        ys = [val(s[1 + j], "halluc_rate") for s in surf]
        ys = [y if y is not None else 0 for y in ys]
        off = (j - 1) * w
        bars = axH.bar(xH + off, ys, w, label=lbl, color=col, edgecolor="white")
        for b, y in zip(bars, ys):
            if y: axH.text(b.get_x() + b.get_width()/2, y + 0.005, f"{y:.3f}",
                           ha="center", fontsize=7, color=col)
    axH.set_xticks(xH); axH.set_xticklabels([s[0] for s in surf], fontsize=8)
    axH.set_ylabel("Confuser hallucination rate (lower better)")
    axH.set_title("(b) Hallucination rate by surface")
    axH.grid(True, alpha=0.3, axis="y", linestyle=":")
    for ax in (axF, axH):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle("Distilled feature-space verifier (mlp_v5) vs MobileNetV3 patch v2, on FT4", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig8_distill_verifier.pdf")
    fig.savefig(OUT_DIR / "fig8_distill_verifier.png")
    print("  ✓ fig8_distill_verifier.pdf")
    plt.close(fig)


def fig_rgb_threestance():
    """Scatter: the recall-vs-hallucination cliff across the 3 RGB training stances.
    Values verified against tab:rgb_comparison (by-category cache svanstrom_1280_by_category.csv)
    + ledger retrainedv2-recall-collapse. x=bird-frame fire rate, y=Svanstrom drone recall."""
    # (label, bird_fire_%, drone_R)
    pts = [("baseline", 94.4, 0.959),
           ("hardneg_v3more", 94.2, 0.950),
           ("retrained_v2", 3.4, 0.306)]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    cols = ["#1f77b4", "#ff7f0e", "#d62728"]
    # label offsets to avoid baseline/hardneg overlap (both near 94% fire)
    offs = [(10, -16), (10, 10), (12, 8)]
    for (lbl, x, y), c, off in zip(pts, cols, offs):
        ax.scatter(x, y, s=140, color=c, zorder=3, edgecolor="white", linewidth=1.2)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=off, fontsize=10, color=c)
    # guide line connecting the trade
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    ax.plot(xs, ys, ":", color="grey", alpha=0.6, zorder=1)
    ax.set_xlabel("Bird-frame fire rate (%, Svanstrom bird-only frames)")
    ax.set_ylabel("Svanstrom drone recall")
    ax.set_title("The recall vs hallucination cliff across RGB training stances")
    ax.set_xlim(-5, 105); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.annotate("suppressing birds\ncollapses small-drone recall", (3.4, 0.306),
                textcoords="offset points", xytext=(40, 40), fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_rgb_threestance.pdf")
    fig.savefig(OUT_DIR / "fig8_rgb_threestance.png")
    print("  fig8_rgb_threestance.pdf")
    plt.close(fig)


def fig_cascade_segment():
    """Dumbbell: Stage-1 RGB vs full cascade, segment-level, 4 RGB variants.
    Values verified against tab:cascade_segment (evals: vid_drone_*, pipe_vid_*_seg,
    pipe_vidconf_*_seg; ledger cascade-segment-recovers / cascade-tightens-variance)."""
    models = ["baseline", "retrained_v2", "selcom_1280", "selcom_640"]
    rgb_f1   = [0.760, 0.605, 0.721, 0.730]
    casc_f1  = [0.826, 0.770, 0.814, 0.816]
    rgb_fpr  = [0.512, 0.196, 0.709, 0.260]
    casc_fpr = [0.162, 0.119, 0.136, 0.126]
    y = np.arange(len(models))[::-1]

    fig, (axF, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    # Panel A: drone F1 RGB -> cascade (gain)
    for yi, a, b in zip(y, rgb_f1, casc_f1):
        axF.plot([a, b], [yi, yi], "-", color="#bbbbbb", zorder=1, linewidth=2)
    axF.scatter(rgb_f1, y, color="#9e9e9e", s=90, label="Stage-1 RGB", zorder=3, edgecolor="white")
    axF.scatter(casc_f1, y, color="#2ca02c", s=90, label="Full cascade", zorder=3, edgecolor="white")
    for yi, b in zip(y, casc_f1):
        axF.annotate(f"{b:.3f}", (b, yi), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#2ca02c")
    axF.set_yticks(y); axF.set_yticklabels(models)
    axF.set_xlabel("Segment-level drone $F1$"); axF.set_xlim(0.55, 0.9)
    axF.set_title("(a) Drone $F1$: cascade lifts every variant")
    axF.legend(loc="lower right", fontsize=9); axF.grid(True, axis="x", alpha=0.3, linestyle=":")

    # Panel B: confuser FPR RGB -> cascade (cut)
    for yi, a, b in zip(y, rgb_fpr, casc_fpr):
        axR.plot([b, a], [yi, yi], "-", color="#bbbbbb", zorder=1, linewidth=2)
    axR.scatter(rgb_fpr, y, color="#9e9e9e", s=90, label="Stage-1 RGB", zorder=3, edgecolor="white")
    axR.scatter(casc_fpr, y, color="#1f77b4", s=90, label="Full cascade", zorder=3, edgecolor="white")
    for yi, a, b in zip(y, rgb_fpr, casc_fpr):
        axR.annotate(f"-{(1-b/a)*100:.0f}%", ((a+b)/2, yi), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="#1f77b4")
    axR.set_yticks(y); axR.set_yticklabels([])
    axR.set_xlabel("Segment-level confuser FPR"); axR.set_xlim(0, 0.78)
    axR.set_title("(b) Confuser FPR: cascade cuts every variant")
    axR.legend(loc="upper right", fontsize=9); axR.grid(True, axis="x", alpha=0.3, linestyle=":")
    for ax in (axF, axR):
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle("Cascade vs Stage-1 RGB on real video (segment grain, sa32 classifier)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "fig8_cascade_segment.pdf")
    fig.savefig(OUT_DIR / "fig8_cascade_segment.png")
    print("  fig8_cascade_segment.pdf")
    plt.close(fig)


def fig_surface_exchange():
    """RQ2 keystone: cascade COSTS F1 on Svanstrom-paired but GAINS on real video.
    Backed: rgb_svan_baseline (S1 0.950), svan_s3_sa32_thr08 (S3 0.896),
    vid_drone_baseline (0.760), pipe_vid_baseline_seg (0.826)."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    surfaces = ["Svanstrom paired\n(in-distribution)", "Real video\n(operational)"]
    s1 = [0.950, 0.760]; casc = [0.895, 0.826]
    x = np.arange(len(surfaces)); w = 0.32
    b1 = ax.bar(x - w/2, s1, w, label="Stage-1 RGB", color="#9e9e9e", edgecolor="white")
    b2 = ax.bar(x + w/2, casc, w, label="Full cascade (S3)", color="#2ca02c", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.006, f"{b.get_height():.3f}",
                    ha="center", fontsize=9)
    # delta arrows
    ax.annotate("", xy=(0+w/2, 0.905), xytext=(0-w/2, 0.945),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
    ax.text(0, 0.965, "$-5.5$ pp", ha="center", color="#d62728", fontsize=9)
    ax.annotate("", xy=(1+w/2, 0.820), xytext=(1-w/2, 0.766),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
    ax.text(1, 0.845, "$+6.6$ pp", ha="center", color="#1f77b4", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(surfaces)
    ax.set_ylabel("Drone $F1$"); ax.set_ylim(0.5, 1.0)
    ax.set_title("The cascade's drone-$F1$ exchange is surface-dependent (RQ2)")
    ax.legend(loc="lower center", fontsize=9); ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_surface_exchange.pdf"); fig.savefig(OUT_DIR / "fig8_surface_exchange.png")
    print("  fig8_surface_exchange.pdf"); plt.close(fig)


def fig_patch_catchbar():
    """H2: patch-v2 per-bucket catch vs the 0.90 decisiveness bar. Backed: patch_catch_v2_svan."""
    buckets = ["Helicopter", "Bird", "Airplane"]
    catch = [0.709, 0.638, 0.517]; medprob = [0.987, 0.904, 0.540]
    y = np.arange(len(buckets))[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(y, catch, 0.55, color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="white")
    for yi, c, mp in zip(y, catch, medprob):
        ax.text(c+0.012, yi, f"{c*100:.0f}% catch  (median $p$={mp:.2f})", va="center", fontsize=9)
    ax.axvline(0.90, color="grey", linestyle="--", lw=1.3)
    ax.text(0.905, max(y)+0.45, "0.90 decisiveness bar", color="grey", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(buckets)
    ax.set_xlabel("Confuser catch rate (patch v2, \\texttt{patch\\_thr}=0.5)")
    ax.set_xlim(0, 1.15)
    ax.text(0.054, -0.05, "(drone-TP veto only 5.4\\%)", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#555")
    ax.set_title("Patch verifier: every bucket below the 0.90 bar; airplane is the gap")
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_patch_catchbar.pdf"); fig.savefig(OUT_DIR / "fig8_patch_catchbar.png")
    print("  fig8_patch_catchbar.pdf"); plt.close(fig)


def fig_perframe_segment():
    """M2: the grain reversal V -- classifier appears to hurt per-frame, segment recovers it.
    Backed: vid_drone_* (RGB), pipe_vid_*_pf (+classifier per-frame), pipe_vid_*_seg (segment)."""
    models = ["baseline", "retrained_v2", "selcom_1280", "selcom_640"]
    rgb = [0.760, 0.605, 0.721, 0.730]
    pf  = [0.586, 0.615, 0.537, 0.568]
    seg = [0.826, 0.770, 0.814, 0.816]
    xs = [0, 1, 2]; xlab = ["RGB stage", "+classifier\n(per-frame)", "full cascade\n(segment)"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m, c, a, b, d in zip(models, cols, rgb, pf, seg):
        ax.plot(xs, [a, b, d], "-o", color=c, label=m, markersize=6)
    ax.set_xticks(xs); ax.set_xticklabels(xlab)
    ax.set_ylabel("Drone $F1$"); ax.set_ylim(0.45, 0.9)
    ax.set_title("Reading per-frame is misleading: the segment grain recovers the drop")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_perframe_segment.pdf"); fig.savefig(OUT_DIR / "fig8_perframe_segment.png")
    print("  fig8_perframe_segment.pdf"); plt.close(fig)


def fig_classifier_reversal():
    """M3: no classifier wins both surfaces. x=OOD-zoo fire (lower safer), y=real-video drone F1.
    Backed: clfzoo_sa32/fnfn/control40 (S2 fire), pipe3clf_sa32/control40/fnfn (real-video F1)."""
    clfs = [("sa32", 0.205, 0.826, "#2ca02c"),
            ("control40", 0.212, 0.644, "#ff7f0e"),
            ("fnfn", 0.016, 0.219, "#1f77b4")]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for lbl, fire, f1, c in clfs:
        ax.scatter(fire, f1, s=150, color=c, edgecolor="white", zorder=3)
        ax.annotate(lbl, (fire, f1), textcoords="offset points", xytext=(10, 6), fontsize=10, color=c)
    ax.set_xlabel("OOD confuser-zoo fire rate (S2, lower = safer)")
    ax.set_ylabel("Real-video drone segment $F1$ (higher = better)")
    ax.set_title("No classifier wins both surfaces (basis for the sa32 production pick)")
    ax.set_xlim(-0.02, 0.26); ax.set_ylim(0.1, 0.92)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.annotate("safe on zoo,\ncollapses on video", (0.016, 0.219), textcoords="offset points",
                xytext=(28, 18), fontsize=8, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", alpha=0.6))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_classifier_reversal.pdf"); fig.savefig(OUT_DIR / "fig8_classifier_reversal.png")
    print("  fig8_classifier_reversal.pdf"); plt.close(fig)


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
    fig_realvideo_pareto()
    fig_cascade_percategory()
    fig_distill_verifier()
    fig_rgb_threestance()
    fig_cascade_segment()
    fig_surface_exchange()
    fig_patch_catchbar()
    fig_perframe_segment()
    fig_classifier_reversal()

    print(f"\nAll figures saved to {OUT_DIR}")
