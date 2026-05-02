"""Generate thesis-ready plots for YouTube OOD IR filter evaluation.
Reads per_video.csv and category_summary.csv from eval_youtube_ir/.

Usage:
    python classifier/generate_youtube_plots.py
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "runs" / "eval_youtube_ir"


def main():
    # ── Load per-video data ──────────────────────────────────────────
    pv_path = OUT / "per_video.csv"
    if not pv_path.exists():
        print(f"Missing {pv_path} — run eval_youtube_ir_filter.py first")
        return
    rows = list(csv.DictReader(open(pv_path)))
    for r in rows:
        r["frames"] = int(r["frames"])
        r["ir_only_det_rate"] = float(r["ir_only_det_rate"])
        r["ir_filter_det_rate"] = float(r["ir_filter_det_rate"])
        r["filter_suppression"] = float(r["filter_suppression"])

    # ── 1. Per-video bar chart: ir_only vs ir_filter ─────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    names = []
    for r in rows:
        q = f" [{r.get('quality','')}]" if r.get("quality") else ""
        names.append(f"{r['video'].replace('yt_','').replace('.mp4','')}\n({r['category']}{q})")
    x = np.arange(len(rows))
    w = 0.35
    raw = [r["ir_only_det_rate"] * 100 for r in rows]
    flt = [r["ir_filter_det_rate"] * 100 for r in rows]
    
    colors_raw = []
    colors_flt = []
    for r in rows:
        if r["category"] == "DRONE":
            colors_raw.append("#2ecc71")  # green
            colors_flt.append("#27ae60")
        else:
            colors_raw.append("#e74c3c")  # red
            colors_flt.append("#c0392b")
    
    bars1 = ax.bar(x - w/2, raw, w, label="ir_only", color=colors_raw, alpha=0.7,
                   edgecolor="black", linewidth=0.3)
    bars2 = ax.bar(x + w/2, flt, w, label="ir_filter", color=colors_flt, alpha=0.9,
                   edgecolor="black", linewidth=0.3)
    
    for i, (v1, v2) in enumerate(zip(raw, flt)):
        if v1 > 0:
            ax.text(x[i] - w/2, v1 + 1, f"{v1:.0f}%", ha="center", fontsize=6)
        if v2 > 0 or v1 > 0:
            ax.text(x[i] + w/2, v2 + 1, f"{v2:.0f}%", ha="center", fontsize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("YouTube OOD — IR Detection Rate: ir_only vs ir_filter\n"
                 "(Green = DRONE, Red = CONFUSER)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT / "per_video_bars.png", dpi=140)
    plt.close(fig)
    print("  per_video_bars.png saved")

    # ── 2. Category summary bar chart ────────────────────────────────
    cat_path = OUT / "category_summary.csv"
    cats = list(csv.DictReader(open(cat_path)))
    for c in cats:
        c["ir_only_det_rate"] = float(c["ir_only_det_rate"]) * 100
        c["ir_filter_det_rate"] = float(c["ir_filter_det_rate"]) * 100
        c["suppression"] = float(c["suppression"]) * 100
        c["total_frames"] = int(c["total_frames"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: detection rates
    cat_names = [c["category"] for c in cats]
    x = np.arange(len(cats))
    w = 0.35
    raw_cat = [c["ir_only_det_rate"] for c in cats]
    flt_cat = [c["ir_filter_det_rate"] for c in cats]
    cat_colors = ["#2ecc71" if c["category"] == "DRONE" else "#e74c3c" for c in cats]

    ax1.bar(x - w/2, raw_cat, w, label="ir_only", color=[c + "99" for c in cat_colors],
            edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, flt_cat, w, label="ir_filter", color=cat_colors,
            edgecolor="black", linewidth=0.5)
    for i, (v1, v2) in enumerate(zip(raw_cat, flt_cat)):
        ax1.text(x[i] - w/2, v1 + 1, f"{v1:.1f}%", ha="center", fontsize=8)
        ax1.text(x[i] + w/2, v2 + 1, f"{v2:.1f}%", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_names, fontsize=10)
    ax1.set_ylabel("Detection Rate (%)")
    ax1.set_title("Detection Rate by Category")
    ax1.legend()
    ax1.set_ylim(0, 55)
    ax1.grid(axis="y", alpha=0.3)

    # Right: suppression rates
    supp = [c["suppression"] for c in cats]
    bars = ax2.bar(x, supp, 0.5, color=cat_colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(supp):
        ax2.text(x[i], v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, fontsize=10)
    ax2.set_ylabel("Suppression Rate (%)")
    ax2.set_title("Filter Suppression Rate\n(Confusers: higher = better | Drone: lower = better)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("YouTube OOD — IR Confuser Filter Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "category_summary_bars.png", dpi=140)
    plt.close(fig)
    print("  category_summary_bars.png saved")

    # ── 3. Confusion-style summary: 2x2 for filter effectiveness ─────
    # Drone CLEAN only
    clean = [r for r in rows if r.get("quality") == "CLEAN"]
    confusers = [r for r in rows if r["category"] != "DRONE"]

    clean_frames = sum(r["frames"] for r in clean) if clean else 0
    clean_raw_det = sum(int(r.get("ir_only_det_frames", 0)) for r in clean)
    clean_flt_det = sum(int(r.get("ir_filter_det_frames", 0)) for r in clean)

    conf_frames = sum(r["frames"] for r in confusers)
    conf_raw_det = sum(int(r.get("ir_only_det_frames", 0)) for r in confusers)
    conf_flt_det = sum(int(r.get("ir_filter_det_frames", 0)) for r in confusers)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ir_only confusion
    ax = axes[0]
    m = np.array([[clean_raw_det, clean_frames - clean_raw_det],
                  [conf_raw_det, conf_frames - conf_raw_det]], dtype=float)
    ax.imshow(m, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Detected", "Not Detected"], fontsize=9)
    ax.set_yticklabels(["DRONE", "CONFUSER"], fontsize=9)
    for ii in range(2):
        for jj in range(2):
            val = int(m[ii, jj])
            ax.text(jj, ii, f"{val}", ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if m[ii, jj] > m.max() * 0.4 else "black")
    ax.set_title("ir_only (no filter)", fontsize=11)

    # ir_filter confusion
    ax = axes[1]
    m = np.array([[clean_flt_det, clean_frames - clean_flt_det],
                  [conf_flt_det, conf_frames - conf_flt_det]], dtype=float)
    ax.imshow(m, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Detected", "Not Detected"], fontsize=9)
    ax.set_yticklabels(["DRONE", "CONFUSER"], fontsize=9)
    for ii in range(2):
        for jj in range(2):
            val = int(m[ii, jj])
            ax.text(jj, ii, f"{val}", ha="center", va="center", fontsize=11,
                    fontweight="bold",
                    color="white" if m[ii, jj] > m.max() * 0.4 else "black")
    ax.set_title("ir_filter (with confuser filter)", fontsize=11)

    plt.suptitle("YouTube OOD — Frame-Level Detection Confusion (DRONE_CLEAN only)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "confusion_filter.png", dpi=140)
    plt.close(fig)
    print("  confusion_filter.png saved")

    print(f"\nAll YouTube plots saved to {OUT}")


if __name__ == "__main__":
    main()
