"""
reporting.py — Output formatting, CSV/JSON writing, console tables, and plotting.

Consolidates all reporting logic used by eval_pipeline.py and eval_model.py.
"""

from __future__ import annotations
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

# Try matplotlib — optional for non-plot runs
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Console output ───────────────────────────────────────────────

def print_metrics_table(rows: list[dict], title: str = ""):
    """Print a formatted metrics table to console."""
    if title:
        print(f"\n{title}")
    print(f"  {'config':<22s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s} "
          f"{'prec':>7s} {'rec':>7s} {'f1':>7s}")
    print("  " + "-" * 80)
    for row in rows:
        tn_str = f"{row.get('TN', 0):>8d}"
        print(f"  {row['config']:<22s} {row['TP']:>8d} {row['FP']:>8d} "
              f"{row['FN']:>8d} {tn_str} {row['precision']:>7.4f} "
              f"{row['recall']:>7.4f} {row['f1']:>7.4f}")


def print_fp_by_category(fp_cats: dict, configs: list[str], categories: list[str],
                          title: str = ""):
    """Print FP-by-category breakdown."""
    if title:
        print(f"\n{title}")
    header = f"  {'config':<22s} " + " ".join(f"{c:>10s}" for c in categories)
    print(header)
    print("  " + "-" * (22 + 11 * len(categories)))
    for cfg in configs:
        vals = " ".join(f"{fp_cats.get(cfg, {}).get(c, 0):>10d}" for c in categories)
        print(f"  {cfg:<22s} {vals}")


def print_size_distribution(dist: dict[str, dict[str, int]], title: str = ""):
    """Print detection size distribution."""
    if title:
        print(f"\n{title}")
    print(f"  {'config':<22s} {'small':>8s} {'medium':>8s} {'large':>8s} {'total':>8s}")
    print("  " + "-" * 55)
    for cfg, sizes in dist.items():
        total = sum(sizes.values())
        print(f"  {cfg:<22s} {sizes.get('small', 0):>8d} "
              f"{sizes.get('medium', 0):>8d} {sizes.get('large', 0):>8d} {total:>8d}")


# ── CSV output ───────────────────────────────────────────────────

def save_metrics_csv(rows: list[dict], path: Path):
    """Save metrics rows to CSV."""
    if not rows:
        return
    fields = ["config", "TP", "FP", "FN", "TN", "precision", "recall", "f1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def save_fp_category_csv(fp_cats: dict, configs: list[str],
                          categories: list[str], path: Path):
    """Save FP-by-category to CSV."""
    fields = ["config", *categories, "total"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg in configs:
            row = {"config": cfg}
            total = 0
            for cat in categories:
                v = fp_cats.get(cfg, {}).get(cat, 0)
                row[cat] = v
                total += v
            row["total"] = total
            w.writerow(row)
    print(f"  Saved: {path}")


def save_jsonl(records: list[str], path: Path, append: bool = False):
    """Write JSONL records to file."""
    mode = "a" if append else "w"
    with open(path, mode) as f:
        f.write("\n".join(records) + "\n")


# ── JSON output ──────────────────────────────────────────────────

def save_json(data: dict, path: Path):
    """Save dict as formatted JSON."""
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved: {path}")


# ── Plotting ─────────────────────────────────────────────────────

def plot_metrics_bars(rows: list[dict], out_dir: Path, title: str,
                       suffix: str = ""):
    """Bar chart of precision/recall/F1 per config."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available for plotting")
        return
    names = [r["config"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1s = [r["f1"] for r in rows]
    x = np.arange(len(names))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#3498db")
    ax.bar(x, rec, w, label="Recall", color="#e74c3c")
    ax.bar(x + w, f1s, w, label="F1", color="#2ecc71")
    for i, vs in enumerate([prec, rec, f1s]):
        for j, v in enumerate(vs):
            ax.text(x[j] + (i - 1) * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{title} — per-detection metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"metrics_bars{suffix}.png", dpi=140)
    plt.close(fig)


def plot_confusion_matrices(rows: list[dict], out_dir: Path, title: str,
                             suffix: str = ""):
    """2×2 confusion matrices per config."""
    if not HAS_MPL:
        return
    n = len(rows)
    cols = min(3, n)
    r_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(r_, cols, figsize=(4 * cols, 3.5 * r_))
    if r_ * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, row in enumerate(rows):
        ax = axes[i]
        tp, fp, fn = row["TP"], row["FP"], row["FN"]
        m = np.array([[tp, fn], [fp, 0]], dtype=float)
        ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred+", "pred-"], fontsize=8)
        ax.set_yticklabels(["GT+", "GT-"], fontsize=8)
        labels = [[f"TP\n{tp}", f"FN\n{fn}"], [f"FP\n{fp}", "—"]]
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, labels[ii][jj], ha="center", va="center",
                        fontsize=9,
                        color="white" if m[ii, jj] > m.max() * 0.5 else "black")
        ax.set_title(f"{row['config']}\nP={row['precision']:.3f} "
                     f"R={row['recall']:.3f} F1={row['f1']:.3f}", fontsize=9)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{title} — confusion (no TN)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion{suffix}.png", dpi=140)
    plt.close(fig)


def plot_pr_curves(
    rgb_records: list, ir_records: list,
    n_rgb_gt: int, n_ir_gt: int,
    out_dir: Path, title: str, patch_thr: float = 0.70,
):
    """PR curves by sweeping conf threshold on cached per-det records.

    records: list of [conf, filter_prob, match_iou, match_iop]
    """
    if not HAS_MPL:
        return
    from metrics import pr_sweep as _pr_sweep

    def _sweep(records, total_gt, match_idx, filter_fn=None):
        recs = [(r[0], r[match_idx]) for r in records
                if (filter_fn is None or filter_fn(r[1]))]
        if not recs or total_gt == 0:
            return np.array([]), np.array([]), np.array([])
        return _pr_sweep(recs, total_gt)

    for rule, m_idx in (("iou", 2), ("iop", 3)):
        fig, ax = plt.subplots(figsize=(8, 6))
        specs = [
            ("rgb_only", rgb_records, n_rgb_gt, None, "#3498db"),
            ("ir_only", ir_records, n_ir_gt, None, "#e67e22"),
            ("rgb_filter", rgb_records, n_rgb_gt, (lambda p: p < patch_thr), "#2980b9"),
            ("ir_filter", ir_records, n_ir_gt, (lambda p: p < patch_thr), "#d35400"),
        ]
        for name, recs, gt, mask, colour in specs:
            if gt == 0 or not recs:
                continue
            pr, rc, th = _sweep(recs, gt, m_idx, mask)
            if len(pr) == 0:
                continue
            ax.plot(rc, pr, label=name, color=colour, linewidth=1.8)
            op = 0.25 if "rgb" in name else 0.40
            if len(th):
                idx = int(np.argmin(np.abs(th - op)))
                ax.scatter([rc[idx]], [pr[idx]], color=colour, s=40, zorder=5,
                           edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left")
        ax.set_title(f"{title} — PR curves ({rule.upper()} match)")
        plt.tight_layout()
        fig.savefig(out_dir / f"pr_curves_{rule}.png", dpi=140)
        plt.close(fig)


def plot_size_distribution(dist: dict[str, dict[str, int]], out_dir: Path,
                            title: str):
    """Stacked bar chart of detection size distribution."""
    if not HAS_MPL:
        return
    configs = list(dist.keys())
    smalls = [dist[c].get("small", 0) for c in configs]
    mediums = [dist[c].get("medium", 0) for c in configs]
    larges = [dist[c].get("large", 0) for c in configs]
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, smalls, label="Small (<0.1%)", color="#e74c3c")
    ax.bar(x, mediums, bottom=smalls, label="Medium (0.1-1%)", color="#f39c12")
    ax.bar(x, larges, bottom=[s + m for s, m in zip(smalls, mediums)],
           label="Large (>1%)", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("Detection count")
    ax.set_title(f"{title} — Detection size distribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "size_distribution.png", dpi=140)
    plt.close(fig)


def plot_youtube_summary(results: list[dict], out_dir: Path):
    """Bar chart of YouTube OOD filter suppression rates."""
    if not HAS_MPL:
        return
    cats = defaultdict(lambda: {"raw": 0, "filt": 0, "frames": 0})
    for r in results:
        c = r["category"]
        cats[c]["raw"] += r.get("raw_det_frames", 0)
        cats[c]["filt"] += r.get("filter_det_frames", 0)
        cats[c]["frames"] += r.get("frames", 0)

    cat_names = sorted(cats.keys())
    raw_rates = [cats[c]["raw"] / max(cats[c]["frames"], 1) for c in cat_names]
    filt_rates = [cats[c]["filt"] / max(cats[c]["frames"], 1) for c in cat_names]

    x = np.arange(len(cat_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, raw_rates, w, label="Raw YOLO", color="#e74c3c")
    ax.bar(x + w / 2, filt_rates, w, label="+ Filter", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names)
    ax.set_ylabel("Detection rate")
    ax.set_title("YouTube OOD — Filter suppression by category")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    fig.savefig(out_dir / "youtube_summary.png", dpi=140)
    plt.close(fig)
