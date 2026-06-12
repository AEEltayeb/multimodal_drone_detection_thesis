"""Generate thesis-ready plots from existing eval CSV/JSONL files.
No YOLO/torch dependency — uses only matplotlib + numpy + csv + json.

Usage:
    python classifier/generate_plots.py
"""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "runs" / "eval_six_configs"
PATCH_THR = 0.70


def plot_metrics_bars(rows, out_dir, title, suffix=""):
    names = [r["config"] for r in rows]
    prec  = [r["precision"] for r in rows]
    rec   = [r["recall"]    for r in rows]
    f1s   = [r["f1"]        for r in rows]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#3498db")
    ax.bar(x,     rec,  w, label="Recall",    color="#e74c3c")
    ax.bar(x + w, f1s,  w, label="F1",        color="#2ecc71")
    for i, vs in enumerate([prec, rec, f1s]):
        for j, v in enumerate(vs):
            ax.text(x[j] + (i - 1) * w, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"{title} — Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"metrics_bars{suffix}.png", dpi=140)
    plt.close(fig)


def plot_confusion(rows, out_dir, title, suffix=""):
    n = len(rows); cols = 3; r_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(r_, cols, figsize=(4 * cols, 3.5 * r_))
    axes = axes.flatten()
    for i, row in enumerate(rows):
        ax = axes[i]
        tp, fp, fn = row["TP"], row["FP"], row["FN"]
        m = np.array([[tp, fn], [fp, 0]], dtype=float)
        ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
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
    for j in range(len(rows), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{title} — Per-Detection Confusion (no TN)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_matrices{suffix}.png", dpi=140)
    plt.close(fig)


def plot_pr_curves(perdet_path, out_dir, ds_name):
    if not perdet_path.exists():
        print(f"  [{ds_name}] no per_det.jsonl, skipping PR curves")
        return
    rgb_records = []; ir_records = []
    n_rgb_gt = 0; n_ir_gt = 0
    for ln in perdet_path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        n_rgb_gt += r["rgb_n_gt"]; n_ir_gt += r["ir_n_gt"]
        rgb_records.extend(r["rgb"]); ir_records.extend(r["ir"])

    def pr_sweep(records, total_gt, match_idx, filter_mask=None):
        recs = [t for t in records if (filter_mask is None or filter_mask(t[1]))]
        recs.sort(key=lambda t: -t[0])
        tp = fp = 0
        precs = []; recs_ = []; threshs = []
        for t in recs:
            if t[match_idx]:
                tp += 1
            else:
                fp += 1
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / total_gt if total_gt > 0 else 0.0
            precs.append(pr); recs_.append(rc); threshs.append(t[0])
        return np.array(precs), np.array(recs_), np.array(threshs)

    for rule, m_idx in (("iou", 2), ("iop", 3)):
        fig, ax = plt.subplots(figsize=(8, 6))
        specs = [
            ("rgb_only",   rgb_records, n_rgb_gt, None,                      "#3498db"),
            ("ir_only",    ir_records,  n_ir_gt,  None,                      "#e67e22"),
            ("rgb_filter", rgb_records, n_rgb_gt, (lambda p: p < PATCH_THR), "#2980b9"),
            ("ir_filter",  ir_records,  n_ir_gt,  (lambda p: p < PATCH_THR), "#d35400"),
        ]
        for name, recs, gt, mask, colour in specs:
            if gt == 0 or not recs:
                continue
            pr, rc, th = pr_sweep(recs, gt, m_idx, mask)
            ax.plot(rc, pr, label=name, color=colour, linewidth=1.8)
            # operating-point dot
            op = 0.25 if "rgb" in name else 0.40
            if len(th):
                idx = int(np.argmin(np.abs(th - op)))
                ax.scatter([rc[idx]], [pr[idx]], color=colour, s=40, zorder=5,
                           edgecolor="black", linewidth=0.6)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="lower left")
        ax.set_title(f"{ds_name} — PR curves ({rule.upper()} match, "
                     f"dots = operating threshold)")
        plt.tight_layout()
        fig.savefig(out_dir / f"pr_curves_{rule}.png", dpi=140)
        plt.close(fig)
    print(f"  [{ds_name}] PR curves saved")


def main():
    for ds in ["antiuav", "svanstrom"]:
        d = OUT_ROOT / ds
        if not d.exists():
            print(f"  SKIP {ds} (dir not found)")
            continue

        for rule in ("iou", "iop"):
            csv_p = d / f"metrics_{rule}.csv"
            if not csv_p.exists():
                print(f"  SKIP {csv_p}")
                continue
            rows = list(csv.DictReader(open(csv_p)))
            for r in rows:
                for k in ("TP", "FP", "FN"):
                    r[k] = int(r[k])
                for k in ("precision", "recall", "f1"):
                    r[k] = float(r[k])

            title = f"{ds} [{rule.upper()}]"
            plot_metrics_bars(rows, d, title, suffix=f"_{rule}")
            plot_confusion(rows, d, title, suffix=f"_{rule}")
            print(f"  [{ds}] {rule}: metrics_bars + confusion_matrices saved")

        plot_pr_curves(d / "per_det.jsonl", d, ds)

    print("\nAll plots generated!")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
