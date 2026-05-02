"""
generate_all_plots.py — Unified plot generator for all eval results.

Computes ALL metrics from per_det.jsonl with correct GT scoping:
  - Single-modality configs: scored against their own GT only
  - Classifier configs: scored against the GT of the trusted modality (per-frame)

Config naming:
  - filter_then_classifier:  filter runs first, classifier routes on filtered features
  - classifier_then_filter:  classifier routes first (raw features), filter vetoes after

Consistent color scheme across all plots.

Usage:
    python classifier/generate_all_plots.py
"""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_ROOT = SCRIPT_DIR / "runs" / "eval_six_configs"
YT_ROOT  = SCRIPT_DIR / "runs" / "eval_youtube_ir"
PATCH_THR = 0.70
RGB_CONF = 0.25
IR_CONF  = 0.40

# ── Config definitions ───────────────────────────────────────────
CONFIG_NAMES = [
    "ir_only", "rgb_only", "classifier",
    "ir_filter", "rgb_filter",
    "filter_then_classifier", "classifier_then_filter",
]
CONFIG_COLORS = {
    "ir_only":                "#e67e22",  # orange
    "rgb_only":               "#3498db",  # blue
    "classifier":             "#9b59b6",  # purple
    "ir_filter":              "#d35400",  # dark orange
    "rgb_filter":             "#2980b9",  # dark blue
    "filter_then_classifier": "#8e44ad",  # dark purple
    "classifier_then_filter": "#2c3e50",  # navy
}
CONFIG_DISPLAY = {
    "filter_then_classifier": "filter→classifier",
    "classifier_then_filter": "classifier→filter",
}

def cfg_label(name):
    return CONFIG_DISPLAY.get(name, name)

def cfg_color(name):
    return CONFIG_COLORS.get(name, "#333333")


# ── Compute ALL metrics from per_det.jsonl (scoped GT) ───────────
SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
def svan_category(key):
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


def compute_all_metrics(perdet_path):
    """Return dict: config -> rule -> {tp, fp, fn, tn, precision, recall, f1}"""
    if not perdet_path.exists():
        return None

    counters = {c: {r: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    for r in ("iou", "iop")}
                for c in CONFIG_NAMES}
    fp_cats = {c: {r: defaultdict(int) for r in ("iou", "iop")} for c in CONFIG_NAMES}

    for ln in perdet_path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        clf_raw = rec["clf_raw"]
        clf_flt = rec["clf_flt"]
        rgb_n_gt = rec["rgb_n_gt"]
        ir_n_gt  = rec["ir_n_gt"]
        cat = svan_category(rec["key"])

        rgb_all = rec["rgb"]  # [conf, fprob, m_iou, m_iop]
        ir_all  = rec["ir"]

        rgb_raw = [d for d in rgb_all if d[0] >= RGB_CONF]
        ir_raw  = [d for d in ir_all  if d[0] >= IR_CONF]
        rgb_flt = [d for d in rgb_raw if d[1] < PATCH_THR]
        ir_flt  = [d for d in ir_raw  if d[1] < PATCH_THR]

        # Config definitions: (rgb_dets, ir_dets, use_rgb_gt, use_ir_gt)
        specs = {
            "ir_only":   ([], ir_raw,  False, True),
            "rgb_only":  (rgb_raw, [], True,  False),
            "ir_filter": ([], ir_flt,  False, True),
            "rgb_filter":(rgb_flt, [], True,  False),
        }

        # classifier: routes on raw features, scoped GT
        use_rgb_clf = clf_raw in (1, 3)
        use_ir_clf  = clf_raw in (2, 3)
        specs["classifier"] = (
            rgb_raw if use_rgb_clf else [],
            ir_raw  if use_ir_clf  else [],
            use_rgb_clf, use_ir_clf,
        )

        # filter_then_classifier: filter first, classifier on filtered features
        use_rgb_ftc = clf_flt in (1, 3)
        use_ir_ftc  = clf_flt in (2, 3)
        specs["filter_then_classifier"] = (
            rgb_flt if use_rgb_ftc else [],
            ir_flt  if use_ir_ftc  else [],
            use_rgb_ftc, use_ir_ftc,
        )

        # classifier_then_filter: classifier routes on raw, then filter
        use_rgb_ctf = clf_raw in (1, 3)
        use_ir_ctf  = clf_raw in (2, 3)
        specs["classifier_then_filter"] = (
            rgb_flt if use_rgb_ctf else [],
            ir_flt  if use_ir_ctf  else [],
            use_rgb_ctf, use_ir_ctf,
        )

        for c_name, (kr, ki, use_rgt, use_igt) in specs.items():
            for rule, m_idx in (("iou", 2), ("iop", 3)):
                tp = fp = fn = 0

                # Score RGB dets
                if use_rgt:
                    for det in kr:
                        if det[m_idx]: tp += 1
                        else: fp += 1
                    rgb_tp = sum(1 for d in kr if d[m_idx])
                    fn += max(0, rgb_n_gt - rgb_tp)
                else:
                    fp += len(kr)

                # Score IR dets
                if use_igt:
                    for det in ki:
                        if det[m_idx]: tp += 1
                        else: fp += 1
                    ir_tp = sum(1 for d in ki if d[m_idx])
                    fn += max(0, ir_n_gt - ir_tp)
                else:
                    fp += len(ki)

                # TN
                has_gt = (use_rgt and rgb_n_gt > 0) or (use_igt and ir_n_gt > 0)
                has_det = len(kr) > 0 or len(ki) > 0
                tn_inc = 1 if (not has_gt and not has_det) else 0

                counters[c_name][rule]["tp"] += tp
                counters[c_name][rule]["fp"] += fp
                counters[c_name][rule]["fn"] += fn
                counters[c_name][rule]["tn"] += tn_inc
                fp_cats[c_name][rule][cat] += fp

    # Compute P/R/F1
    results = {}
    for c in CONFIG_NAMES:
        results[c] = {}
        for rule in ("iou", "iop"):
            d = counters[c][rule]
            tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
            results[c][rule] = {
                "config": c, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": p, "recall": r, "f1": f1,
                "fp_cats": dict(fp_cats[c][rule]),
            }
    return results


# ── Plot: metrics bars ───────────────────────────────────────────
def plot_metrics_bars(results, out_dir, title, rule):
    rows = [results[c][rule] for c in CONFIG_NAMES]
    names = [cfg_label(c) for c in CONFIG_NAMES]
    prec  = [r["precision"] for r in rows]
    rec   = [r["recall"]    for r in rows]
    f1s   = [r["f1"]        for r in rows]
    colors = [cfg_color(c) for c in CONFIG_NAMES]
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - w, prec, w, label="Precision", color=colors, alpha=0.6,
           edgecolor="black", linewidth=0.3)
    ax.bar(x,     rec,  w, label="Recall",    color=colors, alpha=0.8,
           edgecolor="black", linewidth=0.3)
    ax.bar(x + w, f1s,  w, label="F1",        color=colors, alpha=1.0,
           edgecolor="black", linewidth=0.3)
    for i, (vs, offset) in enumerate([(prec, -w), (rec, 0), (f1s, w)]):
        for j, v in enumerate(vs):
            ax.text(x[j] + offset, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score")
    ax.set_title(f"{title} — Precision / Recall / F1", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend([Patch(facecolor="gray", alpha=a) for a in [0.6, 0.8, 1.0]],
              ["Precision", "Recall", "F1"], loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / f"metrics_bars_{rule}.png", dpi=160)
    plt.close(fig)


# ── Plot: confusion matrices ─────────────────────────────────────
def plot_confusion(results, out_dir, title, rule):
    rows = [results[c][rule] for c in CONFIG_NAMES]
    n = len(rows); cols = 4; r_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(r_, cols, figsize=(4.2 * cols, 3.8 * r_))
    axes = axes.flatten()
    for i, row in enumerate(rows):
        ax = axes[i]
        tp, fp, fn, tn = row["TP"], row["FP"], row["FN"], row["TN"]
        m = np.array([[tp, fn], [fp, tn]], dtype=float)
        ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred +", "Pred −"], fontsize=8)
        ax.set_yticklabels(["GT +", "GT −"], fontsize=8)
        labels = [[f"TP\n{tp:,}", f"FN\n{fn:,}"],
                  [f"FP\n{fp:,}", f"TN\n{tn:,}"]]
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, labels[ii][jj], ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if m[ii, jj] > m.max() * 0.5 else "black")
        color = cfg_color(row["config"])
        ax.set_title(f"{cfg_label(row['config'])}\n"
                     f"P={row['precision']:.3f} R={row['recall']:.3f} "
                     f"F1={row['f1']:.3f}", fontsize=9, color=color,
                     fontweight="bold")
    for j in range(len(rows), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f"{title} — Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / f"confusion_matrices_{rule}.png", dpi=160)
    plt.close(fig)


# ── Plot: PR curves ──────────────────────────────────────────────
def plot_pr_curves(perdet_path, out_dir, ds_name):
    if not perdet_path.exists():
        return

    # Collect per-detection records and scoped GT for all 7 configs
    # Each config accumulates: list of (conf, match_iou, match_iop) + total_gt per rule
    cfg_data = {c: {"dets": [], "gt_iou": 0, "gt_iop": 0}
                for c in CONFIG_NAMES}

    for ln in perdet_path.read_text().splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        clf_raw = rec["clf_raw"]
        clf_flt = rec["clf_flt"]
        rgb_n_gt = rec["rgb_n_gt"]
        ir_n_gt  = rec["ir_n_gt"]

        rgb_all = rec["rgb"]   # [conf, fprob, m_iou, m_iop]
        ir_all  = rec["ir"]

        # For PR sweep: use ALL detections (no conf filter) so curves
        # extend fully; the operating point is marked with a dot.
        rgb_flt_all = [d for d in rgb_all if d[1] < PATCH_THR]
        ir_flt_all  = [d for d in ir_all  if d[1] < PATCH_THR]

        use_rgb_clf = clf_raw in (1, 3)
        use_ir_clf  = clf_raw in (2, 3)
        use_rgb_ftc = clf_flt in (1, 3)
        use_ir_ftc  = clf_flt in (2, 3)
        use_rgb_ctf = clf_raw in (1, 3)
        use_ir_ctf  = clf_raw in (2, 3)

        # (config_name, rgb_dets, ir_dets, use_rgb_gt, use_ir_gt)
        frame_specs = {
            "ir_only":                ([], ir_all,      False, True),
            "rgb_only":               (rgb_all, [],     True,  False),
            "ir_filter":              ([], ir_flt_all,   False, True),
            "rgb_filter":             (rgb_flt_all, [],  True,  False),
            "classifier":             (rgb_all if use_rgb_clf else [],
                                       ir_all  if use_ir_clf  else [],
                                       use_rgb_clf, use_ir_clf),
            "filter_then_classifier": (rgb_flt_all if use_rgb_ftc else [],
                                       ir_flt_all  if use_ir_ftc  else [],
                                       use_rgb_ftc, use_ir_ftc),
            "classifier_then_filter": (rgb_flt_all if use_rgb_ctf else [],
                                       ir_flt_all  if use_ir_ctf  else [],
                                       use_rgb_ctf, use_ir_ctf),
        }

        for c_name, (kr, ki, use_rgt, use_igt) in frame_specs.items():
            cd = cfg_data[c_name]
            scoped_gt = 0
            if use_rgt:
                scoped_gt += rgb_n_gt
            if use_igt:
                scoped_gt += ir_n_gt
            cd["gt_iou"] += scoped_gt
            cd["gt_iop"] += scoped_gt
            # Collect all kept detections with (conf, match_iou, match_iop)
            for d in kr:
                cd["dets"].append((d[0], d[2], d[3]))
            for d in ki:
                cd["dets"].append((d[0], d[2], d[3]))

    def pr_sweep(dets, total_gt, match_idx):
        """match_idx: 1=iou, 2=iop in the (conf, m_iou, m_iop) tuple"""
        recs = sorted(dets, key=lambda t: -t[0])
        tp = fp = 0; precs = []; recalls = []; threshs = []
        for t in recs:
            if t[match_idx]:
                tp += 1
            else:
                fp += 1
            pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rc = tp / total_gt if total_gt > 0 else 0.0
            precs.append(pr); recalls.append(rc); threshs.append(t[0])
        return np.array(precs), np.array(recalls), np.array(threshs)

    for rule, m_idx, gt_key in (("iou", 1, "gt_iou"), ("iop", 2, "gt_iop")):
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_order = [
            ("rgb_only",               "-",  2.0),
            ("ir_only",                "-",  2.0),
            ("rgb_filter",             "--", 2.0),
            ("ir_filter",              "--", 2.0),
            ("classifier",             "-",  2.5),
            ("filter_then_classifier", "--", 2.0),
            ("classifier_then_filter", ":",  2.0),
        ]
        for name, ls, lw in plot_order:
            cd = cfg_data[name]
            gt = cd[gt_key]
            if gt == 0 or not cd["dets"]:
                continue
            pr, rc, th = pr_sweep(cd["dets"], gt, m_idx)
            colour = cfg_color(name)
            ax.plot(rc, pr, label=cfg_label(name), color=colour,
                    linewidth=lw, linestyle=ls)
            # Operating threshold dot — use 0.25 for rgb, 0.40 for ir,
            # and the lower of the two (0.25) for mixed configs
            if "rgb" in name and "ir" not in name:
                op = 0.25
            elif "ir" in name and "rgb" not in name:
                op = 0.40
            else:
                op = 0.25  # mixed classifier configs
            if len(th):
                idx = int(np.argmin(np.abs(th - op)))
                ax.scatter([rc[idx]], [pr[idx]], color=colour, s=50, zorder=5,
                           edgecolor="black", linewidth=0.8)
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
        ax.set_title(f"{ds_name} - PR Curves ({rule.upper()} match, "
                     f"dots = operating threshold)", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_dir / f"pr_curves_{rule}.png", dpi=160)
        plt.close(fig)
    print(f"    PR curves saved")


# ── Plot: YouTube OOD ────────────────────────────────────────────
def plot_youtube(yt_dir):
    pv_path = yt_dir / "youtube_per_video.csv"
    if not pv_path.exists():
        print("  [youtube] no youtube_per_video.csv - skip")
        return
    rows = list(csv.DictReader(open(pv_path, encoding="utf-8-sig")))
    for r in rows:
        r["frames"] = int(r["frames"])
        r["ir_only_det_rate"] = float(r["ir_only_det_rate"])
        r["ir_filter_det_rate"] = float(r["ir_filter_det_rate"])

    # Per-video bars
    fig, ax = plt.subplots(figsize=(14, 6))
    names = []
    for r in rows:
        q = f" [{r.get('quality','')}]" if r.get("quality") else ""
        names.append(f"{r['video'].replace('yt_','').replace('.mp4','')}\n"
                     f"({r['category']}{q})")
    x = np.arange(len(rows)); w = 0.35
    raw = [r["ir_only_det_rate"] * 100 for r in rows]
    flt = [r["ir_filter_det_rate"] * 100 for r in rows]

    drone_mask = [r["category"] == "DRONE" for r in rows]
    c_raw = [cfg_color("ir_only") if not d else "#2ecc71" for d in drone_mask]
    c_flt = [cfg_color("ir_filter") if not d else "#27ae60" for d in drone_mask]

    ax.bar(x - w/2, raw, w, label="ir_only", color=c_raw, alpha=0.7,
           edgecolor="black", linewidth=0.3)
    ax.bar(x + w/2, flt, w, label="ir_filter", color=c_flt, alpha=0.9,
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
                 "(Green = DRONE, Orange/Brown = CONFUSER)", fontsize=12)
    ax.legend(loc="upper right"); ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    fig.savefig(yt_dir / "per_video_bars.png", dpi=160); plt.close(fig)

    # Category summary
    cat_path = yt_dir / "category_summary.csv"
    cats = list(csv.DictReader(open(cat_path, encoding="utf-8-sig")))
    for c in cats:
        c["ir_only_det_rate"] = float(c["ir_only_det_rate"]) * 100
        c["ir_filter_det_rate"] = float(c["ir_filter_det_rate"]) * 100
        c["suppression"] = float(c["suppression"]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cat_names = [c["category"] for c in cats]
    x = np.arange(len(cats)); w = 0.35
    raw_cat = [c["ir_only_det_rate"] for c in cats]
    flt_cat = [c["ir_filter_det_rate"] for c in cats]
    cat_colors = ["#2ecc71" if c["category"] == "DRONE" else cfg_color("ir_only")
                  for c in cats]

    ax1.bar(x - w/2, raw_cat, w, label="ir_only",
            color=[c + "99" for c in cat_colors], edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, flt_cat, w, label="ir_filter",
            color=cat_colors, edgecolor="black", linewidth=0.5)
    for i, (v1, v2) in enumerate(zip(raw_cat, flt_cat)):
        ax1.text(x[i] - w/2, v1 + 1, f"{v1:.1f}%", ha="center", fontsize=8)
        ax1.text(x[i] + w/2, v2 + 1, f"{v2:.1f}%", ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(cat_names, fontsize=10)
    ax1.set_ylabel("Detection Rate (%)"); ax1.set_title("Detection Rate by Category")
    ax1.legend(); ax1.set_ylim(0, 55); ax1.grid(axis="y", alpha=0.3)

    supp = [c["suppression"] for c in cats]
    ax2.bar(x, supp, 0.5, color=cat_colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(supp):
        ax2.text(x[i], v + 1, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(cat_names, fontsize=10)
    ax2.set_ylabel("Suppression Rate (%)")
    ax2.set_title("Filter Suppression Rate\n(Confusers: higher=better | Drone: lower=better)")
    ax2.set_ylim(0, 100); ax2.grid(axis="y", alpha=0.3)
    plt.suptitle("YouTube OOD — IR Confuser Filter Evaluation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(yt_dir / "category_summary_bars.png", dpi=160); plt.close(fig)

    # Frame-level confusion matrices (ir_only vs ir_filter)
    # Confuser frames: detection = FP, no detection = TN
    # CLEAN drone frames only: detection = TP, no detection = FN
    # (LABELS drone videos excluded — drone not visible on most frames)
    confuser_cats = {"AIRPLANE", "BIRD", "HELICOPTER"}
    configs_yt = {
        "ir_only": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        "ir_filter": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
    }
    for r in rows:
        frames = r["frames"]
        ir_det = int(r["ir_only_det_frames"])
        flt_det = int(r["ir_filter_det_frames"])
        if r["category"] in confuser_cats:
            configs_yt["ir_only"]["fp"]  += ir_det
            configs_yt["ir_only"]["tn"]  += frames - ir_det
            configs_yt["ir_filter"]["fp"] += flt_det
            configs_yt["ir_filter"]["tn"] += frames - flt_det
        elif r.get("quality", "").strip() == "CLEAN":
            # Only CLEAN drone videos for TP/FN
            configs_yt["ir_only"]["tp"]  += ir_det
            configs_yt["ir_only"]["fn"]  += frames - ir_det
            configs_yt["ir_filter"]["tp"] += flt_det
            configs_yt["ir_filter"]["fn"] += frames - flt_det

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for idx, (cname, cnts) in enumerate(configs_yt.items()):
        ax = axes[idx]
        tp, fp, fn, tn = cnts["tp"], cnts["fp"], cnts["fn"], cnts["tn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r_ / (p+r_) if (p+r_) > 0 else 0.0
        m = np.array([[tp, fn], [fp, tn]], dtype=float)
        ax.imshow(m, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred +", "Pred −"], fontsize=9)
        ax.set_yticklabels(["Drone", "Confuser"], fontsize=9)
        labels = [[f"TP\n{tp:,}", f"FN\n{fn:,}"],
                  [f"FP\n{fp:,}", f"TN\n{tn:,}"]]
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, labels[ii][jj], ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if m[ii, jj] > m.max() * 0.5 else "black")
        color = cfg_color(cname)
        ax.set_title(f"{cname}\n"
                     f"P={p:.3f} R={r_:.3f} F1={f1:.3f}",
                     fontsize=10, color=color, fontweight="bold")
    plt.suptitle("YouTube OOD — Frame-Level Confusion (Drone vs Confuser)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(yt_dir / "confusion_matrices.png", dpi=160); plt.close(fig)

    print("    YouTube plots saved")


# ── Main ─────────────────────────────────────────────────────────
def main():
    # Delete old plots
    for d in [OUT_ROOT / "antiuav", OUT_ROOT / "svanstrom", YT_ROOT]:
        if not d.exists():
            continue
        for f in d.glob("*.png"):
            f.unlink()
            print(f"  deleted {f.name}")

    for ds in ["antiuav", "svanstrom"]:
        d = OUT_ROOT / ds
        if not d.exists():
            print(f"  SKIP {ds}")
            continue
        print(f"\n[{ds}]")

        # Compute ALL metrics from per_det.jsonl with scoped GT
        results = compute_all_metrics(d / "per_det.jsonl")
        if not results:
            print(f"  No per_det.jsonl — skip")
            continue

        # Print summary (ASCII names for console compatibility)
        console_names = {
            "filter_then_classifier": "filter->classifier",
            "classifier_then_filter": "classifier->filter",
        }
        for rule in ("iou", "iop"):
            print(f"  {rule.upper()}:")
            for c in CONFIG_NAMES:
                r = results[c][rule]
                cname = console_names.get(c, c)
                print(f"    {cname:<25s}  TP={r['TP']:>8,}  FP={r['FP']:>8,}  "
                      f"FN={r['FN']:>8,}  TN={r['TN']:>8,}  "
                      f"P={r['precision']:.4f}  R={r['recall']:.4f}  F1={r['f1']:.4f}")

        for rule in ("iou", "iop"):
            title = f"{ds} [{rule.upper()}]"
            plot_metrics_bars(results, d, title, rule)
            plot_confusion(results, d, title, rule)

            # Write corrected CSV (scoped GT, with TN)
            csv_path = d / f"metrics_scoped_{rule}.csv"
            with open(csv_path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=[
                    "config", "TP", "FP", "FN", "TN",
                    "Precision", "Recall", "F1"])
                w.writeheader()
                for c in CONFIG_NAMES:
                    r = results[c][rule]
                    w.writerow({
                        "config": c,
                        "TP": r["TP"], "FP": r["FP"],
                        "FN": r["FN"], "TN": r["TN"],
                        "Precision": round(r["precision"], 4),
                        "Recall": round(r["recall"], 4),
                        "F1": round(r["f1"], 4),
                    })
            print(f"    {rule}: metrics_bars + confusion_matrices + CSV saved")

        plot_pr_curves(d / "per_det.jsonl", d, ds)

    # YouTube
    if YT_ROOT.exists():
        print(f"\n[youtube]")
        plot_youtube(YT_ROOT)

    print(f"\nAll plots generated!")


if __name__ == "__main__":
    main()
