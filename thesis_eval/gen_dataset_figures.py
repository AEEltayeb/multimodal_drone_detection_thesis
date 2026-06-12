"""gen_dataset_figures.py — three dataset figures for the thesis (notes round 1).

(1) fig_datasets_pie     two pies: training corpora vs canonical evaluation surfaces (source frames).
                         Counts are the documented totals from the methodology/appendix dataset tables
                         (tab:ds_rgb_components 172,022; tab:ds_ir_components 129,130; tab:ds_confusers
                         21,784+2,607 train+val / 2,633 test; appendix: Anti-UAV 85,374, Svanström
                         28,710, SelCom 2,076/311 val, IR_confusers evaluated split 5,237, YouTube
                         evaluated 1,359 drone + 1,250 confuser; rgb_dataset test 17,209 and
                         ir_dset_final test 9,612 from the Tier-1 cache n_source).
(2) fig_confuser_examples  image grid: what the confuser corpora look like (RGB + thermal rows).
(3) fig_confuser_fp_examples  real ft4 hallucinations on rgb_confusers_merged/test, drawn from the
                         Tier-1 cache (no detector run): context crop around each FP box with the
                         detector confidence and the confuser filter's P(drone) — every example shown
                         is suppressed by the production filter (P(drone) < 0.25).

Writes figures/*.{pdf,png} into the Overleaf folder.
  py -u thesis_eval/gen_dataset_figures.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
for _sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))
from pipeline_eval_unified import load_verifiers, batch_probs, RGB_THR_MLP   # noqa: E402
from notes_round1_replays import cat_of                                       # noqa: E402

FIGDIR = REPO / "docs/thesis_working_distilling_overleaf/figures"
CACHE = REPO / "thesis_eval/cache"
RGB_CONF_DIR = Path("G:/drone/rgb_confusers_merged/images/test")
IR_CONF_DIR = Path("G:/drone/IR_confusers/images/train")

TRAIN = [("Composite RGB corpus", 172_022), ("ir\\_dset\\_final (IR corpus)", 129_130),
         ("RGB confuser corpus\n(train+val)", 24_391), ("SelCom CCTV", 2_076)]
EVAL = [("Anti-UAV RGBT (paired)", 85_374), ("Svanström (paired)", 28_710),
        ("rgb\\_dataset test", 17_209), ("ir\\_dset\\_final test", 9_612),
        ("IR\\_confusers", 5_237), ("rgb\\_confusers test", 2_633),
        ("YouTube video clips", 2_609), ("SelCom val", 311)]


def pie(ax, data, title, cmap):
    names = [n.replace("\\_", "_") for n, _ in data]
    vals = [v for _, v in data]
    colors = plt.get_cmap(cmap)(np.linspace(0.25, 0.85, len(vals)))
    wedges, _ = ax.pie(vals, startangle=120, colors=colors,
                       wedgeprops={"edgecolor": "white", "linewidth": 1})
    total = sum(vals)
    ax.legend(wedges, [f"{n} — {v:,} ({v/total*100:.1f}%)" for n, v in zip(names, vals)],
              loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=7.6, frameon=False)
    ax.set_title(title, fontsize=10)


def fig_pies():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    pie(a1, TRAIN, f"(a) training corpora — {sum(v for _, v in TRAIN):,} frames", "Blues")
    pie(a2, EVAL, f"(b) evaluation surfaces — {sum(v for _, v in EVAL):,} source frames", "Greens")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_datasets_pie.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_datasets_pie")


def find_img(d, stem):
    for ext in (".jpg", ".png", ".jpeg"):
        p = d / (stem + ext)
        if p.exists():
            return p
    hits = list(d.glob(stem + ".*"))
    return hits[0] if hits else None


def fig_examples():
    """One example per category per modality. Picks the highest-confidence FIRED frame per category
    from the Tier-1 cache (representative hard negatives, deterministic); falls back to the first
    stem on disk for categories the detector never fires on."""
    rows = [("RGB confusers (rgb_confusers_merged)", RGB_CONF_DIR, "rgb_confuser", "rgb",
             ["airplane", "bird", "helicopter", "other"]),
            ("Thermal confusers (IR_confusers)", IR_CONF_DIR, "ir_confusers", "ir",
             ["airplane", "bird", "helicopter"])]
    picks = []
    for title, d, cache_name, slot, cats in rows:
        bycat = {}
        frames = pickle.load(open(CACHE / f"{cache_name}.pkl", "rb"))["frames"]
        for fr in frames:
            confs = fr[slot]["confs"]
            if not len(confs):
                continue
            c = cat_of(fr["key"])
            if c in cats and (c not in bycat or float(confs.max()) > bycat[c][1]):
                bycat[c] = (fr["key"], float(confs.max()))
        stems = sorted(p.stem for p in d.glob("*.*"))
        for s in stems:
            c = cat_of(s)
            if c in cats and c not in bycat:
                bycat[c] = (s, 0.0)
        picks.append((title, d, [(c, bycat.get(c, (None, 0))[0]) for c in cats]))
    ncol = max(len(p[2]) for p in picks)
    fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol, 5.6))
    for r, (title, d, items) in enumerate(picks):
        for c in range(ncol):
            ax = axes[r][c]; ax.axis("off")
            if c >= len(items) or items[c][1] is None:
                continue
            cat, stem = items[c]
            p = find_img(d, stem)
            if p is None:
                continue
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(cat, fontsize=9)
        axes[r][0].text(-0.08, 0.5, title, transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_confuser_examples.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_confuser_examples")


def fig_fp_examples():
    d = pickle.load(open(CACHE / "rgb_confuser.pkl", "rb"))
    frames = d["frames"]
    verifs = load_verifiers()
    probs = batch_probs(frames, "rgb", verifs["mlp_v5"])
    # per category: highest-confidence detection whose whole frame is suppressed by the filter
    best = {}
    for i, fr in enumerate(frames):
        confs = fr["rgb"]["confs"]
        if not len(confs) or (probs[i] >= RGB_THR_MLP).any():
            continue
        c = cat_of(fr["key"])
        j = int(np.argmax(confs))
        if c not in best or confs[j] > best[c][2]:
            best[c] = (i, j, float(confs[j]))
        # keep a second, lower-conf example per category for variety
        key2 = c + "#2"
        if c in best and i != best[c][0] and (key2 not in best or confs[j] > best[key2][2]):
            best[key2] = (i, j, float(confs[j]))
    order = ["airplane", "bird", "helicopter", "airplane#2", "bird#2", "helicopter#2"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.2))
    for k, name in enumerate(order):
        ax = axes[k // 3][k % 3]; ax.axis("off")
        if name not in best:
            continue
        i, j, conf = best[name]
        fr = frames[i]
        p = find_img(RGB_CONF_DIR, fr["key"])
        if p is None:
            continue
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = fr["rgb"]["boxes"][j]
        # context crop: 6x the box size, clamped to the frame
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 24) * 3
        H, W = img.shape[:2]
        a, b = int(max(cx - s, 0)), int(max(cy - s, 0))
        c2, d2 = int(min(cx + s, W)), int(min(cy + s, H))
        crop = img[b:d2, a:c2]
        ax.imshow(crop)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x1 - a, y1 - b), x2 - x1, y2 - y1,
                               fill=False, edgecolor="red", linewidth=1.6))
        ax.set_title(f"{name.split('#')[0]} — det conf {conf:.2f}, "
                     f"filter P(drone) {float(probs[i][j]):.3f}", fontsize=8.5)
    fig.suptitle("ft4 hallucinations on the OOD confuser corpus — all suppressed by the MLP "
                 "confuser filter (P(drone) < 0.25)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_confuser_fp_examples.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_confuser_fp_examples")


if __name__ == "__main__":
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig_pies()
    fig_examples()
    fig_fp_examples()
