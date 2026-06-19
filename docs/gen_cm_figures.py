"""docs/gen_cm_figures.py — confusion-matrix figures for the three production learned models,
read from the frozen held-out JSONs (thesis_eval/results/per_model_heldout/). Zero-GPU.
Writes cm_router.png + cm_filters.png straight into the thesis figures dir.
Run AFTER: py thesis_eval/eval_router_heldout.py ; py thesis_eval/eval_filter_heldout_cm.py
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
try:
    from thesis_style import apply_theme, CMAP
    apply_theme()
except Exception:
    CMAP = "BuPu"

RES = REPO / "thesis_eval" / "results" / "per_model_heldout"
FIGS = REPO / "docs" / "thesis_working_distilling_overleaf" / "figures"


def heat(ax, cm, xlabels, ylabels, title):
    cm = np.array(cm, float)
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)        # row-normalised for colour only
    ax.imshow(rn, cmap=CMAP, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10.5)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if rn[i, j] > 0.55 else "#222222", fontsize=9.5, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)


# ── router 3x3 ──
r = json.load(open(RES / "router_heldout.json"))
short = {"trust_rgb": "rgb", "trust_ir": "ir", "both": "both"}
labs = [short[l] for l in r["labels"]]
fig, ax = plt.subplots(figsize=(4.0, 3.7))
heat(ax, r["confusion_matrix"], labs, labs, f"Trust router (held-out, acc {r['accuracy']:.3f})")
fig.tight_layout(); fig.savefig(FIGS / "cm_router.png", dpi=300); plt.close(fig)

# ── filters 2x2 each (rows: drone/confuser ; cols: kept/vetoed) ──
f = json.load(open(RES / "filter_heldout_cm.json"))


def fcm(d, c):
    return [[d["kept_TP"], d["vetoed_FN"]], [c["kept_FP"], c["vetoed_TN"]]]


fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
heat(axes[0], fcm(f["rgb_filter"]["drone_rgb_dataset_test"], f["rgb_filter"]["confuser_rgb_confuser"]),
     ["kept", "vetoed"], ["drone", "confuser"], "RGB filter (v4) @0.25")
heat(axes[1], fcm(f["ir_filter"]["drone_ir_dset_final"], f["ir_filter"]["confuser_ir_confusers"]),
     ["kept", "vetoed"], ["drone", "confuser"], "IR filter (thermal-only) @0.05")
fig.tight_layout(); fig.savefig(FIGS / "cm_filters.png", dpi=300); plt.close(fig)
print("wrote cm_router.png + cm_filters.png to", FIGS)
