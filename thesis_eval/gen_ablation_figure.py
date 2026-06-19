"""gen_ablation_figure.py — fig:pipeline_ablation for the thesis, from tier1_results.json.

(a) drone-positive paired surfaces: F1 bars (bare -> +router -> production) with 95% CI whiskers.
(b) confuser surfaces: frame fire-rate bars (bare / patch / mlp / router / composed).
Writes figures/fig_pipeline_ablation.{pdf,png} into the Overleaf folder.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
R = json.load(open(REPO / "thesis_eval/results/tier1_results.json"))
FIGDIR = REPO / "docs/thesis_working_distilling_overleaf/figures"

CELLS_A = [("bare", "bare"), ("clf[robust8]", "+ router"), ("clf->filt[robust8]", "+ filter\n(production)")]
SURF_A = [("svanstrom", "Svanström (IoP@0.5)"), ("antiuav", "Anti-UAV (IoU@0.5)")]
CELLS_B = [("bare", "bare"), ("filt_patch", "patch"), ("filt_mlp", "mlp_v5_v4"),
           ("clf[robust8]", "router"), ("clf->filt[robust8]", "composed")]
SURF_B = [("rgb_confuser", "RGB confusers"), ("ir_confusers", "IR confusers")]
C = {"bare": "#9e9e9e", "patch": "#7fa8d9", "mlp_v5_v4": "#2e7d32", "router": "#f0a050", "composed": "#1a4e8a",
     "+ router": "#f0a050", "+ filter\n(production)": "#1a4e8a"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})

# (a) paired drone surfaces
W, X0 = 0.26, 0.0
xt, xl = [], []
for si, (s, slabel) in enumerate(SURF_A):
    B = R[s]["B_pipeline"]
    for ci, (cell, clabel) in enumerate(CELLS_A):
        p = B[cell]; x = X0 + si * 1.1 + ci * W
        ci_lo, ci_hi = p.get("f1_ci", [p["f1"], p["f1"]])
        ax1.bar(x, p["f1"], width=W * 0.92, color=C[clabel] if clabel in C else "#888",
                yerr=[[p["f1"] - ci_lo], [ci_hi - p["f1"]]], capsize=3, error_kw={"lw": 1})
        ax1.text(x, 0.30, clabel.replace("\n", " "), rotation=90, ha="center", va="bottom",
                 fontsize=7.5, color="white", fontweight="bold")
        ax1.text(x, p["f1"] + 0.012, f"{p['f1']:.3f}", ha="center", fontsize=7)
    xt.append(X0 + si * 1.1 + W); xl.append(slabel)
ax1.set_xticks(xt); ax1.set_xticklabels(xl, fontsize=9)
ax1.set_ylim(0.25, 1.04); ax1.set_ylabel("drone F1 (trust-aware)")
ax1.set_title("(a) drone-positive paired surfaces", fontsize=10)
ax1.spines[["top", "right"]].set_visible(False)

# (b) confuser surfaces
for si, (s, slabel) in enumerate(SURF_B):
    Cc = R[s]["C_confuser"]
    for ci, (cell, clabel) in enumerate(CELLS_B):
        if cell not in Cc:
            continue
        p = Cc[cell]; x = si * 1.45 + ci * 0.24
        ax2.bar(x, p["fire_rate"], width=0.22, color=C[clabel])
        ax2.text(x, p["fire_rate"] + 0.004, f"{p['fire_rate']*100:.1f}".rstrip("0").rstrip(".") + "%",
                 ha="center", fontsize=6.6, rotation=0)
ax2.set_xticks([si * 1.45 + 0.48 for si in range(len(SURF_B))])
ax2.set_xticklabels([s for _, s in SURF_B], fontsize=9)
ax2.set_ylabel("frame fire rate"); ax2.set_ylim(0, 0.40)
ax2.set_title("(b) confuser surfaces (no drones; lower is better)", fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for _, k in CELLS_B]
ax2.legend(handles, [k for _, k in CELLS_B], fontsize=8, ncol=2, frameon=False, loc="upper right")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_pipeline_ablation.{ext}", dpi=200, bbox_inches="tight")
print("wrote", FIGDIR / "fig_pipeline_ablation.{pdf,png}")
