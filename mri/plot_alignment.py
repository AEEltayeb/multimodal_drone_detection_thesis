"""
plot_alignment — Proof #2 figure: "grayscale transfers to thermal".

From mri/results/modality_align/features.npz (yolo-only features per
modality x class) + modality_align.json, render a 2-panel thesis figure:

  (A) Per-modality z-scored features projected on the drone-vs-confuser LDA
      axis (fit on the combined aligned data). If gray-drone overlaps
      thermal-drone and gray-confuser overlaps thermal-confuser, the modality
      gap has collapsed -> the two modalities are aligned.
  (B) gray->thermal transfer AUROC: raw (no align) vs per-modality z-score vs
      CORAL, against the within-modality ceiling.

CPU only (no detector). Output: mri/docs/images/mri_gray_thermal_alignment.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "mri" / "results" / "modality_align"
OUT = REPO / "mri" / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    z = np.load(SRC / "features.npz")
    Td, Tc = z["thermal_drone"], z["thermal_conf"]
    Gd, Gc = z["gray_drone"], z["gray_conf"]
    # per-modality z-score (align each modality to its own mean/std)
    sct = StandardScaler().fit(np.vstack([Td, Tc]))
    scg = StandardScaler().fit(np.vstack([Gd, Gc]))
    Tdz, Tcz = sct.transform(Td), sct.transform(Tc)
    Gdz, Gcz = scg.transform(Gd), scg.transform(Gc)
    # LDA drone-vs-confuser on the combined aligned data
    X = np.vstack([Tdz, Tcz, Gdz, Gcz])
    y = np.r_[np.ones(len(Tdz)), np.zeros(len(Tcz)), np.ones(len(Gdz)), np.zeros(len(Gcz))]
    lda = LinearDiscriminantAnalysis().fit(X, y)
    proj = {"thermal drone": lda.transform(Tdz).ravel(),
            "thermal confuser": lda.transform(Tcz).ravel(),
            "gray drone": lda.transform(Gdz).ravel(),
            "gray confuser": lda.transform(Gcz).ravel()}

    res = json.loads((SRC / "modality_align.json").read_text())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
    # Panel A — aligned LDA histograms
    styles = {"thermal drone": ("#1a7f1a", "-"), "gray drone": ("#69d069", "--"),
              "thermal confuser": ("#b41a1a", "-"), "gray confuser": ("#e87a7a", "--")}
    for name, v in proj.items():
        c, ls = styles[name]
        axA.hist(v, bins=60, density=True, histtype="step", lw=2.0, color=c, ls=ls, label=name)
    axA.set_title("(A) Drone vs confuser LDA after per-modality z-score\n"
                  "thermal & grayscale land in the SAME regions", fontsize=11)
    axA.set_xlabel("LDA axis (drone-vs-confuser)"); axA.set_ylabel("density")
    axA.legend(fontsize=8); axA.grid(alpha=0.25)

    # Panel B — transfer AUROC bars
    bars = [("raw\n(no align)", res["raw_gray_to_thermal"], "#b41a1a"),
            ("CORAL", res["coral_gray_to_thermal"], "#d98a00"),
            ("per-modality\nz-score", res["permod_gray_to_thermal"], "#1a7f1a"),
            ("ceiling\n(thermal CV)", res["ceiling_thermal"], "#888888")]
    xs = np.arange(len(bars))
    axB.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for x, b in zip(xs, bars):
        axB.text(x, b[1] + 0.012, f"{b[1]:.3f}", ha="center", fontsize=10, fontweight="bold")
    axB.axhline(0.5, color="k", ls=":", lw=1, alpha=0.6)
    axB.set_xticks(xs); axB.set_xticklabels([b[0] for b in bars], fontsize=9)
    axB.set_ylim(0.4, 1.02); axB.set_ylabel("gray->thermal transfer AUROC")
    axB.set_title("(B) Grayscale-trained confuser filter transfers to thermal\n"
                  "chance -> near-ceiling after per-modality z-score", fontsize=11)
    axB.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = OUT / "mri_gray_thermal_alignment.png"
    fig.savefig(p, dpi=140); print(f"wrote {p}")


if __name__ == "__main__":
    main()
