#!/usr/bin/env python
"""gen_drone_size_hist.py - the resolution-argument figure: distribution of drone
sqrt(area) in native pixels for Svanstrom vs Anti-UAV, with the imgsz resolvable
floors marked. Reads GT YOLO labels (class 0 = drone) directly. CPU, ~seconds.

  py docs/gen_drone_size_hist.py
Outputs: docs/figures/fig_drone_size_hist.{pdf,png}
"""
import os, glob, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(0)
OUT = "docs/figures/fig_drone_size_hist"

# dataset -> (labels dir, native W, H), class 0 = drone
DSETS = {
    "Svanström (640$\\times$512)": ("G:/drone/svanstrom_paired/RGB/labels", 640, 512),
    "Anti-UAV (1920$\\times$1080)": ("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels", 1920, 1080),
}
SAMPLE = 9000  # label files per dataset


def sqrt_areas(labels_dir, W, H):
    files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if len(files) > SAMPLE:
        files = random.sample(files, SAMPLE)
    out = []
    for fp in files:
        try:
            for ln in open(fp, encoding="utf-8"):
                p = ln.split()
                if len(p) >= 5 and p[0] == "0":
                    w, h = float(p[3]) * W, float(p[4]) * H
                    if w > 0 and h > 0:
                        out.append((w * h) ** 0.5)
        except Exception:
            pass
    return np.array(out)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.logspace(np.log10(2), np.log10(400), 45)
    colors = ["#1f77b4", "#d62728"]
    for (label, (d, W, H)), c in zip(DSETS.items(), colors):
        a = sqrt_areas(d, W, H)
        if a.size == 0:
            print("  no boxes:", label); continue
        ax.hist(a, bins=bins, density=True, alpha=0.55, color=c,
                label=f"{label}  (median {np.median(a):.0f} px, n={a.size})")
        ax.axvline(np.median(a), color=c, ls=":", lw=1)
    # imgsz resolvable floors for Svanstrom-native (p3 stride 8 at model input)
    for px, txt in [(8, "imgsz=640 floor"), (4, "imgsz=1280 floor")]:
        ax.axvline(px, color="black", ls="--", lw=1.1)
        ax.text(px, ax.get_ylim()[1] * 0.92, " " + txt, rotation=90,
                va="top", ha="left", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"drone $\sqrt{\mathrm{area}}$ in native pixels (log scale)")
    ax.set_ylabel("density")
    ax.set_title("Drone size by dataset: why Svanström needs imgsz=1280")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    os.makedirs("docs/figures", exist_ok=True)
    fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=160)
    print("wrote", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
