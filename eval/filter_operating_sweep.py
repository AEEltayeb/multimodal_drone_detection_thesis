"""
Filter operating-point sweep (thesis figure `fig_filter_operating`). ZERO-GPU.

For each confuser filter (RGB mlp_v5 / IR-thermal aligned) sweep the P(drone)
threshold and plot, on one axis pair, the drone DETECTION recall (fraction of GT-matched true-drone
detections retained, pooled over that filter's drone surfaces) and the confuser FIRE-RATE (frame-level,
the matching confuser surface). The shipped operating point is marked. Reads the unified caches and
recomputes P(drone) from the cached 517-D features, so nothing is re-detected.

Run from repo root:  py eval/filter_operating_sweep.py
Outputs: docs/thesis_working_distilling_overleaf/figures/fig_filter_operating.{pdf,png}
         eval/results/filter_operating_sweep.json  (caption numbers)
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "thesis_eval"))
sys.path.insert(0, str(REPO / "eval"))
import pipeline_eval_unified as U          # noqa: E402
from metrics import score_detections_detailed  # noqa: E402

CACHE = REPO / "thesis_eval" / "cache"
FIGDIR = REPO / "docs" / "thesis_working_distilling_overleaf" / "figures"
OUTJSON = REPO / "eval" / "results" / "filter_operating_sweep.json"
GRID = np.round(np.arange(0.02, 0.951, 0.01), 3)

# label -> (verifier key, slot, [drone surfaces], confuser surface, shipped thr, panel title)
FILTERS = [
    ("RGB mlp_v5",        "mlp_v5",       "rgb", ["svanstrom", "antiuav", "dut_antiuav_960"], "rgb_confuser",  0.25),
    ("IR-thermal aligned","aligned",      "ir",  ["svanstrom", "antiuav", "ir_dset_final"],   "ir_confusers",  0.05),
]


def load(name):
    import pickle
    return pickle.load(open(CACHE / f"{name}.pkl", "rb"))


def drone_pos_probs(verif, slot, surfaces):
    """P(drone) of detections that match a real drone GT (true-drone detections), pooled."""
    out = []
    for s in surfaces:
        d = load(s); meta, frames = d["meta"], d["frames"]; rule = meta["rule"]
        probs = U.batch_probs(frames, slot, verif)
        gt_key = "rgb_gt" if slot == "rgb" else "ir_gt"
        for i, fr in enumerate(frames):
            dets = U.dets2(fr[slot])
            if not dets:
                continue
            det = score_detections_detailed(dets, U.gts(fr[gt_key]))
            for j, dd in enumerate(det):
                if (dd["matched_iop"] if rule == "iop" else dd["matched_iou"]):
                    out.append(float(probs[i][j]))
    return np.array(out)


def confuser_frame_probs(verif, slot, surface):
    """List of per-frame P(drone) arrays on a confuser surface (every detection is a false positive)."""
    d = load(surface); frames = d["frames"]
    probs = U.batch_probs(frames, slot, verif)
    return [probs[i] for i in range(len(frames))], len(frames)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=str(FIGDIR), help="figure output dir (default = committed)")
    ap.add_argument("--json", default=str(OUTJSON), help="caption-numbers JSON path")
    a = ap.parse_args()
    figdir, outjson = Path(a.fig_dir), Path(a.json)
    verifs = U.load_verifiers("cpu")
    figdir.mkdir(parents=True, exist_ok=True); outjson.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    summary = {}
    for ax, (name, vk, slot, dsurf, csurf, shipped) in zip(axes, FILTERS):
        pos = drone_pos_probs(verifs[vk], slot, dsurf)
        cfr, nfr = confuser_frame_probs(verifs[vk], slot, csurf)
        recall = np.array([(pos >= t).mean() if len(pos) else 0.0 for t in GRID])
        fire = np.array([np.mean([1.0 if (len(a) and (a >= t).any()) else 0.0 for a in cfr]) for t in GRID])

        ax.plot(GRID, recall, color="#1f77b4", lw=2, label="drone recall")
        ax.set_xlabel(r"filter threshold  $P(\mathrm{drone})$"); ax.set_ylim(0, 1.02)
        ax.set_ylabel("drone detection recall", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(GRID, fire, color="#d62728", lw=2, ls="--", label="confuser fire")
        ax2.set_ylabel("confuser fire-rate", color="#d62728"); ax2.set_ylim(0, max(0.05, fire.max() * 1.1))
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.axvline(shipped, color="0.4", ls=":", lw=1.5)
        si = int(np.argmin(np.abs(GRID - shipped)))
        ax.plot([shipped], [recall[si]], "o", color="#1f77b4", ms=6)
        ax2.plot([shipped], [fire[si]], "s", color="#d62728", ms=6)
        ax.set_title(f"{name}  (shipped $t={shipped}$)", fontsize=10)
        # caption numbers
        def at(t):
            i = int(np.argmin(np.abs(GRID - t)))
            return round(float(recall[i]), 3), round(float(fire[i]), 4)
        summary[name] = {"slot": slot, "drone_surfaces": dsurf, "confuser_surface": csurf,
                         "n_pos": int(len(pos)), "n_conf_frames": nfr, "shipped_thr": shipped,
                         "shipped": at(shipped), "t0.05": at(0.05), "t0.10": at(0.10), "t0.25": at(0.25)}

    fig.tight_layout()
    fig.savefig(figdir / "fig_filter_operating.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fig_filter_operating.png", dpi=150, bbox_inches="tight")
    json.dump(summary, open(outjson, "w"), indent=2)
    print("WROTE", figdir / "fig_filter_operating.pdf")
    for k, v in summary.items():
        print(f"  {k:20s} POS={v['n_pos']} shipped(t={v['shipped_thr']}) recall/fire={v['shipped']}"
              f"  | t0.05={v['t0.05']}  t0.10={v['t0.10']}  t0.25={v['t0.25']}")


if __name__ == "__main__":
    main()
