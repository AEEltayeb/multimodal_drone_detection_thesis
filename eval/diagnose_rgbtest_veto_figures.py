"""diagnose_rgbtest_veto_figures.py — figures for the rgb_dataset_test veto
mechanism (companion to diagnose_rgbtest_veto_mechanism.py). ZERO-GPU.

Produces, in docs/analysis/images/:
  2026-06-17_rgbtest_veto_mechanism.png  (3 panels)
    (a) train-drone vs train-confuser log_area, with rgb_test VETOED drones
        overlaid -> shows the vetoed drones sit at the SMALL end where training
        drones are thin and confusers dominate (the size-skew coverage gap).
    (b) nearest-train-drone OOD ratio, KEPT vs VETOED (rgb_test) + svanstrom
        VETOED control -> shows rgb_test vetoed drones are far OOD (~3.3x the
        internal train-drone scale) vs svanstrom's mildly-OOD tail (~1.4x).
    (c) per-size filter veto-rate among detector-matched drones (from the
        2026-06-17 imgsz JSON) -> the veto is concentrated on small drones.

Also writes 2026-06-17_rgbtest_veto_stats.json with the cited numbers.

  py eval/diagnose_rgbtest_veto_figures.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
import diagnose_rgbtest_veto_mechanism as vm           # noqa: E402  (reuse match/real_drone_feats)
from eval_v4_vs_patch import MLPv4Verifier             # noqa: E402
from sklearn.neighbors import NearestNeighbors         # noqa: E402

IMG = REPO / "docs" / "analysis" / "images"; IMG.mkdir(parents=True, exist_ok=True)
IMGSZ_JSON = REPO / "docs" / "analysis" / "2026-06-17_rgbtest_imgsz_640_vs_1280.json"
OUT_PNG = IMG / "2026-06-17_rgbtest_veto_mechanism.png"
OUT_JSON = REPO / "docs" / "analysis" / "2026-06-17_rgbtest_veto_stats.json"


def main():
    mlp = MLPv4Verifier(vm.MLP_V5, device="cpu")
    mean = mlp.scaler_mean.cpu().numpy().ravel()
    scale = mlp.scaler_scale.cpu().numpy().ravel()
    def sc(X): return (X - mean) / scale

    tr = np.load(vm.TRAIN_NPZ); Xtr, ytr = tr["X"].astype(np.float32), tr["y"].astype(int)
    Xtr_s = sc(Xtr)
    drone_s = Xtr_s[ytr == 1]
    # internal train-drone NN scale (exclude self)
    nn2 = NearestNeighbors(n_neighbors=2).fit(drone_s)
    rng = np.random.RandomState(0)
    samp = drone_s[rng.choice(len(drone_s), min(2000, len(drone_s)), replace=False)]
    dd, _ = nn2.kneighbors(samp)
    internal = float(np.median(dd[:, 1]))
    nn1 = NearestNeighbors(n_neighbors=1).fit(drone_s)

    def ood_ratio(X):
        d, _ = nn1.kneighbors(sc(X)); return d[:, 0] / internal

    # rgb_test + svanstrom real-drone splits
    Xr, pr = vm.real_drone_feats("rgb_dataset_test", "iou", mlp)
    keptr = pr >= vm.THR
    Xsv, psv = vm.real_drone_feats("svanstrom", "iop", mlp)
    keptsv = psv >= vm.THR

    la = 1  # meta-first: idx[1] = log_area
    tr_drone_la, tr_conf_la = Xtr[ytr == 1, la], Xtr[ytr == 0, la]
    veto_la = Xr[~keptr, la]
    ood_kept, ood_veto = ood_ratio(Xr[keptr]), ood_ratio(Xr[~keptr])
    ood_sv_veto = ood_ratio(Xsv[~keptsv])

    veto_med = float(np.median(veto_la))
    frac_tr_below = float((tr_drone_la < veto_med).mean())
    frac_conf_below = float((tr_conf_la < veto_med).mean())

    # per-size veto from imgsz JSON (640)
    jj = json.loads(IMGSZ_JSON.read_text())
    b640 = jj["640"]["buckets"]
    order = ["<16px", "16-32px", "32-64px", ">=64px"]
    veto_rate = [1 - b640[k]["kept"] / max(b640[k]["matched"], 1) for k in order]

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(min(tr_drone_la.min(), veto_la.min()),
                       np.percentile(tr_drone_la, 99.5), 50)
    ax[0].hist(tr_drone_la, bins=bins, density=True, alpha=.55, label=f"train drones (n={len(tr_drone_la)})", color="tab:blue")
    ax[0].hist(tr_conf_la, bins=bins, density=True, alpha=.45, label=f"train confusers (n={len(tr_conf_la)})", color="tab:orange")
    ax[0].hist(veto_la, bins=bins, density=True, alpha=.65, label=f"rgb_test VETOED drones (n={len(veto_la)})", color="tab:red", hatch="//")
    ax[0].axvline(veto_med, color="tab:red", ls="--", lw=1.4, label=f"vetoed median log_area={veto_med:.2f}")
    ax[0].set_xlabel("log bbox area"); ax[0].set_ylabel("density")
    ax[0].set_title(f"(a) Size skew: only {frac_tr_below:.0%} of train drones are\nas small as the median vetoed drone")
    ax[0].legend(fontsize=8)

    obins = np.linspace(0, 6, 40)
    ax[1].hist(ood_kept, bins=obins, density=True, alpha=.6, label=f"rgb_test KEPT (med {np.median(ood_kept):.2f}x)", color="tab:green")
    ax[1].hist(ood_veto, bins=obins, density=True, alpha=.6, label=f"rgb_test VETOED (med {np.median(ood_veto):.2f}x)", color="tab:red")
    ax[1].hist(ood_sv_veto, bins=obins, density=True, alpha=.45, label=f"svanstrom VETOED (med {np.median(ood_sv_veto):.2f}x)", color="tab:gray")
    ax[1].axvline(1.0, color="k", ls=":", lw=1, label="train-drone internal scale (1x)")
    ax[1].set_xlabel("nearest-train-drone distance / internal scale")
    ax[1].set_title("(b) Coverage gap: vetoed rgb_test drones are\nfar OOD from the training drone manifold")
    ax[1].legend(fontsize=8)

    x = np.arange(len(order))
    bars = ax[2].bar(x, [v * 100 for v in veto_rate], color=["tab:red", "tab:orange", "gold", "tab:green"])
    ax[2].set_xticks(x); ax[2].set_xticklabels(order)
    ax[2].set_ylabel("filter veto rate among matched drones (%)")
    ax[2].set_title("(c) The veto is concentrated on small drones\n(rgb_dataset_test, imgsz 640)")
    for b, v in zip(bars, veto_rate):
        ax[2].text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=9)

    fig.suptitle("rgb_dataset_test: the shipped mlp_v5 filter vetoes small, coverage-gap drones (not genuine confusers)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    stats = {
        "internal_train_drone_scale": internal,
        "rgb_test": {"n_real_drones": int(len(pr)), "kept": int(keptr.sum()), "vetoed": int((~keptr).sum()),
                     "veto_rate": float((~keptr).mean()),
                     "ood_ratio_kept_med": float(np.median(ood_kept)), "ood_ratio_vetoed_med": float(np.median(ood_veto))},
        "svanstrom": {"n_real_drones": int(len(psv)), "vetoed": int((~keptsv).sum()),
                      "ood_ratio_vetoed_med": float(np.median(ood_sv_veto))},
        "vetoed_median_log_area": veto_med,
        "frac_train_drones_below_vetoed_median": frac_tr_below,
        "frac_train_confusers_below_vetoed_median": frac_conf_below,
        "persize_veto_rate_640": dict(zip(order, veto_rate)),
        "figure": str(OUT_PNG),
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"saved {OUT_PNG}\nsaved {OUT_JSON}")
    print(f"\nKEY NUMBERS:")
    print(f"  rgb_test veto {(~keptr).mean():.1%} ({(~keptr).sum()}/{len(pr)}); OOD ratio kept {np.median(ood_kept):.2f}x vs vetoed {np.median(ood_veto):.2f}x")
    print(f"  svanstrom vetoed OOD ratio {np.median(ood_sv_veto):.2f}x (control)")
    print(f"  only {frac_tr_below:.1%} of train drones are <= vetoed median log_area {veto_med:.2f} (vs {frac_conf_below:.1%} of train confusers)")
    print(f"  per-size veto (640): " + ", ".join(f"{k} {v:.0%}" for k, v in zip(order, veto_rate)))


if __name__ == "__main__":
    main()
