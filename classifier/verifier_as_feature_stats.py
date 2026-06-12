"""verifier_as_feature_stats.py — Phase 0 (verifier-score variant).

Question: can the VERIFIER's P(drone) — the MLP hooked to YOLO's P3/P5 backbone —
serve as an input FEATURE for the trust classifier, and does it survive grayscale
where the raw IR box features (ir_max_conf etc.) collapsed to chance (AUROC ~0.51,
see 2026-06-01_robust6_phase0_results.md)?

Zero GPU. Reuses the offline pipeline detection caches (eval/results/_offline_pipeline/
cache/*.pkl) which already store, per detection: the 517-D backbone feature vector,
box, conf, patch score, + per-frame GT. We run the already-trained verifier MLP over the
cached features (CPU) to get P(drone) per detection, label each detection drone(1)/
confuser-or-FP(0) by GT match, group by verifier context (RGB / thermal-IR / gray-IR),
and compute:

  - AUROC(verifier P(drone))  vs  AUROC(raw box conf)   per context  (does it beat conf?)
  - the same split thermal-IR vs gray-IR                (does it SURVIVE grayscale?)
  - Spearman corr(verifier, conf)                       (redundant with robust6's conf?)
  - bootstrap 95% CI on every AUROC                     (gray pool is small -> be honest)

  py classifier/verifier_as_feature_stats.py
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))
sys.path.insert(0, str(REPO / "eval"))
from metrics import score_detections_detailed            # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                # noqa: E402

CACHE = REPO / "eval" / "results" / "_offline_pipeline" / "cache"
IMG = REPO / "docs" / "analysis" / "images"
OUT_CSV = REPO / "classifier" / "fusion_models" / "optimal_v1" / "verifier_feature_stats.csv"
IMG.mkdir(parents=True, exist_ok=True)

# verifier context per modality (the "hooked to the brain" score)
VERIF = {
    "rgb":  REPO / "models/verifiers/rgb_v5/mlp_v5.pt",
    "ir":   REPO / "models/verifiers/ir_aligned/mlp_aligned.pt",
    "gray": REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt",
}
CONTEXT = {"rgb": "RGB (mlp_v5)", "ir": "thermal-IR (aligned)", "gray": "gray-IR (aligned_gray)"}
_vcache: dict = {}


def get_verif(modality):
    p = str(VERIF[modality])
    if p not in _vcache:
        _vcache[p] = MLPv4Verifier(Path(p), device="cpu")
    return _vcache[p]


def boot_auroc(y, s, n=1000, seed=0):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    idx = np.arange(len(y))
    accs = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        accs.append(roc_auc_score(y[b], s[b]))
    if not accs:
        return (np.nan, np.nan)
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def collect():
    """Per-detection table: modality, verifier P(drone), conf, label(drone=1)."""
    recs = []
    for pkl in sorted(CACHE.glob("*.pkl")):
        d = pickle.load(open(pkl, "rb"))
        meta, frames = d["meta"], d["frames"]
        modality, rule, has_drones = meta["modality"], meta["rule"], meta["has_drones"]
        verif = get_verif(modality)
        # verifier P(drone) once over all dets, split per frame
        counts = [len(f["confs"]) for f in frames]
        if sum(counts):
            allf = np.concatenate([f["feats"] for f in frames if len(f["feats"])], axis=0)
            pall = verif.predict_drone_probs(allf)
        else:
            pall = np.zeros(0, np.float32)
        i = 0
        for fr in frames:
            n = len(fr["confs"])
            if n == 0:
                continue
            pv = pall[i:i + n]; i += n
            if has_drones:
                dets = [(tuple(fr["boxes"][j]), float(fr["confs"][j])) for j in range(n)]
                det = score_detections_detailed(dets, [tuple(g) for g in fr["gt_boxes"]],
                                                 iou_thr=0.5, iop_thr=0.5)
                key = "matched_iop" if rule == "iop" else "matched_iou"
                labels = [r[key] for r in det]
            else:
                labels = [0] * n            # confuser surface: every det is a non-drone
            for j in range(n):
                recs.append(dict(surface=meta["name"], modality=modality,
                                 context=CONTEXT[modality], verifier_pdrone=float(pv[j]),
                                 conf=float(fr["confs"][j]), label=int(labels[j])))
    return pd.DataFrame(recs)


def main():
    df = collect()
    df.to_csv(OUT_CSV, index=False)
    print(f"Collected {len(df):,} detections across {df.surface.nunique()} surfaces\n")

    rows = []
    for ctx_key in ["rgb", "ir", "gray"]:
        sub = df[df.modality == ctx_key]
        y = sub.label.to_numpy()
        if len(y) == 0 or len(np.unique(y)) < 2:
            print(f"[{CONTEXT[ctx_key]}] n={len(y)} pos={int(y.sum())} -> skip (need both classes)")
            continue
        a_v = roc_auc_score(y, sub.verifier_pdrone)
        a_c = roc_auc_score(y, sub.conf)
        lo_v, hi_v = boot_auroc(y, sub.verifier_pdrone.to_numpy())
        lo_c, hi_c = boot_auroc(y, sub.conf.to_numpy())
        rho = spearmanr(sub.verifier_pdrone, sub.conf).correlation
        rows.append(dict(context=CONTEXT[ctx_key], n=len(y), pos=int(y.sum()), neg=int((y == 0).sum()),
                         auroc_verifier=a_v, v_lo=lo_v, v_hi=hi_v,
                         auroc_conf=a_c, c_lo=lo_c, c_hi=hi_c,
                         lift=a_v - a_c, spearman_v_conf=rho))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== Verifier P(drone) vs raw box conf — AUROC (drone vs confuser/FP), by context ===")
    print(res[["context", "n", "pos", "neg", "auroc_verifier", "v_lo", "v_hi",
               "auroc_conf", "c_lo", "c_hi", "lift", "spearman_v_conf"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nKEY READS:")
    g = res[res.context.str.startswith("gray")]
    t = res[res.context.str.startswith("thermal")]
    if len(g):
        gr = g.iloc[0]
        print(f"  gray-IR: verifier AUROC {gr.auroc_verifier:.3f} [{gr.v_lo:.3f},{gr.v_hi:.3f}] "
              f"vs conf {gr.auroc_conf:.3f} [{gr.c_lo:.3f},{gr.c_hi:.3f}]  (lift {gr.lift:+.3f})")
        print(f"           recall the raw ir_max_conf gray AUROC was ~0.51 (chance) in the full56 split.")
    if len(t):
        tr = t.iloc[0]
        print(f"  thermal-IR: verifier {tr.auroc_verifier:.3f} vs conf {tr.auroc_conf:.3f} (lift {tr.lift:+.3f})")

    # ── PLOT: verifier vs conf AUROC by context, with CIs ─────────────
    ctxs = res.context.tolist()
    x = np.arange(len(ctxs)); w = 0.38
    plt.figure(figsize=(9, 5.5))
    plt.bar(x - w/2, res.auroc_verifier, w, color="#9467bd", label="verifier P(drone)",
            yerr=[res.auroc_verifier - res.v_lo, res.v_hi - res.auroc_verifier], capsize=4)
    plt.bar(x + w/2, res.auroc_conf, w, color="#7f7f7f", label="raw box conf",
            yerr=[res.auroc_conf - res.c_lo, res.c_hi - res.auroc_conf], capsize=4)
    plt.axhline(0.5, color="k", lw=0.8, ls="--")
    for xi, r in zip(x, res.itertuples()):
        plt.text(xi, 0.03, f"n={r.n}\npos={r.pos}", ha="center", fontsize=8)
    plt.xticks(x, ctxs); plt.ylim(0, 1.02); plt.ylabel("AUROC (drone vs confuser/FP)")
    plt.title("Verifier P(drone) as a feature: discriminative power by context (95% CI)")
    plt.legend()
    plt.tight_layout(); plt.savefig(IMG / "verifier_feature_auroc.png", dpi=160); plt.close()

    res.to_csv(REPO / "models/routers/optimal_v1/verifier_feature_auroc.csv", index=False)
    print(f"\nSaved: {OUT_CSV.name}, verifier_feature_auroc.csv")
    print("Plot: docs/analysis/images/verifier_feature_auroc.png")


if __name__ == "__main__":
    main()
