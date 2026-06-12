"""failopen_expanded_ref.py — does expanding the confuser REFERENCE fix fail-open's
svanstrom precision backfire WITHOUT losing OOD-drone recall?

Fail-open keeps a det if (mlp keep) OR (OOD-from-confusers > tau). The backfire: svan
clutter was "OOD" only because the reference (rgb_confusers) lacked svan-style clutter.
Test: build OOD reference = rgb_confusers + svan clutter from a HELD-OUT half of svan
frames, then evaluate fail-open on the OTHER half. Compare the recall/precision frontier
(sweep tau) for ORIGINAL ref vs EXPANDED ref vs full-veto. Hypothesis: expanded-ref
dominates (recovers precision at the same recall) because drones stay far from confusers
while now-known clutter gets vetoed.

CPU/offline. Frame split is index-parity (mild same-sequence leakage caveat). Figure:
docs/analysis/images/failopen_expanded_ref_svan.png

  py eval/failopen_expanded_ref.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from metrics import score_detections, compute_prf
from eval_v4_vs_patch import MLPv4Verifier
C = REPO / "eval/results/_offline_pipeline/cache"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
mlp = MLPv4Verifier(REPO / "models/verifiers/rgb_v5/mlp_v5.pt", device="cpu")
THR = 0.25


def iop(d, g):
    x1, y1 = max(d[0], g[0]), max(d[1], g[1]); x2, y2 = min(d[2], g[2]), min(d[3], g[3])
    i = max(0., x2-x1)*max(0., y2-y1); da = (d[2]-d[0])*(d[3]-d[1]); return i/da if da > 0 else 0.


def make_ood(ref_feats):
    mu, sd = ref_feats.mean(0), ref_feats.std(0) + 1e-6
    nn = NearestNeighbors(n_neighbors=min(5, len(ref_feats))).fit((ref_feats - mu) / sd)
    return lambda X: nn.kneighbors((X - mu) / sd)[0].mean(1) if len(X) else np.zeros(0)


def score_frames(frames, keep_fn):
    tp = fp = fn = 0
    for fr in frames:
        n = len(fr["feats"])
        keep = keep_fn(fr) if n else np.zeros(0, bool)
        kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
        t, f, fnn = score_detections(kept, [tuple(g) for g in fr["gt_boxes"]], rule="iop", iou_thr=0.5, iop_thr=0.5)
        tp += t; fp += f; fn += fnn
    return compute_prf(tp, fp, fn)


def main():
    svan = pickle.load(open(C / "svanstrom.pkl", "rb"))["frames"]
    rgb_conf = np.array([f for fr in pickle.load(open(C / "rgb_confuser.pkl", "rb"))["frames"] for f in fr["feats"]])
    # split svan frames: even=reference (clutter source), odd=test
    ref_frames = svan[0::2]; test_frames = svan[1::2]
    # svan clutter from reference frames = non-GT-matching dets
    svan_clutter = []
    for fr in ref_frames:
        for i, b in enumerate(fr["boxes"]):
            if max((iop(b, g) for g in fr["gt_boxes"]), default=0) < 0.5:
                svan_clutter.append(fr["feats"][i])
    svan_clutter = np.array(svan_clutter)
    print(f"test frames={len(test_frames)}  rgb_conf ref={len(rgb_conf)}  svan clutter added={len(svan_clutter)}")

    ood_orig = make_ood(rgb_conf)
    ood_exp = make_ood(np.vstack([rgb_conf, svan_clutter]))

    # precompute per-test-frame mlp prob + both ood scores
    for fr in test_frames:
        fr["_p"] = mlp.predict_drone_probs(fr["feats"]) if len(fr["feats"]) else np.zeros(0)
        fr["_oo"] = ood_orig(fr["feats"]); fr["_oe"] = ood_exp(fr["feats"])

    # baselines
    bare = score_frames(test_frames, lambda fr: np.ones(len(fr["feats"]), bool))
    fullveto = score_frames(test_frames, lambda fr: fr["_p"] >= THR)
    print(f"\nbare:      P={bare['precision']:.3f} R={bare['recall']:.3f} F1={bare['f1']:.3f}")
    print(f"full-veto: P={fullveto['precision']:.3f} R={fullveto['recall']:.3f} F1={fullveto['f1']:.3f}")

    # tau sweeps -> frontiers
    def frontier(key):
        allv = np.concatenate([fr[key] for fr in test_frames if len(fr["feats"])])
        taus = np.quantile(allv, np.linspace(0.0, 1.0, 40))
        pts = []
        for t in taus:
            m = score_frames(test_frames, lambda fr, t=t, key=key: (fr["_p"] >= THR) | (fr[key] > t))
            pts.append((m["recall"], m["precision"], m["f1"]))
        return np.array(pts)
    fo, fe = frontier("_oo"), frontier("_oe")

    # headline: at the tau recovering recall ~ bare (full recovery), compare precision
    def at_recall(pts, target):
        ok = pts[pts[:, 0] >= target]
        return ok[ok[:, 1].argmax()] if len(ok) else pts[pts[:, 0].argmax()]
    tgt = min(bare["recall"], 0.90)
    ro, re_ = at_recall(fo, tgt), at_recall(fe, tgt)
    print(f"\n@recall>={tgt:.2f}:")
    print(f"  failopen ORIGINAL ref:  R={ro[0]:.3f} P={ro[1]:.3f} F1={ro[2]:.3f}")
    print(f"  failopen EXPANDED ref:  R={re_[0]:.3f} P={re_[1]:.3f} F1={re_[2]:.3f}")
    print(f"  full-veto:              R={fullveto['recall']:.3f} P={fullveto['precision']:.3f}")

    plt.figure(figsize=(7, 5.5))
    plt.plot(fo[:, 0], fo[:, 1], "-o", ms=3, label="fail-open (original ref: rgb_confusers)", color="red")
    plt.plot(fe[:, 0], fe[:, 1], "-o", ms=3, label="fail-open (expanded ref: + svan clutter)", color="green")
    plt.scatter([fullveto["recall"]], [fullveto["precision"]], c="black", s=90, marker="*", label="full-veto (mlp_v5)", zorder=5)
    plt.scatter([bare["recall"]], [bare["precision"]], c="gray", s=70, marker="s", label="bare (no verifier)", zorder=5)
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Svanstrom fail-open: original vs expanded confuser reference")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(IMG / "failopen_expanded_ref_svan.png", dpi=160); plt.close()
    print(f"\nfigure -> {IMG/'failopen_expanded_ref_svan.png'}")


if __name__ == "__main__":
    main()
