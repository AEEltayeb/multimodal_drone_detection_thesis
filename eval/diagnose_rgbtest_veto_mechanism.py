"""diagnose_rgbtest_veto_mechanism.py — WHY does mlp_v5 veto real drones on
rgb_dataset_test but not on Svanstrom/Selcom?  (zero-GPU, from cached feats)

Correct feature layout: meta FIRST [0]=conf [1]=log_area [2]=aspect [3]=rel_cx
[4]=rel_cy, then p3[5:261], p5[261:517]. (The older diagnose_mlp_recall_drop.py
used meta-LAST and so mislabelled conf/log_area.)

For each surface, take real-drone detections (det matches a GT box), score
mlp_v5, split KEPT (P>=0.25) vs VETOED (P<0.25 = recall loss). Then locate the
two groups relative to the filter's TRAINING manifold (scaled feature space):

  - conf / log_area means (are vetoed drones small/low-conf?)
  - distance to train-DRONE centroid vs train-CONFUSER centroid
    (closer to confuser => "this dataset's drones look like the confusers the
     filter learned to reject"; the boundary is doing its job)
  - kNN (k=20) class purity among nearest training samples
  - OOD distance: nearest train-DRONE distance vs the train-drone internal NN
    scale (far => an appearance the filter never saw = coverage gap)

Svanstrom (mostly kept) is the control.

  py eval/diagnose_rgbtest_veto_mechanism.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from eval_v4_vs_patch import MLPv4Verifier  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

CACHE = REPO / "eval/results/_offline_pipeline/cache"
MLP_V5 = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
TRAIN_NPZ = REPO / "eval/results/_v5_selcom_pure_1x8/training_data.npz"
THR = 0.25


def match(d, g, rule):
    x1, y1 = max(d[0], g[0]), max(d[1], g[1]); x2, y2 = min(d[2], g[2]), min(d[3], g[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    da = (d[2] - d[0]) * (d[3] - d[1]); ga = (g[2] - g[0]) * (g[3] - g[1])
    return inter / da if rule == "iop" and da > 0 else (
        inter / (da + ga - inter) if da + ga - inter > 0 else 0.)


def real_drone_feats(name, rule, mlp):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    feats, probs = [], []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
            continue
        p = mlp.predict_drone_probs(np.array(fr["feats"]))
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g, rule) for g in fr["gt_boxes"]), default=0) >= 0.5:
                feats.append(fr["feats"][i]); probs.append(float(p[i]))
    return np.array(feats, dtype=np.float32), np.array(probs)


def main():
    mlp = MLPv4Verifier(MLP_V5, device="cpu")
    mean = mlp.scaler_mean.cpu().numpy().ravel()
    scale = mlp.scaler_scale.cpu().numpy().ravel()
    def sc(X): return (X - mean) / scale

    # training manifold (scaled)
    tr = np.load(TRAIN_NPZ); Xtr, ytr = tr["X"].astype(np.float32), tr["y"].astype(int)
    Xtr_s = sc(Xtr)
    Cd = Xtr_s[ytr == 1].mean(0); Cc = Xtr_s[ytr == 0].mean(0)
    nn_drone = NearestNeighbors(n_neighbors=21).fit(Xtr_s[ytr == 1])
    nn_all = NearestNeighbors(n_neighbors=20).fit(Xtr_s)
    # train-drone internal NN scale (exclude self -> 2nd neighbor)
    dd, _ = nn_drone.kneighbors(Xtr_s[ytr == 1][np.random.RandomState(0).choice((ytr == 1).sum(), 2000, replace=False)])
    train_internal = float(np.median(dd[:, 1]))
    print(f"train manifold: {(ytr==1).sum()} drones, {(ytr==0).sum()} confusers; "
          f"median train-drone NN dist = {train_internal:.2f}")

    for name, rule in [("rgb_dataset_test", "iou"), ("svanstrom", "iop")]:
        X, p = real_drone_feats(name, rule, mlp)
        kept = p >= THR
        Xs = sc(X)
        print(f"\n=== {name} ({rule}) ===  real drones {len(p)}  | "
              f"KEPT {kept.sum()}  VETOED {(~kept).sum()}  (loss {(~kept).mean():.1%})")
        for grp, m in (("KEPT", kept), ("VETOED", ~kept)):
            if m.sum() < 5:
                print(f"  {grp}: n<5"); continue
            xg = Xs[m]
            dD = np.linalg.norm(xg - Cd, axis=1)        # to drone centroid
            dC = np.linalg.norm(xg - Cc, axis=1)        # to confuser centroid
            knn_d, _ = nn_all.kneighbors(xg)
            # class purity of 20-NN
            _, idx = nn_all.kneighbors(xg)
            conf_frac = (ytr[idx] == 0).mean(1)
            ood, _ = nn_drone.kneighbors(xg)             # nearest train drone
            print(f"  {grp:<7} conf={X[m,0].mean():.3f}  log_area={X[m,1].mean():.2f}  "
                  f"| dist->drone {dD.mean():.2f}  dist->confuser {dC.mean():.2f}  "
                  f"closer-to-confuser {(dC<dD).mean():.0%}")
            print(f"          20-NN confuser-fraction {conf_frac.mean():.0%}  "
                  f"| nearest-train-drone dist {np.median(ood[:,0]):.2f} "
                  f"(train internal {train_internal:.2f}, "
                  f"ratio {np.median(ood[:,0])/train_internal:.2f}x)")


if __name__ == "__main__":
    main()
