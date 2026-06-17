"""rgbtest_fix_separability_check.py — PRE-TRAINING statistical check: do the
517-D features SUPPORT the RGB balanced-retrain fix?  ZERO-GPU.

The vetoed rgb_test drones sit in a region currently OWNED by small confusers
(55% NN-confuser). The fix (add small/wosdetc drones) only works if the 517-D
features can actually SEPARATE those vetoed drones from the small confusers — if
they overlap, no amount of rebalancing helps. We test:

  1. AUROC(vetoed rgb_test drones  vs  small training confusers)  -- the crux:
     can a boundary separate them at all? (5-fold CV, logistic on scaled 517-D)
  2. AUROC(known small training drones vs small training confusers) -- baseline.
  3. A small-specialist classifier (trained ONLY on small drones vs small
     confusers) applied to the vetoed drones -> % it would KEEP as drone. This is
     "would a size-aware filter rescue them?"
  4. Are the vetoed drones closer to the small-DRONE centroid or small-CONFUSER
     centroid (scaled space)?

PASS (features support the fix) iff (1) AUROC high AND (3) most vetoed drones
classified drone by the small-specialist.

  py eval/rgbtest_fix_separability_check.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from diagnose_rgbtest_veto_mechanism import real_drone_feats, MLP_V5, TRAIN_NPZ  # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                                       # noqa: E402
from sklearn.linear_model import LogisticRegression                             # noqa: E402
from sklearn.preprocessing import StandardScaler                                # noqa: E402
from sklearn.model_selection import cross_val_score                            # noqa: E402

SMALL_PX = 16


def px(la):
    return np.sqrt(np.exp(np.asarray(la, dtype=np.float64)))


def auroc_cv(A, B):
    X = np.vstack([A, B]).astype(np.float32)
    y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    Xs = StandardScaler().fit_transform(X)
    return float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                 Xs, y, cv=5, scoring="roc_auc").mean())


def main():
    mlp = MLPv4Verifier(MLP_V5, device="cpu")
    Xr, pr = real_drone_feats("rgb_dataset_test", "iou", mlp)
    veto = Xr[pr < 0.25]
    print(f"rgb_test real drones {len(pr)} | vetoed {len(veto)} ({(pr<0.25).mean():.1%})")

    z = np.load(TRAIN_NPZ); X = z["X"].astype(np.float32); y = z["y"].astype(int)
    p = px(X[:, 1])
    sc = X[(y == 0) & (p < SMALL_PX)]      # small training confusers
    sd = X[(y == 1) & (p < SMALL_PX)]      # small training drones
    veto_small = veto[px(veto[:, 1]) < SMALL_PX]
    print(f"train small(<16px): drones {len(sd)} | confusers {len(sc)} ; vetoed small {len(veto_small)}")

    # 1 + 2 separability
    a_veto = auroc_cv(veto, sc)
    a_base = auroc_cv(sd, sc)
    print(f"\n[1] AUROC vetoed-drones vs small-confusers   = {a_veto:.3f}  (crux: can features separate them?)")
    print(f"[2] AUROC small-drones  vs small-confusers   = {a_base:.3f}  (baseline)")

    # 3 small-specialist applied to vetoed
    Xtr = np.vstack([sd, sc]); ytr = np.r_[np.ones(len(sd)), np.zeros(len(sc))]
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    keep_frac = float((clf.predict(scaler.transform(veto)) == 1).mean())
    print(f"[3] small-specialist keeps {keep_frac:.1%} of vetoed drones as DRONE "
          f"(vs the shipped filter which vetoes 100% of them)")

    # 4 centroid proximity (scaled by full-train scaler)
    mean = mlp.scaler_mean.cpu().numpy().ravel(); scale = mlp.scaler_scale.cpu().numpy().ravel()
    sca = lambda M: (M - mean) / scale
    Cd, Cc = sca(sd).mean(0), sca(sc).mean(0)
    vv = sca(veto)
    closer_drone = float((np.linalg.norm(vv - Cd, axis=1) < np.linalg.norm(vv - Cc, axis=1)).mean())
    print(f"[4] vetoed drones closer to small-DRONE centroid than small-CONFUSER: {closer_drone:.0%}")

    ok = (a_veto >= 0.85) and (keep_frac >= 0.6)
    print("\n" + ("PASS -- 517-D features SUPPORT the fix: vetoed drones are separable from "
                  "small confusers, so adding small/wosdetc drone coverage will let the retrain keep them."
                  if ok else
                  "WEAK -- features only partially separate vetoed drones from small confusers; "
                  "rebalancing will help but may not fully close the gap (consider size-aware threshold too)."))


if __name__ == "__main__":
    main()
