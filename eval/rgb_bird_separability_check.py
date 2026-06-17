"""rgb_bird_separability_check.py — is the bird-FP fixable by adding bird confusers
(v3), or is it an inherent drone<->bird overlap?  ZERO-GPU.

The v2 balanced RGB filter keeps more rgb_bird_confuser fires (FP 199->374). Test
whether the birds it falsely keeps are SEPARABLE from real drones in 517-D:
  high AUROC + a separator that flags them as bird => MOVABLE (v3: add bird confusers
  to training teaches the rejection). low AUROC / they look like drones => inherent
  overlap (no retrain fixes it; accept the trade or size-gate).

  py eval/rgb_bird_separability_check.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from diagnose_rgbtest_veto_mechanism import real_drone_feats          # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                            # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402
from sklearn.model_selection import cross_val_score                 # noqa: E402

CACHE = REPO / "eval/results/_offline_pipeline/cache"
V2 = REPO / "eval/results/_v5_balanced_v2/classifiers/mlp_v5_balanced_v2.pt"
THR = 0.25


def to_px(la): return np.sqrt(np.exp(np.asarray(la, np.float64)))


def auroc_cv(A, B):
    X = np.vstack([A, B]).astype(np.float32); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    return float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                 StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean())


def main():
    mlp = MLPv4Verifier(V2, device="cpu")
    Xd, _ = real_drone_feats("rgb_dataset_test", "iou", mlp)                 # real drones (kept by v2)
    d = pickle.load(open(CACHE / "rgb_bird_confuser.pkl", "rb"))
    Xb = np.array([f for fr in d["frames"] for f in fr["feats"]], np.float32)  # all bird fires
    pb = mlp.predict_drone_probs(Xb)
    kept = Xb[pb >= THR]                                                     # the bird FP (falsely kept)
    print(f"drones {len(Xd)} (med px {np.median(to_px(Xd[:,1])):.0f}) | bird fires {len(Xb)} "
          f"(med px {np.median(to_px(Xb[:,1])):.0f}) | bird FP kept@{THR} {len(kept)} "
          f"(med px {np.median(to_px(kept[:,1])):.0f})")

    a_all = auroc_cv(Xd, Xb)
    a_kept = auroc_cv(Xd, kept) if len(kept) >= 10 else float("nan")
    print(f"\n[1] AUROC drones vs ALL birds        = {a_all:.3f}")
    print(f"[2] AUROC drones vs KEPT birds (FP)  = {a_kept:.3f}  (high => the FP birds ARE separable => MOVABLE)")

    # separator trained drones-vs-birds: what % of the kept-bird FP does it correctly call BIRD?
    Xtr = np.vstack([Xd, Xb]); ytr = np.r_[np.ones(len(Xd)), np.zeros(len(Xb))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    bird_frac = float((clf.predict(sc.transform(kept)) == 0).mean()) if len(kept) else float("nan")
    print(f"[3] a drone-vs-bird separator flags {bird_frac:.1%} of the kept-bird FP as BIRD")

    mean = mlp.scaler_mean.cpu().numpy().ravel(); scale = mlp.scaler_scale.cpu().numpy().ravel()
    scl = lambda M: (M - mean) / scale
    Cd, Cb = scl(Xd).mean(0), scl(Xb).mean(0); kk = scl(kept)
    closer_bird = float((np.linalg.norm(kk - Cb, axis=1) < np.linalg.norm(kk - Cd, axis=1)).mean())
    print(f"[4] kept-bird FP closer to BIRD centroid than DRONE: {closer_bird:.0%}")

    movable = (not np.isnan(a_kept)) and a_kept >= 0.85 and bird_frac >= 0.6
    print("\n" + ("MOVABLE -- the bird FP are separable from drones; v3 (add bird confusers to the distill "
                  "corpus) should teach the filter to reject them without losing the small-drone recall."
                  if movable else
                  "OVERLAP -- the kept birds look like the (bird-adjacent wosdetc/AirBird) drones we now keep; "
                  "adding bird confusers will fight the drone recall. Accept the bird-FP trade or size-gate."))


if __name__ == "__main__":
    main()
