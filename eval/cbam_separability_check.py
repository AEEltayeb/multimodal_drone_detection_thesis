"""cbam_separability_check.py — would a CBAM train/test split FIX the IR drone-recall
collapse, or do CBAM's airplane-like DRONES genuinely overlap CBAM's AIRPLANES?
ZERO-GPU (from cbam.pkl). Same logic that predicted the bird win.

high AUROC(CBAM drones vs CBAM confusers) + the vetoed drones flagged as DRONE by a
drone-vs-confuser separator => MOVABLE (CBAM-split retrain teaches keep-drone/reject-
airplane). low / vetoed drones look like confusers => OVERLAP (no retrain fixes it).

  py eval/cbam_separability_check.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from diagnose_rgbtest_veto_mechanism import match                  # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                          # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from sklearn.model_selection import cross_val_score               # noqa: E402

CACHE = REPO / "eval/results/_offline_pipeline/cache"
BAL = REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt"
THR = 0.05


def auroc_cv(A, B):
    if len(A) < 10 or len(B) < 10:
        return float("nan")
    X = np.vstack([A, B]).astype(np.float32); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    return float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                 StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean())


def main():
    d = pickle.load(open(CACHE / "cbam.pkl", "rb")); fr = d["frames"]; rule = d["meta"]["rule"]
    mlp = MLPv4Verifier(BAL, device="cpu")
    drones, confs, drone_p = [], [], []
    for f in fr:
        n = len(f["confs"])
        if n == 0:
            continue
        p = mlp.predict_drone_probs(np.array(f["feats"]))
        gtb = [tuple(g) for g in f["gt_boxes"]]
        for i, box in enumerate(f["boxes"]):
            is_drone = len(gtb) and max((match(box, g, rule) for g in gtb), default=0) >= 0.5
            (drones if is_drone else confs).append(f["feats"][i])
            if is_drone:
                drone_p.append(float(p[i]))
    drones = np.array(drones, np.float32); confs = np.array(confs, np.float32); drone_p = np.array(drone_p)
    veto = drones[drone_p < THR]
    print(f"CBAM: {len(drones)} drone dets ({(drone_p<THR).mean():.0%} vetoed by balanced@{THR}) | {len(confs)} confuser dets")

    a_all = auroc_cv(drones, confs)
    a_veto = auroc_cv(veto, confs)
    print(f"\n[1] AUROC CBAM drones vs CBAM confusers       = {a_all:.3f}")
    print(f"[2] AUROC VETOED CBAM drones vs CBAM confusers = {a_veto:.3f}  (high => MOVABLE)")

    Xtr = np.vstack([drones, confs]); ytr = np.r_[np.ones(len(drones)), np.zeros(len(confs))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    drone_frac = float((clf.predict(sc.transform(veto)) == 1).mean()) if len(veto) else float("nan")
    print(f"[3] drone-vs-confuser separator keeps {drone_frac:.1%} of the vetoed CBAM drones as DRONE")

    mean = mlp.scaler_mean.cpu().numpy().ravel(); scale = mlp.scaler_scale.cpu().numpy().ravel()
    scl = lambda M: (M - mean) / scale
    Cd, Cc = scl(drones).mean(0), scl(confs).mean(0); vv = scl(veto)
    closer_d = float((np.linalg.norm(vv - Cd, axis=1) < np.linalg.norm(vv - Cc, axis=1)).mean()) if len(veto) else float("nan")
    print(f"[4] vetoed CBAM drones closer to DRONE centroid than CONFUSER: {closer_d:.0%}")

    movable = (not np.isnan(a_veto)) and a_veto >= 0.85 and drone_frac >= 0.6
    print("\n" + ("MOVABLE -- CBAM drones ARE separable from CBAM airplanes; a CBAM train/test split "
                  "(add CBAM-train drones+confusers, eval on CBAM-test) should recover the recall like birds did."
                  if movable else
                  "OVERLAP -- the vetoed CBAM drones sit in the airplane cluster; CBAM-split training will "
                  "fight itself (keep-drone vs reject-airplane on the SAME features). Likely can't fully fix."))


if __name__ == "__main__":
    main()
