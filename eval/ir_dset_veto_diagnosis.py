"""ir_dset_veto_diagnosis.py — the balanced IR filter loses ir_dset_final recall.
Are the lost drones genuinely INSIDE the thermal-airplane/confuser cluster
(ambiguous -> up-weighting / retrain can't easily recover them) or just nudged
across a movable boundary (-> up-weight retrain WILL recover them)?  ZERO-GPU.

  py eval/ir_dset_veto_diagnosis.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from diagnose_rgbtest_veto_mechanism import match                  # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                          # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402
from sklearn.model_selection import cross_val_score               # noqa: E402

CACHE = REPO / "eval/results/_offline_pipeline/cache"
BAL = REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt"
GOOD = ["antiuav_ir", "svanstrom_ir", "ir_video"]   # surfaces where recall HELD
THR = 0.05


def drone_feats_probs(name, mlp):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb")); rule = d["meta"]["rule"]
    F, P = [], []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
            continue
        p = mlp.predict_drone_probs(np.array(fr["feats"]))
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g, rule) for g in fr["gt_boxes"]), default=0) >= 0.5:
                F.append(fr["feats"][i]); P.append(float(p[i]))
    return np.array(F, np.float32), np.array(P)


def conf_feats(name):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    return np.array([f for fr in d["frames"] for f in fr["feats"]], np.float32)


def auroc_cv(A, B):
    X = np.vstack([A, B]).astype(np.float32); y = np.r_[np.ones(len(A)), np.zeros(len(B))]
    return float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                                 StandardScaler().fit_transform(X), y, cv=5, scoring="roc_auc").mean())


def main():
    mlp = MLPv4Verifier(BAL, device="cpu")
    Xd, Pd = drone_feats_probs("ir_dset_final", mlp)
    veto = Xd[Pd < THR]; kept = Xd[Pd >= THR]
    print(f"ir_dset_final drones {len(Pd)} | VETOED(<{THR}) {len(veto)} | kept {len(kept)}")
    Xconf = conf_feats("ir_confusers")                       # thermal confusers (~76% airplane)
    Xgood = np.vstack([drone_feats_probs(n, mlp)[0] for n in GOOD])   # held thermal drones

    # 1) can features separate the LOST ir_dset drones from the airplane/confuser cluster?
    a_veto = auroc_cv(veto, Xconf)
    a_good = auroc_cv(Xgood, Xconf)
    print(f"\n[1] AUROC vetoed-ir_dset vs thermal-confusers = {a_veto:.3f}  (high => separable => MOVABLE)")
    print(f"[2] AUROC held-drones   vs thermal-confusers = {a_good:.3f}  (reference ceiling)")

    # 3) a drone-vs-confuser separator (held drones vs confusers): does it call the vetoed drones DRONE?
    Xtr = np.vstack([Xgood, Xconf]); ytr = np.r_[np.ones(len(Xgood)), np.zeros(len(Xconf))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    drone_frac = float((clf.predict(sc.transform(veto)) == 1).mean())
    print(f"[3] drone-vs-confuser separator keeps {drone_frac:.1%} of vetoed ir_dset drones as DRONE")

    # 4) centroid proximity in the balanced filter's scaled space
    mean = mlp.scaler_mean.cpu().numpy().ravel(); scale = mlp.scaler_scale.cpu().numpy().ravel()
    scl = lambda M: (M - mean) / scale
    Cd, Cc = scl(Xgood).mean(0), scl(Xconf).mean(0); vv = scl(veto)
    closer_drone = float((np.linalg.norm(vv - Cd, axis=1) < np.linalg.norm(vv - Cc, axis=1)).mean())
    print(f"[4] vetoed ir_dset drones closer to DRONE centroid than CONFUSER: {closer_drone:.0%}")

    movable = (a_veto >= 0.85) and (drone_frac >= 0.6)
    print("\n" + ("MOVABLE -- the lost drones are separable from airplanes; an up-weight retrain (or lower thr) "
                  "should recover ir_dset recall while keeping the suppression win."
                  if movable else
                  "AMBIGUOUS -- the lost ir_dset drones sit in the thermal-airplane cluster; up-weighting will "
                  "fight the suppression. Prefer the thr~0.002 ship point (recall held, modest suppression) "
                  "or accept a small ir_dset recall cost for big suppression."))


if __name__ == "__main__":
    main()
