"""rgb_balanced_threshold_sweep.py — map the RGB balanced filter's recall/FP across
thresholds, to see whether any single operating point holds rgb_dataset_test's win
while recovering selcom recall and controlling bird-confuser FP. ZERO-GPU.

  py eval/rgb_balanced_threshold_sweep.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections                      # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE   # noqa: E402

SHIP = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
BAL = str(REPO / "eval/results/_v5_balanced_remine/classifiers/mlp_v5_balanced.pt")
DRONE = ["rgb_dataset_test", "svanstrom", "selcom_val", "antiuav_rgb"]
CONF = ["rgb_confuser", "rgb_bird_confuser"]
THRS = [0.05, 0.10, 0.15, 0.25, 0.40]


def evalw(weight, thr):
    mlp = get_mlp(weight); rec = {}
    for n in DRONE:
        d = pickle.load(open(CACHE / f"{n}.pkl", "rb")); fr = d["frames"]; rule = d["meta"]["rule"]
        probs = mlp_probs_per_frame(fr, mlp)
        tp = fp = fn = 0
        for fi, f in enumerate(fr):
            nn = len(f["confs"]); keep = probs[fi] >= thr if nn else np.zeros(0, bool)
            kept = [(tuple(f["boxes"][i]), float(f["confs"][i])) for i in range(nn) if keep[i]]
            t, f_, n_ = score_detections(kept, [tuple(g) for g in f["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
            tp += t; fp += f_; fn += n_
        rec[n] = compute_prf(tp, fp, fn)["recall"]
    fps = {}
    for n in CONF:
        d = pickle.load(open(CACHE / f"{n}.pkl", "rb")); fr = d["frames"]
        probs = mlp_probs_per_frame(fr, mlp)
        fps[n] = sum(int((probs[fi] >= thr).sum()) for fi in range(len(fr)))
    return rec, fps


def main():
    sr, sf = evalw(SHIP, 0.25)
    print("SHIPPED @0.25 (bars):")
    print("  " + "  ".join(f"{n}={sr[n]:.3f}" for n in DRONE) + "  | " + " ".join(f"{n}FP={sf[n]}" for n in CONF))
    print(f"\nBALANCED sweep:")
    print(f"{'thr':>6} " + " ".join(f"{n.split('_')[0][:7]:>8}" for n in DRONE) + "  " + " ".join(f"{n.split('_')[1][:4]+'FP':>7}" for n in CONF))
    for thr in THRS:
        r, fp = evalw(BAL, thr)
        print(f"{thr:>6.2f} " + " ".join(f"{r[n]:>8.3f}" for n in DRONE) + "  " + " ".join(f"{fp[n]:>7}" for n in CONF))
    print(f"\n(bars: rgb_dataset_test MUST be > {sr['rgb_dataset_test']:.3f}; selcom/svan/antiuav within -1pp; "
          f"rgb_confuser FP <= {sf['rgb_confuser']}, bird FP <= {sf['rgb_bird_confuser']})")


if __name__ == "__main__":
    main()
