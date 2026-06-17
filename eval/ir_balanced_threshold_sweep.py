"""ir_balanced_threshold_sweep.py — find the operating point for a candidate IR
filter that HOLDS thermal drone recall AND still beats shipped on confuser FP.
ZERO-GPU. The acceptance eval rejected at the shipped deploy thr (0.05) because the
candidate is more aggressive; sweep to find where both bars pass.

  py eval/ir_balanced_threshold_sweep.py                                  # default: balanced
  py eval/ir_balanced_threshold_sweep.py --weight mri/results/ir_aligned_cbam_thermalonly/classifiers/mlp_aligned.pt --label cbam
NOTE: the ir_confusers FP column uses the CACHED ir_confusers (= the TRAIN split the
balanced/cbam filters trained on) => LEAKY/optimistic; cross-ref the held-out val/test
numbers from eval/eval_ir_heldout.py. The DRONE recall columns are NOT leaky.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections                      # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE   # noqa: E402

SHIP = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
BAL = str(REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt")
DRONE = ["antiuav_ir", "ir_dset_final", "svanstrom_ir", "ir_video"]
CONF = "ir_confusers"
THRS = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]


def recall_fp(weight, thr):
    mlp = get_mlp(weight)
    recs = {}
    for n in DRONE:
        d = pickle.load(open(CACHE / f"{n}.pkl", "rb")); fr = d["frames"]; rule = d["meta"]["rule"]
        probs = mlp_probs_per_frame(fr, mlp)
        tp = fp = fn = 0
        for fi, f in enumerate(fr):
            nn = len(f["confs"]); keep = probs[fi] >= thr if nn else np.zeros(0, bool)
            kept = [(tuple(f["boxes"][i]), float(f["confs"][i])) for i in range(nn) if keep[i]]
            t, f_, n_ = score_detections(kept, [tuple(g) for g in f["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
            tp += t; fp += f_; fn += n_
        recs[n] = compute_prf(tp, fp, fn)["recall"]
    d = pickle.load(open(CACHE / f"{CONF}.pkl", "rb")); fr = d["frames"]
    probs = mlp_probs_per_frame(fr, mlp)
    fpc = sum(int((probs[fi] >= thr).sum()) for fi in range(len(fr)))
    return recs, fpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default=BAL, help="candidate filter weight (default: balanced)")
    ap.add_argument("--label", default="BALANCED", help="label for the candidate column header")
    a = ap.parse_args()
    cand = str(REPO / a.weight) if not Path(a.weight).is_absolute() and not Path(a.weight).exists() else a.weight
    sr, sfp = recall_fp(SHIP, 0.05)
    print("SHIPPED @0.05 (the bar to beat):")
    print(f"  " + "  ".join(f"{n.split('_')[0]}={sr[n]:.3f}" for n in DRONE) + f"  | ir_confusers FP={sfp}")
    bar_rec = {n: sr[n] for n in DRONE}; bar_fp = sfp
    print(f"\n{a.label} filter sweep (PASS = every drone recall >= shipped-0.01 AND FP < {bar_fp}):")
    print(f"{'thr':>6} " + " ".join(f"{n.split('_')[0]:>9}" for n in DRONE) + f" {'confFP':>7}  verdict")
    for thr in THRS:
        r, fp = recall_fp(cand, thr)
        rec_ok = all(r[n] >= bar_rec[n] - 0.01 for n in DRONE)
        fp_ok = fp < bar_fp
        ok = rec_ok and fp_ok
        print(f"{thr:>6.3f} " + " ".join(f"{r[n]:>9.3f}" for n in DRONE) +
              f" {fp:>7}  {'PASS' if ok else ('recall' if not rec_ok else 'fp')}")
    print(f"\n(bare ir_dset_final recall ceiling ~0.969; shipped ir_dset_final={bar_rec['ir_dset_final']:.3f})")


if __name__ == "__main__":
    main()
