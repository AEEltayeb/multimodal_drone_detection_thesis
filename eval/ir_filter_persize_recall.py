"""ir_filter_persize_recall.py — does the IR aligned filter have a hidden
SMALL-thermal-drone veto (the RGB pathology) on the thermal path?  ZERO-GPU.

For each thermal drone surface, take GT-matched real-drone detections, bucket by
detection short-side px, and report the filter veto rate per size bucket (aligned
@0.05 deploy + @0.25; native @0.25 for contrast). If sub-16px veto stays low,
the IR filter has no coverage-gap recall hole and does NOT need drone balancing.

  py eval/ir_filter_persize_recall.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE   # noqa: E402
from diagnose_rgbtest_veto_mechanism import match                        # noqa: E402

ALIGNED = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
NATIVE = str(REPO / "eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt")
DRONE = ["antiuav_ir", "ir_dset_final", "svanstrom_ir", "ir_video"]
EDGES = [0, 16, 32, 64, 1e9]; NAMES = ["<16px", "16-32", "32-64", ">=64"]


def sbin(box):
    s = min(box[2] - box[0], box[3] - box[1])
    for i in range(4):
        if EDGES[i] <= s < EDGES[i + 1]:
            return i
    return 3


def collect(name, mlp):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    rule = d["meta"]["rule"]; frames = d["frames"]
    probs = mlp_probs_per_frame(frames, mlp)
    bins, ps = [], []
    for fi, fr in enumerate(frames):
        if len(fr["gt_boxes"]) == 0:
            continue
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g, rule) for g in fr["gt_boxes"]), default=0) >= 0.5:
                bins.append(sbin(box)); ps.append(float(probs[fi][i]))
    return np.array(bins), np.array(ps)


def report(tag, mlp_path, thr):
    mlp = get_mlp(mlp_path)
    allb, allp = [], []
    for n in DRONE:
        b, p = collect(n, mlp); allb.append(b); allp.append(p)
    b = np.concatenate(allb); p = np.concatenate(allp)
    print(f"\n=== {tag} @ thr {thr} (pooled thermal drones, n={len(b)}) ===")
    print(f"{'bucket':<8} {'matched':>8} {'kept':>6} {'veto%':>7}")
    for i, nm in enumerate(NAMES):
        m = b == i; n_m = int(m.sum())
        if n_m == 0:
            print(f"{nm:<8} {0:>8} {'-':>6} {'-':>7}"); continue
        kept = int((p[m] >= thr).sum())
        print(f"{nm:<8} {n_m:>8} {kept:>6} {100*(1-kept/n_m):>6.1f}%")
    print(f"{'TOTAL':<8} {len(b):>8} {int((p>=thr).sum()):>6} {100*(1-(p>=thr).mean()):>6.1f}%")


def main():
    report("aligned (shipped)", ALIGNED, 0.05)
    report("aligned (shipped)", ALIGNED, 0.25)
    report("native (mlp_v5_ir)", NATIVE, 0.25)


if __name__ == "__main__":
    main()
