"""negative_frame_fire.py — v3b fire rate on ir_dset_final's NEGATIVE frames (notes round 2, N11).

The IR corpus is 27.5% negatives (mined sea / sky / building scenes). Its test cache stores
per-frame detections + GT, so the detector's fire rate on the frames with EMPTY ground truth is an
exact zero-GPU replay: it measures whether the production IR detector hallucinates on the very
scene types its negatives were mined from. Also prints the +filter rate (aligned thermal scaler)
and, for context, the same numbers on rgb_dataset_test's negative frames (ft4 + mlp_v5).

  py -u thesis_eval/negative_frame_fire.py
"""
from __future__ import annotations
import json, pickle
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
for _sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))
from pipeline_eval_unified import (load_verifiers, batch_probs,            # noqa: E402
                                   IR_THR_MLP, RGB_THR_MLP)

CACHE = REPO / "thesis_eval/cache"
OUT = REPO / "thesis_eval/results"

SURFACES = [("ir_dset_final", "ir", "ir_gt", "aligned", IR_THR_MLP),
            ("rgb_dataset_test", "rgb", "rgb_gt", "mlp_v5", RGB_THR_MLP)]


def main():
    verifs = load_verifiers()
    out = {}
    for name, slot, gkey, vkey, thr in SURFACES:
        d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
        frames = [fr for fr in d["frames"] if len(fr[gkey]) == 0]   # negative frames only
        probs = batch_probs(frames, slot, verifs[vkey])
        n = len(frames)
        fired = sum(1 for fr in frames if len(fr[slot]["confs"]) > 0)
        fp = sum(len(fr[slot]["confs"]) for fr in frames)
        fired_f = sum(1 for i, fr in enumerate(frames) if (probs[i] >= thr).any())
        fp_f = sum(int((probs[i] >= thr).sum()) for i in range(n))
        out[name] = {"n_negative_frames": n, "n_total": d["meta"]["n"],
                     "bare": {"FP": fp, "fired": fired, "fire_rate": round(fired / max(n, 1), 4)},
                     "filt": {"FP": fp_f, "fired": fired_f, "fire_rate": round(fired_f / max(n, 1), 4)},
                     "verifier": vkey, "thr": thr}
        print(f"[{name}] negatives {n}/{d['meta']['n']}  bare fire {out[name]['bare']['fire_rate']} "
              f"({fp} FP dets)  +filt {out[name]['filt']['fire_rate']} ({fp_f})")
    json.dump(out, open(OUT / "negative_frame_fire.json", "w"), indent=2)
    print("DONE ->", OUT / "negative_frame_fire.json")


if __name__ == "__main__":
    main()
