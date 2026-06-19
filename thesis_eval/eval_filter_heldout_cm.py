"""thesis_eval/eval_filter_heldout_cm.py — detection-level confusion matrix for the production
confuser filters by RE-SCORING the cached 517-D detector features with the production weights
(zero-GPU). Drone side = clean held-out TEST split; confuser side flagged for train-overlap.

RGB filter: models/verifiers/rgb_v5/mlp_v5_v4.pt        @ P(drone) >= 0.25  (ft4 features)
IR  filter: models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt @ 0.05  (v3b features)

The filter keeps a detection iff P(drone) >= thr.  On a DRONE surface, a true-drone detection
kept = TP, vetoed = FN.  On a CONFUSER surface (no drones), kept = FP, vetoed = TN.
Cross-check: drone recall here should match the model cards (rgb_dataset_test 0.874, ir_dset 0.928).

Run: py thesis_eval/eval_filter_heldout_cm.py
"""
from __future__ import annotations
import pickle, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
for sub in ("eval", "classifier"):
    sys.path.insert(0, str(REPO / sub))
from metrics import score_detections                    # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier               # noqa: E402

CACHE = REPO / "thesis_eval" / "cache"
RGB_FILTER = REPO / "models/verifiers/rgb_v5/mlp_v5_v4.pt"
IR_FILTER = REPO / "models/verifiers/ir_aligned/mlp_aligned_thermalonly.pt"


def load(name):
    return pickle.load(open(CACHE / f"{name}.pkl", "rb"))


def gts(arr):
    return [(float(g[0]), float(g[1]), float(g[2]), float(g[3])) for g in arr]


def dets2(slot, mask=None):
    b, c = slot["boxes"], slot["confs"]
    idx = range(len(c)) if mask is None else np.where(mask)[0]
    return [((float(b[i][0]), float(b[i][1]), float(b[i][2]), float(b[i][3])), float(c[i])) for i in idx]


def probs_for(frames, slot, ver):
    out = []
    for fr in frames:
        if len(fr[slot]["confs"]):
            out.append(ver.predict_drone_probs(np.asarray(fr[slot]["feats"], np.float32)))
        else:
            out.append(np.zeros(0, np.float32))
    return out


def drone_cm(cache, slot, gtk, ver, thr, conf_gate):
    rule = cache["meta"]["rule"]; frames = cache["frames"]; P = probs_for(frames, slot, ver)
    tpB = fnB = tpF = 0
    for fr, p in zip(frames, P):
        gt = gts(fr[gtk])
        if not gt:
            continue
        cmask = np.asarray(fr[slot]["confs"], np.float32) >= conf_gate
        t, _, n = score_detections(dets2(fr[slot], cmask), gt, rule=rule); tpB += t; fnB += n
        t2, _, _ = score_detections(dets2(fr[slot], cmask & (p >= thr)), gt, rule=rule); tpF += t2
    tot = tpB + fnB
    return dict(rule=rule, conf_gate=conf_gate, gt_drone_dets=tot, kept_TP=tpF, vetoed_FN=tpB - tpF,
                missed_bare_FN=fnB, recall_filtered=round(tpF / max(tot, 1), 4))


def confuser_cm(cache, slot, ver, thr, conf_gate):
    frames = cache["frames"]; P = probs_for(frames, slot, ver); n = kept = 0
    for fr, p in zip(frames, P):
        c = np.asarray(fr[slot]["confs"], np.float32) >= conf_gate
        n += int(c.sum()); kept += int((c & (p >= thr)).sum())
    return dict(confuser_dets=n, kept_FP=kept, vetoed_TN=n - kept,
                veto_rate=round((n - kept) / max(n, 1), 4))


def main():
    print("== RGB confuser filter  mlp_v5_v4  @0.25  (ft4 features, detector conf>=0.25) ==")
    v = MLPv4Verifier(RGB_FILTER, device="cpu")
    rgb_d = drone_cm(load("rgb_dataset_test"), "rgb", "rgb_gt", v, 0.25, 0.25)
    rgb_c = confuser_cm(load("rgb_confuser"), "rgb", v, 0.25, 0.25)
    print("  drone   [rgb_dataset_test, clean held-out]:", rgb_d)
    print("  confuser[rgb_confuser, OOD; partial train-overlap]:", rgb_c)
    print("\n== IR confuser filter  mlp_aligned_thermalonly  @0.05  (v3b features, DEPLOY conf>=0.40) ==")
    vi = MLPv4Verifier(IR_FILTER, device="cpu")
    ir_d = drone_cm(load("ir_dset_final"), "ir", "ir_gt", vi, 0.05, 0.40)
    ir_c = confuser_cm(load("ir_confusers"), "ir", vi, 0.05, 0.40)
    print("  drone   [ir_dset_final test, clean held-out]:", ir_d)
    print("  confuser[ir_confusers, OOD; partial train-overlap]:", ir_c)

    out = {"rgb_filter": {"model": "mlp_v5_v4", "thr": 0.25, "conf_gate": 0.25,
                          "drone_rgb_dataset_test": rgb_d, "confuser_rgb_confuser": rgb_c},
           "ir_filter": {"model": "mlp_aligned_thermalonly", "thr": 0.05, "conf_gate": 0.40,
                         "drone_ir_dset_final": ir_d, "confuser_ir_confusers": ir_c}}
    od = REPO / "thesis_eval" / "results" / "per_model_heldout"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "filter_heldout_cm.json", "w"), indent=2)
    print(f"\nwrote {od / 'filter_heldout_cm.json'}")


if __name__ == "__main__":
    main()
