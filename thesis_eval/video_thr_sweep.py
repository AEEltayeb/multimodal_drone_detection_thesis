"""video_thr_sweep.py — verifier-threshold sweep on the CONSECUTIVE video caches (zero GPU).

The temporal table showed benchmark-tuned verifier thresholds (rgb 0.25 / gray 0.25) over-vetoing
OOD drones at the segment grain. The probabilities are cached, so the video-tuned operating point
is a pure replay: sweep a shared verifier threshold t over both channels and report, per t,
window-level drone P/R/F1 (video_drone) and window fire (video_confuser) for:
  filt(t)                  — verifier only
  clf->filt[robust8](t)    — production composition
  clf->filt[robust6](t)    — best-router composition on this surface
Reference rows: bare and clf-only (t = none).

  py -u thesis_eval/video_thr_sweep.py
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "thesis_eval"))
from pipeline_eval_unified import load_classifiers, load_verifiers, batch_labels, batch_probs  # noqa: E402
from temporal_replay import window_vote, prf  # noqa: E402
import pickle  # noqa: E402

CACHE_DIR = REPO / "thesis_eval" / "cache"
OUT_DIR = REPO / "thesis_eval" / "results"
THRS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.25]


def load(name):
    d = pickle.load(open(CACHE_DIR / f"{name}.pkl", "rb"))
    return d["meta"], d["frames"]


def main():
    clfs, verifs = load_classifiers(), load_verifiers()
    surfaces = {}
    for name in ("video_drone", "video_confuser"):
        meta, frames = load(name)
        F8, F32 = meta["F8"], meta["F32"]
        F8m = np.stack([f["f8_all"] for f in frames])
        F32m = np.stack([f["f32_all"] for f in frames])
        labels = {c: batch_labels(clf, F8m, F32m, F8, F32) for c, clf in clfs.items()}
        rp = batch_probs(frames, "rgb", verifs["mlp_v5"])
        ip = batch_probs(frames, "ir", verifs["aligned_gray"])
        surfaces[name] = {
            "seqs": [f["seq"] for f in frames],
            "pos": np.array([len(f["rgb_gt"]) > 0 for f in frames]),
            "rgb_any": np.array([len(f["rgb"]["confs"]) > 0 for f in frames]),
            "ir_any": np.array([len(f["ir"]["confs"]) > 0 for f in frames]),
            "rp": rp, "ip": ip, "labels": labels, "n": meta["n"],
        }

    def fire(S, t, router=None):
        n = len(S["pos"])
        rgb = (np.array([bool((S["rp"][i] >= t).any()) for i in range(n)]) if t is not None else S["rgb_any"])
        ir = (np.array([bool((S["ip"][i] >= t).any()) for i in range(n)]) if t is not None else S["ir_any"])
        if router is None:
            return rgb | ir
        lab = S["labels"][router]
        return (np.isin(lab, [1, 3]) & rgb) | (np.isin(lab, [2, 3]) & ir)

    def row(t, router):
        D, Cn = surfaces["video_drone"], surfaces["video_confuser"]
        wa, wp = window_vote(fire(D, t, router), D["pos"], D["seqs"])
        p, r, f = prf(wa, wp)
        wc, _ = window_vote(fire(Cn, t, router), Cn["pos"], Cn["seqs"])
        return p, r, f, round(float(wc.mean()), 4)

    L = ["# Video verifier-threshold sweep (segment 2-of-3; zero-GPU replay)",
         f"{time.strftime('%Y-%m-%d %H:%M')} | shared thr on mlp_v5 (RGB) + aligned_gray (gray channel)",
         "", "| cell | thr | window P | window R | window F1 | confuser window fire |",
         "|---|---|---|---|---|---|"]
    results = {}
    for router in (None, "robust8", "robust6"):
        rn = router or "filt-only"
        p, r, f, c = row(None, router)
        tag = "bare" if router is None else f"clf[{router}] (no filt)"
        L.append(f"| {tag} | --- | {p} | {r} | {f} | {c} |")
        results[f"{rn}@none"] = [p, r, f, c]
        for t in THRS:
            p, r, f, c = row(t, router)
            cell = f"filt({t})" if router is None else f"clf->filt[{router}]({t})"
            L.append(f"| {cell} | {t} | {p} | {r} | {f} | {c} |")
            results[f"{rn}@{t}"] = [p, r, f, c]
    (OUT_DIR / "video_thr_sweep.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(OUT_DIR / "video_thr_sweep.json", "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
