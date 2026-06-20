"""
thesis_eval/temporal_threshold_sweep.py — DIAGNOSTIC sweep of the per-frame FILTER threshold on the
real-video segment surface (tab:temporal_production), to locate the video-tuned operating point.

Reads thesis_eval/cache/{video_drone,video_confuser}.pkl. Router = robust8_nr_drop (shipped).
The video regime is ft4-on-RGB + v3b-on-gray(RGB); the per-frame veto is mlp_v5@t_rgb (RGB) |
aligned_gray@t_gray (gray). Router labels + per-detection P(drone) are computed ONCE, then the
threshold is swept by pure re-thresholding (zero GPU). Window = 2-of-3 per clip (table semantics).

Reports window 2-of-3: video_drone P/R/F1 ; video_confuser window fire (suppression = 1-fire).
Baselines (threshold-free): bare, clf-only[router]. Three sweeps: joint, gray-only, rgb-only.

  py -u thesis_eval/temporal_threshold_sweep.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "thesis_eval"))
from pipeline_eval_unified import load_classifiers, load_verifiers, batch_labels, batch_probs  # noqa: E402
from temporal_replay import window_vote, prf  # noqa: E402

CACHE_DIR = REPO / "thesis_eval" / "cache"
ROUTER = "robust8_nr_drop"
THRS = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]


def prep(name, clfs, verifs):
    d = pickle.load(open(CACHE_DIR / f"{name}.pkl", "rb"))
    meta, frames = d["meta"], d["frames"]
    seqs = [f["seq"] for f in frames]
    pos = np.array([len(f["rgb_gt"]) > 0 for f in frames])
    F8, F32 = meta["F8"], meta["F32"]
    F8m = np.stack([f["f8_all"] for f in frames])
    F32m = np.stack([f["f32_all"] for f in frames])
    lab = batch_labels(clfs[ROUTER], F8m, F32m, F8, F32)
    trgb, tir = np.isin(lab, [1, 3]), np.isin(lab, [2, 3])
    rgb_any = np.array([len(f["rgb"]["confs"]) > 0 for f in frames])
    ir_any = np.array([len(f["ir"]["confs"]) > 0 for f in frames])
    rp = batch_probs(frames, "rgb", verifs["mlp_v5"])
    ip = batch_probs(frames, "ir", verifs["aligned_gray"])
    return dict(seqs=seqs, pos=pos, trgb=trgb, tir=tir, rgb_any=rgb_any, ir_any=ir_any,
                rp=rp, ip=ip, n=meta["n"], clips=len(set(seqs)))


def cell_at(D, t_rgb, t_gray):
    n = len(D["pos"])
    rgb_mlp = np.array([bool((D["rp"][i] >= t_rgb).any()) for i in range(n)])
    ir_mlp = np.array([bool((D["ip"][i] >= t_gray).any()) for i in range(n)])
    return (D["trgb"] & rgb_mlp) | (D["tir"] & ir_mlp)


def win_prf(al, D):
    wa, wp = window_vote(al, D["pos"], D["seqs"])
    return prf(wa, wp)


def win_fire(al, D):
    wa, _ = window_vote(al, D["pos"], D["seqs"])
    return round(float(wa.mean()), 4)


def main():
    clfs, verifs = load_classifiers(), load_verifiers()
    Dd, Dc = prep("video_drone", clfs, verifs), prep("video_confuser", clfs, verifs)
    print(f"\nvideo_drone n={Dd['n']} clips={Dd['clips']} | video_confuser n={Dc['n']} clips={Dc['clips']}")

    bare_d = Dd["rgb_any"] | Dd["ir_any"]
    clf_d = (Dd["trgb"] & Dd["rgb_any"]) | (Dd["tir"] & Dd["ir_any"])
    bare_c = Dc["rgb_any"] | Dc["ir_any"]
    clf_c = (Dc["trgb"] & Dc["rgb_any"]) | (Dc["tir"] & Dc["ir_any"])
    print("\nBASELINES (no filter):")
    print(f"  bare              drone win P/R/F1 = {win_prf(bare_d, Dd)}   conf win fire = {win_fire(bare_c, Dc)}")
    print(f"  clf-only[{ROUTER}] drone win P/R/F1 = {win_prf(clf_d, Dd)}   conf win fire = {win_fire(clf_c, Dc)}")

    for title, mk in (("JOINT  (rgb=gray=t)", lambda t: (t, t)),
                      ("GRAY-ONLY (rgb=0.25, gray=t)", lambda t: (0.25, t)),
                      ("RGB-ONLY  (rgb=t, gray=0.25)", lambda t: (t, 0.25))):
        print(f"\nSWEEP clf->filt[{ROUTER}] — {title}")
        print(f"  {'t':>5} | {'drnP':>6} {'drnR':>6} {'drnF1':>6} | {'confFire':>8} {'suppr':>6}")
        for t in THRS:
            tr, tg = mk(t)
            p, r, f = win_prf(cell_at(Dd, tr, tg), Dd)
            cf = win_fire(cell_at(Dc, tr, tg), Dc)
            tag = "  <- SHIPPED" if (tr, tg) == (0.25, 0.25) else ""
            print(f"  {t:>5.2f} | {p:>6.3f} {r:>6.3f} {f:>6.3f} | {cf:>8.3f} {1 - cf:>6.3f}{tag}")

    # ── channel isolation: effect of DROPPING the grayscale fallback channel ──────────────────────
    print("\nCHANNEL ISOLATION — drone clips: which channel contributes recall vs confuser fire")
    print(f"  {'config':>20} | {'drnP':>6} {'drnR':>6} {'drnF1':>6} | {'confFire':>8} {'suppr':>6}")

    def chan(D, which, t_rgb, t_gray):
        n = len(D["pos"])
        rgb = D["trgb"] & np.array([bool((D["rp"][i] >= t_rgb).any()) for i in range(n)])
        gry = D["tir"] & np.array([bool((D["ip"][i] >= t_gray).any()) for i in range(n)])
        return {"rgb": rgb, "gray": gry, "fused": rgb | gry}[which]

    channels = {}
    for t in (0.25, 0.05):
        for nm in ("rgb", "gray", "fused"):
            p, r, f = win_prf(chan(Dd, nm, t, t), Dd)
            cf = win_fire(chan(Dc, nm, t, t), Dc)
            label = {"rgb": "RGB-only (drop gray)", "gray": "gray-only", "fused": "fused (shipped)"}[nm]
            print(f"  {label+' @'+format(t, '.2f'):>20} | {p:>6.3f} {r:>6.3f} {f:>6.3f} | {cf:>8.3f} {1 - cf:>6.3f}")
            channels[f"{nm}@{t:.2f}"] = {"drone_window_PRF": [p, r, f], "confuser_window_fire": cf}

    # ── persist the load-bearing cells for the thesis caption / % source ──────────────────────────
    out = {
        "regime": "grayrgb_paired (ft4 RGB + v3b-on-gray); router=robust8_nr_drop; window 2-of-3",
        "weights": {"rgb_filter": "mlp_v5_v4", "gray_filter": "mlp_aligned_gray_balanced"},
        "joint_sweep": {f"{t:.2f}": {"drone_window_PRF": list(win_prf(cell_at(Dd, t, t), Dd)),
                                      "confuser_window_fire": win_fire(cell_at(Dc, t, t), Dc)} for t in THRS},
        "channel_isolation": channels,
    }
    op = REPO / "thesis_eval" / "results" / "temporal_filter_sweep.json"
    json.dump(out, open(op, "w"), indent=2, default=float)
    print(f"\nDONE -> {op}")


if __name__ == "__main__":
    main()
